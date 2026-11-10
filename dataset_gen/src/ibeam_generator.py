#!/usr/bin/env python3
"""I-beam training samples (host-side driver): DIN I-beam segments, their three plates and their envelopes, all built in
FreeCAD (dataset_gen/src/ibeam_freecad.py run by freecadcmd) and registered with src/sample_writer.finish_sample.

Per beam (a profile from src/din_profiles.py, length L = U(0.5, 2) h, a random rotation, optionally a mitred end):
  * the beam itself:     topology "box", subclass "ibeam", fillet true (root fillets r; the tapered I series also toe
                         radii) - or, with probability --nofillet-prob, the same beam without fillets (fillet false);
  * its three plates:    topology "box", subclass "full", source "ibeam_plate", role "flange" | "web" (plate
                         "top_flange" | "bottom_flange" | "web") - each a Part.makeBox in the plate's own sample,
                         kept with probability --plate-fraction. These carry the thin-plate aspect ratios
                         (b : tf 5-22, web depth : tw 7-47 over the table) the old box data lacks;
  * its envelope:        topology "box", subclass "full", source "ibeam_envelope" (the b x h x L bounding box: the
                         hard negative), kept with probability --envelope-fraction.
Every sample gets its own in-scale draw (size fraction, position) on the 40 mm grid; the STL FreeCAD writes is already
placed in that grid, which is what finish_sample voxelises (.bin), fields (.field.npz) and registers
(<out>/shard<K>/box/dataset_metadata.jsonl, readable by training/manifest.py).

Record extras: `ibeam` (profile, series, standard, end cut, beam_id), `boxes` (the three plates in WORLD coordinates of
the placed sample: centre, half extents, quaternion [qx, qy, qz, qw]; mean-thickness plates for the tapered series; for a
mitred beam the plates of the uncut prism, `boxes_exact` false), `freecad` (volume, validity), `beam_id` on every sample
of one beam (keep them on one side of a split), and placement.freecad_volume_mm3 (placed volume, for the voxel check).

Resolution: the web of an HE-A/IPE (tw / h = 0.02-0.03) is only 1-1.5 voxels at 64^3 even when a short segment fills the
grid. plan_beam draws size and length so the thinnest wall is >= --min-thin-vox voxels where that is possible; the rest
are skipped (or, with --keep-thin-misses, written with placement.thin_ok false); placement.tw_vox / tf_vox (beam) and thickness_vox (plates, envelopes) record it.

    python3 -m src.ibeam_generator --out local_dataset_ibeam/small_train --n-per-profile 1 --seed 1 --shard 0
(run from dataset_gen/).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import din_profiles as dp                                                    # noqa: E402
from src.dataset_utils import Rotation, clean_subprocess_env, sample_rotation        # noqa: E402

FREECAD_SCRIPT = Path(__file__).resolve().parent / "ibeam_freecad.py"
GRID_MM = 40.0
PLATES = ("top_flange", "bottom_flange", "web")

# the default selection: spread over the series and over h (about 20 profiles), incl. the frame's HEA 500
DEFAULT_PROFILES = ("IPE 80", "IPE 160", "IPE 270", "IPE 400", "IPE 600",
                    "HEA 100", "HEA 200", "HEA 360", "HEA 500", "HEA 600",
                    "HEB 100", "HEB 200", "HEB 300", "HEB 400", "HEB 600",
                    "HEM 100", "HEM 300", "HEM 600",
                    "I 100", "I 200")


# ------------------------------------------------------------------------------------------------ pure planning

def _norm(name: str) -> str:
    return name.replace(" ", "").upper()


def resolve_profiles(spec: Optional[str]) -> List[dp.IProfile]:
    """"default" / None -> DEFAULT_PROFILES; "all" -> every profile; otherwise a comma list of profile names ("HEA 500",
    "HEA500") and/or series names ("IPE")."""
    if spec is None or spec == "default":
        return [dp.PROFILES[n] for n in DEFAULT_PROFILES]
    if spec == "all":
        return dp.select()
    by_norm = {_norm(n): p for n, p in dp.PROFILES.items()}
    out: List[dp.IProfile] = []
    for token in (t.strip() for t in spec.split(",")):
        if not token:
            continue
        if token.upper() in {"IPE", "HEA", "HEB", "HEM", "I"}:
            out.extend(dp.select(token.upper()))
        elif _norm(token) in by_norm:
            out.append(by_norm[_norm(token)])
        else:
            raise ValueError(f"unknown profile or series {token!r}")
    return out


def quat_matrix(q: Sequence[float]) -> np.ndarray:
    """[qx, qy, qz, qw] -> 3x3 rotation (the convention of FreeCAD's Rotation(qx, qy, qz, qw) and the box macro)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotated_extent(half: Sequence[float], quat: Sequence[float]) -> np.ndarray:
    """World bounding-box extent of an axis-aligned local box (half extents) after the rotation: 2 |R| half."""
    return 2.0 * np.abs(quat_matrix(quat)) @ np.asarray(half, float)


def _sample_boxes(profile: dp.IProfile, length: float) -> Dict[str, Tuple[np.ndarray, float]]:
    """{sample kind/plate: (local half extents of its bounding box, its thinnest wall in mm)}."""
    p = profile
    env = np.array([p.b / 2, p.h / 2, length / 2])
    out = {"ibeam": (env, min(p.tw, p.tf)), "envelope": (env, 2 * float(env.min()))}
    for name, (_, half) in local_plates(p, length).items():
        out[name] = (half, 2 * float(half.min()))
    return out


def plan_beam(rng: random.Random, profile: dp.IProfile, beam_id: str, *, length_range=(0.5, 2.0),
              nofillet_prob: float = 0.2, mitre_prob: float = 0.25, plate_fraction: float = 1.0,
              envelope_fraction: float = 1.0, size_fraction=(0.5, 0.95), min_thin_vox: float = 1.2,
              resolution: int = 64, grid_mm: float = GRID_MM, max_tries: int = 30) -> Dict:
    """All random draws of one beam and of the samples made from it (no FreeCAD).

    min_thin_vox: the thinnest wall of every sample (the web / flange of the beam, the plate thickness) should be at
    least this many voxels, or the voxel grid breaks the body into pieces (measured: an I 200 web at 0.84 voxels gave
    3 components; an HEA 500 web at 1.09 stayed one piece). Per sample the size fraction is drawn from
    [max(lo, needed), hi]; if even hi is not enough, the beam length is redrawn (a shorter segment has a smaller
    extent) up to max_tries times; a sample that still misses it keeps hi and is flagged (thin_ok false)."""
    h = profile.h
    rotation = sample_rotation(rng, -180.0, 180.0)
    quat = [rotation.qx, rotation.qy, rotation.qz, rotation.qw]
    v = grid_mm / resolution
    fillet = rng.random() >= nofillet_prob
    kinds = ["ibeam"] + [pl for pl in PLATES if rng.random() < plate_fraction] \
        + (["envelope"] if rng.random() < envelope_fraction else [])

    def needed(length):
        out = {}
        for kind, (half, thin) in _sample_boxes(profile, length).items():
            out[kind] = min_thin_vox * v * float(rotated_extent(half, quat).max()) / (grid_mm * thin)
        return out

    length = rng.uniform(*length_range) * h
    for _ in range(max_tries):
        need = needed(length)
        if all(need[k] <= size_fraction[1] for k in kinds):
            break
        length = rng.uniform(*length_range) * h
    need = needed(length)
    mitre = None
    if rng.random() < mitre_prob:
        axis = rng.choice(("x", "y"))
        angle = rng.uniform(30.0, 60.0)
        extent = profile.h if axis == "x" else profile.b
        if extent * math.tan(math.radians(angle)) < 0.7 * length:
            mitre = dict(axis=axis, angle=angle)

    samples = []
    for kind in kinds:
        lo = max(size_fraction[0], need[kind])
        thin_ok = lo <= size_fraction[1]
        frac = rng.uniform(lo, size_fraction[1]) if thin_ok else size_fraction[1]
        s = dict(kind="plate", plate=kind) if kind in PLATES else dict(kind=kind)
        s.update(size_fraction=frac, center_unit=[rng.uniform(-1, 1) for _ in range(3)], thin_ok=thin_ok,
                 thin_vox_expected=frac * min_thin_vox / need[kind] if need[kind] > 0 else float("inf"))
        samples.append(s)
    return dict(beam_id=beam_id, profile=profile, length=length, rotation=rotation, fillet=fillet, mitre=mitre,
                samples=samples)


def local_plates(profile: dp.IProfile, length: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return dict(zip(PLATES, dp.three_boxes(profile, length)))


def world_boxes(profile: dp.IProfile, length: float, quat: Sequence[float], k: float, T: Sequence[float]) -> List[Dict]:
    """The three plates in world coordinates of a sample placed as world = k R local + T."""
    R = quat_matrix(quat)
    out = []
    for name, (c, half) in local_plates(profile, length).items():
        out.append(dict(plate=name, center=[float(v) for v in k * (R @ c) + np.asarray(T, float)],
                        half_extents=[float(v) for v in k * half], quaternion=[float(v) for v in quat]))
    return out


def profile_parameters(profile: dp.IProfile, length: float, fillet: bool, mitre: Optional[Dict]) -> Dict[str, float]:
    p = profile
    return dict(h=p.h, b=p.b, tw=p.tw, tf=p.tf, r=p.r, area_cm2=p.area_cm2, flange_slope=p.flange_slope,
                length=float(length), tw_ratio=p.tw / p.h, tf_ratio=p.tf / p.h, b_ratio=p.b / p.h, r_ratio=p.r / p.h,
                length_ratio=float(length) / p.h, fillet=1.0 if fillet else 0.0,
                mitre_angle=float(mitre["angle"]) if mitre else 0.0)


def build_jobs(plan: Dict, box_dir: Path, first_index: int, grid_mm: float = GRID_MM) -> List[Dict]:
    """FreeCAD jobs plus the record skeleton of each sample of one planned beam."""
    p: dp.IProfile = plan["profile"]
    L = plan["length"]
    rot: Rotation = plan["rotation"]
    quat = [rot.qx, rot.qy, rot.qz, rot.qw]
    slug = _norm(p.name)
    plates = local_plates(p, L)
    jobs = []
    for i, s in enumerate(plan["samples"]):
        index = first_index + i
        if s["kind"] == "ibeam":
            tag = "ibeam" if plan["fillet"] else "ibeamnf"
            job = dict(kind="ibeam", profile=dict(asdict(p), toe_radius=dp.toe_radius(p)), length=L,
                       fillet=plan["fillet"], mitre=plan["mitre"])
        elif s["kind"] == "plate":
            tag = s["plate"]
            c, half = plates[s["plate"]]
            job = dict(kind="box", center=c.tolist(), half_extents=half.tolist())
        else:
            tag = "envelope"
            job = dict(kind="box", center=[0.0, 0.0, 0.0], half_extents=[p.b / 2, p.h / 2, L / 2])
        stl = box_dir / f"box_i{index:07d}_{tag}_{slug}.stl"
        job.update(id=index, quaternion=quat, stl=str(stl), deflection=0.01, grid_mm=grid_mm,
                   size_fraction=s["size_fraction"], center_unit=s["center_unit"])
        job["_sample"] = s
        jobs.append(job)
    return jobs


def make_record(plan: Dict, job: Dict, result: Dict, seed: int, resolution: int = 64) -> Dict:
    """The record handed to finish_sample (it adds stl_path, voxel_path, voxels, field, labels)."""
    from src import sdf_primitives as sp
    p: dp.IProfile = plan["profile"]
    L = plan["length"]
    rot: Rotation = plan["rotation"]
    quat = [rot.qx, rot.qy, rot.qz, rot.qw]
    R = quat_matrix(quat)
    s = job["_sample"]
    k, T = float(result["scale_factor"]), np.asarray(result["translation_mm"], float)
    scale_mm = p.h / 4.0                         # the unit of the *_ratio parameters of all samples of this beam
    v = GRID_MM / resolution
    centre, extent = result["center_mm"], result["extent_mm"]
    placement = dict(mode="fixed", grid_mm=GRID_MM, size_fraction=s["size_fraction"], scale_factor=k,
                     center_mm=centre, extent_mm=extent, translation_mm=T.tolist(),
                     center_vox=[resolution / 2 + centre[2] / v, resolution / 2 - centre[1] / v, resolution / 2 + centre[0] / v],
                     extent_vox=[extent[2] / v, extent[1] / v, extent[0] / v],
                     freecad_volume_mm3=result["volume_mm3"], thin_ok=bool(s["thin_ok"]), cuts=[])
    if s["kind"] == "ibeam":
        placement.update(tw_vox=k * p.tw / v, tf_vox=k * p.tf / v)
    else:
        half_ = local_plates(p, L)[s["plate"]][1] if s["kind"] == "plate" else np.array([p.b, p.h, L]) / 2
        placement["thickness_vox"] = float(2 * k * half_.min() / v)
    record = dict(index=job["id"], mode="type", topology="box", rotation=asdict(rot), scale_mm=scale_mm, seed=seed,
                  beam_id=plan["beam_id"], placement=placement,
                  ibeam=dict(profile=p.name, series=p.series, standard=p.standard, length_mm=L,
                             end_cut="mitre" if plan["mitre"] else "square", mitre=plan["mitre"]),
                  freecad=dict(volume_mm3=result["volume_mm3"], local_volume_mm3=result["local_volume_mm3"],
                               valid=result["valid"], n_solids=result["n_solids"], facets=result["facets"],
                               builder="src/ibeam_freecad.py"))
    if s["kind"] == "ibeam":
        record.update(subclass="ibeam", fillet=bool(plan["fillet"]), parameters=profile_parameters(p, L, plan["fillet"], plan["mitre"]),
                      boxes=world_boxes(p, L, quat, k, T), boxes_exact=plan["mitre"] is None and p.flange_slope == 0)
        half = np.array([p.b / 2, p.h / 2, L / 2])
        record["_shape"] = dict(prim=sp.Box(half=(k * half).tolist()), R=R, t=T.tolist())        # labels: the envelope
    else:
        if s["kind"] == "plate":
            c, half = local_plates(p, L)[s["plate"]]
            record.update(source="ibeam_plate", role="web" if s["plate"] == "web" else "flange", plate=s["plate"])
        else:
            c, half = np.zeros(3), np.array([p.b / 2, p.h / 2, L / 2])
            record.update(source="ibeam_envelope")
        record.update(subclass="full", parameters=dict(
            x_half_ratio=float(half[0] / scale_mm), y_half_ratio=float(half[1] / scale_mm),
            z_half_ratio=float(half[2] / scale_mm), aspect_max_min=float(half.max() / half.min())))
        record["_shape"] = dict(prim=sp.Box(half=(k * half).tolist()), R=R, t=(k * (R @ c) + T).tolist())
    return record


# ------------------------------------------------------------------------------------------------ FreeCAD

def run_freecad(jobs: List[Dict], freecad: Optional[str] = None, workdir: Optional[Path] = None) -> List[Dict]:
    """Build every job in ONE freecadcmd process; returns the per-job results (ok / error, volume, placement).
    `freecad`: None - src.freecad_path's rule ($SHAPE_RECO_FREECADCMD, PATH, /usr/bin/freecadcmd)."""
    from src.freecad_path import freecadcmd_or_default
    freecad = freecad or freecadcmd_or_default()
    with tempfile.TemporaryDirectory(prefix="ibeam_", dir=workdir) as tmp:
        job_path, result_path = Path(tmp) / "jobs.json", Path(tmp) / "results.json"
        job_path.write_text(json.dumps([{k: v for k, v in j.items() if not k.startswith("_")} for j in jobs]))
        env = clean_subprocess_env()
        env["IBEAM_JOB_JSON"], env["IBEAM_RESULT_JSON"] = str(job_path), str(result_path)
        script = str(FREECAD_SCRIPT)
        code = f"exec(compile(open({script!r}, encoding='utf-8').read(), {script!r}, 'exec'))"
        proc = subprocess.run([freecad, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"freecadcmd failed ({proc.returncode}):\n{proc.stdout[-3000:]}")
        return json.loads(result_path.read_text())


# ------------------------------------------------------------------------------------------------ driver

def generate(out: Path, profiles: List[dp.IProfile], n_per_profile: int, seed: int, shard: int = 0, *,
             freecad: Optional[str] = None, resolution: int = 64, with_field: bool = True,
             finish: Optional[Callable] = None, run: Callable = run_freecad, log=print, keep_thin_misses: bool = False,
             noise: Optional[Dict] = None, **plan_kwargs) -> Dict:
    """Plan -> build (one freecadcmd per profile) -> finish_sample per sample. Returns counts and timings.
    Samples whose thinnest wall cannot reach min_thin_vox voxels are skipped (counted in "skipped_thin") unless
    keep_thin_misses: below about one voxel the web breaks into pieces and the sample is no longer one manifold."""
    if finish is None:
        from src.sample_writer import finish_sample as finish
    out_root = Path(out) / f"shard{shard}"
    box_dir = out_root / "box"
    box_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed * 1000 + shard)
    index = shard * 1_000_000
    counts: Dict[str, int] = {}
    failures: List[Dict] = []
    skipped: Dict[str, int] = {}
    t_fc = t_fin = 0.0
    for profile in profiles:
        plans, jobs = [], []
        for b in range(n_per_profile):
            plan = plan_beam(rng, profile, f"s{seed}_k{shard}_{_norm(profile.name)}_{b}", resolution=resolution,
                             **plan_kwargs)
            if not keep_thin_misses:            # a wall below min_thin_vox breaks into voxel pieces: not one manifold
                for s in plan["samples"]:
                    if not s["thin_ok"]:
                        skipped[s.get("plate") or s["kind"]] = skipped.get(s.get("plate") or s["kind"], 0) + 1
                plan["samples"] = [s for s in plan["samples"] if s["thin_ok"]]
            pj = build_jobs(plan, box_dir, index)
            index += len(pj)
            plans.append(plan)
            jobs.extend((plan, j) for j in pj)
        t0 = time.time()
        if not jobs:
            log(f"{profile.name}: no sample reaches the minimum wall thickness, skipped")
            continue
        results = {r["id"]: r for r in run([j for _, j in jobs], freecad)}
        t_fc += time.time() - t0
        for plan, job in jobs:
            res = results.get(job["id"])
            if not res or not res.get("ok") or res.get("n_solids") != 1 or not res.get("valid"):
                failures.append(dict(id=job["id"], profile=profile.name, kind=job["_sample"]["kind"],
                                     error=(res or {}).get("error", "invalid or multi-solid")))
                continue
            t0 = time.time()
            rec = make_record(plan, job, res, seed, resolution)
            finish(Path(job["stl"]), rec, out_root, resolution=resolution, with_field=with_field,
                   **({"noise": noise} if noise else {}))
            t_fin += time.time() - t0
            key = rec["subclass"] if rec["subclass"] == "ibeam" else rec["source"]
            counts[key] = counts.get(key, 0) + 1
        log(f"{profile.name}: {len(jobs)} samples, FreeCAD {t_fc:.1f} s, finish {t_fin:.1f} s cumulative")
    n = sum(counts.values())
    return dict(counts=counts, skipped_thin=skipped, failures=failures, freecad_s=t_fc, finish_s=t_fin, samples=n)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="dataset root; samples go to <out>/shard<K>/box/")
    ap.add_argument("--n-per-profile", type=int, default=1, help="beams per profile")
    ap.add_argument("--profiles", default="default",
                    help='"default" (20 spread over the series), "all", or a comma list of names/series ("HEA 500,IPE")')
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--freecad", default=None, help="default: $SHAPE_RECO_FREECADCMD, then PATH, then /usr/bin/freecadcmd")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--no-field", action="store_true", help="skip the .field.npz")
    ap.add_argument("--nofillet-prob", type=float, default=0.2)
    ap.add_argument("--mitre-prob", type=float, default=0.25)
    ap.add_argument("--plate-fraction", type=float, default=1.0, help="probability each of the 3 plates becomes a sample")
    ap.add_argument("--envelope-fraction", type=float, default=1.0)
    ap.add_argument("--size-fraction", type=float, nargs=2, default=(0.5, 0.95))
    ap.add_argument("--min-thin-vox", type=float, default=1.2, help="thinnest wall in voxels (see plan_beam)")
    ap.add_argument("--keep-thin-misses", action="store_true",
                    help="also write samples whose thinnest wall stays below --min-thin-vox (flagged placement.thin_ok false)")
    ap.add_argument("--length-range", type=float, nargs=2, default=(0.5, 2.0), help="L / h")
    from src.voxel_noise import add_noise_arguments, noise_spec_from_args
    add_noise_arguments(ap)
    args = ap.parse_args(argv)
    profiles = resolve_profiles(args.profiles)
    t0 = time.time()
    summary = generate(args.out, profiles, args.n_per_profile, args.seed, args.shard, freecad=args.freecad,
                       resolution=args.resolution, with_field=not args.no_field, nofillet_prob=args.nofillet_prob,
                       mitre_prob=args.mitre_prob, plate_fraction=args.plate_fraction,
                       envelope_fraction=args.envelope_fraction, size_fraction=tuple(args.size_fraction),
                       length_range=tuple(args.length_range), min_thin_vox=args.min_thin_vox,
                       keep_thin_misses=args.keep_thin_misses, noise=noise_spec_from_args(args))
    summary["wall_s"] = time.time() - t0
    print(json.dumps(summary, indent=1))
    if summary["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
