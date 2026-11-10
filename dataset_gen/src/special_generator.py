#!/usr/bin/env python3
"""
Host driver of the special-shape subclasses: hollow (all five parents), tube (cylinder), rhs (box) and frame (all five
parents). Every solid is built by FreeCAD (dataset_gen/src/special_freecad.py under freecadcmd: Part solids, Part.cut,
multiFuse, exported to STL already placed in the in-scale grid); this side draws the random numbers, then hands the STL
to src/sample_writer.finish_sample, which voxelises it, writes the .field.npz and appends the dataset_metadata.jsonl
record (training/manifest.py reads it).

    python3 -m src.special_generator --out local_dataset_v5/small/train --subclasses box/frame cylinder/tube \
        --n-per-subclass 20 --seed 1000 --shard 0 --workers 8          (run from dataset_gen/)

Output: <out>/shard<K>/<parent>/<subclass>_<index>.stl (+ .bin, .field.npz) and <out>/shard<K>/<parent>/
dataset_metadata.jsonl. Records carry topology = parent, subclass, sample_kind "special", the parent's dims_mm /
center_mm / rotation_canonical (the OUTER body; a frame's labels describe the tessellated parent), the FreeCAD volume
and check, and per subclass: hollow / tube / rhs `wall_mm` (+ in voxels); frame `member_kind`, `tessellation_tol`,
`n_members`, `members` (world endpoints, section kind and dims per strut).

Randomness: one random.Random per sample, seeded from (seed, shard, subclass, i), so a sample does not depend on the
worker count or order; disjoint --seed values per split give disjoint data. Frame member kinds cycle through
MEMBER_KINDS by the sample's index (offset by the parent), so every n consecutive frames are balanced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

SRC = Path(__file__).resolve().parent
FREECAD_SCRIPT = SRC / "special_freecad.py"
if str(SRC.parent) not in sys.path:
    sys.path.insert(0, str(SRC.parent))

from src.dataset_utils import quaternion_to_euler_xyz, random_unit_quaternion  # noqa: E402

PARENTS = ("box", "cylinder", "cone", "sphere", "torus")
SUBCLASSES = {                                 # the fixed label vocabulary (= sample_writer.SUBCLASSES)
    "box": ("full", "hollow", "ibeam", "rhs", "frame"),
    "cylinder": ("full", "sector", "hollow", "tube", "frame"),
    "cone": ("full", "sector", "hollow", "frame"),
    "sphere": ("full", "sector", "band", "hollow", "frame"),
    "torus": ("full", "sector", "band", "hollow", "frame"),
}
OWNED = tuple([f"{p}/hollow" for p in PARENTS] + ["cylinder/tube", "box/rhs"] + [f"{p}/frame" for p in PARENTS])
MEMBER_KINDS = ("rod", "bar", "tube", "rhs", "ibeam")
NOMINAL_MM = 20.0                              # parents are drawn at this size, FreeCAD scales them into the grid
MAX_EDGES: Dict[str, int] = {}                 # per parent override (the torus uses a u/v grid, 3 n_u n_v edges <= 60)
DEFAULT_MAX_EDGES = 60
MIN_STRUT_VOXELS = dict(rod=3.0, bar=3.0, tube=4.0, rhs=4.0, ibeam=4.0)   # member size in voxels of the placed grid
MIN_MEMBER_WALL_VOXELS = 1.0                   # tube / rhs member wall (capped at 0.35 of the member size)


# ------------------------------------------------------------------------------------------------ draws (pure python)

def parse_subclass(spec: str) -> Tuple[str, str]:
    parent, _, sub = spec.partition("/")
    if parent not in SUBCLASSES or sub not in SUBCLASSES[parent]:
        raise ValueError(f"unknown subclass {spec!r}; the vocabulary is {SUBCLASSES}")
    if spec not in OWNED:
        raise ValueError(f"{spec!r} is not built by this generator (it builds {OWNED})")
    return parent, sub


def member_kind_for(parent: str, index: int) -> str:
    """Balanced member kinds: cycle by the sample's global index, offset by the parent (so one frame per parent in a
    smoke run already covers all five kinds)."""
    return MEMBER_KINDS[(index + PARENTS.index(parent)) % len(MEMBER_KINDS)]


def draw_dims(rng: random.Random, parent: str, sub: str) -> Dict:
    """Parent dimensions in mm at NOMINAL_MM. Aspect ratios bounded (<= 4) so a hollow wall or a strut stays resolvable
    after the whole part is scaled into the grid; rhs has its long axis on z (the open direction)."""
    S = NOMINAL_MM
    if parent == "box":
        if sub == "rhs":
            hx, hy = rng.uniform(0.3, 0.8) * S, rng.uniform(0.3, 0.8) * S
            return dict(half=[hx, hy, rng.uniform(1.3, 3.0) * max(hx, hy)])
        lo = 0.4 if sub == "frame" else 0.3
        return dict(half=[rng.uniform(lo, 1.0) * S for _ in range(3)])
    if parent == "sphere":
        return dict(radius=S)
    if parent == "cylinder":
        r = rng.uniform(0.35, 1.0) * S
        return dict(radius=r, height=rng.uniform(max(0.6, 0.5 if sub != "tube" else 1.0), 3.0) * r)
    if parent == "cone":
        big = rng.uniform(0.5, 1.0) * S
        small = 0.0 if (sub != "hollow" and rng.random() < 0.25) else rng.uniform(0.2, 0.7) * big
        r1, r2 = (big, small) if rng.random() < 0.5 else (small, big)
        return dict(radius1=r1, radius2=r2, height=rng.uniform(0.7, 2.5) * big)
    if parent == "torus":
        major = rng.uniform(0.6, 1.0) * S
        return dict(major=major, minor=rng.uniform(0.25, 0.6) * major)
    raise ValueError(parent)


def draw_frame(rng: random.Random, parent: str, member_kind: str) -> Dict:
    from src import din_profiles
    fr = dict(member_kind=member_kind,
              tess_tol_frac=rng.uniform(0.10, 0.25),
              tess_angular=rng.uniform(1.2, 2.2),
              max_edges=MAX_EDGES.get(parent, DEFAULT_MAX_EDGES),
              # strut thickness >= 0.05 of the part's extent (>= 3 voxels at 64^3 when the part fills the grid); an I-beam's
              # size is its depth h, drawn larger so its flanges are resolvable
              thickness_frac=rng.uniform(0.08, 0.12) if member_kind == "ibeam" else rng.uniform(0.05, 0.09),
              aspect=rng.uniform(1.0, 1.6), member_wall_frac=rng.uniform(0.12, 0.2),
              node_spheres=rng.random() < 0.5, node_scale=rng.uniform(1.2, 1.6),
              fillet=rng.random() < 0.5)          # I-beam root fillets: half the samples (they triple the fuse time)
    if member_kind == "ibeam":
        names = sorted(din_profiles.PROFILES)
        p = din_profiles.PROFILES[rng.choice(names)]
        fr["ibeam_profile"] = asdict(p)
    return fr


def quat_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sample_rng(seed: int, shard: int, spec: str, i: int, redraw: int = 0) -> random.Random:
    return random.Random(f"special:{seed}:{shard}:{spec}:{i}:{redraw}")


def make_job(rng: random.Random, parent: str, sub: str, index: int, stl_out: Path, grid_mm: float,
             size_fraction: Tuple[float, float], resolution: int) -> Dict:
    """Everything FreeCAD needs for one sample (all random draws happen here)."""
    q = random_unit_quaternion(rng)
    job = dict(id=index, parent=parent, subclass=sub, dims=draw_dims(rng, parent, sub), quaternion=q,
               R=quat_to_matrix(q).tolist(),
               placement=dict(grid_mm=grid_mm, size_fraction=rng.uniform(*size_fraction),
                              pos_unit=[rng.random() for _ in range(3)]),
               stl_out=str(stl_out), deflection=0.05, max_attempts=4,
               min_wall_mm=1.5 * grid_mm / resolution)
    if sub == "frame":
        job["frame"] = draw_frame(rng, parent, member_kind_for(parent, index))
        job["frame"].update(voxel_mm=grid_mm / resolution, min_strut_voxels=MIN_STRUT_VOXELS,
                            min_member_wall_voxels=MIN_MEMBER_WALL_VOXELS)
        job["angular_deflection"] = 0.3
    else:
        job["wall_frac"] = rng.uniform(0.05, 0.20)
    return job


# ------------------------------------------------------------------------------------------------ labels / record

def parent_primitive(parent: str, dims: Dict, k: float):
    """The OUTER body as an sdf_primitives primitive at world size (for sample_writer's label derivation)."""
    from src import sdf_primitives as sp
    if parent == "box":
        return sp.Box([h * k for h in dims["half"]])
    if parent == "sphere":
        return sp.Sphere(dims["radius"] * k)
    if parent == "cylinder":
        return sp.Cylinder(dims["radius"] * k, dims["height"] * k)
    if parent == "cone":
        return sp.Cone(dims["radius1"] * k, dims["radius2"] * k, dims["height"] * k)
    if parent == "torus":
        return sp.Torus(dims["major"] * k, dims["minor"] * k)
    raise ValueError(parent)


def build_record(job: Dict, res: Dict, seed: int, shard: int, resolution: int, redraws: int,
                 with_shape: bool = True) -> Dict:
    """The manifest record (before sample_writer adds stl/voxel paths, voxel counts and the field)."""
    parent, sub, dims = job["parent"], job["subclass"], job["dims"]
    q = job["quaternion"]
    rx, ry, rz = quaternion_to_euler_xyz(q)
    k = float(res["scale_factor"])
    g = float(job["placement"]["grid_mm"])
    v = g / resolution
    c, e = res["center_mm"], res["extent_mm"]
    info = res.get("info", {})
    placement = dict(mode="fixed", grid_mm=g, size_fraction=job["placement"]["size_fraction"], scale_factor=k,
                     center_mm=c, extent_mm=e, stl_placed=True, cuts=[],
                     center_vox=[resolution / 2 + c[2] / v, resolution / 2 - c[1] / v, resolution / 2 + c[0] / v],
                     extent_vox=[e[2] / v, e[1] / v, e[0] / v],
                     solid=dict(volume_mm3=res["volume_mm3"], n_solids=res["n_solids"], n_shells=res["n_shells"],
                                valid=res["valid"], mesh_closed=res.get("mesh_closed")))
    params = {"nominal_mm": NOMINAL_MM, **{key: (val if not isinstance(val, list) else list(val)) for key, val in dims.items()}}
    record = dict(index=job["id"], mode="type", topology=parent, subclass=sub, sample_kind="special",
                  source="special_generator", parameters=params,
                  rotation=dict(rx=rx, ry=ry, rz=rz, qx=q[0], qy=q[1], qz=q[2], qw=q[3]), scale_mm=NOMINAL_MM,
                  seed=seed, shard=shard, placement=placement, freecad_volume_mm3=res["volume_mm3"],
                  freecad=dict(attempts=res.get("attempts", []), n_attempts=res.get("n_attempts", 1), host_redraws=redraws,
                               seconds_build=res.get("seconds_build"), seconds_export=res.get("seconds_export"),
                               n_facets=res.get("n_facets")))
    if sub == "frame":
        record["thickness_voxels"] = (info.get("thickness_mm") or 0.0) / v
        for key in ("member_kind", "tessellation_tol", "tessellation_tol_frac", "tessellation_angular",
                    "tessellation_method", "tessellation_grid", "fillet", "build_attempt", "n_members",
                    "n_vertices", "members", "thickness_mm", "node_spheres", "node_radius", "ibeam_builder"):
            record[key] = info.get(key)
        if info.get("member_kind") == "ibeam":
            record["ibeam_profile"] = job["frame"]["ibeam_profile"]["name"]
    else:
        record["wall_mm"] = info.get("wall_mm")
        record["wall_voxels"] = info.get("wall_mm", 0.0) / v
        record["wall_frac"] = info.get("wall_frac_effective")
    if with_shape:
        prim = parent_primitive(parent, dims, k)
        R = np.asarray(job["R"], float)
        centre = np.asarray(res["parent_center_mm"], float)
        if parent in ("cylinder", "cone"):             # sdf cylinder / cone stand on z = 0; ours are centred
            centre = centre - R @ np.array([0.0, 0.0, dims["height"] * k / 2])
        record["_shape"] = dict(prim=prim, R=R, t=centre)
    return record


# ------------------------------------------------------------------------------------------------ FreeCAD call

def run_freecad(jobs: List[Dict], freecad: Optional[str] = None, timeout: float = 300) -> List[Dict]:
    """Run special_freecad.py on `jobs` in one freecadcmd process; returns its per-job results."""
    from src.closest_point_field import find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("freecadcmd not found")
    with tempfile.TemporaryDirectory(prefix="special_") as tmp:
        jp, rp = Path(tmp) / "jobs.json", Path(tmp) / "results.json"
        jp.write_text(json.dumps(jobs))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env.update(SPECIAL_JOB_JSON=str(jp), SPECIAL_RESULT_JSON=str(rp), SPECIAL_SRC_DIR=str(SRC))
        try:
            if os.environ.get("SHAPE_RECO_FREECAD_WORKER") == "1":     # persistent freecadcmd (reco_field/freecad_worker.py)
                import types
                repo = str(Path(__file__).resolve().parent.parent.parent)
                if repo not in sys.path:
                    sys.path.insert(0, repo)
                from reco_field import freecad_worker
                rc, out = freecad_worker.run_script(freecad, FREECAD_SCRIPT, env, timeout=timeout)
                if rc == -9:
                    raise subprocess.TimeoutExpired([freecad, str(FREECAD_SCRIPT)], timeout)
                proc = types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
            else:
                proc = subprocess.run([freecad, str(FREECAD_SCRIPT)], capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return [dict(id=j.get("id"), ok=False, error=f"freecadcmd timeout ({timeout:.0f} s)") for j in jobs]
        if not rp.is_file():
            raise RuntimeError(f"freecadcmd produced no result (exit {proc.returncode}): {proc.stdout[-2000:]} {proc.stderr[-2000:]}")
        return json.loads(rp.read_text())


# ------------------------------------------------------------------------------------------------ one sample

def default_finish_sample() -> Callable:
    from src.sample_writer import finish_sample
    return finish_sample


def make_sample(spec: str, i: int, index: int, out_root: Path, seed: int, shard: int, grid_mm: float = 40.0,
                size_fraction=(0.5, 0.9), resolution: int = 64, with_field: bool = True, field_points: int = 2048,
                max_redraws: int = 3, freecad: Optional[str] = None, builder: Optional[Callable] = None,
                finish: Optional[Callable] = None, noise: Optional[Dict] = None) -> Dict:
    """Build, register and time one sample. `builder(jobs) -> results` and `finish(stl, record, out_root, ...)` default to
    freecadcmd and sample_writer.finish_sample (tests stub them). A failed FreeCAD build is redrawn (new parameters) up to
    `max_redraws` times. Returns {record, timing, failures}."""
    parent, sub = parse_subclass(spec)
    builder = builder or (lambda jobs: run_freecad(jobs, freecad))
    finish = finish or default_finish_sample()
    stl = Path(out_root) / parent / f"{sub}_{index:06d}.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    t0 = time.time()
    for redraw in range(max_redraws + 1):
        rng = sample_rng(seed, shard, spec, i, redraw)
        job = make_job(rng, parent, sub, index, stl, grid_mm, size_fraction, resolution)
        res = builder([job])[0]
        if res.get("ok"):
            break
        failures.append(res.get("error") or "unknown")
    else:
        raise RuntimeError(f"{spec} #{i}: FreeCAD failed {len(failures)} times: {failures}")
    t_freecad = time.time() - t0
    record = build_record(job, res, seed, shard, resolution, redraw)
    record["freecad"]["failed_draws"] = failures
    t1 = time.time()
    written = finish(stl, record, out_root, resolution=resolution, with_field=with_field, field_points=field_points,
                     **({"noise": noise} if noise else {}))
    t_finish = time.time() - t1
    t_field = float(((written.get("placement") or {}).get("field") or {}).get("seconds") or 0.0)
    timing = dict(spec=spec, freecad=t_freecad, freecad_build=res.get("seconds_build"), voxels=t_finish - t_field,
                  field=t_field, total=time.time() - t0, fc_attempts=res.get("n_attempts", 1), redraws=redraw)
    return dict(record=written, timing=timing, failures=failures)


def _worker(args):
    try:
        return make_sample(*args[0], **args[1])
    except Exception as exc:                    # reported, the run goes on
        return dict(error=f"{type(exc).__name__}: {exc}", spec=args[0][0], i=args[0][1])


def generate(out: Path, specs: List[str], n_per_subclass: int, seed: int, shard: int = 0, workers: int = 1,
             **kw) -> List[Dict]:
    out_root = Path(out) / f"shard{shard}"
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    for s_i, spec in enumerate(specs):
        parse_subclass(spec)
        for i in range(n_per_subclass):
            gi = shard * n_per_subclass + i                              # global index within the subclass: balance key
            index = (s_i * 1_000_000) + gi
            tasks.append(((spec, gi, index, out_root, seed, shard), kw))
    results = []
    if workers <= 1:
        for t in tasks:
            results.append(_worker(t))
            _progress(results[-1], len(results), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for fut in as_completed([pool.submit(_worker, t) for t in tasks]):
                results.append(fut.result())
                _progress(results[-1], len(results), len(tasks))
    timings = [r["timing"] for r in results if "timing" in r]
    (out_root / "special_timing.json").write_text(json.dumps(dict(timings=timings, summary=summarise(timings),
                                                                  errors=[r for r in results if "error" in r]), indent=1))
    return results


def _progress(r: Dict, done: int, total: int) -> None:
    if "error" in r:
        print(f"[{done}/{total}] {r['spec']} #{r['i']}: ERROR {r['error']}", flush=True)
    else:
        t = r["timing"]
        print(f"[{done}/{total}] {t['spec']}: freecad {t['freecad']:.1f}s voxels {t['voxels']:.1f}s field {t['field']:.1f}s"
              f" (attempts {t['fc_attempts']}, redraws {t['redraws']})", flush=True)


def summarise(timings: List[Dict]) -> Dict:
    out = {}
    for t in timings:
        out.setdefault(t["spec"], []).append(t)
    return {spec: {key: round(float(np.mean([t[key] for t in ts])), 2) for key in ("freecad", "voxels", "field", "total")}
            | {"n": len(ts)} for spec, ts in out.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="split directory; samples go to <out>/shard<K>/<parent>/")
    ap.add_argument("--subclasses", nargs="+", default=list(OWNED), help=f"parent/subclass list (default all: {OWNED})")
    ap.add_argument("--n-per-subclass", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="use disjoint seeds per split")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--grid-mm", type=float, default=40.0)
    ap.add_argument("--size-fraction", type=float, nargs=2, default=[0.5, 0.9],
                    help="largest extent as a fraction of the grid (default 0.5-0.9: walls and struts must stay resolvable)")
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--field-points", type=int, default=2048)
    ap.add_argument("--no-field", action="store_true")
    ap.add_argument("--freecad", default=None)
    from src.voxel_noise import add_noise_arguments, noise_spec_from_args
    add_noise_arguments(ap)
    a = ap.parse_args(argv)
    res = generate(a.out, a.subclasses, a.n_per_subclass, a.seed, a.shard, a.workers, grid_mm=a.grid_mm,
                   size_fraction=tuple(a.size_fraction), resolution=a.resolution, with_field=not a.no_field,
                   field_points=a.field_points, freecad=a.freecad, noise=noise_spec_from_args(a))
    timings = [r["timing"] for r in res if "timing" in r]
    print(json.dumps(summarise(timings), indent=1))
    n_err = sum("error" in r for r in res)
    if n_err:
        print(f"{n_err} sample(s) failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
