#!/usr/bin/env python3
"""
The tree study on REAL trees (the user's request 2026-09-30: "real trees, not saplings - make it big"): every program
our reconstruction produced, written as the same solid in several boolean tree shapes and benchmarked in aegir's
straight-line scan, beside the synthetic stacked set of the aegir checkout (benchmarks/stacked, at most 32 leaves).

Sources (whatever exists):
  abc      ~/shape_reco_results/real_trees/abc/<run>/<part>/program.json   (fetched from server_a: the 352 known ABC
           parts with and without sub-parts, and the pilots; ReconstructionResult dicts)
  liner    the run store's liner runs (13 to 60 features, composites = hole stacks and tools)
  review   ~/shape_reco_results/frame_review_fix1/union/rNN_*/run.json      (15 generated frames, member programs)
  frame    ~/shape_reco_results/real_trees/frame40/result.json               (the 40-add frame, exact IoU 0.9655)
           and docs/data/frame_cascade_program.json                           (the 32-add frame of the first study)

Tree shapes (aegir's scripts/sandbox_geometry/make_gdml.py): --union as-is (the program's own tree: the loop's
left-deep fold with its composites), binary (every union a left-deep chain), balanced (halved recursively), multi
(G4MultiUnion of leaves) - each with --cuts as-is or grouped (the base minus the union of all its tools). Shapes whose
tree metrics equal an earlier shape of the same program are the same tree and are dropped.

Placement: the frame and the liner keep their own guns and placements of the first study; every other part is
centred on its bounds, turned so that its thinnest direction is the beam, and scaled uniformly into the 'study' gun's
volume (300 x 160 mm across, 100 mm along). A uniform scale leaves the tree and the relative cost of its shapes alone.
That rule ("bounds") stays wherever the gun sees the part; where a simulated validation scan (the gun's 20000 lines,
sphere-traced on the program's SDF) predicts no entry, the gun is aimed at the material instead ("material", T-33): the
bounds of points sampled in the solid give axis, scale and origin. "bounds" sees no entry for 14 of the 641 programs of
the first study (11 recorded none, 3 crash on a G4Torus before): its origin is make_gdml's default, the centre of EVERY
leaf's bounds (cut tools of metres included), and its scale came from balls around the added leaves (an intersection's
huge operand included). Each row records placement_rule and predicted_entries (the simulated scan, None: own gun).

    python3 scripts/real_tree_study.py [--aegir ~/Documents/Sandbox_aegir/aegir] [--only abc,liner,...] [--jobs 8]
                                       [--series REGEX --name realtree_t33]       # a rerun of a few, beside the study

Writes <aegir>/geometry/sandbox/realtree/<series>/*.gdml (+ .metrics.json), <aegir>/geometry/sandbox/realtree_index.json
(the rows the benchmark reads: geometry, part = the gun, series, shape, same_solid, the tree metrics, the source) and
~/shape_reco_results/real_trees/programs/<series>.json (the program trees). Benchmark on server_b:
INDEX=geometry/sandbox/realtree_index.json OUT=benchmarks/realtree/runs bash benchmarks/stacked/run_workers.sh
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]
RESULTS = Path.home() / "shape_reco_results"
WORK = RESULTS / "real_trees"
FORMS = [(u, c) for c in ("as-is", "grouped") for u in ("as-is", "binary", "balanced", "multi")]
STUDY_BOX = np.array([300.0, 160.0, 100.0])        # the 'study' gun: across x, across y, along the beam
BEAM_AXES = {"z": np.eye(3), "x": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
             "y": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)}          # as make_gdml.BEAM_AXES
STUDY_GUN = dict(vertex_z=-1000.0, max_theta=0.35)  # aegir's workflows/sandbox.libsonnet, guns.study
SCAN_EVENTS = 20000                                 # aegir's validation scan (run_study.py --validation-events)
FRAME = ["--beam-axis", "x", "--origin", "602", "6000", "-3548"]              # make_parts_index.FRAME
LINER = ["--beam-axis", "y", "--origin", "0", "-96", "808.35"]                # HANDOVER.md, the liner levels


# ------------------------------------------------------------------------------------------------ sources
def _geometry_only(p):
    """A primitive dict without what the geometry does not need (stage sources, alternatives; a node the run store
    recovered has none)."""
    if isinstance(p, dict):
        return {k: (_geometry_only(v) if k == "parts" else v) for k, v in p.items() if k not in ("sources", "alternatives")}
    if isinstance(p, list):
        return [_geometry_only(x) for x in p]
    return p


def _result(d: dict):
    from reco_engine.result_schema import ReconstructionResult
    ops = [dict(o, primitive=_geometry_only(o["primitive"])) for o in d.get("operations", [])]
    return ReconstructionResult.from_dict(dict(d, operations=ops))


def _flat_ops(ops: list) -> list:
    """The review frames' flat operations (kind, center_mm, ... at the top) as schema operations."""
    keys = ("kind", "quaternion", "center_mm", "dims_mm", "label")
    return [dict(op=o["op"], primitive={k: o[k] for k in keys if k in o}) for o in ops]


def sources(only):
    """(series, gun, extra make_gdml args or None, program dict or result, facts) for every real program found."""
    from reco_engine.result_to_program import program_dict
    out = []
    if "abc" in only:
        for prog in sorted((WORK / "abc").glob("*/*/program.json")):
            run, part = prog.parent.parent.name, prog.parent.name
            d = json.loads(prog.read_text())
            if not d.get("operations"):
                continue
            series = f"{run.replace('abc_', '')}_{part.split('_')[0]}_{part.split('_')[-1]}"
            out.append((series, "study", None, program_dict(_result(d)),
                        dict(source=f"abc/{run}/{part}", ops=len(d["operations"]), exact_iou=d.get("exact_iou"))))
    if "liner" in only:
        from reco_engine import run_store
        seen = set()
        for rj in sorted(RESULTS.glob("Liner_LINER_001*/*/run.json")):
            d = run_store.load_run(rj.parent)             # a program cut when it was saved: its recovery, if written
            ops = d.get("program") or []
            if len(ops) < 2 or len(ops) in seen or not d.get("finished", True):
                continue
            if run_store.truncated(ops):
                print(f"skip {rj.parent.name}: the run store cut its program (scripts/recover_run_program.py)", file=sys.stderr)
                continue
            seen.add(len(ops))
            try:
                prog = program_dict(_result(dict(input_path=d.get("input", {}).get("path", ""), operations=ops)))
            except Exception as exc:                          # a run of an older schema: say so, go on
                print(f"skip {rj.parent.name}: {exc}", file=sys.stderr)
                continue
            out.append((f"liner_{len(ops):03d}ops", "liner", LINER, prog,
                        dict(source=str(rj.parent), ops=len(ops), exact_iou=(d.get("metrics") or {}).get("exact_iou"))))
    if "review" in only:
        for rj in sorted((RESULTS / "frame_review_fix1" / "union").glob("r*/run.json")):
            d = json.loads(rj.read_text())
            if not d.get("operations"):
                continue
            prog = program_dict(_result(dict(input_path=d.get("name", ""), operations=_flat_ops(d["operations"]))))
            out.append((f"review_{rj.parent.name}", "study", None, prog,
                        dict(source=str(rj.parent), ops=len(d["operations"]))))
    if "frame" in only:
        f40 = WORK / "frame40" / "result.json"
        if f40.exists():
            d = json.loads(f40.read_text())
            out.append(("frame_40adds", "frame", FRAME, program_dict(_result(d)),
                        dict(source=str(f40), ops=len(d["operations"]), exact_iou=d.get("exact_iou"))))
        f32 = ROOT / "docs" / "data" / "frame_cascade_program.json"
        out.append(("frame_32adds", "frame", FRAME, json.loads(f32.read_text()), dict(source=str(f32), ops=32)))
    return out


# ------------------------------------------------------------------------------------------------ placement
def _size(prim: dict) -> float:
    """The largest distance of a primitive's material from its placement point, from its size fields."""
    vals = []
    for k, v in prim.items():
        if k in ("type", "u", "v", "u0", "u1", "v0", "v1"):
            continue
        if isinstance(v, (int, float)):
            vals.append(abs(float(v)))
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            vals.append(float(np.linalg.norm(v)))
    return float(sum(vals)) if vals else 1.0


def _bounds(prog: dict):
    """Rough bounds of the program's ADDED material: every placed leaf that is not under the right operand of a
    subtraction, as a ball of its translation and its size (enough to choose the beam axis and the scale)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)

    def walk(n, cut=False):
        nonlocal lo, hi
        if not isinstance(n, dict):
            return
        if "a" in n and "b" in n:
            walk(n["a"], cut)
            walk(n["b"], cut or n.get("type") == "Subtraction")
            return
        if cut or n.get("type") != "Placed" or not isinstance(n.get("primitive"), dict):
            return
        if n["primitive"].get("type") in ("HalfSpace", "Halfspace", "Plane"):
            return
        c, r = np.asarray(n.get("translation", (0, 0, 0)), float), _size(n["primitive"])
        lo, hi = np.minimum(lo, c - r), np.maximum(hi, c + r)
    walk(prog)
    return lo, hi


def placement_bounds(prog: dict):
    """make_gdml args for a part on the 'study' gun: beam along its thinnest direction, scaled into the gun's box."""
    lo, hi = _bounds(prog)
    ext = np.maximum(hi - lo, 1e-6)
    if not np.all(np.isfinite(ext)):
        return ["--beam-axis", "z"]
    axis = "xyz"[int(np.argmin(ext))]
    world = np.abs(BEAM_AXES[axis]) @ ext
    scale = float(np.min(STUDY_BOX / world))
    return ["--beam-axis", axis, "--scale", f"{scale:.6g}"]


def _model(d: dict):
    """The program dict as an SDF tree of dataset_gen/src/sdf_primitives (the inverse of
    reco_field.export.program_to_dict)."""
    from src import sdf_primitives as sp
    t = d["type"]
    if t == "Placed":
        return sp.Placed(_model(d["primitive"]), np.asarray(d["rotation"], float), np.asarray(d["translation"], float))
    if t in ("Union", "Intersection", "Subtraction"):
        return getattr(sp, t)(_model(d["a"]), _model(d["b"]))
    keys = dict(Box=("half",), Cylinder=("radius", "height", "u", "inner_radius"),
                Cone=("radius1", "radius2", "height", "u"), Sphere=("radius", "u", "v", "inner_radius", "latitude"),
                Torus=("major", "minor", "u", "v"), HalfSpace=("normal", "offset"))[t]
    return getattr(sp, t)(**{k: tuple(d[k]) if k in ("u", "v") else d[k] for k in keys if k in d})


def _material(prog: dict, n: int = 4096, want: int = 1000, rounds: int = 4, budget: int = 2_000_000) -> np.ndarray:
    """Points of the program's SOLID (source frame): n random points in the box of every leaf that adds material (not
    under the right operand of a subtraction), kept where the whole program is solid; every point of the solid lies in
    such a leaf. Random, not a grid: a regular grid can step over a thin wall that a cut left (a 1 mm tube of 100 mm
    found no grid point at 12^3). Fewer than `want` points found: again with 8 times as many (at most `budget` points
    a round), up to `rounds` times."""
    boxes = []

    def walk(d, R, t, cut):
        if d["type"] == "Placed":
            walk(d["primitive"], R @ np.asarray(d["rotation"], float), R @ np.asarray(d["translation"], float) + t, cut)
        elif d["type"] in ("Union", "Intersection", "Subtraction"):
            walk(d["a"], R, t, cut)
            walk(d["b"], R, t, cut or d["type"] == "Subtraction")
        elif not cut and d["type"] != "HalfSpace":
            boxes.append(_model(d).bounds() + (R, t))
    walk(prog, np.eye(3), np.zeros(3), False)
    model, rng, found = _model(prog), np.random.default_rng(0), np.zeros((0, 3))
    for k in range(rounds):
        m = min(n * 8 ** k, budget // max(len(boxes), 1))
        p = np.concatenate([rng.uniform(lo, hi, (m, 3)) @ R.T + t for lo, hi, R, t in boxes] or [found])
        found = np.concatenate([found, p[model.sdf(p) <= 0]])
        if len(found) >= want or not boxes or m < n * 8 ** k:
            break
    return found


def placement_material(points: np.ndarray):
    """make_gdml args that aim the 'study' gun at the material: the bounds of the solid's points (not of the leaves'
    balls, not of the cut tools) give the beam axis (the thinnest direction), the scale (into the gun's box) and the
    origin (their centre, so the whole solid lies in the gun's box)."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    axis = "xyz"[int(np.argmin(ext))]
    scale = float(np.min(STUDY_BOX / (np.abs(BEAM_AXES[axis]) @ ext)))
    return ["--beam-axis", axis, "--scale", f"{scale:.6g}", "--origin"] + [f"{x:.6g}" for x in scale * 0.5 * (lo + hi)]


def scan_entries(prog: dict, args: list, gdml, points: np.ndarray, rays: int = SCAN_EVENTS, seed: int = 1) -> int:
    """How many of `rays` lines of the 'study' gun enter the solid as make_gdml places it with `args`: the events with
    an entry that aegir's validation scan should record. The lines are drawn as aegir's particle_gun_source draws them
    (from (0, 0, vertex_z), theta uniform up to max_theta, phi uniform) and sphere-traced on the program's SDF, a lower
    bound of the distance outside a CSG tree, so no step jumps over the solid. `points`: _material(prog)."""
    if len(points) == 0:
        return 0
    G = BEAM_AXES[args[args.index("--beam-axis") + 1]]
    scale = float(args[args.index("--scale") + 1]) if "--scale" in args else 1.0
    if "--origin" in args:
        origin = np.array([float(x) for x in args[args.index("--origin") + 1:][:3]])
    else:                                                    # make_gdml's default: the centre of EVERY leaf's bounds
        probe = gdml.Program(np.eye(3), np.zeros(3), scale, "as-is")
        probe.build(prog)
        origin = 0.5 * (probe.lo + probe.hi)
    model = _model(prog)
    lo, hi = model.bounds()                                  # the tree's bounds hold the solid (Subtraction: its base)
    if np.all(np.abs([lo, hi]) < 1e8):                       # (a half space in a union has none)
        w = (scale * np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
             - origin) @ G.T
    else:
        w = (scale * points - origin) @ G.T
        w = np.stack([w.min(axis=0) - 0.25 * np.ptp(w, axis=0) - 1.0, w.max(axis=0) + 0.25 * np.ptp(w, axis=0) + 1.0])
    lo, hi = w.min(axis=0) - 1e-3, w.max(axis=0) + 1e-3
    rng = np.random.default_rng(seed)
    theta, phi = rng.uniform(0.0, STUDY_GUN["max_theta"], rays), rng.uniform(0.0, 2 * np.pi, rays)
    d = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1)
    o = np.array([0.0, 0.0, STUDY_GUN["vertex_z"]])
    with np.errstate(divide="ignore", invalid="ignore"):     # slab test: the lines that cross those bounds
        t1, t2 = (lo - o) / d, (hi - o) / d
    t0 = np.nanmax(np.minimum(t1, t2), axis=1).clip(0.0)
    t_out = np.nanmin(np.maximum(t1, t2), axis=1)
    live = np.flatnonzero(t0 < t_out)
    t, hit = t0[live], np.zeros(len(live), bool)
    for _ in range(400):
        on = ~hit & (t < t_out[live])
        if not on.any():
            break
        x = o + t[on, None] * d[live[on]]
        s = scale * model.sdf(((x @ G) + origin) / scale)    # the SDF in the world frame (G is a rotation)
        hit[np.flatnonzero(on)[s < 1e-3]] = True
        t[on] += np.maximum(s, 1e-3)
    return int(hit.sum())


def placement(prog: dict, gdml):
    """(make_gdml args, rule, predicted events with an entry) for a part on the 'study' gun. The rule of the first
    study ("bounds") stays where the gun sees the part; where a simulated validation scan predicts no entry
    (T-33: the origin came from every leaf's bounds, huge cut tools included, or the scale from an inflated ball),
    the gun is aimed at the material ("material")."""
    points = _material(prog)
    args = placement_bounds(prog)
    seen = scan_entries(prog, args, gdml, points)
    if seen > 0 or len(points) == 0:
        return args, "bounds", seen
    args = placement_material(points)
    return args, "material", scan_entries(prog, args, gdml, points)


# ------------------------------------------------------------------------------------------------ writing
def _write_one(job):
    """One program in every tree shape: [(row), ...]; a shape whose metrics repeat an earlier one is dropped."""
    aegir, name, series, gun, extra, prog_path, facts = job
    sys.path.insert(0, str(Path(aegir) / "scripts" / "sandbox_geometry"))
    import make_gdml
    out_dir = Path(aegir) / "geometry" / "sandbox" / name / series
    out_dir.mkdir(parents=True, exist_ok=True)
    prog = json.loads(Path(prog_path).read_text())
    args, rule, entries = (extra, "own gun", None) if extra is not None else placement(prog, make_gdml)
    rows, seen = [], set()
    for union, cuts in FORMS:
        stem = f"{series}__{union}_{cuts}".replace("as-is", "asis")
        gdml, metrics = out_dir / f"{stem}.gdml", out_dir / f"{stem}.metrics.json"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = make_gdml.main(["csg", str(prog_path), str(gdml), "--union", union, "--cuts", cuts,
                                     "--metrics", str(metrics)] + list(args))
        except SystemExit as exc:
            rc = exc.code
        except Exception as exc:                               # one shape failing must not stop the batch
            rc = f"{type(exc).__name__}: {exc}"
        if rc not in (0, None) or not metrics.exists():
            rows.append(dict(error=str(rc), series=series, shape=f"{union}/{cuts}", log=buf.getvalue()[-500:]))
            continue
        m = json.loads(metrics.read_text())
        key = json.dumps({k: m.get(k) for k in ("leaves", "nodes", "nodes_by_kind", "depth", "leaf_depth_mean",
                                                "imbalance")}, sort_keys=True)
        if key in seen:                                        # the same tree as an earlier shape: not written twice
            gdml.unlink(missing_ok=True); metrics.unlink(missing_ok=True)
            continue
        seen.add(key)
        row = {k: v for k, v in m.items() if k != "geometry"}          # the tree metrics first, ours on top
        row.update(facts)
        row.update(geometry=f"{name}/{series}/{stem}", part=gun, series=series,
                   shape=f"{union}" + ("+grouped" if cuts == "grouped" else ""), union=union, cuts=cuts,
                   same_solid=series, placement=args, placement_rule=rule, predicted_entries=entries)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aegir", type=Path, default=Path.home() / "Documents" / "Sandbox_aegir" / "aegir")
    ap.add_argument("--only", default="abc,liner,review,frame")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="at most this many programs (a trial run)")
    ap.add_argument("--series", help="only the series this regular expression finds (e.g. a rerun of a few)")
    ap.add_argument("--name", default="realtree", help="geometry/sandbox/<name>/ and <name>_index.json (default "
                    "realtree; another name leaves the study's geometries and index alone)")
    a = ap.parse_args()
    only = set(a.only.split(","))
    progs = WORK / "programs"
    progs.mkdir(parents=True, exist_ok=True)
    jobs = []
    found = [s for s in sources(only) if not a.series or re.search(a.series, s[0])]
    for series, gun, extra, prog, facts in found[: a.limit]:
        p = progs / f"{series}.json"
        p.write_text(json.dumps(prog))
        jobs.append((str(a.aegir), a.name, series, gun, extra, str(p), facts))
    print(f"{len(jobs)} programs", flush=True)
    rows, errors = [], []
    with ProcessPoolExecutor(a.jobs) as pool:
        for i, res in enumerate(pool.map(_write_one, jobs, chunksize=4)):
            for r in res:
                (errors if "error" in r else rows).append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(jobs)} programs, {len(rows)} geometries, {len(errors)} errors", flush=True)
    base = dict(geometry="study_baseline", part="study", series="baseline", shape="chain", leaves=1, nodes=0,
                nodes_by_kind={}, depth=0, leaf_depth_mean=0.0, imbalance=0.0, subtypes={"box/full": 1},
                same_solid="baseline")
    index = a.aegir / "geometry" / "sandbox" / f"{a.name}_index.json"
    index.write_text(json.dumps([base] + rows))
    err = WORK / ("errors.json" if a.name == "realtree" else f"errors_{a.name}.json")
    err.write_text(json.dumps(errors, indent=1))
    leaves = [r["leaves"] for r in rows]
    rules = {}
    for r in rows:
        rules.setdefault(r["placement_rule"], set()).add(r["series"])
    print(f"written {index}: {len(rows)} geometries of {len(jobs)} programs, {len(errors)} errors "
          f"(details {err}); leaves per tree: median {int(np.median(leaves)) if leaves else 0}, "
          f"max {max(leaves) if leaves else 0}; placement: "
          + ", ".join(f"{k} {len(v)}" for k, v in sorted(rules.items()))
          + "".join(f"\n  material: {s}" for s in sorted(rules.get("material", ()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
