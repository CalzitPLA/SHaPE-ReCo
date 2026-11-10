#!/usr/bin/env python3
"""
The exact IoU of a stored run's model after the run, by the incremental difference (2026-10-01): the global boolean of
input and model fails once many feature tools are in the model (liner, 60 and 86 features: the general fuse refused,
the booleans disagree), so the difference is built up as the feature phase builds it - the rough model's exact
difference first (general fuse, then the booleans), then the run's tools applied to its pieces in chunks
(reco_field.exact_check.incremental_residual), each chunk checked by the volume budget and the sampled intersection.

    python3 scripts/exact_of_run.py <run directory | run.json> --input <part.step> --out <directory>
                                    [--rough N] [--chunk K] [--timeout S] [--build-timeout S]

--rough N: the first N operations are the rough model (default: the operations before the first one whose source is
the feature phase, "feature:..."). --chunk K: K tools per incremental difference (default 0: all in one).
--snap MM: every tool whose cylinder and cone apex lie within MM of an axis and an apex of the input's faces is moved
rigidly onto them first (snap_tool); the moves are recorded, and the IoU is that of the moved program.
--grow MM: every tool grown by MM (grow), so that no face of a tool lies within MM of the face it reproduces: the model
cut by the grown tools lies inside the run's model, so its exact r+ is an upper bound of the run's r+ (0 is the lower
one), and the run's r- = r+ + V_model - V_in. Hence the bracket V_in / V_model >= IoU >= (V_in - r+') / (V_model + r+')
(liner, 86 features, 2026-10-01). Shrinking (MM < 0) is accepted but parts a tool's fused leaves into separate solids
(gaps of 2 MM): no use for the bracket. --global-first: the global difference of the whole model is tried first.
Written into <directory>: rough.step / model_<k>.step (the models built in FreeCAD), state_<k>/ (the difference's
solids), exact_of_run.json (the run, the input, per stage: the model's volume and validity, method, r+, r-, IoU,
budget error, unsure pairs, seconds; `final`: the last stage that answered, `complete`: it is the whole program).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]


def commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        return out or (ROOT / ".commit").read_text().strip()
    except OSError:
        return "unknown"


def load_result(spec: str):
    """The run's ReconstructionResult (geometry only: sources and alternatives dropped) and the run's dict."""
    from reco_engine import run_store
    from reco_engine.result_schema import ReconstructionResult
    path = Path(spec)
    run = run_store.load_run(path.parent if path.name == "run.json" else path)

    def walk(p):
        yield p
        for q in p.get("parts") or []:
            yield from (walk(q) if isinstance(q, dict) else [q])

    if any(isinstance(q, str) for o in run.get("program") or [] for q in walk(o["primitive"])):
        raise SystemExit("this run's program has composite operands stored as text: it cannot be built")

    def bare(p):
        return dict(p, sources=[], alternatives=[], parts=[bare(q) for q in p.get("parts") or []])

    ops = [dict(o, primitive=bare(o["primitive"])) for o in run.get("program") or []]
    result = ReconstructionResult.from_dict(dict(input_path=(run.get("input") or {}).get("path") or "part",
                                                 operations=ops))
    return result, run


def feature_start(run: dict) -> int:
    """The index of the first operation the feature phase made (its primitive's sources name "feature:...")."""
    for i, o in enumerate(run.get("program") or []):
        modes = [str(s.get("mode", "")) for s in (o["primitive"].get("sources") or []) if isinstance(s, dict)]
        if any(m.startswith("feature:") for m in modes):
            return i
    return len(run.get("program") or [])


def _apex(leaf, R):
    """The apex of a cone leaf's surface (where its radius runs out), its base at centre - axis * half_length with
    radius1, its top at centre + axis * half_length with radius2; None for a cylinder-like cone."""
    import numpy as np
    d, c = R[:, 2], np.asarray(leaf.center_mm, float)
    r1, r2, h = leaf.dims_mm["radius1"], leaf.dims_mm["radius2"], leaf.dims_mm["half_length"]
    if abs(r1 - r2) < 1e-9:
        return None
    return c - d * h + d * (2.0 * h * r1 / (r1 - r2))


def snap_tool(prim, faces, tol: float = 1e-3, angle: float = 1e-5):
    """The tool moved rigidly onto the input's own axes: the pattern copies of the liner's tools sit 1e-7 to 4e-5 mm
    off the faces they reproduce (run of 2026-09-30 08:07), where OCCT's booleans neither see the faces as one nor as
    apart. The first cylinder leaf whose radius (1e-4), direction (`angle`) and axis line (`tol` mm) match a
    cylindrical face of the input sets the direction and the line; a cone leaf whose surface apex lies on that line
    within `tol` of a conical face's apex sets the position along it. Returns (the moved primitive or the same one,
    dict(matched, axial, shift (mm, the largest move of a leaf centre), angle (rad)))."""
    import numpy as np
    from reco_engine.result_to_program import rotation_matrix, transform_primitive

    def leaves(p):
        if p.kind == "composite":
            for q in p.parts:
                yield from leaves(q)
        else:
            yield p

    cyl = [f for f in faces if f["type"] == "cylinder" and f.get("axis") and f.get("center")]
    cones = [f for f in faces if f["type"] == "cone" and f.get("axis") and f.get("apex")]
    match = None
    for leaf in leaves(prim):
        if leaf.kind != "cylinder":
            continue
        R = rotation_matrix(leaf)
        d, c = R[:, 2], np.asarray(leaf.center_mm, float)
        for f in cyl:
            if abs(f["radius"] - leaf.dims_mm["radius"]) > 1e-4:
                continue
            df, pf = np.asarray(f["axis"], float), np.asarray(f["center"], float)
            df = df if df @ d > 0 else -df
            if np.linalg.norm(np.cross(d, df)) > angle:
                continue
            v = c - pf
            dist = np.linalg.norm(v - df * (v @ df))
            if dist <= tol and (match is None or dist < match[0]):
                match = (dist, d, df, pf, c)
        if match is not None:
            break
    if match is None:
        return prim, dict(matched=False)
    _, d, df, pf, c = match
    axis = np.cross(d, df)
    s, co = np.linalg.norm(axis), float(np.clip(d @ df, -1.0, 1.0))
    if s < 1e-15:
        Q = np.eye(3)
    else:
        k = axis / s
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Q = np.eye(3) + s * K + (1 - co) * (K @ K)                 # Rodrigues: Q d = df
    v = c - pf
    t = (c - Q @ c) + (pf + df * (v @ df) - c)                      # rotate about c, then onto the line
    axial = False
    for leaf in leaves(prim):
        if leaf.kind != "cone":
            continue
        a = _apex(leaf, rotation_matrix(leaf))
        if a is None:
            continue
        a = Q @ a + t
        for f in cones:
            fa, fd = np.asarray(f["apex"], float), np.asarray(f["axis"], float)
            w = fa - pf
            if np.linalg.norm(np.cross(fd, df)) > angle or np.linalg.norm(w - df * (w @ df)) > tol:
                continue
            gap = (fa - a) @ df
            if abs(gap) <= tol:
                t = t + df * gap
                axial = True
                break
        if axial:
            break
    if not axial:                       # a line alone leaves a turn about it and a slide along it open (the liner's
        return prim, dict(matched=True, axial=False, moved=False)     # pocket copies): such a tool stays where it is
    moved = transform_primitive(prim, Q, t)
    shift = max(float(np.linalg.norm(np.asarray(m.center_mm) - np.asarray(o.center_mm)))
                for m, o in zip(leaves(moved), leaves(prim)))
    return moved, dict(matched=True, axial=axial, moved=True, shift=shift, angle=float(np.arcsin(min(1.0, s))))


def grow(p, delta: float):
    """The primitive grown (delta > 0) or shrunk (delta < 0) by |delta| mm, leaf by leaf: a box's halves, a cylinder's
    radius and half length, a sphere's radius, a torus's minor radius, a cone's surface moved along its normal and its
    ends along the axis, a half-space's plane; in a composite the subtracted operand the other way. The grown tool
    contains the tool and the shrunk one is contained in it (unions, intersections and differences are monotone), so
    a model cut by the grown tools lies inside the run's model: its exact r+ bounds the run's from above (`--grow`),
    with no face of a tool left within delta of the face it reproduces. Shrunk leaves that were fused face to face
    come apart (liner, 2026-10-01: blind holes and stacks in 2 to 4 solids, the pocket tools not valid)."""
    import copy
    import math
    import numpy as np
    from reco_engine.result_to_program import rotation_matrix
    q = copy.copy(p)
    q.alternatives = []
    if p.kind == "composite":
        a, b = p.parts
        q.parts = [grow(a, delta), grow(b, -delta if p.combine == "cut" else delta)]
        return q
    d = dict(p.dims_mm)
    if p.kind == "box":
        for k in ("x_half", "y_half", "z_half"):
            d[k] = max(d[k] + delta, 1e-6)
    elif p.kind == "cylinder":
        d["radius"], d["half_length"] = max(d["radius"] + delta, 1e-6), max(d["half_length"] + delta, 1e-6)
    elif p.kind == "sphere":
        d["radius"] = max(d["radius"] + delta, 1e-6)
    elif p.kind == "torus":
        d["minor_radius"] = max(d["minor_radius"] + delta, 1e-6)
    elif p.kind == "cone":
        r1, r2, h = d["radius1"], d["radius2"], d["half_length"]
        s = (r2 - r1) / (2.0 * h)                                    # the radius along the axis: mid + s t, |t| <= h
        mid, n = 0.5 * (r1 + r2), delta * math.sqrt(1.0 + s * s)
        h2 = max(h + delta, 1e-6)
        d.update(radius1=max(mid - s * h2 + n, 0.0), radius2=max(mid + s * h2 + n, 0.0), half_length=h2)
    elif p.kind == "halfspace":                                     # {x : n.x <= n.c}: the plane moves along n
        n = rotation_matrix(p)[:, 2]
        q.center_mm = tuple(float(x) for x in np.asarray(p.center_mm, float) + delta * n)
    else:
        raise ValueError(f"grow: no rule for a {p.kind}")
    q.dims_mm = d
    return q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--input", required=True, help="the part's STEP (the run's input)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rough", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=3600, help="per difference (the incremental one: + 10 s per tool)")
    ap.add_argument("--build-timeout", type=int, default=3600)
    ap.add_argument("--check-points", type=int, default=1500, help="seeded points of the sampled intersection")
    ap.add_argument("--snap", type=float, default=0.0, metavar="MM",
                    help="move every feature tool rigidly onto the input's own axes when they lie within MM (snap_tool); "
                         "the moves are recorded (`snap`), the IoU is then that of the moved program")
    ap.add_argument("--grow", type=float, default=0.0, metavar="MM",
                    help="every feature tool grown (> 0) or shrunk (< 0) by |MM| (grow): one side of a bracket")
    ap.add_argument("--global-first", action="store_true",
                    help="try the global difference (general fuse, then the booleans) of the whole model first")
    args = ap.parse_args(argv)

    from reco_engine.result_to_program import primitive_to_sp, program_model
    from reco_field.exact_check import exact_faces, exact_residual, incremental_residual
    from reco_field.export import build_in_freecad, program_to_dict

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    result, run = load_result(args.run)
    ops = [o for o in result.operations if o.accepted]
    n_rough = args.rough if args.rough is not None else feature_start(run)
    rec = dict(run=str(args.run), input=str(args.input), commit=commit(), host=os.uname().nodename,
               started=time.strftime("%Y-%m-%dT%H:%M:%S"), operations=len(ops), rough_operations=n_rough,
               features=len(ops) - n_rough, chunk=args.chunk, timeout=args.timeout, stages=[], final=None,
               complete=False)

    def save():
        (out / "exact_of_run.json").write_text(json.dumps(rec, indent=1, default=str))

    if args.snap > 0:
        fx = exact_faces(args.input)
        if "error" in fx:
            raise SystemExit(f"--snap: {fx['error']}")
        moves = []
        for i, o in enumerate(ops[n_rough:], start=n_rough):
            o.primitive, info = snap_tool(o.primitive, fx["faces"], tol=args.snap)
            moves.append(dict(operation=i, label=(o.primitive.label or "")[:80], **info))
        moved = [m for m in moves if m.get("moved")]
        rec["snap"] = dict(tolerance=args.snap, tools=len(moves), moved=len(moved),
                           max_shift=max((m["shift"] for m in moved), default=0.0),
                           max_angle=max((m["angle"] for m in moved), default=0.0), moves=moves)
        save()
    if args.grow:
        for o in ops[n_rough:]:
            o.primitive = grow(o.primitive, args.grow)
        rec["grow"] = args.grow
        save()

    def build(upto, name):
        t = time.time()
        ex = build_in_freecad(program_model(result, upto=upto), out, name, timeout=args.build_timeout)
        return ex, dict(model=str(ex.step), operations=upto, volume=float(ex.volume), valid=bool(ex.valid),
                        solids=int(ex.n_solids), faces=int(ex.n_faces), build_seconds=time.time() - t)

    def keep(r):
        return {k: r.get(k) for k in ("method", "iou", "r_plus", "r_minus", "input_volume", "model_volume",
                                      "budget_error", "balance_error", "fuzzy", "sampled_common", "tools", "booleans",
                                      "repairs", "r_plus_solids", "r_minus_solids", "model_genus", "seconds", "error",
                                      "tried")} | dict(unsure=len(r.get("unsure") or []),
                                                       unsure_detail=(r.get("unsure") or [])[:20])

    if args.global_first:                                            # the whole model in one difference
        ex, st = build(len(ops), "model")
        t = time.time()
        r = exact_residual(args.input, str(ex.step), timeout=args.timeout)
        st.update(keep(r), stage="global", wall=time.time() - t)
        rec["stages"].append(st)
        print(json.dumps({k2: st.get(k2) for k2 in ("stage", "iou", "r_plus", "r_minus", "error", "wall")}), flush=True)
        if "error" not in r:
            rec.update(final=st, complete=True, ended=time.strftime("%Y-%m-%dT%H:%M:%S"))
            save()
            print(json.dumps(dict(complete=True, operations=len(ops), iou=st.get("iou"), r_plus=st.get("r_plus"),
                                  r_minus=st.get("r_minus"))))
            return 0
        save()

    # the rough model: the global difference, its solids kept as the state
    ex, st = build(n_rough, "rough")
    t = time.time()
    r = exact_residual(args.input, str(ex.step), timeout=args.timeout, state_dir=out / "state_0")
    st.update(keep(r), stage="rough", wall=time.time() - t)
    rec["stages"].append(st)
    save()
    if "error" in r:
        print(json.dumps(dict(stage="rough", error=r["error"][-300:])))
        return 1
    rec["final"] = st
    state = out / "state_0"
    bounds = list(range(n_rough, len(ops), args.chunk))[1:] + [len(ops)] if args.chunk > 0 else [len(ops)]
    lo = n_rough
    for k, hi in enumerate(bounds, start=1):
        if hi <= lo:
            continue
        ex, st = build(hi, f"model_{k}")
        tools = [dict(op=o.op, program=program_to_dict(primitive_to_sp(o.primitive))) for o in ops[lo:hi]]
        t = time.time()
        r = incremental_residual(args.input, str(ex.step), state, tools, state_dir=out / f"state_{k}",
                                 timeout=int(args.timeout + 10 * len(tools)), check_points=args.check_points)
        st.update(keep(r), stage=f"tools {lo}..{hi - 1}", wall=time.time() - t)
        if r.get("tool_log"):
            (out / f"tool_log_{k}.json").write_text(json.dumps(r["tool_log"], indent=1))
        rec["stages"].append(st)
        save()
        print(json.dumps({k2: st.get(k2) for k2 in ("stage", "iou", "r_plus", "r_minus", "budget_error", "error", "wall")}),
              flush=True)
        if "error" in r:
            break
        rec["final"], state, lo = st, out / f"state_{k}", hi
    rec["complete"] = rec["final"] is not None and rec["final"].get("operations") == len(ops)
    rec["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save()
    f = rec["final"] or {}
    print(json.dumps(dict(complete=rec["complete"], operations=f.get("operations"), iou=f.get("iou"),
                          r_plus=f.get("r_plus"), r_minus=f.get("r_minus"))))
    return 0 if rec["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
