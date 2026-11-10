#!/usr/bin/env python3
"""
One part in detail instead of a batch (the user, 2026-10-29: "create a complex part, store its tree, then create two
parallel trees for the same geometry, compare all their IoU and then reconstruct and compare the reconstruction details
accurately; then find out on which step the software lacks; a link to compute on the server and preview live").

    python3 scripts/detail_study.py make bracket            # the part, its tree T1 and two parallel trees T2, T3;
                                                            # STEP + STL of each, the exact IoU of every pair
    python3 scripts/remote_run.py start bracket ...         # the reconstruction on a server, followed live (the GUI)
    python3 scripts/detail_study.py compare bracket <run dir>   # the reconstruction against T1, T2 and T3 in detail
    python3 scripts/detail_study.py lacks bracket <run dir>     # the step at which the software went wrong, and why

Files: ~/shape_reco_results/detail/<name>/ (truths/<name>_T<k>.json in reco_engine.truth's format, step/, stl/,
the run) and docs/data/detail/<name>/{trees,compare,lacks}.json (the numbers of the study, in the repository).

The part "bracket" (mm, axis-aligned, 14 leaves in T1): a plate 120 x 80 x 12 with a cylindrical boss R 16 on top and a
bore R 8 through boss and plate, a rib between the boss and a pocket 32 x 24 x 6, four corner holes R 4 with counterbores
R 7 x 4 (one feature four times: the mirror partners of the symmetry rule of 2026-10-29) and a rebate 8 x 4 along the
whole front edge.
  T1  the machinist's tree: plate + boss + rib, then the eleven cuts one after the other
  T2  an additive tree: the rebate is no cut - the plate is two slabs; each corner is one tool (hole + counterbore) and
      the four corner tools are one union subtracted once
  T3  a reordered, split tree: the cuts on the plate first, then the boss and the rib, then the bore as two cylinders
      (plate part, boss part); the plate as the intersection of two boxes; the pocket as two overlapping boxes
The three are the same solid by construction - `make` checks it with OCCT's exact booleans (every pair's IoU).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen"), str(ROOT / "scripts")]

HOME = Path.home() / "shape_reco_results" / "detail"
DATA = ROOT / "docs" / "data" / "detail"
I3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


# ------------------------------------------------------------------------------------------------ tree builders

def box(cx, cy, cz, hx, hy, hz, label=None):
    """A box by its centre and half extents (program_to_dict's format)."""
    d = {"type": "Placed", "primitive": {"type": "Box", "half": [hx, hy, hz]}, "rotation": I3, "translation": [cx, cy, cz]}
    if label:
        d["label"] = label
    return d


def box_span(x0, x1, y0, y1, z0, z1, label=None):
    return box((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, label)


def cyl_z(cx, cy, z0, z1, r, label=None):
    """A cylinder along z from z0 to z1 (sdf_primitives.Cylinder runs from z = 0 to `height`, FreeCAD's makeCylinder:
    the translation is its base)."""
    d = {"type": "Placed", "primitive": {"type": "Cylinder", "radius": r, "height": z1 - z0, "u": [0.0, 2 * math.pi],
                                         "inner_radius": 0.0},
         "rotation": I3, "translation": [cx, cy, z0]}
    if label:
        d["label"] = label
    return d


def op(kind, a, b):
    return {"type": kind, "a": a, "b": b}


def chain(kind, first, rest):
    out = first
    for x in rest:
        out = op(kind, out, x)
    return out


CORNERS = [(50.0, 30.0), (-50.0, 30.0), (-50.0, -30.0), (50.0, -30.0)]


def bracket_trees() -> dict:
    """{"T1": program, "T2": ..., "T3": ...} of the bracket (see the module docstring)."""
    plate = box_span(-60, 60, -40, 40, 0, 12, "plate")
    boss = cyl_z(-30, 0, 11, 34, 16.0, "boss")
    rib = box_span(-15, 10, -2, 2, 11, 20, "rib")
    bore = cyl_z(-30, 0, -1, 35, 8.0, "bore")
    pocket = box_span(12, 44, -6, 18, 6, 13, "pocket")
    holes = [cyl_z(x, y, -1, 13, 4.0, f"hole {k + 1}") for k, (x, y) in enumerate(CORNERS)]
    cbores = [cyl_z(x, y, 8, 13, 7.0, f"counterbore {k + 1}") for k, (x, y) in enumerate(CORNERS)]
    rebate = box_span(-61, 61, -41, -32, 8, 13, "rebate")
    t1 = chain("Subtraction", chain("Union", plate, [boss, rib]), [bore, pocket] + holes + cbores + [rebate])

    lower = box_span(-60, 60, -40, 40, 0, 8, "lower slab")
    upper = box_span(-60, 60, -32, 40, 8, 12, "upper slab")
    corner_tools = [op("Union", h, c) for h, c in zip(holes, cbores)]
    t2 = chain("Subtraction", chain("Union", lower, [upper, boss, rib]),
               [bore, pocket, chain("Union", corner_tools[0], corner_tools[1:])])

    plate_i = op("Intersection", box_span(-60, 60, -40, 40, -5, 12, "slab below"),
                 box_span(-70, 70, -50, 50, 0, 20, "slab above"))
    pocket_a, pocket_b = box_span(12, 29, -6, 18, 6, 13, "pocket left"), box_span(27, 44, -6, 18, 6, 13, "pocket right")
    on_plate = chain("Subtraction", plate_i, [rebate, pocket_a, pocket_b] + cbores + holes)
    bore_plate, bore_boss = cyl_z(-30, 0, -1, 12.5, 8.0, "bore in the plate"), cyl_z(-30, 0, 11, 35, 8.0, "bore in the boss")
    t3 = chain("Subtraction", chain("Union", on_plate, [boss, rib]), [bore_plate, bore_boss])
    return {"T1": t1, "T2": t2, "T3": t3}


PARTS = {"bracket": bracket_trees}


def tree_facts(program: dict) -> dict:
    """Leaves, depth, operation counts and leaf kinds of a program."""
    counts = {"Union": 0, "Subtraction": 0, "Intersection": 0}
    kinds = {}

    def walk(node, depth):
        t = node["type"]
        if t == "Placed":
            k = node["primitive"]["type"]
            kinds[k] = kinds.get(k, 0) + 1
            return 1, depth
        counts[t] = counts.get(t, 0) + 1
        la, da = walk(node["a"], depth + 1)
        lb, db = walk(node["b"], depth + 1)
        return la + lb, max(da, db)
    leaves, depth = walk(program, 0)
    return dict(leaves=leaves, depth=depth, ops=counts, kinds=kinds)


# ------------------------------------------------------------------------------------------------ commands

def folder(name: str) -> Path:
    return HOME / name


def cmd_make(a):
    from reco_field.export import build_program
    from reco_field.exact_check import exact_residual
    trees = PARTS[a.name]()
    out = folder(a.name)
    for sub in ("truths", "step", "stl", "build"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    facts = {}
    for k, prog in trees.items():
        stem = f"{a.name}_{k}"
        truth = dict(name=stem, source="detail", meta=dict(part=a.name, tree=k, **tree_facts(prog)), program=prog)
        (out / "truths" / f"{stem}.json").write_text(json.dumps(truth, indent=1))
        t0 = time.time()
        ex = build_program(prog, out / "build" / k, stem, deflection=a.deflection)
        Path(ex.step).replace(out / "step" / f"{stem}.step")
        Path(ex.stl).replace(out / "stl" / f"{stem}.stl")
        facts[k] = dict(tree_facts(prog), volume=ex.volume, valid=ex.valid, solids=ex.n_solids, faces=ex.n_faces,
                        seconds=round(time.time() - t0, 1))
        print(f"{k}: {facts[k]['leaves']} leaves, depth {facts[k]['depth']}, volume {facts[k]['volume']}, "
              f"valid {facts[k]['valid']}, solids {facts[k]['solids']}")
    pairs = {}
    keys = list(trees)
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            r = exact_residual(str(out / "step" / f"{a.name}_{ka}.step"), str(out / "step" / f"{a.name}_{kb}.step"))
            pairs[f"{ka}-{kb}"] = dict(iou=r.get("iou"), r_plus=r.get("r_plus"), r_minus=r.get("r_minus"),
                                       method=r.get("method"), error=r.get("error"))
            print(f"exact IoU {ka} / {kb}: {r.get('iou')} (missing {r.get('r_plus')}, invented {r.get('r_minus')} mm^3, "
                  f"{r.get('method')}{', ' + r['error'] if r.get('error') else ''})")
    DATA.joinpath(a.name).mkdir(parents=True, exist_ok=True)
    (DATA / a.name / "trees.json").write_text(json.dumps(dict(part=a.name, trees=facts, exact_pairs=pairs,
                                                              created=time.strftime("%Y-%m-%dT%H:%M:%S")), indent=1))
    print(f"-> {out}, {DATA / a.name / 'trees.json'}")


def _labels(program: dict):
    """The leaves' labels in the order a compiled tree lists them (left to right, depth first)."""
    out = []

    def walk(node):
        if node["type"] == "Placed":
            out.append(node.get("label") or node["primitive"]["type"].lower())
            return
        walk(node["a"])
        walk(node["b"])
    walk(program)
    return out


def leaf_errors(truth, ops, labels) -> list:
    """Every truth leaf against the nearest reconstructed primitive of its kind (validation's leaf descriptors): centre
    distance (mm), the largest size difference (mm), the axis angle (deg), and whether validation's matching pairs them
    (cost <= 1: within 2 % of the diagonal in centre and size, 5 deg in axis)."""
    import numpy as np
    from reco_engine.validation import _leaf_descriptor, _pair_cost, compile_program
    model = compile_program(ops)
    M = [(i, _leaf_descriptor(l)) for i, l in enumerate(model.leaves)]
    M = [(i, d) for i, d in M if d is not None]
    rows = []
    for ti, leaf in enumerate(truth.compiled.leaves):
        t = _leaf_descriptor(leaf)
        if t is None:
            continue
        same = [(i, d) for i, d in M if d["kind"] == t["kind"] and len(d["dims"]) == len(t["dims"])]
        row = dict(leaf=ti, label=labels[ti] if ti < len(labels) else None, kind=t["kind"], role=t["role"],
                   centre=[round(float(x), 3) for x in t["centre"]], dims=[round(float(x), 3) for x in t["dims"]])
        if same:
            mi, m = min(same, key=lambda x: float(np.linalg.norm(x[1]["centre"] - t["centre"])))
            ang = None
            if t["axes"].shape[1] and m["axes"].shape[1]:
                cosv = abs(float(t["axes"][:, 0] @ m["axes"][:, 0]))
                ang = round(float(np.degrees(np.arccos(min(1.0, cosv)))), 3)
            row.update(nearest_op=int(m["owner"]), nearest_role=m["role"],
                       centre_mm=round(float(np.linalg.norm(m["centre"] - t["centre"])), 3),
                       size_mm=round(float(np.max(np.abs(m["dims"] - t["dims"]))), 3), axis_deg=ang,
                       cost=round(float(_pair_cost(t, m, truth.diag)), 3))
        rows.append(row)
    return rows


def _run_parts(run_dir: Path):
    import json as _json
    from reco_engine.validation import load_program
    result, log = load_program(run_dir)
    record = _json.loads((run_dir / "record.json").read_text()) if (run_dir / "record.json").is_file() else None
    return result, log, record


def cmd_compare(a):
    from reco_engine.truth import truth_from_json
    from reco_engine.validation import program_ops, rate
    out = folder(a.name)
    run_dir = Path(a.run).expanduser()
    result, log, record = _run_parts(run_dir)
    ops = program_ops(result)
    sources = {}
    for e in (log.final if log is not None else []):
        sources[e.get("op_index")] = e.get("source") or (e.get("primitive") or {}).get("label")
    per_tree = {}
    for k in ("T1", "T2", "T3"):
        tj = out / "truths" / f"{a.name}_{k}.json"
        truth = truth_from_json(tj)
        truth.input_path = str(out / "stl" / f"{a.name}_T1.stl")
        r = rate(truth, result, log, record, model_mesh=run_dir / "model.stl" if (run_dir / "model.stl").is_file() else None)
        prog = json.loads(tj.read_text())["program"]
        per_tree[k] = dict(volume=r["vector"]["volume"], hausdorff=r["vector"]["surface"].get("hausdorff"),
                           rms=r["vector"]["surface"].get("rms"), coverage=r["vector"]["surface"].get("coverage"),
                           tier=r["grade"]["tier"], limiting=r["grade"].get("limiting"), tree=r["vector"]["tree"],
                           trajectory=r["vector"]["trajectory"], by_op=r["attribution"]["by_op"],
                           classes=r["attribution"]["classes"], separation=r["attribution"]["separation"],
                           first_defect=r["attribution"]["first_defect"],
                           regions=[x for x in r["attribution"]["regions"] if x.get("significant")],
                           leaves=leaf_errors(truth, ops, _labels(prog)), hard=r["hard"], validity=r["vector"]["validity"])
    op_rows = []
    traj = per_tree["T1"]["trajectory"]
    for i, (op_, prim) in enumerate(ops):
        before = traj[i - 1]["iou"] if i > 0 else 0.0
        op_rows.append(dict(index=i, op=op_, kind=type(prim).__name__, source=sources.get(i), iou_after=traj[i]["iou"],
                            delta_iou=traj[i]["iou"] - before,
                            matched={k: next((p["truth"] for p in v["tree"]["pairs"] if p["model"] == i), None)
                                     for k, v in per_tree.items()}))
    rec = record or {}
    res = dict(part=a.name, run=str(run_dir), host=rec.get("host"), commit=rec.get("commit"),
               seconds=(rec.get("seconds") or {}), exact=(rec.get("comparison") or {}).get("exact"),
               model=rec.get("model"), stopped=(rec.get("run") or {}).get("stopped"), ops=op_rows, trees=per_tree)
    DATA.joinpath(a.name).mkdir(parents=True, exist_ok=True)
    (DATA / a.name / f"compare{'_' + a.label if a.label else ''}.json").write_text(json.dumps(res, indent=1, default=str))
    ex = res["exact"] or {}
    print(f"{a.name}: {len(ops)} operations; exact IoU {ex.get('iou')}; one valid solid "
          f"{(res['model'] or {}).get('one_valid_solid')}; stopped: {res['stopped']}")
    for k, v in per_tree.items():
        t = v["tree"]
        print(f"  {k}: tier {v['tier']}, volume IoU {v['volume']['iou']:.4f}, Hausdorff/D {v['hausdorff']:.4f}, "
              f"leaves matched {t['matched']}/{t['truth_leaves']} (model {t['model_leaves']}), separation "
              f"{v['separation'].get('cls') if v['separation'] else None} at op {v['separation'].get('culprit') if v['separation'] else None}")
    print("  operation  kind        IoU after  change   matched T1/T2/T3   source")
    for r in op_rows:
        print(f"  {r['index']:3d} {r['op']:5s} {r['kind']:10s} {r['iou_after']:9.4f} {r['delta_iou']:+8.4f}   "
              f"{str(r['matched']['T1']):>4s}/{str(r['matched']['T2']):>4s}/{str(r['matched']['T3']):>4s}   {r['source']}")
    print("  T1 leaves: label (kind, role) -> nearest op, centre mm, size mm, axis deg, matched")
    for x in per_tree["T1"]["leaves"]:
        print(f"    {str(x['label']):16s} ({x['kind']}, {x['role']}) -> op {x.get('nearest_op')}, {x.get('centre_mm')}, "
              f"{x.get('size_mm')}, {x.get('axis_deg')}, {'yes' if (x.get('cost') or 9) <= 1 else 'no'}")
    return res


STAGES = (("feature_rejected", "feature phase: the check refused the piece's best tool"),
          ("feature_candidate", "feature phase: no guess fitted the piece"),
          ("candidate", "rough loop: the judge of a candidate"),
          ("rejected", "rough loop: a candidate refused"),
          ("done", "the run stopped (budget, tolerance or time)"),
          ("feature_done", "the feature phase stopped (time budget or max iterations)"),
          ("accepted", "rough loop: an accepted operation that did harm"),
          ("feature_accepted", "feature phase: an accepted tool that did harm"),
          ("add", "the first add (the rough cover)"))


def cmd_lacks(a):
    """The steps at which the software went wrong: every significant defect region of the rating (against T1; the
    classes against T2 and T3 beside it), with the step that caused or failed to repair it and that step's reason."""
    cmp_path = DATA / a.name / f"compare{'_' + a.label if a.label else ''}.json"
    if not cmp_path.is_file():
        raise SystemExit("run `compare` first")
    res = json.loads(cmp_path.read_text())
    t1 = res["trees"]["T1"]
    stage_of = dict(STAGES)
    rows = []
    for reg in sorted(t1["regions"], key=lambda x: -x["volume_share"]):
        ev = reg.get("event") or {}
        kind = ev.get("kind")
        rows.append(dict(cls=reg["cls"], volume_share=reg["volume_share"], depth=reg.get("depth"), culprit=reg.get("culprit"),
                         step=ev.get("index"), stage=stage_of.get(kind, kind), cause=reg.get("cause"),
                         reason=ev.get("reason") or ev.get("stopped"), source=ev.get("source"), decisive=reg.get("decisive"),
                         centre=reg.get("centre")))
    by_stage = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0.0) + r["volume_share"]
    classes = {k: res["trees"][k]["classes"] for k in res["trees"]}
    out = dict(part=a.name, separation=t1["separation"], first_defect=t1["first_defect"], regions=rows,
               share_by_stage=dict(sorted(by_stage.items(), key=lambda kv: -kv[1])), classes_by_tree=classes)
    (DATA / a.name / f"lacks{'_' + a.label if a.label else ''}.json").write_text(json.dumps(out, indent=1, default=str))
    s = t1["separation"] or {}
    print(f"separation point: {s.get('cls')} (op {s.get('culprit')}, step {(s.get('event') or {}).get('index')}): "
          f"{s.get('cause')}; {100 * (s.get('volume_share') or 0):.2f} % of the volume")
    print("defects by the stage that caused them (share of the part's volume):")
    for k, v in out["share_by_stage"].items():
        print(f"  {100 * v:7.3f} %  {k}")
    print("significant regions:")
    for r in rows[:15]:
        print(f"  {r['cls']:14s} {100 * r['volume_share']:7.3f} %  op {r['culprit']}, step {r['step']}, {r['stage']}: "
              f"{r['cause']}{' - ' + str(r['reason'])[:90] if r['reason'] else ''}{' [decisive]' if r['decisive'] else ''}")
    print("classes by tree (the tree form names a defect, the geometry does not change):")
    for k, v in classes.items():
        print(f"  {k}: " + ", ".join(f"{c} {100 * x:.3f} %" for c, x in v.items()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("name", choices=sorted(PARTS))
    m.add_argument("--deflection", type=float, default=0.02)
    for name in ("compare", "lacks"):
        q = sub.add_parser(name)
        q.add_argument("name", choices=sorted(PARTS))
        q.add_argument("run", nargs="?" if name == "lacks" else None, help="the run directory (remote_run.py fetch: <mirror>/run)")
        q.add_argument("--label", default="", help="a name for this run's files (compare_<label>.json, lacks_<label>.json)")
    a = p.parse_args(argv)
    return dict(make=cmd_make, compare=cmd_compare, lacks=cmd_lacks)[a.cmd](a)


if __name__ == "__main__":
    main()
