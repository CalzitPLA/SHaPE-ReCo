#!/usr/bin/env python3
"""
A stored run's program as a GDML file for Geant4 / ROOT (reco_field/gdml_export.py).

    python3 scripts/export_gdml.py <run directory | part:latest | part:<n> | operations.json | program.json> [--out part.gdml]
                                   [--material iron|aluminium] [--tree grouped-balanced|grouped-multi|as-is] [--check]

program.json: a program tree (reco_field.export.program_to_dict; the real-tree study's programs/<series>.json).
--tree: the shape of the boolean tree (reco_field/gdml_export.py, TREE SHAPES; default grouped cuts, balanced unions).
--check reads the written file back and compares the solid with the program at 200 000 points.
--points N writes N points with the program's verdict for the check by ROOT (scripts/gdml_check_root.py).
"""

import argparse
import sys
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

import numpy as np  # noqa: E402

from reco_engine import run_store  # noqa: E402
from reco_engine.result_schema import ReconstructionResult  # noqa: E402
from reco_engine.result_to_program import program_model  # noqa: E402
from reco_field import gdml_export as ge  # noqa: E402

MATERIALS = {"iron": ge.STEEL, "aluminium": ge.ALUMINIUM}


def _check(model, out, n=200000) -> int:
    """The written file read back, judged against the program at n points in its bounds: 1 if any point differs."""
    lo, hi = (np.asarray(b, float) for b in model.bounds())
    pts = np.random.default_rng(0).uniform(lo, hi, size=(n, 3))
    a, b = np.asarray(model.sdf(pts)), np.asarray(ge.read_gdml(out).sdf(pts))
    sure = np.abs(a) > 1e-6
    differ = int(((a < 0) != (b < 0))[sure].sum())
    print(f"check: {int((a < 0).sum())} of {len(pts)} points inside the program, {differ} judged differently by the file read back")
    return 1 if differ else 0


def _program_tree(model, source: Path, args) -> int:
    out = Path(args.out) if args.out else source.with_suffix(".gdml")
    info = ge.export_gdml(model, out, name=source.stem, material=MATERIALS[args.material], tree=args.tree)
    print(f"{info['path']}: {info['primitives']} primitives, {info['booleans']} booleans "
          f"({', '.join(f'{n} {k}' for k, n in sorted(info['solids'].items()))}), tree {info['tree']}")
    return _check(model, out) if args.check else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--out", default=None, help="the GDML file (default: <run directory>/model.gdml)")
    ap.add_argument("--material", choices=sorted(MATERIALS), default="iron")
    ap.add_argument("--tree", choices=ge.TREE_SHAPES, default=ge.TREE_SHAPES[0],
                    help="the shape of the boolean tree (default: grouped cuts, balanced unions; as-is: the program's own tree)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--points", type=int, default=0, metavar="N",
                    help="also write <out>.points.txt: N points with the program's verdict, for scripts/gdml_check_root.py")
    args = ap.parse_args(argv)
    if str(args.run).endswith(".json") and Path(args.run).name != "run.json":      # a list of operations, or a dict with one
        import json
        data = json.loads(Path(args.run).read_text())
        if isinstance(data, dict) and data.get("type") in ("Placed", "Union", "Subtraction", "Intersection"):
            return _program_tree(ge.model_of_program(data), Path(args.run), args)       # a program tree
        ops = data if isinstance(data, list) else data.get("operations") or data.get("program") or []
        # an operation with its primitive's fields beside "op" (the frame report's format) is taken too
        ops = [o if "primitive" in o else dict(op=o["op"], accepted=o.get("accepted", True),
                                               primitive={k: v for k, v in o.items() if k not in ("op", "accepted")})
               for o in ops]
        run = dict(program=ops, part=Path(args.run).stem, path=str(Path(args.run).parent))
    else:
        run = run_store.load_run(run_store.resolve(args.run))
    cut = run_store.truncated(run.get("program"))
    if cut:
        print(f"the run store cut {len(cut)} node(s) of this program down to text when it saved it (before 2026-10-01; "
              f"first: {cut[0]}): it cannot be built. scripts/recover_run_program.py recovers it from the feature "
              f"phase's record when that is still in the cache", file=sys.stderr)
        return 2
    if run.get("program_source") not in (None, run_store.RUN_JSON):
        print(f"the program of {run['program_source']} (recovered; run.json's was cut)", file=sys.stderr)
    def bare(p):                                          # the geometry alone (a recovered node has no sources)
        return dict(p, sources=[], alternatives=[], parts=[bare(q) for q in p.get("parts") or []])
    ops = [dict(o, primitive=bare(o["primitive"])) for o in run.get("program") or []]
    result = ReconstructionResult.from_dict(dict(input_path=(run.get("input") or {}).get("path") or run.get("part") or "part",
                                                 operations=ops))
    out = Path(args.out) if args.out else Path(run["path"]) / "model.gdml"
    info = ge.export_result(result, out, name=run.get("part"), material=MATERIALS[args.material], tree=args.tree)
    print(f"{info['path']}: {info['primitives']} primitives, {info['booleans']} booleans "
          f"({', '.join(f'{n} {k}' for k, n in sorted(info['solids'].items()))}), volume {info['volume']} of {info['material']}, "
          f"tree {info['tree']}")
    if args.points:
        model = program_model(result)
        lo, hi = (np.asarray(b, float) for b in model.bounds())
        pad = 0.1 * float(np.max(hi - lo))
        pts = np.random.default_rng(1).uniform(lo - pad, hi + pad, size=(args.points, 3))
        s = np.asarray(model.sdf(pts))
        keep = np.abs(s) > 1e-3                           # 1 micrometre off the surface: there the sign is rounding
        np.savetxt(str(out)[:-5] + ".points.txt", np.column_stack([pts[keep], (s[keep] < 0).astype(int)]), fmt="%.9g")
        print(f"points: {int(keep.sum())} written, {int((s[keep] < 0).sum())} of them inside")
    if args.check:
        return _check(program_model(result), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
