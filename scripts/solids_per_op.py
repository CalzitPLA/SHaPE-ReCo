#!/usr/bin/env python3
"""Which operation of a program first splits the model into several solids (2026-10-16): every prefix of the accepted
operations of a run's program.json is built in freecadcmd and its solid count printed. The loop's solid gate (D5) judges
a cut on the voxel grid (26-connectivity); the exact build can still sever the model - this script names the operation.

    python3 scripts/solids_per_op.py <run dir with program.json> [--from K] [--timeout S]
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="a run directory (program.json) or the program.json itself")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first prefix length to build (default 1)")
    ap.add_argument("--timeout", type=int, default=300, help="freecadcmd timeout per build (s)")
    ap.add_argument("--out", default=None, help="where the prefix builds go (default: <run>/solids_per_op)")
    a = ap.parse_args()
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.result_to_program import program_model
    from reco_field.export import build_in_freecad
    run = Path(a.run)
    prog = run if run.is_file() else run / "program.json"
    result = ReconstructionResult.from_dict(json.loads(prog.read_text()))
    ops = [o for o in result.operations if o.accepted]
    out = Path(a.out) if a.out else (run if run.is_dir() else run.parent) / "solids_per_op"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ops)} accepted operations of {prog}")
    print(f"{'k':>3s} {'op':9s} {'kind':10s} {'solids':>6s} {'valid':>5s} {'s':>6s}  label")
    prev = None
    for k in range(max(1, a.start), len(ops) + 1):
        o = ops[k - 1]
        t = time.time()
        try:
            ex = build_in_freecad(program_model(result, k), out, f"p{k:02d}", timeout=a.timeout)
            n, valid = int(ex.n_solids), bool(ex.valid)
        except Exception as e:                                     # a crash of freecadcmd is an answer too
            n, valid = -1, False
            print(f"    build error at {k}: {str(e)[:160]}")
        mark = ""
        if prev is not None and n != prev:
            mark = "  <-- splits" if n > prev else "  <-- joins"
        label = o.primitive.label or ""
        print(f"{k:3d} {o.op:9s} {o.primitive.kind:10s} {n:6d} {str(valid):>5s} {time.time() - t:6.1f}  {label[:70]}{mark}", flush=True)
        prev = n
    return 0


if __name__ == "__main__":
    sys.exit(main())
