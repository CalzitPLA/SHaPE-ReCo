#!/usr/bin/env python3
"""
Headless entry point of the reconstruction: the residual add-cut loop (reco_engine/residual_loop.py) with the
stage-registry cascade, STL or STEP input, built and graded in FreeCAD. Prints the result_schema JSON.

This is the thing to point a cluster batch job (or a CI/regression check) at: one process, one exit code, one JSON
result on stdout - nothing about it depends on an interactive FreeCAD session.

Run as:
    python3 -m reco_engine.reconstruct_cli --input-stl part.stl --workdir /tmp/reco
    python3 -m reco_engine.reconstruct_cli --mode add_cut --input-stl part.step --workdir /tmp/reco

The legacy modes 'loop' and 'ensemble' (the voxel greedy drivers, route C) were removed on 2026-09-30; they are in
git history (the commits before the tidy branch).

Prints a human-readable summary to stderr as it runs, and exactly one JSON object to stdout on success.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="add_cut", choices=["add_cut"],
                        help="'add_cut': the residual add-cut loop with the stage-registry cascade (the only mode).")
    parser.add_argument("--input-stl", type=Path, required=True, help="STL or STEP input.")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Also write the JSON result here (in addition to stdout).")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    from reco_engine.residual_loop import ResidualConfig, reconstruct
    _log(f"[reconstruct_cli] mode=add_cut input={args.input_stl}")
    try:
        result = reconstruct(args.input_stl, ResidualConfig(out_dir=args.workdir))
    except Exception as exc:  # surface a clean one-line failure on top of the traceback
        _log(f"[reconstruct_cli] FAILED: {exc}")
        raise
    payload = result.to_json(indent=2)
    if args.output:
        args.output.write_text(payload)
        _log(f"[reconstruct_cli] wrote result to {args.output}")
    print(payload)


if __name__ == "__main__":
    main()
