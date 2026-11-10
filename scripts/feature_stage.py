#!/usr/bin/env python3
"""
The feature stage on one STEP part: a rough model, its exact difference to the input, one feature per piece.

    python3 scripts/feature_stage.py <step> --out <dir> [--fcstd] [--sources "step faces,fit,cascade"]
                                     [--rough-step model.step] [--max-pieces 500] [--all-sources]

Writes into <dir>: rough.step / rough.stl / rough.json (the program) / rough_parameters.json, pieces/ (every piece of
the difference as STEP and STL, pieces.json with its exact faces), features.json (the summary and the feature list)
and, with --fcstd, features.FCStd (input, rough model and every piece as a labelled object - open it in FreeCAD).
Prints the feature table and the time of the three phases. See reco_engine/feature_stage.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine.feature_stage import feature_stage, format_features, write_fcstd  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--sources", default="step faces,fit,cascade", help="the proposers, comma separated")
    ap.add_argument("--rough-step", default=None, help="a STEP to take as the rough model (skips the rough stage)")
    ap.add_argument("--max-pieces", type=int, default=500, help="the biggest pieces per sign that get a feature")
    ap.add_argument("--all-sources", action="store_true", help="ask every source for every piece (default: stop at "
                                                               "the first source whose candidate scores >= 0.99)")
    ap.add_argument("--fcstd", action="store_true", help="also write features.FCStd")
    args = ap.parse_args(argv)
    out = feature_stage(args.input, args.out, max_pieces=args.max_pieces, rough_step=args.rough_step,
                        **(dict(stop_at=None) if args.all_sources else {}),
                        sources=tuple(s.strip() for s in args.sources.split(",") if s.strip()))
    print(format_features(out))
    s = out["summary"]["seconds"]
    print(f"\nrough {s['rough']:.1f} s, difference {s['difference']:.1f} s, guess {s['guess']:.1f} s")
    print(f"written {Path(args.out) / 'features.json'}")
    if args.fcstd:
        print(f"written {write_fcstd(out, Path(args.out) / 'features.FCStd')}")
    return out


if __name__ == "__main__":
    main()
