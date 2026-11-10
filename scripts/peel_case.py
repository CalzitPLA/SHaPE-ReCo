#!/usr/bin/env python3
"""
Run the peeling loop on one STL/STEP file and print every stage: for studying a single part (a failure) without the GUI.

    python3 scripts/peel_case.py part.stl [--max-ops N] [--grade] [--all]

Prints the stage log (default: only the candidates, verdicts and the end; --all: every stage), the program with its parameters, and, with --grade,
the two-way chamfer. Case study: docs/FRAME_FAILURE.md.

--max-ops defaults to a PREDICTION from the part's own genus (`reco_field.peel.predict_ops_budget`), not a fixed number - pass an
int to override it with the old fixed behaviour.
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import INVENTED_SLACK, peel_steps, predict_ops_budget  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--max-ops", type=int, default=None, help="default: predicted from the part's genus, see the module docstring")
    ap.add_argument("--grade", action="store_true", help="two-way chamfer at the end (seconds; about 90 s on a part several metres across)")
    ap.add_argument("--all", action="store_true", help="print every stage, including field and cluster")
    ap.add_argument("--bounding-box", action="store_true",
                    help="use reco_field.fit.fit_box_bounding (covers every point of a cluster, no Tukey-IRLS "
                         "refinement) instead of the converged fit for a box base/cut/add - docs/FRAME_FAILURE.md, "
                         "'Bounding-box fits for the acceptance-gated cover'. Off by default.")
    ap.add_argument("--invented-slack", type=float, default=None,
                    help="override reco_field.peel.INVENTED_SLACK (default 0.02); --bounding-box needs this opened "
                         "well past its default before it accepts what it proposes - 0.2+ on the frame part")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    field = load_field(args.path)
    lo, hi = field.bounds
    genus = field.genus()
    if genus != genus:                                                 # NaN: not closed
        genus_note = "not closed (genus undefined)"
    elif genus <= 1:
        genus_note = f"genus {genus:.0f}"
    else:
        genus_note = f"genus {genus:.0f} — more than a plain frame/ring, expect many independent through-features"
    print(f"{Path(args.path).name}: {len(field.faces)} triangles, bounding box {np.round(hi - lo, 1)}, {genus_note}")
    if args.max_ops is None:
        budget = predict_ops_budget(field)
        print(f"  predicted budget: {budget.max_ops} ops ({budget.reason}; plateau genus {budget.plateau_genus:.0f} at "
             f"resolution {budget.plateau_resolution}, exact genus {budget.exact_genus:.0f})")
    else:
        print(f"  budget: {args.max_ops} ops (given explicitly, not predicted)")
    t = time.time()
    last = None
    for st in peel_steps(field, max_ops=args.max_ops, grade=args.grade, bounding_box=args.bounding_box,
                        invented_slack=args.invented_slack if args.invented_slack is not None else INVENTED_SLACK):
        last = st
        if args.all or st.stage in ("base", "candidate", "accepted", "rejected", "grade", "done"):
            extra = ""
            if st.stage == "candidate" and st.candidate.op in ("add", "cut"):
                l, h = st.candidate.primitive.bounds()
                extra = f"  [fit half {np.round(st.candidate.fit.half, 1)}, primitive extent {np.round(h - l, 1)}]"
            print(f"{time.time() - t:6.1f}s {st.stage:9s} {st.message}{extra}", flush=True)
    print("program:")
    for i, o in enumerate(last.ops):
        params = {k: round(float(v), 2) for k, v in o.fit.params.items()}
        print(f"  {i}. {o.op}:{o.fit.kind} {params} -> {o.explained_after:.1%}")
    if last.grade:
        g = last.grade
        print(f"chamfer: input->model mean {g['input_to_model']['mean']:.4g}, model->input mean {g['model_to_input']['mean']:.4g}, "
              f"symmetric {g['symmetric_mean']:.4g}, worst {g['hausdorff']:.4g}")


if __name__ == "__main__":
    main()
