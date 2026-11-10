#!/usr/bin/env python3
"""
Standalone CLI wrapping reco_engine.predict's in-process functions,
meant to be invoked as a subprocess by a Python that does NOT have
torch (see reco_engine/model_registry.py's subprocess mode), using a
Python that DOES.

Run as: python3 -m reco_engine.predict_cli --checkpoint-root PATH
    --task {type,rotation,dimension} --voxel-npy PATH [--topology NAME]

Prints one JSON object to stdout. Voxel input is a .npy file holding
the CNN-ready {-1,+1} float32 array (see reco_engine.predict._to_tensor)
- not a raw {0,1} .bin - so no re-normalization happens on either side
of the subprocess boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="best_model.pth")
    parser.add_argument("--task", required=True, choices=["type", "rotation", "dimension"])
    parser.add_argument("--voxel-npy", type=Path, required=True)
    parser.add_argument("--topology", default=None, help="required for rotation/dimension tasks")
    parser.add_argument("--allowed", default=None, help="comma separated topologies the Type CNN may predict")
    args = parser.parse_args()

    from reco_engine import model_registry
    if not model_registry.TORCH_AVAILABLE:
        print(f"predict_cli: torch is not importable under {sys.executable} either.", file=sys.stderr)
        sys.exit(1)

    from reco_engine.predict import predict_dimensions, predict_rotation, predict_type

    models = model_registry.load_models(args.checkpoint_root, checkpoint_name=args.checkpoint_name,
                                        allowed_topologies=args.allowed)
    voxel = np.load(args.voxel_npy)

    if args.task == "type":
        prediction = predict_type(models, voxel)
        result = {
            "topology": prediction.topology,
            "confidence": prediction.confidence,
            "top5": prediction.top5,
        }
    else:
        if not args.topology:
            print(f"predict_cli: --topology is required for task={args.task}", file=sys.stderr)
            sys.exit(1)
        if args.task == "rotation":
            result = {"quaternion": predict_rotation(models, voxel, args.topology)}
        else:
            result = {"parameters": predict_dimensions(models, voxel, args.topology)}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
