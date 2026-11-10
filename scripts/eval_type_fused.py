#!/usr/bin/env python3
"""
Held-out Type CNN accuracy by input kind (training/field_input.py): occupancy vs sdf vs field vs fused, same backbone,
same held-out test sets the project already uses for rotation eval (never seen in training).

    python3 scripts/eval_type_fused.py --model occupancy=checkpoints_fused/type_occupancy/type_cnn --model fused=checkpoints_fused/type_fused/type_cnn
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.field_input import CHANNELS, encode  # noqa: E402
from training.manifest import load_manifest  # noqa: E402
from training.topology_schema import TOPOLOGY_TO_IDX  # noqa: E402
from training.train_type_cnn import TypeCNN  # noqa: E402

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"


def load_net(path: Path, device):
    cfg = json.loads((path.parent / "config.json").read_text())
    net = TypeCNN(in_channels=CHANNELS[cfg.get("input", "occupancy")])
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state_dict"])
    return net.to(device).eval(), cfg


@torch.no_grad()
def predict(net, volumes, device, input_kind: str, batch: int = 8):
    out = []
    for i in range(0, len(volumes), batch):
        chunk = volumes[i:i + batch]
        if input_kind == "occupancy":
            x = torch.from_numpy(np.stack(chunk).astype(np.float32) * 2 - 1).unsqueeze(1)
        else:
            x = torch.from_numpy(np.stack([encode(v.astype(np.float32) * 2 - 1, input_kind) for v in chunk]))
        out.append(net(x.to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=DIR", help="dir holding best_model.pth + config.json")
    ap.add_argument("--sets", nargs="+", default=["test_mixed", "test_cyl"], help="topology test sets to pool (box+sphere, cylinder)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data/type_fused_eval.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recs = []
    for name in args.sets:
        recs += load_manifest(DATA / name, require_voxel=True)
    y = np.array([TOPOLOGY_TO_IDX[r.topology] for r in recs])
    vols = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in recs]
    import collections
    print(f"{len(recs)} test samples: {dict(collections.Counter(r.topology for r in recs))}")

    result = {}
    for spec in args.model:
        name, directory = spec.split("=")
        net, cfg = load_net(Path(directory) / "best_model.pth", device)
        pred = predict(net, vols, device, cfg.get("input", "occupancy"))
        acc = float((pred == y).mean())
        per_class = {t: float((pred[y == i] == i).mean()) for t, i in TOPOLOGY_TO_IDX.items() if (y == i).any()}
        confusions = collections.Counter((int(a), int(b)) for a, b in zip(y, pred) if a != b)
        result[name] = dict(input=cfg.get("input", "occupancy"), n=len(recs), accuracy=acc, per_class=per_class,
                            confusions={f"{list(TOPOLOGY_TO_IDX)[a]}->{list(TOPOLOGY_TO_IDX)[b]}": c for (a, b), c in confusions.items()})
        print(f"{name:12s} input={cfg.get('input', 'occupancy'):10s} accuracy {100 * acc:5.1f} %  per-class {per_class}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
