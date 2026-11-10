#!/usr/bin/env python3
"""
Held-out evaluation of box orientation estimators (symmetry-aware angular error, degrees).

Test boxes: dataset_gen/local_dataset_inscale/{test,test_ff}, full-field cubes excluded (no orientation).
The error of an estimate E against the label M is min over the admissible symmetry group of the geodesic
angle between E and M @ g (training/rotation_symmetry.py): 4 half-turns for a generic box, 8 / 24 when
dimensions are nearly equal. Methods:

  random        uniformly random rotation                           (chance level)
  pca           second-moment principal axes of the largest connected component (classical, no learning)
  old-cnn       the earlier Rotation CNN, its world quaternion converted to the canonical frame WITH the true
                dimensions (a favour: the old network cannot know the axis order)
  <name>        each --model NAME=DIR (a trained SymRotationCNN)
  <name>+tta    the same, averaged over the 24 grid rotations of the input (test-time augmentation)
  <name>+pca    gate: take the PCA estimate when it agrees with the network within --gate degrees, else the
                network (PCA is very precise on an uncut box, the network survives cuts)
Writes --out (per-method summary, per-cut breakdown, per-sample errors for paired tests).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_type_cnn import mcnemar_exact, placements_by_voxel_path, wilson  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402
from training.manifest import load_manifest  # noqa: E402
from training.train_rotation_sym import SymRotationCNN  # noqa: E402

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"


def project_to_rotation(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def load_net(path: Path, device):
    cfg = json.loads((path.parent / "config.json").read_text())
    from training.field_input import CHANNELS
    net = SymRotationCNN(cfg["rep"], in_channels=CHANNELS[cfg.get("input", "occupancy")])
    net.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state_dict"])
    return net.to(device).eval(), cfg


def _encode(volumes, input_kind: str):
    from training.field_input import encode
    if input_kind == "occupancy":
        return np.stack(volumes).astype(np.float32) * 2 - 1
    return np.stack([encode(v.astype(np.float32) * 2 - 1, input_kind) for v in volumes])


@torch.no_grad()
def predict_matrices(net, volumes, device, batch=8, input_kind: str = "occupancy"):
    out = []
    for i in range(0, len(volumes), batch):
        x = torch.from_numpy(_encode(volumes[i:i + batch], input_kind))
        if input_kind == "occupancy":
            x = x.unsqueeze(1)
        out.append(net(x.to(device)).double().cpu().numpy())
    return np.concatenate(out)


def tta(net, volumes, device, input_kind: str = "occupancy"):
    """(N, 3, 3): per volume, the consensus of the 24 grid-rotated predictions mapped back (g^T P_g)."""
    votes = np.stack([np.einsum("ji,njk->nik", g, predict_matrices(net, [rs.rotate_volume(v, g) for v in volumes], device, input_kind=input_kind))
                      for g in rs.GROUP], axis=1)                                  # (N, 24, 3, 3)
    d2 = rs.GROUP[[k for k in range(24) if rs.admissible_group([3.0, 2.0, 1.0])[k]]]   # half-turns: generic box
    out = []
    for v in votes:
        pair = np.array([[np.min(np.degrees(rs.geodesic_angle(a[None], b[None] @ d2))) for b in v] for a in v])
        centre = v[int(np.argmin(pair.sum(axis=1)))]
        aligned = []
        for m in v:
            best = d2[int(np.argmin(rs.geodesic_angle(centre[None], m[None] @ d2)))]
            aligned.append(m @ best)
        out.append(project_to_rotation(np.mean(aligned, axis=0)))
    return np.stack(out)


def summarize(err: np.ndarray, cuts: np.ndarray, kind=None, noisy=None) -> dict:
    def block(e):
        n = len(e)
        return dict(n=n, median=float(np.median(e)), mean=float(e.mean()),
                    **{f"within{t}": float((e < t).mean()) for t in (5, 10, 20)},
                    within10_ci95=wilson(int((e < 10).sum()), n))
    out = dict(all=block(err))
    for c in (0, 1, 2):
        if (cuts == c).any():
            out[f"cuts{c}"] = block(err[cuts == c])
    if kind is not None:
        for k in ("plain", "cut", "carved"):
            if (kind == k).any():
                out[f"kind_{k}"] = block(err[kind == k])
    if noisy is not None and (~noisy).any() and noisy.any():
        out["noisy"], out["clean"] = block(err[noisy]), block(err[~noisy])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=DIR", help="dir holding best_model.pth + config.json")
    ap.add_argument("--old", type=Path, default=ROOT / "checkpoints/rotation_cnn/best_model.pth")
    ap.add_argument("--gate", type=float, default=10.0)
    ap.add_argument("--tta", nargs="*", default=[], help="model names to also evaluate with test-time augmentation")
    ap.add_argument("--sets", nargs="+", default=["test", "test_ff"])
    ap.add_argument("--align", choices=["pca", "frame"], default="pca",
                    help="alignment applied before a PCA-aligned residual network (rotation_symmetry.align_volume)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data/rotation_eval.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recs, placements = [], {}
    for name in args.sets:
        placements.update(placements_by_voxel_path(DATA / name))
        recs += load_manifest(DATA / name, require_voxel=True)
    recs = [r for r in recs if r.topology == "box" and not (placements.get(str(r.voxel_path)) or {}).get("full_field")]
    vols = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in recs]
    cuts = np.array([len((placements.get(str(r.voxel_path)) or {}).get("cuts", [])) for r in recs])
    kind = np.array(["carved" if "carve" in (placements.get(str(r.voxel_path)) or {}) else "cut" if c else "plain"
                     for r, c in zip(recs, cuts)])
    noisy = np.array([(placements.get(str(r.voxel_path)) or {}).get("noise", True) for r in recs])
    targets = [rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")], [r.parameters[k] for k in rs.DIM_KEYS]) for r in recs]
    M = np.stack([t[0] for t in targets]); mask = np.stack([t[1] for t in targets])
    print(f"{len(recs)} test boxes; cuts 0/1/2: {[(cuts == c).sum() for c in (0, 1, 2)]}; "
          f"admissible group sizes: {dict(zip(*np.unique(mask.sum(1), return_counts=True)))}")

    est = {}
    rng = np.random.default_rng(0)
    est["random"] = np.stack([project_to_rotation(rng.normal(size=(3, 3))) for _ in recs])
    pca = [rs.pca_orientation(v) for v in vols]
    est["pca"] = np.stack([p if p is not None else np.eye(3) for p in pca])
    if args.old.exists():
        from training.train_rotation_cnn import RotationCNN
        from training.topology_schema import TOPOLOGY_TO_IDX
        old = RotationCNN().to(device).eval()
        old.load_state_dict(torch.load(args.old, map_location=device, weights_only=True)["model_state_dict"])
        qs = []
        with torch.no_grad():
            for i in range(0, len(vols), 8):
                chunk = vols[i:i + 8]
                x = torch.from_numpy(np.stack(chunk).astype(np.float32) * 2 - 1).unsqueeze(1).to(device)
                qs.append(old(x, torch.full((len(chunk),), TOPOLOGY_TO_IDX["box"], device=device)).cpu().numpy())
        q = np.concatenate(qs)
        est["old-cnn"] = np.stack([rs.canonical_orientation(rs.quat_to_matrix(qq), [r.parameters[k] for k in rs.DIM_KEYS])
                                   for qq, r in zip(q, recs)])
    for spec in args.model:
        name, directory = spec.split("=")
        net, cfg = load_net(Path(directory) / "best_model.pth", device)
        ik = cfg.get("input", "occupancy")
        if cfg.get("align"):                                    # PCA-aligned residual model: estimate = E @ residual
            aligned = [rs.align_volume(v, args.align) for v in vols]
            est[name] = np.einsum("nij,njk->nik", np.stack([e for _, e in aligned]),
                                  predict_matrices(net, [a for a, _ in aligned], device, input_kind=ik))
            continue
        est[name] = predict_matrices(net, vols, device, input_kind=ik)
        if name in args.tta:
            est[name + "+tta"] = tta(net, vols, device, input_kind=ik)
    for name in [n for n in est if n not in ("random", "pca", "old-cnn")]:
        gated = est[name].copy()
        for i, p in enumerate(est["pca"]):
            if pca[i] is not None and np.degrees(rs.symmetric_error(p[None], est[name][i][None], rs.admissible_group([3.0, 2.0, 1.0])[None])[0]) < args.gate:
                gated[i] = p
        est[name + "+pca"] = gated

    errors = {name: rs.symmetric_error(e, M, mask) for name, e in est.items()}
    result = dict(n=len(recs), cuts={str(c): int((cuts == c).sum()) for c in (0, 1, 2)}, gate_deg=args.gate,
                  methods={n: summarize(e, cuts, kind, noisy) for n, e in errors.items()},
                  errors={n: e.round(3).tolist() for n, e in errors.items()})
    args.out.write_text(json.dumps(result, indent=1))
    print(f"{'method':22s} {'median':>7s} {'mean':>7s} {'<5':>6s} {'<10':>6s} {'<20':>6s} | median by cuts 0/1/2 | by kind plain/cut/carved | noisy / clean")
    for n, s in result["methods"].items():
        a = s["all"]
        print(f"{n:22s} {a['median']:7.1f} {a['mean']:7.1f} {a['within5']:6.2f} {a['within10']:6.2f} {a['within20']:6.2f} | "
              + " / ".join(f"{s[f'cuts{c}']['median']:.1f}" for c in (0, 1, 2) if f"cuts{c}" in s)
              + " | " + " / ".join(f"{s[f'kind_{k}']['median']:.1f}" for k in ("plain", "cut", "carved") if f"kind_{k}" in s)
              + " | " + (f"{s['noisy']['median']:.1f} / {s['clean']['median']:.1f}" if "noisy" in s else "-"))


if __name__ == "__main__":
    main()
