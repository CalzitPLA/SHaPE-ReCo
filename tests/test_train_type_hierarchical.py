"""training/train_type_cnn.py --hierarchical end to end on a synthetic 16^3 dataset (CPU, 2 epochs): the masked
subclass CE, both network inputs (occupancy and vf_exact from a fake .field.npz), the checkpoint round trip through
reco_engine (a hierarchical checkpoint still answers the old 17-way type interface, plus `subclass`), --oversample,
and model_card.json."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training import train_type_cnn as ttc
from training.subclass_schema import FLAT_SUBCLASSES, PARENT_MASK, PARENTS
from training.topology_schema import NUM_TOPOLOGIES, TOPOLOGY_TO_IDX

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.field_channels import GridSpec, save_field_npz  # noqa: E402

R = 16
SAMPLES = [("box", "full"), ("box", "ibeam"), ("box", "hollow"), ("cylinder", "full"), ("cylinder", "tube"),
           ("sphere", "band"), ("torus", "sector"), ("cone", "frame"), ("box", None), ("polycone", None)]


def _write_root(root: Path, seed: int, samples=SAMPLES):
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    lines = []
    for i, (topology, subclass) in enumerate(samples):
        occ = (rng.random((R, R, R)) > 0.6).astype(np.uint8)
        occ.tofile(root / f"s{i}.bin")
        (root / f"s{i}.stl").write_bytes(b"solid fake\nendsolid fake\n")
        channels = np.concatenate([occ[None].astype(np.float32), rng.normal(0, 1, (4, R, R, R)).astype(np.float32)])
        save_field_npz(root / f"s{i}.field.npz", channels, GridSpec((-20.0,) * 3, (20.0,) * 3, R), source="mesh")
        rec = dict(index=i, mode="type", topology=topology, parameters={}, rotation={}, stl_path=f"s{i}.stl",
                   voxel_path=f"s{i}.bin", placement={"cuts": [1]} if i % 3 == 0 else {})
        if subclass:
            rec["subclass"] = subclass
        lines.append(json.dumps(rec))
    (root / "dataset_metadata.jsonl").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    base = tmp_path_factory.mktemp("hier_data")
    return dict(train=_write_root(base / "train", 0), val=_write_root(base / "val", 1), test=_write_root(base / "test", 2))


def _train(roots, out: Path, *extra):
    argv = ["--dataset-root", str(roots["train"]), "--val-roots", str(roots["val"]), "--test-roots", str(roots["test"]),
            "--checkpoint-dir", str(out), "--resolution", str(R), "--epochs", "2", "--batch-size", "4",
            "--num-workers", "0", "--device", "cpu", "--patience", "0", "--lr", "1e-3", *extra]
    ttc.main(argv)
    return out


def test_masked_subclass_ce_leaves_other_parents_logits_alone():
    torch.manual_seed(0)
    parent_logits = torch.randn(6, 5, requires_grad=True)
    sub_logits = torch.randn(6, 24, requires_grad=True)
    parent = torch.tensor([0, 1, 2, 3, 4, 0])
    sub = torch.tensor([FLAT_SUBCLASSES.index(n) for n in
                        ("box/ibeam", "cylinder/tube", "cone/frame", "sphere/band", "torus/sector", "box/full")])
    loss, p_loss, s_loss = ttc.hierarchical_loss(parent_logits, sub_logits, parent, sub, sub_weight=1.0)
    assert torch.isfinite(loss) and torch.isfinite(s_loss)
    loss.backward()
    grad = sub_logits.grad
    own = torch.as_tensor(PARENT_MASK)[parent]
    assert (grad[~own] == 0).all()                          # other parents' subclass logits get no gradient
    assert (grad[own].abs() > 0).any(-1).all() if own.any() else True
    # the subclass CE equals a plain CE over the true parent's subclasses only
    row = 0
    idx = PARENT_MASK[parent[row]].nonzero()[0]
    ref = torch.nn.functional.cross_entropy(sub_logits[row:row + 1, idx], torch.tensor([list(idx).index(sub[row].item())]))
    single = torch.nn.functional.cross_entropy(ttc.masked_subclass_logits(sub_logits[:1], parent[:1]), sub[:1])
    assert torch.allclose(ref, single)


def test_forward_is_the_old_17_way_interface():
    model = ttc.HierTypeCNN(in_channels=1).eval()
    x = torch.randn(2, 1, R, R, R)
    with torch.no_grad():
        logits17 = model(x)
        parent, sub = model.heads(x)
    assert logits17.shape == (2, NUM_TOPOLOGIES) and sub.shape == (2, 24)
    cols = [TOPOLOGY_TO_IDX[p] for p in PARENTS]
    assert torch.equal(logits17[:, cols], parent)
    others = [i for i in range(NUM_TOPOLOGIES) if i not in cols]
    assert torch.isinf(logits17[:, others]).all()
    probs = torch.softmax(logits17, 1)
    assert torch.allclose(probs.sum(1), torch.ones(2)) and (probs[:, others] == 0).all()


def test_oversample_parsing_and_matching(roots):
    from training.manifest import load_manifest
    records = load_manifest(roots["train"])
    terms = ttc.parse_oversample(["subclass=hollow:3,cut=1"], default_factor=2)
    assert terms == [("subclass", "hollow", 3), ("cut", "1", 2)]
    factors = ttc.oversample_factors(records, terms)
    by_name = {r.stl_path.stem: f for r, f in zip(records, factors)}
    assert by_name["s2"] == 3                                 # box/hollow
    assert by_name["s0"] == 2 and by_name["s3"] == 2          # cuts present (i % 3 == 0)
    assert by_name["s1"] == 1
    assert len(ttc.oversampled_indices(records, terms)) == sum(factors)
    with pytest.raises(ValueError):
        ttc.parse_oversample(["subclass"])


@pytest.mark.parametrize("input_kind,channels", [("occupancy", 1), ("vf_exact", 5)])
def test_train_two_epochs_checkpoint_and_model_card(roots, tmp_path, input_kind, channels):
    out = _train(roots, tmp_path / "type_cnn", "--hierarchical", "--input", input_kind, "--oversample", "subclass=hollow:2")
    hist = json.loads((out / "training_history.json").read_text())
    assert len(hist["train_loss"]) == 2
    for key in ("train_loss", "val_loss", "train_sub_loss", "val_acc", "val_sub_acc", "val_top2", "val_sub_acc_pred_parent"):
        assert all(np.isfinite(hist[key])), key
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["hierarchical"] is True and cfg["input"] == input_kind
    assert cfg["parents"] == list(PARENTS) and cfg["subclasses"]["box"] == ["full", "hollow", "ibeam", "rhs", "frame"]
    assert cfg["n_train"] == 9 and cfg["n_train_draws"] == 10                    # polycone skipped, box/hollow doubled

    card = json.loads((out / "model_card.json").read_text())
    assert card["input_kind"] == input_kind and card["seed"] == 42 and card["hierarchical"]
    assert "commit" in card["git"]
    assert card["counts"]["train"]["box/full"] == 2 and card["counts"]["train"]["box/ibeam"] == 1
    m = card["metrics"][str(roots["test"])]
    assert m["n"] == 9 and m["skipped"] == {"topology:polycone": 1}
    for key in ("parent_acc", "parent_top2", "sub_acc", "sub_acc_pred_parent"):
        assert 0.0 <= m[key] <= 1.0
    assert m["parent_top2"] >= m["parent_acc"] and m["sub_acc"] >= m["sub_acc_pred_parent"]
    assert set(m["confusion_ibeam_vs_box"]) <= {"box/full", "box/ibeam"}
    assert set(m["confusion_segment_vs_full"]) <= set(PARENTS)

    # checkpoint round trip: through reco_engine's loader (given a plain TypeCNN, as the cascade does)
    from reco_engine import model_registry
    net = model_registry._load_one(ttc.TypeCNN(in_channels=channels), out / "best_model.pth", torch.device("cpu"))
    assert isinstance(net, ttc.HierTypeCNN)
    ref = ttc.load_checkpoint_model(out / "best_model.pth", channels, torch.device("cpu"))
    x = torch.randn(1, channels, R, R, R)
    with torch.no_grad():
        assert torch.equal(net(x), ref(x))


def test_predict_type_gives_parent_top5_and_subclass(roots, tmp_path):
    out = _train(roots, tmp_path / "type_cnn", "--hierarchical", "--input", "occupancy")   # the predict API, not the input
    from reco_engine import model_registry
    from reco_engine.predict import TypePrediction, _predict_type_in_process
    net = model_registry._load_one(ttc.TypeCNN(in_channels=1), out / "best_model.pth", torch.device("cpu"))
    models = SimpleNamespace(type_cnn=net, device=torch.device("cpu"), allowed_topologies=None, type_input_kind="occupancy")
    vol = np.where(np.random.default_rng(3).random((R, R, R)) > 0.5, 1.0, -1.0).astype(np.float32)
    pred = _predict_type_in_process(models, vol)
    assert pred.topology in PARENTS and {t for t, _ in pred.top5} <= set(PARENTS) and len(pred.top5) == 5
    assert abs(sum(p for _, p in pred.top5) - 1.0) < 1e-5
    assert pred.subclass == pred.subclasses[pred.topology][0][0]
    for parent, subs in pred.subclasses.items():
        assert abs(sum(p for _, p in subs) - 1.0) < 1e-5
    # masking still works on the 17-way space
    models.allowed_topologies = ("box", "sphere")
    pred = _predict_type_in_process(models, vol)
    assert {t for t, _ in pred.top5} <= {"box", "sphere"}
    # the old constructor (no subclass) still works
    assert TypePrediction("box", 1.0, [("box", 1.0)]).subclass is None


def test_non_hierarchical_path_is_unchanged(roots, tmp_path):
    """No new config keys except the input contract's resolution (R-05c), a plain TypeCNN checkpoint (no sub head),
    records of all 17 topologies accepted."""
    out = tmp_path / "type_cnn"
    ttc.main(["--dataset-root", str(roots["train"]), "--checkpoint-dir", str(out), "--resolution", str(R), "--epochs", "1",
              "--batch-size", "2", "--num-workers", "0", "--device", "cpu", "--val-fraction", "0.3"])
    cfg = json.loads((out / "config.json").read_text())
    assert list(cfg) == ["input", "sdf_truncation", "field_resolution", "max_per_class", "roots", "epochs", "lr", "batch_size",
                         "weight_decay", "seed", "val_fraction", "n_train", "n_val", "train_per_class", "val_per_class",
                         "resolution"]
    assert cfg["resolution"] == R
    state = torch.load(out / "best_model.pth", weights_only=True)["model_state_dict"]
    assert not ttc.is_hierarchical_state(state) and state["classifier.weight"].shape[0] == NUM_TOPOLOGIES
    assert not (out / "model_card.json").exists()
