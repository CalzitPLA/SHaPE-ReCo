"""scripts/dataset_hygiene.py: the filtered view keeps every record but the cones under the taper limit, and
training/manifest.load_manifest reads the view like the source root."""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sample(d: Path, stem: str, topology: str, dims: dict, subclass: str = "full") -> dict:
    (d / topology).mkdir(parents=True, exist_ok=True)
    for ext in (".stl", ".bin"):
        (d / topology / f"{stem}{ext}").write_bytes(b"x")
    return dict(topology=topology, subclass=subclass, mode="type", parameters={}, rotation={}, dims_mm=dims,
                stl_path=f"{topology}/{stem}.stl", voxel_path=f"{topology}/{stem}.bin", placement=dict(grid_mm=40.0, voxels=999))


def test_view_drops_only_unresolvable_cones(tmp_path):
    src = tmp_path / "set" / "train" / "inc1"
    shard = src / "shard200"
    pitch = 40.0 / 64
    entries = [
        _sample(shard, "a", "cone", dict(half_length=10.0, radius1=10.0, radius2=10.0 - 1.9 * pitch)),          # dropped
        _sample(shard, "b", "cone", dict(half_length=10.0, radius1=10.0, radius2=10.0 - 2.1 * pitch)),          # kept
        _sample(shard, "c", "cylinder", dict(half_length=10.0, radius=10.0)),                                    # kept
        _sample(shard, "d", "cone", dict(half_length=10.0, radius1=4.0, radius2=4.0 + 0.5 * pitch), "sector"),  # dropped
    ]
    (shard / "dataset_metadata.jsonl").write_text(json.dumps(dict(record_type="dataset_header", grid_mm=40.0)) + "\n")
    for topo in ("cone", "cylinder"):
        (shard / topo / "dataset_metadata.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in entries if e["topology"] == topo))
    out = tmp_path / "view"
    subprocess.run([sys.executable, str(REPO / "scripts" / "dataset_hygiene.py"), "--src", str(src), "--out", str(out),
                    "--base", str(tmp_path / "set")], check=True)

    report = json.loads((out / "hygiene_report.json").read_text())["roots"]["train/inc1"]
    assert report["kept"] == 2 and report["dropped"] == 2
    assert report["dropped_by_label"] == {"cone/full": 1, "cone/sector": 1}
    # the header file is mirrored, the sample files are not copied
    assert (out / "train/inc1/shard200/dataset_metadata.jsonl").is_file()
    assert not list(out.rglob("*.bin"))

    sys.path.insert(0, str(REPO))
    from training.manifest import load_manifest
    view = load_manifest(out / "train/inc1")
    assert sorted(r.voxel_path.name for r in view) == ["b.bin", "c.bin"]
    assert all(r.voxel_path.is_file() and src in r.voxel_path.parents for r in view)        # the source's own files
    assert len(load_manifest(src)) == 4                                                       # the source is untouched


def test_zero_limit_keeps_everything(tmp_path):
    src = tmp_path / "set" / "val"
    e = _sample(src, "a", "cone", dict(half_length=10.0, radius1=10.0, radius2=10.0))
    (src / "cone" / "dataset_metadata.jsonl").write_text(json.dumps(e) + "\n")
    out = tmp_path / "view"
    subprocess.run([sys.executable, str(REPO / "scripts" / "dataset_hygiene.py"), "--src", str(src), "--out", str(out),
                    "--min-cone-taper-vox", "0"], check=True)
    assert json.loads((out / "hygiene_report.json").read_text())["roots"]["val"]["dropped"] == 0
