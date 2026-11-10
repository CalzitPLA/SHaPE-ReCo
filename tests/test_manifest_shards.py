"""training.manifest resolves a record's paths against the metadata file's own output root, so a dataset
generated in parallel shards (dataset_root/shardK/<topology>/...) loads like an ordinary one."""

import json
from pathlib import Path

import numpy as np

from training.manifest import load_manifest


def write_sample(output_root: Path, topology: str, name: str) -> dict:
    (output_root / topology).mkdir(parents=True, exist_ok=True)
    stl, binp = output_root / topology / f"{name}.stl", output_root / topology / f"{name}.bin"
    stl.write_text("solid x")
    np.zeros(8, dtype=np.uint8).tofile(binp)
    record = dict(index=0, mode="type", topology=topology, parameters={}, rotation={}, scale_mm=10.0,
                  stl_path=f"{topology}/{name}.stl", seed=1, voxel_path=f"{topology}/{name}.bin")
    with (output_root / topology / "dataset_metadata.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def test_ordinary_layout_still_loads(tmp_path):
    write_sample(tmp_path, "box", "b0")
    write_sample(tmp_path, "sphere", "s0")
    records = load_manifest(tmp_path)
    assert sorted(r.topology for r in records) == ["box", "sphere"]
    assert all(r.voxel_path.is_file() for r in records)


def test_sharded_layout_loads(tmp_path):
    for k in range(3):
        write_sample(tmp_path / f"shard{k}", "box", f"b{k}")
        write_sample(tmp_path / f"shard{k}", "sphere", f"s{k}")
    records = load_manifest(tmp_path)
    assert len(records) == 6
    assert {r.voxel_path.parent.parent.name for r in records} == {"shard0", "shard1", "shard2"}


def test_a_missing_voxel_file_is_still_skipped(tmp_path):
    record = write_sample(tmp_path / "shard0", "box", "b0")
    (tmp_path / "shard0" / record["voxel_path"]).unlink()
    assert load_manifest(tmp_path) == []
