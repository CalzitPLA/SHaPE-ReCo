"""training/subclass_schema.py: the 5-parent / 24-subclass label space of the hierarchical TypeNet, the label
derivation for new (explicit `subclass`) and old in-scale records, the parent mask, and skipping of other topologies."""
import json
from pathlib import Path

import numpy as np
import pytest

from training import subclass_schema as ss
from training.manifest import load_manifest


def test_flat_order_is_order_of_appearance():
    assert ss.NUM_PARENTS == 5 and ss.NUM_SUBCLASSES == 24
    assert ss.FLAT_SUBCLASSES[:5] == ("box/full", "box/hollow", "box/ibeam", "box/rhs", "box/frame")
    assert ss.FLAT_SUBCLASSES[5] == "cylinder/full" and ss.FLAT_SUBCLASSES[-1] == "torus/frame"
    assert ss.FLAT_SUBCLASSES.index("cone/frame") == 13


def test_mask_rows_partition_the_subclasses():
    mask = ss.parent_mask()
    assert mask.shape == (5, 24) and mask.dtype == bool
    assert mask.sum(1).tolist() == [5, 5, 4, 5, 5]
    assert (mask.sum(0) == 1).all()                    # every subclass has exactly one parent
    for s, p in enumerate(ss.SUB_PARENT):
        assert mask[p, s] and ss.FLAT_SUBCLASSES[s].startswith(ss.PARENTS[p] + "/")
    np.testing.assert_array_equal(mask, ss.PARENT_MASK)


def test_parents_are_dataset_gen_topologies():
    from training.topology_schema import TOPOLOGY_TO_IDX, labels_of
    assert labels_of is ss.labels_of
    assert all(p in TOPOLOGY_TO_IDX for p in ss.PARENTS)


@pytest.mark.parametrize("topology,subclass", [("box", "ibeam"), ("box", "frame"), ("cylinder", "tube"),
                                               ("sphere", "band"), ("torus", "sector"), ("cone", "hollow")])
def test_new_records_use_their_subclass(topology, subclass):
    parent, flat = ss.labels_of({"topology": topology, "subclass": subclass,
                                 "placement": {"hollow": {"wall_voxels": 3}}, "segment": {"u": [0, 1]}})
    assert ss.PARENTS[parent] == topology
    assert ss.FLAT_SUBCLASSES[flat] == f"{topology}/{subclass}"


def test_old_in_scale_records_are_derived():
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "box", "placement": {"size_vox": 30}})[1]] == "box/full"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "box", "placement": {"hollow": {"wall_voxels": 4}}})[1]] == "box/hollow"
    # a failed hollowing is a full body
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "sphere", "placement": {"hollow_failed": True}})[1]] == "sphere/full"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "cylinder", "segment": {"u": [0.0, 3.0]}})[1]] == "cylinder/sector"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "sphere", "segment": {"v": [-0.5, 0.5]}})[1]] == "sphere/band"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "torus", "segment": {"u": [0, 2], "v": [0, 1]}})[1]] == "torus/sector"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "cone", "sample_kind": "segment"})[1]] == "cone/sector"
    assert ss.FLAT_SUBCLASSES[ss.labels_of({"topology": "cone", "sample_kind": "whole"})[1]] == "cone/full"
    assert ss.labels_of({"topology": "torus"}) == (4, ss.FLAT_SUBCLASSES.index("torus/full"))


def test_invalid_subclass_for_the_parent_raises():
    with pytest.raises(ValueError):
        ss.labels_of({"topology": "cylinder", "subclass": "band"})
    with pytest.raises(ValueError):
        ss.labels_of({"topology": "cone", "segment": {"v": [0, 1]}})


def test_unknown_topologies_are_skipped_with_a_count():
    records = [{"topology": "box"}, {"topology": "cube"}, {"topology": "cube"}, {"topology": "torus_segment"},
               {"topology": "cylinder", "subclass": "band"}, {"topology": "sphere", "subclass": "frame"}]
    with pytest.raises(ss.UnknownTopology):
        ss.labels_of(records[1])
    kept, skipped = ss.filter_records(records)
    assert [r["topology"] for r in kept] == ["box", "sphere"]
    assert skipped == {"topology:cube": 2, "topology:torus_segment": 1, "subclass:cylinder/band": 1}
    assert ss.subclass_counts(records) == {"box/full": 1, "sphere/frame": 1}


def test_labels_from_manifest_records(tmp_path):
    """load_manifest keeps the extra JSONL keys (Record.meta), which is where subclass/segment live."""
    lines = []
    for i, extra in enumerate([dict(topology="box", subclass="ibeam", member_kind=None),
                               dict(topology="sphere", segment={"v": [0, 1]}),
                               dict(topology="box", placement={"hollow": {"wall_voxels": 2}}),
                               dict(topology="polycone")]):
        (tmp_path / f"s{i}.stl").write_bytes(b"solid\n")
        (tmp_path / f"s{i}.bin").write_bytes(b"\0" * 8)
        lines.append(json.dumps(dict(index=i, mode="type", parameters={}, rotation={}, stl_path=f"s{i}.stl",
                                     voxel_path=f"s{i}.bin", **extra)))
    (tmp_path / "dataset_metadata.jsonl").write_text("\n".join(lines) + "\n")
    records = load_manifest(Path(tmp_path))
    assert records[0].meta["subclass"] == "ibeam" and "topology" not in records[0].meta
    kept, skipped = ss.filter_records(records)
    assert [ss.FLAT_SUBCLASSES[ss.labels_of(r)[1]] for r in kept] == ["box/ibeam", "sphere/band", "box/hollow"]
    assert skipped == {"topology:polycone": 1}
