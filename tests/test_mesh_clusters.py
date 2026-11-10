"""reco_field/peel.py's `mesh_clusters` option (docs/FRAME_FAILURE.md, "Full-mesh component clustering"):
full-mesh triangle-connectivity clustering (`trimesh.Trimesh.split`) as an opt-in alternative to
`_side_clusters`'s Euclidean-distance clustering of the sparse sample. Off by default - the ordinary
peel()/peel_steps() behaviour must not change at all when it is left False (checked directly here and,
more broadly, against all 41 example parts in docs/data/peel_examples.json, unaffected by this feature)."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import _mesh_clusters, peel, peel_steps  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402


def box_with_a_hole_and_a_boss():
    """A box with a through-hole cut (cutter material) AND a small boss added on the side (addition
    material) - unlike `two_bosses_on_a_slab` (additions only), this has BOTH sides so cuts_first's
    ordering actually means something."""
    box = trimesh.creation.box(extents=(30.0, 20.0, 10.0))
    hole = trimesh.creation.cylinder(radius=2.0, height=12.0)
    hole.apply_translation((-8.0, 0.0, 0.0))
    cut = box.difference(hole)
    boss = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    boss.apply_translation((8.0, 0.0, 6.5))
    mesh = cut.union(boss)
    return SurfaceField(mesh.vertices, mesh.faces)


def two_bosses_on_a_slab():
    """A slab with two small, separate cube bosses far apart on its top face - one connected mesh (the
    bosses share the slab's material), but the ADDED material (outside the slab's own box) is two distinct,
    topologically unconnected components."""
    slab = trimesh.creation.box(extents=(40.0, 20.0, 4.0))
    boss1 = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    boss1.apply_translation((-15.0, 0.0, 3.5))
    boss2 = trimesh.creation.box(extents=(3.0, 3.0, 3.0))
    boss2.apply_translation((15.0, 0.0, 3.5))
    mesh = trimesh.util.concatenate([slab, boss1, boss2])
    mesh = mesh.process(validate=True)
    field = SurfaceField(mesh.vertices, mesh.faces)
    slab_model = sp.Box((20.0, 10.0, 2.0))
    return field, slab_model


def test_mesh_clusters_finds_the_two_separate_bosses_as_distinct_components():
    field, slab_model = two_bosses_on_a_slab()
    tol = 1e-3 * field.area ** 0.5
    components = _mesh_clusters(slab_model, field, tol, extent=40.0, min_faces=3)
    additions = [(samples, n) for samples, inside, n in components if not inside]
    assert len(additions) == 2
    for samples, n in additions:
        assert len(samples.surface) > 0


def test_mesh_clusters_off_by_default_does_not_change_peel_result():
    """The exact property the 41-example regression check (docs/data/peel_examples.json) verifies at scale:
    leaving `mesh_clusters` at its default False must be identical to not having the parameter at all."""
    field, _slab_model = two_bosses_on_a_slab()
    default = peel(field, max_ops=6)
    explicit_off = peel(field, max_ops=6, mesh_clusters=False)
    assert default.explained == explicit_off.explained
    assert [o.fit.kind for o in default.ops] == [o.fit.kind for o in explicit_off.ops]


def test_mesh_clusters_mode_reconstructs_the_slab_with_both_bosses():
    field, _slab_model = two_bosses_on_a_slab()
    result = peel(field, max_ops=6, mesh_clusters=True)
    assert result.explained > 0.9
    kinds = [o.fit.kind for o in result.ops]
    assert kinds.count("box") >= 2                     # the slab base plus at least one boss add


def test_mesh_clusters_cuts_first_tries_cutters_before_additions():
    """A regression guard for a real inversion the ultrareview cloud review caught: the cuts_first sort
    inside the mesh_clusters branch used `(c[1], -c[2])`, which - since `c[1]` (inside) is True for a
    cutter - sorted ADDITIONS first, the opposite of what cuts_first (on by default here, since
    bounding_box=True) promises and what `_side_clusters`/the octree branch both actually do."""
    field = box_with_a_hole_and_a_boss()
    result = peel(field, max_ops=4, mesh_clusters=True, bounding_box=True)
    first_op = result.ops[1].op                         # ops[0] is the base; the first REAL step
    assert first_op in ("cut", "clip")                  # both take material away: a plane clip trims the bounding box
    assert "add" in [o.op for o in result.ops[2:]]      # and the addition comes after them


def test_mesh_clusters_yields_real_cluster_indices_not_none():
    """The other half of the same review finding: `cluster`/`candidate`/`rejected` PeelSteps in mesh_clusters
    mode used to always carry `cluster=None` (no indices into `s.surface` existed for a mesh component, a
    freshly-drawn dense sample), which silently blanked the GUI's Cluster panel and 3D highlight for every
    step of a mesh_clusters run. They should now carry a real (if approximate) index array instead."""
    field = box_with_a_hole_and_a_boss()
    seen_a_populated_cluster = False
    for step in peel_steps(field, max_ops=3, mesh_clusters=True):
        if step.stage in ("cluster", "candidate", "rejected") and step.cluster is not None:
            seen_a_populated_cluster = True
            assert len(step.cluster) > 0
    assert seen_a_populated_cluster
