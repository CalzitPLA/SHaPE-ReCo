#!/usr/bin/env python3
"""
The ONE field encoding of a solid on a voxel grid, used both to write training samples (dataset_generator.py
--with-field -> `<sample>.field.npz`) and by the residual add-cut loop at inference (reco_engine/residual_loop.py), so
a network never sees a differently computed input than the one it was trained on.

Five channels per voxel centre, in this order (`CHANNELS`):

  occupancy   1.0 inside, 0.0 outside                       = (sdf > 0)
  sdf         signed distance to the surface, mm            POSITIVE inside, NEGATIVE outside
  vx, vy, vz  V(x) = c(x) - x, mm                           c(x) the nearest surface point; |V| = |sdf|

Sign CONVENTION = `closest_point_field.Query.signed_distance` (positive inside), the plan's choice for the stored channels.
The sign itself is COMPUTED by scanline ray parity (`parity_occupancy`), not from the nearest face normal: that one was
measured wrong in 8.7 % of the voxel centres of a FreeCAD cylinder with the fast query, and still wrong near edges and
apexes with the exact one (see `mesh_channels`). `sdf_primitives` and training/field_input.py's "fused" kind use the
OPPOSITE sign (negative inside); `network_input` flips to theirs, so a "vf_exact" network input has exactly the "fused"
layout, sign and scaling (occupancy, sdf, V; array frame) - only computed from the mesh instead of the grid.

Array axes are WORLD axes: `channels[:, i, j, k]` belongs to the voxel centre `spec.centres()[i, j, k]` = (x_i, y_j, z_k),
the convention of `reco_field.peel._grid_centres` - NOT the stltovoxel/.bin order [z, -y, x] (docs/FRAME_FAILURE.md,
"Voxels as the ground truth occupancy source"; `reco_field.peel.real_voxel_occupancy` for the y flip). A stored .bin
grid converts with `bin_to_world(raw)`; `network_input` converts the other way, into the array frame the rotation
labels and the 24-element grid augmentation of training/train_rotation_sym.py live in.

Open meshes (`SurfaceField.is_closed()` False) get the same parity sign - as good as any ray-parity test there. Not the
stltovoxel path: that needs a temporary file and an external process per call; the scanline parity is in-process and
evaluates exactly the voxel centres. It does NOT equal an stltovoxel .bin: stltovoxel rounds its anchor to whole voxels,
so its grid sits up to half a voxel off (scripts/verify_inscale_dataset.py therefore only asks for >= 99 % agreement and
no disagreement farther than one voxel from the surface; IoU 0.84 median on v5 small, docs/training_plan_v5.tex F7).
Since D-vox (2026-09-23) this parity is the ground truth; voxelization._voxelize_parity writes it as the .bin when
SHAPE_RECO_VOXELIZER=parity (not yet the default: task T-19).

`parity_axes` is the ONE parity implementation. Its callers: `parity_occupancy` (here), `parity_of_file` (the generator's
parity .bin, inside bits and field sign), reco_engine.residual_loop.parity_occupancy (the loop's grid),
reco_field.peel.real_voxel_occupancy, local_grids.iter_slabs, reco_engine.run_tensors.parity_grid. Each caller shifts the
rows itself by the same 1.2345e-4 / 2.3456e-4 of a pitch (local_grids.ROW_SHIFT names the pair; the others repeat the
literals): keep them equal, or the "same grid" statement between generator and loop no longer holds.

    spec = grid_spec(-20, 20, 64)                    # the in-scale grid: 40 mm, centred on the world origin
    ch = mesh_channels(field, spec)                  # (5, 64, 64, 64) float32 from an STL/STEP mesh
    ch = voxel_channels(occ, spec)                   # the same from a boolean component (a residual, a carve bite)
    pts = surface_points(field, 2048)                # (2048, 6) xyz + outward unit normal
    x = network_input(ch, spec)                      # (5, R, R, R) normalised, array frame - what the networks read
    x = dirlog_input(ch, spec, curvature=True)       # (8, R, R, R) material, unit normal, log distance, kappa1, kappa2,
                                                     # collapse (edge/vertex) - "dirlog_exact" (5) / "dirlog_curv" (8)
    x = dirlog_residual(s_model, ch[0] > 0.5, spec)  # (7, R, R, R) input flag + the MODEL's dirlog + residual sign (add/cut)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .closest_point_field import SurfaceField, field_backend, voxel_surface_mesh

CHANNELS = ("occupancy", "sdf", "vx", "vy", "vz")
DEFAULT_TRUNCATION = 8.0         # voxels; the same truncation training/field_input.py and sdf_input.py use


@dataclass(frozen=True)
class GridSpec:
    """A cubic grid of `resolution`^3 cells over [lo, hi] (mm, world axes). Voxel centres are
    `lo + (index + 0.5) * pitch`, identical to reco_field.peel._grid_centres(lo, hi, resolution)."""
    lo: Tuple[float, float, float]
    hi: Tuple[float, float, float]
    resolution: int

    @property
    def pitch(self) -> float:
        """From the x side only: `grid_spec` guarantees a cube; a hand-built non-cubic GridSpec gets a wrong pitch."""
        return float(self.hi[0] - self.lo[0]) / self.resolution

    def centres(self) -> np.ndarray:
        """(R, R, R, 3) world coordinates of the voxel centres, indexed [i, j, k] = (x, y, z)."""
        lo, hi, r = np.asarray(self.lo, float), np.asarray(self.hi, float), self.resolution
        # the same arithmetic as peel._grid_centres (linspace edges + half a step), so the two agree bit for bit
        ax = [np.linspace(lo[i], hi[i], r + 1) for i in range(3)]
        step = np.array([a[1] - a[0] for a in ax])
        return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)[:-1, :-1, :-1] + step / 2.0

    def to_record(self) -> dict:
        return dict(lo=[float(v) for v in self.lo], hi=[float(v) for v in self.hi], resolution=int(self.resolution),
                    axes="world_xyz", channels=list(CHANNELS), sign="positive_inside", units="mm")

    @classmethod
    def from_record(cls, d) -> "GridSpec":
        if isinstance(d, np.ndarray):                      # as stored in an .npz (a 0-d object/str array)
            import json
            d = json.loads(str(d))
        return cls(tuple(float(v) for v in d["lo"]), tuple(float(v) for v in d["hi"]), int(d["resolution"]))


def grid_spec(lo, hi, resolution: int) -> GridSpec:
    """A CUBE grid covering [lo, hi]: if the box is not cubic it is widened about its centre to its largest side, so
    every voxel is a cube (the networks and the distance units assume one pitch). `lo`/`hi` may be scalars."""
    lo = np.broadcast_to(np.asarray(lo, float), (3,))
    hi = np.broadcast_to(np.asarray(hi, float), (3,))
    side = float((hi - lo).max())
    if side <= 0:
        raise ValueError(f"grid_spec: empty box lo={lo} hi={hi}")
    centre = (lo + hi) / 2.0
    return GridSpec(tuple(centre - side / 2), tuple(centre + side / 2), int(resolution))


def inscale_spec(grid_mm: float = 40.0, resolution: int = 64) -> GridSpec:
    """The in-scale dataset's grid (dataset_gen/src/inscale.py, stltovoxel anchor="origin"): centred on the world origin."""
    return grid_spec(-grid_mm / 2.0, grid_mm / 2.0, resolution)


def grid_shape(spec) -> Tuple[int, int, int]:
    """(nx, ny, nz) of a grid: `spec.shape` for a box grid (reco_engine.cascade.BoxGridSpec), else resolution^3."""
    shape = getattr(spec, "shape", None)
    if shape is not None:
        return tuple(int(n) for n in shape)
    r = int(spec.resolution)
    return (r, r, r)


def _pack(occ: np.ndarray, sdf: np.ndarray, vec: np.ndarray, r) -> np.ndarray:
    """`r`: the cube's resolution or the grid's shape (nx, ny, nz)."""
    shape = (r, r, r) if np.isscalar(r) else tuple(r)
    out = np.empty((5,) + shape, np.float32)
    out[0] = occ.reshape(shape)
    out[1] = sdf.reshape(shape)
    out[2:] = vec.reshape(shape + (3,)).transpose(3, 0, 1, 2)
    return out


REFINE_EDGE_VOXELS = 4.0         # mesh_channels: triangles are split until no edge is longer than this many voxels
REFINE_K = 64                    # ... and the one-pass query checks this many nearest centroids (SurfaceField's k)


def mesh_channels(field: SurfaceField, spec: GridSpec, exact: bool = False, sign: str = "parity",
                  inside: Optional[np.ndarray] = None) -> np.ndarray:
    """(5, R, R, R) float32 [occupancy, sdf, vx, vy, vz] of a mesh on `spec`.

    `inside` (bool (R, R, R), world axes): a precomputed `parity_occupancy(field, spec)` for the "parity" sign, so a
    caller that already has it (sample_writer via `parity_of_file`, the residual loop) does not pay for it again.
    Under SHAPE_RECO_FIELD_BACKEND=igl the query is exact on the mesh AS GIVEN (closest_point_field._query_igl), so
    the refinement below is skipped: measured 12-19 s -> 0.03-0.35 s per 64^3 grid on the CPU (docs/data/field_igl_bench.json).

    |sdf| and V: `SurfaceField.query` on a REFINED copy of the mesh (`refined_field`: every edge at most
    REFINE_EDGE_VOXELS voxels). Why: CAD tessellations are long sliver triangles, and the one-pass query (exact=False)
    proposes candidates by triangle CENTROID, which on a sliver can be far from the part of it that is nearest. Measured on
    dataset_gen/examples/cy_cyl.stl at 32^3: on the raw mesh 16.5 % of the voxel centres got a distance off by > 0.01 mm
    (max 3.2 mm, V off by up to 11.9 mm); after the split (912 -> 31k faces) max 0.027 mm, V p99 0.007 mm, 0.5 s - while
    exact=True on the raw mesh is exact but takes 12 s (32^3) to about 60-100 s (64^3), the bounding proof degenerating
    with the sliver radius. A 256-section trimesh cylinder (slivers 0.12 mm wide) still left max 0.09 mm with k = 24
    candidates after the split; with REFINE_K = 64 the max error is 0.0005 mm (1.2 s at 32^3) - on the same cylinder
    moved off the grid axes a few centres are still off by up to 0.05 mm (0.07 voxel; 99.9 % within 0.01 mm). `exact` is still passed on
    to the query (exact=True stays slow even on the split mesh: 74 s at 32^3 there).

    Sign (and the occupancy channel), `sign`:
      "parity" (default): scanline ray parity along x, `parity_occupancy` - what a voxelizer does, exact for a closed mesh
                 at every voxel centre not on the surface, independent of which face the query picked.
      "normal":  the face-normal sign of `Query.signed_distance`, as closest_point_field defines it. Kept for comparison
                 only: it needs the TRUE nearest face (with exact=False on the raw cy_cyl 8.7 % of the centres were
                 wrong), and even with the exact face it is ambiguous where the nearest point is an edge or a vertex
                 (co_cone: 1465 voxels wrong with exact=True, up to 6.4 mm from the surface, measured by the residual-loop
                 agent) - the plan's own "face-normal sign" risk, which is why parity is the default.
    The convention of the stored channel is unchanged either way: sdf POSITIVE inside.

    Second-order fields (closest_point_field.load_field with SHAPE_RECO_MESH_ORDER=2, a quadratic_mesh.QuadraticField):
    only the igl backend queries the quadratic patches. On any other backend `refined_field` rebuilds a plain SurfaceField
    from `field.vertices/faces`, which for a QuadraticField are the flat four-way subdivision of the six-node triangles,
    so the result is the first-order field of that subdivision, without a message (review R-04, F4)."""
    pts = spec.centres().reshape(-1, 3)
    src = field if field_backend() == "igl" else refined_field(field, REFINE_EDGE_VOXELS * spec.pitch)
    q = src.query(pts, exact=exact)
    if sign == "parity":
        inside = (parity_occupancy(field, spec) if inside is None else np.asarray(inside, bool)).reshape(-1)
        sdf = np.where(inside, q.distance, -q.distance)
    elif sign == "normal":
        sdf = q.signed_distance
    else:
        raise ValueError(f"sign must be 'parity' or 'normal', got {sign!r}")
    return _pack(sdf > 0, sdf, q.vector, grid_shape(spec))


def refined_field(field: SurfaceField, max_edge: float) -> SurfaceField:
    """A SurfaceField of the same surface whose triangles have no edge longer than `max_edge` (long ones split with
    trimesh.remesh.subdivide_to_size - the geometry is unchanged, only the triangulation is finer; the split mesh has
    T-junctions, so it is used for distances and V only, never for the inside test or `is_closed`), with REFINE_K
    candidates per query and NO brute-force "large triangle" list. That list is what makes SurfaceField exact on raw CAD
    meshes, but it is chosen relative to the median triangle, so on a carved box (a fine spherical bite next to big flat
    faces) hundreds of faces land on it and every voxel centre is projected onto all of them: measured 157 s for one
    64^3 grid of a 2500-face carved box. After the split no triangle is large any more."""
    tri = field.vertices[field.faces]
    longest = float(np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max()) if len(tri) else 0.0
    v, f = field.vertices, field.faces
    if longest > max_edge:
        import trimesh
        v, f = trimesh.remesh.subdivide_to_size(field.vertices, field.faces, max_edge=max_edge)
    return SurfaceField(v, f, k=REFINE_K, large_factor=1e30)


_FILE_CACHE: dict = {}           # (resolved path, size, mtime_ns, what...) -> SurfaceField | parity; the last few files
_FILE_CACHE_MAX = 6              # the key does not hold SHAPE_RECO_MESH_ORDER or the STEP deflection: change them only
                                 # between processes (or clear the dict), or a cached field of the other setting is served


def _file_key(path) -> tuple:
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), int(st.st_size), int(st.st_mtime_ns))


def _cached(key: tuple, make):
    hit = _FILE_CACHE.get(key)
    if hit is None:
        hit = make()
        while len(_FILE_CACHE) >= _FILE_CACHE_MAX:
            del _FILE_CACHE[next(iter(_FILE_CACHE))]
        _FILE_CACHE[key] = hit
    return hit


def field_of_file(path) -> SurfaceField:
    """`closest_point_field.load_field(path)`, cached per process by (path, size, mtime): one generated sample's STL is
    read by the parity voxelizer, the inside bits and the field writer in turn (`sample_writer.finish_sample`)."""
    from .closest_point_field import load_field
    return _cached(_file_key(path) + ("field",), lambda: load_field(str(path)))


def parity_of_file(path, spec: GridSpec) -> np.ndarray:
    """`parity_occupancy(load_field(path), spec)`, cached like `field_of_file` (a read-only array): its three consumers
    per generated sample - voxelization._voxelize_parity, inside_bits.compute_inside, sample_writer.write_field -
    compute it once (measured 0.03-0.09 s each on v5 solids, plus the STL read; identical arrays by construction)."""
    def make():
        out = parity_occupancy(field_of_file(path), spec)
        out.flags.writeable = False
        return out
    return _cached(_file_key(path) + ("parity", tuple(spec.lo), tuple(spec.hi), int(spec.resolution)), make)


def parity_occupancy(field: SurfaceField, spec: GridSpec) -> np.ndarray:
    """(R, R, R) bool, world axes: inside/outside of every voxel centre of `spec` by scanline ray parity along x (what a
    voxelizer does): each triangle is hit by the grid rows (y_j, z_k) inside its yz projection, and a centre is inside
    when an odd number of hits lie at smaller x. The rows are shifted by a tiny irrational fraction of a voxel so no row
    runs exactly through a mesh edge or vertex (CAD parts put vertices on round coordinates, as the grid does). Needs a
    closed mesh; on an open one it is as good as any ray-parity test (trimesh `contains` included). Written by the
    residual-loop agent (reco_engine/residual_loop.py) and moved here so the generator and the loop share one
    implementation. Measured there: matches exact-mode signs on four example parts, agrees with trimesh `contains` on
    co_cone where the face-normal sign does not, and costs a fraction of `contains` (cy_box_bore 48^3: 17 s).
    Nothing here checks that the mesh is closed. See `parity_axes` for the two degenerate cases."""
    c = spec.centres()
    xs, ys, zs = c[:, 0, 0, 0], c[0, :, 0, 1] + 1.2345e-4 * spec.pitch, c[0, 0, :, 2] + 2.3456e-4 * spec.pitch
    return parity_axes(field.vertices, field.faces, xs, ys, zs)


def parity_axes(vertices: np.ndarray, faces: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """(len(xs), len(ys), len(zs)) bool: the scanline ray parity of `parity_occupancy` on an arbitrary axis-aligned grid
    of centres (sorted `xs`, `ys`, `zs`; any box, not only a cube). The caller shifts `ys`/`zs` off round coordinates.

    Degenerate cases (not covered by a test): a triangle whose yz projection has |det| < 1e-15 (an ABSOLUTE threshold,
    in the mesh's units squared) is skipped as edge-on; a row through an edge shared by two triangles counts on both
    (barycentric >= 0 on all three) and flips the parity of everything beyond it - the callers' irrational row shift
    makes this unlikely, not impossible. The hit counter is a dense (len(xs) + 1, len(ys), len(zs)) int32 array: large
    box grids go through local_grids.iter_slabs."""
    xs, ys, zs = np.asarray(xs, float), np.asarray(ys, float), np.asarray(zs, float)
    r = len(xs)
    tri = np.asarray(vertices, float)[np.asarray(faces)]                # (f, 3, 3)
    a, b, cc = tri[:, 0], tri[:, 1], tri[:, 2]
    j0 = np.searchsorted(ys, tri[:, :, 1].min(axis=1)); j1 = np.searchsorted(ys, tri[:, :, 1].max(axis=1))
    k0 = np.searchsorted(zs, tri[:, :, 2].min(axis=1)); k1 = np.searchsorted(zs, tri[:, :, 2].max(axis=1))
    keep = (j1 > j0) & (k1 > k0)
    counts = np.zeros((r + 1, len(ys), len(zs)), np.int32)
    for f in np.nonzero(keep)[0]:
        jj, kk = np.meshgrid(np.arange(j0[f], j1[f]), np.arange(k0[f], k1[f]), indexing="ij")
        py, pz = ys[jj], zs[kk]
        # barycentric coordinates of (py, pz) in the triangle's yz projection
        d = (b[f, 1] - a[f, 1]) * (cc[f, 2] - a[f, 2]) - (cc[f, 1] - a[f, 1]) * (b[f, 2] - a[f, 2])
        if abs(d) < 1e-15:
            continue
        l1 = ((py - a[f, 1]) * (cc[f, 2] - a[f, 2]) - (cc[f, 1] - a[f, 1]) * (pz - a[f, 2])) / d
        l2 = ((b[f, 1] - a[f, 1]) * (pz - a[f, 2]) - (py - a[f, 1]) * (b[f, 2] - a[f, 2])) / d
        hit = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        if not hit.any():
            continue
        x = a[f, 0] + l1[hit] * (b[f, 0] - a[f, 0]) + l2[hit] * (cc[f, 0] - a[f, 0])
        np.add.at(counts, (np.searchsorted(xs, x), jj[hit], kk[hit]), 1)
    return (np.cumsum(counts, axis=0)[:r] % 2) == 1


def voxel_field(occ: np.ndarray, spec: GridSpec) -> Optional[SurfaceField]:
    """The SurfaceField of a boolean grid's blocky boundary (`voxel_surface_mesh`) in world mm, with every face turned
    OUTWARD. `voxel_surface_mesh` itself does not orient its quads consistently (measured: its signed distance agrees with
    the occupancy at only 47 % of the voxel centres of a notched block), so each face is checked here: the point half a
    voxel behind its centroid, against its normal, must be material; otherwise the face is flipped. None if `occ` is empty."""
    occ = np.asarray(occ, bool)
    v, f = voxel_surface_mesh(occ)
    if v is None:
        return None
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    behind = np.floor(tri.mean(axis=1) - 0.5 * n).astype(int)                 # the voxel on the -normal side (index units)
    inb = np.all((behind >= 0) & (behind < np.array(occ.shape)), axis=1)
    solid = np.zeros(len(f), bool)
    solid[inb] = occ[behind[inb, 0], behind[inb, 1], behind[inb, 2]]
    f = np.where(solid[:, None], f, f[:, ::-1])
    return SurfaceField(v * spec.pitch + np.asarray(spec.lo, float), f)


def voxel_channels(occ: np.ndarray, spec: GridSpec) -> np.ndarray:
    """(5, R, R, R) float32, the same channels from a boolean component `occ` (world axes, the shape of `spec`): distance
    and V to its blocky voxel surface, sign from `occ` itself (a voxel centre is never on that surface, it is half a voxel
    away at least, so the sign is exact). An empty `occ` gives occupancy 0, V = 0 and sdf = minus the grid diagonal
    (finite: "far outside everything")."""
    occ = np.asarray(occ, bool)
    r = grid_shape(spec)
    if occ.shape != r:
        raise ValueError(f"voxel_channels: occupancy {occ.shape} does not match the grid {r}")
    field = voxel_field(occ, spec)
    if field is None:
        far = float(np.linalg.norm(np.asarray(spec.hi, float) - np.asarray(spec.lo, float)))
        return _pack(occ, np.full(occ.size, -far), np.zeros((occ.size, 3)), r)
    q = field.query(spec.centres().reshape(-1, 3), exact=False)
    sdf = np.where(occ.reshape(-1), q.distance, -q.distance)
    return _pack(occ, sdf, q.vector, r)


def surface_points(field: SurfaceField, n: int = 2048, seed: int = 0) -> np.ndarray:
    """(n, 6) float32: `n` area-weighted points ON the surface and the unit normal of the face each lies on (as the mesh
    orients it - outward on a closed, consistently wound mesh). The same sampling as `SurfaceField.sample_points` with
    band 0, but keeping each point's face, so no closest-point query is needed: a query of an on-surface point on a raw
    CAD mesh has to prove distance 0 against every sliver triangle (measured: 2 minutes for 2048 points on a carved box)."""
    rng = np.random.default_rng(seed)
    a, b, c = field._a, field._b, field._c
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(field.faces), size=n, p=area / area.sum())
    r1, r2 = rng.random((n, 1)), rng.random((n, 1))
    flip = (r1 + r2) > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    on = a[f] + (b[f] - a[f]) * r1 + (c[f] - a[f]) * r2
    return np.concatenate([on, field.face_normals[f]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------------------------------- conversions

def bin_to_world(raw: np.ndarray) -> np.ndarray:
    """A stltovoxel/.bin grid in its array order [z, -y, x] -> world axes [x, y, z] (peel.real_voxel_occupancy's flip)."""
    return np.ascontiguousarray(np.asarray(raw)[:, ::-1, :].transpose(2, 1, 0))


def world_to_bin(world: np.ndarray) -> np.ndarray:
    """Inverse of `bin_to_world`: world axes [x, y, z] -> the .bin array order [z, -y, x]."""
    return np.ascontiguousarray(np.asarray(world).transpose(2, 1, 0)[:, ::-1, :])


def network_input(channels: np.ndarray, spec: GridSpec, truncation: float = DEFAULT_TRUNCATION) -> np.ndarray:
    """(5, R, R, R) float32 network input ("vf_exact") from `mesh_channels`/`voxel_channels`, laid out EXACTLY like
    training/field_input.py's "fused" kind, so the two are interchangeable in any network and in comparisons:
      channel 0  occupancy in {-1, +1}
      channel 1  sdf / (truncation voxels), clipped to [-1, 1], NEGATIVE inside - the "fused" sign, i.e. the stored
                 (positive-inside) channel negated. Chosen so that vf_exact differs from "fused" only in HOW the numbers
                 were computed (exact mesh field vs grid-derived), never in what they mean - a controlled comparison
                 (Step 4) then compares the computation, not a sign flip the network would have to relearn;
      2..4       V / (truncation voxels), length clipped to 1 (|V| = |sdf| as in "fused"; V points to the surface in both);
    all in the ARRAY frame [z, -y, x] of every other input kind (rotation labels, `rotation_symmetry.rotate_volume`, the
    checkpoints): the grid is transposed/flipped and the vector turned with it, array components (a0, a1, a2) = (vz, -vy, vx)."""
    scale = truncation * spec.pitch
    occ = channels[0] * 2.0 - 1.0
    sdf = -np.clip(channels[1] / scale, -1.0, 1.0)
    vec = channels[2:5] / scale
    length = np.linalg.norm(vec, axis=0, keepdims=True)
    vec = vec / np.maximum(length, 1.0)
    vec_arr = np.stack([vec[2], -vec[1], vec[0]])
    out = np.concatenate([occ[None], sdf[None], vec_arr], axis=0)
    return np.ascontiguousarray(np.stack([world_to_bin(c) for c in out]).astype(np.float32))


DEFAULT_LOG_SCALE = 1.0          # voxels: d_log is ~linear within this distance of the surface (training/field_input.py)
CURVATURE_SCALE_MM = 2.0         # kappa is fed as arctan(kappa * this) / (pi / 2): a radius of 2 mm maps to 0.5
CURVATURE_MIN_DISTANCE = 0.3     # voxels: closer to the surface kappa = (1/lambda - 1)/d is ill-conditioned -> 0


def closest_point_spectrum(channels: np.ndarray, spec: GridSpec):
    """(lambda (2, R, R, R) ascending, kappa (2, R, R, R) in 1/mm, valid (R, R, R) bool) from the STORED exact field:
    the tangential eigenvalues of the Jacobian of the closest-point map c(x) = x + V(x), by central differences between
    voxel centres (world axes). Off the medial axis lambda_i = 1/(1 + d kappa_i), d the distance, positive OUTSIDE,
    kappa_i > 0 where the solid is convex; lambda ~ 0 across an edge, both ~ 0 at a vertex (scripts/field_atlas.py, where
    this is checked against analytic solids). No extra closest-point query: everything comes from `channels`."""
    r, h = spec.resolution, spec.pitch
    V = channels[2:5].astype(np.float64)
    c = np.moveaxis(spec.centres(), -1, 0) + V
    J = np.stack([np.stack(np.gradient(c[i], h), axis=0) for i in range(3)], axis=0)       # J[i, k] = dc_i / dx_k
    S = 0.5 * (J + np.swapaxes(J, 0, 1))
    S = np.moveaxis(S.reshape(3, 3, -1), -1, 0)                                            # (N, 3, 3)
    w, vec = np.linalg.eigh(S)
    n = np.moveaxis(V.reshape(3, -1), 0, -1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    drop = np.abs(np.einsum("nij,ni->nj", vec, n)).argmax(axis=1)
    keep = np.array([[j for j in range(3) if j != i] for i in range(3)])[drop]
    lam = np.take_along_axis(w, keep, axis=1)                                              # (N, 2) ascending
    d = -channels[1].astype(np.float64).reshape(-1)                                        # positive OUTSIDE, mm
    valid = np.abs(d) > CURVATURE_MIN_DISTANCE * h
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(valid[:, None] & (lam > 0.25), (1.0 / lam - 1.0) / d[:, None], 0.0)
    kappa = np.sort(kappa, axis=1)
    return lam.T.reshape(2, r, r, r), kappa.T.reshape(2, r, r, r), valid.reshape(r, r, r)


def dirlog_input(channels: np.ndarray, spec: GridSpec, log_scale: float = DEFAULT_LOG_SCALE,
                 curvature: bool = False) -> np.ndarray:
    """The "dirlog_exact" (5 channels) / "dirlog_curv" (8) network input from `mesh_channels` output, in the array
    frame like `network_input`:
      0     material {0, 1}
      1..3  UNIT outward normal of the level set through the voxel (V / |V| turned outward: full weight at every
            distance; `network_input` scales V by the distance, so it fades to 0 exactly at the surface)
      4     d_log = sign(d) log(1 + |d|/s) / log(1 + R/s), d in voxels, NEGATIVE inside (the "fused" sign), no truncation
      5, 6  (curvature) kappa_1 <= kappa_2 of the surface at c(x), as arctan(kappa * CURVATURE_SCALE_MM) / (pi/2)
      7     (curvature) collapse = clip(1 - 2 lambda_min, 0, 1): 1 where c(x) is an edge or vertex (lambda = 0), 0 on a
            face unless d kappa >= 1 (a convex face farther away than its own radius)
    The same encoding as training/field_input.py's grid-derived "dirlog", but from the exact mesh field."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from training.field_input import log_distance
    occ = channels[0] > 0.5
    V = channels[2:5].astype(np.float64)
    length = np.linalg.norm(V, axis=0, keepdims=True)
    outward = np.where(occ[None], 1.0, -1.0) * V / np.maximum(length, 1e-12)
    tiny = length[0] < 1e-6 * spec.pitch
    if tiny.any():                                                                          # on the surface: grad of sdf
        g = -np.stack(np.gradient(channels[1].astype(np.float64)), axis=0)
        g /= np.maximum(np.linalg.norm(g, axis=0, keepdims=True), 1e-12)
        outward[:, tiny] = g[:, tiny]
    dlog = log_distance(-channels[1] / spec.pitch, spec.resolution, log_scale)
    parts = [occ.astype(np.float32)[None], np.stack([outward[2], -outward[1], outward[0]]), dlog[None]]
    if curvature:
        lam, kappa, _ = closest_point_spectrum(channels, spec)
        k = np.arctan(kappa * CURVATURE_SCALE_MM) / (np.pi / 2)
        collapse = np.clip(1.0 - 2.0 * lam[0], 0.0, 1.0)
        parts += [k, collapse[None]]
    out = np.concatenate(parts, axis=0)
    return np.ascontiguousarray(np.stack([world_to_bin(ch) for ch in out]).astype(np.float32))


RESIDUAL_CHANNELS = ("input", "model", "nx", "ny", "nz", "l_model", "residual")


def dirlog_residual(s_model: Optional[np.ndarray], occ_input: np.ndarray, spec: GridSpec,
                    log_scale: float = DEFAULT_LOG_SCALE) -> np.ndarray:
    """(7, R, R, R) float32, array frame: the dirlog of the current MODEL combined with whether each voxel centre - the
    base point x of the model's closest-point vector - lies inside the INPUT mesh. The add-cut loop's view of one step:

      0  m_in      x inside the input (world-axes bool `occ_input`, e.g. `parity_occupancy` / `mesh_channels`[0])
      1  m_model   x inside the model (s_model > 0)
      2..4 n       unit outward normal of the model's level set through x (-grad s_model / |grad s_model|); 0 while the
                   model is empty
      5  l_model   log distance to the model surface (training/field_input.log_distance), NEGATIVE inside the model;
                   +1 (far) while the model is empty
      6  r         m_in - m_model: +1 material the model still lacks (an ADD region), -1 material the model has and the
                   input lacks (a CUT region), 0 explained

    `s_model`: the model's signed distance at the voxel centres of `spec`, mm, POSITIVE inside (reco_engine.residual_loop
    `_State.s_model`; -inf or None = nothing built yet). For a Boolean tree it is the min/max composition of the primitive
    SDFs: exact near the surface, a bound farther away, and its gradient is the normal of the face that bounds x. The
    vector channels are 2..4 (`rotate_network_input(x, g, (2, 5))`)."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from training.field_input import log_distance
    occ_in = np.asarray(occ_input, bool)
    r = spec.resolution
    out = np.zeros((7, r, r, r), np.float64)
    out[0] = occ_in
    if s_model is None or not np.isfinite(s_model).any():
        out[5] = 1.0
    else:
        s = np.where(np.isfinite(s_model), s_model, -np.sqrt(3.0) * (spec.hi[0] - spec.lo[0])).astype(np.float64)
        out[1] = s > 0
        g = -np.stack(np.gradient(s, spec.pitch), axis=0)                       # outward = against the inside-positive SDF
        norm = np.linalg.norm(g, axis=0, keepdims=True)
        n = np.where(norm > 1e-9, g / np.maximum(norm, 1e-12), 0.0)
        out[2:5] = np.stack([n[2], -n[1], n[0]])                                # world (x, y, z) -> array components
        out[5] = log_distance(-s / spec.pitch, r, log_scale)
    out[6] = out[0] - out[1]
    return np.ascontiguousarray(np.stack([world_to_bin(c) for c in out]).astype(np.float32))


def rotate_network_input(x: np.ndarray, g: np.ndarray, vector: Tuple[int, int] = (2, 5)) -> np.ndarray:
    """Rotate a `network_input` tensor by the grid rotation `g` (a signed permutation, rotation_symmetry.GROUP): every
    channel is moved like `rotate_volume` moves an occupancy grid, and the vector channels x[vector[0]:vector[1]] are also
    turned, V -> g V - the equivariant counterpart of "rotate the occupancy, then encode" for an input that cannot be
    re-derived from the grid. `vector` = (1, 4) for `dirlog_input` (every other channel there is a scalar)."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parent.parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from training.rotation_symmetry import rotate_volume
    out = np.stack([rotate_volume(c, g) for c in x])
    a, b = vector
    out[a:b] = np.einsum("ij,j...->i...", g, out[a:b])
    return np.ascontiguousarray(out.astype(np.float32))


# ---------------------------------------------------------------------------------------------------- storage

def save_field_npz(path, channels: np.ndarray, spec: GridSpec, points: Optional[np.ndarray] = None, **meta) -> None:
    """Write one sample's field: `channels` (float16, compressed), `points` (n, 6) float32, `spec` (JSON text) and any extra
    scalar/str metadata (`source` = "mesh" | "voxel", `closed`, `seconds`). float16 precision, measured by review R-04
    (2026-10-01): abs error up to 7.8e-3 mm for |d| of 14-28 mm, 6e-4 mm below 1 mm; its effect on the curvature channels
    of `dirlog_input(curvature=True)` (central differences of the stored V) was not measured."""
    import json
    arrays = dict(channels=np.asarray(channels, np.float16), spec=np.array(json.dumps(spec.to_record())))
    if points is not None:
        arrays["points"] = np.asarray(points, np.float32)
    for k, v in meta.items():
        arrays[k] = np.array(v)
    np.savez_compressed(path, **arrays)


def load_field_npz(path):
    """(channels float32 (5, R, R, R), spec GridSpec, npz dict of the rest)."""
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return data["channels"].astype(np.float32), GridSpec.from_record(data["spec"]), data
