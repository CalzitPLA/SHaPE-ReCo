# The vector-field route: reconstruction from the closest-surface-point field of an STL/STEP model

Branch `vector-field` was renamed `main`. This document describes the peel loop (route A): the field, the fits, the
loop, its export, its GUI mode and its measured limits. Streamlined on 2026-09-29: nothing was re-measured for that.

## Status (2026-09-29)

| | state | date, source |
|---|---|---|
| role | the **comparison route**. The thesis main line is the cascade add-cut loop (route B: `reco_engine/residual_loop.py`, `reco_engine/cascade.py`, `reco_engine/ibeam.py`). | D-002, 2026-09-24 (CLAUDE.md) |
| 41 example parts (20-35 mm across) | every part comes out as the expected program, 100 % of the surface explained, mean chamfer 0.0012 mm, worst single deviation 0.24 mm, 0.6-34.4 s per part | `docs/data/peel_examples.json`, 2026-09-22 |
| frame `Frame_4_x_6`, this route | not reconstructed: 72.5 % of the surface explained, chamfer 34 mm mean (STL) and 47.46 mm (STEP); exact IoU not measured | 2026-09-22, `docs/FRAME_FAILURE.md` |
| frame, route B | counts as solved by the user's decision of 2026-09-28: exact OCCT IoU 0.9655, 40 adds, genus 17 = the input's, surplus 0, chamfer 0.31 mm; the root fillets (r = 27 mm, 3.04e7 mm3 in 60 pieces) are missing | 2026-09-24 (S6), `docs/FRAME_FAILURE.md` |
| liner | not run with this route | - |

**Superseded (2026-09-28):** the status of 2026-09-24 in this file said "Neither route solves the frame: route B reaches
an exact OCCT IoU of 0.910". The measurement 0.910 stands for the 32-add program of 2026-09-23. The current result and
the decision are in the table above.

**Open:** plan items 2 (rest), 3, 4, 6, 8 and 9 below. None is scheduled; the work is on route B and the feature phase.

The direction of 2026-09-21 was: reconstruct a part from the geometry of an STL or STEP file with a vector field that
points from every point in space to the nearest surface point, without the voxel grid and the three CNNs as the base.
CLAUDE.md now states that voxels are the ground truth for where material is, and that the field supports them.

## The field

For a query point `x`, `V(x) = c(x) - x`, where `c(x)` is the point of the surface nearest to `x`.

- `|V(x)|` is the unsigned distance; where the SDF `f` is differentiable, `V = -f * grad f` (the field carries the distance
  and its gradient in one vector), so `V` points to the surface from both sides.
- At a surface point `V = 0`, and just off it `V` is along the surface normal.
- It says WHERE on the surface the nearest point is, which a plain SDF does not.
- It is discontinuous on the medial axis (points with two nearest surface points): the closest POINT jumps there, the distance
  does not. A faceted mesh makes this worse deep inside a body; near the surface the field is accurate.
- The sign (inside/outside) is not needed for the vector itself. `query()` also reports `signed_distance`, the distance signed
  by the STL's own face normal at `c(x)` (negative outside, positive inside: `sign(dot(V, normal))`, since `V` points back
  towards the surface, i.e.\ along the outward normal iff `x` is inside) - meaningful only for a closed mesh with consistently
  oriented normals (`is_closed()`). Note the opposite convention from `sdf_primitives.py`'s analytic `Primitive.sdf` (negative
  *inside*, the usual computer-graphics convention) - the two are not interchangeable without flipping a sign.

## Base: STL or STEP

`dataset_gen/src/closest_point_field.py`: `load_field("part.stl" | "part.step")` returns a `SurfaceField` with exact
closest-point queries on the triangle set (verified against `trimesh` to 1e-15; a KD-tree over small triangles, brute force over
the few large ones, and a bounding argument that proves nothing closer was missed). A STEP model is tessellated by FreeCAD
(`freecadcmd`, deflection in the model's units). About 11 000 queries/s near the surface; `exact=False` (one pass) for far points
such as a full voxel grid. Helpers: `sample_points` (both sides of the surface), `grid_vectors` (the field on a voxel grid centred
on the mesh's bounding box, array frame, in voxels; not the in-scale grid of a placed sample), `is_closed`.
`SHAPE_RECO_FIELD_BACKEND=igl` answers both query modes with libigl's exact AABB tree: 0.03-0.35 s per 64^3 grid against 12-19 s
for the refined one-pass query on the CPU (`docs/data/field_igl_bench.json`, 2026-09-24); the dataset generator uses it by default
when libigl is installed, the library default is `auto` (cuda, torch, numpy). `SHAPE_RECO_MESH_ORDER=2` gives a STEP file a
second-order field (six-node triangles from gmsh, `dataset_gen/src/quadratic_mesh.py`; measured in `docs/data/second_order_mesh.json`
and `second_order_liner.json`, not on the frame and not in this route's loop).

A STEP file also holds the exact B-rep: every face has a surface type (plane, cylinder, sphere, cone, torus) and its
parameters. This route does not use it (an STL has no such information). Route B reads the faces since 2026-09-26
(`reco_engine/brep.step_faces`, `scripts/freecad_brep_faces.py`).

## Reconstruction by fitting and peeling

`reco_field/fit.py` fits analytic primitives (`dataset_gen/src/sdf_primitives.py`, validated against FreeCAD Part) to the field:
sphere, box, cylinder, cone, torus, plane (a half space, the flat face of a cut). The residual of a primitive with SDF `f` is `f(c_i)` on
surface points and `| f(x_j) | - |V(x_j)|` on off-surface points (this pins the size; used for the whole part only, not for a
cluster). Starting points come from the exact normals: a box frame from a Manhattan-frame fit to the normals and its faces
from the pair of opposite planes with the most support (a start that survives a union), a cylinder from RANSAC on pairs of
oriented points (two normals give axis, centre and radius; several cylinders in one leftover each yield a hypothesis), a sphere
from the algebraic fit (with a RANSAC start from pairs of oriented points when part of the surface is not on the sphere), a cone
from RANSAC on TRIPLES of oriented points (the normals of a cone make one angle with its axis, so they lie on a small circle of the
unit sphere; three normals give the axis, the tangent planes all pass through the apex), a torus from a scan of the signed minor
radius r (shifted by r along the normals, the points of a torus fall on its core circle; the sign of r tells a solid from a
groove). The objective is a Tukey-biweight IRLS, which ignores clear outliers; a soft-L1 loss only discounts them
and settles on a compromise between the primitive and the unmodelled surface (measured: on the stepped box it PREFERRED a wrong
box, cost 0.99 against 1.62 for the true one). Tolerance is 0.4 % of the model extent.

### Segments: everything with a circle has a u parameter

Sphere, cylinder, cone and torus are fitted as bodies of revolution WITH their sweep angle `u = (0, w)` about the axis; the sphere
also with a latitude range `v` (`latitude="plane"` is FreeCAD's `makeSphere`, `"cone"` the true parameter region), the torus with
the minor-circle range `v`. The mechanism (`fit._segment`): fit the full body; take the angles of the surface points on its
lateral surface about the axis; an empty arc (larger than 0.25 rad and 10 mean spacings) is a candidate cut-open sector; turn the
frame about the axis so that the sector starts at `u = 0`; refine `u` and every other parameter against the side faces; keep the
segment only if it explains at least 0.5 % more surface than the full body. That last rule is what separates a sector from a
cut: the side faces of a sector lie on planes through the axis and carry data, whereas the empty arc of a cylinder with a flat
(`cy_flat`) has no faces through the axis, gets no segment, and is left to the peeling loop (a clip by a plane). A sphere's axis
is not defined by its lateral surface: it comes from the cut faces (dominant plane normals, the direction perpendicular to two
of them, the axis of a cone through the leftover).

A body of revolution whose radius is more than four times the part's extent is a plane in disguise (or, as a segment, the wedge
between two planes) and explains nothing. Measured before that rule: `l_bracket` and `side_slot` (a box with a quarter or a slot
removed) came out as a box minus a huge cylinder, which the chamfer could not see (0.006 mm) because the cylinder's surface
never comes near the part; with the rule they are box minus box.

`reco_field/peel.py` is the loop. Fit the base to the whole part; group the surface it does not explain into connected clusters;
the unexplained points inside the material of the current model (its SDF is negative there) are the walls of pockets, holes and steps and are
fitted as CUTTERS (a flat one as a CLIP by a half space), those outside as ADDITIONS (the two sides are clustered separately, so a
cluster never mixes them); accept an operation if it explains more of the input surface and does not make the model invent geometry
(below); repeat, up to a budget of operations that by default (`max_ops=None`) is PREDICTED from the part's own genus rather than a
fixed number (`predict_ops_budget`, below; pass an int for the old fixed behaviour, still `MAX_OPS` = 24). A cutter's hidden faces are not observable: an end of a box, cylinder or cone with no surface data ON it
(points on the end plane whose normals run along the axis) is open and is pushed out of the body, so the boolean is clean; a cone
keeps its surface when extended and stops at its apex.

`invented(model, field, tol, extent)` is the model -> input half of that grade, cheap enough to run inside the loop: random points around the
model are moved onto its surface (two Newton steps) and checked against the field; the result is the share of the model's surface
farther than 2 tol from the input. An operation is accepted only if its gain passes the floor AND it raises this share by at most 0.02
(`invented_slack`). The gain floor was 0.5 % of the whole surface until 2026-09-22; since then it is `min_gain * (1 - explained)`, relative to
what is unexplained. Without
the check a one-way "explained" gain accepted a cylinder of radius 3000 mm on a picture-frame STL (below): it touched more of the input's
surface and invented a ball-sized solid.

### Predicting how many operations a part needs (2026-09-22)

*(The rationale "plateau = the number of large features" is retracted: `docs/FRAME_FAILURE.md`, item 2, B-062. The
function and its measured cost stand.)*

- `predict_ops_budget(field)` sets the loop's budget. It is the default (`peel` / `peel_steps` with `max_ops=None`), in
  `peel_case.py` and behind the GUI's "Predict from genus" button. An int gives the old fixed behaviour (`MAX_OPS` = 24).
- It sweeps `SurfaceField.coarse_genus` at growing resolutions and stops when the running plateau has caught up with
  the exact genus (`SurfaceField.genus()`, milliseconds) or a wall-clock cap is hit.
- `max_ops = 1 (base) + ceil(plateau genus) + margin`, margin 4, cap 60. Margin 4 covers the 41 example parts: none
  needs more than 2 operations beyond base + genus (`docs/data/peel_examples.json`).
- On the 41 example parts the result (program, explained share, chamfer) is identical to the fixed `MAX_OPS = 24`. The
  sweep returns after one or two resolutions, in under a second.
- On the frame the sweep adds 16.8 s (STL, 6 554 triangles) to 64.3 s (STEP, 15 218 triangles) and changes no
  operation, share or chamfer (`docs/FRAME_FAILURE.md`, "Benchmark"). It predicts 14.
- Known gap, not fixed: a handle sealed inside a cavity is invisible to `coarse_genus` at every resolution (`.fill()`
  erases the cavity). Such a part pays the full `max_seconds` and is under-predicted
  (`tests/test_closest_point_field.py`). No part of this project has this shape.
- Tests: `tests/test_reco_field_peel.py`.

The grade is geometric, in the input's units and in both directions (`chamfer`): the model's surface against the input mesh and
back, with exact queries. A one-way "explained" share cannot see geometry the model invents; the chamfer can. (Measured: with
the approximate query mode the chamfer reported a phantom 2.9 mm error on the tube; the approximate mode is unreliable on CAD meshes,
whose cylinder walls are long sliver triangles.)

Results on the 41 example STLs (30 box, sphere and cylinder parts - including 3 hollow/shell ones, below - 11 cone, torus and
segment parts; no voxels, no networks, 0.6-34.4 s per part; `docs/data/peel_examples.json`, regenerated by
`scripts/eval_peel_examples.py`): every part comes out as the expected CSG program, 100 % of the surface explained, mean
chamfer 0.0012 mm over the set, worst single deviation 0.24 mm on parts 20-35 mm across. (The first run, on the 27 earlier
parts, had two wrong programs, `l_bracket` and `side_slot`, hidden behind a good chamfer; see the segment section.) The
three hollow parts (a closed internal cavity, not a bite off one side - see `docs/data/hollow_box.md`) are unremarkable in
program, explained share and chamfer, but are the three slowest of the whole set (25-35 s against 10.5 s for the next
slowest): the cavity's own surface adds a second cluster of unexplained points the loop must fit after the outer shell,
roughly doubling the number of points the fitter has to consider.

| part | program (op:primitive) | explained | chamfer mean (both ways) | max |
|---|---|---|---|---|
| `blind_hole` | box + cut:cylinder | 100 % | 0.0007 mm | 0.07 mm |
| `box_i000000` | box | 100 % | 0.0016 mm | 0.24 mm |
| `box_with_bore` | box + cut:cylinder | 100 % | 0.0015 mm | 0.08 mm |
| `bs_box` | box | 100 % | 0.0006 mm | 0.08 mm |
| `bs_box_dome` | box + add:sphere | 100 % | 0.0015 mm | 0.08 mm |
| `bs_box_sphere_cavity` | box + cut:sphere | 100 % | 0.0006 mm | 0.03 mm |
| `bs_box_sphere_pocket` | box + cut:sphere | 100 % | 0.0008 mm | 0.04 mm |
| `bs_box_step` | box + add:box | 100 % | 0.0012 mm | 0.09 mm |
| `bs_box_two_pockets` | box + cut:sphere + cut:sphere | 100 % | 0.0009 mm | 0.04 mm |
| `bs_hollow_box` | box + cut:box | 100 % | 0.0013 mm | 0.10 mm |
| `bs_hollow_box_thin` | box + cut:box | 100 % | 0.0011 mm | 0.08 mm |
| `bs_hollow_sphere` | sphere + cut:sphere | 100 % | 0.0012 mm | 0.00 mm |
| `bs_sphere` | sphere | 100 % | 0.0006 mm | 0.01 mm |
| `bs_sphere_flat` | sphere(v=-88.7..30°, plane) | 100 % | 0.0008 mm | 0.06 mm |
| `cavity` | box + cut:box | 100 % | 0.0007 mm | 0.10 mm |
| `co_cone` | cone | 100 % | 0.0009 mm | 0.08 mm |
| `co_cone_apex` | cone | 100 % | 0.0008 mm | 0.18 mm |
| `co_countersink` | box + cut:cone + cut:cylinder | 100 % | 0.0016 mm | 0.04 mm |
| `corner_pocket` | box + cut:box | 100 % | 0.0009 mm | 0.09 mm |
| `cross_hole` | cylinder + cut:cylinder | 100 % | 0.0013 mm | 0.10 mm |
| `cs_cone_sector` | cone(u=200°) | 100 % | 0.0016 mm | 0.15 mm |
| `cs_cyl_sector` | cylinder(u=120°) | 100 % | 0.0005 mm | 0.07 mm |
| `cs_sphere_band` | sphere(v=-89..20°, plane) | 100 % | 0.0005 mm | 0.03 mm |
| `cs_sphere_sector` | sphere(u=270°) | 100 % | 0.0008 mm | 0.07 mm |
| `cs_torus_half` | torus(u=180°) | 100 % | 0.0008 mm | 0.09 mm |
| `cy_box_bore` | box + cut:cylinder | 100 % | 0.0012 mm | 0.04 mm |
| `cy_box_on_cyl` | cylinder + add:box | 100 % | 0.0022 mm | 0.14 mm |
| `cy_cyl` | cylinder | 100 % | 0.0007 mm | 0.10 mm |
| `cy_cyl_long` | cylinder | 100 % | 0.0003 mm | 0.05 mm |
| `cy_flat` | cylinder + clip:plane | 100 % | 0.0017 mm | 0.05 mm |
| `cy_on_box` | box + add:cylinder | 100 % | 0.0014 mm | 0.08 mm |
| `cy_t` | cylinder + add:cylinder | 100 % | 0.0008 mm | 0.09 mm |
| `l_bracket` | box + cut:box | 100 % | 0.0012 mm | 0.06 mm |
| `plate_4holes` | box + cut:cylinder + cut:cylinder + cut:cylinder + cut:cylinder | 100 % | 0.0019 mm | 0.06 mm |
| `side_slot` | box + cut:box | 100 % | 0.0012 mm | 0.04 mm |
| `three_holes` | box + cut:cylinder + cut:cylinder + cut:cylinder | 100 % | 0.0015 mm | 0.04 mm |
| `to_elbow` | torus(u=90°) | 100 % | 0.0011 mm | 0.09 mm |
| `to_ring_groove` | cylinder + cut:torus | 100 % | 0.0033 mm | 0.12 mm |
| `to_torus` | torus | 100 % | 0.0006 mm | 0.00 mm |
| `tube` | cylinder + cut:cylinder | 100 % | 0.0022 mm | 0.11 mm |
| `two_bores` | box + cut:cylinder + cut:cylinder | 100 % | 0.0013 mm | 0.05 mm |

Closed-loop tests on analytically built solids (ground truth known) recover the operations and their parameters
(`tests/test_reco_field_peel.py`).

## GUI

`python3 -m reco_gui` has a method switch at the top of the settings dock (`--method cascade|field`): the cascade loop (route B, the
panel's default) and **Vector field** (an STL or a STEP file, no networks, no voxels); the voxel mode was removed on 2026-09-30. The vector-field mode steps through `reco_field.peel.peel_steps`, the generator behind `peel`
(`peel` is its last step, so the two cannot differ): `field` (samples drawn), `base` (whole-part fit), then per operation `cluster`
(the unexplained surface grouped, one cluster chosen: cutter or addition), `candidate` (a fit of it, with what the model would then
explain), `accepted` / `rejected`, and at the end `grade` (the two-way chamfer, optional) and `done`. Space steps, F5 continues to the
breakpoints of the method (default: base, accepted, rejected), the timeline shows any earlier step again.

The **Vector field** tab draws in the part's own units (mm): the input mesh (grey), the field as lines from sample points to their
closest surface points (V = c - x), the model built so far (blue), the input surface it does not explain (red), the cluster being fitted
(yellow) and the candidate primitive (orange; a cutter pink, a rejected one dark red; an open cutter is cropped to the part's box).
The detail panel lists the program with every primitive's parameters, the cluster, the candidate and the grade.
`reco_gui/field_session.py` (Qt-free) folds the steps into states and meshes the primitives in the worker thread; `viewer_field.py`
draws; `tests/test_reco_gui_field.py` covers the steps, the states, the report and a headless window run.

## What changes and what stays (state of 2026-09-21)

This table compares the voxel pipeline of tag `archive/master-2026-09-18` with the vector-field route. It does not
describe route B.

| stage | before | vector-field route |
|---|---|---|
| input | voxel grid (fit / in-scale), from an STL by `stltovoxel` | the geometry itself: STL or STEP |
| recognition | Type CNN, Rotation CNN, Dimension CNN | fitting primitives to the field, kind chosen by how much surface each explains |
| placement | initial estimate + local voxel search (IoU) | least squares on field residuals, in mm, no discretisation |
| loop | cover (add) then cut (subtract) on voxel residuals | peel: fit, remove the explained surface, fit the rest as an addition or a cut |
| check | voxel IoU (and a voxel-vs-real mismatch that was never resolved) | distance of the model to the input surface, both ways (Chamfer / field error) |
| build | FreeCAD macros, `assemble_reconstruction` | the program is built directly with Part shapes and booleans (`reco_field/export.py`); the macros are not used |
| training data | generator + voxelizer + noise | not needed for the fit; the generator and the CNNs remain for scans or as initialisers |

Unchanged and reusable: the FreeCAD macros and assembly, the SDF library, the STL/STEP I/O, the grading of a reconstruction
against the input, the GUI (worker, timeline and pacing are shared with the new mode), the test fixtures.

## Export to FreeCAD

`reco_field/export.py` turns `peel(...).model` into a FreeCAD solid. The primitive tree is serialised to JSON (`program_to_dict`) and
rebuilt by `reco_field/freecad_builder.py`, run inside `freecadcmd`, with the Part module's own shapes and booleans (fuse, cut, common),
so the result is an exact B-rep, written as STEP and as STL (tessellated at 0.02 mm). The Part conventions of
`docs/archive/freecad_part_shapes.tex` are applied: the box is centred here and has its corner at the origin in FreeCAD; a sector (u0, u1) is a
sweep of u1 - u0 turned by u0; the sphere's latitude range is `makeSphere(angle1, angle2)` (planes); the torus minor range `v` is built with
the `Part::Torus` feature as `[Angle1, Angle2] = (-v1, -v0)` (two pies when the range wraps past 180 degrees); a half space is a very large box on
the -normal side of its plane. Not exportable, with a clear error: a sphere with a cone-limited latitude (Part builds only plane-limited ones),
the ellipsoid, the convex polyhedron, a hollow torus.

```python
res = peel(load_field("part.stl"))
out = build_in_freecad(res.model, "out_dir", "part")          # part.step, part.stl, part.json (the program), volume, valid
export_check(out, load_field("part.stl"))                      # two-way chamfer of the FreeCAD solid to the input, mm
```
`python3 reco_field/export.py part.stl --out out_dir` does both. Measured: the FreeCAD solid built from the reconstruction of `co_countersink`,
`to_elbow` and `cs_sphere_band` has the volume of the original Part shape to 0.1 % (7253.5 against 7256.8, 444.0 against 444.1, 3125.4 against
3127.0 mm^3; the mesh of the original is tessellated) and a chamfer to the input below 0.005 mm. `tests/test_reco_field_export.py` (23) checks eleven
analytic primitives against the SDF model (sectors that do not start at 0, a tube sector, a wrapped torus range, a slanted half space, a
rotated cutter), the documented torus pie volume (725.49 mm^3), and eight example parts against the input (chamfer, volume, one valid solid).

## FreeCAD nodes

Three nodes of the Nodes workbench wrap the route (`freecad_nodes/shape_reco/shape_reco_field_peel.py`, `_export.py`, `_grade.py`; the logic is in `reco_field/node_support.py`,
tested without FreeCAD): `Peel (Vector Field)` (STL or STEP -> program as JSON, share explained, summary; Run button), `Export Program` (program -> STEP and STL with FreeCAD's
Part module; Run button) and `Field Grade` (two-way chamfer of two meshes in mm). `freecad_nodes/examples/example_vector_field.json` wires them (12 nodes, 11 edges).
`scripts/make_node_graph_figure.py` draws a graph (`docs/figures/vf_nodes.pdf`). FreeCAD runs the installed COPY of the nodes (`bash freecad_nodes/install.sh`), so after
new nodes are added the install must be repeated; `scripts/check_nodes_install.py` (run with `freecadcmd`) compares the copy with the repo and loads every example graph in the real workbench
(an outdated copy shows up as `OpCode ... not registered` with a broken `%d` message and `unexpected keyword argument 'socket_type'`). Tested in `tests/test_freecad_nodes.py` (16): socket names, the Run-button behaviour, a box through
peel -> export -> grade, the graph wiring and the idempotence of the update.

## Training data

This route's reconstruction needs no training data: it fits primitives to the field of the one mesh in front of it by least
squares (Tukey IRLS with RANSAC starts) and grades the result geometrically. The fit learns nothing, and `reco_field/` caches
nothing on disk. (Checked 2026-10-01; the earlier sentence "there is no vector-field training data" no longer holds for the
project, see the second and third items.) What exists:

- Field on demand: `load_field(stl_or_step)` then `query` (exact), `sample_points` (both sides of the surface), `grid_vectors`
  (the field on a voxel grid around the mesh, in voxels, array frame). Measured on a 20 mm part: a 64^3 grid 1.8 s (approximate
  mode), 20 000 exact near-surface samples 7.6 s (before the igl backend, see "Base: STL or STEP").
- Stored fields, for route B's networks (not read by this route): since 2026-09-24 (commit 0b9f2da2) every sample written by
  `dataset_generator.py --with-field` or `sample_writer.finish_sample` carries `<stem>.field.npz` (occupancy, signed distance
  and V on the in-scale grid, float16, world axes, plus 2048 surface points with normals; `dataset_gen/src/field_channels.py`)
  and `<stem>.inside.npy` (parity bits). The encoded network inputs are cached under `~/.cache/shape_reco/field_kinds/` and
  `voxfield/` (`training/field_input.py`, `training/inputs.py`; keys in `docs/CODEMAP.md` section 8).
- This route's point-cloud networks (`training/field_points.py`, `train_field_*.py`) are trained on surface samples of the
  dataset STLs (`local_dataset_type_pointcnn_big`, 3000 whole primitives; thesis chapter 12); as hints
  (`reco_field/field_hints.py`) they changed no fit on the frame (2026-09-22, plan item 8 below).
- Ground truth for evaluation: the datasets `dataset_gen/local_dataset_*` (gitignored; in the main checkout) hold an STL and a voxel `.bin` per sample plus JSONL
  labels, one record per sample in `<shard>/<topology>/dataset_metadata.jsonl`: `parameters` (size ratios), `placement` (centre and
  extent in mm, the noise flag, and for a carved sample the `carve` record: cutter kind, its centre and radius, the removed
  fraction), `rotation` (quaternion and Euler angles). 16 400 STLs in `local_dataset_inscale`. They were made for the CNNs and
  are unchanged. Their labels, cutter included, give the true program of each part, so the peeling could be graded on thousands
  of parts instead of the example parts; that comparison has not been run (the sizes are ratios of `scale_mm`, and a
  comparison needs a rule for when two programs are the same).
- Example parts with a known program: `dataset_gen/examples/*.stl` (`scripts/make_test_parts.py`), evaluated in
  `docs/data/peel_examples.json`.

A network would come back only where fitting is not enough: scans (noise, holes, outliers), to propose the base primitive or a
starting pose, or a learned field. Then the dataset would be a generator step: sample the field of each STL (points in the band
around the surface with V) plus the label; roughly 2 s per part for a 64^3 grid, so 16 400 parts are about 9 hours on one core,
less with the approximate mode restricted to a band (an estimate of 2026-09-21, not measured). Not started for this route beyond
the point-cloud networks above. For route B the field is a network input since 2026-09-23 (`docs/DIRLOG.md`), stored per sample as
described above.

## The field's derivative, and the field as network input (2026-09-23)

The closest-point map c = x + V has the Jacobian dc/dx with tangential eigenvalues 1/(1 + d kappa_i) (d distance,
kappa_i principal curvatures at c(x)), 0 across an edge, 0 twice at a vertex. So the derivative of the field classifies
every near-surface point as plane / cyl / dome / saddle / edge / vertex without a fit - `scripts/field_atlas.py`,
thesis chapter `docs/thesis/chapters/09b_field_analysis.tex`. This works from the exact field (analytic, mesh, or the
exact V sampled on the grid as in `.field.npz`), not from a V derived from voxel occupancy (staircase noise).
For network input the old layout (`fused`, `vf_exact`) weights the direction by the distance; the `dirlog` kinds keep
material, unit direction and log distance apart (`training/field_input.py`, `field_channels.dirlog_input`).

## Known limits (measured)

- The frame `Frame_4_x_6` (1000 x 6700 x 4980 mm; the material fills 2.6 % of its bounding box) is not reconstructed by
  this route. STL run of 2026-09-21: 71.3 % explained, chamfer mean 35 mm, worst 237 mm; STEP run: 66.4 %, 47 mm, worst
  362 mm. Measured cause: every addition is judged alone, and a slab or disc refills the opening (explained 70 % ->
  39 %). Measured not to be the cause: the tessellation, the open-end extension, the base. The part is two I-beam
  rings joined by 16 plates (genus 17). All measurements: `docs/FRAME_FAILURE.md`.
- Sphere, box, cylinder, cone, torus and plane; no ellipsoid, no convex polyhedron (wedge, prism), no tube as ONE primitive
  (a tube is a cylinder minus a cylinder), no torus/cone/cylinder with an inner radius.
- A segment is only recognised when its side faces carry data; a segment whose end is where it enters another body (its
  boundary is not a plane through the axis) is fitted as a full body and needs the other operations to trim it.
- A cutter that is a segment keeps its sector boundary; the u-ends of a partial cutter are not extended like the axial ends of a
  cylinder or cone.
- The base is the one primitive explaining most of the surface; a part whose largest feature is an added one (a body built from
  parts) is decomposed by the same rule and may come out as a different, equally correct program (not unique).
- The program is not minimal or canonical: two programs can describe the same solid.
- Only clean CAD meshes were tried (tessellation error only). A scan (noise, holes, outliers) needs an outlier model, a noise-
  aware tolerance and RANSAC starts for boxes and spheres as well.
- The field is defined for a closed OR an open mesh, but a cut needs to know which side is material: closed meshes only, or
  the normals' orientation.

## Plan (written 2026-09-21/22; states checked 2026-09-29)

1. ~~Peeling loop~~ (done: base, cut, clip, add; open cutter ends; two-way chamfer).
2. ~~Cone, torus, segments with u and v~~ (done). Still to fit: tube as one primitive (inner radius), ellipsoid, convex solids
   (wedge, prism); model selection by explained surface with a penalty per parameter (today: rounded explained share, then the
   parameter count).
3. Robustness for scans: noise-aware tolerance, an outlier model, RANSAC starts for boxes and spheres, open meshes.
4. Program clean-up: merge equal cutters, drop operations that change nothing, express a through-hole cutter by its true length,
   choose a canonical program.
5. ~~Export~~ (done: `reco_field/export.py` builds the program in FreeCAD as a STEP and an STL, see "Export to FreeCAD"). Still to do: hand it to the
   existing `assemble_reconstruction` macros / the FreeCAD nodes, and a cone-latitude sphere.
6. STEP B-rep: read the face surface types and parameters directly where a STEP file is the input. Not done in this route; done in
   route B (`reco_engine/brep.py`, 2026-09-26).
7. ~~Adapt the GUI to step through the peeling loop~~ (done: `python3 -m reco_gui`, method "Vector field"; see "GUI" above).
8. Decide the role of the CNNs (initialisation for scans, or retired). Decided for the project by D-002 and D-007 (2026-09-24): the CNNs
   belong to route B. For this route the point-cloud hints (`reco_field/field_hints.py`) changed no fit on the frame (2026-09-22).
9. The frame (`docs/FRAME_FAILURE.md`): evaluate an addition together with the cut that cancels its overlap (slab minus window as one
   candidate): not tested. The frame counts as solved by route B (status above), so this item has no priority.

## Files

All paths checked on 2026-09-29. The test counts in brackets are those recorded on 2026-09-21/22; they were not re-counted.

`dataset_gen/src/closest_point_field.py`, `dataset_gen/src/sdf_primitives.py`, `reco_field/fit.py`, `reco_field/peel.py`;
`reco_field/export.py`, `reco_field/freecad_builder.py`; `scripts/eval_peel_examples.py` (regenerates `docs/data/peel_examples.json`); `scripts/peel_case.py` (one part, every stage; the frame: `docs/FRAME_FAILURE.md`); `scripts/genus_sweep.py` (exact
and coarse-resolution genus of one part) and `scripts/inspect_step_faces.py` (a STEP file's exact B-rep face inventory, needs `freecadcmd`) — both from the frame
case study; `docs/archive/slides_vectorfield_gui.tex` (Beamer slides: technique and GUI; screenshots `docs/figures/gui_*.png` from `scripts/make_gui_slide_figures.py`, needs a display); the LaTeX versions `docs/archive/vectorfield_reconstruction.tex` (method, measurements;
numbers, table and figures from `scripts/make_vectorfield_tex.py`) and `docs/archive/freecad_part_shapes.tex`; `docs/archive/freecad_part_shapes.md` (the Part shapes against the
wiki and FreeCAD 1.0.0: parameters, frames, the mirrored torus `v`, what is still missing); test parts made by `scripts/make_test_parts.py` (FreeCAD;
`co_`, `to_`, `cs_` are the cone, torus and segment parts); tests `tests/test_closest_point_field.py` (17),
`tests/test_reco_field_fit.py` (30), `tests/test_reco_field_peel.py` (44), `tests/test_reco_field_export.py` (23), `tests/test_sdf_primitives.py` (49), `tests/test_reco_gui_field.py` (9).
