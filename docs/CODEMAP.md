# SHaPe ReCo — Codemap

Architecture of the shape-reconstruction project. This file is the source of truth for the thesis documents
(`docs/thesis/`, `docs/archive/thesis_chapter.tex`). Update it first when the architecture changes.
Commands, options and expected output: [`RUNNING.md`](RUNNING.md).

State described: branch `bolt-holes` (2026-09-29), which contains `main`. Function and module names were checked
against the code on 2026-09-29. Line numbers are not given, because they go stale; use `grep -n "def <name>"`.

Contents: 1 Goal · 2 Routes · 3 Main line · 4 Packages · 5 `reco_engine` · 6 `reco_field` · 7 `reco_gui` ·
8 `dataset_gen` · 9 `training` · 10 `scripts`, `server` · 11 `freecad_nodes` · 12 Tests · 13 Design rules ·
14 Measured findings of the legacy voxel route · 15 Not in this checkout

## 1. Goal

Given an STL or STEP file of a machined part, reconstruct it as a sequence of parametric CAD primitives (FreeCAD
Part solids: box, cylinder, cone, sphere, torus, half-space, and boolean trees of them) combined by booleans
(add, cut, intersect). The result is a program with type, dimensions and placement per primitive, not a mesh.
The reconstruction does not need to recognise a feature by name (an I-beam, a bolt hole); primitives and booleans
that explain the volume are the goal.

## 2. Routes

Three routes exist side by side. Frame = `Frame_4_x_6` (STEP, and its STL at 1/100 scale). Liner =
`Liner_LINER_001` (STEP).

| route | code | entry points | role | measured on the frame |
|---|---|---|---|---|
| **B. cascade add-cut loop with the feature phase** | `reco_engine/residual_loop.py`, `cascade.py`, `ibeam.py`, `start_shape.py`, `feature_stage.py`, `hole_stack.py`, `tools.py`, `feature_groups.py` | `residual_loop.residual_steps`, `reconstruct`; `scripts/time_cascade.py`, `scripts/eval_add_cut.py`, `scripts/g3_run.py`; `python3 -m reco_engine.reconstruct_cli --mode add_cut`; GUI `--method cascade` | **main line** (D-002, 2026-09-24) | exact OCCT IoU 0.9655, 40 adds, genus 17 = the input's, surplus 0, chamfer 0.31 mm; the root fillets are missing (r = 27 mm, 3.04e7 mm³ in 60 pieces). Member sub-loop with `member_min_length_ratio` 1.5 (S6). The user counts the frame as solved on these numbers (decision of 2026-09-28, `CLAUDE.md`) |
| **A. peel / vector field** | `reco_field/` | `reco_field.peel.peel_steps`, `peel`; `scripts/peel_case.py`; GUI `--method field` | comparison route (D-002) | 72.5 % of the surface explained; chamfer 34 mm (STL) / 47.46 mm (STEP); exact IoU not measured (`docs/FRAME_FAILURE.md`, "Benchmark", 2026-09-22) |
| **C. voxel greedy drivers** | removed 2026-09-30 (git history; §5.3) | — | — | legacy (D-002 names only A and B) | not measured on the frame | — |

Earlier frame numbers of route B, kept for the record:

| date | setting | exact IoU | source |
|---|---|---|---|
| 2026-09-24 | cube grid 96³, 32 adds, `min_length_ratio` 2.0 | 0.910 (7.93e7 mm³ missing: 62 % transverse plates, 38 % root fillets) | `docs/data/local_grid_residual.json`, `docs/FRAME_FAILURE.md` banner |
| 2026-09-24 | grid `tau`, 88 x 562 x 419 cells | 0.9445 | `docs/FRAME_FAILURE.md`, "The cascade loop on the wall-thickness grid" |
| 2026-09-24 | S6, `member_min_length_ratio` 1.5, STEP input; cube grid (pitch 10.1 × the thinnest wall: the exact judge decides) | 0.9655 | `docs/data/s6_frame_plates.json`, `docs/FRAME_FAILURE.md` "S6" (the G3 commit of 2026-09-27 is a later run of the same branch) |

Defaults that differ between entry points:

| setting | `ResidualConfig` (library) | GUI (`reco_gui.setup.RunSettings`) | `scripts/time_cascade.py` |
|---|---|---|---|
| method | — | panel preselects "Cascade (add-cut)" (first item of `settings_panel.py`); `RunSettings.method` = "cascade"; `--method` overrides (checked 2026-10-01) | cascade only |
| resolution / max | 48 / 128 | 96 / 192 | 96 / 192 |
| grid | "cube" | "tau" | "cube" |
| field backend | environment's | "igl" | "igl" |
| start shape | "legacy" | "legacy" | "legacy" |
| feature phase | off | on | off (`--features`) |
| FreeCAD build at the end | on (`freecad=True`) | off (`cascade_freecad_final`; toolbar button instead) | on |

Other entry points (checked 2026-10-01): `reco_engine/reconstruct_cli.py` builds `ResidualConfig(out_dir=...)`, i.e.
the library defaults with grid "cube"; `feature_stage.ROUGH` sets grid "cube" (96 / 192); `scripts/run_part.py` (the ABC
runner) sets grid "tau". The library default "cube" is the fixed part-scale grid D-005 warns against (review R-01,
2026-09-30, F1): on the frame the cube gave exact IoU 0.910 with 32 adds where the τ grid gave 0.9445
(`docs/FRAME_FAILURE.md`, "The cascade loop on the wall-thickness grid"); the volume-ratio refinement and the
blindness report (`grid_pitch_over_tau`) are the only guards. Not changed by the Document step.

Open blockers are tracked in `.claude/guru/planning/BACKLOG.md` (ids B-nnn). Blockers that the code has since
closed: D-003 (the order is a flag, `ResidualConfig.cuts_first`), B-060 (the loop passes
`member_min_length_ratio` = 1.5), B-090 (grid "tau").

## 3. The main line: cascade add-cut loop with the feature phase

`reco_engine.residual_loop.residual_steps(input_path, config)` yields one `ResidualStep` per stage.
`reconstruct` runs it to the end and returns a `result_schema.ReconstructionResult` in world mm.
The loop works on grids in world axes x, y, z (`dataset_gen/src/field_channels.py`), not on stltovoxel `.bin` arrays.

| # | stage | what happens | code | step kinds |
|---|---|---|---|---|
| 1 | input and grid | STL/STEP → `SurfaceField` → channels occupancy (ray parity), SDF, V on a grid. Grid "cube" (library default): `resolution` cells per side, refined ×1.5 up to `max_resolution` until the parity volume reaches `min_volume_ratio` (0.5) of the mesh volume; the run ends ("walls thinner than a voxel") only when no voxel is inside at any resolution tried. Grid "tau": a box of cells at the finer of the wall-thickness pitch (`tau_fraction` 1.0 × τ, the 5th percentile; D-005 licenses τ/3) and the cube pitch, coarsened ×1.25 while above `max_cells` (40M). `_grid_blindness` = pitch / τ (5th percentile); above 1 the type stage is skipped and the "done" message says the grid is blind | `_load`, `_spec_for`, `_tau_spec`, `_channels`, `parity_occupancy`, `_grid_blindness` | `input` |
| 2 | start | `first_guess` (user's shapes, `user_guess.py`) when given and an add of it is kept. `start_shape` "cubes" (2026-10-10): the voxel-cube cover `_cube_cover_add` - the largest completely filled boxes of the input occupancy (`largest_boxes`), snapped to the surface, as the first adds, no sub-loop, the rest left to the residual phase. Else `start_shape` "legacy" (default): the type stage on the whole input gives a subclass hint; member sub-loop or hollow sub-loop (hint or analytic gates), else the graded first add: the cascade's answer (`first_pass_iou` 0.9) and up to 3 of its alternatives compete with every bounding cover under `_grade_cover` (containment ≥ `COVER_MIN_INSIDE` 0.995). "auto" or a named type: the starting-shape stage instead; when no candidate passes, the run ends with an empty program (no fallback, no grade) | `_first_guess_phase`, `_member_phase`, `_hollow_phase`, `_first_add`, `_bounding_cover`, `_grade_cover`, `_start_add`; `ibeam.py`, `start_shape.py` | `candidate`, `member`, `ibeam_box`, `member_op`, `hollow`, `add` |
| 3 | residual | voxels deeper than pitch/2 inside the input and outside the model (missing), and the reverse (invented) | `_State` | `residual` |
| 4 | components | connected components of the residual, one cascade run each, plus extra candidates: strict cylinders and planes, intersect with the covers, composites, largest filled boxes, boolean tools, varied views on failure | `_components`, `_best_candidate`, `_evaluate`, `_tool_candidates`, `_component_views`, `largest_boxes`; `cascade.run_cascade` | `component`, `candidate`, `accepted`, `rejected` |
| 5 | rough rule | only with the feature phase on: a later cut (add) stays in the rough model only if its gain is ≥ `rough_cut_grade` (0.10) of the first cut's (add's) gain. Member operations are never graded | `_residual_phase`, `_rough_text` | — |
| 6a | feature phase (if on) | exact difference model ↔ input by OCCT, split into pieces; per piece or congruence group a guess; validated by a real FreeCAD build. Section 5.2 | `_feature_phase` | `feature_*` |
| 6b | voxel fallback (if the phase is off) | components nothing explains become voxel cubes merged into boxes, at most `max_fallback_boxes` | `voxel_boxes`, `_fallback_box` | `fallback` |
| 6c | polish (`ResidualConfig.polish`, on since 2026-10-02) | every accepted sphere, cylinder, cone and torus refitted to the input's VERTICES: the input faces of its curved surface (centroid within the band, normal facing its way), robust least squares of the distance to that surface alone; only the curved surface moves (caps and planes stay). Kept if the vertices' RMS falls to at most 0.25 of what it was (`CONSISTENT`) and the program's surface RMS over all vertices does not rise. Why: a CAD tessellation's vertices lie on the surface (v5: within 1e-6 mm), its triangles sag inside a convex face, and the loop fits the triangles - v5 spheres came out 6-11 um small (IoU 0.998 against the exact tree, tier B). Re-rated finished runs (calib + eval, 74): A 24 -> 56, no tier and no IoU worse; aegir pilot (13) unchanged (`docs/data/polish_2026-10-02.json`). Cylinder/cone SECTORS keep a B: their side planes pass through the axis and are not part of the fit | `polish.polish_result` | `done` (`info["polish"]`) |
| 7 | build and grade | the program is built in FreeCAD (`reco_field/export.py`) and graded: two-way chamfer (10 000 points), validity (exactly one solid), the exact residual by OCCT (`exact_grade`, `exact_check.exact_residual`) cross-checked with `exact_check.sampled_iou`: a disagreement over 0.05 reports the sampled IoU instead (`ReconstructionResult.exact_method`); STL inputs over 12 000 triangles get the sampled IoU only; genus of the model and of the input. A failed build is a result (`valid=False`, the error), not an exception. Then the statistics vector (`solution.solution_stats`, voxel based) in the `done` step | `freecad_grade`, `_grade`, `_exact_grade`; `reco_field/exact_check.py`, `solution.py` | `grade`, `done` |

Order of operations (`ResidualConfig.cuts_first`, D-003): `True` takes the first acceptable cut component before any
add; `False` takes all adds before the cuts; `None` (default) is cuts first after a starting-shape start and adds
first on the legacy path, so a default run (legacy) adds first. D-003 asks for both orders to be measured: no
recorded measurement compares them on the legacy path (searched `docs/data/`, `RUNNING.md`, `FRAME_FAILURE.md`,
`LINER.md`, `FRAMEWORK_CHANGES_2026-09-26.md` on 2026-10-01). The classic solution of `solution_choice.py` runs with
cuts first. `op_guard` (default off, `$SHAPE_RECO_OP_GUARD`) bypasses the order: every round takes whichever
component gains most.

Known limit of the voxel part: a wall thinner than the pitch leaves no residual voxel (B-059). The exact grade and
the feature phase see such material; the voxel loop does not.

**The judges of a run** (route B; review R-02, 2026-09-30). Each sees something the others cannot; none is complete.

| judge | code | measures | blind to (measured or read) |
|---|---|---|---|
| banded voxel residual \|R\| | `_State.residual` | voxels deeper than pitch/2 inside the input and outside the model, and the reverse | walls thinner than the pitch: the frame on the cube grid had residual 0 / 0 at exact IoU 0.910 (`docs/FRAME_FAILURE.md`) |
| candidate checks | `_evaluate` | net gain ≥ `min_gain`; for a cut the strict void check on the eroded interior and the sign check | the same pitch limit |
| sampled exact judge | `ibeam.ExactJudge` | 1500 points per primitive, input by winding number, model by its SDF: precision, new, gain | sampling error (binomial, about 0.006 at 0.95 for 1500 points, computed) |
| exact residual by OCCT | `exact_check.exact_residual` | r+, r−, IoU = (V_in − r+)/(V_in + r−), pieces, genus; general fuse first, then separate booleans | its sample check (1500 points in the joint box, tolerance 4σ + 0.2 % V_in) cannot refuse a failed boolean on a part that fills little of its box: review r13 gave IoU 1.0e-7 without an error against sampled 0.9366 (`~/shape_reco_results/compare_server_a_20260929/summary_all.md`; the tolerance, 1.75e8 mm³, exceeds V_in = 7.53e7 mm³) |
| the loop's exact grade | `residual_loop._exact_grade` | the OCCT number unless `sampled_iou` disagrees by more than 0.05 | the feature phase's own differences and the standalone scripts have no such cross-check |
| surface measures | `surface_orders.containment`, `coverage` | share of the input's surface on/inside the model; orders 0/1/2 of surface agreement | which side the material is on (normals compared up to sign), tolerances of 1e-3 × the diagonal |
| difference grade | `difference_grade.grade_difference` | the exact difference's pieces as skin / feature / open; `reach` (a prediction) | inherits the exact residual's IoU as given |
| statistics vector | `solution.solution_stats` | voxel IoU, covered share, coverage, position / orientation terms, score | OCCT (never called); `score` silently 0 when the surface terms fail |

Agreement where both exist: on the ABC run of 2026-09-30 (`abc_known`), exact minus sampled IoU over 218 parts had a
median of +1.4e-4; 9 parts disagreed by more than 4 standard errors + 0.002; the exact difference gave no IoU for 64
parts, each refused by one of its own checks (22 an invalid piece, 21 the balances, 7 Null shape, 4 the sample, 10
other; `docs/reports/abc_benchmark/abc_benchmark.tex` and `data/exact_fail.tex`, uncommitted in the main checkout on
2026-10-01; raw records `~/shape_reco_results/abc/abc_known/`).

**Known limits of the loop** (review R-01, 2026-09-30; documented, not changed): the default grid is the cube (§2);
the default order is adds first (above); `start_shape="auto"` has no fallback (row 2); the type stage degrades
silently - `cascade.load_default_registry` falls back to the network-free default on any error, a net or descriptor
that fails to load is cached as None for the process, and only the entries that voted appear in the `input` step;
route B imports route A (`reco_field.peel.predict_ops_budget`, `reco_field.fit`, `export`, `exact_check`) and, on the
starting-shape path, the GUI package (`reco_gui.model_mesh`, `reco_gui.field_session`); every step carries a deep copy
of the result (`_snapshot`), cost not measured; `_feature_phase` is one function of about 1 530 lines.

## 4. Packages

Dependency order, lowest first. Nothing imports `reco_gui` except `reco_engine/start_shape.py` (lazily, on the
starting-shape path: `reco_gui.model_mesh.primitive_solid`, `reco_gui.field_session.MeshData`; checked 2026-10-01).

| package | purpose | tracked files |
|---|---|---|
| `dataset_gen/` | training data from FreeCAD macros and Part solids; the field, parity and SDF libraries all other packages use (imported as `src`) | `src/`, `FCMacros/`, `examples/` |
| `training/` | network definitions, input encodings, trainers | 28 modules |
| `reco_field/` | route A, and the FreeCAD builder and exact judge that route B also uses | 10 modules |
| `reco_engine/` | routes B and C, run store, timing, cache | 54 modules |
| `reco_gui/` | Qt supervision window for all three routes; dataset-generation window | 27 modules |
| `freecad_nodes/` | nodes for FreeCAD's Nodes workbench | 30 node modules |
| `scripts/` | command-line drivers, evaluations, report generators | 144 files |
| `server/` | setup and job runners for workstation and the institute servers | 15 files |
| `models/registry.json` | the stage registry of the cascade | 1 file |
| `tests/` | pytest suite | 111 files |
| `docs/` | Markdown notes, LaTeX thesis, slides, reports, measured data (`docs/data/`) | — |

The import graph is generated, not typed: `scripts/codemap_graph.py`, figures by `scripts/make_codemap_tex.py`
(`docs/archive/codemap_modules.tex`).

## 5. `reco_engine/`

### 5.1 Main line modules (route B)

| module | purpose | main functions and classes | tests |
|---|---|---|---|
| `residual_loop.py` | the add-cut loop and its feature phase | `ResidualConfig`, `ResidualStep`, `residual_steps`, `reconstruct`, `reconstruct_variants`, `parity_occupancy`, `voxel_boxes`, `largest_boxes`, `select_for_target`, `freecad_grade`; constants `STEP_KINDS`, `FEATURE_KINDS`, `THIN_WALL` | `test_add_cut_loop.py`, `test_feature_phase.py`, `test_target_iou.py`, `test_booleans.py`, `test_box_grid.py`, `test_excess_boxes.py` |
| `cascade.py` | one primitive for one target through the stages TYPE → ROTATION → DIMENSIONS → POSITION; each stage tries the registry's sources in cost order, a source that reaches its gate ends the stage | `run_cascade`, `type_stage`, `LocalGridSpec`, `BoxGridSpec`, `box_grid_spec`, `grid_centres`, `primitive_occupancy`, `inscale_target`, `oriented_bounding_box`, `bounding_cylinder`, `bounding_candidates`, `strict_cylinders`, `strict_planes`, `strict_spheres`, `composites`, `surface_agreement`, `near_surface_agreement`, `load_default_registry`. Type stage: the live entries fused by the weighted mean (weights = val parent accuracy), `legacy` entries skipped (D-007), the target scale-normalised onto the 64³ in-scale grid (`inscale_target`, mode "always"). Pose CNN entries run only for callables passed in `models` (`_cnn_pose`). `branch_iou` has no caller; `LocalGridSpec` is the ImportError fallback of `grid_spec` | `test_cascade.py`, `test_type_stage_legacy.py`, `test_inscale.py`, `test_bounding_box_mode.py` |
| `stage_registry.py` | which source answers which stage for which topology; reads `models/registry.json` (one flat list `entries`, each with `id`, `stage`, `kinds`, `mode`, `path`, `input_kind`, `cost`, `gate`, `metrics`, optional `legacy`). Live type entries (2026-10-01): `type_descriptors_v5` and `type_hier_dirlog_inside_medium`, fused, test parent accuracy 0.802 against 0.768 and 0.667 alone (`docs/data/gate_t_medium.md`); the other six type entries are `legacy`. The keys `ensembles`, `type_stage` are not read by the loop; `default_type_entry` only by `model_registry.py` | `StageEntry`, `load_registry`, `load_registry_meta`, `entries_for` | `test_stage_registry.py` |
| `type_descriptors.py` | network-free type source: 27 rotation-invariant descriptors of a 64³ grid and gradient-boosted trees over the v5 hierarchy (5 parents, 24 subclasses) | `descriptors`, `DescriptorModel`, `fit`, `evaluate`, `save`, `load` | `test_type_descriptors.py` |
| `ibeam.py` | straight prismatic members (I-beam, rod, bar, tube, rhs) and closed cavities, found in exact mesh sections, not in voxels; no world axis assumed | `detect_members`, `fit_ibeam_section`, `snap_profile`, `classify_section`, `ibeam_boxes`, `member_operations`, `find_cavities`, `ExactJudge`; `UNIT_SCALES`, `SNAP_MAX_REL` | `test_ibeam.py`, `test_beam_parts.py` |
| `scripts/path_atlas.py` | the path atlas: every run's tree stored per (domain, part, path variant), trees compared across paths (fingerprints, best path), the parts x paths map per domain (2026-10-10) | `build`, `compare`, `map` | - |
| `reco_gui/server_run.py`, `reco_gui/server_result.py` | server runs from the window for any part (the run bar's "on"): the settings as run_part options, start in a thread, poll, fetch, store (`run_store.import_run_part`), the alternatives together on the server; the result panel (2026-10-29, RUNNING §3.8) | `server_args`, `ServerRuns`, `import_mirror`, `with_variant`; `result_html`, `fetch` | `test_server_run.py`, `test_server_follow.py` |
| `scripts/detail_study.py`, `scripts/remote_run.py`, `reco_gui/server_follow.py` | one part in detail (three equivalent trees, exact pairwise IoU, the reconstruction against each tree, the lacking steps) and a server run followed live (ssh tail -F of path.jsonl, the model rebuilt per operation; 2026-10-29, RUNNING §3.8, §4.10) | `bracket_trees`, `leaf_errors`, `cmd_compare`, `cmd_lacks`; `remote_command`, `start_mirror`; `ServerFollow`, `model_mesh` | `test_server_follow.py` |
| `stats_tests.py`, `scripts/sci_test.py` | the reconstruction protocol (2026-10-29, RUNNING §4.9): paired arms on frozen corpora against the replicate's noise floor; freeze / power / analyze / waste / launch, the R7 verdict | `mcnemar_exact`, `paired_bootstrap_delta`, `wilcoxon_paired`, `holm`, `wilson`, `mcnemar_sample_size`, `flip_rate`, `net_change_floor`; `compare`, `verdict`, `waste_of_run` | `test_stats_tests.py`, `test_sci_test.py` |
| `prune.py`, `reco_gui/postprocess.py`, `scripts/minimize_parts.py` | the parts of a reconstruction on one set of sample points (2026-10-30, RUNNING §3.9): mass share, unique volume and fit per part, sampled IoU of any keep/drop combination, minimize the number of parts within an IoU tolerance; the post-processing window (tree with mass ratios, select, benchmark, decide, exact check, export); the loop's prune step (`ResidualConfig.prune_parts`, off; protocol prune_v1) | `sample_program`, `part_table`, `metrics`, `minimize`, `edited_result`; `PostprocessWindow`; `residual_loop._prune_parts` | `test_postprocess.py` |
| `leaf_recognition.py`, `scripts/leaf_recognition_study.py` | the primitives inside a compound STEP part from its own surfaces (2026-10-30, RUNNING §4.12): a hypothesis per surface over its own extent (convex add, concave cut, whole or sector; a dominating surface also a cover; boxes from orthogonal plane pairs), verified against the input, the fewest selected by a greedy CSG and minimized; the loop's start 'leaves' (`residual_loop._leaves_add`, protocol leaves_v1) | `surfaces_of`, `hypotheses`, `verify`, `select`, `recognize` | `test_leaf_recognition.py` |
| `scripts/grade_methods.py` | every method graded against the bare loop of its own batch (A confirmed, B better, C neutral, D worse, E unmeasured) and the keep/drop decision; `path_policy.VARIANTS` (offered) / `RETIRED` (2026-10-30) | `paired`, `grade`, `decide` | `test_path_policy.py` |
| `shape_class.py`, `scripts/shape_atlas.py` | shape classes `CLASS_V1` (Betti numbers, surface family, tree complexity) and the atlas of results per class: capability and coverage gaps (2026-10-29, §5.5) | `describe`, `tree_facts`, `classify`; `describe`, `calibrate`, `join`, `report`, `select` | `test_shape_class.py` |
| `path_policy.py` | the path variants a run can take (bare, sub-parts, cubes, auto, descriptors, cnn, tree: `reco_gui.setup` overrides and `run_part.py` options) ranked by their measured share of good runs (`docs/data/path_weights.json`, `scripts/path_weights.py`); the GUI asks after a low-IoU run (`MainWindow._offer_retry`, 2026-10-10) | `VARIANTS`, `variant_of`, `apply`, `rank`, `iou_of_done`, `describe` | `test_path_policy.py` |
| `benchmark.py`, `event_time.py` | the benchmark of a part's stored runs (2026-10-11): rows of what each run measured, its path, its aegir rating against a known tree, its nephogram figures (p95 missing / invented) and its event time; ranked for accuracy, design, simplicity, speed or event time. `event_time.py` (2026-10-29): the aegir navigation time per event of a stored run as the deck measured it (difference quotient of 50k / 300k events minus an empty baseline in the same world and gun, median of 3 rounds), any part placed in aegir's fixed guns (beam along the thinnest extent, the tightest covering cone) | `benchmark`, `row_of`, `rank`, `choose`, `read_nephogram`; `event_time`, `placement`, `navigation`, `read_event_time` | `test_benchmark.py`, `test_event_time.py` |
| `residual_tree.py` | the difference-driven decision tree (2026-10-15, `ResidualConfig.mode = "tree"`): every round one explicit decision on the difference between model and input (D0 accept, D1 carve the excess, D2 add the missing, D3 leave the rest to the feature phase, D6 stop), cuts first; a component no primitive explains becomes a CHILD - its inside boxes as one composite operation, else the cover child (its bounding box, the material taken wrongly re-added by a grandchild, depth-limited); the solid gate D5 (`residual_loop._severs`) refuses a cut that leaves more pieces than the input has, and D5 exact (`residual_loop._exact_severs`, 2026-10-16) builds the chosen cut with the program in freecadcmd and refuses it when the solid count grows (the grid cannot see pieces touching along the cutter's faces; `scripts/solids_per_op.py` names the splitting operation of a run); every child counts once against the op budget, a work budget counts components | `classify`, `Decision`, `Node`, `tree_loop`, `fuse_chain`, `is_child`; `BRANCHES` | `test_residual_tree.py` |
| `shape_search.py` | the dumb shape search (2026-10-16, the user's rule, `ResidualConfig.mode = "search"`): per residual component the input's exact surface samples near it (the samples whose voxel lies in the component dilated by one), then for EVERY base shape guesses from those points alone (centroid, principal and world axes, extents, radial distances) hill-climbed by random perturbation on the one grade, the coverage = the share of the nearby samples on the shape's surface (|sdf| <= `search_tolerance_share` 0.004 x the part's extent); `best_candidate` returns `_best_candidate`'s dict, every kind's best judged by the loop's `_evaluate`, the covered passing one wins; sources `search:coverage`; the start stays the legacy one |
| `repair.py` | the repair pass after the loop's rounds (2026-10-15, `ResidualConfig.repair`, on in mode "tree"): every accepted operation left out once on the final model - one whose absence changes nothing within `repair_tolerance` is taken back, and when the model is in more pieces than the input the operation whose absence restores the count at the least IoU cost; `accepted = False` with a "repair:" label, a `repair` step | `repair` | `test_repair.py` |
| `start_shape.py` | the starting shape: holds the whole part, highest score = covered share of the generated volume × surface coverage (mode "product"; "band": least excess near the best coverage). The excess needs `reco_gui.model_mesh` (manifold3d); without it no candidate passes and the run ends empty | `surface_inventory`, `start_candidates`, `grade_start`, `choose_start`, `vary_start`, `dominating_cover` | `test_start_shape.py` |
| `feature_stage.py` | rough model, exact difference in pieces, guesses per piece; shared with the loop's feature phase | section 5.2 | `test_feature_stage.py`, `test_contact_tool.py`, `test_contact_grading.py`, `test_thin_wall.py` |
| `hole_stack.py` | hole stacks: per hole axis the cylinders and cones a piece's real faces name, each step with its own reach | `hole_stacks`, `stack_parts`, `stack_side`, `stack_label`, `stacks_tool` | `test_hole_stack.py` |
| `polish.py` | the polish after the loop (stage 6c): curved primitives refitted to the input's vertices | `polish_result`, `surface_distance`, `mesh_arrays`; `CURVED`, `FREE`, `CONSISTENT` | `test_polish.py` |
| `torus_guard.py` | the torus guard (T-29, 2026-10-01): a torus is rejected when r >= R (spindle/horn: Geant4's G4Torus refuses it), (R - r) / R < `NEAR_HORN_GAP` 0.01 (near-horn) or R + r > `OVERSIZE_RATIO` 1.5 x the part's bounding-box diagonal (oversized). Applied wherever a torus is proposed or accepted: the cascade (`_Context.admit`: the mode has no answer, its branch falls), the loop's `_evaluate` (fails with "torus guard: ..."), the feature sources (`feature_stage.torus_gate`: not offered, the reason under `failed`), `subpart_cover.torus_of`; `rounded_corners` gives a horn or near-horn corner face the thinned torus (`thinned`: inside the face's, same outer equator, (R' - r') / R' = 0.02; the liner's slot ends R = r = 2) and leaves any other rejected corner as the box made it. Thresholds from the 1321 torus faces of the 500 ABC STEP files (`docs/data/torus_faces_abc500.json`: 43 horn faces, smallest other (R - r) / R 0.085, largest (R + r) / diagonal 1.12) and the stored programs; counted by `scripts/torus_guard_census.py`: 18 of the 679 program trees hit (spindle/horn 7, near-horn 1, oversized 10) | `torus_defect`, `primitive_defect`, `thinned`, `tori`, `program_tori`, `TorusDefect`; `NEAR_HORN_GAP`, `OVERSIZE_RATIO`, `REPAIR_GAP` | `test_torus_guard.py`, `test_prismatic_groups.py` |
| `turned_contour.py` | the turned contour of a rough cylinder, from the input's coaxial cylinders and cones: per run of profile pieces one cut (a cylinder 5 % larger than the rough one minus the stack of the profile's cylinders and cones); a ring is cut only when at most 0.2 % of 20 000 sampled points of it lie in the input. Only when the first accepted operation is a cylinder add and the only add (`ResidualConfig.feature_contour`). `feature_stage._snap_sector` turns a sector's end planes onto the input's planes through the axis | `coaxial_segments`, `contour`, `turned_contour` | `test_turned_contour.py` |
| `prismatic_groups.py` | a piece that is a union of prismatic features (step, groove, branch, slits): one channel per pair of facing real planes, with floor, ends, rounded edges and corners from the faces around it; the channels' tools fused and offered to `guess_candidates` with the hole stacks (sources "tool (prismatic groups)", "tool (prismatic groups + hole stacks)"). A cut and thin material are told apart by the sign of the distance between the two facing planes, never by thickness. Measured 2026-09-29 on the liner's removed solid of 101 133.2 mm3, exact: 100 940.0 mm3 covered, 191.92 mm3 left, harm 0.000 mm3; sampled against the input's mesh the same tool has precision 0.997, below the loop's 0.999 (open) | `channels`, `channel`, `fused`, `prismatic_tools`, `candidate_tools`, `with_face_boxes` | `test_prismatic_groups.py` |
| `reco_field/exact_check.py`: the incremental difference (2026-09-29) | OCCT's boolean of the input against a model with features fails silently (liner, 39 features: common 0.00 mm3) or raises (52 features: Null shape). `incremental_residual` keeps the solids of the last difference and updates them by the tools applied since: a cut T takes T from r-, and adds what T takes of the input to r+; every pair is split both ways, the volumes must add up, the volume budget V_model - V_in = r- - r+ is checked. The loop chooses incremental, then global, then the sampled judge (`ResidualConfig.feature_incremental`). Measured 2026-09-29 under load on the liner: 39 features 172 s, exact IoU 0.974032, r+ 0, r- 1 224 974 mm3; 52 features 193 s, exact IoU 0.976852, r+ 0, r- 1 088 795 mm3 (before: no answer in 300 s, and Null shape) | `exact_residual(state_dir=)`, `incremental_residual` | `test_exact_check.py`, `test_feature_phase.py` |
| `reco_field/exact_check.py`: the general fuse (2026-09-29) | the exact difference's first method: one general fuse of input and model, the pieces by the map of their origin; the shared volume from the two balances V_in - r+ and V_model - r-, which must agree within 1e-5 V_in and with the sample; else the separate booleans. Faster and right on the liner with 39 and 52 features, wrong on the frame with a tolerance, the liner with 60 features, review r13 and r08 (`RUNNING.md`, the general-fuse table). Every result passes the checks before it is returned, but the sample check is vacuous on a part that fills a small share of its bounding box ("The judges of a run", §3): the checks do not guarantee a right number there | `partition`, `by_general_fuse` (in `SCRIPT`), `exact_residual(general_fuse=)` | `test_exact_check.py` |
| `scripts/exact_of_run.py` (2026-10-01) | the exact IoU of a stored run after the run: the rough model's exact difference, then the run's tools by the incremental difference; `--grow` brackets the IoU where the tools sit a hair off the input's faces (grown tools: an upper bound of r+), `--snap` moves them onto the input's axes. Liner, 86 features: the model's own difference fails, the bracket is 0.99995 to 0.99998 (`docs/data/liner_exact_2026-10-01.json`) | `load_result`, `feature_start`, `grow`, `snap_tool` | |
| `scripts/count_holes.py` (2026-10-01) | the holes of a STEP part from its B-rep: coaxial cylinders and cones whose extents touch, with a concave cylinder closing the full circle. Liner 122, TARGET 74 (`docs/data/target_holes_2026-10-01.json`) | `holes_of` | |
| `scripts/run_part.py`, `scripts/parts_table.py`, `scripts/abc_lists.py`, `scripts/abc_manifest.py`, `scripts/compare_pair.py`, `server/server_a/run_parts.sh`, `server/server_a/fetch_abc.sh`, `scripts/server_parts.sh` | the ABC test set on the institute server: download from the source, manifest, one record per part, the table with robustness, quality by class and triage (RUNNING.md) | | |
| `surface_orders.py` | the second quality measure (the user's rule of 2026-09-29): **containment** - the share of the input's surface that lies on or inside the model, 1.0 is the aim; **exact surface coverage** - the share of the input's surface that the model's surfaces reproduce in position (order 0), normal (order 1, within 2 degrees) and curvature (order 2, the shape operator by finite differences of the analytic distance function), and the same the other way round. Free-form faces are counted apart. The orders compare normals up to sign: the side of the material is not judged (a boss and a bore of the same radius reach order 2); tolerances 1e-3 × the diagonal | `containment`, `coverage`, `orders`, `input_surfaces`, `primitive_surfaces`, `shape_operator` | `test_surface_orders.py` |
| `subpart_cover.py` | the rough model as several sub-parts that are ADDED before any cut (the user's rule of 2026-09-29). Every primitive type is a function with a space of solutions (all boxes, all cylinders, all cones, all spheres, all tori); per sub-part each space is searched for the instance that holds the sub-part and reproduces the most of the input's surface exactly, the instances named by the input's own faces. Greedy splits at the input's planes, a split must gain `min_gain` exact coverage or save `min_saving` volume. `ResidualConfig.rough_subparts` (off by default), taken only with containment >= 0.995. Measured 2026-09-29 on ONE part (ABC 00000424, a disc of 32 faces, features off): sampled IoU 0.823 against 0.529 of the single box, exact coverage 0.84 against 0.47 | `cover`, `directions`, `candidates`, `split_planes`, `box_of`, `cylinder_of`, `cone_of`, `sphere_of`, `torus_of` | `test_subpart_cover.py`, `test_feature_phase.py` |
| `feature_stage.subpart_tool`, `ResidualConfig.feature_subparts` | the same cover for a feature piece that no tool explains, before the voxel fallback: the cover's primitives fused to one tool. They bound their sub-parts, so the tool is taken only when the sampled share of it inside the piece reaches `feature_min_precision`; then checked and validated like every tool. Off by default; not yet measured on a real part | `subpart_tool`, `subparts` (in `_feature_phase`) | `test_feature_phase.py` |
| `residual_loop.py`, feature phase: rules of 2026-09-29 | **groups**: a failed representative is followed by full guesses of the next members, at most `feature_group_tries` = 3; a member that finds a tool hands it on, also to members that failed before it. **Voxel fallback** (`feature_stage.voxel_tool`): the last resort of a piece, after every tool, source, group guess and the split: boxes of voxels at pitch tau/3 over 2 inside the piece, at most `feature_fallback_max_boxes` = 16, rejected under `feature_fallback_min_recall` = 0.25; its step lists what was tried. **Small pieces**: kept when congruent with an accepted piece or a stored tool (`feature_keep_known`) or when a real curved face of radius r has a volume of at least pi r^3 (`feature_min_radius_depth`). **Checks**: the per-feature check uses the builder's checked booleans and a volume balance; a failed check is an error in the step, not an empty row. Measured 2026-09-29 under load on stored liner pieces: fallback of a bolt hole 16 boxes, recall 0.729, harm 0.0 mm3; the 40 bores under the share floor all kept | `rounds`, `fallback`, `run_checks` | `test_feature_phase.py`, `test_feature_stage.py`, `test_exact_check.py` |
| `member_store.py` | members stored and checked for congruency (the user's rule of 2026-09-29; D-004: by geometry): the members of `detect_members` grouped by kind, section dimensions and length (2e-3 relative); per operation the first member that passes the exact judge is the source, its primitive is moved onto the others and judged there; a copy that fails is judged by itself, a failed member never rejects its group; the groups go into the tool store (`created:member <kind>`) and into the steps' info. Measured 2026-09-29 under load: frame 32 of 40 operations by congruency, program unchanged; r06 28 of 34, r14 6 of 12, r10 3 of 11 | `MemberStore`, `member_motion`, `move_primitive` | `test_member_alternatives.py` |
| `solution_choice.py` | the classic solution before the fallback (the user's rule of 2026-09-29): after a first-add sub-loop and the add/cut loop, before the voxel-cube fallback or the feature phase, the classic solution (plain first add, add/cut with cuts first) is computed on a fresh state; both are judged by the sampled exact IoU (`ibeam.ExactJudge`), the classic one replaces the first only when higher by more than `solution_margin`. `ResidualConfig.solution_alternatives`: auto, always, never. Only on the legacy path (`state.solution_ctx`); the margin 0.01 is a setting, the sampling's variance was not measured. Measured 2026-09-29 under load: never chosen on r06, r10, r14, r08 (classic sampled IoU 0.03, 0.00, 0.05, 0.00), 9-43 s more per part; the frame is not touched (recorded here and in `RUNNING.md` for r14 only; no data file) | `before_fallback`, `sampled_measure`, `reasons_to_try`, `open_share` | `test_member_alternatives.py` |
| `tools.py` | the type set of standard tools: 18 parametric boolean trees of Part-native primitives (holes, pockets, edges, turned features), their fitters from exact faces, the session's tool store | `Tool`, `Param`, `fit_piece`, `fit_revolved`, `fit_prismatic`, `fit_edge`, `fit_box_edges`, `catalogue_type_of`, `tree_signature`, `ToolRecord`, `ToolStore`; `CLASSES` | `test_tools.py` |
| `feature_groups.py` | pieces grouped by tool type: the analytic rule table, the `FeatureNet` interface (stub `NoOpinion`), their fusion | `describe`, `classify_analytic`, `piece_tensor`, `FeatureNet`, `NoOpinion`, `fuse`, `PieceType`, `group_pieces`; `RULES` | `test_feature_groups.py` |
| `difference_grade.py` | grades a guess by what its difference to the input consists of: skin (depth ≤ δ0 = min(1e-3 diagonal, τ/10)), feature (score ≥ 0.9), open; `reach` is a prediction. Takes the exact residual's IoU as given; `apply_features` returns `iou` also for an invalid build | `grade_difference`, `classify`, `piece_depth`, `default_delta0`, `apply_features`, `format_grade` | `test_feature_stage.py` (`classify`; needs FreeCAD) |
| `user_guess.py` | the first guess by hand: shapes the user places roughly, refined against the mesh | `UserShape`, `save_first_guess`, `load_first_guess`, `as_operations`, `refine` | `test_user_guess.py` |
| `advice.py` | asking the user when a guess is hard; Qt-free protocol | `Question`, `Answer`, `Adviser`, `NoAdviser`, `ScriptedAdviser`, `QueueAdviser`, `AdviceConfig` | `test_advice.py` |
| `result_schema.py` | the one result schema of cascade and loop (mm, world frame) | `StageSource`, `Primitive`, `Operation`, `ReconstructionResult`; `OPS` = add, cut, intersect | — |
| `result_to_program.py` | schema → `reco_field` program, so that loop results are built by the same FreeCAD builder as route A | `primitive_to_sp`, `program_model`, `program_dict`, `primitive_from_fit`, `transform_primitive` | `test_booleans.py` |
| `brep.py` | exact surface inventory of a STEP part (runs `scripts/freecad_brep_faces.py`; cached by file bytes, deflection, script). Raises (RuntimeError, `TimeoutExpired`), unlike the judge | `is_step`, `step_faces` | — |
| `solution.py` | statistics vector of a whole solution (`docs/thesis/chapters/10b_solution_spaces.tex`): voxel IoU on the loop's grid, covered share, strict and soft coverage, position and orientation terms, score = covered × coverage. Never OCCT; `score` is 0 when the surface terms fail (`surface_error`) | `surface_terms`, `solution_stats` | — (only stored and read back: `test_run_store.py`) |
| `run_store.py` | stored runs, one directory per run, and their comparison. A program is written whole at any depth (`plain(limit=None)`, no recursion; until 2026-10-01 composites were cut to text below 12, then 400 JSON levels); `truncated` finds the nodes an older store cut, `load_run` then reads `program_recovered.json` (`scripts/recover_run_program.py`: from the feature phase's `features_job.json` in the cache) | `save_run`, `load_run`, `list_runs`, `resolve`, `previous_run`, `diff_runs`, `store_root`, `truncated` | `test_run_store.py` |
| `run_tensors.py` | the run's input tensors computed once and stored sparse in a band around the surface. **Not used by the loop** (its measuring script `scripts/measure_run_tensors.py` was removed 2026-09-30; in git history) | `SparseGrid`, `RunTensors`, `piece_channels`, `parity_grid`, `surface_band` | `test_run_tensors.py` |
| `timing.py` | named, nested wall/CPU sections; free unless a `Recorder` is active | `timed`, `timed_fn`, `Recorder`, `format_sections`, `format_summary` | `test_timing.py` |
| `cache.py` | content-addressed cache of builds, voxelizations and predictions; the wall-thickness cache | `enable_cache`, `disable_cache`, `default_cache`, `Cache`, `wall_thickness_cached`, `wall_cache_dir` | `test_cache.py` |
| `cascade_node_support.py` | logic of the "Cascade Reconstruct" FreeCAD node | `cascade_program`, `run_cascade`, `residual_config`, `summarize` | `test_cascade_node.py` |

**The starting-shape stage** (`start_shape.py`, the user's rule of 2026-09-26; `ResidualConfig.start_shape`,
GUI "starting shape", `--start`). The starting shape holds the whole part and is at least partly exact: some of the
part's real surfaces lie on it. Excess is tolerated; cuts remove it.

| `start_shape` | what runs at the start | order of operations after it (`cuts_first=None`) |
|---|---|---|
| `legacy` (default everywhere) | type stage → member or hollow sub-loop → graded first add (the cascade's answer and alternatives among the bounding covers, `_grade_cover`) | adds first |
| `auto` | the starting-shape stage; the sub-loops never run at the start. No candidate passes → empty program, no grade | cuts first |
| a named type of `START_TYPES` | that candidate, forced; the run ends with "no starting shape of type ..." if it does not exist | cuts first |

| step | function | what it does |
|---|---|---|
| 1 inventory | `surface_inventory` | STEP: the B-rep faces from `freecadcmd` (`brep.step_faces`, `scripts/freecad_brep_faces.py`; cached in `~/.cache/shape_reco/brep/`, `$SHAPE_RECO_BREP_CACHE`), grouped by surface, each with kind, area share and `dominating` (the whole part on its material side). STL: strict RANSAC, cylinders and spheres first, then planes |
| 2 candidates | `start_candidates`, `dominating_cover` | the dominating cover (intersection of the dominating surfaces' solids and a closing oriented box), the dominating major surfaces, the bounding shapes (oriented box, world box, cylinder, cylinder sector, sphere, box & cylinder), the cascade's answer |
| 3 grade | `grade_start` | coverage (area share of the part's surface on the candidate), excess (volume(candidate) − volume(part), exact by manifold3d), contains (share of the part's surface samples inside), score |
| 4 choice | `choose_start` | only candidates with contains ≥ `MIN_CONTAINS` (0.995) and every composite operand carrying ≥ `MIN_OPERAND_SHARE` (0.01) of the surface. `start_score` "product" (default): the highest score = covered share of the generated volume × surface coverage. "band": within `COVERAGE_BAND` (0.05) of the best coverage, the least excess. A candidate without an exact excess (no manifold3d, a solid that failed) is dropped |
| 5 variation | `vary_start` | coordinate search in the shape's parameters at the grid pitch, at most `start_vary_rounds` (4) rounds (`start_vary`, default on). Operands made from an exact surface are never varied. Skipped when the feature phase is on with `rough_cut_grade` > 0 |

`reconstruct_variants(input, config, k)` gives k solutions from the k best starting shapes, with the statistics
vector of `solution.py`. Measured results: `RUNNING.md`, section 4.2, and `docs/FRAMEWORK_CHANGES_2026-09-26.md`.
The comparison matrix over parts × starting shapes (`scripts/start_matrix_cell.py`, `start_matrix_report.py`) had
11 of 153 cells done on 2026-09-26; it has not been completed. The scripts name `docs/START_SHAPES.md`, which does
not exist.

Details of the loop that are not in the table:

- **Members.** A member is kept only if its length is at least `min_length_ratio` × its largest section side.
  `ibeam.detect_members` defaults to 2.0; the loop passes `ResidualConfig.member_min_length_ratio` = 1.5, because
  2.0 dropped the frame's 8 transverse plates (712 x 444 x 20 mm, ratio 1.6). Sections snap to
  `dataset_gen/src/din_profiles.py` within `SNAP_MAX_REL` = 0.05; the frame STL matches at unit scale × 100.
  `ExactJudge` judges adds against the exact mesh (igl winding number), because a 12 mm web gains no voxels on a
  whole-part grid. Sub-loops can be switched off: `$SHAPE_RECO_IBEAM=0`, `$SHAPE_RECO_HOLLOW=0`;
  `hint_subclass` selects one.
- **First cover** (`_bounding_cover`, `_grade_cover`): every covering mode is built and graded (oriented and
  world-axis box, bounding cylinder as sector and full, bounding sphere, box & cylinder). The grade samples each
  candidate's own surface: `on` the input's surface, `far`, `near`; score `on − near`. The cascade's answer and up to
  three alternatives are graded with them (since 2026-09-25; before, a cascade answer at IoU ≥ 0.9 was taken
  ungraded). Only candidates containing `COVER_MIN_INSIDE` (0.995) of the surface samples count, a composite only
  when each operand has ≥ `COMPOSITE_PART_ON` (0.5) of its surface on the input; ties within 0.02 go to a single
  primitive, fewer excess pieces, the smaller volume; none passes → the oriented bounding box. Liner: the R 192
  half-cylinder, voxel IoU 0.702 → 0.870 (`docs/LINER.md`, 2026-09-25, measured with the rule before this grade: the
  cylinder won by ≥ 5 points of surface share).
- **Cuts chosen by the surface** (`surface_choice`, default on): strict RANSAC cylinders as extra cut candidates;
  among passing candidates with ≥ 1/4 of the best gain, the one most of the nearby input surface lies on wins.
  Liner bore R 116.6 → R 125.5 (`docs/LINER.md`).
- **More booleans** (`more_booleans`, default on): half-space cuts, "intersect" with the part's covers
  (`_covers`), composites box & cylinder and box − cylinder (D-001). Each needs ≥ `boolean_min_surface` (0.2) of the
  nearby surface on it. Primitive kinds `halfspace` and `composite` (`parts` + `combine`).
- **Boolean tools** (`boolean_tools`, default on): pairwise booleans of a piece's best single candidates, and
  every accepted operation re-placed at the piece (`tool_library_size` 6).
- **Operation budget.** `max_ops=None` takes `reco_field.peel.predict_ops_budget`; on the starting-shape path the
  budget also grows with the surface inventory (`budget_from_surfaces`, `_ops_budget`).
- **Grid "tau"** (`cascade.BoxGridSpec`): one pitch, its own cell count per axis. The loop fits components on a
  window (`_window`, `WINDOW_MARGIN`), keeps float32 SDFs, rejects adds that leave the part's box.
- **Parity.** `residual_loop.parity_occupancy` calls `field_channels.parity_axes`; it is not a second
  implementation.
- **Timing.** The loop, `cascade.type_stage`, `run_cascade`, `ibeam.detect_members`, `ExactJudge` and the GUI's
  `CascadeFolder` are instrumented.

### 5.2 The feature phase and the feature stage

Two callers share `feature_stage.py`:

| caller | input | what it does with a guess | entry |
|---|---|---|---|
| feature stage | STEP only | lists the features, does not apply them; writes `rough.step`, `rough.json`, `rough_parameters.json`, `pieces/`, `features.json`, optionally `features.FCStd` | `feature_stage.feature_stage`, `scripts/feature_stage.py` |
| feature phase of the loop | STEP; STL up to `feature_stl_max_faces` (12000) triangles | applies it and validates it by a FreeCAD build | `residual_loop._feature_phase`, `ResidualConfig.feature_phase` |

Functions of `feature_stage.py`:

| group | functions |
|---|---|
| rough model | `rough_stage` (the loop up to its first `residual` step), `snap_to_faces`, `extend_members` |
| difference | `difference_pieces` (calls `reco_field.exact_check.exact_residual(..., pieces_dir=...)`) |
| candidate sources | `metadata_candidates` (the piece's exact STEP faces), `fit_candidates` (`reco_field.fit.fit_best`), `cascade_candidates` (the cascade on the piece's parity voxels at pitch ≤ τ/3), `hole_tools`, `contact_tool`, `tool_candidates`, `rounded_box`, `rounded_corners` |
| scoring | `sampled_score` (volume IoU), `PartScorer` (precision and recall as the operation), `excess_region`, `SampledJudge`, `contact_grade` |
| search | `guess_candidates`, `guess_feature`, `extend_ends` |
| congruent pieces | `piece_descriptor`, `congruent`, `group_congruent`, `transfer_candidates`, `best_transfer` |
| splitting | `piece_grid`, `piece_resolution`, `segment_piece`, `segment_necks`, `segment_surfaces`, `segment_axis`, `merge_small` |
| thin walls | `thin_wall_threshold`, `piece_thickness`, `suppressed_reason`, `open_thin`; `THIN_WALL` = 2e-3 of the input's diagonal |
| output | `format_features`, `write_fcstd`; `Candidate`, `Feature` |

One iteration of the feature phase:

1. Build the program (`feature_build`, default `reco_field.export.build_in_freecad`).
2. Exact difference with pieces (`feature_difference`, default `exact_check.exact_residual`). One difference writes
   at most `feature_max_written_pieces` (200) pieces.
3. Drop pieces under `feature_min_share` (1e-5) of the input, pieces tried before, and suppressed thin pieces.
4. Group the pieces by tool type and congruence (`feature_groups.group_pieces`; step `feature_groups`). Order the
   groups by `feature_piece_order`: "volume per face" (default) or "volume".
5. With `feature_target_iou` set, take the groups with the most volume first and stop taking when the IoU
   estimated from the volumes, (V_in − missing) / (V_in + invented), reaches the target (`select_for_target`). The
   estimate counts a piece as explained in full by its tool.
6. Per representative piece, `guess_candidates` asks the sources in this order. The first passing one ends the
   search.

   | # | source | source label in the step |
   |---|---|---|
   | 1 | the session's tools of the piece's type | `session:<id>` |
   | 2 | the catalogue's fits of the ranked types | `catalogue:<type>` |
   | 3 | hole stacks; with other real faces, fused with the contact tool | `tool (hole stack)`, `tool (contact faces + hole stacks)` |
   | 4 | the contact-face tool | `tool (contact faces)` |
   | 5 | exact STEP faces, surface fitters, the cascade | `step faces`, `fit`, `cascade` |
   | 6 | boolean tools (rounded box, pairwise booleans; at most `feature_tool_max_operands` 9) | created tools |
   | 7 | splitting the piece on its own grid, as a fallback pass | step `feature_split` |

   Sources 3 and 4 need a STEP input ("step faces" among the sources) and `feature_scoring` "part". A partial hole
   stack is judged by its precision alone; the rest of the piece comes back in the next difference.
7. Congruent pieces get the representative's primitive moved by the rigid motion of centroid and principal axes
   (source `pattern:<id>`); a full guess when that scores below `feature_min_score`.
8. A guess is applied when its score ≥ `feature_min_score` (0.85) and its precision ≥ `feature_min_precision`
   (0.999). Its ends are extended (`extend_ends`).
9. Exact check per feature before the batch is validated (`exact_check.feature_checks`): a feature whose harm
   exceeds `feature_check_tolerance` (1e-3) × its volume is rejected.
10. Validation: one build and one exact difference for all guesses of an iteration (`feature_batch`); a failing
    batch is halved until the failing operation is found. Kept if one valid solid and the exact residual dropped
    by ≥ `feature_min_gain` (0.25) × the pieces' volume. When OCCT fails on a pair, `SampledJudge` judges.

Stop reasons: "converged" (residual ≤ `feature_tolerance` × input, or no piece above the floor), "nothing applied",
"max iterations" (3), "time budget" (900 s, checked between pieces and sources), "max pieces"
(`feature_max_total_pieces` 40 full guesses), "max applied" (60), "error: ...". The phase never raises. Every
FreeCAD call runs under `feature_call_timeout` (300 s).

Step kinds (the contract with `reco_gui`), all with `info["phase"] == "feature"`: `feature_residual`,
`feature_groups`, `feature_piece`, `feature_feedback`, `feature_split`, `feature_candidate`, `feature_accepted`,
`feature_rejected`, `advice_asked`, `advice_given`, `feature_done` (`FEATURE_KINDS`). A piece's mesh in a step has at most 20000 triangles.

Hooks, no-ops by default: `first_guess` (`user_guess.py`), `adviser` / `advice` (`advice.py`; steps `advice_asked`,
`advice_given`), `feature_tool_store` (a `tools.ToolStore` or its JSON path), `piece_feedback` (a
`piece_feedback.json` file or a run directory; no module of `reco_gui` on this branch writes that file),
`feature_net` (the module `reco_engine/feature_net.py` is not on this branch; a value is reported and ignored).

Output: without `out_dir` the phase writes `~/.cache/shape_reco/features/<part>_<time>/features/`, with
`features.FCStd` (`exact_check.write_features_document`).

Measured results of the phase: `RUNNING.md`, "Feature phase", and `docs/reports/hole_stacks/hole_stacks.tex`.

### 5.3 Voxel building blocks (from the former route C)

Route C, the greedy voxel loop and its drivers, was removed on 2026-09-30 (branch `tidy`): `greedy_reconstruct`,
`pipeline`, `void_pipeline`, `octree_search`, `octree_void_pipeline`, `notebook_pipeline`, `live_pipeline`,
`live_void_pipeline`, `ensemble`, `ensemble_live`, `live_gui`, `fit_score`, `fit_choice`, `shape_probe`,
`node_support`, the loop guards of `loop_guards`, the GUI's voxel mode, its FreeCAD nodes and scripts. They are in git
history (`git log --diff-filter=D --name-only`; `git checkout <commit>^ -- <path>` restores one); the thesis chapter
08 and §14 below record what they did and measured. The modules below stay because the main line uses them.

| module | purpose | main functions and classes |
|---|---|---|
| `model_registry.py` | loads CNN checkpoints; in-process or via subprocess when torch is missing. The Type CNN's input kind comes from its own `config.json`; `exact_field` is refused | `Models`, `load_models`, `registry_type_dir` |
| `predict.py`, `predict_cli.py` | inference wrappers and their subprocess form (the cascade's CNN sources) | `predict_type`, `predict_rotation`, `predict_dimensions`, `TypePrediction` |
| `geometry.py` | builds a primitive STL with the macros that generate the training data | `build_primitive_stl`, `assemble_reconstruction_stl` |
| `compare.py` | voxel IoU, precision, recall, subtraction | `compare`, `metrics_bool`, `subtract`, `BooleanMetrics` |
| `placement.py` | voxel ↔ world conversion (dataset generation, the cascade) | `estimate_placement`, `array_to_world`, `array_to_world_position` |
| `local_search.py` | bounded search of position and scale on voxel arrays; `spill_fraction` | `refine_position`, `LocalSearchConfig`, `SearchObjective` |
| `reconstruct_cli.py` | headless entry of the add-cut loop, JSON result | `--input-stl`, `--workdir` (`--mode add_cut` is the only mode) |
| `loop_guards.py` | the largest completely filled box: the box cover of the voxel-cube fallback | `largest_box_in_blocks`, `max_rectangle` |
| `box_orientation.py`, `sphere_fit.py`, `cylinder_fit.py` | analytic fits on a voxel grid (in-scale mode): cascade sources | `BoxOrienter`, `refine_box`, `SphereFitter`, `CylinderFitter` |
| `fused_predict.py` | inference of `FusedNet` (registry entry `net_fused`) | `predict_fused`, `fused_estimate_for_topology`, `load_fused_model` |
| `fit_grade.py` | tolerance-aware grade of a fit (used by `void_fit`) | `grade`, `Grader`, `FitGrade`, `GradeConfig` |
| `void_fit.py`, `void_quality.py` | a void fitted to the grade; exact classifier of a void (the add-cut loop's cut checks) | `fit_void`, `assess_void`, `VoidAssessment` |
| `cache.py` | content-addressed cache of predictions and voxelizations | `Cache`, `cached_predictions`, `enable_cache` |

### 5.4 The validation cycle: reconstructions of shapes with a known CSG tree (since 2026-10-01)

Shapes whose CSG tree is known exactly are reconstructed by the main line (`scripts/run_part.py`, on the server), and every
run is rated against its tree: a multivariate metric, a tier, and the separation point - the step of the run's path whose
operation caused the defect. Branch `validation`; plan `~/.claude/plans/sleepy-coalescing-conway.md`.

| module | purpose | main functions and classes |
|---|---|---|
| `csg_surface.py` | exact boundary samples of an sp tree: area-uniform on every leaf's candidate surfaces, kept where the leaf's and the tree's sign change and the leaf decides the tree (internal and coincident faces drop out); outward normals, owner op; bracketed distances (\|sdf\| below, nearest sample / Newton projection above) | `compile_tree`, `compile_program`, `boundary_samples`, `surface_distances`, `CompiledTree.evaluate` (value + deciding leaf) |
| `truth.py` | the known trees: aegir's stacked set (`program_to_dict` JSON) and exact v5 test records (plain / sector / band primitives, carved), `truth_gate` (tree and mesh are the same solid) | `truth_from_aegir`, `truth_from_v5`, `v5_exact`, `truth_gate`, `program_from_dict`, `program_json` |
| `run_path.py` | the run's path, written while it runs (`path.jsonl`): operations entering / leaving the program per step (snapshot diff per branch), candidates with primitive, gain, reason | `PathRecorder`, `load_path`, `program_from_path` |
| `validation.py` | the rating: volume history (each op folded once; per-prefix trajectory), surface metrics (std, RMS, max, coverage both ways over a tau ladder), leaf matching (diagnostic), hard conditions, `GRADE_V1` tiers, attribution (over-cut, over-clip, omitted add / cut, uncut excess, over-add, surface deviation) and the separation point | `rate_run`, `rate`, `volume_history`, `surface_metrics`, `attribute`, `grade`, `regrade` |

Driver: `scripts/validate_known.py` (`select`, `build`, `start`, `status`, `rate`, `regrade`, `pull`, `report`, `review`);
`server/server_a/run_parts.sh` takes `TRUTHS` (rates every run) and `CPUSET`. The tree comparison is diagnostic only - the
same solid has equivalent trees - so the separation point is defined geometrically, by the write history of the volume
points and the path's candidates.

### 5.5 Shape classes and the shape-class atlas (since 2026-10-29)

Every part gets a versioned class key `CLASS_V1` = topology . surface family . tree complexity, so reconstruction
results are counted per kind of geometry instead of per part: where the method lacks, and where the tests lack. Branch
`shape-atlas`; plan `~/.claude/plans/functional-leaping-papert.md`.

| module | purpose | main functions |
|---|---|---|
| `shape_class.py` | the facts of one part and its key. **T**: Betti numbers of the solid (b0 solids, b1 genus summed over the shells, b2 voids; `multi`, else genus bins g0 / g1 / g2-3 / g4-9 / g10+, suffix `v` for voids); b1 split into `b1_holes` (handles a through hole stack explains: `hole_stack.hole_stacks` on the part's concave faces) and `b1_rest` (window/frame handles). **S**: P planes only, A turned (curved share >= 1/3, >= 80 % of it on one axis line), D sphere + torus >= 5 %, M planes with curved features, F free-form faces; flag `blend` (cylinder/torus strips <= 0.3 of their surface). **C**: leaf bins of the known tree (`tree_facts`: leaves, depth, ops, subtypes, partial leaves, axis directions), else the leaves estimated from the carrier surfaces by a calibration on the known trees, else carrier bins K (`c_source` says which). STEP: one freecadcmd pass (`exact_check.exact_facts`), cached by the file's bytes; STL: a marked mesh fallback (S "?"); a known tree without a STEP is built from its program (`step_of_program`) | `describe`, `tree_facts`, `classify`, `surface_facts`, `mesh_descriptors`, `mesh_betti`, `step_of_program` |
| `reco_field/exact_check.py` | `betti(shape, tol, per_shell)`: b0, b1, b2 (and the genus per shell) from the solid's merged tessellation; `genus` is its b1. `exact_facts(path, stl_out)`: the faces as `exact_faces` gives them, plus the Betti numbers and topology counts. The Betti numbers are computed BEFORE the face pass, because the face-wise tessellations left the solid's open | `betti`, `genus`, `exact_facts` |

Driver: `scripts/shape_atlas.py` (`describe` per corpus / glob / list, resumable and parallel; `calibrate` the leaf
estimator; `join` the descriptors with run directories (`path_atlas._row`) and the ABC table; `report` the cells with
Wilson intervals, failure causes, capability and coverage gaps, and the Firth logistic model; `select` class-stratified
lists). `validate_known.py fork --strata-from` takes the classes as strata, also from an unbuilt pool.

## 6. `reco_field/` (route A, and FreeCAD services for route B)

| module | purpose | main functions and classes | tests |
|---|---|---|---|
| `fit.py` | fits sphere, box, cylinder, cone, torus, plane to the closest-point field (Tukey IRLS, RANSAC starts); round shapes with sweep `u`, sphere and torus also `v` | `fit_best`, `fit_sphere`, `fit_plane`, `fit_box`, `fit_box_bounding`, `fit_cylinder`, `fit_cone`, `fit_torus`, `rank`, `FitResult`, `Samples`, `make_samples` | `test_reco_field_fit.py` |
| `peel.py` | the peeling loop: base, cuts, clips, additions; two-way chamfer | `peel_steps`, `peel`, `PeelStep`, `PeelResult`, `Op`, `predict_ops_budget`, `OpsBudget`, `invented`, `input_occupancy`, `real_voxel_occupancy`, `build_model`, `mesh_from_sdf`, `chamfer`; `MAX_OPS` = 24, `INVENTED_SLACK` = 0.02 | `test_reco_field_peel.py`, `test_bounding_box_mode.py`, `test_mesh_clusters.py`, `test_octree_clusters.py`, `test_real_voxel_occupancy.py`, `test_verify_freecad.py` |
| `export.py` | builds a program in FreeCAD as STEP and STL; checks it against the input | `build_in_freecad`, `build_program`, `build_groups`, `build_split_in_freecad`, `program_to_dict`, `mesh_chamfer`, `export_check`, `Exported` | `test_reco_field_export.py`, `test_step_export.py`, `test_freecad_groups.py` |
| `gdml_export.py` | the program as a GDML file for Geant4 / ROOT: primitives as GDML solids, operations as boolean solids; a boolean's rotation is written as the angles of R, a physical volume's as those of R^T (Geant4's reader inverts only the latter); the tree shape (`tree=`): grouped cuts + balanced unions (default since 2026-10-01), grouped cuts + multiUnion (Geant4 only), or the program's own tree; a full turn is written as 2 pi to the last digit (G4Torus, G4Polycone) | `export_gdml`, `export_result`, `shape_tree`, `model_of_program`, `gdml_angles`, `matrix_of_angles`, `read_gdml` | `test_gdml_export.py` |
| `freecad_builder.py` | runs inside `freecadcmd`; no project imports | `build`, `boolean`, `main`, `main_groups`, `primitive_object`, `placed_leaves`; `BOOLEAN_LOG` | — |
| `exact_check.py` | the exact judge: residual of a model against the input by OCCT booleans (general fuse, then separate booleans), genus, pieces with their faces; `incremental_residual` for a model with features; `sampled_iou` as the independent cross-check the loop's `_exact_grade` uses | `exact_residual`, `incremental_residual`, `exact_faces`, `sampled_iou`, `feature_checks`, `write_features_document` | `test_exact_check.py` (16; no test reaches a refusal branch of the general fuse or the separate booleans) |
| `freecad_worker.py`, `freecad_worker_server.py` | one long-lived `freecadcmd` per host process; opt-in `SHAPE_RECO_FREECAD_WORKER=1`; restart every `RESTART_EVERY` = 50 jobs | `enabled`, `run_script` | `test_freecad_worker.py` |
| `field_hints.py` | the point-cloud networks as an additional start of `fit_box` | `FieldNetHints` | `test_field_hints.py` |
| `node_support.py` | logic of the vector-field FreeCAD nodes | `peel_program`, `export_program`, `field_grade` | — |

- **Acceptance in the peeling loop:** an operation must explain at least `min_gain` (0.005) more of the input surface
  and must not raise `invented` (the share of the model's surface off the input) by more than `invented_slack`.
  Unexplained points inside and outside the model are clustered separately (cutters, additions).
- **Budget:** `max_ops=None` predicts the budget from the part's genus (`predict_ops_budget`). Its rationale is
  retracted (B-062, `docs/FRAME_FAILURE.md`).
- **Options of `peel_steps`:** `bounding_box`, `cuts_first`, `max_candidate_invented`, `split_multibody`,
  `mesh_clusters`, `octree`, `octree_real_voxels`, `verify_freecad` (`PeelStep.freecad`). All opt-in.
- **`real_voxel_occupancy`** uses ray parity (`field_channels.parity_axes`) since 2026-09-24. The earlier
  stltovoxel version clipped parts far from the origin; every `octree_real_voxels` measurement made with it is void
  (the function's docstring, `docs/FRAME_FAILURE.md`).
- **Checked booleans** (`freecad_builder.boolean`): every boolean is computed plain and fuzzy (1e-4). Where the
  volumes differ, 256 seeded points classified against the operands decide, and `BOOLEAN_LOG` gets an entry. OCCT's
  plain cut and common failed silently on the liner's tools.
- **Groups** (`export.build_groups`, `freecad_builder.main_groups`): every operation's tool as an object of one
  document, grouped into adds and cuts, no final boolean. `tools="solid"`: each tool with its own booleans.
  `tools="primitives"`: parametric Part primitives with their placements only.
- **Consistency of the exact residual:** V_in − r+ = V_model − r− and, as a third check, the intersection's volume.
  The first alone also holds for garbage (measured on `s03_rod`: IoU 0.09 for a model that was the part to
  1e-4 mm). A total failure (common 0, r+ = V_in, r− = V_model) passes all three; only the sample of the shared
  volume (1500 points in the joint box) can refuse it, and it cannot when the part fills a small share of the box
  (review r13, §3 "The judges of a run"). Balance tolerances: 1e-5 V_in (general fuse), 1e-4 V_in (separate
  booleans), 1e-5 V_in for the incremental budget; fuzzy tolerances relative to the diagonal, in the incremental
  path absolute (1e-4 mm to 1e-2 mm).

Design and measured results: `docs/VECTORFIELD.md`. Part shapes against FreeCAD 1.0.0:
`docs/archive/freecad_part_shapes.md`.

## 7. `reco_gui/`

A Qt window (PyQt5 + VTK) that runs a reconstruction one stage at a time. It observes and paces; it cannot
override a decision. Use: `RUNNING.md`, "GUI".

| module | Qt-free | purpose | main names | tests |
|---|---|---|---|---|
| `__main__.py` | — | entry point and options; forces the X11 Qt platform when `DISPLAY` exists | `main` | — |
| `app.py` | no | main window: toolbar, docks, heartbeat, child-process handling | `MainWindow`, `running_children`, `terminate_children` | `test_reco_gui_window.py` |
| `decision_bar.py`, `nephogram.py` | no | the non-blocking "keep or try another path?" bar above the viewport (2026-10-15; with `RunSettings.alternatives` > 0 the window goes on by itself instead, 2026-10-29) and the comparison export it offers: the guess and the input as STL and as PLY coloured by the exact surface distance (HSV nephogram, scalar field `distance`), stats, README and ALWAYS the band histograms both ways (`histogram.png`, opened by the FreeCAD macro too); `stored_nephogram` makes one for a stored run (the program built when the run has no model mesh) | `DecisionBar`, `ExportThread`; `write_nephogram`, `write_histogram`, `stored_nephogram`, `hsv_colours`, `open_in_freecad`, `open_in_cloudcompare` | `test_decision_bar.py`, `test_nephogram.py` |
| `benchmark_panel.py` | no | the Benchmark tab (2026-10-11): a part's stored runs ranked; since 2026-10-29 the comparison of the alternatives - check boxes (the newest 3 checked), us/event and p95 columns, 'Event time (aegir)' and 'Nephogram' in the background, the selected run's histograms and the checked runs' accuracy-vs-cost plot | `BenchmarkPanel` (`set_checked`, `event_time`, `nephogram`), `scatter_png`, `row_texts` | `test_reco_gui_benchmark.py` |
| `logo_token.py` | no | the logo's badge turning slowly like a token (12 s a turn, perspective, both faces the right way round, rim, sheen, shadow; QPainter, no GL) as the 3D views' background while nothing is on screen and no run computes (2026-11-10); the views stack it in front of the VTK widget (`FieldView3D._stacked`, `set_running`), its timer runs only while visible; image: `packaging/logo/shape-reco-3d-badge.png` (1024 px), else the 256 px icon | `LogoToken`, `token_stack`, `badge_path` | `test_logo_token.py` |
| `settings_panel.py` | no | settings dock, collected into a `RunSettings`; the detail slider | `SettingsPanel` (`target_iou`, `set_target_iou`, `pick_random`, `set_input`) | `test_reco_gui_window.py` |
| `flow_bar.py`, `start_part_dialog.py` | no | the window's flow as one line of steps (Part > Starting part > Reconstruct > Compare > Choose & export, done / current / open, Next; 2026-10-29) and the starting-part dialog: automatic, a basic shape around the part or a sub-part (`user_guess.bounding_shape`), or an exact shape (`user_guess.exact_shape`), passed as `ResidualConfig.first_guess` | `FlowBar`; `StartPartDialog` (`entry`, `summary`) | `test_start_part.py` |
| `setup.py` | yes | `RunSettings` (method `cascade` by default, or `field`), discovery of parts, `make_session` | `RunSettings`, `discover_examples`, `discover_test_parts`, `type_cnn_input_kind`, `make_session` | `test_reco_gui_setup.py` |
| `session.py` | yes | advances a step generator and folds each step (with the mode's own fold) into an immutable state | `StepSession` | `test_reco_gui_window.py` |
| `worker.py` | no | the engine in a `QThread` | `Worker` | — |
| `cascade_session.py` | yes | cascade mode: folds `residual_steps` into `CascadeState`; passes the feature settings the config has a field for | `CascadeFolder`, `CascadeState`, `FeaturePiece`, `FeatureView`, `make_cascade_session`, `feature_config`, `config_defaults`; `FEATURE_FIELDS`, `CASCADE_PACED_STAGES` | `test_reco_gui_cascade.py`, `test_reco_gui_feature.py` |
| `model_mesh.py` | yes | cascade mode: model and candidate meshes by exact mesh booleans (manifold3d); SDF fallback without it | `ModelMesher`, `primitive_solid`, `available` | `test_reco_gui_cascade.py` |
| `viewer_cascade.py` | no | 3D view of the cascade mode, in mm | `CascadeView3D`, `layer_meshes` | `test_reco_gui_cascade_view.py` |
| `feature_panel.py` | `FeatureTrace` yes | the dock "Feature (live)": the piece in work alone, FreeCAD's face metadata, every guess with precision, recall and verdict | `FeatureTrace`, `FeaturePanel`, `PieceRecord` | `test_reco_gui_feature_panel.py` |
| `op_tree.py` | no | Operations dock: the program as Fusion / Cut / Common nodes, and every step by component | `csg_tree`, `OperationTree` | `test_reco_gui_cascade_view.py` |
| `stats_panel.py` | no | live statistics of the run: cores, memory, steps on one time axis | `StatsPanel` | — |
| `rebuild.py` | `rebuild`, `groups` yes | FreeCAD build on demand: the folded solid with its chamfer, or the grouped tools | `rebuild`, `groups`, `groups_text`, `grade_text`, `FreecadRebuild` | `test_reco_gui_rebuild.py` |
| `open_run.py` | `load_stored` yes | the Open dialog and what the window shows of a stored run | `list_rows`, `load_stored`, `summary_html`, `OpenRunDialog` | `test_reco_gui_open_run.py` |
| `preview.py` | no | preview of a picked part before a run, loaded in its own thread | `load_preview`, `PreviewLoader` | `test_reco_gui_preview.py` |
| `field_session.py` | yes | vector-field mode: folds `peel_steps` into `FieldState` | `FieldFolder`, `FieldState`, `make_field_session` | `test_reco_gui_field.py` |
| `viewer_field.py` | no | 3D view of the vector-field mode, in mm | `FieldView3D` | `test_reco_gui_field.py` |
| `report.py` | yes | a state as the detail panel's HTML; log lines | `render_state`, `render_cascade_state`, `render_field_state`, `render_run_diff`, `feature_log_lines`, `candidates_summary` | — |
| `dataset_gen_setup.py`, `dataset_gen_panel.py`, `dataset_gen_window.py`, `dataset_gen_main.py` | setup yes | second window: a form around `src.dataset_generator`, one `QProcess` per shard | `DatasetGenSettings`, `build_args`, `shard_jobs`, `summarize_output`, `DatasetGenWindow` | `test_reco_gui_dataset_gen_setup.py`, `test_reco_gui_dataset_gen_window.py` |

Limits: the cascade and vector-field views are real geometry (the voxel mode was removed 2026-09-30). `model_mesh`'s SDF fallback loses walls thinner
than a 72³ cell (the frame's model was empty from the 4th operation on).

## 8. `dataset_gen/`

`src/dataset_generator.py` is the CLI. For a topology it samples parameters (`src/dataset_utils.py`,
`PARAMETER_SPECS`) and a rotation, runs the matching macro of `FCMacros/` and voxelizes the STL. Every sample is
recorded in `dataset_metadata.jsonl`, the one manifest schema all downstream code reads.

| mode | varies | fixed per group | trains |
|---|---|---|---|
| `type` | parameters and rotation | — | Type CNN |
| `measure` | rotation | parameters | Dimension CNN |
| `rotation` | parameters | rotation | Rotation CNN |

| module (`dataset_gen/src/`) | purpose | main functions and classes |
|---|---|---|
| `dataset_generator.py` | the generator CLI; `--export-slurm-job` writes a SLURM array job | `execute_macro`, `resolve_macro`, `make_sample_stl`, `finish_sample` |
| `dataset_utils.py` | `TOPOLOGIES` (17 names), `PARAMETER_SPECS`, `MACRO_MAP`, sampling | `sample_parameters`, `sample_rotation`, `Rotation`, `Sample` |
| `voxelization.py` | STL → `.bin`. Default (since 2026-10-01, T-19): ray parity at the voxel centres of the in-scale grid (scale mode `fixed`, anchor `origin`; D-vox). stltovoxel is legacy: on request (`voxelizer=`, `SHAPE_RECO_VOXELIZER=stltovoxel`, `--voxelizer stltovoxel`), and for the fit/uniform/centred grids, which only it makes (a `LegacyVoxelizerWarning`; an explicit parity request there raises). `report["voxelizer"]` is the one that ran; the generators record it as `placement.voxelizer` | `voxelize_stl_to_bin`, `voxelizer_for`, `assemble_pngs_to_bin`, `assemble_pngs_in_scale` |
| `cache_keys.py` | one key recipe for the on-disk caches (voxel cache, voxfield, field_kinds, sdf, exact field): each input file's identity (path, size, mtime), the voxelizer, a code fingerprint (module source without docstrings and comments, plus its live UPPER_CASE constants) and the parameters | `cache_key`, `code_fingerprint`, `file_identity` |
| `voxel_noise.py` | noise in the voxelization layer: salt, pepper, foreign blobs, one dilate/erode pass; pure numpy; seeded from the generator's `--seed`; applied after either voxelizer | `VoxelNoiseConfig`, `apply_voxel_noise`, `add_noise_arguments`, `noise_spec_from_args` |
| `inscale.py` | in-scale samples: random size and position (`sample_placement`); cuts, carves, hollow bodies and the manifold check work on voxel arrays | `sample_placement`, `sample_cuts`, `apply_cuts`, `carve`, `hollow`, `manifold_report`, `thickness` |
| `sample_writer.py` | the one writer of a registered sample, shared by all generators: voxels (ray parity by default since T-19; `placement.voxelizer` = the one that ran), `placement.degenerate` (< 200 voxels), optional noise with `<stem>.clean.bin`, inside bits, field, record. Sets `SHAPE_RECO_FIELD_BACKEND=igl` for its whole process when libigl is installed and the variable is unset | `finish_sample`, `voxelize`, `write_field`, `append_record` |
| `solid_builder.py` | exact solids for `--with-field`, `--segment-prob`, `--emit-negative-residual` | `base_program`, `placed_program`, `build_stl` |
| `ibeam_generator.py`, `ibeam_freecad.py` | I-beam samples: DIN segments, their three plates, envelopes | `generate`, `make_ibeam_solid` |
| `special_generator.py`, `special_freecad.py` | special subclasses: hollow, tube, rhs, frame | `generate`, `make_sample` |
| `din_profiles.py` | hot-rolled I sections (DIN 1025). Its docstring states that the numbers were written from memory and are self-checked by area | `IProfile`, `select`, `nearest`, `three_boxes`, `self_check` |
| `sdf_primitives.py` | analytic signed distance fields of the Part solids with segment parameters; placement and booleans; negative inside | `Box`, `Sphere`, `Cylinder`, `Cone`, `Torus`, `Ellipsoid`, `HalfSpace`, `ConvexPolyhedron`, `Wedge`, `Prism`, `Placed`, `Union`, `Intersection`, `Subtraction`, `sdf_grid`, `occupancy_grid` |
| `closest_point_field.py` | the closest-surface-point field of an STL or STEP model; exact genus; backends numpy, torch, cuda, igl (`SHAPE_RECO_FIELD_BACKEND`, default `auto`: cuda, else torch, else numpy); `SHAPE_RECO_MESH_ORDER=2` gives a STEP file's second-order field | `SurfaceField` (`query`, `genus`, `coarse_genus`, `genus_by_resolution`, `grid_vectors`), `load_field`, `load_mesh`, `tessellate_step`, `field_backend`, `mesh_order`, `kdtree_workers`, `voxel_surface_mesh` |
| `freecad_path.py` | which `freecadcmd` runs, for the whole project: `$SHAPE_RECO_FREECADCMD`, then the PATH, then `/usr/bin/freecadcmd`; standard library only | `find_freecadcmd`, `freecadcmd_or_default` |
| `quadratic_mesh.py` | second-order surface meshes of a STEP model (six-node triangles from gmsh, nodes on the exact surface); closest point by an igl start and Gauss-Newton on the patch | `mesh_step`, `mesh_step_second_order`, `QuadraticSurface`, `QuadraticField`, `load_quadratic_field` |
| `field_channels.py` | the one field encoding of a solid on a voxel grid, used by the generator and by the loop; also the one ray-parity implementation (`parity_axes`) | `CHANNELS`, `GridSpec`, `grid_spec`, `inscale_spec`, `mesh_channels`, `refined_field`, `voxel_channels`, `parity_occupancy`, `parity_axes`, `parity_of_file`, `field_of_file`, `surface_points`, `bin_to_world`, `world_to_bin`, `network_input`, `dirlog_input`, `dirlog_residual`, `closest_point_spectrum`, `rotate_network_input`, `save_field_npz`, `load_field_npz` |
| `inside_bits.py` | the base-point bits `<stem>.inside.npy`: parity of the placed solid per voxel centre, packed, in `.bin` order [z, −y, x]; the 6th channel of `dirlog_inside`. `read_inside` ignores the stored size (B-051) | `compute_inside`, `write_inside`, `read_inside` |
| `local_grids.py` | thickness-adaptive parity grids and their topology without a dense array | `wall_thickness`, `adaptive_pitch`, `BoxGrid`, `box_grid`, `iter_slabs`, `slab_topology`, `grid_topology` |

- **Sign conventions.** `sdf_primitives`: negative inside. `field_channels` SDF channel: positive inside (mm).
  `SurfaceField.query` `signed_distance`: negative outside, positive inside.
- **Parity (D-vox).** Inside/outside is decided by scanline ray parity at the voxel centres. `parity_of_file`
  computes it once per (file, size, mtime, grid) and process. `field_channels.parity_axes` is the one implementation;
  its callers are the generator (`_voxelize_parity`, `inside_bits.compute_inside`, the field's sign), the loop
  (`residual_loop.parity_occupancy`), `reco_field.peel.real_voxel_occupancy`, `local_grids.iter_slabs` and
  `run_tensors.parity_grid`. Each caller shifts the rows by the same 1.2345e-4 / 2.3456e-4 of a pitch
  (`local_grids.ROW_SHIFT` names the pair, the others repeat the literals). Nothing checks that the mesh is closed.
- **Voxels: state of 2026-10-01 (review R-03, `.claude/guru/project/R-03-voxels.md`).** D-vox says new data are parity
  grids; in the code parity is opt-in. `SHAPE_RECO_VOXELIZER` defaults to `stltovoxel` in `voxelization.py`,
  `sample_writer.py` and `dataset_generator.py`; `server/run_plan.sh` and `server/workstation/xgen.sh` set `parity`,
  `scripts/mpi_dataset_gen.py --voxelizer` passes it. `placement.voxelizer` records the variable, not the path taken
  (the parity path exists only for `fixed` + `origin`). An stltovoxel `.bin` sits up to half a voxel off the exact
  grid: IoU 0.84 median against the exact solid, 0 on a 26-voxel torus (`docs/training_plan_v5.tex`, F7). Existing
  sets: `scripts/revoxelize_parity.py` (keeps the old grid as `<stem>.stltovoxel.bin`). `<stem>.inside.npy` is parity
  whatever the voxelizer. Without `--with-field` the generator's cuts, carves and hollows edit the voxel array, so
  that `.bin` is the parity of no mesh and has no inside bits. T-19 makes parity the default and puts the voxelizer
  and a code version into the cache keys; until then `Cache.voxel_key` (`reco_engine/cache.py`) keys the STL bytes,
  resolution, scale mode, `scale_mm` and anchor, not the voxelizer (it has no caller outside its module and tests).
- **`local_grids` is used** by the loop's grid "tau" (`residual_loop._tau_spec`, `cache.wall_thickness_cached`), the
  feature stage, `difference_grade`, `exact_check`, the GUI and `scripts/voxel_iou.py`. `adaptive_pitch` returns
  `fraction` × the q-th percentile of the wall thickness (defaults 1/3 and 5). Measured on the frame: Betti numbers
  (1, 17, 0), the exact values, at every pitch τ/1 … τ/4 with τ = 12.0 mm (0.98–1.03 of the volume); wrong at
  64³–256³ (`docs/data/local_grid_frame.json`). The loop's grid "tau" calls `adaptive_pitch` with
  `ResidualConfig.tau_fraction` = 1.0, i.e. pitch τ₅ itself, not D-005's τ/3; `_grid_blindness` reports pitch/τ₅ and
  calls a grid blind only above 1. Whether 1.0 is intended is an open decision (R-03, 5.1).
- **Surface fields: state of 2026-10-01 (review R-04, `.claude/guru/project/R-04-surface-fields.md`).** Stored
  sign and occupancy are parity; the field supplies |d| and the direction only. Backend of new data: libigl's exact
  query on the raw mesh whenever libigl is installed (`sample_writer.write_field`, user decision 2026-09-24; recorded
  as `placement.field.backend`); without it, `mesh_channels` refines the mesh to edges ≤ 4 voxels and runs the
  one-pass query with k = 64. The GUI and `time_cascade.py` pass igl; `ResidualConfig`'s backend `None` keeps the
  environment's. `.field.npz` channels are float16: error ≤ 7.8e-3 mm at 14–28 mm, 6e-4 mm below 1 mm (R-04 probe);
  the effect on `dirlog_curv`'s curvature is not measured. Second order (`SHAPE_RECO_MESH_ORDER=2`, STEP only):
  `mesh_channels` queries the quadratic patches only under igl; on any other backend `refined_field` silently uses the
  flat four-way subdivision. Measured: sphere r = 10 mm, 320 six-node triangles, distance error 5.2e-4 mm mean
  against 6.1e-3 for 8000 flat ones (`docs/data/second_order_mesh.json`); liner, 1500 points within 2 mm against the
  B-rep, 3.2e-4 against 4.2e-3 mm mean, the same maximum (0.115 against 0.118 mm), gmsh 73 s against 3.9 s
  (`docs/data/second_order_liner.json`). Not measured: the frame, the effect on the loop.
- **Field caches.** In-process: `field_channels._FILE_CACHE` (path, size, mtime; not the mesh order or STEP
  deflection). On disk under `$SHAPE_RECO_CACHE` or `~/.cache/shape_reco`: `field_kinds/` (`field_input`:
  `.field.npz` path and mtime, kind, manual `EXACT_KIND_CACHE_VERSION`; the encoding constants are not in the key) and
  `voxfield/` (`training/inputs.py`: path, size and mtime of the `.bin`, inside bits and `.field.npz`, saturation, log
  scale, manual `CACHE_VERSION`). `~/.cache/shape_reco/wall_thickness/` (`$SHAPE_RECO_WALL_CACHE`;
  `reco_engine/cache.py`: mesh bytes, n, seed and the source of `wall_thickness`). `sdf_input` (path and truncation) and `exact_field_input` (STL path, resolution, grid mm) have no
  mtime in their keys. T-19 adds the voxelizer and a code version.
- **Genus.** `SurfaceField.genus()` is exact (Euler characteristic, correct for several closed shells).
  `coarse_genus` is the genus of a voxelization at a coarse resolution. The frame has exact genus 17 and reads as
  genus 1 at a coarse scale; the intermediate plateau of 9 is a grid-sampling effect (`docs/FRAME_FAILURE.md`).
- **Macros in this checkout:** box, cone, cylinder (`zylinder_gdml_...`), sphere, torus, and
  `assemble_reconstruction.FCMacro`. `MACRO_MAP` names 17 topologies; the other 12 macros are on tag
  `archive/master-2026-09-18`. Audit results: section 14.6.

## 9. `training/`

| module | purpose | main names |
|---|---|---|
| `topology_schema.py` | imports `TOPOLOGIES` / `PARAMETER_SPECS` from `dataset_gen`; [-1, 1] normalisation; per-topology parameter mask; categorical parameters snap to a valid choice | `normalize_params`, `denormalize_params`, `param_mask` |
| `subclass_schema.py` | label space of the hierarchical Type CNN: 5 parents, 24 subclasses | `labels_of`, `parent_mask`, `subclass_name` |
| `manifest.py` | reads `dataset_metadata.jsonl`; splits train/val by group | `Record`, `load_manifest`, `group_split` |
| `voxel_dataset.py` | Dataset over records and their `.bin` volumes; `input_kind` | `VoxelDataset`, `load_voxel` |
| `backbone.py` | shared 3D-conv backbone ending in `AdaptiveAvgPool3d`; Kaiming init, default BatchNorm init | `Voxel3DBackbone`, `ConvBlock` |
| `train_common.py` | shared loop: AdamW, cosine schedule, gradient clipping, NaN/Inf guard, checkpoints, speed options | `run_training_loop`, `load_split`, `check_finite`, `Speed`, `add_speed_arguments`, `configure_speed`, `prepare_input`, `autocast` |
| `train_type_cnn.py` | Type CNN, 17-way or hierarchical (`--hierarchical`); `--input`, `--topologies`, `--max-per-class` | `TypeCNN`, `HierTypeCNN`, `predict_records` |
| `train_rotation_cnn.py` | quaternion regression, topology-conditioned, geodesic loss 1 − \|q̂·q\| | `RotationCNN`, `geodesic_quat_loss` |
| `train_dimension_cnn.py` | masked parameter regression, topology-conditioned | `DimensionCNN`, `masked_smooth_l1` |
| `rotation_symmetry.py`, `train_rotation_sym.py` | box orientation with its symmetry group: canonical frame, symmetric loss, 6D output, alignment | `record_dims`, `admissible_group`, `symmetric_error`, `align_volume`, `SymRotationCNN` |
| `box_alignment.py` | frame fit to the surface normals instead of PCA | `fit_frame`, `frame_orientation` |
| `train_box_dims.py` | box half extents, longest first, in voxels | `BoxDimsCNN` |
| `train_axis.py` | axis of cylinder, cone, torus in the PCA frame (`--align` required) | `AxisCNN`, `label_axis`, `pca_axis` |
| `field_input.py` | grid-derived input kinds (`occupancy`, `sdf`, `field`, `fused`, `voxel_field`, `dirlog`, `dirlog_inside`: `encode`, the one function training and `predict` call); kinds read from `.field.npz` (`vf_exact`, `dirlog_exact`, `dirlog_curv`, `voxfield`: `load_exact_kind`, refused by `encode`); `dirlog_residual` is built by the loop | `encode`, `dirlog_channels`, `dirlog_inside_channels`, `log_distance`, `load_exact_kind`, `exact_kind_cache_path` |
| `sdf_input.py` | SDF from the occupancy grid; float16 cache keyed by path and truncation only | `sdf_from_occupancy`, `load_sdf_input` |
| `exact_field_input.py` | legacy: the mesh's own exact field as input; needs the STL; not loadable into `reco_engine`; no production caller; its grid is centred on the mesh's bounding box, not on the in-scale grid of a placed sample | `field_resolution_for`, `compute_exact_field`, `precompute` |
| `inputs.py` | the "voxfield" input: noisy grid, unit normal and log distance of the exact field, parity bit; 6 channels; sparse band cache in `~/.cache/shape_reco/voxfield/`; voxel-only fallback without a `.field.npz`. Shared encoder, but no Type-net inference path (`encode` refuses the kind); used at inference only by `feature_groups.piece_tensor` (fallback) | `voxfield_channels`, `voxfield_from_world`, `load_voxfield`, `encode_sample`, `voxfield_cache_path` |
| `gpu_encode.py` | occupancy, dirlog, dirlog_inside for a whole batch on the GPU | `encode_batch`, `dirlog_batch`, `rotate_batch`, `load_raw` |
| `train_fused.py` | `FusedNet`: one backbone, type, box-pose and size heads | `FusedNet`, `losses` |
| `field_points.py`, `field_point_backbone.py`, `field_point_nets.py`, `train_field_type.py`, `train_field_rotation.py`, `train_field_dimension.py`, `train_field_position.py` | point-cloud networks of the vector-field route | `PointCloudBackbone`, `FieldTypeNet`, `FieldBoxPoseNet`, `FieldAxisNet`, `FieldDimensionNet`, `FieldPositionNet` |

- The registry's CNN checkpoints marked `legacy` were trained on stltovoxel grids (B-066; D-007 asks for retraining
  on parity data). The registry's type rule skips them: "fused: descriptors + type_hier_dirlog_inside_medium"
  (`models/registry.json`, `type_stage`).
- Inventory of checkpoints: `docs/CNN_INVENTORY.md`. Training status and results: section 14.5 and `RUNNING.md`.
- Field analysis and the `dirlog` encodings: `docs/DIRLOG.md`.
- **Input contract: state of 2026-10-01 (review R-05, `.claude/guru/project/R-05-input-contract.md`).**
  - *Which kinds reach a Type net at inference.* `predict._type_tensor` encodes the cascade's grid with
    `field_input.encode`, so only grid-derived kinds work. `model_registry._type_cnn_input_kind` accepts every kind of
    `field_input.CHANNELS` except `exact_field`; the `.field.npz` kinds then raise at the first prediction (probe
    2026-10-01). The cascade skips `vf_exact`, `fused`, `points` and `exact_field` entries (`VOXEL_UNREADABLE`) unless
    the caller hands in a model per entry.
    `stage_registry.INPUT_KINDS` has no `voxfield`. `registry_type_dir` (the `$SHAPE_RECO_REGISTRY` override) admits
    occupancy, sdf and fused only; the cascade's own loader `_load_type_net` has no such check.
  - *The registry's type network* `type_hier_dirlog_inside_medium` reads `dirlog_inside`: the five dirlog channels of
    the (noisy) grid plus the base-point bit. Training reads the bit as `<stem>.inside.npy`, else `<stem>.clean.bin`,
    else the grid itself (implemented three times: `voxel_dataset.VoxelDataset._load`, `gpu_encode.load_raw`,
    `inputs._sample_files`; a noisy sample without either file silently gets the noisy grid). Inference:
    `cascade.inscale_target` resamples the loop's parity grid by nearest neighbour onto 64³ over 40 mm (type stage:
    mode "always", the target scaled to 0.7 of the grid) and passes the same grid as the bit. So the encoder is
    shared, the grid is not: where the loop's pitch is not 40/64 mm the input is a resample, not parity of the mesh on
    the training grid. R-05c (2026-10-01, v5 medium test, n = 672) measured that resample against the clean 64³ grid:
    +0.006 (descriptors) and +0.012 (CNN) parent accuracy, both inside the 95 % interval. `default_type_entry` is still the legacy `type_v3_B`.
  - *What a checkpoint records.* `type_cnn/config.json`: `input`, `sdf_truncation`, `field_resolution`, roots,
    counts and `resolution` (every run since 2026-10-02; before that only `--hierarchical`); no grid mm, voxelizer,
    channel order or code version.
    `model_card.json` (only with `--test-roots`) adds `resolution` and the git commit. `server/run_plan.sh` pins an
    `input_fingerprint` (sha256 of `field_input.py`, `field_channels.py`, `voxel_dataset.py`, `gpu_encode.py`; it
    omits `inputs.py` and `inside_bits.py`) and checks it before each training stage; nothing reads it at load or
    inference time. Any edit to those four files, docstrings included, changes the fingerprint
    (`ALLOW_INPUT_CHANGE=1` re-pins). Since 2026-10-02 the resolution is checked: `Models.type_resolution`
    (`model_registry._type_cnn_resolution`: config, else model card, else None = legacy, unchecked) and
    `predict.check_type_grid` refuse another grid; `cascade.check_descriptor_grid` refuses one of another resolution
    than the descriptor fit's (in the cascade: `type_descriptors.py`'s bytes are the fitted model's fingerprint); the cascade builds each source's target at that source's resolution. Before, a 32³ grid ran
    silently and cost the CNN 0.23 parent accuracy (R-05c).
  - *Voxelizer of the training data.* The registry entry says "parity medium v5"; no trainer reads or asserts
    `placement.voxelizer`.

## 10. `scripts/` and `server/`

| group | scripts |
|---|---|
| main line: run, time, judge | `time_cascade.py`, `eval_add_cut.py`, `g3_run.py`, `g3_report.py`, `s6_frame_eval.py`, `s6_g3_compare.py`, `feature_stage.py`, `grade_difference.py`, `start_matrix_cell.py`, `start_matrix_report.py`, `review_frames.py`, `review_frames_sheet.py`, `compare_runs.py` |
| exact measures (`freecadcmd`) | `freecad_iou.py`, `freecad_residual.py`, `freecad_brep_faces.py`, `freecad_crop_topology.py`, `inspect_step_faces.py`, `step_feature_census.py`, `voxel_iou.py`, `local_grid_frame.py`, `local_grid_residual.py`, `torus_guard_census.py` (the stored programs and real torus faces per torus-guard rule) |
| route A | `peel_case.py`, `eval_peel_examples.py`, `debug_freecad_solid_loop.py`, `genus_sweep.py`, `eval_field_nets.py` |
| test parts (`freecadcmd`) | `make_test_parts.py`, `make_beam_test_parts.py`, `make_frame_review_set.py`, `make_chamfer_pockets.py` |
| datasets | `make_dataset_v5.sh`, `mpi_dataset_gen.py`, `run_small_training.sh`, `add_inside_bits.py`, `add_voxel_noise.py`, `revoxelize_parity.py`, `verify_inscale_dataset.py`, `timing_report.py` |
| network evaluation | `evaluate_on_val.py`, `eval_type_cnn_v3.py`, `eval_type_fused.py`, `eval_fused_net.py`, `eval_rotation.py`, `eval_alignment.py`, `eval_stage_ensemble.py`, `eval_minkowski_baseline.py`, `eval_descriptor_baseline_v5.py`, `report_type_cnn.py`, `check_size_observability.py`, `check_observability_v5.py`, `bench_field_backend.py`, `bench_training_speed.py` |
| field and dirlog studies | `field_atlas.py`, `field_slices.py`, `field_encoding_stats.py`, `energetic_descriptors.py`, `sdf_reference_freecad.py`, `measure_second_order.py`, `chamfer_study.py` |
| reports and LaTeX generators | `frame_report.py`, `frame_report_fillets.py`, `frame_report_nephogram.py`, `frame_report_tex.py`, `make_*_tex.py`, `make_*_figures.py`, `make_report_artifacts.py`, `make_training_data_slides.py`, `codemap_graph.py` |
| machines and jobs | `workstation_run.sh`, `workstation_exec.sh`, `workstation_home.sh`, `workstation_queue_runner.sh`, `workstation_queue.txt`, `run_slot.sh`, `resource_budget.py`, `suite_by_file.py` |
| FreeCAD nodes | `check_nodes_install.py`, `make_node_graph_figure.py` |

`server/`: `run_plan.sh` (the resumable job list of training plan v5), `environment.yml`, `requirements.txt`,
`setup_venv.sh` (venv on the institute servers, `docs/SERVER_TRAINING.md`), `wheels_for_workstation.sh`, `workstation/` (chain,
watchdog and environment scripts that run on workstation).

## 11. `freecad_nodes/shape_reco/`

Nodes for FreeCAD's **Nodes** workbench (github.com/j8sr0230/Nodes). 7 node modules (the voxel nodes of route C -
load models, voxelize, the three CNNs, placement, local search, the consensus nodes and the reconstruct/ensemble
loops - were removed on 2026-09-30):

| group | node modules (`shape_reco_<name>.py`) |
|---|---|
| input and output | `load_mesh`, `load_shape`, `export_stl` |
| route A | `field_peel`, `field_export`, `field_grade` |
| route B | `cascade` |

- Slow nodes have their own Run button (`_bootstrap.RunButtonContent`), because the workbench evaluates every node
  on file open and on undo/redo.
- `install.sh` copies the nodes into FreeCAD's addon directory and writes `_sandbox_root.txt` beside them.
  `_bootstrap.py` finds the repository by `SHAPE_RECO_SANDBOX_ROOT`, then that file, then its own location.
- Example graphs: `freecad_nodes/examples/example_cascade.json`, `example_vector_field.json`.

## 12. Tests

`tests/conftest.py` runs the tests CPU-only unless `SHAPE_RECO_TEST_GPU=1` and marks tests measured at ≥ 20 s as
`slow`. The option `--run-slow` is not defined in this repository: it comes from the installed plugin
`pytest-astropy` (`/usr/lib/python3/dist-packages/pytest_astropy/plugin.py` on the laptop), which skips tests marked
`slow` unless `--run-slow` is given. Without that plugin, slow tests run unless deselected with `-m "not slow"`.
Tests use a private wall-thickness cache. The test files per module are in the tables above.
Commands: `RUNNING.md`, "Tests".

## 13. Design rules

- **One label-space source.** `dataset_gen/src/dataset_utils.py` defines `TOPOLOGIES` and `PARAMETER_SPECS`;
  `training/topology_schema.py` imports them.
- **One primitive-building implementation.** Routes A and B: `reco_field/export.py` → `freecad_builder.py`
  (`reco_engine/geometry.py` still builds single primitives with the generator's macros).
- **One field encoding.** `dataset_gen/src/field_channels.py` serves training and inference.
- **One result schema** for cascade, loop, GUI, nodes and JSON: `reco_engine/result_schema.py`.
- **Voxels are the ground truth for where material is, at a pitch licensed by the local thickness** (D-vox, D-005):
  ray parity at the voxel centres, pitch ≤ τ/3. In the code (2026-10-01) parity is opt-in for generated data
  (`SHAPE_RECO_VOXELIZER`, T-19) and the loop's grid "tau" uses τ₅ itself (`tau_fraction` 1.0); see §8.
- **Every operation is a real solid.** Programs are built in `freecadcmd`; validity means exactly one solid.
- **Group-aware train/val split.** A group shares one value; splitting by sample would leak it.
- **Quaternion geodesic loss**, not Euler angles. For symmetric shapes the loss is the minimum over the symmetry
  group.
- **Fail fast on NaN/Inf loss.**
- **Voxel convention of route C.** `.bin` files are `{0, 1}` uint8, indexed `[z, y, x]`, and the y axis runs
  backwards. CNN input tensors are `{-1, +1}` float32; the conversion happens once at the model input. Signed
  quantities go to world coordinates through `placement.array_to_world_position` (order reversed and y negated);
  unsigned extents through `array_to_world`. Verify any new conversion on an asymmetric, off-centre shape
  (`tests/test_real_voxel_occupancy.py`, `tests/test_array_axes.py`).

## 14. Measured findings of the legacy voxel route and the training

All numbers in this section were measured before 2026-09-24 with the smoke-trained `checkpoints/`, unless a date
is given. They are kept because the thesis cites them. Route C and most scripts named here were removed on 2026-09-30; they are in git history
(§5.3), and these records say what they measured. One run per part; the runs are deterministic.

### 14.1 Octree candidate proposal

`scripts/compare_strategies.py`; fit thresholds 0.05; eleven parts of `dataset_gen/examples/`. Final IoU of the
assembled result against the input:

| part | default | fixed depth 1 | fixed depth 2 | adaptive (good_fit 0.5) |
|---|---|---|---|---|
| `box_with_bore` | 0.846 | 0.882 (3 cuts) | 0.834 | **0.891** (3 cuts) |
| `two_bores` | 0.803 | 0.803 | 0.809 (3 cuts) | 0.803 |
| `corner_pocket` | 0.789 | 0.869 (3 cuts) | 0.800 (3 cuts) | 0.869 (3 cuts) |
| `blind_hole` | 0.801 | 0.840 (3 cuts) | 0.804 (1 cut) | 0.840 (3 cuts) |
| `side_slot` | 0.790 | 0.808 (3 cuts) | 0.799 (3 cuts) | 0.808 (3 cuts) |
| `l_bracket` | 0.773 | 0.803 (3 cuts) | 0.785 (3 cuts) | **0.813** (3 cuts) |
| `tube` | 0.526 | 0.570 (3 cuts) | 0.526 | 0.570 (3 cuts) |
| `cross_hole` | 0.653 | 0.645 (3 cuts) | 0.661 (3 cuts) | **0.671** (3 cuts) |
| `three_holes` | 0.771 | 0.783 (2 cuts) | 0.779 (3 cuts) | 0.783 (2 cuts) |
| `cavity` | 0.846 (2 cuts) | 0.848 (1 cut) | 0.848 (3 cuts) | 0.848 (1 cut) |
| `plate_4holes` | 0.849 | 0.849 | 0.856 (3 cuts) | 0.849 |

| strategy | mean IoU change, 11 parts (wins / ties / losses at ±0.005) | on the 7 held-out parts | mean wall time, cold |
|---|---|---|---|
| default | — | — | 4 s |
| fixed depth 1 | +0.023 (7/3/1) | +0.012 (3/3/1) | 34–36 s |
| fixed depth 2 | +0.005 (7/3/1) | +0.007 (5/2/0) | 34–36 s |
| adaptive | +0.027 (8/3/0) | +0.017 (4/3/0) | 22 s |

- `good_fit` was tuned on four parts (`box_with_bore`, `corner_pocket`, `tube`, `three_holes`); the gain there was
  +0.045. The held-out +0.017 is the size of the effect.
- On `box_with_bore` the bore is recovered as a chain of three cubes. The default strategy's best cut candidate
  fails the precision gate (0.775 < 0.8).
- The first version ranked cells by their own recall and had no whole-grid candidate; on `box_with_bore` it gave
  IoU 0.442.

### 14.2 Cache

Per candidate: freecadcmd build ≈ 68 %, stltovoxel ≈ 27 %, CNN inference ≈ 10 ms on the GPU. The 44-run sweep
(11 parts × 4 strategies): 10 min 30 s cold, 2 min 44 s warm, 1 min 00 s warm after the faster search scorer. The
cached results reproduce the uncached ones bit for bit on all 44. Keys: for a built primitive the topology,
parameters, quaternion, scale, resolution and the macro's content; for a voxelization the STL's bytes, the grid, the
voxelizer that runs and its code fingerprint (`dataset_gen/src/cache_keys.py`, since 2026-10-01; older entries are never
read again); for a prediction the input grid and the checkpoint files' size and mtime.

### 14.3 The Type CNN on trained shapes and on slices

- **Bodies of trained shapes only** (eight `bs_*` parts, `--topologies box sphere`): a plain axis-aligned box is
  called a sphere (0.48); cover-alone IoU 0.55–0.96; cut candidates are weak (leaf fits 0.2–0.4). Training saw only
  randomly rotated boxes; that this is the cause is not verified.
- **Slices** (`reco_engine/shape_probe.py`, `scripts/probe_type_cnn_slices.py`; 465 slices, in scale):

  | finding | numbers |
  |---|---|
  | only box and sphere are known | P(true class) of the whole shape: sphere 0.91, box 0.02, cylinder 0.01, cone 0.01, torus 0.00 |
  | sphere recognised from 0.8 of its diameter, all six directions alike | 0.64–0.78 at 0.8; 0.88–0.91 from 0.9; direction inconsistency 0.04 |
  | in the stretched `fit` mode the same sweep depended on the cut axis | hemisphere P(sphere) 0.70 / 0.59 / 0.41 / 0.20; inconsistency ≈ 0.3 |
  | box recognised at slice thickness up to 0.5 | P 0.84–0.97; the whole cube is "sphere" (P(box) 0.02) |
  | the answer follows the fill of the grid | below 40 % fill all 348 slices are "box"; a single threshold (fill ≥ 0.635) reproduces the answer on 94 % of the slices |

  Every slice is out of the training distribution (whole shapes only), so this measures extrapolation.
- **In-scale Type CNN** (`checkpoints_inscale/type_cnn`; 1,600 samples, 800 box and 800 sphere, 63 % with at least
  one cut): best validation accuracy 0.975 (epoch 28). On the same slices in the fixed frame: whole sphere 0.94,
  whole cube P(box) 0.85; sphere slices "sphere" on 92 % of caps and 93 % of slabs; box slices "box" on 92 % and
  87 %. The old network calls everything a box there (P(sphere) 0.00 on the whole sphere). A fill threshold
  separates the in-scale classes on 70 % of the samples.
- **Hollow bodies** (`inscale.hollow`): adding hollow data raised the Type CNN's accuracy on held-out hollow samples
  from 91.2 % to 98.8 % (box 89.5 → 98.0, sphere 93.0 → 99.5), final checkpoint after 40 epochs. Validation accuracy
  dropped to 40–48 % repeatedly during training. `docs/data/hollow_box.md`.

### 14.4 Voids and the y axis

- **Void quality.** The default cut phase keeps mostly `hits_material` voids (purity 0.73–0.97; e.g. 191 material
  voxels destroyed by a void removing 6357 excess voxels). The strict rule rejects nearly all of them;
  `bs_box_sphere_pocket` 0.891 → 0.857. The bottleneck is the proposals, not the gate.
- **The voxel array's y axis runs backwards.** `stltovoxel` writes image rows top to bottom. Before the fix every
  signed voxel-space quantity was mapped to world with the wrong sign of y. `corner_pocket`: the octree's cuts
  raised the voxel IoU 0.789 → 0.869 and lowered the real IoU 0.830 → 0.724. After the fix
  (`placement.array_to_world_position`): real IoU 0.840 → 0.917; footprints agree at 0.93–0.97 (cover boxes) and
  0.82–0.86 (cuts). Every voxel IoU in this document stands; every real-geometry number for an asymmetric part from
  before the fix does not. Not done over all parts before the route was removed.
- **Placement.** A synthetic test showed IoU 0.003 → 0.818 for an identical shape once placed by `placement.py`.
- **Mesh booleans are not used** for the assembly: fusing two real candidate tori by mesh left the result's volume
  equal to one input alone. `assemble_reconstruction.FCMacro` builds Part solids from parameters.

### 14.5 Training status (box and sphere, with voxel noise)

| network | result | baseline | verdict |
|---|---|---|---|
| Type CNN | 100 % held-out accuracy (40/40); validation loss 0.5737 | label-smoothing floor 0.5729 | solved for this pair |
| Rotation CNN, 16 groups | 116.3° mean error | 126.5° random | not learned |
| Rotation CNN, 400 groups × 2 | 122.2° (box 124.1°, sphere 120.5°) | 126.5° random | not learned; group count refuted as the cause |
| Rotation, box, modulo the symmetry group | 77.5° | 75.2° random, scored alike | no orientation signal |
| Dimension CNN, 16 groups | MAE 0.496 (box), 0.300 (sphere) | 0.549, 0.345 (training mean) | not learned |
| Dimension CNN, 400 groups × 2 | +6.9 % (box), +0.3 % (sphere) against the constant predictor | — | not learned |

- **Dimension: the input does not contain the target.** 800 spheres with a 16× radius range occupy
  118,949–123,100 voxels (3.4 % spread, correlation with r³ 0.03; `scripts/check_size_observability.py`). The `fit`
  voxelizer erases absolute size. The in-scale mode keeps it.
- **Rotation of the raw-quaternion network is unexplained.** The symmetry-aware networks and the network-free frame
  fit replaced it for boxes; their results are in `RUNNING.md`, "Symmetry-aware Rotation CNN" and "Face-normal frame
  alignment".
- **BatchNorm defect (fixed).** An earlier backbone zero-initialised the BatchNorm gamma without skip connections;
  every block returned zero. Type CNN: 7.5 % validation accuracy before, 67.5 % after the fix on the same data.
- **Fused input.** A controlled swap of `--input occupancy` for `fused` lowers type accuracy and box-orientation
  accuracy; `FusedNet` recovers part of the type accuracy and is worse on box orientation. Numbers: `RUNNING.md`,
  "Fused input".
- **Analytic dims prior:** second moments, h = sqrt(3 λ), near exact on intact boxes. The registry's
  `box_dims_measured` (whole-voxel extents) is biased by up to a voxel per half extent
  (`scripts/check_observability_v5.py`).

### 14.6 Macros and Part solids

Macro audit (made on the pre-vector-field line; of these files only the five in use are in this checkout):

| macro | finding |
|---|---|
| box, cylinder, cone, sphere, torus | correct, volume-verified, in use |
| sphere | `Part.makeSphere(..., 0.0, 360.0, 360.0)` built a degenerate sphere (volume ≈ 5.6e-15); fixed to `(-90.0, 90.0, 360.0)`. Sphere samples from before the fix are degenerate |
| polycone | unfinished stub: a syntax error, no `DATASET_*` interface |
| torus_segment | ignores the `DATASET_*` interface; always writes its own 343-combination sweep |
| cone_segment | `Part.makeCone` fails for equal radii with a partial sweep; the error message names the outer cone also when the inner one failed |
| the other 9 | ran once with mid-range parameters and gave a plausible STL; not a correctness audit. `paraboloid` gave ~45,000 triangles |

FreeCAD-native solids (FreeCAD 1.0, volumes checked against closed forms):

| primitive | parameters | volume check | in the project |
|---|---|---|---|
| `Part::Box` | Length, Width, Height | exact | `box` |
| `Part::Cylinder` | Radius, Height, Angle | exact, also with a partial `Angle` | `cylinder` |
| `Part::Sphere` | Radius, Angle1–3 | exact | `sphere` |
| `Part::Cone` | Radius1, Radius2, Height, Angle | exact | `cone` |
| `Part::Torus` | Radius1, Radius2, Angle1–3 | exact | `torus` |
| `Part::Ellipsoid` | Radius1–3 | within 0.04 % | no macro |
| `Part::Prism` | Polygon, Circumradius, Height | exact for n = 3, 4, 6, 12 | no macro |
| `Part::Wedge` | 10 extents | exact for box, taper, shear | no macro |

There is no `Part::Tube`; a tube is a cylinder minus a cylinder. An equal-radius "cone" with a partial sweep should
be built as `Part::Cylinder` with `Angle`. Symmetry of the rotation label: sphere all of SO(3); cylinder, cone,
torus continuous about the axis; box and ellipsoid with distinct sides 4 proper rotations; regular n-prism 2n.

### 14.7 Exact-field speed-ups (2026-09-24, CPU-measured)

`docs/data/field_igl_bench.json`, `tests/test_field_speedups.py`.

| measure | before | after |
|---|---|---|
| closest-point query per 64³ grid, torch CPU → `SHAPE_RECO_FIELD_BACKEND=igl` | 12–19 s | 0.03–0.35 s |
| cKDTree query, 16,384 centres, 13k faces, single thread → `SHAPE_RECO_KDTREE_WORKERS=-1` | 0.30 s | 0.05 s |
| 4 generated samples, `SHAPE_RECO_FREECAD_WORKER=1` | 85.6 s | 30.8 s |

- The one-pass query's k = 64 candidate set is proven exact for 60 % of the centres. Where it misses it is an
  upper bound (cylinder: 20 centres off by up to 0.044 mm; `box_with_bore`: 6798 off by up to 0.24 mm). igl gives
  the exact minimum. Use one backend for a whole dataset.
- Not applied: the EDT, no-host-sync and `--bn-momentum` items of
  `.claude/guru/training/2026-09-24_training_throughput.md`. `--cudnn-benchmark` is opt-in and not measured.
- `run_tensors` (2026-09-27, `docs/data/run_tensors.json`): liner at 2.4 mm (164 x 84 x 668), igl, band 9:
  175.5 MB dense against 117.9 MB sparse; build peak RSS 1513 MB against 481 MB; difference 0 inside the band. At
  9.6 mm the band is 99 % of the grid and nothing is saved.

## 15. Not in this checkout

These directories were described here before. They are not tracked in this repository.

| directory | where it is | what it was |
|---|---|---|
| `FeatureNet/` | untracked in `~/Documents/Sandbox`; tracked on tag `archive/master-2026-09-18` | fork of github.com/zibozzb/FeatureNet; `ShaPe_Integra_v2.py` / `v3.py` were the reference for `training/backbone.py`. v2's history showed NaN loss from epoch 1 |
| `FCMacros/` (top level) | untracked in `~/Documents/Sandbox`; tracked on the tag | the macro set; the five in use are in `dataset_gen/FCMacros/` |
| `SHaPe_Reco_Dashboard/` | on the tag only | early single-file explorations |
| `archiv/` | untracked in `~/Documents/Sandbox` | retired prototypes |
