#!/usr/bin/env python3
"""
The residual add-cut loop (plan, Step 5): reconstruct an STL/STEP as a CSG program of the cascade's primitives by subtracting
what the model already explains from the input, step by step, on voxel tensors of the SAME encoding the networks read
(dataset_gen/src/field_channels.py: occupancy, SDF positive inside, V; array axes = world x, y, z).

  1. to tensor   STL/STEP -> SurfaceField (load_field; STEP tessellated by freecadcmd) -> T_in = mesh_channels with the sign
                 and the occupancy from scanline ray parity (`parity_occupancy`, D-vox). `grid`: "cube" (the library
                 default; `resolution` cells per side, refined x1.5 up to `max_resolution` until the parity volume reaches
                 `min_volume_ratio` of the mesh volume) or "tau" (one pitch, the finer of `tau_fraction` x the 5th-percentile
                 wall thickness and the cube's pitch, its own cell count per axis; B-090. With tau_fraction 1.0 the pitch is
                 tau, coarser than the tau / 3 D-005 licenses). The GUI and scripts/run_part.py set "tau";
                 reconstruct_cli, scripts/time_cascade.py (by default) and feature_stage.ROUGH run the cube. A cube grid
                 coarser than the thinnest wall aliases it (`_grid_blindness`, reported in the "done" step); on such a
                 grid the type stage is skipped.
  2. start       one of three paths, in this order:
                 - `first_guess` (reco_engine/user_guess.py): the user's shapes, judged by the loop's own checks; they
                   replace the start and the sub-loops when at least one add is kept;
                 - `start_shape` != "legacy": the starting-shape stage (reco_engine/start_shape.py: surface inventory,
                   graded candidates, coordinate search). When no candidate passes `choose_start` the run ends with an
                   EMPTY program (a "done" step, no grade): there is no fallback to the legacy first add;
                 - "legacy" (the default): the type stage on the whole input gives a subclass hint; then a SUB-LOOP when
                   the hint or the mesh calls for one (reco_engine/ibeam.py): members - straight prismatic runs found in
                   the mesh's exact sections (I-beam -> 3 box adds, rod -> cylinder, bar -> box, tube -> cylinder add +
                   cut, rhs -> box add + cut), judged by the exact (mesh) volume they explain, since webs far below the
                   pitch gain nothing on the grid; or a hollow body - closed cavities (inner shells) -> the outer shell's
                   primitive, then a cut of the same kind per cavity. Otherwise the graded first add (`_first_add`): the
                   cascade's answer (`first_pass_iou`) and up to three of its alternatives compete with every bounding
                   cover (oriented / world box, cylinder and sector, sphere, box & cylinder) under one grade of how the
                   candidate's surface lies against the input's (`_grade_cover`, containment >= COVER_MIN_INSIDE) -
                   "add a bounding part first".
  3. difference  the model's own analytic SDF (dataset_gen/src/sdf_primitives.py, the tree result_to_program builds) on the
                 same grid -> R = occ_in - occ_model. A voxel counts only when BOTH surfaces are more than half a voxel away
                 (the input's sign from ray parity): the one-voxel staircase where a correct primitive meets the input is
                 not residual.
  4. leftovers   connected components of R > 0 (material still missing: adds) and R < 0 (material the model has and the
                 input lacks: cuts). Each is encoded with voxel_channels and passed to the same cascade, scored inside the
                 region where it matters (outside the model for an add, inside it for a cut). Further candidates: void_fit
                 for a cut (a void fitted to the input's mismatch, free to protrude into air), strict RANSAC cylinders and
                 planes (surface_choice), half-spaces, intersections with the part's covers and composites
                 (more_booleans), the largest filled boxes of a cut piece (excess_boxes), boolean tools, varied views on
                 failure. Box and cylinder faces that touch no input material are pushed outwards (a cut) or one voxel
                 into the model (an add), so FreeCAD never meets coplanar faces.
     accept      iff |R| shrinks by at least `min_gain` voxels, and for a cut iff void_quality.assess_void passes strictly
                 against the input's interior voxels AND the input's sign channel says no material deeper than half a voxel
                 is removed.
  5. order       `cuts_first` (D-003): True takes the first acceptable cut component before any add, False all adds
                 before the cuts; None (default) is cuts first after a starting shape and ADDS FIRST on the legacy path,
                 so the default run is adds first. No recorded measurement compares the two orders on the legacy path
                 (checked 2026-10-01). $SHAPE_RECO_OP_GUARD=1 (or `op_guard=True`) bypasses the order: every iteration
                 takes whichever component, add or cut, gains most.
  6. stop        |R| <= `tolerance` x |input|, the operation budget (`max_ops`, else reco_field.peel.predict_ops_budget,
                 12 when that fails; grown by the surface inventory on the starting-shape path) is used, or no component
                 is left that a primitive explains; with the feature phase also the rough rule (`rough_cut_grade`,
                 `rough_max_ops`).
  7. solution    (legacy path only, ResidualConfig.solution_alternatives) the classic solution - plain first add, cuts
                 first - computed on a fresh state and kept when its sampled exact IoU is higher by more than
                 `solution_margin` (reco_engine/solution_choice.py).
  8. fallback    components nothing explains are covered by voxel cubes merged into few boxes (loop_guards.largest_box_in_blocks),
                 fused for R > 0, cut for R < 0, sources "voxel_fallback". Not when the feature phase is on.
  9. features    (ResidualConfig.feature_phase, off by default; replaces the fallback of 8) the program built in FreeCAD,
                 its EXACT difference to the input by OCCT, every solid of it a piece; per piece a primitive guessed
                 (reco_engine.feature_stage: exact STEP faces, surface fitters, the cascade), applied when it scores, kept
                 only when a real FreeCAD build stays one valid solid and the exact residual drops; iterated under limits.
Build + grade in FreeCAD at the end (`_grade`: reco_field.export build_in_freecad, export_check two-way chamfer, valid =
exactly one solid), optionally per step; then the exact grade (`_exact_grade`: the OCCT residual against the input by
reco_field.exact_check.exact_residual, cross-checked with exact_check.sampled_iou - a disagreement over 0.05 reports the
sampled IoU instead, `exact_method` says which) and the statistics vector (reco_engine/solution.py) in the "done" step.

    for step in residual_steps("part.stl", ResidualConfig()):
        print(step.kind, step.info.get("message", ""))
    result = reconstruct("part.stl")                    # reco_engine.result_schema.ReconstructionResult
"""

from __future__ import annotations
import tempfile

import copy
import os
from contextlib import contextmanager
import sys
import time
from dataclasses import dataclass, field as dc_field, replace
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from reco_engine.timing import timed, timed_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import cascade  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource, UNKNOWN  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, program_model, quaternion_of, rotation_matrix  # noqa: E402

STEP_KINDS = ("input", "member", "ibeam_box", "member_op", "hollow", "add", "residual", "decision", "component", "candidate", "accepted",
              "rejected", "repair", "solution", "fallback", "prune", "feature_residual", "feature_groups", "feature_piece", "feature_feedback",
              "feature_split", "feature_confirm", "feature_confirmed",
              "feature_candidate",
              "feature_accepted", "advice_asked", "advice_given",
              "feature_rejected", "feature_done", "grade", "done")
FEATURE_KINDS = ("feature_residual", "feature_groups", "feature_piece", "feature_feedback", "feature_split",
                 "feature_candidate", "feature_accepted", "feature_rejected", "advice_asked", "advice_given", "feature_done",
                 "feature_confirm", "feature_confirmed")      # the feature phase's steps (info["phase"] == "feature"; the contract with reco_gui)
MEMBER_SUBCLASSES = {"ibeam": {"ibeam"}, "tube": {"tube"}, "rhs": {"rhs"}, "frame": None}   # TypeNet subclass -> member kinds


THIN_WALL = 2e-3                    # the super-thin threshold, a share of the input's diagonal (feature_stage.py)


@dataclass
class ResidualStep:
    kind: str                         # one of STEP_KINDS
    result: ReconstructionResult      # the program so far (a snapshot: later steps do not change it)
    info: Dict = dc_field(default_factory=dict)


@dataclass
class ResidualConfig:
    resolution: int = 48              # grid cells per side (a cube around the part)
    # A part whose walls are thinner than the voxel pitch (the frame: 12-23 mm webs and flanges on a 6.7 m part) falls
    # between the voxel centres and has (almost) no input voxels. The grid is then refined, up to max_resolution, until
    # the occupied volume is at least min_volume_ratio of the mesh's own volume (measured on the frame: 0 voxels at 48^3,
    # 8 at 64^3, 1376 = 0.83 of the volume at 96^3).
    max_resolution: int = 128
    min_volume_ratio: float = 0.5
    # grid "tau" (B-090, D-005; the user's rule 2026-09-24: always compute on the more detailed voxelisation): a box of
    # cells over the part at the FINER of two pitches - `tau_fraction` x the 5th-percentile wall thickness
    # (dataset_gen/src/local_grids.py) and the cube's pitch at `resolution` - coarsened by 1.25 steps while it would have
    # more than `max_cells` cells. The frame: tau = 0.12 STL units < the 96^3 cube's 0.81 -> 88 x 562 x 419 cells, 0.99 of
    # the volume, where the cube had 1376 voxels (0.83, by aliasing). A chunky part (box_with_bore, tau = 12.1) keeps the
    # cube's pitch (0.48) - tau alone would have given 8 x 8 x 6 cells. "cube": the refined cube above.
    grid: str = "cube"
    # surface_choice (docs/LINER.md): a cut also gets the strict-tolerance RANSAC cylinders of the input's surface as
    # candidates (cascade.strict_cylinders), and among the passing candidates whose gain is at least
    # surface_min_gain_share x the best gain, the one more of the nearby input surface lies on wins
    # (cascade.near_surface_agreement). False: the largest voxel gain wins, as before.
    surface_choice: bool = True
    surface_min_gain_share: float = 0.25
    # more_booleans: besides add (union) and cut (difference) the loop tries, per component, half-space cuts from strict
    # planes of the nearby surface (cascade.strict_planes), "intersect" with the part's bounding covers (model & P, judged
    # by the cut checks on what it removes) and composites of two primitives judged as one (cascade.composites: box &
    # cylinder, box - cylinder). False: add and cut of single primitives only, as before.
    more_booleans: bool = True
    # A new-boolean candidate (intersect, half-space, composite) must also explain the surface: at least this share of
    # the nearby input surface on it. Measured 2026-09-25: the accepted ones that helped had 94-100 % (cy_flat), the
    # bounding-cylinder intersects that broke bs_box_step and bs_box_dome had 0 % - passing only because nothing else did.
    boolean_min_surface: float = 0.2
    # vary_on_failure (the user's rule 2026-09-25: "when guessing the next shape fails, start varying"): a component none
    # of whose candidates passes is asked again on varied views - its voxels opened / closed by one voxel, resampled at
    # half and double the pitch, and turned by `vary_rotations` fixed random rotations (with the input surface) - every
    # answer mapped back and judged by the same checks on the loop's grid.
    # start_shape (the user's rule 2026-09-26): "legacy" = sub-loops first, then the graded first add (as before);
    # "auto" = the starting-shape stage FIRST (reco_engine/start_shape.py: holds the part, most surface on it, least
    # excess), the sub-loops never run at the start; or a named type forced by the user (start_shape.START_TYPES).
    start_shape: str = "legacy"
    # start_shape "cubes" (the user's option of 2026-10-10): the largest completely filled axis-aligned boxes of the input
    # occupancy, grown from the voxels one after the other (largest_boxes), are the first adds - inside the material by
    # construction, so the cover has no excess; what they do not reach is left to the loop as additional primitives.
    cubes_max: int = 8                 # boxes at most
    cubes_min_share: float = 0.05      # stop when the next box holds less than this share of the first one's voxels
    cubes_rest: float = 0.02           # stop when the uncovered share of the input is below this
    cubes_min_voxels: int = 27         # a box needs at least this many voxels
    cubes_snap: bool = True            # each box face moved onto the input surface where samples support it (_snap_box_faces)
    # start_shape "compartments" (the user, 2026-10-29: "look for mass-population compartments on the whole, then add-cuts
    # until we got good fits"): the input occupancy split into boxes that hold its mass with little empty space
    # (reco_engine/mass_compartments.py: least waste per cut, else the largest step of the mass profile); the filled ones
    # are the first adds, snapped to the surface - an over-cover the loop then cuts and adds to until its fits converge.
    compartments_fill: float = 0.85     # a compartment filled at least this much is not split further
    compartments_max: int = 12          # compartments at most
    compartments_min_voxels: int = 27   # bits smaller than this are dropped first; boxes smaller are not split
    compartments_min_fill: float = 0.25  # a compartment filled less than this is left to the loop
    # start_shape "leaves" (the user, 2026-10-30: "how could we get the method of recognizing subshapes - parts or whole
    # primitives - in higher ordered shapes?"): the primitives the STEP part's own surfaces name
    # (reco_engine/leaf_recognition.py: a hypothesis per surface over its own extent, verified against the input, the
    # fewest selected by a greedy CSG and minimized) are the first operations; the loop and the feature phase explain the
    # rest. STEP only (an STL or a recognition under leaves_min_iou: the legacy start takes over). Measured offline on
    # the gate's 36 STEP parts (docs/data/leaf_recognition_study_gate.md): leaf recall 0.28 against the loop's 0.17,
    # 2.5 parts against 10.7; union leaves 0.54 / 0.32 / 0.19 against 0.00 / 0.03 / 0.00.
    leaves_min_iou: float = 0.2
    # The objective in the shape space (the user's rule 2026-09-26, docs/thesis/chapters/10b_solution_spaces.tex):
    # score = (covered share of the generated volume) x (surface coverage); start_score "product" picks the highest
    # score among the candidates that hold the part, "band" the rule before (best coverage band, least excess).
    # start_vary: the chosen shape is then varied in its parameters (centre, sizes, frame; each operand of a composite)
    # at the grid pitch, a coordinate search that keeps every step raising the score, at most start_vary_rounds steps.
    start_score: str = "product"
    start_vary: bool = True
    start_vary_rounds: int = 4
    # cuts_first (D-003, "cuts first" stays the rule; the order is a flag so both can be measured): each round takes the
    # first acceptable CUT component (largest first) and an add only when no cut passes. False: all adds before the cuts
    # (the order before 2026-09-26). None: cuts first after a starting-shape start (a superset of the part: the excess is
    # cut away before anything is added - note 20 section 3), adds first on the legacy path.
    cuts_first: Optional[bool] = None
    # mode "tree" (2026-10-15, reco_engine/residual_tree.py): every round an explicit decision on the difference (D0
    # accept / D1 carve / D2 add / D3 features / D6 stop), and a component no primitive explains becomes a child - its
    # inside boxes as ONE composite operation, else the cover child (its bounding box, the material taken wrongly
    # re-added by a grandchild, depth-limited). "loop": the rounds as before. The tree is meant for the starting-shape
    # start (start_shape "auto": a cover, then cuts first); it runs after the legacy start too.
    mode: str = "loop"
    # mode "search" (reco_engine/shape_search.py, the user's rule 2026-10-16): per component every base shape, guesses
    # from the nearby exact surface samples, hill-climbed on the share of them on the shape (the coverage); the loop's
    # material checks judge each kind's best, the covered passing one wins. The start stays the legacy one.
    search_kinds: Optional[Tuple[str, ...]] = None                      # None: all of result_schema.KINDS
    search_evals: int = 120                                             # random neighbours per refined guess and kind
    search_tolerance_share: float = 0.004                               # x the part's extent: "on the surface"
    search_samples: int = 30000                                         # exact surface samples of the input
    search_min_coverage: float = 0.3                                    # a kind below it is not taken (the step part:
                                                                        # two torus cuts at 8 % and 16 %, 2026-10-16)
    tree_child_boxes: int = 8            # a child's cube cover: at most this many largest boxes
    tree_primitive_share: float = 0.5    # a primitive that explains this share of a component is taken without a child
    # D1c (2026-10-15, cy_t): a cover whose excess is more than tree_max_cover_excess of the input is replaced by the
    # cascade's best contained primitive when that explains tree_contained_min_explained of the input with
    # tree_contained_min_precision of its own material inside it - a union of bars is built by adds, not carved
    tree_contained_start: bool = True
    tree_max_cover_excess: float = 0.3
    tree_contained_min_explained: float = 0.05   # a sanity floor; the decision is the margin below
    tree_contained_min_precision: float = 0.95
    tree_contained_margin: float = 0.8           # swap when the primitive's missing material < margin x the cover's excess
    tree_cover_child: bool = True        # the cover child (the component's bounding box + the grandchild's re-add)
    tree_max_depth: int = 2              # a grandchild at depth 1 may have a cover child of its own up to this depth
    tree_cover_leftover: float = 0.1     # a cover child passes when at most this share of its wrong material stays wrong
    tree_work_budget: int = 60           # components solved at the root before the tree stops (the work budget)
    tree_max_children: int = 12          # box children per program at most (24 of them crashed freecadcmd's build)
    # the repair pass after the loop's rounds (reco_engine/repair.py): operations the final model does not need, and the
    # one whose absence makes a multi-piece model whole again, are taken back. None: in mode "tree" only.
    # the memory watcher (reco_engine/memory_guard.py): the loop stops with the model it has when the machine's
    # available memory falls below this (MB; None: $SHAPE_RECO_MIN_FREE_MB or 1200; 0: off), checked between rounds
    # and between the feature phase's pieces - the user's rule 2026-10-15, a killed process keeps nothing
    memory_min_free_mb: Optional[float] = None
    repair: Optional[bool] = None
    repair_tolerance: float = 1e-4
    repair_max_loss: float = 0.02        # the solid rule gives up at most this much IoU (a floating add is no severing)
    # D5, the solid gate: a cut (or intersect) that leaves the model in more connected pieces than the input has is
    # rejected. In `_evaluate` for every candidate on grids up to connectivity_max_cells (a labelling of the grid per
    # candidate; the frame's 20.7M cells would cost seconds each), always on the tree's chosen candidate.
    connectivity_gate: bool = True
    connectivity_max_cells: int = 4_000_000
    # The exact solid gate (2026-10-16): the grid's labelling cannot see a cut that leaves pieces touching the kept
    # model along the cutter's faces (the gate corpus of 10-15: 20 of 60 tree models in 2-6 solids, each split by a
    # composite tool cut the grid had passed - scripts/solids_per_op.py names the operation), so a chosen cut /
    # intersect is also built in freecadcmd with the program so far (cfg.feature_build injectable) and refused when
    # the solid count grows; the next passing candidate by gain is taken. At most exact_solids_max_checks builds per
    # component, none once the program has exact_solids_max_ops operations; a build that fails or times out leaves the
    # grid's answer standing. Adds are judged too (floating or merely touching material is another solid). Off without a build (no freecadcmd and no injected cfg.feature_build).
    exact_solids_gate: bool = True
    exact_solids_max_checks: int = 3
    exact_solids_max_ops: int = 40
    exact_solids_timeout: int = 60
    # A cut piece also gets its `excess_boxes` largest completely filled axis-aligned boxes as candidates (largest_boxes:
    # exact box cuts - the frame's windows and flange recesses after a box start), and void_fit runs only on pieces of at
    # most `void_fit_max_voxels` (measured 2026-09-26: 700 s of void_fit on the frame's whole excess, one piece).
    excess_boxes: int = 3
    void_fit_max_voxels: int = 300_000
    # The op budget on the starting-shape path also grows with the surface inventory: at least 1 (the start) +
    # budget_surface_margin + the number of plane / cylinder / sphere surfaces with >= budget_surface_share of the area,
    # capped at budget_cap. predict_ops_budget knows only the genus: the liner (genus 0, 19 faces >= 0.5 %) got 5 and
    # stopped after 4 cuts, the voxel fallback then spent 16 boxes of 12-180 voxels (measured 2026-09-26, docs/LINER.md).
    budget_from_surfaces: bool = True
    budget_surface_share: float = 0.005
    budget_surface_margin: int = 4
    budget_cap: int = 60
    # The rough stage (the user's decision 2026-09-26: small features are left to a second loop): a residual piece
    # smaller than rough_min_piece_share x the input voxels is not explained by this loop - it stays in the residual
    # for the feature stage. The liner's last cuts were 28-32-voxel specks (0.06 %) and they made the solid invalid.
    rough_min_piece_share: float = 0.001
    # boolean_tools (the user's rule 2026-09-26: "a boolean result is a tool itself"): per piece also the pairwise
    # booleans of its best single candidates (A & B, A - B, B - A, A | B - composites nest, so a tool can be a boolean
    # of booleans), and every operation accepted so far re-placed at the piece (the tool library: repeated features).
    boolean_tools: bool = True
    tool_library_size: int = 6
    vary_on_failure: bool = True
    vary_rotations: int = 3
    tau_fraction: float = 1.0
    max_cells: int = 40_000_000
    pad_fraction: float = 0.08        # grid margin around the part's bounding box, share of its largest side
    exact_channels: bool = False      # SurfaceField.query exact=True for T_in (slower; exact far from the surface too)
    # The closest-point backend for T_in and the grade's chamfer (src.closest_point_field.field_backend), None: the
    # environment's ($SHAPE_RECO_FIELD_BACKEND, default auto = cuda). "igl" is exact and on the frame at 96^3 took 0.58 s
    # where the default took 313 s (the GPU brute-force k-nearest search over the 537k-triangle refined mesh).
    field_backend: Optional[str] = None
    registry: Optional[list] = None   # stage registry entries; None: cascade.load_default_registry()
    models: Optional[object] = None   # passed to run_cascade (CNN entries)
    top_k: int = 2
    first_pass_iou: float = 0.9       # the first add is the cascade's primitive when it explains the input this well
    component_pass_iou: float = 0.6   # a leftover's primitive must reach this IoU (in its scope) to be a candidate
    min_component: int = 6            # smaller leftover components are noise (voxels)
    min_gain: int = 3                 # an operation must shrink |R| by at least this many voxels
    tolerance: float = 0.005          # stop when |R| <= tolerance x input voxels
    max_ops: Optional[int] = None     # None: predict_ops_budget(field).max_ops
    op_guard: Optional[bool] = None   # None: $SHAPE_RECO_OP_GUARD (default off: the order of `cuts_first`)
    purity_min: Optional[float] = None  # void_quality purity for cuts; None: void_quality.PURITY_MIN
    use_void_fit: bool = True
    voxel_fallback: bool = True
    max_fallback_boxes: int = 16      # the voxel fallback: the largest filled boxes first (largest_boxes), few elements
    freecad: bool = True              # build + grade the final program
    grade_each_step: bool = False     # also build + grade after every accepted operation
    out_dir: Optional[Path] = None    # FreeCAD output (default: a temp directory)
    builder: Optional[Callable] = None  # (sp model, out_dir, name, field) -> dict(valid, n_solids, step, stl, chamfer); tests
    # The sub-loops of the first-add phase (reco_engine/ibeam.py). `ibeam`: straight members found in the mesh's sections
    # (I-beam -> 3 boxes, rod, bar, tube, rhs); $SHAPE_RECO_IBEAM=0 disables. `hollow`: a closed cavity (an inner shell)
    # -> outer body add + inner body cut; $SHAPE_RECO_HOLLOW=0 disables. `hint_subclass`: the TypeNet subclass
    # ("box/ibeam", "*/frame", "*/hollow", "cylinder/tube", "box/rhs"), when given, selects the sub-loop instead of the
    # analytic gate; any other value ("box/full", "*/sector", ...) runs none. None: the registry's type stage on the
    # whole input provides it (residual_steps); a hinted sub-loop that finds nothing falls back to the analytic gate.
    ibeam: bool = True
    hollow: bool = True
    # the cavity of a hollow body cut first (2026-10-29, the gate's unions): the cavity's own shape is fitted next to the
    # outer shell's kind (a spherical cavity in an oriented box got only a box cutter), and a cavity cut that also takes
    # input material - other bodies of a union reaching into the cavity - is accepted at a precision of at least
    # hollow_cut_first_precision (not member_min_precision) when its exact gain reaches min_gain: the residual phase adds
    # the taken material back, as the CSG tree does (the cavity cut before the bodies that reach into it). On the gate four
    # unions had the hollow leaf's inner sphere refused at 63-84 % and the model filled their cavities completely.
    # MEASURED, then off (2026-10-29, d8367ee6, server_c/server_d, both arms at once): worse on all three unions that ended -
    # aegir_n4_U_000153 exact IoU 0.496 -> 0.361, n5_U_000906 0.354 -> 0.340, n6_U_000303 0.352 -> 0.323 (the single hollow
    # sphere stays 1.000): the residual phase (adds first, budget 5) spent every operation on boxes of 134-212 voxels
    # putting the intruders' material back and never cut the cover's excess (177701 voxels; off: five cuts, the first
    # -48486). It needs the intruding bodies re-added whole, not as fragments.
    hollow_cut_first: bool = False
    hollow_cut_first_precision: float = 0.6
    hint_subclass: Optional[str] = None
    member_min_length_ratio: float = 1.5     # 2.0 missed the frame's 8 transverse plates (712 x 444 x 20 mm: 1.6); S6
    member_min_precision: float = 0.95  # share of an add inside the input / of a cut's removal outside it (exact, mesh)
    member_samples: int = 1500          # points per primitive for that exact volume accounting
    member_min_new: float = 0.2         # an add must bring at least this share of its own volume as new input material
    # Members are stored and checked for congruency (the user's rule 2026-09-29; reco_engine/member_store.py): members
    # equal in kind, section dimensions and length (within member_congruent_tol, relative) form a group; the first one
    # whose operation passes when judged in full is the group's source, the others get its primitive moved by the
    # rigid motion between the members and judged by the same exact judge - a copy that fails is judged in full by
    # itself, and a member that fails never rejects its group. The groups go into the session's tool store
    # (feature_tool_store) as created tools and into the steps' info (group, by_congruency). False: every member by
    # itself, as before.
    member_congruency: bool = True
    member_congruent_tol: float = 2e-3
    # Other solutions before the fallback (the user's rule 2026-09-29; reco_engine/solution_choice.py): after a
    # first-add sub-loop and the add/cut loop, the CLASSIC solution (no sub-loop: the plain first add, then the add/cut
    # loop with cuts first) is computed on a copy of the state, both are judged by the sampled exact IoU on the mesh
    # and the better one is kept - before the voxel-cube fallback or the feature phase. "auto": only when the member
    # phase was asked for and found nothing usable, or the voxel residual is above `tolerance`, or the exact account
    # of the accepted operations leaves more than solution_exact_tolerance of the input's volume open (None: that
    # test is off); "always"; "never". solution_exact_tolerance 0.05: the frame's members leave 0.0345 (exact IoU
    # 0.9655, the root fillets) and the frame counts as solved - its run must not pay for an alternative; the review
    # set's r14 (mitred) leaves 0.114, r10 0.075, r06 0.062 (frame_review_fix1, 2026-09-29). The classic solution
    # replaces the first one only when its IoU is higher by more than solution_margin (the sampling's noise).
    solution_alternatives: str = "auto"
    solution_exact_tolerance: Optional[float] = 0.05
    solution_margin: float = 0.01
    solution_samples: int = 1500        # points per add of a solution for the sampled exact IoU
    # The exact judge (reco_field.exact_check): after the FreeCAD build, the residual against the INPUT itself by OCCT
    # booleans (exact IoU, missed/invented volume, pieces) and the genus of the result. The loop's own voxel residual
    # is blind below its pitch (the frame: 0 voxels at 48^3, 9 % missed exactly).
    exact_grade: bool = True
    # The feature phase (the user's specification 2026-09-27: "rough shape - liner = possible features; segment, guess
    # shapes on them, score, limit the usage, validate, iterate - inside the existing loop"): after the voxel add/cut
    # loop the program is built in FreeCAD and its exact difference to the input taken by OCCT - missing = input - model
    # (adds), invented = model - input (cuts), every solid a piece. The biggest pieces get a primitive from
    # reco_engine.feature_stage.guess_candidates (the input's exact STEP faces, the surface fitters, the cascade on the
    # piece's parity voxels; cheapest first, stop at a score of 0.99), scored by its sampled volume IoU against the piece.
    # A guess scoring >= feature_min_score is applied and validated by a real FreeCAD build: one valid solid, and the
    # exact residual (missing + invented) must drop by >= feature_min_gain x the piece's volume - else it is reverted.
    # The voxel-cube fallback (6) does NOT run when the phase is on: on the liner (docs/LINER.md, 2026-09-27) the
    # fallback's boxes gained 0.006 voxel IoU with 9 boxes, and at the same piece threshold nothing at all, while what is
    # left are bores of R 2-10 mm below the pitch - material the exact difference sees and the fallback's voxel blocks
    # cannot fit. Off by default: every measurement before 2026-09-27 ran without it.
    feature_phase: bool = False
    # The polish (reco_engine/polish.py, 2026-10-02): after the feature phase, every accepted curved primitive refitted
    # to the input's VERTICES - a CAD tessellation's vertices lie on the surface, its triangles sag inside a convex face,
    # and the loop fits the triangles. On v5 spheres the loop's radius was 6-11 um short (IoU 0.998 against the exact
    # tree, tier B); polished, exact (1.00000). On by default since 2026-10-02: re-rating the finished runs of calib
    # and eval polished (scripts/polish_runs.py; docs/data/polish_2026-10-02.json): A 24 -> 56 of 74, no tier and no
    # IoU worse.
    polish: bool = True
    # Prune (the user, 2026-10-30: "minimize the number of parts: simple trees for simple parts instead of complex trees
    # for simple parts"): at the end of the run, after the polish, the parts the shape does not need are dropped -
    # reco_engine/prune.minimize on sample points against the input (one at a time, the best program first, while the
    # sampled IoU stays within prune_tolerance of the program as it came) - and the pruned program is built in FreeCAD:
    # it must stay one valid solid, else the longest prefix of the drops that does is kept. A pruned part stays in the
    # program with accepted = False and "pruned" in its label. Measured offline on the gate's 60 stored runs of
    # adc2d6c3 (defaults on; scripts/minimize_parts.py, docs/data/minimize_parts_gate.json): 445 -> 271 parts at 0.002
    # (median 7.5 -> 4.0 per run; a 2-3-leaf union 12.0 -> 5.8), mean sampled IoU +0.015 - a quarter of the parts
    # changed nothing or harmed. Off until a protocol accepts it.
    prune_parts: bool = False
    prune_tolerance: float = 0.002
    prune_samples: int = 100_000
    # Rough means rough (the user's diagnosis of the live liner run, 2026-09-27: "the rough reconstruction was too
    # detailed and started wrong cuts"; the rule is the user's: "grade the cuts in the rough step based on the size of
    # the first cut"). With the feature phase on, the first cut of the voxel loop sets the scale: its gain g1 (the
    # liner's bore, 39321 voxels). Every later cut is graded gain / g1 and belongs to the rough model only at a grade
    # >= rough_cut_grade; the first one below ends the rough stage, its component and everything smaller are left for
    # the feature phase (the liner's later cuts graded 0.023, 0.020, 0.008, 0.007, 0.003, 0.002, 0.002, 0.002 - the
    # composites fitted to voxel blobs). Adds after the first add are graded the same way against the first add's gain
    # (the input voxels the cover explains) - the user named cuts only; the symmetric rule for adds is our reading.
    # After the first operation of its kind a component tries single primitives first, composites and boolean tools
    # only when no single one reaches the grade. rough_min_gain_share (share of the input voxels, None: off) is an
    # extra absolute floor; rough_max_ops caps the loop's operations (None: the usual budget). Without the feature
    # phase none of this applies; rough_cut_grade 0 switches the rule off (the loop as before). Default 0.10 - the
    # user's decision 2026-09-28 after two live liner runs ("the first cut been again too detailed"): the rough model
    # is the cover and the operations of the first cut's order of magnitude, the rest belongs to the feature phase.
    # (Measured before, with single-primitive features only, final exact IoU after a 900 s phase: rule off 0.9632,
    # grade 0.02 0.9402, share 0.05 of the input 0.8874 - docs/RUNNING.md.) With the phase on the first add is also
    # ONE plain primitive (no composite cover, no variation rounds) and the first cut a single primitive.
    rough_cut_grade: float = 0.1
    rough_min_gain_share: Optional[float] = None
    rough_max_ops: Optional[int] = None
    feature_max_pieces: int = 8          # pieces guessed per iteration (the biggest ones not tried before)
    # Two caps (2026-09-28: in the user's live liner run 28 rejections used up a single cap of 40): full guesses in the
    # whole phase (runtime; a transfer to a congruent piece costs a score and does not count) -> stopped "max pieces",
    # and operations applied -> stopped "max applied".
    feature_max_total_pieces: int = 40
    # The pieces one exact difference writes out (each with its STEP, mesh and faces). It was feature_max_total_pieces:
    # the liner's 36 congruent bolt holes came as 24 + 12, the 12 waiting for a second difference that never came
    # (2026-09-29). A congruence group is guessed once, so many small congruent pieces cost little.
    feature_max_written_pieces: int = 200
    # The level of detail (the user's rule 2026-09-29: "a slider for the max IoU we need to gain, which allows us to
    # create more or less detailed models"): the feature phase takes the congruence groups with the most volume first
    # and stops taking when the IoU estimated from the volumes, (V_in - missing) / (V_in + invented), reaches this
    # target; it ends when a difference measures it. None: every piece (the most detailed model). The estimate counts
    # a piece as explained in full by its tool; a rejected or partial tool leaves the model below the target.
    feature_target_iou: Optional[float] = None
    # The rough model as several sub-parts (the user's rule 2026-09-29: "for one part of the rough guess add, assume we
    # add more subparts before cutting rough"): the first add is a cover of a few primitives added to each other
    # (reco_engine/subpart_cover.py), chosen by the exact surface coverage (reco_engine/surface_orders.py: the share of
    # the input's surface the cover reproduces in position, normal and curvature) under full containment; the cuts
    # begin after it. STEP inputs only (the exact faces). Taken instead of the single first add when it has more of
    # that coverage (by rough_subparts_min_gain) or less volume (by rough_subparts_min_saving). Off by default until
    # measured on a set: on one ABC part (00000424, a disc of 32 faces) the single oriented box reproduces 0.46 of the
    # surface exactly, four sub-parts 0.84, in 15 s.
    rough_subparts: bool = False
    rough_subparts_max: int = 6
    rough_subparts_min_gain: float = 0.03
    rough_subparts_min_saving: float = 0.1
    feature_max_applied: int = 60
    feature_max_iterations: int = 3      # difference -> guesses -> validation rounds -> "max iterations"
    feature_time_budget: float = 900.0   # s, wall clock of the whole phase, checked between pieces and between sources
    feature_call_timeout: int = 300      # s, every FreeCAD subprocess of the phase (build, exact difference)
    # a piece below this share of the input volume is dropped, unless known or a feature by its absolute size (below).
    # 1e-3 since 2026-10-30, by protocol focus_min_share_v1 (docs/data/protocol/focus_min_share_v1, kind "saving"): on
    # TEST (202 known trees, server_a, both arms at once) good 89 = 89, rescued 0, broken 0, one valid solid 189 -> 191,
    # frame 0.9666 and liner 0.9952 unchanged, median paired time -16.7 s per part (p < 1e-4; median 444 -> 412 s),
    # time-budget stops 19 -> 9; DEV (the gate) good 29 = 29, about 10 % faster. Before: 1e-5 (computed, not measured: a
    # R 2 mm bore 67 mm long - the liner's wall - is 1.8e-5 of the liner's 4.6e7 mm^3).
    feature_min_share: float = 1e-3
    # Small pieces (2026-09-29): the liner's four holes R 2.459 x 15 in its top face are pieces of 293.9 mm^3 each,
    # under the share's 459 mm^3, and were dropped while the same hole was cut elsewhere. A piece under
    # feature_min_share is kept when it is KNOWN (feature_keep_known: congruent with a piece accepted in this phase or
    # with a created tool of the session's store, or a catalogue fit of its faces has the type and size of a tool of
    # the store) or when it is a feature by its ABSOLUTE size (feature_min_radius_depth, 0 = off: it has a real curved
    # face - one on the input's surface - of radius r and at least that many times pi r^3 of volume, a bore that
    # many radii deep; feature_stage.feature_sized). Under feature_small_share of the input nothing is written or
    # looked at. Measured on the liner's rough difference: 56 solids, none between 1 and 293 mm^3 - the four holes
    # are what this adds (19.8 x pi r^3 each).
    feature_keep_known: bool = True
    feature_min_radius_depth: float = 1.0
    feature_small_share: float = 1e-6
    feature_min_score: float = 0.85      # sampled volume IoU of the guess against its piece
    feature_min_gain: float = 0.25       # the exact residual must drop by this share of the piece's volume
    feature_tolerance: float = 0.001     # "converged": missing + invented <= this share of the input volume
    # One FreeCAD build + one exact difference per iteration for all its guesses; when that fails, the batch is halved
    # and each half validated on its own down to the single operation that fails (the step stream names it). False:
    # one build + difference per guess (the honest, dearer order).
    feature_batch: bool = True
    # Applied at once (the user's rule 2026-09-30: "if a feature is accepted, apply it immediately"): a group's
    # guesses are checked in FreeCAD as soon as the group is guessed (the exact per-feature check: what the tool
    # shares with its piece, what it harms); a tool that passes and gains enough joins the program there and then,
    # its step says "applied at once". The build of the whole model and its difference against the input follow once
    # per batch as the confirmation (step "feature_confirm" before, "feature_confirmed" after); when that fails,
    # the features applied at once are taken back and validated in halves as before.
    feature_apply_at_once: bool = False
    # Congruent pieces reuse one guess (the user's verdict on the live liner run, 2026-09-27: nine pocket pieces of
    # 86483.945 mm^3 and three bore pieces of 7716.676 mm^3, each guessed from scratch at ~30 s): pieces equal in
    # volume, area, principal moments and exact face counts form a group; the first is guessed, the others get its
    # primitive moved by the rigid motion of centroid and principal axes (feature_stage.transfer_candidates), scored
    # on themselves (source "pattern:<id>"), a full guess when that scores below feature_min_score. A group counts
    # once against feature_max_pieces (the batch of one validation); feature_max_total_pieces caps the pieces taken.
    feature_patterns: bool = True
    # A failed representative does not condemn its group (the user, 2026-09-29: "investigate the pattern it seems to
    # be an error source"; liner run 20260928-220854: 35 of 45 rejections were "the congruent piece found no
    # primitive", untried): the next member is guessed in full, at most feature_group_tries members per group; a
    # member that finds a tool hands it to the ones that failed before it. The rest of a group without a tool is not
    # guessed (a group of 36 does not cost 36 full guesses) and goes to the voxel fallback.
    feature_group_tries: int = 3
    # The voxel fallback (the user's rules 2026-09-29: "when reconstruction fails, always use voxels as compound, for
    # stability" and "before going into fallback check other solution"): the LAST resort of a piece. Only after its
    # typed tools, contact-face and hole tools, primitive sources, boolean tools, the full guesses of its congruence
    # group and the split found nothing that passes, the piece's own material becomes a compound of voxel cubes at
    # the licensed pitch (feature_stage.voxel_tool: tau / 3 or finer, D-005), merged into at most
    # feature_fallback_max_boxes boxes, inside the piece only; its step names what was tried before, with the best
    # precision / recall, and what the boxes leave of the piece. Checked and validated like every tool (it owes the
    # gain of what it covers). Below feature_fallback_min_recall of the piece's volume it is not worth an operation.
    # Sub-parts before the voxels (the user's rule 2026-09-29: "for not reconstructable parts of features ... assume we
    # add more subparts"): a piece without a tool is covered by up to feature_subparts_max primitives, each searched
    # in the space of every primitive type (feature_stage.subpart_tool, reco_engine/subpart_cover.py), fused to one
    # tool. The primitives bound their sub-parts, so the tool is taken only when the share of it inside the piece
    # reaches feature_min_precision; it is then checked and validated like every tool. Off by default until measured.
    feature_subparts: bool = False
    feature_subparts_max: int = 6
    feature_voxel_fallback: bool = True
    # cut_grow (the user, 2026-10-29: "our cut method needs to be more accurate, often two cuts could be done by only
    # one; check the sizing of cuts"): a box cut's faces grow outwards layer by layer while the layer holds no input
    # material (a cell within half a voxel of it stops the face); a face that reaches the grid's edge is open and ends
    # three cells beyond it; the grown faces are snapped onto the surface again. Kept when the checks pass and the gain
    # does not drop. Measured before: a quarter of all box-cut faces of the gate's runs stopped short of open space
    # (scripts/cut_audit.py), and the bracket's cuts 1 and 6 were one box. Off: the old push of three cells.
    # MEASURED, then switched off (2026-10-29, the bracket on server_a, 4481a471 + cut_grow): the grown cuts overlap ("10404
    # voxels cut again"), the bigger first cut raises the rough rule's bar (0.1 x 26776 instead of 18822), the rough phase
    # stops after 3 cuts instead of 7 and the run ends at exact IoU 0.8585 instead of 0.9992. Kept as an option;
    # cut_open_faces (below) is the part of it that helps.
    cut_grow: bool = False
    # cut_open_faces: a box cut's face that sees only open space up to the grid's edge (no input material in the layers
    # beyond it) ends three cells beyond the edge; every other face stays where the fit put it. A cut face snapped onto
    # the cover's own face (the bracket's boss top at z 34) left a zero-thickness skin: Hausdorff/D 0.1485 at exact IoU
    # 0.9992.
    cut_open_faces: bool = False      # MEASURED, then off (2026-10-29, the bracket, 928694c5): on this part nearly every
                                      # face of the first cut is open, so it grew as far as with cut_grow - IoU 0.8585 again
    cut_merge: bool = True            # merged cuts after the rough phase (reco_engine/cut_merge.py, 7b of residual_steps)
    # cut_tidy (2026-10-29): after the rough phase and after the feature phase, cut faces within half a cell of a face of
    # the cover (the first add) move three cells past it (cut_merge.lift_cover_faces) and two cuts whose faces lie less
    # than half a cell apart are made to overlap where no input material lies between them, else the free part of the
    # seam gets its own seam cut (close_seams). Both remove skins and walls of no volume that the IoU does not see. The
    # bracket, offline on the stored programs: with the merged cuts (bracket_T1_tidy) Hausdorff/D 0.1485 -> 0.0020, RMS/D
    # 0.00883 -> 0.00013, tier E -> B, IoU 0.99939 -> 0.99944 (6 seam fixes; the 0.031 mm wall between the merged cut and
    # the rough guess's box, the rib between them); without (bracket_T1_rough) 0.1486 -> 0.0118, tier E -> C.
    cut_tidy: bool = True
    # the rough guess (the user, 2026-10-29: "often the first rough guesses work well, but then the last guess is a really
    # detailed part and to find a solution is impossible - rough guess those parts"): a piece that no tool, no split and
    # no congruent piece explained gets the best single candidate tried for it - before the sub-parts and the voxel
    # fallback - when its score, precision and recall reach the rough_* bounds; it is applied only if the exact residual
    # drops by feature_rough_min_drop of the piece and it harms at most feature_rough_max_harm of its own volume, and its
    # source is "rough:<source>". The bracket of the detail study (2026-10-29): piece #3, 8682 mm^3, was left at exact
    # IoU 0.9292 because its box - cylinder (score 0.973, precision 0.980) would cut 133 mm^3 of the rib.
    feature_rough: bool = True
    feature_rough_min_score: float = 0.5
    feature_rough_min_precision: float = 0.9
    feature_rough_max_harm: float = 0.05
    feature_rough_min_drop: float = 0.5
    # ... and at most this share of the INPUT's volume (the check's tolerance and the batch's exact difference): protocol
    # rough_tidy_v1, DEV (2026-10-29), two parts destroyed - an over-covered piece 5x (aegir_n6_I_000121) and 11x
    # (aegir_n8_U_000040) the part's volume got a box | sphere cut that removed all of the part; the residual still
    # dropped by half the piece and 5 % of the giant tool's volume exceeded the part. The bracket's rough guess harms 0.12 %.
    feature_rough_max_harm_input: float = 0.01
    # Symmetry (the user's rule 2026-10-29: "always check if symmetrical features exist"): the input's mirror planes and
    # rotation axes (reco_engine/symmetry.py) are found at the start of the phase and named in the feature_groups step;
    # a congruent piece that a symmetry maps the representative onto gets the representative's tool copied EXACTLY by
    # that symmetry (source "symmetry:<id>:<name>") before the moment-based transfer, whose float noise made a corner
    # tool of ABC part 00009283_..._step_001 leave a 0.00094 mm^3 sliver (refused as "2 solids").
    feature_symmetry: bool = True
    feature_fallback_max_boxes: int = 16
    feature_fallback_min_recall: float = 0.25
    # Splitting (the user's decision 2026-09-27: "split the excess and guess shapes on the segments of the voxel
    # field"): a piece of at least feature_split_min_share of the input whose best single primitive fails is put on its
    # own grid (tau / 3 of its thinnest wall, <= 96 cells per side) with its field channels computed once, cut into at
    # most feature_max_segments segments (feature_stage.segment_piece: "necks" - watershed on the inside distance;
    # "surfaces" - boundary patches by the field's normals; "axis" - one cut across the longest axis; "auto" tries them
    # in this order), segments under feature_segment_min_voxels join a neighbour, and every segment is guessed and
    # scored as PART of the excess: precision (share of the primitive that removes no input material) >=
    # feature_min_precision and recall (share of the segment covered) >= feature_min_recall; a guess short of the
    # precision by at most 0.1 is shrunk first. A failing segment is split again while the depth is below
    # feature_split_depth. The validation is the same as for whole pieces.
    # The FreeCAD check of every feature (the user's request 2026-09-27: "use compound explode to check with FreeCAD
    # our results for features"): before a batch is validated, reco_field.exact_check.feature_checks builds each
    # feature and its piece and measures common(feature, piece), feature - piece, piece - feature and the input
    # material inside feature - piece; a feature whose harm exceeds feature_check_tolerance x its volume (and 1e-6 of
    # the input) is rejected - the exact version of the sampled precision. At the end the phase writes
    # <out_dir>/features/features.FCStd (exact_check.write_features_document, under feature_document_timeout; a
    # failure is reported, never raised). feature_checks: an injectable (input, items, timeout) -> {id: dict} (tests).
    # How a whole piece's guesses are scored: "part" - precision and recall as the operation (feature_stage.PartScorer,
    # 2026-09-27: the exact STEP faces lost to fits under the volume IoU on pieces closed by a curved model face);
    # "iou" - the sampled volume IoU against the piece, as before.
    feature_scoring: str = "part"
    # Boolean tools as a feature source (the user's rule 2026-09-28): a piece whose single primitives all fail gets the
    # composites of feature_stage.tool_candidates - the rounded box of its planes and cylinder faces (box minus, per
    # rounded corner, corner square minus cylinder: exact for any radius; a pocket with four rounded corners is 9
    # operands, hence the default 9 where 3 was proposed), then pairwise &, |, - - fewest operands first, the first
    # passing one ends the search; at most feature_tool_candidates per piece. Scored and checked like a primitive;
    # congruent pieces get it by the pattern transfer (every operand moved).
    feature_tools: bool = True
    # Tool types (the user's request 2026-09-28: "a type set of tools ... a FeatureNet which groups the found features,
    # and based on the grouping it uses a tool, creates one by itself, or ... recycles an old tool of the session"):
    # the pieces are grouped by tool type (reco_engine.feature_groups: the analytic rule table, the FeatureNet when
    # given) and congruence, a "feature_groups" step lists them, and every representative tries, in this order, the
    # session's tools of its type (source "session:<id>", `feature_tool_store`), the catalogue's fits of its ranked
    # types (reco_engine.tools, "catalogue:<type>"), and only then the primitive sources and the composite search
    # ("created"; an accepted composite is stored, as a catalogue type when its tree has one's shape).
    # feature_tool_store: a tools.ToolStore (kept by the caller across runs: the GUI's window session) or the path of
    # its JSON file (loaded before, saved after the phase); None: a store for this run only.
    feature_types: bool = True
    feature_tool_store: Optional[object] = None
    # Hooks for work of other sessions (2026-09-28), no-ops by default: `feature_net` the checkpoint of the FeatureNet
    # (reco_engine/feature_net.py, its classify_pieces re-ranks the types feature_groups admits), `first_guess` the
    # user's first operations (reco_engine/user_guess.py), `adviser` who answers the loop's questions
    # (reco_engine/advice.py). user_guess.py and advice.py exist; reco_engine/feature_net.py does not (checked
    # 2026-10-01): a `feature_net` value is reported in the feature_done message and ignored.
    feature_net: Optional[str] = None
    first_guess: Optional[object] = None
    adviser: Optional[object] = None
    advice: Optional[object] = None      # an advice.AdviceConfig (the thresholds and the budget); None: its defaults
    # The GUI's piece feedback (reco_gui/pieces_view.py writes iteration_<n>/piece_feedback.json beside pieces.json):
    # a file, or a run directory searched for every such file. A later run of the same input matches its pieces by
    # sign, volume (1e-3) and centre (1e-3 of the piece's size) and acts on the verdict - "leave" / "not_a_feature":
    # skipped with that reason; "split": the split pass at once; "voxels": the voxel fallback at once
    # (feature_voxel_fallback; left when that is off); a tool class in "kind": that type tried first - and says so in a "feature_feedback" step.
    piece_feedback: Optional[object] = None
    feature_tool_max_operands: int = 9
    feature_tool_candidates: int = 12
    # Snap the rough model onto the input's exact faces before its first difference (STEP input;
    # feature_stage.snap_to_faces: box faces and cylinder ends onto the nearest parallel planes, cylinders onto the
    # input's axis and radius). Measured 2026-09-28 on the liner with the coarse rough rule: the plain cover's end lay
    # 0.011 mm from the liner's end plane and OCCT's difference failed even fuzzy; snapped (1.3 mm shift, R 192.3 ->
    # 192, bore R 125.5 -> 125) the plain booleans agree: exact IoU 0.860, 52 pieces in 69 s.
    feature_snap: bool = True
    # After the snap a rough cylinder takes the input's turned contour (reco_engine/turned_contour.py): the coaxial
    # cylinders and cones of the input give a profile r(t), every ring between it and the rough radius that holds none
    # of the input's material is cut. Measured 2026-09-29 on the liner's rough model (add sector R 192, cut R 125):
    # the shoulder R 175 with its chamfer and the bottom chamfer were 5.2e5 mm3 too much and, with the sector's wedges,
    # joined the pieces at the split plane and the top into one of 1.19e6 mm3 with 676 faces.
    feature_contour: bool = True
    # Contact grading (the user's rule 2026-09-28: "surfaces which will touch the input, and surfaces that only exist on
    # the tool"): every graded candidate reports its contact / free / harm areas and its contact recall and precision
    # (feature_stage.contact_grade) in its step; with feature_contact_grading its score is also the lower of those two,
    # so a tool whose real faces are wrong loses to one whose free faces are longer. Off: today's score (compare).
    feature_contact_grading: bool = False
    # The grade as a rank (the user's rule, 2026-09-28): among the candidates that pass, the one whose surface
    # reproduces more of the piece's contact faces and harms less (contact recall x contact precision, in steps of
    # 0.01) comes first, before the fewer operands. Off: the order without the grade.
    feature_contact_ranking: bool = True
    # The contact-face guess source (2026-09-28): before a piece is guessed as a whole piece or split, the tool of its
    # contact faces alone (feature_stage.contact_tool: the faces on the input's surface; open - run on past the piece -
    # wherever only faces of the current model close it) is offered, source "tool (contact faces)", after the typed
    # candidates; judged like every candidate (sampled precision / recall, then the exact check and the validation).
    # Needs the piece's exact faces (a STEP input, "step faces" among the sources) and feature_scoring "part".
    feature_contact_tool: bool = True
    # The order of the pieces (the user's rule, 2026-09-28: "volume share / number of faces", a congruence group as
    # one unit): "volume per face" - the groups by their members' summed volume share over the representative's
    # number of faces, the most volume explained per face to get right first; "volume" - by the largest member's
    # volume, as before. Read here as the quotient; it decides who is tried first under the caps and the time budget.
    feature_piece_order: str = "volume per face"
    # Super-thin material (the user's rule, 2026-09-28; feature_stage.THIN_WALL): the threshold as a share of the
    # input's diagonal, or absolute (feature_thin_wall_mm); 0 = off. A piece thin as a whole (the 95th percentile of
    # its wall thickness below the threshold) is not guessed, gridded or split: it is listed as suppressed (steps
    # feature_residual / feature_done: `suppressed`, `suppressed_volume`) and its volume stays in the exact residual.
    # A thick piece's grid takes its pitch from the walls at or above the threshold and is opened at it.
    feature_thin_wall: float = THIN_WALL
    feature_thin_wall_mm: Optional[float] = None
    feature_check: bool = True
    feature_check_tolerance: float = 1e-3
    feature_document: bool = True
    feature_document_timeout: int = 900
    feature_checks: Optional[Callable] = None
    feature_split: bool = True
    feature_split_method: str = "auto"
    feature_split_min_share: float = 1e-3
    feature_max_segments: int = 6
    feature_segment_min_voxels: int = 50
    feature_split_depth: int = 1
    # 0.999 = 1 - feature_check_tolerance: the sampled precision estimates what the exact check judges; at 0.98 the
    # liner's pocket box (sampled 0.994) passed, the check rejected it (440 mm^3 of input removed at the rounded
    # corners) and the tool that is right - the rounded box, harm 0.0 mm^3 - was never tried (2026-09-28)
    feature_min_precision: float = 0.999
    feature_min_recall: float = 0.5
    feature_sources: Optional[Tuple[str, ...]] = None   # None: step faces, fit, cascade (STEP); fit, cascade (STL)
    # An STL input becomes an OCCT solid of its facets (reco_field.exact_check); measured 2026-09-27: box_with_bore.stl
    # (932 triangles) differenced with its box in 4.1 s. Above this many triangles the phase is skipped (as _exact_grade).
    feature_stl_max_faces: int = 12000
    # The FreeCAD calls of the phase, injectable (tests): build (result, out_dir, name, timeout) -> dict(valid, n_solids,
    # step); difference (input, model_step, pieces_dir, max_pieces, min_solid, timeout) -> an exact_residual dict; guess
    # (piece, sources, deadline, registry, models) -> a feature_stage.guess_candidates dict. None: the real ones.
    feature_build: Optional[Callable] = None
    feature_mesh_fallback: bool = True   # the meshes' difference (manifold3d) when OCCT's disagrees or times out (2026-10-15)
    feature_difference: Optional[Callable] = None
    feature_guess: Optional[Callable] = None
    # The difference of a model WITH features is not computed from input and model (measured 2026-09-29 on the liner,
    # 39 features: every OCCT intersection of the two came back empty, plain and fuzzy) but from the difference of
    # the model before and the tools applied since (reco_field.exact_check.incremental_residual). The call, injectable:
    # (input, model_step, state, tools [dict(op, program)], state_dir, pieces_dir, max_pieces, min_solid, timeout) ->
    # an exact_residual dict. None: the real one - with an injected feature_difference none (the fake world has no
    # state). False as feature_incremental: the global difference always.
    feature_incremental: bool = True
    feature_incremental_difference: Optional[Callable] = None
    # its timeout: feature_call_timeout + this per tool (measured on the liner: 3 s per tool, 6.5 s under load)
    feature_incremental_tool_seconds: float = 10.0


# ------------------------------------------------------------------------------------------------ helpers

@timed_fn("load input")
def _load(input_path):
    from src.closest_point_field import SurfaceField, load_field
    if isinstance(input_path, SurfaceField):
        return input_path, "<field>"
    return load_field(str(input_path)), str(input_path)


@timed_fn("tau grid")
def _tau_spec(field, cfg: ResidualConfig):
    """(BoxGridSpec at the wall-licensed pitch over the part, info dict) - see ResidualConfig.grid."""
    from src import local_grids
    from reco_engine.cache import wall_thickness_cached
    tau = wall_thickness_cached(field.vertices, field.faces)          # once per mesh, also across runs
    licensed = local_grids.adaptive_pitch(tau, fraction=cfg.tau_fraction)
    lo, hi = np.asarray(field.bounds[0], float), np.asarray(field.bounds[1], float)
    cube = _spec_for(field, cfg)
    cube_pitch = float(cube.hi[0] - cube.lo[0]) / cfg.resolution
    pitch = min(licensed, cube_pitch)                    # the more detailed of the two
    while np.prod(np.ceil((hi - lo) / pitch) + 4) > cfg.max_cells:
        pitch *= 1.25
    finite = tau[np.isfinite(tau) & (tau > 0)]
    info = dict(tau_p5=float(np.percentile(finite, 5)), tau=tau, licensed_pitch=float(licensed), cube_pitch=cube_pitch,
                pitch=float(pitch), by="tau" if licensed <= cube_pitch else "cube",
                capped=bool(pitch > min(licensed, cube_pitch) * 1.0001))
    return cascade.box_grid_spec(lo, hi, pitch), info


def _spec_for(field, cfg: ResidualConfig):
    lo, hi = np.asarray(field.bounds[0], float), np.asarray(field.bounds[1], float)
    pad = cfg.pad_fraction * float((hi - lo).max())
    return cascade.grid_spec(lo - pad, hi + pad, cfg.resolution)


@timed_fn("parity voxels")
def parity_occupancy(field, spec) -> np.ndarray:
    """Inside/outside of every voxel centre by scanline ray parity along x (what a voxelizer does): each triangle is hit by the
    grid rows (y_j, z_k) inside its yz projection, and a centre is inside when an odd number of hits lie at smaller x. The
    rows are shifted by a tiny irrational fraction of a voxel so no row runs exactly through a mesh edge or vertex (CAD parts
    put vertices on round coordinates, as the grid does). Pure numpy; needs a closed mesh."""
    from src.field_channels import parity_axes
    c = cascade.grid_centres(spec)
    pitch = float(c[1, 0, 0, 0] - c[0, 0, 0, 0])
    xs, ys, zs = c[:, 0, 0, 0], c[0, :, 0, 1] + 1.2345e-4 * pitch, c[0, 0, :, 2] + 2.3456e-4 * pitch
    return parity_axes(field.vertices, field.faces, xs, ys, zs)          # any box of cells, not only a cube


@timed_fn("field arrows")
def field_arrows(t_in: np.ndarray, centres: np.ndarray, pitch: float, band: float = 3.0, max_n: int = 4000,
                 seed: int = 0) -> Dict[str, np.ndarray]:
    """A light sample of the input's closest-point field for display: voxel centres within `band` voxels of the surface
    (|sdf| <= band * pitch), at most `max_n` of them (seeded), each with its vector V = c(x) - x to the nearest surface
    point and the signed distance (positive inside). World mm, the grid's own centres - what the GUI draws as arrows."""
    sdf = t_in[1]
    near = np.abs(sdf) <= band * pitch
    idx = np.flatnonzero(near)
    if len(idx) > max_n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
    pts = centres.reshape(-1, 3)[idx]
    vec = np.stack([t_in[2].ravel()[idx], t_in[3].ravel()[idx], t_in[4].ravel()[idx]], axis=1)
    return dict(points=pts.astype(np.float32), vectors=vec.astype(np.float32), sdf=sdf.ravel()[idx].astype(np.float32))


@contextmanager
def _field_backend(name: Optional[str]):
    """$SHAPE_RECO_FIELD_BACKEND = `name` for the duration (None: unchanged); the field reads it at each query."""
    if not name:
        yield
        return
    old = os.environ.get("SHAPE_RECO_FIELD_BACKEND")
    os.environ["SHAPE_RECO_FIELD_BACKEND"] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SHAPE_RECO_FIELD_BACKEND", None)
        else:
            os.environ["SHAPE_RECO_FIELD_BACKEND"] = old


def _channels(field, spec, exact, backend: Optional[str] = None):
    """mesh_channels, with the sign made trustworthy. The one-pass query (exact=False) can pick a wrong nearest face far
    from the surface, and the sign comes from that face's normal: measured on cy_box_bore (48^3), 9646 voxels OUTSIDE the
    part were marked inside (0 with exact=True, but that costs 59 s instead of 1.5 s; trimesh `contains` 17 s). So the sign
    is taken from scanline parity (`parity_occupancy`), the magnitude and V from the fast query."""
    from src.field_channels import mesh_channels
    inside = parity_occupancy(field, spec)                              # once; mesh_channels used to compute its own too
    with timed("field channels"), _field_backend(backend):
        ch = mesh_channels(field, spec, exact=exact, inside=inside)
    if exact or not field.is_closed():                                  # an open mesh already got ray-parity signs
        return ch
    ch[1] = np.where(inside, 1.0, -1.0) * np.abs(ch[1])
    ch[0] = inside
    return ch


@timed_fn("voxel channels")
def _voxel_channels(occ, spec):
    from src.field_channels import voxel_channels
    return voxel_channels(occ, spec)


@timed_fn("components")
def _components(mask: np.ndarray, min_voxels: int) -> List[np.ndarray]:
    from scipy import ndimage
    labels, n = ndimage.label(mask)
    if n == 0:
        return []
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    order = np.argsort(-sizes)
    return [labels == (i + 1) for i in order if sizes[i] >= min_voxels]


def _bbox(mask: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    pts = np.argwhere(mask)
    return tuple(int(v) for v in pts.min(axis=0)), tuple(int(v) for v in pts.max(axis=0))


def _signature(op: str, mask: np.ndarray):
    lo, hi = _bbox(mask)
    return op, lo, hi, int(mask.sum()) // 4                     # tolerant to a voxel or two of change elsewhere


def _snapshot(result: ReconstructionResult) -> ReconstructionResult:
    return copy.deepcopy(result)


def _cuts_first(cfg: ResidualConfig) -> bool:
    return cfg.cuts_first if cfg.cuts_first is not None else cfg.start_shape != "legacy"


def _op_guard(cfg: ResidualConfig) -> bool:
    if cfg.op_guard is not None:
        return bool(cfg.op_guard)
    return os.environ.get("SHAPE_RECO_OP_GUARD", "").strip().lower() in ("1", "on", "true", "yes")


class _State:
    """Input and model on the grid, with the banded residual."""

    def __init__(self, t_in: np.ndarray, spec):
        self.spec = spec
        self.centres = cascade.grid_centres(spec)
        self.pitch = float(self.centres[1, 0, 0, 0] - self.centres[0, 0, 0, 0])
        self.band = 0.5 * self.pitch
        self.t_in = t_in                                        # the input channels (the bounding covers for "intersect")
        self.covers = None                                      # [(name, Primitive)] computed on the first cut component
        self.tools: List[Primitive] = []                        # accepted operations, reusable as tools (boolean_tools)
        self.occ_in = t_in[0] > 0.5
        # float32 on a box grid (20.7M cells on the frame: 83 MB per SDF instead of 166; a candidate step holds several):
        # coordinates <= 1e2 resolve to ~1e-5, far below the half-voxel band the loop tests against. Cubes: float64.
        self.dtype = np.float32 if isinstance(spec, cascade.BoxGridSpec) else np.float64
        self.s_in = t_in[1].astype(self.dtype)                  # positive inside
        from scipy import ndimage
        self.core_in = ndimage.binary_erosion(self.occ_in)      # voxel-only interior: the strict void check's "material"
        self.s_model = np.full(self.occ_in.shape, -np.inf, self.dtype)   # positive inside; nothing built yet
        # The cut history: how often each voxel has been emptied by a cut. Every accepted cut adds 1 wherever its cutter
        # covers the grid - also where an earlier cut had already removed the material - so a count of 2 or more marks
        # area cut again (overlapping or redundant cutters). An add does not clear it: it is a history, not a state.
        self.cut_history = np.zeros(self.occ_in.shape, np.int16)
        self._part_diagonal: Optional[float] = None

    @property
    def part_diagonal(self) -> float:
        """The diagonal of the input's bounding box on the grid (its inside voxels, whole cells): the torus guard's
        yardstick (reco_engine/torus_guard.py)."""
        if self._part_diagonal is None:
            idx = np.argwhere(self.occ_in)
            if len(idx):
                self._part_diagonal = float(np.linalg.norm((idx.max(axis=0) - idx.min(axis=0) + 1) * self.pitch))
            else:
                self._part_diagonal = float(np.linalg.norm(np.array(self.occ_in.shape) * self.pitch))
        return self._part_diagonal

    @timed_fn("commit")
    def commit(self, s_new: np.ndarray, s_prim: np.ndarray, op: str) -> int:
        """Accept an operation: the new model SDF, and for a cut or an intersect the history update (an intersect empties
        the model where it lies outside the primitive). Returns the voxels emptied that were already cut before."""
        if op == "intersect":
            covered = (s_prim < 0) & (self.s_model >= 0)
        elif op == "cut":
            covered = s_prim >= 0
        else:
            self.s_model = s_new
            return 0
        self.s_model = s_new
        recut = int(np.count_nonzero(covered & (self.cut_history > 0)))
        self.cut_history[covered] += 1
        return recut

    WINDOW_CELLS = 4                                             # box grids: the SDF is exact this far around a primitive

    @timed_fn("primitive sdf")
    def prim_sdf(self, p: Primitive) -> np.ndarray:
        """The primitive's SDF on the grid, positive inside. On a box grid (the wall-thickness grid, 20.7M cells on the
        frame) only inside the primitive's bounding box plus WINDOW_CELLS cells; everywhere else -(WINDOW_CELLS - 1)
        pitches ("clearly outside"). Every test the loop makes on these values - the residual at +-band (half a voxel),
        the model at >= 0, a cutter's cover at >= 0 - gives the same answer as with the exact SDF (the true distance
        out there is larger), and for a cut the model is clamped at +3 pitches, still above those thresholds. The cube
        path is unchanged."""
        sp_p = primitive_to_sp(p)
        if not isinstance(self.spec, cascade.BoxGridSpec):
            return -sp_p.sdf(self.centres)
        w = self.WINDOW_CELLS
        lo_w, hi_w = (np.asarray(b, float) for b in sp_p.bounds())
        c0 = self.centres[0, 0, 0]
        shape = np.array(self.occ_in.shape)
        i0 = np.clip(np.floor((lo_w - c0) / self.pitch).astype(int) - w, 0, shape)
        i1 = np.clip(np.ceil((hi_w - c0) / self.pitch).astype(int) + w + 1, 0, shape)
        out = np.full(self.occ_in.shape, -(w - 1) * self.pitch, self.dtype)
        if np.all(i1 > i0):
            win = (slice(i0[0], i1[0]), slice(i0[1], i1[1]), slice(i0[2], i1[2]))
            out[win] = -sp_p.sdf(self.centres[win])
        return out

    @staticmethod
    def apply(s_model: np.ndarray, s_prim: np.ndarray, op: str) -> np.ndarray:
        if op == "add":
            return np.maximum(s_model, s_prim)
        if op == "intersect":                                   # model & P: inside both (positive inside)
            return np.minimum(s_model, s_prim)
        return np.minimum(s_model, -s_prim)

    @timed_fn("residual")
    def residual(self, s_model: Optional[np.ndarray] = None):
        s_m = self.s_model if s_model is None else s_model
        pos = (self.s_in > self.band) & (s_m < -self.band)
        neg = (s_m > self.band) & (self.s_in < -self.band)
        return pos, neg

    def size(self, s_model: Optional[np.ndarray] = None) -> int:
        if s_model is None or s_model is self.s_model:           # the current model's residual: counted once per model
            ref = getattr(self, "_size_ref", None)                # (every candidate's evaluation asks for it; the model
            if ref is self.s_model:                               # array is replaced, never changed in place - 30 M
                return self._size_val                             # cells on the ABC part of 2026-10-29 12:45)
            pos, neg = self.residual()
            self._size_ref, self._size_val = self.s_model, int(pos.sum() + neg.sum())
            return self._size_val
        pos, neg = self.residual(s_model)
        return int(pos.sum() + neg.sum())

    @property
    def occ_model(self) -> np.ndarray:
        return self.s_model >= 0

    def dirlog_residual(self, s_model: Optional[np.ndarray] = None) -> np.ndarray:
        """(7, R, R, R) network view of this step (field_channels.dirlog_residual): input flag, the model's material,
        unit normal and log distance, and the residual sign (+1 add, -1 cut)."""
        from src.field_channels import dirlog_residual
        return dirlog_residual(self.s_model if s_model is None else s_model, self.occ_in, self.spec)


# ------------------------------------------------------------------------------------------------ candidate evaluation

WINDOW_MARGIN = 8                  # cells around a component that its cascade / void fit sees on a box grid


def _window(state: "_State", mask: np.ndarray, margin: int = WINDOW_MARGIN):
    """(spec, slices) of the sub-grid a component is fitted on. A box grid (the wall-thickness grid): the component's
    bounding box plus `margin` cells - its channels, the cascade's IoUs and the void fit then cost the window, not the
    20.7M cells of the frame (the whole-grid voxel channels of one component took the run past 5 GB). The answer is a
    primitive in world coordinates, so the window does not change where it lies. A cube grid: the whole grid, unchanged."""
    full = (slice(None),) * 3
    if not isinstance(state.spec, cascade.BoxGridSpec) or not mask.any():
        return state.spec, full
    pts = np.argwhere(mask)
    shape = np.array(mask.shape)
    i0 = np.maximum(pts.min(axis=0) - margin, 0)
    i1 = np.minimum(pts.max(axis=0) + margin + 1, shape)
    sub = cascade.BoxGridSpec(tuple(float(state.spec.lo[d] + i0[d] * state.spec.pitch) for d in range(3)),
                              tuple(int(n) for n in i1 - i0), state.spec.pitch)
    return sub, tuple(slice(int(a), int(b)) for a, b in zip(i0, i1))


def _n_pieces(mask: np.ndarray) -> int:
    """Connected pieces (26-connectivity) of a boolean grid."""
    from scipy import ndimage
    return int(ndimage.label(mask, structure=np.ones((3, 3, 3), bool))[1])


def _severs(state: "_State", s_new: np.ndarray, before: Optional[np.ndarray] = None) -> bool:
    """D5: whether the model `s_new` (an SDF, positive inside) is in more connected pieces than the input (the
    input's own count is cached on the state); `before`: compared with that model instead when given."""
    if not hasattr(state, "input_pieces"):
        state.input_pieces = max(1, _n_pieces(state.occ_in))
    allowed = state.input_pieces
    if before is not None:
        allowed = max(allowed, _n_pieces(before >= 0))
    return _n_pieces(s_new >= 0) > allowed


_FREECADCMD_FOUND: Optional[bool] = None


def _freecadcmd_available() -> bool:
    """Whether freecadcmd can be run ($SHAPE_RECO_FREECADCMD or `freecadcmd` on the PATH); looked up once."""
    global _FREECADCMD_FOUND
    if _FREECADCMD_FOUND is None:
        import shutil
        exe = os.environ.get("SHAPE_RECO_FREECADCMD") or "freecadcmd"
        _FREECADCMD_FOUND = bool(shutil.which(exe) or (os.path.sep in exe and os.access(exe, os.X_OK)))
    return _FREECADCMD_FOUND


def _exact_gate_on(cfg: "ResidualConfig") -> bool:
    """The exact solid gate runs when it is wanted and a build exists: an injected one, or freecadcmd itself (the
    final-grade flag cfg.freecad is not the condition - run_part.py and the GUI build the final model themselves)."""
    return bool(cfg.exact_solids_gate and (cfg.feature_build is not None or _freecadcmd_available()))


@timed_fn("exact solid gate")
def _exact_severs(state: "_State", result: ReconstructionResult, cfg: "ResidualConfig", prim: Primitive, op: str,
                  name: str = "gate") -> Optional[str]:
    """D5 exact: the program's accepted operations plus `op prim` built in FreeCAD (cfg.feature_build, else the real
    build); the reason when the build has more solids than allowed (the input's pieces, or the model's own exact count
    when that is already higher), else None - also None when the gate is off, the program is too long, or the build
    fails / times out (the grid's answer stands). Every operation is judged: a cut that severs, and an add whose
    material floats or only touches the model (the box children of 2026-10-16: 54 solids on one gate part when only
    the cuts were gated). The checks, refusals and seconds are counted on state.exact_gate."""
    if not _exact_gate_on(cfg):
        return None
    ops = [o for o in result.operations if o.accepted]
    if not ops or len(ops) >= cfg.exact_solids_max_ops:
        return None
    acc = state.__dict__.setdefault("exact_gate", dict(checks=0, rejected=0, failed=0, seconds=0.0))
    out = getattr(state, "exact_gate_dir", None)
    if out is None:
        out = state.exact_gate_dir = Path(cfg.out_dir or tempfile.mkdtemp(prefix="add_cut_")) / "solids_gate"
        out.mkdir(parents=True, exist_ok=True)
    tmp = ReconstructionResult(input_path=result.input_path, operations=ops + [Operation(op, prim)])
    build = cfg.feature_build or _feature_build_default
    t = time.time()
    acc["checks"] += 1
    try:
        b = build(tmp, out, f"{name}_{acc['checks']:03d}", cfg.exact_solids_timeout)
        n = int(b.get("n_solids") or 0)
    except Exception as e:                                           # a failed or timed-out build is no answer
        acc["failed"] += 1
        acc["seconds"] += time.time() - t
        acc["last_error"] = str(e)[:200]
        return None
    acc["seconds"] += time.time() - t
    if n <= 0:
        acc["failed"] += 1
        return None
    allowed = max(1, int(getattr(state, "input_pieces", 1)), int(getattr(state, "exact_solids", 1)))
    if n > allowed:
        acc["rejected"] += 1
        return f"severs the model (D5 exact: {n} solids, {allowed} allowed)"
    # the build answers more than the count: a solid OCCT calls invalid, or one with a non-positive volume (an inverted
    # shell: the gate corpus of 10-16 had a tree model of volume -178634 at voxel IoU 0.793 and exact IoU 0.002, and
    # 2 of 60 tree_l / 4 of 60 tree_c final models invalid, bare 0), is no model either
    vol = b.get("volume")
    if "valid" in b and not b.get("valid"):
        acc["rejected"] += 1
        return "makes the solid invalid (D5 exact)"
    if vol is not None and float(vol) <= 0:
        acc["rejected"] += 1
        return f"gives a non-positive volume (D5 exact: {float(vol):.0f})"
    return None


def _unsevered_best(state: "_State", result: ReconstructionResult, cfg: "ResidualConfig", chosen: Dict, op: str) -> Optional[Tuple]:
    """The loop's chosen candidate after the exact solid gate: `chosen["best"]` when its exact build keeps the solid
    count, else the next passing candidate by gain that does (at most cfg.exact_solids_max_checks builds; when the
    budget is used the grid's answer stands). A refused candidate's evaluation says why. None: every one severs."""
    best = chosen.get("best")
    if best is None or not _exact_gate_on(cfg):
        return best
    rest = sorted([(p, e) for p, e in chosen["tried"] if e["ok"] and e.get("s_new") is not None and p is not best[0]],
                  key=lambda t: -t[1]["gain"])
    checks = 0
    for p, e in [best] + rest:
        if checks >= cfg.exact_solids_max_checks:
            return (p, e)
        checks += 1
        why = _exact_severs(state, result, cfg, p, e.get("op", op), name="loop")
        if why:
            e["ok"], e["reason"] = False, why
            continue
        return (p, e)
    chosen["why"] = "every passing candidate severs the model (D5 exact)"
    return None


@timed_fn("evaluate candidate")
def _evaluate(state: _State, prim: Primitive, op: str, purity_min: float, connectivity: Optional[bool] = None) -> Dict:
    """Net gain and, for a cut, the two material checks (voxels: void_quality strict; sign: input SDF). A primitive
    that holds a torus the torus guard rejects (reco_engine/torus_guard.py) fails before anything is measured."""
    from reco_engine.void_quality import assess_void
    from reco_engine.torus_guard import primitive_defect, tori
    bad = prim.defect()                                  # Primitive.defect: nothing degenerate reaches an SDF or FreeCAD
    if bad is not None:
        return dict(gain=0, residual_after=state.size(), ok=False, reason=f"invalid primitive: {bad}",
                    s_new=state.s_model, s_prim=None, guard="defect")
    found = primitive_defect(prim, state.part_diagonal) if tori(prim) else None
    if found is not None:
        return dict(gain=0, residual_after=state.size(), ok=False, reason=f"torus guard: {found.text}",
                    s_new=state.s_model, s_prim=None, guard=found.rule)
    if op == "add" and isinstance(state.spec, cascade.BoxGridSpec):
        # the box grid hugs the part (2 cells of margin): material an add puts outside it is never counted as excess, so
        # an add must stay inside the part's box. Measured on the frame before this check: two plate boxes fitted to 40
        # surface points came out 127 and 62 STL units long on a part 10 wide, and were accepted (chamfer 1.55, max 58.6).
        lo_p, hi_p = (np.asarray(b, float) for b in primitive_to_sp(prim).bounds())
        tol = state.pitch
        if np.any(lo_p < np.asarray(state.spec.lo) - tol) or np.any(hi_p > np.asarray(state.spec.hi) + tol):
            return dict(gain=0, residual_after=state.size(), ok=False, reason="reaches outside the part's box",
                        s_new=state.s_model, s_prim=None)
    s_c = state.prim_sdf(prim)
    new = state.apply(state.s_model, s_c, op)
    before, after = state.size(), state.size(new)
    out = dict(gain=before - after, residual_after=after, ok=True, reason="", s_new=new, s_prim=s_c)
    if op in ("cut", "intersect") and not (new >= 0).any():
        out.update(ok=False, reason=f"{op} would remove the whole model")
        return out
    if op in ("cut", "intersect"):
        cand = s_c >= 0 if op == "cut" else s_c < 0             # what it removes: the cutter / everything outside P
        occ_model = state.occ_model
        a = assess_void(cand, occ_model, state.core_in, purity_min=purity_min)
        removed_sign = int(np.count_nonzero(cand & occ_model & (state.s_in > state.band)))
        out.update(verdict=a.verdict, purity=float(a.purity), removed_material_voxels=int(a.removed_material),
                   removed_material_sign=removed_sign)
        if a.verdict != "good":
            out.update(ok=False, reason=f"void check: {a.verdict} (purity {a.purity:.3f})")
        elif removed_sign > max(2, int(0.01 * a.removes)):
            out.update(ok=False, reason=f"sign check: removes {removed_sign} voxels of input material")
        elif (connectivity if connectivity is not None else
              (getattr(state, "connectivity_gate", True) and new.size <= getattr(state, "connectivity_max_cells", 4_000_000))) \
                and _severs(state, new):
            out.update(ok=False, reason="severs the model (D5: more pieces than the input)")
    return out


def _moved_box(p: Primitive, axis: int, sign: float, delta: float) -> Primitive:
    R = rotation_matrix(p)
    key = ("x_half", "y_half", "z_half")[axis]
    dims = dict(p.dims_mm)
    dims[key] = dims[key] + delta / 2.0
    c = np.asarray(p.center_mm, float) + R[:, axis] * sign * delta / 2.0
    return _replace(p, c, dims)


def _replace(p: Primitive, centre, dims) -> Primitive:
    q = copy.copy(p)
    q.center_mm, q.dims_mm, q.alternatives = tuple(float(x) for x in centre), dims, []
    return q


def _moved_cylinder_cap(p: Primitive, sign: float, delta: float) -> Primitive:
    R = rotation_matrix(p)
    dims = dict(p.dims_mm)
    dims["half_length"] = dims["half_length"] + delta / 2.0
    return _replace(p, np.asarray(p.center_mm, float) + R[:, 2] * sign * delta / 2.0, dims)


def _grow_cut_box(state: _State, prim: Primitive, open_only: bool = False) -> Optional[Primitive]:
    """A world-axis box cut with every face moved outwards as far as the input lets it (ResidualConfig.cut_grow): layer
    by layer along each axis, the layer being the cells the box's current cross-section overlaps; a layer with a cell
    within half a voxel of input material stops the face before it; a face that reaches the grid's last layer is open
    and ends three cells beyond it. `open_only`: only such open faces move (ResidualConfig.cut_open_faces). None when the
    box is not world-axis aligned or no face moved."""
    if prim.kind != "box":
        return None
    R = rotation_matrix(prim)
    P = np.round(R)
    if not (np.allclose(np.abs(P).sum(axis=0), 1) and np.allclose(np.abs(P).sum(axis=1), 1) and np.max(np.abs(R - P)) < 1e-6):
        return None
    half_local = np.array([prim.dims_mm["x_half"], prim.dims_mm["y_half"], prim.dims_mm["z_half"]], float)
    hw = np.abs(P) @ half_local
    c = np.asarray(prim.center_mm, float)
    lo, hi = c - hw, c + hw
    c0, pitch = np.asarray(state.centres[0, 0, 0], float), float(state.pitch)
    n = np.array(state.occ_in.shape)
    solid = getattr(state, "_solid_cells", None)
    if solid is None:
        solid = state._solid_cells = np.asarray(state.s_in > -state.band)
    moved = False

    def span(a):                                       # the cells the box overlaps along axis a (touching is not)
        i0 = int(np.floor((lo[a] - 0.5 * pitch - c0[a]) / pitch + 1e-9)) + 1
        i1 = int(np.ceil((hi[a] + 0.5 * pitch - c0[a]) / pitch - 1e-9)) - 1
        return max(0, i0), min(int(n[a]) - 1, i1)
    for a in range(3):
        for side in (1, -1):
            sl = [slice(*(lambda r: (r[0], r[1] + 1))(span(k))) for k in range(3)]
            if any(x.start >= x.stop for x in sl):
                continue
            j = int(np.floor((hi[a] - c0[a]) / pitch)) + 1 if side > 0 else int(np.ceil((lo[a] - c0[a]) / pitch)) - 1
            steps = range(max(j, 0), int(n[a])) if side > 0 else range(min(j, int(n[a]) - 1), -1, -1)
            stop = None
            for k in steps:
                sl[a] = slice(k, k + 1)
                if solid[tuple(sl)].any():
                    stop = k
                    break
            if open_only and stop is not None:                # the face sees material: it stays
                continue
            if side > 0:
                new = (c0[a] + (stop - 0.5) * pitch) if stop is not None else (c0[a] + (n[a] - 0.5 + 3) * pitch)
                if new > hi[a] + 1e-9:
                    hi[a], moved = new, True
            else:
                new = (c0[a] + (stop + 0.5) * pitch) if stop is not None else (c0[a] - (0.5 + 3) * pitch)
                if new < lo[a] - 1e-9:
                    lo[a], moved = new, True
    if not moved:
        return None
    q = copy.copy(prim)
    q.center_mm = tuple(float(x) for x in (lo + hi) / 2)
    hl = np.abs(P).T @ ((hi - lo) / 2)
    q.dims_mm = dict(x_half=float(hl[0]), y_half=float(hl[1]), z_half=float(hl[2]))
    q.alternatives = []
    return q


@timed_fn("extend open faces")
def _extend_open_faces(state: _State, prim: Primitive, op: str, purity_min: float, base: Dict) -> Tuple[Primitive, Dict]:
    """Push every box face / cylinder cap that can move without costing anything: outwards by three voxels for a cut (a cutter
    ending exactly on the part's face leaves a skin or a coplanar-face boolean in FreeCAD), one voxel into the model for an add
    (the same at the joint). A move is kept when the gain does not drop and the checks still pass - so a face that bounds real
    input geometry (a pocket floor, the top of a stacked block) stays where the fit put it."""
    if prim.kind not in ("box", "cylinder"):
        return prim, base
    if op == "cut" and prim.kind == "box" and (getattr(state, "cut_grow", False) or getattr(state, "cut_open_faces", False)):
        grown = _grow_cut_box(state, prim, open_only=not getattr(state, "cut_grow", False))
        if grown is not None:
            snapped = _snap_box_faces(state, grown, None)               # faces that grew to the surface onto it
            for trial in ([snapped] if snapped is not None else []) + [grown]:
                ev = _evaluate(state, trial, op, purity_min)
                if ev["ok"] and ev["gain"] >= base["gain"]:
                    trial.label = (prim.label or prim.kind) + (", grown to the open space" if getattr(state, "cut_grow", False)
                                                               else ", open faces past the part")
                    return trial, ev
    delta = (3.0 if op == "cut" else 1.0) * state.pitch
    best, best_eval = prim, base
    moves = [(a, s) for a in range(3) for s in (1.0, -1.0)] if prim.kind == "box" else [(2, 1.0), (2, -1.0)]
    for axis, sign in moves:
        trial = _moved_box(best, axis, sign, delta) if prim.kind == "box" else _moved_cylinder_cap(best, sign, delta)
        ev = _evaluate(state, trial, op, purity_min)
        if ev["ok"] and ev["gain"] >= best_eval["gain"]:
            if op == "add":                                         # only into the model: the new slab must be built already
                grown = (state.prim_sdf(trial) >= 0) & ~(state.prim_sdf(best) >= 0)
                if np.count_nonzero(grown & ~state.occ_model) > 0.1 * max(1, np.count_nonzero(grown)):
                    continue
            best, best_eval = trial, ev
    return best, best_eval


@timed_fn("void fit")
def _void_fit_primitive(state: _State, region: np.ndarray, purity_min: float) -> Optional[Primitive]:
    """void_fit's answer as a schema Primitive. void_fit reports its orientation as a world quaternion of the [z, -y, x] array
    convention (rotation_symmetry); our arrays ARE world axes, so the array-frame orientation M it fitted is recovered as
    A R_w A^T and used directly."""
    from reco_engine.void_fit import fit_void
    from scipy.spatial.transform import Rotation
    from training.rotation_symmetry import A_WORLD_TO_ARRAY as A
    sub, sl = _window(state, region)
    vf = fit_void(state.occ_in[sl].astype(np.uint8), state.occ_model[sl].astype(np.uint8), region[sl].astype(np.uint8),
                  voxel_mm=state.pitch, min_gain=3)
    if vf is None:
        return None
    Rw = Rotation.from_quat(vf.quaternion).as_matrix()
    M = A @ Rw @ A.T                                                # columns: the fitted axes in OUR array (= world) frame
    centre = cascade.to_world(sub, vf.center_vox)
    scale = 10.0                                                    # void_fit ratios: mm = ratio * 10 with voxel_mm = pitch
    src = [StageSource(s, "analytic:void_fit", 1.0) for s in ("type", "rotation", "dimensions", "position")]
    if vf.topology == "box":
        P = vf.parameters
        return Primitive(kind="box", quaternion=quaternion_of(M), center_mm=tuple(float(x) for x in centre),
                         dims_mm=dict(x_half=P["x_half_ratio"] * scale, y_half=P["y_half_ratio"] * scale,
                                      z_half=P["z_half_ratio"] * scale), confidence=1.0, sources=src)
    if vf.topology == "sphere":
        return Primitive(kind="sphere", center_mm=tuple(float(x) for x in centre),
                         dims_mm=dict(radius=vf.parameters["radius_ratio"] * scale), confidence=1.0, sources=src)
    # cylinder: axis_quaternion turned local +z onto the axis
    axis = M[:, 2]
    from reco_engine.cylinder_fit import _frame_with_axis
    F = _frame_with_axis(axis)
    R = np.stack([F[:, 1], F[:, 2], F[:, 0]], axis=1)
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in centre),
                     dims_mm=dict(radius=vf.parameters["radius_ratio"] * scale,
                                  half_length=vf.parameters["half_length_ratio"] * scale), confidence=1.0, sources=src)


# ------------------------------------------------------------------------------------------------ voxel-cube fallback

def voxel_boxes(mask: np.ndarray, max_boxes: int = 16, min_voxels: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Cover `mask` greedily with its largest completely filled axis-aligned boxes (loop_guards.largest_box_in_blocks,
    exact), biggest first: [(index_lo, index_hi inclusive)]. Stops at `max_boxes` or when what is left is smaller than
    `min_voxels` - few boxes for a blob, not one cube per voxel."""
    from reco_engine.loop_guards import largest_box_in_blocks
    rest = np.asarray(mask, bool).copy()
    boxes = []
    while len(boxes) < max_boxes and rest.sum() >= min_voxels:
        pts = np.argwhere(rest)
        lo, hi = pts.min(axis=0), pts.max(axis=0) + 1
        crop = rest[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        vol, b = largest_box_in_blocks(crop)
        if vol < min_voxels or b is None:
            break
        blo = lo + np.array([b[0], b[2], b[4]])
        bhi = lo + np.array([b[1], b[3], b[5]])
        boxes.append((blo, bhi))
        rest[blo[0]:bhi[0] + 1, blo[1]:bhi[1] + 1, blo[2]:bhi[2] + 1] = False
    return boxes


@timed_fn("largest boxes")
def largest_boxes(mask: np.ndarray, count: int = 3, max_side: int = 48, min_voxels: int = 6
                  ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The `count` largest completely filled axis-aligned boxes of `mask`, one after the other with the previous ones
    taken out: [(index_lo, index_hi inclusive)]. Bounded cost for any size: found on blocks of f^3 voxels (a block
    counts when all of it is in the mask; at most `max_side` blocks per side), then each face grown one voxel layer at a
    time while the new layer is completely in the mask. The frame's excess after a box start is its windows and the
    recesses between the I-beam flanges - each an exact box cut."""
    from reco_engine.loop_guards import largest_box_in_blocks
    rest = np.asarray(mask, bool).copy()
    out = []
    while len(out) < count and rest.sum() >= min_voxels:
        pts = np.argwhere(rest)
        lo0, hi0 = pts.min(axis=0), pts.max(axis=0) + 1
        crop = rest[lo0[0]:hi0[0], lo0[1]:hi0[1], lo0[2]:hi0[2]]
        f = max(1, int(np.ceil(max(crop.shape) / max_side)))
        n = [int(np.ceil(s / f)) for s in crop.shape]
        pad = np.zeros([k * f for k in n], bool)
        pad[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
        blocks = pad.reshape(n[0], f, n[1], f, n[2], f).all(axis=(1, 3, 5))
        order = np.argsort(blocks.shape)                           # the shortest axis first: its ranges are the loop
        vol, b = largest_box_in_blocks(np.transpose(blocks, order))
        if not vol:
            break
        blo_t, bhi_t = np.array([b[0], b[2], b[4]]), np.array([b[1], b[3], b[5]])
        blo, bhi = np.empty(3, int), np.empty(3, int)
        blo[order], bhi[order] = blo_t, bhi_t
        lo, hi = lo0 + blo * f, lo0 + bhi * f + f - 1
        grew = True
        while grew:                                                # grow at full resolution, face by face
            grew = False
            for ax in range(3):
                for side in (-1, 1):
                    edge = (lo[ax] - 1) if side < 0 else (hi[ax] + 1)
                    if edge < 0 or edge >= rest.shape[ax]:
                        continue
                    sl = [slice(lo[k], hi[k] + 1) for k in range(3)]
                    sl[ax] = slice(edge, edge + 1)
                    if rest[tuple(sl)].all():
                        lo[ax], hi[ax] = (edge, hi[ax]) if side < 0 else (lo[ax], edge)
                        grew = True
        if int(np.prod(hi - lo + 1)) < min_voxels:
            break
        out.append((lo.copy(), hi.copy()))
        rest[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = False
    return out


def _snap_box_faces(state: "_State", b: Primitive, field, min_samples: int = 20) -> Optional[Primitive]:
    """An axis-aligned box whose faces lie on voxel boundaries (largest_boxes, the voxel fallback) moved onto the input's
    surface: each face goes to the median of the input surface samples within one pitch of it whose normal is the face's
    and which lie over the face. None when no face had enough samples. The largest-box cuts of bs_box_step gave chamfer
    0.018 with 0 % surface share before this; the fitted box 0.0002 (2026-09-27)."""
    if b.kind != "box" or np.linalg.norm(np.asarray(b.quaternion, float) - np.array([0, 0, 0, 1.0])) > 1e-6:
        return None
    if field is None and getattr(state, "_samples", None) is None:     # the samples of an earlier snap serve too
        return None
    if not hasattr(state, "_samples"):
        try:
            state._samples = cascade._field_samples(field)
        except Exception:
            state._samples = None
    if state._samples is None:
        return None
    P, N = np.asarray(state._samples.surface, float), np.asarray(state._samples.normals, float)
    c = np.asarray(b.center_mm, float)
    half = np.array([b.dims_mm["x_half"], b.dims_mm["y_half"], b.dims_mm["z_half"]], float)
    lo, hi = c - half, c + half
    pitch = float(state.pitch)
    moved = 0
    for ax in range(3):
        others = [k for k in range(3) if k != ax]
        over = np.all([(P[:, k] >= lo[k] - pitch) & (P[:, k] <= hi[k] + pitch) for k in others], axis=0)
        for side, x in ((0, lo[ax]), (1, hi[ax])):
            sel = over & (np.abs(P[:, ax] - x) <= pitch) & (np.abs(N[:, ax]) > 0.9)
            if int(sel.sum()) >= min_samples:
                x_new = float(np.median(P[sel, ax]))
                if side == 0:
                    lo[ax] = x_new
                else:
                    hi[ax] = x_new
                moved += 1
    if not moved or np.any(hi - lo <= 0.25 * pitch):
        return None
    q = copy.copy(b)
    q.center_mm = tuple(float(v) for v in (lo + hi) / 2.0)
    q.dims_mm = dict(x_half=float((hi - lo)[0] / 2), y_half=float((hi - lo)[1] / 2), z_half=float((hi - lo)[2] / 2))
    q.label = f"a largest box of the piece, {moved} face(s) snapped to the surface"
    return q


def _cube_primitive(state: _State, blo, bhi) -> Primitive:
    lo = cascade.to_world(state.spec, blo) - state.pitch / 2
    hi = cascade.to_world(state.spec, bhi) + state.pitch / 2
    h = (hi - lo) / 2
    return Primitive(kind="box", center_mm=tuple(float(x) for x in (lo + hi) / 2),
                     dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])), confidence=0.0,
                     sources=[StageSource(s, "voxel_fallback", 0.0) for s in ("type", "rotation", "dimensions", "position")])


@timed_fn("fallback box")
def _fallback_box(state: _State, blo, bhi, op: str, field, registry, cfg: ResidualConfig, purity_min: float):
    """One piece of the last resolution: the voxel cube, and the box the cascade fits to the input surface around that same
    piece (a ring-shaped step is no single primitive, but each of its straight pieces is an exact box); the one that gains
    more and passes the checks, its open faces extended. (None, None) when neither gains."""
    piece = np.zeros_like(state.occ_in)
    piece[blo[0]:bhi[0] + 1, blo[1]:bhi[1] + 1, blo[2]:bhi[2] + 1] = True
    cands = [_cube_primitive(state, blo, bhi)]
    if int(piece.sum()) >= cfg.min_component:
        sub, sl = _window(state, piece)
        scope = (~state.occ_model if op == "add" else state.occ_model)[sl]
        fitted = cascade.run_cascade(_voxel_channels(piece[sl], sub), sub, registry, field=field, top_k=1,
                                     models=cfg.models, scope=scope, pass_iou=cfg.component_pass_iou, kinds=("box",))
        if not fitted.is_unknown:
            fitted.alternatives = []
            cands.insert(0, fitted)
    best = (None, None)
    for c in cands:
        ev = _evaluate(state, c, op, purity_min)
        if not ev["ok"] or ev["gain"] <= 0:
            continue
        c, ev = _extend_open_faces(state, c, op, purity_min, ev)
        if best[1] is None or ev["gain"] > best[1]["gain"]:
            best = (c, ev)
    return best


def select_for_target(groups: Sequence[Tuple[float, float]], missing: float, invented: float, v_in: float,
                      target: Optional[float]) -> Tuple[List[int], float]:
    """(the indices of the groups to take, the IoU estimated with them): `groups` [(missing volume, invented volume)
    of a congruence group], the ones with the most volume first until (v_in - missing) / (v_in + invented) reaches
    `target`. None: all of them."""
    def iou(m, i):
        return (v_in - m) / (v_in + i) if v_in + i > 0 else 0.0
    m, i = float(missing), float(invented)
    if target is None:
        for gm, gi in groups:
            m, i = max(0.0, m - gm), max(0.0, i - gi)
        return list(range(len(groups))), iou(m, i)
    keep = []
    for k in sorted(range(len(groups)), key=lambda k: -(groups[k][0] + groups[k][1])):
        if iou(m, i) >= target:
            break
        keep.append(k)
        m, i = max(0.0, m - groups[k][0]), max(0.0, i - groups[k][1])
    return sorted(keep), iou(m, i)


# ------------------------------------------------------------------------------------------------ FreeCAD

def freecad_grade(model, out_dir, name: str, field) -> Dict:
    """Build the program in FreeCAD (reco_field.export.build_in_freecad) and grade it against the input (two-way chamfer)."""
    from reco_field.export import build_in_freecad, export_check
    with timed("freecadcmd build"):
        ex = build_in_freecad(model, out_dir, name)
    with timed("chamfer"):
        ch = export_check(ex, field, n=10000)
    return dict(valid=bool(ex.valid and ex.n_solids == 1), n_solids=int(ex.n_solids), step=str(ex.step), stl=str(ex.stl),
                chamfer=float(ch["mean"]), chamfer_max=float(ch["max"]), volume=float(ex.volume))


@timed_fn("FreeCAD build + grade")
def _grade(result: ReconstructionResult, field, cfg: ResidualConfig, name: str) -> Dict:
    model = program_model(result)
    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(tempfile.mkdtemp(prefix="add_cut_"))
    try:
        with _field_backend(cfg.field_backend):
            g = (cfg.builder or freecad_grade)(model, out_dir, name, field)
    except Exception as exc:                                         # a failed build is a result, not a crash of the loop
        g = dict(valid=False, error=str(exc)[-400:])
    result.chamfer_mm = g.get("chamfer")
    result.step_path, result.stl_path = g.get("step"), g.get("stl")
    accepted = [o for o in result.operations if o.accepted]
    if accepted:
        accepted[-1].freecad_valid = bool(g.get("valid"))
    if cfg.exact_grade and g.get("step") and Path(str(result.input_path)).suffix.lower() in (".step", ".stp", ".stl"):
        g.update(_exact_grade(result, field, g))
    return g


def _exact_grade(result: ReconstructionResult, field, g: Dict, max_faces: int = 12000, max_disagreement: float = 0.05) -> Dict:
    """The exact judge on a built program, measured two independent ways (reco_field.exact_check):

    - OCCT booleans of the built model against the input file (exact IoU, missed / invented volume and pieces), for a
      STEP input and for an STL of at most `max_faces` triangles (a finely tessellated sphere or torus takes minutes
      to become a solid);
    - a sampled IoU: ray parity of the input mesh and the program's CSG signed distance on one grid at the thinnest
      wall's pitch. If the two disagree by more than `max_disagreement` the OCCT booleans failed silently (measured:
      coincident faces made both cuts return the whole shapes, IoU ~ 0 at a 0.0004 chamfer) and the sampled value is
      reported instead, with the error recorded. `exact_method` says which number `exact_iou` is.
    The genus of the result comes from its B-rep tessellation in FreeCAD. Never raises."""
    from reco_field.exact_check import exact_residual, sampled_iou
    out: Dict = {}
    booleans = Path(str(result.input_path)).suffix.lower() in (".step", ".stp") or len(field.faces) <= max_faces
    ex = exact_residual(result.input_path, g["step"], booleans=booleans)
    try:
        sm = sampled_iou(field, program_model(result))
    except Exception as exc:
        sm = dict(error=f"sampled IoU: {exc}")
    occt_ok = booleans and "error" not in ex
    if occt_ok and "error" not in sm and abs(ex["iou"] - sm["iou"]) > max_disagreement:
        out["exact_error"] = f"OCCT IoU {ex['iou']:.4f} disagrees with the sampled {sm['iou']:.4f}: booleans failed"
        occt_ok = False
    elif "error" in ex:
        out["exact_error"] = ex["error"]
    if occt_ok:
        result.exact_iou, result.exact_r_plus, result.exact_r_minus = ex["iou"], ex["r_plus"], ex["r_minus"]
        result.exact_r_plus_solids, result.exact_method = int(ex["r_plus_solids"]), "occt"
    elif "error" not in sm:
        result.exact_iou, result.exact_method = sm["iou"], f"sampled (pitch {sm['pitch']:.3g})"
    out.update(exact_iou=result.exact_iou, exact_method=result.exact_method, exact_r_plus=result.exact_r_plus,
               exact_r_minus=result.exact_r_minus, exact_r_plus_solids=result.exact_r_plus_solids,
               sampled_iou=sm.get("iou"))
    if "model_genus" not in ex:                                     # the booleans failed: the model alone still has a genus
        ex = dict(ex, **{k: v for k, v in exact_residual(result.input_path, g["step"], booleans=False).items()
                         if k == "model_genus"})
    result.genus = ex.get("model_genus")
    try:
        result.input_genus = float(field.genus())
    except Exception:
        result.input_genus = None
    out.update(genus=result.genus, input_genus=result.input_genus)
    return out


# ------------------------------------------------------------------------------------------------ first-add sub-loops

def _env_off(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("0", "off", "false", "no")


def _subloop_plan(cfg: ResidualConfig, hint_subclass: Optional[str] = None) -> Dict:
    """Which first-add sub-loop may run: dict(members=False | None (all kinds) | set of kinds, hollow=bool, forced=bool).
    With a TypeNet subclass hint ("box/ibeam", "*/frame", "*/hollow", "cylinder/tube", "box/rhs") the hint decides and the
    member sub-loop is forced; without one the analytic gates decide (see `_members_worth_it`, `ibeam.find_cavities`)."""
    members_on = cfg.ibeam and not _env_off("SHAPE_RECO_IBEAM")
    hollow_on = cfg.hollow and not _env_off("SHAPE_RECO_HOLLOW")
    hint = (hint_subclass if hint_subclass is not None else cfg.hint_subclass) or ""
    hint = hint.strip().lower()
    if not hint:
        return dict(members=None if members_on else False, hollow=hollow_on, forced=False)
    sub = hint.rpartition("/")[2]
    if sub in MEMBER_SUBCLASSES:
        return dict(members=MEMBER_SUBCLASSES[sub] if members_on else False, hollow=False, forced=True)
    if sub == "hollow":
        return dict(members=False, hollow=hollow_on, forced=True)
    # any other hint ("full", "sector", "band") must not veto the analytic gates: the TypeNet sees the whole part at 64^3
    # and calls a frame of I-beams "box/full"; switching the member sub-loop off on that hint collapsed the frame from 32
    # accepted operations (chamfer 0.033) to 3 (12.9) in gate G3 (docs/data/night_g3, 2026-09-24)
    return dict(members=None if members_on else False, hollow=hollow_on, forced=False)


def _members_worth_it(members) -> bool:
    """The analytic gate of the member sub-loop: an I, tube or rhs member, or at least two rod/bar members (a frame). A
    single rod or bar is one primitive - the cascade's first add already is that (the example parts stay unchanged)."""
    kinds = [m.kind for m in members]
    return any(k in ("ibeam", "tube", "rhs") for k in kinds) or sum(k in ("rod", "bar") for k in kinds) >= 2


@timed_fn("exact eval (mesh)")
def _exact_eval(state: _State, judge, result: ReconstructionResult, prim: Primitive, op: str, cfg: ResidualConfig,
                seed: int = 0, exclude: Optional[Primitive] = None) -> Dict:
    """An operation judged on the grid (the new model SDF, the voxel gain, the whole-model-cut guard) AND on the mesh
    (ibeam.ExactJudge: sampled volume in the input / in the model): walls thinner than the pitch gain nothing on the grid,
    so the exact accounting decides - precision (share of an add inside the input, of a cut's removal outside it) at
    least member_min_precision, and for an add new input material of at least member_min_new of its own volume, for a
    cut an exact gain of at least min_gain voxel volumes. `exclude`: the cutter paired with this add (a tube's bore, an
    rhs's hole), whose volume the add is not judged on. The gain is reported in voxel volumes (volume / pitch^3)."""
    s_c = state.prim_sdf(prim)
    new = state.apply(state.s_model, s_c, op)
    out = dict(grid_gain=state.size() - state.size(new), s_new=new, s_prim=s_c, ok=True, reason="")
    if op == "cut" and not (new >= 0).any():
        out.update(ok=False, reason="cut would remove the whole model", gain=0.0, precision=0.0)
        return out
    model = program_model(result) if any(o.accepted for o in result.operations) else None
    j = judge.judge(prim, op, model, seed=seed, exclude=primitive_to_sp(exclude) if exclude is not None else None)
    gain = float(j["gain_volume"]) / state.pitch ** 3
    out.update(exact=j, gain=gain, precision=float(j["precision"]))
    if j["precision"] < cfg.member_min_precision:
        what = "of it lies in the input" if op == "add" else "of what it removes is empty in the input"
        out.update(ok=False, reason=f"only {j['precision']:.1%} {what} (needs {cfg.member_min_precision:.0%})")
    elif op == "add" and (j["new"] < cfg.member_min_new or gain <= 0):
        out.update(ok=False, reason=f"brings only {j['new']:.1%} of its volume as new material")
    elif op == "cut" and gain < cfg.min_gain:
        out.update(ok=False, reason=f"exact gain {gain:.1f} voxel volumes < {cfg.min_gain}")
    return out


def _commit_member_op(state: _State, result: ReconstructionResult, op: str, prim: Primitive, ev: Dict, voxels: int):
    recut = state.commit(ev["s_new"], ev["s_prim"], op)
    result.operations.append(Operation(op, prim, component_voxels=int(voxels), net_gain=float(ev["gain"]),
                                       recut_voxels=recut))
    return recut


def _member_phase(state: _State, field, result: ReconstructionResult, cfg: ResidualConfig, registry, purity_min: float,
                  kinds, forced: bool = False):
    """The member sub-loop (reco_engine/ibeam.py): every straight member found in the mesh's sections becomes its
    primitives - I-beam: 3 box adds (top flange, bottom flange, web), rod: a cylinder, bar: a box, tube: cylinder add +
    cylinder cut, rhs: box add + box cut - each judged by `_exact_eval`. All member adds come first, then their cuts.
    A member whose section is none of these takes the old path on its region (the cascade on its voxels, else its exact
    bounding box, judged on the grid). Returns the number of accepted operations (0: nothing done, the old first add runs)."""
    from reco_engine import ibeam as ib
    try:
        members = ib.detect_members(field, min_length_ratio=cfg.member_min_length_ratio)
    except Exception as exc:                                          # detection is advisory: the old first add runs
        yield ResidualStep("member", _snapshot(result), dict(members=[], message=f"member detection failed: {exc}"))
        return 0
    if kinds is not None:
        members = [m for m in members if m.kind in kinds or m.kind == "other"]
    if not members or not (forced or _members_worth_it(members)):
        return 0
    judge = ib.ExactJudge(field, n=cfg.member_samples)
    accepted, cuts = 0, []
    n = len(members)
    report = state.member_report = dict(asked=True, detected=n, accepted=0, store=None)
    store = None
    if cfg.member_congruency:                                         # stored, grouped by congruency (member_store.py)
        from reco_engine import member_store as ms
        store = ms.MemberStore(members, tol=cfg.member_congruent_tol)

    def slot(i, m, mask, k, op, own, seed, exclude=None):
        """One operation of a member: the group's source primitive moved onto it first (by congruency), its own
        primitive when there is no source yet or the copy fails. Returns the number accepted (0 or 1)."""
        moved = store.moved(i, k, op) if store is not None else None
        extra = dict(index=i, slot=k) if store is None else dict(index=i, slot=k, group=store.group(i).id,
                                                                 group_size=len(store.group(i).members))
        if moved is not None:
            src = store.source(i, k)
            ev = {}
            ok = yield from _member_op(state, judge, result, cfg, m, mask, op, moved, seed=seed, exclude=exclude,
                                       extra=dict(extra, by_congruency=True, source_member=src), verdict=ev)
            if ok:
                store.accept(i, k, op, moved, True, ev.get("precision", 1.0))
                return 1
            store.failed_copy(i)                                      # a copy that fails is judged in full by itself
        ev = {}
        ok = yield from _member_op(state, judge, result, cfg, m, mask, op, own, seed=seed, exclude=exclude,
                                   extra=dict(extra, by_congruency=False, after_failed_copy=moved is not None), verdict=ev)
        if store is not None:
            if ok:
                store.accept(i, k, op, own, False, ev.get("precision", 1.0))
            else:
                store.reject(i, k)
        return ok

    for i, m in enumerate(members):
        mask = ib.member_prism_mask(m, state.centres)
        sec = m.section
        label = sec.label if sec is not None else m.kind
        ibs = sec.ibeam if sec is not None else None
        info = dict(member=m, section=sec, kind=m.kind, axis=tuple(float(x) for x in m.axis), extent=(m.start, m.end),
                    length=m.length, profile=label, index=i, members=n, mask=mask, voxels=int(mask.sum()),
                    message=f"member {i + 1}/{n}: {m.kind} {label}, length {m.length:.4g} along "
                            f"({', '.join(f'{x:+.3f}' for x in m.axis)})")
        if store is not None:
            info.update(store.info(i))
            info["message"] += f"; {store.text(i)}"
        if ibs is not None:
            info.update(ibeam=dict(h=ibs.h, b=ibs.b, tw=ibs.tw, tf=ibs.tf, r=ibs.r, measured=dict(ibs.measured),
                                   profile=ibs.profile, standard=ibs.standard, snapped=ibs.snapped,
                                   rel_distance=ibs.rel_distance, unit_scale=ibs.unit_scale, confidence=ibs.confidence))
        yield ResidualStep("member", _snapshot(result), info)
        ops = ib.member_operations(m)
        if not ops:                                                   # not a known section: the old path for the region
            region = state.occ_in & mask
            prim = None
            if int(region.sum()) >= cfg.min_component:
                sub, sl = _window(state, region)
                prim = cascade.run_cascade(_voxel_channels(region[sl], sub), sub, registry, field=field,
                                           top_k=cfg.top_k, models=cfg.models, pass_iou=cfg.first_pass_iou,
                                           part_diagonal=state.part_diagonal)
                if prim.is_unknown:
                    prim = ib.member_bounding_box(m)
                prim.alternatives = []
            if prim is None:
                continue
            ev = _evaluate(state, prim, "add", purity_min)
            ok = ev["ok"] and ev["gain"] >= cfg.min_gain
            yield ResidualStep("candidate", _snapshot(result), dict(op="add", primitive=prim, gain=ev["gain"], ok=ok,
                                                                    reason=ev["reason"], member=m,
                                                                    message=f"add {prim.kind} for the {m.kind} member: gain {ev['gain']}"))
            if ok:
                recut = state.commit(ev["s_new"], ev["s_prim"], "add")
                result.operations.append(Operation("add", prim, component_voxels=int(region.sum()),
                                                   net_gain=float(ev["gain"]), recut_voxels=recut))
                accepted += 1
                yield ResidualStep("accepted", _snapshot(result), dict(op="add", primitive=prim, gain=ev["gain"], member=m,
                                                                       cut_history=state.cut_history.copy(),
                                                                       message=f"add {prim.kind}: |R| -{ev['gain']}"))
            else:
                yield ResidualStep("rejected", _snapshot(result), dict(op="add", primitive=prim, member=m,
                                                                       message=f"add {prim.kind}: gains {ev['gain']} voxels"))
            continue
        pair = next((p for o, p in ops if o == "cut"), None)
        for k, (op, prim) in enumerate(ops):
            if op == "cut":
                cuts.append((i, m, mask, k, prim))
                continue
            accepted += yield from slot(i, m, mask, k, op, prim, seed=i, exclude=pair)
    for i, m, mask, k, prim in cuts:
        accepted += yield from slot(i, m, mask, k, "cut", prim, seed=1000)
    report["accepted"] = accepted
    if store is not None:
        report["store"] = store.summary()
        if accepted:                                                  # the groups as created tools of the session
            try:
                from reco_engine.tools import ToolStore
                ts = cfg.feature_tool_store
                if isinstance(ts, (str, Path)):
                    loaded = ToolStore.load(ts)
                    report["tools"] = store.to_tools(loaded, result.input_path)
                    loaded.save(ts)
                elif isinstance(ts, ToolStore):
                    report["tools"] = store.to_tools(ts, result.input_path)
            except Exception as exc:                                  # the store is bookkeeping, never a verdict
                report["tools_error"] = str(exc)[-200:]
    return accepted


def _member_op(state, judge, result, cfg, m, mask, op, prim, seed=0, exclude=None, extra=None, verdict=None):
    """One member primitive: its candidate step ("ibeam_box" for an I-beam's boxes, else "member_op") and the verdict.
    `extra`: more info for its steps (the member's group, by_congruency: the primitive is a congruent member's, moved
    here - reco_engine/member_store.py); `verdict`: a dict that receives the judge's gain and precision."""
    kind = "ibeam_box" if m.kind == "ibeam" else "member_op"
    extra = dict(extra or {})
    ev = _exact_eval(state, judge, result, prim, op, cfg, seed=seed, exclude=exclude)
    if verdict is not None:
        verdict.update(gain=ev.get("gain", 0.0), precision=ev.get("precision", 0.0), ok=ev["ok"])
    what = prim.label or prim.kind
    how = ""
    if extra.get("by_congruency"):
        how = f" (by congruency: member {extra['source_member'] + 1}'s, group {extra.get('group')})"
    elif extra.get("after_failed_copy"):
        how = " (the copy failed: judged in full by itself)"
    yield ResidualStep(kind, _snapshot(result), dict(
        extra, op=op, primitive=prim, gain=ev["gain"], grid_gain=ev["grid_gain"], precision=ev["precision"], ok=ev["ok"],
        reason=ev["reason"], member=m, label=prim.label,
        message=f"{op} {what}{how}: exact gain {ev['gain']:.1f} voxel volumes, {ev['precision']:.1%} "
                + ("in the input" if op == "add" else "outside the input")
                + (f", {ev['reason']}" if ev["reason"] else "")))
    if not ev["ok"]:
        yield ResidualStep("rejected", _snapshot(result), dict(extra, op=op, primitive=prim, member=m, reason=ev["reason"],
                                                               message=f"{op} {what}{how}: {ev['reason']}"))
        return 0
    recut = _commit_member_op(state, result, op, prim, ev, int(mask.sum()))
    yield ResidualStep("accepted", _snapshot(result), dict(extra, op=op, primitive=prim, gain=ev["gain"], grid_gain=ev["grid_gain"],
                                                           precision=ev["precision"], recut=recut, member=m,
                                                           cut_history=state.cut_history.copy(),
                                                           message=f"{op} {what}{how}: exact gain {ev['gain']:.1f} voxel volumes"))
    return 1


def _hollow_phase(state: _State, field, result: ReconstructionResult, cfg: ResidualConfig, registry, spec):
    """The hollow sub-loop: the mesh has closed cavities (inner shells, ibeam.find_cavities). The outer shell alone is the
    first add (the cascade's primitive for it, else its bounding box); each cavity shell is then cut with the cascade's
    primitive of the SAME kind fitted to that shell (any kind when that fails), judged by `_exact_eval`. Returns the number
    of accepted operations (0: no cavity, the old first add runs)."""
    from reco_engine import ibeam as ib
    from src.closest_point_field import SurfaceField
    try:
        cav = ib.find_cavities(field)
    except Exception:
        cav = None
    if cav is None:
        return 0
    outer = SurfaceField(*cav.outer)
    t_out = _channels(outer, spec, cfg.exact_channels, cfg.field_backend)
    occ_out = t_out[0] > 0.5
    if not occ_out.any():
        return 0
    yield ResidualStep("hollow", _snapshot(result), dict(shell="outer", mask=occ_out, voxels=int(occ_out.sum()),
                                                         cavities=len(cav.inner),
                                                         message=f"hollow body: the outer shell and {len(cav.inner)} "
                                                                 f"cavity shell(s)"))
    first = cascade.run_cascade(t_out, spec, registry, field=outer, top_k=cfg.top_k, models=cfg.models,
                                pass_iou=cfg.first_pass_iou, part_diagonal=state.part_diagonal)
    how = f"cascade ({first.kind}) on the outer shell"
    if first.is_unknown:
        obb = cascade.oriented_bounding_box(t_out, spec, field=outer)
        if obb is None:
            return 0
        obb.alternatives = [first] + list(first.alternatives)
        first, how = obb, "oriented bounding box of the outer shell"
    s_first = state.prim_sdf(first)
    new = state.apply(state.s_model, s_first, "add")
    gain = state.size() - state.size(new)
    state.s_model = new
    result.operations.append(Operation("add", first, component_voxels=int(occ_out.sum()), net_gain=float(gain)))
    yield ResidualStep("add", _snapshot(result), dict(primitive=first, message=f"first add (hollow body): {how}"))
    judge = ib.ExactJudge(field, n=cfg.member_samples)
    done = 1
    for k, shell in enumerate(cav.inner):
        inner = SurfaceField(*shell)
        t_i = _channels(inner, spec, cfg.exact_channels, cfg.field_backend)
        occ_i = t_i[0] > 0.5
        yield ResidualStep("hollow", _snapshot(result), dict(shell="inner", index=k, mask=occ_i, voxels=int(occ_i.sum()),
                                                             message=f"cavity {k + 1}/{len(cav.inner)}: "
                                                                     f"{int(occ_i.sum())} voxels"))
        if not occ_i.any():
            continue
        cands = []
        for kinds in ((first.kind,), cascade.KINDS if hasattr(cascade, "KINDS") else None):
            kw = dict(kinds=kinds) if kinds is not None else {}
            p = cascade.run_cascade(t_i, spec, registry, field=inner, top_k=cfg.top_k, models=cfg.models,
                                    pass_iou=cfg.component_pass_iou, part_diagonal=state.part_diagonal, **kw)
            if not p.is_unknown:
                p.alternatives = []
                if not any(c.kind == p.kind for c in cands):
                    cands.append(p)
                if not cfg.hollow_cut_first:                         # the outer's kind only, when it fits
                    break
        best = None
        for c in cands:
            ev = _exact_eval(state, judge, result, c, "cut", cfg, seed=k)
            if (not ev["ok"] and cfg.hollow_cut_first and "exact" in ev
                    and cfg.hollow_cut_first_precision <= ev["precision"] < cfg.member_min_precision
                    and ev["gain"] >= cfg.min_gain):
                ev = dict(ev, ok=True, reason=f"cut first: {ev['precision']:.1%} of what it removes is empty, the input "
                                              f"material it takes (bodies reaching into the cavity) is left to the adds")
            yield ResidualStep("member_op", _snapshot(result), dict(
                op="cut", primitive=c, gain=ev["gain"], grid_gain=ev["grid_gain"], precision=ev["precision"],
                ok=ev["ok"], reason=ev["reason"],
                message=f"cut {c.kind} for cavity {k + 1}: exact gain {ev['gain']:.1f} voxel volumes"
                        + (f", {ev['reason']}" if ev["reason"] else "")))
            if ev["ok"] and (best is None or ev["gain"] > best[1]["gain"]):
                best = (c, ev)
        if best is None:
            yield ResidualStep("rejected", _snapshot(result), dict(op="cut", message=f"cavity {k + 1}: no cutter passes"))
            continue
        c, ev = best
        recut = _commit_member_op(state, result, "cut", c, ev, int(occ_i.sum()))
        done += 1
        yield ResidualStep("accepted", _snapshot(result), dict(op="cut", primitive=c, gain=ev["gain"], recut=recut,
                                                               cut_history=state.cut_history.copy(),
                                                               message=f"cut {c.kind} (cavity {k + 1}): exact gain "
                                                                       f"{ev['gain']:.1f} voxel volumes"))
    return done


# ------------------------------------------------------------------------------------------------ the loop

def _plan_after_miss(plan: Dict, cfg: ResidualConfig) -> Dict:
    """The analytic plan (`_subloop_plan` without a hint) after a HINTED sub-loop found nothing, minus what the hinted
    plan already tried: a member phase over every kind is not repeated, nor a hollow phase."""
    fb = _subloop_plan(cfg, "")
    if plan["members"] is None:
        fb["members"] = False
    if plan["hollow"]:
        fb["hollow"] = False
    return fb


def residual_steps(input_path, config: Optional[ResidualConfig] = None,
                   hint_subclass: Optional[str] = None) -> Iterator[ResidualStep]:
    """The loop's steps (see the module docstring). `hint_subclass` (or config.hint_subclass): the TypeNet subclass that
    selects the first-add sub-loop. None: the type stage of the registry (cascade.type_stage on the whole input: the
    descriptors' subclass or a hierarchical CNN's, whichever the fused winner came from) gives the hint - ibeam / frame /
    tube / rhs -> the member sub-loop first, hollow -> the hollow sub-loop first, full / sector / band -> the plain first
    add (the exact-field fit finds a segment's u/v itself); when the hinted sub-loop finds nothing the analytic detection
    runs as before. Without a subclass from any type entry the analytic gates of `_subloop_plan` decide, as before."""
    cfg = config or ResidualConfig()
    t_start = time.time()
    field, path_str = _load(input_path)
    name = Path(path_str).stem if path_str != "<field>" else "field"
    registry = cfg.registry if cfg.registry is not None else cascade.load_default_registry()
    from reco_engine.void_quality import PURITY_MIN
    purity_min = cfg.purity_min if cfg.purity_min is not None else PURITY_MIN
    resolution, tried = cfg.resolution, []
    tri = field.vertices[field.faces]
    mesh_volume = abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)
    grid_text, tau_info = None, None
    while True:
        if cfg.grid == "tau":
            spec, tau_info = _tau_spec(field, cfg)
            resolution = max(spec.shape)
            grid_text = ("x".join(str(n) for n in spec.shape) + f" at pitch {spec.pitch:.4g} (tau {tau_info['tau_p5']:.4g}, "
                         + ("the wall pitch" if tau_info["by"] == "tau" else f"the {cfg.resolution}^3 cube's pitch")
                         + (", coarsened: too many cells" if tau_info["capped"] else "") + ")")
        elif cfg.grid == "cube":
            spec = _spec_for(field, replace(cfg, resolution=resolution))
        else:
            raise ValueError(f"ResidualConfig.grid must be 'cube' or 'tau', got {cfg.grid!r}")
        t_in = _channels(field, spec, cfg.exact_channels, cfg.field_backend)
        state = _State(t_in, spec)
        state.connectivity_gate, state.connectivity_max_cells = bool(cfg.connectivity_gate), int(cfg.connectivity_max_cells)
        state.cut_grow, state.cut_open_faces = bool(cfg.cut_grow), bool(cfg.cut_open_faces)
        from reco_engine.memory_guard import MemoryGuard
        state.memory_guard = MemoryGuard(cfg.memory_min_free_mb)
        state.mesh = None                                            # the part's surface triangles (the statistics)
        n_in = int(state.occ_in.sum())
        ratio = n_in * state.pitch ** 3 / mesh_volume if mesh_volume > 0 else 1.0
        tried.append(f"{grid_text or f'{resolution}^3'}: {n_in} voxels ({ratio:.2f} of the volume)")
        if cfg.grid == "tau" or ratio >= cfg.min_volume_ratio or resolution >= cfg.max_resolution:
            break
        resolution = min(cfg.max_resolution, int(round(resolution * 1.5 / 16)) * 16)
    result = ReconstructionResult(input_path=path_str)
    if n_in == 0:                                                     # nothing to reconstruct at any resolution tried
        yield ResidualStep("input", _snapshot(result), dict(spec=spec, input_voxels=0, message="no input voxels: " + "; ".join(tried)))
        yield ResidualStep("done", _snapshot(result), dict(stopped="input not resolved on the grid (walls thinner than a voxel): "
                                                           + "; ".join(tried), seconds=time.time() - t_start, grade=None,
                                                           cut_history=state.cut_history.copy()))
        return

    inventory = None                                                  # (surfaces, mesh) of the starting-shape stage
    if cfg.start_shape not in ("legacy", "cubes", "compartments"):
        from reco_engine import start_shape as ss
        with timed("surface inventory"):
            inventory = ss.surface_inventory(path_str if path_str != "<field>" else None, field, channels=t_in, spec=spec)
    if cfg.max_ops is not None:
        max_ops, budget_reason = int(cfg.max_ops), "given"
    else:
        try:
            from reco_field.peel import predict_ops_budget
            with timed("ops budget"):
                b = predict_ops_budget(field, max_seconds=5.0)
            max_ops, budget_reason = int(b.max_ops), b.reason
        except Exception as exc:                                      # budget prediction is advisory
            max_ops, budget_reason = 12, f"default ({exc})"
        if inventory is not None:
            max_ops, budget_reason = _ops_budget(max_ops, inventory[0], cfg, budget_reason)
    if inventory is not None:
        state.mesh = inventory[1]
    elif field is not None and getattr(field, "vertices", None) is not None:
        state.mesh = dict(vertices=np.asarray(field.vertices, float), triangles=np.asarray(field.faces), face_id=None,
                          volume=None)

    # the type stage on the whole input: the sub-loop hint (its subclass) when none was given; reused by the first add
    hint = hint_subclass if hint_subclass is not None else cfg.hint_subclass
    type_result, type_info, type_text = None, None, ""
    # On a grid coarser than the thinnest wall the input tensor is aliasing noise, outside every network's training
    # scope: its type hint must not steer the sub-loops (the frame on 64^3: 2.63 x its volume in voxels, "torus/band"
    # at 0.60, which vetoed the member detection on the exact mesh). The analytic gates decide instead.
    state.blind = _grid_blindness(state, field)
    if hint is None and state.blind is not None and state.blind > 1.0:
        type_text = f", type stage skipped: the grid is blind (pitch {state.blind:.1f} x the thinnest wall)"
    elif hint is None:
        try:
            type_result = cascade.type_stage(t_in, spec, registry, field=field, models=cfg.models)
        except Exception as exc:                                      # the hint is advisory: the analytic gates decide
            type_text = f", type stage failed ({exc})"
        if type_result is not None and type_result.probs:
            type_info = type_result.as_dict()
            hint = type_result.hint
            type_text = f", type {type_result.parent}" + (f"/{type_result.subclass}" if type_result.subclass else "") \
                + f" ({type_result.mode} {type_result.probs[type_result.parent]:.2f})"
    yield ResidualStep("input", _snapshot(result), dict(spec=spec, input_voxels=n_in, max_ops=max_ops, budget=budget_reason,
                                                        tried=tried, resolution=resolution, occ_in=state.occ_in.copy(),
                                                        field_arrows=field_arrows(t_in, state.centres, state.pitch),
                                                        type_stage=type_info, hint_subclass=hint,
                                                        wall_tau=tau_info["tau"] if tau_info else None,
                                                        message=f"{n_in} input voxels on {grid_text or f'{resolution}^3'}"
                                                                + (f" (refined: {'; '.join(tried)})" if len(tried) > 1 else "")
                                                                + f", budget {max_ops}" + type_text
                                                                + (f"; {_rough_text(cfg, n_in)}" if cfg.feature_phase else "")))

    if cfg.first_guess is not None:                                   # the user's first guess (reco_engine/user_guess.py)
        n_adds = yield from _first_guess_phase(state, field, result, cfg, purity_min, path_str)
        if n_adds:                                                    # it replaces the start stage and the sub-loops
            yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops, n_in, name, t_start)
            return
    if cfg.start_shape == "compartments":                             # the mass compartments replace the sub-loops
        n_comp = yield from _compartment_add(state, field, result, cfg, n_in, t_start)
        if not result.operations:
            return
        yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops + max(0, n_comp - 1), n_in,
                                   name, t_start)
        return
    if cfg.start_shape == "leaves":                                   # the recognized leaves replace the sub-loops
        n_leaves = yield from _leaves_add(state, field, result, cfg, n_in, path_str)
        if n_leaves:
            yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops + max(0, n_leaves - 1), n_in,
                                       name, t_start)
            return                                                    # none recognized: the legacy start below takes over
    if cfg.start_shape == "cubes":                                    # the voxel-cube cover replaces the sub-loops
        n_cubes = yield from _cube_cover_add(state, field, result, cfg, n_in, t_start)
        if n_cubes:
            yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops + max(0, n_cubes - 1), n_in,
                                       name, t_start)
            return                                                    # no cube: the legacy start below takes over
    elif cfg.start_shape != "legacy":                                 # the starting-shape stage replaces the sub-loops
        yield from _start_add(state, field, result, cfg, registry, t_in, spec, n_in, name, t_start, path_str, type_result,
                              inventory=inventory)
        if not result.operations:
            return
        yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops, n_in, name, t_start)
        return

    # 2. first add: a sub-loop (members, hollow body) when the hint / the mesh calls for one, else one primitive / the
    #    bounding box; a hinted sub-loop that finds nothing hands over to the analytic detection
    plan = _subloop_plan(cfg, hint)
    sub_ops = 0
    # what the first solution came from, for the choice of solutions before the fallback (solution_choice.py); a
    # hinted member phase that detects nothing was asked for too (r08 of the frame review: 4 tubes, 0 operations)
    first_from, asked = "classic", bool(plan["forced"] and plan["members"] is not False)
    if cfg.member_congruency and cfg.feature_phase and cfg.feature_tool_store is None:
        from reco_engine.tools import ToolStore                      # one store for the run: the members' tools reach
        cfg = replace(cfg, feature_tool_store=ToolStore())            # the feature phase
    for attempt in range(2):
        if plan["members"] is not False:
            sub_ops = yield from _member_phase(state, field, result, cfg, registry, purity_min, plan["members"],
                                               forced=plan["forced"])
            first_from = "members" if sub_ops else first_from
        if sub_ops == 0 and plan["hollow"]:
            sub_ops = yield from _hollow_phase(state, field, result, cfg, registry, spec)
            first_from = "hollow" if sub_ops else first_from
        if sub_ops or not plan["forced"]:
            break
        plan = _plan_after_miss(plan, cfg)
    report = getattr(state, "member_report", None) or (dict(asked=True, detected=0, accepted=0, store=None) if asked else None)
    state.solution_ctx = dict(first=first_from, members=report, budget=max_ops, sub_ops=sub_ops, type_result=type_result)
    if sub_ops:
        max_ops += sub_ops
    else:
        yield from _first_add(state, field, result, cfg, registry, t_in, spec, n_in, name, t_start, type_result)
        if not result.operations:
            return

    yield from _residual_phase(state, field, result, cfg, registry, purity_min, max_ops, n_in, name, t_start)


def _prune_parts(result, field, cfg: ResidualConfig, name: str = "prune", build=None):
    """ResidualConfig.prune_parts: the accepted parts the shape does not need are marked accepted = False. The drops of
    reco_engine/prune.minimize in their order; the pruned program built in FreeCAD (`build(result, out_dir, name,
    timeout)` -> dict(valid, n_solids), default the feature phase's build) must be one valid solid - else the longest
    prefix of the drops that is (binary search, a build per probe). Yields one "prune" step; returns the stop text."""
    import tempfile
    from reco_engine import prune as pr
    acc = [o for o in result.operations if o.accepted]
    s = pr.sample_program(result, (np.asarray(field.vertices, float), np.asarray(field.faces)), n=cfg.prune_samples)
    keep0 = np.ones(len(acc), bool)
    before = pr.metrics(s, keep0)
    _, log = pr.minimize(s, keep0, tol=cfg.prune_tolerance)
    if not log:
        return ""
    build = build or (lambda r, out, nm, t: _feature_build_default(r, out, nm, t))
    out_dir = Path(cfg.out_dir) if cfg.out_dir else Path(tempfile.mkdtemp(prefix="prune_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    builds = []

    def ok(n):                                                   # the first n drops: one valid solid?
        k = keep0.copy()
        k[[d["index"] for d in log[:n]]] = False
        try:
            b = build(pr.edited_result(result, k), out_dir, f"{name}_prune_{n}", cfg.feature_call_timeout)
        except Exception as exc:
            b = dict(valid=False, n_solids=0, error=str(exc)[-120:])
        builds.append((n, bool(b.get("valid")) and int(b.get("n_solids") or 0) == 1))
        return builds[-1][1]

    n = len(log)
    if not ok(n):
        lo, hi = 0, n                                            # ok(lo) is assumed (the program as it came)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if ok(mid) else (lo, mid)
        n = lo
    dropped = [d["index"] for d in log[:n]]
    keep = keep0.copy()
    keep[dropped] = False
    after = pr.metrics(s, keep)
    for i in dropped:
        acc[i].accepted = False
        acc[i].primitive.label = (acc[i].primitive.label + ", " if acc[i].primitive.label else "") + "pruned"
    text = (f"pruned {n} of {len(acc)} parts (sampled IoU {before['iou']:.4f} -> {after['iou']:.4f}, tolerance "
            f"{cfg.prune_tolerance:g}; {len(builds)} FreeCAD build(s))" if n else
            f"prune: none of the {len(log)} proposed drops keeps one valid solid")
    yield ResidualStep("prune", _snapshot(result), dict(dropped=[i + 1 for i in dropped], proposed=len(log), parts=len(acc),
                                                        iou_before=before["iou"], iou_after=after["iou"],
                                                        builds=[dict(drops=a, one_valid_solid=b) for a, b in builds],
                                                        message=text))
    return f"; {text}"


def _tidy_cuts(result, field, state):
    """ResidualConfig.cut_tidy: the cover's faces lifted, the seams between cuts closed (reco_engine/cut_merge.py) on the
    accepted operations; a "solution" step says what changed. Returns the text for the run's `stopped`."""
    from reco_engine.cut_merge import close_seams, lift_cover_faces
    acc = [o for o in result.operations if o.accepted]
    if sum(o.op == "cut" for o in acc) == 0:
        return ""
    pitch = float(getattr(state, "pitch", 0.0) or 0.0)
    if pitch <= 0:
        return ""
    try:
        with timed("tidy cuts"):
            ops = [(o.op, o.primitive) for o in acc]
            ops, lifted = lift_cover_faces(ops, margin=3.0 * pitch)          # one to one: no operation added
            ops, seams, origin = close_seams(ops, field.vertices, field.faces, tol=0.5 * pitch, overlap=pitch,
                                             return_origin=True)            # seam cuts are inserted
    except Exception as exc:                                         # a tidy-up must never end the run
        return f"; tidying the cuts failed: {str(exc)[-80:]}"
    if not lifted and not seams:
        return ""
    rebuilt = []
    for (op_, prim), src in zip(ops, origin):                        # by origin, never by position
        if src is None:
            rebuilt.append(Operation(op_, prim, component_voxels=0, net_gain=0.0))
        else:
            acc[src].primitive = prim
            rebuilt.append(acc[src])
    result.operations = rebuilt
    yield ResidualStep("solution", _snapshot(result), dict(
        kept="tidied cuts", solutions=[], measure="the cover's faces and the input mesh's inside test (cut_merge)",
        tidied=dict(cover_faces=lifted, seams=seams),
        message=f"cuts tidied: {lifted} cut face(s) past the cover's face, {seams} seam(s) between cuts closed"))
    return f"; cuts tidied ({lifted} faces past the cover, {seams} seams)"


def _first_guess_phase(state, field, result, cfg, purity_min, path_str):
    """The user's first guess (the contract in reco_engine/user_guess.py): the rough shapes refined by
    user_guess.refine (an entry with "refined": true is taken as it is; for a STEP input onto its exact faces too),
    then each judged by the LOOP'S OWN checks on its grid (`_evaluate`: the voxel gain >= min_gain, for a cut the void
    and sign checks) - refine's own checks are approximate - in the contract's order; a shape that fails is reported
    ("rejected", reason) and dropped, never forced. Returns the number of adds kept (0: the loop starts as usual)."""
    from reco_engine import user_guess as ug
    fg = cfg.first_guess
    entries, ordered = [], False
    if isinstance(fg, (str, Path)):
        shapes, meta = ug.load_first_guess(fg)
        ordered = bool(meta.get("ordered"))
        flags = meta.get("refined") or []               # one flag per shape (a list: bool(list) took every shape as given)
        entries = [(s, bool(flags[i]) if i < len(flags) else False) for i, s in enumerate(shapes)]
    else:
        for e in fg:
            if isinstance(e, ug.UserShape):
                entries.append((e, False))
            elif isinstance(e, dict):
                entries.append((ug.UserShape.from_dict(e), bool(e.get("refined"))))
            else:                                                     # (op, Primitive): taken as it is
                entries.append((ug.UserShape.from_primitive(*e), True))
    faces = None
    if path_str.lower().endswith((".step", ".stp")):
        try:
            from reco_field.exact_check import exact_faces
            inv = exact_faces(path_str)
            faces = inv.get("faces")
        except Exception:
            faces = None
    todo = [i for i, (shape, done) in enumerate(entries) if not done and shape.role != "cover"]   # a cover bounds
                                                                  # the part already: a refit to the surface only shrinks it
    if todo and field is not None:
        try:
            refined = ug.refine([entries[i][0] for i in todo], field.vertices, field.faces, exact_faces=faces)
            for i, r in zip(todo, refined):
                entries[i] = (r.shape, True)
        except Exception:                                             # the refinement is advisory: the rough shape
            pass
    order = list(range(len(entries)))
    if not ordered:
        order = [i for i in order if entries[i][0].op == "add"] + [i for i in order if entries[i][0].op != "add"]
    n_adds = 0
    for i in order:
        shape = entries[i][0]
        prim, op = shape.primitive(), shape.op
        prim.sources = list(prim.sources) + [StageSource("type", "user", 1.0)]
        try:
            ev = _evaluate(state, prim, op, purity_min)
        except Exception as exc:
            ev = dict(ok=False, gain=0, reason=f"error: {str(exc)[-150:]}")
        cover = shape.role == "cover" and op == "add"
        if cover and ev["ok"]:                                    # an over-cover by design: judged by what it holds
            ev = dict(ev, **_cover_check(state, ev["s_prim"], shape.region_mm))
        ok = bool(ev["ok"]) and (cover or ev["gain"] >= cfg.min_gain)
        reason = ev.get("reason") or ("" if ok else f"gains {ev['gain']} voxels, needs {cfg.min_gain}")
        what = _what(prim) + (f" (cover, holds {ev['held']:.1%} of the input material"
                              + (" in its region" if shape.region_mm else "") + ")" if cover and "held" in ev else "")
        yield ResidualStep("candidate", _snapshot(result), dict(op=op, primitive=prim, gain=ev["gain"], ok=ok, reason=reason,
                                                                source="user", first_guess=i,
                                                                message=f"first guess {i + 1}: {op} {what}: gain {ev['gain']}"
                                                                        + (f", {reason}" if reason else "")))
        if not ok:
            yield ResidualStep("rejected", _snapshot(result), dict(op=op, primitive=prim, reason=reason, source="user",
                                                                   first_guess=i,
                                                                   message=f"first guess {i + 1}: {op} {what} dropped - {reason}"))
            continue
        recut = state.commit(ev["s_new"], ev["s_prim"], op)
        result.operations.append(Operation(op, prim, component_voxels=0, net_gain=float(ev["gain"]), recut_voxels=recut))
        n_adds += op == "add"
        yield ResidualStep("accepted", _snapshot(result), dict(op=op, primitive=prim, gain=ev["gain"], recut=recut,
                                                               source="user", first_guess=i, cut_history=state.cut_history.copy(),
                                                               message=f"first guess {i + 1}: {op} {what}: |R| {-ev['gain']:+d}"))
    return n_adds


COVER_MIN_HELD = 0.99      # a user's cover must hold this share of the input material (of its region) on the loop's grid


def _cover_check(state, s_prim: np.ndarray, region_mm=None) -> Dict:
    """The user's cover (user_guess: role "cover", the dialog's "around the part / sub-part"): an over-cover by design,
    so its net voxel gain says nothing - on the ABC plate 00009283 (12.7 mm thick at pitch 11.66) a box around the whole
    part gained 0 voxels and a sub-part box of 00005438 -872, and both were dropped (the user's session of 2026-10-29).
    It is kept when it holds COVER_MIN_HELD of the input material on the grid (within half a voxel; for a sub-part the
    material whose voxel centres lie in its region); the loop cuts its excess away afterwards. dict(ok, reason, held)."""
    inside = s_prim >= -state.band
    material = state.occ_in
    if region_mm is not None:
        lo, hi = (np.asarray(b, float) for b in region_mm)
        c = state.centres
        material = material & np.all((c >= lo) & (c <= hi), axis=-1)     # a voxel centre beyond a region plane belongs
                                                                            # to the neighbour (the cover ends at the plane)
    n = int(np.count_nonzero(material))
    if n == 0:
        return dict(ok=False, held=0.0, reason="its region holds no input material on the loop's grid")
    held = float(np.count_nonzero(material & inside)) / n
    if held < COVER_MIN_HELD:
        return dict(ok=False, held=held, reason=f"a cover holds {held:.1%} of the input material"
                    + (" of its region" if region_mm is not None else "") + f", needs {COVER_MIN_HELD:.0%}")
    return dict(ok=True, held=held, reason="")


@timed_fn("starting shape")
def _ops_budget(predicted: int, surfaces, cfg: ResidualConfig, reason: str = "") -> Tuple[int, str]:
    """The budget with the inventory's major surfaces counted in (ResidualConfig.budget_from_surfaces): each plane,
    cylinder or sphere with >= budget_surface_share of the area is a feature the loop may need one operation for."""
    if not cfg.budget_from_surfaces:
        return predicted, reason
    n_major = sum(1 for s in surfaces if s.kind in ("plane", "cylinder", "sphere") and s.area_share >= cfg.budget_surface_share)
    budget = min(cfg.budget_cap, max(predicted, 1 + cfg.budget_surface_margin + n_major))
    return budget, f"{reason}; {n_major} surfaces >= {cfg.budget_surface_share:.1%} -> {budget}"


def _start_add(state, field, result, cfg, registry, t_in, spec, n_in, name, t_start, path, type_result=None, inventory=None):
    """The first add from the starting-shape stage (reco_engine/start_shape.py): every candidate graded (coverage,
    excess, contains); cfg.start_shape "auto" takes the rule's choice, a named type is taken as the user forced it.
    When nothing is picked (no candidate passes choose_start - e.g. every `excess` is None because manifold3d is
    missing or the solid failed - or the named type is not among the candidates) a "done" step ends the run with an
    empty program: no fallback to `_first_add`, no FreeCAD grade (review R-01, 2026-09-30). The cascade runs first
    even for a named type (its answer is one of the candidates)."""
    from reco_engine import start_shape as ss
    first = cascade.run_cascade(t_in, spec, registry, field=field, top_k=cfg.top_k, models=cfg.models,
                                pass_iou=cfg.first_pass_iou, type_result=type_result)
    state.root_cascade = first                                      # reused by the tree's cover check (25 s on cy_t otherwise)
    surfaces, mesh = inventory if inventory is not None else \
        ss.surface_inventory(path if path != "<field>" else None, field, channels=t_in, spec=spec)
    cands = ss.start_candidates(surfaces, field, t_in, spec, first=first)
    if cfg.feature_phase and cfg.rough_cut_grade > 0:                   # rough: one plain primitive, no composite cover
        cands = [(n, p) for n, p in cands if p.kind not in ("composite", "halfspace")] or cands
    graded = [(n, p, ss.grade_start(p, field, mesh)) for n, p in cands]
    if cfg.start_shape == "auto":
        pick = ss.choose_start(graded, mode=cfg.start_score)
    else:
        pick = next((g for g in graded if g[0] == cfg.start_shape), None)
    if pick is None:
        yield ResidualStep("done", _snapshot(result), dict(stopped=f"no starting shape of type {cfg.start_shape!r}",
                                                           grade=None, seconds=time.time() - t_start,
                                                           cut_history=state.cut_history.copy()))
        return
    kind, prim, g = pick
    varied = ""
    if cfg.start_vary and not (cfg.feature_phase and cfg.rough_cut_grade > 0):     # rough: no variation rounds
        with timed("vary start"):
            prim2, g2, steps = ss.vary_start(prim, field, mesh, float(state.pitch), rounds=cfg.start_vary_rounds)
        if steps:
            varied = f"; varied {len(steps)} step(s), score {g['score']:.3f} -> {g2['score']:.3f} ({'; '.join(steps)})"
            prim, g = prim2, g2
    s_first = state.prim_sdf(prim)
    gain = state.size() - state.size(state.apply(state.s_model, s_first, "add"))
    state.s_model = state.apply(state.s_model, s_first, "add")
    result.operations.append(Operation("add", prim, component_voxels=n_in, net_gain=float(gain)))
    table = [(n, {k: v for k, v in gg.items()}) for n, _, gg in graded]
    dom = sum(1 for s in surfaces if s.dominating)
    yield ResidualStep("add", _snapshot(result), dict(
        primitive=prim, start_grades=table,
        message=(f"starting shape ({cfg.start_shape}): {kind}, score {g['score']:.3f}, {g['coverage']:.0%} of the surface on it, excess "
                 + (f"{g['excess_share']:+.0%}" if g.get("excess_share") is not None else "n/a")
                 + f" of the part, {g['operands']} operand(s){varied}; {len(surfaces)} surfaces ({dom} dominating); "
                 + "graded (coverage/excess/score): "
                 + ", ".join(f"{n} {gg['coverage']:.2f}/{(gg['excess_share'] if gg.get('excess_share') is not None else float('nan')):+.2f}/{gg['score']:.2f}"
                             for n, _, gg in graded))))


def _first_add(state, field, result, cfg, registry, t_in, spec, n_in, name, t_start, type_result=None):
    """The first add, the most important decision of the run: the cascade's answer (and its alternatives) and every
    covering mode compete under one grade (`_bounding_cover`: how each candidate's surface lies against the input's,
    containment required). Without a field: the cascade's primitive when it explains the whole input, else the oriented
    bounding box."""
    first = cascade.run_cascade(t_in, spec, registry, field=field, top_k=cfg.top_k, models=cfg.models,
                                pass_iou=cfg.first_pass_iou, type_result=type_result)
    grades = []
    if field is not None:
        extra = ([] if first.is_unknown else [(f"cascade {first.kind}", first)]) + \
            [(f"cascade alternative {a.kind}", a) for a in first.alternatives if not a.is_unknown][:3]
        chosen, how_chosen = _bounding_cover(t_in, spec, field, state.pitch, grades_out=grades, extra=extra)
        if chosen is not None:
            if chosen is not first:
                # the others, each without alternatives of its own: `chosen` may BE one of first's alternatives, and
                # first with its list among chosen's made the primitive refer to itself - the result could not be
                # written (RecursionError on 4 of the 20 parts of the ABC pilot, 2026-09-29)
                others = ([] if first.is_unknown else [first]) + list(first.alternatives)
                chosen.alternatives = [replace(x, alternatives=[]) for x in others if x is not chosen]
            first, how = chosen, how_chosen
        elif first.is_unknown:
            yield ResidualStep("done", _snapshot(result), dict(stopped="no first add: nothing covers the input",
                                                               grade=None, seconds=time.time() - t_start,
                                                               cut_history=state.cut_history.copy()))
            return
        else:
            how = f"cascade ({first.kind}), no cover graded"
    elif first.is_unknown:
        obb, how_cover = _bounding_cover(t_in, spec, field, state.pitch, grades_out=grades)
        if obb is None:
            yield ResidualStep("done", _snapshot(result), dict(stopped="no first add: the cascade found nothing and there "
                                                               "is no bounding box to fall back to", grade=None,
                                                               seconds=time.time() - t_start, cut_history=state.cut_history.copy()))
            return
        obb.alternatives = [first] + list(first.alternatives)
        first, how = obb, how_cover
    else:
        how = f"cascade ({first.kind})"
    subparts, cover_text = [], ""
    if cfg.rough_subparts and field is not None and Path(str(result.input_path)).suffix.lower() in (".step", ".stp"):
        try:
            subparts, cover_text = _subpart_cover(first, field, result.input_path, cfg)
        except Exception as exc:                                      # the cover is an offer: the single add stands
            cover_text = f"; sub-part cover failed: {str(exc)[-120:]}"
        if subparts:
            subparts[0].alternatives = [first] + list(first.alternatives)
            first, how = subparts[0], f"sub-part cover{cover_text}"
            cover_text = ""
    s_first = state.prim_sdf(first)
    gain = state.size() - state.size(state.apply(state.s_model, s_first, "add"))
    state.s_model = state.apply(state.s_model, s_first, "add")
    result.operations.append(Operation("add", first, component_voxels=n_in, net_gain=float(gain)))
    if cfg.freecad and cfg.grade_each_step:
        _grade(result, field, cfg, f"{name}_step1")
    yield ResidualStep("add", _snapshot(result), dict(
        primitive=first, cover_grades=[(n, {k: v for k, v in g.items() if not isinstance(v, np.ndarray)}) for n, g in grades],
        message=f"first add: {how}{cover_text}" + ("; graded: " + ", ".join(f"{n} {g.get('score', -1):+.2f}" for n, g in grades)
                                                   if grades else "")))
    for k, prim in enumerate(subparts[1:], start=2):                  # the further sub-parts: adds before any cut
        s_p = state.prim_sdf(prim)
        new = state.apply(state.s_model, s_p, "add")
        gain = state.size() - state.size(new)
        recut = state.commit(new, s_p, "add")
        result.operations.append(Operation("add", prim, component_voxels=n_in, net_gain=float(gain), recut_voxels=recut))
        yield ResidualStep("accepted", _snapshot(result), dict(
            op="add", primitive=prim, gain=gain, subpart=k, subparts=len(subparts), cut_history=state.cut_history.copy(),
            message=f"add {prim.kind}: sub-part {k} of {len(subparts)} of the rough cover, |R| {-gain:+d}"))


def _compartment_add(state, field, result, cfg: ResidualConfig, n_in: int, t_start: float):
    """The mass compartments (start_shape "compartments"; reco_engine/mass_compartments.py): the input occupancy split
    into boxes that hold its mass with little empty space, each added (a face moved onto the input surface where the
    samples support it, `_snap_box_faces`). The cover holds what the compartments do not fill: the residual phase cuts
    it and adds what they left out, until its fits converge. Returns the number of compartments added; with none, a
    "done" step ends the run."""
    from reco_engine.mass_compartments import compartments
    occ = np.asarray(state.occ_in, bool)
    with timed("mass compartments"):
        leaves = compartments(occ, fill_good=cfg.compartments_fill, max_leaves=max(1, int(cfg.compartments_max)),
                              min_voxels=max(1, int(cfg.compartments_min_voxels)), min_fill=cfg.compartments_min_fill)
    if not leaves:
        yield ResidualStep("done", _snapshot(result), dict(stopped="no mass compartment: the input occupancy is empty or "
                                                           "scattered", grade=None, seconds=time.time() - t_start,
                                                           cut_history=state.cut_history.copy()))
        return 0
    prims, snapped = [], 0
    for k, c in enumerate(leaves, start=1):
        p = _cube_primitive(state, np.asarray(c["lo"]), np.asarray(c["hi"]))
        q = _snap_box_faces(state, p, field) if field is not None else None
        if q is not None:
            snapped += 1
            p = q
        p.label = f"mass compartment {k} of {len(leaves)} (filled {c['fill']:.0%})"
        p.sources = [StageSource(s, "mass_compartments", 0.0) for s in ("type", "rotation", "dimensions", "position")]
        prims.append(p)
    held = sum(c["voxels"] for c in leaves) / max(1, n_in)
    fills = ", ".join(f"{c['fill']:.0%}" for c in leaves)
    first = prims[0]
    s_first = state.prim_sdf(first)
    gain = state.size() - state.size(state.apply(state.s_model, s_first, "add"))
    state.s_model = state.apply(state.s_model, s_first, "add")
    result.operations.append(Operation("add", first, component_voxels=n_in, net_gain=float(gain)))
    yield ResidualStep("add", _snapshot(result), dict(
        primitive=first, compartments=leaves,
        message=f"first add: mass compartments: {len(leaves)} box(es) holding {held:.1%} of the input, filled {fills}; "
                f"{snapped} snapped to the surface; the loop cuts their excess and adds the rest"))
    for k, prim in enumerate(prims[1:], start=2):                     # the further compartments: adds before any cut
        s_p = state.prim_sdf(prim)
        new = state.apply(state.s_model, s_p, "add")
        gain = state.size() - state.size(new)
        recut = state.commit(new, s_p, "add")
        result.operations.append(Operation("add", prim, component_voxels=n_in, net_gain=float(gain), recut_voxels=recut))
        yield ResidualStep("accepted", _snapshot(result), dict(
            op="add", primitive=prim, gain=gain, subpart=k, subparts=len(prims), cut_history=state.cut_history.copy(),
            message=f"add box: {prim.label}, |R| {-gain:+d}"))
    return len(prims)


def _leaves_add(state, field, result, cfg: ResidualConfig, n_in: int, input_path):
    """The recognized leaves as the first operations (start_shape "leaves", reco_engine/leaf_recognition.py): the
    selected program's adds, cuts and intersections in its order. Returns their number; 0 (with a "fallback" step) for an
    STL input, a failed recognition or one under leaves_min_iou - then the legacy start takes over."""
    from reco_engine.brep import is_step

    def fallback(why):
        return ResidualStep("fallback", _snapshot(result), dict(reason=why, message=f"leaves: {why} - the legacy start takes over"))
    if input_path is None or not is_step(input_path) or field is None:
        yield fallback("the recognition needs a STEP input")
        return 0
    from reco_engine import leaf_recognition as lr
    try:
        with timed("leaf recognition"):
            rec = lr.recognize(input_path, mesh=(np.asarray(field.vertices, float), np.asarray(field.faces)))
    except Exception as exc:
        yield fallback(f"the recognition failed ({str(exc)[-120:]})")
        return 0
    prog, info = rec["program"], rec["info"]
    if not prog.operations or (info.get("iou") or 0.0) < cfg.leaves_min_iou:
        yield fallback(f"nothing recognized well enough (sampled IoU {info.get('iou')}, at least {cfg.leaves_min_iou:g})")
        return 0
    for k, o in enumerate(prog.operations):
        s_p = state.prim_sdf(o.primitive)
        new = state.apply(state.s_model, s_p, o.op)
        gain = state.size() - state.size(new)
        if k == 0:
            state.s_model, recut = new, 0
        else:
            recut = state.commit(new, s_p, o.op)
        result.operations.append(Operation(o.op, o.primitive, component_voxels=n_in, net_gain=float(gain), recut_voxels=recut))
        text = (f"first add: recognized leaves - {len(rec['surfaces'])} surfaces, {len(rec['hypotheses'])} hypotheses, "
                f"{len(rec['passed'])} verified, {len(prog.operations)} selected (sampled IoU {info.get('iou', 0):.4f})"
                if k == 0 else f"{o.op} {o.primitive.kind}: recognized leaf {k + 1} of {len(prog.operations)}, |R| {-gain:+d}")
        yield ResidualStep("add" if k == 0 else "accepted", _snapshot(result), dict(
            op=o.op, primitive=o.primitive, gain=gain, recognized=info if k == 0 else None,
            cut_history=state.cut_history.copy(), message=text))
    return len(prog.operations)


def _cube_cover_add(state, field, result, cfg: ResidualConfig, n_in: int, t_start: float):
    """The voxel-cube cover (start_shape "cubes"): the largest completely filled axis-aligned boxes of the input occupancy,
    grown from the voxels one after the other (`largest_boxes`, the voxel fallback's search), as the first adds, each
    face moved onto the input surface where the samples support it (`_snap_box_faces`). The boxes lie inside the material
    by construction: the cover has no excess to cut, and what it does not reach is left to the residual phase as
    additional primitives (the cascade on the leftover pieces) and to the feature phase. Returns the number of boxes
    added; with none, a "fallback" step says so and the legacy start takes over."""
    occ = np.asarray(state.occ_in, bool)
    boxes = largest_boxes(occ, count=int(cfg.cubes_max), max_side=48, min_voxels=max(1, int(cfg.cubes_min_voxels)))
    kept, rest, first_n = [], occ.copy(), None
    for lo, hi in boxes:
        n = int(np.prod(hi - lo + 1))
        if first_n is None:
            first_n = n
        elif n < cfg.cubes_min_share * first_n:
            break
        kept.append((lo, hi, n))
        rest[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] = False
        if int(rest.sum()) <= cfg.cubes_rest * n_in:
            break
    if not kept:                # B5 of the user's session of 2026-10-29: the ABC part 00005438 (a 1.6 mm plate, 2-3 voxels
        # thick at pitch 0.7) has no such box, and the run ended with 0 operations and 'residual +0 / -0'; now the
        # legacy start takes over (residual_steps)
        yield ResidualStep("fallback", _snapshot(result), dict(
            reason="no voxel cube",
            message=f"no voxel cube: no completely filled box of at least {cfg.cubes_min_voxels} voxels in the input "
                    f"(thinner than {round(cfg.cubes_min_voxels ** (1 / 3))} voxels everywhere) - the legacy start takes over"))
        return 0
    prims, snapped = [], 0
    for k, (lo, hi, n) in enumerate(kept, start=1):
        p = _cube_primitive(state, lo, hi)
        q = _snap_box_faces(state, p, field) if cfg.cubes_snap and field is not None else None
        if q is not None:
            snapped += 1
            p = q
        p.label = f"voxel cube {k} of {len(kept)}" + (f" ({q.label.split(', ', 1)[1]})" if q is not None else "")
        p.sources = [StageSource(s, "cube_cover", 0.0) for s in ("type", "rotation", "dimensions", "position")]   # not the fallback
        prims.append(p)
    covered = 1.0 - int(rest.sum()) / max(1, n_in)
    sizes = ", ".join("x".join(str(int(v)) for v in (hi - lo + 1)) for lo, hi, n in kept)
    first = prims[0]
    s_first = state.prim_sdf(first)
    gain = state.size() - state.size(state.apply(state.s_model, s_first, "add"))
    state.s_model = state.apply(state.s_model, s_first, "add")
    result.operations.append(Operation("add", first, component_voxels=n_in, net_gain=float(gain)))
    yield ResidualStep("add", _snapshot(result), dict(
        primitive=first, cubes=[(lo.tolist(), hi.tolist(), n) for lo, hi, n in kept],
        message=f"first add: voxel cube cover: {len(kept)} box(es) grown from the voxels ({sizes} voxels), {covered:.1%} of the "
                f"input covered, {snapped} snapped to the surface; the rest is left to the loop as additional primitives"))
    for k, prim in enumerate(prims[1:], start=2):                     # the further boxes: adds before any cut
        s_p = state.prim_sdf(prim)
        new = state.apply(state.s_model, s_p, "add")
        gain = state.size() - state.size(new)
        recut = state.commit(new, s_p, "add")
        result.operations.append(Operation("add", prim, component_voxels=n_in, net_gain=float(gain), recut_voxels=recut))
        yield ResidualStep("accepted", _snapshot(result), dict(
            op="add", primitive=prim, gain=gain, subpart=k, subparts=len(prims), cut_history=state.cut_history.copy(),
            message=f"add box: voxel cube {k} of {len(prims)} of the cover, |R| {-gain:+d}"))
    return len(prims)


def _subpart_cover(first: Primitive, field, input_path, cfg: ResidualConfig):
    """([the cover's primitives] or [] when the single first add stays, the text for the step's message)."""
    from reco_engine import subpart_cover as sc
    from reco_engine import surface_orders as so
    from reco_field.exact_check import exact_faces
    with timed("sub-part cover"):
        faces = exact_faces(str(input_path)).get("faces") or []
        V, F = np.asarray(field.vertices, float), np.asarray(field.faces)
        single = so.coverage(first, faces, V, F, n=4000)
        single_in = so.containment(primitive_to_sp(first), V, F)["inside"]
        c = sc.cover(V, F, faces, max_parts=cfg.rough_subparts_max, min_gain=cfg.rough_subparts_min_gain,
                     min_saving=cfg.rough_subparts_min_saving)
    gain = c["exact"] - single["exact"]
    text = (f": {len(c['primitives'])} sub-part(s) ({', '.join(p.kind for p in c['primitives'])}), exact surface "
            f"coverage {c['exact']:.3f} against the single {first.kind}'s {single['exact']:.3f}, containment "
            f"{c['containment']:.4f} against {single_in:.4f}")
    better = gain >= cfg.rough_subparts_min_gain or (gain > -0.01 and single_in < COVER_MIN_INSIDE <= c["containment"])
    if c["containment"] < COVER_MIN_INSIDE or not better:
        return [], f"; sub-part cover not taken{text}"
    return list(c["primitives"]), text


COVER_MIN_INSIDE = 0.995             # a bounding cover must contain this share of the input's surface samples


def _grade_cover(prim: Primitive, spec, centres: np.ndarray, occ_in: np.ndarray, field, pitch: float,
                 min_component: int = 6, max_points: int = 20000) -> Dict[str, float]:
    """How a first-add candidate's surface lies against the input's, locally (the user's rule 2026-09-25: "50 % exact same
    surface and 50 % far away could be a good result - maybe only one cut is needed to finish"). Points on the
    candidate's own surface (its boundary voxels on a grid over ITS OWN bounds - the loop's grid hugs the part and would
    miss the overhang, the very surface that is "far" - projected onto its exact surface) are sorted by their
    distance d to the input surface:
      on    d <= max(0.05 pitch, 0.002 x the part's diagonal)   - this surface is explained
      far   d >= 2 pitch                                        - material a cut can take away
      near  in between                                          - slivers that need many small cuts
    score = on - near. Also: the share of the input's surface samples it contains (`inside`), how many separate pieces
    of excess it leaves (`pieces`, a rough count of the cuts still needed) and its volume in voxels."""
    from scipy import ndimage
    sp_p = primitive_to_sp(prim)
    inside_grid = -sp_p.sdf(centres) >= 0                                # on the loop's grid: the excess pieces
    diag = float(np.linalg.norm(np.asarray(field.bounds[1], float) - np.asarray(field.bounds[0], float)))
    part_lo, part_hi = np.asarray(field.bounds[0], float), np.asarray(field.bounds[1], float)
    lo, hi = (np.asarray(b, float) for b in sp_p.bounds())
    lo = np.maximum(lo, part_lo - diag)                                  # a half-space is unbounded: keep it near the part
    hi = np.minimum(hi, part_hi + diag)
    own_pitch = max(pitch, float(np.prod(hi - lo) / 2e6) ** (1.0 / 3.0))
    own = cascade.box_grid_spec(lo, hi, own_pitch).centres()
    own_inside = -sp_p.sdf(own) >= 0
    boundary = own_inside & ~ndimage.binary_erosion(own_inside, border_value=0)
    pts = own[boundary]
    eps_pitch = own_pitch
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    eps = 0.25 * eps_pitch
    for _ in range(2):                                                   # Newton steps onto the candidate's surface
        d = sp_p.sdf(pts)
        g = np.stack([(sp_p.sdf(pts + eps * e) - sp_p.sdf(pts - eps * e)) / (2 * eps) for e in np.eye(3)], axis=1)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        pts = pts - d[:, None] * g
    pts = pts[np.abs(sp_p.sdf(pts)) < 0.1 * eps_pitch]
    if not len(pts):
        return dict(on=0.0, near=1.0, far=0.0, score=-1.0, inside=0.0, pieces=0, volume=int(inside_grid.sum()))
    import igl
    d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(pts, dtype=np.float64),
                                               np.ascontiguousarray(field.vertices, dtype=np.float64),
                                               np.ascontiguousarray(field.faces, dtype=np.int64))
    d = np.sqrt(np.maximum(np.asarray(d2, float), 0.0))
    tol_on, tol_far = max(0.05 * pitch, 0.002 * diag), 2.0 * pitch
    on, far = float(np.mean(d <= tol_on)), float(np.mean(d >= tol_far))
    near = 1.0 - on - far
    part_on = None
    if prim.kind == "composite" and len(prim.parts) == 2:
        # which operand's surface each point lies on (the smaller |sdf|), and how much of that operand's share is "on"
        which = np.argmin(np.stack([np.abs(primitive_to_sp(q).sdf(pts)) for q in prim.parts]), axis=0)
        part_on = [float(np.mean(d[which == k] <= tol_on)) if np.any(which == k) else 0.0 for k in (0, 1)]
    _, inside = cascade.surface_agreement(prim, field, tol_on)
    labels, n = ndimage.label(inside_grid & ~occ_in)
    sizes = np.bincount(labels.ravel())[1:] if n else np.zeros(0)
    return dict(on=on, near=near, far=far, score=on - near, inside=inside, pieces=int((sizes >= min_component).sum()),
                volume=int(inside_grid.sum()), part_on=part_on)


COMPOSITE_PART_ON = 0.5            # a composite cover: each operand's surface at least this share on the input


@timed_fn("bounding cover")
def _bounding_cover(t_in, spec, field, pitch: float, grades_out: Optional[list] = None, extra=None):
    """(primitive, how) for the first add: every covering mode of cascade.bounding_candidates (oriented / world-axis box, bounding cylinder sector and full, bounding sphere,
    box & cylinder), graded by `_grade_cover`; the best score among those containing COVER_MIN_INSIDE of the surface,
    ties (within 0.02) to a single primitive over a composite, then fewer excess pieces, then the smaller volume.
    `grades_out` receives (name, grade) of every candidate; `extra` [(name, primitive)] joins the candidates (the
    cascade's own answer and its alternatives - graded like every cover, 2026-09-25: on cy_flat the cascade's box
    had passed IoU 0.9 and was taken ungraded, a box for a cylinder with a flat). Without a field: the oriented
    bounding box."""
    obb = cascade.oriented_bounding_box(t_in, spec, field=field)
    if field is None:
        return obb, "oriented bounding box"
    centres = cascade.grid_centres(spec)
    occ_in = np.asarray(t_in[0]) > 0.5
    graded = []
    for name, cand in list(extra or []) + cascade.bounding_candidates(t_in, spec, field=field):
        try:
            g = _grade_cover(cand, spec, centres, occ_in, field, pitch)
        except Exception as exc:                                         # one broken mode must not stop the first add
            g = dict(score=-1.0, inside=0.0, error=str(exc))
        graded.append((name, cand, g))
        if grades_out is not None:
            grades_out.append((name, g))
    # a composite must be justified part by part: each operand's own surface mostly on the input (box & cylinder on a
    # flat-sided part rounds off a corner the input never had - measured on bs_box_step / bs_box_dome, 2026-09-25)
    ok = [t for t in graded if t[2].get("inside", 0.0) >= COVER_MIN_INSIDE
          and (t[2].get("part_on") is None or min(t[2]["part_on"]) >= COMPOSITE_PART_ON)]
    if not ok:
        return obb, "oriented bounding box (no candidate contains the input)"
    top = max(t[2]["score"] for t in ok)
    near_top = [t for t in ok if t[2]["score"] >= top - 0.02]
    name, best, g = min(near_top, key=lambda t: (t[1].kind == "composite", t[2].get("pieces", 0), t[2].get("volume", 0)))
    return best, (f"{name} ({g['on']:.0%} of its surface on the input, {g['near']:.0%} near, {g['far']:.0%} far, "
                  f"{g['pieces']} excess piece(s))")


def _rough_text(cfg: ResidualConfig, n_in: Optional[int] = None, first_gain: Optional[Dict] = None) -> str:
    """Which rough rule is in force (ResidualConfig.rough_cut_grade): '' without the feature phase."""
    if not cfg.feature_phase:
        return ""
    if cfg.rough_cut_grade <= 0 and cfg.rough_min_gain_share is None:
        return "rough rule idle (rough_cut_grade 0)"
    g1 = (first_gain or {}).get("cut")
    text = (f"rough rule: a cut gains >= {cfg.rough_cut_grade:g} x the first cut's gain"
            + (f" ({g1:.0f} voxels)" if g1 else "") + ", an add the same of the first add's, single primitives first")
    if cfg.rough_min_gain_share is not None:
        text += f", and >= {cfg.rough_min_gain_share:g} of the input" + (
            f" ({int(round(cfg.rough_min_gain_share * n_in))} voxels)" if n_in else "")
    return text + (f", at most {cfg.rough_max_ops} operations" if cfg.rough_max_ops is not None else "")


def _residual_phase(state, field, result, cfg, registry, purity_min, max_ops, n_in, name, t_start):
    """Steps 3-9 of the module docstring: the add/cut loop on the leftovers, the classic solution (legacy path), the
    voxel-cube fallback or the feature phase, the FreeCAD and exact grade, the statistics vector."""
    guard = _op_guard(cfg)
    rough = cfg.feature_phase and (cfg.rough_cut_grade > 0 or cfg.rough_min_gain_share is not None)  # rough_cut_grade
    # adds: the largest add so far - the member sub-loop's adds (the frame: 40 of them, before this phase and never
    # graded) are not all of the first one's size; the scale is the biggest member, not whichever came first
    first_gain = dict(add=max((float(o.net_gain) for o in result.operations if o.accepted and o.op == "add"
                               and o.net_gain > 0), default=None), cut=None)
    single = replace(cfg, more_booleans=False, boolean_tools=False) if rough else cfg
    end_rough = None

    def kind_of(op):
        return "add" if op == "add" else "cut"

    def floor_of(op) -> float:
        """The least gain an operation of this kind needs under the rough rule (0: no rule yet / no phase)."""
        if not rough:
            return 0.0
        g = first_gain[kind_of(op)]
        f = cfg.rough_cut_grade * g if g else 0.0
        if cfg.rough_min_gain_share is not None:
            f = max(f, cfg.rough_min_gain_share * n_in)
        return f
    tried = set()
    stopped = ""
    blind = getattr(state, "blind", None)
    if blind is None:
        blind = state.blind = _grid_blindness(state, field)
    result.grid_pitch_over_tau = blind
    tree = cfg.mode == "tree"
    if tree:                                                        # the decision tree's rounds (residual_tree.py)
        from reco_engine import residual_tree
        stopped = yield from residual_tree.tree_loop(sys.modules[__name__], state, field, result, cfg, registry, purity_min,
                                                     max_ops, n_in, floor_of, kind_of, first_gain if rough else None, name)
    while not tree:
        low = state.memory_guard.check() if getattr(state, "memory_guard", None) is not None else None
        if low:
            stopped = f"{low}: the loop stops with the model it has"
            yield ResidualStep("rejected", _snapshot(result), dict(op=None, bbox=None, reason=low, message=stopped))
            break
        pos, neg = state.residual()
        size = int(pos.sum() + neg.sum())
        result.residual_voxels_pos, result.residual_voxels_neg = int(pos.sum()), int(neg.sum())
        result.recut_voxels = int(np.count_nonzero(state.cut_history >= 2))
        yield ResidualStep("residual", _snapshot(result), dict(pos=pos, neg=neg, size=size, cut_history=state.cut_history.copy(),
                                                               message=f"|R| = {size} (+{int(pos.sum())} / -{int(neg.sum())})"))
        if size <= cfg.tolerance * n_in:
            stopped = "residual within tolerance"
            break
        if len([o for o in result.operations if o.accepted]) >= max_ops:
            stopped = "operation budget used"
            break
        if rough and cfg.rough_max_ops is not None and len([o for o in result.operations if o.accepted]) >= cfg.rough_max_ops:
            stopped = f"rough operation budget ({cfg.rough_max_ops}) used"
            break
        min_piece = max(cfg.min_component, int(round(cfg.rough_min_piece_share * n_in)))
        comps = [("add", c) for c in _components(pos, max(min_piece, int(np.ceil(floor_of("add")))))] + \
                [("cut", c) for c in _components(neg, max(min_piece, int(np.ceil(floor_of("cut")))))]
        comps = [(op, c) for op, c in comps if _signature(op, c) not in tried]
        if not comps and rough:                                     # what is left is below the grade: say so
            under = [(int(c.sum()), op, c) for op, m in (("cut", neg), ("add", pos)) if first_gain[kind_of(op)]
                     for c in _components(m, min_piece)[:1]]
            if under:
                n_vox, op, c = max(under, key=lambda t: t[0])
                g1 = first_gain[kind_of(op)]
                grade = n_vox / g1
                end_rough = (f"rough: grade <= {grade:.3f} of the first {kind_of(op)} ({g1:.0f}) < "
                             f"{cfg.rough_cut_grade:g}: left for the feature phase")
                lo, hi = _bbox(c)
                yield ResidualStep("rejected", _snapshot(result), dict(
                    op=op, bbox=(lo, hi), primitive=None, gain=None, voxels=n_vox, grade=grade, first_cut_gain=g1,
                    reason=end_rough, message=f"{op} component of {n_vox} voxels: {end_rough}"))
                stopped = end_rough
                break
        if not comps:
            stopped = ("no component left that a primitive explains" if min_piece <= cfg.min_component
                       else f"no piece of >= {min_piece} voxels left (the rest is for the feature "
                            f"{'phase' if rough else 'stage'})")
            break
        if not guard:                                               # one op kind before the other (cfg.cuts_first)
            first = "cut" if _cuts_first(cfg) else "add"
            comps.sort(key=lambda t: (t[0] != first, -int(t[1].sum())))
        best = None                                                 # (gain, op, prim, eval, comp)
        for op, comp in comps:
            lo, hi = _bbox(comp)
            yield ResidualStep("component", _snapshot(result), dict(op=op, mask=comp, bbox=(lo, hi), voxels=int(comp.sum()),
                                                                    message=f"{op} component: {int(comp.sum())} voxels"))
            floor = floor_of(op)
            use_single = rough                                          # single primitives first, the first cut too
            with _field_backend(cfg.field_backend):                     # the pieces' voxel channels too (a GPU
                if cfg.mode == "search":                                # the dumb search: every base shape by coverage
                    from reco_engine import shape_search
                    with timed("shape search"):
                        chosen = shape_search.best_candidate(state, comp, op, field, cfg, purity_min, _evaluate)
                else:
                    chosen = _best_candidate(state, comp, op, field, registry, single if use_single else cfg,
                                             purity_min)                # brute force can stall)
                if cfg.mode != "search" and use_single and first_gain[kind_of(op)] is not None and \
                        (chosen["best"] is None or chosen["best"][1]["gain"] < floor):   # never for the first cut / add
                    full = _best_candidate(state, comp, op, field, registry, cfg, purity_min)   # composites, tools
                    chosen = dict(full, tried=chosen["tried"] + full["tried"])
            chosen["best"] = _unsevered_best(state, result, cfg, chosen, op)      # D5 exact (the FreeCAD build)
            for prim, ev in chosen["tried"]:
                c_op = ev.get("op", op)
                yield ResidualStep("candidate", _snapshot(result), dict(
                    op=c_op, primitive=prim, gain=ev["gain"], ok=ev["ok"], reason=ev["reason"],
                    surface=ev.get("surface"),
                    message=f"{c_op} {_what(prim)}: gain {ev['gain']}"
                            + (f", {ev['surface']:.0%} of the nearby surface on it" if ev.get("surface") is not None else "")
                            + (f", {ev['reason']}" if ev["reason"] else "")))
            if chosen["best"] is None:
                tried.add(_signature(op, comp))
                yield ResidualStep("rejected", _snapshot(result), dict(op=op, bbox=(lo, hi), message=f"{op}: {chosen['why']}"))
                continue
            prim, ev = chosen["best"]
            if rough and floor and ev["gain"] < floor:                  # below the grade: the rough stage ends here
                k_op = kind_of(ev.get("op", op))
                grade = ev["gain"] / first_gain[k_op]
                end_rough = (f"rough: grade {grade:.3f} of the first {k_op} ({first_gain[k_op]:.0f}) < "
                             f"{cfg.rough_cut_grade:g}: left for the feature phase")
                yield ResidualStep("rejected", _snapshot(result), dict(
                    op=op, bbox=(lo, hi), primitive=prim, gain=ev["gain"], grade=grade, first_cut_gain=first_gain[k_op],
                    reason=end_rough, message=f"{op} {_what(prim)}: {end_rough}"))
                break
            if best is None or ev["gain"] > best[0]:
                best = (ev["gain"], ev.get("op", op), prim, ev, comp)
            if not guard:
                break                                               # default order: the first acceptable component is taken
        if end_rough:
            stopped = end_rough
            break
        if best is None:
            stopped = "no component left that a primitive explains"
            break
        gain, op, prim, ev, comp = best
        grade_info = {}
        if rough:                                                   # the grade of every rough operation, kept with it
            k_op = kind_of(op)
            if first_gain[k_op] is None:
                first_gain[k_op] = float(gain)
            grade = float(gain) / first_gain[k_op] if first_gain[k_op] else 1.0
            grade_info = dict(grade=grade, first_cut_gain=first_gain[k_op])
            prim.sources = list(prim.sources) + [StageSource("dimensions", "rough_grade", float(min(grade, 1.0)),
                                                             detail=dict(grade=grade, first_gain=first_gain[k_op], op=k_op))]
        recut = state.commit(ev["s_new"], ev["s_prim"], op)
        if op in ("add", "cut") and prim.kind != "halfspace":
            state.tools.append(prim)                                      # reusable as a tool on later pieces
        result.operations.append(Operation(op, prim, component_voxels=int(comp.sum()), net_gain=float(gain),
                                           recut_voxels=recut))
        if cfg.freecad and cfg.grade_each_step:
            _grade(result, field, cfg, f"{name}_step{len(result.operations)}")
        yield ResidualStep("accepted", _snapshot(result), dict(op=op, primitive=prim, gain=gain, recut=recut,
                                                               cut_history=state.cut_history.copy(), **grade_info,
                                                               message=f"{op} {_what(prim)}: |R| -{gain}"
                                                                       + (f", grade {grade_info['grade']:.3f}"
                                                                          if grade_info else "")
                                                                       + (f", {recut} voxels cut again" if recut else "")))

    if cfg.repair if cfg.repair is not None else tree:                # the repair pass (reco_engine/repair.py)
        from reco_engine.repair import repair as repair_pass
        with timed("repair"):
            dropped = repair_pass(state, result, tolerance=cfg.repair_tolerance, max_loss=cfg.repair_max_loss)
        if dropped:
            stopped += f"; repair: {len(dropped)} operation(s) taken back"
            for i, reason, before, after in dropped:
                yield ResidualStep("repair", _snapshot(result), dict(index=i, op=result.operations[i].op,
                                                                     primitive=result.operations[i].primitive, iou_before=before,
                                                                     iou_after=after, reason=reason,
                                                                     message=f"repair: {result.operations[i].op} {_what(result.operations[i].primitive).split(';')[0]} {reason}"))
    if getattr(state, "trial", False):                                # a solution computed for the comparison
        return stopped                                                # (solution_choice.py): its loop only
    # other solutions before the fallback (the user's rule 2026-09-29): the classic solution, the better one is kept
    if cfg.solution_alternatives != "never" and getattr(state, "solution_ctx", None) is not None:
        from reco_engine import solution_choice
        stopped = yield from solution_choice.before_fallback(state, field, result, cfg, registry, purity_min, n_in, name,
                                                             t_start, stopped)

    # 6. the last resolution: voxel cubes for what no primitive explained
    n_boxes = 0
    if cfg.voxel_fallback and not cfg.feature_phase:                   # the feature phase replaces the fallback
        pos, neg = state.residual()
        enough = state.size() <= cfg.tolerance * n_in                 # the loop's own stop, checked before every box
        min_piece = max(cfg.min_component, int(round(cfg.rough_min_piece_share * n_in)))   # the rough stage's scale
        for op, mask in (("add", pos), ("cut", neg)):
            for comp in _components(mask, min_piece):
                # the biggest filled box first, grown to its full extent, bounded cost (the user's rule 2026-10-15: the
                # fewest elements; largest_boxes searches on blocks of <= 48 per side where voxel_boxes was exact on
                # the whole crop - cubic in the component's extent)
                for blo, bhi in largest_boxes(comp, count=cfg.max_fallback_boxes - n_boxes, max_side=48,
                                              min_voxels=max(2, min_piece // 4)):
                    if enough or n_boxes >= cfg.max_fallback_boxes:
                        break
                    prim, ev = _fallback_box(state, blo, bhi, op, field, registry, cfg, purity_min)
                    if prim is None:
                        continue
                    gain = ev["gain"]
                    recut = state.commit(ev["s_new"], ev["s_prim"], op)
                    n_boxes += 1
                    enough = state.size() <= cfg.tolerance * n_in
                    result.operations.append(Operation(op, prim, component_voxels=int(comp.sum()), net_gain=float(gain),
                                                       recut_voxels=recut))
                    yield ResidualStep("fallback", _snapshot(result), dict(op=op, primitive=prim, gain=gain, recut=recut,
                                                                           cut_history=state.cut_history.copy(),
                                                                           message=f"fallback {op} {n_boxes}/{cfg.max_fallback_boxes}: {prim.sources[1].mode} box for {int(np.prod(bhi - blo + 1))} voxels, |R| = {state.size()}"))
                if enough or n_boxes >= cfg.max_fallback_boxes:
                    break
            if enough or n_boxes >= cfg.max_fallback_boxes:
                break
        if n_boxes:
            stopped += f"; {n_boxes} fallback boxes" + (" to the tolerance" if enough else "")
    # 7b. merged cuts (ResidualConfig.cut_merge): two box cuts of one frame whose common box holds no input material
    # (the input mesh itself, reco_engine/cut_merge.py) and with no add between them reaching into it are one cut
    if cfg.cut_merge and field is not None and sum(o.op == "cut" and o.accepted for o in result.operations) >= 2:
        try:
            from reco_engine.cut_merge import merge_cuts
            with timed("merge cuts"):
                acc = [o for o in result.operations if o.accepted]
                merged, rep = merge_cuts([(o.op, o.primitive) for o in acc], field.vertices, field.faces)
            if rep["merges"] or rep["dropped"]:
                by_prim = {id(o.primitive): o for o in acc}
                result.operations = [by_prim[id(pr)] if id(pr) in by_prim else
                                     Operation(op_, pr, component_voxels=0, net_gain=0.0) for op_, pr in merged]
                for o in result.operations:
                    if id(o.primitive) not in by_prim:
                        o.primitive.label = (o.primitive.label or "box") + ", merged cuts"
                yield ResidualStep("solution", _snapshot(result), dict(
                    merged_cuts=rep, kept="merged cuts", solutions=[], measure="the input mesh's inside test (cut_merge)",
                    message=f"cuts merged: {len(rep['merges'])} pair(s) into one cut each, "
                                             f"{len(rep['dropped'])} redundant cut(s) dropped - {len(acc)} -> "
                                             f"{len(result.operations)} operations"))
                stopped += f"; {len(rep['merges'])} cut pair(s) merged"
        except Exception as exc:                                     # a simplification must never end the run
            stopped += f"; merging cuts failed: {str(exc)[-80:]}"
    if cfg.cut_tidy and field is not None:
        stopped += yield from _tidy_cuts(result, field, state)
    # 8. the feature phase: the exact difference of the built program, a primitive per piece, validated in FreeCAD
    if cfg.feature_phase:
        feats = yield from _feature_phase(result, cfg, result.input_path, registry, field=field, state=state)
        stopped += f"; feature phase: {feats['stopped']}, {feats['applied']} applied"
        if cfg.cut_tidy and field is not None:
            stopped += yield from _tidy_cuts(result, field, state)
    # 9. the polish: the curved primitives refitted to the input's vertices (before the FreeCAD build and the grade)
    polish = None
    if cfg.polish and [o for o in result.operations if o.accepted]:
        from reco_engine.polish import polish_result
        with timed("polish"):
            polish = polish_result(result, field.vertices, field.faces)
        if polish.kept:                                                # nothing refitted: the run reads as before
            stopped += f"; polish: {polish.summary()}"
    # 10. the prune: the parts the shape does not need dropped (one valid solid checked in FreeCAD)
    if cfg.prune_parts and field is not None and len([o for o in result.operations if o.accepted]) > 1:
        try:
            with timed("prune"):
                stopped += yield from _prune_parts(result, field, cfg, name)
        except Exception as exc:                                     # a simplification must never end the run
            stopped += f"; prune failed: {str(exc)[-80:]}"
    pos, neg = state.residual()
    result.residual_voxels_pos, result.residual_voxels_neg = int(pos.sum()), int(neg.sum())
    result.recut_voxels = int(np.count_nonzero(state.cut_history >= 2))

    if blind is not None and blind > 1.0:
        stopped += (f" (the grid is blind here: pitch {blind:.1f} x the thinnest wall, so voxel residuals below it do not "
                    f"show; see the exact grade)")
    grade = None
    if cfg.freecad:
        grade = _grade(result, field, cfg, name)
        yield ResidualStep("grade", _snapshot(result), dict(grade=grade, message=f"FreeCAD: valid {grade.get('valid')}, "
                                                                                 f"chamfer {grade.get('chamfer')}"))
    from reco_engine.solution import solution_stats
    with timed("solution statistics"):
        stats = solution_stats(result, state.occ_in, state.occ_model, getattr(state, "mesh", None), float(state.pitch),
                               grade=grade, seconds=time.time() - t_start)
    if getattr(state, "exact_gate", None):
        stats["exact_gate"] = dict(state.exact_gate)                       # D5 exact: checks, rejected, failed, seconds
    rough_text = _rough_text(cfg, n_in, first_gain)
    eg = getattr(state, "exact_gate", None)
    if eg:
        rough_text = (rough_text + "; " if rough_text else "") + \
            f"D5 exact: {eg['checks']} build(s), {eg['rejected']} refused, {eg['failed']} without answer, {eg['seconds']:.0f} s"
    yield ResidualStep("done", _snapshot(result), dict(cut_history=state.cut_history.copy(), stopped=stopped, seconds=time.time() - t_start, grade=grade,
                                                       input_voxels=n_in, fallback_boxes=n_boxes, stats=stats,
                                                       rough_rule=rough_text or None,
                                                       members=(getattr(state, "member_report", None) or {}).get("store"),
                                                       solution=(getattr(state, "solution_ctx", None) or {}).get("first"),
                                                       solution_account=(getattr(state, "solution_ctx", None) or {}).get("account"),
                                                       polish=polish.as_dict() if polish is not None else None,
                                                       message=f"done: {stopped}{'; ' + rough_text if rough_text else ''}; IoU {stats['iou']:.4f}, coverage {stats.get('coverage', float('nan')):.3f}, "
                                                               f"score {stats['score']:.3f}"))


# ------------------------------------------------------------------------------------------------ the feature phase

FEATURE_MESH_TRIANGLES = 20000     # a piece's mesh in a feature_piece step (the GUI drops arrays over 100k elements)
FEATURE_CANDIDATES_SHOWN = 6       # feature_candidate steps per piece, best first


def _feature_build_default(result: ReconstructionResult, out_dir, name: str, timeout: int) -> Dict:
    from reco_field.export import build_in_freecad
    with timed("feature build"):
        ex = build_in_freecad(program_model(result), out_dir, name, timeout=timeout)
    return dict(valid=bool(ex.valid), n_solids=int(ex.n_solids), step=str(ex.step), stl=str(ex.stl), volume=float(ex.volume))


def _feature_difference_default(input_path, model_step, pieces_dir, max_pieces: int, min_solid, timeout: int,
                                state_dir=None) -> Dict:
    from reco_field.exact_check import exact_residual
    with timed("feature difference"):
        return exact_residual(input_path, model_step, timeout=timeout, min_solid=min_solid, pieces_dir=pieces_dir,
                              max_pieces=max(1, int(max_pieces)), state_dir=state_dir)


def _feature_incremental_default(input_path, model_step, state, tools, state_dir, pieces_dir, max_pieces: int,
                                 min_solid, timeout: int) -> Dict:
    from reco_field.exact_check import incremental_residual
    with timed("feature difference/incremental"):
        return incremental_residual(input_path, model_step, state, tools, state_dir=state_dir, timeout=timeout,
                                    min_solid=min_solid, pieces_dir=pieces_dir, max_pieces=max(1, int(max_pieces)))


def _feature_guess_default(piece: Dict, sources, deadline: float, registry, models) -> Dict:
    from reco_engine.feature_stage import guess_candidates, segment_candidates
    if piece.get("segment") is not None:                        # a segment of a split piece: its voxels and field
        with timed("feature guess/segment"):
            return segment_candidates(piece["segment"], registry, models, sources, deadline=deadline)
    sc = piece.get("scoring") or {}
    with timed("feature guess"):
        return guess_candidates(piece, registry, models, sources, deadline=deadline, op=sc.get("op"),
                                inside_input=sc.get("inside_input"), inside_region=sc.get("inside_region"),
                                tools=sc.get("tools"), typed=sc.get("typed"), contact=sc.get("contact"),
                                contact_faces=bool(sc.get("contact_faces")), thin=float(sc.get("thin") or 0.0),
                                part_diagonal=sc.get("part_diagonal"))


def _cluster_decimate(V: np.ndarray, F: np.ndarray, max_faces: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering decimation: vertices merged per cell of a grid over the mesh's box (the cell's mean), faces
    that collapse dropped, the grid coarsened by 1.5 until at most `max_faces` remain. A random face subsample would
    keep up to three vertices per face (60k for 20k triangles, over the GUI's array limit) and leave holes."""
    lo, size = V.min(axis=0), float(np.ptp(V, axis=0).max()) or 1.0
    n = 256.0
    while True:
        key = np.floor((V - lo) / (size / n)).astype(np.int64)
        _, cell, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        cell = cell.ravel()
        Vc = np.zeros((len(counts), 3))
        np.add.at(Vc, cell, V)
        Vc /= counts[:, None]
        Fc = cell[F]
        Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])]
        Fc = np.unique(Fc, axis=0)
        if len(Fc) <= max_faces or n <= 4:
            used, inv = np.unique(Fc, return_inverse=True)
            return Vc[used], inv.reshape(-1, 3)
        n /= 1.5


def _feature_piece_mesh(stl) -> Tuple[Optional[Dict], bool]:
    """(dict(vertices float32, faces int32), truncated) of a piece's STL for display, at most FEATURE_MESH_TRIANGLES
    triangles (decimated above that, `_cluster_decimate`); (None, False) without a readable file."""
    if not stl or not Path(str(stl)).is_file():
        return None, False
    try:
        import trimesh
        m = trimesh.load(str(stl), process=True, force="mesh")
        V, F = np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)
    except Exception:
        return None, False
    truncated = len(F) > FEATURE_MESH_TRIANGLES
    if truncated:
        V, F = _cluster_decimate(V, F, FEATURE_MESH_TRIANGLES)
    return dict(vertices=V.astype(np.float32), faces=F.astype(np.int32)), truncated


def _piece_signature(p: Dict, v_in: float):
    """A piece met again in a later iteration (the model changed elsewhere): its sign, its centre to 1e-3 of the
    input's size and its volume to 5 %."""
    q = 1e-3 * max(float(v_in), 1e-30) ** (1.0 / 3.0)
    return (p["sign"], tuple(int(round(float(x) / q)) for x in p["centre"]),
            int(round(20.0 * np.log(max(float(p["volume"]), 1e-30)))))


def _mesh_volume(field) -> Optional[float]:
    if field is None or getattr(field, "vertices", None) is None:
        return None
    tri = np.asarray(field.vertices, float)[np.asarray(field.faces)]
    return abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


def _feature_phase(result: ReconstructionResult, cfg: ResidualConfig, input_path, registry, field=None, state=None,
                   out_dir=None):
    """The feature phase (ResidualConfig.feature_phase; a generator of ResidualSteps of FEATURE_KINDS, returning the
    info of its "feature_done" step). Per iteration:

      1. difference  the current program built in FreeCAD, its exact difference to the input by OCCT: missing (input -
                     model, adds) and invented (model - input, cuts); every solid one piece
      2. ranking     the biggest first; pieces under feature_min_share of the input volume (unless known or a
                     feature by their own size: feature_keep_known, feature_min_radius_depth), pieces already tried and
                     pieces over feature_max_pieces (per iteration) / feature_max_total_pieces (phase) are dropped
      3. guess       per piece the scored candidates of feature_stage.guess_candidates, the best one's ends extended
                     where that changes nothing of the part (feature_stage.extend_ends: no coplanar faces for OCCT)
      4. apply       a best guess scoring >= feature_min_score joins the program (sources "feature:<source>")
      5. validate    one FreeCAD build + one exact difference for the iteration's guesses (feature_batch; halved on a
                     failure down to the operation that fails): one valid solid (no worse than the rough model if that
                     was not one already) and the residual down by >= feature_min_gain x the pieces' volume, else reverted
      6. iterate     until converged, nothing applied, or a limit

    Never raises: an error ends the phase with stopped "error: ..."; one piece failing is a rejection, the phase goes
    on. The voxel state (when given) takes every applied operation too, so the loop's statistics see it."""
    from reco_engine.result_schema import StageSource
    t0 = time.time()
    deadline = t0 + float(cfg.feature_time_budget)
    limits = dict(max_pieces=cfg.feature_max_pieces, max_total_pieces=cfg.feature_max_total_pieces,
                  max_iterations=cfg.feature_max_iterations, max_applied=cfg.feature_max_applied, time_budget=cfg.feature_time_budget,
                  call_timeout=cfg.feature_call_timeout, min_share=cfg.feature_min_share, min_score=cfg.feature_min_score,
                  min_gain=cfg.feature_min_gain, tolerance=cfg.feature_tolerance, batch=cfg.feature_batch,
                  split=cfg.feature_split, split_min_share=cfg.feature_split_min_share,
                  max_segments=cfg.feature_max_segments, segment_min_voxels=cfg.feature_segment_min_voxels,
                  split_depth=cfg.feature_split_depth, min_precision=cfg.feature_min_precision,
                  min_recall=cfg.feature_min_recall)
    tally = dict(iterations=0, applied=0, rejected=0, skipped=0, differences=0, groups=0, transferred=0,
                 transfer_fallbacks=0, renewed=0, split_pieces=0, segments=0, segments_accepted=0, tools={},
                 group_retries=0, voxel_fallbacks=0, voxel_fallbacks_applied=0, subparts=0, subparts_applied=0,
                 methods={})
    from reco_engine.tools import ToolStore
    store_path = cfg.feature_tool_store if isinstance(cfg.feature_tool_store, (str, Path)) else None
    store = cfg.feature_tool_store if isinstance(cfg.feature_tool_store, ToolStore) else \
        (ToolStore.load(store_path) if store_path else ToolStore())
    store_before = len(store)
    holder = dict(group_source={}, group_tool={}, group_params={}, ptype={}, desc={}, net=None, skip_groups=set(),
                  tried={}, best={}, rejected_ids=set(), accepted=[], symmetry=None)
    hooks_note = ""
    if cfg.feature_symmetry and field is not None:                  # always look for symmetric features (2026-10-29)
        try:
            from reco_engine import symmetry as SY
            holder["symmetry"] = SY.detect(np.asarray(field.vertices, float), np.asarray(field.faces))
        except Exception as exc:                                     # a check, never a reason to stop
            hooks_note += f"; symmetry not checked: {str(exc)[-120:]}"
    if cfg.feature_net:                                              # the FeatureNet hook (its module: another session)
        try:
            from reco_engine import feature_net                     # noqa: F401
            holder["net"] = feature_net.load(cfg.feature_net) if hasattr(feature_net, "load") else None
        except ImportError:
            hooks_note += "; feature_net given but reco_engine/feature_net.py is not on this branch"
    feedback = _load_piece_feedback(cfg.piece_feedback)
    adviser, budget, n_q = None, None, [0]
    if cfg.adviser is not None:                                      # reco_engine/advice.py's contract
        from reco_engine import advice as AD
        adviser, budget = cfg.adviser, AD.Budget(cfg.advice)
    from reco_engine.feature_stage import sampled_score as score_fn
    now = dict(missing=None, invented=None, iou=None)
    build = cfg.feature_build or _feature_build_default
    difference = cfg.feature_difference or _feature_difference_default
    incremental = cfg.feature_incremental_difference if cfg.feature_incremental else None
    if incremental is None and cfg.feature_incremental and cfg.feature_difference is None:
        incremental = _feature_incremental_default
    guess = cfg.feature_guess or _feature_guess_default
    path = str(input_path) if input_path is not None else ""
    suffix = Path(path).suffix.lower()
    sources = tuple(cfg.feature_sources) if cfg.feature_sources else \
        (("fit", "cascade") if suffix == ".stl" else ("step faces", "fit", "cascade"))

    def step(kind, **info):
        info["phase"] = "feature"
        return ResidualStep(kind, _snapshot(result), info)

    records: Dict[int, Dict] = {}                                   # per piece / segment: what the document shows

    def note(pid, **kw):
        records.setdefault(pid, {}).update({k: v for k, v in kw.items() if v is not None})

    def rejected(it, piece, reason, op=None, prim=None, source=None, score=None, validation=None):
        if piece["id"] not in holder["rejected_ids"]:                # a piece counts once (its fallback may fail too)
            tally["rejected"] += 1
        holder["rejected_ids"].add(piece["id"])
        note(piece["id"], verdict="rejected", reason=reason, source=source, score=score,
             kind=getattr(prim, "kind", None), best=prim)
        return step("feature_rejected", iteration=it, piece_id=piece["id"], op=op or piece["op"], primitive=prim,
                    source=source, score=score, reason=reason, validation=validation, **tool_info(piece, source),
                    message=f"piece #{piece['id']}: rejected - {reason}")

    inside = None
    inside_part = lambda pts: np.zeros(len(pts), bool)              # noqa: E731  (no input mesh: nothing to harm)
    if field is not None and getattr(field, "vertices", None) is not None:
        from reco_engine.feature_stage import _inside_mesh, input_inside
        V_in, F_in = np.asarray(field.vertices, float), np.asarray(field.faces)
        inside = lambda pts: _inside_mesh(V_in, F_in, pts)          # noqa: E731
        inside_part = input_inside(V_in, F_in)
    seg_sources = tuple(x for x in sources if x != "step faces" or suffix in (".step", ".stp"))
    judge_cache = {}

    def judge_fn():
        """The sampled judge (feature_stage.SampledJudge) over the input, built on first use; None without a mesh."""
        if "j" not in judge_cache:
            judge_cache["j"] = None
            if inside is not None:
                from reco_engine.feature_stage import SampledJudge
                with timed("feature sampled judge"):
                    judge_cache["j"] = SampledJudge(inside_part, V_in.min(axis=0), V_in.max(axis=0))
        return judge_cache["j"]

    def commit(g, it):
        """One validated operation into the program (and the voxel state)."""
        prim, op = g["prim"], g["op"]
        gain = 0.0
        if state is not None:
            try:
                s_c = state.prim_sdf(prim)
                new = state.apply(state.s_model, s_c, op)
                gain = float(state.size() - state.size(new))
                state.commit(new, s_c, op)
            except Exception:                                        # the voxel view is secondary to the exact one
                pass
        result.operations.append(Operation(op, prim, component_voxels=0, net_gain=gain, freecad_valid=True))

    def make_pending(d, prim, source, score, it):
        """A guess that passed: its ends extended where that changes nothing, named by its source."""
        from reco_engine.feature_stage import extend_ends
        prim = copy.deepcopy(extend_ends(prim, d["op"], inside))
        prim.alternatives = []
        prim.sources = list(prim.sources) + [StageSource(
            "type", f"feature:{source}", float(score),
            detail=dict(piece=d["id"], side=d["side"], volume=d["volume"], iteration=it))]
        prim.label = ((prim.label + ", ") if prim.label else "") + f"feature ({source})"
        return dict(piece=d, op=d["op"], prim=prim, source=source, score=float(score))

    def tool_info(d, source=None, tool_type=None):
        """tool_type, group, tool_source of a step: the piece's predicted type, its group, where its tool came from."""
        ts = None
        if source:
            ts = ("session" if source.startswith("session:") else "catalogue" if source.startswith("catalogue:")
                  else "created" if source.startswith("tool (") else
                  "fallback" if source.startswith("voxel fallback") else
                  "sub-parts" if source.startswith("sub-parts") else
                  holder["group_source"].get(d.get("group")) if source.startswith("pattern:") else None)
        return dict(tool_type=tool_type or d.get("tool_type"), group=d.get("group"), tool_source=ts)

    def piece_step(d, it, k, n):
        mesh, truncated = _feature_piece_mesh(d["stl"])
        return step("feature_piece", iteration=it, piece=d, index=k, of=n, mesh=mesh, mesh_truncated=truncated,
                    **tool_info(d),
                    message=f"piece {k}/{n} (#{d['id']}): {d['side']} {d['volume']:.4g} ({d['share']:.3%} of the input) "
                            f"-> {d['op']}" + (f", {d['tool_type']}" if d.get("tool_type") else "")
                            + (f", congruent with #{d['group']}" if d.get("group") != d["id"] else ""))

    def candidate_step(it, d, c):
        pr, rc = getattr(c, "precision", None), getattr(c, "recall", None)
        ct = getattr(c, "contact", None) or {}
        if records.get(d["id"], {}).get("score") is None or float(c.score) > records[d["id"]]["score"]:
            note(d["id"], kind=c.primitive.kind, source=c.source, score=float(c.score), precision=pr, recall=rc)
        return step("feature_candidate", iteration=it, piece_id=d["id"], op=d["op"], primitive=c.primitive,
                    source=c.source, score=float(c.score), seconds=float(c.seconds), precision=pr, recall=rc,
                    contact_area=ct.get("contact_area"), free_area=ct.get("free_area"), harm_area=ct.get("harm_area"),
                    contact_recall=ct.get("contact_recall"), contact_precision=ct.get("contact_precision"),
                    **tool_info(d, c.source, getattr(c, "tool_type", None)),
                    message=f"  {c.source}: {d['op']} {_what(c.primitive)}, score {c.score:.3f}"
                            + (f" (precision {pr:.3f}, recall {rc:.3f})" if pr is not None and rc is not None else "")
                            + (f" [contact {ct['contact_area']:.4g} mm2, free {ct['free_area']:.4g}, harm {ct['harm_area']:.3g}; "
                               f"contact recall {ct['contact_recall']:.3f}, precision {ct['contact_precision']:.3f}]" if ct else "")
                            + f" ({c.seconds:.1f} s)")

    def tried_rows(d, cands, failed=None):
        """What a piece's guess tried, for the fallback's step: per source its best candidate's kind, score, precision
        and recall, and the sources that gave nothing."""
        rows = {}
        for c in cands:
            rows.setdefault(c.source, dict(solution=c.source, kind=c.primitive.kind, score=float(c.score),
                                           precision=getattr(c, "precision", None), recall=getattr(c, "recall", None)))
        holder["tried"].setdefault(d["id"], []).extend(
            list(rows.values()) + [dict(solution=k2, failed=str(v)[-120:]) for k2, v in (failed or {}).items()])
        for c in cands:                                             # the rough guess's candidate: the best by score
            prev = holder["best"].get(d["id"])
            if prev is None or float(c.score) > float(prev.score):
                holder["best"][d["id"]] = c

    def tried_text(rows):
        def one(r):
            if r.get("members") is not None:
                return f"{r['solution']} ({', '.join('#' + str(m) for m in r['members'])})"
            if r.get("failed") or r.get("note"):
                return f"{r['solution']}: {r.get('failed') or r.get('note')}"
            pr, rc = r.get("precision"), r.get("recall")
            return (f"{r['solution']} {r.get('kind') or ''}".rstrip() + (f" score {r['score']:.3f}" if r.get("score") is not None else "")
                    + (f" (precision {pr:.3f}, recall {rc:.3f})" if pr is not None and rc is not None else ""))
        return "; ".join(one(r) for r in rows) or "nothing"

    def advise(d, cands, copies=0, giving_up=False):
        """Ask the adviser about a hard piece (advice.hardness decides, the Budget limits); returns (Answer or None,
        the candidates in the question's order). Steps advice_asked (before ask) and advice_given."""
        if adviser is None:
            return None, []
        from reco_engine import advice as AD
        shown = sorted(cands, key=lambda c: -c.score)[:(cfg.advice or AD.AdviceConfig()).max_candidates]
        rows = [dict(kind=c.primitive.kind, op=d["op"], score=c.score, precision=getattr(c, "precision", None),
                     recall=getattr(c, "recall", None), source=c.source, primitive=c.primitive,
                     label=f"{d['op']} {getattr(c, 'tool_type', None) or _what(c.primitive)} ({c.source})") for c in shown]
        n_q[0] += 1
        q = AD.hardness(dict(id=d["id"], share=d["share"], side=d["side"], op=d["op"], volume=d["volume"], bbox=d["bbox"]),
                        rows, cfg.feature_min_score, cfg.advice, copies=copies, giving_up=giving_up,
                        precision_threshold=cfg.feature_min_precision, question_id=f"q{n_q[0]}")
        if q is None or not budget.spend(q):
            return None, shown
        mesh, _ = _feature_piece_mesh(d.get("stl"))
        yield step("advice_asked", question=q.as_dict(), piece_id=d["id"], mesh=mesh, candidates=[c.primitive for c in shown],
                   message=q.text)
        a = adviser.ask(q)
        usable = a.choice not in ("adjusted", "voxels") and not (
            a.choice == "candidate" and not (0 <= int((a.parameters or {}).get("index", -1)) < len(shown)))
        yield step("advice_given", answer=a.as_dict(), question_id=q.id, source=a.source,
                   message=f"advice on piece #{d['id']}: {a.choice} ({a.source})"
                           + ("" if usable else " - not usable in the feature phase, taken as the default"))
        if not usable:
            a = q.default_answer(source=a.source, note="not usable in the feature phase")
        return a, shown

    def act(a, shown, d, p, it):
        """What an answer does to a piece: (pending or None, a verdict step already given?, "stop" / None)."""
        ch = a.choice
        if ch == "candidate":
            c = shown[int(a.parameters["index"])]
            pend = make_pending(d, c.primitive, f"{c.source} (chosen by the user)", c.score, it)
            pend.update(tool_type=getattr(c, "tool_type", None), tool_params=getattr(c, "tool_params", None))
            return pend, False, None
        if ch == "tool":
            from reco_engine import tools as TL
            V = piece_vertices(p)
            fits = TL.fit_piece(p, V, d["op"], types=[a.parameters.get("class")]) if V is not None else []
            if fits:
                lo, hi = (np.asarray(b, float) for b in d["bbox"])
                pend = make_pending(d, fits[0].build(0.02 * float(np.linalg.norm(hi - lo))),
                                    f"catalogue:{fits[0].tool} (named by the user)", 1.0, it)
                pend.update(tool_type=fits[0].tool, tool_params={k: v for k, v in fits[0].params.items() if not k.startswith("_")})
                return pend, False, None
            return None, f"the user named {a.parameters.get('class')}, which does not fit the piece's faces", None
        if ch in ("skip", "leave"):
            return None, "skipped by the user" if ch == "skip" else "left by the user", None
        if ch == "skip_group":
            holder["skip_groups"].add(d.get("group"))
            return None, "skipped by the user (with its group)", None
        if ch == "stop":
            return None, "stopped by the user", "stop"
        return None, False, None                                     # "default": as the algorithm does by itself

    def full_guess(d, p, it, state_out, may_split=False, typed=None, copies=0):
        """The scored guesses of one piece; returns (the pending entry or None, the best passing Candidate or None).
        `may_split`: a piece whose best guess scores too low is not rejected but returned as (None, "split"). A piece
        the adviser's answer rejects is (None, "left")."""
        try:
            part = cfg.feature_scoring == "part" and inside is not None
            region = None
            if part and holder.get("diff_model") is not None:
                from reco_engine.feature_stage import excess_region
                region = excess_region(d["op"], inside_part, holder["diff_model"], *d["bbox"])
            tools = dict(min_score=cfg.feature_min_score, min_precision=cfg.feature_min_precision,
                         max_operands=cfg.feature_tool_max_operands,
                         max_candidates=cfg.feature_tool_candidates) if part and cfg.feature_tools else None
            contact = dict(V=V_in, F=F_in, inside=inside_part, score=cfg.feature_contact_grading,
                           rank=cfg.feature_contact_ranking) if part else None
            g = guess(dict(p, scoring=dict(op=d["op"], inside_input=inside_part if part else None, inside_region=region,
                                           tools=tools, typed=typed, contact=contact,
                                           contact_faces=bool(part and cfg.feature_contact_tool),
                                           thin=holder.get("thin_threshold") or 0.0,
                                           part_diagonal=holder.get("part_diagonal"))),
                      sources, deadline, registry, cfg.models)
            cands = list(g.get("candidates") or [])
            tried_rows(d, cands, g.get("failed"))
            for c in cands[:FEATURE_CANDIDATES_SHOWN]:
                yield candidate_step(it, d, c)
            state_out["out_of_time"] = state_out["out_of_time"] or bool(g.get("timed_out"))
            if not cands:
                failed = "; ".join(f"{k2}: {v}" for k2, v in (g.get("failed") or {}).items())
                yield rejected(it, d, "no candidate" + (f" ({failed})" if failed else "")
                               + (" - the time budget ended the search" if g.get("timed_out") else ""))
                return None, None
            best = cands[0]
            from reco_engine.feature_stage import precise_enough
            imprecise = not precise_enough(best, cfg.feature_min_precision)
            if (best.score < cfg.feature_min_score or imprecise) and adviser is not None:
                a, shown = yield from advise(d, cands, copies=copies, giving_up=not may_split)
                if a is not None and a.choice != "default":
                    pend, verdict, stop = act(a, shown, d, p, it)
                    if stop:
                        state_out["stop"] = True
                    if pend is not None:
                        return pend, None
                    yield rejected(it, d, verdict if isinstance(verdict, str) else f"advice: {a.choice}")
                    return None, "left"                              # the user's verdict: no fallback either
            if best.score < cfg.feature_min_score or imprecise:
                if imprecise and best.score >= cfg.feature_min_score and not may_split:
                    yield rejected(it, d, f"precision {best.precision:.3f} < {cfg.feature_min_precision}",
                                   prim=best.primitive, source=best.source, score=float(best.score))
                    return None, None
                if may_split:
                    return None, "split"
                yield rejected(it, d, f"best score {best.score:.3f} < {cfg.feature_min_score}", prim=best.primitive,
                               source=best.source, score=float(best.score))
                return None, None
            pend = make_pending(d, best.primitive, best.source, best.score, it)
            pend.update(tool_type=getattr(best, "tool_type", None), tool_params=getattr(best, "tool_params", None))
            if getattr(best, "partial", False) and best.recall is not None:
                pend["need_volume"] = float(best.recall) * float(d["volume"])
            if pend["tool_type"] is None and best.primitive.kind == "composite":
                # a created tool is a catalogue type when its tree has that type's shape AND a catalogue fitter read
                # the type from the piece's faces - the shape alone is too weak (any box - cylinder has a fillet's)
                from reco_engine.tools import catalogue_type_of
                ct = catalogue_type_of(best.primitive)
                pt = holder["ptype"].get(d["id"])
                read = {f.tool for f in pt.fits} if pt is not None else set()
                pend["tool_type"] = ct if ct in read else f"created:{best.primitive.label or 'tool'}"
            holder["group_source"][d.get("group")] = tool_info(d, best.source)["tool_source"]
            holder["group_tool"][d.get("group")] = pend["tool_type"]
            holder["group_params"][d.get("group")] = pend["tool_params"]
            return pend, best
        except Exception as exc:                                     # one piece failing ends nothing
            yield rejected(it, d, f"error: {str(exc)[-200:]}")
            return None, None

    def transfer(d, p, rep_d, rep_desc, desc, best, it, state_out, guess_after=True):
        """The representative's primitive moved onto a congruent piece (feature_stage.transfer_candidates) and scored
        there; below feature_min_score the piece gets a full guess instead (`guess_after`; without it None)."""
        from reco_engine.feature_stage import Candidate, best_transfer
        t = time.time()
        sym = holder.get("symmetry")
        if sym and rep_desc is not None and desc is not None:       # an exact copy by a symmetry of the part first
            from reco_engine import symmetry as SY
            try:
                from src.closest_point_field import load_field
                fld = load_field(p["stl"])
                V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
                tol = float(sym["tol"]) + 0.05 * float(d.get("volume") or 0.0) ** (1.0 / 3.0)
                for name, T in SY.partner_transforms(sym, rep_desc["centre"], desc["centre"], tol)[:3]:
                    q = SY.apply_transform(best.primitive, T)
                    c = Candidate(f"symmetry:{rep_d['id']}:{name}", q, float(score_fn(q, V, F, n=20000)), time.time() - t)
                    tried_rows(d, [c])
                    yield candidate_step(it, d, c)
                    if c.score >= cfg.feature_min_score:
                        tally["symmetric"] = tally.get("symmetric", 0) + 1
                        pend = make_pending(d, q, c.source, c.score, it)
                        pend.update(tool_type=holder["group_tool"].get(d.get("group")),
                                    tool_params=holder["group_params"].get(d.get("group")))
                        return pend
            except Exception as exc:
                state_out["symmetry_error"] = str(exc)[-200:]
        try:
            from src.closest_point_field import load_field
            fld = load_field(p["stl"])
            V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
            sc, q = best_transfer(best.primitive, rep_desc, desc, lambda q2: score_fn(q2, V, F, n=20000))
        except Exception as exc:
            sc, q = 0.0, None
            state_out["transfer_error"] = str(exc)[-200:]
        c = Candidate(f"pattern:{rep_d['id']}", q if q is not None else best.primitive, float(sc), time.time() - t)
        tried_rows(d, [c])
        yield candidate_step(it, d, c)
        if q is not None and sc >= cfg.feature_min_score:
            tally["transferred"] += 1
            pend = make_pending(d, q, f"pattern:{rep_d['id']}", sc, it)
            pend.update(tool_type=holder["group_tool"].get(d.get("group")),
                        tool_params=holder["group_params"].get(d.get("group")))
            return pend
        if not guess_after:
            return None
        tally["transfer_fallbacks"] += 1
        pend, _ = yield from full_guess(d, p, it, state_out)
        return pend

    def rough(d, p, it):
        """The rough guess of a piece nothing else explained (ResidualConfig.feature_rough): the best single candidate
        tried for it, as a pending entry marked rough - the batch validation then wants the exact residual to drop by
        feature_rough_min_drop of the piece, the check allows feature_rough_max_harm. None when that candidate is too
        weak (score, precision, recall)."""
        best = holder["best"].get(d["id"])
        if best is None:
            return None
        pr, rc = getattr(best, "precision", None), getattr(best, "recall", None)
        if float(best.score) < cfg.feature_rough_min_score or (pr or 0.0) < cfg.feature_rough_min_precision \
                or (rc or 0.0) < cfg.feature_min_recall:
            return None
        c = copy.copy(best)
        c.source = f"rough:{best.source}"
        s = candidate_step(it, d, c)
        s.info["rough"] = True
        s.info["message"] = (f"  rough guess for piece #{d['id']}: {d['op']} {best.primitive.kind} from {best.source} "
                             f"(score {float(best.score):.3f}, precision {pr:.3f}, recall {rc:.3f}) - applied if the exact "
                             f"residual drops by {cfg.feature_rough_min_drop:.0%} of the piece "
                             f"({cfg.feature_rough_min_drop * float(d['volume']):.4g}) and it harms at most "
                             f"{cfg.feature_rough_max_harm:.0%} of its volume")
        yield s
        tally["rough"] = tally.get("rough", 0) + 1
        pend = make_pending(d, best.primitive, f"rough:{best.source}", float(best.score), it)
        pend.update(tool_type=None, tool_params=None, rough=True,
                    need_volume=cfg.feature_rough_min_drop / max(cfg.feature_min_gain, 1e-9) * float(d["volume"]))
        return pend

    def fallback(d, p, it):
        """The last resort of a piece (ResidualConfig.feature_voxel_fallback): its material as boxes of voxel cubes
        (feature_stage.voxel_tool), after everything in holder["tried"] failed. Returns the pending entry or None; its
        candidate step (source "voxel fallback") carries `fallback`: the boxes, the pitch, what was tried before and
        what is left of the piece."""
        from reco_engine import feature_stage as fs
        rows = list(holder["tried"].get(d["id"], []))
        before = tried_text(rows)
        last = records.get(d["id"], {}).get("reason")
        t = time.time()
        tally["voxel_fallbacks"] += 1
        try:
            from src.closest_point_field import load_field
            fld = load_field(p["stl"])
            V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
            desc = holder["desc"].get(d["id"]) or fs.piece_descriptor(p)
            frames = dict(world=np.eye(3))
            if desc is not None:
                frames["principal"] = desc["axes"]
            with timed("feature voxel fallback"):
                tau = fs.piece_thickness(fld, holder.get("thin_threshold") or 0.0)["tau"]
                r = fs.voxel_tool(V, F, tau=tau, frames=frames, max_boxes=max(1, cfg.feature_fallback_max_boxes))
                precision = None
                if r is not None:
                    sdf = primitive_to_sp(r["primitive"])
                    lo, hi = (np.asarray(b, float) for b in sdf.bounds())
                    pts = lo + np.random.default_rng(0).random((20000, 3)) * (hi - lo)
                    pts = pts[np.asarray(sdf.sdf(pts)) <= 0]
                    precision = float(fs._inside_mesh(V, F, pts).mean()) if len(pts) else None
        except Exception as exc:
            yield rejected(it, d, f"voxel fallback failed: {str(exc)[-200:]}; tried before: {before}")
            return None
        recall = float(r["covered_volume"]) / float(d["volume"]) if r is not None and d["volume"] > 0 else 0.0
        if r is None or recall < cfg.feature_fallback_min_recall:
            yield rejected(it, d, ("voxel fallback: no voxel cube lies inside the piece" if r is None else
                                   f"voxel fallback: {r['boxes']} box(es) at pitch {r['pitch']:.3g} cover {recall:.3f} of "
                                   f"the piece, under {cfg.feature_fallback_min_recall}") + f"; tried before: {before}")
            return None
        fb = dict(boxes=int(r["boxes"]), max_boxes=int(cfg.feature_fallback_max_boxes), cubes=int(r["box_voxels"]),
                  voxels=int(r["voxels"]), pitch=r["pitch"], licensed_pitch=r["licensed_pitch"], relaxed=r["relaxed"],
                  tau=r["tau"], frame=r["frame"], covered_volume=float(r["covered_volume"]),
                  left_volume=float(d["volume"]) - float(r["covered_volume"]), recall=recall, precision=precision,
                  tried=rows, rejected_before=last)
        c = fs.Candidate("voxel fallback", r["primitive"], recall, time.time() - t)
        c.precision, c.recall = precision, recall
        s = candidate_step(it, d, c)
        s.info["fallback"] = fb
        s.info["message"] = (
            f"  voxel fallback for piece #{d['id']}: {d['op']} {fb['boxes']} box(es) (limit {fb['max_boxes']}) of "
            f"{fb['cubes']} cube(s) at pitch {fb['pitch']:.3g} (licensed {fb['licensed_pitch']:.3g}"
            + (", RELAXED: coarser than tau / 3" if fb["relaxed"] else "") + f", {fb['frame']} axes), covers "
            f"{fb['covered_volume']:.4g} of {d['volume']:.4g} (recall {recall:.3f}"
            + (f", sampled precision {precision:.3f}" if precision is not None else "")
            + f"), left {fb['left_volume']:.4g}" + ("" if fb["voxels"] == fb["cubes"] else
                                                      f" ({fb['voxels'] - fb['cubes']} cube(s) inside are in no box)")
            + f"; tried before: {before}" + (f"; last verdict: {last}" if last else ""))
        yield s
        pend = make_pending(d, r["primitive"], "voxel fallback", recall, it)
        pend.update(tool_type=None, tool_params=None, need_volume=float(r["covered_volume"]), fallback=fb)
        return pend

    def subparts(d, p, it):
        """A piece without a tool as several sub-parts (ResidualConfig.feature_subparts), before the voxel fallback.
        Returns the pending entry or None; its candidate step (source "sub-parts") carries `subparts`."""
        from reco_engine import feature_stage as fs
        rows = holder["tried"].setdefault(d["id"], [])
        t = time.time()
        tally["subparts"] += 1
        try:
            from src.closest_point_field import load_field
            fld = load_field(p["stl"])
            V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
            with timed("feature sub-parts"):
                r = fs.subpart_tool(p, V, F, max_parts=max(1, cfg.feature_subparts_max),
                                    part_diagonal=holder.get("part_diagonal"))
        except Exception as exc:
            rows.append(dict(solution="sub-parts", failed=str(exc)[-160:]))
            return None
        if r is None:
            rows.append(dict(solution="sub-parts", failed="no cover"))
            return None
        pr, rc = r["precision"], r["recall"]
        score = pr * rc / max(1e-12, pr + rc - pr * rc)
        rows.append(dict(solution="sub-parts", kind=" | ".join(r["kinds"]), score=score, precision=pr, recall=rc))
        if pr < cfg.feature_min_precision or rc < cfg.feature_min_recall:
            return None
        c = fs.Candidate("sub-parts", r["primitive"], score, time.time() - t)
        c.precision, c.recall = pr, rc
        s = candidate_step(it, d, c)
        s.info["subparts"] = dict(parts=r["parts"], kinds=r["kinds"], exact=r["exact"], containment=r["containment"],
                                  precision=pr, recall=rc, history=r["history"])
        s.info["message"] = (f"  sub-parts for piece #{d['id']}: {d['op']} {r['parts']} primitive(s) ("
                             + ", ".join(r["kinds"]) + f"), exact surface coverage {r['exact']:.3f}, containment "
                             f"{r['containment']:.4f}, sampled precision {pr:.4f}, recall {rc:.4f}")
        yield s
        pend = make_pending(d, r["primitive"], "sub-parts", score, it)
        pend.update(tool_type=None, tool_params=None, subparts=s.info["subparts"])
        return pend

    def split_piece(d, p, it, state_out):
        """The piece on its own grid, its field channels once, segmented (feature_stage.segment_piece) and every
        segment guessed as part of the excess; returns the pending entries of its passing segments (or None when no
        split was possible - then the piece is rejected here)."""
        from reco_engine import feature_stage as fs
        from src.closest_point_field import load_field
        t = time.time()
        try:
            fld = load_field(p["stl"])
            region = None
            if holder.get("diff_model") is not None and inside is not None:
                region = fs.excess_region(d["op"], inside_part, holder["diff_model"], *d["bbox"])
            with timed("feature split/grid"):
                spec, ch, ginfo = fs.piece_grid(fld, cfg.field_backend or "igl", region=region,
                                                bbox=d["bbox"] if region is not None else None,
                                                thin=holder.get("thin_threshold") or 0.0)
            occ = ch[0] > 0.5
            with timed("feature split/segment"):
                labels, method = fs.segment_piece(occ, ch[1], ch[2:5], spec.pitch, cfg.feature_split_method,
                                                  cfg.feature_segment_min_voxels, cfg.feature_max_segments)
        except Exception as exc:
            yield rejected(it, d, f"split failed: {str(exc)[-200:]}")
            return None
        if labels.max() < 2:
            yield rejected(it, d, f"best single primitive below {cfg.feature_min_score}, and the split found one "
                                  f"segment only")
            return None
        tally["split_pieces"] += 1
        holder["split_note"] = (f"pitch {spec.pitch:.3g} > tau / 3 = {ginfo['licensed_pitch']:.3g}: thin parts of this "
                                f"piece are not resolved" if ginfo["relaxed"] else "")
        if ginfo.get("thin_share"):
            holder["split_note"] = "; ".join(x for x in (holder["split_note"], (
                f"{ginfo['thin_share']:.0%} of its surface is super thin (< {ginfo['thin_threshold']:.3g}) and "
                f"licenses no pitch, {ginfo['opened_cells']} cell(s) opened away")) if x)
        pend = yield from split_mask(d, p, it, state_out, spec, labels, method, 1, time.time() - t)
        return pend

    def split_mask(d, p, it, state_out, spec, labels, method, depth, seconds):
        from reco_engine import feature_stage as fs
        c = cascade.grid_centres(spec)
        segs = []
        for i in range(1, int(labels.max()) + 1):
            m = labels == i
            if not m.any():
                continue
            idx = np.argwhere(m)
            lo_w, hi_w = c[tuple(idx.min(axis=0))] - spec.pitch / 2, c[tuple(idx.max(axis=0))] + spec.pitch / 2
            vol = float(m.sum()) * spec.pitch ** 3
            segs.append((dict(id=holder["next_id"], parent=d["id"], side=d["side"], op=d["op"], volume=vol,
                              share=vol / holder["v_in"], voxels=int(m.sum()),
                              bbox=(tuple(float(x) for x in lo_w), tuple(float(x) for x in hi_w)), stl=None,
                              group=holder["next_id"]), m))
            holder["next_id"] += 1
        for sd, _ in segs:
            note(sd["id"], id=sd["id"], iteration=it, side=sd["side"], op=sd["op"], volume=sd["volume"],
                 share=sd["share"], parent=d["id"], step=records.get(d["id"], {}).get("step"), verdict="")
        tally["segments"] += len(segs)
        pitch_note = holder.pop("split_note", "") if depth == 1 else ""
        yield step("feature_split", iteration=it, piece_id=d["id"], method=method, pitch=float(spec.pitch),
                   grid=tuple(int(n) for n in labels.shape), seconds=seconds, depth=depth, pitch_note=pitch_note or None,
                   segments=[{k: v for k, v in sd.items() if k in ("id", "parent", "voxels", "volume", "share", "bbox")}
                             for sd, _ in segs],
                   message=f"piece #{d['id']} split by {method} into {len(segs)} segment(s) on "
                           f"{'x'.join(str(n) for n in labels.shape)} cells at pitch {spec.pitch:.3g} "
                           f"({seconds:.1f} s)" + (f"; {pitch_note}" if pitch_note else ""))
        pending = []
        for k, (sd, m) in enumerate(segs):
            if time.time() > deadline:
                state_out["out_of_time"] = True
                tally["skipped"] += len(segs) - k
                break
            V, F = fs.mask_mesh(m, spec)
            mesh, truncated = None, False
            if len(F):
                truncated = len(F) > FEATURE_MESH_TRIANGLES
                if truncated:
                    V, F = _cluster_decimate(V, F, FEATURE_MESH_TRIANGLES)
                mesh = dict(vertices=V.astype(np.float32), faces=F.astype(np.int32))
            yield step("feature_piece", iteration=it, piece=sd, index=k + 1, of=len(segs), mesh=mesh,
                       mesh_truncated=truncated,
                       message=f"segment {k + 1}/{len(segs)} (#{sd['id']}) of piece #{d['id']}: {sd['voxels']} voxels, "
                               f"{sd['volume']:.4g} ({sd['share']:.3%} of the input) -> {sd['op']}")
            lo_w, hi_w = (np.asarray(b, float) for b in sd["bbox"])
            scorer = fs.PartScorer(d["op"], fs.voxel_lookup(m, spec), lo_w, hi_w, sd["volume"], inside_part)
            ctx = dict(mask=m, spec=spec, faces=p.get("faces"), scorer=scorer, op=d["op"],
                       min_precision=cfg.feature_min_precision, part_diagonal=holder.get("part_diagonal"))
            try:
                g = guess(dict(sd, segment=ctx), seg_sources, deadline, registry, cfg.models)
                cands = list(g.get("candidates") or [])
            except Exception as exc:
                yield rejected(it, sd, f"error: {str(exc)[-200:]}")
                continue
            for cnd in cands[:FEATURE_CANDIDATES_SHOWN]:
                yield candidate_step(it, sd, cnd)
            state_out["out_of_time"] = state_out["out_of_time"] or bool(g.get("timed_out"))
            best = cands[0] if cands else None
            ok = best is not None and (best.precision or 0.0) >= cfg.feature_min_precision \
                and (best.recall or 0.0) >= cfg.feature_min_recall
            root = d.get("root", d["id"])                            # the piece of the difference this is a part of
            sd["root"] = root
            holder["tried"].setdefault(root, []).append(
                dict(solution=f"split ({method}), segment #{sd['id']}", note="no candidate") if best is None else
                dict(solution=f"split ({method}), segment #{sd['id']}: {best.source}", kind=best.primitive.kind,
                     score=float(best.score), precision=best.precision, recall=best.recall))
            if ok:
                pending.append(make_pending(sd, best.primitive, best.source, best.score, it))
                continue
            if depth < cfg.feature_split_depth and sd["voxels"] >= 2 * cfg.feature_segment_min_voxels:
                sub, sub_method = fs.segment_piece(m, np.where(m, sd_sdf(spec, m), 0.0), np.zeros((3,) + m.shape),
                                                   spec.pitch, "necks" if cfg.feature_split_method == "auto" else
                                                   cfg.feature_split_method, cfg.feature_segment_min_voxels,
                                                   cfg.feature_max_segments)
                if sub.max() >= 2:
                    more = yield from split_mask(sd, p, it, state_out, spec, sub, sub_method, depth + 1, 0.0)
                    pending += more or []
                    continue
            reason = "no candidate" if best is None else (
                f"best precision {best.precision:.3f} (needs {cfg.feature_min_precision}), recall {best.recall:.3f} "
                f"(needs {cfg.feature_min_recall})")
            yield rejected(it, sd, reason, prim=best.primitive if best else None, source=best.source if best else None,
                           score=float(best.score) if best else None)
        for g2 in pending:
            g2.setdefault("parent", d["id"])
        holder["parents"][d["id"]] = dict(d=d, n=len(segs), ids=[sd["id"] for sd, _ in segs])
        return pending

    def sd_sdf(spec, m):
        """The inside distance of a segment's own voxels (mm) for a second split (its channels are voxel ones)."""
        from scipy import ndimage
        return ndimage.distance_transform_edt(m) * spec.pitch

    def run_checks(pending, it):
        """The exact FreeCAD checks of a batch's features against their pieces (ResidualConfig.feature_check);
        returns the features that pass. Without a piece STEP (a fake difference) a feature is not checked."""
        items = []
        for g in pending:
            st = records.get(g["piece"]["id"], {}).get("step")
            if st and g.get("checks") is None:                      # checked before (applied at once): not again
                try:
                    from reco_field.export import program_to_dict
                    items.append(dict(id=g["piece"]["id"], op=g["op"], piece_step=st,
                                      program=program_to_dict(primitive_to_sp(g["prim"]))))
                except Exception:
                    continue
        if not (cfg.feature_check and items and (cfg.feature_checks is not None or cfg.feature_difference is None)):
            return pending
        from reco_field.exact_check import feature_checks
        with timed("feature checks"):
            res = (cfg.feature_checks or feature_checks)(path, items, cfg.feature_call_timeout)
        if "error" in res and not isinstance(res.get("error"), dict):
            tally["check_failures"] = tally.get("check_failures", 0) + 1     # advisory when FreeCAD fails - but said
            holder["check_error"] = str(res["error"])[-200:]
            return pending
        kept = []
        for g in pending:
            c = res.get(g["piece"]["id"])
            if c and "error" in c:                                  # no numbers for this one: said, not a verdict
                note(g["piece"]["id"], check_error=str(c["error"])[-200:])
            if c and "error" not in c:
                g["checks"] = c
                note(g["piece"]["id"], **{k: c[k] for k in ("common_volume", "beyond_piece_volume",
                                                            "left_of_piece_volume", "harm_volume")})
                share = cfg.feature_rough_max_harm if g.get("rough") else cfg.feature_check_tolerance
                tol = max(share * c["feature_volume"], 1e-6 * holder["v_in"])
                if g.get("rough"):                                  # a giant tool's 5 % may exceed the whole part
                    tol = max(min(tol, cfg.feature_rough_max_harm_input * holder["v_in"]), 1e-6 * holder["v_in"])
                if c["harm_volume"] > tol:
                    yield rejected(it, g["piece"], f"check: {'removes' if g['op'] == 'cut' else 'adds'} "
                                                   f"{c['harm_volume']:.4g} mm3 of {'input material' if g['op'] == 'cut' else 'material outside the input'}"
                                                   f" beyond its piece (tolerance {tol:.3g})",
                                   prim=g["prim"], source=g["source"], score=g["score"])
                    continue
            kept.append(g)
        return kept

    def parent_verdicts(pending, it, only=None):
        """After a batch: every split piece whose segments were in it gets its verdict (`only`: that piece now)."""
        done_parents = {g["parent"] for g in pending if g.get("parent") is not None}
        for pid in list(holder["parents"]):
            info = holder["parents"][pid]
            if pid not in done_parents and pid != only:
                continue
            acc = [g for g in pending if g.get("parent") == pid and g.get("accepted")]
            del holder["parents"][pid]
            if acc:
                tally["segments_accepted"] += len(acc)
                yield step("feature_accepted", iteration=it, piece_id=pid, op=info["d"]["op"], primitive=None,
                           source="split", score=None, validation=acc[-1].get("validation"),
                           segments_accepted=[g["piece"]["id"] for g in acc],
                           message=f"piece #{pid}: {len(acc)} of {info['n']} segment(s) accepted")
            else:
                yield rejected(it, info["d"], f"split: 0 of {info['n']} segments accepted")

    def reaches_out(g, pieces_of_round) -> bool:
        """Does an accepted primitive reach into another piece of the same difference (whose volume it then changed)?
        Bounding boxes first; with meshes, sampled points of the primitive inside the other piece."""
        try:
            lo, hi = (np.asarray(b, float) for b in primitive_to_sp(g["prim"]).bounds())
        except Exception:
            return True
        pts = None
        for d, p in pieces_of_round:
            if d["id"] == g["piece"]["id"]:
                continue
            blo, bhi = (np.asarray(b, float) for b in d["bbox"])
            if np.any(hi < blo) or np.any(lo > bhi):
                continue
            if not d.get("stl") or not Path(str(d["stl"])).is_file():
                return True
            if pts is None:
                rng = np.random.default_rng(0)
                pts = lo + rng.random((4000, 3)) * (hi - lo)
                pts = pts[np.asarray(primitive_to_sp(g["prim"]).sdf(pts)) <= 0]
            from src.closest_point_field import load_field
            from reco_engine.feature_stage import _inside_mesh
            o = load_field(d["stl"])
            sel = pts[np.all((pts >= blo) & (pts <= bhi), axis=1)]
            if len(sel) and _inside_mesh(np.asarray(o.vertices, float), np.asarray(o.faces), sel).any():
                return True
        return False

    def thickness_of(p):
        """feature_stage.piece_thickness of a piece against the super-thin threshold; None when the suppression is
        off, the piece has no mesh, or the measure fails (then the piece is guessed as before)."""
        thin = holder.get("thin_threshold") or 0.0
        if thin <= 0 or not p.get("stl") or not Path(str(p["stl"])).is_file():
            return None
        try:
            from reco_engine.feature_stage import piece_thickness
            from src.closest_point_field import load_field
            with timed("feature thickness"):
                return piece_thickness(load_field(p["stl"]), thin)
        except Exception:
            return None

    def piece_vertices(p):
        if not p.get("stl") or not Path(str(p["stl"])).is_file():
            return None
        from src.closest_point_field import load_field
        return np.asarray(load_field(p["stl"]).vertices, float)

    def typed_order(taken, descs):
        """[[index, ...]] groups of the same tool type and congruent (feature_groups.group_pieces), then the pieces of
        no type by congruence alone; {index: PieceType}. Largest first by their first member."""
        from reco_engine import feature_groups as FG
        from reco_engine.feature_stage import group_congruent
        entries = []
        for i, (p, _, _) in enumerate(taken):
            V = piece_vertices(p)
            if V is not None:
                entries.append(dict(id=i, piece=p, V=V, op="add" if p["sign"] == "plus" else "cut", descriptor=descs[i]))
        with timed("feature groups"):
            fg = FG.group_pieces(entries, net=holder.get("net"))
        order = [list(g["members"]) for g in fg["groups"]]          # no type: congruence alone, as before
        seen = {i for g in order for i in g}
        rest = [i for i in range(len(taken)) if i not in seen]
        order += [[rest[j] for j in g] for g in group_congruent([descs[i] for i in rest])]
        order.sort(key=lambda g: min(g))
        return order, {i: fg["per_piece"].get(i) for i in range(len(taken))}

    def feedback_of(p):
        """The feedback entry for a piece of this run: the same sign, volume within 1e-3, centre within 1e-3 of its size."""
        if not feedback:
            return None
        c = np.asarray(p.get("centre") or [0, 0, 0], float)
        lo, hi = (np.asarray(b, float) for b in p["bbox"])
        size = float(np.linalg.norm(hi - lo))
        for e in feedback:
            q = e.get("piece") or {}
            if q.get("sign") != p.get("sign") or not q.get("volume"):
                continue
            if abs(float(q["volume"]) - float(p["volume"])) <= 1e-3 * float(p["volume"]) and \
                    np.linalg.norm(np.asarray(q.get("centre") or [np.inf] * 3, float) - c) <= 1e-3 * size + 1e-9:
                return e
        return None

    def typed_candidates(d, p, desc):
        """The representative's typed candidates, recycle before use: the session's tools matching the catalogue
        fits of its ranked types (the stored standard size) and the session's created tools congruent with the piece
        (moved onto it), then the catalogue fits themselves; at most feature_tool_candidates."""
        from reco_engine import tools as TL
        pt = holder["ptype"].get(d["id"])
        lo, hi = (np.asarray(b, float) for b in d["bbox"])
        over = 0.02 * float(np.linalg.norm(hi - lo))
        out, cat = [], []
        for f in (pt.fits if pt is not None else [])[:cfg.feature_tool_candidates]:
            prm = {k: v for k, v in f.params.items() if not k.startswith("_")}
            rec = store.match(f.tool, prm)
            try:
                if rec is not None:
                    out.append(dict(source=f"session:{rec.id}", primitive=TL.TOOLS[f.tool].build(rec.params, f.R, f.origin, over),
                                    tool_type=f.tool, tool_source="session", params=rec.params))
                else:
                    cat.append(dict(source=f"catalogue:{f.tool}", primitive=f.build(over), tool_type=f.tool,
                                    tool_source="catalogue", params=prm))
            except (ValueError, KeyError):
                continue
        if desc is not None:
            from reco_engine.feature_stage import best_transfer, congruent, sampled_score
            for rec in store.records:
                if rec.construction is None or rec.descriptor is None:
                    continue
                src_desc = TL._descriptor_from_json(rec.descriptor)
                if not congruent(src_desc, desc):
                    continue
                V = piece_vertices(p)
                if V is None:
                    continue
                from src.closest_point_field import load_field
                fld = load_field(p["stl"])
                F = np.asarray(fld.faces)
                sc, q = best_transfer(TL.construction_primitive(rec), src_desc, desc,
                                      lambda q2: sampled_score(q2, V, F, n=20000))
                out.append(dict(source=f"session:{rec.id}", primitive=q, tool_type=rec.tool, tool_source="session",
                                params=None))
        return (out + cat)[:max(1, cfg.feature_tool_candidates)]

    def small_kept(p):
        """Why a piece under feature_min_share is kept (ResidualConfig.feature_keep_known, feature_min_radius_depth);
        None: dropped as too small."""
        from reco_engine import feature_stage as fs
        from reco_engine import tools as TL
        try:
            if cfg.feature_keep_known:
                desc = fs.piece_descriptor(p)
                for pid, known in holder["accepted"]:
                    if fs.congruent(known, desc):
                        return f"congruent with the accepted piece #{pid}"
                for rec in store.records:
                    if rec.descriptor is not None and fs.congruent(TL._descriptor_from_json(rec.descriptor), desc):
                        return f"congruent with the session's tool {rec.id}"
                V = piece_vertices(p)
                if V is not None and len(store):
                    for f in TL.fit_piece(p, V, "add" if p["sign"] == "plus" else "cut"):
                        rec = store.match(f.tool, {k: v for k, v in f.params.items() if not k.startswith("_")})
                        if rec is not None:
                            return f"the size of the session's tool {rec.id}"
            return fs.feature_sized(p, cfg.feature_min_radius_depth)
        except Exception:                                            # a piece that cannot be read is not kept
            return None

    def tools_of(grp):
        from reco_field.export import program_to_dict
        return [dict(op=g["op"], program=program_to_dict(primitive_to_sp(g["prim"]))) for g in grp]

    def differ(model_step, tools, pieces_dir, max_pieces, min_solid, tag):
        """One exact difference of input and `model_step`: incremental when the state of the model before is there
        (holder["state"]: dict(dir, model), `tools` what was applied since - [] for the state's own model), else the
        global boolean. Returns (the exact_residual dict with "method", "seconds", "timeout" and - on a failure of the
        first method - "fallback_from", the state of `model_step` or None)."""
        t, timeout = time.time(), cfg.feature_call_timeout
        st, r, first = holder.get("state"), None, None
        new = (holder["state_root"] / tag) if holder.get("state_root") is not None else None
        if incremental is not None and st is not None and tools is not None:
            timeout = int(timeout + cfg.feature_incremental_tool_seconds * len(tools))
            try:
                r = incremental(path, model_step, st["dir"], tools, new, pieces_dir, max_pieces, min_solid, timeout)
            except Exception as exc:
                r = dict(error=f"incremental difference: {str(exc)[-200:]}")
            r.setdefault("method", "incremental")
            if "error" in r:
                first, r = r, None
        if r is None:
            timeout = cfg.feature_call_timeout
            if cfg.feature_difference is None:
                r = difference(path, model_step, pieces_dir, max_pieces, min_solid, timeout,
                               state_dir=new if incremental is not None else None)
            else:
                r = difference(path, model_step, pieces_dir, max_pieces, min_solid, timeout)
            r.setdefault("method", "occt")
            if first is not None:
                r["fallback_from"] = first["error"]
        if "error" in r and cfg.feature_mesh_fallback and cfg.feature_difference is None:
            # the ladder's middle rung (2026-10-15): the meshes' exact difference (manifold3d) where OCCT disagreed
            # or gave no answer - the phase goes on with mesh pieces (no STEP, no typed faces) instead of ending
            from reco_field.exact_check import mesh_residual
            model_stl = Path(str(model_step)).with_suffix(".stl")
            occt_error = r["error"]
            if model_stl.is_file():
                mesh_in = (field.vertices, field.faces) if field is not None and getattr(field, "vertices", None) is not None else None
                with timed("feature difference/mesh"):
                    r2 = mesh_residual(path, model_stl, pieces_dir, max_pieces, min_solid, input_mesh=mesh_in)
                if "error" not in r2:
                    r = dict(r2, fallback_from=occt_error, state=None)
                else:
                    r["error"] = f"{occt_error}; mesh residual: {r2['error']}"
            else:
                r["error"] = f"{occt_error}; no model STL for the mesh residual"
        r["seconds"] = dict(r["seconds"], wall=time.time() - t) if isinstance(r.get("seconds"), dict) else \
            dict(wall=time.time() - t)
        r["timeout"] = timeout
        if "error" in r or incremental is None or new is None:
            return r, None
        return r, dict(dir=str(new), model=str(model_step))

    def how(r):
        """The method of a difference for a step's message."""
        return (f"{r.get('method')} difference, {r['seconds']['wall']:.0f} s (timeout {r['timeout']} s)"
                + (f"; the incremental one failed: {str(r['fallback_from'])[-160:]}" if r.get("fallback_from") else ""))

    def rounds(holder):
        """The differences; returns the `stopped` text. One exact difference serves all its pieces: they are
        disjoint solids, so a validated operation leaves the others unchanged - unless it reaches into one of them,
        then the next difference is computed at once."""
        from reco_engine.feature_stage import group_congruent, piece_descriptor
        if cfg.feature_difference is None and suffix not in (".step", ".stp", ".stl"):
            return f"error: the exact difference needs the input as a STEP or STL file, got {path or 'a field'}"
        if cfg.feature_difference is None and suffix == ".stl" and field is not None \
                and len(field.faces) > cfg.feature_stl_max_faces:
            return (f"error: skipped - an STL of {len(field.faces)} triangles (> feature_stl_max_faces "
                    f"{cfg.feature_stl_max_faces}) is no OCCT solid in bounded time; give the STEP")
        if cfg.feature_build is None or cfg.feature_difference is None:
            from reco_field.export import find_freecadcmd
            if find_freecadcmd() is None:
                return "error: freecadcmd not found (the phase builds and differences in FreeCAD)"
        # without an out_dir: ~/.cache/shape_reco/features/<part>_<time>/ - a reboot wiped /tmp and the user's
        # features.FCStd with it (2026-09-28)
        default = Path.home() / ".cache" / "shape_reco" / "features" / \
            f"{Path(path).stem or 'part'}_{time.strftime('%Y%m%d-%H%M%S')}"
        out = Path(out_dir or cfg.out_dir or default) / "features"
        out.mkdir(parents=True, exist_ok=True)
        holder["state_root"] = out / "state"
        if cfg.feature_snap and suffix in (".step", ".stp") and cfg.feature_build is None:
            from reco_engine.feature_stage import snap_to_faces
            from reco_field.exact_check import exact_faces
            with timed("feature snap"):
                inv = exact_faces(path, timeout=cfg.feature_call_timeout)
                if "error" not in inv:
                    moved = snap_to_faces(result, inv["faces"])
                    holder["snapped"] = moved
                    if cfg.feature_contour and field is not None:
                        from reco_engine.feature_stage import input_inside
                        from reco_engine.turned_contour import turned_contour
                        turned = turned_contour(result, inv["faces"], input_inside(field.vertices, field.faces))
                        holder["contour"] = turned
                        moved = moved or [t for t in turned if t["op"]]
                    if state is not None and moved:                 # the voxel view of the moved program
                        try:
                            state.s_model = -program_model(result).sdf(state.centres).astype(state.s_model.dtype)
                        except Exception:
                            pass
        base = build(result, out, "feature_model_0", cfg.feature_call_timeout)
        if not base.get("step"):
            return f"error: the rough model was not built ({base.get('error', 'no STEP')})"
        base_valid = bool(base.get("valid"))
        # one valid solid - or, if the rough model already was not, no worse than it (else nothing could ever pass)
        holder["max_solids"] = 1 if base_valid and int(base.get("n_solids") or 0) == 1 else max(1, int(base.get("n_solids") or 1))
        holder["base"] = base
        v_est = _mesh_volume(field)
        keeps_small = cfg.feature_keep_known or cfg.feature_min_radius_depth > 0
        floor = (min(cfg.feature_min_share, cfg.feature_small_share) if keeps_small else cfg.feature_min_share) \
            * v_est if v_est else None
        tried, total, n_build = set(), 0, 1
        while True:
            if time.time() > deadline:
                return "time budget"
            low = state.memory_guard.check() if state is not None and getattr(state, "memory_guard", None) is not None else None
            if low:
                return f"{low}: the phase stops with what it applied"
            tally["iterations"] = it = tally["iterations"] + 1
            t_it = time.time()
            st = holder.get("state")                                # the state of another model is none of this one
            if st is not None and st["model"] != str(base["step"]):
                holder["state"] = None
            r, st = differ(base["step"], [], out / f"iteration_{it}",
                           max(cfg.feature_max_total_pieces, cfg.feature_max_written_pieces) + len(tried), floor,
                           f"iteration_{it}")
            if st is not None:
                holder["state"] = st
            tally["differences"] += 1
            tally["methods"][r.get("method")] = tally["methods"].get(r.get("method"), 0) + 1
            try:
                holder["diff_model"] = program_model(result)            # the model this difference is of
            except Exception:
                holder["diff_model"] = None
            if "error" in r:
                return f"error: {r['error']} ({how(r)})"
            v_in = holder["v_in"] = float(r["input_volume"])
            holder.setdefault("iterations", []).append(dict(iteration=it, model_step=str(base["step"]),
                                                            fuzzy=r.get("fuzzy"), iou=r.get("iou"),
                                                            method=r.get("method")))
            now.update(missing=float(r["r_plus"]), invented=float(r["r_minus"]), iou=r.get("iou"))
            holder["residual"] = now["missing"] + now["invented"]
            left = r.get("left_out_count") or {}
            dropped_small = int(sum((left.get("small") or {}).values()))
            dropped_limit = int(sum((left.get("capped") or {}).values()))
            taken, seen_before, above = [], 0, 0
            suppressed, kept_small = [], []
            for p in sorted(r.get("pieces") or [], key=lambda p: -float(p["volume"])):
                share = float(p["volume"]) / v_in if v_in > 0 else 0.0
                if share < cfg.feature_min_share:
                    why = small_kept(p) if share >= min(cfg.feature_min_share, cfg.feature_small_share) else None
                    if why is None:
                        dropped_small += 1
                        continue
                    kept_small.append(dict(piece=p.get("id"), sign=p["sign"], volume=float(p["volume"]), share=share,
                                           reason=why))
                sig = _piece_signature(p, v_in)
                if sig not in holder["thin"]:
                    holder["thin"][sig] = thickness_of(p)
                th = holder["thin"][sig]
                if th is not None and th["suppressed"]:             # super-thin as a whole: listed, not guessed
                    from reco_engine.feature_stage import suppressed_reason
                    suppressed.append(dict(piece=p.get("id"), sign=p["sign"], volume=float(p["volume"]), share=share,
                                           reason=suppressed_reason(th), thickness=th, step=p.get("step")))
                    continue
                above += 1
                if sig in tried:
                    seen_before += 1
                    continue
                taken.append((p, sig, share))
            tally["skipped"] += seen_before
            holder["suppressed"] = suppressed
            descs = [piece_descriptor(p) if cfg.feature_patterns else None for p, _, _ in taken]
            t_g = time.time()
            ptypes: Dict[int, object] = {}
            if cfg.feature_types:
                order, ptypes = typed_order(taken, descs)             # type + congruence (feature_groups)
            else:
                order = group_congruent(descs)                      # [[index, ...]] the representative first
            if cfg.feature_piece_order == "volume per face":        # a congruence group is one unit: its volume
                def weight(g):
                    return sum(taken[i][2] for i in g) / max(1, len(taken[g[0]][0].get("faces") or []))
                order.sort(key=lambda g: (-weight(g), min(g)))
            room = max(0, cfg.feature_max_total_pieces - total)     # the guess cap counts groups (full guesses)
            over_total = len(order) > room
            if over_total:
                dropped_limit += sum(len(g) for g in order[room:])
                order = order[:room]
            dropped_target, target_text = 0, ""
            if cfg.feature_target_iou is not None and v_in > 0:
                est_now = (v_in - now["missing"]) / (v_in + now["invented"])
                if (now.get("iou") if now.get("iou") is not None else est_now) >= cfg.feature_target_iou:
                    return f"target IoU {cfg.feature_target_iou:.3f} reached"
                vols = [(sum(float(taken[i][0]["volume"]) for i in g if taken[i][0]["sign"] == "plus"),
                         sum(float(taken[i][0]["volume"]) for i in g if taken[i][0]["sign"] != "plus")) for g in order]
                keep, est = select_for_target(vols, now["missing"], now["invented"], v_in, cfg.feature_target_iou)
                dropped_target = sum(len(order[k]) for k in range(len(order)) if k not in set(keep))
                order = [order[k] for k in keep]
                target_text = (f"; target IoU {cfg.feature_target_iou:.3f}: {sum(len(g) for g in order)} piece(s) in "
                               f"{len(order)} group(s) taken for an estimated {est:.4f}, {dropped_target} left as detail")
            pieces, groups = [], []
            for grp in order:
                rep_id = holder["next_id"]
                members = []
                for i in grp:
                    p, sig, share = taken[i]
                    lo, hi = p["bbox"]
                    side = "missing" if p["sign"] == "plus" else "invented"
                    pt = ptypes.get(i)
                    d = dict(id=holder["next_id"], side=side, op="add" if side == "missing" else "cut",
                             volume=float(p["volume"]), share=share,
                             bbox=(tuple(float(x) for x in lo), tuple(float(x) for x in hi)), stl=p.get("stl"), group=rep_id,
                             tool_type=getattr(pt, "tool_type", None))
                    holder["ptype"][holder["next_id"]] = pt
                    holder["desc"][holder["next_id"]] = descs[i]
                    holder["next_id"] += 1
                    note(d["id"], id=d["id"], iteration=it, side=side, op=d["op"], volume=d["volume"], share=share,
                         sign=p["sign"], centre=list(p.get("centre") or []), step=p.get("step"), verdict="")
                    pieces.append((d, p))
                    members.append((d, p, sig, descs[i]))
                groups.append(members)
            tally["groups"] += len(groups)
            iou = now["iou"]
            yield step("feature_residual", iteration=it, missing_volume=now["missing"], invented_volume=now["invented"],
                       input_volume=v_in, exact_iou=iou, pieces=[d for d, _ in pieces], dropped_small=dropped_small,
                       piece_volumes=sorted((float(p["volume"]) for p in r.get("pieces") or []), reverse=True),
                       groups=[[m[0]["id"] for m in g] for g in groups],
                       suppressed=suppressed, suppressed_volume=float(sum(q["volume"] for q in suppressed)),
                       kept_small=kept_small, dropped_limit=dropped_limit, limits=dict(limits), seconds=time.time() - t_it,
                       difference_method=r.get("method"), difference_seconds=r["seconds"]["wall"],
                       difference_timeout=r["timeout"], difference_fallback_from=r.get("fallback_from"),
                       message=f"features, iteration {it}: missing {now['missing']:.4g}, invented {now['invented']:.4g} "
                               f"(input {v_in:.4g}" + (f", exact IoU {iou:.4f}" if iou is not None else "") + f"); "
                               f"{len(pieces)} piece(s) in {len(groups)} congruence group(s), {dropped_small} too small"
                               + (f" ({len(kept_small)} under the share kept: " + "; ".join(sorted({q["reason"] for q in kept_small}))
                                  + ")" if kept_small else "") + ", "
                               f"{dropped_limit} over the limits, {seen_before} tried before"
                               + (f", {len(suppressed)} super-thin suppressed "
                                  f"({sum(q['volume'] for q in suppressed):.4g})" if suppressed else "")
                               + target_text + f"; {how(r)}"
                               + ("; the rough model snapped onto the input's faces first ("
                                  + ", ".join(f"{m['label']} moved {m['shift']:.3g}, size {m['size_change']:.3g}"
                                              + (f", sector {m['sector_deg_before']:.2f} -> {m['sector_deg']:.2f} deg"
                                                 if "sector_deg" in m else "")
                                              for m in holder.pop("snapped")) + ")" if holder.get("snapped") else "")
                               + ("; the rough model's turned contour ("
                                  + "; ".join(f"{m['label']}, {m['volume']:.4g}" if m["op"] else m["label"]
                                              for m in holder.pop("contour")) + ")" if holder.get("contour") else ""))
            if cfg.feature_types:
                glist, ungrouped = [], []
                for g in groups:
                    d0 = g[0][0]
                    pt = holder["ptype"].get(d0["id"])
                    if pt is None or pt.tool_type is None:
                        ungrouped += [m[0]["id"] for m in g]
                        continue
                    glist.append(dict(id=d0["id"], tool_type=pt.tool_type, family=pt.tool_type,
                                      members=[m[0]["id"] for m in g], representative=d0["id"],
                                      confidence=pt.confidence, source=pt.source))
                from reco_engine.symmetry import describe as sym_text
                yield step("feature_groups", iteration=it, groups=glist, ungrouped=ungrouped, seconds=time.time() - t_g,
                           symmetry=sym_text(holder.get("symmetry")) if cfg.feature_symmetry else None,
                           message="tool types: " + (", ".join(f"{g['tool_type']} x{len(g['members'])}" for g in glist)
                                                       or "none") + f"; {len(ungrouped)} piece(s) of no catalogue type"
                           + (f"; symmetry of the part: {sym_text(holder.get('symmetry'))}" if cfg.feature_symmetry else ""))
            if holder["residual"] <= cfg.feature_tolerance * v_in:
                return "converged"
            if not pieces:
                if over_total:
                    return "max pieces"
                return "converged" if above == 0 else "nothing applied"
            applied_before = tally["applied"]
            flags = dict(out_of_time=False)
            k = 0
            deferred, renewed = [], False
            hopeless: Dict[int, tuple] = {}                          # pieces without a tool: the voxel fallback's
            piece_of = {d["id"]: p for d, p in pieces}
            sig_of = {m[0]["id"]: m[2] for g in groups for m in g}
            for c0 in range(0, len(groups), max(1, cfg.feature_max_pieces)):   # a group counts once against the limit
                pending = []
                at_once = []                                        # guesses that went the immediate way
                for members in groups[c0:c0 + max(1, cfg.feature_max_pieces)]:
                    rep_d, rep_p, rep_sig, rep_desc = members[0]
                    n_before = len(pending)
                    best = None
                    tries, guessed, waiting = 0, [], []              # the group's full guesses; members without a tool
                    for j, (d, p, sig, desc) in enumerate(members):
                        if time.time() > deadline:
                            flags["out_of_time"] = True
                            break
                        k += 1
                        tried.add(sig)
                        if tally["applied"] + len(pending) >= cfg.feature_max_applied:   # a batch stays under the cap
                            flags["max_applied"] = True
                            break
                        full = best is None or best == "split"      # no tool in the group yet: this member's own guess
                        given_up = best is None and tries >= max(1, cfg.feature_group_tries)
                        if full and not given_up:                   # a full guess costs; a transfer does not
                            total += 1
                        yield piece_step(d, it, k, len(pieces))
                        may_split = bool(cfg.feature_split and d["share"] >= cfg.feature_split_min_share and d.get("stl"))
                        if given_up:                                # a hopeless group of 36 is not 36 full guesses
                            holder["tried"].setdefault(d["id"], []).append(
                                dict(solution="the full guesses of its congruence group", members=list(guessed)))
                            yield rejected(it, d, f"pattern: {tries} congruent piece(s) ("
                                                  + ", ".join(f"#{m}" for m in guessed) + ") were guessed in full and "
                                                  "found no tool")
                            hopeless[d["id"]] = (d, p, sig)
                            continue
                        if full:                                    # each copy on its own until one finds a tool
                            tries += 1
                            guessed.append(d["id"])
                            if j > 0 and best is None:
                                tally["group_retries"] += 1
                            if d.get("group") in holder["skip_groups"]:
                                yield rejected(it, d, "skipped by the user (with its group)")
                                continue
                            fb = feedback_of(p)
                            if fb is not None:
                                v, kind = fb.get("verdict"), fb.get("kind")
                                yield step("feature_feedback", iteration=it, piece_id=d["id"], verdict=v, feedback_kind=kind,
                                           note=fb.get("note"), source=fb.get("_file"),
                                           message=f"piece #{d['id']}: feedback used - {v}" + (f", {kind}" if kind else "")
                                                   + (f" ({fb['note']})" if fb.get("note") else ""))
                                if v == "voxels" and cfg.feature_voxel_fallback and d.get("stl"):
                                    holder["tried"].setdefault(d["id"], []).append(
                                        dict(solution="feedback", note="voxels asked for by the user"))
                                    hopeless[d["id"]] = (d, p, sig)
                                    continue
                                if v in ("leave", "not_a_feature", "voxels"):
                                    reason = {"leave": "left (feedback)", "not_a_feature": "not a feature (feedback)",
                                              "voxels": "voxels (feedback) - the voxel fallback is off or the piece "
                                                        "has no mesh: left"}[v]
                                    yield rejected(it, d, reason)
                                    continue
                                if v == "split":
                                    deferred.append((d, p, sig))
                                    continue
                                from reco_engine.tools import TOOLS as _TOOLS
                                if kind in _TOOLS:                  # that tool type first
                                    V_ = piece_vertices(p)
                                    from reco_engine.tools import fit_piece
                                    fits = fit_piece(p, V_, d["op"], types=[kind]) if V_ is not None else []
                                    pt = holder["ptype"].get(d["id"])
                                    if fits and pt is not None:
                                        pt.fits = fits + [f for f in pt.fits if f.tool != kind]
                                        pt.ranked = [(kind, 1.0)] + [r for r in pt.ranked if r[0] != kind]
                            typed = typed_candidates(d, p, desc) if cfg.feature_types else None
                            pend, found = yield from full_guess(d, p, it, flags, may_split=may_split, typed=typed,
                                                                copies=len(members) - 1 - j)
                            if flags.get("stop"):
                                return "stopped by the user"
                            if found == "left":                     # the user's answer: neither a guess nor voxels
                                continue
                            if found == "split" and pend is None:   # split later: every group's cheap guess first
                                best = "split"
                                deferred.append((d, p, sig))
                                continue
                            if found is not None and found != "split":
                                # this member's tool is the group's: the members that failed before it get it too
                                best, rep_d, rep_desc = found, d, desc
                                for d2, p2, sig2, desc2 in waiting:
                                    pend2 = yield from transfer(d2, p2, rep_d, rep_desc, desc2, best, it, flags,
                                                                guess_after=False)
                                    if pend2 is not None:
                                        hopeless.pop(d2["id"], None)
                                        pending.append(pend2)
                                waiting = []
                            elif pend is None:
                                waiting.append((d, p, sig, desc))
                        else:
                            pend = yield from transfer(d, p, rep_d, rep_desc, desc, best, it, flags)
                        if pend is None:
                            hopeless[d["id"]] = (d, p, sig)
                        else:
                            pending.append(pend)
                    if cfg.feature_apply_at_once and len(pending) > n_before:
                        mine = pending[n_before:]
                        at_once.extend(mine)
                        mine = yield from run_checks(mine, it)
                        pending[n_before:] = yield from apply_now(mine, it)
                    if flags["out_of_time"] or flags.get("max_applied"):
                        break
                guesses = at_once + [g for g in pending if not any(g is a for a in at_once)]
                pending = yield from run_checks(pending, it)
                batches = [pending] if cfg.feature_batch else [[g] for g in pending]
                first = True
                for grp in batches:
                    if grp:
                        n_build = yield from validate(grp, it, base, holder, out, n_build, v_in, force=first)
                        first = False
                n_build = yield from confirm(it, base, holder, out, n_build, v_in)
                yield from parent_verdicts(guesses if cfg.feature_apply_at_once else pending, it)
                for g in guesses:                                    # a tool the check or the validation rejected
                    pid = g["piece"]["id"]
                    if not g.get("accepted") and pid in piece_of:
                        holder["tried"].setdefault(pid, []).append(
                            dict(solution=f"{g['source']}, rejected", note=records.get(pid, {}).get("reason") or ""))
                        hopeless[pid] = (g["piece"], piece_of[pid], sig_of.get(pid))
                base = holder.get("base", base)
                if flags["out_of_time"] or time.time() > deadline:
                    tally["skipped"] += len(pieces) - k
                    return "time budget"
                if flags.get("max_applied") or tally["applied"] >= cfg.feature_max_applied:
                    tally["skipped"] += len(pieces) - k
                    return "max applied"
                if holder["residual"] <= cfg.feature_tolerance * v_in:
                    return "converged"
                if any(g.get("accepted") and reaches_out(g, pieces) for g in guesses):
                    tally["renewed"] += 1
                    tally["skipped"] += len(pieces) - k             # their pieces come back in the next difference
                    renewed = True
                    break
            # the fallback pass (2026-09-28): the pieces no single primitive or tool explains are split only now -
            # on the liner, splitting the three largest first took 644 s of grids and left the 8 congruence groups
            # (pockets, bores) unguessed when the budget ran out
            for n_def, (d, p, sig) in enumerate(sorted(deferred, key=lambda t: -t[0]["volume"])):
                if renewed:                                          # the next difference has them again
                    tried.discard(sig)
                    continue
                if time.time() > deadline:
                    tally["skipped"] += len(deferred) - n_def
                    return "time budget"
                more = yield from split_piece(d, p, it, flags)
                if not more:
                    if more is not None:                            # split, nothing passed: the verdict now
                        yield from parent_verdicts([], it, only=d["id"])
                    hopeless[d["id"]] = (d, p, sig)
                    continue
                more = yield from run_checks(more, it)
                if more:
                    n_build = yield from validate(more, it, base, holder, out, n_build, v_in, force=True)
                yield from parent_verdicts(more, it)
                if not any(g.get("accepted") for g in more):
                    hopeless[d["id"]] = (d, p, sig)
                base = holder.get("base", base)
                if holder["residual"] <= cfg.feature_tolerance * v_in:
                    return "converged"
            # the last resort (2026-09-29): what no tool, no congruent piece's tool and no split explained becomes
            # voxel cubes - one batch, checked and validated like every tool
            def last_resort(d, p, it, has_stl):
                """The sub-parts, else the voxel fallback of a piece; its pending entry or None."""
                pend = None
                if cfg.feature_subparts and has_stl:
                    pend = yield from subparts(d, p, it)
                if pend is None and cfg.feature_voxel_fallback and has_stl:
                    pend = yield from fallback(d, p, it)
                return pend

            def settle(batch):
                """Check and validate a last-resort batch; "converged" when the residual is within tolerance."""
                nonlocal n_build, base
                if batch:
                    batch = yield from run_checks(batch, it)
                    if batch:
                        n_build = yield from validate(batch, it, base, holder, out, n_build, v_in, force=True)
                    base = holder.get("base", base)
                    if holder["residual"] <= cfg.feature_tolerance * v_in:
                        return "converged"
                return None

            fallen, roughs = [], []
            for n_h, (d, p, sig) in enumerate(hopeless.values()):
                has_stl = bool(d.get("stl") and Path(str(d["stl"])).is_file())    # the sub-parts and the voxels need
                if not (cfg.feature_rough or ((cfg.feature_voxel_fallback or cfg.feature_subparts) and has_stl)):
                    continue                                        # the piece's mesh; the rough guess does not
                if renewed:                                          # the next difference has them again
                    if sig is not None:
                        tried.discard(sig)
                    continue
                if time.time() > deadline or tally["applied"] + len(fallen) >= cfg.feature_max_applied:
                    tally["skipped"] += len(hopeless) - n_h
                    break
                pend = (yield from rough(d, p, it)) if cfg.feature_rough else None
                if pend is not None:
                    roughs.append((pend, d, p, has_stl))
                else:
                    pend = yield from last_resort(d, p, it, has_stl)
                if pend is not None:
                    fallen.append(pend)
            if (yield from settle(fallen)) == "converged":
                return "converged"
            # a rough guess the check or the validation refused still leaves the piece its last resort, in this round
            # (protocol rough_tidy_v2, DEV: aegir_n6_I_000121's refused guess left its 64066 mm^3 piece without the voxel
            # fallback, the round applied nothing and the phase stopped - exact IoU 0.160 against 0.531 without the guess)
            again = []
            for pend, d, p, has_stl in roughs:
                if pend.get("accepted") or time.time() > deadline \
                        or tally["applied"] + len(again) >= cfg.feature_max_applied:
                    continue
                q = yield from last_resort(d, p, it, has_stl)
                if q is not None:
                    again.append(q)
            if (yield from settle(again)) == "converged":
                return "converged"
            if tally["applied"] == applied_before:
                return "nothing applied"
            if over_total or total >= cfg.feature_max_total_pieces:
                return "max pieces"
            if it >= cfg.feature_max_iterations:
                return "max iterations"

    def remember(g):
        """An accepted tool into the session's store (catalogue / session: its type and size; created: its tree and
        the piece it was fitted to); tools_by_source counts it."""
        ts = tool_info(g["piece"], g["source"])["tool_source"]
        tally["tools"][ts or "primitive"] = tally["tools"].get(ts or "primitive", 0) + 1
        tt = g.get("tool_type")
        if not tt or not cfg.feature_types:
            return
        try:
            if ts in ("catalogue", "session") and g.get("tool_params"):
                store.add_use(tt, path, g["piece"]["id"], g["score"], "pattern" if g["source"].startswith("pattern:") else ts,
                              params=g["tool_params"])
            elif ts == "created" or (ts is None and g["prim"].kind == "composite"):
                desc = holder["desc"].get(g["piece"]["id"])        # the descriptor the pieces were grouped by
                store.add_use(tt, path, g["piece"]["id"], g["score"], "created", construction=g["prim"], descriptor=desc)
            else:
                store.add_use(tt, path, g["piece"]["id"], g["score"], ts or "pattern")
        except Exception:                                           # the store is bookkeeping, never a verdict
            pass

    def accept(g, it, val, before, n):
        """A feature into the program, the books and the step that says so."""
        commit(g, it)
        g["accepted"] = True
        c = g.get("checks") or {}
        g["validation"] = dict(val, **{k: c.get(k) for k in ("common_volume", "beyond_piece_volume",
                                                             "left_of_piece_volume", "harm_volume")})
        remember(g)
        if g["piece"]["id"] in holder["rejected_ids"]:      # a verdict of before (a failed guess) is replaced
            holder["rejected_ids"].discard(g["piece"]["id"])
            tally["rejected"] -= 1
        if g.get("subparts"):
            tally["subparts_applied"] += 1
        if g.get("fallback"):
            tally["voxel_fallbacks_applied"] += 1
        elif holder["desc"].get(g["piece"]["id"]) is not None:   # known from now on (small_kept)
            holder["accepted"].append((g["piece"]["id"], holder["desc"][g["piece"]["id"]]))
        note(g["piece"]["id"], tool_type=g.get("tool_type"))
        note(g["piece"]["id"], verdict="accepted", reason="", kind=g["prim"].kind, source=g["source"],
             score=g["score"], applied=g["prim"])
        tally["applied"] += 1
        return step("feature_accepted", iteration=it, piece_id=g["piece"]["id"], op=g["op"], primitive=g["prim"],
                   source=g["source"], score=g["score"], validation=dict(g["validation"]),
                   fallback=g.get("fallback"),
                   **tool_info(g["piece"], g["source"], g.get("tool_type")),
                   message=f"{g['op']} {_what(g['prim'])} for piece #{g['piece']['id']} from {g['source']} "
                           f"(score {g['score']:.3f}): exact residual {before:.4g} -> {val['residual_after']:.4g}"
                           + (f" - FALLBACK, {g['fallback']['left_volume']:.4g} of the piece left"
                              if g.get("fallback") else "")
                           + (" [feature check, applied at once]" if val.get("method") == "feature check" else
                              f" [{val.get('method')}, {val.get('difference_seconds') or 0.0:.0f} s of "
                              f"{val.get('difference_timeout')} s]")
                           + (f" (validated together with {n - 1} more)" if n > 1 else ""))

    def apply_now(grp, it):
        """ResidualConfig.feature_apply_at_once: the features of `grp` whose exact check has numbers and whose gain
        (what the tool shares with its piece less what it harms) is enough join the program now; returns the others
        (no check: they wait for the batch's validation)."""
        left = []
        for g in grp:
            c = g.get("checks")
            if not c or c.get("common_volume") is None:
                left.append(g)
                continue
            gain = float(c["common_volume"]) - float(c.get("harm_volume") or 0.0)
            need = cfg.feature_min_gain * g.get("need_volume", g["piece"]["volume"])
            if gain < need and g["score"] >= cfg.feature_min_score:
                # the check and the sampled score disagree (liner, 2026-09-30: the check's boolean gave a shared
                # volume of 0.0 for a tool of sampled precision 1.0, recall 0.98): no verdict here, the batch's
                # build and difference decide
                g["check_disagrees"] = f"the check gives a gain of {gain:.4g}, the sampled score is {g['score']:.3f}"
                note(g["piece"]["id"], check_error=g["check_disagrees"])
                left.append(g)
                continue
            if gain < need:
                yield rejected(it, g["piece"], f"check: the tool gains {gain:.4g} of its piece, it must {need:.4g}",
                               prim=g["prim"], source=g["source"], score=g["score"])
                continue
            if not holder["unconfirmed"]:                           # what a failed confirmation goes back to
                holder["before_at_once"] = dict(
                    residual=holder["residual"], now=dict(now), operations=len(result.operations),
                    s_model=None if state is None else state.s_model.copy(),
                    cut_history=None if state is None else state.cut_history.copy())
            before = holder["residual"]
            val = dict(valid=None, n_solids=None, residual_before=before, residual_after=before - gain, exact_iou=None,
                       seconds=0.0, batch=1, method="feature check", confirmed=False)
            s_acc = accept(g, it, val, before, 1)
            holder["residual"] = before - gain
            mine, other = ("invented", "missing") if g["op"] == "cut" else ("missing", "invented")
            now[mine] = max(0.0, now[mine] - float(c["common_volume"]))
            now[other] = now[other] + float(c.get("harm_volume") or 0.0)
            now["iou"] = None                                        # not measured until the confirmation
            holder["unconfirmed"].append(g)
            yield s_acc
        return left

    def confirm(it, base, holder, out, n_build, v_in):
        """The features applied at once, confirmed: one build of the program, one difference. Returns the build
        counter."""
        grp = list(holder["unconfirmed"])
        if not grp:
            return n_build
        base = holder.get("base", base)
        yield step("feature_confirm", iteration=it, features=len(grp),
                   message=f"confirming {len(grp)} feature(s) applied at once: the whole model is built in FreeCAD and "
                           f"differenced against the input (each at most {cfg.feature_call_timeout} s) ...")
        t, reason, b, d, st, n_solids = time.time(), None, None, None, None, 0
        try:
            b = build(result, out, f"feature_model_{n_build}", cfg.feature_call_timeout)
            n_build += 1
            n_solids = int(b.get("n_solids") or 0)
            if not b.get("step"):
                reason = f"FreeCAD built nothing ({b.get('error', 'no STEP')})"
            elif not b.get("valid") and base.get("valid"):
                reason = "FreeCAD: the solid is not valid"
            elif n_solids > holder["max_solids"]:
                reason = f"FreeCAD: {n_solids} solids"
            else:
                if holder.get("state") is not None and holder["state"]["model"] != str(base["step"]):
                    holder["state"] = None
                d, st = differ(b["step"], tools_of(grp), None, 0, None, f"model_{n_build - 1}")
        except Exception as exc:
            reason = f"FreeCAD: {str(exc)[-200:]}"
        holder["unconfirmed"] = []
        back = holder.pop("before_at_once", None)
        if reason is not None and back is not None:                  # taken back, then validated in halves
            del result.operations[back["operations"]:]
            if state is not None and back["s_model"] is not None:
                state.s_model, state.cut_history = back["s_model"], back["cut_history"]
            holder["residual"] = back["residual"]
            now.update(back["now"])
            for g in grp:
                g["accepted"] = False
                tally["applied"] -= 1
                if g.get("subparts"):
                    tally["subparts_applied"] -= 1
                if g.get("fallback"):
                    tally["voxel_fallbacks_applied"] -= 1
                holder["accepted"] = [a for a in holder["accepted"] if a[0] != g["piece"]["id"]]
            yield step("feature_confirmed", iteration=it, features=len(grp), confirmed=False, reason=reason,
                       seconds=time.time() - t,
                       message=f"not confirmed ({reason}): the {len(grp)} feature(s) are taken back and validated in "
                               f"halves")
            n_build = yield from validate(grp, it, base, holder, out, n_build, v_in, force=True)
            return n_build
        holder["base"] = dict(step=b["step"], valid=bool(b.get("valid")), n_solids=n_solids)
        if d is not None and "error" not in d:
            holder["state"] = st
            now.update(missing=float(d["r_plus"]), invented=float(d["r_minus"]), iou=d.get("iou"))
            holder["residual"] = now["missing"] + now["invented"]
            text = (f"exact residual {holder['residual']:.4g}"
                    + (f", exact IoU {d['iou']:.4f}" if d.get("iou") is not None else "")
                    + f" [{d.get('method')}, {d['seconds']['wall']:.0f} s]")
        else:
            holder["state"] = None
            text = (f"the model is one valid solid; the difference gave no answer "
                    f"({str((d or {}).get('error'))[-160:]}), the residual {holder['residual']:.4g} is the sum of the "
                    f"features' exact checks")
        yield step("feature_confirmed", iteration=it, features=len(grp), confirmed=True, seconds=time.time() - t,
                   exact_iou=now.get("iou"), residual=holder["residual"], method=(d or {}).get("method"),
                   message=f"confirmed, {len(grp)} feature(s), {time.time() - t:.0f} s: {text}")
        return n_build

    def validate(grp, it, base, holder, out, n_build, v_in, force=False):
        """Validate a group of guesses on top of the current model; bisect on a failure. Returns the build counter."""
        if holder.get("unconfirmed"):                               # first what was applied at once
            n_build = yield from confirm(it, base, holder, out, n_build, v_in)
        base = holder.get("base", base)
        before = holder["residual"]
        if not force and time.time() > deadline:
            for g in grp:
                yield rejected(it, g["piece"], "time budget: not validated", prim=g["prim"], source=g["source"],
                               score=g["score"])
            return n_build
        trial = _snapshot(result)
        for g in grp:
            trial.operations.append(Operation(g["op"], g["prim"], component_voxels=0))
        t = time.time()
        val = dict(valid=False, n_solids=0, residual_before=before, residual_after=None, exact_iou=None, seconds=0.0,
                   batch=len(grp))
        reason, b, d, trial_state = None, None, None, None
        try:
            b = build(trial, out, f"feature_model_{n_build}", cfg.feature_call_timeout)
            n_build += 1
            val.update(valid=bool(b.get("valid")), n_solids=int(b.get("n_solids") or 0))
            if not b.get("step"):
                reason = f"FreeCAD built nothing ({b.get('error', 'no STEP')})"
            elif not b.get("valid") and base.get("valid"):
                reason = "FreeCAD: the solid is not valid"
            elif val["n_solids"] > holder["max_solids"]:
                reason = f"FreeCAD: {val['n_solids']} solids"
            else:
                if holder.get("state") is not None and holder["state"]["model"] != str(base["step"]):
                    holder["state"] = None                          # the state of another model
                d, trial_state = differ(b["step"], tools_of(grp), None, 0, None, f"model_{n_build - 1}")
                val.update(difference_seconds=d["seconds"]["wall"], difference_timeout=d["timeout"],
                           difference_fallback_from=d.get("fallback_from"))
                # a partial tool (hole stacks of a piece with more in it) owes the gain of what it covers
                need = sum(cfg.feature_min_gain * g.get("need_volume", g["piece"]["volume"]) for g in grp)
                if "error" in d and judge_fn() is not None:
                    # OCCT failed on this pair (coincident faces): the sampled judge, before and after on one point set
                    J, occt_error = judge_fn(), d["error"]
                    mb, ib = J.residual(program_model(result))
                    ma, ia = J.residual(program_model(trial))
                    val.update(method="sampled", occt_error=occt_error[-200:], residual_after=before + (ma + ia) - (mb + ib),
                               sampled_before=mb + ib, sampled_after=ma + ia)
                    d = dict(r_plus=now["missing"] + ma - mb, r_minus=now["invented"] + ia - ib, iou=None)
                    if (mb + ib) - (ma + ia) < need:
                        reason = (f"the sampled residual drops by {(mb + ib) - (ma + ia):.4g}, it must by {need:.4g} "
                                  f"(OCCT failed: {occt_error[-120:]})")
                elif "error" in d:
                    reason = f"exact difference: {d['error']}"
                else:
                    after = float(d["r_plus"]) + float(d["r_minus"])
                    val.update(residual_after=after, exact_iou=d.get("iou"), method=d.get("method") or "occt")
                    if before - after < need:
                        reason = f"the exact residual drops by {before - after:.4g}, it must by {need:.4g}"
                if reason is None and any(g.get("rough") for g in grp) and now["missing"] is not None:
                    # a rough guess may not trade the part for its piece (also when the FreeCAD check gave no numbers)
                    lost = float(d["r_plus"]) - float(now["missing"])
                    grown = float(d["r_minus"]) - float(now["invented"])
                    cap = cfg.feature_rough_max_harm_input * v_in
                    if max(lost, grown) > cap:
                        reason = (f"rough guess: {'removes' if lost >= grown else 'adds'} {max(lost, grown):.4g} mm3 of "
                                  f"{'the part' if lost >= grown else 'material outside it'}, at most "
                                  f"{cfg.feature_rough_max_harm_input:.0%} of the input ({cap:.4g})")
        except Exception as exc:                                     # a timeout or a crash of the build is a verdict
            reason = f"FreeCAD: {str(exc)[-200:]}"
        val["seconds"] = time.time() - t
        if reason is None:
            for g in grp:
                yield accept(g, it, val, before, len(grp))
            holder["residual"] = val["residual_after"]
            holder["base"] = dict(step=b["step"], valid=bool(b.get("valid")), n_solids=val["n_solids"])
            holder["state"] = trial_state                           # None after a sampled verdict: no exact pieces
            now.update(missing=float(d["r_plus"]), invented=float(d["r_minus"]), iou=d.get("iou"))
            return n_build
        if len(grp) == 1:
            g = grp[0]
            yield rejected(it, g["piece"], reason, prim=g["prim"], source=g["source"], score=g["score"], validation=val)
            return n_build
        if time.time() > deadline:
            for g in grp:
                yield rejected(it, g["piece"], f"time budget: its batch of {len(grp)} failed ({reason}) and was not "
                                               f"bisected", prim=g["prim"], source=g["source"], score=g["score"],
                               validation=val)
            return n_build
        half = len(grp) // 2
        n_build = yield from validate(grp[:half], it, base, holder, out, n_build, v_in, force=True)
        n_build = yield from validate(grp[half:], it, base, holder, out, n_build, v_in, force=True)
        return n_build

    holder.update(residual=None, max_solids=1, next_id=1, v_in=1.0, parents={}, thin={}, suppressed=[], unconfirmed=[])
    # the torus guard's yardstick (reco_engine/torus_guard.py): the input's diagonal, from its mesh, else the grid's
    V_d = getattr(field, "vertices", None) if field is not None else None
    V_d = np.asarray(V_d, float) if V_d is not None else None
    if V_d is not None and len(V_d):
        holder["part_diagonal"] = float(np.linalg.norm(V_d.max(axis=0) - V_d.min(axis=0)))
    elif state is not None:
        holder["part_diagonal"] = state.part_diagonal
    if cfg.feature_thin_wall_mm is not None:
        holder["thin_threshold"] = float(cfg.feature_thin_wall_mm)
    elif field is not None and cfg.feature_thin_wall > 0:
        V_f = np.asarray(field.vertices, float)                       # the input's diagonal, from its mesh
        if len(V_f):
            holder["thin_threshold"] = float(cfg.feature_thin_wall) * float(np.linalg.norm(V_f.max(axis=0) - V_f.min(axis=0)))
    stopped = None
    try:
        stopped = yield from rounds(holder)
    except Exception as exc:
        stopped = f"error: {str(exc)[-300:]}"
    fcstd, doc_note = None, ""
    if cfg.feature_document and holder.get("iterations") and cfg.feature_difference is None:
        try:
            fcstd, doc_note = _write_document(cfg, path, holder, records, now)
        except Exception as exc:                                     # the document never stops the loop
            doc_note = f"; features.FCStd not written: {str(exc)[-200:]}"
    info = dict(iterations=tally["iterations"], applied=tally["applied"], rejected=tally["rejected"],
                skipped=tally["skipped"], differences=tally["differences"], groups=tally["groups"],
                transferred=tally["transferred"], transfer_fallbacks=tally["transfer_fallbacks"],
                group_retries=tally["group_retries"], voxel_fallbacks=tally["voxel_fallbacks"],
                voxel_fallbacks_applied=tally["voxel_fallbacks_applied"],
                subparts=tally["subparts"], subparts_applied=tally["subparts_applied"],
                split_pieces=tally["split_pieces"], segments=tally["segments"],
                tools_by_source=dict(tally["tools"]), store_size=len(store), store_new=len(store) - store_before,
                segments_accepted=tally["segments_accepted"], difference_methods=dict(tally["methods"]),
                stopped=stopped or "error: no verdict", seconds=time.time() - t0, fcstd=fcstd,
                exact_iou=now["iou"], missing_volume=now["missing"], invented_volume=now["invented"],
                thin_threshold=holder.get("thin_threshold") or 0.0, suppressed=list(holder["suppressed"]),
                suppressed_pieces=len(holder["suppressed"]), check_failures=tally.get("check_failures", 0),
                check_error=holder.get("check_error"),
                suppressed_volume=float(sum(q["volume"] for q in holder["suppressed"])))
    iou = info["exact_iou"]
    yield step("feature_done", **info,
               message=f"feature phase: {info['stopped']}; {info['applied']} applied, {info['rejected']} rejected, "
                       f"{info['skipped']} skipped in {info['iterations']} iteration(s), {info['transferred']} by "
                       f"pattern, {info['differences']} exact difference(s) with pieces"
                       + (" (" + ", ".join(f"{v} {k}" for k, v in info["difference_methods"].items()) + ")"
                          if info["difference_methods"] else "") + f", {info['seconds']:.0f} s"
                       + (f", exact IoU {iou:.4f}" if iou is not None else "")
                       + (f"; tools {', '.join(f'{k} {v}' for k, v in info['tools_by_source'].items())}"
                          if info["tools_by_source"] else "")
                       + f"; tool store {info['store_size']} ({info['store_new']:+d})"
                       + (f"; voxel fallback for {info['voxel_fallbacks']} piece(s), {info['voxel_fallbacks_applied']} "
                          f"applied" if info["voxel_fallbacks"] else "")
                       + (f"; {info['suppressed_pieces']} super-thin piece(s) suppressed, "
                          f"{info['suppressed_volume']:.4g} left in the residual" if info["suppressed_pieces"] else "")
                       + (f"; {info['check_failures']} FreeCAD check call(s) failed, their features went unchecked "
                          f"({info['check_error']})" if info["check_failures"] else "")
                       + (f"; FreeCAD document {fcstd}" if fcstd else "") + doc_note + hooks_note)
    if store_path:
        try:
            store.save(store_path)
        except OSError:
            pass
    return info


def _load_piece_feedback(src) -> List[Dict]:
    """The GUI's piece feedback entries (reco_gui/pieces_view.py's piece_feedback.json: {piece id: dict(verdict, kind,
    note, piece: dict(number, sign, volume, centre, faces, group))}), from a file or every such file under a directory;
    each entry gets "_file". [] without a source."""
    import json as _json
    if not src:
        return []
    paths = [Path(src)] if Path(src).is_file() else sorted(Path(src).rglob("piece_feedback.json"))
    out = []
    for f in paths:
        try:
            data = _json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        items = data.values() if isinstance(data, dict) else data
        for e in items:
            if isinstance(e, dict):
                out.append(dict(e, _file=str(f)))
    return out


SUMMARY_COLUMNS = ("id", "iteration", "parent", "side", "op", "volume", "share", "kind", "source", "score", "precision",
                   "recall", "verdict", "reason", "common_volume", "beyond_piece_volume", "left_of_piece_volume",
                   "harm_volume")


def _write_document(cfg: ResidualConfig, input_path: str, holder: Dict, records: Dict, now: Dict):
    """features.FCStd next to the phase's files (reco_field.exact_check.write_features_document): (path, note)."""
    from reco_field.exact_check import write_features_document
    from reco_field.export import program_to_dict
    its = holder["iterations"]
    out = Path(its[0]["model_step"]).parent
    iou_before = its[0].get("iou")

    def label(r):
        return (f"piece {r['id']:03d} {r.get('op', '')} {r.get('volume', 0.0):.4g} mm3 {r.get('verdict') or 'not guessed'}"
                + (f" {r['kind']}" if r.get("kind") else "") + (f" score {r['score']:.3f}" if r.get("score") is not None else ""))

    def prim_job(prim):
        """The primitive for the document; a composite as its tree, every leaf with its own program."""
        d = dict(kind=prim.kind, dims_mm=dict(prim.dims_mm or {}), center_mm=list(prim.center_mm),
                 quaternion=list(prim.quaternion), u=prim.u, v=prim.v, label=prim.label)
        if prim.kind == "composite":
            d.update(combine=prim.combine, parts=[prim_job(q) for q in prim.parts])
        else:
            d["program"] = program_to_dict(primitive_to_sp(prim))
        return d
    iterations = []
    for itn in its:
        pieces = [dict(id=r["id"], sign=r["sign"], volume=r["volume"], centre=r["centre"], label=label(r),
                       verdict=r.get("verdict") or "not guessed")
                  for r in records.values() if r.get("iteration") == itn["iteration"] and r.get("sign")]
        iterations.append(dict(itn, pieces=pieces))
    features, rejected = [], []
    for r in records.values():
        prim = r.get("applied") if r.get("verdict") == "accepted" else r.get("best")
        if prim is None or prim.kind in ("halfspace", UNKNOWN):
            continue
        entry = dict(id=r["id"], primitive=prim_job(prim), program=program_to_dict(primitive_to_sp(prim)),
                     label=f"piece {r['id']:03d} {r.get('op', '')} {r.get('tool_type') or prim.kind} from {r.get('source')} "
                           f"score {r.get('score') or 0:.3f}", piece_step=r.get("step"))
        (features if r.get("verdict") == "accepted" else rejected).append(entry)
    rows = [{k: r.get(k) for k in SUMMARY_COLUMNS} for r in sorted(records.values(), key=lambda r: r["id"])]
    job = dict(input=str(input_path), rough=its[0]["model_step"], final=str(holder["base"]["step"]),
               iterations=iterations, features=features, rejected=rejected, rows=rows, columns=list(SUMMARY_COLUMNS),
               iou_before=iou_before, iou_after=now.get("iou"), path=str(out / "features.FCStd"),
               macro=str(out / "features_colours.FCMacro"))
    with timed("feature document"):
        r = write_features_document(job, timeout=cfg.feature_document_timeout)
    if "error" in r:
        return None, f"; features.FCStd not written: {r['error'][-200:]}"
    holder["document"] = r
    return r["path"], ""


@timed_fn("best candidate")
def _clip_to_part(state: "_State", p: Primitive, comp: Optional[np.ndarray] = None) -> Primitive:
    """An add box whose axes run along the world axes, clipped to the component it explains (its voxels' bounding box
    plus one voxel; without `comp`, the part's box) when it reaches beyond. Measured on the frame (2026-09-25, pitch
    0.1200): two transverse plates were fitted to ~40 surface points as boxes far longer than the part and rejected by
    the outside check (the plates stayed missing); clipped to the part's box they came out 10.6 wide where the plates are
    7.1 and needed four repair cuts. Any other primitive is returned as it is."""
    if p.kind != "box" or not isinstance(state.spec, cascade.BoxGridSpec):
        return p
    from reco_engine.result_to_program import rotation_matrix
    R = rotation_matrix(p)
    A = np.abs(R)
    if not np.all(A.max(axis=0) > 1 - 1.5e-6):                        # not axis-aligned: no exact clip
        return p
    perm = A.argmax(axis=0)                                            # local axis j runs along world axis perm[j]
    h = np.array([p.dims_mm["x_half"], p.dims_mm["y_half"], p.dims_mm["z_half"]])
    c = np.asarray(p.center_mm, float).copy()
    lo, hi = np.asarray(state.spec.lo, float), np.asarray(state.spec.hi, float)
    if comp is not None and comp.any():
        idx = np.argwhere(comp)
        c_lo = state.centres[tuple(idx.min(axis=0))] - 1.5 * state.pitch     # the component's cells, + one voxel
        c_hi = state.centres[tuple(idx.max(axis=0))] + 1.5 * state.pitch
        lo, hi = np.maximum(lo, c_lo), np.minimum(hi, c_hi)
    changed = False
    for j in range(3):
        w = perm[j]
        a, b = max(c[w] - h[j], lo[w]), min(c[w] + h[j], hi[w])
        if b <= a:
            return p
        if a > c[w] - h[j] + 1e-12 or b < c[w] + h[j] - 1e-12:
            c[w], h[j], changed = (a + b) / 2.0, (b - a) / 2.0, True
    if not changed:
        return p
    q = Primitive(kind="box", quaternion=p.quaternion, center_mm=tuple(float(x) for x in c),
                  dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])), confidence=p.confidence,
                  sources=list(p.sources), label=(p.label + ", " if p.label else "") + "clipped to its component")
    return q


def _component_views(state: "_State", comp_win: np.ndarray, sub, field, cfg: ResidualConfig):
    """[(view name, channels, spec, field, (R, t))] of one component's window: the varied inputs of `vary_on_failure`.
    (R, t) maps a primitive found in the view back to the input's frame (x -> R x + t); None for the unrotated views."""
    from scipy import ndimage
    from scipy.spatial.transform import Rotation
    from src.closest_point_field import SurfaceField
    views = []
    for name, mask in (("opened", ndimage.binary_opening(comp_win)), ("closed", ndimage.binary_closing(comp_win))):
        if mask.sum() >= cfg.min_component and not np.array_equal(mask, comp_win):
            views.append((name, _voxel_channels(mask, sub), sub, field, None))
    for z in (2.0, 0.5):                                                   # finer / coarser quality of the same piece
        m = ndimage.zoom(comp_win.astype(np.uint8), z, order=0) > 0
        if m.sum() >= cfg.min_component:
            zs = cascade.BoxGridSpec(tuple(sub.lo), tuple(int(n) for n in m.shape), sub.pitch / z)
            views.append((f"pitch x{1 / z:g}", _voxel_channels(m, zs), zs, field, None))
    if field is not None and cfg.vary_rotations > 0:
        rng = np.random.default_rng(0)
        c = np.asarray(sub.lo, float) + np.asarray(sub.shape, float) * sub.pitch / 2.0     # the window's centre
        n = int(np.ceil(np.linalg.norm(sub.shape)))                           # a rotated window holds the whole piece
        for k in range(cfg.vary_rotations):
            R = Rotation.random(random_state=rng).as_matrix()
            rs = cascade.BoxGridSpec(tuple(c - n * sub.pitch / 2.0), (n, n, n), sub.pitch)
            x_rot = rs.centres().reshape(-1, 3)
            x = (x_rot - c) @ R + c                                            # R^T (x' - c) + c: back into the input frame
            idx = (x - np.asarray(sub.lo, float)) / sub.pitch - 0.5
            m = ndimage.map_coordinates(comp_win.astype(np.uint8), idx.T, order=0, mode="constant").reshape(n, n, n) > 0
            if m.sum() < cfg.min_component:
                continue
            rf = SurfaceField((np.asarray(field.vertices, float) - c) @ R.T + c, np.asarray(field.faces))
            views.append((f"rotation {k + 1}", _voxel_channels(m, rs), rs, rf, (R.T, c - R.T @ c)))
    return views


def _tool_candidates(state: "_State", comp: np.ndarray, single: List[Primitive], cfg: ResidualConfig) -> List[Primitive]:
    """Boolean results as tools for one piece: the pairwise booleans of its best three single candidates (A & B,
    A - B, B - A, A | B) and the library of accepted operations, each moved to the piece's centre (orientation kept)."""
    import copy
    out: List[Primitive] = []
    best = [c for c in single if c.kind in ("box", "cylinder", "sphere", "cone", "composite")][:3]
    for i in range(len(best)):
        for j in range(len(best)):
            if i == j:
                continue
            a, b = best[i], best[j]
            pairs = [("common", "&"), ("cut", "-")] + ([("fuse", "|")] if i < j else [])
            if i > j:
                pairs = [("cut", "-")]                                    # & and | are symmetric: once is enough
            for combine, sym in pairs:
                out.append(Primitive(kind="composite", center_mm=a.center_mm, dims_mm={}, parts=[a, b], combine=combine,
                                     label=f"tool ({a.kind} {sym} {b.kind})"))
    if state.tools:
        idx = np.argwhere(comp)
        centre = (state.centres[tuple(idx.min(axis=0))] + state.centres[tuple(idx.max(axis=0))]) / 2.0
        for t in state.tools[-cfg.tool_library_size:]:
            moved = copy.deepcopy(t)
            shift = centre - np.asarray(t.center_mm, float)
            from reco_engine.result_to_program import transform_primitive
            moved = transform_primitive(moved, np.eye(3), shift)
            moved.label = f"library tool ({_what(t)})"
            out.append(moved)
    return out


def _what(p: Primitive) -> str:
    """A primitive's name in messages: its kind, a composite's label ("box & cylinder"), a cover's label."""
    if p.kind == "composite":
        return p.label or f"composite ({p.combine})"
    if p.kind == "halfspace":
        return "half-space"
    return p.kind + (f" ({p.label})" if p.label in ("oriented bounding box", "bounding cylinder") else "")


@timed_fn("bounding covers")
def _covers(state: _State, field) -> List[Tuple[str, Primitive]]:
    """The part's bounding covers (oriented box, bounding cylinder), computed once per run: the primitives an "intersect"
    trims the model to. The one the first add already is gains nothing and is rejected by min_gain."""
    if state.covers is None:
        covers = []
        for name, make in (("oriented bounding box", cascade.oriented_bounding_box), ("bounding cylinder", cascade.bounding_cylinder)):
            try:
                p = make(state.t_in, state.spec, field=field)
            except Exception:                                          # a cover is advisory
                p = None
            if p is not None:
                p.label = name
                covers.append((name, p))
        state.covers = covers
    return state.covers


def _grid_blindness(state, field) -> Optional[float]:
    """The loop grid's pitch over the part's thinnest wall tau (dataset_gen.src.local_grids: 5th percentile of the
    thickness along the inward normal). Above 1 a wall can fall between voxel centres, and the voxel residual cannot
    see it (D-005: voxels are ground truth only at pitch <= tau / 3). None if it cannot be measured."""
    try:
        from src.local_grids import wall_thickness
        _, _, tau = wall_thickness(field.vertices, field.faces, n=2000)
        tau = tau[np.isfinite(tau) & (tau > 0)]
        if not len(tau):
            return None
        return float(state.pitch / np.percentile(tau, 5))
    except Exception:
        return None


@timed_fn("best candidate")
def _best_candidate(state: _State, comp: np.ndarray, op: str, field, registry, cfg: ResidualConfig, purity_min: float) -> Dict:
    """Every candidate for one component, each with its own operation: the cascade's winner and its alternatives, void_fit's
    void for a cut, and with cfg.more_booleans also strict-tolerance cylinders, half-space cuts from strict planes,
    "intersect" with the part's bounding covers and two-primitive composites (box & / - cylinder). Boxes and cylinders get
    their open faces extended. The passing one with the largest net gain wins, if it gains at least cfg.min_gain; for a
    cut component with cfg.surface_choice the one most of the nearby surface lies on among those with >= a share of the
    best gain. `best` is (primitive, evaluation); the evaluation's "op" is the operation to apply."""
    sub, sl = _window(state, comp)
    channels = _voxel_channels(comp[sl], sub)
    scope = (~state.occ_model if op == "add" else state.occ_model)[sl]
    prim = cascade.run_cascade(channels, sub, registry, field=field, top_k=cfg.top_k, models=cfg.models,
                               scope=scope, pass_iou=cfg.component_pass_iou, part_diagonal=state.part_diagonal)
    single = ([] if prim.is_unknown else [prim]) + [a for a in prim.alternatives if not a.is_unknown]
    n_comp = int(comp.sum())
    if op == "cut" and cfg.use_void_fit and n_comp <= cfg.void_fit_max_voxels:
        vf = _void_fit_primitive(state, comp, purity_min)
        if vf is not None:
            single.append(vf)
    if cfg.excess_boxes:                                          # cuts since 2026-09-26, adds too since 2026-10-15
        for blo, bhi in largest_boxes(comp, cfg.excess_boxes, min_voxels=cfg.min_component):
            b = _cube_primitive(state, blo, bhi)
            b.label = "a largest box of the piece"
            snapped = _snap_box_faces(state, b, field)                  # its faces onto the part's faces next to them
            if snapped is not None:
                single.append(snapped)
            single.append(b)
    surface = cfg.surface_choice and op == "cut" and field is not None
    booleans = cfg.more_booleans and field is not None
    strict = []
    if surface or booleans:
        with timed("strict cylinders"):
            strict = cascade.strict_cylinders(channels, sub, field, scope=scope)
    if surface and op == "cut":
        single += strict
    cands = [(op, c) for c in single]
    if booleans:
        with timed("boolean candidates"):
            cyls = [c for c in single + strict if c.kind == "cylinder"][:4]
            spheres = cascade.strict_spheres(channels, sub, field, scope=scope)
            # the piece's own extent is the natural box operand: the leftover above a dome is its bounding box minus the
            # dome's sphere (the cascade's box fit of the piece was a different box: box - sphere cut into the dome)
            idx = np.argwhere(comp)
            lo_w = state.centres[tuple(idx.min(axis=0))] - state.pitch / 2.0
            hi_w = state.centres[tuple(idx.max(axis=0))] + state.pitch / 2.0
            extent = Primitive(kind="box", center_mm=tuple(float(x) for x in (lo_w + hi_w) / 2),
                               dims_mm=dict(zip(("x_half", "y_half", "z_half"), (float(v) for v in (hi_w - lo_w) / 2))),
                               label="the piece's extent")
            boxes = [extent] + [c for c in single if c.kind == "box"][:2]
            cands += [(op, c) for c in cascade.composites(boxes, cyls + spheres)]
            if op == "cut":
                cands += [("cut", h) for h in cascade.strict_planes(channels, sub, field, scope=scope)]
                cands += [("intersect", c) for _, c in _covers(state, field)]
        if cfg.boolean_tools:
            with timed("boolean tools"):
                cands += [(op, t) for t in _tool_candidates(state, comp, single, cfg)]
    tried, best = [], None
    for cand_op, c in cands:
        if cand_op == "add":
            c = _clip_to_part(state, c, comp)
        ev = _evaluate(state, c, cand_op, purity_min)
        if cand_op in ("add", "cut") and ev["ok"] and ev["gain"] >= cfg.min_gain:
            c, ev = _extend_open_faces(state, c, cand_op, purity_min, ev)
        ev["op"] = cand_op
        new_boolean = cand_op == "intersect" or c.kind in ("halfspace", "composite")
        if surface or (booleans and new_boolean):
            with timed("surface agreement"):
                ev["surface"] = cascade.near_surface_agreement(c, channels, sub, field, scope=scope)
        if new_boolean and ev["ok"] and (ev.get("surface") or 0.0) < cfg.boolean_min_surface:
            ev.update(ok=False, reason=f"only {ev.get('surface') or 0.0:.0%} of the nearby surface on it "
                                       f"(a new boolean needs {cfg.boolean_min_surface:.0%})")
        tried.append((c, ev))
        if ev["ok"] and ev["gain"] >= cfg.min_gain and (best is None or ev["gain"] > best[1]["gain"]):
            best = (c, ev)
    if best is None and cfg.vary_on_failure:
        from reco_engine.result_to_program import transform_primitive
        with timed("varied views"):
            for view, v_ch, v_spec, v_field, back in _component_views(state, comp[sl], sub, field, cfg):
                try:
                    v = cascade.run_cascade(v_ch, v_spec, registry, field=v_field, top_k=cfg.top_k, models=cfg.models,
                                            pass_iou=cfg.component_pass_iou, part_diagonal=state.part_diagonal)
                except Exception:                                          # a view is advisory
                    continue
                for c in ([] if v.is_unknown else [v]) + [a for a in v.alternatives if not a.is_unknown]:
                    if back is not None:
                        c = transform_primitive(c, *back)
                    c.label = ((c.label + ", ") if c.label else "") + f"from the {view} view"
                    if op == "add":
                        c = _clip_to_part(state, c, comp)
                    ev = _evaluate(state, c, op, purity_min)
                    if ev["ok"] and ev["gain"] >= cfg.min_gain:
                        c, ev = _extend_open_faces(state, c, op, purity_min, ev)
                    ev["op"] = op
                    if surface:
                        ev["surface"] = cascade.near_surface_agreement(c, channels, sub, field, scope=scope)
                    tried.append((c, ev))
                    if ev["ok"] and ev["gain"] >= cfg.min_gain and (best is None or ev["gain"] > best[1]["gain"]):
                        best = (c, ev)
    if surface and best is not None:
        floor = cfg.surface_min_gain_share * best[1]["gain"]
        passing = [(c, ev) for c, ev in tried if ev["ok"] and ev["gain"] >= max(cfg.min_gain, floor)
                   and ev.get("surface") is not None]
        if passing:
            best = max(passing, key=lambda t: (round(t[1]["surface"], 3), t[1]["gain"]))
    if best is not None:
        best[0].alternatives = [c for c, _ in tried if c is not best[0]][:3]
        for a in best[0].alternatives:
            a.alternatives = []
    why = "the cascade found no kind" if not cands else "no candidate gains and passes the checks"
    return dict(best=best, tried=tried, why=why)


def reconstruct_variants(input_path, config: Optional[ResidualConfig] = None, k: int = 3) -> List[Dict]:
    """k solutions of one part (docs/thesis/chapters/10b_solution_spaces.tex, "a set of solutions"): the loop run from
    the automatic start, then from each of the next best-scoring starting shapes of its graded list, each solution with
    its statistics vector. [dict(start, chosen, result, stats, stopped)] in the order run; the selection is the
    caller's (the rule of choose_start on the statistics, or the user in the GUI)."""
    cfg = config or ResidualConfig()
    if cfg.start_shape == "legacy":
        cfg = ResidualConfig(**{**cfg.__dict__, "start_shape": "auto"})
    out, grades = [], None
    def run(start):
        c = ResidualConfig(**{**cfg.__dict__, "start_shape": start})
        add, done = None, None
        for st in residual_steps(input_path, c):
            if st.kind == "add" and add is None:
                add = st
            done = st
        return add, done
    add, done = run(cfg.start_shape)
    if add is None or done is None:
        return out
    grades = add.info.get("start_grades") or []
    chosen = add.info["primitive"].label if add.info.get("primitive") is not None else None
    out.append(dict(start=cfg.start_shape, chosen=chosen, result=done.result, stats=done.info.get("stats"),
                    stopped=done.info.get("stopped")))
    ranked = sorted([(n, g) for n, g in grades if g.get("contains", 0) >= 0.995 and not g.get("unjustified")],
                    key=lambda t: -(t[1].get("score") or 0.0))
    seen = {chosen}
    for name, g in ranked:
        if len(out) >= k:
            break
        if name in seen or name == "cascade" and chosen and chosen.startswith("cascade"):
            continue
        seen.add(name)
        a2, d2 = run(name)
        if d2 is not None:
            out.append(dict(start=name, chosen=name, result=d2.result, stats=d2.info.get("stats"),
                            stopped=d2.info.get("stopped")))
    return out


def reconstruct(input_path, config: Optional[ResidualConfig] = None) -> ReconstructionResult:
    """The finished program: the result of the last step of `residual_steps`."""
    last = None
    for step in residual_steps(input_path, config):
        last = step
    return last.result
