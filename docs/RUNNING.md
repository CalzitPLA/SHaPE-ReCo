# SHaPe ReCo — How to run

Which command, with what, and what you should see. Architecture: [`CODEMAP.md`](CODEMAP.md).
All commands run from the repository root. State described: branch `bolt-holes` (2026-09-29).

Options were checked against `python3 -m reco_gui --help` and the argparse code of the scripts on 2026-09-29.
Quoted outputs and measured numbers carry their date or source. They were not re-run for this revision.

Contents: 1 Prerequisites · 2 Tests · 3 GUI · 4 Main line: cascade add-cut loop · 5 workstation and the servers ·
6 Vector-field route · 7 Legacy voxel route · 8 Inside the FreeCAD GUI · 9 Diagnostics ·
10 Data generation and training · 11 Known limits

Routes: B = cascade add-cut loop with the feature phase (the thesis main line, D-002); A = vector-field peeling
loop (comparison route); C = voxel greedy drivers (legacy).

## 1. Prerequisites

| needs | used for | check |
|---|---|---|
| Python 3.13 with `torch`, `numpy`, `Pillow` | inference, training, voxel math | `python3 -c "import torch, numpy, PIL"` |
| `freecadcmd` (FreeCAD 1.0): `$SHAPE_RECO_FREECADCMD`, else `freecadcmd` on the PATH, else `/usr/bin/freecadcmd` (`dataset_gen/src/freecad_path.py`) | every build of a solid, STEP tessellation, exact booleans | `freecadcmd --version` |
| `libigl` (`pip install --user --break-system-packages libigl`) | exact closest point (`field_backend="igl"`), winding-number inside test of `ibeam.ExactJudge`, wall thickness | `python3 -c "import igl"` |
| `manifold3d` (same way; installed 2026-09-24) | the GUI's cascade model meshes by exact mesh booleans; without it the SDF path, which drops walls thinner than a grid cell | `python3 -c "import manifold3d"` |
| PyQt5 and vtk | the GUI | `python3 -c "import PyQt5, vtk"` |
| `models/registry.json` and the checkpoints it names | route B's cascade stages | `ls models/registry.json` |
| `stltovoxel` (`~/.local/bin/stltovoxel`) | the legacy voxelizer (`SHAPE_RECO_VOXELIZER=stltovoxel`, `--voxel-scale-mode fit`); not needed for the parity default | `which stltovoxel` |
| checkpoints in `checkpoints/{type,rotation,dimension}_cnn/best_model.pth` | route C | `ls checkpoints/*/best_model.pth` |
| FreeCAD GUI (`/usr/bin/freecad`) | only section 8; the user starts it | — |

Checkpoints and datasets are not in git. Train them (section 10) or copy them in.

Environment variables:

| variable | effect | default |
|---|---|---|
| `SHAPE_RECO_FIELD_BACKEND` | closest-point backend: `numpy`, `torch`, `cuda`, `auto`, `igl` | `auto` |
| `SHAPE_RECO_KDTREE_WORKERS` | threads of the CPU tree query | -1 (all cores) |
| `SHAPE_RECO_FIELD_GPU_MB` | cap on the GPU memory of the field's neighbour search | 30 % of the free memory |
| `SHAPE_RECO_FREECADCMD` | the `freecadcmd` to run (a path or a command name); one that names no executable means no FreeCAD, never another one | `freecadcmd` on the PATH, then `/usr/bin/freecadcmd` |
| `SHAPE_RECO_FREECAD_WORKER=1` | one persistent `freecadcmd` per process | off |
| `SHAPE_RECO_VOXELIZER=parity\|stltovoxel` | voxelizer of the in-scale grids (generators, `voxelize_stl_to_bin`); `stltovoxel` is legacy; recorded per sample as `placement.voxelizer` | `parity` (since 2026-10-01) |
| `SHAPE_RECO_CACHE=<dir>` | build, voxelization and prediction cache (route C); also the root of the training caches `voxfield/`, `field_kinds/` (keys: input files' identity + code fingerprint, `dataset_gen/src/cache_keys.py`; entries from before 2026-10-01 are never read again and can be deleted) | route C cache off; scripts that enable it, and the training caches, use `~/.cache/shape_reco` |
| `SHAPE_RECO_WALL_CACHE=<dir>\|off` | wall-thickness cache | `~/.cache/shape_reco/wall_thickness/` |
| `SHAPE_RECO_RESULTS` | the run store | `~/shape_reco_results/` |
| `SHAPE_RECO_TIMINGS_DIR` | the GUI's timing tables | `~/.cache/shape_reco/timings/` |
| `SHAPE_RECO_TEST_PARTS` | parts of the GUI's Random button | `~/shape_reco_parts/external_parts` |
| `SHAPE_RECO_IBEAM=0`, `SHAPE_RECO_HOLLOW=0` | switch a sub-loop of route B off | on |
| `SHAPE_RECO_OP_GUARD` | per-operation guard of route B | off |
| `SHAPE_RECO_VOXEL_MODE`, `SHAPE_RECO_VOXEL_SCALE_MM` | route C voxelizer mode `fit`, `uniform`, `fixed` | `fit` |
| `SHAPE_RECO_BOX_ORIENTER=frame\|<dir>` | route C: network-free box orienter, or a checkpoint directory | off |
| `SHAPE_RECO_FUSED_NET=<dir>` | route C: a `FusedNet` checkpoint as pose and size source | off |
| `SHAPE_RECO_TOPOLOGIES` | route C: restrict the Type CNN | none |
| `SHAPE_RECO_PYTHON` | a torch-capable Python for subprocess prediction, used only when torch is not importable | `/usr/bin/python3` |
| `SHAPE_RECO_SANDBOX_ROOT` | repository root for the FreeCAD nodes and the macro | found by `_sandbox_root.txt` |
| `SHAPE_RECO_TEST_GPU=1` | tests may use the GPU | CPU only |

## 2. Tests

**Do not run the whole suite by default** (more than 15 min on the laptop). Run the test files of the modules you
changed. Heavy jobs run one after the other, never in parallel. Check first:

```bash
pgrep -af "pytest|time_cascade|peel_case"
python3 -m pytest tests/test_reco_gui_cascade.py tests/test_timing.py -q
```

| fact | value | source |
|---|---|---|
| tests collected | 888, of them 762 with `-m "not slow"` | counted 2026-09-24; not counted since |
| expected pass count | not stated | the suite has not been run in full since that count |
| `slow` marker | tests measured at ≥ 20 s | `tests/conftest.py` |
| `--run-slow` | runs the tests marked `slow`; without it they are skipped. The option comes from the installed plugin `pytest-astropy`, not from this repository | `/usr/lib/python3/dist-packages/pytest_astropy/plugin.py` |
| GPU | CPU only unless `SHAPE_RECO_TEST_GPU=1` | `tests/conftest.py` |

Use `--run-slow` only for the one module that needs it. Whether the option exists in workstation's venv was not checked.

Some tests are end-to-end runs (for example `tests/test_add_cut_loop.py`). They need `freecadcmd`, the example parts
or the registry checkpoints and are **skipped** when one is missing or without `--run-slow`;
`python3 -m pytest tests/test_add_cut_loop.py -q -rs` prints why.

| command | what | time |
|---|---|---|
| `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_reco_gui_feature.py -q` | the GUI's feature phase on a fake step stream, no FreeCAD | 37 tests, 11 s (2026-09-27) |
| `python3 -m pytest tests/test_feature_phase.py -q` | the feature phase with fakes | about 1 s (2026-09-27) |
| `python3 -m pytest tests/test_feature_phase.py -q --run-slow` | plus loop runs and the STEP end-to-end | about 70 s (2026-09-27) |
| `python3 -m pytest tests/test_feature_stage.py tests/test_exact_check.py -q` | feature stage and exact judge | about 25 s (2026-09-27) |
| `python3 -m pytest tests/test_g3_harness.py -q` | G3 harness on a stub part | about 1 s |
| `python3 -m pytest tests/test_closest_point_field.py tests/test_reco_field_fit.py tests/test_reco_field_peel.py` | route A | about 7 min |
| `python3 -m pytest tests/test_reco_gui_field.py` | the GUI's vector-field mode, headless | about 40 s |
| `python3 scripts/suite_by_file.py --out <dir> [--jobs 6]` | the whole suite, one process per file | not measured here; ask `resource-guru` first |

## 3. GUI

The GUI needs the user's display. Start it in the background and let the user look; do not block on it.

### 3.1 Start

```bash
python3 -m reco_gui --stl part.stl                                   # the window, part preselected
python3 -m reco_gui --method cascade --run --pace 0.6 --stl part.step   # live: starts by itself, runs to the end
python3 -m reco_gui --random                                         # a random STEP of the downloaded test parts
python3 -m reco_gui --open Frame_4_x_6:latest                        # show a stored run
python3 -m reco_gui.dataset_gen_main                                 # the dataset-generation window
```

| option | meaning | default |
|---|---|---|
| `--stl STL` | part to preselect: an STL, or a STEP in the vector-field and cascade modes | none |
| `--random` | preselect a random STEP of the test parts (ignored with `--stl`) | off |
| `--method {field,voxel,cascade}` | method to preselect | the panel's first item, the vector field |
| `--run` | start at once and run to the end without breakpoints | off |
| `--pace PACE` | seconds to wait after each stage | the toolbar's 0.5 |
| `--verbose` | the log also goes to the terminal, with a heartbeat line every 10 s and a line per candidate | off |
| `--open RUN` | open a stored run: its directory, `part:latest`, `part:previous` or `part:<n>`; ignored with `--run` | none |
| `--benchmark` | open the Benchmark tab (3.7): the stored runs of the chosen part, ranked, rated with aegir on demand | off |
| `--benchmark-run RUN` | with `--benchmark`: show that stored run at once (directory or `part:latest`), with its trees | none |
| `--target-iou X` | cascade: the level of detail, the IoU the feature phase works up to | all details |
| `--features` / `--no-features` | cascade: the feature phase after the rough model | on |
| `--voxels N` | cascade: voxels per side of the cube pitch | 96 |
| `--start START` | cascade: the starting shape, one of `reco_engine.start_shape.START_TYPES` (`auto`, `legacy`, `dominating`, `dominating major`, `cascade`, `oriented box`, `world box`, `cylinder`, `cylinder sector`, `sphere`, `box & cylinder`) | `legacy` |
| `--max-ops N` | cascade: operations at most | the loop's budget |

### 3.2 Desktop launcher

`~/.local/bin/shape-reco-gui` and `~/.local/share/applications/shape-reco-gui.desktop` (a copy on `~/Desktop`) are
outside the repository. The launcher always passes `--method cascade` and, since 2026-10-29, `--window maximized`
(`python3 -m reco_gui --window normal|maximized|full`: the start size, the docks fitted to it). The desktop entry's
actions: *Full screen*, *Run the chosen part live*, *Liner, live*, *Frame, live*.

| call | effect |
|---|---|
| `shape-reco-gui` | the window; the part is chosen in the settings |
| `shape-reco-gui part.stl\|part.step` | that part preselected |
| `shape-reco-gui --live part` | that part, with `--features --run --pace 0.6 --verbose` |
| `shape-reco-gui --liner` | `benchmark_parts/2.0/Liner_LINER_001-LINER_001.step`, live |
| `shape-reco-gui --frame` | `benchmark_parts/1.0/Frame_4_x_6.stp`, live |
| `shape-reco-gui --full [part]` | full screen (F11 toggles), the part preselected if given |

Which checkout it starts, in this order:

1. `$SHAPE_RECO_DIR`, if set.
2. The first line of `~/.config/shape_reco/gui_dir`, if that directory has `reco_gui/__main__.py`. Since the merge of
   2026-10-29 the file holds `~/Documents/Sandbox` (main).
   The log line of each start names the checkout and its commit.
3. Otherwise `~/Documents/Sandbox` or the worktree under `~/Documents/Sandbox-wt/` with the most recently modified
   file in `reco_gui/` or `reco_engine/`.

Its output goes to `~/.cache/shape_reco/reco_gui_launcher.log`. Since 2026-10-10 it exports `CUDA_VISIBLE_DEVICES=""`:
a run in the window crashed with `Fatal Python error: _PyThreadState_Attach: non-NULL old thread state` in
`reco_engine/predict.py` (`_type_tensor`, the `.to(cuda)` of the type input from the worker thread; Python 3.13.5,
torch 2.7.1, PyQt5 5.15.11, on the HEA beam part at 11:13). On the CPU the type net answers in well under a second,
and the cascade's field uses igl, so the window loses nothing. `python3 -m reco_gui --retry-iou X` sets the
"ask for another path" threshold from the command line (§3.4).

**Icon and logo (since 2026-11-05).** The window icon of both windows (`packaging/shape-reco.png`) and the desktop
icon `install.sh` installs (`packaging/shape-reco.svg` to `~/.local/share/icons/hicolor/scalable/apps/`, the PNG to
`256x256/apps/`) are the logo's badge: a poison dart frog in the crow's nest with a spyglass, the skeletal formula of
its poison Batrachotoxin in the sky, name and sum formula on the ring. While no part is loaded and no run computes,
the 3D views show the badge turning slowly like a token (`reco_gui/logo_token.py`, since 2026-11-10). One build writes them and the full logo
(`packaging/logo/shape-reco-3d-logo.svg`, badge and wordmark, and `shape-reco-3d-logo-dark.svg` with a light wordmark
for dark backgrounds; `README.md` shows both as 600 px PNGs, the one matching the viewer's theme, because
GitHub serves SVGs with a CSP that may block the badge's embedded JPEG):

```bash
# the render (Cycles, ~20 s on server_a's GPU; portable Blender 4.3.2 in /data/user/software); also <out>.json
blender -b -P packaging/logo/blender_logo.py -- --out packaging/logo/shape-reco-3d.png --res 2048 --samples 512
python3 packaging/logo/make_logo.py --3d packaging/logo/shape-reco-3d.png   # badge, molecule, wordmark, icons
```

The skeletal formula is Wikipedia's (since 2026-11-10): `packaging/logo/molecule_2d.py` fetches the drawing the article
shows (Wikimedia Commons `File:Batrachotoxin_skeletal.svg`, Ed (Edgar181), public domain), checks its SHA-1 and caches
it unchanged in `packaging/logo/batrachotoxin-skeletal-wikipedia.svg` (source and licence in `batrachotoxin.json`);
the badge keeps its geometry and only restyles it (cream lines thickened from 0.035 to 0.07 bond lengths, O orange,
N pink, a navy halo). Name and sum formula on the ring come from PubChem (CID 6324647) through the MoleculMaker
FreeCAD macro, cached in `packaging/logo/batrachotoxin-2d.sdf`. `python3 packaging/logo/molecule_2d.py [--refetch]`
writes the formula alone (`batrachotoxin-2d.svg`); `python3 packaging/logo/make_logo.py` alone writes the earlier
flat 2D logo.

### 3.3 The window

**The flow (since 2026-10-29; the user: "make it linear; the application fits the screen in full screen").** One
line of steps at the top of the window (`reco_gui/flow_bar.py`):

| step | what it does | done when |
|---|---|---|
| 1 Part | the Settings dock; *Browse ...* when no part is chosen | a part is chosen (its name is the step's note) |
| 2 Starting part | the dialog `reco_gui/start_part_dialog.py`: *automatic* (the path's own start), *around the part* (a box, cylinder, cone, sphere or torus that bounds the part or a sub-part, i.e. the part inside a box region in mm; axis x / y / z / automatic) or *exact* (centre, angles about x / y / z, sizes); "let the reconstruction fit it to the surface" refines it, unticked it is used as given. The hint shows the share of the part the shape holds and the share of the shape inside the part | the dialog is accepted |
| 3 Reconstruct | Start; the alternatives (3.7) follow by themselves | the last alternative has finished |
| 4 Compare | the Benchmark tab: nephograms, histograms, event time, ranking | a result is chosen (*Use this result*) |
| 5 Choose & export | a menu: use the selected result, FreeCAD rebuild, FreeCAD groups, GDML, nephogram, save | - |

Done steps carry a check mark, the current one is highlighted; *Next* does the current step; step 3 runs the
reconstruction to the end (pace and breakpoints apply; a session waiting at "Ready" goes on).

**When the part is reconstructed** (since 2026-10-29), a bar above the 3D view names the result, e.g. "<part>
reconstructed on server_a: exact IoU 0.9992, 20 operations", with **Compare ▸** and **Benchmark** (`reco_gui/result_bar.py`).
It appears after a local run, after a chain of alternatives (the best one first) and after a server run, followed or
started here.
- **Compare** (also flow step 4) is the compare mode, `enter_compare_mode`:
  - the Benchmark tab with the run or runs checked and the histograms of the selected one;
  - the input coloured by its distance to the model in the 3D view (the "Distance" layer), from the stored run's
    nephogram, made in a thread if the run has none (local and server runs alike);
  - for a run in the window without a stored copy, the comparison of the session (`compare_export`). The starting part is the
loop's first guess (`RunSettings.cascade_first_guess` -> `ResidualConfig.first_guess`, the contract in
`reco_engine/user_guess.py`): the first add, judged by the loop's own checks, dropped when it fails. A shape around
the part or a sub-part is a **cover** (`UserShape.role = "cover"`, `region_mm`): it is kept when it holds 99 % of the
input material on the loop's grid (`residual_loop._cover_check`; for a sub-part the material whose voxel centres lie
in the region), never refitted, and the loop cuts its excess. Before 2026-10-29 it was judged by the add's net voxel
gain, which an over-cover cannot have: the box around the ABC plate 00009283 "gains 0 voxels" (12.7 mm thick at pitch
11.66) and a sub-part box of 00005438 "-872" were dropped. An exact shape is a cover when its tick is set (Fill from
'around' sets it), else it must gain material like any add. Under the flow bar
is one compact run bar (Start, Step, Continue, Pause, pace, *Break at*, *follow live*); everything else is in the menus
**File** (Open run, Save run, Compare with the previous run, Dataset generation, Close), **View** (Details, Timeline
and Benchmark, the docks, *Full screen* F11) and **Tools** (Starting part, Compare with the input = the nephogram,
FreeCAD rebuild, FreeCAD groups, GDML export). The one row of 31 toolbar actions it replaced was 2292 px wide and hid
Open, Save, GDML and the rebuild behind the overflow arrow on a 1920 px screen; the window's least width fell from 2292
to 1111 px. The docks are sized to the window (`fit_docks`: Settings 21 %, the right column 19 %, the output 16 % of
the height; again after F11); the Settings dock has no horizontal scrollbar (measured at 1920, 1366 and 1280 px), and
the 3D views' rows of switches wrap instead of scrolling.

The **Reconstruction** switch at the top of the settings dock chooses the method: Cascade (add-cut; the default) or
Vector field. Toolbar, timeline, log, pacing and the *Break at* menu are the same for both. (The voxel mode, route C,
was removed on 2026-09-30.)

| toolbar item | key | what it does |
|---|---|---|
| Start / restart | Ctrl+R | starts the run |
| Step | Space | advances one stage |
| Continue | F5 | runs to the next stage ticked under *Break at*, waiting *pace* seconds after each stage |
| Pause | Esc | stops after the current computation |
| Dataset generation ... | — | opens the dataset-generation window (3.8) |
| FreeCAD groups | — | cascade: the tools as grouped objects of a FreeCAD document (3.5) |
| FreeCAD rebuild | — | cascade: the program as one solid, with its chamfer (3.5) |
| Open | Ctrl+O | shows a stored run (3.7) |
| Benchmark | — | the Benchmark tab of the bottom panel: this part's stored runs, ranked and rated (3.7) |
| Save | Ctrl+S | stores the run (3.7) |
| Compare | — | the run just saved against the newest earlier run of the same input (3.7) |
| Close | Ctrl+Q | closes the window, also while a step computes |

- **Part.** Pick from the list, Browse (STL and STEP), or **Random**. Random draws from
  `reco_gui.setup.discover_test_parts`: every STEP below `$SHAPE_RECO_TEST_PARTS`; of a directory with a
  `manifest.csv` only the rows of status `ok`; every part once before one comes again. A picked part is shown at
  once as a preview with its size and triangle count (`reco_gui/preview.py`).
- **Timeline.** Click a row to look at an earlier step while the run goes on.
- **Heartbeat** (since 2026-09-27). While a step computes, the status bar is refreshed every 2 s, e.g.
  `Computing ...   [41 s, now 3.2 cores, in residual/fit (12 s), running freecadcmd 8 s]`. The cores are given as
  `now 2.1 cores + children 1.0`. "now 0.0 cores" without a child process over several refreshes is a step that
  hangs.
- **Close.** The worker gets 5 s to return; then the child processes (`freecadcmd`) are ended; if the worker still
  does not return within 3 s, the process itself is ended.
- The GUI observes and paces. It does not change a decision.

### 3.4 Cascade mode and the feature phase

```bash
python3 -m reco_gui --method cascade --run --pace 0.6 --verbose --stl part.step   # the feature phase is on by default
python3 -m reco_gui --method cascade --no-features --stl part.stl                 # the rough loop only
python3 -m reco_gui --method cascade --target-iou 0.95 --stl part.step            # fewer, larger features
```

Settings of the cascade mode (`reco_gui.setup.RunSettings`):

| setting | default | meaning |
|---|---|---|
| voxels per side / max voxels per side | 96 / 192 | the cube pitch and its refinement limit |
| grid | finest: wall thickness or cube (`tau`) | `cube` gives the behaviour before 2026-09-24 |
| starting shape | `legacy` | sub-loops first |
| field backend | `igl` | exact closest point |
| type source | `registry` | or `descriptors`, `cnn` |
| FreeCAD rebuild at the end of the run (slow) | off | the toolbar button does it on demand |
| save every finished run | on | section 3.7 |

Group **Feature phase** (behind *More settings*). 0 means the loop's own limit. The GUI passes a setting only when
`ResidualConfig` has a field for it (`reco_gui/cascade_session.py`, `FEATURE_FIELDS`); what it cannot pass is said in
the Output panel.

| setting | loop default (`ResidualConfig`) |
|---|---|
| feature phase after the rough model | on in the GUI; off in the library |
| max pieces per iteration | 8 |
| max pieces in all | 40 |
| max iterations | 3 |
| time budget | 900 s |
| min score | 0.85 |
| detail: target IoU | all details |

**Level of detail** (the slider "detail: target IoU"; `--target-iou`; `ResidualConfig.feature_target_iou`). The slider
runs from 0.800 to 0.995 in steps of 0.005; its right end is "all details". The feature phase takes the congruence
groups with the most volume first and stops taking when the IoU estimated from the volumes,
(V_in − missing) / (V_in + invented), reaches the target (`residual_loop.select_for_target`). The pieces left over
are reported as detail. The estimate counts a piece as explained in full by its tool. The effect of the slider on a
real part is not measured.

Stages of the phase in the *Break at* menu and the timeline (`cascade_session.FEATURE_STAGES`):

| stage | content |
|---|---|
| `feature_residual` | the difference of an iteration: volumes, exact IoU, the pieces, the limits |
| `feature_piece` | the piece in work |
| `feature_split` | a piece no single primitive explains is split on its voxel field (method necks / surfaces / axis) |
| `feature_candidate` | a guess with its source and score |
| `feature_accepted` / `feature_rejected` | with the numbers of the FreeCAD build, or the reason |
| `feature_done` | the summary and which limit ended the phase |

The pace is not waited after `candidate` and `feature_candidate` steps. The Output panel sums the candidates of a
component or piece up in one line; its checkbox **log every candidate** (on with `--verbose`) gives a line per
candidate.

The **Cascade** tab draws every piece of the iteration over the input and the model. Layers (*Debug layers* menu):

| layer | what | colour |
|---|---|---|
| Pieces: pending | pieces not looked at yet (bounding boxes) | grey |
| Piece in work | the piece being guessed, its real mesh | yellow |
| Pieces: guessed | a guess waits for its FreeCAD build | light blue |
| Pieces: accepted | the guess was applied and passed the build | green |
| Pieces: rejected | the guesses were rejected | orange |
| Pieces: skipped | left without a verdict (a limit ended the phase) | dark grey |
| Segments | the segments of a split piece; the opacity is the state | a 10-colour palette |
| Segments: state | the segments' boxes as outlines | the state colours above |
| Feature guess | the guessed primitive, wireframe | orange (add), pink (cut), dark red (rejected) |

Other panels:

| panel | content |
|---|---|
| detail panel | sections *Feature phase*, *Piece in work*, *Pieces of this iteration*, *Feature iterations*, *Feature phase: summary* |
| Operations dock | the program as a tree; an applied feature is a leaf labelled with its source and piece ("feature (fit), piece 7"); the History has one group per piece |
| Feature (live) dock | the piece in work alone, FreeCAD's face metadata of it, every guess with precision, recall and verdict (`reco_gui/feature_panel.py`) |
| Stats dock | cores, memory and steps on one time axis (`reco_gui/stats_panel.py`) |
| Output panel | every step's message and detail lines; at the end the timing table |

The camera is set once per run. Iso / Front / Top / Side fit the part again.

**Another path when the IoU is too low (since 2026-10-10; the user's rule: the first run takes the framework bare, then
vary the path, weighted).** A finished cascade run whose IoU (the FreeCAD grade's exact IoU when graded, else the loop's
voxel IoU) is below the setting "ask for another path below IoU" (`retry_iou`, 0.95; 0 = never) opens a question: try
the next path variant? The variants are `reco_engine/path_policy.py`'s: bare, sub-parts, cubes (the voxel-cube cover),
auto, descriptors, cnn, tree (the decision tree on the auto start, 2026-10-15), analytic (`models/registry_analytic.json`, the AI-free type stage) - ranked by their
measured RESCUE rate: the share of the parts the bare run failed on (no single valid solid, or IoU < 0.95) that the
variant got right, on the same parts; where no paired runs exist, by the share of good runs; a variant with fewer than
5 measured paired failures (or runs) ranks last and is marked so. `docs/data/path_weights.json` holds the numbers, written
by `python3 scripts/path_weights.py` from the recorded runs. "Try <variant>" sets the path on the settings panel and
starts; "Choose..." lists the others; at most `retry_max` (3) further paths per part; the log keeps "paths tried on
this part: bare 0.712, cubes 0.98". Since 2026-10-15 the question is a BAR above the viewport (`reco_gui/decision_bar.py`), not a modal box: the viewport,
the docks and the toolbar stay usable while it is open. Its buttons: Try <best>, Choose..., Keep this result, and
"Export for comparison" (`reco_gui/nephogram.py`, in a thread): the guess and the input as `input.stl` / `model.stl`
and as `input_to_model.ply` / `model_to_input.ply`, every vertex coloured by its exact distance to the other surface
(HSV hue 240 deg blue at 0 to 0 deg red at the scale, default the 95th percentile of the larger way) and carrying the
scalar field `distance`, with `stats.json` (mean, median, p95, max each way, the two-way chamfer) and a README, under
`~/.cache/shape_reco/compare/<part>_<time>/`. The export also writes
`nephogram.FCMacro`, and the setting "comparison opens in" (default FreeCAD, the user's preference) opens the tool at
once: FreeCAD runs the macro (`freecad nephogram.FCMacro`: the two PLYs as Mesh objects with the view property
Coloring on, so the per-vertex colours show; the plain STLs beside them, hidden), CloudCompare loads the PLYs with
the scalar field. The bar's buttons "Open in FreeCAD" / "Open in CloudCompare" / "Open folder" do the same by hand. Since 2026-10-15 (the
user's rule: between STEP files, not meshes) the distances are EXACT where the input's file and the program's STEP
exist: every vertex of the input's tessellation to the model's B-rep and every vertex of the model's to the input's,
by OCCT in freecadcmd (`reco_field/exact_check.point_distances`: distToShape per vertex, about 3 ms each, the material
flag by isInside), the model built as STEP from the program first; at most 50 000 exact vertices per direction (the
rest from the mesh field, the README says so); an STL input is the solid of its facets. The toolbar's
"Compare" runs the same export at any time, and in every case the 3D view gets the layer "Distance": the input coloured
per triangle by its distance to the model (the "inside the framework" option of the setting shows only that).
The voxel fallback of the loop (since 2026-10-15, the user's rule) takes the largest filled boxes first
(`largest_boxes`, bounded cost: blocks of at most 48 per side, then grown at full resolution) so that a leftover
becomes few elements; the feature phase's piece fallback did so already. Headless runs never ask.
Weights of 2026-10-10: sub-parts rescued 0.144 of the
167 known-ABC parts the bare run failed on (and lost 0.108 of the 139 it got right); the analytic type stage rescued
0.100 of 20 A/B parts (lost 0.033 of 60); the cube cover was good on 0.67 of 12 example parts against bare's 0.92 on
the same parts (one paired failure, rescued); auto, descriptors and cnn were not measured then (the gate of 2026-10-10 measured them).

**The memory watcher (since 2026-10-15; the user's rule: the framework must not be killed by the machine running
out of memory).** `reco_engine/memory_guard.py` reads the available memory (psutil, else /proc/meminfo) and, below
`memory_min_free_mb` (`ResidualConfig`; the GUI setting "pause below free memory", 1200 MB; `$SHAPE_RECO_MIN_FREE_MB`;
0 = off), first frees what it can (the window's mesh caches, the garbage collector) and then acts: the loop stops a
run with the model it has between rounds (the loop, the tree) and between the feature phase's pieces, the record's
`stopped` text says "memory low: N MB available"; the window pauses the run at its next step and says so in the
status bar - Continue goes on once memory is back. A reading that fails never stops anything.

**The decision tree (since 2026-10-15; the user's rule: a decision tree on the difference between the rough guess and
the input, with children, one repetitive algorithm, on the auto start).** `ResidualConfig.mode = "tree"`
(`reco_engine/residual_tree.py`; GUI row "rounds", `scripts/run_part.py --mode tree`, `time_cascade.py --mode tree`,
path variant `tree` = `--start auto --mode tree`). After the starting shape every round is one explicit decision on
the loop grid's residual, a step of kind `decision`: D0 the residual is within tolerance (accept); D1 an excess component
above the floor (cuts first, D-003): carve it; D2 a missing component: add it; D3 only pieces below the rough floor
remain: the feature phase; D6 the budget is used or nothing made progress: stop. A component is solved by the same
procedure at every depth: first the loop's own candidates (the cascade, the void fit, the largest boxes, strict
surfaces, booleans, tools), judged as before; when none passes, the component becomes a CHILD - the largest boxes inside
it as ONE composite operation (inside the excess or the missing material by construction, so the material checks pass;
one operation against the budget however many boxes); when no box fits, the cover child: the component's bounding box
as the operation and the material it takes wrongly re-added by a grandchild at depth + 1 (the alternating sum of
volumes; `tree_max_depth` 2, accepted only when the net gain passes the floor and at most `tree_cover_leftover` 0.1 of
the wrong material stays). A cut that leaves the model in more connected pieces than the input has is refused (D5,
`residual_loop._severs`; on grids up to `connectivity_max_cells` 4e6 cells for every candidate, on the tree's chosen
candidate always). **D5 exact (since 2026-10-16):** the grid's labelling cannot see a cut that leaves pieces touching
the kept model along the cutter's faces - in the gate corpus of 2026-10-15, 20 of 60 tree models ended in 2-6 solids
and in every one measured the split came from a composite tool cut (`box - cone`, `box - sphere`) the grid had passed;
so the chosen operation - a cut that severs, an add that floats or only touches (the box children of a wrong
cover: 54 solids on one gate part when only the cuts were gated) - (the tree: every passing candidate in gain order,
the box child, the cover child; the loop: its chosen candidate) is also built in freecadcmd with the program so far (`residual_loop._exact_severs`,
`cfg.feature_build` injectable) and refused when the solid count grows beyond the input's pieces, when OCCT calls the solid invalid or when its
volume is not positive (an inverted shell: a tree_l model of 2026-10-16 had volume -178634 at voxel IoU 0.793 and exact
IoU 0.002; 2 of 60 tree_l and 4 of 60 tree_c final builds were invalid, bare 0); the next passing candidate by gain
is taken. `exact_solids_max_checks` 3 builds per component, none once the program has
`exact_solids_max_ops` 40 operations, `exact_solids_timeout` 60 s; a build that fails or times out leaves the grid's
answer standing; `exact_solids_gate` False switches it off (it is off by itself only without freecadcmd and without an injected build; the final-grade flag `freecad` is not its condition). The counts go to the done
step's `stats.exact_gate` and its message (checks, refused, without answer, seconds; `run_part.py`: `run.exact_gate`). `scripts/solids_per_op.py <run dir>` builds every
prefix of a run's `program.json` and names the operation that first splits the model (measured 2026-10-16 on the
gate's `aegir_n3_S_000341_L`: op 2 of 12, `cut tool (box - cone)`, 1 -> 3 solids, 0.6 s per prefix build).
`tree_work_budget` 60 components per run. The children's operations carry the source mode
`tree:carve` / `tree:add` / `tree:cover` (`run_part.py` counts them as `by_source.tree`). Measured: see the paired
numbers below this paragraph as they come in (the first part, cy_t at 64^3 without the feature phase, 2026-10-15).

**The mesh residual and the repair pass (since 2026-10-15).** When OCCT's exact difference disagrees or gives no
answer (7.1 % of the ABC batch, 11 of 60 gate-bare runs), the feature phase no longer ends: the meshes' exact difference
by manifold3d (`reco_field/exact_check.mesh_residual`, `ResidualConfig.feature_mesh_fallback`) gives r+ and r- and the
pieces as STL files (no STEP, no typed faces: the fit and cascade sources work on them, hole stacks and step faces skip
them); the record's method reads `mesh` and keeps the OCCT error as `fallback_from` (`run_part.py`: `occt_error`,
`--no-mesh-residual` switches it off). After the loop's rounds the repair pass (`reco_engine/repair.py`, `repair` None =
in mode "tree" only) leaves every accepted operation out once on the final model: an operation whose absence changes
nothing within `repair_tolerance` 1e-4 of the input's voxels is taken back, and a model in more connected pieces than
the input loses the one operation whose absence restores the count at the least IoU cost; the operations stay in the
program with `accepted = False` and a "repair:" label, each a step of kind `repair`.

**The path atlas (since 2026-10-10; the user's idea: store the trees, compare them, a two-dimensional multi-domain
view, the aegir trees as the benchmark).** `python3 scripts/path_atlas.py build <domain>=<runs glob>[@variant/part]
... --truths <dir> --out docs/data/path_atlas.jsonl` stores every run's tree (the program's operations with their
fingerprints), IoU, tier and separation class as one row per (domain, part, path variant); `compare` lists, per
part, which paths gave identical trees (by fingerprint), how many operations every path has in common and the best
path (tier, then IoU); `map --out fig.png` draws parts x paths per domain, a cell per run coloured by the tier against
the known tree (else the IoU), "x" where the model is not one valid solid. The gate corpus for this is
`~/shape_reco_results/validation/gate_paths` (60 shapes of the calibration corpus, 23 v5 primitives and 37 aegir
trees), run once per path variant with `gate_paths/gate_launch.sh <variant> <jobs> -` on the servers
(`runs_<variant>/`); `scripts/path_weights.py` reads those runs as paired rescue rates. `bash server/gate_pull.sh [host]`
pulls the per-part JSON of every variant from the NFS dir into the laptop corpus and rebuilds the weights, the gate atlas
(`docs/data/path_atlas_gate.jsonl`, `_compare.md`) and the map `docs/reports/path_atlas/path_atlas_gate.png` (4 s); it is
safe to run while the servers still work (partial rows, unrated parts count as running).

**Gate corpus readout of 2026-10-16 (good = one valid solid and exact IoU >= 0.95; 60 shapes; `gate_agg.py` on the
NFS dir, `path_weights.py` for the paired rates).** The tree after the legacy start WITH the exact solid gate
(`runs_tree_lg`, server_a + server_b, checkout 26d0d758): good 30 (tree_l without the gate 29, bare 28, tree_c 27),
one solid 53 (tree_l 46, bare 57; one VALID solid 51 - two of the 53 are invalid, §4.9), IoU >= 0.95 30, median exact IoU 0.958 (bare 0.873), mean 0.752; rescued 4 of
bare's 32 failures, broke 2 of its 28 good ones. The gate's cost: 40 of 60 parts needed a build, 279 checks, 63
refused, 0 failed builds, 273 s in all, median 5.7 s per part, 57.5 s at most. Three runs crashed on a bug of the
box-child branch (the gate's verdict stored in the explanation string, fixed 8f3724e1) and were rerun on the fixed
checkout: n5_I_082 0.538 in 2 solids (bare 0.839 in 5), n5_I_124 0.419 (bare 0.523), v5_cone_sector_2175 0.782 (bare
0.654). The breaks the gate did not cure have one measured cause (`path.jsonl` of n4_S_230, bare 0.999 -> 0.091 in 2
solids, and n3_U_399, tree_l 0.942 -> 0.323 in 2): D1c replaces a cover whose excess is large (213 %, 352 % of the
input) by a contained box explaining 9-10 % even when the carve branch had a passing cut child (n4_S_230: "cut child
8 boxes, gain 77897, ok" was found first; bare kept the bounding cylinder sector and reached 0.999 by cuts); after
the swap the part must be rebuilt by adds, and the gate refuses every add child or box as floating ("D5 exact: 5
solids, 2 allowed") - tree_l had accepted those floating adds and joined them later. The D1c rule is open (gate the
swap on the carve branch's result, or run both as alternatives and keep the better). The GUI never ran the gate
before 39e1b5bc: without `cfg.out_dir` (only the GUI) its temp-dir fallback hit a missing import at the first cut;
the gate test covers that case now. Host rule: the corpus is comparable only on server_a + server_b - a bare-with-gate
run on server_c + server_d (`runs_bare_g_mm`, 4 jobs x 2 threads + freecadcmd on 16 cores, load 17) was 3-4x slower in
every stage including the gate-free comparison (median reconstruction 1138 s vs 324, sampled comparison 265 s vs 62),
24 feature phases stopped by the time budget (bare 3), 4 reconstruction timeouts; its numbers (good 28 = bare,
median 0.865) measure the hosts, not the gate (the gate there: 39 parts, 145 checks, 7 refused, 512 s).
Bare WITH the gate rerun on server_a + server_b (`runs_bare_g`, 2026-10-16 18:25-19:40): good 28 = bare, one solid 58
(bare 57, the repeat bare_b 56), IoU >= 0.95 28, median 0.878 (bare 0.873, bare_b 0.910), 6 feature phases stopped by
the time budget (bare 3, bare_b 4); paired against bare: rescued 1 of 32, broke 1 of 28 - the same as the repeat
bare_b, i.e. within the run-to-run noise of the loop (bare is not reproducible: n4_S_230 bare 0.999 with strict
cylinders, bare_b 0.988 with four fallback boxes). The gate in bare: 43 of 60 parts needed a build, 164 checks, 10
refused on 4 parts, 0 failed builds, 129 s in all, median 1.5 s per part, 23.3 s at most. Where it refused: n2_I_457
(bare 0.000 in 2 solids, bare_b 0.887 in 2, bare_g 0.882 in ONE), n8_U_040 (bare 2 solids, bare_g 0.084 in 2),
n2_I_025 (bare_g 0.758 in 1), n5_I_082 (bare 0.839 in 5 solids, bare_g 0.588 in 2). So the exact gate costs the loop
nothing it can measure and gives the tree 7 more one-solid models (46 -> 53) and one more good one (29 -> 30); the
one-solid gap to bare (53 vs 57) is the D1c swap above, not the gate.

**Mode "search" - the dumb shape search (the user's rule 2026-10-16; `reco_engine/shape_search.py`; `--mode search`
in `run_part.py`, `time_cascade.py`, `python3 -m reco_gui`, the GUI's mode row).** No network, no RANSAC: per residual
component the input's exact surface samples near it (their voxel in the component dilated by one), then for EVERY base
shape guesses from those points and from the component's own voxel centres (centroid, principal and world axes,
extents, radial distances), each hill-climbed by random perturbation of centre, axis and dims (`search_evals` 120
neighbours per refined guess, the two best guesses of a kind). The grade is the coverage: the share of the nearby
samples on the shape's surface (|sdf| <= `search_tolerance_share` 0.004 x the part's extent); the search's score is
the coverage minus the share of the whole part's surface samples the shape swallows (strictly inside it: material a
cutter removes, air an add fills - without that term every cut of the example parts failed the void check at purity
0.81-0.96). Every kind's best is judged by the loop's own checks (`_evaluate`: gain, void purity, sign, severing) and
the covered passing one wins, gain among equal coverages; a passing kind under `search_min_coverage` 0.3 is refused
(without the floor the step part took two torus cuts at 8 % and 16 % coverage to IoU 0.887 - junk that gains
voxels). The start stays the legacy one. Measured 2026-10-16 (48^3, no feature phase, no fallback, laptop):
`bs_box_two_pockets` IoU 0.983 in 4 operations (torus, torus, cone cuts) in 20 s; `to_ring_groove` 0.981 in 2 (one
torus cut) in 16 s; `bs_box_step` 0.686 with no cut taken in 9 s - the legacy start's oriented box (62 % of its
surface on the input) leaves a wedge-shaped excess whose best box covers 45 % of the nearby surface but swallows
13.6 % of the part's (purity 0.80); the loop mode's cascade cuts the step from the voxels. Tests:
`tests/test_shape_search.py` (7, the pockets run among them, 21 s).

### 3.5 FreeCAD groups and FreeCAD rebuild (since 2026-09-29)

A cascade run ends without building its program in FreeCAD. Two toolbar buttons do it on demand, in the background
(`reco_gui/rebuild.py`).

| button | what it does | measured on the liner, 41 operations, 2026-09-29 |
|---|---|---|
| FreeCAD groups | every operation's tool as an object of `<part>_groups.FCStd`, grouped into adds and cuts, no final boolean (`reco_field.export.build_groups`). The button uses tools "primitives": parametric Part primitives with their placements, no boolean at all | 0.8 s, 361 primitives. Tools "solid" (each tool with its own booleans): 12.6 s |
| FreeCAD rebuild | the program folded into one solid (all booleans) and its chamfer distance; without the exact grade | 23.5 s for the build. The exact grade gave no answer in 300 s and failed after 1268 s |

The groups document is written to `~/.cache/shape_reco/groups/<part>_<time>/`. Its path is in the status line and the
log. Open it in FreeCAD yourself.

### 3.6 Vector-field mode

**Vector-field mode.** Settings group **Vector-field loop**: max operations (24), stop at explained (0.985), surface
and field points (4000 / 2500), vectors drawn (300), the STEP deflection (0.02), chamfer at the end (a few seconds;
about 15–90 s on a part several metres across).

| stage | content |
|---|---|
| `field` | samples drawn |
| `base` | every kind fitted to the whole part; the best is the base |
| `cluster` | the unexplained surface grouped; one cluster chosen |
| `candidate` | a fit of that cluster folded into a trial model |
| `accepted` / `rejected` | with the reason: too little gain, or invented geometry |
| `grade`, `done` | the two-way chamfer, the end |

*Break at* defaults to base, accepted, rejected. The **Vector field** tab draws in mm: input mesh (grey), field lines
(teal), model (blue), unexplained input surface (red points), cluster (yellow), candidate (orange for an addition,
pink for a cutter, dark red once rejected). Code: `reco_gui/field_session.py`, `reco_gui/viewer_field.py`.

The voxel mode (route C) was removed on 2026-09-30; see section 7.

#### Members by congruency, and the classic solution before the fallback (2026-09-29)

Both rules are on by default (`ResidualConfig.member_congruency`, `solution_alternatives="auto"`); nothing to pass.
The step stream shows them: a member's step names its group (`group M1, 2 of 4 congruent members`), an operation
taken from a congruent member says `by congruency: member 1's, group M1`, and when the classic solution was tried its
steps start with `[classic]` and one `solution` step states both measures and which was kept. From Python:

```python
from reco_engine.residual_loop import ResidualConfig, residual_steps
cfg = ResidualConfig(member_min_length_ratio=1.5, solution_alternatives="always")   # "auto" | "always" | "never"
for s in residual_steps("dataset_gen/examples/frame_review/r14_mitred_rect_union.step", cfg):
    if s.kind == "solution" or s.info.get("by_congruency"):
        print(s.kind, s.info["message"])
```

Expected on r14: 6 of the 12 operations by congruency; `solution: members kept - the classic solution's exact IoU
0.0472 is not above its 0.8880 by more than 0.01 ...`. Tests: `python3 -m pytest tests/test_member_alternatives.py -q`
(20 tests, about a minute, no FreeCAD).

**The test suite on the institute server** (since 2026-09-29; the user's rule: "lets run every test on the server").
The whole suite runs on server_a (32 cores, 377 GB, shared with others), one pytest process per test file, started by
**GNU parallel** (O. Tange, GNU Parallel, 2011-, https://www.gnu.org/software/parallel/): a file starts only while the
1-minute load is below `MAX_LOAD` (default 75 % of the cores) and `MIN_FREE` (16G) of memory are free, at most `JOBS`
(8) at once, niced, each with a timeout. The run is detached; a rerun into the same directory repeats only what failed
or did not run. The server has no batch system (no HTCondor, no Slurm; `at` / `batch` only, whose load limit of 1.5
is the daemon's and would never start on a machine in use).

```
scripts/server_tests.sh sync      # this checkout, the checkpoints and models -> /data/user/shape_reco/tests_checkout
scripts/server_tests.sh setup     # once: venv_tests (torch cu128, requirements, PyQt5, vtk), GNU parallel, freecadcmd wrapper
scripts/server_tests.sh start     # every tests/test_*.py; or: start tests/test_hole_stack.py tests/test_ibeam.py
scripts/server_tests.sh status    # load, files through, failed so far, what runs
scripts/server_tests.sh fetch     # table.md, summary.json, the failed files' logs -> ~/shape_reco_results/server_tests/<run>/
scripts/server_tests.sh stop
```

On the server the slow tests run too: `--run-slow` belongs to the laptop's pytest-astropy plugin, which the server's
venv does not have, so nothing is skipped as slow. FreeCAD there is the extracted 1.0 AppImage behind
`/data/user/shape_reco/bin/freecadcmd`; the runner names it in `SHAPE_RECO_FREECADCMD` (and keeps it first on the
PATH). server_a has no `/usr/bin/freecadcmd`; server_b's is the system's FreeCAD 0.20.2, which the variable keeps out.
CPU only by default (`GPU=1 scripts/server_tests.sh start` lets the tests use CUDA). Pinned to some cores
(`taskset`), give `MAX_LOAD` as a number: GNU parallel takes a percentage of the cores it may use. The NFS home
(`/nfs/home/user`) is over its disk quota (2026-10-01): a test must not write below `~` (give the loop an
`out_dir`; without one the feature phase writes `~/.cache/shape_reco/features`).
The runner's `TMPDIR` is a short local directory (`/tmp/srt.*`, removed at the end) and pytest's directories lie
below `<out>/tmp/<file>` (`--basetemp`): under the long NFS path the AF_UNIX sockets of multiprocessing exceed
the 108-byte limit, and a DataLoader worker's batch is then dropped (`AF_UNIX path too long`) while the test waits
for ever (2026-10-01). Process pools that run torch go through `spawn`, not `fork` (`exact_field_input.precompute`):
a forked child of a process that has used torch's OpenMP hangs in its first parallel op.

**The ABC test set on the institute server** (since 2026-09-29). Chunk 0000 of the ABC dataset (10 000 STEP files,
https://deep-geometry.github.io/abc-dataset/) is downloaded ON the server from its source, its md5 checked against
the dataset's list, unpacked, and every file listed in a manifest (`server/server_a/fetch_abc.sh`,
`scripts/abc_manifest.py`: solids, faces, free-form faces, box, volume, status). Every part with status `ok` (one valid
solid) runs through `scripts/run_part.py`: the input's facts, the reconstruction (the GUI's cascade defaults, feature
phase on), the FreeCAD build, the exact difference, the sampled comparison (`scripts/compare_pair.py`, the part scaled
to a box diagonal of 1000), and a record per part. GNU parallel starts the parts as it starts the tests.

Since 2026-10-15 `run_part.py` records a TERM from outside (the batch's timeout) as `state: "killed"` with the stage it
was in (before, such a run stayed `"running"` and could not be told from one still going; the gate of 2026-10-10 lost
25 of its 36 timeouts in the sampled comparison at 2e6 points, not in the loop), `--reconstruction-timeout S` ends the
loop with the model it has (`run.timed_out`, the partial model is still built and measured), `--compare-timeout S`
(900) makes a slow sampled comparison an error instead of a kill, `--mode loop|tree` selects the rounds. The batch
runner `server/server_a/run_parts.sh` passes `RECO_TIMEOUT` 1800, `COMPARE_TIMEOUT` 600, `POINTS` 500000 (2e6 on
request) and no memory guard (`--memfree` killed the youngest job silently; set `MIN_FREE` to bring it back).
`scripts/parts_table.py` lists killed runs apart from running ones; `scripts/path_weights.py` counts neither as
evidence.

```
scripts/server_parts.sh sync                 # this checkout, checkpoints, models
scripts/server_parts.sh fetch-abc            # download, md5, unpack, manifest (detached; 12 min to the unpacked files)
scripts/server_parts.sh lists                # pilot.txt (20, by number of faces), known.txt (the subset of 2026-09-28), rest.txt
scripts/server_parts.sh start <run> pilot    # then: known, rest - into the same run directory
scripts/server_parts.sh status <run>
scripts/server_parts.sh fetch <run>          # the records and table.md -> ~/shape_reco_results/abc/<run>/
python3 scripts/parts_table.py <run directory> [--line 0.95]
```

Options of a run go through `EXTRA`, e.g. `EXTRA=--subparts scripts/server_parts.sh start <run> pilot` (the rough model
as sub-parts), `EXTRA=--feature-subparts` (pieces without a tool as sub-parts before the voxels), `--no-features`.
One run at a time: a part's job uses 3 to 4 cores and FreeCAD's booleans more, two runs of 6 jobs gave a load of 77 on
the 32 cores (2026-09-29), which inflates every time in the records.

The table has three readings. **Robustness**: did the run end by itself with ONE valid solid. **Quality**: the
distribution of the sampled IoU, for the parts in scope (planes, cylinders, cones, spheres, tori only) and out of scope
(free-form faces) apart; no pass mark. **Triage**, in this order: 1. what crashed, timed out or gave no valid solid,
grouped by its text; 2. parts whose exact and sampled IoU disagree (a boolean failure, not a reconstruction
failure); 3. out of scope: listed, not investigated; 4. in scope and below the line. Failures are investigated by
class, the largest class first, on one representative; a fix is measured on all parts of its class.

**The exact difference by general fuse** (since 2026-09-29; `reco_field/exact_check.py`, `partition`): ONE general
fuse of input and model (what FreeCAD's boolean XOR is built on) gives the pieces of both differences. It is tried
first and accepted only when both volume balances agree within 1e-5 of the input's volume and with the sample; else the
separate booleans follow as before (`exact_residual(..., general_fuse=False)` skips it). Measured 2026-09-29 on server_a:

| pair | general fuse | before |
|---|---|---|
| liner, rough model | right, 5 s | right |
| frame, 40 boxes | right without a tolerance (IoU 0.965543, 1 s); with a fuzzy tolerance nothing is fused | 0.9655 |
| liner + 39 features | r- 1 224 974.95 mm3 in 31 s | the cuts: no answer; incremental difference 1 224 974.39 in 172 s |
| liner + 52 features | r- 1 088 794.47 mm3 in 40 s | the cuts: Null shape; incremental 1 088 794.72 in 193 s |
| liner + 60 features | refused: the balances disagree by 3 683 mm3 | not measured |
| review r13, r08 | wrong | wrong |

It is faster, and right on the liner where the cuts fail; it is not more stable in general.

**The exact IoU of a stored run, after the run** (since 2026-10-01; `scripts/exact_of_run.py`). The rough model's
exact difference (general fuse), then the run's feature tools applied to its pieces by the incremental difference, in
one chunk or `--chunk K`; `--global-first` tries the whole model's global difference first. Where the tools sit a
hair off the faces they reproduce, the booleans fail; `--grow MM` grows every tool by MM, and the model cut by the
grown tools gives an upper bound of the run's missing volume r+, hence V_in / V_model >= IoU >= (V_in - r+') /
(V_model + r+'). `--snap MM` moves the tools onto the input's axes instead (another model; recorded).

```bash
python scripts/exact_of_run.py <run directory> --input <part.step> --out <dir> [--grow 1e-3] [--global-first] [--snap 1e-3]
python scripts/count_holes.py <part.step> [--json out.json] [--small 7.0]     # the holes of a STEP from its faces
```

Measured 2026-10-01 on server_b, the liner run of 2026-09-30 08:07 (86 features, budget 3000 s;
`docs/data/liner_exact_2026-10-01.json`): global difference and incremental difference both fail (pattern copies
4e-8 to 4e-5 mm off the input's faces); grown by 1e-3 mm: r+ 631.4 mm3, budget error -17.8 mm3, exact IoU between
0.99995 and 0.99998; by 2e-4 mm: r+ 39.1, budget error -107.7, between 0.999975 and 0.999982; by 1e-4 mm: fails.
Each call about 6 min (the model's build 2 min, the difference 4 min). `count_holes.py`: the liner has 122 holes, the
second target TARGET 74 (`docs/data/target_holes_2026-10-01.json`).

**aegir, the step after the reconstruction** (since 2026-09-29). aegir is the SHiP experiment's simulation framework
(Phlex + Geant4; checkout `~/Documents/Sandbox_aegir/aegir` or `$AEGIR_ROOT`, run through its pixi environment
`bench`). `scripts/aegir_benchmark.py` writes the program as GDML in the frame aegir's gun expects (beam along z,
`PartVolume` of G4_Fe in a `World` of G4_AIR), lets aegir load it and times its straight-line scan (muon neutrinos,
crossing detector, no output: the cost of an event is the navigation through the solid), beside the part's mesh where
aegir has one. Nothing is written into the aegir checkout.

```
python3 scripts/aegir_benchmark.py <run directory | part:latest | operations.json> --part liner|frame|blind_hole \
        [--events 200000] [--rounds 3] [--threads 1] [--mesh liner_mesh] [--upto N] [--dry-run]
```

Result: `~/shape_reco_results/aegir/<stem>_<time>/aegir_benchmark.json` (seconds per round, mean, standard deviation,
events per second, the load at each round's start). Round 0 is a warm-up. The run waits for the shared lock of heavy
jobs. Measured so far: only a smoke run of 2000 events under load (2026-09-29; the liner's 54 operations loaded and
scanned, exit 0) - start-up dominates it, it is no benchmark. `--upto N` takes the first N operations: a level of
detail.

**GDML export** (Geant4, ROOT; since 2026-09-29). The program as a GDML file: every primitive a GDML solid (box, tube,
cone, orb / sphere, torus), every operation a boolean solid, one logical volume in a world box
(`reco_field/gdml_export.py`: `export_gdml`, `export_result`; GUI: toolbar "GDML").

```
python3 scripts/export_gdml.py <run directory | part:latest | part:<n> | operations.json | program.json> --out part.gdml \
        [--tree grouped-balanced|grouped-multi|as-is] [--check] [--points 20000]
python3 scripts/gdml_check_root.py <folder>        # on a machine with ROOT: every point of *.points.txt judged by TGeoManager
```

Tree shape (`--tree`, `export_gdml(..., tree=)`; also `aegir_benchmark.py --tree`; since 2026-10-01): the default
`grouped-balanced` writes every chain of cuts as one cut by the union of its tools and halves every tree of unions;
`grouped-multi` writes a union of primitives as one multiUnion (Geant4 only: ROOT 6.40 does not read multiUnion);
`as-is` is the program's own tree (the export before). The same solid in every shape; the choice is speed: in aegir's
scan of 593 programs (T-28, `docs/reports/real_trees/`) both grouped shapes took a median 0.58 of the own tree's
navigation time. Measured 2026-10-01 (aegir's scan on server_b, liner program of 64 operations, 200 000 lines): the
three shapes give the same 86 918 entry points (largest distance 7.5e-5 mm), none off the program's exact surface by
more than 1 um. The same scan found that a full turn written as 6.283185307 (2 pi rounded to 9 digits) leaves Geant4's
G4Torus and G4Polycone open by a hairline slit at phi = 0: 1815 of 88 490 entries lay off the program's solid by up to
13 mm, inside the liner's holes and fillets. A full turn is written as 2 pi to the last digit since (`FULL_TURN`); ROOT
6.40 reads the files and judges 20 000 points per file as the program does (14 files, server_b's aegir environment).

Measured 2026-09-29: the liner (54 operations: 493 primitives, 492 booleans) and the frame (40 boxes, 39 unions), and 13
test solids, tilted and nested - ROOT 6.38.00 (server_a, LCG_109) judged 0 of 20 000 points per file differently from
the program. Geant4 itself was not run (its GDML reader's source was read for the rotation convention; the check
program did not link on server_a). A run stored before 2026-10-01 may have its composite tools cut down to text by the
run store (below 12 levels until 2026-09-29, below 400 until 2026-10-01); such a run is exported from its
`program_recovered.json` when there is one (section 3.7), otherwise refused.


### 3.7 Run store: Save, Compare, Open (since 2026-09-27; Open since 2026-09-29)

Runs are stored outside the repository, in `$SHAPE_RECO_RESULTS` or `~/shape_reco_results/`:
`<store>/<part stem>/<YYYYmmdd-HHMMSS>_<short commit>[_<label>]/`.

| action | effect |
|---|---|
| **Save** | stores the run as it is; asks for an optional label; an unfinished run is stored with `"finished": false` |
| **save every finished run** (settings, on by default) | stores every run at "Run finished". `python3 -m reco_gui` allows it (`MainWindow(autosave=True)`) |
| **Compare** | the run just saved against the newest earlier run of the same input file (by sha256), in the detail panel |
| **Open** | a table of every stored run, newest first, with a filter. Shows the run's model in the 3D views, its summary (metrics, program, settings) in the detail panel, its steps in the log (`reco_gui/open_run.py`). A stored run is looked at, not continued. Open switches "follow live" off (a later step would replace the summary); Start switches it on again (since 2026-10-16: before, a run started after Open filled the timeline while the 3D view and the detail panel stayed) |

```bash
python3 scripts/compare_runs.py list [part]                          # the stored runs, newest first
python3 scripts/compare_runs.py diff Frame_4_x_6:previous Frame_4_x_6:latest     # or two run directories, or part:<n>
python3 scripts/compare_runs.py diff <run a> <run b> --json          # the raw dict; --all: every metric; --tolerance T
python3 scripts/compare_runs.py --store <dir> list                   # another store
```

| file of a run | content |
|---|---|
| `run.json` | `schema`, `part`, `label`, `timestamp`, `host`, `input` (path, size, sha256), `code` (commit, branch, `dirty`), `method`, `finished`, `steps`, `settings`, `program`, `metrics`, `timings`, `model`, `view`. A metric that was not computed is `null` |
| `steps.jsonl` | one line per step |
| `model.step`, `model.stl` | copies of the built model, when there is one |
| `view.png` | the 3D view |

A run directory is never overwritten and is written atomically. From Python:
`reco_engine.run_store.save_run(session_or_result, settings, input_path, label=None)`.

The program is written whole at any depth (since 2026-10-01). Runs stored earlier may have the deeper operands of
their composite tools cut down to text (`run_store.truncated(program)` lists them; the GUI's Open says so). Such a
program is recovered from the feature phase's own record of the run (`~/.cache/shape_reco/features/<part>_<start>/
features/features_job.json`), checked node by node against what run.json kept, and written beside run.json, which
stays untouched; `load_run` (Open, export_gdml, aegir_benchmark, real_tree_study) then uses it:

```bash
python3 scripts/recover_run_program.py <run directory> [...]           # check only
python3 scripts/recover_run_program.py <run directory> [...] --write   # writes <run>/program_recovered.json
```

**Benchmark tab (since 2026-10-11; the user: the benchmark is run with aegir on the framework's own results after
the run, not on files from the servers, and lets the user choose the most suitable result for his application).**
Toolbar "Benchmark" (or `--benchmark`) raises a tab of the bottom panel (`reco_gui/benchmark_panel.py`, Qt-free core
`reco_engine/benchmark.py`) with the STORED RUNS of the current input (the run store, matched by the file's content; the
GUI saves every finished run), one row per run: when it was saved, its label, its path variant (`path_policy.variant_of`
of the saved settings), operations, one valid solid, exact and voxel IoU, chamfer, missing and invented volume, seconds,
and - once rated - the aegir tier, separation class and failed criteria. The rows are ranked for the chosen preference:
*accuracy* (valid, then the exact IoU, then fewer operations), *design* (the aegir tier, then its IoU against the known
tree), *simplicity* (fewest operations among the valid models within 0.02 of the best IoU), *speed* (the fastest of
those). **Rate with aegir** rates the unrated runs in the background against the part's known tree - `<stem>.json`
beside the input, or `truths/<stem>.json` beside it or one level up (the validation corpora's layout); without one it
says so (a real part has no known tree: its rows carry the measurements against the input only). A rating is cached as
`rating.json` in the run's directory. A click on a row shows the run as Open does (model, summary, log) plus its rating
in the detail panel and its TREES in the Operations dock (`OperationTree.show_benchmark`): "Model" = the program as the
boolean tree, "Known tree" = the truth (Union / Intersection / Subtraction over the placed primitives, tools in red).
**Use this result** (or a double click) labels the selected run "chosen" in the store and shows it; FreeCAD rebuild and
groups then work on it. The tab refreshes after every finished run and opens by itself when the retry policy (above)
has run several paths on the part. The path atlas of the server corpora stays a script-side tool
(`scripts/path_atlas.py`, `scripts/path_atlas_all.sh`, 3.4); the GUI does not read it.

**Alternatives, event time, nephogram + histograms (since 2026-10-29).** The user asked for three things:
- "show iteratively different reconstructions, start with 3 to select by default";
- "benchmark their event time with aegir";
- "select via nephogram / histogram and benchmark like the presentation", where "each nephogram has always
  corresponding histograms".

How the GUI does it:
- **Alternatives.** The setting *reconstructions per part* (`RunSettings.alternatives`, default 3; 0 = off) makes a
  finished cascade run go on by itself. The next run uses the untried path variant with the best measured rescue rate
  (`path_policy.rank`). This repeats until three DIFFERENT programs of the part exist; a path that repeats an earlier
  program does not count. There is no question in between, also not after a good run: the user compares. Then the
  Benchmark tab opens with those runs checked, and the settings panel shows the user's own path again. With 0, the
  decision bar asks below the IoU threshold as before. Since the user's session of 2026-10-29, when bare reached IoU
  1.000 on the ABC plate 00009283 and sub-parts (104.6 s) and analytic (65.7 s) rebuilt its program in full:
  - no alternatives after a run of IoU >= `path_policy.EXACT_IOU` (0.999);
  - an alternative whose path acts before the feature phase only (`path_policy.rough_only`: start shape, mode) and
    whose rough model equals a tried path's is stopped at its first feature step and counted as that path's program;
  - the chain ends when another part is chosen (it had started "cubes" on a part the user had opened meanwhile);
  - an alternative runs through, as flow step 3 does.
- **Nephogram + histograms per run.** Every stored run gets its nephogram in a thread, written to `<run>/nephogram/`
  by the decision bar's export with the exact STEP distances (PLYs, `stats.json`, macro). It always comes with
  `histogram.png`: the share of the vertices per band, both ways (missing / invented). The FreeCAD macro opens the
  histogram image beside the coloured meshes.
- **Benchmark tab.**
  - Each row has a check box; the newest 3 runs of a part are checked by default.
  - New columns: *us/event*, *p95 missing*, *p95 invented*.
  - **Event time (aegir)** scans the checked runs that have no event time yet. **Nephogram** opens the selected run's
    nephogram in the comparison tool; an older run that has none gets one from its program built in FreeCAD.
  - Below the table: the selected run's histograms, and the checked runs' IoU against their event time.
  - New preference *event time*: the cheapest model among the valid ones within 0.02 of the best IoU.
- **Event time** (`reco_engine/event_time.py`, the deck's method):
  - Each run's program is written as GDML (`PartVolume` of G4_Fe in G4_AIR) and scanned by aegir's straight-line
    workflow at 50 000 and 300 000 events.
  - Navigation per event = the difference quotient of the two counts, minus an empty baseline. The baseline is the
    same world and gun, with the part replaced by a 0.1 mm box beside the beam. The result is the median of 3
    measured rounds after a warm-up.
  - It is stored as `<run>/aegir/event_time.json` and waits for the shared heavy lock. It takes about a minute per run
    on the laptop.
  - Placement for any part: the beam runs along the thinnest extent, the part is centred, and the gun is the tightest
    of aegir's fixed cones that covers the cross-section at the front face. Every run of a part gets the input's
    placement, so their event times compare.
  - Measured 2026-10-29 on ABC plate `00009283_..._step_001` (9 primitives, gun "liner"): navigation 0.63 us against a
    baseline of 13.1 us per event (one round of 20k/100k events, 13 s).

```
python3 -c "from reco_engine.event_time import event_time; print(event_time('<run dir>')['navigation_us'])"
python3 -c "from reco_gui.nephogram import stored_nephogram; stored_nephogram('<run dir>')"   # an older run's nephogram
```

### 3.8 Dataset-generation window

```bash
python3 -m reco_gui.dataset_gen_main
```

A form around `src.dataset_generator` (section 10). Settings: mode and topologies, output directory, macro
directory, `freecadcmd`, scale, seed, **shards** (N parallel runs `output_dir/shardI` with `seed+i`; 1 is a single
run), voxelization, the in-scale extras, the noise flags. **Generate** starts one `QProcess` per shard and streams
its output into the log, prefixed `[shardK]`. **Cancel** kills the running ones. The summary panel counts samples
per topology and hollow / carved / cut from every `dataset_metadata.jsonl` under the output directory; it updates
every 2 s while running.

## 4. Main line: the cascade add-cut loop (route B)

`reco_engine/residual_loop.py`, `cascade.py`, `ibeam.py`. Architecture: `CODEMAP.md`, section 3.

```bash
D="$HOME/shape_reco_parts/benchmark_parts/1.0"
python3 scripts/time_cascade.py "$D/Frame_4_x_6.stp" --json out.json                # one part, with the timing table
python3 scripts/eval_add_cut.py "$D/Frame_4_x_6.stp" "$D/Unbenannt-Frame_4_x_6.stl" # several parts; --json <file>
python3 -m reco_engine.reconstruct_cli --mode add_cut --input-stl part.stl --workdir /tmp/reco
python3 -m reco_gui --method cascade --stl "$D/Unbenannt-Frame_4_x_6.stl"
python3 scripts/local_grid_residual.py --program docs/data/frame_cascade_program.json --json docs/data/local_grid_residual.json
```

`eval_add_cut.py` caches in `~/.cache/shape_reco/add_cut` (`--no-cache`, `--cache-dir`); its default resolution is 48.
Exact IoU of a built model: `scripts/freecad_iou.py` (its docstring). Voxel IoU of two closed meshes:
`python3 scripts/voxel_iou.py <model> <input> --pitch 2.0`.

Frame status. The user counts the frame as solved (decision of 2026-09-28, `CLAUDE.md`) on this measurement: 40
adds, exact OCCT IoU 0.9655, genus 17 = the input's, surplus 0, chamfer 0.31 mm; the root fillets are missing
(r = 27 mm, 3.04e7 mm³ in 60 pieces). Earlier: 0.910 on the cube grid with `min_length_ratio` 2.0 (2026-09-24; the
8 transverse plates and the fillets missing), 0.9445 on the grid `tau` (2026-09-24).

### 4.1 Options of `scripts/time_cascade.py`

| option | default | meaning |
|---|---|---|
| `--resolution`, `--max-resolution` | 96, 192 | the GUI's defaults |
| `--top-k` | 2 | branches kept per stage |
| `--field-backend` | `igl` | or `cuda`, `torch`, `numpy`, `''` (the environment's) |
| `--grid {cube,tau}` | `cube` | the loop's grid |
| `--start` | `legacy` | the starting shape |
| `--order {default,cuts-first,adds-first}` | `default` | `ResidualConfig.cuts_first`; default: cuts first after a starting shape |
| `--excess-boxes N` | 3 (the loop's) | largest filled boxes per cut piece; 0 = off |
| `--no-vary-start`, `--variants N` | — | no coordinate search on the start; N variants |
| `--max-ops N` | the loop's budget | stop after N operations |
| `--no-tools`, `--no-vary`, `--no-more-booleans`, `--no-surface-choice` | — | switch a candidate source off |
| `--features` | off | the feature phase (4.3) |
| `--gui-fold` | off | also fold the steps as the GUI does (its meshing) |
| `--json FILE` | none | the summary, with `peak_rss_gb` |

### 4.2 Grid, first fit, booleans

**The grid (since 2026-09-24, B-090).** Grid `tau` is a box of cells over the part at the finer of the thinnest
walls (5th-percentile wall thickness) and the cube's pitch.

| frame, 2026-09-24 | cube 96³ | grid `tau` |
|---|---|---|
| cells | 96³ at pitch 0.81 | 88 x 562 x 419 at pitch 0.12 (STL units) |
| exact IoU | 0.9101 | 0.9445 |
| chamfer (STL units) | 0.033 | 0.0052 |
| time, peak memory | 27 s | 98 s, 5.1 GB |

Source: `docs/FRAME_FAILURE.md`, "The cascade loop on the wall-thickness grid". The frame on `tau` needs about 5 GB
free: check `free -g` first. The wall thickness is measured once per mesh (`reco_engine.cache.wall_thickness_cached`,
4000 rays, libigl; 0.1 s on `bs_hollow_sphere` where trimesh took 33.8 s) and kept in
`~/.cache/shape_reco/wall_thickness/`. `bs_hollow_sphere`, whole run: 51.6 s before, 13.9 s cold, 12.9 s cached.

**Starting shape (since 2026-09-26; `reco_engine/start_shape.py`).** `--start` of `time_cascade.py` and of
`reco_gui`, "starting shape" in the GUI's settings, `ResidualConfig.start_shape`. Stage and grading rule:
`CODEMAP.md`, section 5.1.

| value | effect |
|---|---|
| `legacy` (default) | member / hollow sub-loops first, then the graded first add; adds before cuts |
| `auto` | the graded starting shape first; cuts before adds |
| `dominating`, `dominating major`, `cascade`, `oriented box`, `world box`, `cylinder`, `cylinder sector`, `sphere`, `box & cylinder` | that candidate, forced; cuts before adds |

```bash
python3 scripts/time_cascade.py part.step --start auto --json out.json
python3 scripts/time_cascade.py part.step --start auto --order adds-first      # the other order, to measure both (D-003)
python3 scripts/time_cascade.py part.step --start auto --no-vary-start         # no coordinate search on the chosen shape
python3 scripts/time_cascade.py part.step --variants 3 --json out.json         # 3 solutions from the 3 best starts
python3 scripts/start_matrix_cell.py <part> <start type> [--out DIR]           # one cell of the comparison matrix
python3 scripts/start_matrix_report.py [--rows DIR] [--md FILE] [--json FILE]
```

The first-add message names the choice, e.g. `starting shape (auto): <kind>, score ..., ..% of the surface on it,
excess ...`. Measured, one short run each (`docs/FRAMEWORK_CHANGES_2026-09-26.md`, 2026-09-26 and its addendum of
2026-09-27):

| part | before | with the starting-shape stage |
|---|---|---|
| `cy_flat` | 21 operations (legacy, 2026-09-24) | 1 operation, IoU 0.9985, chamfer 0.0055, 16 s; start = box & cylinder |
| `bs_box_dome` | 16 operations, chamfer 0.10 | 2 operations, IoU 0.983, chamfer 0.73 (2026-09-26); IoU 0.972 (2026-09-27) |
| `bs_sphere_flat` | 1 operation | 2 operations, IoU 0.971 (2026-09-26); 1 operation, IoU 0.998 (2026-09-27) |
| liner, pitch 9.6 mm | 0.933 (surface choice) | 0.938 with `auto` (budget 5, 16 fallback cubes); 0.933 with budget 21 (16 cuts, 0 cubes) |
| frame, box start, 4 operations | one candidate step 844 s, killed at 9.8 GB | 3 cuts remove 86 % of the excess in 359 s; not pursued further (the user's decision) |

The comparison matrix had 11 of 153 cells done on 2026-09-26 and has not been completed. `auto` against `legacy`
on the full part set is not measured.

**First fit (2026-09-25).** When no primitive explains the whole input, every covering mode is built and graded:
oriented and world-axis box, bounding cylinder (sector and full), bounding sphere, box & cylinder. The grade
samples each candidate's own surface: `on`, `far`, `near`; score `on − near`. A composite needs each operand's
surface ≥ 50 % on. Liner: sector cylinder +0.58, boxes +0.03.

**Booleans** (`--no-more-booleans` switches them off): half-space cuts, "intersect" with the part's covers, box & /
− cylinder composites; each needs ≥ 20 % of the nearby surface on it. Measured on 14 parts, one run each,
2026-09-25: `cy_flat` 21 → 11 operations, chamfer 0.146 → 0.066; the other 12 small parts and the liner unchanged.

**Voxel-cube cover (`--start cubes`, since 2026-10-10; the user's option "grow from the voxels the biggest cubes, then
recognise the rest as additional").** The largest completely filled axis-aligned boxes of the input occupancy, grown
from the voxels one after the other (`residual_loop.largest_boxes`, the voxel fallback's search), are the first adds,
each face moved onto the input surface where samples support it; at most `cubes_max` (8) boxes, stopping when the
uncovered share of the input is below `cubes_rest` (2 %) or the next box holds less than `cubes_min_share` (5 %) of the
first one's voxels. The boxes lie inside the material by construction, so the cover has no excess to cut (the
dominant failure of the covers, docs/reports/server_runs_2026-10-10); what they do not reach is left to the residual
phase as additional primitives and to the feature phase. The boxes follow the grid's axes: a part that is not aligned
with the world axes gets a staircase. `python3 scripts/run_part.py part.step --out <dir> --start cubes`; in the GUI
the start combo lists it. Measured 2026-10-10 on 12 example parts against the default start, same code (server_a,
`~/data/shape_reco/cubes_examples/`, records in `~/shape_reco_results/cubes_examples/`): good (one valid solid,
IoU >= 0.95) on 8 of 12 against 11 of 12; it rescued the one part the default failed on (`cy_t` 0.752 -> 0.997) and lost
4 of the 11 the default got right (`bs_hollow_box` 0.717 and two solids - the hollow sub-loop does not run after a cube
start; `cs_cyl_sector` 0.940 with 24 operations; `co_countersink` 0.887; `three_holes` 0.903). It wins where the part is
box-like: `bs_box_step` 2 operations instead of 5 at IoU 1.0, `cy_on_box` 0.9998 against 0.980, `side_slot` and
`l_bracket` in a fifth of the time. On the liner (`liner_compare/cubes_k4`) 0.689 and not one valid solid. It is a
second path for the retry rule of §3.4, not a default.

### 4.3 The feature phase: settings and measurements

After the voxel loop the program is built in FreeCAD. Its exact difference to the input falls into pieces. A piece
gets a tool or primitive; a guess is applied and kept only when a real FreeCAD build stays one valid solid and the
exact residual drops. The phase replaces the voxel-cube fallback. STL input: fitters and cascade only; skipped above
12000 triangles. Order of the sources and stop reasons: `CODEMAP.md`, section 5.2.

```bash
python3 scripts/time_cascade.py part.step --features --json out.json
python3 scripts/time_cascade.py part.step --features --feature-time-budget 900 --feature-max-pieces 8 \
    --feature-max-iterations 3 --feature-call-timeout 300 --feature-min-score 0.85 --no-feature-batch --json out.json
```

| option of `time_cascade.py` | `ResidualConfig` field | default |
|---|---|---|
| `--features` | `feature_phase` | off |
| `--feature-max-pieces` | `feature_max_pieces` | 8 per iteration |
| `--feature-max-total-pieces` | `feature_max_total_pieces` | 40 full guesses |
| `--feature-max-iterations` | `feature_max_iterations` | 3 |
| `--feature-time-budget` | `feature_time_budget` | 900 s |
| `--feature-call-timeout` | `feature_call_timeout` | 300 s per FreeCAD call |
| `--feature-min-share` | `feature_min_share` | 1e-5 of the input volume |
| `--feature-min-score` | `feature_min_score` | 0.85 |
| `--feature-min-gain` | `feature_min_gain` | 0.25 of the piece |
| `--feature-sources "step faces,fit"` | `feature_sources` | STEP: step faces, fit, cascade; STL: fit, cascade |
| `--feature-scoring {part,iou}` | `feature_scoring` | `part` |
| `--no-feature-batch` | `feature_batch` | one validation per iteration |
| `--no-feature-types` | `feature_types` | tool types on |
| `--tool-store FILE` | `feature_tool_store` | a store for this run only |
| `--no-feature-split` | `feature_split` | splitting on |
| `--no-feature-patterns` | `feature_patterns` | congruent pieces reuse one guess |
| `--no-feature-check` | `feature_check` | exact check per feature on |
| `--no-feature-document` | `feature_document` | `features.FCStd` written |
| `--rough-cut-grade` | `rough_cut_grade` | 0.10 |
| `--rough-min-gain-share` | `rough_min_gain_share` | off |
| `--rough-max-ops` | `rough_max_ops` | the usual budget |

Fields without a command-line option (set them in Python): `feature_target_iou` (GUI: `--target-iou`),
`feature_max_written_pieces` (200), `feature_max_applied` (60), `feature_min_precision` (0.999),
`feature_contact_tool` (on), `feature_contact_ranking` (on), `feature_contact_grading` (off),
`feature_piece_order` ("volume per face"), `feature_thin_wall` (2e-3 of the diagonal), `feature_snap` (on),
`feature_tools` (on).

Output: without an `out_dir` the phase writes `~/.cache/shape_reco/features/<part>_<time>/features/` with
`features.FCStd` (input, rough and final model, each difference exploded into labelled pieces, the features as
Part objects, their checks, a Summary sheet). Run `features_colours.FCMacro` from that directory once in the
FreeCAD GUI for the colours.

Measured on the liner (`--grid tau --start auto --resolution 192 --max-resolution 384`, one run each). Voxel IoU by
`scripts/voxel_iou.py --pitch 2.0`. The input has 45,948,204 mm³.

| run | date | rough model | features applied | exact IoU (OCCT) | voxel IoU | invented (mm³) | time | stopped |
|---|---|---|---|---|---|---|---|---|
| phase off | 2026-09-27 | 10 ops | — | 0.9475 | 0.9301 | 2.53e6 | 131 s | — |
| phase on, detailed rough stage | 2026-09-27 | 10 ops, 0.9475 | 12 (9 box cuts, 3 cylinder cuts R 6.92) | 0.9632 | 0.9449 | 1.74e6 | 1043 s | time budget |
| coarse rough, all on (5aff03a8) | 2026-09-28 | cover + bore, snapped, 0.8600 | 10 (9 rounded boxes: 1 tool + 8 by pattern; 1 box) | 0.8952 | 0.8919 | 5.38e6 | 1263 s | time budget |
| before hole stacks | 2026-09-28 22:08 | — | 13 (8 by pattern), 45 pieces rejected | 0.9723 | not measured | 1.298e6 (sampled) | not recorded here | — |
| with hole stacks | 2026-09-29 09:23 | — | 39 (31 by pattern), 7 pieces rejected | not available | not measured | 1.218e6 (sampled) | not recorded here | the second iteration's exact difference ran into the 300 s limit |

- The last two rows are from `docs/reports/hole_stacks/hole_stacks.tex`. For the model with 39 features FreeCAD's
  exact difference gave no answer within 300 s and failed after 1268 s. The IoU from the volumes of the two solids
  is 0.9740; it holds only if the model contains the whole input, which was judged by sampling, not exactly.
- Row 2, peak memory 7.2 GB (phase off: 6.8 GB). Its phase time: exact differences 436 s (5 calls), guesses 389 s
  (cascade 213 s, fitters 123 s for 16 pieces), builds 3 s. 12 applied, 4 rejected, 16 skipped.
- Row 3: the pockets are rounded boxes (precision 1.000, exact harm 0 mm³, where a plain box removed 440 mm³ of the
  input). Missing against row 2: the cover's excess over the liner's outer shape (five pieces, 14 % of the input)
  and the hole groups.
- The hand-made CSG of the liner reaches voxel IoU 0.962 (`docs/LINER.md`).

**Hole stacks** (since 2026-09-29; `reco_engine/hole_stack.py`). A hole is guessed as the stack of cylinders and
cones its piece's real faces name. Sources in the steps: "tool (hole stack)" and "tool (contact faces + hole
stacks)". On the stored pieces of the run of 2026-09-28 the tool reaches precision 1.000 and recall 1.000 in less
than one second per piece (`docs/reports/hole_stacks/hole_stacks.tex`).

### 4.4 Feature stage (STEP input; features listed, not applied)

```bash
python3 scripts/feature_stage.py dataset_gen/examples/beams/m02_bolt_holes.step --out /tmp/m02 --fcstd
python3 scripts/feature_stage.py part.step --out out --all-sources        # ask every source for every piece
python3 scripts/feature_stage.py part.step --out out --rough-step my_rough_model.step   # a given rough model
python3 scripts/grade_difference.py /tmp/m02 --apply      # -> /tmp/m02/grade.json; --apply builds rough + features
```

Options of `feature_stage.py`: `--out` (required), `--sources` (default "step faces,fit,cascade"), `--rough-step`,
`--max-pieces` (500 per sign), `--all-sources`, `--fcstd`. Options of `grade_difference.py`: `--apply`, `--delta0`,
`--json`.

| output file | content |
|---|---|
| `rough.step`, `rough.stl`, `rough.json` | the rough model and the builder's program |
| `rough_parameters.json` | every operation's kind, pose, dimensions, label, sources |
| `pieces/`, `pieces.json` | each piece as STEP and STL, with its exact faces and where each face comes from |
| `features.json` | summary and feature list |
| `features.FCStd` | with `--fcstd`: input, rough model and every piece as a labelled object |

Measured on `m02_bolt_holes.step` (HEA 300 x 2 m, 4 bolt holes d 22 through both flanges), 2026-09-27: rough model
3 boxes, exact IoU 0.9425; 12 pieces: 8 cuts, each a through cylinder of radius 11 and length 14 read from the STEP
faces (score 1.000), and 4 adds of 312,889 mm³ each, the root fillets, which no primitive describes (best score
0.24; D-001).

Difference grade, measured on workstation, 2026-09-27, one run per part (`reach` = the predicted IoU with the features
applied, `measured` = the exact IoU of that build):

| part | IoU | skin | feature | open | pieces s/f/o | reach | measured |
|---|---|---|---|---|---|---|---|
| s03_rod | 1.0000 | 0 | 0 | 0 | 0/0/0 | 1.0000 | 1.0000 |
| s05_rhs | 1.0000 | 0 | 0 | 0 | 0/0/0 | 1.0000 | 1.0000 |
| m01_endplate | 0.9688 | 0 | 0 | 0.0312 | 0/0/4 | 0.9688 | 0.9688 |
| s01_ipe200 | 0.9566 | 0 | 0 | 0.0434 | 0/0/4 | 0.9566 | 0.9566 |
| m02_bolt_holes | 0.9425 | 0 | 0.0019 | 0.0556 | 0/8/4 | 0.9443 | 0.9443 |
| m03_mixed | 0.9562 | 0 | 0.0055 | 0.0382 | 0/8/12 | 0.9615 | 0.9615 |
| s04_tube | 0.1422 | 0 | 0.6432 | 0.2146 | 0/1/4 | 0.3984 | 0.3986 |
| frame (STEP) | 0.9655 | 0 | 0 | 0.0345 | 0/0/60 | 0.9655 | 0.9655 |

No skin occurs in these runs; the skin class is tested on a synthetic part only (`tests/test_feature_stage.py`).

### 4.5 Timing: where the loop spends its time

```bash
python3 scripts/time_cascade.py "$D/Unbenannt-Frame_4_x_6.stl" --gui-fold --json docs/data/timing_frame_cascade.json
```

Output: one line per step, then every section (`reco_engine/timing.py`: nested, wall and CPU including child
processes) with calls, share and self time. The GUI does the same: each step's log line gets its biggest sections,
and at the end the table goes into the Output panel and `~/.cache/shape_reco/timings/<part>_<time>.json`.
New sections: `with timed("name"):` or `@timed_fn("name")` (not on generators).

**Run timing jobs alone.** A second job on the laptop inflates every number.

Frame STL, GUI settings (96³ → 192³, top-k 2), one run each, measured 2026-09-24:

| | whole run | biggest part |
|---|---|---|
| before (cuda field backend, trimesh `contains`) | 421 s headless + 19 s GUI meshing | `field channels` 313 s |
| `field_backend="igl"` | 120 s (not run alone) | `exact eval (mesh)` 65.6 s |
| + igl winding number in `ibeam.ExactJudge` | 46.7 s | `gui fold/mesh model` 16.8 s |
| + exact model meshes by mesh booleans | 26.9 s (run alone) | `ops budget` 7.7 s, `exact eval (mesh)` 7.8 s |

Same program in all four: 32 operations, residual 0 / 0, chamfer 0.032973 (STL units). The type stage gave
torus/band 0.915. What is left in the last row: `ops budget` 7.7 s (29 %), `exact eval (mesh)` 7.8 s, `detect
members` 3.8 s, `type stage` 3.0 s (of which the descriptor model load is about 2.6 s, once per process), `freecadcmd
build` 2.2 s. The frame's 32-add model has volume 802.68 of 882.14 (STL units³), 0.910 of the input.

### 4.6 Gate G3: the end-to-end suite

60 parts (the frame STL, `dataset_gen/examples/beams/*.stl`, `dataset_gen/examples/*.stl` except `box_i000000`), one
process per part. Defaults (`scripts/g3_run.py`, `DEFAULTS`): type source `registry`, max resolution 128, 2 jobs x 2
threads, nice 10, timeout 1500 s (frame, beams) / 600 s. Resumable: rerunning a label skips the parts that have a
record. Check with `resource-guru` before a full run.

```bash
python3 scripts/g3_run.py --label <name>                         # -> docs/data/g3/<name>/<parent>__<stem>.json + run.json
python3 scripts/g3_run.py --label <name> --list                  # what is still pending
python3 scripts/g3_run.py --label smoke --jobs 1 --parts dataset_gen/examples/blind_hole.stl
python3 scripts/g3_report.py --base docs/data/night_g3/new_fix --new docs/data/g3/<name> \
    --json docs/data/g3_report.json --md docs/data/g3_report.md --tex docs/data/g3_macros.tex
```

Per part: exact IoU (OCCT, or `sampled` with its pitch), r+ / r− in mm³, genus of the result and of the input,
chamfer in mm (the frame STL's units × 100, D-008, asserted by its extent), FreeCAD validity, number of operations,
CPU and wall seconds, commit. The report's verdict is the chamfer rule of `docs/data/night_g3_eval.md` (worse if
new > 1.5 × base + 0.05). An IoU verdict is given only when both runs measured the part by OCCT (`--iou-tol`,
default 0.01).

### 4.7 The validation cycle (shapes with a known CSG tree; since 2026-10-01)

Selection on the laptop (light), everything heavy on server_b, only the small JSON files back (CODEMAP §5.4):

```
python3 scripts/validate_known.py select --out <corpus> --aegir-per-class 3 --v5-per-stratum 1 --seed 0
python3 scripts/validate_known.py start  <corpus> --server server_b --cpuset 8-13 --jobs 6   # build + gate + runs
python3 scripts/validate_known.py status <corpus>
python3 scripts/validate_known.py pull   <corpus>
python3 scripts/validate_known.py report <corpus>          # tiers, metrics, separation classes per stratum
python3 scripts/validate_known.py review <corpus> <stem>   # <corpus>/review/<stem>/*_groups.FCStd: open in FreeCAD
```

The regression gate (2026-10-02): freeze a corpus's shapes once, rerun them with the checkout under test (`start` syncs
the checkout that runs it), compare paired by stem:

```
python3 scripts/validate_known.py fork    <corpus> --out <gate> --subset 60           # stratified, deterministic, no runs
python3 scripts/validate_known.py start   <gate>  --server server_b --cpuset 8-13       # from the worktree under test
python3 scripts/validate_known.py compare <corpus> <gate> --out cmp.json               # tier moves, share A, median IoU
python3 scripts/validate_known.py compare <corpus> <corpus> --rating-b rating_polished.json   # two ratings, same runs
```

Keep a change only if `compare` resolves its gain (the interval excludes 0) and lists no tier worse and no IoU fallen
beyond the noise.

Finished runs polished after the fact (stage 6c; the same programs re-rated, nothing reconstructed again; on the server,
a few seconds per run): `python3 scripts/polish_runs.py ~/data/shape_reco/validation/<corpus> --workers 4`
writes `program_polished.json`, `polish.json`, `rating_polished.json` per run and `<corpus>/polish_summary.json`. Set
`OMP_NUM_THREADS=1` per worker on a shared server (server_f reached load 18 on 12 cores without it, 2026-10-02).

One part by hand: `python3 scripts/run_part.py <shape.stl|step> --out <dir> --truth <truth.json>` writes `path.jsonl`
and `rating.json`; `python3 -c "from reco_engine.validation import rate_run; rate_run('<dir>', '<truth.json>')"` re-rates.
The server's models are its own copies (`/data/user/shape_reco/tests_checkout/{models,checkpoints*}`, linked by
`start`); without them the type stage falls back silently and the rating measures a weaker framework.

### 4.8 The shape-class atlas (since 2026-10-29)

Every part classified by `CLASS_V1` = topology . surface family . tree complexity (CODEMAP §5.5), joined with the stored
runs, counted per class. Describing is light (one freecadcmd read per part, no reconstruction): 60 gate parts in 34 s
on the laptop (3 jobs), the 6054 ABC parts on server_a with 12 jobs. The heavy part of the atlas is the class-stratified
reruns, which go to server_a + server_b like every gate run.

```
python3 scripts/shape_atlas.py describe --corpus ~/shape_reco_results/validation/gate_paths --domain gate \
        --out ~/shape_reco_results/shape_atlas/desc_gate.jsonl --jobs 3
python3 scripts/shape_atlas.py describe --list real_list.txt --domain real --out .../desc_real.jsonl   # name<TAB>path
python3 scripts/shape_atlas.py calibrate .../desc_gate.jsonl .../desc_validation.jsonl .../desc_pool.jsonl
python3 scripts/shape_atlas.py join .../desc_*.jsonl \
        --runs "gate=$HOME/shape_reco_results/validation/gate_paths/runs_*/*@variant/part" \
               "validation=$HOME/shape_reco_results/validation/calib_20261002/runs/*@variant/part#val_1002" \
        --table abc_0930=$HOME/shape_reco_results/abc/abc_paths/table.csv --out docs/data/shape_atlas.jsonl
python3 scripts/shape_atlas.py report docs/data/shape_atlas.jsonl --variant tree_lg --md out.md --fig out.pdf
python3 scripts/shape_atlas.py select .../desc_pool.jsonl --per-cell 8 --in-scope --out list.txt --strata-out strata.json
python3 scripts/validate_known.py fork <pool> --out <corpus> --subset 150 --strata-from strata.json
```

On a server: `describe_run.sh <name> <jobs> <describe args>` in `~/data/shape_reco/shape_atlas/` (niced,
`OMP_NUM_THREADS=1`, the cache in `/data/user/shape_reco/cache/shape_class` - `~/data` is 93 % full;
checkout `/data/user/shape_reco/wt/shape_atlas_20261029` with a `COMMIT` file). Paths with spaces in a `--list`:
tab-separated lines. Checks of 2026-10-29: the B-rep genus equals Euler-Poincare (`docs/reports/step_construction_review/
step_facts.py`) on 60 of 60 gate parts and `truth.mesh_genus` on the 53 one-shell parts (the 7 others have voids, which
`mesh_genus` refuses); the frame STEP has b1 17, all window handles (b1_rest), the liner b1 36, all through holes; the
join reproduces every gate variant's good count (tree_lg 30, bare 28, tree_l 29, cnn 29, cubes 19). Its one-solid
counts are one VALID solid: tree_l 44, tree_lg 51, tree_c 38. That is 2 below the gate readouts, which counted the
invalid inverted builds too.

**Describing, 2026-10-29.** All parts were described in 2131 s on server_a (12 jobs). 4 failed, all on the 900 s
FreeCAD timeout.
- **Memory:** two of those failures held 52 and 77 GB in freecadcmd until the timeout. Since then `describe_run.sh`
  caps every process with `ulimit -v` (`SHAPE_ATLAS_MEM_GB`, default 12).
- **Genus without a closed tessellation:** 109 parts had a tessellation that did not close. The Euler-Poincare fallback
  gives 102 of them a genus; 7 stay `g?` because their B-rep is invalid.
- **Other corpora:** the validation corpora (698 of 701 described; 3 truths FreeCAD cannot build) and the aegir pool
  (1099 trees, 19 strata) ran on server_b. The frame, the liner, the 11 NIST AP203 parts and the 72 machined trees
  ran on the laptop.
- **Leaf calibration:** fitted on 1867 known trees, `R2` 0.694, 5-fold bin accuracy 0.633. Its mean log2 residual is
  -0.03 on aegir and -0.37 on the machined trees, i.e. it underestimates machined parts. A fit on the machined trees
  alone (`docs/data/shape_class_calib_machined.json`) has `R2` 0.877. Rerunning the ABC report with that fit
  (`shape_atlas_abc_machcal.md`) moves no family or flag odds ratio by more than 0.04.

**Atlas v1, from the stored runs** (`docs/data/shape_atlas_*.md|json`, figures `docs/figures/shape_atlas_*`, thesis
§14 `14d_shape_class_atlas.tex`, tables by `scripts/thesis/make_shape_atlas_tables.py`).

ABC batch of 30 Sep, in scope: 4469 runs, good 0.600. The joint model (AUC 0.821), as odds ratios:

| covariate | odds ratio |
|---|---|
| per doubling of the leaves | x0.385 |
| family D (against P) | x0.10 |
| family A (against P) | x0.32 |
| family M (against P) | x0.50 |
| blend | x0.51 |
| thin | x0.41 |
| void | x0.40 |
| per doubling of the through-hole handles | x1.44 |
| per doubling of the window handles | x0.94 +- 0.04 |

- **Capability gaps:** M at >= 8 leaves in every genus bin; D and A at >= 8 leaves. The failures there are invented
  volume first, then not one solid.
- **Coverage gaps:** g4-9.M (706 real parts, 15.8 %) and g10+.M (338, 7.5 %) had no known tree. With
  `scripts/make_machined_trees.py` (72 trees) each now has 6.

**Atlas v2 (running since 2026-10-29 04:18).** These are the class-stratified reruns with checkout
`shape_atlas_20261029b|c` (= main 4ef6212e + the atlas). All use tree_lg, i.e. `--mode tree` with the gate's settings
(`server/shape_atlas/rerun.sh`).
- **server_a:** `runs_abc_tree_lg`, 312 ABC parts (26 classes x 12, classes with >= 30 parts), 12 jobs.
- **server_b:** `known_chain.sh` builds and runs `class_known_20261029` (149 pool trees, 43 classes x 4; 130 passed the
  truth gate). `machined_20261029` (72) is queued after it.

**The load breaks the timing.** server_a sat at load 51-65 on 32 cores. A part takes far more than its 2 threads in
bursts (one run_part at 1480 % CPU; probably libigl's and OCCT's own thread pools), so GNU parallel `--load` holds new
jobs back. The feature phase's 900 s budget and the 1800 s reconstruction timeout bite harder under that load, so the
v2 readout must count stops and timeouts per class before it is compared with the gate. Next time use 6-8 jobs on
server_a.

### 3.8 Server runs: started in the window, followed live, in the Benchmark (since 2026-10-29)

**The part library** (the part field; `setup.discover_library`): after the examples come
- the benchmark parts (frame, liner);
- the detail study (`~/shape_reco_results/detail/*/step`);
- the known-tree corpora of the protocol: gate (60, DEV), class_known (130, TEST) and machined (72, TEST), in
  `~/shape_reco_results/{validation/gate_paths,protocol/corpora/*}` with STEP and STL, 437 MB.

Each part is listed once, as STEP where there is one. A part with a known tree (`truths/<stem>.json` one level up) is
rated in the Benchmark tab and, on a server, run with `--truth`.

**In the window, for any part.** Choose the part (step 1) and the starting part (step 2), then set the run bar's
**on** to *server_a* or *server_b* and press Start (or *Next* at step 3). Then:
- The laptop's commit is deployed to `/data/user/shape_reco/wt/gui_<commit>` once per commit. Uncommitted changes
  are not on the server, and the log says so.
- `run_part` starts there, niced, with the window's path (start shape, mode, sub-parts, type source, registry), the
  feature phase and its budget, the starting part (`--first-guess`) and, when the part has a known tree, `--truth`.
- The window follows the run live (the user, 2026-10-29: "not 100 % live, but roughly show which shapes we are
  investigating"):
  - the steps in the log;
  - the model in blue over the grey part;
  - the newest primitive the run tried, in the candidate layer (pink a cut, orange an add), with what happened to it
    in the caption and the status line ("step 82: accepted cut box (gain 9242)", "refused cut cylinder - void check");
  - the leftover component it works on, as a yellow box.
  The newest try is shown and older ones are skipped, so the view never falls behind (20 ms per candidate mesh).
- **Speed** (the user, 2026-10-29: "really slow").
  - A run started in the window runs at nice 0, the batch jobs at nice 10, so it comes first on a shared server.
  - *on: a server: the least loaded* picks the server where the run should finish first: the measured speed factor
    (server_a and server_b 1, server_c and server_d 3; §3.4) times max(1, load per core). Each alternative picks again, so
    alternatives spread over the servers.
  - The live view stays open while the server process lives, however long a step takes. The first version ended after
    120 s of one 863 s candidate search.
  - The slow run of 12:45 (ABC 00000393, 6852 x 5144 x 3226 units) took 1629 s on server_a. 863 s of it went into one add
    component's candidate search on a 436 x 329 x 208 grid (30 M cells, already coarsened and blind: pitch 1.3 x the
    thinnest wall). Every candidate evaluation passes over the whole grid. The run went twice, from a window older than
    the start guard below, beside three batch arms.
- A second Start while a server run of the part is starting or running only says so. In the session of 10:41 the deploy
  took 20 s without a sign, and the same run went to server_a twice. *Tools > Stop server runs* stops the runs started
  from the window, by their process ids.
- When the run ends, the window fetches it, shows the exact model and the summary (record, program with the IoU after
  every operation, rating), and **stores it in the run store** (`run_store.import_run_part`). The Benchmark tab, Open,
  Compare, *Nephogram* and *Event time (aegir)* then treat it as any run; its row's label is "server <host> <path>".
- With *reconstructions per part* > 1, the alternatives start together on the server after a first run that is not
  exact (IoU < 0.999). The Benchmark tab opens with all of them checked, and step 4 compares them as for local runs.
- Closing the window does not stop a server run. File > *Follow a server run* opens it again later; a finished run is
  then shown and stored at once.

Measured (2026-10-29 10:36, the ABC plate 00009283 through the window offscreen, server_a under the load of the focus
arms): deploy and start, 53 s of reconstruction, exact IoU 1.0000, one valid solid. It was stored and listed in the
Benchmark (variant bare, with model), and no alternatives were started after the exact result. Everything together
took 188 s.

**From the command line** (the same machinery):

The user, 2026-10-29: "a link which we could use to compute on the server but preview live on the computer". The
reconstruction runs on server_a or server_b; the window on the laptop shows its steps and its model while it runs.

```bash
python3 scripts/remote_run.py start <part.step> --host server_a [--truth tree.json] [--extra "--mode tree"] [--first-guess fg.json] [--gui]
python3 -m reco_gui --follow ~/shape_reco_results/remote/<id>      # or File > Follow a server run ...
python3 scripts/remote_run.py status [<mirror>]                    # host, pid, state, steps mirrored
python3 scripts/remote_run.py fetch <mirror>                       # record, program, rating, model.stl -> <mirror>/run
python3 scripts/remote_run.py stop <mirror>                        # ends the server run by its pid
```

- **What happens.** `start` copies the STEP (its STL and the truth) to `~/data/shape_reco/remote/<id>/`, which
  is the same on every server over NFS. It starts `scripts/run_part.py` there from a /data checkout (default
  `protocol_e6aa19ed`), niced, in the batch environment of `server/server_a/run_parts.sh`, and prints the host's load.
  A local `ssh <host> tail -F` mirrors the run's `path.jsonl` to `~/shape_reco_results/remote/<id>/` until the run's
  process ends; the file is flushed per step (`reco_engine/run_path.py`).
- **What the window shows.** `reco_gui/server_follow.py` reads the new lines every second. Each step becomes a log line
  `[server] ...`. An operation entering or leaving the program rebuilds the model (`program_from_path` ->
  `program_model` -> an SDF mesh at 96^3, in a thread), shown in blue over the grey part.
- **What it is not.** The preview is not the exact model.
- **When the run ends.** The window fetches the run's files itself (`reco_gui/server_result.py`: record, program,
  rating, `model.stl` into `<mirror>/run`). It then shows the exact model over the part and, in the detail panel, the
  record, the program with the IoU after every operation, the rating (tier, separation point, defect classes) and, when
  `scripts/detail_study.py` compared the run, the leaves against the trees and the steps where the software lacked.
  Following a finished mirror shows its results at once.
- **First use** (the bracket of §4.10, server_a, 2026-10-29 10:15): the mirror had 94 steps after 60 s of computing and
  the window followed it.

### 3.9 Post-processing: the tree, the mass ratios, fewer parts (since 2026-10-30)

The user, 2026-10-30: "a postprocessing window which shows the user the tree of the reconstruction, the mass ratio of the
subparts; interactively the user is able to select parts and then benchmark, decide and export" and "minimize the
number of parts: simple trees for simple parts instead of complex trees for simple parts".

Open it with **Tools > Post-process ...**, the flow's step 5 or **Post-process ▸** on the result bar: the newest result
of the session, else the result bar's stored run. The tree is the program's boolean tree (Fusion / Cut / Common); per
part:

| column | meaning |
|---|---|
| Mass | an add: its share of the final volume, shared material split equally (the adds add up to 100 %); a cut: the volume it removes for good |
| Unique mm³ | the volume only this part provides: what disappears when it is dropped |
| Fit | an add: the share of its mass inside the input; a cut: the share of what it removes that lies outside the input |
| Decision | keep / drop / drop (proposed) |

Select parts (a group selects its parts): the 3D view shows them orange, the benchmark compares the program as it came,
without the selection, as decided and with the proposal (sampled IoU, missing and invented volume, parts). **Minimize
parts** drops one part at a time, the best program first, while the sampled IoU stays within the tolerance of the program
as it came, and proposes the drops; **Accept proposal** decides them. **Exact check** builds the decided program in
FreeCAD (one valid solid?). Export: STEP, FreeCAD groups, GDML, the program as JSON, **Save as run** (label
"post-processed", the dropped parts in the run's `extra`). Everything runs on 150 000 sample points
(`reco_engine/prune.py`): a part's inside mask once, then any combination is a fold over the masks - seconds, no FreeCAD.

**How many parts the shape does not need** (`scripts/minimize_parts.py`), the gate's 60 stored runs of adc2d6c3 with the
defaults on (`docs/data/minimize_parts_gate.json`, 100 000 points, sampled IoU):

| tolerance | parts | dropped | median per run | mean IoU change |
|---|---|---|---|---|
| 0 | 445 -> 343 | 22.9 % | 7.5 -> 4.5 | +0.0150 |
| 0.002 | 445 -> 271 | 39.1 % | 7.5 -> 4.0 | +0.0146 |
| 0.005 | 445 -> 253 | 43.1 % | 7.5 -> 3.0 | +0.0139 |

Per run, true leaves | reconstructed | after minimize at 0.002: unions of 2-3 leaves 2.5 | 12.0 | 5.8, intersections
2.5 | 9.5 | 6.2, subtractions 2.5 | 6.2 | 3.5, single primitives (v5) 1.3 | 2.2 | 2.0. A quarter of the parts change
nothing or harm. Sampling does not see connectivity: dropping a part can split the model. The loop's **prune step**
(`ResidualConfig.prune_parts`, `run_part --prune`, off) therefore builds the pruned program in FreeCAD and keeps the
longest prefix of the drops that stays one valid solid; protocol `prune_v1` (kind "simpler": non-inferior good rate,
fewer parts) judges it on the gate, then the test set, on server_b.

### 4.9 The reconstruction protocol (since 2026-10-29)

The user, 2026-10-29: "develop a scientific testing method based on all we got already", and "the framework should
work more focused through its different options instead of wandering around and wasting time and energy". A change of
the reconstruction is judged by a pre-registered, paired comparison of two arms on frozen corpora, against the
measured run-to-run noise, with its cost (`scripts/sci_test.py`, `reco_engine/stats_tests.py`; the thesis: chapter 12,
"The reconstruction protocol").

```bash
python3 scripts/sci_test.py freeze <name> --hypothesis "..." --arm-a bare --arm-b tree_lg [--kind saving] \
    --dev <corpus> --test <corpus> [...] --desc ~/shape_reco_results/shape_atlas/desc_*.jsonl --settings '{...}'
git add docs/data/protocol/<name> && git commit     # BEFORE the runs (R1)
python3 scripts/sci_test.py power <name> --flip 0.033
python3 scripts/sci_test.py launch <name> B test    # the server command of an arm
python3 scripts/sci_test.py analyze <name> --label test --split test --a "<runs A>/*" --b "<runs B>/*" \
    --replicate "<runs A>/*" "<runs A'>/*" [--anchors-a <glob> --anchors-b <glob>]
python3 scripts/sci_test.py waste "<runs>/*" --desc ... --json out.json
```

| rule | what it means |
|---|---|
| R1 pre-registration | hypothesis, arms, metric, corpora, alpha 0.05, margin 0.03, decision rule in `docs/data/protocol/<name>/protocol.json` + `manifest.json`, committed before the runs; deviations listed there |
| R2 frozen corpora | DEV = the gate (60, looked at many times: decides nothing); TEST = class_known + machined (202 known trees, no aegir scene shared with DEV); REAL = the 312 ABC parts (no truth); anchors frame and liner |
| R3 paired design | same parts, same checkout (a COMMIT file), same host(s), jobs, load cap, budgets; one factor changed; arms at the same time or one after the other on the same host, never mixed hosts |
| R4 noise floor | the flip rate of a replicate (A against A': one checkout, one host, the same time); the net change rescues - breaks counts only above 1.96 sqrt(rate n). The gate's bare vs bare_b (2 flips of 60, 0.033) is NOT a replicate - other host, three other commits, 500k instead of 2M points - so that floor is provisional; the clean replicate focus_dev_A vs focus_dev_A2 runs on server_a (2026-10-29) |
| R5 confounds | stops per arm (time budget, feature phase error, exact residual timeout, crash) and the analysis without them; hosts and commits per arm |
| R6 statistics | good = one valid solid and IoU >= 0.95; exact McNemar (one-sided for "B better"); the paired bootstrap interval of the good rate; Wilcoxon on IoU and seconds; per CLASS_V1 cell Holm-corrected; McNemar power (Connor) |
| R7 decision | accept B only if McNemar p < 0.05 and the net change exceeds the noise floor and no anchor falls below its band (0.01) and no class cell (n >= 8) is worse and the one-solid count is not lower; a time-saving change: the good-rate interval above -0.03 and fewer seconds, the same other checks |
| R8 reporting | every arm reported (`result_<label>.{json,md}`), numbers from the result files only |

**First readouts (2026-10-29).**
- *tree_vs_bare_v1, DEV (the gate).* It reproduces §3.4: good 28 vs 30, rescued 4, broken 2, two-sided p 0.69;
  the net +2 is below the provisional noise floor 2.77 (from the repeat bare_b, which differs in host, code and
  points). Decision: reject or back to DEV (DEV decides nothing anyway).
  - One valid solid 57 vs 51. §3.4's "53" counted two invalid single solids.
  - Bare's feature phase never ran on 22 of 60 parts (9 exact-residual timeouts, 11 boolean disagreements, 2 STLs too
    big), against 3 for tree_lg. The two arms are six days of code apart (73eae617 / 26d0d758), the confound R3
    forbids.
  - TEST: the tree_lg arm ran on server_b at 81efefb2 before the freeze (unanalysed; a deviation on record). The bare arm
    runs on the same host, checkout, jobs and load cap after it (`queue_bare_test_20261029.sh`).
- *tree_vs_bare_v1, TEST (the confirmatory readout, 2026-10-29 15:00; server_b, 81efefb2, 202 known trees; anchors
  from server_c).* Decision: **reject**; the decision tree (tree_lg) does not beat the bare loop, so bare stays the
  default.
  - Good: 92 (bare) against 94 (tree_lg), rescued 8, broken 6, McNemar p 0.40; change of the good rate +0.010
    [-0.025, +0.045].
  - Failed checks: one valid solid 191 against 180; the liner anchor 0.9952 against 0.9627 (the frame 0.96657 in both).
  - Time-budget stops: 8 against 19.
  - Noise: the clean replicate of focus_min_share_v1 (A against A2) flips 0 of 60.
- *Power.* With the measured flip rate, the 202 TEST pairs resolve a change of the good rate of 0.05; the gate's 60
  pairs only 0.13.
- *Waste (`sci_test waste`, gate).* Half of the reconstruction time is on pieces and candidates that end rejected
  (bare 47.8 %, tree_lg 50.7 % of 7.1 / 8.3 h). The feature phase's guesses take half of all time. tree_lg makes 21
  candidates per accepted operation, bare 9.5.
  - Pieces under 0.1 % of the input take 20.5 % (bare) and 29.1 % (tree_lg) of the time. Only 7 % and 23 % of them are
    accepted, and each can move the IoU by at most about 0.001.
  - So the first focus arm is `focus_min_share_v1`: `run_part --feature-min-share 0.001` against 1e-5. It is a
    time-saving change, judged by non-inferiority. DEV runs on server_a with both arms at the same time, 3 jobs each,
    load cap 30, checkout `protocol_e6aa19ed`.

### 4.10 One part in detail (since 2026-10-29)

The user, 2026-10-29: "instead of analyzing in bulk, analyze in detail: create a complex part, store its tree, create
two parallel trees for the same geometry, compare all their IoU, reconstruct and compare the details accurately; then
find out on which step the software lacks" (`scripts/detail_study.py`).

```bash
python3 scripts/detail_study.py make bracket                       # T1, T2, T3; STEP/STL; the exact IoU of every pair
python3 scripts/remote_run.py start ~/shape_reco_results/detail/bracket/step/bracket_T1.step \
    --truth ~/shape_reco_results/detail/bracket/truths/bracket_T1.json --host server_a --gui
python3 scripts/remote_run.py fetch <mirror>
python3 scripts/detail_study.py compare bracket <mirror>/run       # against T1, T2, T3 -> docs/data/detail/bracket/compare.json
python3 scripts/detail_study.py lacks bracket                      # -> docs/data/detail/bracket/lacks.json
```

**The bracket.** A plate 120 x 80 x 12 mm with a boss R 16 and a bore R 8 through boss and plate, a rib, a pocket
32 x 24 x 6, four corner holes R 4 with counterbores R 7 x 4, and a rebate 8 x 4 along the front edge. Three trees:

| tree | leaves | depth | how |
|---|---|---|---|
| T1 | 14 | 13 | the machinist's tree: plate + boss + rib, then 11 cuts |
| T2 | 14 | 6 | additive: the plate as two slabs (no rebate cut), each corner one tool, the four corners one union |
| T3 | 17 | 16 | reordered and split: the cuts on the plate first, the plate as an intersection of two boxes, pocket and bore in two pieces |

Measured (`docs/data/detail/bracket/trees.json`): each tree builds one valid solid of 114702.0148 mm^3, and the exact
IoU of every pair is 1.0 (missing 0, invented 0; OCCT general fuse). `sdf_primitives.Cylinder` runs from z = 0 to its
height (FreeCAD's makeCylinder): a first build with centred cylinders floated the boss above the plate (2 solids).

`compare` rates the run against each tree (`validation.rate`). Volume and surface metrics do not depend on the tree
form; the leaf matching and the names of the defect classes do. Then every truth leaf is set against the nearest
reconstructed primitive of its kind (centre and size in mm, axis angle). `lacks` lists the significant defect regions
with the step and stage that caused or failed to repair them and that step's reason.

**The bracket reconstructed (server_a, `run_part` bare, 2026-10-29 10:15, 258 s; `docs/data/detail/bracket/{compare,lacks}.json`).**
Overall: 18 operations, one valid solid, exact IoU 0.9292 (missing 52 mm^3, invented 8682 mm^3), tier E against every
tree. The volume IoU is 0.9295 against T1 and T2 and 0.9292 against T3. Leaves matched: T1 6/14, T2 6/14, T3 5/17 (24
model leaves).

What came out exact (T1's leaves against the nearest reconstructed primitive):
- the pocket (centre 0.08 mm);
- the four counterbores: two from the catalogue and the 4-fold pattern, two from the step faces (0.27 and 0.40 mm);
- the bore: radius exact, length 39.2 instead of the tree's 36 with its overshoot (3.2 mm, just outside the matching's
  2 % of D);
- the holes and the rebate are geometrically right but come as pieces of the counterbored tools and of the rebate tool
  ("contact faces + 2 hole stacks"). They are shorter than the tree's leaves, which overshoot by 1 mm: a matching
  artefact of equivalent trees, not a defect.

What went wrong, step by step (the separation point: uncut excess of 7.48 % of the volume next to the boss's +x side
and the rib, "the cover's excess was never cut"):
1. Step 2, the first add, is the bounding box of the whole part including the boss height (120 x 80 x 34). The type
   stage said cone/sector (0.48). The plate is no separate leaf, the boss no add.
2. Steps 4-96: the rough loop carves the air around the boss with seven largest-box cuts. Its cylinder cuts all fail
   the void check (purity about 0.5) or the sign check, so the boss comes out a box.
3. Step 97: the rough rule ("a cut gains >= 0.1 x the first cut's gain") hands 6161 excess voxels to the feature phase.
4. Steps 104-177, feature piece #3 (8682 mm^3, the boss's +x corners and the rib's surroundings):
   - the right tool, box - cylinder, scores 0.973 but has precision 0.980: it would cut 133 mm^3 of the rib, which
     touches the boss there, and the 0.999 precision rule refuses it;
   - the three split segments reach precision 0.774 and 0.945, and the third one's cut does not lower the exact
     residual at all;
   - the voxel fallback (16 boxes) builds an invalid solid.
   The boss's -x corners were rounded by two box - cylinder tools, because no rib is in the way there.
So the step that lacks is the first add plus the rough loop: no plate + boss decomposition, no cylinder as an add. The
feature phase then has no tool for "round a corner except where a neighbour touches it" (box - cylinder - rib). The
results are in the window: File > Follow a server run on the mirror (the run's files are fetched at the end; RUNNING
§3.8).

### 4.11 Cuts and rough guesses (since 2026-10-29)

The user, 2026-10-29: "our cut method needs to be more accurate, often two cuts could be done by only one; check the
sizing of cuts"; "often the first rough guesses work well, but then the last guess is a really detailed part and to
find a solution is impossible - rough guess those parts".

**The cut audit** (`scripts/cut_audit.py "<runs>/*"`, `reco_engine/cut_merge.py`; judged on the input mesh's winding
number, not on the voxels):

| gate runs | strictly mergeable cut pairs | cuts in such pairs | box-cut faces short of open space |
|---|---|---|---|
| bare | 2 of 60 runs | 4 of 219 (2 %) | 255 of 1026 (25 %) |
| tree_lg | 3 of 60 runs | 8 of 176 (5 %) | 99 of 390 (25 %) |

"Strictly mergeable" means the box around the two holds no input material and no add between them reaches into it.
On the bracket, cuts 1 and 6 are such a pair.

**What runs in the loop now:**

| setting | default | what it does |
|---|---|---|
| `feature_rough` | on | a piece that no tool, no split and no congruent piece explained gets the best candidate tried for it, before the sub-parts and the voxels: score >= 0.5, precision >= 0.9, recall >= 0.5; applied only if the exact residual drops by half the piece and it harms at most 5 % of its volume and at most 1 % of the input's (`feature_rough_max_harm_input`, since adc2d6c3); source "rough:..."; a refused guess leaves the piece its sub-parts and voxel fallback in the same round (since e9cf7e2d) |
| `cut_merge` | on | after the rough phase, strictly mergeable cut pairs become one cut and redundant cuts go |
| `cut_tidy` | on | cut faces in a cover face move past it; seams between cuts are closed or their free part cut; a seam cut that a cut already holds is not added again (since adc2d6c3) |
| `cut_open_faces` | off | a box cut's face that sees only open space up to the grid's edge ends three cells past it; the other faces stay; measured, see below |
| `cut_grow` | off | every face of a box cut grows until material; measured worse, see below |
| `hollow_cut_first` | off | the hollow sub-loop fits a cavity's own shape next to the outer shell's kind, and accepts a cavity cut that also takes input material (bodies of a union reaching into the cavity) at precision >= 0.6 when its exact gain is enough; the residual phase adds the material back; measured worse, see below (`run_part --hollow-cut-first`) |

**Protocol rough_tidy (docs/data/protocol/rough_tidy_v1, \_v2).** v1 (the gate, server_b): good 28 -> 29, rescued 1,
broken 0, p 0.50 - and two parts destroyed by the rough guess: a box | sphere cut against an over-covered piece 5x
(aegir_n6_I_000121, exact IoU 0.531 -> 0) and 11x (aegir_n8_U_000040, 0.084 -> 0) the part's volume removed all of the
part, because the harm was judged against the tool's own volume. v1 ended at DEV, its TEST stopped unread. v2 (the 1 %
cap, no repeated seam cut): good 28 -> 29 again, rescued 1, broken 0; aegir_n8_U_000040 back at 0.0838; aegir_n6_I_000121
at 0.160 against 0.531 - its refused guess left the piece without the voxel fallback (fixed in e9cf7e2d, after v2's
freeze). v2's TEST runs on server_b.

**Hollow leaves (2026-10-29).** 144 of the 262 protocol parts have a leaf with an inner radius (195 tube, 95 hollow
sphere leaves). Pooled, they look far worse (bare: good 3/49 against 16/26 at 4-7 leaves, odds ratio 5.4 stratified
by size), but stratified by source x operation x size the odds ratio is 1.8: the hollow leaves sit mostly in aegir
unions, which fail with or without them (1 good of 82 unions with >= 2 leaves). On the gate's 24 hollow-leaf parts their
empty interiors hold 3.5 % of the invented volume; in unions the model fills a closed cavity completely because the
hollow sub-loop refuses the cavity cut (63-84 % of what it removes is empty, the bar is 95 %: other bodies reach into the
cavity). `hollow_cut_first` accepts that cut; measured off against on (d8367ee6, server_c/server_d, both arms at once):

| part | off | on | |
|---|---|---|---|
| aegir_n1_P_000760 (one hollow sphere) | 1.0000 | 1.0000 | control |
| aegir_n4_U_000153 | 0.4963 | 0.3613 | the residual phase (adds first, budget 5) spends every operation on boxes of 134-212 voxels putting the intruders' material back, the cover's excess (177701 voxels) is never cut |
| aegir_n5_U_000906 | 0.3538 | 0.3404 | |
| aegir_n6_U_000303 | 0.3524 | 0.3233 | |
| aegir_n16_U_000158 | 0.3707 | crashed | a cone cutter for the cavity: the exact judge's samplers knew only box, cylinder and sphere (fixed: rejection on the SDF) |

Off by default. It needs the bodies that reach into the cavity re-added whole, not as fragments.

**Measured on the bracket** (one part, server_a / server_b, the same code but for the setting named; `docs/data/detail/bracket/`):

| run | exact IoU | operations | seconds | note |
|---|---|---|---|---|
| bare (10:15) | 0.9292 | 18 | 258 | piece #3 left (8682 mm^3): its box - cylinder (precision 0.980) cuts 133 mm^3 of the rib |
| + rough guess (13:49) | **0.99922** | 20 | 184 | piece #3's box - cylinder as a rough guess: residual 9538 -> 1551 mm^3, iteration 2 repairs the rest; a weaker guess (piece #10) refused (4423 mm^3 outside) |
| + rough + cut_grow (13:59) | 0.8585 | - | 181 | the grown cuts overlap (10404 voxels cut again), the bigger first cut (26776 instead of 18822 voxels) raises the rough rule's bar, the rough phase stops after 3 cuts instead of 7 |
| + rough + cut_open_faces (14:05) | 0.8585 | - | 181 | on this part nearly every face of the first cut is open: the same growth |
| + rough + merge + tidy (14:47) | **0.99945** | 25 | 255 | **tier B**, Hausdorff/D 0.0020, RMS/D 0.00013, invented 11.6 mm^3; 1 cut pair merged, 4 faces past the cover, 6 seam fixes |

The tier E of the rough-guess run (Hausdorff/D 0.1485 at IoU 0.9992) came from walls of no volume:
- a wedge where the rough guess's box face lay 0.007 mm off the cover's y = 40 face and tilted 0.07 deg;
- a 0.019 / 0.031 mm wall between a largest-box cut and the rough guess's box, with the rib between them.

`cut_open_faces` and `cut_grow` stay off.

Two mistakes of the first tidy versions are on record:
- testing only the shared cross-section of a seam cut 716 mm^3 of the boss (offline IoU 0.9935);
- writing the tidied primitives back by position after inserting seam cuts gave exact IoU 0.704 on server_a (14:36),
  fixed by rebuilding the operations by their origin (test `test_the_loops_tidy_keeps_every_operation_in_its_place`).

### 4.12 Recognized leaves and the method grades (2026-10-30)

**Recognizing the primitives inside a compound part** (`reco_engine/leaf_recognition.py`; the user: "how could we get the
method of recognizing subshapes - parts or whole primitives - in higher ordered shapes?"). Every exact surface of a STEP
part names a primitive over its own extent (its triangles' axial range and angular coverage): a convex surface an add, a
concave one a cut, as the whole turn and, for a partial turn, the sector; a dominating convex surface also a cover (first
operation or intersection); three orthogonal plane pairs a box. Verified against the input (winding number: an add >= 98 %
inside, its volume in a verified cut counting as explained; a cut >= 98 % outside; a cover holds >= 98 % of the input),
the fewest selected by a greedy CSG on sample points and minimized. The start of the loop had used only the dominating
surfaces, which no union leaf has.

`python3 scripts/leaf_recognition_study.py --parts <corpus> --loop <runs>`, the gate's 36 STEP parts
(`docs/data/leaf_recognition_study_gate.md`; leaf recall by `validation.match_leaves` with the rating's visibility rule):

| | leaf recall | parts | IoU >= 0.95 |
|---|---|---|---|
| the recognition alone (no loop) | 0.28 | 2.5 | 6 |
| today's loop (rough_tidy_v2 DEV, stored ratings) | 0.17 | 10.7 | 6 |

Union leaves 0.54 / 0.32 / 0.19 (2-3 / 4-7 / 8+ leaves) against 0.00 / 0.03 / 0.00. As the start of the loop
(`run_part --start leaves`, STEP only; an STL or a recognition under `leaves_min_iou` falls back to the legacy start), on
`aegir_n2_U_000153_L` (two 270-degree cone sectors): exact IoU 1.000, 2 operations, tier A, both leaves - the legacy start
0.146, 10 operations, tier E. Protocol `leaves_v1` judges it (gate, then the test set, server_a).

Found on the way: OCCT's fuse of a cylinder and a rotated cone sector whose flat face lies in the cylinder's seam plane
returns the cylinder alone - valid, without an error; the builder does not check that no operand was lost.

**The method grades** (`scripts/grade_methods.py`, `docs/data/method_grades.md`), the batch of 2026-10-30 on today's code
(every path and four stage ablations on the gate against the bare control of their own host and time). The bare loop on
server_b and on server_a flips 4 of 60 parts: differences of a part or two are noise.

| method | grade | gate | decision |
|---|---|---|---|
| compartments | D | good 29 -> 1, valid solids 57 -> 33 | retired |
| cubes | D | 27 -> 20 (1 rescued / 8 broken), solids 58 -> 41 | retired |
| tree | D | 27 -> 29 (5 / 3), 3 fewer valid solids; rejected on TEST | retired |
| descriptors, cnn, analytic | C | +-1 part; analytic +182 s | not offered |
| auto | C | +-1, unions +0.10 IoU | kept, for unions |
| sub-parts | B | IoU +0.031 (p 0.02) | kept |
| feature phase | B | 27 -> 29, valid solids 45 -> 57, +291 s | kept |
| hollow sub-loop | B | IoU +0.013 (p 0.046) | kept |
| polish, member sub-loop | C | no effect on the gate | kept (tier gains; the frame) |

The window offers bare, sub-parts, auto and leaves (`path_policy.VARIANTS`); the retired paths stay known
(`path_policy.RETIRED`: stored runs keep their name, `run_part` takes their options).

# on workstation: python3 -m venv ~/shape_reco_venv && ~/shape_reco_venv/bin/pip install --no-index --find-links ~/wheels -r ~/shape_reco/server/requirements.txt torch
```

Known limits on workstation (2026-09-27):

- The descriptor type model is pickled with numpy 2 and does not load with the venv's numpy 1.24.2 ("No module named
  'numpy._core'"); the type stage then runs on the CNN entry alone. `BRUTUS_NUMPY2=1` puts `~/shape_reco_site_np2`
  (numpy 2.2.4, scipy 1.15.3) on the PYTHONPATH for one run. It is not the default, because the system's compiled
  packages fail with numpy 2 (27 tests of the suite).
- The GUI on workstation's own screen:
  `DISPLAY=:0 XAUTHORITY=~/.Xauthority python3 -m reco_gui --method cascade --run --pace 0.6 --stl <part>` in that tree.

Institute servers (server_a and others): `docs/SERVER_TRAINING.md`, `server/setup_venv.sh`, `server/run_plan.sh`
(section 10.2).

## 6. Vector-field route (route A, the comparison route)

The field `V(x) = c(x) − x` (c = the nearest surface point) of the mesh is the input. Primitives are fitted to it
and a peeling loop builds the CSG program. Nothing is trained. An operation is accepted when it explains at least
0.5 % more of the input surface and does not raise the share of the model's surface off the input by more than
0.02. The operation budget is predicted from the part's genus (`max_ops=None`); pass an int for a fixed budget
(`peel.MAX_OPS` = 24). Design and results: `docs/VECTORFIELD.md`. The frame: `docs/FRAME_FAILURE.md`.

```bash
python3 scripts/peel_case.py part.stl --grade             # one part: every stage, the program, the chamfer
python3 scripts/eval_peel_examples.py                    # every dataset_gen/examples/*.stl -> docs/data/peel_examples.json (about 5 min)
python3 scripts/eval_peel_examples.py co_cone to_elbow   # a few parts, printed only
python3 scripts/genus_sweep.py part.stl                  # exact genus and the genus at coarse resolutions
freecadcmd scripts/inspect_step_faces.py part.stp        # a STEP file's exact B-rep face inventory
python3 reco_field/export.py part.stl --out out_dir       # reconstruct and build in FreeCAD: part.step, part.stl, chamfer
python3 -m reco_gui --method field --stl part.stl        # the GUI
```

Options of `peel_case.py`: `--max-ops`, `--grade`, `--all` (print every stage), `--bounding-box`,
`--invented-slack`. The `benchmark-frame` skill and `peel_case.py` run route A only.

```python
from src.closest_point_field import load_field          # sys.path: repo root and dataset_gen/
from reco_field.peel import peel, peel_steps, chamfer
from reco_field.export import build_in_freecad

field = load_field("dataset_gen/examples/co_countersink.stl")     # or a .step (tessellated by freecadcmd)
res = peel(field)
for o in res.ops:                                        # e.g. base:box, cut:cone, cut:cylinder
    print(o.op, o.fit.kind, o.fit.params, o.fit.u, o.fit.v)
print(res.explained, res.stopped)                        # "target reached" | "max operations" | "nothing helps"
print(chamfer(res.model, field))                         # two-way distance model <-> input in mm
for step in peel_steps(field, n_vectors=100, grade=True):
    print(step.stage, step.message)                      # field, base, cluster, candidate, accepted / rejected, grade, done
build_in_freecad(res.model, "out_dir", "part")            # -> out_dir/part.step, part.stl, part.json
```

Documents: `python3 scripts/make_vectorfield_tex.py` (numbers for `docs/archive/vectorfield_reconstruction.tex`),
`python3 scripts/make_gui_slide_figures.py` (GUI screenshots; needs a display).

Training data: none is needed for this route. A field can be computed from any dataset STL on demand
(`field.grid_vectors(64)` about 2 s; 20,000 exact near-surface samples about 8 s).

## 7. Legacy voxel route (route C): removed

Route C, the voxel greedy drivers (`greedy_reconstruct`, `pipeline`, `void_pipeline`, the octree variants,
`notebook_pipeline`, the `live_*` FreeCAD wrappers, `ensemble*`, `fit_score`, `fit_choice`), its GUI mode, its FreeCAD
nodes and its scripts (`reconstruct_to_stl.py`, `compare_strategies.py`, `compare_in_freecad.py`,
`live_in_freecad.py`, `ShapeReco_Reconstruct.FCMacro`, the positioning benchmark and the Type-CNN probes) were removed
on 2026-09-30 (branch `tidy`; D-002 names only routes A and B). The building blocks the main line uses stay:
`void_fit`, `void_quality`, `compare`, `local_search`, `placement`, `fit_grade`, the analytic fits, `predict`,
`model_registry`, `cache`, and `loop_guards.largest_box_in_blocks` (the voxel-cube fallback). The removed files are in
git history: `git log --diff-filter=D --name-only -- reco_engine scripts reco_gui freecad_nodes` lists them,
`git checkout <commit>^ -- <path>` restores one. The thesis chapter on the route (08_voxel_route.tex) describes it as
it was.

## 8. Inside the FreeCAD GUI

The user starts FreeCAD. Do not launch it from a session; provide the file, JSON or macro.

**The Nodes workbench** — 7 node modules in `freecad_nodes/shape_reco/` (cascade, vector-field peel, export and grade, load mesh / shape, export STL; the voxel nodes of route C were removed on 2026-09-30):

```bash
bash freecad_nodes/install.sh        # copies the nodes into FreeCAD's Nodes addon
```

Restart FreeCAD → *Nodes* workbench → new node editor → right-click → *Shape Reco*. Slow nodes have their own Run
button. Requires FreeCAD's *Nodes* addon (Addon Manager; `FREECAD_NODES_MOD` if it lives elsewhere). Node list and
example graphs: [`freecad_nodes/README.md`](../freecad_nodes/README.md). `install.sh` against a real Nodes addon was
not run when this was written.

A `ModuleNotFoundError: FreeCADGui` from plain `python3` is expected.

## 9. Diagnostics

```bash
python3 scripts/evaluate_on_val.py --checkpoint-root checkpoints_box_sphere_groups \
    --rotation-dataset-root dataset_gen/local_dataset_bs_groups/rotation \
    --dimension-dataset-root dataset_gen/local_dataset_bs_groups/measure --summary-only
# -> rotation 122.23 deg vs 126.48 deg random guessing; dimension +6.9% (box) / +0.3% (sphere) vs the mean
python3 scripts/check_size_observability.py --dataset-root dataset_gen/local_dataset_bs_groups/measure
python3 scripts/resource_budget.py                               # machine-wide resource report
```

Use the dataset the checkpoint was trained on, with the same `--seed` and `--val-fraction`.
`checkpoints_box_sphere_groups/` has no Type CNN; omit `--type-dataset-root` for it.

The torus guard (`reco_engine/torus_guard.py`, T-29) over the stored programs and the real torus faces:

```bash
python3 scripts/torus_guard_census.py --diag Liner_LINER_001-LINER_001=1648.2   # 5 s, no FreeCAD
# -> program trees: 679, 61 with a torus, 18 hit (spindle/horn 7, near-horn 1, oversized 10); run store: the liner's
#    90-operation run; real torus faces (docs/data/torus_faces_abc500.json): 43 horn faces of 1321, nothing else
python3 scripts/torus_guard_census.py --faces <dir of .step> --faces-json out.json      # the face census (freecadcmd)
```

## 10. Data generation and training

Check `df -h .` before generating datasets. Heavy runs go to workstation or the servers.

### 10.1 Quick pass and the three modes

```bash
bash scripts/run_small_training.sh     # generate type/measure/rotation sets, train all three CNNs -> checkpoints/

cd dataset_gen
python3 -m src.dataset_generator --mode type --topologies box sphere \
    --count-per-type 40 --macro-dir ./FCMacros --output-dir local_dataset/type \
    --freecad /usr/bin/freecadcmd --voxelize --voxel-resolution 64 --resume
cd ..
python3 -m training.train_type_cnn --dataset-root dataset_gen/local_dataset/type \
    --checkpoint-dir checkpoints/type_cnn --resolution 64 --epochs 50 --patience 15
```

`--mode measure` and `--mode rotation` work the same way. Noise flags (`--noise-*`) and cluster generation
(`--export-slurm-job`): [`dataset_gen/README.md`](../dataset_gen/README.md). These commands were taken from
`scripts/run_small_training.sh` and the `--help` output.

### 10.2 Training plan v5 (`docs/training_plan_v5.tex`)

Laptop side (CPU, short):

```bash
python3 scripts/timing_report.py dataset_gen/local_dataset_v5/small --run-log dataset_gen/local_dataset_v5_small_run.log \
    --json docs/data/v5_small_timing.json
python3 scripts/check_observability_v5.py dataset_gen/local_dataset_v5/small --json docs/data/v5_small_observability.json
python3 scripts/bench_field_backend.py dataset_gen/local_dataset_v5/small/train --json docs/data/field_backend_bench.json   # on an idle machine
python3 -m training.train_axis --dataset-root .../small/train --val-root .../small/val --align --epochs 15 \
    --checkpoint-dir checkpoints/axis_v5_t0_align
python3 scripts/make_training_plan_v5_tex.py && (cd docs && pdflatex training_plan_v5.tex)
```

Server side, one resumable runner (`server/environment.yml`; stamps under `$OUT/.done`):

```bash
bash server/run_plan.sh check      # versions, freecadcmd, stltovoxel, GPUs, field backend
bash server/run_plan.sh smoke      # pytest + 2 samples/subclass through every generator + 1 epoch of every trainer
bash server/run_plan.sh pilot      # val (30/subclass), test (50), train increments inc1..inc3 (20/+40/+120), noise 50 %
bash server/run_plan.sh curve      # every net on inc1, inc1+2, inc1+2+3
bash server/run_plan.sh full 5     # more increments
bash server/run_plan.sh final 5 occupancy   # the chosen configurations, 3 seeds
```

The runner sets `SHAPE_RECO_VOXELIZER=parity` (the library default too since 2026-10-01) and `SHAPE_RECO_FREECAD_WORKER=1`.
Datasets generated before that without the runner have stltovoxel grids (`placement.voxelizer`): convert them with
`scripts/revoxelize_parity.py`. `FAST=1` passes
`--gpu-encode --amp`.

| speed-up | measured | note |
|---|---|---|
| `SHAPE_RECO_FIELD_BACKEND=igl` | 0.03–0.35 s per 64³ grid instead of 12–19 s (2026-09-24) | exact where the one-pass query misses by up to 0.05 mm (v5 cylinder); use one backend for a whole dataset (`docs/data/field_igl_bench.json`) |
| `SHAPE_RECO_FREECAD_WORKER=1` | one `freecadcmd` per process instead of about 3 launches of 1.75 s per sample | outputs identical |
| `--gpu-encode` | dirlog on the CPU 94–374 ms/sample, on the GPU 7–9 ms | laptop GPU |
| `--amp` | training step 301 → 88 ms | laptop GPU |
| `--cudnn-benchmark` | not measured | opt-in |

`python3 scripts/bench_training_speed.py <train root> --json docs/data/training_speed_bench.json` refuses a busy GPU.
Hide the GPU from a generator next to a training run with `CUDA_VISIBLE_DEVICES=""`.
`--min-voxels 200` (default): parts under 200 clean voxels are recorded as `placement.degenerate` and skipped by the
trainers. A noised sample keeps its clean grid as `<stem>.clean.bin`; every sample with a placed solid stores its
base-point bits `<stem>.inside.npy`.

**Where a type net is wrong, and the hygiene view (since 2026-10-10).** `scripts/type_predictions.py --checkpoint
<dir>/type_cnn --roots <root> ... --out pred.jsonl` writes one line per sample: true and predicted parent/subclass, the
parent probabilities and the generator's record (dims, segment, cuts, carve, voxels), so error rates can be tabulated
against the geometry. Measured on the clean v5 set (k4 dirlog_inside on test+val, k3 on the unseen inc4): cones whose
radii differ by under 2 voxels are named right in 36–50 % of the cases (unresolvable at the grid); cylinders with
half-length/radius < 2 in 0–8 % (called cone); sectors spanning under 135° in 15–60 %. `scripts/dataset_hygiene.py
--src <roots> --out <view> --base <parent>` writes a filtered *view* of a dataset (manifests only, absolute paths,
nothing copied or deleted) without the cones under `--min-cone-taper-vox` (2.0); `load_manifest` reads the view like a
root. `server/server_a/v5_clean_enhance.sh` is the chain that made the view of the clean v5 set, generated the
cylinder/cone increment `train/t1` and trained k4 / k4h / k5t (two seeds) on server_a.

### 10.3 In-scale training data

`--voxel-scale-mode fixed`: one shape per file with a random size and position in a grid that is only partly
populated, optional cuts, and the noise flags.

```bash
cd dataset_gen
python3 -m src.dataset_generator --mode type --topologies box sphere --count-per-type 100 \
    --macro-dir ./FCMacros --output-dir local_dataset_inscale/type/shard0 --freecad /usr/bin/freecadcmd \
    --voxelize --voxel-resolution 64 --voxel-scale-mode fixed --grid-mm 40 --size-fraction 0.25 0.9 \
    --position random --cut-prob 0.6 --cut-keep 0.35 0.95 \
    --noise-salt-prob 0.0005 --noise-pepper-prob 0.02 --noise-blob-count 2 --noise-dilate-prob 0.25 \
    --noise-erode-prob 0.25 --seed 2000
```

| option | effect |
|---|---|
| shards | run several `--output-dir shardK` with different `--seed` in parallel (8 shards, 1,600 samples in about 4 min on the laptop's 16 threads); the trainer loads a root recursively |
| `--full-field-prob P` | that share of the box samples becomes the completely filled grid; for Type-CNN data only |
| `--hollow-prob P`, `--hollow-wall a b` | box and sphere: a shell by binary erosion; label = the outer shape; drawn before carve/cut |
| `--carve-prob P` | a box minus one sphere, a sphere minus one box; label = the base shape |
| `--require-manifold` | the clean part must be one uniform manifold; a hollow sample is accepted with its one cavity |
| `--noise-prob P` | noises a sample by its own coin, independent of class and kind |
| `--cut-min-thickness` | default 4 voxels: a cut that would leave a plate is skipped, like one that leaves fewer than 300 voxels |

Every draw is recorded per sample (`placement` in the metadata). Verify a carved set with
`python3 scripts/verify_inscale_dataset.py <root>`. Hollow data results: `docs/data/hollow_box.md`. Write-up of the
in-scale training: `docs/archive/thesis_inscale_training.tex` (`python3 scripts/report_type_cnn.py && python3
scripts/make_report_artifacts.py`).

### 10.4 Symmetry-aware Rotation CNN for boxes

`training/rotation_symmetry.py`: canonical frame (body axes longest first, `M = A·R`, `A` = world→array =
`[z, -y, x]`), symmetry group per sample, symmetric loss, 6D output, alignment.

```bash
python3 -m training.train_rotation_sym --dataset-root dataset_gen/local_dataset_inscale/type \
    --extra-root dataset_gen/local_dataset_inscale/rot --checkpoint-dir checkpoints_rot_sym/G/rotation_cnn \
    --rep 6d --align --jitter-deg 20 --epochs 40          # about 50 s/epoch on 3.9k boxes
python3 scripts/eval_rotation.py --model G=checkpoints_rot_sym/G/rotation_cnn    # -> docs/data/rotation_eval.json
```

346 held-out boxes, symmetry-aware angular error in degrees (`docs/data/rotation_eval.json`):

| method | median | within 10° | median, 0 / 1 / 2 cuts |
|---|---|---|---|
| random | 74.8 | 0 % | 77.5 / 73.7 / 69.4 |
| old Rotation CNN (given the true dimension order) | 76.7 | 0 % | 78.1 / 77.3 / 71.8 |
| network on the raw grid (6D, sym. loss) | 33.3 | 6 % | 26.1 / 38.7 / 46.2 |
| + random grid-rotation augmentation | 31.4 | 9 % | 24.9 / 32.9 / 44.6 |
| PCA (no learning) | 6.6 | 58 % | 0.8 / 16.8 / 27.3 |
| PCA-aligned residual CNN, 1.3k boxes | 3.0 | 70 % | 0.7 / 7.5 / 15.7 |
| + jitter augmentation (20°) | 2.5 | 75 % | 1.0 / 5.0 / 14.7 |
| + 3.9k boxes (`G_more`) | **1.9** | **82 %** | 0.8 / 3.7 / 7.1 |

`H_manifold` is trained on manifold-verified data with carves (`carve` + `rot2`, 3.2k train boxes). Two fresh test
sets, 200 boxes each:

| test set | method | median | within 5° | within 10° | plain / cut / carved median | noisy / clean median |
|---|---|---|---|---|---|---|
| `test_carve` (93 % of boxes carved) | PCA | 18.8 | 22 % | 30 % | 1.2 / 18.6 / 20.7 | 21.0 / 15.5 |
| | `G_more` | 1.9 | 86 % | 94 % | 1.4 / 2.0 / 1.9 | 1.9 / 1.9 |
| | `H_manifold` | **1.5** | **94 %** | **96 %** | 1.6 / 1.7 / 1.5 | 1.5 / 1.5 |
| `test_mixed` (plain, cut, carved) | PCA | 12.4 | 30 % | 44 % | 0.6 / 20.4 / 19.6 | 13.1 / 11.3 |
| | `G_more` | 2.0 | 77 % | 82 % | 0.9 / 4.7 / 1.9 | 2.4 / 1.9 |
| | `H_manifold` | **1.7** | 74 % | 83 % | 1.1 / 8.6 / 1.3 | 1.8 / 1.5 |

The same 400 boxes taken together (`docs/data/rotation_eval_manifold.json`):

| | median | within 10° | median: plain / planar cut / carved | noisy / clean |
|---|---|---|---|---|
| PCA | 16.1 | 37 % | 0.8 / 19.8 / 20.6 | 17.6 / 15.1 |
| `G_more` (no carve data) | 1.9 | 88 % | 1.0 / 4.3 / 1.9 | 2.0 / 1.9 |
| `H_manifold` | **1.6** | **90 %** | 1.1 / 6.6 / **1.5** | 1.6 / 1.5 |

Carved boxes are as easy as plain ones (95 % within 10°). Noise does not matter. Planar-cut boxes remain the weak
spot: 24 % above 30°, and 64 % of those errors sit at 75–105°, an axis swap. Single training run; differences of a
few tenths of a degree are not significant.

Not wired into the reconstruction pipeline: the estimate is in the longest-first canonical frame, so the
Dimension CNN has to be retrained on the same convention first.

### 10.5 Face-normal frame alignment for boxes

`training/box_alignment.py` fits an orthonormal frame to the body's surface normals. About 55 ms per volume (PCA:
4 ms). `rotation_symmetry.align_volume(volume, "frame")`; `--align frame` in `eval_rotation.py` and
`eval_box_dims.py`; `--align-method frame` in `train_rotation_sym.py`.

```bash
python3 scripts/eval_alignment.py --sets test test_carve test_mixed
python3 -m training.train_rotation_sym --dataset-root dataset_gen/local_dataset_inscale/carve \
    --extra-root dataset_gen/local_dataset_inscale/rot2 --checkpoint-dir checkpoints_rot_sym/I_frame/rotation_cnn \
    --rep 6d --align --align-method frame --jitter-deg 20 --epochs 40
python3 scripts/eval_rotation.py --old /nonexistent --align frame --sets test test_carve test_mixed \
    --model H=checkpoints_rot_sym/H_manifold/rotation_cnn --model I=checkpoints_rot_sym/I_frame/rotation_cnn
```

600 held-out boxes (`docs/data/alignment_eval.json`, `rotation_eval_frame.json`, `rotation_eval_I_frame.json`):

| | median | within 10° | median by 0 / 1 / 2 cuts | within 10° by 0 / 1 / 2 cuts |
|---|---|---|---|---|
| PCA | 11.7 | 45 % | 9.2 / 17.4 / 22.4 | 51 / 40 / 13 % |
| `H_manifold` on PCA (as trained) | 1.6 | 87 % | 1.3 / 3.9 / 9.9 | |
| `H_manifold` on frame alignment (no retraining) | 1.4 | 89 % | 1.1 / 2.9 / 9.6 | 99 / 75 / 50 % |
| `I_frame` (same net, trained on frame alignment) | 1.7 | 93 % | 1.5 / 2.1 / 3.5 | 98 / 82 / 79 % |
| **frame fit alone, no network** | **1.3** | **94 %** | **1.2 / 1.4 / 2.0** | |

Paired on the same boxes: `I_frame` beats `H_manifold` (557 vs 536 of 600 within 10°, McNemar 26 vs 5, p = 2e-4).
The frame fit alone is not worse than `I_frame` (561 vs 557, p = 0.125) and has the lower typical error (median 1.29
vs 1.65°, Wilcoxon p = 7e-16). For boxes the residual network adds nothing measurable on top of the frame fit.
Modulo the full cube group 98 % of the boxes are within 10°. Classical extents after frame alignment: 0.82 voxel
median (2.83 with PCA); the dimension network: 2.5. One training run per network.

### 10.6 SDF primitives

`dataset_gen/src/sdf_primitives.py` (pure numpy). `u` = sweep angle about the axis, `v` = sphere latitude / torus
minor angle, in radians (`deg()` converts).

```python
from src.sdf_primitives import Sphere, Torus, Cylinder, Cone, Box, Wedge, Prism, Placed, deg, occupancy_grid, sdf_grid
seg = Sphere(12.0, u=deg(0, 250), v=deg(-30, 60), latitude="plane")      # "plane" = FreeCAD's makeSphere angles
tor = Torus(14.0, 4.0, u=deg(0, 200), v=deg(-60, 100))
bite = Box((10, 10, 10)) - Sphere(8.0)
occ = occupancy_grid(Placed(seg, translation=(2, 0, 0)), 64, 40.0)        # uint8 grid, array frame [z, -y, x]
sdf = sdf_grid(tor, 64, 40.0)
```

Validated against FreeCAD: `freecadcmd scripts/sdf_reference_freecad.py` builds 21 Part shapes with 1,050 points
each (`tests/data/sdf_reference.json`); `tests/test_sdf_primitives.py` checks it. Findings about Part: `makeSphere`
gives a full sphere by default and cuts between planes, not cones; the function `Part.makeTorus` does not give a
clean tube segment for its minor angles (volume 2210 for angles −30..100 where the pie is 725); the feature
`Part::Torus` does, with the angle measured the other way round. `docs/archive/freecad_part_shapes.md`.

Coverage of the 17 topologies: box, cube, sphere, cylinder, cone, torus, ellipsoid (approximate distance), tube,
torus_segment; cone_segment by subtraction. Not yet: elliptical_cone, elliptical_tube, paraboloid, polycone,
polyhedron with a phi sector, general_trapezoid, parallelepiped.

### 10.7 Fused input and the multi-task network

`training/field_input.py` builds a five-channel input from a binary grid: occupancy, signed distance (truncated at
8 voxels), and the vector to the nearest surface point. 0.1 s per grid. Against the exact mesh field at 64³: median
direction error 5.3° (box), 3.0° (cylinder), 2.2° (sphere). Kinds: `occupancy`, `sdf`, `field`, `fused`.

```bash
python3 -m training.train_type_cnn --dataset-root dataset_gen/local_dataset_inscale/carve --extra-root dataset_gen/local_dataset_inscale/cyl \
    --checkpoint-dir checkpoints_fused/type_fused/type_cnn --input fused --epochs 40
python3 -m training.train_rotation_sym --dataset-root dataset_gen/local_dataset_inscale/type --extra-root dataset_gen/local_dataset_inscale/rot \
    --checkpoint-dir checkpoints_fused/rot_fused/rotation_cnn --rep 6d --augment --input fused --epochs 40
python3 -m training.train_fused --dataset-root dataset_gen/local_dataset_inscale/carve --extra-root dataset_gen/local_dataset_inscale/cyl \
    --checkpoint-dir checkpoints_fused/net_fused --input fused --augment --epochs 40
python3 scripts/eval_rotation.py --model occupancy=checkpoints_fused/rot_occupancy/rotation_cnn --model fused=checkpoints_fused/rot_fused/rotation_cnn
python3 scripts/eval_type_fused.py --model occupancy=checkpoints_fused/type_occupancy/type_cnn --model fused=checkpoints_fused/type_fused/type_cnn
python3 scripts/eval_fused_net.py --model checkpoints_fused/net_fused
```

Held-out results, 2026-09-22 (`docs/data/rotation_eval_fused.json`, `docs/data/type_fused_eval.json`):

| task | network and input | result |
|---|---|---|
| box rotation (40 epochs, raw grid; 346 test boxes) | per-task, occupancy | median 24.3°, 23 % within 10° |
| | per-task, fused | median 25.8°, 19 % within 10° |
| | `FusedNet` | median 34.0°, 5.5 % within 10° |
| type (800 test samples) | per-task, occupancy (15 epochs) | 92.4 % (box 89.5, cylinder 91.8, sphere 96.5) |
| | per-task, fused (15 epochs) | 85.9 % (box 81.5, cylinder 83.8, sphere 94.5) |
| | `FusedNet` (40 epochs) | 87.4 % (box 83.5, cylinder 85.5, sphere 95.0) |

The fused input does not help on this data and these epoch budgets. The likely cause is optimisation, not
information; it was not tuned further. `FusedNet` favours type at the expense of pose; no per-head loss curves or
capacity ablation were run. Write-up: `docs/archive/thesis_fused_input.tex`.

### 10.8 Field analysis and the `dirlog` inputs

Findings and every command: `docs/DIRLOG.md`.

```bash
python3 scripts/field_encoding_stats.py          # -> docs/data/field_encoding_stats.json
python3 scripts/field_atlas.py                   # -> docs/data/field_atlas.json
python3 scripts/eval_minkowski_baseline.py       # -> docs/data/type_minkowski_baseline.json
python3 -m training.train_type_cnn --dataset-root dataset_gen/local_dataset_inscale/carve --extra-root dataset_gen/local_dataset_inscale/cyl \
    --checkpoint-dir checkpoints_fused/type_dirlog/type_cnn --input dirlog --epochs 15
python3 -m training.train_type_cnn --input dirlog_curv --dataset-root dataset_gen/local_dataset_v5/small/train \
    --val-roots dataset_gen/local_dataset_v5/small/val --test-roots dataset_gen/local_dataset_v5/small/test --epochs 15 \
    --checkpoint-dir checkpoints_fielddirlog/v5_dirlog_curv/type_cnn     # needs <sample>.field.npz (--with-field)
python3 scripts/field_slices.py
python3 scripts/energetic_descriptors.py
python3 scripts/thesis/make_field_analysis.py
```

Input kinds: `dirlog` (5 channels, from the grid), `dirlog_exact` (5) and `dirlog_curv` (8) from the `.field.npz`;
the latter two are cached under `~/.cache/shape_reco/field_kinds` (first read 1–10 s per sample).

### 10.9 Exact mesh field input

`training/exact_field_input.py` queries the mesh directly. Its resolution is independent of the occupancy grid's;
the default is twice the grid's (`field_resolution_for`, `--field-resolution`).

```bash
python3 -m training.train_type_cnn --dataset-root dataset_gen/local_dataset_inscale/carve --extra-root dataset_gen/local_dataset_inscale/cyl \
    --checkpoint-dir checkpoints_exact_field/type_cnn --input exact_field --field-resolution 128 --batch-size 4 --epochs 15 \
    --max-per-class 300
```

Cost: about 7.7 s per mesh at resolution 128, single-threaded; always cached (`--field-cache`). At thousands of
samples that is 1–9 hours, and 10× slower beside another job on the same cores. Do not launch it at full scale
without checking that the cores are free. `--max-per-class` caps the records per topology before the input is
computed. A checkpoint with `input: "exact_field"` cannot be loaded into `reco_engine`.

## 11. Known limits

- **Five topologies have a macro in this checkout** (box, cone, cylinder, sphere, torus) plus
  `assemble_reconstruction`. The other 12 are on tag `archive/master-2026-09-18`.
- **The `checkpoints/` of route C are smoke-trained.** The Rotation and Dimension CNNs have not learned their tasks
  (`CODEMAP.md`, section 14.5). `search scale=1.851` (the ×1.08⁸ growth cap) is a common sight.
- **The cut phase of route C rarely accepts** with these models (section 7.1).
- **Route C erases absolute size** in the `fit` voxel mode; `scale_mm` only affects reported millimetres.
- **The exact grade can fail on a model with many features**: on the liner with 39 features FreeCAD's exact
  difference gave no answer in 300 s (2026-09-29).
- **The registry's CNN checkpoints marked legacy** were trained on stltovoxel grids (B-066, D-007).
