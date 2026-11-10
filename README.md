<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="packaging/logo/shape-reco-3d-logo-dark.png">
    <img src="packaging/logo/shape-reco-3d-logo.png" width="300"
         alt="SHaPe ReCo: a poison dart frog in a crow's nest looking out through a spyglass; Batrachotoxin in the sky">
  </picture>
</p>

# SHaPe ReCo — machined parts reconstructed as CSG programs

SHaPe ReCo takes the mesh (STL) or the B-rep (STEP) of a machined part and reconstructs it as a **sequence of
parametric CAD primitives** — boxes, cylinders, cones, spheres, tori and their sectors — combined by booleans. Every
step is materialised as a real solid in FreeCAD, so the result is an editable CAD model, not a mesh: a list of
operations ("add a cylinder of radius r and height h at this placement", "cut a box of these sizes rotated by ..."),
each with its own local coordinate system.

This is the research code of a master's thesis that is still being written. It is published as it is used: the
measured state, its gaps included, is documented below and in `docs/`.

**Contents:** [Features](#features) · [Measured state](#measured-state) · [Installation](#installation-linux) ·
[Usage: the window](#usage-the-window) · [Usage: the command line](#usage-the-command-line) ·
[Outputs](#what-a-run-writes) · [Data and training](#dataset-generation-and-training) ·
[Configuration](#configuration) · [Repository layout](#repository-layout) · [Licence](#licence)

## Features

### Input

- **STEP** (preferred) or **STL**. A STEP file is a B-rep: the feature phase reads the part's real faces (planes,
  cylinders, cones) from it, so a STEP input gives exact holes and pockets where an STL gives tessellated guesses.
- Lengths and volumes stay in the file's units.
- The **ground truth for where material is** is a voxel grid computed by ray parity of the mesh at the voxel centres,
  at a pitch fine enough for the part's thinnest wall (pitch ≤ wall thickness / 3); a fixed part-scale grid would
  alias thin walls.

### The reconstruction: two routes

- **The cascade add-cut loop with the feature phase** (`reco_engine/`, the main line).
  1. *Starting shape.* The loop begins with a cover of the part: automatic, a basic shape around the part or around a
     sub-part, or an exact shape you give (window step 2, or `--first-guess first_guess.json`).
  2. *The cascade.* A chain of stages — type, rotation, dimensions, position — proposes the next primitive for the
     unexplained material. Each stage is a trained network where a checkpoint is available
     (`models/registry.json`) and an analytic voxel or surface fit otherwise (`models/registry_analytic.json`).
  3. *The residual loop.* Adds and cuts are proposed until the difference between model and input is explained;
     cuts come first. Every candidate is built as a solid and checked against the exact geometry (OpenCASCADE
     booleans through `freecadcmd`); a candidate that does not pay for itself is rejected. Sub-loops handle prismatic
     members (profile-agnostic: no hard-coded I-beams) and hollow parts.
  4. *The feature phase.* The pieces of the exact difference that are left are explained with tools read from the
     part's own faces: holes and **hole stacks** (a hole as the stack of cylinders and cones its faces name), pockets,
     box − cylinder fillets, and rough guesses for pieces nothing else explains. The part's mirror planes and
     rotation axes are detected first; a piece a symmetry maps onto another gets that piece's tool copied exactly.
     The **level of detail** is yours: the window's detail slider (`--target-iou 0.95`) stops the feature phase once
     the model reaches that IoU (fewer, larger features); by default it explains every piece above 0.1 % of the
     volume.
  5. *Polish and tidy.* Curved primitives are refitted to the input's vertices; cuts are merged and tidied; with
     `--prune` the parts the shape does not need are dropped at the end (checked to stay one valid solid).
- **The vector-field peeling loop** (`reco_field/`, the comparison route). Primitives are fitted to the closest-surface-
  point field of the input and peeled off one by one, each step accepted only when it explains material
  (`docs/VECTORFIELD.md`).

### Checking the result

- **Exact comparison**: the IoU of model and input by exact OpenCASCADE booleans, the model's genus and the volume it
  misses.
- **Sampled comparison**: two-way surface distances (mean, median, p95, max, chamfer) from up to millions of points.
- **Nephogram**: the input coloured by its distance to the model, as STL and as PLY with a scalar field, plus the
  distance-band histograms both ways — open it in FreeCAD or CloudCompare.
- **Known trees**: for a shape whose CSG tree is known (`--truth tree.json`), a rating with tiers, defect classes and
  the step where the reconstruction separated from the truth (`reco_engine/validation.py`).

### The window (`reco_gui/`, PyQt5 + VTK)

- A **linear flow** of five steps — Part, Starting part, Reconstruct, Compare, Choose & export — in one bar, with a
  *Next* button; full screen with F11.
- **Live 3D view** of the shapes under investigation: the input, the model so far, the candidate the loop is trying,
  the residual; a timeline of every step you can click back through while the run goes on.
- **Pacing and breakpoints**: Step, Continue to the next ticked stage, Pause, a pace in seconds per stage.
- **Alternatives**: several reconstructions of the same part, then a **Benchmark** tab that ranks every stored run of
  the part (accuracy against cost, histograms, nephograms) so you choose the one you want.
- **Post-processing**: the program's boolean tree with each part's share of the mass and the volume only it provides;
  select parts, benchmark, drop the ones the shape does not need, export.
- **Export**: the tools as grouped objects of a FreeCAD document (**FreeCAD groups**: the editable parametric
  primitives, adds and cuts), the program folded into one solid with its distance to the input (**FreeCAD rebuild**),
  **GDML** for Geant4 and ROOT (every primitive a GDML solid, every operation a boolean solid), and the model and the
  input as STL / PLY with the nephogram. `shape-reco run` writes the model as STEP and STL.
- **Run store**: every finished run is saved with its settings, program and steps; open, compare and rate them later.
- A second window for **dataset generation**.

### Around the reconstruction

- **Dataset generation** (`dataset_gen/`): FreeCAD Part solids and booleans written by macros, parity voxels,
  closest-point fields, optional noise in the voxelisation layer (`--noise-*`), a `dataset_metadata.jsonl` manifest
  per set.
- **Training** (`training/`) of the cascade's voxel / field networks (type, rotation, dimensions).
- **Evaluation**: a pre-registered testing protocol (`scripts/sci_test.py`: frozen corpora, paired arms, exact
  McNemar, paired bootstrap, a noise floor from replicates, a decision rule fixed before the runs), and every method
  of the framework graded by its measured effect (`scripts/grade_methods.py`).
- **FreeCAD nodes** (`freecad_nodes/`): the stages as nodes for FreeCAD's Nodes workbench.

## Measured state

On 202 known CSG trees (random trees of primitives and machined blocks, prisms and turned parts), the loop reconstructs
**89 of 202** as one valid solid with an IoU of at least 0.95 (protocol `focus_min_share_v1`, test split, 2026-10-30).
Single primitives and machined parts are reconstructed well; **unions of rotated primitives are the main gap**: their
bounding cover leaves an excess the cuts do not remove (1 good reconstruction of 82 such unions with two or more leaves).
Every method of the framework is graded by its measured effect (`scripts/grade_methods.py`).

## Installation (Linux)

You need **Python 3.11 or newer** and **FreeCAD 1.0**. Every solid, boolean and STEP file goes through FreeCAD's
command-line interpreter `freecadcmd`:

- Debian / Ubuntu: `sudo apt install python3-venv freecad` (FreeCAD 1.0 is in Debian 13 and Ubuntu 25.04; older
  releases ship 0.20 - then take the AppImage);
- otherwise the AppImage from <https://www.freecad.org>. Unpack it (`./FreeCAD_1.0*.AppImage --appimage-extract`) and set
  `SHAPE_RECO_FREECADCMD=/path/to/squashfs-root/usr/bin/freecadcmd`.

Then, in the cloned repository:

```bash
./install.sh                 # a virtual environment with the packages, the command shape-reco, a desktop entry
shape-reco check             # imports, FreeCAD, and one reconstruction of an example part (a few minutes)
```

| option | effect |
|---|---|
| (none) | the window and the command line; packages from PyPI |
| `--with-torch` | plus PyTorch, for the trained networks (their checkpoints are not part of this repository) |
| `--system-site-packages` | the environment sees the distribution's packages (PyQt5, VTK, torch) |
| `--venv DIR` | the environment in `DIR` (default `~/.local/share/shape-reco/venv`) |
| `--no-gui` | the command line only (no PyQt5 / VTK, no desktop entry) |
| `--check` | check an installation: imports, FreeCAD, one reconstruction of an example part |
| `--uninstall` | remove the command, the desktop entry, the icon and the environment |

`install.sh` installs the checkout in place and writes only to the environment's directory and `~/.local` (the
command `~/.local/bin/shape-reco`, the desktop entry "SHaPe ReCo" and its icon). Without trained checkpoints the cascade
uses its analytic stages — everything works, the networks are an option.

Without the installer: `python3 -m pip install -r requirements.txt -r requirements-gui.txt` in an environment of your
choice, then `python3 -m reco_gui` from the repository's root.

## Usage: the window

```bash
shape-reco                                   # the window; or "SHaPe ReCo" in the desktop menu
shape-reco part.step                         # that part preselected (STEP or STL)
shape-reco part.step --full                  # ... full screen
shape-reco part.step --target-iou 0.95       # fewer, larger features
shape-reco part.step --run --pace 0.6        # starts at once and runs to the end, 0.6 s per stage
shape-reco --open part:latest                # look at the newest stored run of a part
```

A walk through one part:

| step | what you do | what you get |
|---|---|---|
| **1 Part** | pick a file (*Browse*, the list, or *Random*) | a preview of the part with its size and triangle count |
| **2 Starting part** | *automatic*, or a box / cylinder / cone / sphere / torus *around* the part or a region of it, or an *exact* shape | the hint shows how much of the part the shape holds |
| **3 Reconstruct** | *Start* (Ctrl+R); *Step* (Space), *Continue* (F5), *Pause* (Esc) | the model grows in the 3D view; the timeline lists every step |
| **4 Compare** | the Benchmark tab | the input coloured by its distance to the model, histograms, the runs ranked |
| **5 Choose & export** | *Use this result*, then the export menu | FreeCAD groups, FreeCAD rebuild, GDML, the nephogram, the saved run |

**Tools > Post-process ...** shows the result's boolean tree with the mass share and unique volume of every part. The
settings dock (left) holds the method (cascade or vector field), the grid (voxels per side, default 96, refined up to
192), the starting shape, the type source and, behind *More settings*, the feature phase. `python3 -m reco_gui --help`
lists every window option; `python3 -m reco_gui.dataset_gen_main` opens the dataset-generation window.

## Usage: the command line

```bash
# one reconstruction with records, the model built in FreeCAD and the exact + sampled comparison
shape-reco run part.step --out out/part
shape-reco run part.step --out out/part --prune --reconstruction-timeout 1800

# the headless loop alone: one process, one exit code, one JSON result on stdout (for batch jobs and CI)
shape-reco reconstruct --input-stl part.step --workdir out/work -o out/result.json

# the vector-field route on one part, with the two-way chamfer at the end
shape-reco peel part.stl --grade

# try it on the bundled example parts
shape-reco run dataset_gen/examples/box_with_bore.stl --out out/box_with_bore
```

Options of `shape-reco run` (`scripts/run_part.py --help` lists all):

| option | effect |
|---|---|
| `--no-features` | the rough model only, no feature phase |
| `--feature-min-share S` | pieces under this share of the input volume are dropped unless a tool knows them (default 0.001) |
| `--feature-time-budget SEC` | time for the feature phase |
| `--first-guess FILE` | your starting part(s), as the window's step 2 writes them |
| `--start START` | the starting shape: `legacy` (default), `auto`, `cubes`, or a named type (`oriented box`, `cylinder`, ...) |
| `--subparts` | the rough model as several sub-parts |
| `--prune` | drop the parts the shape does not need at the end (checked: one valid solid) |
| `--no-polish` | skip the refit of curved primitives to the input's vertices |
| `--type-source registry\|descriptors\|cnn` | which type sources vote: all, the descriptor model only, or the networks only |
| `--registry FILE` | another stage registry, e.g. `models/registry_analytic.json` |
| `--truth FILE` | the shape's known CSG tree: rate the run against it |
| `--reconstruction-timeout SEC` | the loop ends with the model it has when the time runs out |
| `--points N` | points of the sampled comparison |

Stored runs and exports:

```bash
python3 scripts/compare_runs.py list [part]                        # the stored runs, newest first
python3 scripts/compare_runs.py diff part:previous part:latest     # two runs side by side
python3 scripts/export_gdml.py part:latest --out part.gdml --check # the program as GDML, checked by sampled points
```

### What a run writes

`shape-reco run ... --out DIR`:

| file | content |
|---|---|
| `record.json` | input facts, settings, timings per stage, the model's and the comparison's numbers; rewritten after every stage, so a killed run still says where it was |
| `program.json` | the operations in full: primitive, parameters, placement, add or cut |
| `model.step`, `model.stl` | the model built in FreeCAD |
| `compare.json`, `exact.json`, `regions.json` | the sampled and the exact comparison, the regions the model misses or adds |
| `steps.jsonl`, `path.jsonl` | the loop's steps, and every operation and candidate with its gain and the reason it was kept or dropped |
| `rating.json` | with `--truth`: tier, defects and the separation point |

The window stores its runs in `$SHAPE_RECO_RESULTS` (default `~/shape_reco_results/<part>/<time>_<commit>/`).

## Dataset generation and training

```bash
bash scripts/run_small_training.sh     # a quick pass: generate small type / measure / rotation sets, train the CNNs

cd dataset_gen
python3 -m src.dataset_generator --mode type --topologies box sphere --count-per-type 40 \
    --macro-dir ./FCMacros --output-dir local_dataset/type --voxelize --voxel-resolution 64 --resume
cd ..
python3 -m training.train_type_cnn --dataset-root dataset_gen/local_dataset/type \
    --checkpoint-dir checkpoints/type_cnn --resolution 64 --epochs 50 --patience 15
```

Every sample is a single FreeCAD Part solid (no compounds) with its parity voxels and, optionally, the closest-point
field; `--mode measure` and `--mode rotation` work the same way. The trained checkpoints enter the cascade through
`models/registry.json`. Datasets and checkpoints are not part of this repository. Details: `docs/RUNNING.md`, section 10.

## Configuration

| variable | effect | default |
|---|---|---|
| `SHAPE_RECO_FREECADCMD` | the `freecadcmd` to run | `freecadcmd` on the PATH, then `/usr/bin/freecadcmd` |
| `SHAPE_RECO_RESULTS` | the run store | `~/shape_reco_results/` |
| `SHAPE_RECO_TEST_PARTS` | the parts the window's *Random* button draws from (every STEP below it) | — |
| `SHAPE_RECO_FIELD_BACKEND` | closest-point backend: `igl`, `numpy`, `torch`, `cuda`, `auto` | `auto` |
| `SHAPE_RECO_CACHE` | build, voxelisation and prediction cache | `~/.cache/shape_reco` where enabled |
| `SHAPE_RECO_FREECAD_WORKER=1` | one persistent `freecadcmd` per process (faster for many builds) | off |

The full list, every command with its expected output and the known limits are in `docs/RUNNING.md`; the
architecture is in `docs/CODEMAP.md`, the vector-field route in `docs/VECTORFIELD.md`.

Tests: run the files of the modules you change — the whole suite takes long.

```bash
~/.local/share/shape-reco/venv/bin/python -m pytest tests/test_feature_phase.py -q
```

## Repository layout

| path | what is in it |
|---|---|
| `reco_engine/` | the main line: cascade, residual add-cut loop, feature phase, hole stacks, validation |
| `reco_field/` | the vector-field route, the FreeCAD export and the GDML export |
| `reco_gui/` | the window and the dataset-generation window |
| `dataset_gen/` | FreeCAD macros, the dataset generator, voxelisation and field channels; `examples/` holds small test parts |
| `training/` | training of the cascade's networks |
| `models/` | the stage registries (which stage uses which network or analytic fit) |
| `scripts/` | command-line tools: `run_part.py`, `peel_case.py`, `compare_runs.py`, `export_gdml.py`, the testing protocol |
| `freecad_nodes/` | the stages as nodes for FreeCAD's Nodes workbench |
| `packaging/` | the icon, the logo and the scripts that build them |
| `docs/` | `RUNNING.md` (every command), `CODEMAP.md` (architecture), `VECTORFIELD.md` |
| `tests/` | the test suite |

## Licence

GPL-3.0 (see `LICENSE`). The GUI uses PyQt5 (GPL-3.0); FreeCAD is LGPL-2.1. The skeletal formula of Batrachotoxin in
the logo is Wikipedia's (Wikimedia Commons `File:Batrachotoxin_skeletal.svg` by Ed (Edgar181), public domain).
