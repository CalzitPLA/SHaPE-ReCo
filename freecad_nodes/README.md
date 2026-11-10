# shape_reco nodes — usage

## Install

```bash
bash freecad_nodes/install.sh
```

Then in FreeCAD: switch to the **Nodes** workbench, create a new node
editor document, right-click the canvas → the nodes below are under
the **Shape Reco** category. `Load Mesh` / the built-in `MViewer` are
useful for visualization but are separate categories (**Shape Reco**
and **Viz**).

Seven nodes remain (2026-09-30): the main line's `Cascade Reconstruct`, the vector-field route's `Peel (Vector
Field)`, `Export Program` and `Field Grade`, and `Load Mesh`, `Load Shape`, `Export STL`. The voxel nodes of the
removed route C (Load Models, Voxelize STL, the three CNN nodes, Place Candidate, Local Position Search, the consensus
nodes, Reconstruct Loop / Ensemble Best-Fit / Reconstruct + Cut Voids and their live variants) are in git history.

**Slow nodes do not run automatically.** `Cascade Reconstruct`, `Peel (Vector Field)` and `Export Program` carry a
**▶ Run** button in the node body: the Nodes workbench evaluates every node on file open (and on undo/redo), and these
run for seconds to minutes.

### After pulling new nodes: re-install and check

The workbench runs the COPY that `install.sh` made under `~/.local/share/FreeCAD/Mod/Nodes/nodes/shape_reco`, not the repo. A graph saved with newer nodes than the
installed copy fails to open, with errors that do not name the cause: `TypeError: %d format: a real number is required, not str` (a node is not registered - the
workbench's own message is broken), `FCNSocketModel.__init__() got an unexpected keyword argument 'socket_type'` (the graph has more input sockets than the
installed node class) and then `KeyError: <id>` in `Edge.deserialize` (the wires of those nodes). Re-run `bash freecad_nodes/install.sh`, restart FreeCAD, and check:

```bash
QT_QPA_PLATFORM=offscreen freecadcmd scripts/check_nodes_install.py     # INSTALL current / REGISTERED n / LOADED <graph> nodes a/a edges b/b OK
```

## Main line: `Cascade Reconstruct`

`Input STL` (an STL or STEP path), optional `Top K`, `Max Ops`, `Registry` (default `models/registry.json`) and
`Op Guard` -> `Program` (the result schema as JSON text), `Residual Pos`, `Residual Neg` (voxels still missing /
invented) and `Summary`. The logic is `reco_engine/cascade_node_support.py` (tested without FreeCAD,
`tests/test_cascade_node.py`). `freecad_nodes/examples/example_cascade.json`: Text nodes -> `Load Mesh` of the input
and `Cascade Reconstruct`; click Run on the cascade node.

## Vector-field route (route A): no voxels, no networks, no checkpoints

Three nodes wrap `reco_field` (logic in `reco_field/node_support.py`, tested without FreeCAD): the mesh of an STL or STEP file is fitted with sphere / box
/ cylinder / cone / torus / plane (the round ones with their sweep angle `u`, sphere and torus also `v`) and a peeling loop returns a CSG program; the
program is built in FreeCAD as a real solid and graded in millimetres. See `docs/vectorfield_reconstruction.tex`.

- **`Peel (Vector Field)`** (Run button; a few seconds) - `Input STL` (a .step is tessellated by `freecadcmd`), optional `Max Ops` (8) and `Target`
  (0.985, the share of the surface to explain) -> `Program` (the primitive tree as JSON text), `Explained` (0..1) and `Summary` (one line per operation,
  with the fitted parameters).
- **`Export Program`** (Run button; shells out to `freecadcmd`) - `Program`, `Workdir` and optional `Name` -> `STEP Path` (the exact B-rep), `STL Path` (feed it
  to `Load Mesh`) and `Summary` (volume, validity, faces). The Part conventions are applied: box corner, sector start, plane-limited sphere latitude, the
  mirrored torus `v` (`docs/freecad_part_shapes.tex`). A sphere with a cone-limited latitude, the ellipsoid and the convex polyhedron are refused with a
  clear error.
- **`Field Grade`** (no Run button) - `Result STL`, `Input STL`, optional `Points` (3000 per direction) -> `Chamfer` (mean, mm) and `Summary` (mean, maximum,
  each direction). Large input->result: the model misses geometry; large result->input: it invents geometry.

`freecad_nodes/examples/example_vector_field.json` (12 nodes, 11 edges): Text (STL, Max Ops, Target) -> `Peel (Vector Field)` -> `Export Program` (+ Workdir, Name)
-> `Load Mesh` of the result and `Field Grade` against the input; a second `Load Mesh` shows the input. Click Run on `Peel`, then on `Export Program`. `python3 scripts/make_node_graph_figure.py` draws a graph as a figure
(`docs/figures/vf_nodes.pdf`).
