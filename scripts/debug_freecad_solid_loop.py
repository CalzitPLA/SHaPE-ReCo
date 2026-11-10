#!/usr/bin/env python3
"""
Debug loop: drive `reco_field.peel.peel_steps` (the real pipeline: predict a primitive, cut/add it, repeat) but
materialize every accepted step as a REAL FreeCAD solid (`reco_field.export.build_split_in_freecad`), not only the
final symbolic SDF tree - so a step is checked (Part's own `isValid`) and, if the boolean actually split the model
into more than one disconnected solid, every piece is kept (not just the biggest, which `peel.py`'s own symbolic
SDF model never notices splitting at all - it just keeps evaluating a tree that happens to describe two disjoint
regions).

    freecadcmd scripts/debug_freecad_solid_loop.py [part.stl] [--max-ops 5]

For each body of the INPUT itself (split first, in case the input file already holds more than one - the same
Trimesh.split connectivity check `reco_field.peel._mesh_clusters` uses) the loop runs up to `--max-ops` steps.
Whatever splits off at the FIRST step that actually produces more than one real solid (the base itself never can -
Box/Sphere/Cylinder/Cone/Torus are each simply connected - it is always a later cut disconnecting a chunk of
material) is kept as "the first split's leftovers" and, once every body's main loop is done, fed back in as its
OWN fresh input to the same process - the small, real fragments a symbolic-only model would silently keep
attached to (or silently omit from) the main tree forever.

Needs FreeCAD's own Python (freecadcmd) for `reco_field.export.build_split_in_freecad`'s subprocess calls to work,
and the project's own modules (numpy/trimesh/reco_field/src) importable from it - both already true on this
machine (freecadcmd 1.0 ships Python 3.13 with numpy/trimesh, and simply adding the project root to sys.path is
enough, see the session's own check)."""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from reco_field.export import build_split_in_freecad, mesh_chamfer  # noqa: E402
from reco_field.peel import mesh_from_sdf, peel_steps  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402

FRAME = Path.home() / "shape_reco_parts/benchmark_parts/1.0/Unbenannt-Frame_4_x_6.stl"


def quick_voxelize(field: SurfaceField, resolution: int = 64):
    """A binary occupancy grid (not the training pipeline's exact `stltovoxel` convention, just a fast, honest
    fit-to-bbox voxelization for this debug loop's own record - `docs/FRAME_FAILURE.md`'s `coarse_genus` uses the
    same trimesh call). Returns (occupied_fraction, grid_shape)."""
    extent = float((field.bounds[1] - field.bounds[0]).max())
    pitch = extent / resolution
    mesh = trimesh.Trimesh(vertices=field.vertices, faces=field.faces, process=False)
    try:
        occ = mesh.voxelized(pitch).fill().matrix
    except Exception as e:                                        # not closed, or too small: report, don't crash the loop
        return None, str(e)
    return float(occ.mean()), occ.shape


def reconstruct_body(vertices, faces, label: str, out_root: Path, max_ops: int = 5) -> list:
    """Runs peel_steps (max_ops) on this body, materializing every accepted step in FreeCAD. Returns the STL paths
    of whatever split off the FIRST accepted step, beyond the biggest piece ("the first step's leftovers")."""
    field = SurfaceField(vertices, faces)
    print(f"\n=== {label}: {len(faces)} faces, extent {float((field.bounds[1]-field.bounds[0]).max()):.1f} ===", flush=True)
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)
    first_step_leftovers: list = []
    n_accepted = 0
    for step in peel_steps(field, max_ops=max_ops, grade=False):
        if step.stage not in ("base", "accepted"):
            continue
        n_accepted += 1
        op = step.ops[-1]
        name = f"step{n_accepted - 1}_{op.op}_{op.fit.kind}"
        ex = build_split_in_freecad(step.model, out_dir, name)
        print(f"[{label}] {name}: explained {step.explained:.1%}  FreeCAD: valid={ex.valid} n_solids={ex.n_solids} "
              f"volume={ex.volume:.3g}", flush=True)
        if ex.n_solids > 1:
            vols = [round(s['volume'], 3) for s in ex.solids]
            print(f"           split into {ex.n_solids} solids, volumes {vols} - keeping the biggest, "
                  f"the rest are real leftovers", flush=True)
            if not first_step_leftovers:                 # the FIRST step at which a real split happens, not step 0
                first_step_leftovers = [s["stl"] for s in ex.solids[1:]]
        biggest_field = load_field(ex.stl)                                    # "remesh the biggest"
        frac, shape = quick_voxelize(biggest_field)                           # "voxelize"
        if frac is not None:
            print(f"           voxel occupancy {frac:.1%} of a {shape} grid", flush=True)
        # sanity: the real FreeCAD solid should agree with the symbolic SDF tree it was built from (not with the
        # input - a cut's real solid is expected to differ from the whole input, that's the point of the cut)
        sdf_v, sdf_f = mesh_from_sdf(step.model)
        c = mesh_chamfer(biggest_field, SurfaceField(sdf_v, sdf_f))
        print(f"           real solid vs. its own symbolic SDF tree: {c['mean']:.4g} mm (mean), {c['max']:.4g} (max)", flush=True)
        if n_accepted >= max_ops:
            break
    return first_step_leftovers


def input_bodies(path: Path):
    field = load_field(path)
    mesh = trimesh.Trimesh(vertices=field.vertices, faces=field.faces, process=False)
    bodies = mesh.split(only_watertight=False)
    print(f"input {path.name}: {len(bodies)} connected component(s)" + (" - splitting" if len(bodies) > 1 else ""))
    return [(b.vertices, b.faces) for b in bodies]


def main():
    args = sys.argv[1:]
    max_ops = 5
    if "--max-ops" in args:
        i = args.index("--max-ops")
        max_ops = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    part = Path(args[0]) if args else FRAME
    out_root = Path(tempfile.mkdtemp(prefix="debug_freecad_loop_"))
    print(f"output: {out_root}")

    all_leftovers = []
    for bi, (v, f) in enumerate(input_bodies(part)):
        leftovers = reconstruct_body(v, f, f"body{bi}", out_root, max_ops)
        all_leftovers += [(f"body{bi}_leftover{i}", stl) for i, stl in enumerate(leftovers)]

    for label, stl in all_leftovers:
        field = load_field(stl)
        reconstruct_body(field.vertices, field.faces, label, out_root, max_ops)

    print(f"\ndone, {len(all_leftovers)} first-step leftover(s) looped, output under {out_root}")


if __name__ == "__main__":
    main()
