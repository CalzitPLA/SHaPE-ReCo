#!/usr/bin/env python3
"""
Build a FreeCAD primitive STL from a predicted (topology, params,
quaternion) triple.

Deliberately reuses dataset_generator.py's execute_macro/resolve_macro
and the same FCMacros/*.FCMacro files used to *generate* training data -
they already implement exactly "topology + params + quaternion -> STL"
via the DATASET_* environment-variable interface, so reconstruction
uses the identical code path as data generation instead of a second,
possibly-diverging implementation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
DATASET_GEN_ROOT = SANDBOX_ROOT / "dataset_gen"

if str(DATASET_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_GEN_ROOT))

from src.dataset_generator import execute_macro, resolve_macro  # noqa: E402
from src.dataset_utils import Rotation, quaternion_to_euler_xyz  # noqa: E402
from src.freecad_path import freecadcmd_or_default  # noqa: E402
from reco_engine.placement import array_to_world, array_to_world_position  # noqa: E402

DEFAULT_MACRO_DIR = DATASET_GEN_ROOT / "FCMacros"
ASSEMBLE_MACRO = DEFAULT_MACRO_DIR / "assemble_reconstruction.FCMacro"


def build_primitive_stl(
    topology: str,
    parameters: Dict[str, float],
    quaternion: List[float],
    output_stl: Path,
    scale_mm: float = 10.0,
    macro_dir: Path = DEFAULT_MACRO_DIR,
    freecad: Optional[str] = None,
) -> Path:
    """Realize a predicted primitive as an STL file. Returns output_stl. `freecad`: None - src.freecad_path's rule."""
    freecad = freecad or freecadcmd_or_default()
    qx, qy, qz, qw = quaternion
    rx, ry, rz = quaternion_to_euler_xyz([qx, qy, qz, qw])

    rotation = Rotation(rx=rx, ry=ry, rz=rz, qx=qx, qy=qy, qz=qz, qw=qw)
    macro_path = resolve_macro(macro_dir, topology)

    execute_macro(
        freecad=freecad,
        macro_path=macro_path,
        macro_dir=macro_dir,
        topology=topology,
        parameters=parameters,
        rotation=rotation,
        output_stl=Path(output_stl),
        scale_mm=scale_mm,
    )
    if topology in OFF_CENTRE_TOPOLOGIES:
        _recentre_stl(Path(output_stl))

    return Path(output_stl)


# Macros that build the body with its BASE at the origin (FreeCAD's Part::Cylinder runs from z = 0 to its height and is rotated about
# the origin), so the body's centre is up to half its length away from the origin. The in-scale voxelizer anchors the origin at the grid
# centre and clips at the grid, which cut a long cylinder in half; box and sphere are centred already.
OFF_CENTRE_TOPOLOGIES = ("cylinder",)


def _recentre_stl(path: Path) -> None:
    """Move the mesh so its bounding-box centre lies on the origin (only the position changes; size and orientation are kept)."""
    try:
        import trimesh
    except ImportError:                                  # a Python without trimesh (FreeCAD's own): leave the file as built
        return
    mesh = trimesh.load(str(path))
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.export(str(path))


def _manifest_entry(primitive, voxel_size_mm: float) -> Dict:
    """A PrimitiveResult (or anything with the same attributes) -> the
    plain dict assemble_reconstruction.FCMacro reads: topology +
    parameters + quaternion (everything needed to build the same Part
    solid build_primitive_stl would have built), plus where it ended up
    and how big it has to be.

    Position and size come from the voxel-space fit, in FreeCAD XYZ
    order (see placement.array_to_world): position_mm is the accepted
    footprint's center, extent_mm its bounding-box size - the footprint
    the fit score was actually computed on, since placement.place_candidate
    stretches every candidate to its target's bounding box, so the
    dimension CNN's own absolute size is not what was validated. The
    assembler stretches the built solid to extent_mm. Without extent_vox
    (a primitive from an older code path) it falls back to the local
    search's uniform scale factor.
    """
    entry = {
        "topology": primitive.topology,
        "parameters": dict(primitive.parameters),
        "quaternion": [float(v) for v in primitive.quaternion],
        "position_mm": array_to_world_position(primitive.position_mm),
        "scale": 1.0,
    }
    extent_vox = list(getattr(primitive, "extent_vox", None) or [])
    if extent_vox:
        entry["extent_mm"] = array_to_world([e * voxel_size_mm for e in extent_vox])
    else:
        placement = getattr(primitive, "placement", None)
        if placement is not None:
            entry["scale"] = float(placement.scale)
    return entry


def assemble_reconstruction_stl(
    added: Sequence,
    output_stl: Path,
    cut: Sequence = (),
    macro_dir: Path = DEFAULT_MACRO_DIR,
    freecad: Optional[str] = None,
    scale_mm: float = 10.0,
    resolution: int = 64,
) -> Path:
    """Combine accepted primitives (PrimitiveResult-like:
    `.topology`, `.parameters`, `.quaternion`, `.position_mm`,
    optionally `.placement.scale`) into ONE output STL, by building
    each as a real Part solid (Part.makeBox/makeCylinder/makeCone/
    makeSphere/makeTorus - the exact formulas each per-topology
    FCMacro itself uses) at its real accepted position/scale, fusing
    the `added` ones together and cutting the `cut` ones out with a
    real Part boolean - not a mesh reimport/reconstruction and not a
    mesh-level boolean (see assemble_reconstruction.FCMacro's module
    docstring: both were tried first, against real reconstructed
    primitives, and both silently produced geometrically wrong volumes
    without raising).

    Does not run any prediction or recognition itself - `added`/`cut`
    must already be the accepted output of a reconstruction run (e.g.
    reco_engine.pipeline.reconstruct's `.primitives`, or
    reco_engine.void_pipeline.reconstruct_with_voids's `.added`/`.cut`).
    """
    if not added:
        raise ValueError("assemble_reconstruction_stl: no accepted 'add' primitives to assemble.")
    freecad = freecad or freecadcmd_or_default()

    output_stl = Path(output_stl)
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    voxel_size_mm = scale_mm / resolution

    manifest = {
        "add": [_manifest_entry(p, voxel_size_mm) for p in added],
        "cut": [_manifest_entry(p, voxel_size_mm) for p in cut],
        "output_stl": str(output_stl.resolve()),
        "scale_mm": scale_mm,
    }

    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = Path(tmp) / "assemble_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        macro_abs = str(ASSEMBLE_MACRO.resolve())
        code = f"exec(compile(open({repr(macro_abs)}, encoding='utf-8').read(), {repr(macro_abs)}, 'exec'))"
        env = {"ASSEMBLE_MANIFEST_JSON": str(manifest_path)}

        process = subprocess.run(
            [freecad, "-c", code],
            cwd=str(macro_dir),
            env={**os.environ, **env},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"assemble_reconstruction_stl: FreeCAD failed.\n\n"
            f"Manifest: {json.dumps(manifest, indent=2)}\n\n{process.stdout}"
        )
    if not output_stl.is_file() or output_stl.stat().st_size <= 0:
        raise RuntimeError(
            f"assemble_reconstruction_stl: FreeCAD completed (exit 0) but did not "
            f"create a non-empty STL: {output_stl}\n\n{process.stdout}"
        )

    return output_stl
