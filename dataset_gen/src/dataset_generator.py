#!/usr/bin/env python3
"""
FreeCAD STL Dataset Generator

This script generates STL datasets for CNN learning by executing FreeCAD macros.
It supports three dataset modes:

TYPE: CNN learns topology/shape type (varying dimensions and rotations)
MEASURE: CNN learns dimensions/proportions (fixed ratios, varying rotations)
ROTATION: CNN learns orientation (fixed rotation, varying dimensions)

The script only generates STL files. Voxelization is performed by a separate pipeline.
"""

from __future__ import annotations

import argparse
import tempfile
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Import shared utilities
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_utils import (
    DEFAULT_SCALE_MM,
    DEFAULT_SEED,
    DEFAULT_ROTATION_MIN,
    DEFAULT_ROTATION_MAX,
    TOPOLOGIES,
    MACRO_MAP,
    PARAMETER_SPECS,
    Rotation,
    Sample,
    clean_subprocess_env,
    sample_parameters,
    sample_rotation,
    build_filename,
    json_dump_compact,
)
from src.voxelization import voxelize_stl_to_bin, check_voxelization_available, voxelizer_for
from src import inscale
from src.voxel_noise import VoxelNoiseConfig
import shutil
from src import field_channels, sample_writer, solid_builder
from src import sdf_primitives as sp
from src.closest_point_field import load_field


# ============================================================================
# FREECAD ENVIRONMENT
# ============================================================================

def build_environment(
    topology: str,
    parameters: Dict[str, Any],
    rotation: Rotation,
    output_stl: Path,
    scale_mm: float,
) -> Dict[str, str]:
    """Build environment variables for FreeCAD macro execution."""
    env = clean_subprocess_env()
    
    params_json = json_dump_compact(parameters)
    rotation_json = json_dump_compact([rotation.qx, rotation.qy, rotation.qz, rotation.qw])
    
    env["DATASET_GENERATOR_MODE"] = "1"
    env["DATASET_TOPOLOGY"] = topology
    env["DATASET_PARAMS_JSON"] = params_json
    env["DATASET_AFFINE_JSON"] = params_json
    env["DATASET_AFFINE_PARAMS_JSON"] = params_json
    env["DATASET_ROTATION_JSON"] = rotation_json
    env["DATASET_ROTATION_QUATERNION_JSON"] = rotation_json
    env["DATASET_ROTATION_X"] = str(rotation.qx)
    env["DATASET_ROTATION_Y"] = str(rotation.qy)
    env["DATASET_ROTATION_Z"] = str(rotation.qz)
    env["DATASET_ROTATION_W"] = str(rotation.qw)
    env["DATASET_SCALE_MM"] = str(scale_mm)
    env["DATASET_REALIZATION_SCALE_MM"] = str(scale_mm)
    env["DATASET_OUT_STL"] = str(output_stl.resolve())
    env["DATASET_OUT_DIR"] = str(output_stl.resolve().parent)
    # elliptical_tube_gdml_dataset_learning_CNN.FCMacro reads this
    # unprefixed name instead of DATASET_OUT_STL - alias it here rather
    # than special-casing that one macro's call site.
    env["OUTPUT_STL"] = str(output_stl.resolve())
    
    # Also expose every parameter individually for compatibility
    for name, value in parameters.items():
        safe_name = "".join(
            c.upper() if c.isalnum() or c == "_" else "_" for c in name
        )
        env[f"DATASET_{safe_name}"] = str(value)
        env[f"DATASET_AFFINE_{safe_name}"] = str(value)
    
    return env


# ============================================================================
# FREECAD EXECUTION
# ============================================================================

def execute_macro(
    freecad: str,
    macro_path: Path,
    macro_dir: Path,
    topology: str,
    parameters: Dict[str, Any],
    rotation: Rotation,
    output_stl: Path,
    scale_mm: float,
    verbose: bool = False,
) -> str:
    """Execute a FreeCAD macro to generate an STL file."""
    output_stl.parent.mkdir(parents=True, exist_ok=True)
    
    env = build_environment(
        topology=topology,
        parameters=parameters,
        rotation=rotation,
        output_stl=output_stl,
        scale_mm=scale_mm,
    )
    
    macro_abs = str(macro_path.resolve())
    code = f"exec(compile(open({repr(macro_abs)}, encoding='utf-8').read(), {repr(macro_abs)}, 'exec'))"
    
    command = [freecad, "-c", code]
    
    if verbose:
        print(f"FreeCAD command: {' '.join(command)}")
        print(f"Parameters: {json.dumps(parameters, sort_keys=True, indent=2)}")
        print(f"Quaternion: {json.dumps([rotation.qx, rotation.qy, rotation.qz, rotation.qw])}")
    
    if os.environ.get("SHAPE_RECO_FREECAD_WORKER") == "1":     # one persistent freecadcmd (reco_field/freecad_worker.py)
        import types
        repo = str(Path(__file__).resolve().parent.parent.parent)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from reco_field import freecad_worker
        rc, out = freecad_worker.run_script(freecad, macro_abs, env, cwd=str(macro_dir))
        process = types.SimpleNamespace(returncode=rc, stdout=out)
    else:
        process = subprocess.run(
            command,
            cwd=str(macro_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    
    if process.returncode != 0:
        raise RuntimeError(
            f"FreeCAD failed.\n\n"
            f"Topology: {topology}\n"
            f"Macro: {macro_path}\n"
            f"Parameters: {json.dumps(parameters, indent=2)}\n"
            f"Rotation: {json.dumps(asdict(rotation), indent=2)}\n"
            f"Output: {output_stl}\n\n"
            f"{process.stdout}"
        )
    
    if not output_stl.is_file():
        raise RuntimeError(
            f"FreeCAD completed (exit 0) but did not create STL: {output_stl}\n\n"
            f"Topology: {topology}\n"
            f"Parameters: {json.dumps(parameters, indent=2)}\n"
            f"Rotation: {json.dumps(asdict(rotation), indent=2)}\n\n"
            f"{process.stdout}"
        )

    if output_stl.stat().st_size <= 0:
        raise RuntimeError(
            f"FreeCAD created an empty STL: {output_stl}\n\n"
            f"Topology: {topology}\n"
            f"Parameters: {json.dumps(parameters, indent=2)}\n"
            f"Rotation: {json.dumps(asdict(rotation), indent=2)}\n\n"
            f"{process.stdout}"
        )
    
    return process.stdout


# ============================================================================
# VOXELIZATION
# ============================================================================

def _noise_config_from_args(args: argparse.Namespace) -> VoxelNoiseConfig:
    return VoxelNoiseConfig(
        salt_prob=args.noise_salt_prob,
        pepper_prob=args.noise_pepper_prob,
        blob_count=args.noise_blob_count,
        blob_min_size=args.noise_blob_min_size,
        blob_max_size=args.noise_blob_max_size,
        dilate_prob=args.noise_dilate_prob,
        erode_prob=args.noise_erode_prob,
    )


def voxelize_sample(args: argparse.Namespace, output_stl: Path, output_root: Path,
                    rng: random.Random = None, topology: str = None, shape: Optional[Dict] = None,
                    extras: Optional[Dict] = None):
    """(voxel path relative to output_root or None, placement record or None).

    --voxel-scale-mode fit (default): as maybe_voxelize always did - no placement.
    --voxel-scale-mode fixed: the sample gets a random size and position (src/inscale.py) and is
    voxelized in scale with the world origin at the grid centre; the draw is returned for the metadata.
    With --full-field-prob, a box sample is (with that probability) the completely filled grid: a cube
    that spans the whole voxel field (then noised like every other sample). The STL next to it is then
    NOT the voxel content, so such samples serve the Type CNN only.

    `shape` ({"prim": local sdf primitive, "R": rotation}, from `make_sample_stl` when the sample was built as an exact
    program) enables the exact-solid extras, returned through the `extras` dict:
      --with-field: the MODIFIED solid (carve / cuts / hollow re-expressed in world mm, src/solid_builder.py) is built in
        FreeCAD and written as <stem>.solid.stl, the stored .bin is the (parity) voxelization of THAT solid (so voxels and
        field describe one body; placement["solid_revoxelized"]), and <stem>.field.npz holds its channels
        (src/field_channels.py) - never the field of the uncut base next to carved voxels;
      labels (dims_mm, center_mm, rotation_canonical, segment) of the placed shape;
      --emit-negative-residual: the carve cutter's voxels inside the clean base (extras["negative_residual"])."""
    if not args.voxelize or args.voxel_scale_mode == "fit":
        return maybe_voxelize(args, output_stl, output_root, rng=rng), None

    if rng is None:
        raise ValueError("voxelize_sample: --voxel-scale-mode fixed draws a size and position and needs an rng.")
    noise_config = _noise_config_from_args(args)
    noise_rng = np.random.default_rng(rng.randrange(2**32)) if noise_config.enabled else None
    if topology == "box" and args.full_field_prob > 0 and rng.random() < args.full_field_prob:
        bin_path = output_stl.with_suffix(".bin")
        volume = np.ones((args.voxel_resolution,) * 3, dtype=np.uint8)
        placement = {"full_field": True, "size_fraction": 1.0, "cuts": [], "voxels_uncut": int(volume.sum()),
                     "voxels": int(volume.sum())}
        if noise_config.enabled:
            from src.voxel_noise import apply_voxel_noise
            volume.astype(np.uint8).tofile(bin_path.with_suffix(".clean.bin"))   # the clean grid
            volume = apply_voxel_noise(volume, noise_config, noise_rng)
        if shape is not None and getattr(args, "with_field", False):
            # the full grid IS the grid cube: built as that box in FreeCAD like every other solid, fielded from its STL
            h = args.grid_mm / 2.0
            with tempfile.TemporaryDirectory(prefix="sample_") as tmp:
                cube = Path(tmp) / "solid.stl"
                solid_builder.build_stl(sp.Box([h, h, h]), cube, freecad=args.freecad)
                _write_field(args, output_stl, cube, placement)
            placement["inside"] = _write_inside(args, output_stl, bin_path)
        volume.astype(np.uint8).tofile(bin_path)
        return str(bin_path.relative_to(output_root)), placement
    # Voxelize CLEAN, then cut, then add noise: speckle and foreign blobs belong in the final grid, not on
    # the side that was cut away.

    bin_path = output_stl.with_suffix(".bin")
    ran: Dict = {}                                     # report["voxelizer"] of voxelize_stl_to_bin: the voxelizer that ran
    exact = shape is not None and getattr(args, "with_field", False)
    work = tempfile.TemporaryDirectory(prefix="sample_") if (exact or shape is not None) else None
    for draw in range(1, (6 if args.require_manifold else 1) + 1):
        bbox_center, extents = inscale.bounds_center_extents(output_stl)
        placement = inscale.sample_placement(rng, extents, args.grid_mm, args.size_fraction[0], args.size_fraction[1],
                                             position=args.position, resolution=args.voxel_resolution)
        with tempfile.TemporaryDirectory() as tmp:
            placed = Path(tmp) / "placed.stl"
            if exact:
                # every solid of a --with-field sample is a FreeCAD Part solid: the placed base too (not the numpy-scaled
                # STL of write_placed_stl), so the voxels, the solid STL and the field all come from one Part shape
                placed = Path(work.name) / "solid.stl"
                solid_builder.build_stl(solid_builder.placed_program(shape["prim"], shape["R"], placement, bbox_center),
                                        placed, freecad=args.freecad)
            else:
                inscale.write_placed_stl(output_stl, placed, placement, bbox_center)
            ok = voxelize_stl_to_bin(
                stl_path=str(placed), output_bin=str(bin_path), resolution=args.voxel_resolution,
                executable=args.stltovoxel_exe, scale_mode="fixed", scale_mm=args.grid_mm, anchor="origin",
                voxelizer=getattr(args, "voxelizer", None), report=ran,
            )
        if not ok:
            raise RuntimeError(f"Voxelization failed for: {output_stl}")

        volume = np.fromfile(bin_path, dtype=np.uint8).reshape((args.voxel_resolution,) * 3)
        base_volume = volume.copy()
        placement["voxels_uncut"] = int(volume.sum())
        carved = False
        if args.hollow_prob > 0 and topology in ("box", "sphere") and rng.random() < args.hollow_prob:
            # empty out the interior: a hollow box or a hollow ball, label = the outer (base) shape. Unlike carve/cut this
            # LEAVES a closed cavity on purpose - the opposite of the "no enclosed void" rule below - so it draws first and,
            # if it succeeds, skips carve/cut for this sample (a hollow body is not also bitten or planar-cut here).
            hollow_volume, hollow_record = inscale.hollow(volume, rng, wall_fraction=args.hollow_wall)
            if hollow_volume is None:
                placement["hollow_failed"] = True
            else:
                volume, carved = hollow_volume, True                 # `carved` also gates the cut branch below: one modification per sample
                placement["hollow"] = hollow_record
        if not carved and args.carve_prob > 0 and topology in ("box", "sphere") and rng.random() < args.carve_prob:
            # box - sphere (label box) or sphere - box (label sphere): ONE cutter, and only accepted if the
            # result is one uniform manifold (one piece, no enclosed void); otherwise the body stays uncarved
            carved_volume, carve_record = inscale.carve(
                volume, rng, "sphere" if topology == "box" else "box", size=args.carve_size, removed=args.carve_removed,
                min_thickness=args.cut_min_thickness)
            if carved_volume is None:
                placement["carve_failed"] = True
            else:
                volume, carved = carved_volume, True
                placement["carve"] = carve_record
        if not carved and args.cut_prob > 0 and rng.random() < args.cut_prob:
            cuts = inscale.sample_cuts(rng, max_cuts=args.max_cuts, keep_min=args.cut_keep[0], keep_max=args.cut_keep[1],
                                       axis_aligned_prob=args.cut_axis_aligned_prob)
            volume, applied = inscale.apply_cuts(volume, cuts, min_thickness=args.cut_min_thickness)
            placement["cuts"] = applied
        if exact and ("hollow" in placement or "carve" in placement or placement.get("cuts")):
            volume = _exact_modified_solid(args, shape, placement, bbox_center, base_volume, volume,
                                           Path(work.name) / "solid.stl", bin_path)
        placement["voxels"] = int(volume.sum())
        placement["manifold"] = inscale.manifold_report(volume)                 # of the clean part, before noise
        is_hollow = "hollow" in placement
        if not args.require_manifold or placement["manifold"]["ok"] or is_hollow:
            break
    else:
        raise RuntimeError(f"no uniform-manifold draw in 6 attempts for: {output_stl}")
    # canon (CLAUDE.md): a clean part under --min-voxels is recorded as placement.degenerate and skipped in training
    placement["degenerate"] = bool(placement["voxels"] < int(getattr(args, "min_voxels", 200)))
    placement["voxelizer"] = ran["voxelizer"]                               # the one that ran, not the env's wish
    placement["draws"] = draw
    if shape is not None and extras is not None:
        extras["labels"] = solid_builder.labels(shape["prim"], shape["R"], placement, bbox_center)
    if exact:
        _write_field(args, output_stl, Path(work.name) / "solid.stl", placement)
        placement["inside"] = _write_inside(args, output_stl, bin_path)
    if getattr(args, "emit_negative_residual", False) and extras is not None and shape is not None and "carve" in placement:
        extras["negative_residual"] = _negative_residual(args, shape, placement, bbox_center)
    if work is not None:
        work.cleanup()
    # Whether a sample is noised is decided by its own coin (noise generator, so the parameter/rotation stream is
    # untouched) and NEVER by what kind of sample it is: every class and every kind (plain, cut, carved) has noisy
    # and clean samples in the same proportion, so noise cannot become a cue for a class or for cut/carve.
    noised = noise_config.enabled and bool(noise_rng.random() < args.noise_prob)
    placement["noise"] = bool(noised)
    if noised:
        from src.voxel_noise import apply_voxel_noise
        volume.astype(np.uint8).tofile(bin_path.with_suffix(".clean.bin"))   # the clean grid (dirlog_inside's bit)
        placement["voxels_clean"] = int(volume.sum())
        volume = apply_voxel_noise(volume, noise_config, noise_rng)
    volume.astype(np.uint8).tofile(bin_path)
    return str(bin_path.relative_to(output_root)), placement


def _exact_modified_solid(args, shape, placement, bbox_center, base_volume, voxel_volume, solid_stl: Path,
                          bin_path: Path) -> np.ndarray:
    """--with-field: the carve / cuts / hollow just drawn on the voxels, re-expressed exactly in world mm, built in FreeCAD
    as one solid (written to `solid_stl`), and that solid voxelized exactly like the base was (same voxelizer). Returns the
    new clean volume, which REPLACES the voxel-space modification, so the .bin and the field are of the same body. The
    draw's acceptance (manifold, --require-manifold redraws) is then judged on this volume."""
    r, g = args.voxel_resolution, args.grid_mm
    program = solid_builder.placed_program(shape["prim"], shape["R"], placement, bbox_center)
    if "hollow" in placement:
        wall_mm = placement["hollow"]["wall_voxels"] * g / r
        inner = solid_builder.hollow_inner(program, wall_mm)
        if inner is None:                            # a segment sphere, or the wall eats the body: keep it solid
            del placement["hollow"]
            placement["hollow_failed"] = True
            return base_volume
        program = sp.Subtraction(program, inner)
        placement["hollow"]["wall_mm"] = float(wall_mm)
    elif "carve" in placement:
        program = sp.Subtraction(program, solid_builder.carve_cutter(placement["carve"], g, r))
    else:
        for half_space in solid_builder.cut_halfspaces(base_volume, placement["cuts"], g):
            program = sp.Intersection(program, half_space)
    built = solid_builder.build_stl(program, solid_stl, freecad=args.freecad)
    ok = voxelize_stl_to_bin(stl_path=str(solid_stl), output_bin=str(bin_path), resolution=r,
                             executable=args.stltovoxel_exe, scale_mode="fixed", scale_mm=g, anchor="origin",
                             voxelizer=getattr(args, "voxelizer", None))
    if not ok:
        raise RuntimeError(f"Voxelization of the exact solid failed: {solid_stl}")
    volume = np.fromfile(bin_path, dtype=np.uint8).reshape((r,) * 3)
    both = int(((volume > 0) | (voxel_volume > 0)).sum())
    placement["solid_revoxelized"] = True
    placement["solid"] = dict(volume_mm3=built["volume"], n_solids=built["n_solids"],
                              voxel_iou_vs_voxel_modification=float(((volume > 0) & (voxel_volume > 0)).sum() / max(both, 1)))
    return volume


def _write_inside(args, output_stl: Path, bin_path: Path) -> str:
    """<stem>.inside.npy (src/inside_bits.py, the user's standard base-point bit) from the PLACED solid <stem>.solid.stl
    that _write_field has just stored. Only for samples with a placed solid (--with-field): without it the record's STL is
    the unplaced macro STL and cuts/carves exist only in voxel space, so no mesh describes the body - no bits are written,
    and readers fall back to the grid."""
    from src.inside_bits import write_inside
    solid = output_stl.with_name(output_stl.stem + ".solid.stl")
    if not solid.exists():
        raise RuntimeError(f"inside bits need the placed solid, missing: {solid}")
    return write_inside(output_stl, bin_path, args.grid_mm, args.voxel_resolution)


def _write_field(args, output_stl: Path, solid_stl: Path, placement: Dict, stem: Optional[str] = None) -> Dict:
    """<stem>.field.npz (+ <stem>.solid.stl, a copy of the placed FreeCAD solid) for one sample, through the shared writer
    (src/sample_writer.write_field: the channels on the in-scale grid, world axes, + --field-points surface points)."""
    stem = stem or output_stl.stem
    keep = output_stl.with_name(stem + ".solid.stl")
    if Path(solid_stl).resolve() != keep.resolve():
        shutil.copyfile(solid_stl, keep)
    record = sample_writer.write_field(keep, output_stl.with_name(stem + ".field.npz"), args.voxel_resolution,
                                       args.grid_mm, args.field_points)
    record["solid_stl"] = keep.name
    placement["field"] = record
    return record


def _negative_residual(args, shape: Dict, placement: Dict, bbox_center) -> Optional[Dict]:
    """The carve cutter intersected with the placed base - Part.common() in FreeCAD (sp.Intersection through
    reco_field/freecad_builder.py), exactly the leftover a perfect first add leaves for this sample. Returns the STL (in a
    temporary directory, placed world mm) and the carve record, or None if FreeCAD finds nothing / it is tiny."""
    tmp = Path(tempfile.mkdtemp(prefix="negres_"))
    program = sp.Intersection(solid_builder.placed_program(shape["prim"], shape["R"], placement, bbox_center),
                              solid_builder.carve_cutter(placement["carve"], args.grid_mm, args.voxel_resolution))
    try:
        built = solid_builder.build_stl(program, tmp / "negres.stl", freecad=args.freecad)
    except Exception as exc:                                               # an empty common() - skip, the base stays
        print(f"    negative residual not built: {exc}")
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    if built["volume"] < 1e-3:
        shutil.rmtree(tmp, ignore_errors=True)
        return None
    return dict(stl=tmp / "negres.stl", carve=placement["carve"], built=built)


def write_negative_residual(args, output_path: Path, output_root: Path, index: int, mode: str, residual: Dict) -> Dict:
    """Register the negative residual of a carve sample as its OWN sample next to it, through the shared writer
    (src/sample_writer.finish_sample): <stem>_negres.stl (the FreeCAD common() of base and cutter, placed, world mm),
    <stem>_negres.bin, <stem>_negres.field.npz, and a record in <output_root>/<cutter kind>/dataset_metadata.jsonl labelled
    with the CUTTER (topology, subclass full, sample_kind "negres", dims_mm, center_mm, rotation, rotation_canonical)."""
    from src.dataset_utils import quaternion_to_euler_xyz
    stl_path = output_path.with_name(output_path.stem + "_negres.stl")
    shutil.copyfile(residual["stl"], stl_path)
    shutil.rmtree(Path(residual["stl"]).parent, ignore_errors=True)
    lab = solid_builder.carve_cutter_labels(residual["carve"], args.grid_mm, args.voxel_resolution)
    q = solid_builder.matrix_to_quat(lab["R"])
    rx, ry, rz = quaternion_to_euler_xyz(q)
    params = ({"x_half_ratio": 1.0, "y_half_ratio": 1.0, "z_half_ratio": 1.0} if lab["topology"] == "box"
              else {"radius_ratio": 1.0})                                      # proportions only; sizes are in dims_mm
    record = {
        "index": index, "mode": mode, "topology": lab["topology"], "subclass": "full", "sample_kind": "negres",
        "parameters": params, "rotation": dict(rx=rx, ry=ry, rz=rz, qx=q[0], qy=q[1], qz=q[2], qw=q[3]),
        "scale_mm": args.scale_mm, "seed": args.seed, **({"seed_offset": args.seed_offset} if args.seed_offset else {}),
        "placement": {"grid_mm": args.grid_mm, "negative_residual": True, "residual_of": index, "stl_placed": True,
                      "solid": dict(volume_mm3=residual["built"]["volume"], n_solids=residual["built"]["n_solids"])},
        "dims_mm": lab["dims_mm"], "center_mm": lab["center_mm"], "rotation_canonical": lab["rotation_canonical"],
    }
    noise_config = _noise_config_from_args(args)                               # negres pieces are noised like the rest
    return sample_writer.finish_sample(stl_path, record, output_root, resolution=args.voxel_resolution,
                                       with_field=True, field_points=args.field_points,
                                       noise=dict(config=noise_config, prob=args.noise_prob) if noise_config.enabled else None,
                                       min_voxels=int(getattr(args, "min_voxels", 200)),
                                       voxelizer=getattr(args, "voxelizer", None))


def maybe_voxelize(args: argparse.Namespace, output_stl: Path, output_root: Path,
                    rng: random.Random = None) -> Optional[str]:
    """Voxelize an STL to a .bin next to it, if --voxelize was requested.

    Returns the .bin path relative to output_root, or None if voxelization
    was skipped/disabled. Raises on failure (a bad voxelization is as fatal
    as a bad STL - it would silently corrupt training data otherwise).

    `rng` is the same seeded random.Random driving parameter/rotation
    sampling for this dataset - reused (via a derived, independent
    np.random.Generator) only when one of the --noise-* flags is set,
    so a --seed run stays fully reproducible; when noise is disabled
    (the default) this never touches rng, leaving the existing
    parameter-sampling sequence exactly as it was before noise support
    existed.
    """
    if not args.voxelize:
        return None

    bin_path = output_stl.with_suffix(".bin")

    noise_config = _noise_config_from_args(args)
    noise_rng = None
    if noise_config.enabled:
        if rng is None:
            raise ValueError("maybe_voxelize: noise is enabled but no rng was provided.")
        noise_rng = np.random.default_rng(rng.randrange(2**32))

    ok = voxelize_stl_to_bin(            # the legacy "fit" grid: stltovoxel's (voxelization.voxelizer_for warns, or
        stl_path=str(output_stl),          # raises for an explicit --voxelizer parity / SHAPE_RECO_VOXELIZER=parity)
        output_bin=str(bin_path),
        resolution=args.voxel_resolution,
        executable=args.stltovoxel_exe,
        noise_config=noise_config,
        noise_rng=noise_rng,
        voxelizer=getattr(args, "voxelizer", None),
    )

    if not ok:
        raise RuntimeError(f"Voxelization failed for: {output_stl}")

    return str(bin_path.relative_to(output_root))


# ============================================================================
# MACRO RESOLUTION
# ============================================================================

def resolve_macro(macro_dir: Path, topology: str) -> Path:
    """Resolve a topology to its FreeCAD macro file."""
    if topology not in MACRO_MAP:
        raise RuntimeError(f"No macro mapping for topology '{topology}'.")
    
    path = macro_dir / MACRO_MAP[topology]
    
    if not path.is_file():
        raise FileNotFoundError(f"Macro does not exist for topology '{topology}':\n{path}")
    
    return path


# ============================================================================
# EXACT-PROGRAM SAMPLES (--with-field / --segment-prob / --emit-negative-residual)
# ============================================================================

def uses_programs(args: argparse.Namespace) -> bool:
    """Any of the new flags: every sample is built as an exact sdf program (src/solid_builder.py) instead of by its
    macro, so its shape, modifications and labels are known exactly. Without them nothing changes."""
    return bool(getattr(args, "with_field", False) or getattr(args, "segment_prob", 0.0) > 0
                or getattr(args, "emit_negative_residual", False))


def make_sample_stl(args: argparse.Namespace, rng: random.Random, topology: str, params: Dict[str, Any],
                    rotation: Rotation, macro_path: Path, output_path: Path):
    """Build the sample's (unplaced) STL. Returns (parameters as built, shape or None). Old path: the macro, shape None.
    Program path (`uses_programs`): with probability --segment-prob a u/v segment is drawn for sphere / cylinder / cone /
    torus, and the shape is built by FreeCAD from its sdf program (a torus is whole unless a segment is drawn - the
    macro's own angle draws are dropped from the parameters, they are not what is built)."""
    if not uses_programs(args):
        execute_macro(freecad=args.freecad, macro_path=macro_path, macro_dir=args.macro_dir, topology=topology,
                      parameters=params, rotation=rotation, output_stl=output_path, scale_mm=args.scale_mm,
                      verbose=args.verbose)
        return params, None
    segment = None
    if args.segment_prob > 0 and topology in solid_builder.SEGMENT_KINDS and rng.random() < args.segment_prob:
        segment = solid_builder.sample_segment(rng, topology)
    if topology == "torus":
        params = {k: v for k, v in params.items() if not k.startswith("angle")}
    prim = solid_builder.local_primitive(topology, params, args.scale_mm, segment)
    R = solid_builder.quat_to_matrix([rotation.qx, rotation.qy, rotation.qz, rotation.qw])
    solid_builder.build_stl(solid_builder.base_program(prim, R), output_path, freecad=args.freecad)
    return params, dict(prim=prim, R=R, segment=segment)


def finish_sample(args: argparse.Namespace, rng: random.Random, topology: str, output_path: Path, output_root: Path,
                  shape: Optional[Dict]):
    """voxelize_sample plus the extras of the program path: (voxel_path, placement, extra JSONL keys, negative residual)."""
    extras: Dict[str, Any] = {}
    voxel_path, placement = voxelize_sample(args, output_path, output_root, rng=rng, topology=topology, shape=shape,
                                            extras=extras)
    extra = dict(extras.get("labels") or {})
    if shape is not None and not extra and not (placement or {}).get("full_field"):
        extra = solid_builder.labels(shape["prim"], shape["R"], None, None)        # fit mode / not voxelized: unplaced
    if shape is not None:
        extra["sample_kind"] = "segment" if shape.get("segment") else "whole"
    extra["subclass"] = subclass_of(shape, placement)
    return voxel_path, placement, extra, extras.get("negative_residual")


def _with_offset(args, extra: Dict) -> Dict:
    if args.seed_offset:
        extra = dict(extra, seed_offset=args.seed_offset)
    return extra


def subclass_of(shape: Optional[Dict], placement: Optional[Dict]) -> str:
    """The fixed subclass vocabulary (src/sample_writer.SUBCLASSES) for this generator's samples: a hollow body is
    "hollow"; a segment is "sector" when it has a u range (about the axis), "band" when it is limited in v only (sphere
    latitude planes, torus tube pie); everything else - including carved and planar-cut bodies - is "full"."""
    if placement and "hollow" in placement:
        return "hollow"
    segment = (shape or {}).get("segment") or {}
    if "u" in segment:
        return "sector"
    if "v" in segment:
        return "band"
    return "full"


# ============================================================================
# METADATA
# ============================================================================

def metadata_path(output_dir: Path) -> Path:
    """Get the path to the metadata file."""
    return output_dir / "dataset_metadata.jsonl"


def append_metadata(path: Path, sample: Sample, extra: Optional[Dict[str, Any]] = None) -> None:
    """Append a sample record to the metadata file. `extra`: further top-level keys (the exact-solid labels of
    --with-field / --segment-prob / --emit-negative-residual runs; nothing without those flags)."""
    record = {
        "index": sample.index,
        "mode": sample.mode,
        "topology": sample.topology,
        "parameters": sample.parameters,
        "rotation": asdict(sample.rotation),
        "scale_mm": sample.scale_mm,
        "stl_path": sample.stl_path,
        "seed": sample.seed,
        "voxel_path": sample.voxel_path,
    }
    if sample.placement is not None:
        record["placement"] = sample.placement
    if extra:
        record.update(extra)
    
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json_dump_compact(record) + "\n")


def write_metadata_header(output_dir: Path, args: argparse.Namespace) -> None:
    """Write a header record to the metadata file."""
    path = metadata_path(output_dir)
    
    if path.exists() and path.stat().st_size > 0:
        return
    
    header = {
        "record_type": "dataset_header",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "topologies": args.topologies,
        "count_per_type": args.count_per_type,
        "scale_mm": args.scale_mm,
        "seed": args.seed,
        **({"seed_offset": args.seed_offset} if args.seed_offset else {}),
        "rotation_min": args.rotation_min,
        "rotation_max": args.rotation_max,
        "voxel_scale_mode": args.voxel_scale_mode,
        "grid_mm": args.grid_mm if args.voxel_scale_mode == "fixed" else None,
        "size_fraction": list(args.size_fraction) if args.voxel_scale_mode == "fixed" else None,
        "position": args.position if args.voxel_scale_mode == "fixed" else None,
        "cut_prob": args.cut_prob if args.voxel_scale_mode == "fixed" else None,
        "full_field_prob": args.full_field_prob if args.voxel_scale_mode == "fixed" else None,
        "carve_prob": args.carve_prob if args.voxel_scale_mode == "fixed" else None,
        "hollow_prob": args.hollow_prob if args.voxel_scale_mode == "fixed" else None,
        "hollow_wall": list(args.hollow_wall) if args.voxel_scale_mode == "fixed" else None,
        "cut_keep": list(args.cut_keep) if args.voxel_scale_mode == "fixed" else None,
        "noise": {k: getattr(args, f"noise_{k}") for k in ("salt_prob", "pepper_prob", "blob_count", "dilate_prob", "erode_prob")},
        "parameter_specs": {
            topology: PARAMETER_SPECS[topology] for topology in args.topologies
        },
    }
    if getattr(args, "voxelize", False):                  # the voxelizer this run uses (the per-sample record has it too)
        header["voxelizer"] = voxelizer_for(*(("fixed", "origin") if args.voxel_scale_mode == "fixed" else ("fit", "center")),
                                            getattr(args, "voxelizer", None), warn=False)
    if uses_programs(args):                               # only with the new flags: old headers stay byte-identical
        header.update(with_field=bool(args.with_field), segment_prob=args.segment_prob,
                      emit_negative_residual=bool(args.emit_negative_residual), field_points=args.field_points,
                      builder="src/solid_builder.py (sdf program -> reco_field/freecad_builder.py)")
    
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json_dump_compact(header) + "\n")


# ============================================================================
# RESUME SUPPORT
# ============================================================================

def collect_existing_indices(output_dir: Path, topology: str) -> set[int]:
    """Collect existing sample indices for a topology."""
    topology_dir = output_dir / topology
    
    if not topology_dir.is_dir():
        return set()
    
    indices = set()
    prefix = topology + "_i"
    
    for path in topology_dir.glob(f"{topology}_i*.stl"):
        name = path.name
        if not name.startswith(prefix):
            continue
        
        remainder = name[len(prefix):]
        digits = []
        for char in remainder:
            if char.isdigit():
                digits.append(char)
            else:
                break
        
        if digits:
            try:
                indices.add(int("".join(digits)))
            except ValueError:
                pass
    
    return indices


# ============================================================================
# TYPE DATASET GENERATION
# ============================================================================

def generate_type_dataset(args: argparse.Namespace, rng: random.Random) -> None:
    """Generate TYPE dataset: topology fixed, ratios vary, rotation varies."""
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("TYPE DATASET")
    print("=" * 60)
    print(f"Output: {output_root}")
    print(f"Topologies: {', '.join(args.topologies)}")
    print(f"Samples per topology: {args.count_per_type}")
    print("=" * 60)
    
    write_metadata_header(output_root, args)
    
    global_index = 0
    
    if args.resume:
        existing = list(output_root.rglob("*.stl"))
        if existing:
            global_index = max(
                int(p.name.split("_i")[1].split("_")[0]) for p in existing
            ) + 1
    
    for topology in args.topologies:
        topology = topology.lower()
        topology_dir = output_root / topology
        topology_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_file = metadata_path(topology_dir)
        
        print(f"\nTopology: {topology}")
        
        generated = 0
        skipped = 0
        failed = 0

        macro_path = resolve_macro(args.macro_dir, topology)

        for _ in range(args.count_per_type):
            params = sample_parameters(topology, rng)
            rotation = sample_rotation(rng, args.rotation_min, args.rotation_max)

            filename = build_filename(
                topology=topology,
                index=global_index,
                scale_mm=args.scale_mm,
                rotation=rotation,
            )

            output_path = topology_dir / filename

            if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                print(f"  SKIP: {filename}")
                skipped += 1
                global_index += 1
                continue

            print(f"  [{global_index:06d}] {filename}")

            try:
                params_built, shape = make_sample_stl(args, rng, topology, params, rotation, macro_path, output_path)
                voxel_path, placement, extra, negres = finish_sample(args, rng, topology, output_path, output_root, shape)
            except Exception as exc:
                print(f"  FAILED: {filename}\n    {exc}")
                failed += 1
                global_index += 1
                continue

            sample = Sample(
                index=global_index,
                mode="type",
                topology=topology,
                parameters=params_built,
                rotation=rotation,
                scale_mm=args.scale_mm,
                stl_path=str(output_path.relative_to(output_root)),
                seed=args.seed,
                voxel_path=voxel_path,
                placement=placement,
            )

            append_metadata(metadata_file, sample, _with_offset(args, extra))
            if negres is not None:
                write_negative_residual(args, output_path, output_root, global_index, "type", negres)

            generated += 1
            global_index += 1

        print(f"  Generated: {generated}")
        print(f"  Skipped: {skipped}")
        if failed:
            print(f"  FAILED: {failed}")


# ============================================================================
# MEASURE DATASET GENERATION
# ============================================================================

def generate_measure_dataset(args: argparse.Namespace, rng: random.Random) -> None:
    """Generate MEASURE dataset: topology fixed, ratio fixed, rotation varies."""
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("MEASURE DATASET")
    print("=" * 60)
    print(f"Output: {output_root}")
    print(f"Topologies: {', '.join(args.topologies)}")
    print(f"Ratios per topology: {args.measure_ratios_per_type}")
    print(f"Rotations per ratio: {args.count_per_measure}")
    print("=" * 60)
    
    write_metadata_header(output_root, args)
    
    global_index = 0
    
    if args.resume:
        existing = list(output_root.rglob("*.stl"))
        if existing:
            global_index = max(
                int(p.name.split("_i")[1].split("_")[0]) for p in existing
            ) + 1
    
    for topology in args.topologies:
        topology = topology.lower()
        macro_path = resolve_macro(args.macro_dir, topology)
        topology_root = output_root / topology
        topology_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTopology: {topology}")
        
        for ratio_index in range(args.measure_ratios_per_type):
            fixed_params = sample_parameters(topology, rng)
            
            ratio_name = "_".join(
                f"{k}{v:.6f}".rstrip("0").rstrip(".") for k, v in fixed_params.items()
            )
            ratio_dir = topology_root / f"measure_{ratio_name}"
            ratio_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = metadata_path(ratio_dir)
            
            print(f"\n  Ratio: {ratio_index}")
            print(f"  Parameters: {json.dumps(fixed_params, sort_keys=True)}")
            
            for rotation_index in range(args.count_per_measure):
                rotation = sample_rotation(rng, args.rotation_min, args.rotation_max)
                
                filename = build_filename(
                    topology=topology,
                    index=global_index,
                    scale_mm=args.scale_mm,
                    rotation=rotation,
                )
                
                output_path = ratio_dir / filename
                
                if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                    print(f"    SKIP: {filename}")
                    global_index += 1
                    continue
                
                print(f"    [{global_index:06d}] {filename}")

                try:
                    fixed_params_built, shape = make_sample_stl(args, rng, topology, fixed_params, rotation, macro_path, output_path)
                    voxel_path, placement, extra, negres = finish_sample(args, rng, topology, output_path, output_root, shape)
                except Exception as exc:
                    print(f"    FAILED: {filename}\n      {exc}")
                    global_index += 1
                    continue

                sample = Sample(
                    index=global_index,
                    mode="measure",
                    topology=topology,
                    parameters=fixed_params_built,
                    rotation=rotation,
                    scale_mm=args.scale_mm,
                    stl_path=str(output_path.relative_to(output_root)),
                    seed=args.seed,
                    voxel_path=voxel_path,
                placement=placement,
                )

                append_metadata(metadata_file, sample, _with_offset(args, extra))
                if negres is not None:
                    write_negative_residual(args, output_path, output_root, global_index, "measure", negres)
                global_index += 1


# ============================================================================
# ROTATION DATASET GENERATION
# ============================================================================

def generate_rotation_dataset(args: argparse.Namespace, rng: random.Random) -> None:
    """Generate ROTATION dataset: topology fixed, rotation fixed, ratios vary."""
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("ROTATION DATASET")
    print("=" * 60)
    print(f"Output: {output_root}")
    print(f"Topologies: {', '.join(args.topologies)}")
    print(f"Fixed rotations per topology: {args.rotation_groups_per_type}")
    print(f"Ratios per rotation: {args.count_per_rotation}")
    print("\nIMPORTANT: Rotation is FIXED inside each folder.")
    print("Only dimension ratios vary.")
    print("=" * 60)
    
    write_metadata_header(output_root, args)
    
    global_index = 0
    
    if args.resume:
        existing = list(output_root.rglob("*.stl"))
        if existing:
            global_index = max(
                int(p.name.split("_i")[1].split("_")[0]) for p in existing
            ) + 1
    
    for topology in args.topologies:
        topology = topology.lower()
        macro_path = resolve_macro(args.macro_dir, topology)
        topology_root = output_root / topology
        topology_root.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTopology: {topology}")
        
        for rotation_group in range(args.rotation_groups_per_type):
            # Generate rotation ONCE for this group
            rotation = sample_rotation(rng, args.rotation_min, args.rotation_max)
            
            rotation_name = f"rx{int(rotation.rx)}_ry{int(rotation.ry)}_rz{int(rotation.rz)}"
            rotation_dir = topology_root / f"rot_{rotation_name}"
            rotation_dir.mkdir(parents=True, exist_ok=True)
            
            metadata_file = metadata_path(rotation_dir)
            
            print(f"\n  Rotation group: {rotation_group}")
            print(f"  Rotation: {rotation_name}")
            
            for _ in range(args.count_per_rotation):
                params = sample_parameters(topology, rng)
                
                filename = build_filename(
                    topology=topology,
                    index=global_index,
                    scale_mm=args.scale_mm,
                    rotation=rotation,
                )
                
                output_path = rotation_dir / filename
                
                if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                    print(f"    SKIP: {filename}")
                    global_index += 1
                    continue
                
                print(f"    [{global_index:06d}] {filename}")

                try:
                    params_built, shape = make_sample_stl(args, rng, topology, params, rotation, macro_path, output_path)
                    voxel_path, placement, extra, negres = finish_sample(args, rng, topology, output_path, output_root, shape)
                except Exception as exc:
                    print(f"    FAILED: {filename}\n      {exc}")
                    global_index += 1
                    continue

                sample = Sample(
                    index=global_index,
                    mode="rotation",
                    topology=topology,
                    parameters=params_built,
                    rotation=rotation,
                    scale_mm=args.scale_mm,
                    stl_path=str(output_path.relative_to(output_root)),
                    seed=args.seed,
                    voxel_path=voxel_path,
                placement=placement,
                )

                append_metadata(metadata_file, sample, _with_offset(args, extra))
                if negres is not None:
                    write_negative_residual(args, output_path, output_root, global_index, "rotation", negres)
                global_index += 1


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate FreeCAD STL datasets for CNN learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset modes:
  TYPE    - CNN learns topology/shape type (varying dimensions and rotations)
  MEASURE - CNN learns dimensions/proportions (fixed ratios, varying rotations)
  ROTATION - CNN learns orientation (fixed rotation, varying dimensions)

Examples:
  # Type dataset
  python3 -m src.dataset_generator --mode type --topologies box cylinder sphere \\
      --count-per-type 100 --macro-dir ./FCMacros --output-dir ./dataset/stl/type

  # Measure dataset
  python3 -m src.dataset_generator --mode measure --topologies box cylinder \\
      --measure-ratios-per-type 10 --count-per-measure 20 \\
      --macro-dir ./FCMacros --output-dir ./dataset/stl/measure

  # Rotation dataset
  python3 -m src.dataset_generator --mode rotation --topologies box cylinder \\
      --rotation-groups-per-type 10 --count-per-rotation 20 \\
      --macro-dir ./FCMacros --output-dir ./dataset/stl/rotation
        """,
    )
    
    parser.add_argument(
        "--mode",
        required=True,
        choices=["type", "measure", "rotation"],
        help="Dataset generation mode",
    )
    
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["box", "cylinder", "sphere", "torus"],
        choices=TOPOLOGIES,
        help="Topologies to generate",
    )
    
    parser.add_argument(
        "--count-per-type",
        type=int,
        default=100,
        help="Samples per topology (for TYPE mode)",
    )
    
    parser.add_argument(
        "--measure-ratios-per-type",
        type=int,
        default=10,
        help="Number of ratio sets per topology (for MEASURE mode)",
    )
    
    parser.add_argument(
        "--count-per-measure",
        type=int,
        default=20,
        help="Rotations per ratio set (for MEASURE mode)",
    )
    
    parser.add_argument(
        "--rotation-groups-per-type",
        type=int,
        default=10,
        help="Number of fixed rotation groups (for ROTATION mode)",
    )
    
    parser.add_argument(
        "--count-per-rotation",
        type=int,
        default=20,
        help="Ratios per rotation group (for ROTATION mode)",
    )
    
    parser.add_argument(
        "--macro-dir",
        required=True,
        type=Path,
        help="Directory containing FreeCAD macros",
    )
    
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Dataset output directory",
    )
    
    parser.add_argument(
        "--freecad",
        default=None,
        help="FreeCAD command-line executable (default: $SHAPE_RECO_FREECADCMD, then freecadcmd on PATH, "
             "then /usr/bin/freecadcmd; src/freecad_path.py)",
    )
    
    parser.add_argument(
        "--scale-mm",
        type=float,
        default=DEFAULT_SCALE_MM,
        help="Physical realization scale in mm",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for reproducibility",
    )
    
    parser.add_argument(
        "--rotation-min",
        type=float,
        default=DEFAULT_ROTATION_MIN,
        help="Minimum rotation angle in degrees",
    )
    
    parser.add_argument(
        "--rotation-max",
        type=float,
        default=DEFAULT_ROTATION_MAX,
        help="Maximum rotation angle in degrees",
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing STL files",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--voxelize",
        action="store_true",
        help="Also voxelize each generated STL to a .bin next to it",
    )

    parser.add_argument(
        "--voxel-resolution",
        type=int,
        default=64,
        help="Voxel grid resolution per axis (default: 64)",
    )

    parser.add_argument(
        "--stltovoxel-exe",
        default="stltovoxel",
        help="stltovoxel executable name/path",
    )

    parser.add_argument(
        "--voxelizer",
        choices=["parity", "stltovoxel"],
        default=None,
        help="parity (the default, also when $SHAPE_RECO_VOXELIZER is unset): ray parity of the placed mesh at the voxel "
             "centres of the in-scale grid (D-vox; --voxel-scale-mode fixed only). stltovoxel: the legacy voxelizer "
             "(the only one for --voxel-scale-mode fit). Recorded per sample as placement.voxelizer.",
    )

    parser.add_argument(
        "--voxel-scale-mode",
        choices=["fit", "fixed"],
        default="fit",
        help="fit (default): the bounding box is stretched onto the whole grid per axis (every existing "
             "dataset). fixed: IN-SCALE samples - each shape gets a random size (--size-fraction) and "
             "position (--position) in a grid of --grid-mm, voxelized at a fixed voxel size with the "
             "world origin at the grid centre, so the grid is only partly populated (src/inscale.py).",
    )
    parser.add_argument("--grid-mm", type=float, default=40.0,
                        help="Physical size of the voxel grid for --voxel-scale-mode fixed. Default 40.")
    parser.add_argument("--size-fraction", type=float, nargs=2, default=[0.25, 0.9], metavar=("MIN", "MAX"),
                        help="Largest extent of a shape as a fraction of the grid, drawn uniformly per "
                             "sample (--voxel-scale-mode fixed). Default 0.25 0.9.")
    parser.add_argument("--full-field-prob", type=float, default=0.0,
                        help="--voxel-scale-mode fixed: probability that a BOX sample is the completely filled "
                             "voxel field (a cube spanning the whole grid, size fraction 1), noised like the "
                             "rest. Teaches that a full grid is a cube. Type-CNN data only (the STL is not "
                             "the voxel content). Default 0 (off).")
    parser.add_argument("--hollow-prob", type=float, default=0.0,
                        help="--voxel-scale-mode fixed: probability that a BOX or SPHERE sample is emptied out into a shell "
                             "(a hollow box / hollow ball, label = the outer shape), drawn before carve/cut so a sample is at "
                             "most one of hollow/carve/cut. Leaves a closed cavity ON PURPOSE (--require-manifold does not "
                             "reject it). Default 0.0 (off).")
    parser.add_argument("--hollow-wall", type=float, nargs=2, default=[0.08, 0.3], metavar=("MIN", "MAX"),
                        help="Wall thickness of a hollow sample as a fraction of the body's half-extent. Default 0.08 0.3.")
    parser.add_argument("--carve-prob", type=float, default=0.0,
                        help="--voxel-scale-mode fixed: probability that a BOX sample is carved by one sphere "
                             "(box - sphere, label box) or a SPHERE sample by one box (sphere - box, label sphere), "
                             "only accepted when the result is one uniform manifold (one piece, no enclosed void). "
                             "A carved sample gets no planar cuts. Voxel-space only (the STL is not carved), so "
                             "for Type/Rotation-CNN data. Default 0 (off).")
    parser.add_argument("--noise-prob", type=float, default=1.0,
                        help="--voxel-scale-mode fixed with --noise-* flags: probability that a sample gets the noise at all, "
                             "drawn per sample independently of its class and of cuts/carves, so every class and kind has "
                             "noisy and clean samples in the same proportion. Default 1 (all noised, as before). "
                             "Recorded as placement.noise.")
    parser.add_argument("--min-voxels", type=int, default=200,
                        help="--voxel-scale-mode fixed: a clean part with fewer voxels is recorded as placement.degenerate "
                             "(skipped by the trainers; the project's canon, CLAUDE.md). Default 200.")
    parser.add_argument("--require-manifold", action="store_true",
                        help="--voxel-scale-mode fixed: redraw size/position/cuts/carve (up to 6 times) until the CLEAN "
                             "part is one uniform manifold (one piece, no enclosed void); the sample fails otherwise.")
    parser.add_argument("--carve-size", type=float, nargs=2, default=[0.18, 0.42], metavar=("MIN", "MAX"),
                        help="Cutter radius / half edge as a fraction of the body's largest extent. Default 0.18 0.42.")
    parser.add_argument("--carve-removed", type=float, nargs=2, default=[0.04, 0.5], metavar=("MIN", "MAX"),
                        help="Accepted range of the fraction of the body a carve removes. Default 0.04 0.5.")
    parser.add_argument("--cut-prob", type=float, default=0.0,
                        help="--voxel-scale-mode fixed: probability that a sample is CUT by one or two random "
                             "planes (a sliced sphere/box; its label stays the base shape). Default 0 (off).")
    parser.add_argument("--cut-keep", type=float, nargs=2, default=[0.35, 0.95], metavar=("MIN", "MAX"),
                        help="Fraction of the shape's extent along the cut normal that is kept, drawn per cut. "
                             "Default 0.35 0.95.")
    parser.add_argument("--cut-min-thickness", type=float, default=4.0,
                        help="A cut is skipped if it would leave a plate thinner than this many voxels along its "
                             "thinnest principal axis (a triangle or layer in one plane). Default 4; 0 = off.")
    parser.add_argument("--max-cuts", type=int, default=2, help="At most this many cuts per sample. Default 2.")
    parser.add_argument("--cut-axis-aligned-prob", type=float, default=0.5,
                        help="Probability that a cut plane is axis-aligned (else a random orientation). Default 0.5.")
    parser.add_argument("--position", choices=["random", "centered"], default="random",
                        help="Where a shape sits in the grid (--voxel-scale-mode fixed): random (anywhere it "
                             "fits entirely) or centered. Default random.")
    parser.add_argument(
        "--noise-salt-prob",
        type=float,
        default=0.0,
        help="Voxel-grid noise (applied in the voxelization layer, after --voxelize): "
             "fraction of EMPTY voxels randomly flipped to occupied. Default 0 (off).",
    )
    parser.add_argument(
        "--noise-pepper-prob",
        type=float,
        default=0.0,
        help="Fraction of OCCUPIED voxels randomly flipped to empty. Default 0 (off).",
    )
    parser.add_argument(
        "--noise-blob-count",
        type=int,
        default=0,
        help="Number of small solid foreign-object cubes spliced into each voxel grid "
             "(simulates neighboring debris/objects, not just isolated speckle). Default 0 (off).",
    )
    parser.add_argument(
        "--noise-blob-min-size",
        type=int,
        default=2,
        help="Minimum blob cube side length in voxels (--noise-blob-count > 0).",
    )
    parser.add_argument(
        "--noise-blob-max-size",
        type=int,
        default=6,
        help="Maximum blob cube side length in voxels (--noise-blob-count > 0).",
    )
    parser.add_argument(
        "--noise-dilate-prob",
        type=float,
        default=0.0,
        help="Probability of one 6-neighborhood dilation pass (thickens surfaces slightly). "
             "Default 0 (off).",
    )
    parser.add_argument(
        "--noise-erode-prob",
        type=float,
        default=0.0,
        help="Probability of one 6-neighborhood erosion pass (thins surfaces slightly). "
             "Default 0 (off).",
    )

    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Split selector for disjoint random streams with the same --seed (e.g. 0 train, 1 val, 2 test). "
                             "0 (default) is the old stream exactly; recorded in the header and in every record's seed_offset.")
    parser.add_argument("--kinds", nargs="+", default=None, choices=list(solid_builder.KINDS),
                        help="The five whole kinds of the cascade (box sphere cylinder cone torus): the same as --topologies, "
                             "restricted to the kinds that have an exact sdf program (src/solid_builder.py). Overrides "
                             "--topologies when given.")
    parser.add_argument("--with-field", action="store_true",
                        help="--voxelize --voxel-scale-mode fixed only: build every sample as an exact program in FreeCAD "
                             "(src/solid_builder.py), build carved / cut / hollow samples as the MODIFIED solid (<stem>.solid.stl) "
                             "and store its stltovoxel voxelization as the .bin, and write <stem>.field.npz: occupancy, sdf "
                             "(positive inside, parity sign), V = c(x) - x on the in-scale grid (world axes, mm) + "
                             "--field-points surface points with normals + the grid spec (src/field_channels.py). Adds the "
                             "labels dims_mm, center_mm, rotation_canonical to the JSONL.")
    parser.add_argument("--field-points", type=int, default=2048, help="--with-field: surface points per sample. Default 2048.")
    parser.add_argument("--segment-prob", type=float, default=0.0,
                        help="Probability that a sphere / cylinder / cone / torus sample is a u/v SEGMENT (a sector about the "
                             "axis; sphere latitude planes; a pie of the torus tube), built with FreeCAD Part through "
                             "reco_field/freecad_builder.py (docs/freecad_part_shapes.md conventions: sector = sweep u1-u0 "
                             "turned by u0, torus v mirrored). Label: the base kind + JSONL `segment` {u, v} (radians). "
                             "Any value > 0 builds every sample as an exact program. Default 0 (off).")
    parser.add_argument("--emit-negative-residual", action="store_true",
                        help="--voxel-scale-mode fixed with --carve-prob: for every carved sample also write the cutter "
                             "intersected with the clean base as its own sample (<stem>_negres.bin/.stl/.field.npz), labelled "
                             "with the CUTTER's kind, pose and size (sample_kind negative_residual) - what a perfect first add "
                             "leaves for the residual loop, encoded like the loop encodes it (field_channels.voxel_channels).")

    parser.add_argument(
        "--export-slurm-job",
        type=Path,
        default=None,
        help=(
            "Write a SLURM array-job script (one array task per topology) "
            "that reproduces this exact command on a cluster, then exit "
            "without generating anything locally."
        ),
    )

    parser.add_argument(
        "--slurm-job-name",
        default="shape_reco_dataset",
        help="Job name for --export-slurm-job",
    )

    parser.add_argument(
        "--slurm-time",
        default="04:00:00",
        help="Walltime (HH:MM:SS) per array task for --export-slurm-job",
    )

    parser.add_argument(
        "--slurm-mem",
        default="8G",
        help="Memory per array task for --export-slurm-job",
    )

    parser.add_argument(
        "--slurm-cpus",
        type=int,
        default=4,
        help="CPUs per array task for --export-slurm-job",
    )

    parser.add_argument(
        "--slurm-partition",
        default=None,
        help="SLURM partition/queue name for --export-slurm-job (optional)",
    )

    args = parser.parse_args()
    from src.freecad_path import freecadcmd_or_default
    args.freecad = (shutil.which(args.freecad) or args.freecad) if args.freecad else freecadcmd_or_default()
    if args.kinds:
        args.topologies = list(args.kinds)
    if uses_programs(args):
        bad = [t for t in args.topologies if t not in solid_builder.KINDS]
        if bad:
            parser.error(f"--with-field / --segment-prob / --emit-negative-residual build exact programs, available for "
                         f"{', '.join(solid_builder.KINDS)} only - not {', '.join(bad)}")
    if (args.with_field or args.emit_negative_residual) and not (args.voxelize and args.voxel_scale_mode == "fixed"):
        parser.error("--with-field / --emit-negative-residual need --voxelize --voxel-scale-mode fixed (the in-scale grid)")
    if not 0.0 <= args.segment_prob <= 1.0:
        parser.error("--segment-prob must be in [0, 1]")
    return args


# ============================================================================
# CLUSTER (SLURM) EXPORT
# ============================================================================

def mode_count_args(args: argparse.Namespace) -> List[str]:
    """CLI flags that carry the per-mode sample counts."""
    if args.mode == "type":
        return ["--count-per-type", str(args.count_per_type)]
    if args.mode == "measure":
        return [
            "--measure-ratios-per-type", str(args.measure_ratios_per_type),
            "--count-per-measure", str(args.count_per_measure),
        ]
    return [
        "--rotation-groups-per-type", str(args.rotation_groups_per_type),
        "--count-per-rotation", str(args.count_per_rotation),
    ]


def write_slurm_script(args: argparse.Namespace, script_path: Path) -> None:
    """Write a SLURM array-job script: one array task per topology.

    Each task runs this same CLI, restricted to one topology, with
    --resume so a partially-completed/re-run job is safe. Designed to be
    submitted from the repo root on a cluster login node with FreeCAD and
    stltovoxel available in PATH (or via `module load`).
    """
    topologies = list(args.topologies)
    topologies_bash = " ".join(topologies)

    voxel_args = ""
    if args.voxelize:
        voxel_args = (
            f" --voxelize --voxel-resolution {args.voxel_resolution}"
            f" --stltovoxel-exe {args.stltovoxel_exe}"
        )
        if getattr(args, "voxelizer", None):
            voxel_args += f" --voxelizer {args.voxelizer}"

    count_args = " ".join(mode_count_args(args))

    partition_line = (
        f"#SBATCH --partition={args.slurm_partition}\n" if args.slurm_partition else ""
    )

    script = f"""#!/bin/bash
#SBATCH --job-name={args.slurm_job_name}
#SBATCH --array=0-{len(topologies) - 1}
#SBATCH --time={args.slurm_time}
#SBATCH --mem={args.slurm_mem}
#SBATCH --cpus-per-task={args.slurm_cpus}
#SBATCH --output={args.slurm_job_name}_%A_%a.out
{partition_line}
# Auto-generated by dataset_generator.py --export-slurm-job.
# One array task per topology; safe to resubmit (uses --resume).
#
# Expects: freecadcmd and stltovoxel available in PATH (module load / venv),
# and this script run from the directory containing the `src/` package
# (i.e. dataset_gen/).

set -euo pipefail

TOPOLOGIES=({topologies_bash})
TOPOLOGY="${{TOPOLOGIES[$SLURM_ARRAY_TASK_ID]}}"

echo "Array task $SLURM_ARRAY_TASK_ID -> topology: $TOPOLOGY"

python3 -m src.dataset_generator \\
    --mode {args.mode} \\
    --topologies "$TOPOLOGY" \\
    {count_args} \\
    --macro-dir {args.macro_dir} \\
    --output-dir {args.output_dir} \\
    --freecad {args.freecad} \\
    --scale-mm {args.scale_mm} \\
    --seed {args.seed} \\
    --rotation-min {args.rotation_min} \\
    --rotation-max {args.rotation_max} \\
    --resume{voxel_args}
"""

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    script_path.chmod(0o755)

    print(f"SLURM job script written: {script_path}")
    print(f"Array size: {len(topologies)} (one task per topology)")
    print(f"Submit with: sbatch {script_path}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Main entry point."""
    args = parse_args()

    if args.export_slurm_job is not None:
        # Cluster export doesn't touch FreeCAD/stltovoxel locally - only
        # macro-dir needs to exist (it's referenced, not executed, here).
        if not args.macro_dir.is_dir():
            raise RuntimeError(f"Macro directory does not exist: {args.macro_dir}")
        write_slurm_script(args, args.export_slurm_job)
        return

    # Validate inputs
    if not args.macro_dir.is_dir():
        raise RuntimeError(f"Macro directory does not exist: {args.macro_dir}")

    if not os.path.isfile(args.freecad):
        raise RuntimeError(f"FreeCAD executable not found: {args.freecad}")

    if args.voxelize:
        # the voxelizer this run uses (raises now, not after the first FreeCAD build, for --voxelizer parity in fit mode)
        mode, anchor = ("fixed", "origin") if args.voxel_scale_mode == "fixed" else ("fit", "center")
        used = voxelizer_for(mode, anchor, args.voxelizer)
        print(f"Voxelizer: {used}")
        if used == "stltovoxel" and not check_voxelization_available(args.stltovoxel_exe):
            raise RuntimeError(
                f"--voxelize with the stltovoxel voxelizer but the executable "
                f"'{args.stltovoxel_exe}' was not found (or Pillow/numpy are "
                f"missing). Install stltovoxel or drop --voxelize."
            )

    # Initialize random number generator
    # --seed-offset 0: exactly the old stream. Otherwise a string seed (hashed by random.Random, version 2) keyed by both
    # numbers, so train / val / test runs with the same --seed and different offsets draw unrelated streams - unlike
    # seed + offset, where seed 12345 offset 1 would repeat the stream of seed 12346 offset 0.
    rng = random.Random(args.seed) if args.seed_offset == 0 else random.Random(f"{args.seed}:{args.seed_offset}")
    
    print("\n" + "=" * 60)
    print("FREECAD STL DATASET GENERATOR")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"FreeCAD: {args.freecad}")
    print(f"Macro directory: {args.macro_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Scale: {args.scale_mm} mm")
    print("=" * 60)
    
    # Dispatch to appropriate generator
    if args.mode == "type":
        generate_type_dataset(args, rng)
    elif args.mode == "measure":
        generate_measure_dataset(args, rng)
    elif args.mode == "rotation":
        generate_rotation_dataset(args, rng)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
