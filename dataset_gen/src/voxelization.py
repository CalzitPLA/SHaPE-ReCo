#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voxelization Module - STL to .bin, by ray parity (the default) or by the stltovoxel CLI (legacy)

Ported from FCMacros/src/voxelization.py (the more robust of two prior
voxelizer scripts): fixed-resolution, validated output, matching the
resolution^3 uint8 tensor contract used throughout the CNN pipeline.

Output convention: each voxel is 0 (empty) or 1 (occupied), NOT 0/255,
in the array order [z, -y, x] (field_channels.world_to_bin).

Which voxelizer (`voxelizer_for`; T-19, 2026-10-01): "parity" is the default - scanline ray parity of the placed mesh at
the voxel centres of the exact in-scale grid (D-vox, CLAUDE.md), the grid of the sample's .inside.npy / .field.npz and of
the reconstruction loop. It exists for the in-scale grid only (scale_mode "fixed", anchor "origin"). "stltovoxel" stays
selectable as legacy: the argument `voxelizer="stltovoxel"`, env SHAPE_RECO_VOXELIZER=stltovoxel, or dataset_generator.py
--voxelizer stltovoxel. The legacy grids (fit, uniform, fixed with anchor center) exist only as stltovoxel's: an explicit
request for parity there raises; with nothing requested they fall back to stltovoxel with a LegacyVoxelizerWarning.
`voxelize_stl_to_bin(..., report={})` tells the caller which one ran (what the generators record in placement.voxelizer).
The legacy stltovoxel grid of the in-scale mode rounds its anchor to whole voxels, so it sits up to half a voxel off
the exact grid by a per-sample amount (measured on v5 small: IoU 0.84 median against the exact solid, 0 on a
26-voxel torus; docs/training_plan_v5.tex, F7). Existing stltovoxel sets: scripts/revoxelize_parity.py.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset_utils import clean_subprocess_env  # noqa: E402

try:
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

DEBUG = False

def log_debug(msg: str):
    if DEBUG:
        print(f"[VOX-DEBUG] {msg}")


VOXELIZERS = ("parity", "stltovoxel")
DEFAULT_VOXELIZER = "parity"          # D-vox (2026-09-23), the default since T-19 (2026-10-01); stltovoxel is legacy
# the modules whose code makes each voxelizer's grid (cache keys: reco_engine/cache.Cache.voxel_key)
VOXELIZER_CODE = {"parity": ("src.voxelization", "src.field_channels", "src.closest_point_field"),
                  "stltovoxel": ("src.voxelization",)}


class LegacyVoxelizerWarning(UserWarning):
    """A grid that parity cannot make (fit / uniform / fixed+center) was voxelized by stltovoxel without being asked to."""


def requested_voxelizer(voxelizer: Optional[str] = None) -> Tuple[str, bool]:
    """(name, explicit): the argument, else $SHAPE_RECO_VOXELIZER, else DEFAULT_VOXELIZER ("parity"); `explicit` is
    False only when nothing asked for one. An unknown name raises."""
    explicit = voxelizer is not None
    if voxelizer is None:
        env = os.environ.get("SHAPE_RECO_VOXELIZER", "").strip().lower()
        explicit = bool(env)
        voxelizer = env or DEFAULT_VOXELIZER
    if voxelizer not in VOXELIZERS:
        raise ValueError(f"voxelizer must be one of {VOXELIZERS} (argument or SHAPE_RECO_VOXELIZER), got {voxelizer!r}")
    return voxelizer, explicit


def voxelizer_for(scale_mode: str = "fit", anchor: str = "center", voxelizer: Optional[str] = None,
                  warn: bool = True) -> str:
    """The voxelizer `voxelize_stl_to_bin` runs for this grid. Parity makes the in-scale grid only (scale_mode "fixed",
    anchor "origin"); the other modes are stltovoxel's own grids (stretched per axis, or centred by its slice count).
    There, an explicit "parity" (argument or env) raises ValueError, and the default falls back to stltovoxel with a
    LegacyVoxelizerWarning (`warn`=False: no warning, for key computations)."""
    name, explicit = requested_voxelizer(voxelizer)
    if name == "stltovoxel" or (scale_mode == "fixed" and anchor == "origin"):
        return name
    if explicit:
        raise ValueError(f"the parity voxelizer makes the in-scale grid only (scale_mode='fixed', anchor='origin'), not "
                         f"scale_mode={scale_mode!r} anchor={anchor!r}; ask for voxelizer='stltovoxel' "
                         f"(SHAPE_RECO_VOXELIZER=stltovoxel, --voxelizer stltovoxel) for this legacy grid")
    if warn:
        warnings.warn(f"voxelization: scale_mode={scale_mode!r} anchor={anchor!r} is a legacy grid that only stltovoxel "
                      f"makes - voxelized by stltovoxel (and recorded so), not by the parity default (D-vox), which "
                      f"needs scale_mode='fixed' anchor='origin'", LegacyVoxelizerWarning, stacklevel=3)
    return "stltovoxel"

def find_stltovoxel(executable: str = "stltovoxel") -> Optional[str]:
    """Find stltovoxel CLI."""
    detected = shutil.which(executable)
    if detected:
        return detected

    for path in [
        os.path.expanduser("~/.local/bin/stltovoxel"),
        "/usr/local/bin/stltovoxel",
        "/usr/bin/stltovoxel",
    ]:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None

def check_voxelization_available(executable: str = "stltovoxel") -> bool:
    if not HAS_DEPS:
        return False
    return find_stltovoxel(executable) is not None

def _natural_key(path: Path):
    """Sort files numerically: slice2.png < slice10.png."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', path.name)]

def _voxelize(stl_path: str, pics_dir: str, resolution: int,
              executable: str) -> Tuple[bool, str]:
    """Run stltovoxel with several output-naming strategies."""
    exe = find_stltovoxel(executable)
    if not exe:
        return False, "stltovoxel not found in PATH"

    os.makedirs(pics_dir, exist_ok=True)

    # The CLI usually wants a filename PATTERN, not a directory.
    # Strategy 1: numbered pattern template
    pattern = os.path.join(pics_dir, "slice.png")
    cmd = [exe, str(stl_path), pattern, "--resolution", str(resolution)]
    log_debug(f"Trying: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=clean_subprocess_env())
    except subprocess.TimeoutExpired:
        return False, "stltovoxel timed out"
    except Exception as exc:
        return False, f"stltovoxel exception: {exc}"

    pngs = _collect_pngs(pics_dir)

    if proc.returncode == 0 and pngs:
        log_debug(f"Strategy 1 OK: {len(pngs)} PNGs")
        return True, f"{len(pngs)} slices"

    # Strategy 2: pass the directory itself
    cmd = [exe, str(stl_path), pics_dir, "--resolution", str(resolution)]
    log_debug(f"Trying strategy 2: {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=clean_subprocess_env())
    except Exception as exc:
        return False, f"stltovoxel exception: {exc}"

    pngs = _collect_pngs(pics_dir)

    if proc.returncode == 0 and pngs:
        log_debug(f"Strategy 2 OK: {len(pngs)} PNGs")
        return True, f"{len(pngs)} slices"

    err = (proc.stderr or proc.stdout or "unknown error")[:500]
    return False, f"stltovoxel failed (rc={proc.returncode}): {err}"

def _collect_pngs(pics_dir: str) -> List[Path]:
    """Collect PNG files, naturally sorted, excluding single-file oddities."""
    paths = [p for p in Path(pics_dir).glob("*.png") if p.is_file()]
    return sorted(paths, key=_natural_key)

def assemble_pngs_to_bin(
    pics_dir: str,
    output_bin: str,
    resolution: int = 64,
    noise_config=None,
    noise_rng=None,
) -> Tuple[bool, str]:
    """
    Assemble PNG slices into a binary uint8 .bin volume of shape
    (resolution, resolution, resolution), values in {0, 1}.

    Handles:
    - more slices than needed (subsample evenly)
    - fewer slices than needed (repeat last slice)
    - slices with wrong pixel size (resized to resolution x resolution)
    - grayscale or RGBA input
    """
    if not HAS_DEPS:
        return False, "Pillow/numpy not installed"

    pngs = _collect_pngs(pics_dir)
    if not pngs:
        return False, f"No PNG slices found in {pics_dir}"

    n_found = len(pngs)
    log_debug(f"Assembling {n_found} slices -> {resolution}^3")

    # Evenly subsample or pad the slice axis to exactly `resolution`
    volume = np.zeros((resolution, resolution, resolution), dtype=np.uint8)

    if n_found >= resolution:
        # Pick evenly spaced slices
        indices = np.linspace(0, n_found - 1, resolution).round().astype(int)
        indices = sorted(set(int(i) for i in indices))
        # Ensure exactly resolution slices
        while len(indices) < resolution:
            indices.append(indices[-1])
        selected = [pngs[i] for i in indices[:resolution]]
    else:
        selected = pngs
        # Pad by repeating the last slice to fill the depth
        while len(selected) < resolution:
            selected.append(pngs[-1])

    for i, png_path in enumerate(selected):
        img = Image.open(png_path).convert("L")

        # Resize if the pixel grid doesn't match resolution
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution),
                             resample=Image.NEAREST)

        arr = np.array(img)

        # Binarize: occupied = 1, empty = 0
        mask = (arr > 127).astype(np.uint8)
        volume[i, :, :] = mask

    if noise_config is not None and noise_config.enabled:
        from src.voxel_noise import apply_voxel_noise
        volume = apply_voxel_noise(volume, noise_config, noise_rng)

    volume.tofile(output_bin)

    if not os.path.isfile(output_bin) or os.path.getsize(output_bin) == 0:
        return False, f".bin not written: {output_bin}"

    filled = float((volume > 0).mean()) * 100.0
    log_debug(f"BIN written: {output_bin}, filled {filled:.1f}%")

    # Sanity check: a completely empty or completely full volume is suspicious
    if filled < 0.5:
        return True, f"WARNING: only {filled:.1f}% filled"
    if filled > 99.5:
        return True, f"WARNING: {filled:.1f}% filled (fully solid?)"

    return True, f"OK ({n_found} slices, {filled:.1f}% filled)"

# ----------------------------------------------------------------------
# Scale modes
#
#   "fit"      (default, what every existing dataset and checkpoint uses)
#              `stltovoxel --resolution N` keeps proportions, but
#              assemble_pngs_to_bin then resizes every slice to a square and
#              forces N slices, so the bounding box is stretched onto the whole
#              grid independently per axis: aspect ratio and absolute size are
#              lost (a thin plate and a cube look alike).
#   "uniform"  normalized: one isotropic voxel size chosen so the LARGEST extent
#              spans the grid; proportions are kept, the part is centred.
#   "fixed"    affine: a fixed physical voxel size, scale_mm / resolution, part
#              centred on the grid, so absolute size is kept too (a part bigger
#              than scale_mm is clipped by the grid, with a warning).
# ----------------------------------------------------------------------

SCALE_MODES = ("fit", "uniform", "fixed")


def mesh_bounds(stl_path: str):
    """(min, max) corners (x, y, z) of an STL's bounding box."""
    from stl import mesh as stl_mesh
    vertices = stl_mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3)
    return vertices.min(axis=0), vertices.max(axis=0)


def mesh_extents(stl_path: str):
    """(x, y, z) size of an STL's bounding box (numpy-stl, which stltovoxel needs anyway)."""
    from stl import mesh as stl_mesh
    vertices = stl_mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3)
    return vertices.max(axis=0) - vertices.min(axis=0)


def _voxelize_in_scale(stl_path: str, pics_dir: str, voxel_size: float,
                       executable: str) -> Tuple[bool, str]:
    """stltovoxel with ONE isotropic physical voxel size and no padding: every axis keeps its scale."""
    exe = find_stltovoxel(executable)
    if not exe:
        return False, "stltovoxel not found in PATH"
    os.makedirs(pics_dir, exist_ok=True)
    cmd = [exe, str(stl_path), os.path.join(pics_dir, "slice.png"),
           "--voxel-size", repr(float(voxel_size)), "--pad", "0"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=clean_subprocess_env())
    except Exception as exc:
        return False, f"stltovoxel exception: {exc}"
    pngs = _collect_pngs(pics_dir)
    if proc.returncode != 0 or not pngs:
        return False, f"stltovoxel failed (rc={proc.returncode}): {(proc.stderr or proc.stdout)[:500]}"
    return True, f"{len(pngs)} slices"


def assemble_pngs_in_scale(pics_dir: str, output_bin: str, resolution: int = 64,
                           noise_config=None, noise_rng=None, offsets=None) -> Tuple[bool, str]:
    """Slices -> a (resolution,)*3 uint8 {0,1} volume, NOT stretched: the slices are stacked as
    they are (array [z, y, x], as in assemble_pngs_to_bin) and centred in the grid; an axis longer
    than the grid is centre-cropped. `offsets` (array order, voxels) instead places the part's first
    voxel at that grid index (anchor="origin"), clipping whatever falls outside."""
    if not HAS_DEPS:
        return False, "Pillow/numpy not installed"
    pngs = _collect_pngs(pics_dir)
    if not pngs:
        return False, f"No PNG slices found in {pics_dir}"
    solid = np.stack([np.array(Image.open(p).convert("L")) > 127 for p in pngs])       # (nz, ny, nx)

    volume = np.zeros((resolution,) * 3, dtype=np.uint8)
    src, dst, clipped = [], [], False
    for axis, n in enumerate(solid.shape):
        if offsets is not None:
            lo = int(offsets[axis])
            s0, d0 = max(0, -lo), max(0, lo)
            length = min(n - s0, resolution - d0)
            if length <= 0:
                src.append(slice(0, 0)); dst.append(slice(0, 0)); clipped = True
                continue
            src.append(slice(s0, s0 + length)); dst.append(slice(d0, d0 + length))
            clipped = clipped or s0 > 0 or length < n
        elif n <= resolution:
            offset = (resolution - n) // 2
            src.append(slice(0, n)); dst.append(slice(offset, offset + n))
        else:
            start = (n - resolution) // 2
            src.append(slice(start, start + resolution)); dst.append(slice(0, resolution)); clipped = True
    volume[tuple(dst)] = solid[tuple(src)]

    if noise_config is not None and noise_config.enabled:
        from src.voxel_noise import apply_voxel_noise
        volume = apply_voxel_noise(volume, noise_config, noise_rng)
    volume.tofile(output_bin)
    if not os.path.isfile(output_bin) or os.path.getsize(output_bin) == 0:
        return False, f".bin not written: {output_bin}"
    filled = float((volume > 0).mean()) * 100.0
    return True, (f"WARNING: part larger than the grid, clipped ({solid.shape} -> {resolution}^3)" if clipped
                  else f"OK (in scale, {solid.shape[0]}x{solid.shape[1]}x{solid.shape[2]} voxels, {filled:.1f}% filled)")


def _voxelize_parity(stl_path: str, output_bin: str, resolution: int, scale_mm: float, noise_config=None,
                     noise_rng=None) -> bool:
    """The parity voxelizer (the default, `voxelizer_for`): the in-scale grid (world origin at the centre, `scale_mm`
    wide) by scanline ray parity of the placed mesh at every voxel CENTRE (field_channels.parity_occupancy, the
    convention of the exact field and of the
    reconstruction loop's inference grid), written in the .bin array order [z, -y, x]. stltovoxel's anchor is rounded to
    whole voxels, which displaced the stored grids by up to about half a voxel per axis, varying per sample (measured on v5
    small: IoU with the exact solid 0.84 median, down to 0 on a 26-voxel torus; docs/training_plan_v5.tex, F7)."""
    import numpy as np
    from src.field_channels import inscale_spec, parity_of_file, world_to_bin
    if not scale_mm or scale_mm <= 0:
        raise ValueError("the parity voxelizer needs scale_mm (the physical size of the grid)")
    world = parity_of_file(stl_path, inscale_spec(scale_mm, resolution))       # cached: the inside bits and field reuse it
    volume = world_to_bin(world).astype(np.uint8)
    if noise_config is not None and noise_config.enabled:
        from src.voxel_noise import apply_voxel_noise
        volume = apply_voxel_noise(volume, noise_config, noise_rng)
    volume.tofile(output_bin)
    return os.path.isfile(output_bin) and os.path.getsize(output_bin) == resolution ** 3


def voxelize_stl_to_bin(
    stl_path: str,
    output_bin: str,
    resolution: int = 64,
    executable: str = "stltovoxel",
    noise_config=None,
    noise_rng=None,
    scale_mode: str = "fit",
    scale_mm: float = None,
    anchor: str = "center",
    voxelizer: Optional[str] = None,
    report: Optional[dict] = None,
) -> bool:
    """Complete pipeline: STL -> .bin volume, by the voxelizer `voxelizer_for(scale_mode, anchor, voxelizer)` picks
    (parity on the in-scale grid by default; stltovoxel's PNG slices for the legacy grids or on request). `report`, if
    given, gets report["voxelizer"] = the voxelizer that ran (set before it runs, so also on a failure).

    `noise_config` (a voxel_noise.VoxelNoiseConfig) optionally corrupts
    the clean occupancy grid - salt/pepper speckle, spliced-in foreign
    object blobs, a dilation/erosion pass - before it is written to
    disk, so a training sample looks like a rough scan rather than
    pristine generated geometry. Off by default; `noise_rng` (a
    np.random.Generator) must be given whenever `noise_config` is
    non-default, for reproducibility.
    """
    if not os.path.isfile(stl_path):
        log_debug(f"STL not found: {stl_path}")
        return False
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    if anchor not in ("center", "origin"):
        raise ValueError(f"anchor must be 'center' or 'origin', got {anchor!r}")
    if anchor == "origin" and scale_mode != "fixed":
        raise ValueError("anchor='origin' needs scale_mode='fixed' (a physical grid to anchor the world origin in)")

    used = voxelizer_for(scale_mode, anchor, voxelizer)
    if report is not None:
        report["voxelizer"] = used
    if used == "parity":
        return _voxelize_parity(stl_path, output_bin, resolution, scale_mm, noise_config, noise_rng)

    with tempfile.TemporaryDirectory() as tmp_dir:
        pics_dir = os.path.join(tmp_dir, "pics")

        if scale_mode == "fit":
            ok, msg = _voxelize(stl_path, pics_dir, resolution, executable)
        else:
            if scale_mode == "fixed":
                if not scale_mm or scale_mm <= 0:
                    raise ValueError("scale_mode='fixed' needs scale_mm (the physical size of the grid)")
                voxel_size = scale_mm / resolution
            else:   # uniform: the largest extent spans the grid (a hair over, so rounding never adds a voxel)
                voxel_size = float(max(mesh_extents(stl_path))) / resolution * 1.001
            ok, msg = _voxelize_in_scale(stl_path, pics_dir, voxel_size, executable)
        if not ok:
            log_debug(f"Slice generation failed: {msg}")
            return False

        extra = {}
        if anchor == "origin":
            # The world origin sits at the grid centre and the part keeps its position. stltovoxel's first
            # voxel is the mesh's min corner for x and z, but for y the array runs BACKWARDS (image rows
            # top to bottom), so its first row is the mesh's max y.
            lo, hi = mesh_bounds(stl_path)
            v = scale_mm / resolution
            extra["offsets"] = (round((lo[2] + scale_mm / 2) / v),      # a0 = z
                                round((scale_mm / 2 - hi[1]) / v),      # a1 = y, reversed
                                round((lo[0] + scale_mm / 2) / v))      # a2 = x
        assemble = assemble_pngs_to_bin if scale_mode == "fit" else assemble_pngs_in_scale
        ok, msg = assemble(pics_dir, output_bin, resolution, noise_config=noise_config, noise_rng=noise_rng, **extra)
        if not ok:
            log_debug(f"Assembly failed: {msg}")
            return False

        # Assembly may return warnings with success
        if "WARNING" in msg:
            log_debug(f"Assembly warning: {msg}")

        return True

# ----------------------------------------------------------------------
# Standalone test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DEBUG = True

    if len(sys.argv) < 2:
        print("Usage: python3 voxelization.py <stl_file> [output.bin] [resolution]")
        print("Example: python3 voxelization.py ../dataset/stl/box/box_99999999.stl /tmp/test.bin 64")
        sys.exit(1)

    stl = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/test_volume.bin"
    res = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    print(f"Tool found:  {find_stltovoxel()}")
    print(f"Deps (PIL/numpy): {HAS_DEPS}")
    print(f"Input STL:   {stl}")

    success = voxelize_stl_to_bin(stl, out, res)

    if success:
        print(f"\n✓ Wrote {out} ({os.path.getsize(out):,} bytes)")
        # Verify by loading it back
        vol = np.fromfile(out, dtype=np.uint8)
        side = int(round(vol.size ** (1 / 3.0)))
        vol = vol[:side**3].reshape(side, side, side)
        print(f"  Loaded back: shape {vol.shape}")
        print(f"  Occupied voxels: {(vol > 0).sum():,} ({(vol > 0).mean()*100:.1f}%)")
    else:
        print("\n✗ Voxelization FAILED")
        sys.exit(1)
