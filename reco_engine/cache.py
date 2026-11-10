#!/usr/bin/env python3
"""
Content-addressed cache for the two expensive, deterministic steps of
proposing a candidate, so experimenting on a fixed shape stops paying for
the same work over and over.

Where a candidate's time goes (scripts/benchmark_positioning.py): ~68 %
building the primitive with freecadcmd, ~27 % voxelizing it with
stltovoxel, ~5 % everything else - the CNN calls are ~10 ms on a GPU.
Both expensive steps are pure functions of their inputs, and those inputs
repeat constantly when one shape is reconstructed with different
strategies, thresholds or depths: the first iteration always asks the
CNNs about the same grid, gets the same answer, and rebuilds the same
primitive.

Three kinds of entry, each keyed by a hash of everything the result depends
on, so a stale hit is not possible without changing an input:

  * a built primitive - keyed by (topology, parameters, quaternion,
    scale_mm, resolution, the CONTENT of the topology's FCMacro). Stores
    the STL (later stages re-read it from disk).
  * a voxelized STL - keyed by the STL's bytes, the resolution and grid mode,
    the voxelizer that runs for that mode (parity by default, stltovoxel for
    the legacy grids or on request) and the code fingerprint of that voxelizer
    (dataset_gen/src/cache_keys.py; for stltovoxel also the executable's
    identity), so it also serves the input part and every assembly rebuild,
    and a grid of one voxelizer or code version never serves another.
  * a CNN prediction - keyed by (the exact input grid, the checkpoint
    files' size + mtime). Stores the top-5 types, quaternion and
    parameters. In subprocess mode (torch not importable here) each call
    costs a process start, so this matters far more there.

Off by default: nothing is cached unless a cache is enabled, either with
`enable_cache(path)` or the SHAPE_RECO_CACHE environment variable. The
development scripts (compare_strategies.py, reconstruct_to_stl.py
--cache) opt in. It is safe to delete the directory at any time.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

CACHE_VERSION = "1"   # bump if the meaning of a cached result changes


def _digest(*parts: bytes) -> str:
    h = hashlib.sha1(CACHE_VERSION.encode())
    for part in parts:
        h.update(len(part).to_bytes(8, "little"))
        h.update(part)
    return h.hexdigest()


def _floats(values) -> bytes:
    # repr round-trips a float exactly, so equal predictions hash equal
    return ",".join(repr(float(v)) for v in values).encode()


# Bump when what build_primitive_stl writes changes for the same inputs (2: cylinders are recentred on the origin).
BUILD_VERSION = "2"


@dataclass
class CacheStats:
    build_hits: int = 0
    build_misses: int = 0
    voxel_hits: int = 0
    voxel_misses: int = 0
    predict_hits: int = 0
    predict_misses: int = 0

    def __str__(self) -> str:
        return (f"builds {self.build_hits} hit / {self.build_misses} miss, "
                f"voxelizations {self.voxel_hits} hit / {self.voxel_misses} miss, "
                f"predictions {self.predict_hits} hit / {self.predict_misses} miss")


class Cache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._macro_hashes = {}
        self._model_hashes = {}

    # ------------------------------------------------------------ primitives

    def _macro_hash(self, macro_path: Path) -> str:
        key = str(macro_path)
        if key not in self._macro_hashes:
            self._macro_hashes[key] = hashlib.sha1(Path(macro_path).read_bytes()).hexdigest()
        return self._macro_hashes[key]

    def build_key(self, topology: str, parameters: dict, quaternion, scale_mm: float,
                  resolution: int, macro_path: Path) -> str:
        params = ";".join(f"{k}={float(parameters[k])!r}" for k in sorted(parameters))
        return _digest(topology.encode(), params.encode(), _floats(quaternion),
                       repr(float(scale_mm)).encode(), str(int(resolution)).encode(),
                       self._macro_hash(macro_path).encode(), BUILD_VERSION.encode())

    def get_build(self, key: str, stl_dest: Path) -> bool:
        """Copy a cached primitive STL to `stl_dest`; False on a miss (or an
        unreadable entry)."""
        try:
            shutil.copyfile(self.root / "build" / f"{key}.stl", stl_dest)
        except OSError:
            self.stats.build_misses += 1
            return False
        self.stats.build_hits += 1
        return True

    def put_build(self, key: str, stl_path: Path) -> None:
        directory = self.root / "build"
        directory.mkdir(exist_ok=True)
        _atomic(directory / f"{key}.stl", lambda tmp: shutil.copyfile(stl_path, tmp))

    # ------------------------------------------------------------- voxelizing

    @staticmethod
    def voxel_key(stl_path: Path, resolution: int, scale_mode: str = "fit", scale_mm: float = None,
                  anchor: str = "center", voxelizer: Optional[str] = None) -> str:
        """Key of the grid `voxelize_stl_to_bin(stl_path, ..., resolution, scale_mode=, scale_mm=, anchor=,
        voxelizer=)` makes: the STL's content, the grid's parameters, the voxelizer that runs for them
        (voxelization.voxelizer_for: $SHAPE_RECO_VOXELIZER when `voxelizer` is None) and its code (cache_keys). Keys of
        the old formula (STL bytes + resolution [+ mode]) are never produced again (T-19, 2026-10-01)."""
        root = str(Path(__file__).resolve().parent.parent / "dataset_gen")
        if root not in sys.path:
            sys.path.insert(0, root)
        from src.cache_keys import cache_key
        from src.voxelization import VOXELIZER_CODE, find_stltovoxel, voxelizer_for
        used = voxelizer_for(scale_mode, anchor, voxelizer, warn=False)
        exe = find_stltovoxel() if used == "stltovoxel" else None
        params = dict(stl_sha1=hashlib.sha1(Path(stl_path).read_bytes()).hexdigest(), resolution=int(resolution),
                      scale_mode=scale_mode, scale_mm=None if scale_mm is None else float(scale_mm), anchor=anchor)
        return cache_key("voxel", files=[exe] if exe else (), voxelizer=used, code=VOXELIZER_CODE[used], params=params)

    def get_voxel(self, key: str) -> Optional[np.ndarray]:
        try:
            voxel = np.load(self.root / "voxel" / f"{key}.npz")["voxel"]
        except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
            self.stats.voxel_misses += 1
            return None
        self.stats.voxel_hits += 1
        return voxel

    def put_voxel(self, key: str, voxel: np.ndarray) -> None:
        directory = self.root / "voxel"
        directory.mkdir(exist_ok=True)
        def write(tmp):
            with open(tmp, "wb") as f:      # a path would make numpy append ".npz" and write elsewhere
                np.savez_compressed(f, voxel=voxel)
        _atomic(directory / f"{key}.npz", write)

    # ----------------------------------------------------------- predictions

    def _model_hash(self, models) -> str:
        # $SHAPE_RECO_REGISTRY can take the Type CNN from a different directory than checkpoint_root/type_cnn
        # (model_registry.registry_type_dir), so the key is built from the directory actually loaded
        from reco_engine.model_registry import registry_type_dir
        type_dir = registry_type_dir() or Path(models.checkpoint_root) / "type_cnn"
        key = (str(models.checkpoint_root), str(type_dir), models.checkpoint_name)
        if key not in self._model_hashes:
            parts = []
            for sub, directory in (("type_cnn", type_dir),
                                   ("rotation_cnn", Path(models.checkpoint_root) / "rotation_cnn"),
                                   ("dimension_cnn", Path(models.checkpoint_root) / "dimension_cnn")):
                st = (directory / models.checkpoint_name).stat()
                parts.append(f"{sub}:{directory}:{st.st_size}:{st.st_mtime_ns}")
            self._model_hashes[key] = ";".join(parts)
        # the restriction changes the answer, so it is part of the key (computed per call: cheap)
        allowed = getattr(models, "allowed_topologies", None)
        if allowed:
            return self._model_hashes[key] + ";allowed=" + ",".join(sorted(allowed))
        return self._model_hashes[key]

    def predict_key(self, models, cnn_input: np.ndarray) -> str:
        return _digest(self._model_hash(models).encode(), np.ascontiguousarray(cnn_input).tobytes())

    def get_prediction(self, key: str):
        try:
            data = json.loads((self.root / "predict" / f"{key}.json").read_text())
        except (OSError, ValueError):
            self.stats.predict_misses += 1
            return None
        self.stats.predict_hits += 1
        return data

    def put_prediction(self, key: str, data: dict) -> None:
        directory = self.root / "predict"
        directory.mkdir(exist_ok=True)
        _atomic(directory / f"{key}.json", lambda tmp: Path(tmp).write_text(json.dumps(data)))


def _atomic(dest: Path, write) -> None:
    """Write via a temp file in the same directory, then rename: a reader
    (or a parallel writer) never sees a half-written entry."""
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# ---------------------------------------------------------------- global switch

_default: Optional[Cache] = None
_default_resolved = False


def enable_cache(path) -> Cache:
    """Use `path` as the process-wide default cache."""
    global _default, _default_resolved
    _default, _default_resolved = Cache(Path(path)), True
    return _default


def disable_cache() -> None:
    global _default, _default_resolved
    _default, _default_resolved = None, True


def default_cache() -> Optional[Cache]:
    """The cache in effect: whatever enable_cache/disable_cache last set,
    else $SHAPE_RECO_CACHE if that is set, else none."""
    global _default, _default_resolved
    if not _default_resolved:
        env = os.environ.get("SHAPE_RECO_CACHE")
        _default = Cache(Path(env)) if env else None
        _default_resolved = True
    return _default


# ------------------------------------------------- the cached prediction step

def cached_predictions(cache: Optional[Cache], models, cnn_input: np.ndarray, predict_all):
    """(TypePrediction, quaternion, parameters) for `cnn_input`. `predict_all`
    computes them on a miss; with no cache it is simply called."""
    from reco_engine.predict import TypePrediction

    if cache is None:
        return predict_all()
    key = cache.predict_key(models, cnn_input)
    hit = cache.get_prediction(key)
    if hit is not None:
        tp = hit["type"]
        return (TypePrediction(tp["topology"], tp["confidence"], [tuple(t) for t in tp["top5"]]),
                hit["quaternion"], hit["parameters"])
    type_pred, quaternion, parameters = predict_all()
    cache.put_prediction(key, dict(
        type=dict(topology=type_pred.topology, confidence=type_pred.confidence,
                  top5=[list(t) for t in type_pred.top5]),
        quaternion=[float(v) for v in quaternion],
        parameters={k: float(v) for k, v in parameters.items()}))
    return type_pred, quaternion, parameters


# ------------------------------------------------- the wall thickness of a part

_WALL_MEMORY: dict = {}


def wall_cache_dir() -> Optional[Path]:
    """Where wall-thickness samples are kept: $SHAPE_RECO_WALL_CACHE (a directory; "0" or "off" disables it), default
    ~/.cache/shape_reco/wall_thickness. Unlike the opt-in cache above this one is on by default: an entry is 32 KB,
    and the rays behind it cost seconds to minutes per part (31 s for bs_hollow_sphere)."""
    env = os.environ.get("SHAPE_RECO_WALL_CACHE", "").strip()
    if env.lower() in ("0", "off", "no", "false"):
        return None
    return Path(env) if env else Path.home() / ".cache" / "shape_reco" / "wall_thickness"


def wall_thickness_cached(vertices, faces, n: int = 4000, seed: int = 0) -> np.ndarray:
    """The thickness samples `tau` (n,) of dataset_gen/src/local_grids.wall_thickness for this mesh, computed once per
    mesh: kept in memory and on disk, keyed by the mesh's bytes, n, seed and the source of wall_thickness itself (a
    changed function never hits an old entry)."""
    import inspect
    from src import local_grids
    V = np.ascontiguousarray(vertices, dtype=np.float64)
    F = np.ascontiguousarray(faces, dtype=np.int64)
    key = _digest(b"wall_thickness", V.tobytes(), F.tobytes(), str((n, seed)).encode(),
                  inspect.getsource(local_grids.wall_thickness).encode())
    hit = _WALL_MEMORY.get(key)
    if hit is not None:
        return hit.copy()
    folder = wall_cache_dir()
    path = folder / f"{key}.npy" if folder is not None else None
    if path is not None and path.is_file():
        try:
            tau = np.load(path)
            _WALL_MEMORY[key] = tau
            return tau.copy()
        except (OSError, ValueError):
            pass                                           # a damaged entry: measure again and overwrite it
    _, _, tau = local_grids.wall_thickness(V, F, n=n, seed=seed)
    tau = np.asarray(tau, np.float64)
    _WALL_MEMORY[key] = tau
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            def write(tmp):
                with open(tmp, "wb") as fh:
                    np.save(fh, tau)
            _atomic(path, write)
        except OSError:
            pass                                           # read-only home etc.: the memory copy still serves this run
    return tau.copy()
