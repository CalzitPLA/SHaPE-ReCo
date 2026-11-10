#!/usr/bin/env python3
"""Single-voxel inference wrappers around the three CNNs.

Each function dispatches to either the in-process torch path or a
`predict_cli.py` subprocess, depending on how `models` was constructed
by reco_engine.model_registry.load_models() (see that module's
docstring for why: torch may not be importable in the current
process, e.g. under a FreeCAD AppImage's isolated Python).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reco_engine.model_registry import Models
from training.topology_schema import IDX_TO_TOPOLOGY, denormalize_params
from training.voxel_dataset import load_voxel

SANDBOX_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))
from src.dataset_utils import clean_subprocess_env  # noqa: E402


@dataclass
class TypePrediction:
    topology: str
    confidence: float
    top5: List[Tuple[str, float]]
    # Hierarchical TypeNet only (training/train_type_cnn.py --hierarchical), else None: the most likely subclass of
    # `topology` (training/subclass_schema.SUBCLASSES, e.g. "full", "hollow", "ibeam", "frame"), and for every parent in
    # top5 its subclasses with probabilities (softmax restricted to that parent), most likely first.
    subclass: Optional[str] = None
    subclasses: Optional[Dict[str, List[Tuple[str, float]]]] = None


def voxel_from_bin(bin_path, resolution: int) -> np.ndarray:
    """Load a .bin voxel file, ready for predict_* (see also
    training.voxel_dataset.load_voxel, which this wraps)."""
    return load_voxel(bin_path, resolution)


# ============================================================================
# SUBPROCESS DISPATCH
# ============================================================================

def _run_predict_cli(models: Models, task: str, voxel: np.ndarray, topology: str = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        npy_path = Path(tmp) / "voxel.npy"
        np.save(npy_path, voxel.astype(np.float32))

        command = [
            models.python_exe, "-m", "reco_engine.predict_cli",
            "--checkpoint-root", str(models.checkpoint_root),
            "--checkpoint-name", models.checkpoint_name,
            "--task", task,
            "--voxel-npy", str(npy_path),
        ]
        if topology is not None:
            command += ["--topology", topology]
        if models.allowed_topologies:
            command += ["--allowed", ",".join(models.allowed_topologies)]

        proc = subprocess.run(
            command, cwd=str(SANDBOX_ROOT),
            capture_output=True, text=True, timeout=120,
            env=clean_subprocess_env(),
        )

    if proc.returncode != 0:
        raise RuntimeError(
            f"predict_cli ({task}) failed via {models.python_exe} "
            f"(exit {proc.returncode}):\n{proc.stderr}"
        )

    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"predict_cli ({task}) produced unparseable output:\n"
            f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        ) from exc


# ============================================================================
# IN-PROCESS (torch) IMPLEMENTATION
# ============================================================================

def _to_tensor(voxel: np.ndarray, device) -> "torch.Tensor":
    import torch
    return torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,R,R,R)


def _type_tensor(models: Models, voxel: np.ndarray, inside: Optional[np.ndarray] = None) -> "torch.Tensor":
    """(1, C, R, R, R) on models.device, encoded the way models.type_cnn was trained (training/field_input.py):
    occupancy stays a bare {-1,+1} grid, the other kinds are derived from that same grid on the fly - no mesh
    needed (models.type_input_kind is never "exact_field": model_registry.load_models refuses to load one, see
    its docstring). The kinds read from a sample's .field.npz (vf_exact, dirlog_exact, dirlog_curv, voxfield) raise
    ValueError in `encode`. `inside`: the base-point bit of "dirlog_inside" (bool grid, same axes: the voxel centre lies in
    the exact solid); None uses the grid's own occupancy, which is that bit wherever the grid is clean parity.
    The grid's shape is checked before (predict_type -> check_type_grid)."""
    import torch
    kind = getattr(models, "type_input_kind", "occupancy")           # test doubles may not set it: occupancy is the default
    if kind == "occupancy":
        return _to_tensor(voxel, models.device)
    from training.field_input import encode
    encoded = encode(voxel, kind, getattr(models, "type_sdf_truncation", 8.0), inside=inside)
    return torch.from_numpy(encoded).unsqueeze(0).to(models.device)  # (1, C, R, R, R)


def _predict_type_in_process(models: Models, voxel: np.ndarray, inside: Optional[np.ndarray] = None) -> TypePrediction:
    import torch

    tensor = _type_tensor(models, voxel, inside)
    sub_logits = None
    with torch.no_grad():
        if getattr(models.type_cnn, "hierarchical", False):
            from training.train_type_cnn import expand_parent_logits
            parent_logits, sub_logits = models.type_cnn.heads(tensor)
            logits = expand_parent_logits(parent_logits)
            sub_logits = sub_logits[0].float().cpu()
        else:
            logits = models.type_cnn(tensor)
        if models.allowed_topologies:
            from training.topology_schema import TOPOLOGY_TO_IDX
            mask = torch.full_like(logits, float("-inf"))
            mask[:, [TOPOLOGY_TO_IDX[name] for name in models.allowed_topologies]] = 0.0
            logits = logits + mask
        probs = torch.softmax(logits, dim=1)[0].cpu()

    # never report a topology that was masked out (top-k would otherwise pad with p = 0)
    k = min(5, int((probs > 0).sum()))
    top5_probs, top5_idx = torch.topk(probs, k)
    top5 = [(IDX_TO_TOPOLOGY[int(i)], float(p)) for i, p in zip(top5_idx, top5_probs)]

    best_idx = int(top5_idx[0])
    subclass, subclasses = (None, None) if sub_logits is None else _subclasses(sub_logits, [name for name, _ in top5])
    return TypePrediction(
        topology=IDX_TO_TOPOLOGY[best_idx],
        confidence=float(top5_probs[0]),
        top5=top5,
        subclass=subclass,
        subclasses=subclasses,
    )


def _subclasses(sub_logits, parents: List[str]):
    """(best subclass of parents[0], {parent: [(subclass, p), ...] sorted}) from the (24,) subclass logits."""
    import torch
    from training.subclass_schema import FLAT_SUBCLASSES, PARENT_MASK, PARENT_TO_IDX
    out = {}
    for parent in parents:
        if parent not in PARENT_TO_IDX:
            continue
        idx = [i for i, on in enumerate(PARENT_MASK[PARENT_TO_IDX[parent]]) if on]
        probs = torch.softmax(sub_logits[idx], dim=0)
        out[parent] = sorted(((FLAT_SUBCLASSES[i].split("/", 1)[1], float(p)) for i, p in zip(idx, probs)),
                             key=lambda kv: -kv[1])
    best = out[parents[0]][0][0] if parents and parents[0] in out else None
    return best, out


def _predict_rotation_in_process(models: Models, voxel: np.ndarray, topology: str) -> List[float]:
    import torch
    from training.topology_schema import TOPOLOGY_TO_IDX

    tensor = _to_tensor(voxel, models.device)
    topo_idx = torch.tensor([TOPOLOGY_TO_IDX[topology]], device=models.device)
    with torch.no_grad():
        quat = models.rotation_cnn(tensor, topo_idx)[0].cpu().numpy()
    return [float(v) for v in quat]


def _predict_dimensions_in_process(models: Models, voxel: np.ndarray, topology: str) -> Dict[str, float]:
    import torch
    from training.topology_schema import TOPOLOGY_TO_IDX

    tensor = _to_tensor(voxel, models.device)
    topo_idx = torch.tensor([TOPOLOGY_TO_IDX[topology]], device=models.device)
    with torch.no_grad():
        normed = models.dimension_cnn(tensor, topo_idx)[0].cpu().numpy().tolist()
    return denormalize_params(topology, normed)


# ============================================================================
# PUBLIC API
# ============================================================================

def check_type_grid(models: Models, voxel: np.ndarray, inside: Optional[np.ndarray] = None) -> None:
    """Raise ValueError unless `voxel` (and `inside`) is the cubic grid the Type CNN was trained on
    (models.type_resolution; a legacy checkpoint without one only needs a cubic 3-D grid). Without it a wrong grid ran
    silently: 32^3 into the 64^3 net cost 0.23 parent accuracy and turned most answers into "box" (R-05c)."""
    shape = np.shape(voxel)
    expected = getattr(models, "type_resolution", None)
    if len(shape) != 3 or len(set(shape)) != 1:
        raise ValueError(f"Type CNN input must be a cubic 3-D grid, got shape {shape}")
    if expected is not None and shape[0] != int(expected):
        raise ValueError(f"Type CNN was trained on {expected}^3 grids ({getattr(models, 'checkpoint_root', '?')}), "
                         f"got {shape[0]}^3 - build the target at the checkpoint's resolution")
    if inside is not None and np.shape(inside) != shape:
        raise ValueError(f"inside bits {np.shape(inside)} do not match the grid {shape}")


def predict_type(models: Models, voxel: np.ndarray, inside: Optional[np.ndarray] = None) -> TypePrediction:
    """`inside`: the base-point bit for a "dirlog_inside" checkpoint (see _type_tensor); ignored by other input kinds
    and in the subprocess mode, where predict_cli encodes from the grid's own occupancy (the same bit for clean parity).
    Raises ValueError for a grid the checkpoint was not trained on (check_type_grid)."""
    check_type_grid(models, voxel, inside)
    if models.is_subprocess_mode:
        result = _run_predict_cli(models, "type", voxel)
        return TypePrediction(
            topology=result["topology"],
            confidence=result["confidence"],
            top5=[tuple(item) for item in result["top5"]],
            subclass=result.get("subclass"),
            subclasses=({k: [tuple(i) for i in v] for k, v in result["subclasses"].items()}
                        if result.get("subclasses") else None),
        )
    return _predict_type_in_process(models, voxel, inside)


def predict_rotation(models: Models, voxel: np.ndarray, topology: str) -> List[float]:
    if models.is_subprocess_mode:
        return _run_predict_cli(models, "rotation", voxel, topology)["quaternion"]
    return _predict_rotation_in_process(models, voxel, topology)


def predict_dimensions(models: Models, voxel: np.ndarray, topology: str) -> Dict[str, float]:
    if models.is_subprocess_mode:
        return _run_predict_cli(models, "dimension", voxel, topology)["parameters"]
    return _predict_dimensions_in_process(models, voxel, topology)
