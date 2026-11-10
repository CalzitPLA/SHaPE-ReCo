#!/usr/bin/env python3
"""
Inference for FusedNet (training/train_fused.py): one forward pass over the occupancy+SDF+vector-field input
(training/field_input.py) gives a topology, a box pose and a size. Two ways to use it, both built on the same
forward pass and the same pose/size -> build_primitive_stl conversion (_pose_and_size):

  predict_fused(...)               standalone: FusedNet's own topology guess too (top5) - for the GUI/FreeCAD
                                    "Fused Net Predict" node, a single recognition shown against the input.
  fused_estimate_for_topology(...) a known-size/pose SOURCE for a topology already decided elsewhere (the Type
                                    CNN) - the same role reco_engine.box_orientation.BoxOrienter /
                                    reco_engine.sphere_fit.SphereFitter / reco_engine.cylinder_fit.CylinderFitter
                                    already play via Models.box_orienter/sphere_fitter/cylinder_fitter;
                                    Models.fused_net plugs into reco_engine.greedy_reconstruct the same way, so
                                    the *existing* multi-primitive loop (and every GUI/node built on it) benefits
                                    without a separate decomposition loop of FusedNet's own.

Not a drop-in replacement for the three separate CNNs: FusedNet only covers box/sphere/cylinder, and only trained
a rotation for boxes (spheres and cylinders never contribute to the pose loss - train_fused.py's FusedDataset).
For sphere the identity quaternion is exactly right (rotation-invariant); for cylinder it is a placeholder - the
axis direction is not predicted, only radius and half length.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FUSED_TOPOLOGIES = ("box", "sphere", "cylinder")
IDENTITY_QUATERNION = [0.0, 0.0, 0.0, 1.0]


@dataclass
class FusedNetHandle:
    model: object          # training.train_fused.FusedNet, eval mode
    input_kind: str        # training/field_input.py KINDS entry it was trained with
    device: object          # torch.device
    sdf_truncation: float = 8.0


@dataclass
class FusedPrediction:
    topology: str
    confidence: float
    top5: List[Tuple[str, float]]
    quaternion: Optional[List[float]]          # world frame, for build_primitive_stl - None if the topology isn't covered
    parameters: Dict[str, float]                # ratio parameters at REALIZATION_SCALE_MM, ready for build_primitive_stl
    size_mm: Dict[str, float] = field(default_factory=dict)   # the same sizes, realized (post in-scale draw), for display


@dataclass
class FusedEstimate:
    quaternion: List[float]
    parameters: Dict[str, float]
    center_vox: np.ndarray
    size_mm: Dict[str, float] = field(default_factory=dict)


def load_fused_model(checkpoint_dir, checkpoint_name: str = "best_model.pth", device=None) -> FusedNetHandle:
    """`checkpoint_dir` is what --checkpoint-dir pointed training/train_fused.py at directly (its own
    config.json/best_model.pth sit right there, unlike the type_cnn/rotation_cnn/dimension_cnn subdirectory
    layout reco_engine.model_registry reads for the three separate CNNs)."""
    import torch
    from training.field_input import CHANNELS
    from training.train_fused import FusedNet

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{config_path}: not a train_fused.py checkpoint directory (no config.json)")
    cfg = json.loads(config_path.read_text())
    kind = cfg.get("input", "fused")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusedNet(in_channels=CHANNELS[kind])
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return FusedNetHandle(model=model, input_kind=kind, device=device, sdf_truncation=cfg.get("sdf_truncation", 8.0))


def _size_parameters(topology: str, size_vox: np.ndarray, voxel_mm: float, realization_scale_mm: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(ratio parameters for build_primitive_stl, size in mm for display) from FusedNet's three log-length slots
    (training/train_fused.py's size_targets: box = sorted half extents, sphere = radius, cylinder = radius + half
    length, all in voxels of the occupancy grid)."""
    size_mm_vals = size_vox * voxel_mm
    ratio = size_mm_vals / realization_scale_mm
    if topology == "box":
        return ({"x_half_ratio": float(ratio[0]), "y_half_ratio": float(ratio[1]), "z_half_ratio": float(ratio[2])},
                {"x_half_mm": float(size_mm_vals[0]), "y_half_mm": float(size_mm_vals[1]), "z_half_mm": float(size_mm_vals[2])})
    if topology == "sphere":
        return {"radius_ratio": float(ratio[0])}, {"radius_mm": float(size_mm_vals[0])}
    if topology == "cylinder":
        return ({"radius_ratio": float(ratio[0]), "half_length_ratio": float(ratio[1])},
                {"radius_mm": float(size_mm_vals[0]), "half_length_mm": float(size_mm_vals[1])})
    raise ValueError(f"FusedNet does not cover topology {topology!r} (only {FUSED_TOPOLOGIES})")


def _pose_and_size(handle: FusedNetHandle, occ01: np.ndarray, topology: str, grid_mm: float, resolution: int,
                   center_vox: np.ndarray, refine: bool):
    """(quaternion, parameters, size_mm, center_vox) for `topology` (must be one of FUSED_TOPOLOGIES) from one
    forward pass of FusedNet on `occ01`. Shared by predict_fused (topology from FusedNet's own type head) and
    fused_estimate_for_topology (topology decided elsewhere)."""
    import torch
    from training.field_input import encode
    from training.rotation_symmetry import array_orientation_to_world_quaternion
    from training.train_box_dims import REALIZATION_SCALE_MM

    volume_pm1 = occ01.astype(np.float32) * 2.0 - 1.0
    x = torch.from_numpy(encode(volume_pm1, handle.input_kind, handle.sdf_truncation)).unsqueeze(0).to(handle.device)
    with torch.no_grad():
        out = handle.model(x)
    voxel_mm = grid_mm / resolution
    size_vox = out["size"][0].double().cpu().numpy()

    if topology == "box":
        M = out["pose"][0].double().cpu().numpy()
        half_vox = np.exp(size_vox)
        if refine:
            from reco_engine.box_orientation import refine_box
            M, half_vox, center_vox, _before, _after = refine_box(occ01, M, half_vox, center_vox)
        parameters, size_mm = _size_parameters(topology, half_vox, voxel_mm, REALIZATION_SCALE_MM)
        quaternion = array_orientation_to_world_quaternion(M)
    else:
        parameters, size_mm = _size_parameters(topology, np.exp(size_vox), voxel_mm, REALIZATION_SCALE_MM)
        quaternion = list(IDENTITY_QUATERNION)     # not predicted for sphere (harmless: rotation-invariant) / cylinder (axis unknown)
    return quaternion, parameters, size_mm, center_vox


def predict_fused(handle: FusedNetHandle, voxel01: np.ndarray, grid_mm: float = 40.0, resolution: int = 64,
                  refine: bool = True) -> FusedPrediction:
    """One forward pass, FusedNet's own topology guess. `voxel01` is a {0,1} occupancy grid (array axes
    [z, -y, x], like every other voxel input in this project). For a box, the orientation is additionally
    polished against the actual voxels with reco_engine.box_orientation.refine_box (cheap, numpy-only coordinate
    descent on IoU) - the same refinement the network-free box orienter uses."""
    import torch
    from training.field_input import encode
    from training.topology_schema import IDX_TO_TOPOLOGY

    occ01 = (np.asarray(voxel01) > 0).astype(np.uint8)
    volume_pm1 = occ01.astype(np.float32) * 2.0 - 1.0
    x = torch.from_numpy(encode(volume_pm1, handle.input_kind, handle.sdf_truncation)).unsqueeze(0).to(handle.device)
    with torch.no_grad():
        out = handle.model(x)
    probs = torch.softmax(out["type"], dim=1)[0].cpu()
    k = min(5, int(probs.numel()))
    top5_probs, top5_idx = torch.topk(probs, k)
    top5 = [(IDX_TO_TOPOLOGY[int(i)], float(p)) for i, p in zip(top5_idx, top5_probs)]
    topology = top5[0][0]

    if topology not in FUSED_TOPOLOGIES:
        # FusedNet's type head still names the full topology list; pose/size heads have nothing to say about one
        # outside box/sphere/cylinder (train_fused.py never trained them on anything else).
        return FusedPrediction(topology=topology, confidence=top5[0][1], top5=top5, quaternion=None, parameters={})

    pts = np.argwhere(occ01)
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0 if len(pts) else np.zeros(3)
    quaternion, parameters, size_mm, _center = _pose_and_size(handle, occ01, topology, grid_mm, resolution, center, refine)
    return FusedPrediction(topology=topology, confidence=top5[0][1], top5=top5,
                           quaternion=quaternion, parameters=parameters, size_mm=size_mm)


def fused_estimate_for_topology(handle: FusedNetHandle, voxel01: np.ndarray, topology: str, grid_mm: float = 40.0,
                                resolution: int = 64, refine: bool = True) -> Optional[FusedEstimate]:
    """Like predict_fused, but for a topology already decided elsewhere (the Type CNN) - the known-size/pose
    source reco_engine.greedy_reconstruct reads from Models.fused_net, exactly like Models.box_orienter /
    sphere_fitter / cylinder_fitter. None when there is too little material to measure, or when `topology` isn't
    one of FUSED_TOPOLOGIES (the caller should simply not call this then, as with the other fitters)."""
    if topology not in FUSED_TOPOLOGIES:
        return None
    occ01 = (np.asarray(voxel01) > 0).astype(np.uint8)
    pts = np.argwhere(occ01)
    if len(pts) < 20:
        return None
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    quaternion, parameters, size_mm, center = _pose_and_size(handle, occ01, topology, grid_mm, resolution, center, refine)
    return FusedEstimate(quaternion=quaternion, parameters=parameters, center_vox=center, size_mm=size_mm)
