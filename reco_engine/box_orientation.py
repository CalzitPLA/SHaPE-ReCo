#!/usr/bin/env python3
"""
Orientation and dimensions of a BOX from a voxel grid, in the convention of training/rotation_symmetry.py.

The Rotation and Dimension CNNs of the original pipeline answer in an arbitrary axis order that the voxels do not
determine (see training/rotation_symmetry.py). For a box this module replaces them by

  1. alignment: the body is rotated so its box axes lie on the grid axes (`align_volume`). Default: the face-normal frame
     fit (training/box_alignment.py), which is robust to planar cuts and bites (median 1.3 deg on plain, cut and carved
     boxes, where PCA is 15-20 deg off on cut and carved ones); `align_method="pca"` gives the second-moment axes;
  2. residual rotation network (training/train_rotation_sym.py): the small correction, estimate = E @ residual;
  3. dimension network (training/train_box_dims.py): the three half extents, longest first, in voxels;

and returns what the macro builder needs: a world quaternion and (x, y, z) half-extent ratios with x >= y >= z, so that
"box with these ratios, rotated by this quaternion" is the body seen in the grid. It expects the IN-SCALE voxel
mode (`SHAPE_RECO_VOXEL_MODE=fixed`, `SHAPE_RECO_VOXEL_SCALE_MM=grid mm`, anchor origin): the stretched default
mode erases the aspect ratio and the size these networks read.

    orienter = BoxOrienter.load(Path("checkpoints_box"))   # rotation/ and dims/ inside, each with config.json
    est = orienter.estimate(voxel01)                        # voxel01: {0, 1} array, [z, -y, x] order
    est.quaternion, est.parameters
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REALIZATION_SCALE_MM = 10.0        # dataset_gen box macro: half extent mm = ratio * 10


def _small_rotation(axis: int, degrees: float) -> np.ndarray:
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    r = np.eye(3)
    i, j = [(1, 2), (0, 2), (0, 1)][axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, (-s if axis != 1 else s), (s if axis != 1 else -s), c
    return r


def refine_box(target: np.ndarray, orientation: np.ndarray, half: np.ndarray, center: np.ndarray,
               max_evaluations: int = 600):
    """Refine a box estimate against the voxels it should explain: coordinate descent on the IoU between the
    ANALYTIC box (orientation columns = its axes in the array frame, half extents in voxels, centre in array
    index coordinates; a voxel is inside when its centre is within the half extent along every axis) and the target.
    Rotation about the three body axes, the three half extents and the centre are moved by shrinking steps
    (2 deg / 1 voxel, then 1 / 0.5, then 0.5 / 0.25) while the IoU improves. Cheap (numpy, on the target's window), no
    rebuilding of geometry. Returns (orientation, half, center, iou_before, iou_after).

    This removes what the networks and the moment estimates leave: a few degrees of tilt (which inflates the measured
    extents and makes the candidate's bounding box larger than the box) and the sub-voxel bias on the thin axis."""
    occ = target > 0
    pts = np.argwhere(occ)
    if len(pts) < 20:
        return orientation, half, center, 0.0, 0.0
    lo = np.maximum(pts.min(axis=0) - 4, 0)
    hi = np.minimum(pts.max(axis=0) + 5, target.shape)
    window = occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    grid = np.stack(np.meshgrid(*[np.arange(l, h) for l, h in zip(lo, hi)], indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float64)
    truth = window.reshape(-1)
    n_truth = int(truth.sum())

    def iou(M, h, c):
        inside = (np.abs((grid - c) @ M) <= h).all(axis=1)
        inter = int(np.count_nonzero(inside & truth))
        return inter / max(int(np.count_nonzero(inside)) + n_truth - inter, 1)

    M, h, c = np.array(orientation, dtype=np.float64), np.array(half, dtype=np.float64), np.array(center, dtype=np.float64)
    before = best = iou(M, h, c)
    evals = 1
    for rot_step, len_step in ((2.0, 1.0), (1.0, 0.5), (0.5, 0.25)):
        improved = True
        while improved and evals < max_evaluations:
            improved = False
            moves = []
            for axis in range(3):
                for sign in (1.0, -1.0):
                    moves.append(("rot", axis, sign * rot_step))
                    moves.append(("half", axis, sign * len_step))
                    moves.append(("center", axis, sign * len_step))
            for kind, axis, delta in moves:
                M2, h2, c2 = M, h, c
                if kind == "rot":
                    M2 = M @ _small_rotation(axis, delta)
                elif kind == "half":
                    h2 = h.copy(); h2[axis] = max(h2[axis] + delta, 1.0)
                else:
                    c2 = c.copy(); c2[axis] += delta
                score = iou(M2, h2, c2)
                evals += 1
                if score > best + 1e-4:
                    M, h, c, best, improved = M2, h2, c2, score, True
    return M, h, c, before, best


@dataclass
class BoxEstimate:
    quaternion: List[float]                 # [qx, qy, qz, qw], world frame, for the macro builder
    parameters: Dict[str, float]            # x_half_ratio >= y_half_ratio >= z_half_ratio
    orientation: np.ndarray                 # (3, 3) array-frame orientation of the longest-first axes
    half_extents_vox: np.ndarray            # (3,) longest first
    pca_orientation: np.ndarray             # the alignment frame E the network started from (name kept from the PCA days)
    center_vox: Optional[np.ndarray] = None # array index coordinates of the box centre (set when refined)
    refined: bool = False


class BoxOrienter:
    def __init__(self, rotation_net, dims_net, device, grid_mm: float = 40.0, resolution: int = 64,
                 align_method: str = "frame", refine: bool = True):
        """`rotation_net` None: the alignment frame IS the orientation (no residual); `dims_net` None: the half extents
        are measured after aligning (see `frame_only` for the mode without any network - it needs no torch)."""
        self.rotation_net, self.dims_net, self.device = rotation_net, dims_net, device
        self.grid_mm, self.resolution = grid_mm, resolution
        self.align_method = align_method     # how the volume is aligned before the residual network (training/box_alignment.py)
        self.refine = refine                 # refine_box against the voxels after the network / measurement

    @classmethod
    def frame_only(cls, grid_mm: float = 40.0, resolution: int = 64, align_method: str = "frame",
                   refine: bool = True) -> "BoxOrienter":
        """The network-free orienter: orientation from the face-normal frame fit (training/box_alignment.py), half extents
        measured on the aligned body, then the IoU refinement. Needs no checkpoints and no torch. Measured on 600
        held-out boxes: as accurate as the residual network on top of it (median 1.3 deg, 94 % within 10 deg; the
        network adds nothing measurable, see docs/RUNNING.md, "Face-normal frame alignment"), and its extents are
        more accurate than the dimension network's."""
        return cls(None, None, None, grid_mm, resolution, align_method, refine)

    @classmethod
    def load(cls, root: Path, device=None, grid_mm: float = 40.0, resolution: int = 64,
             align_method: Optional[str] = None) -> "BoxOrienter":
        """`align_method` None: the one recorded in the rotation network's config.json (`align_method`, written by
        train_rotation_sym for networks trained on it), else "frame" - measured to be better than PCA even for
        networks that were trained on PCA-aligned volumes (docs/data/rotation_eval_frame.json)."""
        import torch
        from training.train_box_dims import BoxDimsCNN
        from training.train_rotation_sym import SymRotationCNN
        root = Path(root)
        device = device or torch.device("cpu")
        rot_cfg = json.loads((root / "rotation" / "config.json").read_text())
        if not rot_cfg.get("align"):
            raise ValueError("the rotation network must be a PCA-aligned residual model (config align=true)")
        rotation = SymRotationCNN(rot_cfg["rep"])
        rotation.load_state_dict(torch.load(root / "rotation" / "best_model.pth", map_location=device, weights_only=True)["model_state_dict"])
        dims = None
        if (root / "dims" / "best_model.pth").is_file():
            dims = BoxDimsCNN()
            dims.load_state_dict(torch.load(root / "dims" / "best_model.pth", map_location=device, weights_only=True)["model_state_dict"])
            dims = dims.to(device).eval()
        # without a dimension network the half extents are measured after aligning the body with the estimated orientation
        return cls(rotation.to(device).eval(), dims, device, grid_mm, resolution,
                   align_method or rot_cfg.get("align_method", "frame"))

    @property
    def voxel_mm(self) -> float:
        return self.grid_mm / self.resolution

    def estimate(self, voxel01: np.ndarray) -> Optional[BoxEstimate]:
        """None when there is too little material to measure."""
        from training import rotation_symmetry as rs
        aligned, E = rs.align_volume((np.asarray(voxel01) > 0).astype(np.uint8), self.align_method)
        if rs.pca_orientation(voxel01) is None:
            return None
        residual, log_h = np.eye(3), None
        if self.rotation_net is not None or self.dims_net is not None:
            import torch
            x = torch.from_numpy(aligned.astype(np.float32) * 2 - 1)[None, None].to(self.device)
            with torch.no_grad():
                if self.rotation_net is not None:
                    residual = self.rotation_net(x)[0].double().cpu().numpy()
                if self.dims_net is not None:
                    log_h = self.dims_net(x)[0].double().cpu().numpy()
        M = E @ residual
        if log_h is not None:
            half_vox = np.exp(log_h)
        else:
            volume = (np.asarray(voxel01) > 0).astype(np.uint8)
            half_vox = rs.measured_half_extents(rs.resample_rotated(volume, M, rs._centroid_of_body(volume)))
        center = None
        if self.refine:
            pts = np.argwhere(np.asarray(voxel01) > 0)
            start_center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
            M, half_vox, center, _, after = refine_box(voxel01, M, half_vox, start_center)
            order = np.argsort(-half_vox, kind="stable")                    # keep the longest-first convention
            M, half_vox = M[:, order], half_vox[order]
            if np.linalg.det(M) < 0:
                M = M.copy(); M[:, 2] *= -1.0
        ratios = half_vox * self.voxel_mm / REALIZATION_SCALE_MM
        return BoxEstimate(
            quaternion=rs.array_orientation_to_world_quaternion(M),
            parameters={"x_half_ratio": float(ratios[0]), "y_half_ratio": float(ratios[1]), "z_half_ratio": float(ratios[2])},
            orientation=M, half_extents_vox=half_vox, pca_orientation=E, center_vox=center, refined=center is not None,
        )
