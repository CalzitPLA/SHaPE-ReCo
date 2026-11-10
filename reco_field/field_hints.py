#!/usr/bin/env python3
"""
Turns the vector-field route's trained point-cloud networks (training/train_field_type.py,
train_field_rotation.py, train_field_dimension.py, train_field_position.py) into the `hint` argument
`reco_field.fit.fit_box` (and, as more topologies get checkpoints, the other `fit_<kind>` functions) already
accepts - an ADDITIONAL starting hypothesis for `_optimise`, never a replacement for the existing PCA/
Manhattan-frame geometric starts (see fit_box's own docstring). Opt-in only: `reco_field.peel.peel_steps`'s
default behaviour is completely unchanged unless a `FieldNetHints` is explicitly passed to it (see
`docs/FRAME_FAILURE.md`'s Strategy discussion and the plan this was built from,
~/.claude/plans/dazzling-prancing-tower.md).

    hints = FieldNetHints.load(Path("checkpoints_field_pertopo"))
    fit_box(samples, hint=hints.box_hint(samples))          # None if no box checkpoint is loaded
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from reco_field.fit import Samples
from training.field_point_backbone import encode_patch
from training.field_point_nets import DIMENSION_KEYS, ROTATION_KINDS, WHOLE_KINDS


def _load_checkpoint(path: Path, ctor, *ctor_args):
    import torch
    net = ctor(*ctor_args)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    return net.eval()


@dataclass
class FieldNetHints:
    """Whichever of the 16 networks (`checkpoints_field_pertopo/`) were found on disk - loading is
    all-or-nothing per topology/role, never partial-but-broken: a topology's hint methods return None
    unless ALL the networks that hint needs are present."""
    type_net: Optional[object] = None
    rotation: Dict[str, object] = None
    dimension: Dict[str, object] = None
    position: Dict[str, object] = None
    n_points: int = 512

    def __post_init__(self):
        self.rotation = self.rotation or {}
        self.dimension = self.dimension or {}
        self.position = self.position or {}

    @classmethod
    def load(cls, root: Path, checkpoint_name: str = "best_model.pth") -> "FieldNetHints":
        import torch
        from training.field_point_nets import FieldAxisNet, FieldBoxPoseNet, FieldDimensionNet, FieldPositionNet, FieldTypeNet
        root = Path(root)
        type_path = root / "type_cnn" / checkpoint_name
        type_net = _load_checkpoint(type_path, FieldTypeNet) if type_path.is_file() else None
        rotation, dimension, position = {}, {}, {}
        for topology in WHOLE_KINDS:
            dim_path = root / f"{topology}_dimension_cnn" / checkpoint_name
            if dim_path.is_file():
                dimension[topology] = _load_checkpoint(dim_path, FieldDimensionNet, len(DIMENSION_KEYS[topology]))
            pos_path = root / f"{topology}_position_cnn" / checkpoint_name
            if pos_path.is_file():
                position[topology] = _load_checkpoint(pos_path, FieldPositionNet)
            if topology in ROTATION_KINDS:
                rot_path = root / f"{topology}_rotation_cnn" / checkpoint_name
                if rot_path.is_file():
                    ctor = FieldBoxPoseNet if topology == "box" else FieldAxisNet
                    rotation[topology] = _load_checkpoint(rot_path, ctor)
        return cls(type_net=type_net, rotation=rotation, dimension=dimension, position=position)

    def type_ranking(self, s: Samples) -> Optional[Tuple[str, ...]]:
        """WHOLE_KINDS reordered by the Type network's predicted probability (most likely first), or None
        if no Type network was loaded. `reco_field.fit.fit_best` already tries the cheapest kind first and
        stops early on a good-enough fit; a caller MAY use this to try the network's top pick first instead -
        not wired into peel.py's default FIT_ORDER, which stays exactly as it always has."""
        if self.type_net is None:
            return None
        import torch
        x = torch.from_numpy(encode_patch(s.surface, s.normals, self.n_points)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.type_net(x), dim=1)[0].numpy()
        return tuple(np.array(WHOLE_KINDS)[np.argsort(-probs)])

    def box_hint(self, s: Samples) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """(centre, rotation matrix, half-extents), all in the SAME world frame `s.surface` is in, or None
        if the box rotation/dimension/position networks aren't all loaded."""
        if not ("box" in self.rotation and "box" in self.dimension and "box" in self.position):
            return None
        import torch
        x = torch.from_numpy(encode_patch(s.surface, s.normals, self.n_points)).unsqueeze(0)
        centroid = s.surface.mean(axis=0)
        extent = float(np.linalg.norm(s.surface.max(axis=0) - s.surface.min(axis=0)))
        with torch.no_grad():
            R = self.rotation["box"](x)[0].double().numpy()
            log_half_mm = self.dimension["box"](x)[0].double().numpy()
            offset = self.position["box"](x)[0].double().numpy()
        centre = centroid + offset * max(extent, 1e-6)
        half = np.exp(log_half_mm)
        return centre, R, half
