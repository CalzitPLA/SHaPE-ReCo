#!/usr/bin/env python3
"""
Load the three trained CNN checkpoints for inference.

Two modes, chosen automatically based on whether torch is importable
in the current process:

- In-process (torch available): loads real nn.Module objects, as
  before. This is what happens under the system-installed FreeCAD
  (/usr/bin/freecad), whose embedded Python is the system Python and
  already has torch.
- Subprocess (torch NOT available): Models just remembers
  checkpoint_root + which python executable has torch, and
  reco_engine.predict's functions shell out to
  `<python_exe> -m reco_engine.predict_cli` for each prediction instead
  of importing torch directly. This is what a FreeCAD AppImage needs -
  AppImages bundle their own isolated Python (e.g. 3.11) that doesn't
  see whatever's installed for the system Python (e.g. 3.13), and a
  compiled package like torch can't be bridged across a Python
  minor-version/ABI gap via sys.path alone (confirmed: importing a
  cp313-built torch under a cp311 interpreter fails with "Failed to
  load PyTorch C extensions").

Either way, the returned Models object works transparently with
reco_engine.predict's predict_type/predict_rotation/predict_dimensions.

Type checkpoint from the stage registry: when $SHAPE_RECO_REGISTRY names a
models/registry.json (reco_engine/stage_registry.py), the Type CNN is read
from the registry's type entry - $SHAPE_RECO_REGISTRY_TYPE (an entry id),
else the file's "default_type_entry" - instead of <checkpoint_root>/type_cnn.
Rotation/dimension CNNs still come from checkpoint_root. Unset (the default),
nothing changes. Note: reco_engine/cache.py keys CNN predictions by
checkpoint_root/type_cnn, so do not combine the override with an enabled
prediction cache across different type entries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import torch
    from training.train_dimension_cnn import DimensionCNN
    from training.train_rotation_cnn import RotationCNN
    from training.train_type_cnn import TypeCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DEFAULT_SYSTEM_PYTHON = os.environ.get("SHAPE_RECO_PYTHON", "/usr/bin/python3")
CHECKPOINT_SUBDIRS = ("type_cnn", "rotation_cnn", "dimension_cnn")


@dataclass
class Models:
    type_cnn: Optional["TypeCNN"]
    rotation_cnn: Optional["RotationCNN"]
    dimension_cnn: Optional["DimensionCNN"]
    device: Optional["torch.device"]
    checkpoint_root: Path
    checkpoint_name: str = "best_model.pth"
    python_exe: Optional[str] = None  # set only in subprocess mode
    # If set, the Type CNN may only answer with one of these topologies: the
    # others' logits are masked out before the softmax. For troubleshooting
    # with the shapes the networks were actually trained on (box, sphere).
    allowed_topologies: Optional[tuple] = None
    # Optional replacement of the Rotation/Dimension CNNs for BOXES (reco_engine/box_orientation.py): PCA alignment +
    # a symmetry-aware residual network + a sorted-dimension network. Set by load_models from
    # $SHAPE_RECO_BOX_ORIENTER (a directory with rotation/ and dims/; or "frame" for the network-free orienter: face-normal
    # frame fit + measured extents, no checkpoints). Needs the in-scale voxel mode.
    box_orienter: Optional[object] = None
    # Analytic sphere fit (reco_engine/sphere_fit.py) for a predicted sphere: $SHAPE_RECO_SPHERE_FIT=1. In-scale voxel mode only.
    sphere_fitter: Optional[object] = None
    # Analytic cylinder fit (reco_engine/cylinder_fit.py) for a predicted cylinder: $SHAPE_RECO_CYLINDER_FIT=1. In-scale voxel mode only.
    cylinder_fitter: Optional[object] = None
    # FusedNet (training/train_fused.py, reco_engine/fused_predict.py): one network's pose + size heads as the
    # known-size/pose source for a predicted box/sphere/cylinder, the same role as box_orienter/sphere_fitter/
    # cylinder_fitter above. $SHAPE_RECO_FUSED_NET = a train_fused.py --checkpoint-dir. In-scale voxel mode only.
    fused_net: Optional[object] = None  # reco_engine.fused_predict.FusedNetHandle
    # What the Type CNN was trained on, read from type_cnn/config.json ("input"), written by training/train_type_cnn.py.
    # The grid-derived kinds of training/field_input.py ("occupancy" (default), "sdf", "field", "fused", "voxel_field",
    # "dirlog", "dirlog_inside") are encoded from the voxel grid by reco_engine.predict before every forward pass; the
    # kinds read from a sample's .field.npz ("vf_exact", "dirlog_exact", "dirlog_curv", "voxfield") load here but
    # field_input.encode refuses them at the first prediction.
    type_input_kind: str = "occupancy"
    type_sdf_truncation: float = 8.0
    # The grid the Type CNN was trained on (config.json "resolution", else model_card.json; _type_cnn_resolution).
    # predict.predict_type refuses a grid of another shape: a 32^3 grid fed to the 64^3 net ran silently and cost 0.23
    # parent accuracy (R-05c, docs/data/r05c_input_contract_2026-10-01.json on branch r05c-analyze). None: a legacy
    # checkpoint that recorded no resolution - not checked.
    type_resolution: Optional[int] = None

    @property
    def is_subprocess_mode(self) -> bool:
        return self.type_cnn is None


def registry_type_dir() -> Optional[Path]:
    """The Type CNN checkpoint directory chosen by $SHAPE_RECO_REGISTRY (see the module docstring), or None when the
    variable is unset. Raises for a missing file, an unknown entry id, or an entry the TypeCNN loader cannot read
    (not a cnn type entry, or a non-TypeCNN checkpoint such as FusedNet or the point-cloud type net). Admits the
    input kinds occupancy, sdf and fused only, so a "dirlog_inside" entry (the registry's own type network) is refused
    on this route; the cascade loads registry entries itself (cascade._load_type_net) without this check."""
    registry_path = os.environ.get("SHAPE_RECO_REGISTRY", "").strip()
    if not registry_path:
        return None
    from reco_engine.stage_registry import REPO_ROOT, entries_for, load_registry, load_registry_meta
    registry = load_registry(Path(registry_path))
    wanted = os.environ.get("SHAPE_RECO_REGISTRY_TYPE", "").strip() or load_registry_meta(Path(registry_path)).get("default_type_entry")
    candidates = [e for e in entries_for(registry, "type") if e.mode == "cnn"]
    if not wanted:
        raise ValueError(f"{registry_path}: no default_type_entry and $SHAPE_RECO_REGISTRY_TYPE unset")
    match = [e for e in candidates if e.id == wanted]
    if not match:
        raise ValueError(f"{registry_path}: no cnn type entry {wanted!r}; known: {[e.id for e in candidates]}")
    entry = match[0]
    if entry.input_kind not in ("occupancy", "sdf", "fused"):
        raise ValueError(f"type entry {entry.id!r} has input {entry.input_kind!r}; reco_engine's TypeCNN loader takes voxel inputs only")
    directory = entry.resolved_path(REPO_ROOT)
    config = directory / "config.json"
    if config.is_file() and "heads" in json.loads(config.read_text()):
        raise ValueError(f"type entry {entry.id!r} ({directory}) is a FusedNet checkpoint; load it via $SHAPE_RECO_FUSED_NET instead")
    return directory


def _type_dir(checkpoint_root: Path) -> Path:
    return registry_type_dir() or Path(checkpoint_root) / "type_cnn"


def _type_cnn_resolution(checkpoint_root: Path) -> Optional[int]:
    """The grid resolution the Type CNN was trained on: config.json "resolution" (every run since 2026-10-02, before
    that only --hierarchical runs), else model_card.json "resolution"; None when neither records it."""
    for name in ("config.json", "model_card.json"):
        path = _type_dir(checkpoint_root) / name
        if path.is_file():
            value = json.loads(path.read_text()).get("resolution")
            if value is not None:
                return int(value)
    return None


def _type_cnn_input_kind(checkpoint_root: Path, checkpoint_name: str) -> tuple:
    """(input_kind, sdf_truncation, in_channels) for the type_cnn checkpoint, from its config.json ("input",
    written by training/train_type_cnn.py) next to the checkpoint file; "occupancy" (1 channel) if there is none
    (an older checkpoint, from before --input existed). Raises for "exact_field": that input needs the STL mesh
    at prediction time, but the engine classifies voxel targets that have no mesh of their own (the cascade's
    components and leftovers; originally the removed route C's cropped sub-volumes, reco_engine/greedy_reconstruct.py)
    - there is no way to compute it there, so a checkpoint trained with --input exact_field cannot be loaded into the
    reconstruction engine. Use it via training/train_type_cnn.py's own eval scripts instead. Every other kind of
    training/field_input.CHANNELS is accepted here, including the .field.npz kinds that predict cannot encode
    (Models.type_input_kind)."""
    from training.field_input import CHANNELS
    from training.exact_field_input import CHANNELS as EXACT_FIELD_CHANNELS
    config_path = _type_dir(checkpoint_root) / "config.json"
    kind, truncation = "occupancy", 8.0
    if config_path.is_file():
        cfg = json.loads(config_path.read_text())
        kind = cfg.get("input", "occupancy")
        truncation = cfg.get("sdf_truncation", 8.0)
    if kind == "exact_field":
        raise ValueError(
            f"{config_path}: this Type CNN was trained with --input exact_field, which needs the STL mesh at "
            "prediction time. reco_engine's reconstruction loop classifies re-voxelized sub-volumes that have no "
            "mesh of their own, so this checkpoint cannot be loaded here - evaluate it with "
            "scripts/eval_type_fused.py or training/train_type_cnn.py instead."
        )
    channels = {**CHANNELS, "exact_field": EXACT_FIELD_CHANNELS}
    if kind not in channels:
        raise ValueError(f"{config_path}: unknown input kind {kind!r}")
    return kind, truncation, channels[kind]


def _load_one(model, checkpoint_path: Path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train it first with the matching training/train_*.py script."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(model, TypeCNN):
        # a hierarchical TypeNet checkpoint (train_type_cnn.py --hierarchical: parent + subclass heads) loads into a
        # HierTypeCNN, whose forward() still returns 17-way topology logits - every caller keeps working unchanged
        from training.train_type_cnn import type_cnn_for_state
        model = type_cnn_for_state(checkpoint["model_state_dict"], model.in_channels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _require_checkpoints_exist(checkpoint_root: Path, checkpoint_name: str) -> None:
    for subdir in CHECKPOINT_SUBDIRS:
        path = (_type_dir(checkpoint_root) if subdir == "type_cnn" else checkpoint_root / subdir) / checkpoint_name
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Train it first with the matching training/train_*.py script."
            )


def _parse_allowed(allowed) -> Optional[tuple]:
    """A tuple of valid topology names, or None for "no restriction"."""
    if allowed is None:
        allowed = os.environ.get("SHAPE_RECO_TOPOLOGIES") or None
    if allowed is None:
        return None
    if isinstance(allowed, str):
        allowed = [name for name in allowed.replace(",", " ").split() if name]
    from training.topology_schema import TOPOLOGY_TO_IDX
    unknown = [name for name in allowed if name not in TOPOLOGY_TO_IDX]
    if unknown:
        raise ValueError(f"unknown topologies {unknown}; known: {sorted(TOPOLOGY_TO_IDX)}")
    return tuple(allowed) or None


def load_models(
    checkpoint_root: Path,
    device=None,
    checkpoint_name: str = "best_model.pth",
    python_exe: Optional[str] = None,
    allowed_topologies=None,
) -> Models:
    """checkpoint_root should contain type_cnn/, rotation_cnn/, dimension_cnn/
    subdirectories, each holding checkpoints from the matching
    training/train_*_cnn.py script (--checkpoint-dir).

    python_exe is only used in subprocess mode (torch not importable
    here) - a Python executable path that DOES have torch. Defaults to
    $SHAPE_RECO_PYTHON or /usr/bin/python3.

    allowed_topologies (default: $SHAPE_RECO_TOPOLOGIES, comma separated,
    else no restriction) limits what the Type CNN may predict - see
    Models.allowed_topologies.
    """
    checkpoint_root = Path(checkpoint_root)
    allowed_topologies = _parse_allowed(allowed_topologies)

    sphere_fitter = None
    if os.environ.get("SHAPE_RECO_SPHERE_FIT", "").strip().lower() in ("1", "on", "true", "yes"):
        from reco_engine.sphere_fit import SphereFitter
        sphere_fitter = SphereFitter(grid_mm=float(os.environ.get("SHAPE_RECO_VOXEL_SCALE_MM", "40")))

    cylinder_fitter = None
    if os.environ.get("SHAPE_RECO_CYLINDER_FIT", "").strip().lower() in ("1", "on", "true", "yes"):
        from reco_engine.cylinder_fit import CylinderFitter
        cylinder_fitter = CylinderFitter(grid_mm=float(os.environ.get("SHAPE_RECO_VOXEL_SCALE_MM", "40")))

    if not TORCH_AVAILABLE:
        _require_checkpoints_exist(checkpoint_root, checkpoint_name)
        type_input_kind, type_sdf_truncation, _ = _type_cnn_input_kind(checkpoint_root, checkpoint_name)
        return Models(
            type_cnn=None, rotation_cnn=None, dimension_cnn=None, device=None,
            checkpoint_root=checkpoint_root, checkpoint_name=checkpoint_name,
            python_exe=python_exe or DEFAULT_SYSTEM_PYTHON, allowed_topologies=allowed_topologies,
            sphere_fitter=sphere_fitter, cylinder_fitter=cylinder_fitter,      # numpy only: they work without torch too
            type_input_kind=type_input_kind, type_sdf_truncation=type_sdf_truncation,
            type_resolution=_type_cnn_resolution(checkpoint_root),
        )

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    box_orienter = None
    orienter_dir = os.environ.get("SHAPE_RECO_BOX_ORIENTER")
    if orienter_dir:
        from reco_engine.box_orientation import BoxOrienter
        grid_mm = float(os.environ.get("SHAPE_RECO_VOXEL_SCALE_MM", "40"))
        if orienter_dir.strip().lower() == "frame":
            box_orienter = BoxOrienter.frame_only(grid_mm=grid_mm)
        else:
            box_orienter = BoxOrienter.load(Path(orienter_dir), torch.device("cpu"), grid_mm=grid_mm)

    fused_net = None
    fused_net_dir = os.environ.get("SHAPE_RECO_FUSED_NET")
    if fused_net_dir:
        from reco_engine.fused_predict import load_fused_model
        fused_net = load_fused_model(Path(fused_net_dir), checkpoint_name, device)

    type_input_kind, type_sdf_truncation, type_in_channels = _type_cnn_input_kind(checkpoint_root, checkpoint_name)
    return Models(
        type_cnn=_load_one(TypeCNN(in_channels=type_in_channels), _type_dir(checkpoint_root) / checkpoint_name, device),
        rotation_cnn=_load_one(RotationCNN(), checkpoint_root / "rotation_cnn" / checkpoint_name, device),
        dimension_cnn=_load_one(DimensionCNN(), checkpoint_root / "dimension_cnn" / checkpoint_name, device),
        device=device,
        checkpoint_root=checkpoint_root,
        checkpoint_name=checkpoint_name,
        allowed_topologies=allowed_topologies,
        box_orienter=box_orienter,
        sphere_fitter=sphere_fitter,
        cylinder_fitter=cylinder_fitter,
        fused_net=fused_net,
        type_input_kind=type_input_kind,
        type_sdf_truncation=type_sdf_truncation,
        type_resolution=_type_cnn_resolution(checkpoint_root),
    )
