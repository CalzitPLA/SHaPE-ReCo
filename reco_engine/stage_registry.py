#!/usr/bin/env python3
"""
One place that says which source answers which cascade stage for which topology.

Before this, every consumer hard-coded its own checkpoint choice: reco_gui/setup.py kept a PREFERRED list of type
directories, the FreeCAD nodes pointed at checkpoints_box, the live_v2/live_v4 bundles were symlink farms, and the
analytic fits (frame orienter, sphere/cylinder fits, placement) were switched on by separate environment variables.
None of them recorded WHY a source was chosen. models/registry.json lists every source - CNN checkpoint, analytic
fit or prior - per stage ("type", "rotation", "dimensions", "position"), with the held-out numbers that justified it
(scripts/eval_stage_ensemble.py -> docs/data/ensemble_selection.json), a cost to order them by (cheap first) and an
optional confidence gate above which a source short-circuits its stage.

Old checkpoint directories stay where they are; the registry only points at them (paths relative to the repo root).

What the cascade (reco_engine/cascade.py) does with the entries (checked 2026-10-01): the type stage skips entries marked
`legacy` (D-007) and fuses the live ones (weights = metrics["parent_acc"]); a "cnn" pose entry (rotation, dimensions)
runs only when the caller passes a callable for its id in `models`, so the registry's pose CNN entries do not run in a
normal loop run. The file's other keys are not read by the loop: "ensembles" and "type_stage" document the type rule
("ensembles" is read by scripts/make_training_plan_v5_tex.py, "type_stage" by no code), "default_type_entry" is read by
reco_engine/model_registry.py only. The cascade loads
the file through cascade.load_default_registry, which falls back to its network-free default on any error, silently.

    from reco_engine.stage_registry import load_registry, entries_for
    reg = load_registry()                        # <repo>/models/registry.json
    for e in entries_for(reg, "rotation", "box"):
        ...                                      # frame fit first, then the residual networks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "models" / "registry.json"

STAGES = ("type", "rotation", "dimensions", "position")
MODES = ("cnn", "analytic", "prior", "descriptor")     # descriptor: reco_engine/type_descriptors.py (type stage only)
INPUT_KINDS = ("occupancy", "sdf", "fused", "vf_exact", "points", "none",
               "dirlog", "dirlog_inside")   # grid-derived (training/field_input); cascade.inscale_target builds both


@dataclass
class StageEntry:
    id: str                 # e.g. "type_v3_B", "box_rot_frame", "box_rot_H_manifold", "cyl_fit_analytic"
    stage: str              # "type" | "rotation" | "dimensions" | "position"
    kinds: tuple            # topologies it answers, e.g. ("box","sphere")
    mode: str               # "cnn" | "analytic" | "prior" | "descriptor"
    path: Optional[str]     # checkpoint dir (cnn) / joblib file (descriptor) relative to repo root, else None
    input_kind: str         # one of INPUT_KINDS: "occupancy" | "sdf" | "fused" | "vf_exact" | "points" | "none" | "dirlog" | "dirlog_inside"
    cost: int               # ordering key: lower runs first
    gate: Optional[float]   # confidence above which this mode short-circuits the stage (None = never)
    metrics: dict = field(default_factory=dict)   # held-out numbers that justified the choice
    legacy: bool = False    # kept for tests/reference, skipped by the cascade's type stage (decision D-007)

    def resolved_path(self, root: Path = REPO_ROOT) -> Optional[Path]:
        """The checkpoint directory as an absolute path (None for analytic/prior modes)."""
        return None if self.path is None else (Path(root) / self.path)


def _entry(raw: dict, where: str) -> StageEntry:
    try:
        e = StageEntry(id=str(raw["id"]), stage=str(raw["stage"]), kinds=tuple(raw["kinds"]), mode=str(raw["mode"]),
                       path=raw.get("path"), input_kind=str(raw.get("input_kind", "none")), cost=int(raw["cost"]),
                       gate=None if raw.get("gate") is None else float(raw["gate"]), metrics=dict(raw.get("metrics") or {}),
                       legacy=bool(raw.get("legacy", False)))
    except KeyError as missing:
        raise ValueError(f"{where}: registry entry {raw.get('id', '?')!r} lacks the field {missing}") from None
    if e.stage not in STAGES:
        raise ValueError(f"{where}: entry {e.id!r} has unknown stage {e.stage!r} (known: {STAGES})")
    if e.mode not in MODES:
        raise ValueError(f"{where}: entry {e.id!r} has unknown mode {e.mode!r} (known: {MODES})")
    if e.input_kind not in INPUT_KINDS:
        raise ValueError(f"{where}: entry {e.id!r} has unknown input_kind {e.input_kind!r} (known: {INPUT_KINDS})")
    if e.mode in ("cnn", "descriptor") and not e.path:
        raise ValueError(f"{where}: {e.mode} entry {e.id!r} needs a checkpoint path")
    if e.mode == "descriptor" and e.stage != "type":
        raise ValueError(f"{where}: descriptor entry {e.id!r} must be a type entry, not {e.stage!r}")
    return e


def load_registry(path: Optional[Path] = None) -> List[StageEntry]:
    """Every entry of the registry file (default <repo>/models/registry.json). The file is either a JSON list of
    entries or an object with an "entries" list (plus free-form keys such as "ensembles" that this loader ignores).
    Raises FileNotFoundError for a missing file and ValueError for a malformed entry or a duplicated id."""
    path = Path(path) if path is not None else DEFAULT_REGISTRY
    if not path.is_file():
        raise FileNotFoundError(f"stage registry not found: {path}")
    data = json.loads(path.read_text())
    raw_entries = data["entries"] if isinstance(data, dict) else data
    entries = [_entry(raw, str(path)) for raw in raw_entries]
    seen = set()
    for e in entries:
        if (e.id, e.stage) in seen:
            raise ValueError(f"{path}: duplicate entry {e.id!r} for stage {e.stage!r}")
        seen.add((e.id, e.stage))
    return entries


def load_registry_meta(path: Optional[Path] = None) -> dict:
    """The registry file's non-entry keys (e.g. "ensembles": the combination rule chosen for the type stage)."""
    path = Path(path) if path is not None else DEFAULT_REGISTRY
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if k != "entries"} if isinstance(data, dict) else {}


def entries_for(registry: List[StageEntry], stage: str, kind: Optional[str] = None) -> List[StageEntry]:
    """Entries of `stage` (answering `kind`, if given), cheapest first; ties keep the file order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (known: {STAGES})")
    hits = [e for e in registry if e.stage == stage and (kind is None or kind in e.kinds)]
    return sorted(hits, key=lambda e: e.cost)
