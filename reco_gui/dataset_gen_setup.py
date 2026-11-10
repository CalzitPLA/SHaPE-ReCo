#!/usr/bin/env python3
"""
Everything the dataset-generation GUI needs that is not Qt: the DatasetGenSettings record, turning it into
`src.dataset_generator` command lines (one per shard - dataset_gen/docs/RUNNING.md's own workflow is always several
parallel shards with `--seed base+i` into `output_dir/shardI`, run in parallel because FreeCAD is single-threaded
per sample), and summarizing what a finished (or still-running) output directory holds.

Kept import-light (stdlib + json only) so it is unit-testable without Qt or a FreeCAD install.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATASET_GEN = ROOT / "dataset_gen"


def default_freecadcmd() -> str:
    """The freecadcmd default of the settings: dataset_gen/src/freecad_path.py's rule ($SHAPE_RECO_FREECADCMD, then
    freecadcmd on PATH, then /usr/bin/freecadcmd). That module is standard library only, so this stays import-light."""
    import sys
    if str(DATASET_GEN) not in sys.path:
        sys.path.insert(0, str(DATASET_GEN))
    from src.freecad_path import freecadcmd_or_default
    return freecadcmd_or_default()


@dataclass
class DatasetGenSettings:
    mode: str = "type"                                  # "type" | "measure" | "rotation"
    topologies: Tuple[str, ...] = ("box", "sphere")
    count_per_type: int = 200                            # mode "type"
    measure_ratios_per_type: int = 20                    # mode "measure"
    count_per_measure: int = 10
    rotation_groups_per_type: int = 20                   # mode "rotation"
    count_per_rotation: int = 10

    output_dir: Path = Path("local_dataset")
    macro_dir: Path = DATASET_GEN / "FCMacros"
    freecad: str = field(default_factory=default_freecadcmd)
    scale_mm: float = 10.0
    seed: int = 0
    shards: int = 1                                       # >1: output_dir/shard0 .. shard{shards-1}, seed+i each
    resume: bool = False

    rotation_min: float = -180.0
    rotation_max: float = 180.0

    voxelize: bool = True
    voxel_resolution: int = 64
    voxel_scale_mode: str = "fixed"                       # "fit" | "fixed"
    grid_mm: float = 40.0
    size_fraction: Tuple[float, float] = (0.25, 0.9)
    position: str = "random"                              # "random" | "centered"

    full_field_prob: float = 0.0
    hollow_prob: float = 0.0
    hollow_wall: Tuple[float, float] = (0.08, 0.3)
    carve_prob: float = 0.0
    carve_size: Tuple[float, float] = (0.18, 0.42)
    carve_removed: Tuple[float, float] = (0.04, 0.5)
    cut_prob: float = 0.0
    cut_keep: Tuple[float, float] = (0.35, 0.95)
    cut_min_thickness: int = 4
    max_cuts: int = 2
    cut_axis_aligned_prob: float = 0.5
    require_manifold: bool = False
    noise_prob: float = 1.0

    noise_salt_prob: float = 0.0
    noise_pepper_prob: float = 0.0
    noise_blob_count: int = 0
    noise_blob_min_size: int = 2
    noise_blob_max_size: int = 5
    noise_dilate_prob: float = 0.0
    noise_erode_prob: float = 0.0

    export_slurm_job: Optional[str] = None                # write a SLURM script and exit, instead of generating


def build_args(s: DatasetGenSettings, output_dir: Path, seed: int) -> List[str]:
    """The `src.dataset_generator` argv for one shard (everything after the module name)."""
    args = ["--mode", s.mode, "--topologies", *s.topologies,
           "--macro-dir", str(s.macro_dir), "--output-dir", str(output_dir), "--freecad", s.freecad,
           "--scale-mm", str(s.scale_mm), "--seed", str(seed),
           "--rotation-min", str(s.rotation_min), "--rotation-max", str(s.rotation_max)]
    if s.mode == "type":
        args += ["--count-per-type", str(s.count_per_type)]
    elif s.mode == "measure":
        args += ["--measure-ratios-per-type", str(s.measure_ratios_per_type), "--count-per-measure", str(s.count_per_measure)]
    else:
        args += ["--rotation-groups-per-type", str(s.rotation_groups_per_type), "--count-per-rotation", str(s.count_per_rotation)]
    if s.resume:
        args.append("--resume")
    if s.export_slurm_job:
        return args + ["--export-slurm-job", s.export_slurm_job]
    if not s.voxelize:
        return args
    args += ["--voxelize", "--voxel-resolution", str(s.voxel_resolution), "--voxel-scale-mode", s.voxel_scale_mode]
    if s.voxel_scale_mode != "fixed":
        return args
    args += ["--grid-mm", str(s.grid_mm), "--size-fraction", str(s.size_fraction[0]), str(s.size_fraction[1]),
             "--position", s.position, "--full-field-prob", str(s.full_field_prob),
             "--hollow-prob", str(s.hollow_prob), "--hollow-wall", str(s.hollow_wall[0]), str(s.hollow_wall[1]),
             "--carve-prob", str(s.carve_prob), "--carve-size", str(s.carve_size[0]), str(s.carve_size[1]),
             "--carve-removed", str(s.carve_removed[0]), str(s.carve_removed[1]),
             "--cut-prob", str(s.cut_prob), "--cut-keep", str(s.cut_keep[0]), str(s.cut_keep[1]),
             "--cut-min-thickness", str(s.cut_min_thickness), "--max-cuts", str(s.max_cuts),
             "--cut-axis-aligned-prob", str(s.cut_axis_aligned_prob), "--noise-prob", str(s.noise_prob),
             "--noise-salt-prob", str(s.noise_salt_prob), "--noise-pepper-prob", str(s.noise_pepper_prob),
             "--noise-blob-count", str(s.noise_blob_count), "--noise-blob-min-size", str(s.noise_blob_min_size),
             "--noise-blob-max-size", str(s.noise_blob_max_size), "--noise-dilate-prob", str(s.noise_dilate_prob),
             "--noise-erode-prob", str(s.noise_erode_prob)]
    if s.require_manifold:
        args.append("--require-manifold")
    return args


@dataclass
class ShardJob:
    label: str          # "shard0" or "(single run)"
    output_dir: Path
    args: List[str]


def shard_jobs(s: DatasetGenSettings) -> List[ShardJob]:
    """One job per shard: `--shards 1` (the default) is a single plain run into `output_dir`; `--shards N>1` is
    dataset_gen/docs/RUNNING.md's own workflow, N parallel `output_dir/shardI` runs with `--seed base+i`, meant to
    be started together (FreeCAD is single-threaded per sample, so shards are how this project uses many cores)."""
    if s.shards <= 1:
        return [ShardJob("(single run)", Path(s.output_dir), build_args(s, Path(s.output_dir), s.seed))]
    jobs = []
    for i in range(s.shards):
        out = Path(s.output_dir) / f"shard{i}"
        jobs.append(ShardJob(f"shard{i}", out, build_args(s, out, s.seed + i)))
    return jobs


def summarize_output(output_dir: Path) -> Dict[str, object]:
    """Per-topology sample counts (and, for in-scale samples, how many were hollowed/carved/cut) from every
    dataset_metadata.jsonl under `output_dir` - works mid-run (only counts what has been written so far) and on a
    finished, possibly multi-shard, directory alike (training/manifest.py's own convention: recurse and read all
    of them, skipping the header record)."""
    per_topology: Dict[str, int] = collections.Counter()
    hollow = carve = cut = 0
    for meta_file in Path(output_dir).rglob("dataset_metadata.jsonl"):
        for line in meta_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("record_type") == "dataset_header":
                continue
            per_topology[entry["topology"]] += 1
            placement = entry.get("placement") or {}
            hollow += "hollow" in placement
            carve += "carve" in placement
            cut += bool(placement.get("cuts"))
    return {"total": sum(per_topology.values()), "per_topology": dict(per_topology),
           "hollow": hollow, "carve": carve, "cut": cut}
