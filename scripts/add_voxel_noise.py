#!/usr/bin/env python3
"""
Add voxel noise to a finished dataset after the fact: the project rule is that every class gets noisy AND clean samples
(CLAUDE.md), and noise lives in the voxelization layer - so a set generated without --noise-* flags (the v5 medium set,
2026-09-23) can be noised later without rebuilding any solid.

Mirrors dataset_gen/src/sample_writer.finish_sample(noise=...) exactly: per sample an rng seeded from crc32 of the STL's
path relative to the split directory; with probability --noise-prob the .bin is replaced by apply_voxel_noise(clean) (the
clean grid is kept as <stem>.clean.bin); the manifest record gets placement.noise (bool), placement.voxels_clean,
placement.voxels, voxel_volume_mm3 and placement.noise_posthoc = the noise config (the marker that makes a second run a
no-op). The .field.npz, if any, stays the clean solid's, as in sample_writer.

Run it only when no generator is still appending to the manifests (the JSONL files are rewritten atomically).

    python3 scripts/add_voxel_noise.py dataset_gen/local_dataset_v5/medium/{train,val,test} \
        --noise-salt-prob 0.0005 --noise-pepper-prob 0.02 --noise-blob-count 2 --noise-dilate-prob 0.25 \
        --noise-erode-prob 0.25 --noise-prob 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.voxel_noise import add_noise_arguments, apply_voxel_noise, noise_spec_from_args  # noqa: E402


def _stl(meta: Path, split_dir: Path, rel: str) -> Path:
    """A record's STL, resolved like training/manifest.py: relative to the metadata file's ancestors, nearest first."""
    for base in [meta.parent, *meta.parents]:
        p = base / rel
        if p.exists():
            return p
        if base == split_dir:
            break
    return split_dir / rel


def noise_split(split_dir: Path, spec: dict, resolution: int) -> Counter:
    stats = Counter()
    cfg = spec["config"]
    for meta in sorted(split_dir.glob("shard*/*/dataset_metadata.jsonl")):
        lines, changed = meta.read_text().splitlines(), False
        out = []
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            if "stl_path" not in rec or not rec.get("voxel_path"):
                out.append(line)
                continue
            placement = dict(rec.get("placement") or {})
            if "noise_posthoc" in placement:
                stats["already"] += 1
                out.append(line)
                continue
            stl = _stl(meta, split_dir, rec["stl_path"])
            bin_path = stl.with_suffix(".bin")
            if not bin_path.exists():
                stats["missing_bin"] += 1
                out.append(line)
                continue
            clean = np.fromfile(bin_path, dtype=np.uint8)
            if clean.size != resolution ** 3:
                stats["bad_size"] += 1
                out.append(line)
                continue
            volume = clean.reshape((resolution,) * 3)
            rng = np.random.default_rng(zlib.crc32(str(stl.relative_to(split_dir)).encode()))
            noisy = bool(rng.random() < float(spec["prob"]))
            placement["voxels_clean"] = int((volume > 0).sum())
            if noisy:
                clean_copy = bin_path.with_name(bin_path.stem + ".clean.bin")
                if not clean_copy.exists():
                    volume.astype(np.uint8).tofile(clean_copy)
                volume = apply_voxel_noise(volume, cfg, rng)
                volume.astype(np.uint8).tofile(bin_path)
            placement["noise"] = noisy
            placement["voxels"] = int((volume > 0).sum())
            placement["noise_posthoc"] = dict(asdict(cfg), prob=float(spec["prob"]))
            grid_mm = float(placement.get("grid_mm", 40.0))
            rec["voxel_volume_mm3"] = float(placement["voxels"] * (grid_mm / resolution) ** 3)
            rec["placement"] = placement
            out.append(json.dumps(rec, separators=(",", ":")))
            changed = True
            stats["noised" if noisy else "clean"] += 1
            stats[f"{rec.get('topology')}/{rec.get('subclass', 'full')}:{'noisy' if noisy else 'clean'}"] += 1
        if changed:
            tmp = meta.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(out) + "\n")
            os.replace(tmp, meta)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="+", type=Path, help="split directories (containing shard*/<topology>/)")
    ap.add_argument("--resolution", type=int, default=64)
    add_noise_arguments(ap)
    args = ap.parse_args()
    spec = noise_spec_from_args(args)
    if spec is None:
        ap.error("all --noise-* flags are off: nothing to do")
    for split in args.splits:
        st = noise_split(split.resolve(), spec, args.resolution)
        per_class = sorted(k for k in st if ":" in k)
        without_both = sorted({k.split(":")[0] for k in per_class}
                              - {k.split(":")[0] for k in per_class if k.endswith("noisy")}
                              | ({k.split(":")[0] for k in per_class} - {k.split(":")[0] for k in per_class if k.endswith("clean")}))
        print(f"{split}: noised {st['noised']}, clean {st['clean']}, already {st['already']}, missing .bin "
              f"{st['missing_bin']}, bad size {st['bad_size']}; classes lacking a noisy or a clean sample: {without_both or 'none'}")


if __name__ == "__main__":
    main()
