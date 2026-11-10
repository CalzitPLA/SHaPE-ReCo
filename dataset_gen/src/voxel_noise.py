#!/usr/bin/env python3
"""
Voxel-grid noise injection, applied in the voxelization layer -
directly to the {0,1} occupancy grid the voxelizer just produced
(stltovoxel + assemble_pngs_to_bin, or voxelization._voxelize_parity;
also sample_writer.finish_sample, which keeps the clean grid as
<stem>.clean.bin, and scripts/add_voxel_noise.py for an existing set),
before it is written to disk as a training sample's
.bin file. Without this, every training sample is pristine,
exactly-as-generated primitive geometry with a perfectly clean
boundary; real scanned/reconstructed input never looks that clean, so
the CNNs never see anything resembling the noise/occlusion/foreign-
object artifacts they'll actually face. Applying it here (not as a
training-time augmentation) means it is baked into the dataset itself
and inspectable/reproducible the same way any other generated sample
is.

Pure numpy, no scipy - matching reco_engine/placement.py and
reco_engine/local_search.py's existing "no scipy" constraint, since
this needs to survive the same isolated-Python environments they do.

Disabled by default: VoxelNoiseConfig()'s own defaults are all
zero/off, and dataset_generator.py only builds a non-default config
when the user passes one of the --noise-* CLI flags, so existing runs
and checkpoints are unaffected unless explicitly opted in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelNoiseConfig:
    """Four independently-tunable kinds of "spice", all off by
    default. Applied once, in this fixed order, so a given rng draw
    sequence always produces the same result: salt -> pepper -> blobs
    -> dilate/erode."""
    salt_prob: float = 0.0     # fraction of EMPTY voxels flipped to occupied (isolated speckle)
    pepper_prob: float = 0.0   # fraction of OCCUPIED voxels flipped to empty (dropout/occlusion)
    blob_count: int = 0        # number of small foreign-object cubes to splice in
    blob_min_size: int = 2     # blob cube side length range (voxels)
    blob_max_size: int = 6
    dilate_prob: float = 0.0   # probability of one 6-neighborhood dilation pass (thickens surfaces)
    erode_prob: float = 0.0    # probability of one 6-neighborhood erosion pass (thins surfaces)

    @property
    def enabled(self) -> bool:
        return (self.salt_prob > 0 or self.pepper_prob > 0 or self.blob_count > 0
                or self.dilate_prob > 0 or self.erode_prob > 0)


def _salt_and_pepper(volume: np.ndarray, salt_prob: float, pepper_prob: float,
                      rng: np.random.Generator) -> np.ndarray:
    out = volume.copy()
    if salt_prob > 0:
        flip = (out == 0) & (rng.random(out.shape) < salt_prob)
        out[flip] = 1
    if pepper_prob > 0:
        flip = (out == 1) & (rng.random(out.shape) < pepper_prob)
        out[flip] = 0
    return out


def _add_blobs(volume: np.ndarray, count: int, min_size: int, max_size: int,
               rng: np.random.Generator) -> np.ndarray:
    """Splice in `count` small solid cubes at random positions/sizes -
    a stand-in for a neighboring object, scan debris, or a scanning
    artifact, not just isolated-voxel speckle."""
    out = volume.copy()
    resolution = out.shape[0]
    for _ in range(count):
        size = min(int(rng.integers(min_size, max_size + 1)), resolution)
        lo = [int(rng.integers(0, resolution - size + 1)) for _ in range(3)]
        out[lo[0]:lo[0] + size, lo[1]:lo[1] + size, lo[2]:lo[2] + size] = 1
    return out


def _shift6(volume: np.ndarray):
    """The same 6-neighborhood (+-X/Y/Z) integer shift local_search.py
    uses for its translation search, reused here for dilate/erode."""
    for axis in range(3):
        for direction in (1, -1):
            shifted = np.zeros_like(volume)
            src, dst = [slice(None)] * 3, [slice(None)] * 3
            if direction == 1:
                src[axis], dst[axis] = slice(0, -1), slice(1, None)
            else:
                src[axis], dst[axis] = slice(1, None), slice(0, -1)
            shifted[tuple(dst)] = volume[tuple(src)]
            yield shifted


def _dilate_once(volume: np.ndarray) -> np.ndarray:
    out = volume
    for shifted in _shift6(volume):
        out = np.maximum(out, shifted)
    return out


def _erode_once(volume: np.ndarray) -> np.ndarray:
    out = volume
    for shifted in _shift6(volume):
        out = np.minimum(out, shifted)
    return out


def apply_voxel_noise(volume: np.ndarray, config: VoxelNoiseConfig,
                       rng: np.random.Generator) -> np.ndarray:
    """Corrupt a clean {0,1} occupancy grid per `config`. Never mutates
    `volume` in place; returns a new array of the same dtype/shape."""
    if not config.enabled:
        return volume

    out = volume
    if config.salt_prob > 0 or config.pepper_prob > 0:
        out = _salt_and_pepper(out, config.salt_prob, config.pepper_prob, rng)
    if config.blob_count > 0:
        out = _add_blobs(out, config.blob_count, config.blob_min_size, config.blob_max_size, rng)
    if config.dilate_prob > 0 and rng.random() < config.dilate_prob:
        out = _dilate_once(out)
    if config.erode_prob > 0 and rng.random() < config.erode_prob:
        out = _erode_once(out)
    return out.astype(volume.dtype)


# ------------------------------------------------------------------------------------------------ shared CLI
# The special-shape and I-beam generators write through sample_writer.finish_sample, which applies the noise itself
# (`noise=`): these helpers give them the SAME --noise-* flags as dataset_generator.py (whose own definitions stay as
# they are), so every generator of the v5 set can produce noisy and clean samples of every class.

def add_noise_arguments(parser) -> None:
    g = parser.add_argument_group("voxel noise (applied to the .bin after voxelization; all off by default)")
    g.add_argument("--noise-salt-prob", type=float, default=0.0, help="fraction of EMPTY voxels flipped to occupied")
    g.add_argument("--noise-pepper-prob", type=float, default=0.0, help="fraction of OCCUPIED voxels flipped to empty")
    g.add_argument("--noise-blob-count", type=int, default=0, help="foreign-object cubes spliced into the grid")
    g.add_argument("--noise-blob-min-size", type=int, default=2)
    g.add_argument("--noise-blob-max-size", type=int, default=6)
    g.add_argument("--noise-dilate-prob", type=float, default=0.0, help="probability of one 6-neighbourhood dilation")
    g.add_argument("--noise-erode-prob", type=float, default=0.0, help="probability of one 6-neighbourhood erosion")
    g.add_argument("--noise-prob", type=float, default=1.0,
                   help="probability that a sample is noised at all (per sample, independent of its class)")


def noise_spec_from_args(args):
    """{"config": VoxelNoiseConfig, "prob": p} for sample_writer.finish_sample(noise=...), or None when every flag is off."""
    cfg = VoxelNoiseConfig(salt_prob=args.noise_salt_prob, pepper_prob=args.noise_pepper_prob,
                           blob_count=args.noise_blob_count, blob_min_size=args.noise_blob_min_size,
                           blob_max_size=args.noise_blob_max_size, dilate_prob=args.noise_dilate_prob,
                           erode_prob=args.noise_erode_prob)
    return dict(config=cfg, prob=float(args.noise_prob)) if cfg.enabled else None
