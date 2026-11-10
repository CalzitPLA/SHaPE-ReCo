#!/usr/bin/env python3
"""Shared training-loop plumbing for the three CNN trainers.

Centralizes the pieces that were duplicated (and buggy) across the old
FeatureNet scripts: argument parsing, the train/val split, the
optimizer/scheduler setup, NaN detection, checkpointing, and history
logging.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from training.manifest import Record, group_split, load_manifest


def common_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=Path, required=True,
                         help="Root of a dataset_gen output tree (voxelized, --voxelize)")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15,
                         help="Early-stopping patience in epochs (0 disables)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default=None,
                         help="cuda/cpu, default: cuda if available else cpu")
    return parser


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_split(dataset_root: Path, val_fraction: float, seed: int) -> tuple[List[Record], List[Record]]:
    records = load_manifest(dataset_root, require_voxel=True)
    if not records:
        raise RuntimeError(
            f"No voxelized samples found under {dataset_root}. "
            f"Generate with dataset_generator.py's --voxelize flag first."
        )
    train, val = group_split(records, val_fraction, seed)
    if not val:
        raise RuntimeError("Validation split is empty - increase --val-fraction or dataset size.")
    return train, val


@dataclass
class TrainState:
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        "train_loss": [], "val_loss": [], "lr": [],
    })
    best_val_loss: float = float("inf")
    epochs_without_improvement: int = 0


# ------------------------------------------------------------------------------------------------ throughput (opt-in)
# --gpu-encode: the loader ships uint8 grids ((B,R,R,R), or (B,2,R,R,R) = [grid, inside bits] for dirlog_inside) and the
# input is encoded on the device by training/gpu_encode.py (CPU dirlog ~374 ms/sample in every epoch -> ~7 ms on the GPU).
# --amp: bf16 autocast (no GradScaler: bf16 needs none; the old NaN history came from fp16 + GradScaler, backbone.py) and
# channels_last_3d memory layout; measured 301 -> 88 ms per step (backbone, 6 channels, batch 16). Losses stay float32.

@dataclass
class Speed:
    gpu_kind: Optional[str] = None
    amp: bool = False
    cudnn_benchmark: bool = False


SPEED = Speed()


def add_speed_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu-encode", action="store_true",
                        help="encode grid-derived inputs (occupancy, dirlog, dirlog_inside) on the GPU (training/gpu_encode.py)")
    parser.add_argument("--amp", action="store_true", help="bf16 autocast + channels_last_3d (opt-in until its A/B passes)")
    parser.add_argument("--cudnn-benchmark", action="store_true",
                        help="torch.backends.cudnn.benchmark: let cuDNN pick the fastest 3-D conv algorithm for the fixed "
                             "64^3 shapes (not bit-identical between algorithms; unmeasured until the GPU is idle - "
                             "scripts/bench_training_speed.py reports it)")


    parser.add_argument("--no-cudnn", action="store_true",
                        help="convolutions by torch's own CUDA kernels instead of cuDNN: for a GPU the installed cuDNN "
                             "has no engine for (workstation's GTX 970: 'GET was unable to find an engine'); measured there "
                             "0.98 s per step (batch 16, 6 channels, fp32) against 2.52 s on its 16 CPU threads")


def configure_speed(args, input_kind: str, model: torch.nn.Module) -> torch.nn.Module:
    """Set SPEED from the parsed flags; returns the model (in channels_last_3d under --amp)."""
    from training.gpu_encode import GPU_KINDS
    if getattr(args, "gpu_encode", False):
        if input_kind not in GPU_KINDS:
            raise ValueError(f"--gpu-encode supports {GPU_KINDS}, not {input_kind!r}")
        SPEED.gpu_kind = input_kind
    SPEED.amp = bool(getattr(args, "amp", False))
    SPEED.cudnn_benchmark = bool(getattr(args, "cudnn_benchmark", False))
    if getattr(args, "no_cudnn", False):
        torch.backends.cudnn.enabled = False
    if SPEED.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    return model.to(memory_format=torch.channels_last_3d) if SPEED.amp else model


def prepare_input(x: torch.Tensor, device) -> torch.Tensor:
    """The network input on `device`: encoded there under --gpu-encode, channels_last_3d under --amp."""
    x = x.to(device, non_blocking=True)
    if SPEED.gpu_kind is not None:
        from training.gpu_encode import encode_batch
        grid, inside = (x[:, 0], x[:, 1]) if x.dim() == 5 else (x, None)
        x = encode_batch(grid, SPEED.gpu_kind, inside=inside)
    return x.contiguous(memory_format=torch.channels_last_3d) if SPEED.amp else x


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give a dataset's own `rng` (numpy) a seed of its worker. Without it every forked worker
    copies the same generator and draws the SAME augmentation sequence (measured: 4 workers, identical rotations), and
    no run is reproducible. torch.initial_seed() differs per worker and per epoch and follows torch.manual_seed."""
    import numpy as np
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)


def loader_kwargs(device, num_workers: int, seed: int) -> dict:
    """pin_memory / persistent_workers / per-worker seeding for a DataLoader, and a seeded generator for the shuffle."""
    g = torch.Generator()
    g.manual_seed(int(seed))
    return dict(pin_memory=getattr(device, "type", str(device)) == "cuda", persistent_workers=num_workers > 0,
                worker_init_fn=seed_worker, generator=g)


def autocast():
    """bf16 autocast under --amp, a no-op otherwise."""
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=SPEED.amp and torch.cuda.is_available())


def check_finite(loss: torch.Tensor, step_desc: str) -> None:
    """Fail fast on NaN/Inf instead of silently logging garbage for the
    rest of training (this is exactly what happened in the old
    ShaPe_Integra_v2.py run: NaN from epoch 1, logged for all 15 epochs
    unnoticed until someone inspected training_history.json).
    """
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"Non-finite loss ({loss.item()}) at {step_desc}. "
            f"Stopping immediately rather than continuing to train on "
            f"garbage - lower --lr, check for bad input data, or disable "
            f"AMP if enabled."
        )


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer, epoch: int, extra: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        **extra,
    }, path)


def save_history(path: Path, history: Dict[str, List[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def run_training_loop(
    *,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_dir: Path,
    train_step: Callable,
    eval_step: Callable,
    extra_metrics_key: Optional[str] = None,
) -> None:
    """Generic epoch loop. train_step/eval_step take (model, batch, device)
    and return a dict with at least {"loss": scalar_tensor}; any other
    keys are averaged and logged into history under the same name.
    """
    state = TrainState()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_totals: Dict[str, float] = {}
        n_train = 0
        for batch in train_loader:
            metrics = train_step(model, batch, device, optimizer)
            bs = batch[0].size(0) if isinstance(batch, (list, tuple)) else 1
            n_train += bs
            for k, v in metrics.items():
                train_totals[k] = train_totals.get(k, 0.0) + float(v) * bs

        model.eval()
        val_totals: Dict[str, float] = {}
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                metrics = eval_step(model, batch, device)
                bs = batch[0].size(0) if isinstance(batch, (list, tuple)) else 1
                n_val += bs
                for k, v in metrics.items():
                    val_totals[k] = val_totals.get(k, 0.0) + float(v) * bs

        train_avg = {k: v / max(n_train, 1) for k, v in train_totals.items()}
        val_avg = {k: v / max(n_val, 1) for k, v in val_totals.items()}

        current_lr = optimizer.param_groups[0]["lr"]
        state.history.setdefault("lr", []).append(current_lr)
        for k, v in train_avg.items():
            state.history.setdefault(f"train_{k}", []).append(v)
        for k, v in val_avg.items():
            state.history.setdefault(f"val_{k}", []).append(v)

        val_loss = val_avg["loss"]
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        extra_str = ""
        if extra_metrics_key and extra_metrics_key in val_avg:
            extra_str = f" {extra_metrics_key}={val_avg[extra_metrics_key]:.4f}"
        print(
            f"epoch {epoch:4d}/{epochs} "
            f"train_loss={train_avg['loss']:.4f} val_loss={val_loss:.4f}"
            f"{extra_str} lr={current_lr:.2e} ({elapsed:.1f}s)"
        )

        save_checkpoint(checkpoint_dir / "checkpoint_latest.pth", model, optimizer, epoch, {"val_loss": val_loss})

        if val_loss < state.best_val_loss:
            state.best_val_loss = val_loss
            state.epochs_without_improvement = 0
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, epoch, {"val_loss": val_loss})
        else:
            state.epochs_without_improvement += 1

        save_history(checkpoint_dir / "training_history.json", state.history)

        if patience > 0 and state.epochs_without_improvement >= patience:
            print(f"Early stopping: no improvement for {patience} epochs.")
            break

    save_checkpoint(checkpoint_dir / "final_model.pth", model, optimizer, epoch, {"val_loss": val_loss})
