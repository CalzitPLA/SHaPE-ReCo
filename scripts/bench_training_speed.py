#!/usr/bin/env python3
"""Throughput of the training input and step, old path vs the opt-in fast path (training/gpu_encode.py, --amp), for the
training plan (docs/training_plan_v5.tex, F8). Needs an IDLE GPU for meaningful numbers: the GPU utilisation at the start
is recorded, and the run refuses above --max-busy unless --force.

    python3 scripts/bench_training_speed.py dataset_gen/local_dataset_v5/small/train --json docs/data/training_speed_bench.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import field_input as fi  # noqa: E402
from training import gpu_encode as ge  # noqa: E402
from training.backbone import Voxel3DBackbone  # noqa: E402


def gpu_busy() -> int:
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return -1


def step_ms(amp: bool, channels: int, batch: int, n: int = 15, cudnn_benchmark: bool = False) -> float:
    dev = "cuda"
    torch.backends.cudnn.benchmark = cudnn_benchmark            # the trainers' opt-in --cudnn-benchmark (train_common)
    m = Voxel3DBackbone(in_channels=channels).to(dev)
    head = torch.nn.Linear(256, 29).to(dev)
    opt = torch.optim.AdamW(list(m.parameters()) + list(head.parameters()), 3e-4)
    x = torch.randn(batch, channels, 64, 64, 64, device=dev)
    y = torch.randint(0, 29, (batch,), device=dev)
    if amp:
        m, x = m.to(memory_format=torch.channels_last_3d), x.contiguous(memory_format=torch.channels_last_3d)
    for i in range(n + 3):
        if i == 3:
            torch.cuda.synchronize()
            t = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            loss = torch.nn.functional.cross_entropy(head(m(x)).float(), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-busy", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    busy = gpu_busy()
    if busy > args.max_busy and not args.force:
        sys.exit(f"GPU is {busy} % busy - rerun on an idle GPU (or --force; the result records the load)")
    bins = [p for p in sorted(args.root.rglob("*.bin")) if not p.name.endswith(".clean.bin")][: args.n]
    grids = np.stack([np.fromfile(p, np.uint8).reshape(64, 64, 64) for p in bins])
    t = time.time()
    for g in grids:
        fi.encode(np.where(g > 0, 1.0, -1.0).astype(np.float32), "dirlog")
    cpu_ms = (time.time() - t) / len(grids) * 1000
    x = torch.as_tensor(grids, device="cuda")
    ge.encode_batch(x[: args.batch], "dirlog")                     # warm-up
    torch.cuda.synchronize()
    t = time.time()
    for lo in range(0, len(grids), args.batch):
        ge.encode_batch(x[lo:lo + args.batch], "dirlog_inside", inside=x[lo:lo + args.batch])
    torch.cuda.synchronize()
    gpu_ms = (time.time() - t) / len(grids) * 1000
    out = dict(gpu=torch.cuda.get_device_name(0), gpu_busy_percent_at_start=busy, n_grids=len(grids), batch=args.batch,
               dirlog_cpu_ms_per_sample=cpu_ms, dirlog_inside_gpu_ms_per_sample=gpu_ms, encode_speedup=cpu_ms / gpu_ms,
               step_ms_fp32=step_ms(False, 6, args.batch), step_ms_bf16_channels_last=step_ms(True, 6, args.batch),
               step_ms_bf16_channels_last_cudnn_benchmark=step_ms(True, 6, args.batch, cudnn_benchmark=True))
    out["step_speedup"] = out["step_ms_fp32"] / out["step_ms_bf16_channels_last"]
    out["cudnn_benchmark_speedup"] = out["step_ms_bf16_channels_last"] / out["step_ms_bf16_channels_last_cudnn_benchmark"]
    print(json.dumps(out, indent=1))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
