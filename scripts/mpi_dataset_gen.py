#!/usr/bin/env python3
"""
MPI dataset generation for the v5 sets: the same three FreeCAD generators as scripts/make_dataset_v5.sh
(dataset_generator.py, ibeam_generator.py, special_generator.py), cut into small tasks and handed out by a dynamic work
queue - rank 0 schedules, every other rank runs one task (one core) at a time.

Why: the bash driver runs one process per generator, so the heavy ones (frames: up to 124 s per sample in FreeCAD) run
long after the light ones are done and most cores idle; and the splits run one after the other. Here every task is a
few samples, the heaviest go first (longest-processing-time order), and a rank asks for the next task as soon as it is free.

Top-up, not restart: rank 0 reads the manifests already on disk and schedules only what is MISSING per split and per
generator unit (a main-generator kind, a special subclass, the I-beam count), so an interrupted run or the bash driver's
partial output is kept. Every task writes to its own shard directory (shard<1000 + task id>) with its own seed range, so
records never collide and seeds stay disjoint from the bash driver's (seed + 0..20).

    mpirun -np 15 python3 scripts/mpi_dataset_gen.py --size medium                 # local: 14 workers + the scheduler
    mpirun --hostfile hosts -np 57 python3 scripts/mpi_dataset_gen.py --size full    # institute servers, see docs/SERVER_TRAINING.md

Every solid is still built by FreeCAD (freecadcmd) inside the generators; this script only schedules them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SIZES = {  # samples per subclass (train, val, test); field or not; I-beams per profile
    "small":  dict(n={"train": 20, "val": 5, "test": 10}, field=True, nprof={"train": 1, "val": 1, "test": 1}),
    "medium": dict(n={"train": 120, "val": 25, "test": 25}, field=False, nprof={"train": 6, "val": 5, "test": 3}),
}
SEED = {"train": 1000, "val": 2000, "test": 3000}
MAIN = [("box", 1, 0.0), ("cylinder", 2, 0.5), ("cone", 2, 0.5), ("sphere", 3, 0.67), ("torus", 3, 0.67)]
SPECIAL = ["box/hollow", "cylinder/hollow", "cone/hollow", "sphere/hollow", "torus/hollow", "cylinder/tube", "box/rhs",
           "box/frame", "cylinder/frame", "cone/frame", "sphere/frame", "torus/frame"]
PROFILES = {"train": "default", "val": "IPE 200,HEA 300,HEB 400,HEM 300,IPE 450",
            "test": "IPE 160,IPE 360,HEA 200,HEA 500,HEB 240,HEB 500,HEM 200,IPE 600,HEA 600,I 200"}
N_DEFAULT_PROFILES = 20
# rough seconds per sample in FreeCAD, for the heaviest-first order (measured: docs/thesis/FACTS.md, special smoke run)
COST = {"frame": 60.0, "hollow": 6.0, "tube": 4.0, "rhs": 4.0, "ibeam": 3.0, "main": 5.0}


def _records(split_dir: Path):
    """(shard name, record) for every manifest record under split_dir."""
    for meta in split_dir.glob("shard*/*/dataset_metadata.jsonl"):
        for line in meta.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "stl_path" in rec:
                yield meta.parent.parent.name, rec


def _unit(shard: str, rec: dict):
    """Which generator unit produced a record: ('main', kind), ('special', 'parent/subclass') or ('ibeam',); negres,
    plates and envelopes are by-products and count for nothing (their generator unit decides how many there are)."""
    if rec.get("sample_kind") == "negres" or rec.get("source") in ("ibeam_plate", "ibeam_envelope"):
        return None
    sub = rec.get("subclass", "full")
    if sub == "ibeam":
        return ("ibeam",)
    if sub in ("hollow", "tube", "rhs", "frame"):          # only the special generator makes these (main runs with --hollow-prob 0)
        return ("special", f"{rec.get('topology')}/{sub}")
    return ("main", rec.get("topology"))


def plan(out: Path, size: str, splits, chunk: int, first_id: int, skip=()):
    cfg = SIZES[size]
    tasks, tid = [], first_id
    for split in splits:
        split_dir = out / split
        have = Counter(u for u in (_unit(s, r) for s, r in _records(split_dir)) if u is not None)
        n = cfg["n"][split]
        base = SEED[split] + 500_000
        want = {("main", k): mult * n for k, mult, _ in MAIN}
        want.update({("special", s): n for s in SPECIAL})
        nprof = cfg["nprof"][split]
        n_prof = N_DEFAULT_PROFILES if PROFILES[split] == "default" else len(PROFILES[split].split(","))
        want[("ibeam",)] = nprof * n_prof
        for unit, target in want.items():
            if "/".join(unit[1:]) in skip or unit[0] in skip:
                continue
            missing = max(0, target - have.get(unit, 0))
            if unit == ("ibeam",):
                reps = -(-missing // n_prof) if missing else 0          # whole passes over the profile list
                for _ in range(reps):
                    tasks.append(dict(id=tid, split=split, unit=unit, n=1, seed=base + tid * 1000,
                                      cost=COST["ibeam"] * n_prof))
                    tid += 1
                continue
            while missing > 0:
                k = min(chunk, missing)
                sub = unit[1].split("/")[-1] if unit[0] == "special" else "main"
                tasks.append(dict(id=tid, split=split, unit=unit, n=k, seed=base + tid * 1000,
                                  cost=COST.get(sub, COST["main"]) * k))
                tid += 1
                missing -= k
    tasks.sort(key=lambda t: -t["cost"])                                  # heaviest first
    return tasks


def command(task, out: Path, size: str, freecad: str, with_field=None, min_voxels: int = 0):
    cfg = dict(SIZES[size])
    if with_field is not None:
        cfg["field"] = bool(with_field)
    split_dir = out / task["split"]
    shard = 1000 + task["id"]
    unit = task["unit"]
    if unit[0] == "main":
        kind = unit[1]
        seg = {k: s for k, _, s in MAIN}[kind]
        return (["python3", "src/dataset_generator.py", "--mode", "type", "--topologies", kind, "--kinds", kind,
                 "--count-per-type", str(task["n"]), "--macro-dir", "FCMacros",
                 "--output-dir", str(split_dir / f"shard{shard}"), "--freecad", freecad,
                 "--seed", str(task["seed"]), "--seed-offset", str(task["seed"]), "--voxelize", "--voxel-scale-mode", "fixed",
                 "--grid-mm", "40", "--segment-prob", str(seg), "--carve-prob", "0.4", "--cut-prob", "0.3",
                 "--hollow-prob", "0", "--emit-negative-residual"] + (["--with-field"] if cfg["field"] else [])
                + (["--min-voxels", str(min_voxels)] if min_voxels else []),
                ROOT / "dataset_gen")
    if unit[0] == "special":
        return (["python3", "-m", "dataset_gen.src.special_generator", "--out", str(split_dir), "--shard", str(shard),
                 "--subclasses", unit[1], "--n-per-subclass", str(task["n"]), "--seed", str(task["seed"]),
                 "--workers", "1", "--freecad", freecad] + ([] if cfg["field"] else ["--no-field"]), ROOT)
    return (["python3", "dataset_gen/src/ibeam_generator.py", "--out", str(split_dir), "--shard", str(shard),
             "--n-per-profile", "1", "--profiles", PROFILES[task["split"]], "--seed", str(task["seed"]),
             "--freecad", freecad, "--plate-fraction", "0.5", "--envelope-fraction", "0.25"]
            + ([] if cfg["field"] else ["--no-field"]), ROOT)


def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size_w = comm.Get_rank(), comm.Get_size()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=sorted(SIZES), required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--chunk", type=int, default=5, help="samples per task (smaller: better balance, more start-up cost)")
    ap.add_argument("--out", type=Path, default=None, help="default dataset_gen/local_dataset_v5/<size>")
    ap.add_argument("--freecad", default=None, help="default: $SHAPE_RECO_FREECADCMD, then PATH, then /usr/bin/freecadcmd")
    ap.add_argument("--first-id", type=int, default=None, help="first task id (default: after the highest shard on disk)")
    ap.add_argument("--dry-run", action="store_true", help="rank 0 prints the plan and stops")
    ap.add_argument("--freecad-worker", action="store_true",
                    help="SHAPE_RECO_FREECAD_WORKER=1: one long-lived freecadcmd per process instead of about 3 launches "
                         "per sample (reco_field/freecad_worker.py; measured 85.6 s -> 30.8 s for 4 samples, identical output)")
    ap.add_argument("--gpu-field", action="store_true",
                    help="let the ranks use CUDA for the field's nearest-centroid search (closest_point_field._knn_gpu). "
                         "Off by default: every rank would open its own CUDA context (a few hundred MB each), and 8+ ranks "
                         "next to a training run exhaust a 6 GB laptop GPU")
    ap.add_argument("--with-field", action="store_true",
                    help="force the vector field (and so the PLACED solid <stem>.solid.stl and the standard inside bits) "
                         "even for sizes whose default is occupancy only; main-generator samples need it for exact data")
    ap.add_argument("--min-voxels", type=int, default=0, help="passed to dataset_generator.py (redraw below it)")
    ap.add_argument("--voxelizer", choices=["stltovoxel", "parity"], default=None,
                    help="SHAPE_RECO_VOXELIZER for every rank (parity: the ground truth since 2026-09-23, CLAUDE.md)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="units not to top up, e.g. torus/frame (measured 2026-09-23: FreeCAD timed out at 300 s x 4 "
                         "attempts or produced open meshes on most torus frames, 20 min lost per failed sample)")
    args = ap.parse_args()
    if args.freecad is None:
        sys.path.insert(0, str(ROOT / "dataset_gen"))
        from src.freecad_path import freecadcmd_or_default
        args.freecad = freecadcmd_or_default()
    out = (args.out or ROOT / "dataset_gen" / "local_dataset_v5" / args.size).resolve()
    logs = out / "logs" / "mpi"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.freecad_worker:
        env["SHAPE_RECO_FREECAD_WORKER"] = "1"
    if args.voxelizer:
        env["SHAPE_RECO_VOXELIZER"] = args.voxelizer
    if not args.gpu_field:
        env["CUDA_VISIBLE_DEVICES"] = ""                                    # CPU cKDTree path, bit-identical output

    if rank == 0:
        first = args.first_id
        if first is None:
            ids = [int(p.name[5:]) - 1000 for p in out.glob("*/shard1*") if p.name[5:].isdigit() and int(p.name[5:]) >= 1000]
            first = max(ids) + 1 if ids else 0
        tasks = plan(out, args.size, args.splits, args.chunk, first, skip=set(args.skip))
        total_cost = sum(t["cost"] for t in tasks)
        print(f"[mpi] {len(tasks)} tasks, {size_w - 1} workers, estimated {total_cost / max(1, size_w - 1) / 60:.0f} min "
              f"(heaviest first); out {out}", flush=True)
        by = Counter((t["split"], t["unit"][0]) for t in tasks)
        print("[mpi] tasks per split/generator: " + ", ".join(f"{s}/{g} {c}" for (s, g), c in sorted(by.items())), flush=True)
        if args.dry_run or size_w < 2:
            for w in range(1, size_w):
                comm.send(None, dest=w)
            return
        logs.mkdir(parents=True, exist_ok=True)
        t0, pending, done, failed, active = time.time(), list(tasks), 0, [], size_w - 1
        status = MPI.Status()
        while active:
            # poll with pauses: a blocking recv spins at full speed in Open MPI and wasted a core (67 % CPU measured)
            while not comm.Iprobe(source=MPI.ANY_SOURCE, status=status):
                time.sleep(0.2)
            msg = comm.recv(source=status.Get_source(), status=status)
            src = status.Get_source()
            if msg is not None:
                done += 1
                if msg["rc"] != 0:
                    failed.append(msg)
                if done % 10 == 0 or msg["rc"] != 0:
                    print(f"[mpi] {done}/{len(tasks)} done, {len(failed)} failed, {time.time() - t0:.0f} s"
                          + (f"  FAILED task {msg['id']} ({msg['what']}) rc {msg['rc']}" if msg["rc"] else ""), flush=True)
            if pending:
                comm.send(pending.pop(0), dest=src)
            else:
                comm.send(None, dest=src)
                active -= 1
        print(f"[mpi] finished {done} tasks in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed", flush=True)
        (logs / "failed.json").write_text(json.dumps(failed, indent=1))
    else:
        comm.send(None, dest=0)                                             # ready
        while True:
            task = comm.recv(source=0)
            if task is None:
                break
            cmd, cwd = command(task, out, args.size, args.freecad, with_field=True if args.with_field else None,
                               min_voxels=args.min_voxels)
            what = f"{task['split']} {'/'.join(task['unit'])} x{task['n']}"
            log = logs / f"task{task['id']:05d}.log"
            with open(log, "w") as fh:
                rc = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
            comm.send(dict(id=task["id"], rc=rc, what=what, host=MPI.Get_processor_name()), dest=0)


if __name__ == "__main__":
    main()
