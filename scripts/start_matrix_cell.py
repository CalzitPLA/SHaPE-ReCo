#!/usr/bin/env python3
"""One cell of the starting-shape comparison: part x starting shape (docs/START_SHAPES.md).

    python3 scripts/start_matrix_cell.py <part> <start type> [--out ~/.cache/shape_reco/prof/start_matrix]

Runs the full loop from that start (scripts/time_cascade.py --grid tau --start <type>) and the voxel IoU of its FreeCAD
result against the part (scripts/voxel_iou.py), both through scripts/run_slot.sh (at most 3 heavy runs at once; the
frame and the liner exclusively). Writes rows/<part>__<type>.json: operations, residual, validity, chamfer, IoU, time,
the start's coverage / excess, the program. Parts: the names of PARTS below.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "dataset_gen" / "examples"
HOME = Path.home()
PARTS = {**{n: (E / f"{n}.stl", [], 0.1, False) for n in (
            "bs_sphere_flat", "cy_flat", "bs_box_dome", "side_slot", "bs_box_step", "box_with_bore", "cross_hole",
            "blind_hole", "co_countersink", "bs_box_sphere_pocket", "corner_pocket", "bs_box_two_pockets")},
         "liner": (HOME / ".cache/shape_reco/parts/Liner_LINER_001.step", ["--resolution", "192", "--max-resolution", "384"], 2.0, True),
         "frame": (HOME / "shape_reco_parts/benchmark_parts/1.0/Frame_4_x_6.stp", [], 4.0, True)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("part", choices=sorted(PARTS))
    ap.add_argument("start")
    ap.add_argument("--out", type=Path, default=HOME / ".cache/shape_reco/prof/start_matrix")
    args = ap.parse_args(argv)
    path, extra, pitch, heavy = PARTS[args.part]
    tag = f"{args.part}__{args.start.replace(' ', '_').replace('&', 'and')}"
    (args.out / "runs").mkdir(parents=True, exist_ok=True)
    (args.out / "rows").mkdir(parents=True, exist_ok=True)
    js, log = args.out / "runs" / f"{tag}.json", args.out / "runs" / f"{tag}.log"
    slot = [str(ROOT / "scripts" / "run_slot.sh")] + (["--exclusive"] if heavy else []) + ["--"]
    t0 = time.time()
    cmd = slot + ["nice", "-n", "10", "timeout", "3000", sys.executable, str(ROOT / "scripts" / "time_cascade.py"), str(path),
                  "--grid", "tau", "--start", args.start, *extra, "--json", str(js)]
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    row = dict(part=args.part, start=args.start, rc=rc, wall_total=time.time() - t0)
    if js.is_file():
        d = json.loads(js.read_text())
        s, g = d["summary"], d["summary"].get("grade") or {}
        first = next((st["message"] for st in d["steps"] if st["kind"] == "add"), "")
        row.update(operations=s["operations"], residual=s["residual"], valid=g.get("valid"), chamfer=g.get("chamfer"),
                   chamfer_max=g.get("chamfer_max"), seconds=s["elapsed"]["wall"], first_add=first[:400],
                   program=[f"{o['op']}:{o['kind']}" for o in d["program"]],
                   stopped=next((st["message"] for st in d["steps"] if st["kind"] == "done"), ""))
        stl = g.get("stl")
        if stl and Path(stl).is_file():
            out = subprocess.run(slot + [sys.executable, str(ROOT / "scripts" / "voxel_iou.py"), stl, str(path), "--pitch", str(pitch)],
                                 cwd=ROOT, capture_output=True, text=True)
            line = next((l for l in out.stdout.splitlines() if l.startswith("VOXEL_IOU")), None)
            if line:
                row["iou"] = json.loads(line.split(" ", 1)[1])
    else:
        row["error"] = log.read_text()[-1500:]
    (args.out / "rows" / f"{tag}.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("part", "start", "rc", "operations", "valid", "chamfer")}),
          "iou", (row.get("iou") or {}).get("iou"))


if __name__ == "__main__":
    main()
