#!/usr/bin/env python3
"""
The frame's root fillets as boolean tools, built and judged exactly (for docs/reports/frame_solution).

Every piece of the exact difference input - model carries its exact faces; a concave cylinder face of the INPUT (radius
27 mm on the frame) is a root fillet. Its tool is a box minus a cylinder (D-001): the square between the cylinder's axis
and the corner the fillet fills, along the axis over the piece's extent, minus the cylinder itself. The tools are fused
to the model by FreeCAD and the result is judged like every model (reco_field.exact_check.exact_residual).

    python3 scripts/frame_report_fillets.py <input.step> <model.step> <exact_residual.json> <out_dir>
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import freecadcmd_or_default  # noqa: E402  ($SHAPE_RECO_FREECADCMD, PATH, /usr/bin)

FREECAD = """
import json, os, FreeCAD, Part
job = json.load(open(os.environ["JOB"]))
model = Part.Shape(); model.read(job["model"])
tools = []
for t in job["tools"]:
    box = Part.makeBox(*t["size"], FreeCAD.Vector(*t["corner"]))
    cyl = Part.makeCylinder(t["radius"], t["length"] + 2 * t["over"], FreeCAD.Vector(*t["start"]), FreeCAD.Vector(*t["axis"]))
    tools.append(box.cut(cyl))
fused = model.fuse(tools).removeSplitter()
fused.exportStep(job["out_step"]); fused.exportStl(job["out_stl"])
Part.makeCompound(tools).exportStep(job["tools_step"])
json.dump(dict(valid=fused.isValid(), solids=len(fused.Solids), volume=fused.Volume, faces=len(fused.Faces),
               tools_volume=sum(x.Volume for x in tools), model_faces=len(model.Faces)), open(job["out_json"], "w"))
"""


def fillet_tools(pieces):
    tools = []
    for p in pieces:
        lo, hi = np.asarray(p["bbox"][0], float), np.asarray(p["bbox"][1], float)
        for f in p["faces"]:
            if f["type"] != "cylinder" or f.get("convex") or f.get("origin") != "input":
                continue
            a = np.asarray(f["axis"], float)
            k = int(np.argmax(np.abs(a)))                       # the frame's members lie along the world axes
            if abs(abs(a[k]) - 1.0) > 1e-6:
                raise ValueError(f"fillet axis {a} is not a world axis: this script handles axis-parallel fillets only")
            r, c, n = float(f["radius"]), np.asarray(f["center"], float), np.asarray(f["normal"], float)
            side = -np.sign(n)                                   # the input's normal points to the axis: material is opposite
            side[k] = 0.0
            corner, size = c.copy(), np.full(3, r)
            for i in range(3):
                if i == k:
                    corner[i], size[i] = lo[i], hi[i] - lo[i]
                elif side[i] < 0:
                    corner[i] = c[i] - r
            start = c.copy()
            start[k] = lo[k] - 1.0
            tools.append(dict(piece=p["id"], corner=corner.tolist(), size=size.tolist(), radius=r, length=float(hi[k] - lo[k]), over=1.0,
                              start=start.tolist(), axis=[float(i == k) for i in range(3)]))
    return tools


def main():
    step_in, model, residual, out = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    tools = fillet_tools(residual["pieces"])
    job = dict(model=str(model), tools=tools, out_step=str(out / "frame_with_fillets.step"), out_stl=str(out / "frame_with_fillets.stl"),
               tools_step=str(out / "fillet_tools.step"), out_json=str(out / "build.json"))
    (out / "job.json").write_text(json.dumps(job))
    (out / "build.py").write_text(FREECAD)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["JOB"] = str(out / "job.json")
    t = time.time()
    proc = subprocess.run([freecadcmd_or_default(), str(out / "build.py")], env=env, capture_output=True, text=True, timeout=1800)
    build_s = time.time() - t
    if not (out / "build.json").exists():
        raise SystemExit("FreeCAD build failed: " + (proc.stderr or proc.stdout)[-600:])
    build = json.loads((out / "build.json").read_text())
    from reco_field.exact_check import exact_residual
    t = time.time()
    res = exact_residual(step_in, job["out_step"], pieces_dir=out / "pieces_after", max_pieces=500)
    summary = dict(tools=len(tools), build_seconds=build_s, judge_seconds=time.time() - t, build=build,
                   after={k: v for k, v in res.items() if not isinstance(v, (list, dict))},
                   before={k: v for k, v in residual.items() if not isinstance(v, (list, dict))},
                   pieces_after=[dict(id=p["id"], sign=p["sign"], volume=p["volume"], bbox=p["bbox"]) for p in res.get("pieces", [])])
    (out / "fillets.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "pieces_after"}, indent=1, default=str))


if __name__ == "__main__":
    main()
