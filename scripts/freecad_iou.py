"""The exact IoU of a built model against the part (freecadcmd, OCCT booleans with a fuzzy tolerance).

    IOU_JOB=job.json freecadcmd scripts/freecad_iou.py

job: {"model": model.step (e.g. the cascade loop's grade output, ReconstructionResult.step_path), "input_step": part.step,
      "scale": 100.0 (model -> input units; the frame STL is at 1/100), "fuzzy": 0.01 (mm, in input units)}
Prints IOU_RESULT {model, input, common, union, iou, missing, invented} (volumes in input units^3). Without the fuzzy
tolerance the booleans of a box program whose faces coincide with the part's (plates on flanges) fail silently (common 0).
Checked 2026-09-24: the 96^3 cube program of the frame gives IoU 0.9101, missing 7.93e7 mm^3 - the recorded values.
"""
import json
import os

import FreeCAD
import Part

job = json.load(open(os.environ["IOU_JOB"]))
mdl = Part.read(job["model"])
scale = float(job.get("scale", 1.0))
if scale != 1.0:
    mdl.scale(scale, FreeCAD.Vector(0, 0, 0))
inp = Part.read(job["input_step"])
mdl = mdl.Solids[0] if len(mdl.Solids) == 1 else mdl
inp = inp.Solids[0] if len(inp.Solids) == 1 else inp
tol = float(job.get("fuzzy", 0.01))
common = mdl.common([inp], tol).Volume
union = mdl.fuse([inp], tol).Volume
print("IOU_RESULT " + json.dumps(dict(model=mdl.Volume, input=inp.Volume, common=common, union=union,
                                      iou=common / union if union > 0 else 0.0, missing=inp.Volume - common,
                                      invented=mdl.Volume - common)))
