"""The EXACT residual of a reconstruction program against its input (freecadcmd): r+ = input - model (material the model
misses), r- = model - input (material it invents), per connected solid, written as STL for voxel comparisons.

    RESIDUAL_JOB=job.json freecadcmd scripts/freecad_residual.py

job: {"input": input.step, "program": program.json (reco_field.freecad_builder format), "out_dir": dir,
      "scale": optional factor applied to the model about the origin (program units -> the STEP's mm)}
out_dir/residual.json: model volume/validity, and for r+ and r- the total volume and every solid's volume, bounding box
and a section estimate (volume / longest bounding-box side); out_dir/r_plus.stl, r_minus.stl, model.stl.
"""

import json
import os

import MeshPart
import Part

HERE = os.path.dirname(os.path.abspath(__file__))
job = json.load(open(os.environ["RESIDUAL_JOB"]))
src = open(os.path.join(HERE, "..", "reco_field", "freecad_builder.py")).read().rsplit("\nmain()", 1)[0]
ns = {"__name__": "freecad_builder_lib"}
exec(compile(src, "freecad_builder.py", "exec"), ns)          # the builder's own build(), without running its main()

inp = Part.read(job["input"])
model = ns["build"](json.load(open(job["program"]))).removeSplitter()
if job.get("scale", 1.0) != 1.0:                 # a program fitted in other units (the frame STL is 1/100 of the STEP)
    model = model.copy()
    model.scale(float(job["scale"]))
os.makedirs(job["out_dir"], exist_ok=True)


def solids_report(shape, name):
    solids = sorted(shape.Solids, key=lambda s: s.Volume, reverse=True)
    rows = []
    for s in solids:
        bb = s.BoundBox
        sides = sorted([bb.XLength, bb.YLength, bb.ZLength])
        rows.append(dict(volume=s.Volume, bbox=[bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax],
                         section=s.Volume / sides[-1] if sides[-1] > 0 else None, sides=sides))
    if solids:
        mesh = MeshPart.meshFromShape(Shape=Part.makeCompound(solids), LinearDeflection=0.5, AngularDeflection=0.1,
                                      Relative=False)
        mesh.write(os.path.join(job["out_dir"], name + ".stl"))
    return dict(volume=sum(r["volume"] for r in rows), n=len(rows), solids=rows)


mesh = MeshPart.meshFromShape(Shape=model, LinearDeflection=0.5, AngularDeflection=0.1, Relative=False)
mesh.write(os.path.join(job["out_dir"], "model.stl"))
out = dict(input_volume=inp.Volume, model_volume=model.Volume, model_valid=bool(model.isValid()),
           model_solids=len(model.Solids),
           r_plus=solids_report(inp.cut(model), "r_plus"), r_minus=solids_report(model.cut(inp), "r_minus"))
json.dump(out, open(os.path.join(job["out_dir"], "residual.json"), "w"), indent=1)
