# freecadcmd script: the manifest of a folder of STEP files (the ABC dataset's chunk), and the mesh of every single
# valid solid. Made 2026-09-28 for the laptop's subset of 500 (external_parts/abc/step_to_stl.py); since 2026-09-29 it
# takes a part of the file list (ABC_PART=i/n), so that several processes share a chunk of 10 000 files, and writes
# manifest.<i>.csv; scripts/abc_manifest_join.py joins them.
#   ABC_DIR=<folder with step/> [ABC_PART=0/8] [ABC_STL=0] freecadcmd scripts/abc_manifest.py
import csv
import glob
import os

import MeshPart
import Part

here = os.environ["ABC_DIR"]
part, of = (int(x) for x in os.environ.get("ABC_PART", "0/1").split("/"))
mesh = os.environ.get("ABC_STL", "1") != "0"
os.makedirs(here + "/stl", exist_ok=True)
FIELDS = ["stem", "step", "stl", "solids", "volume", "bbox", "faces", "free_form_faces", "status"]
paths = sorted(glob.glob(here + "/step/*.step"))[part::of]
name = here + ("/manifest.csv" if of == 1 else "/manifest.%d.csv" % part)
with open(name, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        row = dict.fromkeys(FIELDS, "")
        row.update(stem=stem, step="step/" + os.path.basename(path))
        try:
            sh = Part.read(path)
            row["solids"], row["faces"] = len(sh.Solids), len(sh.Faces)
            row["free_form_faces"] = sum(type(x.Surface).__name__ not in ("Plane", "Cylinder", "Cone", "Sphere", "Toroid")
                                         for x in sh.Faces)
            bb = sh.BoundBox
            row["bbox"] = "%.4g x %.4g x %.4g" % (bb.XLength, bb.YLength, bb.ZLength)
            if len(sh.Solids) != 1:
                row["status"] = "not_single_solid"
            elif not sh.Solids[0].isValid():
                row["status"] = "invalid_solid"
            else:
                row["volume"] = "%.6g" % sh.Solids[0].Volume
                if mesh:
                    m = MeshPart.meshFromShape(Shape=sh.Solids[0], LinearDeflection=max(bb.DiagonalLength * 1e-4, 1e-6),
                                               AngularDeflection=0.35)
                    m.write(here + "/stl/" + stem + ".stl")
                    row["stl"] = "stl/" + stem + ".stl"
                row["status"] = "ok"
        except Exception as e:
            row["status"] = "error: %s" % str(e)[:120]
        w.writerow(row)
        f.flush()
print("ABC_DONE", name, len(paths))
