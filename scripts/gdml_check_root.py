#!/usr/bin/env python3
"""
A GDML file judged by ROOT: TGeoManager reads the file, and for every point of `<file>.points.txt` (x y z inside, mm,
written by scripts/export_gdml.py --points) ROOT says whether it finds the point inside the part.

    python3 scripts/gdml_check_root.py <folder with *.gdml and *.points.txt>

Needs ROOT with its Python module (2026-09-29 on the institute server server_a:
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-ubuntu2204-gcc11-opt/setup.sh, ROOT 6.38.00). No import of this
project: the script is copied to the machine that has ROOT.
"""
import glob
import os
import sys

import ROOT

ROOT.gErrorIgnoreLevel = ROOT.kWarning
print("ROOT", ROOT.gROOT.GetVersion())
total = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.gdml"))):
    name = os.path.basename(f)[:-5]
    geo = ROOT.TGeoManager.Import(f)
    if not geo:
        print(name, "NOT READ")
        total += 1
        continue
    unit = 0.1                                   # TGeo's length unit is cm, the file's is mm
    n = inside = differ = 0
    for line in open(f[:-5] + ".points.txt"):
        x, y, z, want = line.split()
        node = geo.FindNode(float(x) * unit, float(y) * unit, float(z) * unit)
        got = bool(node) and node.GetVolume().GetName().endswith("_volume")
        n += 1
        inside += int(got and want == "1")
        differ += int(got != (want == "1"))
    print("%-30s points %6d  inside %6d  judged differently by ROOT %d" % (name, n, inside, differ))
    total += differ
print("TOTAL differing:", total)
