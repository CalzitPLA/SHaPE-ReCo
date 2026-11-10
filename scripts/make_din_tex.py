#!/usr/bin/env python3
"""docs/generated/din_table.tex: the profile selection of dataset_gen/src/din_profiles.py with its self-check.

Columns: profile, h, b, tw, tf, r, tabulated area, area from the dimensions (parallel-flange formula) and its deviation,
the area of the section as FreeCAD builds it (dataset_gen/src/ibeam_freecad.make_ibeam_solid, 100 mm long, volume / 100;
measured with freecadcmd when it is installed, "--" otherwise) and its deviation, status (ok / flagged).
The file holds table ROWS only (its last row does not end with \\); docs/din_ibeam_profiles.tex wraps them.

    python3 scripts/make_din_tex.py [--no-freecad]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import din_profiles as dp  # noqa: E402

OUT = ROOT / "docs" / "generated" / "din_table.tex"


def freecad_areas(freecad: str) -> dict:
    """{name: section area in cm^2 of the FreeCAD-built solid (fillets, taper, toe radii)}."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "areas.json"
        code = (f"import sys, json; sys.path.insert(0, {str(ROOT / 'dataset_gen')!r})\n"
                "from src import din_profiles as dp, ibeam_freecad as fc\n"
                "res = {n: fc.make_ibeam_solid(dict(p.__dict__, toe_radius=dp.toe_radius(p)), 100.0).Volume / 1e4"
                " for n, p in dp.PROFILES.items()}\n"
                f"json.dump(res, open({str(out)!r}, 'w'))\n")
        script = Path(tmp) / "areas.py"
        script.write_text(code)
        subprocess.run([freecad, "-c", f"exec(open({str(script)!r}).read())"], check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
        return json.loads(out.read_text())


def _area(a: float) -> str:
    s = f"{a:.2f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def num(x: float, nd: int = 1) -> str:
    s = f"{x:.{nd}f}"
    return s[:-2] if nd == 1 and s.endswith(".0") else s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-freecad", action="store_true")
    args = ap.parse_args()
    from src.freecad_path import find_freecadcmd
    fc = find_freecadcmd()
    areas = freecad_areas(fc) if (fc and not args.no_freecad) else {}
    rows, last_series = [], None
    worst = {}
    for p in dp.select():
        dev = dp.area_deviation(p)
        a_fc = areas.get(p.name)
        dev_fc = (a_fc / p.area_cm2 - 1) if a_fc else None
        status = dp.self_check()[p.name][1]
        prefix = r"\midrule " if last_series is not None and p.series != last_series else ""
        last_series = p.series
        rows.append(prefix + " & ".join([
            p.name, num(p.h), num(p.b), num(p.tw), num(p.tf), num(p.r), _area(p.area_cm2),
            f"{dp.area_from_dims(p):.2f}", f"{100 * dev:+.2f}", f"{a_fc:.2f}" if a_fc else "--",
            f"{100 * dev_fc:+.2f}" if dev_fc is not None else "--", status]))
        key = "tapered" if p.flange_slope else "parallel"
        for tag, d in (("formula", dev), ("freecad", dev_fc)):
            if d is not None:
                worst[(key, tag)] = max(worst.get((key, tag), 0.0), abs(d))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(" \\\\\n".join(rows) + "\n")
    print(f"wrote {OUT} ({len(rows)} rows)")
    for k, v in sorted(worst.items()):
        print(f"max |deviation| {k}: {100 * v:.2f} %")


if __name__ == "__main__":
    main()
