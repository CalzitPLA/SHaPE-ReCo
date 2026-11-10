#!/usr/bin/env python3
"""Batrachotoxin - the poison of the Phyllobates dart frogs - as a 2D skeletal formula for the SHaPe ReCo logo.

The drawing is PubChem's (the user, 2026-11-10: "copy exactly the formula from pubchem"): CID 6324647, its 2D record
(record_type=2d), whose coordinates are the ones PubChem's own structure image is drawn from - PubChem serves that image
only as PNG, so it is redrawn here as vectors from the same coordinates, with PubChem's conventions:

- carbons are line vertices; a hydrogen is drawn (with its bond and an "H") when it sits on O or N or its bond is a
  stereo bond, every other one is left out; O, N and H are labels, the bonds stop short of them;
- a ring double bond is the bond plus a shorter inner line towards the ring's centre, any other double bond a centred
  pair; wedges (towards the viewer) and hashes (away) start narrow at the stereo centre (molfile convention);
- every bond is coloured half by each of its atoms (PubChem: C black, O red, N blue, H grey).

Only the proportions differ: PubChem draws 0.014-bond lines and 0.15-bond labels, which vanish in the badge, so the
badge uses the sizes below (labels and gaps about 2.6 x PubChem's) and its own colours on the navy sky (C and H cream,
O orange, N pink) with a navy halo. `python3 packaging/logo/molecule_2d.py` writes the formula alone in PubChem's
colours (batrachotoxin-2d.svg) for a side-by-side check against PubChem's image.

The record and the formula come through the PubChem access and SDF parser of the MoleculMaker FreeCAD macro
(~/Documents/MoleculMaker/MoleculMaker.FCMacro, usable without FreeCAD) and are cached next to this file
(batrachotoxin-2d.sdf, batrachotoxin.json), so the logo builds offline. The same macro built
Batrachotoxin/Batrachotoxin.FCStd: 81 atoms and 87 bonds (5 double), C31H42N2O6, as in the PubChem record.

  python3 packaging/logo/molecule_2d.py [--refetch]   fetch (if not cached) and write batrachotoxin-2d.svg
"""
from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "batrachotoxin"
SDF = HERE / "batrachotoxin-2d.sdf"
META = HERE / "batrachotoxin.json"
MACRO = Path.home() / "Documents" / "MoleculMaker" / "MoleculMaker.FCMacro"

CREAM, NAVY, PINK, ORANGE = "#FFF1DA", "#0E1B3D", "#FF4FA3", "#FF8A1F"
BADGE = {"C": CREAM, "H": CREAM, "O": ORANGE, "N": PINK}                 # on the navy sky
PUBCHEM = {"C": "#000000", "H": "#668B8B", "O": "#FF0000", "N": "#0000FF"}   # PubChem's image (sampled)

# sizes in bond lengths (PubChem's image: line 0.014, label 0.15, gap 0.10, pair 0.047, ring line 0.075, wedge 0.065)
LINE = 0.05         # bond line width
FONT = 0.40         # label font size
GAP = 0.26          # bond end clearance at a label
PAIR = 0.065        # half distance of the two lines of a centred double bond
RING = 0.19         # offset of the inner line of a ring double bond
SHORTEN = 0.15      # the inner line is shorter by this share at each end
WEDGE = 0.085       # half width of a stereo bond's wide end
HALO = 0.07         # the navy halo beyond the ink


# ---------------------------------------------------------------- data
def moleculmaker(path=MACRO):
    loader = importlib.machinery.SourceFileLoader("moleculmaker", str(path))
    spec = importlib.util.spec_from_loader("moleculmaker", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


def hill_formula(symbols):
    n = Counter(symbols)
    order = ["C", "H"] + sorted(e for e in n if e not in ("C", "H"))
    return "".join(f"{e}{n[e] if n[e] > 1 else ''}" for e in order if n[e])


def fetch(refetch=False):
    """PubChem 2D record + formula via MoleculMaker, cached; the formula is checked against the parsed atoms."""
    if SDF.exists() and META.exists() and not refetch:
        return json.loads(META.read_text())
    mm = moleculmaker()
    base = f"{mm.PUBCHEM}/name/{urllib.parse.quote(NAME)}"
    with urllib.request.urlopen(f"{base}/SDF?record_type=2d", timeout=30) as r:
        sdf = r.read().decode("utf-8")
    mol = mm.parse_sdf(sdf)
    with urllib.request.urlopen(f"{base}/property/MolecularFormula,MolecularWeight,IUPACName/JSON", timeout=30) as r:
        prop = json.load(r)["PropertyTable"]["Properties"][0]
    formula = hill_formula(sym for sym, _ in mol.atoms)
    if formula != prop["MolecularFormula"]:
        raise RuntimeError(f"PubChem formula {prop['MolecularFormula']} != atoms of the record {formula}")
    meta = {"name": "Batrachotoxin", "cid": prop["CID"], "formula": formula, "weight": prop["MolecularWeight"],
            "atoms": len(mol.atoms), "bonds": len(mol.bonds), "iupac": prop["IUPACName"],
            "source": "PubChem PUG REST, record_type=2d, fetched with MoleculMaker.FCMacro"}
    SDF.write_text(sdf)
    META.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def load():
    """Atoms (symbol, x, y) and bonds (i, j, order, stereo) of the cached V2000 record (stereo: 1 wedge, 6 hash)."""
    lines = SDF.read_text().splitlines()
    na, nb = int(lines[3][0:3]), int(lines[3][3:6])
    atoms = [(ln[31:34].strip(), float(ln[0:10]), float(ln[10:20])) for ln in lines[4:4 + na]]
    bonds = [(int(ln[0:3]) - 1, int(ln[3:6]) - 1, int(ln[6:9]), int(ln[9:12])) for ln in lines[4 + na:4 + na + nb]]
    return atoms, bonds


# ---------------------------------------------------------------- PubChem's depiction
def pubchem_drawing(refetch=False):
    """The formula as PubChem draws it, in the record's units with y down (as on screen): coloured line segments,
    wedge polygons, hash lines and labels (each with the symbol whose colour it takes), the bond length and points
    for the fit (bond lengths, y up)."""
    meta = fetch(refetch)
    atoms, bonds = load()
    sym = [a[0] for a in atoms]
    pos = [(a[1], -a[2]) for a in atoms]
    nbr = defaultdict(set)
    for i, j, _, _ in bonds:
        nbr[i].add(j)
        nbr[j].add(i)
    stereo_h = {k for i, j, _, st in bonds if st for k in (i, j) if sym[k] == "H"}
    shown = {k for k in range(len(sym))
             if sym[k] != "H" or k in stereo_h or any(sym[m] != "C" for m in nbr[k])}
    labelled = {k for k in shown if sym[k] != "C"}
    bond = statistics.median(math.dist(pos[i], pos[j]) for i, j, _, _ in bonds if sym[i] != "H" and sym[j] != "H")

    def ring_centre(a, b):
        """Centroid of the shortest ring through bond a-b (heavy atoms, up to 8), or None."""
        prev, q = {a: None}, deque([a])
        while q:
            u = q.popleft()
            for v in nbr[u]:
                if sym[v] == "H" or v in prev or (u == a and v == b):
                    continue
                prev[v] = u
                q.append(v)
        if b not in prev:
            return None
        path, u = [], b
        while u is not None:
            path.append(u)
            u = prev[u]
        if len(path) > 8:
            return None
        return (sum(pos[k][0] for k in path) / len(path), sum(pos[k][1] for k in path) / len(path))

    segs, wedges, hashes = [], [], []

    def line(p, q, mid, a, b):
        """p-q coloured as atom a up to the bond's midpoint `mid` (a share of p-q) and as atom b after it."""
        if sym[a] == sym[b] or not 0 < mid < 1:
            segs.append((p, q, sym[a] if mid >= 1 else sym[b]))
            return
        m = (p[0] + (q[0] - p[0]) * mid, p[1] + (q[1] - p[1]) * mid)
        segs.append((p, m, sym[a]))
        segs.append((m, q, sym[b]))

    for i, j, order, st in bonds:
        if i not in shown or j not in shown:
            continue
        (x0, y0), (x1, y1) = pos[i], pos[j]
        d = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / d, (y1 - y0) / d
        nx, ny = -uy, ux
        g0 = GAP if i in labelled else 0.0
        g1 = GAP if j in labelled else 0.0

        def along(t0, t1, off=0.0):
            """The bond from distance t0 to d - t1, shifted sideways by off; and where the midpoint falls on it."""
            p = (x0 + ux * t0 + nx * off, y0 + uy * t0 + ny * off)
            q = (x1 - ux * t1 + nx * off, y1 - uy * t1 + ny * off)
            return p, q, (d / 2 - t0) / max(d - t0 - t1, 1e-9)

        if st in (1, 6):                                  # narrow at i (the stereo centre), wide at j
            p, q, mid = along(0.0, g1)
            if st == 1:
                w = WEDGE
                tip, l1, r1 = p, (q[0] + nx * w, q[1] + ny * w), (q[0] - nx * w, q[1] - ny * w)
                if sym[i] == sym[j]:
                    wedges.append(([tip, l1, r1], sym[i]))
                else:                                     # split at the bond's midpoint like the lines
                    lm = (p[0] + (l1[0] - p[0]) * mid, p[1] + (l1[1] - p[1]) * mid)
                    rm = (p[0] + (r1[0] - p[0]) * mid, p[1] + (r1[1] - p[1]) * mid)
                    wedges.append(([tip, lm, rm], sym[i]))
                    wedges.append(([lm, l1, r1, rm], sym[j]))
            else:
                n = 8
                for k in range(1, n + 1):
                    t = k / n
                    cx, cy = p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t
                    w = WEDGE * t
                    hashes.append(((cx - nx * w, cy - ny * w), (cx + nx * w, cy + ny * w), sym[i] if t < mid else sym[j]))
        elif order == 2:
            c = ring_centre(i, j)
            if c is not None:                             # the bond, and a shorter inner line towards the ring
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                side = 1.0 if (c[0] - mx) * nx + (c[1] - my) * ny > 0 else -1.0
                line(*along(g0, g1), i, j)
                t = SHORTEN * d
                line(*along(max(t, g0), max(t, g1), side * RING), i, j)
            else:                                         # a centred pair (the ester's C=O)
                line(*along(g0, g1, PAIR), i, j)
                line(*along(g0, g1, -PAIR), i, j)
        else:
            line(*along(g0, g1), i, j)
    labels = [(pos[k], sym[k]) for k in sorted(labelled)]

    pts = set()
    for p, q, _ in segs:
        n = max(1, math.ceil(math.dist(p, q) / (0.25 * bond)))
        for k in range(n + 1):
            t = k / n
            pts.add((round((p[0] + (q[0] - p[0]) * t) / bond, 3), round(-(p[1] + (q[1] - p[1]) * t) / bond, 3)))
    for poly, _ in wedges:
        pts.update((round(x / bond, 3), round(-y / bond, 3)) for x, y in poly)
    for p, q, _ in hashes:
        pts.update((round(x / bond, 3), round(-y / bond, 3)) for x, y in (p, q))
    for (x, y), _ in labels:                              # the glyph's box around the atom
        r = 0.42 * FONT * bond
        pts.update((round((x + r * math.cos(a * math.pi / 4)) / bond, 3), round(-(y + r * math.sin(a * math.pi / 4)) / bond, 3))
                   for a in range(8))
    return {"segs": segs, "wedges": wedges, "hashes": hashes, "labels": labels, "bond": bond, "points": sorted(pts),
            "meta": meta, "shown": len(shown), "hydrogens": sorted(k for k in shown if sym[k] == "H")}


def drawing_svg(dr, theta, s, tx, ty, palette=BADGE, halo=NAVY, halo_opacity=0.45, font="Arial"):
    """SVG elements of the drawing placed by angle theta (rad), scale s (px per bond) and translation (tx, ty) of its
    origin - the placement fit() returns: a halo layer, then the ink coloured per atom by `palette`."""
    def n(v):
        return f"{v:.4f}"

    b = dr["bond"]
    tf = f"translate({tx:.2f},{ty:.2f}) rotate({-math.degrees(theta):.3f}) scale({s / b:.5f})"
    fs = FONT * b
    out = []
    for layer in ("halo", "ink"):
        extra = 2 * HALO * b if layer == "halo" else 0.0

        def col(e):
            return halo if layer == "halo" else palette.get(e, palette["C"])

        g = [f'<g transform="{tf}" opacity="{halo_opacity if layer == "halo" else 1}" stroke-linecap="round" '
             f'stroke-linejoin="round">']
        by = defaultdict(str)
        for p, q, e in dr["segs"]:
            by[col(e)] += f"M{n(p[0])},{n(p[1])}L{n(q[0])},{n(q[1])}"
        for c, d in by.items():
            g.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{n(LINE * b + extra)}"/>')
        hs = defaultdict(str)
        for p, q, e in dr["hashes"]:
            hs[col(e)] += f"M{n(p[0])},{n(p[1])}L{n(q[0])},{n(q[1])}"
        for c, d in hs.items():
            g.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{n(0.6 * LINE * b + extra)}"/>')
        for poly, e in dr["wedges"]:
            d = "M" + "L".join(f"{n(x)},{n(y)}" for x, y in poly) + "Z"
            g.append(f'<path d="{d}" fill="{col(e)}" stroke="{col(e)}" stroke-width="{n(0.4 * LINE * b + extra)}"/>')
        for (x, y), e in dr["labels"]:
            paint = (f'fill="{col(e)}" stroke="{col(e)}" stroke-width="{n(extra + 0.03 * b)}"' if layer == "halo"
                     else f'fill="{col(e)}"')
            g.append(f'<text x="{n(x)}" y="{n(y + 0.36 * fs)}" text-anchor="middle" font-family="{font}" '
                     f'font-size="{n(fs)}" {paint}>{e}</text>')
        g.append("</g>")
        out += g
    return out


# ---------------------------------------------------------------- placement in the badge
def fit(points, allowed, angles_deg=range(-20, 21, 4), s_range=(6.0, 30.0), shrink=0.9, step=2):
    """Largest placement of the formula inside a free region, as (theta, s, tx, ty, s_max).

    points: (x, y) on the drawing's ink in bond lengths, y up (pubchem_drawing()["points"]); allowed: a 2D distance
    map (pixels to the nearest forbidden pixel, numpy, [y, x]); every point keeps half a line and the halo clear of
    the forbidden pixels. Per angle the scale is found by bisection over translations on a step-pixel grid, restricted
    to the window where the rotated formula fits the free region's bounding box; the final scale is shrink x the best
    one, at the translation with the most clearance."""
    import numpy as np
    xy = np.array(points, dtype=float)[:, :2]
    hgt, wid = allowed.shape
    ys, xs = np.nonzero(allowed > 0)
    box = (xs.min(), xs.max(), ys.min(), ys.max())

    def clearance(theta, s):
        """(translations, worst clearance over the points per translation)."""
        ct, st = math.cos(theta), math.sin(theta)
        P = np.stack([s * (xy[:, 0] * ct - xy[:, 1] * st), -s * (xy[:, 0] * st + xy[:, 1] * ct)], axis=1)
        x0, x1 = box[0] - P[:, 0].min(), box[1] - P[:, 0].max()
        y0, y1 = box[2] - P[:, 1].min(), box[3] - P[:, 1].max()
        if x1 < x0 or y1 < y0:
            return None, np.array([-1.0])
        gy, gx = np.mgrid[y0:y1 + 1e-9:step, x0:x1 + 1e-9:step]
        T = np.stack([gx.ravel(), gy.ravel()], axis=1)
        Q = np.rint(T[:, None, :] + P[None, :, :]).astype(np.int32)
        qx, qy = np.clip(Q[..., 0], 0, wid - 1), np.clip(Q[..., 1], 0, hgt - 1)
        return T, allowed[qy, qx].min(axis=1) - ((LINE / 2 + HALO) * s + 1.0)

    best = None
    for a in sorted(angles_deg, key=abs):          # upright first: a tilt must be clearly larger to win
        th = math.radians(a)
        lo, hi = s_range
        if clearance(th, lo)[1].max() < 0:
            continue
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if clearance(th, mid)[1].max() >= 0 else (lo, mid)
        if best is None or lo > best[1] + 0.25:
            best = (th, lo)
    if best is None:
        raise RuntimeError("no room for the molecule")
    th, s = best[0], best[1] * shrink
    T, c = clearance(th, s)
    k = int(np.argmax(c))
    return th, s, float(T[k, 0]), float(T[k, 1]), best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refetch", action="store_true", help="fetch from PubChem even if cached")
    a = ap.parse_args()
    dr = pubchem_drawing(a.refetch)
    meta = dr["meta"]
    xs = [p[0] for p in dr["points"]]
    ys = [p[1] for p in dr["points"]]
    s, m = 60.0, 0.5
    w, h = (max(xs) - min(xs) + 2 * m) * s, (max(ys) - min(ys) + 2 * m) * s
    body = drawing_svg(dr, 0.0, s, (m - min(xs)) * s, (max(ys) + m) * s, palette=PUBCHEM, halo="#F5F5F5",
                       halo_opacity=1.0)
    out = HERE / "batrachotoxin-2d.svg"         # text outlined by inkscape when available (no font needed)
    out.write_text("\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
                              f'width="{w:.0f}" height="{h:.0f}">',
                              f"<title>{meta['name']} {meta['formula']}: PubChem's 2D depiction (CID {meta['cid']})</title>",
                              f'<rect width="100%" height="100%" fill="#F5F5F5"/>', *body, "</svg>", ""]))
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", str(out), "--export-text-to-path", "--export-plain-svg",
                        f"--export-filename={out}"], check=True, capture_output=True)
    print(f"{meta['name']}: {meta['formula']}, CID {meta['cid']}, {meta['atoms']} atoms, {meta['bonds']} bonds; "
          f"drawn: {dr['shown']} atoms ({len(dr['hydrogens'])} H), {len(dr['segs'])} line segments, "
          f"{len(dr['wedges'])} wedge parts, {len(dr['hashes'])} hash lines, {len(dr['labels'])} labels, "
          f"bond {dr['bond']:.3f} -> {out.name}")


if __name__ == "__main__":
    main()
