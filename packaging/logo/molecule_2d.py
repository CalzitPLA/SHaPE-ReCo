#!/usr/bin/env python3
"""Batrachotoxin - the poison of the Phyllobates dart frogs - as a 2D skeletal formula for the SHaPe ReCo logo.

The drawing is Wikipedia's: the skeletal formula the article "Batrachotoxin" shows, Wikimedia Commons
File:Batrachotoxin_skeletal.svg by Ed (Edgar181), public domain (no attribution required; credited here anyway). It is
cached unchanged next to this file (batrachotoxin-skeletal-wikipedia.svg; source, licence and SHA-1 in
batrachotoxin.json["skeletal"]), so the logo builds offline. Its geometry stays as drawn (bonds, wedges, hashes, the
gaps where bonds cross, the Arial labels at their places); for the badge it is only restyled: cream ink with O orange
and N pink, every outline thickened by BOLD (Wikipedia's lines are 0.035 bond lengths wide and vanish at icon size),
a navy halo HALO wide.

The name and sum formula on the badge's ring come from PubChem (CID 6324647) through the PubChem access and SDF
parser of the MoleculMaker FreeCAD macro (~/Documents/MoleculMaker/MoleculMaker.FCMacro, usable without FreeCAD),
cached in batrachotoxin-2d.sdf and batrachotoxin.json. The same macro built Batrachotoxin/Batrachotoxin.FCStd: 81
atoms and 87 bonds (5 double), C31H42N2O6, as in the PubChem record.

  python3 packaging/logo/molecule_2d.py [--refetch]   fetch (if not cached) and write batrachotoxin-2d.svg
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import re
import shutil
import statistics
import subprocess
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAME = "batrachotoxin"
SDF = HERE / "batrachotoxin-2d.sdf"
META = HERE / "batrachotoxin.json"
MACRO = Path.home() / "Documents" / "MoleculMaker" / "MoleculMaker.FCMacro"
WIKI_FILE = "File:Batrachotoxin_skeletal.svg"            # the article's infobox image
WIKI_SVG = HERE / "batrachotoxin-skeletal-wikipedia.svg"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "shape-reco-logo-build/1.0"                  # Wikimedia asks every client for one

CREAM, NAVY, PINK, ORANGE = "#FFF1DA", "#0E1B3D", "#FF4FA3", "#FF8A1F"
ELEMENT_COLOUR = {"O": ORANGE, "N": PINK}
BOLD = 0.035        # stroke laid on every outline, in bond lengths: the lines 0.035 -> 0.07 wide
HALO = 0.08         # the halo beyond the ink, in bond lengths
ARIAL = {"H": 0.722, "O": 0.778, "N": 0.722}               # advance widths (em) of the label glyphs; cap height 0.716
SVG_NS = "{http://www.w3.org/2000/svg}"


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


def _meta():
    return json.loads(META.read_text()) if META.exists() else {}


def fetch(refetch=False):
    """PubChem 2D record + formula via MoleculMaker, cached; the formula is checked against the parsed atoms."""
    if SDF.exists() and META.exists() and not refetch:
        return _meta()
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
    meta = _meta()                                       # keeps the "skeletal" entry
    meta.update({"name": "Batrachotoxin", "cid": prop["CID"], "formula": formula, "weight": prop["MolecularWeight"],
                 "atoms": len(mol.atoms), "bonds": len(mol.bonds), "iupac": prop["IUPACName"],
                 "source": "PubChem PUG REST, record_type=2d, fetched with MoleculMaker.FCMacro"})
    SDF.write_text(sdf)
    META.write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_wikipedia(refetch=False):
    """Wikipedia's skeletal formula (the Commons file), cached unchanged; source and licence go to
    batrachotoxin.json["skeletal"], the download is checked against the SHA-1 Commons reports."""
    meta = _meta()
    if WIKI_SVG.exists() and "skeletal" in meta and not refetch:
        return meta["skeletal"]
    q = urllib.parse.urlencode({"action": "query", "titles": WIKI_FILE, "prop": "imageinfo",
                                "iiprop": "url|sha1|extmetadata", "format": "json"})
    page = next(iter(json.loads(_get(f"{COMMONS_API}?{q}"))["query"]["pages"].values()))
    info = page["imageinfo"][0]
    url = info["url"].split("?")[0]
    data = _get(url)
    if hashlib.sha1(data).hexdigest() != info["sha1"]:
        raise RuntimeError(f"{WIKI_FILE}: SHA-1 of the download differs from Commons' {info['sha1']}")

    def plain(key):
        return re.sub(r"<[^>]+>", "", info["extmetadata"].get(key, {}).get("value", "")).strip()

    meta["skeletal"] = {"file": WIKI_FILE, "page": f"https://commons.wikimedia.org/wiki/{WIKI_FILE}", "url": url,
                        "sha1": info["sha1"], "artist": plain("Artist"), "licence": plain("LicenseShortName"),
                        "shown_in": "https://en.wikipedia.org/wiki/Batrachotoxin"}
    WIKI_SVG.write_bytes(data)
    META.write_text(json.dumps(meta, indent=1) + "\n")
    return meta["skeletal"]


# ---------------------------------------------------------------- Wikipedia's drawing
def _subpaths(d):
    """The coordinate pairs of an absolute M/L/Q/C/Z path (what Batik writes), one list per subpath."""
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:[eE]-?\d+)?", d)
    out, nums = [], []
    for t in tokens:
        if t.isalpha():
            if t not in "MLQCZ":
                raise ValueError(f"path command {t} not handled")
            if t == "M":
                out.append([])
            continue
        nums.append(float(t))
        if len(nums) == 2:
            out[-1].append(tuple(nums))
            nums = []
    return out


def wikipedia_drawing(refetch=False):
    """Wikipedia's formula as drawn, in the file's pixels (y down): its filled shapes (bonds, wedges and hashes;
    Batik writes even the lines as outlines) with their clip, the labels, the clip paths that cut the gaps where
    bonds cross (the canvas-sized no-op clips are dropped), the bond length (px), the source and points for the fit
    (bond lengths, y up)."""
    import xml.etree.ElementTree as ET
    source = fetch_wikipedia(refetch)
    root = ET.parse(WIKI_SVG).getroot()
    clips = {}
    for k, cp in enumerate(root.iter(SVG_NS + "clipPath")):
        d = cp.find(SVG_NS + "path").get("d")
        if len(_subpaths(d)) > 1:                       # the bond outline cut out of the canvas
            clips[cp.get("id")] = (f"btx-gap{k}", d)
    shapes, labels = [], []

    def walk(e, off, size, family):
        for c in e:
            tag = c.tag[len(SVG_NS):]
            if tag in ("defs", "clipPath"):
                continue
            o = off
            if c.get("transform"):
                m = re.fullmatch(r"translate\((-?[\d.]+),(-?[\d.]+)\)", c.get("transform"))
                if not m:
                    raise ValueError(f"transform {c.get('transform')} not handled")
                o = (off[0] + float(m[1]), off[1] + float(m[2]))
            sz = float(c.get("font-size", f"{size}px").rstrip("px"))
            fam = c.get("font-family", family).strip("'")
            cid = re.fullmatch(r"url\(#(.+)\)", c.get("clip-path", "url(#-)"))[1]
            if tag == "path":
                if o != (0.0, 0.0):
                    raise ValueError("translated path not handled")
                shapes.append((c.get("d"), clips.get(cid, (None,))[0]))
            elif tag == "text":
                labels.append((o[0] + float(c.get("x", 0)), o[1] + float(c.get("y", 0)), c.text.strip(), sz, fam))
            walk(c, o, sz, fam)

    walk(root, (0.0, 0.0), 12.0, "Arial")
    firsts = []                                         # a bond outline starts with one of its long sides
    for d, _ in shapes:
        for sp in _subpaths(d):
            if len(sp) >= 2 and math.dist(sp[0], sp[1]) > 20:
                firsts.append(math.dist(sp[0], sp[1]))
    bond = statistics.median(firsts)
    pts = set()
    for d, _ in shapes:
        for sp in _subpaths(d):
            for p, q in zip(sp, sp[1:] + sp[:1]):
                n = max(1, math.ceil(math.dist(p, q) / (0.25 * bond)))
                for i in range(n):
                    t = i / n
                    pts.add((round((p[0] + (q[0] - p[0]) * t) / bond, 2), round(-(p[1] + (q[1] - p[1]) * t) / bond, 2)))
    for x, y, text, size, _ in labels:
        w = sum(ARIAL.get(ch, 0.75) for ch in text) * size
        for i in range(5):
            for j in range(3):
                pts.add((round((x + w * i / 4) / bond, 2), round(-(y - 0.716 * size * j / 2) / bond, 2)))
    return {"shapes": shapes, "labels": labels, "clips": list(clips.values()), "bond": bond, "source": source,
            "points": sorted(pts)}


def drawing_svg(dr, theta, s, tx, ty, colour=CREAM, halo=NAVY, halo_opacity=0.45):
    """SVG elements of the drawing placed by angle theta (rad), scale s (px per bond) and translation (tx, ty) of
    its origin - the same placement fit() returns: a halo layer, then the ink (labels per element coloured)."""
    def n(v):
        return f"{v:.3f}"

    b = dr["bond"]
    tf = f"translate({tx:.2f},{ty:.2f}) rotate({n(-math.degrees(theta))}) scale({s / b:.5f})"
    out = ["<defs>", *(f'<clipPath id="{cid}" clipPathUnits="userSpaceOnUse"><path d="{d}"/></clipPath>'
                       for cid, d in dr["clips"]), "</defs>"]
    for ink, width, opacity in ((halo, (BOLD + 2 * HALO) * b, halo_opacity), (None, BOLD * b, 1.0)):
        base = ink or colour
        g = [f'<g opacity="{opacity}" transform="{tf}" fill="{base}" stroke="{base}" stroke-width="{n(width)}" '
             f'stroke-linejoin="round" stroke-linecap="round">']
        for d, cid in dr["shapes"]:
            g.append(f'<path d="{d}"' + (f' clip-path="url(#{cid})"' if cid else "") + "/>")
        for x, y, text, size, family in dr["labels"]:
            body = text if ink else "".join(
                f'<tspan fill="{c}" stroke="{c}">{ch}</tspan>' for ch in text
                for c in (ELEMENT_COLOUR.get(ch, colour),))
            g.append(f'<text x="{n(x)}" y="{n(y)}" font-family="{family}" font-size="{n(size)}">{body}</text>')
        g.append("</g>")
        out += g
    return out


# ---------------------------------------------------------------- placement in the badge
def fit(points, allowed, angles_deg=range(-20, 21, 4), s_range=(6.0, 30.0), shrink=0.9, step=2):
    """Largest placement of the formula inside a free region, as (theta, s, tx, ty, s_max).

    points: (x, y) on the drawing's ink in bond lengths, y up (wikipedia_drawing()["points"]); allowed: a 2D distance
    map (pixels to the nearest forbidden pixel, numpy, [y, x]); every point keeps the ink's thickening and the halo
    clear of the forbidden pixels. Per angle the scale is found by bisection over translations on a step-pixel grid,
    restricted to the window where the rotated formula fits the free region's bounding box; the final scale is
    shrink x the best one, at the translation with the most clearance."""
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
        return T, allowed[qy, qx].min(axis=1) - ((BOLD / 2 + HALO) * s + 1.0)

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
    ap.add_argument("--refetch", action="store_true", help="fetch from PubChem and Wikimedia Commons even if cached")
    a = ap.parse_args()
    meta = fetch(a.refetch)
    dr = wikipedia_drawing(a.refetch)
    xs = [p[0] for p in dr["points"]]
    ys = [p[1] for p in dr["points"]]
    s, m = 40.0, 0.6
    w, h = (max(xs) - min(xs) + 2 * m) * s, (max(ys) - min(ys) + 2 * m) * s
    body = drawing_svg(dr, 0.0, s, (m - min(xs)) * s, (max(ys) + m) * s, colour=NAVY, halo="#FFFFFF",
                       halo_opacity=1.0)
    src = dr["source"]
    out = HERE / "batrachotoxin-2d.svg"         # text outlined by inkscape when available (no font needed)
    out.write_text("\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
                              f'width="{w:.0f}" height="{h:.0f}">',
                              f"<title>{meta['name']} {meta['formula']}: the skeletal formula of Wikipedia "
                              f"({src['file']}, {src['artist']}, {src['licence']})</title>",
                              *body, "</svg>", ""]))
    if shutil.which("inkscape"):
        subprocess.run(["inkscape", str(out), "--export-text-to-path", "--export-plain-svg",
                        f"--export-filename={out}"], check=True, capture_output=True)
    print(f"{meta['name']}: {meta['formula']}, CID {meta['cid']}; drawing {src['file']} ({src['artist']}, "
          f"{src['licence']}): {len(dr['shapes'])} shapes, {len(dr['labels'])} labels, {len(dr['clips'])} gap clips, "
          f"bond {dr['bond']:.1f} px, {len(dr['points'])} fit points -> {out.name}")


if __name__ == "__main__":
    main()
