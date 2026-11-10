#!/usr/bin/env python3
"""
Known CSG trees of machined parts, for the shape-class atlas's coverage gaps (2026-10-29: the real parts' largest classes
g4-9.M and g10+.M - 23 % of the in-scope ABC parts - have no known tree at all, because aegir's random primitive scenes
make no drilled blocks). A validate_known corpus (selection.json, truths/) that `validate_known.py build` builds and
gates and `known_chain.sh` runs, rated against the tree.

    python3 scripts/make_machined_trees.py --out ~/shape_reco_results/validation/machined_20261029 --n 60 --seed 29

Three families, each drawn to a target number of through holes (genus bins 1, 2-3, 4-9, 10+) and of leaves (4-7, 8-15,
16-31), round robin over the targets:
  block    a box (a plate to a block) with through holes (single, rows, grids), blind holes, pockets, bosses - planes and
           cylinders on several axes, the M family
  prism    the same box with box features only: through windows, pockets, steps - the P family
  turned   a cylinder with a coaxial step, an axial bore, grooves (tubes cut away), a flange with a bolt circle - the A family
Features stay inside the face with a margin and apart from each other, so a through feature adds one handle; the
measured class (scripts/shape_atlas.py describe) is what counts, the targets only spread the draws. Half of the parts are
turned by a random rotation (every leaf's placement composed with it); ABC parts are mostly axis-aligned.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

FULL = [0.0, 2 * math.pi]
HOLES = ((1, 1), (2, 3), (4, 9), (10, 16))
LEAVES = ((4, 7), (8, 15), (16, 31))
FAMILIES = ("block", "prism", "turned")
MARGIN = 3.0


def _leaf(prim: dict, R=None, t=(0.0, 0.0, 0.0)) -> dict:
    return dict(type="Placed", primitive=prim, rotation=(np.eye(3) if R is None else np.asarray(R)).tolist(),
                translation=[float(x) for x in t])


def _box(hx, hy, hz, c):
    return _leaf(dict(type="Box", half=[float(hx), float(hy), float(hz)]), t=c)


def _cyl(r, h, base, axis="z", inner=0.0):
    R = {"z": np.eye(3), "x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float),
         "y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)}[axis]
    return _leaf(dict(type="Cylinder", radius=float(r), height=float(h), u=FULL, inner_radius=float(inner)), R, base)


class Face:
    """The footprints on a rectangle |x| <= a, |y| <= b: circles and rectangles that keep MARGIN to the edge and each other."""
    def __init__(self, a, b):
        self.a, self.b, self.items = a, b, []

    def free(self, x, y, hx, hy):
        if abs(x) + hx > self.a - MARGIN or abs(y) + hy > self.b - MARGIN:
            return False
        return all(abs(x - u) >= hx + hu + MARGIN or abs(y - v) >= hy + hv + MARGIN for u, v, hu, hv in self.items)

    def take(self, x, y, hx, hy):
        self.items.append((x, y, hx, hy))


def _place(rng, face, hx, hy, tries=60):
    for _ in range(tries):
        x, y = rng.uniform(-face.a, face.a), rng.uniform(-face.b, face.b)
        if face.free(x, y, hx, hy):
            face.take(x, y, hx, hy)
            return x, y
    return None


def block(rng, holes, leaves, boxes_only=False):
    a, b = rng.uniform(40, 100), rng.uniform(30, 80)
    c = rng.uniform(6, 30)
    base = _box(a, b, c, (0, 0, 0))
    face = Face(a, b)
    adds, cuts, through = [], [], 0
    rmax = max(2.0, min(a, b) / 8)
    while through < holes:                                        # through features: single, a row or a grid
        left = holes - through
        if boxes_only:
            hx, hy = rng.uniform(3, a / 4), rng.uniform(3, b / 4)
            p = _place(rng, face, hx, hy)
            if p is None:
                break
            cuts.append(_box(hx, hy, c + 1, (p[0], p[1], 0)))
            through += 1
            continue
        r = rng.uniform(1.5, rmax)
        k = int(min(left, rng.choice([1, 2, 3, 4, 6, 8])))
        cols = int(math.ceil(math.sqrt(k))) if k > 3 else k
        rows = int(math.ceil(k / cols))
        pitch = 2 * r + MARGIN + rng.uniform(1, 6)
        w, h = (cols - 1) * pitch / 2 + r, (rows - 1) * pitch / 2 + r
        p = _place(rng, face, w, h)
        if p is None:
            if k == 1:
                break
            continue
        for i in range(k):
            x = p[0] - w + r + (i % cols) * pitch
            y = p[1] - h + r + (i // cols) * pitch
            cuts.append(_cyl(r, 2 * c + 2, (x, y, -c - 1)))
            through += 1
    while 1 + len(adds) + len(cuts) < leaves:                    # blind features: no handle
        kind = rng.choice(["pocket", "pocket", "step"] if boxes_only else ["blind", "blind", "pocket", "boss"])
        if kind in ("pocket", "step"):
            hx, hy = rng.uniform(3, a / 3), rng.uniform(3, b / 3)
            d = rng.uniform(0.2, 0.7) * 2 * c
            p = _place(rng, face, hx, hy)
            if p is None:
                break
            cuts.append(_box(hx, hy, d / 2 + 0.5, (p[0], p[1], c - d / 2 + 0.5)))
        elif kind == "blind":
            r = rng.uniform(1.5, rmax)
            d = rng.uniform(0.2, 0.7) * 2 * c
            p = _place(rng, face, r, r)
            if p is None:
                break
            cuts.append(_cyl(r, d + 1, (p[0], p[1], c - d)))
        else:
            r = rng.uniform(3, rmax * 1.5)
            p = _place(rng, face, r, r)
            if p is None:
                break
            adds.append(_cyl(r, rng.uniform(3, 20), (p[0], p[1], c - 0.5)))   # 0.5 into the top: one solid
    return base, adds, cuts


def turned(rng, holes, leaves):
    R, H = rng.uniform(15, 50), rng.uniform(30, 140)
    base = _cyl(R, H, (0, 0, 0))
    adds, cuts = [], []
    Rf, hf = R * rng.uniform(1.6, 2.4), rng.uniform(5, 15)        # a flange at the bottom, for the bolt circle
    adds.append(_cyl(Rf, hf, (0, 0, -hf + 0.5)))
    bore = rng.uniform(0.25, 0.5) * R
    cuts.append(_cyl(bore, H + hf + 32, (0, 0, -hf - 1)))        # the axial bore, through a top step too: one handle
    n = max(0, holes - 1)
    if n:
        rb = rng.uniform(1.5, min((Rf - R) / 3, math.pi * (R + Rf) / 2 / n / 2.5))
        rho = (R + Rf) / 2
        for i in range(n):
            phi = 2 * math.pi * i / n
            cuts.append(_cyl(rb, hf + 2, (rho * math.cos(phi), rho * math.sin(phi), -hf - 1)))
    if 1 + len(adds) + len(cuts) < leaves and rng.random() < 0.7:  # a coaxial step at the top
        adds.append(_cyl(R * rng.uniform(0.6, 0.85), rng.uniform(5, 30), (0, 0, H - 0.5)))
    zs = []
    while 1 + len(adds) + len(cuts) < leaves:                    # grooves round the shaft
        w, d = rng.uniform(1.5, 6), rng.uniform(0.1, 0.3) * (R - bore)
        z = rng.uniform(5, H - 5 - w)
        if any(abs(z - q) < w + 4 for q in zs):
            if len(zs) > H / 10:
                break
            continue
        zs.append(z)
        cuts.append(_cyl(R + 1, w, (0, 0, z), inner=R - d))
    return base, adds, cuts


def _rotate(node: dict, Rg: np.ndarray, tg: np.ndarray) -> dict:
    if node["type"] == "Placed":
        R, t = np.asarray(node["rotation"], float), np.asarray(node["translation"], float)
        return dict(node, rotation=(Rg @ R).tolist(), translation=(Rg @ t + tg).tolist())
    return dict(node, a=_rotate(node["a"], Rg, tg), b=_rotate(node["b"], Rg, tg))


def tree(base, adds, cuts) -> dict:
    node = base
    for x in adds:
        node = dict(type="Union", a=node, b=x)
    for x in cuts:
        node = dict(type="Subtraction", a=node, b=x)
    return node


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from scipy.spatial.transform import Rotation
    out = Path(a.out).expanduser()
    if out.exists():
        raise SystemExit(f"{out} exists")
    (out / "truths").mkdir(parents=True)
    rng = np.random.default_rng(a.seed)
    targets = [(f, h, l) for f in FAMILIES for h in HOLES for l in LEAVES]
    items = []
    for i in range(a.n):
        fam, (h0, h1), (l0, l1) = targets[i % len(targets)]
        holes, leaves = int(rng.integers(h0, h1 + 1)), int(rng.integers(l0, l1 + 1))
        leaves = max(leaves, holes + 1 if fam != "turned" else 2)
        base, adds, cuts = turned(rng, holes, leaves) if fam == "turned" else block(rng, holes, leaves, fam == "prism")
        prog = tree(base, adds, cuts)
        rotated = bool(rng.random() < 0.5)
        if rotated:
            prog = _rotate(prog, Rotation.random(random_state=int(rng.integers(1 << 31))).as_matrix(), rng.uniform(-50, 50, 3))
        n_leaves = 1 + len(adds) + len(cuts)
        stem = f"mach_{fam}_h{h0}_l{l0}_{i:03d}"
        meta = dict(stratum=f"{fam}/h{h0}-{h1}/l{l0}-{l1}", family=fam, target_holes=holes, target_leaves=leaves,
                    leaves=n_leaves, rotated=rotated, seed=a.seed, index=i)
        (out / "truths" / f"{stem}.json").write_text(json.dumps(dict(name=stem, source="machined", meta=meta, program=prog)))
        items.append(dict(stem=stem, source="machined", stratum=meta["stratum"], leaves=n_leaves, gate=None))
    sel = dict(name=out.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=a.seed, generator="make_machined_trees.py",
               items=items, excluded=[])
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    (out / "list.txt").write_text("")
    print(f"{len(items)} machined trees -> {out} (leaves {min(i['leaves'] for i in items)}-{max(i['leaves'] for i in items)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
