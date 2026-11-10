#!/usr/bin/env python3
"""
The torus guard (T-29, 2026-10-01): tori the reconstruction must not propose or accept. FreeCAD builds every torus
below, but Geant4 refuses or mis-navigates them (the real-tree study, docs/reports/real_trees/real_trees.tex,
section "Defects"), so a program that holds one cannot go to the simulation.

Rules, for a torus of sweep (major) radius R and tube (minor) radius r:

  spindle/horn  r >= R. G4Torus::SetAllParameters refuses it (GeomSolids0002, "Invalid swept radius"). Seven of
                the 679 stored program trees hold one (R = 123, r = 1410 in known_00001964_001; the liner's
                90-operation run R = r = 2 mm, in the slot ends of its prismatic tool). The input's own B-rep can
                hold horn faces: 43 of the 1321 torus faces of the 500 ABC STEP files of the external set are horns
                (R = r exactly, in 9 parts), none is a spindle (census of 2026-10-01,
                docs/data/torus_faces_abc500.json); the liner's slot ends are such faces.
  near-horn     R - r < NEAR_HORN_GAP x R: the hole in the middle is a pin-hole. known_00002029_003 cuts a sphere of
                radius 31.8 with R = 2753, r = 2750 ((R - r) / R = 1.1e-3) - in effect "everything but a cylinder of
                radius 3" - and Geant4 navigates it inconsistently between equivalent trees.
  oversized     R + r > OVERSIZE_RATIO x the diagonal of the part's bounding box: a torus of which the part sees an
                arc so short that it stands in for a cylinder, a plane or the complement of a pin.

The thresholds lie between what real parts hold and what the defects hold (measured 2026-10-01; the census of the
real faces: FreeCAD on the 500 ABC STEP files of the external set, 134 parts with 1321 torus faces):

  NEAR_HORN_GAP = 0.01   real torus faces with r < R (1278): smallest (R - r) / R 0.085 (00006044, R 6.94, r 6.35);
                         the near-horn defect of the program trees 1.1e-3; their smallest admitted torus 0.185.
                         0.01 is about the geometric middle of 1.1e-3 and 0.085.
  OVERSIZE_RATIO = 1.5   real torus faces: largest (R + r) / diagonal 1.12 (00000812, a ring of R 293 on a part
                         of diagonal 283); the program trees: every torus above 1.08 that is no spindle comes from an
                         input without a single torus face, the smallest of them 1.84 (known_00002728_014). An arc of
                         a real ring of angle theta inside the part gives (R + r) / diagonal <= 1 / (2 sin(theta / 2)):
                         1.5 keeps every ring of which the part holds 39 degrees or more.

Counted by scripts/torus_guard_census.py (2026-10-01): of the 679 program trees 61 hold a torus and 18 are hit -
spindle/horn 7 (14 tori), near-horn 1, oversized 10 (14 tori); the run store's 20 runs: the liner's 90-operation run
(8 horn tori, all corner tori of its slot tools: thinned now, see below). Among the tori the ABC runs were offered
(their programs' `alternatives` too), 179 of 659 results hold a guarded one. Those offers also fill the space
between the thresholds and the real faces' edges ((R - r) / R 0.011 - 0.085, (R + r) / diagonal 1.12 - 1.31)
without matching a torus face of their input; the guard leaves them to the loop's checks.

A guarded torus is rejected, and its caller takes its normal path for a rejected candidate: the cascade's branch has
no answer, the loop's candidate fails its checks, a feature source's candidate is not offered, the sub-part cover's
torus is not that type's solid. A torus' parameters are its surface, so no repair keeps a horn or spindle surface.

One repair is natural all the same, `thinned`: the corner torus of a rounded corner (feature_stage.rounded_corners,
the contact tool and the prismatic groups), built from the input's own horn face - a slot whose round end's radius
is twice its floor's (the liner's slot ends, R = r = 2 mm). There only the torus' outer half counts (a cylinder of
radius R fills its inside), and a thinner tube about a wider centre circle with the same outer equator,
R' + r' = R + r and (R' - r') / R' = REPAIR_GAP, lies inside the horn torus (R' - R = r - r' = d: a point within r'
of the new centre circle is within r' + d = r of the old one), touches the wall where the face does and moves the
fillet by at most d, about NEAR_HORN_GAP x R (0.020 mm on the liner's R = r = 2). Measured 2026-10-01 on the test
slot of tests/test_prismatic_groups.py (the liner's: 8 wide, floor R 2, round ends R 4; 4e6 samples): the tool with
the 4 thinned corners is 0.40 mm3 smaller than with the horn tori, nothing of it outside them, and stays whole.
Rejected instead, the corners stayed as the box made them: the tool then cut 13.6 mm3 more (0.3 % of it, below the
feature phase's precision 0.999) and became partial.

Measured end to end 2026-10-01 on server_b (scripts/run_part.py, the ABC runs' settings, 1e5 sampled points), the
main line before the guard (bc2c371) against the guarded code, exact IoU (OCCT):
  00006874_001  two oversized tori (R 274 / 371, r 2) of straight bores: 0.99949 -> 1.00000, five cylinder cuts
  00003444_002  two oversized tori (R 315 / 238, r 2.4):                   0.99957 -> 0.99991, cylinders instead
  00002029_003  the near-horn torus (here R 1750, r 1747) that cut the bounding sphere down to the part's pin:
                0.533 -> 0.009 - without it the rough loop (no booleans in the rough stage) has nothing that removes
                the sphere's bulk (model 93 919 mm3 for an input of 872 mm3). The first add, not the guard, is wrong
                there; the torus hid it.

`scripts/torus_guard_census.py` counts the stored programs per rule with these functions.
"""

from __future__ import annotations

import math
from typing import Iterable, List, NamedTuple, Optional

NEAR_HORN_GAP = 0.01            # (R - r) / R below this: near-horn
OVERSIZE_RATIO = 1.5            # R + r above this x the part's bounding-box diagonal: oversized
REPAIR_GAP = 2.0 * NEAR_HORN_GAP  # (R' - r') / R' of a thinned corner torus: clear of the near-horn limit
RULES = ("degenerate", "spindle/horn", "near-horn", "oversized")


class TorusDefect(NamedTuple):
    rule: str                   # one of RULES
    text: str                   # the reason, for a rejected step / a failed source

    def __str__(self) -> str:
        return self.text


def torus_defect(major: float, minor: float, diagonal: Optional[float] = None) -> Optional[TorusDefect]:
    """The first rule a torus of sweep radius `major` (R) and tube radius `minor` (r) breaks, or None. `diagonal`:
    the part's bounding-box diagonal (same unit); without it the size rule is not applied."""
    R, r = float(major), float(minor)
    if not (math.isfinite(R) and math.isfinite(r)) or r <= 0.0:
        return TorusDefect("degenerate", f"degenerate torus: R = {R:.6g}, r = {r:.6g}")
    if r >= R:
        return TorusDefect("spindle/horn", f"{'horn' if r == R else 'spindle'} torus: tube radius r = {r:.6g} >= "
                                           f"sweep radius R = {R:.6g} (Geant4 refuses it)")
    if R - r < NEAR_HORN_GAP * R:
        return TorusDefect("near-horn", f"near-horn torus: R - r = {R - r:.4g} < {NEAR_HORN_GAP:g} R "
                                        f"(R = {R:.6g}, r = {r:.6g})")
    if diagonal is not None and math.isfinite(float(diagonal)) and float(diagonal) > 0.0 \
            and R + r > OVERSIZE_RATIO * float(diagonal):
        return TorusDefect("oversized", f"oversized torus: R + r = {R + r:.4g} > {OVERSIZE_RATIO:g} x the part's "
                                        f"diagonal {float(diagonal):.4g}")
    return None


def thinned(major: float, minor: float) -> Optional[tuple]:
    """(R', r') of the repair of a horn or near-horn corner torus (see the module's text): the same outer equator
    R' + r' = R + r, (R' - r') / R' = REPAIR_GAP; R' >= R and r' <= r, so the thinned torus lies inside the given
    one. None when the torus is admissible as it is or is no horn / near-horn (a spindle, r > R: no repair)."""
    R, r = float(major), float(minor)
    found = torus_defect(R, r)
    if found is None or found.rule not in ("spindle/horn", "near-horn") or r > R:
        return None
    S = R + r
    R2 = S / (2.0 - REPAIR_GAP)
    return R2, S - R2


def tori(p) -> List:
    """Every torus leaf of a schema Primitive (itself, or the leaves of a composite's tree)."""
    if p is None:
        return []
    if getattr(p, "kind", None) == "torus":
        return [p]
    out = []
    for q in getattr(p, "parts", None) or []:
        out += tori(q)
    return out


def primitive_defect(p, diagonal: Optional[float] = None) -> Optional[TorusDefect]:
    """The defect of the first guarded torus in a schema Primitive (a leaf or anywhere in a composite), or None."""
    for t in tori(p):
        d = t.dims_mm or {}
        found = torus_defect(d.get("major_radius", float("nan")), d.get("minor_radius", float("nan")), diagonal)
        if found is not None:
            return found
    return None


def admissible(p, diagonal: Optional[float] = None) -> bool:
    return primitive_defect(p, diagonal) is None


def diagonal_of(points) -> Optional[float]:
    """The bounding-box diagonal of a point set (N x 3), None for an empty one."""
    import numpy as np
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(P):
        return None
    return float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))


def program_tori(node, cut: bool = False) -> Iterable[tuple]:
    """(major, minor, under a cut) of every Torus leaf of a reco_field program dict (the trees of
    reco_engine.result_to_program.program_dict: Placed / Union / Subtraction / Intersection nodes)."""
    if not isinstance(node, dict):
        return
    if node.get("type") == "Placed":
        prim = node.get("primitive") or {}
        if prim.get("type") == "Torus":
            yield float(prim["major"]), float(prim["minor"]), cut
        else:
            yield from program_tori(prim, cut)
        return
    if "a" in node and "b" in node:
        yield from program_tori(node["a"], cut)
        yield from program_tori(node["b"], cut or node.get("type") == "Subtraction")
        return
    for key in ("children", "parts", "items"):
        for child in node.get(key) or []:
            yield from program_tori(child, cut)
