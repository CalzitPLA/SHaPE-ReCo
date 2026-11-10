#!/usr/bin/env python3
"""Standard hot-rolled I-beam sections (DIN 1025-1...5, today EN 10365) for the I-beam training data.

!! PROVENANCE: every number in PROFILES was written down FROM MEMORY (no standard sheet was available to the
!! author). They are checked in only two ways:
!!   1. area consistency: for every parallel-flange row (IPE, HEA, HEB, HEM) the section area computed from the
!!      dimensions, area_from_dims = 2 b tf + (h - 2 tf) tw + (4 - pi) r^2, lies within 1 % of the tabulated area
!!      (self_check() below; tests/test_din_profiles.py enforces it). A wrong h, b, tw, tf or r would in most cases
!!      break that - but not always (two compensating errors, or a wrong area copied with the same wrong dimension);
!!   2. the frame measurement: Frame_4_x_6.stp contains a 490 x 300, tf 23 section of area 197.6 cm^2, which is HEA 500
!!      as listed here (197.5 cm^2).
!! A human should spot-check the rows against EN 10365 (or a producer catalogue such as ArcelorMittal's) before the
!! values are quoted anywhere as authoritative.

Conventions (mm, cm^2): h depth, b flange width, tw web thickness, tf flange thickness, r root radius (web-flange
fillet). Series names: "IPE" (DIN 1025-5), "HEA" (DIN 1025-3, old IPBl), "HEB" (DIN 1025-2, IPB), "HEM" (DIN 1025-4,
IPBv), "I" (DIN 1025-1, INP, narrow I beams with TAPERED flanges).

The "I" series: the inner flange faces slope at 14 % (flange_slope = 0.14), tf is the MEAN flange thickness (the value
DIN 1025-1 tabulates), r is the root radius r1 (= tw for this series) and the flange toes carry a second radius
r2 = 0.6 tf (toe_radius()). The parallel-flange area formula does not describe that section; it overshoots the
tabulated area by about 2 % for every row, consistently, so these rows are FLAGGED (FLAGS) rather than checked. Their
real check is the FreeCAD-built section (dataset_gen/src/ibeam_freecad.py), which models the taper and both radii.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class IProfile:
    name: str
    series: str                 # "IPE" | "HEA" | "HEB" | "HEM" | "I"
    standard: str               # e.g. "DIN 1025-3 (IPBl) / EN 10365"
    h: float
    b: float
    tw: float
    tf: float
    r: float                    # root radius
    area_cm2: float
    flange_slope: float = 0.0   # 0.14 for the DIN 1025-1 narrow I series (tapered flanges), else 0


STANDARDS = {
    "I": "DIN 1025-1 (I, INP) / EN 10365",
    "HEB": "DIN 1025-2 (IPB) / EN 10365",
    "HEA": "DIN 1025-3 (IPBl) / EN 10365",
    "HEM": "DIN 1025-4 (IPBv) / EN 10365",
    "IPE": "DIN 1025-5 (IPE) / EN 10365",
}

#            name      h      b     tw     tf    r    A [cm^2]
_ROWS = {
    "IPE": """
    IPE 80     80    46    3.8    5.2    5    7.64
    IPE 100   100    55    4.1    5.7    7   10.3
    IPE 120   120    64    4.4    6.3    7   13.2
    IPE 140   140    73    4.7    6.9    7   16.4
    IPE 160   160    82    5.0    7.4    9   20.1
    IPE 180   180    91    5.3    8.0    9   23.9
    IPE 200   200   100    5.6    8.5   12   28.5
    IPE 220   220   110    5.9    9.2   12   33.4
    IPE 240   240   120    6.2    9.8   15   39.1
    IPE 270   270   135    6.6   10.2   15   45.9
    IPE 300   300   150    7.1   10.7   15   53.8
    IPE 330   330   160    7.5   11.5   18   62.6
    IPE 360   360   170    8.0   12.7   18   72.7
    IPE 400   400   180    8.6   13.5   21   84.5
    IPE 450   450   190    9.4   14.6   21   98.8
    IPE 500   500   200   10.2   16.0   21  115.5
    IPE 550   550   210   11.1   17.2   24  134.4
    IPE 600   600   220   12.0   19.0   24  156.0
    """,
    "HEA": """
    HEA 100    96   100    5.0    8.0   12   21.2
    HEA 120   114   120    5.0    8.0   12   25.3
    HEA 140   133   140    5.5    8.5   12   31.4
    HEA 160   152   160    6.0    9.0   15   38.8
    HEA 180   171   180    6.0    9.5   15   45.3
    HEA 200   190   200    6.5   10.0   18   53.8
    HEA 220   210   220    7.0   11.0   18   64.3
    HEA 240   230   240    7.5   12.0   21   76.8
    HEA 260   250   260    7.5   12.5   24   86.8
    HEA 280   270   280    8.0   13.0   24   97.3
    HEA 300   290   300    8.5   14.0   27  112.5
    HEA 320   310   300    9.0   15.5   27  124.4
    HEA 340   330   300    9.5   16.5   27  133.5
    HEA 360   350   300   10.0   17.5   27  142.8
    HEA 400   390   300   11.0   19.0   27  159.0
    HEA 450   440   300   11.5   21.0   27  178.0
    HEA 500   490   300   12.0   23.0   27  197.5
    HEA 550   540   300   12.5   24.0   27  211.8
    HEA 600   590   300   13.0   25.0   27  226.5
    """,
    "HEB": """
    HEB 100   100   100    6.0   10.0   12   26.0
    HEB 120   120   120    6.5   11.0   12   34.0
    HEB 140   140   140    7.0   12.0   12   43.0
    HEB 160   160   160    8.0   13.0   15   54.3
    HEB 180   180   180    8.5   14.0   15   65.3
    HEB 200   200   200    9.0   15.0   18   78.1
    HEB 220   220   220    9.5   16.0   18   91.0
    HEB 240   240   240   10.0   17.0   21  106.0
    HEB 260   260   260   10.0   17.5   24  118.4
    HEB 280   280   280   10.5   18.0   24  131.4
    HEB 300   300   300   11.0   19.0   27  149.1
    HEB 320   320   300   11.5   20.5   27  161.3
    HEB 340   340   300   12.0   21.5   27  170.9
    HEB 360   360   300   12.5   22.5   27  180.6
    HEB 400   400   300   13.5   24.0   27  197.8
    HEB 450   450   300   14.0   26.0   27  218.0
    HEB 500   500   300   14.5   28.0   27  238.6
    HEB 550   550   300   15.0   29.0   27  254.1
    HEB 600   600   300   15.5   30.0   27  270.0
    """,
    "HEM": """
    HEM 100   120   106   12.0   20.0   12   53.2
    HEM 200   220   206   15.0   25.0   18  131.3
    HEM 300   340   310   21.0   39.0   27  303.1
    HEM 400   432   307   21.0   40.0   27  325.8
    HEM 500   524   306   21.0   40.0   27  344.3
    HEM 600   620   305   21.0   40.0   27  363.7
    """,
    # DIN 1025-1: tf = mean flange thickness, r = r1 (root), slope 14 %; see the module docstring
    "I": """
    I 80       80    42    3.9    5.9   3.9    7.57
    I 100     100    50    4.5    6.8   4.5   10.6
    I 120     120    58    5.1    7.7   5.1   14.2
    I 160     160    74    6.3    9.5   6.3   22.8
    I 200     200    90    7.5   11.3   7.5   33.4
    I 240     240   106    8.7   13.1   8.7   46.1
    I 300     300   125   10.8   16.2  10.8   69.0
    """,
}

AREA_TOLERANCE = 0.01          # the parallel-flange self-check


def _parse() -> Dict[str, IProfile]:
    out: Dict[str, IProfile] = {}
    for series, block in _ROWS.items():
        for line in block.strip().splitlines():
            parts = line.split()
            name = f"{parts[0]} {parts[1]}"
            h, b, tw, tf, r, a = map(float, parts[2:8])
            out[name] = IProfile(name=name, series=series, standard=STANDARDS[series], h=h, b=b, tw=tw, tf=tf, r=r,
                                 area_cm2=a, flange_slope=0.14 if series == "I" else 0.0)
    return out


def area_from_dims(p: IProfile) -> float:
    """Section area in cm^2 of an ideal parallel-flange I section with four root fillets of radius r:
    2 b tf + (h - 2 tf) tw + (4 - pi) r^2 (each fillet adds r^2 - pi r^2 / 4). For the tapered "I" series with
    tf = mean thickness this is only an approximation (no toe radii, no taper-fillet interaction)."""
    return (2.0 * p.b * p.tf + (p.h - 2.0 * p.tf) * p.tw + (4.0 - math.pi) * p.r ** 2) / 100.0


def area_deviation(p: IProfile) -> float:
    """(area_from_dims - tabulated) / tabulated."""
    return area_from_dims(p) / p.area_cm2 - 1.0


def toe_radius(p: IProfile) -> float:
    """Flange-toe radius r2: 0.6 tf for the DIN 1025-1 I series, 0 (sharp toe) for the parallel-flange series."""
    return 0.6 * p.tf if p.series == "I" else 0.0


def self_check(profiles: Optional[Dict[str, IProfile]] = None) -> Dict[str, Tuple[float, str]]:
    """{name: (relative area deviation, status)} with status "ok", "FAIL" (parallel flange, |dev| > 1 %) or
    "flagged" (tapered series, not checkable with the parallel-flange formula)."""
    out = {}
    for name, p in (profiles or PROFILES).items():
        dev = area_deviation(p)
        if p.flange_slope > 0:
            status = "flagged"
        else:
            status = "ok" if abs(dev) <= AREA_TOLERANCE else "FAIL"
        out[name] = (dev, status)
    return out


_ALL = _parse()
# rows that fail the parallel-flange check are dropped here (and listed in DROPPED) - never kept silently
DROPPED: Dict[str, float] = {n: area_deviation(p) for n, p in _ALL.items()
                             if p.flange_slope == 0 and abs(area_deviation(p)) > AREA_TOLERANCE}
PROFILES: Dict[str, IProfile] = {n: p for n, p in _ALL.items() if n not in DROPPED}
FLAGS: Dict[str, str] = {
    n: "tapered flanges (14 %), tf = mean thickness, toe radius 0.6 tf: the parallel-flange area formula "
       f"overshoots by {100 * area_deviation(p):+.1f} %; checked against the FreeCAD-built section instead"
    for n, p in PROFILES.items() if p.flange_slope > 0}


def select(series=None, h_min=None, h_max=None) -> List[IProfile]:
    """Profiles filtered by series (a name or an iterable of names) and depth range [h_min, h_max] in mm,
    sorted by series then h."""
    if isinstance(series, str):
        series = {series}
    elif series is not None:
        series = set(series)
    out = [p for p in PROFILES.values()
           if (series is None or p.series in series)
           and (h_min is None or p.h >= h_min) and (h_max is None or p.h <= h_max)]
    order = {s: i for i, s in enumerate(("IPE", "HEA", "HEB", "HEM", "I"))}
    return sorted(out, key=lambda p: (order.get(p.series, 99), p.h, p.b))


def nearest(h, b, tw, tf, series=None) -> Tuple[IProfile, float]:
    """The profile closest to a measured section, and its relative distance = max over (h, b, tw, tf) of
    |measured - tabulated| / tabulated. A distance of 0.2 on tw alone (the frame's 350 x 300 x 12 x 17.5 against
    HEA 360's tw = 10) is the typical "right envelope, non-standard web" case."""
    candidates = select(series)
    if not candidates:
        raise ValueError(f"no profiles for series {series!r}")
    best, best_d = None, math.inf
    for p in candidates:
        d = max(abs(h - p.h) / p.h, abs(b - p.b) / p.b, abs(tw - p.tw) / p.tw, abs(tf - p.tf) / p.tf)
        if d < best_d:
            best, best_d = p, d
    return best, float(best_d)


def three_boxes(p: IProfile, length) -> List[Tuple[np.ndarray, np.ndarray]]:
    """The section as three plates, [(centre, half_extents)] in the beam's LOCAL frame: beam axis = local z (the beam
    runs from -length/2 to +length/2), depth h along local y, flange width b along local x. Order: top flange
    (+y), bottom flange (-y), web. Flanges are b x tf, the web is tw x (h - 2 tf) between them; the root fillets
    are excluded (they are the (4 - pi) r^2 remainder of area_from_dims). For the tapered series tf is the mean
    thickness, so the boxes are the mean-thickness approximation."""
    L = float(length)
    yf = (p.h - p.tf) / 2.0
    return [
        (np.array([0.0, +yf, 0.0]), np.array([p.b / 2.0, p.tf / 2.0, L / 2.0])),
        (np.array([0.0, -yf, 0.0]), np.array([p.b / 2.0, p.tf / 2.0, L / 2.0])),
        (np.array([0.0, 0.0, 0.0]), np.array([p.tw / 2.0, (p.h - 2.0 * p.tf) / 2.0, L / 2.0])),
    ]


if __name__ == "__main__":
    for name, (dev, status) in self_check().items():
        print(f"{name:10s} {100 * dev:+6.2f} %  {status}")
    print("dropped:", DROPPED or "none")
