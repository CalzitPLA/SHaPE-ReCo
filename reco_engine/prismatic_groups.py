#!/usr/bin/env python3
"""
Prismatic groups: a piece that is a UNION of several prismatic features (a step, a groove in its floor, a branch of
the groove, slits into it) read from its exact faces, one tool per feature, the tools fused.

    chans = channels(piece["faces"], V)              # one per pair of facing walls
    tools = prismatic_tools(piece, V, keep)           # [(source, primitive, partial)] like feature_stage.hole_tools

Measured 2026-09-29 on the liner: the features of its split face (plane y = 0) are, per side, ONE connected removed
solid of 101 298 / 101 133 mm^3 with 247 / 252 faces: a step 1.6 deep, in its floor a groove 8 wide with floor
radius R 2 and round ends R 4, a branch of the groove running outwards, and 36 slits 0.35 wide from the bore to the
groove. The contact tool (feature_stage.contact_tool) reads ONE box from all faces of a piece; here every feature
needs its own.

The common element of all four classes is the CHANNEL: two real planes that face each other across the piece
(outward normals n and -n, the piece between them, width W = the distance), a depth direction d square to n (the
floor's outward normal) and the sweep direction s = d x n. Around the pair the faces name the rest:

    floor    a real plane with normal d where the walls end, or blend cylinders along s tangent to a wall there
             (floor radius r: the floor lies r deeper; r = W / 2: a round floor, the slit); none: open (a through slot)
    ends     along +-s: a real plane facing that way (a cap, at most W past the walls' end: a step's or pocket's end)
             or cylinders along d tangent to a wall (a rounded corner; r = W / 2: a slot's round end); none: OPEN, the
             tool runs `open_by` of its length past the walls (into the air, or into the feature it opens into)
    mouth    opposite the floor, always open (a real plane there: d is not this pair's depth direction)

The tool is the box of these sides with the rounded edges (feature_stage.rounded_box) and the corners where two
rounded edges meet (rounded_corners) - Part-native boxes, cylinders and tori. A round floor without rounded ends is
box | cylinder (2 operands instead of 5).

Pairs are taken narrowest first, every wall once; a pair of planes that are another channel's caps is that channel
again (a pocket has two such pairs) and is left out. Planes that look at each other with the piece OUTSIDE (W < 0)
are no channel: between them is the part's material - a rib or a thin wall. This is how a thin CUT (the slit of
0.35 mm, D-005: below any voxel pitch, so it can only come from faces) is told from thin MATERIAL (the user's rule
of thin wall suppression): by the sign of the distance between the two facing real planes, never by the thickness.

Coplanar walls of several features are one wall here (the step's wall is cut into 37 faces by the slits). When the
judge `keep` refuses such a channel and its walls have a gap along s, the channel is split at the largest gap and
both parts are tried (two pockets in line, a rib between them).

Not read (open): spheres (the corner of three rounded edges), concave fillets where one channel opens into another,
walls that are no planes (a curved slot), a wall shared by two channels of different depth.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from reco_engine.result_schema import Primitive
from reco_engine.result_to_program import quaternion_of
from reco_engine.tools import AX_COS, REAL, _unit

SQUARE = 0.02                       # two directions are square to each other: |cos| below this
OPEN_BY = 0.05                      # an open side runs this share of the channel's size along that axis past the walls
TANGENT = 0.03                      # a cylinder is tangent to a wall: within this share of its radius
MAX_DEPTHS = 8                      # depth directions tried, the largest floors first
MAX_SPLITS = 4                      # a refused channel is split at its gaps at most this deep


def face_points(f: Dict) -> np.ndarray:
    """Points that span a face: its `outline` (points on its edges, reco_field/exact_check.py's face_info), without
    one the corners of its `bbox` (exact for a feature along the axes, too large for a tilted one), without a box
    the face's own point. The piece's mesh is no source: inside a tilted face's box lie vertices of its neighbours
    in the same plane (measured in the test: the slits' ends on the groove's wall made the groove 1.6 deeper)."""
    if f.get("outline"):
        return np.asarray(f["outline"], float)
    bb = f.get("bbox")
    if bb is None:
        return np.asarray([f["point"]], float)
    lo, hi = np.asarray(bb[0], float), np.asarray(bb[1], float)
    return np.array([[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _span(pts: np.ndarray, u: np.ndarray) -> tuple:
    q = pts @ u
    return float(q.min()), float(q.max())


def _overlaps(pts: np.ndarray, u: np.ndarray, lo: float, hi: float, eps: float) -> bool:
    a, b = _span(pts, u)
    return b > lo + eps and a < hi - eps


def walls_of(faces: Sequence[Dict], eps: float) -> List[Dict]:
    """The real planes of a piece, coplanar faces with one normal as ONE wall: dict(n, pos (along n), faces
    (indices), parts (the points of every face), area)."""
    out: List[Dict] = []
    for i, f in enumerate(faces):
        if f["type"] != "plane" or f.get("origin") not in REAL:
            continue
        n = _unit(f["normal"])
        pos = float(n @ np.asarray(f["point"], float))
        pts = face_points(f)
        for w in out:
            if float(w["n"] @ n) >= AX_COS and abs(w["pos"] - pos) <= eps:
                w["faces"].append(i)
                w["parts"].append(pts)
                w["area"] += float(f["area"])
                break
        else:
            out.append(dict(n=n, pos=pos, faces=[i], parts=[pts], area=float(f["area"])))
    return out


def depth_directions(faces: Sequence[Dict], walls: Sequence[Dict]) -> List[np.ndarray]:
    """Candidates for the depth direction, the largest floors first: the outward normals of the real planes by their
    area, then what a wall and a real cylinder square to it name (n x axis, both signs: a slit's round floor has no
    plane)."""
    found: List[list] = []

    def add(d, area):
        for e in found:
            if float(e[0] @ d) >= AX_COS:
                e[1] += area
                return
        found.append([d, area])
    for w in walls:
        add(w["n"], w["area"])
    found.sort(key=lambda e: -e[1])
    planes = len(found)
    for f in faces:
        if f["type"] != "cylinder" or f.get("origin") not in REAL or not f.get("convex", True):
            continue
        a = _unit(f["axis"])
        for w in walls:
            if abs(float(w["n"] @ a)) <= SQUARE:
                for sign in (1.0, -1.0):
                    add(sign * _unit(np.cross(w["n"], a)), 0.0)
    return [e[0] for e in found[:max(MAX_DEPTHS, planes)]][:MAX_DEPTHS]


def _cylinder(f: Dict, a: np.ndarray, half: float, along: float) -> Primitive:
    from reco_engine.feature_stage import _faces_source, _frame
    c = np.asarray(f["center"], float)
    c = c + a * (along - float(c @ a))
    return Primitive(kind="cylinder", quaternion=quaternion_of(_frame(a)), center_mm=tuple(float(v) for v in c),
                     dims_mm=dict(radius=float(f["radius"]), half_length=float(half)), confidence=1.0,
                     sources=_faces_source())


def _box(R: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Primitive:
    from reco_engine.feature_stage import _faces_source
    half = 0.5 * (hi - lo)
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in R @ (0.5 * (lo + hi))),
                     dims_mm=dict(x_half=float(half[0]), y_half=float(half[1]), z_half=float(half[2])),
                     confidence=1.0, sources=_faces_source())


def channel(faces: Sequence[Dict], A: Dict, B: Dict, d: np.ndarray, pts: Dict[int, np.ndarray], eps: float,
            open_by: float = OPEN_BY, tol: float = TANGENT) -> Optional[Dict]:
    """The channel of two facing walls A (outward normal n) and B (-n) with the depth direction d, or None when the
    faces do not make one (the mouth is closed by a real plane, the walls share no reach). `pts`: the points of every
    face by its index. See the module's text for the sides."""
    from reco_engine.feature_stage import compose, guard_note, rounded_box, rounded_corners
    n = A["n"]
    s = _unit(np.cross(d, n))
    a, b = A["pos"], -B["pos"]
    W = a - b
    P = np.vstack(A["parts"] + B["parts"])
    (t0, t1), (s0, s1) = _span(P, d), _span(P, s)
    if W <= eps or t1 - t0 <= eps or s1 - s0 <= eps:
        return None
    for w in (A, B):                                                 # both walls reach into the common part
        Q = np.vstack(w["parts"])
        if not (_overlaps(Q, d, t0, t1, eps) and _overlaps(Q, s, s0, s1, eps)):
            return None
    real = [(i, f) for i, f in enumerate(faces) if f.get("origin") in REAL]
    walls = set(A["faces"]) | set(B["faces"])
    used = set(walls)

    def between(i):                                                  # the face reaches between the two walls
        return _overlaps(pts[i], n, b, a, eps)

    def tangent(f, r):                                               # its axis lies r inside of wall A or wall B
        cn = float(np.asarray(f["center"], float) @ n)
        return abs(a - r - cn) <= tol * r + eps or abs(cn - b - r) <= tol * r + eps
    rounds = [(i, f, _unit(f["axis"]), float(f["radius"])) for i, f in real
              if f["type"] == "cylinder" and f.get("convex", True) and "center" in f]
    # ---- the floor
    blends = [(i, f, r) for i, f, ax, r in rounds if abs(float(ax @ s)) >= AX_COS and r <= 0.5 * W * (1 + tol) + eps
              and tangent(f, r) and abs(float(np.asarray(f["center"], float) @ d) - t1) <= tol * r + eps
              and _overlaps(pts[i], s, s0, s1, eps)]
    r_floor = max((r for _, _, r in blends), default=0.0)
    blends = [q for q in blends if abs(q[2] - r_floor) <= tol * r_floor]
    floor = t1 + r_floor
    plane = [i for i, f in real if f["type"] == "plane" and i not in walls and float(_unit(f["normal"]) @ d) >= AX_COS
             and abs(float(np.asarray(f["point"], float) @ d) - floor) <= tol * r_floor + eps and between(i)
             and _overlaps(pts[i], s, s0 - W, s1 + W, eps)]
    floor_closed = bool(blends or plane)
    if not floor_closed:
        floor = t1 + open_by * (t1 - t0)
    used.update(plane)
    used.update(i for i, _, _ in blends)
    # ---- the mouth: a real plane that closes it says that d is not the depth of this pair
    for i, f in real:
        if f["type"] == "plane" and i not in walls and float(_unit(f["normal"]) @ d) <= -AX_COS and between(i) \
                and -eps <= t0 - float(np.asarray(f["point"], float) @ d) <= W + eps and _overlaps(pts[i], s, s0, s1, eps):
            return None
    # ---- the two ends along s
    ends, closed, caps, corners = {}, {}, [], []
    for sign in (1.0, -1.0):
        e = s1 if sign > 0 else s0
        best = None
        for i, f in real:
            if f["type"] != "plane" or i in walls or float(_unit(f["normal"]) @ s) * sign < AX_COS:
                continue
            gap = sign * (float(np.asarray(f["point"], float) @ s) - e)
            if -eps <= gap <= W * (1 + tol) + eps and between(i) and _overlaps(pts[i], d, t0, floor, eps) \
                    and (best is None or gap < best[0]):
                best = (gap, i)
        here = [(i, f, r) for i, f, ax, r in rounds if abs(float(ax @ d)) >= AX_COS and r <= 0.5 * W * (1 + tol) + eps
                and tangent(f, r) and abs(float(np.asarray(f["center"], float) @ s) - e) <= tol * r + eps
                and _overlaps(pts[i], d, t0, floor, eps)]
        if best is not None:
            ends[sign], closed[sign] = e + sign * best[0], True
            caps.append(best[1])
        elif here:
            ends[sign], closed[sign] = e + sign * max(r for _, _, r in here), True
        else:
            ends[sign], closed[sign] = e + sign * open_by * (s1 - s0), False
        corners += here
    used.update(caps)
    used.update(i for i, _, _ in corners)
    # ---- the tool
    mouth = t0 - open_by * (floor - t0)
    R = np.stack([n, s, d], axis=1)
    lo, hi = np.array([b, ends[-1.0], mouth]), np.array([a, ends[1.0], floor])
    box = _box(R, lo, hi)
    centre = R @ (0.5 * (lo + hi))
    size = hi - lo
    tori = []
    for i, f in real:                                                # the corners of two rounded edges, inside the box
        if f["type"] == "torus" and "center" in f and \
                np.all(np.abs(R.T @ (np.asarray(f["center"], float) - centre)) <= 0.5 * size + eps):
            tori.append((i, f))
    round_floor = bool(blends) and abs(2 * r_floor - W) <= tol * W and not corners
    guarded: Dict[str, List[str]] = {}                               # corner tori the torus guard thinned or rejected
    if round_floor:                                                  # the slit: box | cylinder
        hi2 = hi.copy()
        hi2[2] = t1
        cyl = _cylinder(blends[0][1], s, 0.5 * size[1], float(centre @ s))
        tool = compose(_box(R, lo, hi2), cyl, "fuse")
        n_edges, n_corners = 0, 0
    else:
        cyls, seen = [], []
        for i, f, ax, r in rounds:                                   # every edge the box can have rounded
            k = int(np.argmax(np.abs(R.T @ ax)))
            if abs(float(R[:, k] @ ax)) < AX_COS:
                continue
            q = R.T @ (np.asarray(f["center"], float) - centre)
            q[k] = 0.0
            if np.any(np.abs(q) > 0.5 * size + eps) or not all(
                    _overlaps(pts[i], R[:, m], lo[m] - eps, hi[m] + eps, 0.0) for m in range(3)):
                continue
            key = (k, r) + tuple(np.round(q / max(eps, 1e-9)).astype(np.int64))
            if key in seen:
                continue
            seen.append(key)
            cyls.append((i, _cylinder(f, R[:, k], 0.5 * size[k], float(centre @ R[:, k]))))
        tool = rounded_box(box, [c for _, c in cyls], tol) if cyls else None
        n_edges = (len(_leaves(tool)) - 1) // 2 if tool is not None else 0
        tool = tool or box
        if n_edges:
            used.update(i for i, _ in cyls)
        n_corners = 0
        for i, f in tori:
            # a torus face can lie in two corners (a round end's half ring): one copy per corner its points reach into
            q = (pts[i] - np.asarray(f["center"], float)) @ R
            for sg in ((x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)):
                if all((q[:, m] * sg[m]).max() > eps for m in range(3)):
                    tool, k = rounded_corners(tool, box, [dict(f, normal=list(R @ np.asarray(sg)))], tol, guard=guarded)
                    n_corners += k
                    if k:
                        used.add(i)
        for i, f in real:                                            # the corner of three rounded edges
            if f["type"] != "sphere" or not f.get("convex", True) or "center" not in f:
                continue
            r, q = float(f["radius"]), R.T @ (np.asarray(f["center"], float) - centre)
            sg = np.sign(q)
            if np.all(np.abs(0.5 * size - np.abs(q) - r) <= tol * r + eps):
                c_lo, c_hi = np.minimum(q, q + sg * r * (1 + tol)), np.maximum(q, q + sg * r * (1 + tol))
                ball = Primitive(kind="sphere", center_mm=tuple(float(v) for v in f["center"]), dims_mm=dict(radius=r),
                                 confidence=1.0, sources=list(box.sources))
                block = _box(R, c_lo + R.T @ centre, c_hi + R.T @ centre)
                tool = compose(tool, compose(block, ball, "cut"), "cut")
                n_corners += 1
                used.add(i)
    length, depth = s1 - s0, floor - t0
    if round_floor and not any(closed.values()):
        kind = "slit"
    elif closed[1.0] and closed[-1.0]:
        kind = "slot" if not caps and corners else "pocket"
    elif not closed[1.0] and not closed[-1.0]:
        kind = "through channel"
    else:
        kind = "open channel"
    tool.label = (f"{kind} {W:.4g} x {length:.4g} x {depth:.4g}" + (f", floor R {r_floor:.4g}" if blends else "")
                  + ("" if floor_closed else ", no floor") + (f", {n_edges} rounded edge(s)" if n_edges else "")
                  + (f", {n_corners} corner(s)" if n_corners else "")
                  + guard_note(guarded))
    return dict(kind=kind, n=n, s=s, d=d, width=float(W), length=float(length), depth=float(depth),
                floor_radius=float(r_floor), floor_closed=floor_closed, closed=(closed[-1.0], closed[1.0]),
                box=(lo, hi), frame=R, faces=sorted(used), walls=(A, B), caps=caps, tool=tool,
                volume=float(np.prod(size)))


def _leaves(p: Optional[Primitive]) -> List[Primitive]:
    from reco_engine.feature_stage import leaves
    return leaves(p) if p is not None else []


def _split(w: Dict, s: np.ndarray, cut: float) -> tuple:
    """The wall's faces below and above `cut` along s, as two walls (None: no face on that side)."""
    out = []
    for side in (0, 1):
        k = [j for j, p in enumerate(w["parts"]) if (0.5 * sum(_span(p, s)) > cut) == bool(side)]
        out.append(dict(w, faces=[w["faces"][j] for j in k], parts=[w["parts"][j] for j in k]) if k else None)
    return tuple(out)


def _largest_gap(A: Dict, B: Dict, s: np.ndarray, eps: float) -> Optional[float]:
    """The middle of the largest stretch along s that no face of the two walls covers (None: no gap)."""
    spans = sorted(_span(p, s) for p in A["parts"] + B["parts"])
    best, reach = None, spans[0][1]
    for lo, hi in spans[1:]:
        if lo - reach > eps and (best is None or lo - reach > best[0]):
            best = (lo - reach, 0.5 * (lo + reach))
        reach = max(reach, hi)
    return None if best is None else best[1]


def channels(faces: Sequence[Dict], V: Optional[np.ndarray] = None, keep: Optional[Callable] = None,
             open_by: float = OPEN_BY, tol: float = TANGENT, eps: Optional[float] = None) -> List[Dict]:
    """The channels of a piece's faces (see the module's text), the narrowest pairs of facing walls first, every wall
    in one channel. `V`: the piece's mesh vertices (not used: see face_points); `keep(tool)`: the judge of one
    channel's tool - a refused channel is split at the gaps of its walls, what stays refused is left out. `eps`: the
    length two positions may differ by (default 1e-5 of the faces' size). Faces without `outline` and `bbox` (pieces
    stored before 2026-09-29: with_face_boxes) give no channel."""
    faces = list(faces)
    if not any(f.get("outline") or f.get("bbox") for f in faces):
        return []
    pts = {i: face_points(f) for i, f in enumerate(faces)}
    if eps is None:
        every = np.vstack(list(pts.values()))
        eps = 1e-5 * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0))) + 1e-9
    walls = walls_of(faces, eps)
    taken: set = set()                                               # faces that are a wall or a cap of a channel
    out: List[Dict] = []

    def attempt(A, B, d, depth=0):
        ch = channel(faces, A, B, d, pts, eps, open_by, tol)
        if ch is None:
            return []
        if keep is None or keep(ch["tool"]):
            return [ch]
        cut = _largest_gap(A, B, ch["s"], eps) if depth < MAX_SPLITS else None
        if cut is None:
            return []
        found = []
        for a2, b2 in zip(_split(A, ch["s"], cut), _split(B, ch["s"], cut)):
            if a2 is not None and b2 is not None:
                found += attempt(a2, b2, d, depth + 1)
        return found
    for d in depth_directions(faces, walls):
        side = [w for w in walls if abs(float(w["n"] @ d)) <= SQUARE]
        pairs = []
        for i, A in enumerate(side):
            for B in side[i + 1:]:
                if float(A["n"] @ B["n"]) <= -AX_COS and A["pos"] + B["pos"] > eps:   # the piece is between them
                    pairs.append((A["pos"] + B["pos"], A, B))
        for _, A, B in sorted(pairs, key=lambda p: p[0]):
            if taken & (set(A["faces"]) | set(B["faces"])):
                continue
            found = attempt(A, B, d)
            for ch in found:
                taken.update(ch["walls"][0]["faces"] + ch["walls"][1]["faces"] + ch["caps"])
            out += found
    return out


def fused(chans: Sequence[Dict]) -> Optional[Primitive]:
    """The channels' tools as one tool (a balanced tree of unions); None without channels."""
    from reco_engine.feature_stage import compose
    parts = [c["tool"] for c in chans]
    if not parts:
        return None
    while len(parts) > 1:
        parts = [compose(parts[i], parts[i + 1], "fuse") if i + 1 < len(parts) else parts[i]
                 for i in range(0, len(parts), 2)]
    kinds: Dict[str, int] = {}
    for c in chans:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    tool = parts[0]
    if len(chans) > 1:
        tool.label = "prismatic groups: " + ", ".join(f"{k} x {v}" for k, v in sorted(kinds.items(), key=lambda q: -q[1]))
    return tool


def prismatic_tools(piece: Dict, V: np.ndarray, keep: Optional[Callable] = None, min_channels: int = 2) -> List[tuple]:
    """The tool of a piece that is a union of prismatic features: [(source, primitive, partial)] (the form of
    feature_stage.hole_tools), empty with fewer than `min_channels` channels (ONE prismatic feature is the contact
    tool's and the catalogue's case). `partial`: the piece has real faces that belong to no channel - the tool explains
    a part of it, and what it leaves is a piece of the next difference."""
    faces = piece.get("faces") or []
    chans = channels(faces, V, keep)
    if len(chans) < min_channels:
        return []
    used = set()
    for c in chans:
        used.update(c["faces"])
    other = [i for i, f in enumerate(faces) if f.get("origin") in REAL and i not in used]
    return [("tool (prismatic groups)", fused(chans), bool(other))]


KEEP_PRECISION = 0.98               # a single channel's sampled precision: below it the pair of walls was no channel
KEEP_POINTS = 5000                  # ... sampled with this many points in the channel's box


def candidate_tools(piece: Dict, V: np.ndarray, scorer=None, stacked: Sequence[tuple] = ()) -> List[tuple]:
    """The candidates of feature_stage.guess_candidates: the prismatic groups of the piece, every channel judged by
    `scorer` (a feature_stage.PartScorer) - kept at a sampled precision of KEEP_PRECISION: a pair of walls that is no
    channel runs through the part's material and harms far more, while an exact slender tool does not reach the
    loop's 0.999 against the input's MESH (measured 2026-09-29: the liner's groove 8 x 1541, exact harm 0 mm^3 by the
    check below, sampled 0.9932) - and, with the piece's hole stacks (`stacked`: hole_tools' answer), both fused.
    What is applied is judged as ever: the fused tool's sampled precision and the exact check."""
    import copy
    from reco_engine.feature_stage import compose
    keep = None
    if scorer is not None:
        light = copy.copy(scorer)
        light.n = KEEP_POINTS
        keep = lambda t: light.score(t)[0] >= KEEP_PRECISION        # noqa: E731
    out = prismatic_tools(piece, V, keep)
    holes = [t for source, t, _ in stacked if source == "tool (hole stack)"]
    if out and holes:
        _, tool, _ = out[0]
        both = compose(tool, holes[0], "fuse", f"{tool.label} + {holes[0].label}")
        out.insert(0, ("tool (prismatic groups + hole stacks)", both, True))
    return out


def with_face_boxes(piece: Dict, freecad: Optional[str] = None) -> Dict:
    """The piece with `bbox` and `outline` on every face, for pieces stored before the faces carried them: the faces
    of the piece's STEP are read again (exact_check.exact_faces) and matched by type, area and point. A face without
    a match keeps none (and spans only its own point)."""
    faces = piece.get("faces") or []
    if all(f.get("outline") for f in faces):
        return piece
    from reco_field.exact_check import exact_faces
    r = exact_faces(piece["step"], freecad)
    if "error" in r:
        raise RuntimeError(r["error"])
    fresh = r["faces"]
    P = np.array([g["point"] for g in fresh], float)
    lo, hi = (np.asarray(b, float) for b in piece["bbox"])
    eps = 1e-5 * float(np.linalg.norm(hi - lo))
    out = []
    for i, f in enumerate(faces):
        order = [i] if i < len(fresh) else []                       # the same order as a rule
        order += list(np.argsort(np.linalg.norm(P - np.asarray(f["point"], float), axis=1))[:4])
        f = dict(f)
        for j in order:
            g = fresh[j]
            if g["type"] == f["type"] and abs(g["area"] - f["area"]) <= 1e-6 * max(f["area"], 1e-9) + eps * eps \
                    and np.linalg.norm(P[j] - np.asarray(f["point"], float)) <= eps:
                f["bbox"], f["outline"] = g["bbox"], g.get("outline")
                break
        out.append(f)
    return dict(piece, faces=out)
