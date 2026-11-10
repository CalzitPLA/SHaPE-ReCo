#!/usr/bin/env python3
"""
Members stored and checked for congruency (the user's rule 2026-09-29: "... and the member stored, to check for
congruency"; D-004: a member is identified by its geometry, never by a profile name).

Every member the member sub-loop finds (reco_engine/ibeam.py: detect_members) is stored with what identifies it up to a
rigid motion - the kind of its section, the section's dimensions, its length - and with its accepted operations.
Congruent members form a group. The first member of a group whose operation was accepted when judged IN FULL (its own
primitive, residual_loop._member_op) is the group's source for that operation; every later member gets the source's
primitive moved by the rigid motion between the two members (x -> R x + t, result_to_program.transform_primitive)
and judged there by the same exact judge (ibeam.ExactJudge through residual_loop._exact_eval). A copy that fails is
judged in full by itself, with its own primitive; a member whose own primitive fails does NOT reject its group - the
next member is judged in full and becomes the source when it passes (the fault of the feature phase on the liner,
2026-09-28: 35 of 45 rejections came from a failed representative rejecting its whole group).

The groups go into the session's tool store (reco_engine.tools.ToolStore) as created tools - the construction (the
member's operations as one boolean tree) and a descriptor in the form of feature_stage.piece_descriptor, computed
from the section (volume, area, principal moments and axes, centroid, face counts) - so that the feature phase
recycles a member for a congruent piece of the exact difference (residual_loop._feature_phase: typed_candidates).

    store = MemberStore(members)
    store.info(i)                         # group id, size, source: for the "member" step
    store.moved(i, k, op)                 # the source's primitive of slot k moved onto member i, or None
    store.accept(i, k, op, prim, by_congruency) / store.reject(i, k)
    store.to_tools(tool_store, part)      # the groups as created tools
    store.summary()                       # for the log / the "done" step
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine.result_schema import Primitive, StageSource

# Relative tolerance of the section's dimensions and the length of two congruent members. Measured 2026-09-29 on the
# frame (STEP, 40 detected members) and the review set: see the commit message of this module / the session's report.
CONGRUENT_MEMBER = 2e-3
MODE_CONGRUENT = "analytic:member_congruent"

FACES = dict(ibeam=dict(plane=14), tube=dict(cylinder=2, plane=2), rhs=dict(plane=10), rod=dict(cylinder=1, plane=2),
             bar=dict(plane=6))          # the exact faces of the member's solid by type (a sharp-cornered section)


# ------------------------------------------------------------------------------------------------ what identifies a member

def section_dims(member) -> Optional[Dict[str, float]]:
    """The section's dimensions that the member's primitives are built from (ibeam.member_operations)."""
    sec = member.section
    if sec is None or sec.kind == "other":
        return None
    if sec.kind == "ibeam":
        ib = sec.ibeam
        return dict(h=float(ib.h), b=float(ib.b), tw=float(ib.tw), tf=float(ib.tf))
    keys = dict(rod=("r",), bar=("hx", "hy"), tube=("r", "r_in"), rhs=("hx", "hy", "ix", "iy")).get(sec.kind)
    if keys is None or any(k not in sec.params for k in keys):
        return None
    return {k: float(sec.params[k]) for k in keys}


def member_descriptor(member) -> Optional[Dict]:
    """dict(kind, dims, length, joined): a member up to a rigid motion. None for a section of no known kind."""
    dims = section_dims(member)
    if dims is None:
        return None
    return dict(kind=member.kind, dims=dims, length=float(member.length), joined=tuple(bool(j) for j in member.joined))


def congruent_members(a: Optional[Dict], b: Optional[Dict], tol: float = CONGRUENT_MEMBER) -> bool:
    """The same kind, every section dimension and the length equal within `tol` (relative)."""
    if a is None or b is None or a["kind"] != b["kind"] or set(a["dims"]) != set(b["dims"]):
        return False
    pairs = [(a["dims"][k], b["dims"][k]) for k in a["dims"]] + [(a["length"], b["length"])]
    return all(abs(x - y) <= tol * max(abs(x), abs(y)) + 1e-12 for x, y in pairs)


def group_members(descriptors: Sequence[Optional[Dict]], tol: float = CONGRUENT_MEMBER) -> List[List[int]]:
    """Indices grouped by congruency, in the given order; a member without a descriptor is a group of its own."""
    groups: List[List[int]] = []
    for i, d in enumerate(descriptors):
        for g in groups:
            if d is not None and congruent_members(descriptors[g[0]], d, tol):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# ------------------------------------------------------------------------------------------------ the rigid motion

def member_frame(member) -> Tuple[np.ndarray, np.ndarray]:
    """(R, c): the member's frame (columns: section x, section y, the axis; ibeam._frame) and its centre (the section's
    centre at the middle of its length) - the local coordinate system its primitives are placed in."""
    from reco_engine import ibeam as ib
    R = ib._frame(member, member.section)
    c = member.to_world(member.section.centre, 0.5 * (member.start + member.end))
    return np.asarray(R, float), np.asarray(c, float)


def cut_transferable(source, target) -> bool:
    """A cutter runs past a free end and stops at a joined one (ibeam.member_operations): it can be moved only onto a
    member with the same ends, in this order or reversed."""
    a, b = tuple(source.joined), tuple(target.joined)
    return a == b or a[::-1] == b


def member_motion(source, target) -> Tuple[np.ndarray, np.ndarray]:
    """(R, t) of x -> R x + t that maps the source member onto the congruent target: frame onto frame, centre onto
    centre; turned by half a turn about the section's x (the axis reversed) when the ends' joined flags agree only
    that way. The sections of all known kinds are symmetric under that turn."""
    Rs, cs = member_frame(source)
    Rt, ct = member_frame(target)
    a, b = tuple(source.joined), tuple(target.joined)
    X = np.diag([1.0, -1.0, -1.0]) if (a != b and a[::-1] == b) else np.eye(3)
    R = Rt @ X @ Rs.T
    return R, ct - R @ cs


def move_primitive(p: Primitive, R: np.ndarray, t: np.ndarray, group: str, source: int) -> Primitive:
    """The primitive moved by x -> R x + t, its own copy (dims, sources), with where it comes from in its sources."""
    from reco_engine.result_to_program import transform_primitive
    q = transform_primitive(copy.deepcopy(p), R, t)
    q.alternatives = []
    q.sources = list(q.sources) + [StageSource("position", MODE_CONGRUENT, 1.0, detail=dict(group=group, source=int(source)))]
    return q


# ------------------------------------------------------------------------------------------------ the member as a piece

def _bodies(member) -> Optional[List[Tuple[float, str, Tuple[float, ...], Tuple[float, float]]]]:
    """[(sign, kind, half dims, (x, y) offset)] of the member's solid in its own frame: disjoint boxes / cylinders
    added (+1) and taken away (-1), all of the member's length."""
    d = section_dims(member)
    if d is None:
        return None
    k = member.kind
    if k == "ibeam":
        yf = 0.5 * (d["h"] - d["tf"])
        return [(1.0, "box", (d["b"] / 2, d["tf"] / 2), (0.0, yf)), (1.0, "box", (d["b"] / 2, d["tf"] / 2), (0.0, -yf)),
                (1.0, "box", (d["tw"] / 2, d["h"] / 2 - d["tf"]), (0.0, 0.0))]
    if k == "rod":
        return [(1.0, "cylinder", (d["r"],), (0.0, 0.0))]
    if k == "bar":
        return [(1.0, "box", (d["hx"], d["hy"]), (0.0, 0.0))]
    if k == "tube":
        return [(1.0, "cylinder", (d["r"],), (0.0, 0.0)), (-1.0, "cylinder", (d["r_in"],), (0.0, 0.0))]
    if k == "rhs":
        return [(1.0, "box", (d["hx"], d["hy"]), (0.0, 0.0)), (-1.0, "box", (d["ix"], d["iy"]), (0.0, 0.0))]
    return None


def member_piece_descriptor(member) -> Optional[Dict]:
    """The member's solid described as feature_stage.piece_descriptor describes a piece of the exact difference
    (volume, area, sorted principal moments, principal axes as columns, centroid, exact face counts, sign "plus" =
    missing material), computed from the section - no mesh is built. The section has sharp corners: a real profile
    with root fillets differs in volume (the frame's 3.4 %) and is then, rightly, not congruent."""
    bodies = _bodies(member)
    if bodies is None:
        return None
    L = float(member.length)
    vol, area, first, I0 = 0.0, 0.0, np.zeros(3), np.zeros((3, 3))
    for sign, kind, half, (ox, oy) in bodies:
        if kind == "box":
            a, b = half
            sec_area, perimeter = 4.0 * a * b, 4.0 * (a + b)
            ixx, iyy = sec_area * b * b / 3.0, sec_area * a * a / 3.0         # second moments of the section about x, y
        else:
            r = half[0]
            sec_area, perimeter = math.pi * r * r, 2.0 * math.pi * r
            ixx = iyy = sec_area * r * r / 4.0
        v = sec_area * L
        o = np.array([ox, oy, 0.0])
        # the inertia tensor of a prism of length L about its own centre (unit density), then the parallel axes
        own = np.diag([ixx * L + v * L * L / 12.0, iyy * L + v * L * L / 12.0, (ixx + iyy) * L])
        I0 += sign * (own + v * (float(o @ o) * np.eye(3) - np.outer(o, o)))
        vol += sign * v
        first += sign * v * o
        area += perimeter * L + sign * 2.0 * sec_area          # a hole's wall is surface too; its ends are not
    if member.kind == "ibeam":                                  # the web meets each flange on tw x L: no surface there
        d = section_dims(member)
        area -= 4.0 * d["tw"] * L
    if vol <= 0:
        return None
    c_loc = first / vol
    I_c = I0 - vol * (float(c_loc @ c_loc) * np.eye(3) - np.outer(c_loc, c_loc))
    moments, vecs = np.linalg.eigh(I_c)
    R, c = member_frame(member)
    axes = R @ vecs
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    return dict(volume=float(vol), area=float(area), moments=np.asarray(moments, float), axes=axes,
                centre=c + R @ c_loc, faces=dict(FACES.get(member.kind, {})), sign="plus")


def member_tool(operations: Sequence[Tuple[str, Primitive]], label: Optional[str] = None) -> Optional[Primitive]:
    """The member's accepted operations as ONE tool: its adds fused, its cuts taken away (a schema composite, which
    result_to_program builds); a single add is the primitive itself."""
    adds = [p for op, p in operations if op == "add"]
    if not adds:
        return None
    tool = copy.deepcopy(adds[0])
    for p in adds[1:]:
        tool = _compose(tool, copy.deepcopy(p), "fuse", label)
    for op, p in operations:
        if op == "cut":
            tool = _compose(tool, copy.deepcopy(p), "cut", label)
    tool.alternatives = []
    return tool


def _compose(a: Primitive, b: Primitive, combine: str, label: Optional[str]) -> Primitive:
    return Primitive(kind="composite", center_mm=tuple(a.center_mm), dims_mm={}, parts=[a, b], combine=combine,
                     label=label, confidence=1.0, sources=list(a.sources))


# ------------------------------------------------------------------------------------------------ the store

@dataclass
class MemberGroup:
    id: str                                   # "M1", "M2", ... in the order the groups were found
    descriptor: Optional[Dict]
    members: List[int]
    sources: Dict[int, int] = dc_field(default_factory=dict)                  # slot -> the member it was judged in full on
    accepted: Dict[int, Dict[int, Tuple[str, Primitive, bool, float]]] = dc_field(default_factory=dict)   # member -> slot -> (op, primitive, by congruency, the judge's precision)
    rejected: Dict[int, List[int]] = dc_field(default_factory=dict)           # member -> the slots nothing passed for
    failed_copies: int = 0                    # moved primitives that failed on their member (then judged in full)


class MemberStore:
    """The members of one run, grouped by congruency, with their accepted operations (see the module docstring).
    A `slot` is the index of an operation in ibeam.member_operations(member): the same for congruent members."""

    def __init__(self, members: Sequence, tol: float = CONGRUENT_MEMBER):
        self.members = list(members)
        self.descriptors = [member_descriptor(m) for m in self.members]
        self.groups = [MemberGroup(f"M{n + 1}", self.descriptors[g[0]], g)
                       for n, g in enumerate(group_members(self.descriptors, tol))]
        self._of = {i: g for g in self.groups for i in g.members}

    def group(self, i: int) -> MemberGroup:
        return self._of[i]

    def info(self, i: int) -> Dict:
        """For the member's step: its group, the group's size and members, its descriptor."""
        g, d = self._of[i], self.descriptors[i]
        return dict(group=g.id, group_size=len(g.members), group_members=list(g.members), group_index=g.members.index(i),
                    descriptor=None if d is None else dict(kind=d["kind"], dims=dict(d["dims"]), length=d["length"]))

    def text(self, i: int) -> str:
        g = self._of[i]
        if len(g.members) < 2:
            return f"group {g.id} (no congruent member)"
        return f"group {g.id}, {g.members.index(i) + 1} of {len(g.members)} congruent members"

    def source(self, i: int, slot: int) -> Optional[int]:
        """The member whose own primitive of this slot was accepted first in i's group; None: none yet, or i itself."""
        j = self._of[i].sources.get(slot)
        return None if j is None or j == i else j

    def moved(self, i: int, slot: int, op: str) -> Optional[Primitive]:
        """The source's accepted primitive of this slot moved onto member i; None when the group has no source yet,
        or for a cutter whose ends do not fit member i (cut_transferable)."""
        j = self.source(i, slot)
        if j is None:
            return None
        if op == "cut" and not cut_transferable(self.members[j], self.members[i]):
            return None
        o, prim = self._of[i].accepted[j][slot][:2]
        if o != op:
            return None
        R, t = member_motion(self.members[j], self.members[i])
        return move_primitive(prim, R, t, self._of[i].id, j)

    def accept(self, i: int, slot: int, op: str, prim: Primitive, by_congruency: bool, precision: float = 1.0) -> None:
        g = self._of[i]
        g.accepted.setdefault(i, {})[slot] = (op, prim, bool(by_congruency), float(precision))
        if not by_congruency and slot not in g.sources and self.descriptors[i] is not None:
            g.sources[slot] = i

    def failed_copy(self, i: int) -> None:
        self._of[i].failed_copies += 1

    def reject(self, i: int, slot: int) -> None:
        self._of[i].rejected.setdefault(i, []).append(slot)

    def operations(self, i: int) -> List[Tuple[str, Primitive]]:
        return [(v[0], v[1]) for _, v in sorted(self._of[i].accepted.get(i, {}).items())]

    def summary(self) -> Dict:
        """Numbers and the groups, JSON-friendly: for the log and the "done" step."""
        groups = []
        for g in self.groups:
            ops = [(i, s, v[2]) for i, slots in g.accepted.items() for s, v in slots.items()]
            d = g.descriptor
            groups.append(dict(id=g.id, kind=d["kind"] if d else "other", dims=dict(d["dims"]) if d else None,
                               length=d["length"] if d else None, members=list(g.members),
                               sources=sorted(set(g.sources.values())),
                               copies=sorted({i for i, _, by in ops if by}),
                               operations=len(ops), by_congruency=sum(1 for _, _, by in ops if by),
                               failed_copies=g.failed_copies,
                               without_operation=[i for i in g.members if i not in g.accepted]))
        return dict(members=len(self.members), groups=groups, n_groups=len(groups),
                    largest_group=max((len(g.members) for g in self.groups), default=0),
                    operations=sum(g["operations"] for g in groups),
                    by_congruency=sum(g["by_congruency"] for g in groups),
                    failed_copies=sum(g["failed_copies"] for g in groups),
                    members_with_operations=sum(len(g.accepted) for g in self.groups))

    def to_tools(self, tool_store, part: str) -> List[str]:
        """Every group with accepted operations as a created tool of the session's store: the first fully explained
        member's operations as one tree (its construction) with its piece descriptor, and one use per member
        (source "created" for members judged in full, "pattern" for copies; the score is the exact judge's precision). Returns the record ids. The store is
        bookkeeping: an error here never fails the run (the caller catches)."""
        ids = []
        for g in self.groups:
            if g.descriptor is None or not g.accepted:
                continue
            need = len(set(s for slots in g.accepted.values() for s in slots))
            full = [i for i in g.members if len(g.accepted.get(i, {})) == need]
            first = (full or sorted(g.accepted))[0]
            tool = member_tool(self.operations(first), label=f"member {g.descriptor['kind']} {g.id}")
            desc = member_piece_descriptor(self.members[first])
            if tool is None or desc is None:
                continue
            for i in sorted(g.accepted):
                copied = all(v[2] for v in g.accepted[i].values())
                score = min(v[3] for v in g.accepted[i].values())    # the exact judge's precision, its worst operation
                rec = tool_store.add_use(f"created:member {g.descriptor['kind']}", part, f"member {i + 1} ({g.id})",
                                         score, "pattern" if copied else "created", construction=tool, descriptor=desc)
            ids.append(rec.id)
        return ids
