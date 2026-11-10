"""
Prismatic groups (reco_engine/prismatic_groups.py): the channels of a piece read from its faces as they come from
reco_field/exact_check.py - here written by hand for a piece like the liner's split-face features, tilted and
off-centre: a step 1.6 deep with rounded corners, in its floor a groove 8 wide with floor radius R 2 and round ends
R 4, and two slits 0.35 wide with a round floor that run from outside the step into the groove. Checked: which
channels are found and their sizes, points inside and outside the fused tool, the groove's volume (sampled in its SDF)
against the closed formula (along the axes from the faces' boxes, tilted from their outlines), that two facing planes with the part's material between them are no channel, that a closed
mouth is none, and that a refused channel is split at the gap of its walls.
"""

import math
import sys
from pathlib import Path

import numpy as np

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine import prismatic_groups as pg  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp  # noqa: E402

ROT, _ = np.linalg.qr(np.array([[2.0, -1.0, 0.5], [0.3, 1.5, -1.0], [1.0, 0.4, 2.0]]))
if np.linalg.det(ROT) < 0:
    ROT[:, 2] *= -1
OFF = np.array([40.0, -25.0, 310.0])
IDENTITY = (np.eye(3), np.zeros(3))
TILTED = (ROT, OFF)


class Piece:
    """Faces written in the feature's own frame (the mouth is the plane y = 0, the depth is +y), placed in the world."""

    def __init__(self, frame, outline=False):
        self.R, self.o = frame
        self.outline = outline
        self.faces, self.V = [], []

    def world(self, pts):
        return self.o + np.asarray(pts, float) @ self.R.T

    def _face(self, pts, normal, origin, **more):
        P = self.world(pts)
        self.V.append(P)
        f = dict(area=1.0, normal=list(self.R @ np.asarray(normal, float)), point=list(P.mean(axis=0)), origin=origin,
                 bbox=[list(P.min(axis=0)), list(P.max(axis=0))], **more)
        if self.outline:
            f["outline"] = [list(q) for q in P]
        self.faces.append(f)
        return f

    def plane(self, normal, lo, hi, origin="input"):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        pts = [[(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        f = self._face(pts, normal, origin, type="plane")
        f["area"] = float(np.prod([s for s in hi - lo if s > 0]))
        return f

    def cylinder(self, axis, centre, r, a0, a1, t0, t1, convex=True, origin="input"):
        """About `axis` (0, 1, 2) through `centre`, the angles a0 .. a1 (degrees, from the next axis towards the one
        after it), from t0 to t1 along the axis."""
        i, j = (axis + 1) % 3, (axis + 2) % 3
        pts = []
        for a in np.radians(np.linspace(a0, a1, 9)):
            for t in (t0, t1):
                q = np.array(centre, float)
                q[i] += r * math.cos(a)
                q[j] += r * math.sin(a)
                q[axis] = t
                pts.append(q)
        mid = np.zeros(3)
        mid[i], mid[j] = math.cos(math.radians((a0 + a1) / 2)), math.sin(math.radians((a0 + a1) / 2))
        e = np.zeros(3)
        e[axis] = 1.0
        f = self._face(pts, mid if convex else -mid, origin, type="cylinder", axis=list(self.R @ e),
                       center=list(self.world(centre)), radius=r, convex=convex)
        f["point"] = list(self.world(pts[8]))
        return f

    def torus(self, centre, major, minor, a0, a1):
        """About y through `centre`: the outer lower quarter of the tube, the angles a0 .. a1 about y."""
        pts = []
        for a in np.radians(np.linspace(a0, a1, 9)):
            for b in np.radians(np.linspace(0.0, 90.0, 5)):
                rho = major + minor * math.cos(b)
                pts.append([centre[0] + rho * math.sin(a), centre[1] + minor * math.sin(b), centre[2] + rho * math.cos(a)])
        m = math.radians((a0 + a1) / 2)
        n = np.array([math.sin(m), 1.0, math.cos(m)]) / math.sqrt(2.0)
        return self._face(pts, n, "input", type="torus", axis=list(self.R @ np.array([0.0, -1.0, 0.0])),
                          center=list(self.world(centre)), major_radius=major, minor_radius=minor)

    def vertices(self):
        return np.vstack(self.V)


SLITS = (50.0, 80.0)


def step_groove_slits(frame, outline=False):
    p = Piece(frame, outline)
    p.plane((0, -1, 0), (25, 0, 6), (55, 0, 200), origin="model")            # the mouth: the model's face
    # the step x 31 .. 55, z 6 .. 200, 1.6 deep, corners R 3; its wall x = 31 is cut by the slits
    cuts = [9.0] + [z + s * 0.175 for z in SLITS for s in (-1, 1)] + [197.0]
    for z0, z1 in zip(cuts[0::2], cuts[1::2]):
        p.plane((-1, 0, 0), (31, 0, z0), (31, 1.6, z1))
        p.plane((0, 1, 0), (31, 1.6, z0), (40, 1.6, z1))
    p.plane((1, 0, 0), (55, 0, 9), (55, 1.6, 197))
    p.plane((0, 0, -1), (34, 0, 6), (52, 1.6, 6))
    p.plane((0, 0, 1), (34, 0, 200), (52, 1.6, 200))
    p.plane((0, 1, 0), (48, 1.6, 6), (55, 1.6, 200))
    for x, z, a in ((34, 9, 90), (52, 9, 0), (52, 197, 270), (34, 197, 180)):  # angles from z towards x, about y
        p.cylinder(1, (x, 0, z), 3.0, a + 90, a + 180, 0.0, 1.6)
    # the groove x 40 .. 48, walls down to y = 3, floor y = 5 with R 2, round ends R 4 about (44, z = 16 / 190)
    p.plane((-1, 0, 0), (40, 1.6, 16), (40, 3, 190))
    p.plane((1, 0, 0), (48, 1.6, 16), (48, 3, 190))
    p.plane((0, 1, 0), (42, 5, 14), (46, 5, 192))
    p.cylinder(2, (42, 3, 0), 2.0, 90, 180, 16, 190)                          # angles from x towards y, about z
    p.cylinder(2, (46, 3, 0), 2.0, 0, 90, 16, 190)
    p.cylinder(1, (44, 0, 16), 4.0, 90, 270, 1.6, 3.0)
    p.cylinder(1, (44, 0, 190), 4.0, -90, 90, 1.6, 3.0)
    p.torus((44, 3, 16), 2.0, 2.0, 90, 270)
    p.torus((44, 3, 190), 2.0, 2.0, -90, 90)
    # the slits: 0.35 wide, x 25 .. 40, walls down to y = 1.762, below them the cylinder R 0.175 along x
    for z in SLITS:
        p.plane((0, 0, -1), (25, 0, z - 0.175), (40, 1.762, z - 0.175))
        p.plane((0, 0, 1), (25, 0, z + 0.175), (40, 1.762, z + 0.175))
        p.cylinder(0, (0, 1.762, z), 0.175, 0, 180, 25, 40)                   # angles from y towards z, about x
        p.cylinder(2, (0, 0, 0), 25.0, -1, 1, z - 0.175, z + 0.175, convex=False, origin="model")
    return p


def inside(tool, p, pts):
    return np.asarray(primitive_to_sp(tool).sdf(p.world(pts))) <= 0


def by_kind(chans):
    out = {}
    for c in chans:
        out.setdefault(c["kind"], []).append(c)
    return out


def check_step_groove_slits(frame, outline):
    p = step_groove_slits(frame, outline)
    chans = pg.channels(p.faces)
    kinds = by_kind(chans)
    assert {k: len(v) for k, v in kinds.items()} == {"pocket": 1, "slot": 1, "slit": 2}
    step, groove = kinds["pocket"][0], kinds["slot"][0]
    assert abs(step["width"] - 24) < 1e-6 and abs(step["depth"] - 1.6) < 1e-6 and step["closed"] == (True, True)
    assert np.allclose(np.sort(np.abs(step["box"][1] - step["box"][0]))[1:], [24, 194], atol=1e-6)
    assert abs(groove["width"] - 8) < 1e-6 and abs(groove["floor_radius"] - 2) < 1e-6
    assert abs(groove["box"][1][2] - groove["box"][0][2] - (3.4 + pg.OPEN_BY * 3.4)) < 1e-6    # floor 5, mouth open
    assert abs(groove["box"][1][1] - groove["box"][0][1] - 182) < 1e-6                         # z 12 .. 194
    for c in kinds["slit"]:
        assert abs(c["width"] - 0.35) < 1e-6 and abs(c["floor_radius"] - 0.175) < 1e-6
        assert c["closed"] == (False, False) and len(pg._leaves(c["tool"])) == 2
        assert abs(c["box"][1][1] - c["box"][0][1] - 15 * (1 + 2 * pg.OPEN_BY)) < 1e-6
    tool = pg.fused(chans)
    pts_in = [(43, 0.8, 100), (44, 4.9, 100), (44, 4.0, 12.5), (44, 2.0, 12.2), (31.2, 0.8, 100), (30, 1.9, 50),
              (30, 1.0, 50.17), (26, 0.1, 80), (32.2, 0.8, 7.2), (40.2, 3.5, 100), (45.9, 4.0, 14.5)]
    pts_out = [(31.3, 0.8, 6.3), (54.7, 0.8, 199.7), (40.1, 4.9, 100), (47.9, 4.9, 100), (41, 4.8, 13.2),
               (40.5, 2.0, 13.0), (30, 1.93, 50.17), (30, 1.0, 50.2), (30, 1.0, 65), (44, 5.1, 100), (44, 3, 11.9),
               (50, 1.7, 100), (30.9, 0.8, 100)]
    assert inside(tool, p, pts_in).all()
    assert not inside(tool, p, pts_out).any()
    return p, chans


def test_step_groove_and_slits_are_one_channel_each():
    check_step_groove_slits(IDENTITY, False)


def test_the_same_tilted_and_off_centre_with_the_outlines_of_the_faces():
    check_step_groove_slits(TILTED, True)


def test_the_grooves_volume_is_the_swept_section_with_its_revolved_ends():
    p = step_groove_slits(IDENTITY)
    groove = by_kind(pg.channels(p.faces))["slot"][0]
    sdf = primitive_to_sp(groove["tool"])
    lo, hi = np.array([39.0, 1.6, 11.0]), np.array([49.0, 5.5, 195.0])
    pts = lo + np.random.default_rng(0).random((400000, 3)) * (hi - lo)
    volume = float((np.asarray(sdf.sdf(pts)) <= 0).mean() * np.prod(hi - lo))
    section = 8 * 1.4 + 4 * 2 + 2 * math.pi                                   # walls, floor, two quarter circles R 2
    ends = math.pi * 16 * 1.4 + math.pi * 4 * 2 + 2 * math.pi * (2 + 8 / (3 * math.pi)) * math.pi
    assert abs(volume - (section * 174 + ends)) < 0.01 * (section * 174 + ends)


def test_the_piece_has_no_other_real_faces_so_the_tool_is_not_partial():
    p = step_groove_slits(IDENTITY)
    (source, tool, partial), = pg.prismatic_tools(dict(faces=p.faces), p.vertices())
    assert source == "tool (prismatic groups)" and not partial
    assert "slit x 2" in tool.label
    p.cylinder(1, (70, 0, 100), 3.0, 0, 360, 0, 10)                           # a hole beside them: another feature
    (_, _, partial), = pg.prismatic_tools(dict(faces=p.faces), p.vertices())
    assert partial


def test_facing_planes_with_the_material_between_them_are_no_channel():
    """A rib 0.35 thick between two pockets: the same two planes as a slit's, their normals towards each other."""
    p = Piece(IDENTITY)
    p.plane((0, 0, 1), (0, 0, 10), (20, 5, 10))                               # the piece ends at z = 10, looking at ...
    p.plane((0, 0, -1), (0, 0, 10.35), (20, 5, 10.35))                        # ... the piece that begins at z = 10.35
    p.plane((0, 1, 0), (0, 5, 0), (20, 5, 10))
    p.plane((0, 1, 0), (0, 5, 10.35), (20, 5, 20))
    assert pg.channels(p.faces) == []
    q = Piece(IDENTITY)                                                       # the slit: the same planes, turned round
    q.plane((0, 0, -1), (0, 0, 10), (20, 5, 10))
    q.plane((0, 0, 1), (0, 0, 10.35), (20, 5, 10.35))
    q.plane((0, 1, 0), (0, 5, 10), (20, 5, 10.35))
    (c,) = pg.channels(q.faces)
    assert abs(c["width"] - 0.35) < 1e-9 and c["floor_closed"] and c["closed"] == (False, False)


def test_a_pair_whose_mouth_is_closed_takes_the_other_depth():
    """A pocket open towards -x only: its walls z = 0 / 10 are closed at y = 0 and y = 5 by real planes."""
    p = Piece(IDENTITY)
    p.plane((0, 0, -1), (0, 0, 0), (20, 5, 0))
    p.plane((0, 0, 1), (0, 0, 10), (20, 5, 10))
    p.plane((0, -1, 0), (0, 0, 0), (20, 0, 10))
    p.plane((0, 1, 0), (0, 5, 0), (20, 5, 10))
    p.plane((1, 0, 0), (20, 0, 0), (20, 5, 10))                               # the floor: x = 20
    p.plane((-1, 0, 0), (0, 0, 0), (0, 5, 10), origin="model")
    (c,) = pg.channels(p.faces)
    assert np.allclose(c["d"], [1, 0, 0]) and c["floor_closed"] and c["closed"] == (True, True)
    assert abs(c["width"] - 5) < 1e-9 and abs(c["length"] - 10) < 1e-9 and abs(c["depth"] - 20) < 1e-9


def test_a_refused_channel_is_split_at_the_gap_of_its_walls():
    """Two pockets in line (z 0 .. 40 and 50 .. 90) share the planes of their walls; the rib between them is the
    part's material: the judge refuses the channel over both, the two halves pass."""
    p = Piece(IDENTITY)
    p.plane((0, -1, 0), (0, 0, 0), (10, 0, 90), origin="model")
    for z0, z1 in ((0, 40), (50, 90)):
        p.plane((-1, 0, 0), (0, 0, z0), (0, 5, z1))
        p.plane((1, 0, 0), (10, 0, z0), (10, 5, z1))
        p.plane((0, 0, -1), (0, 0, z0), (10, 5, z0))
        p.plane((0, 0, 1), (0, 0, z1), (10, 5, z1))
        p.plane((0, 1, 0), (0, 5, z0), (10, 5, z1))
    rib = np.array([[5.0, 2.0, 45.0]])

    def keep(tool):
        return float(np.asarray(primitive_to_sp(tool).sdf(rib))[0]) > 0

    assert len(pg.channels(p.faces)) == 1                                     # unjudged: one channel over both
    chans = pg.channels(p.faces, keep=keep)
    assert sorted(round(c["length"]) for c in chans) == [40, 40]
    assert all(c["kind"] == "pocket" and c["closed"] == (True, True) for c in chans)
    tool = pg.fused(chans)
    assert inside(tool, p, [(5, 2, 20), (5, 2, 70)]).all() and not inside(tool, p, rib).any()
