#!/usr/bin/env python3
"""SHaPe ReCo logo: a poison dart frog in the crow's nest (Ausguck) of a ship, looking out through a spyglass
at the CAD primitives (box, cylinder, cone, sphere) on the horizon.

Writes, next to this script:
  shape-reco-emblem.svg  the round badge alone (no text)
  shape-reco-logo.svg    badge + wordmark, text converted to paths by inkscape (renders without the Inter font)
  shape-reco-logo-text.svg  the same with live text (editable; needs Inter installed)
and, with --3d <render.png> (the Blender render of blender_logo.py, 2048 px):
  shape-reco-3d-logo.svg   round badge cut from the render (embedded JPEG) + wordmark, text outlined
  shape-reco-3d-logo-dark.svg  the same with a light wordmark, for dark backgrounds
  shape-reco-3d-logo.png, shape-reco-3d-logo-dark.png  both as 600 px PNGs: README.md shows these (GitHub serves an
                           SVG with a CSP that may block the embedded JPEG, the badge's picture)
  shape-reco-3d-badge.png  the badge alone, 1024 px, transparent outside the circle
  ../shape-reco.svg, ../shape-reco.png  the application icon (install.sh, the GUI's window icon): the badge,
                           SVG with outlined text and a 256 px PNG
  previews/icon_sizes.png  the icon at the sizes a desktop shows it (128 to 16 px), to judge what stays readable
  The badge carries Batrachotoxin, the frog's poison (molecule_2d.py): its skeletal formula as PubChem draws it (CID
  6324647, the 2D record, via the MoleculMaker macro) in the free sky the spyglass points at, its name and sum formula
  along the ring. The disc and the
  free sky come from <render>.json (blender_logo.py).

Run: python3 packaging/logo/make_logo.py [--3d shape-reco-3d.png]
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# palette: navy ink, frog blue, pink and orange highlights, wood and brass
NAVY = "#0E1B3D"
PINK = "#FF4FA3"
ORANGE = "#FF8A1F"
ROPE = "#E7A85E"
ROPE_DARK = "#A4652E"
ROPE_LIGHT = "#F7CF95"
SHROUD = "#2A1A12"

C = 256                       # badge centre
R_OUT, R_ROPE, R_IN = 254, 238, 226
HORIZON = 372
EYE = (262.0, 178.0)          # spyglass pivot (eyepiece at the frog's right eye)
TILT = -6.0                   # spyglass tilt, degrees (negative: far end up)


def rot(p, ang=TILT, c=EYE):
    a = math.radians(ang)
    x, y = p[0] - c[0], p[1] - c[1]
    return (c[0] + x * math.cos(a) - y * math.sin(a), c[1] + x * math.sin(a) + y * math.cos(a))


def f(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


DEFS = f"""<defs>
  <clipPath id="scene"><circle cx="{C}" cy="{C}" r="{R_IN}"/></clipPath>
  <linearGradient id="sky" gradientUnits="userSpaceOnUse" x1="0" y1="30" x2="0" y2="{HORIZON}">
    <stop offset="0" stop-color="#78BFFF"/><stop offset=".48" stop-color="#B7B9F4"/>
    <stop offset=".8" stop-color="#FFB2D4"/><stop offset="1" stop-color="#FFD49C"/>
  </linearGradient>
  <linearGradient id="sea" gradientUnits="userSpaceOnUse" x1="0" y1="{HORIZON}" x2="0" y2="482">
    <stop offset="0" stop-color="#3175DC"/><stop offset="1" stop-color="#10307A"/>
  </linearGradient>
  <radialGradient id="sun"><stop offset="0" stop-color="#FFF2C4"/><stop offset=".6" stop-color="#FFC56C"/>
    <stop offset="1" stop-color="#FF8E6C"/></radialGradient>
  <radialGradient id="glow"><stop offset=".5" stop-color="#FFE0A8" stop-opacity=".95"/>
    <stop offset="1" stop-color="#FFB2D4" stop-opacity="0"/></radialGradient>
  <linearGradient id="wood" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#6A3C1D"/><stop offset=".35" stop-color="#C08048"/>
    <stop offset=".55" stop-color="#DDA164"/><stop offset=".75" stop-color="#A56733"/>
    <stop offset="1" stop-color="#5A3218"/></linearGradient>
  <linearGradient id="brass" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#FFE6A0"/><stop offset=".32" stop-color="#F8B846"/>
    <stop offset=".7" stop-color="#DC8A22"/><stop offset="1" stop-color="#A65B10"/></linearGradient>
  <linearGradient id="leather" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#3D6FD0"/><stop offset=".45" stop-color="#2350A8"/>
    <stop offset="1" stop-color="#132D6C"/></linearGradient>
  <radialGradient id="lens" cx=".35" cy=".3" r=".9">
    <stop offset="0" stop-color="#F2FCFF"/><stop offset=".45" stop-color="#8EDCFF"/>
    <stop offset="1" stop-color="#2A86D6"/></radialGradient>
  <radialGradient id="frog" gradientUnits="userSpaceOnUse" cx="196" cy="176" r="125">
    <stop offset="0" stop-color="#7CC4FF"/><stop offset=".45" stop-color="#2F86F0"/>
    <stop offset="1" stop-color="#1346B0"/></radialGradient>
  <linearGradient id="belly" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#FFA8D6"/><stop offset="1" stop-color="#FF4FA3"/></linearGradient>
  <linearGradient id="pennant" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{PINK}"/><stop offset=".6" stop-color="#FF6F7E"/>
    <stop offset="1" stop-color="{ORANGE}"/></linearGradient>
  <linearGradient id="lower" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{PINK}"/><stop offset="1" stop-color="{ORANGE}"/></linearGradient>
  <linearGradient id="face-top" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#F4F8FF"/><stop offset="1" stop-color="#CFE0FF"/></linearGradient>
</defs>"""


def badge_ring():
    out = [f'<circle cx="{C}" cy="{C}" r="{R_OUT}" fill="{NAVY}"/>',
           f'<circle cx="{C}" cy="{C}" r="{R_ROPE}" fill="none" stroke="{ROPE}" stroke-width="20"/>']
    n = 72
    for i in range(n):
        t = 2 * math.pi * i / n
        for (dt, col, w) in ((0.0, ROPE_DARK, 3.4), (math.pi / n, ROPE_LIGHT, 2.0)):
            a = (C + 229 * math.cos(t + dt), C + 229 * math.sin(t + dt))
            b = (C + 247 * math.cos(t + dt + 0.075), C + 247 * math.sin(t + dt + 0.075))
            out.append(f'<line x1="{f(a[0])}" y1="{f(a[1])}" x2="{f(b[0])}" y2="{f(b[1])}" stroke="{col}" '
                       f'stroke-width="{w}" stroke-linecap="round"/>')
    return out


def wave(x0, y, n, w=20, amp=5.5, col="#8EC6FF", op=0.85, sw=3):
    d = f"M{f(x0)},{f(y)} " + " ".join(f"q{f(w / 2)},{f(-amp)} {f(w)},0" for _ in range(n))
    return (f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{sw}" stroke-linecap="round" '
            f'stroke-linejoin="round" opacity="{op}"/>')


def cube(x, y, a=13, b=6.5, h=17):
    pts = lambda *ps: " ".join(f"{f(px)},{f(py)}" for px, py in ps)
    left = pts((x - a, y - b), (x, y), (x, y - h), (x - a, y - b - h))
    right = pts((x, y), (x + a, y - b), (x + a, y - b - h), (x, y - h))
    top = pts((x, y - h), (x + a, y - b - h), (x, y - 2 * b - h), (x - a, y - b - h))
    st = f'stroke="{NAVY}" stroke-width="2.4" stroke-linejoin="round"'
    return [f'<polygon points="{left}" fill="#7EA8EC" {st}/>',
            f'<polygon points="{right}" fill="#3B64B6" {st}/>',
            f'<polygon points="{top}" fill="url(#face-top)" {st}/>']


def cylinder(x, y, r=10, h=24, ry=4):
    st = f'stroke="{NAVY}" stroke-width="2.4" stroke-linejoin="round"'
    body = (f'M{f(x - r)},{f(y - h)} L{f(x - r)},{f(y)} A{r},{ry} 0 0 0 {f(x + r)},{f(y)} '
            f'L{f(x + r)},{f(y - h)} Z')
    return [f'<path d="{body}" fill="#5D86D6" {st}/>',
            f'<path d="M{f(x + 3)},{f(y - h + 3)} L{f(x + 3)},{f(y + 2)}" stroke="#2F56A8" stroke-width="5" '
            f'opacity=".6"/>',
            f'<ellipse cx="{f(x)}" cy="{f(y - h)}" rx="{r}" ry="{ry}" fill="url(#face-top)" {st}/>']


def cone(x, y, r=12, h=32, ry=4.2):
    st = f'stroke="{NAVY}" stroke-width="2.4" stroke-linejoin="round"'
    d = f'M{f(x - r)},{f(y)} L{f(x)},{f(y - h)} L{f(x + r)},{f(y)} A{r},{ry} 0 0 1 {f(x - r)},{f(y)} Z'
    return [f'<path d="{d}" fill="#6E98E2" {st}/>',
            f'<path d="M{f(x)},{f(y - h)} L{f(x + 6)},{f(y + 3.5)}" stroke="#2F56A8" stroke-width="5" '
            f'opacity=".55"/>',
            f'<path d="{d}" fill="none" {st}/>']


def sphere(x, y, r=10):
    return [f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="#5D86D6" stroke="{NAVY}" stroke-width="2.4"/>',
            f'<circle cx="{f(x - 3)}" cy="{f(y - 3.5)}" r="{f(r * 0.42)}" fill="#E8F0FF" opacity=".9"/>']


def background():
    out = [f'<rect x="0" y="0" width="512" height="{HORIZON}" fill="url(#sky)"/>',
           f'<circle cx="378" cy="{HORIZON}" r="98" fill="url(#glow)"/>',
           f'<circle cx="378" cy="{HORIZON}" r="50" fill="url(#sun)"/>',
           f'<rect x="0" y="{HORIZON - 1}" width="512" height="{512 - HORIZON + 1}" fill="url(#sea)"/>',
           f'<line x1="0" y1="{HORIZON}" x2="512" y2="{HORIZON}" stroke="#FFE4B8" stroke-width="1.6" opacity=".8"/>']
    for y, x0, x1, col in ((383, 344, 414, "#FFC66E"), (391, 354, 404, "#FF9CCF"), (400, 362, 396, "#FFC66E"),
                           (410, 368, 390, "#FF9CCF"), (421, 373, 385, "#FFC66E")):
        out.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{col}" stroke-width="3.2" '
                   f'stroke-linecap="round" opacity=".9"/>')
    out += [wave(78, 398, 5), wave(250, 414, 4), wave(110, 436, 6), wave(300, 452, 6),
            wave(56, 472, 5), wave(232, 490, 7), wave(380, 486, 3)]
    # the primitives on the horizon, what the lookout is searching for
    out += cube(318, HORIZON + 4) + cylinder(351, HORIZON + 4) + cone(386, HORIZON + 4) + sphere(418, HORIZON - 7)
    out.append(wave(300, HORIZON + 9, 6, w=21, amp=3.5, col="#B9DCFF", op=.9, sw=2.4))
    # gulls
    for (x, y, s) in ((352, 252, 1.0), (392, 230, 0.75), (318, 300, 0.6)):
        d = f"M{f(x)},{f(y)} q{f(9 * s)},{f(-10 * s)} {f(18 * s)},0 q{f(9 * s)},{f(-10 * s)} {f(18 * s)},0"
        out.append(f'<path d="{d}" fill="none" stroke="{NAVY}" stroke-width="{f(3.6 * max(s, .8))}" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')
    return out


def rigging():
    out = []
    pairs = (((140, 344), (34, 540)), ((166, 352), (102, 540))), (((270, 344), (376, 540)), ((244, 352), (308, 540)))
    for (a0, a1), (b0, b1) in pairs:
        for p, q in ((a0, a1), (b0, b1)):
            out.append(f'<line x1="{p[0]}" y1="{p[1]}" x2="{q[0]}" y2="{q[1]}" stroke="{SHROUD}" '
                       f'stroke-width="4.2" stroke-linecap="round"/>')
        for y in range(378, 520, 24):
            xa = a0[0] + (a1[0] - a0[0]) * (y - a0[1]) / (a1[1] - a0[1])
            xb = b0[0] + (b1[0] - b0[0]) * (y - b0[1]) / (b1[1] - b0[1])
            out.append(f'<line x1="{f(xa)}" y1="{y}" x2="{f(xb)}" y2="{y + 2}" stroke="{SHROUD}" '
                       f'stroke-width="2.4" stroke-linecap="round"/>')
    return out


def mast():
    st = f'stroke="{NAVY}" stroke-width="4"'
    return [
        f'<rect x="194" y="330" width="22" height="210" fill="url(#wood)" {st}/>',
        f'<rect x="199" y="58" width="12" height="220" fill="url(#wood)" {st}/>',
        f'<circle cx="205" cy="56" r="8.5" fill="url(#brass)" stroke="{NAVY}" stroke-width="3.6"/>',
        # pennant
        f'<path d="M211,66 C244,58 268,74 300,66 C318,61 334,62 348,66 L320,80 L350,93 C334,97 318,94 300,92 '
        f'C268,88 244,100 211,92 Z" fill="url(#pennant)" stroke="{NAVY}" stroke-width="4" stroke-linejoin="round"/>',
        f'<path d="M220,74 C246,68 266,80 296,74" fill="none" stroke="#FFD3E8" stroke-width="3" '
        f'stroke-linecap="round" opacity=".7"/>',
        # struts under the barrel
        f'<path d="M150,350 L199,404 M260,350 L211,404" stroke="{NAVY}" stroke-width="11" stroke-linecap="round"/>',
        f'<path d="M150,350 L199,404 M260,350 L211,404" stroke="#A86A36" stroke-width="5" stroke-linecap="round"/>',
    ]


RIM = dict(cx=205, cy=268, rx=83, ry=15)
BOT = dict(cy=352, rx=71, ry=12)


def barrel_back():
    return [f'<ellipse cx="{RIM["cx"]}" cy="{RIM["cy"]}" rx="{RIM["rx"]}" ry="{RIM["ry"]}" fill="#3A2112" '
            f'stroke="{NAVY}" stroke-width="5"/>']


def barrel_front():
    cx, cy, rx, ry = RIM["cx"], RIM["cy"], RIM["rx"], RIM["ry"]
    by, brx, bry = BOT["cy"], BOT["rx"], BOT["ry"]
    body = (f"M{cx - rx},{cy} A{rx},{ry} 0 0 0 {cx + rx},{cy} L{cx + brx},{by} "
            f"A{brx},{bry} 0 0 1 {cx - brx},{by} Z")
    out = [f'<path d="{body}" fill="url(#wood)"/>']
    for k in range(-4, 5):  # staves
        phi = k * math.pi / 11
        x0, y0 = cx + rx * math.sin(phi), cy + ry * math.cos(phi)
        x1, y1 = cx + brx * math.sin(phi), by + bry * math.cos(phi)
        out.append(f'<line x1="{f(x0)}" y1="{f(y0)}" x2="{f(x1)}" y2="{f(y1)}" stroke="#4A2812" '
                   f'stroke-width="2" opacity=".55"/>')
    for t in (0.24, 0.8):  # iron hoops with brass rivets
        y = cy + (by - cy) * t
        r = rx + (brx - rx) * t
        e = ry + (bry - ry) * t
        hoop = f"M{f(cx - r)},{f(y)} A{f(r)},{f(e)} 0 0 0 {f(cx + r)},{f(y)}"
        out.append(f'<path d="{hoop}" fill="none" stroke="{NAVY}" stroke-width="9"/>')
        out.append(f'<path d="{hoop}" fill="none" stroke="#33406A" stroke-width="5"/>')
        for k in (-3, -1, 1, 3):
            phi = k * math.pi / 8
            out.append(f'<circle cx="{f(cx + r * math.sin(phi))}" cy="{f(y + e * math.cos(phi))}" r="2.2" '
                       f'fill="#FFC56C"/>')
    out.append(f'<path d="{body}" fill="none" stroke="{NAVY}" stroke-width="5" stroke-linejoin="round"/>')
    lip = f"M{cx - rx},{cy} A{rx},{ry} 0 0 0 {cx + rx},{cy}"
    out.append(f'<path d="{lip}" fill="none" stroke="{NAVY}" stroke-width="13" stroke-linecap="round"/>')
    out.append(f'<path d="{lip}" fill="none" stroke="#D09356" stroke-width="7" stroke-linecap="round"/>')
    out.append(f'<path d="M{cx - 50},{cy + 11} A{rx},{ry} 0 0 0 {cx + 10},{cy + 15}" fill="none" '
               f'stroke="#F2C48E" stroke-width="2.4" stroke-linecap="round" opacity=".8"/>')
    return out


HEAD = ("M140,296 L137,252 C138,224 148,205 161,194 C176,186 192,184 205,184 "
        "C218,184 234,186 249,194 C262,205 272,224 273,252 L270,296 Z")
EYES = ((172, 174, 23), (238, 172, 23))


def frog_body():
    out = []
    for fill_layer in (False, True):  # outline trick: fat navy copies first, plain fills on top (no seams)
        paint = 'fill="url(#frog)"' if fill_layer else f'fill="{NAVY}" stroke="{NAVY}" stroke-width="10" stroke-linejoin="round"'
        out.append(f'<path d="{HEAD}" {paint}/>')
        for (x, y, r) in EYES:
            out.append(f'<circle cx="{x}" cy="{y}" r="{r}" {paint}/>')
    out += [
        # throat and belly
        f'<ellipse cx="205" cy="262" rx="44" ry="21" fill="url(#belly)"/>',
        f'<path d="M176,252 Q205,262 234,252" fill="none" stroke="#FFD0E8" stroke-width="2.4" '
        f'stroke-linecap="round" opacity=".8"/>',
        # orange dorsolateral stripes
        f'<path d="M153,190 C145,206 142,228 143,262" fill="none" stroke="{ORANGE}" stroke-width="6.5" '
        f'stroke-linecap="round"/>',
        f'<path d="M257,190 C265,206 268,228 267,262" fill="none" stroke="{ORANGE}" stroke-width="6.5" '
        f'stroke-linecap="round"/>',
        f'<path d="M163,158 C168,152 178,151 185,156" fill="none" stroke="{ORANGE}" stroke-width="4.5" '
        f'stroke-linecap="round"/>',
        f'<path d="M226,155 C232,150 242,149 249,154" fill="none" stroke="{ORANGE}" stroke-width="4.5" '
        f'stroke-linecap="round"/>',
    ]
    for (x, y, r) in ((189, 194, 3.6), (221, 193, 3.0), (156, 228, 4.4), (254, 226, 3.8), (164, 247, 3.0),
                      (248, 246, 3.4), (150, 210, 2.6), (205, 190, 2.2)):  # navy spots
        out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{NAVY}" opacity=".88"/>')
    out += [
        # gloss
        f'<ellipse cx="199" cy="197" rx="19" ry="5.5" fill="#FFFFFF" opacity=".35"/>',
        f'<ellipse cx="166" cy="163" rx="7" ry="3.5" fill="#FFFFFF" opacity=".5" transform="rotate(-25 166 163)"/>',
        # left eye shut, concentrating
        f'<path d="M159,176 Q172,187 186,175" fill="none" stroke="{NAVY}" stroke-width="4.6" stroke-linecap="round"/>',
        f'<path d="M157,171 L153,167 M186,170 L190,166" stroke="{NAVY}" stroke-width="3" stroke-linecap="round"/>',
        # right eye at the eyepiece
        f'<circle cx="242" cy="172" r="14.5" fill="#0A1230"/>',
        f'<circle cx="237" cy="166" r="4.4" fill="#FFFFFF"/>',
        f'<circle cx="246" cy="178" r="1.8" fill="#FFFFFF" opacity=".9"/>',
        # nostrils, smile
        f'<circle cx="197" cy="205" r="2.2" fill="{NAVY}"/><circle cx="214" cy="204" r="2.2" fill="{NAVY}"/>',
        f'<path d="M165,220 Q205,244 247,218" fill="none" stroke="{NAVY}" stroke-width="4.6" stroke-linecap="round"/>',
        f'<path d="M163,216 Q162,221 166,224 M249,214 Q251,219 246,222" fill="none" stroke="{NAVY}" '
        f'stroke-width="3" stroke-linecap="round"/>',
    ]
    return out


def limb(d, w=13):
    return [f'<path d="{d}" fill="none" stroke="{NAVY}" stroke-width="{w + 7}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>',
            f'<path d="{d}" fill="none" stroke="#2A7BE8" stroke-width="{w}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>']


def fingers(base, tips, w=6.5):
    out = []
    for t in tips:
        out += limb(f"M{f(base[0])},{f(base[1])} L{f(t[0])},{f(t[1])}", w)
    for t in tips:
        out.append(f'<circle cx="{f(t[0])}" cy="{f(t[1])}" r="5.6" fill="{PINK}" stroke="{NAVY}" stroke-width="2.6"/>')
        out.append(f'<circle cx="{f(t[0] - 1.5)}" cy="{f(t[1] - 1.6)}" r="1.7" fill="#FFE0F0"/>')
    return out


def left_hand():
    # resting on the rim of the crow's nest, fingers over the lip
    return limb("M163,232 Q141,246 138,266") + fingers((138, 266), ((125, 282), (136, 286), (148, 284)))


def spyglass_and_right_arm():
    wrist_local = (301.0, 201.0)
    w = rot(wrist_local)
    out = limb(f"M247,236 Q292,266 {f(w[0])},{f(w[1])}", 13)
    g = [f'<g transform="rotate({TILT} {EYE[0]} {EYE[1]})">']
    y = EYE[1]
    st = f'stroke="{NAVY}" stroke-width="3.6" stroke-linejoin="round"'

    def sec(x0, x1, hh, fill, rx=2.5):
        return f'<rect x="{x0}" y="{f(y - hh)}" width="{x1 - x0}" height="{f(2 * hh)}" rx="{rx}" fill="{fill}" {st}/>'

    g += [sec(341, 397, 16, "url(#brass)"),
          sec(394, 410, 19.5, "url(#brass)"),
          f'<rect x="399" y="{f(y - 19.5)}" width="5" height="39" fill="{PINK}" stroke="{NAVY}" stroke-width="2"/>',
          sec(299, 344, 13, "url(#leather)"),
          f'<path d="M305,{f(y - 9)} L338,{f(y - 9)}" stroke="#7FA6F0" stroke-width="2.2" stroke-linecap="round" opacity=".8"/>',
          sec(336, 344, 15.5, "url(#brass)", 1.5),
          sec(264, 301, 10, "url(#brass)"),
          sec(295, 302, 12.5, "url(#brass)", 1.5),
          sec(253, 266, 8.5, "#1B2550", 3),
          f'<path d="M346,{f(y - 11)} L392,{f(y - 11)}" stroke="#FFF3CF" stroke-width="2.6" stroke-linecap="round" opacity=".8"/>',
          f'<ellipse cx="410" cy="{y}" rx="5.5" ry="17" fill="url(#lens)" stroke="{NAVY}" stroke-width="3.2"/>']
    g += fingers(wrist_local, ((293.0, 186.0), (303.0, 182.5), (313.0, 186.0)))
    g.append("</g>")
    out += g
    # glint on the lens
    sx, sy = rot((414.0, 170.0))
    s = 15
    star = (f"M{f(sx)},{f(sy - s)} Q{f(sx)},{f(sy)} {f(sx + s)},{f(sy)} Q{f(sx)},{f(sy)} {f(sx)},{f(sy + s)} "
            f"Q{f(sx)},{f(sy)} {f(sx - s)},{f(sy)} Q{f(sx)},{f(sy)} {f(sx)},{f(sy - s)} Z")
    out.append(f'<path d="{star}" fill="#FFFFFF" stroke="{NAVY}" stroke-width="1.6" stroke-linejoin="round"/>')
    return out


def emblem_body():
    parts = badge_ring()
    parts.append('<g clip-path="url(#scene)">')
    parts += background() + rigging() + mast() + barrel_back() + frog_body() + barrel_front()
    parts += left_hand() + spyglass_and_right_arm()
    parts.append("</g>")
    parts.append(f'<circle cx="{C}" cy="{C}" r="{R_IN}" fill="none" stroke="{NAVY}" stroke-width="7"/>')
    return parts


def svg(view_h, body, title):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 512 {view_h}" width="512" height="{view_h}">',
        f"<title>{title}</title>", DEFS, *body, "</svg>", ""])


def wordmark(ink=NAVY, sub="#2350A8"):
    """The name and the subtitle under the badge: `ink` the capitals, `sub` the subtitle (light ones on dark grounds)."""
    word = [("S", ink), ("H", ink), ("a", "url(#lower)"), ("P", ink), ("e", "url(#lower)"),
            ("R", ink), ("e", "url(#lower)"), ("C", ink), ("o", "url(#lower)")]
    # the word gap as dx on the R: a tspan holding only a space is collapsed away by the renderers
    spans = "".join(f'<tspan fill="{c}"{" dx=\"20\"" if ch == "R" else ""}>{ch}</tspan>' for ch, c in word)
    return [
        f'<text x="256" y="616" text-anchor="middle" font-family="Inter" font-weight="800" font-size="80" '
        f'letter-spacing="-1">{spans}</text>',
        f'<path d="M40,642 q9,-7 18,0 t18,0 t18,0" fill="none" stroke="{ORANGE}" stroke-width="3.4" '
        f'stroke-linecap="round"/>',
        f'<path d="M418,642 q9,-7 18,0 t18,0 t18,0" fill="none" stroke="{ORANGE}" stroke-width="3.4" '
        f'stroke-linecap="round"/>',
        f'<text x="256" y="647" text-anchor="middle" font-family="Inter" font-weight="600" font-size="18" '
        f'letter-spacing="3.2" fill="{sub}">SHAPE RECONSTRUCTION</text>',
    ]


def outline_text(src, dst):
    inkscape = shutil.which("inkscape")
    if not inkscape:
        print(f"inkscape not found: {dst.name} not written (text not outlined)")
        return False
    subprocess.run([inkscape, str(src), "--export-text-to-path", "--export-plain-svg", f"--export-filename={dst}"],
                   check=True, capture_output=True)
    return True


R_DISC = 230      # radius of the render disc inside the navy ring


def molecule_overlay(meta):
    """Batrachotoxin, the dart frog's poison, in the badge: PubChem's skeletal formula in the free sky the spyglass
    points at (fitted around the keep-out circles blender_logo.py exports: glass, frog, eyes, mast), and its name and
    sum formula along the bottom of the ring."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    import molecule_2d as m2

    info = m2.fetch()
    cx, cy, r = meta["cx"], meta["cy"], meta["r"]

    def badge(u, v):
        return C + (u - cx) / r * R_DISC, C + (v - cy) / r * R_DISC

    yy, xx = np.mgrid[0:512, 0:512] + 0.5
    free = (xx - C) ** 2 + (yy - C) ** 2 <= (R_DISC - 9) ** 2
    for u, v, rad in meta["keepout"]:
        x, y = badge(u, v)
        free &= (xx - x) ** 2 + (yy - y) ** 2 > (rad / r * R_DISC + 5) ** 2
    (g0x, g0y), (g1x, g1y) = (badge(*q) for q in meta["glass"])
    nx, ny = g1y - g0y, g0x - g1x                       # normal of the glass line, turned to the sky side
    if ny > 0:
        nx, ny = -nx, -ny
    free &= (xx - g0x) * nx + (yy - g0y) * ny > 0
    (m0x, m0y), (m1x, m1y) = (badge(*q[:2]) for q in meta["mast"])
    hw = meta["mast"][0][2] / r * R_DISC
    off = ((xx - m0x) * (m1y - m0y) - (yy - m0y) * (m1x - m0x)) / math.hypot(m1x - m0x, m1y - m0y)
    free &= np.abs(off) > hw + 5                         # never across the mast
    dist = distance_transform_edt(free)
    dr = m2.pubchem_drawing()
    theta, s, tx, ty, s_max = m2.fit(dr["points"], dist)
    print(f"molecule: {s:.1f} px per bond (room for {s_max:.1f}), {math.degrees(theta):.0f} deg")
    sub = '<tspan font-size="8.5" baseline-shift="sub">{}</tspan>'
    formula = "".join(sub.format(t) if t.isdigit() else t
                      for t in re.findall(r"[A-Z][a-z]?|\d+", info["formula"]))
    rr = R_DISC + 13
    return (m2.drawing_svg(dr, theta, s, tx, ty) +
            [f'<path id="ringtext" d="M{C - rr},{C} A{rr},{rr} 0 0 0 {C + rr},{C}" fill="none"/>',
             f'<text font-family="Inter" font-weight="700" font-size="11.5" letter-spacing="1.6" fill="#E8BC66">'
             f'<textPath xlink:href="#ringtext" startOffset="50%" text-anchor="middle">'
             f'{info["name"].upper()}  \u00b7  {formula}</textPath></text>'])


def badge_3d_body(render, molecule=True):
    """The round badge: the render cut to a disc (embedded as JPEG), a navy ring with a thin brass line."""
    w = int(subprocess.run(["magick", "identify", "-format", "%w", str(render)], check=True, capture_output=True,
                           text=True).stdout)
    meta = render.with_suffix(".json")             # written by blender_logo.py: the disc as fractions of the width
    d = json.loads(meta.read_text()) if meta.exists() else {"cx": 0.508, "cy": 0.537, "r": 0.459}
    cx, cy, r = round(d["cx"] * w), round(d["cy"] * w), round(d["r"] * w)
    with tempfile.TemporaryDirectory() as td:
        jpg = Path(td) / "disc.jpg"
        subprocess.run(["magick", str(render), "-crop", f"{2 * r}x{2 * r}+{cx - r}+{cy - r}", "+repage",
                        "-resize", "1200x1200", "-depth", "8", "-quality", "90", str(jpg)], check=True)
        b64 = base64.b64encode(jpg.read_bytes()).decode()
    return [f'<clipPath id="disc"><circle cx="{C}" cy="{C}" r="{R_DISC}"/></clipPath>',
            f'<circle cx="{C}" cy="{C}" r="{R_OUT}" fill="{NAVY}"/>',
            f'<image xlink:href="data:image/jpeg;base64,{b64}" x="{C - R_DISC}" y="{C - R_DISC}" '
            f'width="{2 * R_DISC}" height="{2 * R_DISC}" preserveAspectRatio="none" clip-path="url(#disc)"/>',
            *(molecule_overlay(d) if molecule and "keepout" in d else []),
            f'<circle cx="{C}" cy="{C}" r="{R_DISC + 1.5}" fill="none" stroke="#E8BC66" stroke-width="3"/>',
            f'<circle cx="{C}" cy="{C}" r="{R_OUT - 5}" fill="none" stroke="#2350A8" stroke-width="2"/>']


def make_3d(render):
    body = badge_3d_body(render)
    with tempfile.TemporaryDirectory() as td:
        for name, mark in (("shape-reco-3d-logo.svg", wordmark()),
                           ("shape-reco-3d-logo-dark.svg", wordmark(ink="#FFF1DA", sub="#8FB1F2"))):
            live = Path(td) / name
            live.write_text(svg(672, body + mark, "SHaPe ReCo"))
            out = HERE / name
            if outline_text(live, out):
                print("wrote", out)
            png = out.with_suffix(".png")
            subprocess.run(["inkscape", str(live), "--export-type=png", f"--export-filename={png}", "-w", "600"],
                           check=True, capture_output=True)
            print("wrote", png)
        badge = Path(td) / "badge.svg"
        badge.write_text(svg(512, body, "SHaPe ReCo"))
        png = HERE / "shape-reco-3d-badge.png"
        subprocess.run(["inkscape", str(badge), "--export-type=png", f"--export-filename={png}", "-w", "1024"],
                       check=True, capture_output=True)
        print("wrote", png)
        # the application icon (install.sh, the GUI's window icon): the badge as SVG with outlined text, and as a
        # 256 px PNG for renderers without clip paths (Qt's SVG Tiny renderer, some icon themes)
        icon_svg, icon_png = HERE.parent / "shape-reco.svg", HERE.parent / "shape-reco.png"
        if outline_text(badge, icon_svg):
            print("wrote", icon_svg)
        subprocess.run(["inkscape", str(badge), "--export-type=png", f"--export-filename={icon_png}", "-w", "256"],
                       check=True, capture_output=True)
        print("wrote", icon_png)
        sizes = HERE / "previews" / "icon_sizes.png"
        sizes.parent.mkdir(exist_ok=True)
        cmd = ["magick", "-background", "#3C3C3C"]
        for px in (128, 64, 48, 32, 24, 16):
            cmd += ["(", str(icon_png), "-resize", f"{px}x{px}", "-gravity", "center", "-extent", f"{px + 12}x140", ")"]
        subprocess.run(cmd + ["+append", str(sizes)], check=True, capture_output=True)
        print("wrote", sizes)


def main():
    emblem = HERE / "shape-reco-emblem.svg"
    emblem.write_text(svg(512, emblem_body(), "SHaPe ReCo"))
    live = HERE / "shape-reco-logo-text.svg"
    live.write_text(svg(672, emblem_body() + wordmark(), "SHaPe ReCo"))
    out = HERE / "shape-reco-logo.svg"
    ok = outline_text(live, out)
    print("wrote", emblem, live, out if ok else "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--3d", dest="render", type=Path, help="Blender render (blender_logo.py) for the 3D badge")
    a = ap.parse_args()
    if a.render:
        make_3d(a.render)
    else:
        main()
