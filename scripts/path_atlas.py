#!/usr/bin/env python3
"""
The path atlas (the user's idea of 2026-10-10: "store the trees and compare them, in a two-dimensional multi-domain
view; benchmark the parts with aegir and compare their benchmark too"): every run's TREE (its program) stored as one
row per (domain, part, path variant), the trees of the same part compared across the paths (identical operations by
fingerprint, same operation kinds, the best path by the rating tier and the IoU), and the two-dimensional map -
parts x paths, one panel per domain, a cell per run coloured by the tier against the known tree where there is one,
else by the IoU.

    python3 scripts/path_atlas.py build <domain>=<runs glob> [...] [--truths <dir>] [--rename DIR=VARIANT ...] --out docs/data/path_atlas.jsonl
    bash scripts/path_atlas_all.sh           # every benchmark with compared variants -> docs/data/path_atlas_all.jsonl (the GUI's Benchmark tab)
    python3 scripts/path_atlas.py compare docs/data/path_atlas.jsonl [--md out.md]
    python3 scripts/path_atlas.py map docs/data/path_atlas.jsonl --out fig.pdf [--domains liner,examples,gate]

A runs glob names run directories (scripts/run_part.py / peel_part.py): its last two path parts are the part and the
variant, e.g. "examples=~/shape_reco_results/cubes_examples/*/*" (part/variant) or
"gate=~/shape_reco_results/validation/gate_paths/runs_*/*@variant/part" (variant/part; a runs/ directory of one
variant is the bare path). A directory
with rating.json (reco_engine/validation.rate_run) contributes the tier and the separation class; --truths adds the
truth tree's leaf count per part (truths/<part>.json).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TIERS = ("A", "B", "C", "D", "E")


def _fingerprints(program: dict):
    from reco_engine.run_path import fingerprint, prim_dict
    out = []
    for o in program.get("operations") or []:
        if o.get("accepted") is False:
            continue
        prim = o.get("primitive") or {}
        out.append((o.get("op"), prim.get("kind"), fingerprint(o.get("op"), prim_dict(prim))))
    return out


def _iou(rec):
    c = (rec or {}).get("comparison") or {}
    e = (c.get("exact") or {}).get("iou")
    s = (c.get("sampled") or {}).get("iou")
    return (float(e), "exact") if e is not None else (float(s), "sampled") if s is not None else (None, None)


def _row(domain, part, variant, d: Path, truths: Path | None):
    rec = json.loads((d / "record.json").read_text()) if (d / "record.json").is_file() else {}
    prog = json.loads((d / "program.json").read_text()) if (d / "program.json").is_file() else {}
    rating = json.loads((d / "rating.json").read_text()) if (d / "rating.json").is_file() else None
    fps = _fingerprints(prog) if prog else []
    m = rec.get("model") or {}
    x, kind = _iou(rec)
    row = dict(domain=domain, part=part, variant=variant, dir=str(d), state=rec.get("state"),
               n_ops=len(fps) if prog else (m.get("operations") or 0),          # older runs: no program.json, the record counts
               ops=[(op, k) for op, k, _ in fps], fingerprints=[f for _, _, f in fps],
               by_source=m.get("by_source"), one_valid_solid=bool(m.get("one_valid_solid")), iou=x, iou_kind=kind,
               seconds=(rec.get("seconds") or {}).get("reconstruction"), host=rec.get("host"),
               tier=(rating or {}).get("grade", {}).get("tier") if rating else None,
               separation=(((rating or {}).get("attribution") or {}).get("separation") or {}).get("cls") if rating else None,
               rating_iou=(((rating or {}).get("vector") or {}).get("volume") or {}).get("iou") if rating else None)
    if truths is not None and (truths / f"{part}.json").is_file():
        t = json.loads((truths / f"{part}.json").read_text())

        def leaves(node):
            if isinstance(node, dict):
                if node.get("parts"):
                    return sum(leaves(p) for p in node["parts"])
                if node.get("operations"):
                    return len(node["operations"])
                return 1
            return 0
        row["truth_leaves"] = leaves(t.get("tree") or t.get("program") or t)
    return row


def cmd_build(a):
    truths = Path(a.truths).expanduser() if a.truths else None
    rename = dict(x.split("=", 1) for x in (a.rename or []))
    rows = []
    for spec in a.specs:
        domain, pattern = spec.split("=", 1)
        layout = a.layout
        if "@" in pattern:                                           # <domain>=<glob>@variant/part
            pattern, layout = pattern.rsplit("@", 1)
        for d in sorted(glob.glob(os.path.expanduser(pattern))):
            d = Path(d)
            if not (d / "record.json").is_file():
                continue
            p1, p2 = d.parent.name, d.name
            if layout == "variant/part":
                part, variant = p2, ("bare" if p1 == "runs" else p1.replace("runs_", ""))
            else:
                part, variant = p1, p2
            variant = rename.get(variant, variant)
            rows.append(_row(domain, part, variant, d, truths))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    by = collections.Counter((r["domain"], r["variant"]) for r in rows)
    print(f"{len(rows)} rows -> {out}; per domain/variant: {dict(by)}")


def _load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _score(r):
    """Higher is better: the tier first (A best), then the IoU, then fewer operations."""
    t = TIERS.index(r["tier"]) if r.get("tier") in TIERS else len(TIERS)
    return (-t, r.get("iou") or r.get("rating_iou") or 0.0, -(r.get("n_ops") or 0))


def cmd_compare(a):
    rows = _load(a.atlas)
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["domain"], r["part"])][r["variant"]] = r
    lines = ["| domain | part | paths | identical trees | best path | best tier / IoU | ops per path | shared ops (fingerprint) |", "|---|---|---|---|---|---|---|---|"]
    summary = collections.defaultdict(collections.Counter)
    for (domain, part), vs in sorted(by.items()):
        names = sorted(vs)
        best = max(names, key=lambda n: _score(vs[n]))
        summary[domain][best] += 1
        fp = {n: set(vs[n]["fingerprints"]) for n in names}
        identical = []
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                if fp[n1] == fp[n2] and fp[n1]:
                    identical.append(f"{n1}={n2}")
        shared = ""
        if len(names) > 1:
            allfp = set.union(*fp.values()) if fp else set()
            common = set.intersection(*[fp[n] for n in names]) if fp else set()
            shared = f"{len(common)} of {len(allfp)} in every path"
        b = vs[best]
        lines.append(f"| {domain} | {part} | {len(names)} | {', '.join(identical) or '-'} | {best} | "
                     f"{b.get('tier') or '-'} / {b['iou']:.4f} | " if b.get("iou") is not None else
                     f"| {domain} | {part} | {len(names)} | {', '.join(identical) or '-'} | {best} | {b.get('tier') or '-'} / - | ")
        lines[-1] += ", ".join(f"{n} {vs[n]['n_ops']}" for n in names) + f" | {shared} |"
    text = "\n".join(lines) + "\n\nBest path per domain (ties to the first in sorted order):\n" + \
        "\n".join(f"- {d}: " + ", ".join(f"{v} {c}" for v, c in cnt.most_common()) for d, cnt in summary.items()) + "\n"
    print(text)
    if a.md:
        Path(a.md).write_text(text)


def cmd_map(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    rows = _load(a.atlas)
    domains = a.domains.split(",") if a.domains else sorted({r["domain"] for r in rows})
    panels = []
    for dom in domains:
        rs = [r for r in rows if r["domain"] == dom]
        if not rs:
            continue
        variants = sorted({r["variant"] for r in rs})
        parts = sorted({r["part"] for r in rs}, key=lambda p: (-(next((r.get("truth_leaves") or 0) for r in rs if r["part"] == p)), p))
        panels.append((dom, parts, variants, {(r["part"], r["variant"]): r for r in rs}))
    if not panels:
        raise SystemExit("nothing to draw")
    widths = [max(2.2, 0.55 * len(v) + 1.2) for _, _, v, _ in panels]
    height = max(3.0, 0.16 * max(len(p) for _, p, _, _ in panels) + 1.6)
    fig, axes = plt.subplots(1, len(panels), figsize=(sum(widths) + 1.5, min(height, 14)), width_ratios=widths, squeeze=False)
    tier_cmap = ListedColormap(["#1c5cab", "#3987e5", "#86b6ef", "#eda100", "#e34948"])     # A..E: blue ramp, then the warnings
    for ax, (dom, parts, variants, cells) in zip(axes[0], panels):
        import numpy as np
        grid = np.full((len(parts), len(variants)), np.nan)
        kind = "tier" if any(c.get("tier") for c in cells.values()) else "iou"
        for i, p in enumerate(parts):
            for j, v in enumerate(variants):
                r = cells.get((p, v))
                if r is None:
                    continue
                if kind == "tier":
                    grid[i, j] = TIERS.index(r["tier"]) if r.get("tier") in TIERS else np.nan
                else:
                    grid[i, j] = r["iou"] if r.get("iou") is not None else (0.0 if r.get("state") == "ended" else np.nan)
        masked = np.ma.masked_invalid(grid)
        if kind == "tier":
            im = ax.imshow(masked, cmap=tier_cmap, vmin=-0.5, vmax=4.5, aspect="auto")
        else:
            im = ax.imshow(masked, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(parts)))
        ax.set_yticklabels([p[:28] for p in parts], fontsize=6 if len(parts) > 30 else 7)
        ax.set_title(f"{dom}: {len(parts)} parts x {len(variants)} paths\n({'tier against the known tree' if kind == 'tier' else 'IoU'})", fontsize=8)
        for i in range(len(parts)):
            for j in range(len(variants)):
                r = cells.get((parts[i], variants[j]))
                if r is not None and not r.get("one_valid_solid") and r.get("state") == "ended":
                    ax.text(j, i, "x", ha="center", va="center", fontsize=6, color="white")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        if kind == "tier":
            cb.set_ticks(range(5))
            cb.set_ticklabels(list(TIERS))
        cb.ax.tick_params(labelsize=7)
    fig.suptitle("Path atlas: the same parts under every path, by domain (x: not one valid solid)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"-> {a.out} ({len(panels)} panels)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("specs", nargs="+", help="<domain>=<runs glob>")
    b.add_argument("--truths", default=None)
    b.add_argument("--layout", choices=["part/variant", "variant/part"], default="part/variant")
    b.add_argument("--rename", nargs="*", default=None, metavar="DIR=VARIANT",
                   help="variant names by directory, e.g. orig=bare aifree=analytic abc_known_subparts=sub-parts")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=cmd_build)
    c = sub.add_parser("compare")
    c.add_argument("atlas")
    c.add_argument("--md", default=None)
    c.set_defaults(fn=cmd_compare)
    m = sub.add_parser("map")
    m.add_argument("atlas")
    m.add_argument("--out", required=True)
    m.add_argument("--domains", default=None)
    m.set_defaults(fn=cmd_map)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
