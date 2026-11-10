#!/usr/bin/env python3
"""
The reconstruction protocol (docs/thesis/chapters/12_datasets_and_protocol.tex, "The reconstruction protocol"; the
user, 2026-10-29: "develop a scientific testing method based on all we got already"): a change of the reconstruction is
judged by a pre-registered, paired comparison of two arms on frozen corpora, against the measured run-to-run noise, with
its cost - not by a look at a few parts.

    python3 scripts/sci_test.py freeze <name> --hypothesis "..." --arm-a bare --arm-b tree_lg \
        --dev <corpus dir> --test <corpus dir> [...] [--real <list>] [--desc desc.jsonl ...] [--margin 0.03]
    python3 scripts/sci_test.py power <name> [--flip 0.033]
    python3 scripts/sci_test.py analyze <name> --a "<runs glob>" --b "<runs glob>" [--split test] \
        [--replicate "<glob A>" "<glob A'>"] [--anchors-a <glob> --anchors-b <glob>] [--desc ...] [--label dev_gate]
    python3 scripts/sci_test.py waste "<runs glob>" [...] [--desc ...] [--json out.json]
    python3 scripts/sci_test.py launch <name> <arm> <split>      # prints the server commands of an arm (R3)

Files: docs/data/protocol/<name>/protocol.json (freeze, committed BEFORE the runs - R1), manifest.json (the parts of
every split with their scene and class), result_<label>.{json,md} (analyze). A runs glob names run directories whose
last path part is the part (scripts/run_part.py's layout: runs_<variant>/<part>/record.json).

The rules (R1-R8 of the chapter):
  good      = one valid solid and IoU >= 0.95, the exact IoU before the sampled one (RUNNING 3.4)
  primary   = exact McNemar on the discordant pairs (one-sided for the hypothesis "B better")
  effect    = the change of the good rate with its paired bootstrap 95 % interval
  noise     = the flip rate of a replicate (A against A', the same code and settings); an effect counts only when the
              net change (rescues - breaks) exceeds what that rate reaches by chance (stats_tests.net_change_floor)
  confounds = budget stops, timeouts and crashes per arm; the sensitivity analysis drops every part with one in either
              arm; the hosts and commits of each arm are reported, a mixed arm is flagged
  per class = the change per CLASS_V1 cell (n >= MIN_CELL), Holm-corrected - exploratory
  cost      = the paired reconstruction seconds (Wilcoxon), co-primary for a change that claims to save time
  verdict   = accept B when: McNemar p < alpha AND the net change exceeds the noise floor AND no anchor falls below its
              band AND no class cell with n >= MIN_CELL is worse (Holm p < alpha) AND the one-solid count is not lower;
              a time-saving change: the good-rate interval above -margin (non-inferior) AND fewer seconds (p < alpha)
              AND the same anchor / cell / one-solid conditions. Anything else: reject or back to DEV.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen"), str(ROOT / "scripts")]

from reco_engine import stats_tests as st  # noqa: E402

GOOD_IOU = 0.95
ALPHA = 0.05
MARGIN = 0.03                  # non-inferiority margin of the good rate for a time-saving change
MIN_CELL = 8                   # a class cell needs this many pairs to take part in the verdict
ANCHOR_BAND = 0.01             # an anchor may not fall more than this below arm A's IoU (its run-to-run noise)
OUT = ROOT / "docs" / "data" / "protocol"
SCHEMA = "sci_test/1"
ACCEPTED_KINDS = ("add", "accepted", "feature_accepted", "member_op")


# ------------------------------------------------------------------------------------------------ parts, classes

def scene_of(stem: str) -> str:
    """The aegir scene of a known tree (equivalent tree forms of one scene share it: aegir_n16_S_000045_T0d4c ->
    aegir_n16_S_000045); any other part is its own scene."""
    m = re.match(r"^(aegir_n\d+_[A-Z]_\d+)_T[0-9a-f]+$", stem)
    return m.group(1) if m else stem


def _corpus_items(corpus: Path):
    """[(stem, file to hash, stratum)] of a validate_known corpus: list.txt, the truth per stem, selection.json's stratum."""
    stems = [s.strip() for s in (corpus / "list.txt").read_text().split() if s.strip()] if (corpus / "list.txt").is_file() else []
    strata = {}
    sel = corpus / "selection.json"
    if sel.is_file():
        items = json.loads(sel.read_text()).get("items") or []
        strata = {it.get("stem"): it.get("stratum") for it in items if isinstance(it, dict)}
    out = []
    for s in stems:
        f = next((p for p in (corpus / "truths" / f"{s}.json", corpus / "step" / f"{s}.step") if p.is_file()), None)
        out.append((s, f, strata.get(s)))
    return out


def _sha(path) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_classes(desc_files, calib=None) -> dict:
    """stem -> CLASS_V1 key (and its axes) from shape_atlas descriptor files (a later file wins)."""
    if not desc_files:
        return {}
    import shape_atlas as sa
    calib = calib if calib is not None else _calib()
    out = {}
    for f in desc_files:
        for line in Path(f).expanduser().read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or not r.get("desc"):
                continue
            try:
                c = sa._class_fields(r, calib)
            except Exception:
                continue
            out[r["name"]] = dict(cls=c["cls"], T=c["T"], S=c["S"], C=c["C"])
    return out


def _calib():
    p = ROOT / "docs" / "data" / "shape_class_calib.json"
    return json.loads(p.read_text()) if p.is_file() else None


# ------------------------------------------------------------------------------------------------ runs

def run_row(d: Path) -> dict:
    """One run directory -> the protocol's row: path_atlas's row (state, one valid solid, IoU, tier, seconds, host) plus
    the stop (time budget, timeout, crash) and the commit."""
    from path_atlas import _row
    r = _row("protocol", d.name, "arm", d, None)
    rec = json.loads((d / "record.json").read_text()) if (d / "record.json").is_file() else {}
    stopped = str((rec.get("run") or {}).get("stopped") or "")
    state = rec.get("state")
    stop = ("crashed" if state == "crashed" else "running" if state == "running" else
            "time budget" if "time budget" in stopped else
            "exact residual timeout" if "no answer within" in stopped else      # the feature phase never ran
            "feature phase error" if "feature phase: error" in stopped else       # boolean disagreement, STL too big
            "timeout" if "timeout" in stopped.lower() else "error" if "error" in stopped.lower() else None)
    r.update(part=d.name, stop=stop, commit=str(rec.get("commit") or "")[:8], started=rec.get("started"),
             good=bool(r.get("one_valid_solid")) and r.get("iou") is not None and r["iou"] >= GOOD_IOU)
    return r


def load_arm(pattern: str) -> dict:
    """part -> row of every run directory the glob names (a directory without record.json counts as not run)."""
    out = {}
    for d in sorted(glob.glob(os.path.expanduser(pattern))):
        d = Path(d)
        if d.is_dir() and (d / "record.json").is_file():
            out[d.name] = run_row(d)
    return out


def _arm_facts(rows) -> dict:
    return dict(n=len(rows), hosts=dict(collections.Counter(r.get("host") for r in rows)),
                commits=dict(collections.Counter(r.get("commit") for r in rows)),
                stops=dict(collections.Counter(r["stop"] for r in rows if r.get("stop"))),
                one_solid=sum(bool(r.get("one_valid_solid")) for r in rows),
                median_seconds=st.median_or_none([r.get("seconds") for r in rows]),
                median_operations=st.median_or_none([r.get("n_ops") for r in rows]),
                median_iou=st.median_or_none([r.get("iou") for r in rows]))


# ------------------------------------------------------------------------------------------------ analysis

def compare(rows_a: dict, rows_b: dict, classes: dict | None = None, alternative: str = "greater",
            margin: float = MARGIN, alpha: float = ALPHA) -> dict:
    """The paired comparison of two arms (dicts part -> row) on their common parts (see the module docstring)."""
    parts = sorted(set(rows_a) & set(rows_b))
    A, B = [rows_a[p] for p in parts], [rows_b[p] for p in parts]
    ga, gb = [r["good"] for r in A], [r["good"] for r in B]
    t = st.discordant(ga, gb) if parts else dict(both=0, breaks=0, rescues=0, neither=0, n=0)
    out = dict(n=len(parts), only_a=sorted(set(rows_a) - set(rows_b)), only_b=sorted(set(rows_b) - set(rows_a)),
               a=dict(_arm_facts(A), good=st.summary_good(ga)), b=dict(_arm_facts(B), good=st.summary_good(gb)),
               table=t, mcnemar_p=st.mcnemar_exact(t["breaks"], t["rescues"], alternative),
               mcnemar_p_two_sided=st.mcnemar_exact(t["breaks"], t["rescues"]),
               rescued=[p for p, x, y in zip(parts, ga, gb) if not x and y],
               broken=[p for p, x, y in zip(parts, ga, gb) if x and not y])
    if parts:
        out["delta_good"] = st.paired_bootstrap_delta(ga, gb)
        out["iou"] = st.wilcoxon_paired([r.get("iou") if r.get("iou") is not None else float("nan") for r in A],
                                        [r.get("iou") if r.get("iou") is not None else float("nan") for r in B])
        out["seconds"] = st.wilcoxon_paired([r.get("seconds") if r.get("seconds") is not None else float("nan") for r in A],
                                            [r.get("seconds") if r.get("seconds") is not None else float("nan") for r in B],
                                            alternative="less")
        out["operations"] = st.wilcoxon_paired([float(r.get("n_ops") or 0) for r in A], [float(r.get("n_ops") or 0) for r in B],
                                               alternative="less")
        clean = [i for i, (x, y) in enumerate(zip(A, B)) if not x.get("stop") and not y.get("stop")]
        if clean:                                                       # R5: without the parts any arm stopped on
            ca, cb = [ga[i] for i in clean], [gb[i] for i in clean]
            ct = st.discordant(ca, cb)
            out["without_stops"] = dict(n=len(clean), table=ct, mcnemar_p=st.mcnemar_exact(ct["breaks"], ct["rescues"],
                                                                                           alternative),
                                        delta_good=st.paired_bootstrap_delta(ca, cb))
    if classes:
        cells = collections.defaultdict(list)
        for p, x, y in zip(parts, ga, gb):
            cells[(classes.get(p) or {}).get("cls", "?")].append((x, y))
        per = {}
        for cell, pairs in sorted(cells.items()):
            ct = st.discordant([x for x, _ in pairs], [y for _, y in pairs])
            per[cell] = dict(n=len(pairs), table=ct, good_a=sum(x for x, _ in pairs), good_b=sum(y for _, y in pairs),
                             p_worse=st.mcnemar_exact(ct["breaks"], ct["rescues"], "less"))
        big = {c: v["p_worse"] for c, v in per.items() if v["n"] >= MIN_CELL}
        for c, q in st.holm(big).items():
            per[c]["p_worse_holm"] = q
        out["classes"] = per
    return out


def verdict(cmp: dict, floor: float | None, anchors: dict | None, kind: str = "improvement", alpha: float = ALPHA,
            margin: float = MARGIN) -> dict:
    """R7: the decision from a compare() result, the replicate's net-change floor and the anchors."""
    t = cmp["table"]
    net = t["rescues"] - t["breaks"]
    cells_worse = [c for c, v in (cmp.get("classes") or {}).items() if v.get("p_worse_holm", 1.0) < alpha]
    anchors_ok = None if not anchors else all(v["ok"] for v in anchors.values())
    checks = dict(anchors=anchors_ok, no_cell_worse=not cells_worse,
                  one_solid_not_lower=cmp["b"]["one_solid"] >= cmp["a"]["one_solid"])
    if kind == "improvement":
        checks.update(mcnemar=cmp["mcnemar_p"] < alpha, above_noise=None if floor is None else net > floor)
    elif kind == "simpler":                                             # as good with fewer parts (the prune, 2026-10-30)
        checks.update(noninferior=st.noninferior(cmp.get("delta_good", {}).get("lo", -1.0), margin),
                      fewer_parts=cmp.get("operations", {}).get("p", 1.0) < alpha)
    else:                                                               # a change that claims to save time
        checks.update(noninferior=st.noninferior(cmp.get("delta_good", {}).get("lo", -1.0), margin),
                      faster=cmp.get("seconds", {}).get("p", 1.0) < alpha)
    missing = [k for k, v in checks.items() if v is None]
    failed = [k for k, v in checks.items() if v is False]
    decision = "reject or back to DEV" if failed else ("undecided: " + ", ".join(missing) + " not measured") if missing \
        else "accept B"
    return dict(kind=kind, net_change=net, noise_floor=floor, cells_worse=cells_worse, checks=checks, failed=failed,
                missing=missing, decision=decision)


def anchor_check(rows_a: dict, rows_b: dict, band: float = ANCHOR_BAND) -> dict:
    out = {}
    for p in sorted(set(rows_a) & set(rows_b)):
        a, b = rows_a[p].get("iou"), rows_b[p].get("iou")
        out[p] = dict(a=a, b=b, band=band, ok=a is not None and b is not None and b >= a - band)
    return out


# ------------------------------------------------------------------------------------------------ waste

def waste_of_run(d: Path) -> dict | None:
    """Where a run's time went (path.jsonl: cumulative seconds per step): seconds per step kind, the share on pieces
    the feature phase rejected in the end and on rejected loop candidates, candidates per accepted operation, and the
    time after the last accepted operation (time without progress)."""
    p = Path(d) / "path.jsonl"
    if not p.is_file():
        return None
    steps = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    steps = [s for s in steps if s.get("event") == "step" and s.get("seconds") is not None]
    if not steps:
        return None
    total = float(steps[-1]["seconds"])
    rejected_pieces = {s.get("piece_id") for s in steps if s.get("kind") == "feature_rejected" and s.get("piece_id") is not None}
    accepted_pieces = {s.get("piece_id") for s in steps if s.get("kind") == "feature_accepted" and s.get("piece_id") is not None}
    lost = rejected_pieces - accepted_pieces
    by_kind = collections.Counter()
    on_rejected, last_gain, prev, n_cand, n_acc = 0.0, 0.0, 0.0, 0, 0
    for s in steps:
        dt = max(0.0, float(s["seconds"]) - prev)
        prev = float(s["seconds"])
        k = s.get("kind")
        by_kind[k] += dt
        if (s.get("piece_id") in lost) or k == "rejected" or (k == "candidate" and s.get("ok") is False):
            on_rejected += dt
        if k in ("candidate", "feature_candidate"):
            n_cand += 1
        if k in ACCEPTED_KINDS:
            n_acc += 1
            last_gain = float(s["seconds"])
    stop = next((str(s.get("stopped")) for s in reversed(steps) if s.get("stopped")), "")
    return dict(part=Path(d).name, seconds=total, by_kind=dict(by_kind), rejected_share=on_rejected / total if total else 0.0,
                candidates=n_cand, accepted=n_acc, candidates_per_accepted=n_cand / n_acc if n_acc else None,
                after_last_gain=total - last_gain, after_last_gain_share=(total - last_gain) / total if total else 0.0,
                time_budget="time budget" in stop)


def waste_summary(runs, classes=None) -> dict:
    rows = [w for w in runs if w]
    if not rows:
        return dict(n=0)
    kinds = collections.Counter()
    for w in rows:
        kinds.update(w["by_kind"])
    total = sum(w["seconds"] for w in rows)
    out = dict(n=len(rows), seconds=total, median_seconds=st.median_or_none([w["seconds"] for w in rows]),
               share_by_kind={k: v / total for k, v in kinds.most_common()},
               rejected_share=sum(w["rejected_share"] * w["seconds"] for w in rows) / total,
               after_last_gain_share=sum(w["after_last_gain"] for w in rows) / total,
               median_candidates_per_accepted=st.median_or_none([w["candidates_per_accepted"] for w in rows]),
               time_budget_stops=sum(w["time_budget"] for w in rows))
    if classes:
        per = collections.defaultdict(list)
        for w in rows:
            per[(classes.get(w["part"]) or {}).get("cls", "?")].append(w)
        out["classes"] = {c: dict(n=len(v), seconds=sum(w["seconds"] for w in v),
                                  rejected_share=sum(w["rejected_share"] * w["seconds"] for w in v) / max(1e-9, sum(w["seconds"] for w in v)),
                                  after_last_gain_share=sum(w["after_last_gain"] for w in v) / max(1e-9, sum(w["seconds"] for w in v)))
                          for c, v in sorted(per.items(), key=lambda kv: -sum(w["seconds"] for w in kv[1]))}
    return out


# ------------------------------------------------------------------------------------------------ commands

def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "?"


def cmd_freeze(a):
    out = OUT / a.name
    if (out / "protocol.json").is_file() and not a.force:
        raise SystemExit(f"{out / 'protocol.json'} exists: a protocol is frozen once (--force only before any run)")
    classes = load_classes(a.desc)
    splits, dev_scenes = {}, set()
    for split, dirs in (("dev", a.dev or []), ("test", a.test or [])):
        items = []
        for c in dirs:
            for stem, f, stratum in _corpus_items(Path(c).expanduser()):
                items.append(dict(stem=stem, corpus=Path(c).name, scene=scene_of(stem), sha256=_sha(f),
                                  cls=(classes.get(stem) or {}).get("cls") or stratum))
        if split == "dev":
            dev_scenes = {it["scene"] for it in items}
        else:                                                            # no scene of DEV in TEST (equivalent tree forms)
            dropped = [it["stem"] for it in items if it["scene"] in dev_scenes]
            items = [it for it in items if it["scene"] not in dev_scenes]
            splits["test_dropped_shared_scene"] = dropped
        splits[split] = items
    if a.real:
        splits["real"] = [dict(stem=s.split()[0]) for s in Path(a.real).expanduser().read_text().splitlines() if s.strip()]
    protocol = dict(schema=SCHEMA, name=a.name, created=datetime.datetime.now().isoformat(timespec="seconds"),
                    frozen_at_commit=_git_head(), hypothesis=a.hypothesis, kind=a.kind,
                    arms=dict(A=a.arm_a, B=a.arm_b), primary=f"good = one valid solid and IoU >= {GOOD_IOU} (exact before sampled)",
                    test="exact McNemar, one-sided (B better)" if a.kind == "improvement" else
                    f"non-inferiority of the good rate (margin {a.margin}) and paired Wilcoxon on the seconds",
                    alpha=ALPHA, margin=a.margin, min_cell=MIN_CELL, anchor_band=ANCHOR_BAND,
                    hosts=a.hosts, settings=a.settings,
                    decision="accept B only if every check of sci_test.verdict holds on TEST; anything else: reject or back "
                             "to DEV; runs outside the paired design (hosts, settings, load cap) are exploratory",
                    splits={k: len(v) for k, v in splits.items()})
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=1))
    (out / "manifest.json").write_text(json.dumps(splits, indent=1))
    print(f"frozen {a.name}: " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()) + f" -> {out}")


def cmd_power(a):
    proto = json.loads((OUT / a.name / "protocol.json").read_text()) if a.name else {}
    flip = a.flip
    rows = []
    for delta in (0.05, 0.10, 0.15, 0.20):
        p_disc = flip + delta                   # the replicate's flips (both ways) plus the change's own discordant pairs
        try:
            n = st.mcnemar_sample_size(min(1.0, p_disc), delta, alpha=proto.get("alpha", ALPHA), two_sided=False)
        except ValueError:
            n = None
        rows.append(dict(delta=delta, p_disc=round(min(1.0, p_disc), 3), pairs=n))
    for r in rows:
        print(f"  change of the good rate {r['delta']:.2f}: discordance {r['p_disc']:.3f} -> {r['pairs']} pairs")
    return rows


def _md(res: dict) -> str:
    c, v = res["compare"], res["verdict"]
    L = [f"# {res['name']} / {res['label']}", "", f"protocol frozen at {res.get('frozen_at_commit')}; analysed "
         f"{res['created']}; arms A = `{res['a']}`, B = `{res['b']}`", "",
         f"**Decision: {v['decision']}** ({v['kind']})", "",
         "| | A | B |", "|---|---|---|",
         f"| good | {c['a']['good']['k']}/{c['a']['good']['n']} ({c['a']['good']['lo']:.2f}-{c['a']['good']['hi']:.2f}) | "
         f"{c['b']['good']['k']}/{c['b']['good']['n']} ({c['b']['good']['lo']:.2f}-{c['b']['good']['hi']:.2f}) |",
         f"| one valid solid | {c['a']['one_solid']} | {c['b']['one_solid']} |",
         f"| median parts | {c['a'].get('median_operations')} | {c['b'].get('median_operations')} |",
         f"| median IoU | {c['a']['median_iou']} | {c['b']['median_iou']} |",
         f"| median seconds | {c['a']['median_seconds']} | {c['b']['median_seconds']} |",
         f"| stops | {c['a']['stops']} | {c['b']['stops']} |",
         f"| hosts | {c['a']['hosts']} | {c['b']['hosts']} |",
         f"| commits | {c['a']['commits']} | {c['b']['commits']} |", "",
         f"pairs {c['n']}; rescued {c['table']['rescues']}, broken {c['table']['breaks']}; McNemar p {c['mcnemar_p']:.4f} "
         f"(two-sided {c['mcnemar_p_two_sided']:.4f})"]
    if "delta_good" in c:
        d = c["delta_good"]
        L.append(f"; change of the good rate {d['delta']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
    if "seconds" in c:
        L.append(f"; seconds: median paired change {c['seconds']['median_diff']:+.1f}, p {c['seconds']['p']:.4f}")
    if "operations" in c:
        L.append(f"; parts: median paired change {c['operations']['median_diff']:+.1f}, p {c['operations']['p']:.4f}")
    if "without_stops" in c:
        w = c["without_stops"]
        L.append(f"; without the parts with a stop ({w['n']} pairs): rescued {w['table']['rescues']}, broken "
                 f"{w['table']['breaks']}, p {w['mcnemar_p']:.4f}")
    L += ["", f"net change {v['net_change']:+d}; noise floor {v['noise_floor'] if v['noise_floor'] is None else round(v['noise_floor'], 2)}"
          + (f" (replicate: {res['replicate']['flips']} flips of {res['replicate']['n']}, rate {res['replicate']['rate']:.3f})"
             if res.get("replicate") else ""), "", "| check | result |", "|---|---|"]
    L += [f"| {k} | {'not measured' if x is None else 'yes' if x else 'NO'} |" for k, x in v["checks"].items()]
    if c.get("classes"):
        L += ["", "| class | n | good A | good B | rescued | broken | p worse (Holm) |", "|---|---|---|---|---|---|---|"]
        for cell, x in sorted(c["classes"].items(), key=lambda kv: -kv[1]["n"]):
            L.append(f"| {cell} | {x['n']} | {x['good_a']} | {x['good_b']} | {x['table']['rescues']} | {x['table']['breaks']} | "
                     f"{x.get('p_worse_holm', '-') if isinstance(x.get('p_worse_holm'), str) else round(x.get('p_worse_holm', float('nan')), 3) if 'p_worse_holm' in x else '-'} |")
    if res.get("anchors"):
        L += ["", "| anchor | A | B | ok |", "|---|---|---|---|"]
        L += [f"| {p} | {x['a']} | {x['b']} | {x['ok']} |" for p, x in res["anchors"].items()]
    L += ["", f"rescued: {', '.join(c['rescued']) or '-'}", "", f"broken: {', '.join(c['broken']) or '-'}", ""]
    return "\n".join(L)


def cmd_analyze(a):
    proto_path = OUT / a.name / "protocol.json"
    proto = json.loads(proto_path.read_text()) if proto_path.is_file() else {}
    kind = proto.get("kind", "improvement")
    rows_a, rows_b = load_arm(a.a), load_arm(a.b)
    if a.split and (OUT / a.name / "manifest.json").is_file():          # only the parts of the frozen split
        keep = {it["stem"] for it in json.loads((OUT / a.name / "manifest.json").read_text()).get(a.split, [])}
        rows_a = {p: r for p, r in rows_a.items() if p in keep}
        rows_b = {p: r for p, r in rows_b.items() if p in keep}
    classes = load_classes(a.desc)
    if not classes and (OUT / a.name / "manifest.json").is_file():
        man = json.loads((OUT / a.name / "manifest.json").read_text())
        classes = {it["stem"]: dict(cls=it.get("cls")) for split in ("dev", "test") for it in man.get(split, []) if it.get("cls")}
    cmp = compare(rows_a, rows_b, classes, margin=proto.get("margin", MARGIN))
    rep, floor = None, None
    if a.replicate:
        r1, r2 = load_arm(a.replicate[0]), load_arm(a.replicate[1])
        common = sorted(set(r1) & set(r2))
        rep = st.flip_rate([r1[p]["good"] for p in common], [r2[p]["good"] for p in common])
        rep["hosts"] = [_arm_facts(list(r1.values()))["hosts"], _arm_facts(list(r2.values()))["hosts"]]
        floor = st.net_change_floor(rep["rate"], cmp["n"])
    anchors = anchor_check(load_arm(a.anchors_a), load_arm(a.anchors_b)) if a.anchors_a and a.anchors_b else None
    res = dict(schema=SCHEMA, name=a.name, label=a.label, created=datetime.datetime.now().isoformat(timespec="seconds"),
               frozen_at_commit=proto.get("frozen_at_commit"), analysed_at_commit=_git_head(), a=a.a, b=a.b,
               split=a.split, compare=cmp, replicate=rep, anchors=anchors,
               verdict=verdict(cmp, floor, anchors, kind, margin=proto.get("margin", MARGIN)))
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"result_{a.label}.json").write_text(json.dumps(res, indent=1, default=str))
    (out / f"result_{a.label}.md").write_text(_md(res))
    print(_md(res))
    return res


def cmd_waste(a):
    classes = load_classes(a.desc)
    runs = [waste_of_run(Path(d)) for pattern in a.runs for d in sorted(glob.glob(os.path.expanduser(pattern)))]
    s = waste_summary(runs, classes)
    if s.get("n"):
        print(f"{s['n']} runs, {s['seconds'] / 3600:.1f} h; on rejected pieces/candidates {s['rejected_share']:.1%}, "
              f"after the last accepted operation {s['after_last_gain_share']:.1%}; median candidates per accepted "
              f"operation {s['median_candidates_per_accepted']}; time-budget stops {s['time_budget_stops']}")
        print("time by step kind: " + ", ".join(f"{k} {v:.1%}" for k, v in list(s["share_by_kind"].items())[:8]))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(summary=s, runs=[w for w in runs if w]), indent=1))
    return s


def cmd_launch(a):
    proto = json.loads((OUT / a.name / "protocol.json").read_text())
    arm = proto["arms"][a.arm]
    s = proto.get("settings") or {}
    print(f"# protocol {a.name}, arm {a.arm} = {arm}, split {a.split}; hosts {proto.get('hosts')}; check load and users first")
    extra = "" if arm == "bare" else s.get("extra", {}).get(arm, "--mode tree")
    print(f"SHAPE_ATLAS_CODE={s.get('code', '<checkout with COMMIT>')} SHAPE_ATLAS_EXTRA=\"{extra}\" "
          f"bash ~/data/shape_reco/shape_atlas/rerun_v2.sh {a.split}_{arm} {s.get('jobs', 6)} <list> <corpus> <truths>")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("name")
    f.add_argument("--hypothesis", required=True)
    f.add_argument("--kind", choices=("improvement", "saving", "simpler"), default="improvement")
    f.add_argument("--arm-a", required=True)
    f.add_argument("--arm-b", required=True)
    f.add_argument("--dev", nargs="*")
    f.add_argument("--test", nargs="*")
    f.add_argument("--real")
    f.add_argument("--desc", nargs="*")
    f.add_argument("--margin", type=float, default=MARGIN)
    f.add_argument("--hosts", nargs="*", default=["server_a", "server_b"])
    f.add_argument("--settings", type=json.loads, default={})
    f.add_argument("--force", action="store_true")
    w = sub.add_parser("power")
    w.add_argument("name", nargs="?")
    w.add_argument("--flip", type=float, default=2 / 60)
    an = sub.add_parser("analyze")
    an.add_argument("name")
    an.add_argument("--a", required=True)
    an.add_argument("--b", required=True)
    an.add_argument("--label", default="test")
    an.add_argument("--split")
    an.add_argument("--replicate", nargs=2)
    an.add_argument("--anchors-a")
    an.add_argument("--anchors-b")
    an.add_argument("--desc", nargs="*")
    ws = sub.add_parser("waste")
    ws.add_argument("runs", nargs="+")
    ws.add_argument("--desc", nargs="*")
    ws.add_argument("--json")
    la = sub.add_parser("launch")
    la.add_argument("name")
    la.add_argument("arm", choices=("A", "B"))
    la.add_argument("split")
    a = p.parse_args(argv)
    return dict(freeze=cmd_freeze, power=cmd_power, analyze=cmd_analyze, waste=cmd_waste, launch=cmd_launch)[a.cmd](a)


if __name__ == "__main__":
    main()
