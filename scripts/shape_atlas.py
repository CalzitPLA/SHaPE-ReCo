#!/usr/bin/env python3
"""
The shape-class atlas (plan of 2026-10-29: "find out on which geometry types we lack; classify shapes by classical
topology mixed with the complexity of the shape tree"): every part of every corpus described and classified
(reco_engine/shape_class.py, CLASS_V1), joined with the stored reconstruction results, and counted per class.

    python3 scripts/shape_atlas.py describe --corpus <validate_known corpus> --domain gate --out desc.jsonl [--jobs 4]
    python3 scripts/shape_atlas.py describe --files '<glob>' --domain abc --out desc.jsonl
    python3 scripts/shape_atlas.py describe --list <file: name path [truth.json] per line, tab-separated if a path has spaces> --domain real --out desc.jsonl
    python3 scripts/shape_atlas.py calibrate desc.jsonl [...] --out docs/data/shape_class_calib.json
    python3 scripts/shape_atlas.py join desc.jsonl [...] --runs gate=<glob>@variant/part [...] --table abc_0930=<table.csv>
                                         [--calib docs/data/shape_class_calib.json] --out docs/data/shape_atlas.jsonl
    python3 scripts/shape_atlas.py report docs/data/shape_atlas.jsonl [--variant tree_lg] [--domain gate] [--md out.md]
                                         [--json out.json] [--fig out.pdf]
    python3 scripts/shape_atlas.py select desc.jsonl [...] --per-cell 20 [--in-scope] [--domain abc] --out list.txt
                                         [--strata-out strata.json]

describe: one row per part (name, domain, path, truth, desc = shape_class.describe, tree = shape_class.tree_facts of the
known tree). A corpus item without a STEP but with its truth gets a STEP built from the truth (the v5 parts come as
meshes); resumable (names already in --out are skipped), one freecadcmd per part, --jobs parts at a time.
calibrate: the leaf estimator for parts without a known tree: log2 leaves = a + b log2 carriers + c log2(1 + concave
curved faces), least squares over the known trees (one tree form per aegir scene); 5-fold bin accuracy and confusion;
ok when the accuracy reaches shape_class.CALIB_MIN_ACCURACY - else the C axis of unknown trees stays carrier bins.
join: run directories (scripts/path_atlas.py's layout and row: state, one valid solid, IoU exact before sampled, tier,
separation class) and the ABC batch table (scripts/parts_table.py) with the class of their part; good = one valid solid
and IoU >= 0.95 (RUNNING 3.4).
report: per class cell (T.S rows x C columns) n, good rate with its Wilson 95 % interval, median IoU and the dominant
cause of the failures (the separation class against a known tree, else invented vs missing); capability gaps (n >= 8,
upper bound below the overall rate, ranked by the deficit times the class's share among real parts) and coverage gaps
(classes of at least 1 % of the real parts with fewer than 5 known trees, on the carrier key T.S.K that both sides
have); the joint logistic model P(good) of the axes (Firth, docs/reports/abc_benchmark/abc_probability.py).
select: a class-stratified list (at most --per-cell parts per class, seeded), for a rerun or `validate_known.py fork
--strata-from`.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen"), str(ROOT / "scripts")]

from reco_engine import shape_class as sc  # noqa: E402

GOOD_IOU = 0.95
MIN_CELL = 8                    # a capability gap needs at least this many runs in its cell
REAL_SHARE = 0.01               # a coverage gap: a class of at least this share of the real parts ...
MIN_KNOWN = 5                   # ... with fewer known trees than this
REAL_DOMAINS = ("abc", "real")


def _load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).expanduser().read_text().splitlines() if l.strip()]
    return rows


def _probability():
    """docs/reports/abc_benchmark/abc_probability.py as a module (logit_fit, predict, cv_scores, wilson)."""
    path = ROOT / "docs" / "reports" / "abc_benchmark" / "abc_probability.py"
    spec = importlib.util.spec_from_file_location("abc_probability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------- describe

def _items(a):
    items = []
    for c in a.corpus or []:
        c = Path(c).expanduser()
        sel = json.loads((c / "selection.json").read_text())
        for it in sel["items"]:
            stem = it["stem"]
            found = [c / d / f"{stem}{e}" for d in ("step", "stl") for e in (".step", ".stp", ".stl")]
            path = next((p for p in found if p.is_file()), None)
            truth = c / "truths" / f"{stem}.json"
            items.append((stem, str(path) if path else None, str(truth) if truth.is_file() else None, a.domain))
    for pattern in a.files or []:
        items += [(Path(p).stem, p, None, a.domain) for p in sorted(glob.glob(os.path.expanduser(pattern)))]
    for lst in a.list or []:
        for line in Path(lst).expanduser().read_text().splitlines():
            parts = line.split("\t") if "\t" in line else line.split()     # tabs: paths with spaces
            if parts and not line.startswith("#"):
                items.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None, a.domain))
    return items


def describe_item(item) -> dict:
    """The descriptor row of one (name, path, truth, domain); never raises."""
    from reco_engine.brep import is_step
    name, path, truth, domain = item
    row = dict(name=name, domain=domain, truth=truth, version=sc.CLASS_VERSION)
    t0 = time.time()
    try:
        program = None
        if truth:
            t = json.loads(Path(truth).read_text())
            program = t["program"] if "program" in t and "type" not in t else t
            row["tree"] = sc.tree_facts(program)
            row["truth_meta"] = t.get("meta")
        if program is not None and (path is None or not is_step(path)):
            path = str(sc.step_of_program(program))          # the exact B-rep of the truth beats its mesh
        if path is None:
            raise FileNotFoundError(f"no STEP, STL or truth for {name}")
        row["path"] = path
        row["desc"] = sc.describe(path)
        if row["desc"].get("error"):
            row["error"] = row["desc"]["error"]
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:400]
    row["seconds"] = round(time.time() - t0, 2)
    return row


def cmd_describe(a):
    out = Path(a.out).expanduser()
    done = set()
    if out.is_file():
        old = _load([out])
        if a.retry_errors:                                       # drop the failed rows, describe those parts again
            old = [r for r in old if not r.get("error")]
            out.write_text("".join(json.dumps(r) + "\n" for r in old))
        done = {(r["name"], r["domain"]) for r in old}
    items = [i for i in _items(a) if (i[0], i[3]) not in done]
    if a.limit:
        items = items[:a.limit]
    print(f"{len(items)} parts to describe ({len(done)} already in {out})", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = err = 0
    t0 = time.time()
    with open(out, "a") as fh:
        if a.jobs > 1:
            from multiprocessing import Pool
            with Pool(a.jobs) as pool:
                rows = pool.imap_unordered(describe_item, items)
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n += 1
                    err += bool(row.get("error"))
                    if n % 25 == 0:
                        print(f"  {n}/{len(items)} ({err} errors) {time.time() - t0:.0f} s", flush=True)
        else:
            for item in items:
                row = describe_item(item)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                err += bool(row.get("error"))
                if n % 25 == 0:
                    print(f"  {n}/{len(items)} ({err} errors) {time.time() - t0:.0f} s", flush=True)
    print(f"described {n} ({err} errors) in {time.time() - t0:.0f} s -> {out}")


# ---------------------------------------------------------------------------------------------------- calibrate

def _features(d: dict):
    return [math.log2(d["carriers"]), math.log2(1 + (d.get("concave_curved") or 0))]


def cmd_calibrate(a):
    import numpy as np
    rows = [r for r in _load(a.descriptors) if r.get("tree") and not r.get("error") and (r.get("desc") or {}).get("carriers")]
    seen, known = set(), []
    for r in rows:                                               # one tree form per scene: they are the same solid
        scene = (r.get("truth_meta") or {}).get("scene") or r["name"]
        if scene not in seen:
            seen.add(scene)
            known.append(r)
    if len(known) < 10:
        raise SystemExit(f"only {len(known)} known trees with descriptors - nothing to calibrate")
    X = np.array([_features(r["desc"]) for r in known])
    y = np.log2([r["tree"]["leaves"] for r in known])
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X1, y, rcond=None)[0]
    res = y - X1 @ beta
    r2 = 1 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    rng = np.random.default_rng(0)                             # 5-fold bin accuracy
    idx = rng.permutation(len(y))
    pred = np.empty(len(y))
    for fold in np.array_split(idx, 5):
        train = np.setdiff1d(idx, fold)
        b = np.linalg.lstsq(X1[train], y[train], rcond=None)[0]
        pred[fold] = X1[fold] @ b
    names = [n for _, _, n in sc.LEAF_BINS]
    true_bins = [sc._bin(r["tree"]["leaves"], sc.LEAF_BINS) for r in known]
    pred_bins = [sc._bin(max(1, int(round(2 ** p))), sc.LEAF_BINS) for p in pred]
    acc = float(np.mean([t == p for t, p in zip(true_bins, pred_bins)]))
    near = float(np.mean([abs(names.index(t) - names.index(p)) <= 1 for t, p in zip(true_bins, pred_bins)]))
    conf = {t: dict(collections.Counter(p for tt, p in zip(true_bins, pred_bins) if tt == t)) for t in names}
    out = dict(version=sc.CLASS_VERSION, created=time.strftime("%Y-%m-%dT%H:%M:%S"), n=len(known),
               features=["log2 carriers", "log2 (1 + concave curved faces)"], a=float(beta[0]), b=float(beta[1]),
               c=float(beta[2]), r2=r2, rmse_log2=float(np.sqrt(np.mean(res ** 2))), accuracy=acc, within_one_bin=near,
               confusion=conf, ok=acc >= sc.CALIB_MIN_ACCURACY, min_accuracy=sc.CALIB_MIN_ACCURACY,
               sources=dict(collections.Counter(r["domain"] for r in known)),
               residual_by_domain={dom: float(np.mean([e for e, r in zip(-res, known) if r["domain"] == dom]))
                                   for dom in sorted({r["domain"] for r in known})})   # estimate - truth, log2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(known)} known trees: log2 leaves = {beta[0]:.3f} + {beta[1]:.3f} log2 carriers + {beta[2]:.3f} "
          f"log2(1 + concave); R2 {r2:.3f}; 5-fold bin accuracy {acc:.3f} (within one bin {near:.3f}) -> "
          f"{'ok' if out['ok'] else 'NOT ok: unknown trees keep carrier bins'}; {a.out}")
    for t in names:
        print(f"  {t:6s} -> {conf[t]}")
    print("  mean log2 residual (estimate - truth) per domain:", {k: round(v, 3) for k, v in out["residual_by_domain"].items()})


# ---------------------------------------------------------------------------------------------------- join

def _class_fields(d: dict, calib) -> dict:
    desc, tree = d.get("desc") or {}, d.get("tree")
    c = sc.classify(desc, tree, calib)
    kb = sc._bin(desc.get("carriers"), sc.CARRIER_BINS) or "K?"
    keep = ("b0", "b1", "b2", "b1_holes", "b1_rest", "family", "faces", "carriers", "axis_lines", "blends",
            "concave_curved", "free_form", "curved_share", "tau_rel", "convexity", "fill_obb", "aspect", "source")
    out = {k: desc.get(k) for k in keep}
    out.update(cls=c["key"], T=c["T"], S=c["S"], C=c["C"], c_source=c["c_source"], leaves=c["leaves"],
               flags=c["flags"], key_brep=f"{c['T']}.{c['S']}.{kb}")
    if tree:
        out.update(tree_depth=tree["depth"], tree_ops=tree["ops"], tree_partial=tree["partial"],
                   tree_axes=tree["axis_dirs"], tree_subtypes=tree["subtypes"])
    return out


def _runs_rows(spec: str):
    from path_atlas import _row
    domain, pattern = spec.split("=", 1)
    forced = None
    if "#" in pattern:                                           # <glob>[@layout]#<variant>: one name for all its runs
        pattern, forced = pattern.rsplit("#", 1)
    layout = "part/variant"
    if "@" in pattern:
        pattern, layout = pattern.rsplit("@", 1)
    rows = []
    for d in sorted(glob.glob(os.path.expanduser(pattern))):
        d = Path(d)
        if not (d / "record.json").is_file():
            continue
        p1, p2 = d.parent.name, d.name
        if layout == "variant/part":
            part, variant = p2, ("bare" if p1 == "runs" else p1.replace("runs_", ""))
        else:
            part, variant = p1, p2
        r = _row(domain, part, forced or variant, d, None)
        rec = json.loads((d / "record.json").read_text())
        ex = ((rec.get("comparison") or {}).get("exact") or {})
        vin = ((rec.get("input_facts") or {}).get("volume") or 0.0)
        if vin and ex.get("missing") is not None:
            r.update(missing=float(ex["missing"]) / vin, invented=float(ex.get("invented") or 0.0) / vin)
        r.pop("ops", None)
        r.pop("fingerprints", None)
        rows.append(r)
    return rows


def _table_rows(spec: str):
    variant, path = spec.split("=", 1)

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    rows = []
    for t in csv.DictReader(open(Path(path).expanduser())):
        e, s = num(t.get("exact_iou")), num(t.get("sampled_iou"))
        rows.append(dict(domain="abc", part=t["name"], variant=variant, state=t.get("state"),
                         one_valid_solid=t.get("one_valid_solid") == "True", iou=e if e is not None else s,
                         iou_kind="exact" if e is not None else "sampled" if s is not None else None,
                         missing=num(t.get("missing_share")), invented=num(t.get("invented_share")),
                         in_scope=t.get("in_scope") == "True", n_ops=num(t.get("operations")),
                         seconds=num(t.get("seconds"))))
    return rows


def cmd_join(a):
    desc = {}
    for r in _load(a.descriptors):
        if not r.get("error"):
            desc[r["name"]] = r
    calib = json.loads(Path(a.calib).read_text()) if a.calib and Path(a.calib).is_file() else None
    rows = []
    for spec in a.runs or []:
        rows += _runs_rows(spec)
    for spec in a.table or []:
        rows += _table_rows(spec)
    have = collections.Counter()
    for r in rows:
        r["good"] = bool(r.get("one_valid_solid") and r.get("iou") is not None and r["iou"] >= GOOD_IOU)
        d = desc.get(r["part"])
        if d is not None:
            r.update(_class_fields(d, calib))
            have[(r["domain"], r["variant"])] += 1
    parts = [dict(domain=d["domain"], part=name, variant=None, **_class_fields(d, calib)) for name, d in desc.items()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows + parts))
    per = collections.Counter((r["domain"], r["variant"]) for r in rows)
    print(f"{len(rows)} runs, {len(parts)} described parts -> {out}; calibration: "
          f"{'ok' if calib and calib.get('ok') else 'none / not ok (carrier bins)'}")
    for k in sorted(per):
        print(f"  {k[0]:10s} {k[1]:16s} runs {per[k]:5d}  classified {have[k]:5d}")


# ---------------------------------------------------------------------------------------------------- report

def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def _cause(r) -> str:
    if r.get("separation"):
        return r["separation"]
    if r.get("state") not in (None, "ended"):
        return f"run {r['state']}"
    if not r.get("one_valid_solid"):
        return "not one solid"
    if r.get("invented") is not None and r.get("missing") is not None:
        return "invented" if r["invented"] >= r["missing"] else "missing"
    return "?"


def _logit(rows, prob):
    """The joint logistic model of good per run: the axes as covariates."""
    import numpy as np
    names = ["log2(1+b1_holes)", "log2(1+b1_rest)", "void", "A", "M", "D", "F", "log2 leaves", "blend", "thin"]
    X, y = [], []
    for r in rows:
        if r.get("b1") is None or r["b1"] < 0 or r.get("S") in (None, "?") or r.get("T") in ("multi", "g?"):
            continue
        lv = r.get("leaves") or (r.get("carriers") and r["carriers"] / 3.0)
        if not lv:
            continue
        holes = r.get("b1_holes") or 0
        rest = r["b1_rest"] if r.get("b1_rest") is not None else r["b1"] - holes
        X.append([math.log2(1 + holes), math.log2(1 + rest),
                  float((r.get("b2") or 0) > 0)] + [float(r["S"] == s) for s in ("A", "M", "D", "F")] +
                 [math.log2(lv), float("blend" in (r.get("flags") or [])), float("thin" in (r.get("flags") or []))])
        y.append(float(r["good"]))
    X, y = np.array(X), np.array(y)
    if len(y) < 30 or y.min() == y.max():
        return None
    keep = [j for j in range(X.shape[1]) if X[:, j].std() > 0]            # a family absent from the corpus: no column
    X = X[:, keep]
    fit = prob.logit_fit(X, y)
    auc_m, auc_s, brier = prob.cv_scores(X, y, reps=5, k=5)
    terms = [dict(term=names[j], beta=float(fit["beta"][i + 1]), se=float(fit["se"][i + 1]),
                  odds=float(math.exp(fit["beta"][i + 1]))) for i, j in enumerate(keep)]
    return dict(n=len(y), good=int(y.sum()), intercept=float(fit["beta"][0]), terms=terms, cv_auc=auc_m,
                cv_auc_sd=auc_s, brier=brier)


def report(rows, real_rows, known_rows, prob):
    runs = [r for r in rows if r.get("cls")]
    k0 = sum(r["good"] for r in runs)
    p0 = k0 / len(runs) if runs else float("nan")
    cells = collections.defaultdict(list)
    for r in runs:
        cells[(f"{r['T']}.{r['S']}", r["C"])].append(r)
    real_parts = {r["part"]: r for r in real_rows}
    real_n = max(len(real_parts), 1)
    real_share = collections.Counter(f"{r['T']}.{r['S']}.{r['C']}" for r in real_parts.values())
    table = []
    for (ts, c), rs in sorted(cells.items()):
        k = sum(r["good"] for r in rs)
        p, lo, hi = prob.wilson(k, len(rs))
        bad = [r for r in rs if not r["good"]]
        causes = collections.Counter(_cause(r) for r in bad)
        table.append(dict(cell=f"{ts}.{c}", TS=ts, C=c, n=len(rs), good=k, rate=p, lo=lo, hi=hi,
                          median_iou=_median([r.get("iou") for r in rs]), causes=dict(causes.most_common(3)),
                          real_share=real_share.get(f"{ts}.{c}", 0) / real_n,
                          parts=sorted({r["part"] for r in rs})[:6]))
    gaps = [dict(t, deficit=p0 - t["rate"]) for t in table if t["n"] >= MIN_CELL and t["hi"] < p0]
    gaps.sort(key=lambda t: -(t["deficit"] * max(t["real_share"], 1.0 / real_n)))
    known = collections.Counter(r["key_brep"] for r in {r["part"]: r for r in known_rows}.values())
    real_brep = collections.Counter(r["key_brep"] for r in real_parts.values())
    coverage = [dict(cls=k, real_share=n / real_n, real=n, known=known.get(k, 0)) for k, n in real_brep.most_common()
                if n / real_n >= REAL_SHARE and known.get(k, 0) < MIN_KNOWN]
    known_ts = collections.Counter(k.rsplit(".", 1)[0] for k in known.elements())
    real_ts = collections.Counter(k.rsplit(".", 1)[0] for k in real_brep.elements())
    coverage_ts = [dict(cls=k, real_share=n / real_n, real=n, known=known_ts.get(k, 0)) for k, n in real_ts.most_common()
                   if n / real_n >= REAL_SHARE and known_ts.get(k, 0) < MIN_KNOWN]
    marginals = {}                                               # one axis or flag at a time
    for axis, value in (("T", lambda r: r["T"]), ("S", lambda r: r["S"]), ("C", lambda r: r["C"]),
                        ("flag", lambda r: r.get("flags") or ["none"])):
        groups = collections.defaultdict(list)
        for r in runs:
            v = value(r)
            for x in (v if isinstance(v, list) else [v]):
                groups[x].append(r)
        marginals[axis] = []
        for x, rs in sorted(groups.items()):
            k = sum(r["good"] for r in rs)
            p, lo, hi = prob.wilson(k, len(rs))
            marginals[axis].append(dict(value=x, n=len(rs), good=k, rate=p, lo=lo, hi=hi,
                                        median_iou=_median([r.get("iou") for r in rs])))
    return dict(version=sc.CLASS_VERSION, runs=len(runs), good=k0, rate=p0, cells=table, capability_gaps=gaps,
                coverage_gaps=coverage, coverage_gaps_ts=coverage_ts, real_parts=len(real_parts),
                known_by_ts=dict(known_ts), real_by_ts=dict(real_ts),
                known_parts=len({r["part"] for r in known_rows}), marginals=marginals, logit=_logit(runs, prob))


def _md(rep, title) -> str:
    f = lambda x: "-" if x is None or (isinstance(x, float) and math.isnan(x)) else \
        f"{(0.0 if abs(x) < 5e-4 else x):.3f}" if isinstance(x, float) else str(x)          # no "-0.000"
    L = [f"# {title}", "", f"CLASS_{rep['version'].upper()}; {rep['runs']} classified runs, {rep['good']} good "
         f"(one valid solid and IoU >= {GOOD_IOU}): overall rate {f(rep['rate'])}. Real parts: {rep['real_parts']}, "
         f"known trees: {rep['known_parts']}.", ""]
    for axis, name in (("T", "topology"), ("S", "surface family"), ("C", "tree complexity"), ("flag", "flags")):
        L += [f"**By {name}:** " + "; ".join(f"{m['value']} {m['good']}/{m['n']} ({f(m['rate'])}, {f(m['lo'])}-{f(m['hi'])})"
                                            for m in rep["marginals"][axis]), ""]
    L += ["## Cells", "",
         "| cell | n | good | rate | 95 % interval | median IoU | real share | causes of the failures |", "|---|---|---|---|---|---|---|---|"]
    for t in rep["cells"]:
        L.append(f"| {t['cell']} | {t['n']} | {t['good']} | {f(t['rate'])} | {f(t['lo'])}-{f(t['hi'])} | {f(t['median_iou'])} | "
                 f"{f(t['real_share'])} | {', '.join(f'{k} {v}' for k, v in t['causes'].items())} |")
    L += ["", f"## Capability gaps (n >= {MIN_CELL}, upper bound below {f(rep['rate'])})", ""]
    L += [f"- **{t['cell']}**: {t['good']}/{t['n']} good ({f(t['rate'])}, {f(t['lo'])}-{f(t['hi'])}), real share "
          f"{f(t['real_share'])}; failures: {', '.join(f'{k} {v}' for k, v in t['causes'].items())}; e.g. {', '.join(t['parts'][:3])}"
          for t in rep["capability_gaps"]] or ["- none at this size"]
    L += ["", f"## Coverage gaps (>= {REAL_SHARE:.0%} of the real parts, fewer than {MIN_KNOWN} known trees)", ""]
    L += [f"- {t['cls']}: {t['real']} real parts ({f(t['real_share'])}), {t['known']} known trees" for t in rep["coverage_gaps"]] or ["- none"]
    L += ["", "On the topology and family level (T.S):", ""]
    L += [f"- {t['cls']}: {t['real']} real parts ({f(t['real_share'])}), {t['known']} known trees" for t in rep["coverage_gaps_ts"]] or ["- none"]
    lg = rep.get("logit")
    if lg:
        L += ["", f"## Logistic model of good (n {lg['n']}, good {lg['good']}; 5 x 5-fold AUC {f(lg['cv_auc'])} +- {f(lg['cv_auc_sd'])}, Brier {f(lg['brier'])})", "",
              "| term | beta | SE | odds ratio |", "|---|---|---|---|"]
        L += [f"| {t['term']} | {t['beta']:+.3f} | {t['se']:.3f} | {t['odds']:.3f} |" for t in lg["terms"]]
    return "\n".join(L) + "\n"


MIN_SHOWN = 5                   # a cell with fewer runs is drawn neutral: its rate says nothing
DIV = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3b1ae", "#e66767", "#b72d2c"]   # blue <-> gray <-> red (dataviz)
INK, INK2, NEUTRAL = "#0b0b0b", "#52514e", "#f0efec"


def _ts_order(ts: str):
    t, s = ts.split(".")
    genus = [n for _, _, n in sc.GENUS_BINS]
    base = t.rstrip("v")
    return (genus.index(base) if base in genus else len(genus), t.endswith("v"),
            sc.FAMILIES.index(s) if s in sc.FAMILIES else len(sc.FAMILIES))


def _figure(rep, path, title):
    """The class cells as a heat map: rows topology . family (genus order, voids after, P A M D F), columns the
    complexity bins; the colour diverges at the overall good rate (red below, blue above), cells under MIN_SHOWN runs
    neutral with their count only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    import numpy as np
    shown = [t for t in rep["cells"] if any(u["TS"] == t["TS"] and u["n"] >= MIN_SHOWN for u in rep["cells"])]
    hidden = len({t["TS"] for t in rep["cells"]}) - len({t["TS"] for t in shown})     # rows of small cells only
    rows_ = sorted({t["TS"] for t in shown}, key=_ts_order)
    cols = [c for c in [n for _, _, n in sc.LEAF_BINS] + [n for _, _, n in sc.CARRIER_BINS] if any(t["C"] == c for t in shown)]
    if not rows_ or not cols:
        return
    p0 = rep["rate"]
    cmap = LinearSegmentedColormap.from_list("div", DIV[::-1])           # red low, blue high
    norm = TwoSlopeNorm(vmin=0.0, vcenter=min(max(p0, 0.01), 0.99), vmax=1.0)
    fig, ax = plt.subplots(figsize=(1.15 * len(cols) + 2.6, 0.36 * len(rows_) + 1.5))
    ax.set_facecolor("#fcfcfb")
    for t in shown:
        i, j = rows_.index(t["TS"]), cols.index(t["C"])
        small = t["n"] < MIN_SHOWN
        fill = NEUTRAL if small else cmap(norm(t["rate"]))
        ax.add_patch(plt.Rectangle((j - 0.5 + 0.02, i - 0.5 + 0.02), 0.96, 0.96, color=fill, lw=0))
        if small:
            ax.text(j, i, f"n={t['n']}", ha="center", va="center", fontsize=6.5, color=INK2)
        else:
            dark = abs(norm(t["rate"]) - 0.5) > 0.35
            ax.text(j, i, f"{t['good']}/{t['n']}", ha="center", va="center", fontsize=7,
                    color="white" if dark else INK)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(rows_) - 0.5, -0.5)
    ax.set_xticks(range(len(cols)), cols, fontsize=8)
    ax.set_yticks(range(len(rows_)), rows_, fontsize=8)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0, colors=INK2)
    ax.set_xlabel("tree complexity (C: leaves, known or estimated; K: carrier surfaces)", fontsize=8, color=INK2)
    ax.set_ylabel("topology . surface family", fontsize=8, color=INK2)
    ax.set_title(f"{title}\ngood / runs per class; colour: good rate against the overall {p0:.2f} (red below, blue above)"
                 f"\ngray: fewer than {MIN_SHOWN} runs" + (f"; {hidden} rows of small cells only not shown" if hidden else ""),
                 fontsize=8, color=INK)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("good rate", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    print(f"figure -> {path}")


def cmd_report(a):
    rows = _load([a.atlas])
    runs = [r for r in rows if r.get("variant") is not None]
    parts = [r for r in rows if r.get("variant") is None]
    if a.variant:
        runs = [r for r in runs if r["variant"] in a.variant.split(",")]
    if a.domain:
        runs = [r for r in runs if r["domain"] in a.domain.split(",")]
    if a.in_scope:                                               # one solid, analytic faces only, genus known
        runs = [r for r in runs if r.get("S") not in ("F", "?") and r.get("T") not in ("multi", "g?")]
    real = [r for r in parts if r["domain"] in REAL_DOMAINS and r.get("S") != "F" and r.get("T") != "multi"]
    known = [r for r in parts if r.get("c_source") == "tree"]
    if a.known_without:                                          # coverage as it was without these known-tree sources
        known = [r for r in known if r["domain"] not in a.known_without.split(",")]
    rep = report(runs, real, known, _probability())
    title = a.title or f"Shape-class atlas ({a.domain or 'all domains'}, {a.variant or 'all variants'}" + \
        (", in scope)" if a.in_scope else ")")
    md = _md(rep, title)
    print(md)
    if a.md:
        Path(a.md).write_text(md)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1, default=str))
    if a.fig:
        _figure(rep, a.fig, title)


# ---------------------------------------------------------------------------------------------------- select

def cmd_select(a):
    calib = json.loads(Path(a.calib).read_text()) if a.calib and Path(a.calib).is_file() else None
    rows = list({r["name"]: r for r in _load(a.descriptors) if not r.get("error")}.values())   # a later file wins
    if a.domain:
        rows = [r for r in rows if r["domain"] in a.domain.split(",")]
    by = collections.defaultdict(list)
    for r in rows:
        c = _class_fields(r, calib)
        if a.in_scope and (c["S"] in ("F", "?") or c["T"] in ("multi", "g?")):
            continue
        by[c["key_brep"] if a.brep_key else c["cls"]].append(r)
    if a.min_cell:                                               # only classes with enough parts to say something
        by = {k: v for k, v in by.items() if len(v) >= a.min_cell}
    rng = random.Random(a.seed)
    picked, strata = [], {}
    for key in sorted(by):
        pool = by[key][:]
        rng.shuffle(pool)
        for r in pool[:a.per_cell]:
            picked.append(r)
            strata[r["name"]] = key
    Path(a.out).write_text("".join(f"{r['name']}\t{r['path']}\n" for r in picked))
    if a.strata_out:
        Path(a.strata_out).write_text(json.dumps(strata, indent=1))
    print(f"{len(picked)} parts from {len(by)} classes (at most {a.per_cell} each) -> {a.out}")
    for key in sorted(by):
        print(f"  {key:22s} {len(by[key]):5d} -> {min(len(by[key]), a.per_cell)}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("describe")
    p.add_argument("--corpus", nargs="*", default=[], help="validate_known corpora (selection.json, truths/, step/, stl/)")
    p.add_argument("--files", nargs="*", default=[], help="globs of STEP/STL files")
    p.add_argument("--list", nargs="*", default=[], help="files of 'name path [truth.json]' lines")
    p.add_argument("--domain", required=True, help="gate, validation, abc, real, aegir, ...")
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--retry-errors", action="store_true", help="describe again the parts whose row has an error")
    p = sub.add_parser("calibrate")
    p.add_argument("descriptors", nargs="+")
    p.add_argument("--out", default=str(ROOT / "docs" / "data" / "shape_class_calib.json"))
    p = sub.add_parser("join")
    p.add_argument("descriptors", nargs="+")
    p.add_argument("--runs", nargs="*", default=[], help="<domain>=<runs glob>[@variant/part][#variant] as scripts/path_atlas.py "
                                                          "build; #variant names all its runs")
    p.add_argument("--table", nargs="*", default=[], help="<variant>=<parts_table.py table.csv> (the ABC batch)")
    p.add_argument("--calib", default=str(ROOT / "docs" / "data" / "shape_class_calib.json"))
    p.add_argument("--out", default=str(ROOT / "docs" / "data" / "shape_atlas.jsonl"))
    p = sub.add_parser("report")
    p.add_argument("atlas")
    p.add_argument("--variant", default="")
    p.add_argument("--domain", default="")
    p.add_argument("--title", default="")
    p.add_argument("--md", default="")
    p.add_argument("--json", default="")
    p.add_argument("--fig", default="")
    p.add_argument("--in-scope", action="store_true", help="only runs of one-solid parts without free-form faces")
    p.add_argument("--known-without", default="", help="known-tree domains left out of the coverage count, e.g. machined")
    p = sub.add_parser("select")
    p.add_argument("descriptors", nargs="+")
    p.add_argument("--per-cell", type=int, default=20)
    p.add_argument("--min-cell", type=int, default=0, help="only classes with at least this many parts")
    p.add_argument("--domain", default="")
    p.add_argument("--in-scope", action="store_true", help="one solid, no free-form faces")
    p.add_argument("--brep-key", action="store_true", help="stratify by T.S.K (carrier bins) instead of the class key")
    p.add_argument("--calib", default=str(ROOT / "docs" / "data" / "shape_class_calib.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--strata-out", default="")
    a = ap.parse_args(argv)
    dict(describe=cmd_describe, calibrate=cmd_calibrate, join=cmd_join, report=cmd_report, select=cmd_select)[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
