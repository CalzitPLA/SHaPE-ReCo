#!/usr/bin/env python3
"""
The validation cycle's driver: shapes whose CSG tree is known are reconstructed on the institute server, every run is
rated against its tree (reco_engine/validation.py), and the ratings are pulled back and reported - tiers, defect classes
and the separation point of every run.

    python3 scripts/validate_known.py select  --out <corpus> [--aegir-per-class 3] [--v5-per-stratum 1] [--seed 0]
    python3 scripts/validate_known.py build   <corpus>              # meshes of the aegir truths + the gate (on the server)
    python3 scripts/validate_known.py start   <corpus> [--server server_b] [--cpuset 8-13] [--jobs 6] [--timeout 2700]
    python3 scripts/validate_known.py status  <corpus> [--server server_b]
    python3 scripts/validate_known.py rate    <corpus>              # (re)rate the runs in place (on the server)
    python3 scripts/validate_known.py regrade <corpus>              # tiers again from the stored vectors (GRADE_V1)
    python3 scripts/validate_known.py pull    <corpus> [--server server_b]
    python3 scripts/validate_known.py report  <corpus>              # per stratum: tiers, metrics, defect classes, separation
    python3 scripts/validate_known.py review  <corpus> <stem>       # FCStd of the program, the culprit labelled, + the truth
    python3 scripts/validate_known.py fork    <corpus> --out <gate> [--subset 60] [--strata-from strata.json]   # the same shapes, no runs: a gate
    python3 scripts/validate_known.py compare <A> <B> [--rating-a rating.json] [--rating-b rating.json]

The regression gate (2026-10-02): `fork` freezes a corpus's shapes (stratified `--subset`, deterministic) into a new
corpus without runs; `start` it from the checkout under test (start syncs the checkout that runs it); `compare` the
baseline's ratings with the new ones, paired by stem: tier moves, share of A and median IoU with paired bootstrap 95 %
intervals, the runs whose IoU fell. `compare <corpus> <corpus> --rating-b rating_polished.json` compares two ratings of
the same runs (scripts/polish_runs.py).

A corpus directory: selection.json (strata, items with their gate, exclusions with the reason), truths/<stem>.json
(reco_engine/truth.program_json), stl/<stem>.stl and, for the aegir trees, step/<stem>.step (built by `build`),
list.txt (the stems that passed the gate), runs/<stem>/ (run_part.py's files, rating.json among them).
On the server the corpus lives in ~/data/shape_reco/validation/<name>/ and the code in
/data/user/shape_reco/wt/validation/; `pull` brings back only the small JSON files.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

AEGIR_STACK = Path.home() / "Documents/Sandbox_aegir/aegir/geometry/sandbox/stack"
V5_TEST = ROOT / "dataset_gen/local_dataset_v5/medium/test"
REMOTE_ROOT = "~/data/shape_reco/validation"
REMOTE_CODE = "/data/user/shape_reco/wt/validation"
VENV = "/data/user/shape_reco/venv_tests/bin/activate"
MODELS_FROM = "/data/user/shape_reco/tests_checkout"     # the server's copy of models/ and checkpoints*/ (2026-09-30)
FREECAD_BIN = "/data/user/shape_reco/bin"                 # FreeCAD 1.0 for every server (server_b's /usr/bin one is 0.20)
SMALL = ("record.json", "rating.json", "program.json", "path.jsonl", "exact.json", "steps.jsonl")


def _load(corpus: Path) -> dict:
    return json.loads((corpus / "selection.json").read_text())


def _save(corpus: Path, sel: dict):
    (corpus / "selection.json").write_text(json.dumps(sel, indent=1, default=str))


# ------------------------------------------------------------------------------------------------------ select

def _scene(stem: str) -> str:
    return "_".join(stem.split("_")[:3])


def _cone_band(tree) -> bool:
    """A sphere limited in latitude by cones: FreeCAD's builder cannot make it (reco_field/freecad_builder.py)."""
    from reco_engine.csg_surface import compile_tree
    from reco_engine.truth import sp
    for leaf in compile_tree(tree).leaves:
        p = leaf.prim
        if isinstance(p, sp.Sphere) and p.latitude == "cone" and not (p.v[0] <= -1.5707963 and p.v[1] >= 1.5707963):
            return True
    return False


def cmd_select(a):
    from reco_engine.truth import iter_v5_records, program_json, truth_from_aegir, truth_from_v5, truth_gate, v5_exact
    corpus = Path(a.out)
    for d in ("truths", "stl", "step"):
        (corpus / d).mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    items, excluded = [], []
    taken_before = set()                                   # stems of other corpora (disjoint splits)
    for other in (a.exclude_from or []):
        taken_before |= {i["stem"] for i in json.loads((Path(other) / "selection.json").read_text())["items"]}
    if a.aegir_per_class > 0:
        classes = set(a.classes.split(",")) if a.classes else None
        for cdir in sorted(p for p in AEGIR_STACK.iterdir() if p.is_dir() and (classes is None or p.name in classes)):
            scenes = collections.defaultdict(list)
            for f in sorted(cdir.glob("*.json")):
                scenes[_scene(f.stem)].append(f)
            keys = sorted(scenes)
            rng.shuffle(keys)
            taken = 0
            for key in keys:
                if taken >= a.aegir_per_class:
                    break
                f = scenes[key][0]                                         # one tree form per scene: the run sees the solid
                stem = f"aegir_{f.stem}"
                if stem in taken_before or any(f"aegir_{g.stem}" in taken_before for g in scenes[key]):
                    continue
                t = truth_from_aegir(f)
                if _cone_band(t.tree):
                    excluded.append(dict(stem=stem, stratum=cdir.name, reason="cone-latitude sphere band (FreeCAD cannot build it)"))
                    continue
                t.meta.update(stratum=cdir.name, scene=key, file=str(f), leaves=len(t.compiled.leaves))
                t.name, t.source = stem, "aegir"
                (corpus / "truths" / f"{stem}.json").write_text(json.dumps(program_json(t)))
                items.append(dict(stem=stem, source="aegir", stratum=cdir.name, leaves=len(t.compiled.leaves), gate=None))
                taken += 1
    if a.v5_per_stratum > 0:
        strata = collections.defaultdict(list)
        for rec, shard in iter_v5_records(a.v5_dir):
            ok, kind = v5_exact(rec)
            if ok:
                strata[(rec["topology"], rec.get("subclass") or "full", kind)].append((rec, shard))
        n_v5 = 0
        for key in sorted(strata):
            if a.v5_max and n_v5 >= a.v5_max:
                break
            pool = strata[key]
            rng.shuffle(pool)
            pool = [(r, sh) for r, sh in pool if f"v5_{key[0]}_{key[1]}_{key[2]}_{Path(sh).name}_{r.get('index')}" not in taken_before]
            for rec, shard in pool[:a.v5_per_stratum]:
                t = truth_from_v5(rec, shard)
                stem = f"v5_{key[0]}_{key[1]}_{key[2]}_{Path(shard).name}_{rec.get('index')}"   # the index repeats across shards
                g = truth_gate(t, t.input_path, volume_ref=t.meta.get("freecad_volume_mm3"))
                if not g["ok"]:
                    excluded.append(dict(stem=stem, stratum="/".join(key), reason="; ".join(g["reasons"])))
                    continue
                shutil.copyfile(t.input_path, corpus / "stl" / f"{stem}.stl")
                t.meta.update(stratum="/".join(key), genus=g["genus"], leaves=len(t.compiled.leaves))
                t.name = stem
                (corpus / "truths" / f"{stem}.json").write_text(json.dumps(program_json(t)))
                items.append(dict(stem=stem, source="v5", stratum="/".join(key), leaves=len(t.compiled.leaves),
                                  gate=dict(ok=True, mismatch=g["mismatch"], genus=g["genus"])))
                n_v5 += 1
    stems = [i["stem"] for i in items]
    if len(stems) != len(set(stems)):
        raise SystemExit(f"duplicate stems: {sorted(x for x in set(stems) if stems.count(x) > 1)[:5]}")
    sel = dict(name=corpus.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), seed=a.seed, aegir_per_class=a.aegir_per_class,
               v5_per_stratum=a.v5_per_stratum, items=items, excluded=excluded)
    _save(corpus, sel)
    (corpus / "list.txt").write_text("".join(f"{i['stem']}\n" for i in items if i["gate"] and i["gate"].get("ok")))
    print(f"{len(items)} shapes ({sum(i['source'] == 'aegir' for i in items)} aegir, {sum(i['source'] == 'v5' for i in items)} v5), "
          f"{len(excluded)} excluded -> {corpus}")


# ------------------------------------------------------------------------------------------------------- build

def cmd_build(a):
    """Meshes of the truths that have none (FreeCAD), then the gate for every unchecked truth; list.txt = what passed."""
    from reco_engine.truth import program_json, truth_from_json, truth_gate
    from reco_field.export import build_in_freecad
    corpus = Path(a.corpus)
    sel = _load(corpus)
    for d in ("step", "stl"):                                    # a fork holds truths/ only
        (corpus / d).mkdir(exist_ok=True)
    for item in sel["items"]:
        gate = item.get("gate") or {}
        if gate.get("ok") is not None and not (a.retry_failed and gate.get("ok") is False):
            continue
        stem = item["stem"]
        tpath = corpus / "truths" / f"{stem}.json"
        t = truth_from_json(tpath)
        try:
            ex = build_in_freecad(t.tree, corpus / "build", stem, timeout=a.timeout)
        except Exception as exc:
            item["gate"] = dict(ok=False, reasons=[f"build: {type(exc).__name__}: {str(exc)[:200]}"])
            continue
        if not ex.valid or ex.n_solids != 1:
            item["gate"] = dict(ok=False, reasons=[f"build: valid={ex.valid} solids={ex.n_solids}"])
            continue
        shutil.move(str(ex.step), corpus / "step" / f"{stem}.step")
        shutil.move(str(ex.stl), corpus / "stl" / f"{stem}.stl")
        g = truth_gate(t, corpus / "stl" / f"{stem}.stl", volume_ref=float(ex.volume))
        item["gate"] = dict(ok=g["ok"], reasons=g["reasons"], mismatch=g["mismatch"], genus=g["genus"], volume=float(ex.volume))
        if g["ok"]:
            d = program_json(t)
            d["meta"].update(genus=g["genus"], volume_occt=float(ex.volume))
            tpath.write_text(json.dumps(d))
        _save(corpus, sel)
    _save(corpus, sel)
    ok = [i["stem"] for i in sel["items"] if i.get("gate") and i["gate"].get("ok")]
    (corpus / "list.txt").write_text("".join(f"{s}\n" for s in ok))
    bad = collections.Counter((i["gate"] or {}).get("reasons", ["?"])[0][:60] for i in sel["items"]
                              if not (i.get("gate") or {}).get("ok"))
    print(f"gate: {len(ok)} of {len(sel['items'])} ok; failures: {dict(bad)}")


# ---------------------------------------------------------------------------------------------- server: start ...

def _sh(cmd, check=True, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check, text=True, **kw)


def cmd_start(a):
    corpus = Path(a.corpus)
    name = corpus.name
    remote = f"{REMOTE_ROOT}/{name}"
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (ROOT / ".commit").write_text(commit + "\n")
    _sh(f"ssh {a.server} 'mkdir -p {a.code} {remote}/runs'")
    _sh(f"rsync -a --delete --exclude .git --exclude 'dataset_gen/local_dataset*' --exclude '/checkpoints*' "
        f"--exclude field_cache --exclude 'sdf_cache*' --exclude archiv --exclude '*.pyc' --exclude /models "
        f"{ROOT}/ {a.server}:{a.code}/")
    # the trained models: the server's own copies, read only (the runs must see the framework's real type stage)
    _sh(f"ssh {a.server} 'cd {a.code} && for d in {MODELS_FROM}/models {MODELS_FROM}/checkpoints*; do "
        f"ln -sfn $d .; done; ls -d models checkpoints* | wc -l'")
    _sh(f"rsync -a --exclude runs {corpus}/ {a.server}:{remote}/")
    env = (f"TRUTHS={remote}/truths CPUSET={a.cpuset} JOBS={a.jobs} GPUS=0 MIN_FREE={a.min_free} TIMEOUT={a.timeout} THREADS={a.threads} "
           f"BUDGET={a.budget} POINTS={a.points} MAX_LOAD={a.max_load}")
    job = (f"cd {a.code} && source {VENV} && export MPLCONFIGDIR=/tmp/mpl_$USER XDG_CACHE_HOME={remote}/cache "
           f"PATH={FREECAD_BIN}:\\$PATH SHAPE_RECO_FREECADCMD={FREECAD_BIN}/freecadcmd && "
           f"taskset -c {a.cpuset} nice -n 10 python scripts/validate_known.py build {remote} && "
           f"{env} bash server/server_a/run_parts.sh {remote}/runs {remote}/list.txt {remote}")
    _sh(f"ssh {a.server} \"nohup bash -c '{job}' > {remote}/start.log 2>&1 < /dev/null & disown; sleep 1; echo started\"")
    print(f"log: {a.server}:{remote}/start.log   runs: {a.server}:{remote}/runs/")


def cmd_status(a):
    remote = f"{REMOTE_ROOT}/{Path(a.corpus).name}"
    script = ("import json,glob,collections,os,sys;"
              "rs=sorted(glob.glob(os.path.expanduser(sys.argv[1])+'/runs/*/record.json'));"
              "st=collections.Counter(json.load(open(r)).get('state') for r in rs);"
              "tr=collections.Counter(json.load(open(r)).get('rating',{}).get('tier') for r in rs);"
              "print(len(rs),'records',dict(st),'tiers',dict(tr))")
    _sh(f"ssh {a.server} \"tail -2 {remote}/start.log; ls {remote}/runs/DONE.* 2>/dev/null; wc -l < {remote}/list.txt; "
        f"python3 -c \\\"{script}\\\" {remote}\"", check=False)


def cmd_rate(a):
    from reco_engine.validation import rate_run
    corpus = Path(a.corpus).expanduser()
    for rd in sorted((corpus / "runs").glob("*/record.json")):
        stem = rd.parent.name
        truth = corpus / "truths" / f"{stem}.json"
        if not truth.is_file():
            continue
        try:
            r = rate_run(rd.parent, truth)
            print(stem, r["grade"]["tier"], (r.get("attribution") or {}).get("separation", {}) and
                  r["attribution"]["separation"].get("cls"), flush=True)
        except Exception as exc:
            print(stem, "rating failed:", type(exc).__name__, str(exc)[:200], flush=True)


def cmd_regrade(a):
    from reco_engine.validation import regrade
    corpus = Path(a.corpus)
    for rp in sorted((corpus / "runs").glob("*/rating.json")):
        r = json.loads(rp.read_text())
        if "vector" not in r:
            continue
        r["grade"] = regrade(r)
        rp.write_text(json.dumps(r, indent=1))
        print(rp.parent.name, r["grade"]["tier"])


def cmd_pull(a):
    corpus = Path(a.corpus)
    remote = f"{REMOTE_ROOT}/{corpus.name}"
    inc = " ".join(f"--include '{f}'" for f in SMALL + ("joblog*", "DONE*", "machine*", "selection.json", "start.log"))
    _sh(f"rsync -a --prune-empty-dirs --include '*/' {inc} --exclude '*' {a.server}:{remote}/ {corpus}/")


# ------------------------------------------------------------------------------------------------------ report

def _med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def cmd_report(a):
    corpus = Path(a.corpus)
    sel = _load(corpus)
    stratum = {i["stem"]: i["stratum"] for i in sel["items"]}
    rows = []
    for stem in [l.strip() for l in (corpus / "list.txt").read_text().splitlines() if l.strip()]:
        rp = corpus / "runs" / stem / "rating.json"
        rec = corpus / "runs" / stem / "record.json"
        r = json.loads(rp.read_text()) if rp.is_file() else None
        state = json.loads(rec.read_text()).get("state") if rec.is_file() else "not run"
        rows.append(dict(stem=stem, stratum=stratum.get(stem, "?"), state=state, rating=r))
    by = collections.defaultdict(list)
    for row in rows:
        by[row["stratum"]].append(row)
    out = dict(corpus=corpus.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), strata={}, overall={})

    def summarise(rs):
        rated = [x["rating"] for x in rs if x["rating"]]
        tiers = collections.Counter(r["grade"]["tier"] for r in rated)
        seps = [((r.get("attribution") or {}).get("separation") or {}) for r in rated]
        vol = [((r.get("vector") or {}).get("volume") or {}) for r in rated]
        surf = [((r.get("vector") or {}).get("surface") or {}) for r in rated]
        return dict(n=len(rs), rated=len(rated), not_rated=collections.Counter(x["state"] for x in rs if not x["rating"]),
                    tiers=dict(sorted(tiers.items())), iou=_med([v.get("iou") for v in vol]),
                    hausdorff=_med([s.get("hausdorff") for s in surf]), rms=_med([s.get("rms") for s in surf]),
                    unmeasured=sum(bool(r.get("unmeasured")) for r in rated),
                    separation_class=dict(collections.Counter(s.get("cls") for s in seps if s)),
                    separation_cause=dict(collections.Counter(s.get("cause") for s in seps if s)),
                    separation_relative_time=_med([s.get("time") / max(r.get("n_ops", 1), 1)
                                                   for s, r in zip(seps, rated) if s and s.get("time") is not None]))
    for k in sorted(by):
        out["strata"][k] = summarise(by[k])
    out["overall"] = summarise(rows)
    (corpus / "report.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"{'stratum':28s} {'n':>3s} {'rated':>5s}  tiers{'':22s} {'IoU':>6s} {'H/D':>8s}  separation (class)")
    for k, s in list(out["strata"].items()) + [("ALL", out["overall"])]:
        tiers = " ".join(f"{t}{n}" for t, n in s["tiers"].items())
        cls = ", ".join(f"{c} {n}" for c, n in sorted(s["separation_class"].items(), key=lambda kv: -kv[1])[:3])
        iou = f"{s['iou']:.4f}" if s["iou"] is not None else "-"
        hd = f"{s['hausdorff']:.2e}" if s["hausdorff"] is not None else "-"
        print(f"{k:28s} {s['n']:3d} {s['rated']:5d}  {tiers:27s} {iou:>6s} {hd:>8s}  {cls}")
    print(f"-> {corpus / 'report.json'}")


# ------------------------------------------------------------------------------------------------------ gate

def cmd_fork(a):
    corpus, out = Path(a.corpus), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    sel = _load(corpus)
    stems = [l.strip() for l in (corpus / "list.txt").read_text().splitlines() if l.strip()]
    stratum = {i["stem"]: i["stratum"] for i in sel["items"]}
    if a.strata_from:                    # strata of another kind, e.g. shape classes (scripts/shape_atlas.py select
        # --strata-out): only the stems it names, also before `build` (an unbuilt pool's list.txt is empty)
        stratum = json.loads(Path(a.strata_from).read_text())
        known = {i["stem"] for i in sel["items"]}
        stems = sorted(s for s in stratum if s in known)
    if a.subset and a.subset < len(stems):
        by = collections.defaultdict(list)
        for s in stems:
            by[stratum.get(s, "?")].append(s)
        rng = random.Random(a.seed)
        for k in by:
            rng.shuffle(by[k])
        picked, k = [], 0                                            # round robin over the strata: every stratum first
        while len(picked) < a.subset:
            for key in sorted(by):
                if k < len(by[key]) and len(picked) < a.subset:
                    picked.append(by[key][k])
            k += 1
        stems = sorted(picked)
    (out / "truths").mkdir(parents=True)
    for s in stems:
        shutil.copy2(corpus / "truths" / f"{s}.json", out / "truths" / f"{s}.json")
    keep = set(stems)
    items = [dict(i, stratum=stratum.get(i["stem"], i["stratum"])) for i in sel["items"] if i["stem"] in keep]
    sel = dict(sel, name=out.name, created=time.strftime("%Y-%m-%dT%H:%M:%S"), forked_from=corpus.name,
               fork_seed=a.seed, strata_from=a.strata_from or None, items=items)
    _save(out, sel)
    (out / "list.txt").write_text("".join(f"{s}\n" for s in stems))
    print(f"{out}: {len(stems)} shapes from {corpus.name} ({len({i['stratum'] for i in sel['items']})} strata)")


def _bootstrap(values_a, values_b, stat, n: int = 10_000, seed: int = 0):
    """(stat(b) - stat(a), 2.5 %, 97.5 %) over paired resamples of the indices."""
    import numpy as np
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = np.array([stat(b[i]) - stat(a[i]) for i in idx])
    return float(stat(b) - stat(a)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare_ratings(pairs) -> dict:
    """pairs: [(stem, rating_a, rating_b)] -> the paired comparison (cmd_compare)."""
    import numpy as np
    tiers = "ABCDE"
    ta = [p[1]["grade"]["tier"] for p in pairs]
    tb = [p[2]["grade"]["tier"] for p in pairs]
    ia = [((p[1].get("vector") or {}).get("volume") or {}).get("iou") for p in pairs]
    ib = [((p[2].get("vector") or {}).get("volume") or {}).get("iou") for p in pairs]
    out = dict(n=len(pairs), tiers_a={t: ta.count(t) for t in tiers}, tiers_b={t: tb.count(t) for t in tiers},
               moves=dict(collections.Counter(f"{x}->{y}" for x, y in zip(ta, tb))),
               better=sum(tiers.index(y) < tiers.index(x) for x, y in zip(ta, tb)),
               worse=sum(tiers.index(y) > tiers.index(x) for x, y in zip(ta, tb)))
    if pairs:
        out["share_A"] = _bootstrap([t == "A" for t in ta], [t == "A" for t in tb], np.mean)
        both = [(x, y) for x, y in zip(ia, ib) if x is not None and y is not None]
        if both:
            xa, xb = zip(*both)
            out["median_iou"] = (float(np.median(xa)), float(np.median(xb)))
            out["median_iou_change"] = _bootstrap(xa, xb, np.median)
            out["iou_fell"] = sorted(((s, x, y) for (s, _, _), x, y in zip(pairs, ia, ib)
                                      if x is not None and y is not None and y < x - 1e-6), key=lambda r: r[2] - r[1])
    return out


def cmd_compare(a):
    ca, cb = Path(a.a).expanduser(), Path(a.b).expanduser()
    pairs = []
    for ra in sorted((ca / "runs").glob(f"*/{a.rating_a}")):
        rb = cb / "runs" / ra.parent.name / a.rating_b
        if rb.is_file():
            pairs.append((ra.parent.name, json.loads(ra.read_text()), json.loads(rb.read_text())))
    c = compare_ratings(pairs)
    print(f"{c['n']} runs paired ({ca.name}/{a.rating_a} -> {cb.name}/{a.rating_b})")
    print(f"tiers  {' '.join(f'{t}{n}' for t, n in c['tiers_a'].items())}  ->  "
          f"{' '.join(f'{t}{n}' for t, n in c['tiers_b'].items())}   better {c['better']}, worse {c['worse']}")
    if c["n"]:
        d, lo, hi = c["share_A"]
        print(f"share A  {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    if "median_iou" in c:
        d, lo, hi = c["median_iou_change"]
        print(f"median IoU  {c['median_iou'][0]:.5f} -> {c['median_iou'][1]:.5f}  ({d:+.2e} [{lo:+.2e}, {hi:+.2e}])")
        for s, x, y in c["iou_fell"]:
            print(f"  IoU fell  {s}: {x:.5f} -> {y:.5f}")
    if a.out:
        Path(a.out).write_text(json.dumps(c, indent=1))
    return c


# ------------------------------------------------------------------------------------------------------ review

def cmd_review(a):
    """The run's program as FreeCAD groups (one object per operation, the culprit labelled) and the truth as a solid,
    for the user to open in FreeCAD: <corpus>/review/<stem>/."""
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.truth import truth_from_json
    from reco_engine.validation import program_ops
    from reco_field.export import build_groups, build_in_freecad
    corpus = Path(a.corpus)
    run = corpus / "runs" / a.stem
    out = corpus / "review" / a.stem
    result = ReconstructionResult.from_dict(json.loads((run / "program.json").read_text()))
    rating = json.loads((run / "rating.json").read_text()) if (run / "rating.json").is_file() else {}
    sep = (rating.get("attribution") or {}).get("separation") or {}
    ops = program_ops(result)
    accepted = [o for o in result.operations if o.accepted]
    labelled = []
    for k, ((op, prim), o) in enumerate(zip(ops, accepted)):
        label = f"{k:02d} {op} {o.primitive.kind}"
        if sep.get("culprit") == k:
            label += f"  <- SEPARATION: {sep.get('cls')} ({sep.get('cause')})"
        labelled.append((op, label, prim))
    g = build_groups(labelled, out, a.stem)
    truth = truth_from_json(corpus / "truths" / f"{a.stem}.json")
    tx = build_in_freecad(truth.tree, out, f"{a.stem}_truth")
    (out / "separation.json").write_text(json.dumps(dict(separation=sep, tier=(rating.get("grade") or {}).get("tier")), indent=1))
    print(f"open in FreeCAD: {g.get('fcstd')}  (truth: {tx.step})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("select")
    p.add_argument("--out", required=True)
    p.add_argument("--aegir-per-class", type=int, default=3)
    p.add_argument("--v5-per-stratum", type=int, default=1)
    p.add_argument("--v5-dir", default=str(V5_TEST))
    p.add_argument("--v5-max", type=int, default=0, help="at most this many v5 shapes in all (0: no limit)")
    p.add_argument("--classes", default="", help="only these aegir classes (comma-separated, e.g. n1_P,n3_S)")
    p.add_argument("--exclude-from", nargs="*", default=[], help="corpora whose shapes this one must not repeat")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("build")
    p.add_argument("corpus")
    p.add_argument("--retry-failed", action="store_true", help="build again the truths whose gate failed")
    p.add_argument("--timeout", type=int, default=300)
    for name in ("start", "status", "pull"):
        p = sub.add_parser(name)
        p.add_argument("corpus")
        p.add_argument("--server", default="server_b")
        if name == "start":
            p.add_argument("--cpuset", default="8-13")
            p.add_argument("--jobs", type=int, default=6)
            p.add_argument("--timeout", type=int, default=2700)
            p.add_argument("--budget", type=int, default=900)
            p.add_argument("--points", type=int, default=2_000_000)
            p.add_argument("--min-free", default="20G")
            p.add_argument("--max-load", default="15")
            p.add_argument("--threads", type=int, default=4, help="OpenMP/BLAS threads per part")
            p.add_argument("--code", default=REMOTE_CODE, help="the server directory of the code this run uses")
    for name in ("rate", "regrade", "report"):
        sub.add_parser(name).add_argument("corpus")
    p = sub.add_parser("review")
    p.add_argument("corpus")
    p.add_argument("stem")
    p = sub.add_parser("fork")
    p.add_argument("corpus")
    p.add_argument("--out", required=True)
    p.add_argument("--subset", type=int, default=0, help="at most this many shapes, round robin over the strata")
    p.add_argument("--strata-from", default="", help="JSON {stem: stratum} (shape_atlas.py select --strata-out): "
                                                     "only these stems, in these strata")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--rating-a", default="rating.json")
    p.add_argument("--rating-b", default="rating.json")
    p.add_argument("--out", default=None, help="write the comparison as JSON")
    a = ap.parse_args(argv)
    dict(select=cmd_select, build=cmd_build, start=cmd_start, status=cmd_status, rate=cmd_rate, regrade=cmd_regrade,
         pull=cmd_pull, report=cmd_report, review=cmd_review, fork=cmd_fork, compare=cmd_compare)[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
