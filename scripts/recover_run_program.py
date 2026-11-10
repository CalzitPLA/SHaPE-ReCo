#!/usr/bin/env python3
"""
The program of a stored run whose composite tools the run store cut down to text when it saved them (before
2026-10-01, reco_engine/run_store.py `plain`: the limit of 12 levels cut every operand below the fifth level of a
composite into the first 200 characters of its text), recovered from the feature phase's own record of the same run:
`features_job.json` in `~/.cache/shape_reco/features/<part>_<start>/features/`, which holds every applied feature's
primitive as its full tree (written for features.FCStd by reco_field/exact_check.write_features_document).

    python3 scripts/recover_run_program.py <run directory> [...] [--job <features_job.json>] [--write]

For each run: the job is the newest one of the same input, started before the run was saved, whose applied features are
the run's feature operations (kind, label, placement of the top primitive); every node the run store kept is compared
with the job's (kind, combine, label, placement, dimensions, u, v: exactly), every node cut to text with the job's by
what its text still shows (kind, quaternion, centre, the dimensions written before the cut). A cut node shorter than 200
characters is its whole text and is read back from it. Nothing is written unless every check passes; --write writes
`program_recovered.json` beside run.json (run.json is never touched), which run_store.load_run then reads in place of
the cut program. Nodes taken from the job carry the geometry alone (no sources, confidence, residual, grade).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine import run_store  # noqa: E402

JOBS = Path.home() / ".cache" / "shape_reco" / "features"
GEOMETRY = ("kind", "combine", "label", "quaternion", "center_mm", "dims_mm", "u", "v")
SCHEMA_ORDER = ("kind", "quaternion", "center_mm", "dims_mm", "u", "v", "confidence", "residual", "grade", "sources",
                "alternatives", "label", "parts", "combine")
CUT = 200                                         # run_store.plain wrote str(obj)[:200]


def _same(a, b) -> bool:
    """Equal as JSON data (a tuple's text against a list, ints against floats of the same value)."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


class Mismatch(Exception):
    pass


def _numbers(text: str) -> str:
    """numpy scalars in a repr (np.float64(x)) as their numbers."""
    return re.sub(r"np\.\w+\(([^()]*)\)", r"\1", text)


def _literal(text: str):
    """A Python repr (of plain's input: tuples, dicts, numpy scalars as np.float64(x)) as data."""
    try:
        return ast.literal_eval(_numbers(text))
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
        raise Mismatch(f"the text cannot be read back ({type(exc).__name__}): {text[:80]!r}") from None


def _prefix_fields(text: str) -> dict:
    """The fields a cut repr still shows completely: the kind and the tuples/dicts closed before the cut."""
    out = {}
    text = _numbers(text)
    m = re.match(r"\{'kind': '([^']*)'", text)
    if m:
        out["kind"] = m.group(1)
    for key in ("quaternion", "center_mm", "u", "v"):
        m = re.search(rf"'{key}': (\([^()]*\)|None)", text)
        if m:
            out[key] = _literal(m.group(1))
    m = re.search(r"'dims_mm': (\{[^{}]*\})", text)
    if m:
        out["dims_mm"] = _literal(m.group(1))
    return out


def _schema_node(job_prim: dict) -> dict:
    """A job primitive (geometry and, for a leaf, its program) as a node of the run store's program."""
    node = dict(kind=job_prim["kind"], quaternion=job_prim["quaternion"], center_mm=job_prim["center_mm"],
                dims_mm=job_prim.get("dims_mm") or {}, u=job_prim.get("u"), v=job_prim.get("v"),
                label=job_prim.get("label"),
                parts=[_schema_node(q) for q in job_prim.get("parts") or []], combine=job_prim.get("combine"))
    return {k: node[k] for k in SCHEMA_ORDER if k in node}


def _top_key(prim: dict) -> tuple:
    return tuple(json.dumps(prim.get(k), sort_keys=True) for k in ("kind", "label", "combine", "center_mm", "quaternion"))


def graft(stored, job: dict, path: str, counts: dict):
    """The stored node with every node cut to text replaced: read back from its text when whole, else the job's node
    after the check of what the text shows. Raises Mismatch where the two disagree."""
    if isinstance(stored, str):
        if len(stored) < CUT:
            node = _literal(stored)
            for k in GEOMETRY:
                if k in job and k in node and not _same(node[k], job[k]):
                    raise Mismatch(f"{path}.{k}: text {node[k]!r} against the job's {job[k]!r}")
            counts["read_back"] += 1
            return run_store.plain(node, limit=None)
        shown = _prefix_fields(stored)
        if "kind" not in shown or "center_mm" not in shown or "quaternion" not in shown:
            raise Mismatch(f"{path}: the text shows too little to be checked: {stored[:80]!r}")
        for k, v in shown.items():
            if k == "dims_mm":
                if not _same(v, job.get("dims_mm") or {}):
                    raise Mismatch(f"{path}.dims_mm: text {v} against the job's {job.get('dims_mm')}")
            elif not _same(v, job.get(k)):
                raise Mismatch(f"{path}.{k}: text {v!r} against the job's {job.get(k)!r}")
        counts["from_job"] += 1
        counts["checked_fields"] += len(shown)
        return _schema_node(job)
    if not isinstance(stored, dict):
        raise Mismatch(f"{path}: {type(stored).__name__} where a primitive is expected")
    for k in GEOMETRY:
        a, b = stored.get(k), job.get(k)
        if isinstance(a, str) and k not in ("kind", "combine", "label"):
            raise Mismatch(f"{path}.{k}: cut to text ({a[:60]!r}) - not handled")
        if k == "dims_mm":
            a, b = a or {}, b or {}
        if not _same(a, b):                                  # a job leaf has no combine: None on both sides
            raise Mismatch(f"{path}.{k}: stored {a!r} against the job's {b!r}")
    counts["kept"] += 1
    out = dict(stored)
    sp, jp = stored.get("parts") or [], job.get("parts") or []
    if isinstance(sp, str) or len(sp) != len(jp):
        raise Mismatch(f"{path}.parts: {len(sp) if isinstance(sp, list) else 'text'} operands against the job's {len(jp)}")
    out["parts"] = [graft(s, j, f"{path}.parts[{i}]", counts) for i, (s, j) in enumerate(zip(sp, jp))]
    if isinstance(stored.get("sources"), list):
        srcs = []
        for i, s in enumerate(stored["sources"]):
            if isinstance(s, str):
                if len(s) >= CUT:
                    raise Mismatch(f"{path}.sources[{i}]: cut at {CUT} characters")
                s = run_store.plain(_literal(s), limit=None)
                counts["read_back"] += 1
            srcs.append(s)
        out["sources"] = srcs
    if isinstance(stored.get("alternatives"), list) and any(isinstance(a, str) for a in stored["alternatives"]):
        out["alternatives"] = [a for a in stored["alternatives"] if not isinstance(a, str)]
        counts["alternatives_dropped"] += sum(isinstance(a, str) for a in stored["alternatives"])
    return out


def candidate_jobs(run: dict):
    """features_job.json files of the same input, started before the run was saved, newest first."""
    saved = float(run.get("time") or 0.0)
    path = (run.get("input") or {}).get("path")
    out = []
    for job in JOBS.glob("*/features/features_job.json"):
        try:
            started = time.mktime(time.strptime(job.parent.parent.name.rsplit("_", 1)[1], "%Y%m%d-%H%M%S"))
        except (ValueError, IndexError):
            continue
        if started <= saved:
            out.append((started, job))
    for started, job in sorted(out, reverse=True):
        data = json.loads(job.read_text())
        if data.get("input") == path:
            yield job, data


def recover(run_dir: Path, job_path=None) -> dict:
    run = json.loads((run_dir / run_store.RUN_JSON).read_text())
    program = run.get("program") or []
    cut = run_store.truncated(program)
    if not cut:
        return dict(run=str(run_dir), status="whole", cut=0)
    jobs = [(Path(job_path), json.loads(Path(job_path).read_text()))] if job_path else candidate_jobs(run)
    tried = []
    for job, data in jobs:
        features = data.get("features") or []
        first = len(program) - len(features)
        keys = [_top_key(o["primitive"]) for o in program[max(first, 0):]]
        if first < 0 or sorted(keys) != sorted(_top_key(f["primitive"]) for f in features):
            tried.append(f"{job.parent.parent.name}: {len(features)} features, not the run's")
            continue
        by_key = {}                                       # each operation's feature by its top primitive
        for f in features:
            by_key.setdefault(_top_key(f["primitive"]), []).append(f)
        features = [by_key[k].pop(0) for k in keys]
        early = [c for c in cut if int(c[1:c.index("]")]) < first and ".sources" not in c]
        if early:
            tried.append(f"{job.parent.parent.name}: cut nodes before the features ({early[:2]})")
            continue
        counts = dict(kept=0, from_job=0, read_back=0, checked_fields=0, alternatives_dropped=0)
        try:
            recovered = []
            for i, o in enumerate(program):
                if i < first:
                    prim = o["primitive"]
                    if run_store.truncated([o]):              # only sources of the rough model can be cut here
                        prim = graft_sources_only(prim, counts)
                    recovered.append(dict(o, primitive=prim))
                else:
                    f = features[i - first]
                    recovered.append(dict(o, primitive=graft(o["primitive"], f["primitive"], f"[{i}].primitive", counts)))
        except Mismatch as exc:
            tried.append(f"{job.parent.parent.name}: {exc}")
            continue
        left = run_store.truncated(recovered)
        if left:
            tried.append(f"{job.parent.parent.name}: still cut after recovery: {left[:3]}")
            continue
        model_step = Path(str(data.get("final") or ""))
        return dict(run=str(run_dir), status="recovered", cut=len(cut), job=str(job),
                    job_sha256=hashlib.sha256(job.read_bytes()).hexdigest(), first_feature=first,
                    features=len(features), counts=counts, final_model_step=str(model_step) if model_step.is_file() else None,
                    program=recovered)
    return dict(run=str(run_dir), status="not recovered", cut=len(cut), tried=tried)


def graft_sources_only(prim: dict, counts: dict) -> dict:
    """A rough-model primitive whose only cut nodes are whole source texts (shorter than the cut)."""
    def walk(p, path):
        if isinstance(p, str):
            raise Mismatch(f"{path}: a rough-model primitive cut to text")
        out = dict(p)
        if isinstance(p.get("sources"), list):
            srcs = []
            for s in p["sources"]:
                if isinstance(s, str):
                    if len(s) >= CUT:
                        raise Mismatch(f"{path}.sources: cut at {CUT} characters")
                    s = run_store.plain(_literal(s), limit=None)
                    counts["read_back"] += 1
                srcs.append(s)
            out["sources"] = srcs
        out["parts"] = [walk(q, f"{path}.parts[{i}]") for i, q in enumerate(p.get("parts") or [])]
        return out
    return walk(prim, "rough")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories of the store")
    ap.add_argument("--job", default=None, help="the features_job.json to use (default: searched in the cache)")
    ap.add_argument("--write", action="store_true", help=f"write {run_store.RECOVERED_PROGRAM} beside run.json")
    args = ap.parse_args(argv)
    bad = 0
    for spec in args.runs:
        run_dir = run_store.resolve(spec)
        r = recover(Path(run_dir), args.job)
        head = f"{run_dir}: {r['status']}, {r['cut']} node(s) cut to text"
        if r["status"] != "recovered":
            print(head + "".join(f"\n    {t}" for t in r.get("tried", [])))
            bad += r["status"] != "whole"
            continue
        print(f"{head}; from {r['job']} ({r['features']} features from operation {r['first_feature']}): {r['counts']}")
        if args.write:
            out = Path(run_dir) / run_store.RECOVERED_PROGRAM
            if out.exists():
                print(f"    {out} exists: left alone")
                continue
            meta = {k: v for k, v in r.items() if k != "program"}
            meta.update(written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), by="scripts/recover_run_program.py",
                        note="nodes taken from the feature job carry the geometry alone (no sources, confidence, residual, grade)")
            out.write_text(json.dumps(dict(recovery=meta, program=r["program"]), indent=1, allow_nan=False))
            print(f"    wrote {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
