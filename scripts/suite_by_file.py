#!/usr/bin/env python3
"""
The whole test suite, one pytest process per test file, several files at a time, with everything kept for analysis.

    python3 scripts/suite_by_file.py --out <dir> [--jobs 6] [--threads 4] [--timeout 3600] [--only tests/test_a.py ...]
    python3 scripts/suite_by_file.py --out <dir> --report          # only re-read <dir> and print the analysis

One process per file: a crash or a hang in one file (a segfault in a native library, a Qt thread that does not end)
costs that file, not the run, and every file has its own wall time. Per file <dir>/<file>.log (pytest -v -rA
--durations=15, stdout and stderr), <file>.xml (junit) and <file>.json (return code, seconds, timed out). Then
<dir>/suite.json: every test with outcome, seconds and message, and the analysis that is printed - totals, the
failures and errors grouped by their exception, the skips grouped by their reason, the slowest tests and files, and
the tests slower than 20 s that tests/conftest.py does not mark `slow`.

Meant for workstation (scripts/workstation_exec.sh): not a timing run - files run side by side, so the seconds are upper
bounds. A failure seen here is confirmed by running its file alone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOW_SECONDS = 20.0


def run_file(path: Path, out: Path, threads: int, timeout: int) -> dict:
    name = path.stem
    log, xml = out / f"{name}.log", out / f"{name}.xml"
    env = dict(os.environ, SHAPE_RECO_TEST_THREADS=str(threads), PYTHONFAULTHANDLER="1")
    cmd = [sys.executable, "-X", "faulthandler", "-m", "pytest", str(path.relative_to(ROOT)), "-v", "-rA",
           "--durations=15", "--no-header", "-p", "no:cacheprovider", f"--junitxml={xml}", "--tb=short"]
    t0, timed_out = time.time(), False
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc, timed_out = -9, True
    info = dict(file=path.name, returncode=rc, seconds=time.time() - t0, timed_out=timed_out)
    (out / f"{name}.json").write_text(json.dumps(info))
    print(f"{path.name:<46} rc {rc:>3} {info['seconds']:>7.1f} s" + (" TIMEOUT" if timed_out else ""), flush=True)
    return info


def _first_line(text: str) -> str:
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    return lines[0][:200] if lines else ""


def _signature(message: str, text: str) -> str:
    """The exception that ended a test: `E   SomeError: ...` of the traceback, numbers and paths blurred, so that
    the same cause in twenty tests is one group."""
    found = re.findall(r"^E\s+([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)\b[^\n]*)", text or "", re.M)
    sig = found[-1] if found else _first_line(message) or _first_line(text)
    sig = re.sub(r"/[\w./\-]+", "<path>", sig)
    sig = re.sub(r"\b\d+(\.\d+)?(e[-+]?\d+)?\b", "<n>", sig)
    return sig[:170]


def collect(out: Path) -> dict:
    tests, files = [], []
    for info_file in sorted(out.glob("test_*.json")):
        info = json.loads(info_file.read_text())
        xml = out / (info_file.stem + ".xml")
        n = 0
        if xml.is_file():
            try:
                for case in ET.parse(xml).getroot().iter("testcase"):
                    n += 1
                    outcome, message, text = "passed", "", ""
                    for kind in ("failure", "error", "skipped"):
                        el = case.find(kind)
                        if el is not None:
                            outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[kind]
                            message, text = el.get("message") or "", el.text or ""
                            break
                    tests.append(dict(file=info["file"], name=case.get("name"), outcome=outcome,
                                      seconds=float(case.get("time") or 0.0), message=message[:400],
                                      signature=_signature(message, text) if outcome in ("failed", "error") else "",
                                      reason=_first_line(message)[:200] if outcome == "skipped" else ""))
            except ET.ParseError:
                pass
        info["tests"] = n
        if n == 0 and info["returncode"] not in (0, 5):       # no junit: the process died before it could write one
            log = (out / (info_file.stem + ".log")).read_text(errors="replace")
            tests.append(dict(file=info["file"], name="<the file itself>", outcome="error", seconds=info["seconds"],
                              message=log[-400:], signature="TIMEOUT" if info["timed_out"] else
                              _signature("", log) or f"process ended with return code {info['returncode']}", reason=""))
        files.append(info)
    return dict(files=files, tests=tests)


def slow_marked() -> tuple:
    src = (ROOT / "tests" / "conftest.py").read_text()
    files = set(re.findall(r'"(test_\w+\.py)",', src.split("SLOW_TESTS")[0]))
    tests = set(re.findall(r'"(test_\w+\.py::\w+)"', src.split("SLOW_TESTS", 1)[1].split("GPU_TESTS")[0]))
    return files, tests


def report(data: dict) -> str:
    tests, files = data["tests"], data["files"]
    c = Counter(t["outcome"] for t in tests)
    L = [f"{len(files)} files, {len(tests)} tests: {c['passed']} passed, {c['failed']} failed, {c['error']} errors, "
         f"{c['skipped']} skipped; {sum(f['seconds'] for f in files):.0f} s of file time, "
         f"{sum(f['timed_out'] for f in files)} files timed out", ""]
    bad = [t for t in tests if t["outcome"] in ("failed", "error")]
    groups = defaultdict(list)
    for t in bad:
        groups[t["signature"]].append(t)
    L.append(f"FAILURES AND ERRORS BY CAUSE ({len(bad)} tests, {len(groups)} causes)")
    for sig, ts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  [{len(ts)}] {sig}")
        per_file = Counter(t["file"] for t in ts)
        for f, n in per_file.most_common():
            names = [t["name"] for t in ts if t["file"] == f]
            L.append(f"        {f} ({n}): " + ", ".join(names[:4]) + (" ..." if len(names) > 4 else ""))
    L.append("")
    skipped = [t for t in tests if t["outcome"] == "skipped"]
    reasons = Counter(re.sub(r"^Skipped: ", "", t["reason"]) or "(no reason given)" for t in skipped)
    L.append(f"SKIPS BY REASON ({len(skipped)} tests)")
    for r, n in reasons.most_common():
        fs = sorted({t["file"] for t in skipped if (re.sub(r'^Skipped: ', '', t['reason']) or '(no reason given)') == r})
        L.append(f"  [{n}] {r[:150]}   ({', '.join(fs[:3])}{' ...' if len(fs) > 3 else ''})")
    L.append("")
    L.append("SLOWEST FILES")
    for f in sorted(files, key=lambda f: -f["seconds"])[:12]:
        L.append(f"  {f['seconds']:>7.1f} s  {f['file']} ({f['tests']} tests)" + ("  TIMEOUT" if f["timed_out"] else ""))
    L.append("")
    marked_files, marked_tests = slow_marked()
    slow = [t for t in tests if t["seconds"] >= SLOW_SECONDS]
    unmarked = [t for t in slow if t["file"] not in marked_files
                and f"{t['file']}::{t['name'].split('[')[0]}" not in marked_tests]
    L.append(f"TESTS OF {SLOW_SECONDS:.0f} s OR MORE: {len(slow)}, of which NOT marked slow in conftest.py: {len(unmarked)}")
    for t in sorted(unmarked, key=lambda t: -t["seconds"])[:25]:
        L.append(f"  {t['seconds']:>7.1f} s  {t['file']}::{t['name']}  ({t['outcome']})")
    took = Counter()
    for t in tests:
        took[t["file"]] += t["seconds"]
    stale = sorted(f for f in marked_files if f in took and took[f] < SLOW_SECONDS)
    if stale:
        L.append("")
        L.append("FILES MARKED slow THAT TOOK LESS THAN 20 s HERE: " + ", ".join(f"{f} ({took[f]:.0f} s)" for f in stale))
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per file")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.report:
        paths = [ROOT / p for p in args.only] if args.only else sorted((ROOT / "tests").glob("test_*.py"))
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(lambda p: run_file(p, out, args.threads, args.timeout), paths))
        print(f"\nwall time {time.time() - t0:.0f} s with {args.jobs} files at a time, {args.threads} threads each\n")
    data = collect(out)
    text = report(data)
    (out / "suite.json").write_text(json.dumps(data, indent=1))
    (out / "suite.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
