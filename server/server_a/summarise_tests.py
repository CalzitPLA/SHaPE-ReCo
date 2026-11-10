#!/usr/bin/env python3
"""The table of a test run of server/server_a/run_tests.sh: per file tests, passed, failed, errors, skipped, seconds, from
the JUnit files and GNU parallel's job log; the failures and the reasons of the skips below it.

    python3 server/server_a/summarise_tests.py <out directory>        ->  <out>/table.md, <out>/summary.json
"""
import csv
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])
files = [Path(l.strip()).stem for l in (out / "files.txt").read_text().splitlines() if l.strip()]
jobs = {}
if (out / "joblog.tsv").is_file():
    for row in csv.DictReader(open(out / "joblog.tsv"), delimiter="\t"):
        name = Path(row["Command"].split()[-1]).stem
        jobs[name] = dict(seconds=float(row["JobRuntime"]), exit=int(row["Exitval"]), signal=int(row["Signal"]))
rows, failures, skips = [], [], Counter()
total = Counter()
for n in files:
    r = dict(file=n, tests=0, passed=0, failed=0, errors=0, skipped=0, seconds=None, state="not run")
    x = out / "xml" / f"{n}.xml"
    if n in jobs:
        r["seconds"] = jobs[n]["seconds"]
        r["state"] = "ok" if jobs[n]["exit"] == 0 else ("timeout or killed" if jobs[n]["exit"] < 0 or jobs[n]["signal"] else "failed")
    if x.is_file():
        try:
            root = ET.parse(x).getroot()
            for case in root.iter("testcase"):
                r["tests"] += 1
                kinds = {c.tag for c in case}
                if "failure" in kinds or "error" in kinds:
                    k = "failed" if "failure" in kinds else "errors"
                    r[k] += 1
                    e = case.find("failure") if "failure" in kinds else case.find("error")
                    failures.append(dict(file=n, test=case.get("name"), kind=k, message=(e.get("message") or "")[:300]))
                elif "skipped" in kinds:
                    r["skipped"] += 1
                    skips[(case.find("skipped").get("message") or "")[:160]] += 1
                else:
                    r["passed"] += 1
        except ET.ParseError:
            r["state"] = "no result (the process died)"
    elif n in jobs:
        r["state"] = "no result (the process died)" if jobs[n]["exit"] else r["state"]
    rows.append(r)
    for k in ("tests", "passed", "failed", "errors", "skipped"):
        total[k] += r[k]
lines = [f"# Test run {out.name}", "", (out / "machine.txt").read_text().strip() if (out / "machine.txt").is_file() else "",
         f"commit {(out / 'commit.txt').read_text().strip() if (out / 'commit.txt').is_file() else 'unknown'}", "",
         f"**{total['tests']} tests in {len(files)} files: {total['passed']} passed, {total['failed']} failed, "
         f"{total['errors']} errors, {total['skipped']} skipped; {sum(r['state'] == 'not run' for r in rows)} files not run, "
         f"{sum(r['state'].startswith('no result') or r['state'].startswith('timeout') for r in rows)} died or timed out.**", "",
         "| file | tests | passed | failed | errors | skipped | seconds | state |", "|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["state"] == "ok" and not r["failed"] and not r["errors"], r["file"])):
    sec = "" if r["seconds"] is None else f"{r['seconds']:.0f}"
    lines.append(f"| {r['file']} | {r['tests']} | {r['passed']} | {r['failed']} | {r['errors']} | {r['skipped']} | {sec} | {r['state']} |")
if failures:
    lines += ["", "## Failures and errors", "", "| file | test | kind | message |", "|---|---|---|---|"]
    lines += [f"| {f['file']} | {f['test']} | {f['kind']} | {f['message'].replace('|', '/').replace(chr(10), ' ')} |" for f in failures]
if skips:
    lines += ["", "## Skips by reason", "", "| count | reason |", "|---|---|"]
    lines += [f"| {c} | {m.replace('|', '/').replace(chr(10), ' ')} |" for m, c in skips.most_common()]
(out / "table.md").write_text("\n".join(lines) + "\n")
(out / "summary.json").write_text(json.dumps(dict(total=dict(total), files=rows, failures=failures,
                                                  skips=[dict(reason=m, count=c) for m, c in skips.most_common()]), indent=1))
print(lines[5])
