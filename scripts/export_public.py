#!/usr/bin/env python3
"""
The public version of the framework, built from this repository's HEAD (the user, 2026-10-30: "publish the framework on
github" - prepare only, stay private; code + tests + user docs; GPL-3.0; the name SHaPe-ReCo). What goes in is
public/manifest.json: the included paths, the excluded globs (with their reasons), markdown sections dropped, the
substitutions (server names, user, paths, industrial part ids - applied to contents AND paths, in order) and the
forbidden patterns (regular expressions). The export fails while a forbidden pattern is left in a text file or a path.
The repository's README.md (written for the public) and public/LICENSE become the export's README.md and LICENSE.

    python3 scripts/export_public.py --out ~/Documents/SHaPe-ReCo-public [--test]

A target that is a git repository already (the public repository's clone, ~/Documents/SHaPe-ReCo-public with its
remote) keeps its history and gets one new commit with the export's changes; any other target becomes a new
repository with one commit and no remote. Nothing is pushed. --test runs the manifest's
verify_tests inside the export (the substitutions must not break the code).
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "public" / "manifest.json"


def _is_text(data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def drop_sections(text: str, headings) -> str:
    """Markdown without the sections that start with one of `headings` (up to the next heading of the same level)."""
    out, skip, level = [], False, 0
    for line in text.splitlines(keepends=True):
        m = re.match(r"(#+) ", line)
        if m and skip and len(m.group(1)) <= level:
            skip = False
        if any(line.startswith(h) for h in headings):
            skip, level = True, len(re.match(r"(#+)", line).group(1))
        if not skip:
            out.append(line)
    return "".join(out)


VERBATIM = (".stl", ".step", ".stp", ".obj", ".ply", ".npz", ".npy")   # geometry and arrays: copied byte for byte


def _key(a: str):
    """A substitution key as a pattern: a number only as a whole number (the first export turned the port's "22" in
    "1.254322" into "1.222" - four example STLs and tests/data/sdf_reference.json were changed)."""
    return re.compile(rf"(?<![\d.]){re.escape(a)}(?![\d])") if re.fullmatch(r"[\d.]+", a) else re.compile(re.escape(a))


def substitute(s: str, subs) -> str:
    for a, b in subs:
        s = _key(a).sub(lambda _m, b=b: b, s)
    return s


def selected(paths, include, exclude):
    for p in paths:
        if not any(p == i or p.startswith(i.rstrip("/") + "/") for i in include):
            continue
        if any(fnmatch.fnmatch(p, e) for e in exclude):
            continue
        yield p


def build(out: Path, man: dict) -> dict:
    files = subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-r", "--name-only", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.splitlines()                  # what HEAD holds, not the index
    keep = list(selected(files, man["include"], man["exclude"]))
    r = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD", "--", *keep], capture_output=True)
    if r.returncode != 0:
        sys.exit(f"git archive: {r.stderr.decode()[-400:]}")
    blob = r.stdout
    kept = out.parent / f".{out.name}.git-kept"         # an existing clone's history survives the rebuild
    if out.exists():
        if (out / ".git").is_dir():
            if kept.exists():
                shutil.rmtree(kept)
            (out / ".git").rename(kept)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if kept.exists():
        kept.rename(out / ".git")
    subs = man["substitute"]
    drops = man.get("drop_markdown_sections", {})
    n_text = n_bin = 0
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            data = tar.extractfile(m).read()
            rel = substitute(m.name, subs)
            dst = out / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if _is_text(data) and not m.name.lower().endswith(VERBATIM):
                text = data.decode("utf-8")
                if m.name in drops:
                    text = drop_sections(text, drops[m.name])
                dst.write_text(substitute(text, subs))
                n_text += 1
            else:
                dst.write_bytes(data)
                n_bin += 1
            dst.chmod(m.mode & 0o777 or 0o644)
    for name in keep:                                                # the verbatim files are what HEAD holds
        if name.lower().endswith(VERBATIM):
            src = subprocess.run(["git", "-C", str(ROOT), "show", f"HEAD:{name}"], capture_output=True, check=True).stdout
            if (out / substitute(name, subs)).read_bytes() != src:
                sys.exit(f"{name}: the export changed a file that must be copied byte for byte")
    shutil.copy(ROOT / "README.md", out / "README.md")
    shutil.copy(ROOT / "public" / "LICENSE", out / "LICENSE")
    return dict(files=len(keep), text=n_text, binary=n_bin)


def scrub_check(out: Path, forbidden) -> list:
    """[(path, pattern, line)] of every forbidden pattern left in a text file's contents or in a path."""
    pats = [(p, re.compile(p)) for p in forbidden]
    hits = []
    for f in sorted(out.rglob("*")):
        if ".git" in f.parts:
            continue
        rel = str(f.relative_to(out))
        for p, rx in pats:
            if rx.search(rel):
                hits.append((rel, p, "(path)"))
        if f.is_file():
            data = f.read_bytes()
            if not _is_text(data):                                   # binaries too: a render's PNG metadata or a
                text = data.decode("latin-1")                        # .blend carried a home path (2026-11-10)
                for p, rx in pats:
                    if rx.search(text):
                        hits.append((rel, p, "(binary)"))
                continue
            for i, line in enumerate(data.decode("utf-8").splitlines(), 1):
                for p, rx in pats:
                    if rx.search(line):
                        hits.append((rel, p, f"{i}: {line.strip()[:120]}"))
    return hits


def git_init(out: Path, commit: str, author: str = ""):
    """One commit (on top of the history when `out` is a clone already; none when nothing changed); `author` ("Name
    <email>", the manifest's commit_author) is its author and committer, so the local login and host of the machine
    that exports never reach the public history."""
    ident = []
    if author:
        name, email = author.rsplit(" <", 1)
        ident = ["-c", f"user.name={name}", "-c", f"user.email={email.rstrip('>')}"]

    def git(*a):
        subprocess.run(["git", *ident, "-C", str(out), *a], check=True, capture_output=True)
    if not (out / ".git").is_dir():
        git("init", "-q", "-b", "main")
    git("add", "-A")
    if subprocess.run(["git", "-C", str(out), "diff", "--cached", "--quiet"]).returncode == 0:
        print("no change against the previous export: no commit")
        return
    git("commit", "-q", "-m", f"SHaPe ReCo, exported from the working repository at {commit[:8]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(Path.home() / "Documents" / "SHaPe-ReCo-public"))
    ap.add_argument("--test", action="store_true", help="run the manifest's verify_tests inside the export")
    a = ap.parse_args(argv)
    man = json.loads(MANIFEST.read_text())
    out = Path(a.out).expanduser()
    if out.resolve() == ROOT.resolve() or ROOT.resolve() in out.resolve().parents:
        sys.exit("the export must lie outside the working repository")
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    facts = build(out, man)
    hits = scrub_check(out, man["forbidden"])
    print(f"exported {facts['files']} files ({facts['text']} text, {facts['binary']} binary) from {head[:8]} to {out}")
    if hits:
        by = {}
        for rel, p, where in hits:
            by.setdefault(p, []).append(f"{rel} {where}")
        for p, v in by.items():
            print(f"FORBIDDEN {p}: {len(v)} hit(s)")
            for x in v[:6]:
                print("   ", x)
        sys.exit(f"{len(hits)} forbidden hit(s) left: not committed")
    git_init(out, head, man.get("commit_author", ""))
    print("scrub check clean; committed in", out)
    if a.test:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *man["verify_tests"]], cwd=out)
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
