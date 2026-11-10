#!/usr/bin/env python3
"""The lists of the ABC runs from a manifest (scripts/abc_manifest.py): every part with status `ok`, split into

  pilot.txt   N parts of the known subset, stratified by their number of faces (equal steps through the sorted list)
  known.txt   the known subset (the laptop's 500 of 2026-09-28, seed 0: `--subset subset.txt`) without the pilot
  rest.txt    every other part of the chunk

    python3 scripts/abc_lists.py <manifest.csv> --subset <subset.txt> --out <directory> [--pilot 20]

One stem per line. A manifest of the known subset alone gives an empty rest.txt."""
import argparse
import csv
from pathlib import Path


def lists(rows, known_stems, pilot: int):
    ok = [r for r in rows if r["status"] == "ok"]
    known = sorted((r for r in ok if r["stem"] in known_stems), key=lambda r: (int(r["faces"] or 0), r["stem"]))
    step = max(1, len(known) // max(1, pilot))
    first = known[step // 2::step][:pilot]
    taken = {r["stem"] for r in first}
    return ([r["stem"] for r in first], [r["stem"] for r in known if r["stem"] not in taken],
            sorted(r["stem"] for r in ok if r["stem"] not in known_stems))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("manifest")
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", type=int, default=20)
    args = ap.parse_args(argv)
    rows = list(csv.DictReader(open(args.manifest)))
    known = {Path(line.strip()).stem for line in open(args.subset) if line.strip()}
    pilot, rest_known, rest = lists(rows, known, args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, stems in (("pilot", pilot), ("known", rest_known), ("rest", rest)):
        (args.out / f"{name}.txt").write_text("".join(s + "\n" for s in stems))
        print(f"{name}: {len(stems)}")


if __name__ == "__main__":
    main()
