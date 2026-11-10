#!/usr/bin/env python3
"""
The step after the reconstruction (the user's request 2026-09-29: "add aegir as a next step, which loads and
benchmarks our result from the reconstruction"): the program is written as GDML, loaded by aegir - the SHiP
experiment's simulation framework (Phlex + Geant4; github.com/ShipSoft/aegir) - and timed in a straight-line scan, beside the
part's mesh where aegir has one.

    python3 scripts/aegir_benchmark.py <run directory | part:latest | operations.json> --part liner
            [--events 200000] [--rounds 3] [--threads 1] [--mesh liner_mesh] [--tree grouped-balanced|grouped-multi|as-is]
            [--out DIR] [--aegir DIR] [--dry-run]

What it does:
  1. the program as `<out>/geometry/sandbox/<stem>.gdml` (reco_field/gdml_export.py) in the frame aegir's gun expects:
     the beam along the world's z, the part's centre in the origin (PLACEMENTS), volume `PartVolume` of G4_Fe in a
     `World` of G4_AIR with 1000 mm around the part - as aegir's own converter writes them
     (scripts/sandbox_geometry/make_gdml.py there);
  2. aegir's scan workflow `workflows/gun_st_sandbox_scan_bench.jsonnet` (threads > 1: `gun_mt_...`): muon neutrinos
     as straight lines through the geometry, a crossing detector, no output - the cost of an event is the navigation
     through the solid. It is started in aegir's checkout through its pixi environment `bench`; NOTHING is written
     into that checkout: the geometry is found through the workflow's variable `root`, which points to `<out>`;
  3. `--mesh STEM`: the same scan on `geometry/sandbox/<STEM>.gdml` of the aegir checkout (the part as one tessellated
     solid), the rounds interleaved with the program's;
  4. `<out>/aegir_benchmark.json`: per geometry the seconds of every round, their mean and standard deviation, the
     events per second, the machine's load when each round began.

Round 0 is a warm-up and is not counted. A benchmark is a heavy job: it waits for the shared lock
(~/.cache/shape_reco/heavy.lock) and reports the load; timings taken beside other jobs are marked by that load.
"""

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

AEGIR = Path(os.environ.get("AEGIR_ROOT") or Path.home() / "Documents" / "Sandbox_aegir" / "aegir")
LOCK = Path.home() / ".cache" / "shape_reco" / "heavy.lock"


def pixi() -> str:
    """The pixi executable: on the PATH, or where its installer puts it (~/.pixi/bin)."""
    import shutil
    found = shutil.which("pixi") or str(Path.home() / ".pixi" / "bin" / "pixi")
    if not Path(found).is_file():
        raise FileNotFoundError("pixi not found (PATH, ~/.pixi/bin): aegir is run through its pixi environment")
    return found

# part -> the gun of aegir's workflows/sandbox.libsonnet, and the placement its geometries have there
# (the header comments of geometry/sandbox/*.gdml in the aegir checkout, read 2026-09-29)
PLACEMENTS = {
    "liner": dict(origin=(0.0, -96.0, 808.35), beam_axis="y"),
    "frame": dict(origin=(602.0, 6000.0, -3548.0), beam_axis="x"),
    "blind_hole": dict(origin=(0.0, 0.0, 0.0), beam_axis="z"),
}


def load_result(spec):
    """The ReconstructionResult of a stored run or of a file of operations (scripts/export_gdml.py's inputs)."""
    from reco_engine import run_store
    from reco_engine.result_schema import ReconstructionResult
    if str(spec).endswith(".json") and Path(spec).name != "run.json":
        data = json.loads(Path(spec).read_text())
        ops = data if isinstance(data, list) else data.get("operations") or data.get("program") or []
        ops = [o if "primitive" in o else dict(op=o["op"], accepted=o.get("accepted", True),
                                               primitive={k: v for k, v in o.items() if k not in ("op", "accepted")})
               for o in ops]
        name, path = Path(spec).stem, "part"
    else:
        run = run_store.load_run(run_store.resolve(spec))
        ops, name, path = run.get("program") or [], run.get("name") or "run", (run.get("input") or {}).get("path") or "part"

    cut = run_store.truncated(ops)
    if cut:
        raise ValueError(f"the run store cut {len(cut)} node(s) of this program down to text when it saved it (before "
                         f"2026-10-01; first: {cut[0]}): it cannot be built (scripts/recover_run_program.py)")

    def bare(p):                                          # the geometry alone (a recovered node has no sources)
        return dict(p, sources=[], alternatives=[], parts=[bare(q) for q in p.get("parts") or []])
    ops = [dict(o, primitive=bare(o["primitive"])) for o in ops]
    return ReconstructionResult.from_dict(dict(input_path=path, operations=ops)), name


def write_geometry(result, out: Path, stem: str, part: str, upto=None, tree: str = "grouped-balanced") -> dict:
    from reco_engine.result_to_program import program_model
    from reco_field.gdml_export import export_gdml
    model = program_model(result, upto=upto) if upto else program_model(result)
    if model is None:
        raise ValueError("the result has no accepted operation")
    place = PLACEMENTS[part]
    return export_gdml(model, out / "geometry" / "sandbox" / f"{stem}.gdml", name=stem, material="G4_Fe", world="G4_AIR",
                       margin_mm=1000.0, origin=place["origin"], beam_axis=place["beam_axis"], volume_name="PartVolume",
                       world_name="World", physvol_name="PartPhys", tree=tree)


def scan_command(aegir: Path, root: Path, part: str, stem: str, events: int, threads: int) -> str:
    wf = "gun_st_sandbox_scan_bench" if threads <= 1 else "gun_mt_sandbox_scan_bench"
    ext = (f"--ext-str root={shlex.quote(str(root))} --ext-str part={part} --ext-str geometry={stem} "
           f"--ext-str events={int(events)}" + (f" --ext-str concurrency={int(threads)}" if threads > 1 else ""))
    return f"phlex -c <(jsonnet {ext} {shlex.quote(str(aegir / 'workflows' / (wf + '.jsonnet')))}) -j {max(1, int(threads))}"


def run_scan(aegir: Path, command: str, log: Path, timeout: float) -> dict:
    """One run through aegir's pixi environment `bench`, in aegir's checkout (its plugins are found from there)."""
    load = open("/proc/loadavg").read().split()[:3]
    full = [pixi(), "run", "--manifest-path", str(aegir / "pixi.toml"), "-e", "bench", "bash", "-c", command]
    t0, c0 = time.perf_counter(), os.times()
    with open(log, "w") as f:
        try:
            rc = subprocess.run(full, cwd=str(aegir), stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    c1 = os.times()
    return dict(seconds=time.perf_counter() - t0, user=c1.children_user - c0.children_user,
                system=c1.children_system - c0.children_system, exit_code=rc, loadavg=[float(x) for x in load], log=str(log))


def summary(rounds: list, events: int) -> dict:
    ok = [r["seconds"] for r in rounds[1:] if r["exit_code"] == 0]              # round 0 is the warm-up
    if not ok:
        return dict(rounds_counted=0)
    mean = statistics.fmean(ok)
    return dict(rounds_counted=len(ok), mean_seconds=mean, stdev_seconds=statistics.stdev(ok) if len(ok) > 1 else None,
                events_per_second=events / mean if mean > 0 else None,
                max_load=max(r["loadavg"][0] for r in rounds[1:]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run")
    ap.add_argument("--part", choices=sorted(PLACEMENTS), required=True, help="the gun and the placement")
    ap.add_argument("--events", type=int, default=200000)
    ap.add_argument("--rounds", type=int, default=3, help="measured rounds (one warm-up round comes before them)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mesh", default=None, metavar="STEM", help="also scan geometry/sandbox/<STEM>.gdml of the aegir checkout")
    ap.add_argument("--upto", type=int, default=None, help="only the first N accepted operations (a level of detail)")
    ap.add_argument("--tree", choices=("grouped-balanced", "grouped-multi", "as-is"), default="grouped-balanced",
                    help="the shape of the boolean tree (reco_field/gdml_export.py; default grouped cuts, balanced unions)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--aegir", default=str(AEGIR))
    ap.add_argument("--timeout", type=float, default=3600.0, help="s per run")
    ap.add_argument("--dry-run", action="store_true", help="write the geometry and print the commands, run nothing")
    ap.add_argument("--no-lock", action="store_true", help="do not wait for the shared lock of heavy jobs")
    args = ap.parse_args(argv)

    aegir = Path(args.aegir).resolve()
    result, name = load_result(args.run)
    stem = "".join(c if c.isalnum() or c == "_" else "_" for c in f"{args.part}_csg_{name}")
    out = Path(args.out) if args.out else Path.home() / "shape_reco_results" / "aegir" / f"{stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    info = write_geometry(result, out, stem, args.part, args.upto, args.tree)
    print(f"geometry: {info['path']} ({info['primitives']} primitives, {info['booleans']} booleans)")
    variants = [(stem, out)] + ([(args.mesh, aegir)] if args.mesh else [])
    for s, root in variants:
        if not (Path(root) / "geometry" / "sandbox" / f"{s}.gdml").is_file():
            print(f"no geometry {root}/geometry/sandbox/{s}.gdml", file=sys.stderr)
            return 2
    commands = {s: scan_command(aegir, root, args.part, s, args.events, args.threads) for s, root in variants}
    for s, c in commands.items():
        print(f"{s}: {c}")
    if args.dry_run:
        return 0
    if not (aegir / "pixi.toml").is_file():
        print(f"no aegir checkout at {aegir}", file=sys.stderr)
        return 2

    lock = None
    if not args.no_lock:
        import fcntl
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        lock = open(LOCK, "w")
        print("waiting for the shared lock of heavy jobs ...", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    rounds = {s: [] for s in commands}
    for r in range(args.rounds + 1):                                     # the variants interleaved, round by round
        for s, c in commands.items():
            res = run_scan(aegir, c, out / f"{s}_r{r}.log", args.timeout)
            rounds[s].append(res)
            print(f"round {r}{' (warm-up)' if r == 0 else ''} {s}: {res['seconds']:.2f} s, exit {res['exit_code']}, "
                  f"load {res['loadavg'][0]:.1f}", flush=True)
    if lock is not None:
        lock.close()
    report = dict(part=args.part, run=str(args.run), events=args.events, threads=args.threads, aegir=str(aegir),
                  geometry=info, upto=args.upto,
                  variants={s: dict(command=commands[s], rounds=rounds[s], **summary(rounds[s], args.events)) for s in commands})
    (out / "aegir_benchmark.json").write_text(json.dumps(report, indent=1))
    for s, v in report["variants"].items():
        if v.get("rounds_counted"):
            sd = f" +- {v['stdev_seconds']:.2f}" if v.get("stdev_seconds") is not None else ""
            print(f"{s}: {v['mean_seconds']:.2f}{sd} s for {args.events} events = {v['events_per_second']:.0f} events/s "
                  f"({v['rounds_counted']} rounds, load up to {v['max_load']:.1f})")
        else:
            print(f"{s}: no round succeeded - see {out}/{s}_r*.log")
    print(f"written {out / 'aegir_benchmark.json'}")
    return 0 if all(v.get("rounds_counted") for v in report["variants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
