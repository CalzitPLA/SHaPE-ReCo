#!/usr/bin/env python3
"""
Compute on a server, look on the laptop (the user, 2026-10-29: "create a link which we could use to compute on the server
but preview live on the computer"). The reconstruction runs on an institute server (scripts/run_part.py in a checkout
on /data/user, niced, the batch environment of server/server_a/run_parts.sh); its path.jsonl - one line per step,
flushed while it runs (reco_engine/run_path.py) - is mirrored to the laptop by `ssh <host> tail -F`, and the window
follows the mirror live (`python3 -m reco_gui --follow <mirror>`: the steps in the log, the model after every operation).

    python3 scripts/remote_run.py start <step> [--host server_a] [--truth t.json] [--extra "--mode tree"] [--name N] [--gui]
    python3 scripts/remote_run.py status [<mirror>]       # the runs started from here, or one: host, pid, steps, state
    python3 scripts/remote_run.py fetch <mirror>          # the finished run's files (record, program, rating, model.stl)
    python3 scripts/remote_run.py stop <mirror>           # ends the server run (by its pid) and the mirror

Mirror: ~/shape_reco_results/remote/<id>/ - link.json (host, remote dir, pid, code, started), input.step (the local
copy the window shows), path.jsonl (growing), run/ after fetch. Remote: ~/data/shape_reco/remote/<id>/ (NFS,
the same on every server). Shared machines: the run is niced; check load and users before (`status` prints the load).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIRRORS = Path.home() / "shape_reco_results" / "remote"
SERVER_ROOT = "/data/user/shape_reco"
REMOTE_BASE = "data/shape_reco/remote"                 # below the home directory on every server (NFS)
DEFAULT_CODE = f"{SERVER_ROOT}/wt/protocol_e6aa19ed"


def ssh(host: str, command: str, timeout: int = 60, check: bool = True) -> str:
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh {host}: {r.stderr.strip()[-400:] or r.returncode}")
    return r.stdout


def remote_command(rdir: str, code: str, extra: str, truth: bool, threads: int = 4, nice: int = 10) -> str:
    """The shell command that starts run_part on the server in the background and prints its pid. `nice`: 10 for a batch
    job, 0 for a run the user waits for in the window (it then comes before the niced batch jobs)."""
    env = (f"PATH={SERVER_ROOT}/bin:{SERVER_ROOT}/venv_tests/bin:$PATH SHAPE_RECO_FREECADCMD={SERVER_ROOT}/bin/freecadcmd "
           f"TMPDIR={rdir}/tmp XDG_CACHE_HOME={rdir}/cache MPLCONFIGDIR={rdir}/cache/matplotlib QT_QPA_PLATFORM=offscreen "
           f"MPLBACKEND=Agg SHAPE_RECO_WALL_CACHE={rdir}/cache/wall_thickness SHAPE_RECO_BREP_CACHE={rdir}/cache/brep "
           f"SHAPE_RECO_CACHE={rdir}/cache/shape_reco XDG_CONFIG_HOME={rdir}/cache/xdg_config XDG_DATA_HOME={rdir}/cache/xdg_data "
           f"OMP_NUM_THREADS={threads} MKL_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} NUMEXPR_NUM_THREADS={threads} "
           f"CUDA_VISIBLE_DEVICES=")
    args = (f"scripts/run_part.py {rdir}/input.step --stl {rdir}/input.stl --out {rdir}/run --feature-time-budget 900 "
            f"--points 500000 --reconstruction-timeout 1800 --compare-timeout 600"
            + (f" --truth {rdir}/truth.json" if truth else "") + (f" {extra}" if extra else ""))
    return (f"mkdir -p {rdir}/run {rdir}/tmp {rdir}/cache && cd {shlex.quote(code)} && "
            f"(env {env} setsid nohup nice -n {int(nice)} python {args} > {rdir}/run.log 2>&1 < /dev/null & echo $! > {rdir}/pid) && "
            f"sleep 1 && cat {rdir}/pid")


def start_mirror(link: dict, mirror: Path) -> int:
    """The local follower: ssh tail -F of the server's path.jsonl into the mirror, until the run's process ends."""
    rdir, pid = link["remote_dir"], link["pid"]
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", link["host"],
           f"until [ -f {rdir}/run/path.jsonl ] || ! kill -0 {pid} 2>/dev/null; do sleep 1; done; "
           f"tail --pid={pid} -n +1 -F {rdir}/run/path.jsonl 2>/dev/null"]
    out = open(mirror / "path.jsonl", "ab")
    p = subprocess.Popen(cmd, stdout=out, stderr=open(mirror / "mirror.err", "ab"), stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return p.pid


def local_head():
    """(full commit, short commit, dirty) of the laptop's checkout; dirty: tracked files with uncommitted changes, which
    a deployed checkout does not have."""
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()
    full = git("rev-parse", "HEAD")
    return full, full[:8], bool(git("status", "--porcelain", "--untracked-files=no"))


def deploy(host: str, full: str, short: str) -> str:
    """The laptop's HEAD as a checkout on /data (shared by every server): the code without docs/ and the example meshes,
    plus docs/data (path weights, calibrations), a COMMIT file, the shared checkpoints and the gitignored models
    (models/*.joblib: the descriptor type model - missing from every checkout from main_20261015 to gui_d8367ee6, whose
    type stage therefore fused the network and the analytic fits only; found 2026-10-30)."""
    d = f"{SERVER_ROOT}/wt/gui_{short}"
    code = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD", "--", ".", ":!docs",
                           ":!dataset_gen/examples"], capture_output=True, check=True).stdout
    data = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", "HEAD", "--", "docs/data"],
                          capture_output=True, check=True).stdout
    for i, blob in enumerate((code, data)):
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                            (f"rm -rf {d}.tmp && mkdir -p {d}.tmp && " if i == 0 else "") + f"tar -x -C {d}.tmp"],
                           input=blob, capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"deploy to {host}: {r.stderr.decode()[-300:]}")
    ssh(host, f"ln -sfn {SERVER_ROOT}/wt/main_20261010/checkpoints {d}.tmp/checkpoints && "
              f"for m in {SERVER_ROOT}/wt/main_20261010/models/*.joblib; do ln -sfn $m {d}.tmp/models/; done && "
              f"test -e {d}.tmp/models/type_descriptors_v5_medium.joblib && echo {full} > {d}.tmp/COMMIT && "
              f"rm -rf {d} && mv {d}.tmp {d}", timeout=120)
    return d


def ensure_code(host: str, say=print) -> str:
    """The checkout of the laptop's HEAD on the servers (deployed once per commit); says when uncommitted changes stay
    behind."""
    full, short, dirty = local_head()
    d = f"{SERVER_ROOT}/wt/gui_{short}"
    if dirty:
        say(f"note: the server runs commit {short}; uncommitted changes of this checkout are not on the server")
    if ssh(host, f"test -f {d}/COMMIT && echo yes || echo no", check=False).strip() != "yes":
        say(f"deploying commit {short} to {d} ...")
        deploy(host, full, short)
    return d


HOSTS = ("server_a", "server_b", "server_c", "server_d")        # server_e has 15 GB: not for an interactive run
# measured speed per run against server_a (RUNNING 3.4: server_c and server_d 3-4x slower in every stage, 2026-10-16)
SPEED = {"server_a": 1.0, "server_b": 1.0, "server_c": 3.0, "server_d": 3.0}


def host_loads(hosts=HOSTS, timeout: int = 15) -> dict:
    """{host: (load per core, load, cores)} of the servers that answer (in parallel, one ssh each)."""
    import concurrent.futures as cf

    def one(h):
        try:
            out = ssh(h, "cat /proc/loadavg; nproc", timeout=timeout, check=False).split()
            load, cores = float(out[0]), int(out[-1])
            return h, (load / cores, load, cores)
        except Exception:
            return h, None
    with cf.ThreadPoolExecutor(len(hosts)) as ex:
        return {h: v for h, v in ex.map(one, hosts) if v is not None}


def pick_host(hosts=HOSTS, say=print) -> str:
    """The server where a run is expected to finish first: its measured speed factor (SPEED) times max(1, load per core) -
    a run started in the window comes first on a loaded machine (nice 0 against the batch's 10), so the load counts only
    beyond one per core. On a tie (within 5 %) the first in HOSTS."""
    loads = host_loads(hosts)
    if not loads:
        raise RuntimeError("no server answers")
    score = {h: SPEED.get(h, 2.0) * max(1.0, v[0]) for h, v in loads.items()}
    low = min(score.values())
    best = next(h for h in hosts if h in score and score[h] <= low * 1.05)
    say("loads: " + ", ".join(f"{h} {v[1]:.0f}/{v[2]} (x{score[h]:.2f})" for h, v in loads.items()) + f" -> {best}")
    return best


def start_run(step, host: str = "server_a", code: str | None = None, truth=None, extra: str = "", name: str | None = None,
              first_guess=None, stl=None, say=print, nice: int = 10) -> Path:
    """Start run_part on `host` for the local STEP `step` and mirror its path.jsonl; returns the mirror directory.
    `code`: a checkout on /data (None: the laptop's HEAD, deployed when needed); `first_guess`: a first_guess.json (the
    window's starting part); `truth`: the part's known tree (the run is rated on the server)."""
    if host == "auto":
        host = pick_host(say=say)
    step = Path(step).expanduser().resolve()
    stl = Path(stl).expanduser().resolve() if stl else step.with_suffix(".stl")
    if not stl.is_file():
        cand = step.parent.parent / "stl" / f"{step.stem}.stl"
        stl = cand if cand.is_file() else None
    code = code or ensure_code(host, say)
    run_id = f"{name or step.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    home = ssh(host, "echo $HOME").strip()
    rdir_abs = f"{home}/{REMOTE_BASE}/{run_id}"
    load = ssh(host, "cat /proc/loadavg; nproc", check=False).split()
    say(f"{host}: load {load[0] if load else '?'} on {load[-1] if load else '?'} cores")
    ssh(host, f"mkdir -p {rdir_abs}")
    files = [(step, "input.step")] + ([(stl, "input.stl")] if stl else []) + \
            ([(Path(truth).expanduser(), "truth.json")] if truth else []) + \
            ([(Path(first_guess).expanduser(), "first_guess.json")] if first_guess else [])
    for src, fname in files:
        subprocess.run(["scp", "-q", str(src), f"{host}:{rdir_abs}/{fname}"], check=True, timeout=300)
    extra = (extra + (f" --first-guess {rdir_abs}/first_guess.json" if first_guess else "")).strip()
    pid = int(ssh(host, remote_command(rdir_abs, code, extra, bool(truth), nice=nice), timeout=120).strip().split()[-1])
    mirror = MIRRORS / run_id
    mirror.mkdir(parents=True, exist_ok=True)
    shutil.copy(step, mirror / "input.step")
    link = dict(id=run_id, host=host, remote_dir=rdir_abs, pid=pid, code=code, extra=extra, step=str(step),
                truth=str(truth) if truth else None, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    link["mirror_pid"] = start_mirror(link, mirror)
    (mirror / "link.json").write_text(json.dumps(link, indent=1))
    say(f"started on {host} (pid {pid}) in {rdir_abs}; mirror {mirror}")
    return mirror


def cmd_start(a):
    mirror = start_run(a.step, a.host, a.code, a.truth, a.extra, a.name, a.first_guess, a.stl)
    print(f"follow: python3 -m reco_gui --follow {mirror}")
    if a.gui:
        subprocess.Popen([sys.executable, "-m", "reco_gui", "--follow", str(mirror)], cwd=str(ROOT), start_new_session=True,
                         stdout=open(mirror / "gui.log", "ab"), stderr=subprocess.STDOUT)
    return mirror


def read_link(mirror) -> dict:
    return json.loads((Path(mirror).expanduser() / "link.json").read_text())


def run_state(link: dict) -> dict:
    out = ssh(link["host"], f"kill -0 {link['pid']} 2>/dev/null && echo running || echo ended; "
                            f"cat {link['remote_dir']}/run/record.json 2>/dev/null | head -c 20000", check=False)
    lines = out.split("\n", 1)
    rec = {}
    if len(lines) > 1 and lines[1].strip():
        try:
            rec = json.loads(lines[1])
        except ValueError:
            rec = {}
    return dict(process=lines[0].strip(), state=rec.get("state"), stage=rec.get("stage"))


def cmd_status(a):
    mirrors = [Path(a.mirror).expanduser()] if a.mirror else sorted(MIRRORS.glob("*/link.json"))
    for m in mirrors:
        m = m.parent if m.name == "link.json" else m
        link = read_link(m)
        n = sum(1 for line in open(m / "path.jsonl") if '"step"' in line) if (m / "path.jsonl").is_file() else 0
        st = run_state(link)
        print(f"{link['id']}: {link['host']} pid {link['pid']}, {st['process']}, record {st['state']}/{st['stage']}, "
              f"{n} steps mirrored")


def cmd_fetch(a):
    m = Path(a.mirror).expanduser()
    link = read_link(m)
    (m / "run").mkdir(exist_ok=True)
    subprocess.run(["rsync", "-a", "--exclude", "loop/", "--exclude", "tmp/", "--exclude", "cache/",
                    f"{link['host']}:{link['remote_dir']}/run/", str(m / "run") + "/"], check=True, timeout=900)
    subprocess.run(["rsync", "-a", f"{link['host']}:{link['remote_dir']}/run.log", str(m) + "/"], check=False, timeout=120)
    print(f"fetched -> {m / 'run'}: {', '.join(sorted(p.name for p in (m / 'run').iterdir()))}")


def cmd_stop(a):
    m = Path(a.mirror).expanduser()
    link = read_link(m)
    ssh(link["host"], f"kill {link['pid']} 2>/dev/null; true", check=False)        # by its pid, never by a pattern
    try:
        os.killpg(link["mirror_pid"], 15)
    except (ProcessLookupError, KeyError, PermissionError):
        pass
    print(f"stopped {link['id']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("step")
    s.add_argument("--stl")
    s.add_argument("--host", default="server_a", choices=("auto", "server_a", "server_b", "server_c", "server_d", "server_e"))
    s.add_argument("--code", default=None, help="a checkout on /data (default: the laptop's HEAD, deployed once per commit)")
    s.add_argument("--first-guess", default=None, help="a first_guess.json (the window's starting part)")
    s.add_argument("--truth")
    s.add_argument("--extra", default="")
    s.add_argument("--name")
    s.add_argument("--gui", action="store_true", help="open the window on the mirror")
    for name in ("status", "fetch", "stop"):
        q = sub.add_parser(name)
        q.add_argument("mirror", nargs="?" if name == "status" else None)
    a = p.parse_args(argv)
    return dict(start=cmd_start, status=cmd_status, fetch=cmd_fetch, stop=cmd_stop)[a.cmd](a)


if __name__ == "__main__":
    main()
