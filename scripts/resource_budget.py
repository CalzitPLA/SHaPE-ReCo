#!/usr/bin/env python3
"""Machine-wide resource budget: report + admission check (proposal for scripts/resource_budget.py).

Stdlib only, never touches CUDA, never signals/renices anything. Read-only except for a tiny
reservation ledger in $XDG_RUNTIME_DIR (so two jobs admitted in the same second don't both take
the last 2 GB).

  python3 scripts/resource_budget.py                       # report, grouped by session
  python3 scripts/resource_budget.py --json                # same, machine readable
  python3 scripts/resource_budget.py --need-ram 2G --need-gpu 1.2G --need-cores 4 --wait --timeout 1800
  python3 scripts/resource_budget.py --need-ram 3G --wait -- python3 -m training.train_type_cnn ...  # admit, then exec
  python3 scripts/resource_budget.py --thermal-gate        # block while GPU >= 85 C (resume below 78 C)

Exit codes: 0 admitted / report ok, 3 not admitted (no --wait, or --timeout reached), 2 usage error.
In Python (e.g. between training epochs):
  from scripts.resource_budget import thermal_gate; thermal_gate(max_c=85, resume_c=78)
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---- defaults tuned for this laptop (16 threads, 15.4 GiB RAM, RTX 4050 6 GiB, see guru_governor.md) ----
RESERVE_RAM = 2.0 * 2**30      # MemAvailable must stay >= this after admission
RESERVE_GPU = 0.4 * 2**30      # leave headroom for CUDA context growth / fragmentation
MAX_LOAD = 14.0                # 1-min loadavg + requested cores must stay <= this
MAX_GPU_C = 83                 # don't START a GPU job above this (hard ceiling 85-87 C = throttle)
MAX_MEM_PSI = 10.0             # memory PSI full avg10 (%) above which nothing new starts (swap thrashing)
MAX_CPU_C = 94                 # k10temp Tctl (Ryzen laptop sits at 90-93 under load; throttles at 95); only gates --need-cores jobs
HOLD_S = 120                   # a reservation counts for this long (time for the job to actually allocate)
LEDGER = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / "shape_reco_budget.json"


def parse_size(s: str) -> float:
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]?)i?B?\s*", s, re.I)
    if not m:
        raise argparse.ArgumentTypeError(f"bad size {s!r} (use e.g. 1.5G, 800M)")
    return float(m.group(1)) * 1024 ** "BKMGT".index((m.group(2) or "B").upper())


def fmt(b: float) -> str:
    return f"{b / 2**30:.2f}G" if abs(b) >= 2**30 else f"{b / 2**20:.0f}M"


# ------------------------------------------------------------------ probes (all read-only)
def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        out[k] = int(v.split()[0]) * 1024
    return out


def loadavg() -> tuple[float, float, float]:
    a = Path("/proc/loadavg").read_text().split()
    return float(a[0]), float(a[1]), float(a[2])


def mem_pressure_full10() -> float | None:
    """PSI 'full avg10' for memory: % of time ALL tasks stalled on memory (swap thrash indicator)."""
    try:
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            if line.startswith("full"):
                return float(line.split()[1].split("=")[1])
    except OSError:
        pass
    return None


def cpu_temp() -> float | None:
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (hw / "name").read_text().strip() in ("k10temp", "coretemp"):
                return int((hw / "temp1_input").read_text()) / 1000.0
        except OSError:
            continue
    return None


def gpu_state() -> dict | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw,"
             "clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip()
        t, used, tot, util, pw, swt, hwt = [x.strip() for x in q.splitlines()[0].split(",")]
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
        per_pid = {}
        for line in apps.splitlines():
            if "," in line:
                p, m = line.split(",")
                try:
                    per_pid[int(p)] = int(m) * 2**20
                except ValueError:
                    pass
        return {"temp": float(t), "used": float(used) * 2**20, "total": float(tot) * 2**20, "util": float(util),
                "power": float(pw) if pw.replace(".", "").isdigit() else None,
                "thermal_throttle": "Active" in (swt, hwt), "per_pid": per_pid}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _stat(pid: int):
    s = Path(f"/proc/{pid}/stat").read_text()
    rest = s[s.rindex(")") + 2:].split()
    return int(rest[1]), int(rest[11]) + int(rest[12]), s[s.index("(") + 1:s.rindex(")")]  # ppid, utime+stime, comm


def processes(sample_s: float = 1.0) -> dict[int, dict]:
    def snap():
        d = {}
        for p in Path("/proc").iterdir():
            if p.name.isdigit():
                try:
                    d[int(p.name)] = _stat(int(p.name))
                except (OSError, ValueError, IndexError):
                    pass
        return d
    a = snap(); t0 = time.monotonic(); time.sleep(sample_s); b = snap(); dt = time.monotonic() - t0
    hz = os.sysconf("SC_CLK_TCK")
    procs = {}
    for pid, (ppid, ticks, comm) in b.items():
        cpu = 100.0 * (ticks - a[pid][1]) / hz / dt if pid in a else 0.0
        info = {"pid": pid, "ppid": ppid, "comm": comm, "cpu": cpu, "rss": 0, "cwd": "?", "cmd": ""}
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    info["rss"] = int(line.split()[1]) * 1024
            info["cmd"] = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")[:300]
            info["cwd"] = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            pass  # other users' processes: rss/cpu still count, cwd unknown
        procs[pid] = info
    return procs


def session_of(pid: int, procs: dict) -> str:
    """Nearest `claude` ancestor, else a scratchpad session UUID in an ancestor's cmdline, else 'other'."""
    seen, p = set(), pid
    uuid = None
    while p in procs and p not in seen and p > 1:
        seen.add(p)
        info = procs[p]
        if info["comm"] == "claude":
            return f"claude[{p}]" + (f" ({uuid[:8]})" if uuid else "")
        m = re.search(r"/tmp/claude-\d+/[^/ ]+/([0-9a-f]{8}-[0-9a-f-]{27})", info["cmd"])
        if m and not uuid:
            uuid = m.group(1)
        p = info["ppid"]
    return f"session {uuid[:8]} (detached)" if uuid else "other"


def job_root(pid: int, procs: dict) -> int:
    """Topmost ancestor that is still a 'job' (python/freecadcmd/prterun/pt_data_worker...), i.e. collapses workers."""
    jobish = re.compile(r"python|freecad|prterun|mpirun|pt_data_worker|pytest|torch")
    root, p = pid, procs[pid]["ppid"]
    while p in procs and jobish.search(procs[p]["comm"] + " " + procs[p]["cmd"][:60]):
        root, p = p, procs[p]["ppid"]
    return root


# ------------------------------------------------------------------ report
def report(as_json: bool, top: int) -> int:
    procs = processes()
    gpu = gpu_state()
    gpu_pid = gpu["per_pid"] if gpu else {}
    mi = meminfo()
    jobs: dict[int, dict] = defaultdict(lambda: {"rss": 0, "cpu": 0.0, "gpu": 0, "n": 0})
    for pid, info in procs.items():
        if not re.search(r"python|freecad|prterun|mpirun|pt_data_worker|pytest|torch", info["comm"] + " " + info["cmd"][:60]):
            continue
        r = job_root(pid, procs)
        j = jobs[r]
        j["rss"] += info["rss"]; j["cpu"] += info["cpu"]; j["gpu"] += gpu_pid.get(pid, 0); j["n"] += 1
    rows = []
    for r, j in jobs.items():
        if j["cpu"] < 1 and j["rss"] < 100 * 2**20 and not j["gpu"]:
            continue
        info = procs[r]
        rows.append({"root_pid": r, "session": session_of(r, procs), "cwd": info["cwd"], "procs": j["n"],
                     "rss": j["rss"], "cpu_pct": round(j["cpu"]), "gpu_mem": j["gpu"], "cmd": info["cmd"][:110]})
    rows.sort(key=lambda x: (x["session"], -x["rss"]))
    state = {"time": time.strftime("%F %T"), "loadavg": loadavg(), "cores": os.cpu_count(),
             "mem_total": mi["MemTotal"], "mem_available": mi["MemAvailable"], "mem_free": mi["MemFree"],
             "swap_used": mi["SwapTotal"] - mi["SwapFree"], "cpu_temp": cpu_temp(),
             "gpu": {k: v for k, v in (gpu or {}).items() if k != "per_pid"}, "jobs": rows[: top or None]}
    if as_json:
        print(json.dumps(state, indent=1, default=str)); return 0
    la = state["loadavg"]
    print(f"{state['time']}  load {la[0]:.1f}/{la[1]:.1f}/{la[2]:.1f} on {state['cores']} threads   "
          f"RAM avail {fmt(mi['MemAvailable'])} (free {fmt(mi['MemFree'])}) of {fmt(mi['MemTotal'])}   "
          f"swap used {fmt(state['swap_used'])}   mem-PSI full10 {mem_pressure_full10()}%   CPU {state['cpu_temp']:.0f} C")
    print("(RSS = sum over the job's processes; shared pages are counted per process, so it is an upper bound)")
    if gpu:
        print(f"GPU {gpu['temp']:.0f} C  util {gpu['util']:.0f}%  mem {fmt(gpu['used'])}/{fmt(gpu['total'])}  "
              f"power {gpu['power']} W  thermal-throttle={'YES' if gpu['thermal_throttle'] else 'no'}")
    per_sess = defaultdict(lambda: [0, 0.0, 0])
    for r in rows:
        s = per_sess[r["session"]]; s[0] += r["rss"]; s[1] += r["cpu_pct"]; s[2] += r["gpu_mem"]
    print(f"\n{'session':32s} {'RSS':>8s} {'CPU%':>6s} {'GPU':>7s}")
    for s, (rss, cpu, g) in sorted(per_sess.items(), key=lambda kv: -kv[1][0]):
        print(f"{s:32s} {fmt(rss):>8s} {cpu:6.0f} {fmt(g):>7s}")
    print(f"\n{'root':>8s} {'n':>3s} {'RSS':>7s} {'CPU%':>5s} {'GPU':>6s}  session / cwd / cmd")
    for r in rows[: top or None]:
        print(f"{r['root_pid']:8d} {r['procs']:3d} {fmt(r['rss']):>7s} {r['cpu_pct']:5d} {fmt(r['gpu_mem']):>6s}  "
              f"{r['session']}  {r['cwd'].replace(str(Path.home()), '~')}\n{'':34s}{r['cmd']}")
    return 0


# ------------------------------------------------------------------ admission
def _ledger(update=None) -> list[dict]:
    """Return live reservations (optionally appending one) under an flock."""
    LEDGER.touch(exist_ok=True)
    with open(LEDGER, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            entries = json.loads(fh.read() or "[]")
        except json.JSONDecodeError:
            entries = []
        now = time.time()
        entries = [e for e in entries if e["until"] > now]
        if update:
            entries.append(update)
        fh.seek(0); fh.truncate(); fh.write(json.dumps(entries))
        return entries


def check(a) -> tuple[bool, list[str]]:
    mi, la, gpu, ct = meminfo(), loadavg(), gpu_state(), cpu_temp()
    held = _ledger()
    held_ram = sum(e["ram"] for e in held); held_gpu = sum(e["gpu"] for e in held); held_cores = sum(e["cores"] for e in held)
    why = []
    if mi["MemAvailable"] - held_ram - a.need_ram < a.reserve_ram:
        why.append(f"RAM: avail {fmt(mi['MemAvailable'])} - held {fmt(held_ram)} - need {fmt(a.need_ram)} < reserve {fmt(a.reserve_ram)}")
    psi = mem_pressure_full10()
    if a.need_ram and psi is not None and psi > a.max_mem_psi:
        why.append(f"memory pressure full avg10 {psi:.1f}% > {a.max_mem_psi}% (swapping)")
    if a.need_cores and la[0] + held_cores + a.need_cores > a.max_load:
        why.append(f"load: {la[0]:.1f} + held {held_cores} + need {a.need_cores} > {a.max_load}")
    if a.need_cores and ct is not None and ct >= a.max_cpu_temp:
        why.append(f"CPU temp {ct:.0f} C >= {a.max_cpu_temp}")
    if a.need_gpu > 0:
        if gpu is None:
            why.append("GPU requested but nvidia-smi unavailable")
        else:
            free = gpu["total"] - gpu["used"] - held_gpu
            if free - a.need_gpu < a.reserve_gpu:
                why.append(f"GPU mem: free {fmt(free)} - need {fmt(a.need_gpu)} < reserve {fmt(a.reserve_gpu)}")
            if gpu["temp"] >= a.max_gpu_temp:
                why.append(f"GPU temp {gpu['temp']:.0f} C >= {a.max_gpu_temp} (thermal pause)")
    return not why, why


def admit(a) -> int:
    t0, delay = time.monotonic(), 5.0
    while True:
        ok, why = check(a)
        if ok:
            _ledger({"pid": os.getppid(), "ram": a.need_ram, "gpu": a.need_gpu, "cores": a.need_cores,
                     "until": time.time() + a.hold, "tag": a.tag})
            print(f"[budget] admitted after {time.monotonic() - t0:.0f}s: ram {fmt(a.need_ram)} gpu {fmt(a.need_gpu)} "
                  f"cores {a.need_cores}", file=sys.stderr)
            if a.cmd:
                os.execvp(a.cmd[0], a.cmd)
            return 0
        waited = time.monotonic() - t0
        if not a.wait or waited >= a.timeout:
            print("[budget] NOT admitted: " + "; ".join(why), file=sys.stderr)
            return 3
        print(f"[budget] waiting ({waited:.0f}s): " + "; ".join(why), file=sys.stderr)
        time.sleep(min(delay * random.uniform(0.8, 1.2), a.timeout - waited))  # jitter: waiters don't stampede
        delay = min(delay * 1.6, 60.0)


def thermal_gate(max_c: float = 85, resume_c: float = 78, poll_s: float = 15, max_wait_s: float = 900) -> float:
    """Block while the GPU is at/above max_c until it cools to resume_c (hysteresis). Returns seconds paused.
    Cheap to call between epochs; a no-op without nvidia-smi."""
    g = gpu_state()
    if not g or g["temp"] < max_c:
        return 0.0
    t0 = time.monotonic()
    while time.monotonic() - t0 < max_wait_s:
        print(f"[budget] GPU {g['temp']:.0f} C >= {max_c}: pausing until <= {resume_c} C", file=sys.stderr)
        time.sleep(poll_s)
        g = gpu_state()
        if not g or g["temp"] <= resume_c:
            break
    return time.monotonic() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="report as JSON")
    ap.add_argument("--top", type=int, default=0, help="report only the N biggest jobs")
    ap.add_argument("--need-ram", type=parse_size, default=0.0)
    ap.add_argument("--need-gpu", type=parse_size, default=0.0)
    ap.add_argument("--need-cores", type=int, default=0, help="threads the job will keep busy")
    ap.add_argument("--reserve-ram", type=parse_size, default=RESERVE_RAM)
    ap.add_argument("--reserve-gpu", type=parse_size, default=RESERVE_GPU)
    ap.add_argument("--max-load", type=float, default=MAX_LOAD)
    ap.add_argument("--max-mem-psi", type=float, default=MAX_MEM_PSI)
    ap.add_argument("--max-gpu-temp", type=float, default=MAX_GPU_C)
    ap.add_argument("--max-cpu-temp", type=float, default=MAX_CPU_C)
    ap.add_argument("--wait", action="store_true", help="wait with backoff (5 s -> 60 s) instead of failing")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--hold", type=float, default=HOLD_S, help="seconds the reservation counts against others")
    ap.add_argument("--tag", default="", help="label stored in the reservation ledger")
    ap.add_argument("--thermal-gate", action="store_true", help="block while GPU >= 85 C, resume at 78 C")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command to exec once admitted")
    a = ap.parse_args(argv)
    a.cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if a.thermal_gate:
        thermal_gate(); return 0
    if a.need_ram or a.need_gpu or a.need_cores or a.cmd:
        return admit(a)
    return report(a.json, a.top)


if __name__ == "__main__":
    sys.exit(main())
