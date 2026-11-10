#!/usr/bin/env python3
"""Numbers of docs/training_plan_v5.tex, as LaTeX macros (docs/generated/tp5_macros.tex) - never typed by hand.

Sources (all under docs/data/, tracked):
  v5_small_timing.json         scripts/timing_report.py                  generation cost of the small v5 set
  field_backend_bench.json     scripts/bench_field_backend.py            numpy vs torch closest-point query
  v5_small_observability.json  scripts/check_observability_v5.py         analytic baselines per stratum
  v5_small_t0.json             this script, from the T0 checkpoints      T0 of the TypeNet (occupancy / vf_exact) and the axis net
  type_v5_small_t0_eval.json   scripts/eval_stage_ensemble.py --extra-test   legacy type nets on the v5 test set
  models/registry.json         scripts/eval_stage_ensemble.py            the registry's current type ensemble (G0)

    python3 scripts/make_training_plan_v5_tex.py            # refresh v5_small_t0.json when the checkpoints exist, write macros
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
OUT = ROOT / "docs" / "generated" / "tp5_macros.tex"

T0_RUNS = {"occ": "checkpoints/type_hier_t0_occupancy/type_cnn", "vf": "checkpoints/type_hier_t0_vf_exact/type_cnn"}
T0_LOGS = {"occ": "logs_t0/type_occupancy.log", "vf": "logs_t0/type_vf_exact.log"}      # per-epoch wall times
AXIS_RUNS = {"raw": "checkpoints/axis_v5_t0", "align": "checkpoints/axis_v5_t0_align"}


def _epoch_seconds(log: Path):
    import re
    return [float(x) for x in re.findall(r"^epoch .*\(([0-9.]+)s\)$", log.read_text(), re.M)] if log.exists() else None


def collect_t0() -> dict:
    """The T0 numbers from the checkpoints (gitignored) into docs/data/v5_small_t0.json (tracked); the stored file is used
    when the checkpoints are gone."""
    path = DATA / "v5_small_t0.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for key, rel in T0_RUNS.items():
        card = ROOT / rel / "model_card.json"
        if card.exists():
            c = json.loads(card.read_text())
            hist = json.loads((ROOT / rel / "training_history.json").read_text())
            test = next(v for k, v in c["metrics"].items() if k != "val")
            out[f"type_{key}"] = dict(input=c["input_kind"], epochs=c["epochs_run"], train_loss=c["train_loss_first_last"],
                                      val=c["metrics"]["val"], test=test, epoch_seconds=_epoch_seconds(ROOT / T0_LOGS[key]),
                                      n_train=sum(c["counts"]["train"].values()))
    for key, rel in AXIS_RUNS.items():
        h, cfg = ROOT / rel / "training_history.json", ROOT / rel / "config.json"
        if h.exists() and cfg.exists():
            hist, conf = json.loads(h.read_text()), json.loads(cfg.read_text())
            out[f"axis_{key}"] = dict(align=conf.get("align", False), n_train=conf["n_train"], n_val=conf["n_val"],
                                      pca_val=conf["pca_baseline_val"], train_loss=[hist["train_loss"][0], hist["train_loss"][-1]],
                                      val_median_deg_last=hist["val_median_deg"][-1])
    path.write_text(json.dumps(out, indent=1))
    return out


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main():
    m = {}
    tim = json.loads((DATA / "v5_small_timing.json").read_text())
    run = tim["run"]
    m["SmallRecords"] = str(run["records"])
    m["SmallWallMin"] = str(run["wall_min"])
    m["SmallCpuSPerRecord"] = fmt(run["wall_s_per_record"] * 16, 0)
    fields = [s["median"] for s in tim["field_s"].values() if s]
    m["FieldMedLo"], m["FieldMedHi"] = fmt(min(fields), 0), fmt(max(fields), 0)
    fr = {k.split("/")[0]: v["median"] for k, v in tim["freecad_special_s"].items() if k.endswith("/frame") and v}
    m["FrameFcLo"], m["FrameFcHi"] = fmt(min(fr.values()), 0), fmt(max(fr.values()), 0)
    ib = [b["per_sample_s"] for b in tim["ibeam"] if b["split"] == "train"]
    m["IbeamSPerSample"] = fmt(ib[0], 0) if ib else "--"

    bench_path = DATA / "field_backend_bench.json"
    for k in ("BenchN", "BenchCudaSpeedup", "BenchTorchSpeedup", "BenchMaxDiff", "BenchNumpyLo", "BenchNumpyHi",
              "BenchCudaLo", "BenchCudaHi"):
        m[k] = "\\textbf{(pending)}"          # scripts/bench_field_backend.py needs an idle machine
    if bench_path.exists():
        b = json.loads(bench_path.read_text())
        m["BenchN"] = str(len(b["rows"]))
        m["BenchCudaSpeedup"] = fmt(b.get("speedup_cuda_median", float("nan")), 1)
        m["BenchTorchSpeedup"] = fmt(b.get("speedup_torch_median", float("nan")), 1)
        diffs = [r[k] for r in b["rows"] for k in r if k.endswith("_max_abs_diff_mm")]
        m["BenchMaxDiff"] = f"{max(diffs):.0e}".replace("e-", "\\times10^{-") + "}" if diffs else "--"
        m["BenchNumpyLo"] = fmt(min(r["numpy_s"] for r in b["rows"]), 0)
        m["BenchNumpyHi"] = fmt(max(r["numpy_s"] for r in b["rows"]), 0)
        m["BenchCudaLo"] = fmt(min(r["cuda_s"] for r in b["rows"]), 1)
        m["BenchCudaHi"] = fmt(max(r["cuda_s"] for r in b["rows"]), 1)

    prof = json.loads((DATA / "generator_profile_knn.json").read_text())
    m["GenKdS"], m["GenKnnS"], m["GenSpeedup"] = fmt(prof["kdtree_cpu_total_s"], 1), fmt(prof["knn_gpu_total_s"], 1), fmt(prof["speedup"], 1)

    obs = json.loads((DATA / "v5_small_observability.json").read_text())
    for st in ("plain", "plate", "carved", "cut", "negres", "envelope"):
        name = st.capitalize()
        if obs["box_dims_rel"].get(st):
            m[f"BoxExt{name}"] = fmt(obs["box_dims_rel"][st]["median"], 3)
            m[f"BoxMom{name}"] = fmt(obs["box_dims_moments_rel"][st]["median"], 3)
            m[f"BoxN{name}"] = str(obs["box_dims_rel"][st]["n"])
    for st in ("plain", "carved", "cut", "negres"):
        if obs["sphere_radius_rel"].get(st):
            m[f"SphR{st.capitalize()}"] = fmt(obs["sphere_radius_rel"][st]["median"], 3)
    c = obs["centre_cog_vox"]
    m["CogPlainMax"] = fmt(max(v["median"] for k, v in c.items() if k.endswith("/plain")), 2)
    m["CogCutLo"] = fmt(min(v["median"] for k, v in c.items() if k.endswith("/cut")), 1)
    m["CogCutHi"] = fmt(max(v["median"] for k, v in c.items() if k.endswith("/cut")), 1)
    m["CogSectorHi"] = fmt(max(v["median"] for k, v in c.items() if k.endswith("/sector")), 1)
    m["CogBoxPlain"], m["CogConePlain"] = fmt(c["box/plain"]["median"], 2), fmt(c["cone/plain"]["median"], 2)

    # cost model: records per subclass-sample (negres pieces, I-beam plates and envelopes come on top), CPU seconds per
    # record (the whole small run: 16 cores, numpy field, several generators sharing them - an upper bound), disk per record
    per_sub = run["records"] / (24 * (20 + 5 + 10))
    cpu = run["wall_s_per_record"] * 16
    disk_mb = sum(sum(f.stat().st_size for f in (ROOT / "dataset_gen/local_dataset_v5/small").rglob(pat))
                  for pat in ("*.bin", "*.stl", "*.field.npz")) / 1e6 / run["records"] \
        if (ROOT / "dataset_gen/local_dataset_v5/small").exists() else 2.1
    m["RecPerSub"], m["DiskMbPerRecord"] = fmt(per_sub, 2), fmt(disk_mb, 1)
    wk = json.loads((DATA / "generator_freecad_worker.json").read_text())
    m["WorkerOffS"], m["WorkerOnS"], m["WorkerFactor"] = fmt(wk["worker_off_wall_s"], 1), fmt(wk["worker_on_wall_s"], 1), fmt(1 / wk["factor"], 1)
    for tag, n_sub in (("Pilot", 180 + 30 + 50), ("Full", 2000 + 30 + 50)):
        rec = 24 * n_sub * per_sub
        m[f"{tag}CpuHFast"] = fmt(rec * cpu * wk["factor"] / 3600, 0)
        m[f"{tag}WallHFast"] = fmt(rec * cpu * wk["factor"] / 3600 / 64, 1)
        m[f"{tag}Records"] = f"{rec / 1000:.1f}k"
        m[f"{tag}CpuH"] = fmt(rec * cpu / 3600, 0)
        m[f"{tag}WallHSixtyFour"] = fmt(rec * cpu / 3600 / 64, 1)
        m[f"{tag}DiskGb"] = fmt(rec * disk_mb / 1000, 0)

    t0 = collect_t0()
    for key, tag in (("type_occ", "Occ"), ("type_vf", "Vf")):
        if key in t0:
            r = t0[key]
            m[f"TZero{tag}Parent"] = fmt(r["test"]["parent_acc"], 3)
            m[f"TZero{tag}Top"] = fmt(r["test"]["parent_top2"], 3)
            m[f"TZero{tag}Sub"] = fmt(r["test"]["sub_acc"], 3)
            m[f"TZero{tag}LossA"], m[f"TZero{tag}LossB"] = fmt(r["train_loss"][0], 2), fmt(r["train_loss"][1], 2)
            m[f"TZero{tag}N"] = str(r["test"]["n"])
            if r.get("epoch_seconds"):
                m[f"TZero{tag}Epoch"] = fmt(statistics.median(r["epoch_seconds"]), 0)
    m["TZeroNTrain"] = str(t0.get("type_occ", {}).get("n_train", "--"))
    card = ROOT / T0_RUNS["occ"] / "model_card.json"
    if card.exists():                                   # majority-parent baseline of the T0 test set
        counts = next(v for k, v in json.loads(card.read_text())["counts"].items() if k not in ("train", "val"))
        per_parent = {}
        for sub, n in counts.items():
            per_parent[sub.split("/")[0]] = per_parent.get(sub.split("/")[0], 0) + n
        t0.setdefault("type_occ", {})["test_majority_parent"] = max(per_parent.values()) / sum(per_parent.values())
        (DATA / "v5_small_t0.json").write_text(json.dumps(t0, indent=1))
    if "test_majority_parent" in t0.get("type_occ", {}):
        m["TZeroMajority"] = fmt(t0["type_occ"]["test_majority_parent"], 2)
    m["TZeroChanceSub"] = fmt(t0.get("type_occ", {}).get("test", {}).get("chance_sub", float("nan")), 2)
    for key, tag in (("axis_raw", "Raw"), ("axis_align", "Align")):
        if key in t0:
            r = t0[key]
            m[f"Axis{tag}Med"] = fmt(r["val_median_deg_last"], 0)
            m[f"Axis{tag}LossA"], m[f"Axis{tag}LossB"] = fmt(r["train_loss"][0], 2), fmt(r["train_loss"][1], 2)
            m["AxisNTrain"], m["AxisPcaMed"] = str(r["n_train"]), fmt(r["pca_val"]["all"]["median_deg"], 1)
            m["AxisPcaWithin"] = fmt(100 * r["pca_val"]["all"]["within5"], 0)

    ev = json.loads((DATA / "type_v5_small_t0_eval.json").read_text())
    legacy = [r["parent_acc"] for root in ev["roots"].values() for name, r in root["metrics"].items()
              if not name.startswith("hier:")]
    if legacy:
        m["LegacyLo"], m["LegacyHi"] = fmt(min(legacy), 2), fmt(max(legacy), 2)

    desc = json.loads((DATA / "type_descriptor_baseline_v5.json").read_text())   # scripts/eval_descriptor_baseline_v5.py
    m["DescParent"], m["DescTop"] = fmt(desc["parent_acc"], 3), fmt(desc["parent_top2"], 3)
    m["DescSub"], m["DescNFeat"] = fmt(desc["sub_acc_true_parent"], 3), str(len(desc["features"]))
    m["DescCone"] = fmt(desc["per_parent"]["cone"]["acc"], 2)
    dm = json.loads((DATA / "type_descriptor_baseline_v5_medium.json").read_text())   # noised medium set
    m["DescMedParent"], m["DescMedSub"], m["DescMedNTrain"] = fmt(dm["parent_acc"], 3), fmt(dm["sub_acc_true_parent"], 3), str(dm["n_train"])
    sp = json.loads((DATA / "training_speed_bench.json").read_text())      # scripts/bench_training_speed.py, idle GPU
    m["SpEncCpu"], m["SpEncGpu"], m["SpEncX"] = fmt(sp["dirlog_cpu_ms_per_sample"], 0), fmt(sp["dirlog_inside_gpu_ms_per_sample"], 1), fmt(sp["encode_speedup"], 1)
    m["SpStepFp"], m["SpStepAmp"], m["SpStepX"] = fmt(sp["step_ms_fp32"], 0), fmt(sp["step_ms_bf16_channels_last"], 0), fmt(sp["step_speedup"], 1)
    reg = json.loads((ROOT / "models" / "registry.json").read_text())["ensembles"]["type"]      # from ensemble_selection.json
    m["GZeroPooled"], m["GZeroTopTwo"] = fmt(reg["pooled_accuracy"], 3), fmt(reg["pooled_top2"], 3)
    m["GZeroCyl"] = fmt(reg["per_set"]["test_cyl"], 3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("% generated by scripts/make_training_plan_v5_tex.py - do not edit\n" +
                   "".join(f"\\newcommand{{\\Tp{k}}}{{{v}}}\n" for k, v in sorted(m.items())))
    print(f"wrote {OUT} ({len(m)} macros)")


if __name__ == "__main__":
    main()
