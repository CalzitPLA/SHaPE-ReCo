#!/usr/bin/env python3
"""
Numbers, tables and figures for docs/slides_training_data.tex (the training-data plan for the add-cut cascade), generated
from what is already on disk - so no number in the deck is typed by hand:

  * the checkpoint histories/configs (checkpoints_*/.../config.json + training_history.json, docs/data/*.json): what each
    earlier data set taught (Type accuracy and rotation error against the number of training samples);
  * the dataset labels (dataset_gen/local_dataset_inscale/*/shard*/<topology>/dataset_metadata.jsonl and the header of
    dataset_gen/local_dataset_type_pointcnn_big): shape variance per kind, size, cut/carve/hollow/noise shares;
  * the measured timings (field: docs/VECTORFIELD.md; generation: STL mtimes of existing shards; training: epoch times
    of existing runs) for the cost estimate of the proposed set.

Read-only on the data; runs in a few seconds:

    python3 scripts/make_training_data_slides.py

writes docs/generated/tdata_macros.tex, tdata_variance_table.tex, tdata_counts_table.tex, tdata_ensemble_table.tex (only
when docs/data/ensemble_selection.json exists) and
docs/figures/tdata_learning.pdf, tdata_variance.pdf.
"""
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, GEN = ROOT / "docs" / "data", ROOT / "docs" / "figures", ROOT / "docs" / "generated"
DS = ROOT / "dataset_gen" / "local_dataset_inscale"
PCNN = ROOT / "dataset_gen" / "local_dataset_type_pointcnn_big"

COL = {"box": "#2c6fbb", "sphere": "#c0392b", "cylinder": "#2e8b57", "cone": "#8e44ad", "torus": "#d68910"}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

MACROS = {}


def macro(name, value):
    MACROS[name] = value


def fmt(x, d=1):
    return f"{x:.{d}f}"


def thousands(n):
    return f"{int(round(n)):,}".replace(",", "\\,")


# --------------------------------------------------------------------------------------------------------------------
# 1. what earlier runs taught: best val metric against training-set size
# --------------------------------------------------------------------------------------------------------------------
def ckpt(d):
    d = ROOT / d
    cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
    hist = json.loads((d / "training_history.json").read_text())
    return cfg, hist


TYPE_RUNS = [  # (label, checkpoint dir, n classes, n_train fallback, input)
    ("in-scale v1", "checkpoints_inscale/type_cnn", 2, 1280, "occ"),
    ("in-scale v2", "checkpoints_inscale_v2/type_cnn", 2, 3200, "occ"),
    ("v3 carve", "checkpoints_type_v3/carve_only/type_cnn", 2, None, "occ"),
    ("v3 carve+v2 (B)", "checkpoints_type_v3/carve_plus_v2data/type_cnn", 2, None, "occ"),
    ("v3 grid SDF", "checkpoints_type_v3/carve_plus_v2data_sdf/type_cnn", 2, None, "sdf"),
    ("v4 (+cyl)", "checkpoints_type_v4/type_cnn", 3, None, "occ"),
    ("hollow", "checkpoints_hollow/type_cnn", 3, None, "occ"),
    ("fused study: occ", "checkpoints_fused/type_occupancy/type_cnn", 3, None, "occ"),
    ("fused study: fused", "checkpoints_fused/type_fused/type_cnn", 3, None, "fused"),
    ("exact field 128$^3$", "checkpoints_exact_field/type_cnn", 3, None, "field"),
    ("point net, 5 kinds", "checkpoints_field_pertopo/type_cnn", 5, None, "field"),
]


def type_points():
    out = []
    for label, d, ncls, n_fb, inp in TYPE_RUNS:
        try:
            cfg, hist = ckpt(d)
        except FileNotFoundError:
            continue
        n = cfg.get("n_train", n_fb)
        per = cfg.get("train_per_class")
        n_per = min(per.values()) if per else n / ncls
        out.append(dict(label=label, n=n, n_per=n_per, ncls=ncls, acc=max(hist["val_acc"]), input=inp,
                        epochs=len(hist["val_acc"])))
    return out


ROT_RUNS = [
    ("B raw grid", "checkpoints_rot_sym/B_6d/rotation_cnn", "box"),
    ("E PCA-aligned", "checkpoints_rot_sym/E_align/rotation_cnn", "box"),
    ("G PCA, 3.9k", "checkpoints_rot_sym/G_align_jit_more/rotation_cnn", "box"),
    ("H manifold", "checkpoints_rot_sym/H_manifold/rotation_cnn", "box"),
    ("I frame", "checkpoints_rot_sym/I_frame/rotation_cnn", "box"),
    ("pts box", "checkpoints_field_pertopo/box_rotation_cnn", "box"),
    ("pts cyl", "checkpoints_field_pertopo/cylinder_rotation_cnn", "cylinder"),
    ("pts cone", "checkpoints_field_pertopo/cone_rotation_cnn", "cone"),
    ("pts torus", "checkpoints_field_pertopo/torus_rotation_cnn", "torus"),
]


def rot_points():
    out = []
    for label, d, kind in ROT_RUNS:
        try:
            cfg, hist = ckpt(d)
        except FileNotFoundError:
            continue
        med = min(hist["val_median_deg"]) if hist.get("val_median_deg") else None
        out.append(dict(label=label, n=cfg["n_train"], mean=min(hist["val_err_deg"]), median=med, kind=kind))
    return out


def epoch_seconds():
    """Seconds per (train+val) sample per epoch, occupancy vs 5-channel input, from existing runs (checkpoint mtimes)."""
    def per_sample(d, epochs=None):
        d = ROOT / d
        cfg, hist = ckpt(d.relative_to(ROOT))
        t = sorted(os.path.getmtime(f) for f in glob.glob(str(d / "*")))
        ep = epochs or len(hist["val_loss"])
        return (t[-1] - t[0]) / ep / (cfg["n_train"] + cfg["n_val"])
    occ = per_sample("checkpoints_fused/type_occupancy/type_cnn")
    fus = per_sample("checkpoints_fused/type_fused/type_cnn")
    return occ, fus


# --------------------------------------------------------------------------------------------------------------------
# 2. shape variance per kind, from the labels
# --------------------------------------------------------------------------------------------------------------------
def load_records():
    recs = []
    for f in sorted(glob.glob(str(DS / "*" / "shard*" / "*" / "dataset_metadata.jsonl"))):
        split = Path(f).parts[-4]
        for line in open(f):
            r = json.loads(line)
            if r.get("record_type") == "dataset_header":
                continue
            r["_split"] = split
            recs.append(r)
    return recs


def tie_class(half):
    """1 = generic box (D2, 4 symmetries), 2 = two near-equal dims (8), 3 = cube-like (24) - TIE_TOLERANCE 0.10 as
    in training/rotation_symmetry.py."""
    d = sorted(half, reverse=True)
    t01 = (d[0] - d[1]) <= 0.10 * d[0]
    t12 = (d[1] - d[2]) <= 0.10 * d[1]
    return 3 if (t01 and t12) else (2 if (t01 or t12) else 1)


def variance(recs):
    kinds = {}
    for r in recs:
        if r["_split"].startswith("test"):
            continue
        k = r["topology"]
        p, pl = r["parameters"], r["placement"]
        e = kinds.setdefault(k, dict(n=0, aspect=[], size=[], vox=[], cut=0, carve=0, hollow=0, noise=0, noise_known=0,
                                     removed=[], resid=[], ties=[0, 0, 0, 0], cutters={}, wall=[]))
        e["n"] += 1
        if k == "box":
            h = [p["x_half_ratio"], p["y_half_ratio"], p["z_half_ratio"]]
            e["aspect"].append(max(h) / min(h))
            e["ties"][tie_class(h)] += 1
        elif k == "cylinder":
            e["aspect"].append(p["half_length_ratio"] / p["radius_ratio"])        # length / diameter
        e["size"].append(pl["size_fraction"] * pl.get("grid_mm", 40.0))
        e["vox"].append(pl["voxels"])
        if pl.get("cuts"):
            e["cut"] += 1
        if "carve" in pl:
            e["carve"] += 1
            c = pl["carve"]
            e["removed"].append(c["removed_fraction"])
            e["cutters"][c["cutter"]] = e["cutters"].get(c["cutter"], 0) + 1
            if not pl.get("noise"):
                e["resid"].append(pl["voxels_uncut"] - pl["voxels"])
        if "hollow" in pl:
            e["hollow"] += 1
            e["wall"].append(pl["hollow"]["wall_fraction"])
        if "noise" in pl:
            e["noise_known"] += 1
            e["noise"] += bool(pl["noise"])
    return kinds


def pcnn_specs():
    hdr = json.loads(open(PCNN / "dataset_metadata.jsonl").readline())
    return hdr["parameter_specs"], hdr["count_per_type"]


# --------------------------------------------------------------------------------------------------------------------
# 3. the proposed set and its cost
# --------------------------------------------------------------------------------------------------------------------
# train samples per kind (whole bodies). Argument on the slides: Type saturates at >=1.3k/class for 2-3 classes;
# box rotation needed ~3k (H: 3.2k -> 9 deg); axis kinds have 2 rotational DOF and 1-2 shape ratios; sphere has none.
PLAN = [  # kind, train, segment share, stages
    ("box", 3000, 0.0, "TRDP"),
    ("sphere", 1000, 0.25, "TDP"),
    ("cylinder", 2000, 0.25, "TRDP"),
    ("cone", 2000, 0.25, "TRDP"),
    ("torus", 2000, 0.25, "TRDP"),
]
VAL_FRAC = 0.15
CARVE_SHARE = 0.40          # of whole bodies: carved by one cutter of another kind -> 1 negative-residual sample each
HOLLOW_SHARE = 0.10
CUT_SHARE = 0.20
TEST_PER_NEW = 400          # new test sets: cone, torus, segments, residual
N_POINTS = 2048
FIELD_GRID_S = 1.8          # docs/VECTORFIELD.md: 64^3 grid, approximate mode, 20 mm part
EXACT_20K_S = 7.6           # docs/VECTORFIELD.md: 20 000 exact near-surface samples
CORES = 14                  # of 16, two left for the desktop
EPOCHS_TYPE = 15
TYPE_PER_KIND = 1200        # Type run: whole bodies per kind + the same number of residual pieces (fits the 1 h budget)
EPOCHS_ROT = 30


def gen_seconds():
    """Wall seconds per sample of the existing generator per shard process (STL mtimes), per kind."""
    out = {}
    for kind, d in (("box", DS / "carve/shard0/box"), ("sphere", DS / "carve/shard0/sphere"),
                    ("cylinder", DS / "cyl/shard0/cylinder"), ("cone", PCNN / "cone"), ("torus", PCNN / "torus")):
        t = sorted(os.path.getmtime(f) for f in glob.glob(str(d / "*.stl")))
        out[kind] = (t[-1] - t[0]) / max(len(t) - 1, 1)
    return out


# --------------------------------------------------------------------------------------------------------------------
def main():
    FIG.mkdir(exist_ok=True)
    GEN.mkdir(exist_ok=True)
    tp, rp = type_points(), rot_points()
    occ_s, fus_s = epoch_seconds()
    recs = load_records()
    var = variance(recs)
    specs, pc_per = pcnn_specs()
    gsec = gen_seconds()

    # ---------- lessons macros
    by = {p["label"]: p for p in tp}
    macro("TdTypeVOneAcc", fmt(100 * by["in-scale v1"]["acc"]))
    macro("TdTypeVOneNper", thousands(by["in-scale v1"]["n_per"]))
    macro("TdTypeVTwoAcc", fmt(100 * by["in-scale v2"]["acc"]))
    macro("TdTypeVTwoNper", thousands(by["in-scale v2"]["n_per"]))
    macro("TdTypeBAcc", fmt(100 * by["v3 carve+v2 (B)"]["acc"]))
    macro("TdTypeBNper", thousands(by["v3 carve+v2 (B)"]["n_per"]))
    macro("TdTypeSdfAcc", fmt(100 * by["v3 grid SDF"]["acc"]))
    macro("TdTypeVFourAcc", fmt(100 * by["v4 (+cyl)"]["acc"]))
    macro("TdTypeVFourNper", thousands(by["v4 (+cyl)"]["n_per"]))
    macro("TdTypeHollowAcc", fmt(100 * by["hollow"]["acc"]))
    macro("TdTypeHollowNper", thousands(by["hollow"]["n_per"]))
    macro("TdTypeFusOccAcc", fmt(100 * by["fused study: occ"]["acc"]))
    macro("TdTypeFusFusAcc", fmt(100 * by["fused study: fused"]["acc"]))
    macro("TdTypePtsAcc", fmt(100 * by["point net, 5 kinds"]["acc"]))
    macro("TdTypePtsNper", thousands(by["point net, 5 kinds"]["n_per"]))
    rb = {p["label"]: p for p in rp}
    macro("TdRotBMean", fmt(rb["B raw grid"]["mean"]))
    macro("TdRotBN", thousands(rb["B raw grid"]["n"]))
    macro("TdRotGMean", fmt(rb["G PCA, 3.9k"]["mean"]))
    macro("TdRotGN", thousands(rb["G PCA, 3.9k"]["n"]))
    macro("TdRotHMean", fmt(rb["H manifold"]["mean"]))
    macro("TdRotHMedian", fmt(rb["H manifold"]["median"]))
    macro("TdRotHN", thousands(rb["H manifold"]["n"]))
    macro("TdRotIMean", fmt(rb["I frame"]["mean"]))
    pts = [p for p in rp if p["label"].startswith("pts")]
    macro("TdRotPtsLo", fmt(min(p["mean"] for p in pts), 0))
    macro("TdRotPtsHi", fmt(max(p["mean"] for p in pts), 0))
    macro("TdRotPtsN", thousands(np.mean([p["n"] for p in pts])))
    old = json.loads((DATA / "type_cnn_eval.json").read_text())["results"]
    macro("TdOldCnnInscaleAcc", fmt(100 * old["old@inscale/test"]["accuracy"]))
    dims = json.loads((DATA / "box_dims_eval.json").read_text())["methods"]
    macro("TdDimsMeasRotRel", fmt(dims["measured+rot"]["rel"]["median"], 2))
    macro("TdDimsCnnRel", fmt(dims["dims_cnn"]["rel"]["median"], 2))

    # ---------- variance table
    kinds_order = ("box", "sphere", "cylinder")
    rows = []
    for k in kinds_order:
        e = var[k]
        size = np.array(e["size"])
        vox = np.array(e["vox"])
        asp = np.array(e["aspect"]) if e["aspect"] else None
        asp_s = f"{np.median(asp):.2f} [{np.percentile(asp, 5):.1f}, {np.percentile(asp, 95):.1f}]" if asp is not None else "--"
        noise = 100 * e["noise"] / e["noise_known"] if e["noise_known"] else float("nan")
        rows.append(f"{k} & {thousands(e['n'])} & {asp_s} & {np.percentile(size, 5):.0f}--{np.percentile(size, 95):.0f} & "
                    f"{thousands(np.median(vox))} & {100 * e['cut'] / e['n']:.0f} & {100 * e['carve'] / e['n']:.0f} & "
                    f"{100 * e['hollow'] / e['n']:.0f} & {noise:.0f}")
    s = specs
    rows.append(f"cone$^*$ & {pc_per} & $h/r_1 \\in$ [{s['cone']['height_ratio']['min'] / s['cone']['radius1_ratio']['max']:.2f}, "
                f"{s['cone']['height_ratio']['max'] / s['cone']['radius1_ratio']['min']:.0f}] & (fit mode) & -- & 0 & 0 & 0 & 0")
    rows.append(f"torus$^*$ & {pc_per} & $r/R \\in$ [{s['torus']['minor_radius_ratio']['min'] / s['torus']['major_radius_ratio']['max']:.3f}, "
                f"{s['torus']['minor_radius_ratio']['max'] / s['torus']['major_radius_ratio']['min']:.1f}] & (fit mode) & -- & 0 & 0 & 0 & 0")
    (GEN / "tdata_variance_table.tex").write_text("% generated by scripts/make_training_data_slides.py - do not edit\n"
                                                  + " \\\\\n".join(rows) + "\n")
    b = var["box"]
    nt = sum(b["ties"])
    macro("TdBoxGeneric", fmt(100 * b["ties"][1] / nt, 0))
    macro("TdBoxTwoTie", fmt(100 * b["ties"][2] / nt, 0))
    macro("TdBoxCube", fmt(100 * b["ties"][3] / nt, 1))
    allrem = np.concatenate([np.array(var[k]["removed"]) for k in var if var[k]["removed"]])
    allres = np.concatenate([np.array(var[k]["resid"]) for k in var if var[k]["resid"]])
    macro("TdCarveRemovedMed", fmt(100 * np.median(allrem), 0))
    macro("TdCarveRemovedLo", fmt(100 * np.percentile(allrem, 5), 0))
    macro("TdCarveRemovedHi", fmt(100 * np.percentile(allrem, 95), 0))
    macro("TdResidVoxMed", thousands(np.median(allres)))
    macro("TdResidVoxLo", thousands(np.percentile(allres, 5)))
    macro("TdResidVoxHi", thousands(np.percentile(allres, 95)))
    macro("TdResidBelowHundred", fmt(100 * np.mean(allres < 100), 0))
    macro("TdCarveN", thousands(len(allrem)))
    ntrain = sum(var[k]["n"] for k in var)
    macro("TdExistingTrainN", thousands(ntrain))
    macro("TdExistingPcnnPer", str(pc_per))
    macro("TdCylAspectMed", fmt(np.median(var["cylinder"]["aspect"]), 2))
    macro("TdBoxAspectMed", fmt(np.median(var["box"]["aspect"]), 2))
    macro("TdBoxAspectNinetyFive", fmt(np.percentile(var["box"]["aspect"], 95), 1))
    walls = np.concatenate([np.array(var[k]["wall"]) for k in var if var[k]["wall"]])
    macro("TdHollowWallMed", fmt(np.median(walls), 2))

    # ---------- proposed counts and cost
    rows = []
    tot = dict(train=0, val=0, resid=0, test=0)
    field_s = FIELD_GRID_S + EXACT_20K_S * N_POINTS / 20000.0
    gen_wall = 0.0
    for kind, ntr, seg, stages in PLAN:
        nval = int(round(ntr * VAL_FRAC))
        nres = int(round(ntr * CARVE_SHARE))                     # every kind can be the base of a carve
        test = {"box": "reuse 4", "sphere": "reuse 3", "cylinder": "reuse 1",
                "cone": f"new {TEST_PER_NEW}", "torus": f"new {TEST_PER_NEW}"}[kind]
        ntest = 0 if kind in ("box", "sphere", "cylinder") else TEST_PER_NEW
        rows.append(f"{kind} & {thousands(ntr)} & {thousands(nval)} & {thousands(nres)} & "
                    f"{'--' if seg == 0 else fmt(100 * seg, 0) + '\\,\\%'} & {stages} & {test}")
        tot["train"] += ntr
        tot["val"] += nval
        tot["resid"] += nres
        tot["test"] += ntest
        n_all = ntr + nval + nres + ntest
        gen_wall += n_all * (gsec[kind] * 2.0 + field_s)          # x2: Part boolean + second tessellation for carved
    tot["test"] += 2 * TEST_PER_NEW                                # segments, residual
    gen_wall += 2 * TEST_PER_NEW * (np.mean(list(gsec.values())) * 2.0 + field_s)
    rows.append(f"\\midrule total & {thousands(tot['train'])} & {thousands(tot['val'])} & {thousands(tot['resid'])} & & & "
                f"+{thousands(tot['test'])}")
    (GEN / "tdata_counts_table.tex").write_text("% generated by scripts/make_training_data_slides.py - do not edit\n"
                                                + " \\\\\n".join(rows) + "\n")
    n_total = tot["train"] + tot["val"] + tot["resid"] + tot["test"]
    macro("TdPlanTrain", thousands(tot["train"]))
    macro("TdPlanVal", thousands(tot["val"]))
    macro("TdPlanResid", thousands(tot["resid"]))
    macro("TdPlanTest", thousands(tot["test"]))
    macro("TdPlanTotal", thousands(n_total))
    macro("TdCarveShare", fmt(100 * CARVE_SHARE, 0))
    macro("TdHollowShare", fmt(100 * HOLLOW_SHARE, 0))
    macro("TdCutShare", fmt(100 * CUT_SHARE, 0))
    macro("TdValFrac", fmt(100 * VAL_FRAC, 0))
    macro("TdTestPerNew", str(TEST_PER_NEW))
    macro("TdNPoints", thousands(N_POINTS))
    macro("TdFieldGridS", fmt(FIELD_GRID_S))
    macro("TdExactTwentyKS", fmt(EXACT_20K_S))
    macro("TdFieldPerSampleS", fmt(field_s))
    macro("TdCores", str(CORES))
    macro("TdGenMinutes", fmt(gen_wall / CORES / 60, 0))
    macro("TdGenBoxS", fmt(gsec["box"], 2))
    macro("TdGenTorusS", fmt(gsec["torus"], 2))
    # storage: V (3x64^3) + SDF (64^3) as float16, points+normals 2048x6 float32; occupancy .bin 64^3 uint8
    npz_mb = (4 * 64 ** 3 * 2 + N_POINTS * 6 * 4) / 2 ** 20
    macro("TdNpzMB", fmt(npz_mb))
    macro("TdDiskGB", fmt(n_total * (npz_mb + 0.25) / 1024, 0))
    ram_gb = (tot["train"] + tot["resid"]) * 5 * 64 ** 3 * 2 / 2 ** 30
    macro("TdRamGB", fmt(ram_gb, 0))
    # training time within 1 h
    macro("TdOccMsPerSample", fmt(1000 * occ_s))
    macro("TdFusMsPerSample", fmt(1000 * fus_s))
    type_n = TYPE_PER_KIND * (len(PLAN) + 1)
    macro("TdTypePerKind", thousands(TYPE_PER_KIND))
    type_n_all = type_n * (1 + VAL_FRAC)
    macro("TdTypeTrainN", thousands(type_n))
    macro("TdTypeRunOccMin", fmt(type_n_all * occ_s * EPOCHS_TYPE / 60, 0))
    macro("TdTypeRunFusMin", fmt(type_n_all * fus_s * EPOCHS_TYPE / 60, 0))
    macro("TdEpochsType", str(EPOCHS_TYPE))
    macro("TdEpochsRot", str(EPOCHS_ROT))
    box_n = 3000 * (1 + VAL_FRAC)
    macro("TdRotBoxRunFusMin", fmt(box_n * fus_s * EPOCHS_ROT / 60, 0))
    macro("TdMaxNFusHour", thousands(3600 / (fus_s * EPOCHS_TYPE)))
    macro("TdMaxNOccHour", thousands(3600 / (occ_s * EPOCHS_TYPE)))

    # ensemble selection (agent 1), if present
    ens = DATA / "ensemble_selection.json"
    macro("TdHaveEnsemble", "1" if ens.exists() else "0")
    if ens.exists():
        esc = lambda t: str(t).replace("_", "\\_")
        val = lambda b: "--" if not isinstance(b.get("value"), (int, float)) else (
            f"{b['value']:.3g}" if b["value"] < 1 or b["metric"] == "accuracy" else f"{b['value']:.1f}")
        rows = []
        for stage, per in json.loads(ens.read_text())["stages"].items():
            for kind, lst in per.items():
                if not isinstance(lst, list) or not lst or kind == "all":
                    continue
                b = lst[0]
                ru = lst[1] if len(lst) > 1 else None
                rows.append(f"{stage} & {kind} & \\code{{{esc(b['id'])}}} ({esc(b['mode'])}) & {esc('peel, 41 parts' if str(b['metric']).startswith('not scored') else b['metric'])} & {val(b)} & "
                            + (f"\\code{{{esc(ru['id'])}}} {val(ru)}" if ru else "--"))
        (GEN / "tdata_ensemble_table.tex").write_text("% generated by scripts/make_training_data_slides.py - do not edit\n"
                                                      + " \\\\\n".join(rows) + "\n")

    # ---------- figures
    fig, ax = plt.subplots(2, 1, figsize=(3.6, 3.9))
    mk = {2: "o", 3: "s", 5: "^"}
    fc = {"occ": "#2c6fbb", "sdf": "#7f8c8d", "fused": "#c0392b", "field": "#2e8b57"}
    for p in tp:
        ax[0].scatter(p["n_per"], 100 * p["acc"], marker=mk[p["ncls"]], color=fc[p["input"]], s=22, zorder=3)
        ax[0].annotate(p["label"], (p["n_per"], 100 * p["acc"]), fontsize=5.2, xytext=(3, -2), textcoords="offset points")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("training samples per class (smallest class)")
    ax[0].set_ylabel("best val. accuracy (%)")
    ax[0].set_title("(a) Type: o 2, s 3, ^ 5 classes", loc="left")
    ax[0].set_ylim(82, 100.5)
    for p in rp:
        c = COL[p["kind"]]
        m = "D" if p["label"].startswith("pts") else "o"
        ax[1].scatter(p["n"], p["mean"], color=c, marker=m, s=22, zorder=3)
        ax[1].annotate(p["label"], (p["n"], p["mean"]), fontsize=5.2, xytext=(3, 1), textcoords="offset points")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("training samples (one kind)")
    ax[1].set_ylabel("best val. mean error (deg)")
    ax[1].set_title("(b) Rotation: o voxel CNN (box), D point net", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "tdata_learning.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.0))
    ax[0].hist(var["box"]["aspect"], bins=np.logspace(0, np.log10(16), 30), color=COL["box"], alpha=0.7, label="box max/min")
    ax[0].hist(var["cylinder"]["aspect"], bins=np.logspace(-1, np.log10(16), 30), color=COL["cylinder"], alpha=0.7,
               label="cyl. L/D")
    ax[0].set_xscale("log")
    ax[0].set_title("(a) aspect ratio", loc="left")
    ax[0].legend(frameon=False)
    for k in kinds_order:
        ax[1].hist(var[k]["size"], bins=np.linspace(8, 38, 31), histtype="step", color=COL[k], label=k)
    ax[1].set_title("(b) largest extent (mm, 40 mm grid)", loc="left")
    ax[1].legend(frameon=False)
    ax[2].hist(allres, bins=np.logspace(1, 5, 30), color="#555555", alpha=0.8)
    ax[2].set_xscale("log")
    ax[2].set_title("(c) cutter$\\cap$base (voxels)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "tdata_variance.pdf")
    plt.close(fig)

    lines = ["% generated by scripts/make_training_data_slides.py - do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in MACROS.items()]
    (GEN / "tdata_macros.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(MACROS)} macros, {len(recs)} records, gen {gen_wall / CORES / 60:.0f} min on {CORES} cores")
    for k, v in MACROS.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
