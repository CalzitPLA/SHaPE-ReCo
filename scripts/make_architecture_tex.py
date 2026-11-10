#!/usr/bin/env python3
"""
Tables, the backbone drawing and number macros for docs/network_architecture.tex, computed from the model classes
themselves (nothing typed by hand):

    python3 scripts/make_architecture_tex.py       ->  docs/generated/arch_*.tex, docs/data/architecture.json

It instantiates the real networks (training.train_*), pushes one voxel grid through them layer by layer to record
tensor shapes, counts parameters and multiply-accumulates (MACs), derives the receptive field from kernel sizes and
strides, and times a forward pass on this machine (CPU and, if present, GPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from training.topology_schema import MAX_PARAMS, NUM_TOPOLOGIES  # noqa: E402
from training.train_dimension_cnn import DimensionCNN  # noqa: E402
from training.train_rotation_cnn import RotationCNN  # noqa: E402
from training.train_rotation_sym import SymRotationCNN  # noqa: E402
from training.train_type_cnn import TypeCNN  # noqa: E402

GEN = ROOT / "docs" / "generated"
R = 64


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")


def fmt_macs(n: float) -> str:
    return f"{n / 1e9:.2f}\\,G" if n >= 1e9 else f"{n / 1e6:.1f}\\,M" if n >= 1e6 else f"{n / 1e3:.1f}\\,k"


def shape_tex(shape) -> str:
    c, *sp = shape
    return f"${c}\\times {sp[0]}^3$" if len(sp) == 3 else f"${c}$"


# --------------------------------------------------------------------------------------------- backbone rows

def backbone_rows(bb):
    """One row per layer: dict(name, op, out shape, params, macs, rf, jump). Shapes come from a real forward pass."""
    x = torch.zeros(1, 1, R, R, R)
    rows, rf, jump = [], 1, 1
    with torch.no_grad():
        for i, block in enumerate(bb.blocks, start=1):
            conv = block.conv
            k, s = conv.kernel_size[0], conv.stride[0]
            x = conv(x)
            rf += (k - 1) * jump
            jump *= s
            macs = x[0].numel() * conv.in_channels * k ** 3
            rows.append(dict(name=f"conv{i}", op=f"Conv3d ${k}^3$, stride {s}, pad {conv.padding[0]}, no bias",
                             shape=tuple(x.shape[1:]), params=n_params(conv), macs=macs, rf=rf, jump=jump,
                             cin=conv.in_channels, cout=conv.out_channels, k=k, s=s))
            x = block.bn(x)
            rows.append(dict(name=f"bn{i}", op="BatchNorm3d (affine)", shape=tuple(x.shape[1:]), params=n_params(block.bn),
                             macs=0, rf=rf, jump=jump))
            x = block.act(x)
            if isinstance(block.pool, nn.MaxPool3d):
                x = block.pool(x)
                rf += (2 - 1) * jump
                jump *= 2
                rows.append(dict(name=f"pool{i}", op="ReLU, then MaxPool3d $2^3$, stride 2", shape=tuple(x.shape[1:]), params=0,
                                 macs=0, rf=rf, jump=jump))
            else:
                rows[-1]["op"] = "BatchNorm3d (affine), ReLU"
        x = bb.global_pool(x).flatten(1)
        rows.append(dict(name="gap", op="AdaptiveAvgPool3d(1)", shape=tuple(x.shape[1:]), params=0, macs=0, rf=None, jump=None))
        lin = bb.proj[0]
        x = bb.proj(x)
        rows.append(dict(name="proj", op="Linear, ReLU, Dropout(0.3)", shape=tuple(x.shape[1:]), params=n_params(bb.proj),
                         macs=lin.in_features * lin.out_features, rf=None, jump=None))
    return rows


def backbone_table(rows):
    lines = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
             "Layer & Operation & Output & Parameters & MACs & Receptive field & Voxels/step \\\\", "\\midrule"]
    for r in rows:
        rf = f"${r['rf']}^3$" if r["rf"] else "--"
        jump = f"{r['jump']}" if r["jump"] else "--"
        macs = fmt_macs(r["macs"]) if r["macs"] else "--"
        lines.append(f"\\texttt{{{r['name']}}} & {r['op']} & {shape_tex(r['shape'])} & {fmt_int(r['params']) if r['params'] else '--'} & {macs} & {rf} & {jump} \\\\")
    lines += ["\\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{Total (backbone)}} & {fmt_int(sum(r['params'] for r in rows))} & "
              f"{fmt_macs(sum(r['macs'] for r in rows))} & & \\\\", "\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


def backbone_tikz(rows):
    """Left-to-right chain of the backbone; the output shape of each stage is written under its box."""
    conv = [r for r in rows if r["name"].startswith("conv")]
    pool = {r["name"][-1]: r for r in rows if r["name"].startswith("pool")}
    gap = [r for r in rows if r["name"] == "gap"][0]
    proj = [r for r in rows if r["name"] == "proj"][0]
    out = ["\\begin{tikzpicture}[x=1cm,y=1cm,",
           "  blk/.style={draw,rounded corners=2pt,align=center,inner sep=3pt,font=\\scriptsize,minimum height=1.25cm,minimum width=1.9cm},",
           "  io/.style={draw,rounded corners=2pt,align=center,inner sep=3pt,font=\\scriptsize,minimum height=1.25cm,fill=black!6},",
           "  ar/.style={-{Stealth[length=1.6mm,width=1.2mm]},line width=0.5pt},",
           "  sh/.style={font=\\scriptsize,text=black!70,align=center}]"]
    stages = [("in", "voxel grid\\\\$\\{-1,+1\\}$", "io", f"$1\\times {R}^3$")]
    for i, c in enumerate(conv, start=1):
        label = f"Conv ${c['k']}^3$" + (f", $s{c['s']}$" if c["s"] > 1 else "") + "\\\\BN, ReLU" + ("\\\\MaxPool $2$" if str(i) in pool else "")
        label += f"\\\\${c['cin']}\\!\\to\\!{c['cout']}$"
        stages.append((f"b{i}", label, "blk", shape_tex(pool[str(i)]["shape"] if str(i) in pool else c["shape"])))
    stages.append(("gap", "global\\\\average pool", "blk", shape_tex(gap["shape"])))
    stages.append(("proj", "Linear, ReLU\\\\Dropout $0.3$", "blk", shape_tex(proj["shape"])))
    stages.append(("feat", "feature\\\\vector", "io", ""))
    x, prev = 0.0, None
    for name, label, style, shape in stages:
        out.append(f"\\node[{style}] ({name}) at ({x:.2f},0) {{{label}}};")
        if shape:
            out.append(f"\\node[sh,below=1pt of {name}] {{{shape}}};")
        if prev:
            out.append(f"\\draw[ar] ({prev}) -- ({name});")
        prev, x = name, x + 2.45
    out.append("\\end{tikzpicture}")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------------- heads

def head_rows(name, model):
    """Layer list of a head (everything except the backbone) with output sizes."""
    rows = []
    if name == "type":
        lin = model.classifier
        rows.append(("classifier", f"Linear ${lin.in_features}\\to {lin.out_features}$", lin.out_features, n_params(lin), lin.in_features * lin.out_features))
    elif name in ("rotation", "dimension"):
        emb = model.topo_embed
        rows.append(("topo\\_embed", f"Embedding ${emb.num_embeddings}\\to {emb.embedding_dim}$ (topology index)", emb.embedding_dim, n_params(emb), 0))
        seq = model.head
        width = model.backbone.feature_dim + emb.embedding_dim
        rows.append(("concat", f"features $\\oplus$ embedding", width, 0, 0))
        for idx, layer in enumerate(seq):
            if isinstance(layer, nn.Linear):
                rows.append((f"head.{idx}", f"Linear ${layer.in_features}\\to {layer.out_features}$", layer.out_features, n_params(layer), layer.in_features * layer.out_features))
            elif isinstance(layer, nn.ReLU):
                rows.append((f"head.{idx}", "ReLU", None, 0, 0))
            elif isinstance(layer, nn.Dropout):
                rows.append((f"head.{idx}", f"Dropout({layer.p})", None, 0, 0))
            elif isinstance(layer, nn.Tanh):
                rows.append((f"head.{idx}", "Tanh (outputs in $[-1,1]$)", None, 0, 0))
        rows.append(("output", "quaternion, normalised to unit length" if name == "rotation" else "normalised parameters, $10$ slots", None, 0, 0))
    else:
        seq = model.head
        for idx, layer in enumerate(seq):
            if isinstance(layer, nn.Linear):
                rows.append((f"head.{idx}", f"Linear ${layer.in_features}\\to {layer.out_features}$", layer.out_features, n_params(layer), layer.in_features * layer.out_features))
            elif isinstance(layer, nn.ReLU):
                rows.append((f"head.{idx}", "ReLU", None, 0, 0))
        rows.append(("output", "6D $\\to$ Gram--Schmidt $\\to$ rotation matrix $3\\times3$" if model.rep == "6d" else "quaternion $\\to$ rotation matrix $3\\times3$", None, 0, 0))
    return rows


def heads_table(models):
    lines = ["\\begin{tabular}{@{}lllrr@{}}", "\\toprule", "Network & Layer & Operation & Parameters & MACs \\\\", "\\midrule"]
    first = True
    for key, title in (("type", "Type CNN"), ("rotation", "Rotation CNN (raw quaternion)"), ("dimension", "Dimension CNN"), ("sym", "Symmetry-aware Rotation CNN")):
        if not first:
            lines.append("\\addlinespace[3pt]")
        first = False
        rows = head_rows(key, models[key])
        for i, (lname, op, _, p, macs) in enumerate(rows):
            title_cell = f"\\textbf{{{title}}}" if i == 0 else ""
            lines.append(f"{title_cell} & \\texttt{{{lname}}} & {op} & {fmt_int(p) if p else '--'} & {fmt_macs(macs) if macs else '--'} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ summary

def timing(model, args, device, reps=30):
    model = model.to(device).eval()
    args = [a.to(device) for a in args]
    with torch.no_grad():
        for _ in range(5):
            model(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(*args)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def cost_figure(rows):
    """Where the parameters and the compute are: share per layer, two panels (one axis each)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#6b6a66", "axes.labelcolor": "#1a1a19", "xtick.color": "#6b6a66",
                         "ytick.color": "#6b6a66", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
                         "axes.axisbelow": True})
    layers = [r for r in rows if r["name"].startswith("conv") or r["name"] == "proj"]
    names = [r["name"] for r in layers]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.0), sharey=True)
    for ax, key, title in ((axes[0], "params", "parameters"), (axes[1], "macs", "multiply-accumulates")):
        vals = np.array([r[key] for r in layers], dtype=float)
        share = 100 * vals / vals.sum()
        y = np.arange(len(layers))[::-1]
        ax.barh(y, share, color="#2a78d6", height=0.6)
        for yi, sh in zip(y, share):
            ax.text(sh + 1.5, yi, f"{sh:.1f} %" if sh >= 0.1 else "<0.1 %", va="center", fontsize=7)
        ax.set_yticks(y)
        ax.set_yticklabels(names, family="monospace", fontsize=7)
        ax.set_xlim(0, 105)
        ax.set_xlabel(f"share of the backbone's {title} (%)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "figures" / "arch_cost.pdf")
    plt.close(fig)


def main():
    torch.manual_seed(0)
    models = {"type": TypeCNN(), "rotation": RotationCNN(), "dimension": DimensionCNN(), "sym": SymRotationCNN("6d")}
    bb = models["type"].backbone
    rows = backbone_rows(bb)
    bb_params, bb_macs = sum(r["params"] for r in rows), sum(r["macs"] for r in rows)
    conv_params = sum(r["params"] for r in rows if r["name"].startswith("conv"))
    conv2_share = [r for r in rows if r["name"] == "conv2"][0]["macs"] / bb_macs
    conv4_params = [r for r in rows if r["name"] == "conv4"][0]["params"]
    last_rf = [r for r in rows if r["name"] == "conv4"][0]["rf"]
    final_rf = [r for r in rows if r["name"] == "pool4"] or [r for r in rows if r["name"] == "pool3"]
    final_rf = final_rf[0]["rf"]
    final_map = [r for r in rows if r["name"] == "pool3"][0]["shape"] if False else [r for r in rows if r["name"].startswith("pool")][-1]["shape"]

    info = {}
    x1 = torch.zeros(1, 1, R, R, R)
    topo = torch.zeros(1, dtype=torch.long)
    x16, topo16 = torch.zeros(16, 1, R, R, R), torch.zeros(16, dtype=torch.long)
    args1 = {"type": (x1,), "rotation": (x1, topo), "dimension": (x1, topo), "sym": (x1,)}
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    for key, m in models.items():
        head_macs = sum(r[4] for r in head_rows(key, m))
        info[key] = dict(params=n_params(m), head_params=n_params(m) - bb_params, macs=bb_macs + head_macs,
                         state_dict_bytes=sum(t.numel() * t.element_size() for t in m.state_dict().values()))
        for d in devices:
            info[key][f"ms_{d.type}"] = timing(m, args1[key], d)
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU"
    # batch-16 training-shaped forward on the GPU, for the cost of one step's forward pass
    if torch.cuda.is_available():
        info["type"]["ms_cuda_b16"] = timing(models["type"], (x16,), torch.device("cuda"), reps=10)
    info["backbone"] = dict(params=bb_params, macs=bb_macs, conv2_mac_share=conv2_share, conv_param_share=conv_params / bb_params,
                            receptive_field_last_conv=last_rf, receptive_field_final=final_rf, final_map=list(final_map),
                            device=cuda_name, torch=torch.__version__)
    (ROOT / "docs" / "data" / "architecture.json").write_text(json.dumps(info, indent=1, default=float))

    GEN.mkdir(exist_ok=True)
    cost_figure(rows)
    (GEN / "arch_backbone_table.tex").write_text(backbone_table(rows))
    (GEN / "arch_backbone_fig.tex").write_text(backbone_tikz(rows))
    (GEN / "arch_heads_table.tex").write_text(heads_table(models))

    # summary table
    lines = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
             "Network & Parameters & thereof head & MACs (1 sample) & CPU ms & GPU ms \\\\", "\\midrule"]
    for key, title in (("type", "Type CNN"), ("rotation", "Rotation CNN (raw)"), ("dimension", "Dimension CNN"), ("sym", "Symmetry-aware Rotation CNN")):
        i = info[key]
        gpu = f"{i['ms_cuda']:.1f}" if "ms_cuda" in i else "--"
        lines.append(f"{title} & {fmt_int(i['params'])} & {fmt_int(i['head_params'])} & {fmt_macs(i['macs'])} & {i['ms_cpu']:.0f} & {gpu} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GEN / "arch_summary_table.tex").write_text("\n".join(lines))

    macros = {
        "archBackboneParams": fmt_int(bb_params), "archBackboneMacs": fmt_macs(bb_macs),
        "archConvParamShare": f"{100 * conv_params / bb_params:.0f}", "archConvTwoMacShare": f"{100 * conv2_share:.0f}",
        "archConvFourParams": fmt_int(conv4_params), "archConvFourShare": f"{100 * conv4_params / bb_params:.0f}",
        "archRFLastConv": last_rf, "archRFFinal": final_rf, "archFinalMap": f"{final_map[0]}\\times {final_map[1]}^3",
        "archNumTopologies": NUM_TOPOLOGIES, "archLastTopology": NUM_TOPOLOGIES - 1, "archMaxParams": MAX_PARAMS,
        "archTypeParams": fmt_int(info["type"]["params"]), "archRotParams": fmt_int(info["rotation"]["params"]),
        "archDimParams": fmt_int(info["dimension"]["params"]), "archSymParams": fmt_int(info["sym"]["params"]),
        "archTypeMB": f"{info['type']['state_dict_bytes'] / 1e6:.1f}",
        "archGpuName": cuda_name.replace("_", "\\_"), "archTorch": torch.__version__.split("+")[0],
        "archTypeMsCpu": f"{info['type']['ms_cpu']:.0f}", "archTypeMsGpu": f"{info['type'].get('ms_cuda', float('nan')):.1f}",
        "archTypeMsGpuBatch": f"{info['type'].get('ms_cuda_b16', float('nan')):.0f}",
    }
    (GEN / "arch_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
    print(json.dumps({k: {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()} for k, v in info.items()}, indent=1))


if __name__ == "__main__":
    main()
