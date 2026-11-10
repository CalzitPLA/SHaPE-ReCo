#!/usr/bin/env python3
"""
Statistics and figures of the frame report (docs/reports/frame_solution): the input, the framework's model, the model
with its fillets and the hand-made GDML model, compared by volume, surface, deviation and cost of a point query.

    python3 scripts/frame_report.py <work dir with the runs> <out dir>

The work directory is what scripts/frame_report_fillets.py and the reproduction run wrote (frame_run.json,
exact_residual.json, fc/, fillets/, handmade_*). Everything here is measured on those files; nothing is typed in.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

BLUE, ORANGE, GREY, INK, MUTED = "#2A78D6", "#EB6834", "#8A8A8A", "#1F1F1F", "#6B6B6B"   # validated pair (dataviz): blue/orange
STEP_IN = "~/shape_reco_parts/benchmark_parts/1.0/Frame_4_x_6.stp"


def mesh_of(path):
    import trimesh
    from src.closest_point_field import load_mesh
    if str(path).lower().endswith((".stp", ".step")):
        v, f = load_mesh(path, 0.02)
        return trimesh.Trimesh(v, f, process=False)
    return trimesh.load(str(path), force="mesh", process=True)


def surface_samples(mesh, n, seed):
    import trimesh
    p, face = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(p), np.asarray(face)


def distance_to(mesh, points):
    import igl
    d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                                               np.ascontiguousarray(mesh.faces, dtype=np.int64))
    return np.sqrt(np.maximum(d2, 0.0))


def inside(mesh, points):
    import igl
    w = igl.fast_winding_number(np.ascontiguousarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(mesh.faces, dtype=np.int64),
                                np.ascontiguousarray(points, dtype=np.float64))
    return np.asarray(w) > 0.5


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class BoxUnion:
    """The model as it is: 40 oriented boxes. Inside = inside any of them."""

    def __init__(self, ops):
        self.c = np.array([o["center_mm"] for o in ops])
        self.h = np.array([[o["dims_mm"]["x_half"], o["dims_mm"]["y_half"], o["dims_mm"]["z_half"]] for o in ops])
        self.r = np.array([quat_matrix(o["quaternion"]) for o in ops])

    def inside(self, p):
        out = np.zeros(len(p), bool)
        for c, h, r in zip(self.c, self.h, self.r):
            out |= (np.abs((p - c) @ r) <= h).all(axis=1)
        return out


class FilletTools:
    """Box minus cylinder, axis-parallel (scripts/frame_report_fillets.fillet_tools)."""

    def __init__(self, tools):
        self.lo = np.array([t["corner"] for t in tools])
        self.hi = self.lo + np.array([t["size"] for t in tools])
        self.k = np.array([int(np.argmax(t["axis"])) for t in tools])
        self.c = np.array([t["start"] for t in tools])
        self.r = np.array([t["radius"] for t in tools])

    def inside(self, p):
        out = np.zeros(len(p), bool)
        for lo, hi, k, c, r in zip(self.lo, self.hi, self.k, self.c, self.r):
            box = ((p >= lo) & (p <= hi)).all(axis=1)
            if box.any():
                q = p[box] - c
                q[:, k] = 0.0
                out[np.nonzero(box)[0]] |= (q * q).sum(axis=1) > r * r
        return out


def timed(fn, *a, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t = time.perf_counter()
        r = fn(*a)
        best = min(best, time.perf_counter() - t)
    return r, best


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    run = json.loads((work / "frame_run.json").read_text())
    res = json.loads((work / "exact_residual.json").read_text())
    fil = json.loads((work / "fillets/fillets.json").read_text())
    job = json.loads((work / "fillets/job.json").read_text())
    hand = json.loads((work / "handmade_residual.json").read_text())
    ops = [o for o in run["operations"] if o["accepted"]]

    meshes = dict(input=mesh_of(STEP_IN), model=mesh_of(work / "fc/Frame_4_x_6.stl"), fillets=mesh_of(work / "fillets/frame_with_fillets.stl"),
                  handmade=mesh_of(work / "handmade_aligned.stl"))
    data = dict(date=time.strftime("%Y-%m-%d"), geometry={}, deviation={}, sampled={}, query={}, pieces=[], run={})
    for k, m in meshes.items():
        data["geometry"][k] = dict(triangles=int(len(m.faces)), vertices=int(len(m.vertices)), area=float(m.area), mesh_volume=float(abs(m.volume)),
                                   extents=[float(x) for x in m.extents], watertight=bool(m.is_watertight),
                                   euler=int(m.euler_number), bounds=np.asarray(m.bounds).tolist())
    data["geometry"]["input"].update(volume=res["input_volume"], genus=int(run["grade"]["input_genus"]))
    data["geometry"]["model"].update(volume=res["model_volume"], genus=res["model_genus"], exact_iou=res["iou"], missing=res["r_plus"], invented=res["r_minus"],
                                     missing_pieces=res["r_plus_solids"], primitives=len(ops), brep_faces=fil["build"]["model_faces"])
    data["geometry"]["fillets"].update(volume=fil["after"]["model_volume"], genus=fil["after"]["model_genus"], exact_iou=fil["after"]["iou"],
                                       missing=fil["after"]["r_plus"], invented=fil["after"]["r_minus"], primitives=len(ops) + fil["tools"],
                                       tools=fil["tools"], brep_faces=fil["build"]["faces"], build_seconds=fil["build_seconds"], valid=fil["build"]["valid"],
                                       solids=fil["build"]["solids"])
    data["geometry"]["handmade"].update(volume=hand["model_volume"], genus=hand["model_genus"], exact_iou=hand["iou"], missing=hand["r_plus"],
                                        invented=hand["r_minus"], primitives=40)

    # surface deviation, both directions, 200000 area-weighted samples each
    n = 200000
    p_in, _ = surface_samples(meshes["input"], n, 0)
    for k in ("model", "fillets", "handmade"):
        d_in = distance_to(meshes[k], p_in)                                   # input surface -> model
        p_m, _ = surface_samples(meshes[k], n, 1)
        d_m = distance_to(meshes["input"], p_m)                               # model surface -> input
        both = np.concatenate([d_in, d_m])
        data["deviation"][k] = dict(samples=n, chamfer_mean=float(0.5 * (d_in.mean() + d_m.mean())), hausdorff=float(both.max()),
                                    input_to_model=dict(mean=float(d_in.mean()), median=float(np.median(d_in)), p95=float(np.percentile(d_in, 95)),
                                                        p99=float(np.percentile(d_in, 99)), max=float(d_in.max()),
                                                        share_within_0_01=float((d_in <= 0.01).mean()), share_within_1=float((d_in <= 1.0).mean())),
                                    model_to_input=dict(mean=float(d_m.mean()), median=float(np.median(d_m)), p95=float(np.percentile(d_m, 95)),
                                                        p99=float(np.percentile(d_m, 99)), max=float(d_m.max()),
                                                        share_within_0_01=float((d_m <= 0.01).mean()), share_within_1=float((d_m <= 1.0).mean())))
        np.save(out / f"dev_input_to_{k}.npy", d_in.astype(np.float32))

    # sampled volumes: an independent check of the exact numbers, 4e6 points in the bounding box
    lo, hi = np.asarray(meshes["input"].bounds)
    rng = np.random.default_rng(0)
    pts = lo - 5.0 + rng.random((4_000_000, 3)) * (hi - lo + 10.0)
    cell = float(np.prod(hi - lo + 10.0)) / len(pts)
    ins = {k: inside(m, pts) for k, m in meshes.items()}
    for k in ("model", "fillets", "handmade"):
        a, b = ins["input"], ins[k]
        data["sampled"][k] = dict(points=len(pts), iou=float((a & b).sum() / max((a | b).sum(), 1)), missing=float((a & ~b).sum() * cell),
                                  invented=float((b & ~a).sum() * cell), input_volume=float(a.sum() * cell))

    # the cost of "is this point in the solid", the question a particle-transport or Monte-Carlo code asks most
    q = pts[:1_000_000]
    boxes, tools = BoxUnion(ops), FilletTools(job["tools"])
    r_box, t_box = timed(boxes.inside, q)
    r_tool, t_tool = timed(lambda x: boxes.inside(x) | tools.inside(x), q)
    data["query"]["points"] = len(q)
    data["query"]["csg_boxes"] = dict(primitives=len(ops), seconds=t_box, agree_with_mesh=float((r_box == ins["model"][:len(q)]).mean()))
    data["query"]["csg_boxes_fillets"] = dict(primitives=len(ops) + len(job["tools"]), seconds=t_tool,
                                              agree_with_mesh=float((r_tool == ins["fillets"][:len(q)]).mean()),
                                              agree_with_input=float((r_tool == ins["input"][:len(q)]).mean()))
    for k in ("input", "model", "fillets"):
        _, t = timed(inside, meshes[k], q)
        data["query"][f"mesh_{k}"] = dict(triangles=int(len(meshes[k].faces)), seconds=t)

    # the missing pieces, grouped
    groups = {}
    for p in res["pieces"]:
        ext = tuple(np.round(np.sort(np.asarray(p["bbox"][1]) - np.asarray(p["bbox"][0])), 0))
        g = groups.setdefault(ext, dict(extent=list(ext), count=0, volume=0.0, fillet_faces=0))
        g["count"] += 1
        g["volume"] += p["volume"]
        g["fillet_faces"] += sum(f["type"] == "cylinder" and not f.get("convex") for f in p["faces"])
    data["pieces"] = sorted(groups.values(), key=lambda g: -g["volume"])
    data["run"] = dict(seconds=run["seconds"], peak_rss_gb=run["peak_rss_gb"], chamfer_loop=run["grade"]["chamfer"], chamfer_max_loop=run["grade"]["chamfer_max"],
                       sampled_iou_loop=run["grade"]["sampled_iou"], members=sum(s["step"] == "member" for s in run["steps"]),
                       input_voxels=run["done"]["input_voxels"], stopped=run["done"]["stopped"],
                       phases={s["step"]: s["t"] for s in run["steps"] if s["step"] in ("input", "residual", "grade", "done")},
                       first_member=min(s["t"] for s in run["steps"] if s["step"] == "member"),
                       operations=[dict(label=o["label"], size=sorted(round(2 * v, 1) for v in o["dims_mm"].values()), mode=o["sources"][0]["mode"]) for o in ops])
    (out / "frame_report.json").write_text(json.dumps(data, indent=1))
    figures(data, meshes, res, job, out)
    for f in out.glob("dev_input_to_*.npy"):                        # only the figure needs them
        f.unlink()
    print(json.dumps({k: data[k] for k in ("sampled", "query", "pieces")}, indent=1)[:3000])
    print(json.dumps(data["deviation"], indent=1)[:2500])


def figures(data, meshes, res, job, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})

    # 1. the two solids, same view
    fig = plt.figure(figsize=(7.2, 3.3))
    for i, (k, title, colour) in enumerate((("input", "Input (STEP, 442 faces)", BLUE), ("model", "Model (40 boxes)", ORANGE))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        m = meshes[k]
        tri = m.vertices[m.faces]
        nrm = m.face_normals
        light = np.array([0.4, -0.5, 0.75]); light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        base = np.array(matplotlib.colors.to_rgb(colour))
        ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(0.35 + 0.65 * shade[:, None] * (0.55 + 0.45 * base), 0, 1), edgecolors="none"))
        lo, hi = m.bounds
        c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
        ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-58); ax.set_axis_off()
        ax.text2D(0.5, 0.93, title, transform=ax.transAxes, ha="center", fontsize=8.5, color=INK)
    fig.subplots_adjust(left=0, right=1, top=1.0, bottom=0, wspace=-0.05)
    fig.savefig(out / "fig_solids.png", dpi=260)
    plt.close(fig)

    # 2. a section through a chord: outlines of input and model, and the corner in detail
    lo, hi = meshes["input"].bounds
    lateral = [t for t in job["tools"] if t["axis"] == [0.0, 1.0, 0.0] and t["length"] > 1500]
    y0 = lateral[0]["corner"][1] + 0.37 * lateral[0]["size"][1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw=dict(width_ratios=[1.15, 1]))
    paths = {}
    for k in ("input", "model"):
        sec = meshes[k].section(plane_origin=[0, y0, 0], plane_normal=[0, 1, 0])
        paths[k] = [np.asarray(sec.vertices[e.points])[:, [0, 2]] for e in sec.entities]
    chord = lateral[0]
    cx, cz = chord["start"][0], chord["start"][2]
    near = [p for p in paths["input"] if np.abs(p[:, 0] - cx).min() < 400 and np.abs(p[:, 1] - cz).min() < 600]
    xs = np.concatenate([p[:, 0] for p in near]); zs = np.concatenate([p[:, 1] for p in near])
    box = (xs.min() - 25, xs.max() + 25, zs.min() - 25, zs.max() + 25)
    for ax, zoom in zip(axes, (False, True)):
        for k, colour, lw, ls in (("input", BLUE, 1.6, "-"), ("model", ORANGE, 1.1, "-")):
            for p in paths[k]:
                ax.plot(p[:, 0], p[:, 1], color=colour, lw=lw, ls=ls, solid_capstyle="round")
        ax.set_aspect("equal")
        if zoom:
            ax.set_xlim(cx - 62, cx + 62); ax.set_ylim(cz - 62, cz + 62)
            ax.set_title("Web-flange corner: the root fillet, r = 27 mm", fontsize=8, color=INK, loc="left")
        else:
            ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
            ax.set_title(f"Section of one chord at y = {y0:.0f} mm", fontsize=8, color=INK, loc="left")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
        ax.grid(color="#E6E6E6", lw=0.4)
    axes[0].plot([], [], color=BLUE, lw=1.6, label="input"); axes[0].plot([], [], color=ORANGE, lw=1.1, label="model")
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_section.pdf"); fig.savefig(out / "fig_section.png", dpi=220)
    plt.close(fig)

    # 3. how far the input's surface is from the model's: share of the surface within a distance
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    d = np.sort(np.load(out / "dev_input_to_model.npy"))
    share = np.arange(1, len(d) + 1) / len(d)
    ax.plot(np.maximum(d, 1e-3), share * 100, color=BLUE, lw=2)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 20); ax.set_ylim(80, 100.5)
    ax.set_xlabel("distance from the input's surface to the model (mm)"); ax.set_ylabel("share of the input's surface (%)")
    on = float((d <= 0.01).mean() * 100)
    ax.annotate(f"{on:.1f} % of the surface lies on the model\n(within 0.01 mm)", xy=(0.01, on), xytext=(0.02, 84.5), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate(f"largest distance {d.max():.2f} mm:\nthe fillet's surface at 45 degrees", xy=(d.max(), 100), xytext=(0.12, 97.2), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.grid(color="#E6E6E6", lw=0.4)
    fig.tight_layout()
    fig.savefig(out / "fig_deviation.pdf"); fig.savefig(out / "fig_deviation.png", dpi=220)
    plt.close(fig)

    # 4. where the missing volume is
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    g = data["pieces"]
    labels = [f"{x['count']} x  {x['extent'][0]:.0f} x {x['extent'][1]:.0f} x {x['extent'][2]:.0f} mm" for x in g]
    vol = np.array([x["volume"] for x in g]) / 1e6
    y = np.arange(len(g))[::-1]
    ax.barh(y, vol, height=0.62, color=ORANGE)
    for yi, v, x in zip(y, vol, g):
        ax.text(v + 0.15, yi, f"{v:.2f}  ({x['fillet_faces']} fillet faces)", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, vol.max() * 1.45)
    ax.set_xlabel("missing volume (10$^6$ mm$^3$)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fig_pieces.pdf"); fig.savefig(out / "fig_pieces.png", dpi=220)
    plt.close(fig)

    # 5. the cost of a point query
    fig, ax = plt.subplots(figsize=(5.6, 2.3))
    qd = data["query"]
    rows = [(f"model as CSG: {qd['csg_boxes']['primitives']} boxes", qd["csg_boxes"]["seconds"], ORANGE),
            (f"model with fillets as CSG: {qd['csg_boxes_fillets']['primitives']} solids", qd["csg_boxes_fillets"]["seconds"], ORANGE),
            (f"model as a mesh: {qd['mesh_model']['triangles']} triangles", qd["mesh_model"]["seconds"], BLUE),
            (f"input as a mesh: {qd['mesh_input']['triangles']} triangles", qd["mesh_input"]["seconds"], BLUE)]
    y = np.arange(len(rows))[::-1]
    us = [r[1] / qd["points"] * 1e6 for r in rows]
    ax.barh(y, us, height=0.6, color=[r[2] for r in rows])
    for yi, v in zip(y, us):
        ax.text(v + max(us) * 0.015, yi, f"{v:.2f}", va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlim(0, max(us) * 1.15)
    ax.set_xlabel("time per point query (microseconds)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "fig_query.pdf"); fig.savefig(out / "fig_query.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
