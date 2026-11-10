#!/usr/bin/env python3
"""
The numbers of docs/reports/frame_solution/frame_solution.tex as LaTeX macros (numbers.tex), read from the measurement
files, so that no number in the report is typed by hand.

    python3 scripts/frame_report_tex.py <work dir> docs/reports/frame_solution
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

STEP_IN = "~/shape_reco_parts/benchmark_parts/1.0/Frame_4_x_6.stp"


def sci(x, digits=2):
    if x == 0:
        return "0"
    e = int(f"{x:e}".split("e")[1])
    return f"{x / 10 ** e:.{digits}f}\\times 10^{{{e}}}"


def main():
    work, out = Path(sys.argv[1]), Path(sys.argv[2])
    d = json.loads((out / "frame_report.json").read_text())
    fil = json.loads((work / "fillets/fillets.json").read_text())
    align = json.loads((work / "handmade_align.json").read_text())
    faces_file = work / "input_faces.json"
    if not faces_file.exists():
        from reco_field.exact_check import exact_faces
        faces_file.write_text(json.dumps(exact_faces(STEP_IN), default=str))
    faces = json.loads(faces_file.read_text())["faces"]
    kinds = collections.Counter(f["type"] for f in faces)
    fillet_faces = [f for f in faces if f["type"] == "cylinder" and not f.get("convex")]
    g, dev, q, run, smp = d["geometry"], d["deviation"], d["query"], d["run"], d["sampled"]
    m = {}
    m["InVolume"] = sci(g["input"]["volume"], 4); m["InArea"] = sci(g["input"]["area"], 3); m["InFaces"] = len(faces)
    m["InPlanes"] = kinds.get("plane", 0); m["InCylinders"] = kinds.get("cylinder", 0); m["InFilletFaces"] = len(fillet_faces)
    m["InOtherFaces"] = len(faces) - kinds.get("plane", 0) - kinds.get("cylinder", 0)
    m["InFilletRadius"] = f"{min(f['radius'] for f in fillet_faces):.0f}" if fillet_faces else "-"
    m["InFilletRadiusMax"] = f"{max(f['radius'] for f in fillet_faces):.0f}" if fillet_faces else "-"
    m["InFilletArea"] = sci(sum(f["area"] for f in fillet_faces), 2)
    m["InFilletAreaShare"] = f"{100 * sum(f['area'] for f in fillet_faces) / g['input']['area']:.1f}"
    m["InTriangles"] = f"{g['input']['triangles']:,}".replace(",", "\\,")
    m["InGenus"] = g["input"]["genus"]
    m["ExtX"], m["ExtY"], m["ExtZ"] = (f"{x:.0f}" for x in g["input"]["extents"])
    for key, name in (("model", "Mo"), ("fillets", "Fi"), ("handmade", "Ha")):
        x = g[key]
        m[name + "Volume"] = sci(x["volume"], 4); m[name + "Area"] = sci(x["area"], 3); m[name + "Genus"] = x["genus"]
        m[name + "IoU"] = f"{x['exact_iou']:.4f}"; m[name + "Missing"] = sci(x["missing"], 2) if x["missing"] > 1 else "0"
        m[name + "Invented"] = sci(x["invented"], 2) if x["invented"] > 1 else "0"
        m[name + "MissingShare"] = f"{100 * x['missing'] / g['input']['volume']:.2f}"
        m[name + "InventedShare"] = f"{100 * x['invented'] / g['input']['volume']:.2f}"
        m[name + "Triangles"] = f"{x['triangles']:,}".replace(",", "\\,"); m[name + "Primitives"] = x["primitives"]
        m[name + "AreaDiff"] = f"{100 * (x['area'] - g['input']['area']) / g['input']['area']:+.2f}"
        m[name + "SampledIoU"] = f"{smp[key]['iou']:.5f}" if smp[key]["iou"] > 0.9999 else f"{smp[key]['iou']:.4f}"
        y = dev[key]
        m[name + "Chamfer"] = f"{y['chamfer_mean']:.3f}" if y["chamfer_mean"] > 0.001 else f"{y['chamfer_mean']:.4f}"
        m[name + "Hausdorff"] = f"{y['hausdorff']:.2f}"
        for way, w in (("input_to_model", "IM"), ("model_to_input", "MI")):
            z = y[way]
            m[name + w + "Mean"] = f"{z['mean']:.3f}" if z["mean"] > 0.001 else f"{z['mean']:.4f}"
            m[name + w + "PNinetyFive"] = f"{z['p95']:.2f}" if z["p95"] > 0.01 else f"{z['p95']:.4f}"
            m[name + w + "PNinetyNine"] = f"{z['p99']:.2f}" if z["p99"] > 0.01 else f"{z['p99']:.4f}"
            m[name + w + "Max"] = f"{z['max']:.2f}" if z["max"] > 0.01 else f"{z['max']:.4f}"
            m[name + w + "On"] = f"{100 * z['share_within_0_01']:.1f}"
            m[name + w + "WithinOne"] = f"{100 * z['share_within_1']:.1f}"
    m["MoFaces"] = g["model"]["brep_faces"]; m["FiFaces"] = g["fillets"]["brep_faces"]; m["FiTools"] = g["fillets"]["tools"]
    m["FiBuildSeconds"] = f"{g['fillets']['build_seconds']:.1f}"; m["FiJudgeSeconds"] = f"{fil['judge_seconds']:.1f}"
    m["FiToolsVolume"] = sci(fil["build"]["tools_volume"], 3); m["FiVolumeDiff"] = f"{abs(fil['build']['volume'] - g['input']['volume']):.1f}"
    m["FiSolids"] = g["fillets"]["solids"]; m["MoMissingPieces"] = g["model"]["missing_pieces"]
    m["MoVolumeShare"] = f"{100 * g['model']['volume'] / g['input']['volume']:.2f}"
    m["HaAlign"] = align["best"]; m["HaAlignLow"] = f"{min(v['iou'] for v in align.values() if isinstance(v, dict) and 'iou' in v):.4f}"
    m["HaSizeX"], m["HaSizeY"], m["HaSizeZ"] = (f"{x:g}" for x in align["handmade_size"])
    m["RunSeconds"] = f"{run['seconds']:.0f}"; m["RunPeak"] = f"{run['peak_rss_gb']:.1f}"; m["RunMembers"] = run["members"]
    m["RunInputSeconds"] = f"{run['phases']['input']:.0f}"; m["RunMembersSeconds"] = f"{run['phases']['residual'] - run['phases']['input']:.0f}"
    m["RunGradeSeconds"] = f"{run['phases']['grade'] - run['phases']['residual']:.0f}"; m["RunVoxels"] = run["input_voxels"]
    m["RunChamfer"] = f"{run['chamfer_loop']:.3f}"; m["RunChamferMax"] = f"{run['chamfer_max_loop']:.2f}"; m["RunSampledIoU"] = f"{run['sampled_iou_loop']:.4f}"
    n = q["points"]
    for key, name in (("csg_boxes", "QBoxes"), ("csg_boxes_fillets", "QTools"), ("mesh_model", "QMeshModel"), ("mesh_input", "QMeshInput"), ("mesh_fillets", "QMeshFillets")):
        m[name] = f"{q[key]['seconds'] / n * 1e6:.2f}"
    m["QToolsOverBoxes"] = f"{q['csg_boxes_fillets']['seconds'] / q['csg_boxes']['seconds']:.1f}"
    m["QMeshInputOverModel"] = f"{q['mesh_input']['seconds'] / q['mesh_model']['seconds']:.1f}"
    m["QMeshFilletsOverModel"] = f"{q['mesh_fillets']['seconds'] / q['mesh_model']['seconds']:.1f}"
    m["QAgreeBoxes"] = f"{100 * q['csg_boxes']['agree_with_mesh']:.4f}"; m["QAgreeTools"] = f"{100 * q['csg_boxes_fillets']['agree_with_input']:.4f}"
    m["QPoints"] = f"{n:,}".replace(",", "\\,"); m["SampledPoints"] = f"{smp['model']['points']:,}".replace(",", "\\,")
    m["TriFilletsOverModel"] = f"{g['fillets']['triangles'] / g['model']['triangles']:.0f}"
    m["TriInputOverModel"] = f"{g['input']['triangles'] / g['model']['triangles']:.1f}"
    m["Date"] = d["date"]
    rows = []
    for p in d["pieces"]:
        e = p["extent"]
        rows.append(f"{p['count']} & ${e[0]:.0f}\\times{e[1]:.0f}\\times{e[2]:.0f}$ & {p['fillet_faces']} & ${sci(p['volume'], 2)}$ & "
                    f"{100 * p['volume'] / g['model']['missing']:.1f} \\\\")
    ops = collections.OrderedDict()
    for o in run["operations"]:
        key = (o["label"], tuple(o["size"]), o["mode"])
        ops[key] = ops.get(key, 0) + 1
    oprows = [f"{c} & {k[0].replace('~', '$\\sim$').replace('_', ' ')} & ${k[1][0]:g}\\times{k[1][1]:g}\\times{k[1][2]:.0f}$ & "
              f"${c * k[1][0] * k[1][1] * k[1][2]:.3e}$".replace("e+0", "\\times 10^{").replace("$ &", "$ &") for k, c in ops.items()]
    oprows = []
    for k, c in ops.items():
        vol = c * k[1][0] * k[1][1] * k[1][2]
        oprows.append(f"{c} & {k[0].replace('~', '$\\sim$')} & ${k[1][0]:g}\\times{k[1][1]:g}\\times{k[1][2]:.0f}$ & ${sci(vol, 2)}$ \\\\")
    lines = ["% generated by scripts/frame_report_tex.py from frame_report.json - do not edit"]
    lines += [f"\\newcommand{{\\F{k}}}{{{v}}}" for k, v in m.items()]
    lines.append("\\newcommand{\\FPieceRows}{%\n" + "\n".join(rows) + "}")
    lines.append("\\newcommand{\\FOperationRows}{%\n" + "\n".join(oprows) + "}")
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"{len(m)} numbers, {len(rows)} piece rows, {len(oprows)} operation rows -> {out / 'numbers.tex'}")


if __name__ == "__main__":
    main()
