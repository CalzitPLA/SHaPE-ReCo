"""(Written 2026-09-29 on the institute server for the comparison batch; in the repository since the ABC runs.)

One pair (input mesh, model mesh): volumes, the sampled comparison, surface distances, and the missing and the
invented material as connected regions. Writes <out>/compare.json and <out>/regions.json.

    python compare_pair.py <name> <input.stl> <model.stl> <out_dir> [--n 20000000] [--surface 1000000] [--pitch 1.0]

The sampled part is the laptop's compare.py (seed 0, chunks of 500 000 points uniform in the joint bounding box
+- 1 mm, inside = |fast winding number| > 0.5): its first 4e6 points are the laptop's points, stored as `replica_4e6`.
"""
import argparse, json, os, sys, time
import numpy as np
import igl
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("input"); ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--n", type=int, default=20_000_000)
ap.add_argument("--surface", type=int, default=1_000_000)
ap.add_argument("--pitch", type=float, default=1.0)
ap.add_argument("--finer", type=float, default=0.5, help="pitch tried first where the budget allows")
ap.add_argument("--budget", type=float, default=2.5e8, help="winding evaluations of the finest level, per pair")
ap.add_argument("--max-voxels", type=float, default=4e7, help="difference voxels kept for the labelling")
ap.add_argument("--top", type=int, default=200)
ap.add_argument("--regions-only", action="store_true", help="compare.json exists: only the regions again")
ap.add_argument("--grow", type=int, default=2)
ap.add_argument("--normalise", type=float, default=0.0,
                help="scale both meshes so that the INPUT's box diagonal is this long (0: as they are). The paddings and "
                     "pitches here are absolute (1 mm): right for parts in mm of decimetre size, wrong for a part of "
                     "unknown unit. The scale is written as `scale`; divide lengths by it, volumes by its cube")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {args.name}: {msg}", flush=True)


def load(path):
    m = trimesh.load(path, force="mesh", process=True)
    V, F = np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64)
    tri = V[F]
    vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)
    area = float(0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())
    return V, F, dict(file=path, triangles=int(len(F)), vertices=int(len(V)), volume_mesh=vol, area=area,
                      watertight=bool(m.is_watertight), bbox=[V.min(0).tolist(), V.max(0).tolist()])


def inside_of(V, F):
    def inside(p):
        p = np.ascontiguousarray(p, np.float64)
        if not len(p):
            return np.zeros(0, bool)
        return np.abs(igl.fast_winding_number(V, F, p)) > 0.5
    return inside


def distance(P, V, F):
    out = igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64), V, F)
    return np.sqrt(np.maximum(np.asarray(out[0]), 0.0))


Vi, Fi, info_i = load(args.input)
Vm, Fm, info_m = load(args.model)
SCALE = 1.0
if args.normalise > 0:
    SCALE = args.normalise / float(np.linalg.norm(Vi.max(0) - Vi.min(0)))
    Vi, Vm = np.ascontiguousarray(Vi * SCALE), np.ascontiguousarray(Vm * SCALE)
    for info in (info_i, info_m):
        info.update(volume_mesh=info["volume_mesh"] * SCALE ** 3, area=info["area"] * SCALE ** 2,
                    bbox=(np.asarray(info["bbox"]) * SCALE).tolist())
in_i, in_m = inside_of(Vi, Fi), inside_of(Vm, Fm)
lo = np.minimum(Vi.min(0), Vm.min(0)) - 1.0
hi = np.maximum(Vi.max(0), Vm.max(0)) + 1.0
box = float(np.prod(hi - lo))
res = dict(name=args.name, scale=SCALE, input=info_i, model=info_m, box=box, box_lo=lo.tolist(), box_hi=hi.tolist(),
           versions=dict(igl=getattr(igl, "__version__", None), numpy=np.__version__, trimesh=trimesh.__version__),
           host=os.uname().nodename)
log(f"meshes {len(Fi)} / {len(Fm)} triangles, box {box:.4g} mm3")

# ------------------------------------------------------------------------------------------------ sampled comparison
def counts_result(n, both, only_i, only_m):
    c = box / n

    def vol(k):
        p = k / n
        return dict(volume=k * c, se=box * float(np.sqrt(p * (1.0 - p) / n)), points=int(k))
    u = both + only_i + only_m
    iou = both / max(u, 1)
    return dict(n=int(n), iou=iou, iou_se=float(np.sqrt(iou * (1.0 - iou) / max(u, 1))),
                common=vol(both), missing=vol(only_i), invented=vol(only_m),
                volume_input_sampled=vol(both + only_i), volume_model_sampled=vol(both + only_m),
                mm3_per_point=c)


if not args.regions_only:
    rng = np.random.default_rng(0)
    both = only_i = only_m = 0
    CH = 500_000
    t = time.time()
    for k in range(0, args.n, CH):
        p = rng.uniform(lo, hi, size=(CH, 3))
        a, b = in_i(p), in_m(p)
        both += int((a & b).sum()); only_i += int((a & ~b).sum()); only_m += int((b & ~a).sum())
        if k + CH == 4_000_000:
            res["replica_4e6"] = counts_result(4_000_000, both, only_i, only_m)
    res["sampled"] = counts_result(args.n, both, only_i, only_m)
    res["sampled"]["seconds"] = time.time() - t
    log(f"sampled IoU {res['sampled']['iou']:.4f} in {time.time() - t:.0f} s")

# ------------------------------------------------------------------------------------------------ surface distances
def surface_points(V, F, n, seed):
    rg = np.random.default_rng(seed)
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rg.choice(len(F), size=n, p=area / area.sum())
    u, v = rg.random(n), rg.random(n); s = np.sqrt(u)
    return tri[idx, 0] * (1 - s)[:, None] + tri[idx, 1] * (s * (1 - v))[:, None] + tri[idx, 2] * (s * v)[:, None]


def stats(d, boot=100):
    n = len(d)
    out = dict(n=int(n), mean=float(d.mean()), mean_se=float(d.std(ddof=1) / np.sqrt(n)),
               median=float(np.median(d)), p95=float(np.percentile(d, 95)), p99=float(np.percentile(d, 99)),
               max=float(d.max()), within_0_05=float((d <= 0.05).mean()), within_1=float((d <= 1.0).mean()))
    for key in ("within_0_05", "within_1"):
        out[key + "_se"] = float(np.sqrt(out[key] * (1.0 - out[key]) / n))
    if boot:
        rg = np.random.default_rng(7)
        q = np.array([np.percentile(d[rg.integers(0, n, n)], [50, 95, 99]) for _ in range(boot)])
        out["median_se"], out["p95_se"], out["p99_se"] = (float(x) for x in q.std(axis=0, ddof=1))
        out["max_se"] = None                     # the maximum of a sample has no standard error of this kind
    return out


def both_ways(n, boot):
    di = distance(surface_points(Vi, Fi, n, 1), Vm, Fm)
    dm = distance(surface_points(Vm, Fm, n, 2), Vi, Fi)
    a, b = stats(di, boot), stats(dm, boot)
    ch = 0.5 * (a["mean"] + b["mean"])
    return dict(input_to_model=a, model_to_input=b, chamfer=ch,
                chamfer_se=0.5 * float(np.hypot(a["mean_se"], b["mean_se"])))


if args.regions_only:
    res = json.load(open(os.path.join(args.out, "compare.json")))
else:
    t = time.time()
    res["replica_surface_2e5"] = both_ways(200_000, 0)
    res["surface"] = both_ways(args.surface, 100)
    res["surface"]["seconds"] = time.time() - t
    log(f"chamfer {res['surface']['chamfer']:.4f} mm in {time.time() - t:.0f} s")
    res["seconds"] = time.time() - T0
    json.dump(res, open(os.path.join(args.out, "compare.json"), "w"), indent=1)

# ------------------------------------------------------------------------------------------------ connected regions
# The classes (in input, in model) at the centres of a regular grid of pitch p. Cells of 64p, 16p, 4p are tested first:
# a cell whose centre is farther from a mesh than its half diagonal lies on one side of that mesh, so its voxels take
# the centre's answer; only the others are divided. The result equals the full grid's.
# The grid's origin is moved off the box corner by an uneven fraction of the pitch: part faces lie at round
# coordinates, and a grid starting at (bounding box - 1 mm) puts voxel centres exactly ON such faces, where the winding
# number is 1/2 and the class is arbitrary (measured on review__r01: 2.3e6 mm3 "invented" against 0 exact).
FACT = (64, 16, 4, 1)
SHIFT = np.array([0.2113, 0.3771, 0.1429])


def grid_classes(p):
    glo = lo - SHIFT * p
    n_fine = np.ceil((hi - glo) / p).astype(np.int64)
    n_pad = ((n_fine + 63) // 64) * 64
    root = n_pad // 64
    g = np.stack(np.meshgrid(*[np.arange(r, dtype=np.int64) for r in root], indexing="ij"), -1).reshape(-1, 3) * 64
    ka = np.full(len(g), -1, np.int8); kb = ka.copy()         # -1 unknown, 0 outside, 1 inside
    blocks, evals = [], 0                                     # (corner, size, a, b) of the settled cells that differ
    for level, s in enumerate(FACT):
        centre = glo + (g + 0.5 * s) * p
        for known, (V, F, ins) in ((ka, (Vi, Fi, in_i)), (kb, (Vm, Fm, in_m))):
            todo = np.flatnonzero(known < 0)
            if s > 1:
                for c0 in range(0, len(todo), 2_000_000):
                    sel = todo[c0:c0 + 2_000_000]
                    d = distance(centre[sel], V, F)
                    far = sel[d > 0.5 * np.sqrt(3.0) * s * p * 1.0001]
                    known[far] = ins(centre[far])
            else:
                evals += len(todo)
                for c0 in range(0, len(todo), 2_000_000):
                    sel = todo[c0:c0 + 2_000_000]
                    known[sel] = ins(centre[sel])
        done = (ka >= 0) & (kb >= 0)
        diff = done & (ka != kb)
        if diff.any():
            blocks.append((g[diff], s, ka[diff].astype(bool)))
        if s == 1:
            break
        keep = ~done
        f = s // FACT[level + 1]
        if keep.sum() * f ** 3 > args.budget * 4:
            return None, dict(pitch=p, refused=f"{keep.sum() * f ** 3:.3g} cells at level {FACT[level + 1]}p")
        off = np.stack(np.meshgrid(*[np.arange(f, dtype=np.int64)] * 3, indexing="ij"), -1).reshape(-1, 3) * FACT[level + 1]
        g = (g[keep][:, None, :] + off[None, :, :]).reshape(-1, 3)
        ka, kb = np.repeat(ka[keep], f ** 3), np.repeat(kb[keep], f ** 3)
        unknown = int((ka < 0).sum() + (kb < 0).sum())
        log(f"pitch {p}: level {FACT[level + 1]}p, {len(g)} cells, {unknown} answers open")
        if FACT[level + 1] == 1 and unknown > args.budget:
            return None, dict(pitch=p, refused=f"{unknown:.3g} winding evaluations at the finest level")
    total = sum(len(c) * s ** 3 for c, s, _ in blocks)
    if total > args.max_voxels:
        return None, dict(pitch=p, refused=f"{total:.3g} difference voxels")
    vox = dict(missing=[], invented=[])
    for c, s, a in blocks:
        off = np.stack(np.meshgrid(*[np.arange(s, dtype=np.int64)] * 3, indexing="ij"), -1).reshape(-1, 3)
        for key, m in (("missing", a), ("invented", ~a)):
            if m.any():
                vox[key].append((c[m][:, None, :] + off[None, :, :]).reshape(-1, 3))
    vox = {k: (np.concatenate(v) if v else np.zeros((0, 3), np.int64)) for k, v in vox.items()}
    return vox, dict(pitch=p, grid_lo=glo.tolist(), grid=n_fine.tolist(), grid_padded=n_pad.tolist(), winding_evaluations_finest=int(evals),
                     full_grid_cells=int(np.prod(n_fine)))


def components(ijk, dims, p, glo, connectivity=26):
    from scipy import sparse
    from scipy.sparse import csgraph
    K = len(ijk)
    if K == 0:
        return dict(voxels=0, volume=0.0, components=0, regions=[])
    dims = np.asarray(dims, np.int64)
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(lin); lin, ijk = lin[order], ijk[order]
    rows, cols = [], []
    offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]
    if connectivity == 6:
        offs = [o for o in offs if sum(abs(x) for x in o) == 1]
    for o in offs:
        nb = ijk + np.asarray(o, np.int64)
        ok = np.all((nb >= 0) & (nb < dims), axis=1)
        ln = (nb[:, 0] * dims[1] + nb[:, 1]) * dims[2] + nb[:, 2]
        pos = np.minimum(np.searchsorted(lin, ln), K - 1)
        hit = ok & (lin[pos] == ln)
        rows.append(np.flatnonzero(hit).astype(np.int32)); cols.append(pos[hit].astype(np.int32))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = sparse.coo_matrix((np.ones(len(rows), np.bool_), (rows, cols)), shape=(K, K)).tocsr()
    n, lab = csgraph.connected_components(graph, directed=False)
    count = np.bincount(lab, minlength=n)
    centre = np.stack([np.bincount(lab, weights=ijk[:, d] + 0.5, minlength=n) / count for d in range(3)], 1)
    o2 = np.argsort(lab, kind="stable"); start = np.concatenate([[0], np.cumsum(count)[:-1]])
    mn = np.minimum.reduceat(ijk[o2], start, axis=0); mx = np.maximum.reduceat(ijk[o2], start, axis=0) + 1
    big = np.argsort(-count, kind="stable")
    regions = []
    for r, c in enumerate(big[:args.top]):
        b0, b1 = glo + mn[c] * p, glo + mx[c] * p
        regions.append(dict(rank=r + 1, voxels=int(count[c]), volume=float(count[c] * p ** 3),
                            bbox_lo=b0.tolist(), bbox_hi=b1.tolist(), bbox_size=(b1 - b0).tolist(),
                            centre=(glo + centre[c] * p).tolist(),
                            fill=float(count[c] / np.prod(mx[c] - mn[c]))))
    vols = np.sort(count)[::-1] * p ** 3
    return dict(voxels=int(K), volume=float(K * p ** 3), components=int(n), connectivity=connectivity,
                components_of_one_voxel=int((count == 1).sum()),
                volume_in_listed=float(vols[:args.top].sum()),
                volume_in_largest_10=float(vols[:10].sum()), regions=regions)


def neighbours(lin, ijk, dims, offs):
    """per offset: (index of the voxel, index of its neighbour) for the neighbours that are in the set"""
    K = len(lin)
    for o in offs:
        nb = ijk + np.asarray(o, np.int64)
        ok = np.all((nb >= 0) & (nb < dims), axis=1)
        ln = (nb[:, 0] * dims[1] + nb[:, 1]) * dims[2] + nb[:, 2]
        pos = np.minimum(np.searchsorted(lin, ln), K - 1)
        hit = ok & (lin[pos] == ln)
        yield np.flatnonzero(hit), pos[hit]


def opened(ijk, dims, p, glo):
    """The regions without the thin sheets that join them: the cores (voxels whose 6 face neighbours are all in the
    set) are labelled by 26-connectivity, the labels grown back `--grow` voxels into the set; what no core reaches is
    `thin` (sheets and wires up to 2 voxels thick: differences of the two tessellations, films along shared faces)."""
    from scipy import sparse
    from scipy.sparse import csgraph
    K = len(ijk)
    if K == 0:
        return dict(components=0, thin_volume=0.0, regions=[])
    dims = np.asarray(dims, np.int64)
    lin = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    order = np.argsort(lin); lin, ijk = lin[order], ijk[order]
    face = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    full = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    nface = np.zeros(K, np.int8)
    for i, _ in neighbours(lin, ijk, dims, face):
        nface[i] += 1
    core = nface == 6
    lab = np.full(K, -1, np.int64)
    n = 0
    if core.any():
        ci = np.flatnonzero(core); clin, cijk = lin[ci], ijk[ci]
        rows, cols = [], []
        for i, j in neighbours(clin, cijk, dims, [o for o in full if o > (0, 0, 0)]):
            rows.append(i.astype(np.int32)); cols.append(j.astype(np.int32))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        n, cl = csgraph.connected_components(
            sparse.coo_matrix((np.ones(len(rows), np.bool_), (rows, cols)), shape=(len(ci), len(ci))).tocsr(), directed=False)
        lab[ci] = cl
    for _ in range(args.grow):
        new = lab.copy()
        for i, j in neighbours(lin, ijk, dims, full):
            take = (lab[i] < 0) & (lab[j] >= 0)
            new[i[take]] = lab[j[take]]
        lab = new
    thin = lab < 0
    out = dict(core="voxels with all 6 face neighbours in the set", grow=args.grow, components=int(n),
               core_voxels=int(core.sum()), thin_voxels=int(thin.sum()), thin_volume=float(thin.sum() * p ** 3),
               volume_in_regions=float((~thin).sum() * p ** 3), regions=[])
    if n:
        kl, kijk = lab[~thin], ijk[~thin]
        count = np.bincount(kl, minlength=n)
        centre = np.stack([np.bincount(kl, weights=kijk[:, d] + 0.5, minlength=n) / count for d in range(3)], 1)
        o2 = np.argsort(kl, kind="stable"); start = np.concatenate([[0], np.cumsum(count)[:-1]])
        mn = np.minimum.reduceat(kijk[o2], start, axis=0); mx = np.maximum.reduceat(kijk[o2], start, axis=0) + 1
        for r, c in enumerate(np.argsort(-count, kind="stable")[:args.top]):
            b0, b1 = glo + mn[c] * p, glo + mx[c] * p
            out["regions"].append(dict(rank=r + 1, voxels=int(count[c]), volume=float(count[c] * p ** 3),
                                       bbox_lo=b0.tolist(), bbox_hi=b1.tolist(), bbox_size=(b1 - b0).tolist(),
                                       centre=(glo + centre[c] * p).tolist(),
                                       fill=float(count[c] / np.prod(mx[c] - mn[c]))))
        vols = np.sort(count)[::-1] * p ** 3
        out["volume_in_listed"] = float(vols[:args.top].sum())
    return out


t = time.time()
reg = dict(name=args.name, method="inside tests (fast winding number) at the centres of a regular grid; cells far "
           "from both surfaces settled by their centre; 26-connected components of the two difference sets",
           box_lo=lo.tolist(), tried=[])
area = info_i["area"] + info_m["area"]
pitches = [q for q in (args.finer, args.pitch, 2.0, 4.0, 8.0, 16.0, 32.0) if q >= min(args.finer, args.pitch)]
pitches = sorted(set(pitches))
vox = None
for p in pitches:
    if 11.0 * area / p ** 2 > 1.5 * args.budget:              # the band of open cells along both surfaces, estimated
        reg["tried"].append(dict(pitch=p, refused=f"estimate {11.0 * area / p ** 2:.3g} winding evaluations"))
        continue
    vox, meta = grid_classes(p)
    reg["tried"].append(meta)
    if vox is not None:
        break
if vox is None:
    reg["error"] = "no pitch within the budget"
else:
    reg.update(meta)
    for key in ("missing", "invented"):
        reg[key] = components(vox[key], meta["grid_padded"], p, np.asarray(meta["grid_lo"]))
        reg[key]["opened"] = opened(vox[key], meta["grid_padded"], p, np.asarray(meta["grid_lo"]))
        log(f"{key}, opened: {reg[key]['opened']['components']} regions, thin {reg[key]['opened']['thin_volume']:.4g} mm3")
        log(f"{key}: {reg[key]['volume']:.4g} mm3 in {reg[key]['components']} regions at pitch {p}")
    np.savez_compressed(os.path.join(args.out, "voxels.npz"), missing=vox["missing"].astype(np.int32),
                        invented=vox["invented"].astype(np.int32), grid_lo=np.asarray(meta["grid_lo"]), pitch=p)
    reg["sampled_missing"] = res["sampled"]["missing"]["volume"]
    reg["sampled_invented"] = res["sampled"]["invented"]["volume"]
reg["seconds"] = time.time() - t
json.dump(reg, open(os.path.join(args.out, "regions.json"), "w"), indent=1)
log(f"done in {time.time() - T0:.0f} s")
