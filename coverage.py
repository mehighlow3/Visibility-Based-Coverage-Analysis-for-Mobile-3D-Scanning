"""Coverage assessment for a mobile LiDAR scan.

Grades every surface voxel by sampling quality and labels the frontier of observed
space as occluded, unsampleable or missed.

Input  <scan>\\       LIDAR_*.bag, recalculate_trj.txt (RayStudio, TUM, 200 Hz)
Output out\\<scan>\\  coverage.json, gaps.json, coverage_voxels.ply, octree.bt

    python coverage.py --scan institutGesamt
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCANS = HERE
OUT = HERE / "out"
TOPIC = "/vanjee_722z"

PC2 = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}

OCCUPIED, FREE, OCCLUDED, MISSED = 0, 1, 2, 3
GOOD, FAIR, POOR = 4, 5, 6
UNSAMPLEABLE = 7
COLORS = {OCCUPIED: (60, 180, 75), FREE: (220, 220, 220),
          OCCLUDED: (130, 130, 130), MISSED: (230, 40, 40),
          GOOD: (40, 170, 70), FAIR: (240, 190, 40), POOR: (235, 110, 30),
          UNSAMPLEABLE: (90, 110, 200)}
NAMES = {OCCUPIED: "occupied", FREE: "free", OCCLUDED: "occluded", MISSED: "missed",
         GOOD: "good", FAIR: "fair", POOR: "poor", UNSAMPLEABLE: "unsampl."}

# voxel key, independent of the grid origin
KX, KY = np.int64(1_000_003), np.int64(1_009)


def vox_key(p, g):
    ijk = np.floor(np.asarray(p) / g).astype(np.int64)
    if ijk.ndim == 1:
        ijk = ijk[None, :]
    return ijk[:, 0] * KX + ijk[:, 1] * KY + ijk[:, 2]


def pc2_to_numpy(msg):
    n, f, o = [], [], []
    for fl in msg.fields:
        n.append(fl.name); f.append(PC2[fl.datatype]); o.append(fl.offset)
    dt = np.dtype({"names": n, "formats": f, "offsets": o, "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def quats_to_R(q):
    """(N,4) xyzw -> (N,3,3)"""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = len(q)
    R = np.empty((N, 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w); R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def slerp(q0, q1, w):
    """Vectorised SLERP. q: (N,4) xyzw, w: (N,)"""
    d = np.sum(q0 * q1, axis=1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.abs(d).clip(-1.0, 1.0)
    th = np.arccos(d)
    st = np.sin(th)
    lin = st < 1e-6                      # nearly parallel: plain lerp
    safe = np.where(lin, 1.0, st)
    s0 = np.where(lin, 1.0 - w, np.sin((1.0 - w) * th) / safe)
    s1 = np.where(lin, w,       np.sin(w * th) / safe)
    q = s0[:, None] * q0 + s1[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def write_ply_rgb(path, xyz, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, c in zip(xyz, rgb):
            fh.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {c[0]} {c[1]} {c[2]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="institutGesamt")
    ap.add_argument("--res", type=float, default=0.10, help="octree resolution [m]")
    ap.add_argument("--grid", type=float, default=0.20, help="analysis grid [m]")
    ap.add_argument("--radius", type=float, default=8.0, help="analysis domain: distance from path [m]")
    ap.add_argument("--max-range", type=float, default=15.0)
    ap.add_argument("--min-range", type=float, default=0.4)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="use every N-th sweep (2 = half the memory and time)")
    ap.add_argument("--no-slerp", action="store_true",
                    help="one rigid pose per sweep, for comparison")
    ap.add_argument("--slerp-chunks", type=int, default=5,
                    help="number of time slices per sweep when inserting")
    ap.add_argument("--fov-lo", type=float, default=None, help="manual lower FOV [deg]")
    ap.add_argument("--fov-hi", type=float, default=None, help="manual upper FOV [deg]")
    ap.add_argument("--p-sample", type=float, default=0.5,
                    help="if P(voxel never sampled) >= this -> sampling-limited")
    ap.add_argument("--n-probe", type=int, default=12,
                    help="number of poses probed per frontier voxel")
    ap.add_argument("--beam-deg", type=float, default=None,
                    help="angular beam spacing [deg]; measured from the data by default")
    ap.add_argument("--min-gap-voxels", type=int, default=8)
    ap.add_argument("--q-density", type=float, default=2000.0,
                    help="good: min point density per m2 of surface (resolution independent)")
    ap.add_argument("--q-range", type=float, default=6.0, help="good: max mean range [m]")
    ap.add_argument("--q-dirs", type=int, default=3,
                    help="good: min distinct viewing directions (octants, max 8)")
    args = ap.parse_args()

    import octomap
    from rosbags.highlevel import AnyReader
    from scipy import ndimage
    from scipy.spatial import cKDTree

    scan_dir = SCANS / args.scan
    out_dir = OUT / args.scan
    out_dir.mkdir(parents=True, exist_ok=True)
    trj_file = scan_dir / "recalculate_trj.txt"
    bags = sorted(scan_dir.rglob("LIDAR_*.bag"))

    if not trj_file.exists():
        print(f"[FAIL] no {trj_file.name} — run RayStudio Data Processing first")
        return 1
    if not bags:
        print(f"[FAIL] no LIDAR bag in {scan_dir}")
        return 1

    print("=" * 74)
    print(f"COVERAGE ANALYSIS   scan={args.scan}")
    print("=" * 74)
    print(f"octree res {args.res} m | analysis grid {args.grid} m | domain radius {args.radius} m")
    print(f"range {args.min_range}..{args.max_range} m\n")

    # 1. trajectory
    T = np.loadtxt(trj_file)
    t_traj, xyz_traj, q_traj = T[:, 0], T[:, 1:4], T[:, 4:8]
    R_traj = quats_to_R(q_traj)
    print(f"[1/6] trajectory : {len(T):,} poses @ {len(T)/(t_traj[-1]-t_traj[0]):.0f} Hz")

    # 2. octree
    tree = octomap.OcTree(args.res)
    t0 = time.time()
    n_sweep = n_pts = 0
    miss_t = 0
    q_key, q_rng, q_dir = [], [], []
    fov_samples = []
    ring_samples = []

    with AnyReader(bags) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        total = sum(c.msgcount for c in conns)
        limit = args.max_frames or total
        print(f"[2/6] inserting {min(limit,total):,} sweeps into the octree ...")

        seen = 0
        for conn, bag_ns, raw in reader.messages(connections=conns):
            if n_sweep >= limit:
                break
            seen += 1
            if (seen - 1) % args.stride:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            a = pc2_to_numpy(msg)

            t_sweep = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
            ok = np.isfinite(xyz).all(1)
            r = np.linalg.norm(xyz, axis=1)
            ok &= (r > args.min_range) & (r < args.max_range)
            xyz = xyz[ok]
            rng = r[ok]
            if len(xyz) < 50:
                continue

            # vertical FOV and ring spacing, measured from the first 50 sweeps
            if len(fov_samples) < 50:
                fov_samples.append(np.degrees(np.arcsin(xyz[:, 2] / rng)))
                if "ring" in a.dtype.names:
                    ring_samples.append(a["ring"][ok].astype(np.int16))

            # one pose per point: 'timestamp' is the offset from the sweep start
            if args.no_slerp:
                j = int(np.clip(np.searchsorted(t_traj, t_sweep), 0, len(t_traj) - 1))
                if abs(t_traj[j] - t_sweep) > 0.05:
                    miss_t += 1
                    continue
                pos = np.repeat(xyz_traj[j][None, :], len(xyz), axis=0)
                world = xyz @ R_traj[j].T + xyz_traj[j]
            else:
                t_pt = t_sweep + a["timestamp"].astype(np.float64)[ok]
                if t_pt.max() < t_traj[0] or t_pt.min() > t_traj[-1]:
                    miss_t += 1
                    continue
                j1 = np.clip(np.searchsorted(t_traj, t_pt), 1, len(t_traj) - 1)
                j0 = j1 - 1
                span = t_traj[j1] - t_traj[j0]
                w = np.clip((t_pt - t_traj[j0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
                pos = xyz_traj[j0] + w[:, None] * (xyz_traj[j1] - xyz_traj[j0])
                Rp = quats_to_R(slerp(q_traj[j0], q_traj[j1], w))
                world = np.einsum("nij,nj->ni", Rp, xyz) + pos

            # the sensor moves during a sweep: insert in time slices, each from its own origin
            nchunk = 1 if args.no_slerp else args.slerp_chunks
            for c in range(nchunk):
                s0 = len(world) * c // nchunk
                s1 = len(world) * (c + 1) // nchunk
                if s1 - s0 < 5:
                    continue
                tree.insertPointCloud(
                    pointcloud=np.ascontiguousarray(world[s0:s1]),
                    origin=np.ascontiguousarray(pos[(s0 + s1) // 2]))

            # per point: voxel, range and viewing octant (sign per axis)
            q_key.append(vox_key(world, args.grid))
            q_rng.append(rng.astype(np.float32))
            d = world - pos
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            q_dir.append(((np.sign(d[:, 0]) + 1) * 9 +
                          (np.sign(d[:, 1]) + 1) * 3 +
                          (np.sign(d[:, 2]) + 1)).astype(np.int8))

            n_sweep += 1
            n_pts += len(world)
            if n_sweep % 300 == 0:
                print(f"        {n_sweep:5d}   {n_pts/1e6:5.2f} M pts   {time.time()-t0:5.1f}s")

    tree.updateInnerOccupancy()
    print(f"        done: {n_sweep:,} sweeps, {n_pts:,} points, {time.time()-t0:.1f}s")
    if miss_t:
        print(f"        ! {miss_t} sweeps without a pose (outside the trajectory)")

    lo, hi = np.asarray(tree.getMetricMin()), np.asarray(tree.getMetricMax())
    print(f"        octree bbox: x {lo[0]:.1f}..{hi[0]:.1f}  y {lo[1]:.1f}..{hi[1]:.1f}  z {lo[2]:.1f}..{hi[2]:.1f}")

    # 3. analysis domain: grid voxels within --radius of the path
    tsub = xyz_traj[::50]
    g = args.grid
    gl = np.maximum(lo, tsub.min(0) - args.radius)
    gh = np.minimum(hi, tsub.max(0) + args.radius)
    ax = [np.arange(gl[k] + g/2, gh[k], g) for k in range(3)]
    GX, GY, GZ = np.meshgrid(*ax, indexing="ij")
    shape = GX.shape
    P = np.stack([GX.ravel(), GY.ravel(), GZ.ravel()], 1)
    print(f"\n[3/6] analysis grid : {shape}  = {len(P):,} voxels")

    kd = cKDTree(tsub)
    dmin, _ = kd.query(P, k=1, workers=-1)
    keep = dmin <= args.radius
    del dmin
    P = P[keep]
    idx_keep = np.flatnonzero(keep)
    print(f"        within the domain (<= {args.radius} m from the path): {len(P):,}")

    # 4. classification: each grid voxel takes the state of the octree cell at its centre
    print(f"\n[4/6] classifying ...")
    state = np.full(len(P), -1, np.int8)
    t0 = time.time()
    for k, p in enumerate(P):
        n = tree.search(p)
        try:
            state[k] = OCCUPIED if tree.isNodeOccupied(n) else FREE
        except Exception:
            state[k] = -1
        if k and k % 200_000 == 0:
            print(f"        {k:,}/{len(P):,}   {time.time()-t0:5.1f}s")

    n_unknown = int((state < 0).sum())
    print(f"        occupied {int((state==OCCUPIED).sum()):,}   "
          f"free {int((state==FREE).sum()):,}   unknown {n_unknown:,}")

    # 5. frontier: unknown voxels next to observed free space
    print(f"\n[5/6] frontier of free space + ray casting ...")
    V = np.full(shape, -9, np.int8).ravel()
    V[idx_keep] = state
    V = V.reshape(shape)

    free_m = (V == FREE)
    unk_m = (V == -1)
    occ_m = (V == OCCUPIED)
    struct = ndimage.generate_binary_structure(3, 1)
    frontier = unk_m & ndimage.binary_dilation(free_m, struct)
    touch_occ = ndimage.binary_dilation(occ_m, struct)

    fidx = np.flatnonzero(frontier.ravel())
    print(f"        frontier voxels: {len(fidx):,}  "
          f"(of {int(unk_m.sum()):,} unknown — the rest is solid or outside)")

    coords = np.unravel_index(fidx, shape)
    FP = np.stack([ax[0][coords[0]], ax[1][coords[1]], ax[2][coords[2]]], 1)
    touches = touch_occ.ravel()[fidx]

    # sensor model from the data: vertical FOV and angular ring spacing
    EL = np.concatenate(fov_samples)
    fov_lo = float(np.percentile(EL, 0.2)) if args.fov_lo is None else args.fov_lo
    fov_hi = float(np.percentile(EL, 99.8)) if args.fov_hi is None else args.fov_hi
    print(f"        measured vertical sensor FOV: {fov_lo:+.1f}° .. {fov_hi:+.1f}°")

    if ring_samples and args.beam_deg is None:
        RG = np.concatenate(ring_samples)
        els = np.array([np.median(EL[RG == g]) for g in np.unique(RG)
                        if (RG == g).sum() > 20])
        beam_deg = float(np.median(np.diff(np.sort(els))))
        print(f"        rings {len(els)}   angular beam spacing {beam_deg:.2f}°")
    else:
        beam_deg = args.beam_deg if args.beam_deg else 2.67
    r_res = args.grid / np.tan(np.radians(beam_deg))
    print(f"        -> at {r_res:.1f} m the ring spacing reaches the voxel size "
          f"({args.grid} m); beyond that one pose no longer guarantees a hit")
    print(f"        probabilistic threshold: P(never sampled) >= {args.p_sample} "
          f"-> sampling-limited   ({args.n_probe} poses per voxel)")

    Rsub = quats_to_R(q_traj[::50])
    kd_pose = cKDTree(tsub)

    t0 = time.time()
    n_blocked = n_clear = n_unsamp = 0
    fstate = np.empty(len(fidx), np.int8)
    for j, p in enumerate(FP):
        near = kd_pose.query_ball_point(p, args.max_range)
        if not near:
            fstate[j] = UNSAMPLEABLE; n_unsamp += 1; continue
        near = np.asarray(near)
        diff = p[None, :] - tsub[near]
        dist = np.linalg.norm(diff, axis=1)
        good = dist > max(args.min_range, 1e-6)
        if not good.any():
            fstate[j] = UNSAMPLEABLE; n_unsamp += 1; continue
        near, diff, dist = near[good], diff[good], dist[good]
        # only poses that have the voxel inside their vertical FOV
        v_s = np.einsum("nji,nj->ni", Rsub[near], diff)
        el = np.degrees(np.arcsin(np.clip(v_s[:, 2] / dist, -1, 1)))
        infov = (el >= fov_lo) & (el <= fov_hi)
        if not infov.any():
            fstate[j] = UNSAMPLEABLE; n_unsamp += 1; continue

        cand = near[infov]; cd = dist[infov]
        order = np.argsort(cd)[:args.n_probe]

        # rings at distance d are s = d*tan(beam) apart, so one clear view hits a
        # voxel of size g with p ~ g/s; P(never sampled) is the product over views
        p_never = 1.0
        clear = False
        for oi in order:
            o = tsub[cand[oi]]
            dd = cd[oi]
            dirv = (p - o) / dd
            end = np.zeros(3)
            if tree.castRay(o, dirv, end, False, dd):
                continue
            # castRay also returns False when it stops in unknown space:
            # the view is clear only if the ray reached the voxel
            if np.linalg.norm(end - o) < dd - (g * 0.87 + args.res):
                continue
            clear = True
            s = dd * np.tan(np.radians(beam_deg))
            p_never *= (1.0 - min(1.0, args.grid / s))
            if p_never < args.p_sample:
                break
        if not clear:
            fstate[j] = OCCLUDED; n_blocked += 1
        elif p_never >= args.p_sample:
            fstate[j] = UNSAMPLEABLE; n_unsamp += 1
        elif touches[j]:
            fstate[j] = OCCLUDED; n_blocked += 1
        else:
            fstate[j] = MISSED; n_clear += 1
        if j and j % 50_000 == 0:
            print(f"        {j:,}/{len(fidx):,}   {time.time()-t0:5.1f}s")

    print(f"        occluded {n_blocked:,}   unsampleable {n_unsamp:,}   "
          f"missed {n_clear:,}   ({time.time()-t0:.1f}s)")

    # frontier labels back into 'state'; other unknown voxels are outside
    state[:] = np.where(state < 0, -9, state)
    flat_to_P = np.full(np.prod(shape), -1, np.int64)
    flat_to_P[idx_keep] = np.arange(len(P))
    for j, fi in enumerate(fidx):
        pi = flat_to_P[fi]
        if pi >= 0:
            state[pi] = fstate[j]

    # 5b. surface quality per occupied voxel
    print(f"\n[5b] quality per voxel ...")
    QK = np.concatenate(q_key); q_key.clear()
    QR = np.concatenate(q_rng); q_rng.clear()
    QD = np.concatenate(q_dir); q_dir.clear()
    uk, inv = np.unique(QK, return_inverse=True)
    del QK
    cnt = np.bincount(inv).astype(np.int64)
    rsum = np.bincount(inv, weights=QR.astype(np.float64))
    rmean = rsum / cnt
    del QR
    combo = np.unique(inv.astype(np.int64) * 27 + QD.astype(np.int64))
    ndir = np.bincount(combo // 27, minlength=len(uk)).astype(np.int64)
    del QD, inv, combo

    # threshold as a density, so it means the same at every grid size
    face = args.grid ** 2
    thr_good = args.q_density * face
    thr_fair = max(2.0, thr_good / 4)
    print(f"        threshold: >= {args.q_density:,.0f} pts/m2  ->  {thr_good:.1f} hits "
          f"per voxel at {args.grid} m  (fair >= {thr_fair:.1f})")

    occ_idx = np.flatnonzero(state == OCCUPIED)
    lut = {int(k): i for i, k in enumerate(uk)}
    n_good = n_fair = n_poor = 0
    qstats = []
    for k in occ_idx:
        j = lut.get(int(vox_key(P[k], args.grid)[0]), -1)
        if j < 0:
            state[k] = POOR; n_poor += 1; continue
        c, rm, nd = int(cnt[j]), float(rmean[j]), int(ndir[j])
        qstats.append((c / face, rm, nd))
        if c >= thr_good and rm <= args.q_range and nd >= args.q_dirs:
            state[k] = GOOD; n_good += 1
        elif c >= thr_fair and rm <= args.q_range * 2:
            state[k] = FAIR; n_fair += 1
        else:
            state[k] = POOR; n_poor += 1
    qs = np.array(qstats) if qstats else np.zeros((1, 3))
    print(f"        good {n_good:,}   fair {n_fair:,}   poor {n_poor:,}")
    print(f"        per voxel: density median {np.median(qs[:,0]):,.0f} pts/m2   "
          f"range median {np.median(qs[:,1]):.1f} m   directions median {np.median(qs[:,2]):.0f}")

    # 6. gaps and output
    print(f"\n[6/6] grouping gaps ...")
    vol = np.zeros(shape, np.int8).ravel()
    vol[:] = -9
    vol[idx_keep] = state
    vol = vol.reshape(shape)
    lab, nlab = ndimage.label(vol == MISSED)
    sizes = ndimage.sum(vol == MISSED, lab, range(1, nlab + 1))
    order = np.argsort(sizes)[::-1]
    vcell = g ** 3

    gaps = []
    for rank, li in enumerate(order, 1):
        nvox = int(sizes[li])
        if nvox < args.min_gap_voxels:
            break
        m = lab == (li + 1)
        ii = np.array(np.nonzero(m))
        c = [float(ax[k][ii[k]].mean()) for k in range(3)]
        ext = [float(np.ptp(ax[k][ii[k]]) + g) for k in range(3)]
        gaps.append({"rank": rank, "voxels": nvox, "volume_m3": round(nvox * vcell, 3),
                     "center": [round(v, 2) for v in c],
                     "extent_m": [round(v, 2) for v in ext]})
        if rank >= 25:
            break

    # analysis domain = observed (surfaces + free) + its frontier
    n_occ = n_good + n_fair + n_poor
    n_free = int((state == FREE).sum())
    tot = n_occ + n_free + n_blocked + n_clear + n_unsamp
    obs = n_occ + n_free + n_blocked + n_unsamp
    metrics = {
        "scan": args.scan,
        "params": {"octree_res": args.res, "grid": g, "radius": args.radius,
                   "max_range": args.max_range, "min_range": args.min_range},
        "sweeps": n_sweep, "points_inserted": n_pts,
        "analysis_domain_voxels": tot,
        "frontier_voxels": int(len(fidx)),
        "surface_voxels": n_occ,
        "surface_good": n_good, "surface_fair": n_fair, "surface_poor": n_poor,
        "surface_good_pct": round(100 * n_good / max(n_occ, 1), 2),
        "surface_poor_pct": round(100 * n_poor / max(n_occ, 1), 2),
        "quality_thresholds": {"density_pts_per_m2": args.q_density,
                               "hits_per_voxel_at_this_grid": round(thr_good, 2),
                               "range_m": args.q_range, "dirs": args.q_dirs},
        "median_surface_density_pts_per_m2": round(float(np.median(qs[:, 0])), 1),
        "free": n_free,
        "occluded": n_blocked,
        "unsampleable": n_unsamp,
        "sensor_fov_deg": [round(fov_lo, 2), round(fov_hi, 2)],
        "beam_spacing_deg": round(beam_deg, 3),
        "single_pose_guarantee_range_m": round(r_res, 2),
        "p_sample_threshold": args.p_sample,
        "poses_probed": args.n_probe,
        "missed": n_clear,
        "coverage_pct": round(100 * obs / tot, 2),
        "missed_pct": round(100 * n_clear / tot, 2),
        "missed_volume_m3": round(n_clear * vcell, 2),
        "n_gaps": len(gaps),
        "largest_gap_m3": gaps[0]["volume_m3"] if gaps else 0.0,
    }

    (out_dir / "coverage.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "gaps.json").write_text(json.dumps(gaps, indent=2), encoding="utf-8")
    tree.writeBinary(str(out_dir / "octree.bt").encode())

    show = np.isin(state, [GOOD, FAIR, POOR, OCCLUDED, UNSAMPLEABLE, MISSED])
    rgb = np.array([COLORS[s] for s in state[show]], np.uint8)
    write_ply_rgb(out_dir / "coverage_voxels.ply", P[show], rgb)

    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print("  SURFACES (graded quality):")
    for s in (GOOD, FAIR, POOR):
        n = int((state == s).sum())
        print(f"    {NAMES[s]:8s} {n:9,}  {100*n/max(n_occ,1):5.1f}% of surface   {n*vcell:7.1f} m3")
    print("  VOLUME:")
    for s in (FREE, OCCLUDED, UNSAMPLEABLE, MISSED):
        n = int((state == s).sum())
        print(f"    {NAMES[s]:8s} {n:9,}  {100*n/tot:5.1f}%            {n*vcell:7.1f} m3")
    print(f"\n  NOT 'missed' (occluded/unsampl. included)  : {metrics['coverage_pct']:.2f} %"
          f"   <- not building coverage, do not quote")
    print(f"  POOR-QUALITY SURFACE                       : {metrics['surface_poor_pct']:.2f} %")
    print(f"  MISSED                                     : {metrics['missed_pct']:.2f} %  "
          f"({metrics['missed_volume_m3']} m3)")
    print(f"  gaps >= {args.min_gap_voxels} voxels: {len(gaps)}")
    for gp in gaps[:5]:
        print(f"     #{gp['rank']}  {gp['volume_m3']:7.2f} m3  centre {gp['center']}  size {gp['extent_m']}")
    print(f"\nsaved to {out_dir}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
