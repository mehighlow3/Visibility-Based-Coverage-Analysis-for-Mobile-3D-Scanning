"""The coverage.py method as functions, for experiments.

One octree build can be classified under several configurations:
  drop=(t1, t2)   removes all sweeps and viewpoints in that time window
  the analysis grid is aligned to multiples of `grid`, so voxels match across runs
  label(model, strict): 'multi' (P(never sampled) over all probes, as coverage.py) or
      'single' (clear view within grid / tan(beam)); strict = the ray must reach the
      voxel (as coverage.py), loose = castRay reports no hit
"""
import time
from pathlib import Path

import numpy as np

SCANS = Path(__file__).resolve().parents[1]
TOPIC = "/vanjee_722z"
PC2 = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}

OCCUPIED, FREE, OCCLUDED, MISSED = 0, 1, 2, 3
GOOD, FAIR, POOR = 4, 5, 6
UNSAMPLEABLE = 7
UNKNOWN = -1          # in the domain, never observed
OUTSIDE = -9          # outside the analysis domain


def pc2_to_numpy(msg):
    n, f, o = [], [], []
    for fl in msg.fields:
        n.append(fl.name); f.append(PC2[fl.datatype]); o.append(fl.offset)
    dt = np.dtype({"names": n, "formats": f, "offsets": o, "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def quats_to_R(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w); R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def slerp(q0, q1, w):
    d = np.sum(q0 * q1, axis=1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.abs(d).clip(-1.0, 1.0)
    th = np.arccos(d); st = np.sin(th)
    lin = st < 1e-6
    safe = np.where(lin, 1.0, st)
    s0 = np.where(lin, 1.0 - w, np.sin((1.0 - w) * th) / safe)
    s1 = np.where(lin, w, np.sin(w * th) / safe)
    q = s0[:, None] * q0 + s1[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def cell_key(ijk):
    """Collision-free int64 key for integer cells within +-2^19 of the origin."""
    ijk = np.asarray(ijk, np.int64) + (1 << 19)
    return (ijk[:, 0] << 40) | (ijk[:, 1] << 20) | ijk[:, 2]


def build(scan, drop=None, res=0.10, grid=0.20, min_range=0.4, max_range=15.0,
          chunks=5, stride=1, log=print):
    import octomap
    from rosbags.highlevel import AnyReader

    sd = SCANS / scan
    T = np.loadtxt(sd / "recalculate_trj.txt")
    t_traj, xyz_traj, q_traj = T[:, 0], T[:, 1:4], T[:, 4:8]
    t0_traj = t_traj[0]
    bags = sorted(sd.rglob("LIDAR_*.bag"))

    tree = octomap.OcTree(res)
    qk, qr, qd = [], [], []
    fov_s, ring_s = [], []
    n_sweep = n_pts = n_drop = n_miss = 0
    t0 = time.time()
    with AnyReader(bags) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        seen = 0
        for conn, _, raw in reader.messages(connections=conns):
            seen += 1
            if (seen - 1) % stride:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            t_sw = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if drop is not None and drop[0] <= t_sw - t0_traj < drop[1]:
                n_drop += 1
                continue
            a = pc2_to_numpy(msg)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
            ok = np.isfinite(xyz).all(1)
            r = np.linalg.norm(xyz, axis=1)
            ok &= (r > min_range) & (r < max_range)
            xyz = xyz[ok]; rng = r[ok]
            if len(xyz) < 50:
                continue
            if len(fov_s) < 50:
                fov_s.append(np.degrees(np.arcsin(xyz[:, 2] / rng)))
                ring_s.append(a["ring"][ok].astype(np.int16))
            t_pt = t_sw + a["timestamp"].astype(np.float64)[ok]
            if t_pt.max() < t_traj[0] or t_pt.min() > t_traj[-1]:
                n_miss += 1
                continue
            j1 = np.clip(np.searchsorted(t_traj, t_pt), 1, len(t_traj) - 1)
            j0 = j1 - 1
            span = t_traj[j1] - t_traj[j0]
            w = np.clip((t_pt - t_traj[j0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
            pos = xyz_traj[j0] + w[:, None] * (xyz_traj[j1] - xyz_traj[j0])
            Rp = quats_to_R(slerp(q_traj[j0], q_traj[j1], w))
            world = np.einsum("nij,nj->ni", Rp, xyz) + pos
            for c in range(chunks):
                s0 = len(world) * c // chunks
                s1 = len(world) * (c + 1) // chunks
                if s1 - s0 < 5:
                    continue
                tree.insertPointCloud(pointcloud=np.ascontiguousarray(world[s0:s1]),
                                      origin=np.ascontiguousarray(pos[(s0 + s1) // 2]))
            qk.append(cell_key(np.floor(world / grid)))
            qr.append(rng.astype(np.float32))
            d = world - pos
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            qd.append(((np.sign(d[:, 0]) + 1) * 9 + (np.sign(d[:, 1]) + 1) * 3 +
                       (np.sign(d[:, 2]) + 1)).astype(np.int8))
            n_sweep += 1
            n_pts += len(world)
    tree.updateInnerOccupancy()
    log(f"    build: {n_sweep:,} sweeps, {n_pts:,} pts, dropped {n_drop}, "
        f"no-pose {n_miss}, {time.time()-t0:.1f}s")

    # probing viewpoints: every 50th pose, without the dropped window
    sub = np.arange(0, len(t_traj), 50)
    if drop is not None:
        trel = t_traj[sub] - t0_traj
        sub = sub[~((trel >= drop[0]) & (trel < drop[1]))]
    EL = np.concatenate(fov_s); RG = np.concatenate(ring_s)
    return dict(scan=scan, tree=tree, res=res, grid=grid, min_range=min_range,
                max_range=max_range, drop=drop, n_sweep=n_sweep, n_pts=n_pts,
                t_traj=t_traj, xyz_traj=xyz_traj, q_traj=q_traj,
                vp_xyz=xyz_traj[sub], vp_R=quats_to_R(q_traj[sub]), vp_t=t_traj[sub] - t0_traj,
                qk=np.concatenate(qk), qr=np.concatenate(qr), qd=np.concatenate(qd),
                EL=EL, RG=RG)


def analyse(B, radius=8.0, n_probe=12, full_domain_vp=None, box=None, log=print):
    """Domain, occupancy, frontier, ray evidence per probe, surface quality.
    full_domain_vp and box come from the full run, so that every run is judged on
    the same domain."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    tree, g = B["tree"], B["grid"]
    vp = B["vp_xyz"]
    dom_vp = vp if full_domain_vp is None else full_domain_vp
    blo, bhi = dom_vp.min(0) - radius, dom_vp.max(0) + radius
    if box is not None:
        blo, bhi = np.maximum(blo, box[0]), np.minimum(bhi, box[1])
    gl = np.floor(blo / g).astype(np.int64)
    gh = np.ceil(bhi / g).astype(np.int64)
    shape = tuple((gh - gl).tolist())
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    IJK = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], 1) + gl
    P = (IJK + 0.5) * g
    dmin, _ = cKDTree(dom_vp).query(P, k=1, workers=-1)
    keep = dmin <= radius
    state = np.full(len(P), OUTSIDE, np.int8)
    t0 = time.time()
    for k in np.flatnonzero(keep):
        n = tree.search(P[k])
        try:
            state[k] = OCCUPIED if tree.isNodeOccupied(n) else FREE
        except Exception:
            state[k] = UNKNOWN
    V = state.reshape(shape)
    st = ndimage.generate_binary_structure(3, 1)
    frontier = (V == UNKNOWN) & ndimage.binary_dilation(V == FREE, st)
    touches = ndimage.binary_dilation(V == OCCUPIED, st).ravel()
    fidx = np.flatnonzero(frontier.ravel())
    log(f"    domain {int(keep.sum()):,} vox, occ {int((state==OCCUPIED).sum()):,} "
        f"free {int((state==FREE).sum()):,} unk {int((state==UNKNOWN).sum()):,} "
        f"frontier {len(fidx):,}  ({time.time()-t0:.1f}s)")

    EL, RG = B["EL"], B["RG"]
    fov_lo, fov_hi = float(np.percentile(EL, 0.2)), float(np.percentile(EL, 99.8))
    els = np.array([np.median(EL[RG == r]) for r in np.unique(RG) if (RG == r).sum() > 20])
    beam = float(np.median(np.diff(np.sort(els))))
    r_res = g / np.tan(np.radians(beam))

    # per frontier voxel: rays from the nearest n_probe viewpoints that have it in the FOV
    kd = cKDTree(vp)
    tol = g * np.sqrt(3) / 2 + B["res"]
    pr_dist = np.full((len(fidx), n_probe), np.nan, np.float32)
    pr_hit = np.zeros((len(fidx), n_probe), bool)
    pr_reach = np.zeros((len(fidx), n_probe), bool)
    n_near = np.zeros(len(fidx), np.int32)
    t0 = time.time()
    end = np.zeros(3)
    for j, fi in enumerate(fidx):
        p = P[fi]
        near = kd.query_ball_point(p, B["max_range"])
        if not near:
            continue
        near = np.asarray(near)
        diff = p[None, :] - vp[near]
        dist = np.linalg.norm(diff, axis=1)
        ok = dist > max(B["min_range"], 1e-6)
        near, diff, dist = near[ok], diff[ok], dist[ok]
        if not len(near):
            continue
        v_s = np.einsum("nji,nj->ni", B["vp_R"][near], diff)
        el = np.degrees(np.arcsin(np.clip(v_s[:, 2] / dist, -1, 1)))
        inf = (el >= fov_lo) & (el <= fov_hi)
        cand, cd = near[inf], dist[inf]
        n_near[j] = len(cand)
        order = np.argsort(cd)[:n_probe]
        for m, oi in enumerate(order):
            o = vp[cand[oi]]; dd = float(cd[oi])
            hit = tree.castRay(o, (p - o) / dd, end, False, dd)
            pr_dist[j, m] = dd
            pr_hit[j, m] = hit
            pr_reach[j, m] = (not hit) and (np.linalg.norm(end - o) >= dd - tol)
    log(f"    probes: {len(fidx):,} frontier vox x <= {n_probe} rays ({time.time()-t0:.1f}s)")

    uk, inv = np.unique(B["qk"], return_inverse=True)
    cnt = np.bincount(inv).astype(np.int64)
    rmean = np.bincount(inv, weights=B["qr"].astype(np.float64)) / cnt
    combo = np.unique(inv.astype(np.int64) * 27 + B["qd"].astype(np.int64))
    ndir = np.bincount(combo // 27, minlength=len(uk))
    occ_idx = np.flatnonzero(state == OCCUPIED)
    pos = np.searchsorted(uk, cell_key(IJK[occ_idx]))
    pos = np.clip(pos, 0, len(uk) - 1)
    found = uk[pos] == cell_key(IJK[occ_idx])
    dens = np.where(found, cnt[pos] / g**2, 0.0)
    rm = np.where(found, rmean[pos], np.inf)
    nd = np.where(found, ndir[pos], 0)

    return dict(B=B, shape=shape, IJK=IJK, P=P, state=state, fidx=fidx, touches=touches[fidx],
                fov=(fov_lo, fov_hi), beam=beam, r_res=r_res, tol=tol,
                pr_dist=pr_dist, pr_hit=pr_hit, pr_reach=pr_reach, n_near=n_near,
                occ_idx=occ_idx, dens=dens, rmean=rm, ndir=nd)


def grade(A, q_density=2000.0, q_range=6.0, q_dirs=3):
    g = A["B"]["grid"]
    thr_good = q_density; thr_fair = max(2.0 / g**2, q_density / 4)
    d, r, n = A["dens"], A["rmean"], A["ndir"]
    gr = np.full(len(d), POOR, np.int8)
    gr[(d >= thr_fair) & (r <= 2 * q_range)] = FAIR
    gr[(d >= thr_good) & (r <= q_range) & (n >= q_dirs)] = GOOD
    return gr


def label(A, model="multi", strict=False, p_sample=0.5):
    """Frontier labels for one sampleability model and ray test."""
    g = A["B"]["grid"]
    dist, hit = A["pr_dist"], A["pr_hit"]
    clear = A["pr_reach"] if strict else (~hit & ~np.isnan(dist))
    valid = ~np.isnan(dist)
    out = np.empty(len(A["fidx"]), np.int8)
    if model == "multi":
        s = dist * np.tan(np.radians(A["beam"]))
        f = np.where(clear, 1.0 - np.minimum(1.0, g / np.where(valid, s, 1.0)), 1.0)
        p_never = np.prod(f, axis=1)
        anyclear = clear.any(1)
        out[:] = MISSED
        out[A["touches"]] = OCCLUDED
        out[p_never >= p_sample] = UNSAMPLEABLE
        out[~anyclear] = OCCLUDED
        out[~valid.any(1)] = UNSAMPLEABLE
    else:
        inr = valid & (dist <= A["r_res"])
        anyclear = (clear & inr).any(1)
        out[:] = MISSED
        out[A["touches"]] = OCCLUDED
        out[~anyclear] = OCCLUDED
        out[~inr.any(1)] = UNSAMPLEABLE
    return out


def full_state(A, flab, grades=None):
    st = A["state"].copy()
    st[A["fidx"]] = flab
    if grades is not None:
        st[A["occ_idx"]] = grades
    return st


def gaps(A, st, min_vox=8):
    from scipy import ndimage
    V = (st == MISSED).reshape(A["shape"])
    lab, n = ndimage.label(V)
    if n == 0:
        return lab.ravel(), []
    sizes = ndimage.sum(V, lab, range(1, n + 1))
    big = [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] >= min_vox]
    return lab.ravel(), big
