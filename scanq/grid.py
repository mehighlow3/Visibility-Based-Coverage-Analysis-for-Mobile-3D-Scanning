"""Dense occupancy labels from the octree, the floor map, and line of sight.

Labels: 1 occupied, 0 free (a ray passed through), -1 unknown (no ray ever).

Floor map (2-D, from the body-height band above the local floor):
    blocked   any occupied cell in the band
    open      no occupied cell and >= OPEN_MIN of the band seen free
    unknown   no occupied cell and >= UNK_MIN of the band unknown
    walk      open cells where a BODY-wide square fits
    reach     walk cells connected to the cells under the trajectory
    cut       walk cells NOT connected to the trajectory, in a region where a floor was seen
              (seen, could be stood in, never entered)
    nofloor   walk cells NOT connected to the trajectory, in a region with no floor seen
              (free space seen through a window: outside, or a void)
    opening   unknown columns next to reach, wide enough to walk into
"""
import numpy as np
from scipy import ndimage

OPEN_MIN = 0.7
UNK_MIN = 0.5
BODY_M = 0.5
MIN_WIDTH_M = 0.6
MIN_BEYOND_M2 = 1.0
BAND = (0.6, 2.0)            # m above the floor
FLOOR_TOL = 0.3              # m: a floor counts as seen if a cell within this of floor height is occupied
SUPPORT_MIN = 0.2            # share of a region's columns with a floor seen, to count as a place to stand
S4 = ndimage.generate_binary_structure(2, 1)
S8 = np.ones((3, 3), bool)


def floor_height(Q, traj_z):
    """Height of the floor: the largest horizontal area below the sensor."""
    m = (np.abs(Q["normal"][:, 2]) > 0.9) & (Q["centre"][:, 2] < np.median(traj_z) - 0.5)
    if m.sum() < 20:
        return float(np.median(traj_z) - 1.5)
    z = Q["centre"][m, 2]
    h, e = np.histogram(z, bins=np.arange(z.min() - 0.05, z.max() + 0.1, 0.05), weights=Q["area"][m])
    h = np.convolve(h, np.ones(3), "same")
    return float(e[np.argmax(h)] + 0.025)


def dense(tree, lo, hi, res):
    """Label every cell of the box [lo, hi) at the octree resolution."""
    gl = np.floor(np.asarray(lo) / res).astype(np.int64)
    gh = np.ceil(np.asarray(hi) / res).astype(np.int64)
    shape = tuple((gh - gl).tolist())
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xy = (np.stack([ii.ravel(), jj.ravel()], 1) + gl[:2] + 0.5) * res
    L = np.empty(shape, np.int8)
    for k in range(shape[2]):
        z = (gl[2] + k + 0.5) * res
        P = np.ascontiguousarray(np.column_stack([xy, np.full(len(xy), z)]))
        L[:, :, k] = np.asarray(tree.getLabels(P)).reshape(shape[:2])
    return dict(L=L, gl=gl, res=res, shape=shape)


def geodesic(mask, seed, res):
    d = np.full(mask.shape, np.inf)
    cur = seed & mask
    d[cur] = 0.0
    k = 0
    while cur.any():
        k += 1
        nxt = ndimage.binary_dilation(cur, S4) & mask & ~np.isfinite(d)
        d[nxt] = k * res
        cur = nxt
    return d


def floor_map(G, floor_z, traj_xy, radius=8.0, band=BAND):
    L, gl, res = G["L"], G["gl"], G["res"]
    k0 = int(np.floor((floor_z + band[0]) / res)) - gl[2]
    k1 = int(np.ceil((floor_z + band[1]) / res)) - gl[2]
    k0, k1 = max(k0, 0), min(k1, L.shape[2])
    S = L[:, :, k0:k1]
    nbz = k1 - k0
    n_occ = (S == 1).sum(2); n_free = (S == 0).sum(2); n_unk = nbz - n_occ - n_free
    shape = n_occ.shape
    ij = np.floor(traj_xy / res).astype(np.int64) - gl[:2]
    ij = ij[((ij >= 0) & (ij < np.array(shape))).all(1)]
    seed = np.zeros(shape, bool); seed[ij[:, 0], ij[:, 1]] = True
    domain = ndimage.distance_transform_edt(~seed) * res <= radius
    blocked = n_occ > 0
    open_ = ~blocked & (n_free >= OPEN_MIN * nbz)
    unknown = ~blocked & (n_unk >= UNK_MIN * nbz) & domain
    b = max(1, int(round(BODY_M / res)))
    walk = ndimage.binary_opening(open_, np.ones((b, b), bool))
    lab, _ = ndimage.label(walk, S4)
    ids = np.unique(lab[seed & walk]); ids = ids[ids > 0]
    reach = np.isin(lab, ids)
    # Floor support. A space the walk is not connected to can only be stood in if a floor
    # was seen under it. Free space seen through a window (the street, a courtyard, the
    # air in front of the next building) has none: the sill hides the ground, or the ground
    # is a storey lower. Decided per connected region, by the share of its columns that
    # have an occupied cell at floor height.
    f0 = max(int(np.floor((floor_z - FLOOR_TOL) / res)) - gl[2], 0)
    f1 = min(int(np.ceil((floor_z + FLOOR_TOL) / res)) - gl[2], L.shape[2])
    floor_seen = (L[:, :, f0:f1] == 1).any(2)
    off = walk & ~reach & domain
    olab, no = ndimage.label(off, S4)
    share = np.asarray(ndimage.mean(floor_seen, olab, np.arange(1, no + 1))) if no else np.zeros(0)
    cut = np.isin(olab, np.flatnonzero(share >= SUPPORT_MIN) + 1)
    nofloor = off & ~cut
    dist = geodesic(reach, seed, res)
    front = unknown & ndimage.binary_dilation(reach, S4)
    ulab, _ = ndimage.label(unknown, S4)
    usize = np.bincount(ulab.ravel())
    flab, _ = ndimage.label(front, S8)
    dnear = ndimage.grey_erosion(np.where(np.isfinite(dist), dist, 1e9), footprint=S8)
    ops = []
    for i, sl in enumerate(ndimage.find_objects(flab), 1):
        m = flab[sl] == i
        ii, jj = np.nonzero(m)
        width = (max(np.ptp(ii), np.ptp(jj)) + 1) * res
        beyond = float(usize[np.unique(ulab[sl][m])].sum()) * res * res
        if width < MIN_WIDTH_M or beyond < MIN_BEYOND_M2:
            continue
        c = (np.array([ii.mean() + sl[0].start, jj.mean() + sl[1].start]) + gl[:2] + 0.5) * res
        ops.append(dict(xy=[round(float(c[0]), 2), round(float(c[1]), 2)], width_m=round(float(width), 1),
                        beyond_m2=round(beyond, 1), walk_m=round(float(dnear[sl][m].min()), 1), _id=i))
    opening = np.isin(flab, [o["_id"] for o in ops])
    clab, nc = ndimage.label(cut, S4)
    return dict(blocked=blocked, open=open_, unknown=unknown, walk=walk, reach=reach, cut=cut,
                nofloor=nofloor, floor_seen=floor_seen,
                floor_seen_share=dict(reach=float(floor_seen[reach].mean()) if reach.any() else 0.0,
                                      cut=float(floor_seen[cut].mean()) if cut.any() else 0.0,
                                      nofloor=float(floor_seen[nofloor].mean()) if nofloor.any() else 0.0),
                cut_lab=clab, opening=opening, openings=ops, dist=dist, domain=domain, seed=seed,
                gl=gl[:2], res=res, floor_z=floor_z,
                traj_on_walk=float(walk[ij[:, 0], ij[:, 1]].mean()) if len(ij) else 0.0)


def los_blocked(G, P0, P1, skip_end=0.25, step=0.07, chunk=400_000):
    """True where the segment P0 -> P1 crosses an occupied cell (the last skip_end m
    before P1 are ignored: that is the target itself)."""
    L, gl, res = G["L"], G["gl"], G["res"]
    sh = np.array(L.shape)
    out = np.zeros(len(P0), bool)
    for s in range(0, len(P0), chunk):
        a, b = P0[s:s + chunk], P1[s:s + chunk]
        v = b - a
        d = np.linalg.norm(v, axis=1)
        u = v / np.maximum(d, 1e-9)[:, None]
        K = int(np.ceil(d.max() / step)) if len(d) else 0
        blk = np.zeros(len(a), bool)
        for k in range(1, K + 1):
            t = k * step
            act = (t < d - skip_end) & ~blk
            if not act.any():
                if t > d.max():
                    break
                continue
            p = a[act] + t * u[act]
            c = np.floor(p / res).astype(np.int64) - gl
            inside = ((c >= 0) & (c < sh)).all(1)
            hit = np.zeros(len(c), bool)
            ci = c[inside]
            hit[inside] = L[ci[:, 0], ci[:, 1], ci[:, 2]] == 1
            idx = np.flatnonzero(act)
            blk[idx[hit]] = True
        out[s:s + chunk] = blk
    return out
