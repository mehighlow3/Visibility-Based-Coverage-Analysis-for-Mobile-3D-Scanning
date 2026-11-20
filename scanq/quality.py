"""Achieved sampling per surface patch, read from a PatchStore.

The measure is the largest point spacing the patch is free of gaps at, not a
density. A spinning multi-line LiDAR puts points close together along a line and
far apart across lines, so points per m^2 can be high while a strip between two
lines has nothing in it. Sampling conditions for reconstruction (epsilon-sampling)
bound the gap, so that is what is measured.

Per patch:
  * a plane (centroid, normal) from the moments of the patch and its 26 neighbours;
  * the patch's sub-cells (nb x nb, projected along the dominant axis of the normal)
    that the plane passes through inside this cell: the EXPECTED sub-cells;
  * which of them received a point (the cell's own mask, OR-ed with the two cells
    in front and behind along that axis, so a wall lying on a cell boundary is not
    split in two);
  * expected sub-cells in observed free space are dropped: the surface ends there;
  * fill = hit / expected at sub-cell sizes g/nb, 2g/nb, 4g/nb, g (2.5, 5, 10, 20 cm);
  * achieved spacing = the finest size with fill >= FILL_OK.
"""
import numpy as np

from .store import DIRS

FILL_OK = 0.9
PLANE_SIGMA = 0.04          # m, a patch thicker than this is not treated as one plane
MIN_EXPECTED = 8            # sub-cells; smaller plane fragments are slivers and are skipped
OFFS27 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
PLANE_AXES = {0: (1, 2), 1: (0, 2), 2: (0, 1)}      # mask d is indexed [a * nb + b]


def _plane(M):
    """centroid, eigenvalues (ascending), eigenvectors from moment rows."""
    n = np.maximum(M[:, 0], 1.0)
    c = M[:, 1:4] / n[:, None]
    C = np.empty((len(M), 3, 3))
    C[:, 0, 0] = M[:, 4] / n - c[:, 0] ** 2
    C[:, 0, 1] = C[:, 1, 0] = M[:, 5] / n - c[:, 0] * c[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = M[:, 6] / n - c[:, 0] * c[:, 2]
    C[:, 1, 1] = M[:, 7] / n - c[:, 1] ** 2
    C[:, 1, 2] = C[:, 2, 1] = M[:, 8] / n - c[:, 1] * c[:, 2]
    C[:, 2, 2] = M[:, 9] / n - c[:, 2] ** 2
    w, v = np.linalg.eigh(C)
    return c, np.maximum(w, 0.0), v, C


def evaluate(store, n_min=3, n_min_nb=20, max_inc_deg=60.0, G=None):
    """Per-patch quality. Returns a dict of arrays over the patches kept.
    G: dense occupancy labels (grid.dense); if given, expected sub-cells that lie in
    observed free space are dropped, so the end of a surface is not counted as a gap."""
    V = store.view()
    g, nb = store.g, store.nb
    M, ijk = V["M"], V["ijk"]
    N = len(M)
    nbr = store.neighbours(OFFS27)
    Mn = np.zeros((N, 10))
    for c in range(27):
        r = nbr[:, c]
        ok = r >= 0
        Mn[ok] += M[r[ok], :10]

    c_own, w_own, v_own, C_own = _plane(M)
    c_nb, w_nb, v_nb, _ = _plane(Mn)
    n_own = M[:, 0]
    use_nb = (np.sqrt(w_nb[:, 0]) < PLANE_SIGMA) & (np.sqrt(w_nb[:, 1]) > 0.05)
    use_own = ~use_nb & (n_own >= 20) & (np.sqrt(w_own[:, 0]) < PLANE_SIGMA) & (np.sqrt(w_own[:, 1]) > 0.02)
    planar = use_nb | use_own
    normal = np.where(use_own[:, None], v_own[:, :, 0], v_nb[:, :, 0])
    pc = np.where(use_own[:, None], c_own, c_nb)                  # a point on the plane
    # normal points toward the side the patch was seen from
    flip = np.einsum("ni,ni->n", normal, M[:, 10:13]) > 0
    normal = np.where(flip[:, None], -normal, normal)
    dom = np.argmax(np.abs(normal), axis=1)

    # expected sub-cells and hits
    E = np.zeros((N, nb, nb), bool)
    H = np.zeros((N, nb, nb), bool)
    FR = np.zeros((N, nb, nb), bool)
    sub =(np.arange(nb) + 0.5) / nb
    rows = np.arange(N)
    for d, (a, b) in PLANE_AXES.items():
        m = dom == d
        if not m.any():
            continue
        A = (ijk[m, a][:, None] + sub[None, :]) * g               # (n, nb)
        B = (ijk[m, b][:, None] + sub[None, :]) * g
        nd = normal[m, d]
        h = pc[m, d][:, None, None] - (normal[m, a][:, None, None] * (A[:, :, None] - pc[m, a][:, None, None])
                                       + normal[m, b][:, None, None] * (B[:, None, :] - pc[m, b][:, None, None])) / nd[:, None, None]
        lo = (ijk[m, d] * g)[:, None, None]
        e = (h >= lo) & (h < lo + g)
        if G is not None:
            # a sub-cell the plane would pass through, but that a ray crossed (octree: free),
            # has no surface in it: the surface ends there. Not a gap.
            P = np.empty(h.shape + (3,))
            P[..., d] = h
            P[..., a] = np.broadcast_to(A[:, :, None], h.shape)
            P[..., b] = np.broadcast_to(B[:, None, :], h.shape)
            # (the plane estimate can sit a centimetre inside the free cell in front of
            # the wall, so "free" also requires no occupied cell one step either way
            # along the dominant axis)
            c = np.floor(P / G["res"]).astype(np.int64) - G["gl"]
            sh = np.array(G["L"].shape)
            inside = ((c >= 0) & (c < sh)).all(-1) & (c[..., d] >= 1) & (c[..., d] < sh[d] - 1)
            free = np.zeros(h.shape, bool)
            ci = c[inside]
            step = np.zeros(3, np.int64); step[d] = 1
            cm, cp = ci - step, ci + step
            L = G["L"]
            free[inside] = ((L[ci[:, 0], ci[:, 1], ci[:, 2]] == 0)
                            & (L[cm[:, 0], cm[:, 1], cm[:, 2]] != 1) & (L[cp[:, 0], cp[:, 1], cp[:, 2]] != 1))
            FR[m] = free
        E[m] = e
        hit = V["mask"][m, d].copy()
        for s in (-1, 1):
            off = [0, 0, 0]; off[d] = s
            r = nbr[m, OFFS27.index(tuple(off))]
            ok = r >= 0
            hit[ok] |= V["mask"][r[ok], d]
        H[m] = hit.reshape(-1, nb, nb)
    if G is not None:
        # ...and only where the surrounding 10 cm block has no point at all. OctoMap turns
        # cells of a surface seen at a grazing angle "free" (more rays pass than end there);
        # those cells do hold points, so they stay expected.
        kb = max(1, int(round(0.10 / (g / nb))))
        if nb % kb == 0:
            blk = H.reshape(N, nb // kb, kb, nb // kb, kb).any(axis=(2, 4))
            blk = np.repeat(np.repeat(blk, kb, 1), kb, 2)
        else:
            blk = H
        E &= ~(FR & ~blk)
    HE = H & E
    n_exp = E.reshape(N, -1).sum(1)
    fills = np.zeros((N, 4))
    levels = []
    k = 1
    for li in range(4):
        if nb % k:
            break
        Ek = E.reshape(N, nb // k, k, nb // k, k).any(axis=(2, 4)).reshape(N, -1)
        Hk = HE.reshape(N, nb // k, k, nb // k, k).any(axis=(2, 4)).reshape(N, -1)
        fills[:, li] = Hk.sum(1) / np.maximum(Ek.sum(1), 1)
        levels.append(g * k / nb)
        k *= 2
    levels = np.array(levels)
    okf = fills[:, :len(levels)] >= FILL_OK
    achieved = np.where(okf.any(1), levels[np.argmax(okf, axis=1)], np.inf)

    # views: closest range among directions within the incidence limit
    cosv = -(DIRS @ normal.T).T                                   # (N, ND): >0 seen from the front
    vr = V["vr"]
    seen = np.isfinite(vr)
    front = seen & (cosv > 0.3)
    back = seen & (cosv < -0.3)
    two_sided = front.any(1) & back.any(1)
    cmin = np.cos(np.radians(max_inc_deg))
    good_dir = seen & (np.abs(cosv) >= cmin)
    r_best = np.where(good_dir, vr, np.inf).min(1)
    r_min = vr.min(1)
    cos_best = np.where(seen, np.abs(cosv), 0.0).max(1)

    sigma = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", normal, C_own, normal), 0.0))
    sigma = np.where(n_own >= 10, sigma, np.nan)
    area = n_exp * (g / nb) ** 2 / np.maximum(np.abs(normal[rows, dom]), 0.5)

    keep = (n_own >= n_min) & (Mn[:, 0] >= n_min_nb) & (n_exp >= MIN_EXPECTED)
    out = dict(row=rows, ijk=ijk, n=n_own, centre=np.where((n_own >= 1)[:, None], c_own, pc),
               normal=normal, planar=planar, dom=dom, n_exp=n_exp, fills=fills, hitmask=H,
               achieved=achieved, area=area, sigma=sigma, two_sided=two_sided,
               r_best=r_best, r_min=r_min, cos_best=cos_best, density=n_own / np.maximum(area, 1e-9))
    out = {k: v[keep] for k, v in out.items()}
    out["levels"] = levels
    return out


def deficient(Q, s_req):
    """Patches that do not reach the required spacing."""
    return Q["achieved"] > s_req + 1e-9
