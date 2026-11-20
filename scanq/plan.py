"""From deficient patches to actions: where would the sensor have to be?

A requirement on a patch (point spacing <= s) is turned into a viewing condition
the sensor can meet: range <= R(s), incidence <= max_inc, inside the elevation
range the sensor covers, with a free line of sight. The set of floor cells that
satisfy it for a patch is that patch's STANDPOINT SET. Its relation to the floor
map decides the action:

    closer    a standpoint on floor reachable from the walk   -> stand there
    enter     only on walkable floor cut off from the walk    -> get into that space
    no_floor  only in free space with no floor seen under it  -> not a place to stand: the patch
                                                                 is outside a window or across a void
    explore   only in space never observed                    -> unknown, go and look
    blocked   candidates exist, all lines of sight obstructed -> go around / move something
    beyond    no floor cell within range and angle at all     -> not reachable with this sensor
                                                                 from the floor (height, range)

Summing the area of deficient patches over the standpoint sets gives, for every
floor cell, the surface area a scan from there would bring up to the requirement.
That field is the utility of a next-best-view planner; the action list is its
greedy maximisation (set cover).
"""
import numpy as np
from scipy import sparse

from . import grid as G_

KINDS = ("closer", "enter", "explore")
REACH, CUT, UNK, AIR = 1, 2, 3, 4
# patch classes, in order of precedence: what is known beats what is not
CLOSER, ENTER, EXPLORE, NO_FLOOR, BLOCKED, BEYOND = 0, 1, 2, 3, 4, 5


def stand_cells(F, cell=0.20):
    """Coarse standpoint lattice from the floor map: class and walking distance per cell."""
    k = max(1, int(round(cell / F["res"])))
    sh = (F["reach"].shape[0] // k, F["reach"].shape[1] // k)
    def pool(a, f):
        return f(a[:sh[0] * k, :sh[1] * k].reshape(sh[0], k, sh[1], k), axis=(1, 3))
    need = (k * k + 1) // 2
    cls = np.zeros(sh, np.int8)
    cls[pool(F["unknown"], np.sum) >= need] = UNK
    cls[pool(F["nofloor"], np.sum) >= need] = AIR
    # a place to stand: every part of the cell is floor a person fits on. (With a majority
    # vote the cell centre could lie on a blocked column next to a wall.)
    cls[pool(F["cut"], np.sum) == k * k] = CUT
    cls[pool(F["reach"], np.sum) == k * k] = REACH
    dist = pool(F["dist"], np.min)
    return dict(cls=cls, dist=dist, k=k, cell=k * F["res"], gl=F["gl"] // k if (F["gl"] % k == 0).all() else None,
                origin=F["gl"] * F["res"])


def standpoints(Q, idx, F, G, R, max_inc_deg=60.0, el=(-30.0, 65.0), sensor_h=1.7, cell=0.20,
                chunk=1500, log=None, Rt=None):
    """Visibility of target patches idx from the standpoint lattice.
    Returns (S, V): V is a sparse bool matrix cells x targets, true where the cell views
    the target within R, under the incidence limit, with a free line of sight.
    Rt: a tighter range limit per target (a patch already seen within R: closer than
    before). S["Vc"] is V restricted to the pairs within Rt."""
    S = stand_cells(F, cell)
    cls = S["cls"]; c = S["cell"]; org = S["origin"]
    sh = cls.shape
    zs = F["floor_z"] + sensor_h
    Rt = np.full(len(idx), float(R)) if Rt is None else np.minimum(np.asarray(Rt, float), R)
    rad = int(np.ceil(R / c))
    oi, oj = np.meshgrid(np.arange(-rad, rad + 1), np.arange(-rad, rad + 1), indexing="ij")
    keep = oi ** 2 + oj ** 2 <= rad ** 2
    oi, oj = oi[keep], oj[keep]
    cmin = np.cos(np.radians(max_inc_deg))
    centre, normal, two = Q["centre"][idx], Q["normal"][idx], Q["two_sided"][idx]
    rows, cols, near = [], [], []
    geo = np.zeros(len(idx), bool)          # some cell meets range/angle (ignoring line of sight)
    for s in range(0, len(idx), chunk):
        cc, nn, tt = centre[s:s + chunk], normal[s:s + chunk], two[s:s + chunk]
        ci = np.floor((cc[:, 0] - org[0]) / c).astype(np.int64)[:, None] + oi[None, :]
        cj = np.floor((cc[:, 1] - org[1]) / c).astype(np.int64)[:, None] + oj[None, :]
        ok = (ci >= 0) & (ci < sh[0]) & (cj >= 0) & (cj < sh[1])
        ti, k = np.nonzero(ok)
        ci, cj = ci[ti, k], cj[ti, k]
        k_cls = cls[ci, cj]
        m = k_cls > 0
        ti, ci, cj = ti[m], ci[m], cj[m]
        P0 = np.column_stack([org[0] + (ci + 0.5) * c, org[1] + (cj + 0.5) * c, np.full(len(ci), zs)])
        v = cc[ti] - P0
        r = np.linalg.norm(v, axis=1)
        cosi = -np.einsum("ni,ni->n", v, nn[ti]) / np.maximum(r, 1e-9)
        cosi = np.where(tt[ti], np.abs(cosi), cosi)
        elev = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        m = (r <= R) & (r >= 0.4) & (cosi >= cmin) & (elev >= el[0]) & (elev <= el[1])
        ti, ci, cj, P0, r = ti[m], ci[m], cj[m], P0[m], r[m]
        geo[s + np.unique(ti)] = True
        blk = G_.los_blocked(G, P0, cc[ti])
        ti, ci, cj, r = ti[~blk], ci[~blk], cj[~blk], r[~blk]
        rows.append(ci * sh[1] + cj); cols.append(ti + s); near.append(r <= Rt[s + ti])
        if log and s % (chunk * 10) == 0:
            log(f"      standpoints {s:,}/{len(idx):,}")
    rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    cols = np.concatenate(cols) if cols else np.zeros(0, np.int64)
    near = np.concatenate(near) if near else np.zeros(0, bool)
    shape = (sh[0] * sh[1], len(idx))
    V = sparse.csr_matrix((np.ones(len(rows), bool), (rows, cols)), shape=shape)
    # Vc: the pairs that are also within the target's own range limit (a better view than before)
    S["Vc"] = sparse.csr_matrix((np.ones(int(near.sum()), bool), (rows[near], cols[near])), shape=shape)
    S["geo"] = geo
    return S, V


def classify(S, V):
    """Class per target patch: the best kind of floor its standpoint set reaches.
    closer > enter > no_floor > explore; with no standpoint at all: blocked or beyond."""
    cls = S["cls"].ravel()
    Vc = V.tocsr()
    kind = np.full(V.shape[1], BLOCKED, np.int8)
    for code, cl in ((EXPLORE, UNK), (NO_FLOOR, AIR), (ENTER, CUT), (CLOSER, REACH)):
        has = np.asarray(Vc[np.flatnonzero(cls == cl)].sum(0)).ravel() > 0
        kind[has] = code
    kind[(kind == BLOCKED) & ~S["geo"]] = BEYOND
    return kind


def actions(S, V, area, kind, centre, min_gain=1.0, zone_m=0.6, max_per_kind=(20, 15, 10), same_frac=None):
    """Greedy set cover, class by class. Each action is a zone of cells around the best one.
    area: what a target is worth (expected area fixed by a full view).
    same_frac: if given, a view that is within R but not within the target's own limit
    (S["Vc"]) is worth this fraction of a full view, and leaves the rest to be claimed.
    Each action carries t (indices of its targets) and t_full (viewed within their limit)."""
    cls = S["cls"].ravel(); sh = S["cls"].shape; c = S["cell"]; org = S["origin"]
    remaining = area.copy().astype(float)
    Vc = V if same_frac is None else S["Vc"]
    a = 1.0 if same_frac is None else float(same_frac)
    Vf = V.astype(np.float32) if same_frac is None else (a * V.astype(np.float32) + (1 - a) * Vc.astype(np.float32))
    zr = int(round(zone_m / c))
    out = []
    covered_by = np.full(len(area), -1, np.int32)
    for code, cl in ((0, REACH), (1, CUT), (2, UNK)):
        cells = cls == cl
        w = np.where(kind == code, remaining, 0.0)
        n0 = len(out)
        while len(out) - n0 < max_per_kind[code]:
            gain = Vf @ w
            gain[~cells] = 0
            b = int(np.argmax(gain))
            if gain[b] < min_gain:
                break
            bi, bj = divmod(b, sh[1])
            i0, i1 = max(bi - zr, 0), min(bi + zr + 1, sh[0]); j0, j1 = max(bj - zr, 0), min(bj + zr + 1, sh[1])
            zi, zj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            z = (zi * sh[1] + zj).ravel()
            z = z[cells[z] & ((zi.ravel() - bi) ** 2 + (zj.ravel() - bj) ** 2 <= zr ** 2)]
            vis = np.asarray(V[z].sum(0)).ravel() > 0
            full = np.asarray(Vc[z].sum(0)).ravel() > 0
            got = vis & (w > 0)
            wg = w[got] * np.where(full[got], 1.0, a)
            g = float(wg.sum())
            tc = (centre[got] * wg[:, None]).sum(0) / max(g, 1e-9)
            xy = [float(org[0] + (bi + 0.5) * c), float(org[1] + (bj + 0.5) * c)]
            d = float(S["dist"][bi, bj])
            out.append(dict(kind=KINDS[code], xy=[round(xy[0], 2), round(xy[1], 2)],
                            gain_m2=round(g, 1), n_patches=int(got.sum()),
                            walk_m=round(d, 1) if np.isfinite(d) else None,
                            target=[round(float(t), 2) for t in tc],
                            target_z=[round(float(centre[got, 2].min()), 2), round(float(centre[got, 2].max()), 2)],
                            cell=[int(bi), int(bj)], t=np.flatnonzero(got), t_full=full[got]))
            # a target belongs to the first action that views it fully, else to the first that views it
            covered_by[got & (full | (covered_by < 0))] = len(out) - 1
            w[got & full] = 0.0
            w[got & ~full] *= 1 - a
    # list order: kind first (what can be done now comes first), then gain
    order = sorted(range(len(out)), key=lambda i: (KINDS.index(out[i]["kind"]), -out[i]["gain_m2"]))
    new = np.empty(len(out), np.int32); new[order] = np.arange(len(out))
    covered_by[covered_by >= 0] = new[covered_by[covered_by >= 0]]
    return [out[i] for i in order], covered_by


def utility(S, V, area, kind=None, code=None):
    w = area if code is None else np.where(kind == code, area, 0.0)
    U = np.asarray(V.astype(np.float32) @ w).reshape(S["cls"].shape)
    U[S["cls"] == AIR] = 0.0            # nowhere to stand
    return U
