"""Building or thing? And how much of the building is hidden behind the things.

The requirement is about the building: walls, floor, ceiling. A scan of a used building
is full of furniture and clutter, which the plane model of quality.py grades badly and
which nobody wants rescanned. This module separates the two with rules on the patches
(no learned model), and estimates the building surface that received no points because
a thing stands on it or in front of it.

Classes per patch:
    wall         flat, vertical                       (cupboard fronts and doors too)
    floor        flat, horizontal, at floor height
    ceiling      flat, horizontal, above 2.2 m
    other_flat   flat, tilted                         (reveals, frames, sloped parts)
    edge         not flat, where two building surfaces meet (wall-floor, wall-ceiling,
                 two walls): the patch holds two planes
    top          flat, horizontal, 0.3-2.2 m above the floor: table tops, seats, shelves
    clutter      not flat and not an edge: chairs, plants, cables, lamps, shelves with things
The first five are BUILDING, the last two are THINGS.

What the rules cannot tell: a cupboard front from a wall (both flat and vertical), a
window sill, a step or a ceiling lower than 2.2 m from a table top (all flat and
horizontal at mid height), a wall whose upper part was not scanned from a low thing.
"""
import numpy as np
from scipy import ndimage

WALL, FLOOR, CEILING, OTHER_FLAT, EDGE, TOP, CLUTTER = range(7)
CLASS_NAMES = ("wall", "floor", "ceiling", "other_flat", "edge", "top", "clutter")
N_BUILDING = 5                      # classes below this are building
EDGE_MIN = 2                        # neighbours of each of the two surfaces
OFFS26 = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
THING_TOP_M = 2.0                   # a thing ends below this height; what goes on above is a wall
WALL_SPAN = (2.0, 3.2)              # m above the floor: a column occupied over this span is a wall
WALL_SPAN_MIN = 0.4                 # ... share of the span occupied
THING_DEPTH_M = 1.5                 # a thing column lies within this of floor a person fits on
FRONT_M = 1.0                       # "behind a thing": something occupied within this in front


def _key(q):
    q = q + (1 << 19)
    return (q[:, 0] << 40) | (q[:, 1] << 20) | q[:, 2]


class Lookup:
    """Row of the patch at a lattice cell, -1 if there is none."""
    def __init__(self, ijk):
        k = _key(ijk)
        self.o = np.argsort(k); self.k = k[self.o]

    def __call__(self, q):
        if not len(self.k):
            return np.full(len(q), -1)
        k = _key(q)
        pos = np.clip(np.searchsorted(self.k, k), 0, len(self.k) - 1)
        return np.where(self.k[pos] == k, self.o[pos], -1)


def classify(Q, floor_z):
    """Class per patch (see the module text)."""
    nz = np.abs(Q["normal"][:, 2]); h = Q["centre"][:, 2] - floor_z
    flat = Q["planar"]
    c = np.full(len(nz), OTHER_FLAT, np.int8)
    c[nz < 0.35] = WALL
    c[(nz > 0.8) & (h < 0.3)] = FLOOR
    c[(nz > 0.8) & (h >= 0.3) & (h < 2.2)] = TOP
    c[(nz > 0.8) & (h >= 2.2)] = CEILING
    c[~flat] = CLUTTER
    # edges: a not-flat patch with at least EDGE_MIN neighbours of each of two building surfaces
    look = Lookup(Q["ijk"])
    N = len(c)
    n_floor = np.zeros(N); n_ceil = np.zeros(N); n_wall = np.zeros(N)
    S = np.zeros((N, 2, 2))                 # scatter of the horizontal normals of wall neighbours
    hn = Q["normal"][:, :2] / np.maximum(np.linalg.norm(Q["normal"][:, :2], axis=1), 1e-9)[:, None]
    for d in OFFS26:
        r = look(Q["ijk"] + np.array(d))
        ok = r >= 0
        t = np.where(ok, c[np.maximum(r, 0)], -1)
        n_floor += t == FLOOR; n_ceil += t == CEILING
        w = t == WALL
        n_wall += w
        v = np.where(w[:, None], hn[np.maximum(r, 0)], 0.0)
        S += v[:, :, None] * v[:, None, :]
    two_walls = np.linalg.eigvalsh(S)[:, 0] >= EDGE_MIN          # two wall directions, each >= EDGE_MIN strong
    edge = ((n_wall >= EDGE_MIN) & ((n_floor >= EDGE_MIN) | (n_ceil >= EDGE_MIN))) | two_walls
    c[~flat & edge] = EDGE
    return c


def is_thing(cls):
    return cls >= N_BUILDING


def columns(G, F):
    """2-D masks at the grid resolution: thing (a low object stands there, next to floor a
    person fits on) and wall (occupied well above head height)."""
    L, gl, res = G["L"], G["gl"], G["res"]
    fz = F["floor_z"]
    def k(z):
        return int(np.clip(np.round((fz + z) / res) - gl[2], 0, L.shape[2]))
    occ = L == 1
    low = occ[:, :, k(0.3):k(THING_TOP_M)].any(2)
    # (a thin wall wanders between neighbouring columns over its height: count a layer as
    # occupied if the column or one next to it is)
    span = ndimage.maximum_filter(occ[:, :, k(WALL_SPAN[0]):k(WALL_SPAN[1])], size=(3, 3, 1))
    wall = span.mean(2) >= WALL_SPAN_MIN
    cand = low & ~wall
    # things stand in the room: connected, through thing columns and gaps too narrow to walk,
    # to floor a person fits on, and not deeper than a piece of furniture
    room = F["reach"] | F["cut"]
    near = ndimage.distance_transform_edt(~room) * res <= THING_DEPTH_M
    region = (cand | (F["open"] & ~F["walk"])) & near
    lab, _ = ndimage.label(region | room)
    ids = np.unique(lab[room]); ids = ids[ids > 0]
    thing = cand & near & np.isin(lab, ids)
    return dict(thing=thing, wall=wall)


def hidden(Q, cls, G, F, g, use=None):
    """Building surface that received no points, estimated. use: patches to start from
    (those inside the analysis domain).
    floor: columns where a thing stands and no floor was seen under it.
    wall:  cells of a wall column (20 cm lattice) with no patch, between the floor and the
           top of the wall seen in that column, and not observed free (free = an opening).
           Split by whether something stands in front of the cell (behind a thing) or not.
    Returns areas in m2 and the 2-D thing mask for the map."""
    L, gl, res = G["L"], G["gl"], G["res"]
    sh = np.array(L.shape)
    C = columns(G, F)
    under = C["thing"] & ~F["floor_seen"]
    out = dict(thing_cols=C["thing"], floor_under_things_m2=float(under.sum()) * res * res,
               things_footprint_m2=float(C["thing"].sum()) * res * res)
    sel = (cls == WALL) & (Q["dom"] < 2)
    w = np.flatnonzero(sel if use is None else sel & use)
    if not len(w):
        return dict(out, wall_behind_things_m2=0.0, wall_not_seen_m2=0.0, wall_too_few_points_m2=0.0)
    ijk = Q["ijk"][w]
    kf = int(np.floor((F["floor_z"] + 0.05) / g))            # lowest lattice layer above the floor
    # wall columns: (i, j, dominant axis); per column the top layer, a point and the normal
    ck = (ijk[:, 0].astype(np.int64) << 22) ^ (ijk[:, 1].astype(np.int64) << 2) ^ Q["dom"][w]
    u, inv = np.unique(ck, return_inverse=True)
    ktop = np.full(len(u), -10 ** 9); np.maximum.at(ktop, inv, ijk[:, 2])
    cnt = np.bincount(inv)
    cxy = np.column_stack([np.bincount(inv, Q["centre"][w, a]) / cnt for a in (0, 1)])
    nrm = np.column_stack([np.bincount(inv, Q["normal"][w, a]) / cnt for a in (0, 1)])
    first = np.zeros(len(u), np.int64); first[inv] = np.arange(len(w))
    col_ij = ijk[first, :2]; col_dom = Q["dom"][w][first]
    # a thin wall seen from both sides: the normals cancel. Use the lattice axis and look
    # for something in front on either side.
    both = np.linalg.norm(nrm, axis=1) < 0.5
    nrm[both] = np.eye(2)[col_dom[both]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
    def key2(ij, dom):
        return ((ij[:, 0] + (1 << 19)) << 22) | ((ij[:, 1] + (1 << 19)) << 2) | dom
    look = Lookup(Q["ijk"])
    behind = 0.0; unseen = 0.0; sparse = 0.0; pts = []
    use = cnt >= 2                                           # a wall, not a single patch
    for k in range(kf, int(ktop.max()) + 1):
        m = use & (ktop > k)
        if not m.any():
            continue
        q = np.column_stack([col_ij[m], np.full(int(m.sum()), k)])
        # no patch here, nor in the cell in front or behind (a wall on a cell boundary
        # falls into one or the other from layer to layer)
        step = np.zeros((int(m.sum()), 3), np.int64)
        step[np.arange(len(step)), col_dom[m]] = 1
        empty = (look(q) < 0) & (look(q + step) < 0) & (look(q - step) < 0)
        # the same wall can form two neighbouring columns (cell boundary): count the cell once
        flagged = key2(q[empty, :2], col_dom[m][empty])
        empty &= ~np.isin(key2(q[:, :2] - step[:, :2], col_dom[m]), flagged)
        if not empty.any():
            continue
        p = np.column_stack([cxy[m][empty], np.full(int(empty.sum()), (k + 0.5) * g)])
        n2 = np.column_stack([nrm[m][empty], np.zeros(int(empty.sum()))])
        def lab(pp):
            c = np.floor(pp / res).astype(np.int64) - gl
            ok = ((c >= 0) & (c < sh)).all(1)
            o = np.full(len(pp), -1, np.int8)
            o[ok] = L[c[ok, 0], c[ok, 1], c[ok, 2]]
            return o
        # observed free at the wall position (at either half of the cell's height): an opening
        free = (lab(p + [0, 0, -0.05]) == 0) | (lab(p + [0, 0, 0.05]) == 0)
        front = np.zeros(len(p), bool)
        two = both[m][empty]
        for t in np.arange(0.2, FRONT_M + 1e-9, 0.1):
            front |= lab(p + t * n2) == 1
            front |= two & (lab(p - t * n2) == 1)
        # points there, but too few for a graded patch: seen, not hidden
        some = (lab(p + [0, 0, -0.05]) == 1) | (lab(p + [0, 0, 0.05]) == 1)
        sparse += float((~free & some).sum()) * g * g
        behind += float((~free & ~some & front).sum()) * g * g
        unseen += float((~free & ~some & ~front).sum()) * g * g
        pts.append(np.column_stack([p, np.where(free, 0, np.where(some, 1, np.where(front, 2, 3)))]))
    return dict(out, wall_behind_things_m2=behind, wall_not_seen_m2=unseen, wall_too_few_points_m2=sparse,
                wall_cells=np.concatenate(pts) if pts else np.zeros((0, 4)))
