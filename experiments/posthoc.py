"""Does any per-voxel measure separate the edge of a deleted stretch from the rest
of the frontier? Reads the arrays saved by ablation.py and writes posthoc.json.

Per frontier voxel of a deletion run:
  y        lies in R (observed in the full run, lost after the deletion)
  touches  next to an occupied voxel
  d_view   distance to the nearest viewpoint with the voxel in its FOV
  d_clear  distance to the nearest viewpoint whose ray reaches it
Reports AUC per measure and the share of R per frontier category.

    python posthoc.py <scan>
"""
import sys, json
from pathlib import Path
import numpy as np

scan = sys.argv[1]
D = Path(__file__).parent / f"abl_{scan}"
F = np.load(D / "full.npz")
r_res = float(F["r_res"])
fr_full = set(F["fidx"].tolist())


def auc(score, y):
    """Mann-Whitney AUC; a higher score should mean 'in R'."""
    score = np.asarray(score, float); y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n1 = y.sum(); n0 = (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


rows = []
for f in sorted(D.glob("win_*.npz"), key=lambda p: int(p.stem.split("_")[1])):
    W = np.load(f)
    fidx = W["fidx"]; R = W["R"]
    y = R[fidx]
    dist = W["pr_dist"].astype(float)
    reach = W["pr_reach"]; hit = W["pr_hit"]
    touches = W["touches"]
    valid = ~np.isnan(dist)
    d_view = np.where(valid, dist, np.inf).min(1)
    d_clear = np.where(reach, dist, np.inf).min(1)
    n_clear = reach.sum(1)
    new = np.array([i not in fr_full for i in fidx.tolist()])

    cat = np.full(len(fidx), "", object)
    cat[touches] = "touches surface"
    nt = ~touches
    cat[nt & ~np.isfinite(d_clear)] = "no clear view"
    cat[nt & np.isfinite(d_clear) & (d_clear > r_res)] = f"clear view only > {r_res:.1f} m"
    cat[nt & np.isfinite(d_clear) & (d_clear <= r_res)] = f"clear view <= {r_res:.1f} m"
    cats = {}
    for c in np.unique(cat):
        m = cat == c
        cats[c] = dict(n=int(m.sum()), in_R=int((m & y).sum()),
                       share_of_R_boundary=round((m & y).sum() / max(y.sum(), 1), 3),
                       precision=round((m & y).sum() / max(m.sum(), 1), 3))
    dv = np.where(np.isfinite(d_view), d_view, 99.0)
    dc = np.where(np.isfinite(d_clear), d_clear, 99.0)
    row = dict(window=W["window"].tolist(), frontier=int(len(fidx)), R_boundary=int(y.sum()),
               new_frontier=int(new.sum()), new_in_R=int((new & y).sum()),
               base_rate=round(float(y.mean()), 4),
               auc_d_view=round(auc(dv, y), 3), auc_d_clear=round(auc(dc, y), 3),
               auc_not_touching=round(auc((~touches).astype(float), y), 3),
               auc_few_clear=round(auc(-n_clear, y), 3),
               categories=cats)
    rows.append(row)
    print(json.dumps(row, indent=1))

(D / "posthoc.json").write_text(json.dumps(rows, indent=2))
