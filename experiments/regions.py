"""Prototype of the two-layer output: one record per region, no claim about why
something is missing.

  surface    clusters of poorly sampled surface voxels -> rescan closer / from another side
  boundary   clusters of frontier voxels by what it would take to see them
      go_around     no clear line of sight from anywhere on the walked path
      go_closer     clear line of sight only from beyond grid / tan(beam)
      sensor_limit  clear line of sight within reach, still no data
Frontier voxels next to a surface are the back of that surface and are not reported.

    python regions.py [scan] [min_voxels]
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import ndimage

H = Path(__file__).parent
scan = sys.argv[1] if len(sys.argv) > 1 else "institutGesamt"
MIN_VOX = int(sys.argv[2]) if len(sys.argv) > 2 else 8
F = np.load(H / f"abl_{scan}" / "full.npz")
g = float(F["grid"]); IJK = F["IJK"]; shape = tuple(F["shape"]); P = (IJK + 0.5) * g
r_res = float(F["r_res"])
fidx = F["fidx"]; dist = F["pr_dist"].astype(float); reach = F["pr_reach"]; touch = F["touches"]
d_clear = np.where(reach, dist, np.inf).min(1)
n_views = (~np.isnan(dist)).sum(1); n_clear = reach.sum(1)

kind = np.full(len(fidx), "", object)
kind[~touch & ~np.isfinite(d_clear)] = "go_around"
kind[~touch & np.isfinite(d_clear) & (d_clear > r_res)] = "go_closer"
kind[~touch & np.isfinite(d_clear) & (d_clear <= r_res)] = "sensor_limit"

# surface grades, same thresholds as coverage.py at 0.20 m
dens, rm, nd = F["dens"], F["rmean"], F["ndir"]; occ = F["occ_idx"]
poor = ~((dens >= 500) & (rm <= 12))
REQ = 2000.0

N = int(np.prod(shape))
st26 = ndimage.generate_binary_structure(3, 3)
regions = []


def clusters(mask_flat):
    lab, n = ndimage.label(mask_flat.reshape(shape), st26)
    return lab.ravel(), n


# boundary layer
for k, action in [("go_around", "needs a viewpoint the path never offered"),
                  ("go_closer", f"approach within {r_res:.1f} m with line of sight"),
                  ("sensor_limit", "in reach and in view but empty - check material "
                                   "(glass, dark, reflective) or a person in the way")]:
    sel = kind == k
    M = np.zeros(N, bool); M[fidx[sel]] = True
    lab, n = clusters(M)
    pos = {v: i for i, v in enumerate(fidx.tolist())}
    for li in range(1, n + 1):
        cells = np.flatnonzero(lab == li)
        if len(cells) < MIN_VOX:
            continue
        j = np.array([pos[c] for c in cells])
        pts = P[cells]
        dc = d_clear[j]
        regions.append({
            "layer": "boundary", "kind": k,
            "center": np.round(pts.mean(0), 2).tolist(),
            "extent_m": np.round(pts.max(0) - pts.min(0) + g, 2).tolist(),
            "voxels": int(len(cells)), "volume_m3": round(len(cells) * g ** 3, 3),
            "view": {"closest_clear_m": None if not np.isfinite(dc.min()) else round(float(dc.min()), 2),
                     "reach_m": round(r_res, 2),
                     "views_probed_median": int(np.median(n_views[j])),
                     "clear_views_median": int(np.median(n_clear[j]))},
            "action": action})

# surface layer
M = np.zeros(N, bool); M[occ[poor]] = True
lab, n = clusters(M)
opos = {v: i for i, v in enumerate(occ.tolist())}
for li in range(1, n + 1):
    cells = np.flatnonzero(lab == li)
    if len(cells) < MIN_VOX:
        continue
    j = np.array([opos[c] for c in cells])
    pts = P[cells]
    md = float(np.median(dens[j])); mr = float(np.median(rm[j][np.isfinite(rm[j])])) if np.isfinite(rm[j]).any() else None
    regions.append({
        "layer": "surface", "kind": "poorly_sampled",
        "center": np.round(pts.mean(0), 2).tolist(),
        "extent_m": np.round(pts.max(0) - pts.min(0) + g, 2).tolist(),
        "voxels": int(len(cells)), "area_m2_approx": round(len(cells) * g ** 2, 2),
        "sampling": {"density_median_pts_m2": round(md), "required_pts_m2": REQ,
                     "deficit": round(REQ / max(md, 1.0), 2),
                     "range_median_m": None if mr is None else round(mr, 1),
                     "directions_median": int(np.median(nd[j]))},
        "action": "rescan closer" if (mr or 0) > 6 else "rescan from another side"})

regions.sort(key=lambda r: -r["voxels"])
for i, r in enumerate(regions, 1):
    r["id"] = i
summary = {"scan": scan, "grid_m": g, "reach_m": round(r_res, 2), "min_voxels": MIN_VOX,
           "frontier_voxels": int(len(fidx)),
           "frontier_back_of_surface": int(touch.sum()),
           "frontier_by_kind": {k: int((kind == k).sum()) for k in ["go_around", "go_closer", "sensor_limit"]},
           "regions_by_kind": {}}
for r in regions:
    summary["regions_by_kind"][r["kind"]] = summary["regions_by_kind"].get(r["kind"], 0) + 1
(H / f"regions_{scan}.json").write_text(json.dumps({"summary": summary, "regions": regions}, indent=1))
print(json.dumps(summary, indent=1))
for k in ["go_around", "go_closer", "sensor_limit", "poorly_sampled"]:
    ex = [r for r in regions if r["kind"] == k][:2]
    for r in ex:
        print(json.dumps(r))
