"""The whole assessment on a built scan: quality -> floor map -> standpoints -> actions."""
import time

import numpy as np
from scipy import ndimage

from . import grid, plan, quality, shell

KIND_NAMES = ("closer", "enter", "explore", "no_floor", "blocked", "beyond")

# Requirement -> viewing condition. R_OF_S is read off the calibration scan
# (experiments/calibrate.py): the largest range up to which a walking operator
# with this sensor leaves >= 85 % of planar patches free of gaps at spacing s.
# 2.5 cm never reaches 85 % on the calibration scan; 1.5 m is the best it does (~82 %).
R_OF_S = {0.025: 1.5, 0.05: 4.0, 0.10: 10.0}
LIMIT_FRAC = 0.5
# What a view of a failing patch is worth: the share expected to meet the requirement
# afterwards. Measured on these two scans, not on other buildings.
#   never seen within R, or seen before and now viewed from closer than CLOSER_FRAC of
#   the closest view so far: P_FIX per class (experiments/gain.py, the two scans: flat
#   building surface 95 % and 84 %; edges 85 % and 88 %, on 10 and 3 m2; table tops 85 %
#   and 72 %; clutter 46 % and 28 %; seen before and later seen from closer, 87 % pooled);
#   seen before and viewed from no closer than before (a second pass): SAME_FRAC of that,
#   about a third (branch test, short corridor branches: 0.36-0.44; second session: 0.30;
#   replay: 60-70 % if the second pass doubles the points, under 10 % if it adds little).
CLOSER_FRAC = 0.75
P_FIX = dict(flat=0.9, edge=0.85, top=0.8, clutter=0.4)
P_FIX_CLASS = np.array([P_FIX["flat"]] * 4 + [P_FIX["edge"], P_FIX["top"], P_FIX["clutter"]])   # by shell class
SAME_FRAC = 0.4
TYPE_NAMES = shell.CLASS_NAMES      # wall, floor, ceiling, other_flat, edge | top, clutter (things)


def el_range(el_hist, p=(0.02, 0.98)):
    c = np.cumsum(el_hist) / max(el_hist.sum(), 1)
    return float(np.searchsorted(c, p[0]) - 90), float(np.searchsorted(c, p[1]) - 90)


def assess(B, s_req=0.05, R=None, max_inc_deg=60.0, radius=8.0, cell=0.20, min_gain=1.0,
           lattice_traj=None, log=print, things=False):
    """B: dict from build/session. lattice_traj: trajectory that fixes the analysis box and
    domain (pass the FULL walk when assessing a scan with part of the walk deleted).
    things: if False (default) the requirement is asked of the building only; furniture
    and clutter (scanq.shell) are left out of the total, the deficit and the places."""
    t0 = time.time()
    R = R_OF_S[s_req] if R is None else R
    Q = quality.evaluate(B["store"], max_inc_deg=max_inc_deg)      # planes only, for the floor
    tq = time.time() - t0
    ltraj = B["traj_full"] if lattice_traj is None else lattice_traj
    floor_z = grid.floor_height(Q, B["traj"][:, 2])
    sensor_h = float(np.median(B["traj"][:, 2]) - floor_z)
    lo = np.array([ltraj[:, 0].min() - radius - 1, ltraj[:, 1].min() - radius - 1, floor_z - 0.4])
    hi = np.array([ltraj[:, 0].max() + radius + 1, ltraj[:, 1].max() + radius + 1, floor_z + 6.0])
    t1 = time.time()
    G = grid.dense(B["tree"], lo, hi, B["res"])
    tg = time.time() - t1
    t1 = time.time()
    Q = quality.evaluate(B["store"], max_inc_deg=max_inc_deg, G=G)
    tq += time.time() - t1
    F = grid.floor_map(G, floor_z, B["traj"][:, :2], radius)
    # analysis domain: patches within `radius` of the walk (of the lattice trajectory)
    Fd = F["domain"] if lattice_traj is None else grid.floor_map(G, floor_z, ltraj[:, :2], radius)["domain"]
    ij = np.floor(Q["centre"][:, :2] / G["res"]).astype(np.int64) - G["gl"][:2]
    ok = ((ij >= 0) & (ij < np.array(Fd.shape))).all(1)
    indom = np.zeros(len(ij), bool)
    indom[ok] = Fd[ij[ok, 0], ij[ok, 1]]
    indom &= (Q["centre"][:, 2] > lo[2]) & (Q["centre"][:, 2] < hi[2])
    cls = shell.classify(Q, floor_z)
    thing = shell.is_thing(cls)
    graded = indom if things else indom & ~thing          # what the requirement is asked of
    defi = quality.deficient(Q, s_req) & graded
    # already seen from well inside the viewing condition and still not resolved:
    # another visit is not expected to help (material, clutter, thin or moving things)
    limit = defi & (Q["r_best"] <= LIMIT_FRAC * R)
    idx = np.flatnonzero(defi & ~limit)
    # new: never seen under the viewing condition. Seen before: only a closer view counts.
    new = ~(Q["r_best"][idx] <= R)
    Rt = np.where(new, R, CLOSER_FRAC * np.where(new, R, Q["r_best"][idx]))
    el = el_range(B["el_hist"])
    t2 = time.time()
    S, V = plan.standpoints(Q, idx, F, G, R, max_inc_deg, el, sensor_h, cell, Rt=Rt)
    kind = plan.classify(S, V)
    area = Q["area"]
    p_fix = P_FIX_CLASS[cls[idx]]
    acts, covered = plan.actions(S, V, area[idx] * p_fix, kind, Q["centre"][idx], min_gain, same_frac=SAME_FRAC)
    tp = time.time() - t2
    typ = cls[idx]
    clear = ndimage.distance_transform_edt(~F["blocked"]) * F["res"]     # m to the nearest obstacle at body height
    for i, a in enumerate(acts):
        t, full = a["t"], a["t_full"]
        ar = area[idx][t]
        ci, cj = np.clip(np.floor(np.array(a["xy"]) / F["res"]).astype(int) - F["gl"], 0, np.array(clear.shape) - 1)
        # area_m2: all failing surface in view. new: never seen within R. closer: seen
        # before, this place is closer. same: seen before from as close (a second pass).
        a.update(id=i + 1, clearance_m=round(float(clear[ci, cj]), 2), area_m2=round(float(ar.sum()), 1),
                 new_m2=round(float(ar[new[t]].sum()), 1), closer_m2=round(float(ar[~new[t] & full].sum()), 1),
                 same_m2=round(float(ar[~full].sum()), 1),
                 types={n: round(float(ar[typ[t] == k].sum()), 1) for k, n in enumerate(TYPE_NAMES)
                        if things or k < shell.N_BUILDING})
    # Outside: deficient surfaces whose only viewing positions have no floor under them
    # (the next building seen through a window). Not part of the scan's job: taken out of
    # the surface total and the deficit, reported on their own.
    outside = np.zeros(len(area), bool)
    outside[idx[kind == plan.NO_FLOOR]] = True
    if not things:
        # the things left out are classified too, only to tell which of them are outside
        # (trees and cars seen through a window are not furniture)
        ix = np.flatnonzero(quality.deficient(Q, s_req) & indom & thing & ~(Q["r_best"] <= LIMIT_FRAC * R))
        if len(ix):
            S2, V2 = plan.standpoints(Q, ix, F, G, R, max_inc_deg, el, sensor_h, cell)
            outside[ix[plan.classify(S2, V2) == plan.NO_FLOOR]] = True
    out_m2 = float(area[outside & graded].sum())
    tot = float(area[graded].sum()) - out_m2
    inside = graded & ~outside
    flat = Q["planar"]
    din = indom & ~outside                                # everything inside, things too
    def share(m):
        return round(100 * float(area[m & ~quality.deficient(Q, s_req)].sum()) / max(float(area[m].sum()), 1e-9), 1)
    Hd = shell.hidden(Q, cls, G, F, B["store"].g, use=din)
    listed = np.zeros(len(idx), bool); listed[covered >= 0] = True
    summary = dict(
        scan=B["scan"], s_req_m=s_req, range_m=R, max_incidence_deg=max_inc_deg,
        elevation_deg=list(el), floor_z=round(floor_z, 2), sensor_height_m=round(sensor_h, 2),
        surface_m2=round(tot, 1), met_m2=round(float(area[graded & ~defi].sum()), 1),
        met_pct=round(100 * float(area[graded & ~defi].sum()) / max(tot, 1e-9), 1),
        deficit_m2=round(float(area[defi].sum()) - out_m2, 1),
        deficit_by_kind=dict({n: round(float(area[idx][kind == k].sum()), 1) for k, n in enumerate(KIND_NAMES)
                              if n != "no_floor"}, limit=round(float(area[limit].sum()), 1)),
        # the same deficit, split by whether the viewing condition was ever met for the patch
        deficit_never_seen_close_m2=round(float(area[defi & inside & ~(Q["r_best"] <= R)].sum()), 1),
        deficit_seen_close_m2=round(float(area[defi & inside & (Q["r_best"] <= R)].sum()), 1),
        closer_not_listed_m2=round(float(area[idx][(kind == plan.CLOSER) & ~listed].sum()), 1),
        met_pct_flat=round(100 * float(area[inside & flat & ~defi].sum()) / max(float(area[inside & flat].sum()), 1e-9), 1),
        # building and things: area and share meeting the requirement per class, whether
        # counted or not; things_counted says if the numbers above include the things
        things_counted=bool(things),
        by_class={n: dict(m2=round(float(area[din & (cls == k)].sum()), 1), met_pct=share(din & (cls == k)))
                  for k, n in enumerate(shell.CLASS_NAMES)},
        things_m2=round(float(area[din & thing].sum()), 1), things_met_pct=share(din & thing),
        building_m2=round(float(area[din & ~thing].sum()), 1), building_met_pct=share(din & ~thing),
        # building surface with no points on it, estimated (not in surface_m2)
        hidden=dict(floor_under_things_m2=round(Hd["floor_under_things_m2"], 1),
                    wall_behind_things_m2=round(Hd["wall_behind_things_m2"], 1),
                    wall_not_seen_m2=round(Hd["wall_not_seen_m2"], 1),
                    things_footprint_m2=round(Hd["things_footprint_m2"], 1)),
        p_fix=P_FIX, closer_frac=CLOSER_FRAC, same_frac=SAME_FRAC,
        outside_m2=round(out_m2, 1),
        achieved_pct={f"{int(round(100 * l * 10)) / 10:g} cm": round(100 * float(area[inside & (Q['achieved'] == l)].sum()) / max(tot, 1e-9), 1)
                      for l in Q["levels"]},
        floor=dict(reachable_m2=round(float(F["reach"].sum()) * G["res"] ** 2, 1),
                   cut_off_m2=round(float(F["cut"].sum()) * G["res"] ** 2, 1),
                   no_floor_m2=round(float(F["nofloor"].sum()) * G["res"] ** 2, 1),
                   floor_seen_share={k: round(v, 2) for k, v in F["floor_seen_share"].items()},
                   openings=len(F["openings"]), traj_on_walk=round(F["traj_on_walk"], 3)),
        actions=len(acts), actions_gain_m2=round(sum(a["gain_m2"] for a in acts), 1),
        time_s=dict(quality=round(tq, 2), dense_grid=round(tg, 1), plan=round(tp, 1),
                    total=round(time.time() - t0, 1)))
    log(f"    assess {B['scan']}: {summary['met_pct']} % of {summary['surface_m2']} m2 {'surface' if things else 'building surface'} meets {100*s_req:g} cm; "
        f"deficit {summary['deficit_m2']} m2 {summary['deficit_by_kind']}; outside (not counted) {summary['outside_m2']} m2; {len(acts)} actions "
        f"({summary['time_s']['total']} s)")
    return dict(Q=Q, G=G, F=F, S=S, V=V, idx=idx, kind=kind, new=new, indom=indom, graded=graded, cls=cls,
                thing_cols=Hd["thing_cols"], hidden_cells=Hd.get("wall_cells", np.zeros((0, 4))), deficient=defi, limit=limit, outside=outside,
                actions=acts, covered=covered, summary=summary, B=B)
