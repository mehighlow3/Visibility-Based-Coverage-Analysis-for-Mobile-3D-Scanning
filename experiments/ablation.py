"""Deletion test: remove known time windows of the walk and check what the tool
calls the edge of the area that is no longer observed.

R = voxels observed (surface or free) in the full run but not after the window's
sweeps and viewpoints are removed. For the frontier inside R the correct label is
MISSED; the test reports how much is labelled MISSED, OCCLUDED or UNSAMPLEABLE.

    python ablation.py <scan> <window_s> <start1,start2,...>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import cov_lib as L

scan = sys.argv[1]
wlen = float(sys.argv[2])
starts = [float(s) for s in sys.argv[3].split(",")]
OUT = Path(__file__).parent / f"abl_{scan}"
OUT.mkdir(exist_ok=True)
CFGS = [("multi", False), ("multi", True), ("single", False), ("single", True)]
NAME = {c: f"{c[0]}-{'strict' if c[1] else 'loose'}" for c in CFGS}
logf = open(OUT / "log.txt", "a", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    logf.write(s + "\n"); logf.flush()


def observed(st):
    return (st == L.OCCUPIED) | (st == L.FREE)


def summarize_full(A):
    res = {}
    gr = L.grade(A)
    for c in CFGS:
        lab = L.label(A, *c)
        st = L.full_state(A, lab)
        _, big = L.gaps(A, st)
        res[NAME[c]] = dict(missed=int((lab == L.MISSED).sum()),
                            occluded=int((lab == L.OCCLUDED).sum()),
                            unsampleable=int((lab == L.UNSAMPLEABLE).sum()),
                            gaps=len(big))
    n = len(gr)
    res["surface"] = dict(voxels=n, good=round(100 * (gr == L.GOOD).mean(), 1),
                          fair=round(100 * (gr == L.FAIR).mean(), 1),
                          poor=round(100 * (gr == L.POOR).mean(), 1),
                          median_density=round(float(np.median(A["dens"])), 0))
    res["sensor"] = dict(fov=[round(x, 2) for x in A["fov"]], beam_deg=round(A["beam"], 3),
                         r_res=round(A["r_res"], 2))
    res["frontier"] = int(len(A["fidx"]))
    return res


log(f"\n===== {scan}  window {wlen}s  starts {starts}  {time.ctime()}")
Bf = L.build(scan, log=log)
vp_full = Bf["vp_xyz"]
BOX = (np.asarray(Bf["tree"].getMetricMin()), np.asarray(Bf["tree"].getMetricMax()))
Af = L.analyse(Bf, full_domain_vp=vp_full, box=BOX, log=log)
full = summarize_full(Af)
log("FULL:", json.dumps(full))
np.savez_compressed(OUT / "full.npz", IJK=Af["IJK"].astype(np.int32), state=Af["state"],
                    fidx=Af["fidx"], pr_dist=Af["pr_dist"], pr_hit=Af["pr_hit"],
                    pr_reach=Af["pr_reach"], touches=Af["touches"], occ_idx=Af["occ_idx"],
                    dens=Af["dens"], rmean=Af["rmean"], ndir=Af["ndir"],
                    beam=Af["beam"], r_res=Af["r_res"], grid=Bf["grid"],
                    traj=Bf["xyz_traj"][::20], traj_t=Bf["t_traj"][::20] - Bf["t_traj"][0],
                    shape=np.array(Af["shape"]))
obsF = observed(Af["state"])
gradeF = np.full(len(Af["state"]), -1, np.int8); gradeF[Af["occ_idx"]] = L.grade(Af)
Tt = Bf["t_traj"] - Bf["t_traj"][0]
del Bf["tree"]

rows = []
for s in starts:
    w = (s, s + wlen)
    log(f"\n--- window {w[0]:.0f}-{w[1]:.0f} s")
    Ba = L.build(scan, drop=w, log=log)
    Aa = L.analyse(Ba, full_domain_vp=vp_full, box=BOX, log=log)
    assert Aa["state"].shape == Af["state"].shape
    stA = Aa["state"]
    R = obsF & ~observed(stA) & (stA != L.OUTSIDE)
    front = np.zeros(len(stA), bool); front[Aa["fidx"]] = True
    FR = front & R
    inw = (Tt >= w[0]) & (Tt < w[1])
    seg = Ba["xyz_traj"][inw]
    path = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum()) if len(seg) > 1 else 0.0
    # surfaces still observed but graded worse
    gradeA = np.full(len(stA), -1, np.int8); gradeA[Aa["occ_idx"]] = L.grade(Aa)
    both = (gradeF >= 0) & (gradeA >= 0)
    rank = {L.GOOD: 0, L.FAIR: 1, L.POOR: 2}
    rf = np.vectorize(rank.get)(gradeF[both]) if both.any() else np.array([])
    ra = np.vectorize(rank.get)(gradeA[both]) if both.any() else np.array([])
    row = dict(window=w, path_m=round(path, 1), removed_vox=int(R.sum()),
               removed_surface=int(((gradeF >= 0) & R).sum()),
               frontier_in_removed=int(FR.sum()),
               surface_downgraded=int((ra > rf).sum()), surface_both=int(both.sum()))
    for c in CFGS:
        lab = L.label(Aa, *c)
        st = L.full_state(Aa, lab)
        mis = st == L.MISSED
        occ = st == L.OCCLUDED
        uns = st == L.UNSAMPLEABLE
        labs, big = L.gaps(Aa, st)
        tp_cl = 0
        for b in big:
            m = labs == b
            if (m & R).sum() >= 0.5 * m.sum():
                tp_cl += 1
        nFR = max(int(FR.sum()), 1)
        row[NAME[c]] = dict(
            missed=int(mis.sum()), missed_in_R=int((mis & R).sum()),
            recall=round((mis & FR).sum() / nFR, 3),
            precision=round((mis & R).sum() / max(int(mis.sum()), 1), 3),
            boundary_as_occluded=round((occ & FR).sum() / nFR, 3),
            boundary_as_unsampleable=round((uns & FR).sum() / nFR, 3),
            gaps=len(big), gaps_true=tp_cl)
    log(json.dumps(row))
    rows.append(row)
    np.savez_compressed(OUT / f"win_{int(w[0])}.npz", state=stA, fidx=Aa["fidx"],
                        pr_dist=Aa["pr_dist"], pr_hit=Aa["pr_hit"], pr_reach=Aa["pr_reach"],
                        touches=Aa["touches"], R=R, occ_idx=Aa["occ_idx"],
                        gradeA=gradeA, window=np.array(w))
    del Ba["tree"]

(OUT / "summary.json").write_text(json.dumps(dict(full=full, windows=rows), indent=2))
log("\nSUMMARY  (recall = boundary of removed region labelled MISSED; "
    "precision = MISSED voxels that are truly removed)")
log(f"{'window':>10s} {'path':>6s} {'removed':>8s} {'bound':>6s} | " +
    " | ".join(f"{NAME[c]:>24s}" for c in CFGS))
for r in rows:
    cells = []
    for c in CFGS:
        d = r[NAME[c]]
        cells.append(f"R{d['recall']:.2f} P{d['precision']:.2f} g{d['gaps_true']}/{d['gaps']:<2d}")
    log(f"{r['window'][0]:>4.0f}-{r['window'][1]:<5.0f} {r['path_m']:>6.1f} {r['removed_vox']:>8d} "
        f"{r['frontier_in_removed']:>6d} | " + " | ".join(f"{x:>24s}" for x in cells))
log("FULL-scan missed / gaps: " + ", ".join(
    f"{NAME[c]} {full[NAME[c]]['missed']}/{full[NAME[c]]['gaps']}" for c in CFGS))
