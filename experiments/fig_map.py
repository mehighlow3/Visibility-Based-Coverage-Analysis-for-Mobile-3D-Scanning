"""Figure: top view of a full run. Left: surface quality. Right: frontier by what it
would take to see it.

    python fig_map.py <scan>
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

scan = sys.argv[1]
H = Path(__file__).parent
F = np.load(H / f"abl_{scan}" / "full.npz")
OUT = H.parent / "figures"; OUT.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
C_GOOD, C_WARN, C_SER = "#0ca30c", "#fab219", "#ec835a"
C_BLUE, C_ORANGE, C_GREY = "#2a78d6", "#eb6834", "#c3c2b7"

g = float(F["grid"]); IJK = F["IJK"]; P = (IJK + 0.5) * g
traj = F["traj"]
zc = np.median(traj[:, 2])
slab = (P[:, 2] > zc - 1.6) & (P[:, 2] < zc + 0.8)       # floor to about head height

# surface grades, same rule as coverage.py
dens, rm, nd = F["dens"], F["rmean"], F["ndir"]
grade = np.full(len(dens), 2)
grade[(dens >= 500) & (rm <= 12)] = 1
grade[(dens >= 2000) & (rm <= 6) & (nd >= 3)] = 0
occ = F["occ_idx"]

# frontier: -1 next to a surface (not drawn), 0 no clear view, 1 only from beyond r_res, 2 within r_res
fidx = F["fidx"]; dist = F["pr_dist"].astype(float); reach = F["pr_reach"]
touch = F["touches"]; r_res = float(F["r_res"])
d_clear = np.where(reach, dist, np.inf).min(1)
fcat = np.full(len(fidx), -1)
fcat[~touch & ~np.isfinite(d_clear)] = 0
fcat[~touch & np.isfinite(d_clear) & (d_clear > r_res)] = 1
fcat[~touch & np.isfinite(d_clear) & (d_clear <= r_res)] = 2

# floor height: densest occupied layer in the lower half
zs = P[F["occ_idx"], 2]
hz, ez = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + g, g))
floor = ez[np.argmax(hz[: len(hz) // 2])] + g / 2
C_VIOLET = "#4a3aa7"

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), dpi=150, sharex=True, sharey=True)
fig.patch.set_facecolor(SURF)
ms = 1.6 if len(P) > 2e6 else 3.2
for ax in axs:
    ax.set_facecolor(SURF); ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)

ax = axs[0]
o = occ[slab[occ]]
go = grade[slab[occ]]
for k, col in [(1, C_WARN), (2, C_SER), (0, C_GOOD)]:
    m = go == k
    ax.scatter(P[o[m], 0], P[o[m], 1], s=ms, c=col, marker="s", lw=0)
ax.plot(traj[:, 0], traj[:, 1], color=INK, lw=0.7, alpha=0.8)
n = len(go)
ax.set_title("Surfaces: how well was each one sampled?", loc="left", color=INK, fontsize=12)
ax.legend(handles=[Line2D([], [], ls="", marker="s", ms=7, color=C_GOOD,
                          label=f"good  {100*(grade==0).mean():.0f} %"),
                   Line2D([], [], ls="", marker="s", ms=7, color=C_WARN,
                          label=f"fair  {100*(grade==1).mean():.0f} %"),
                   Line2D([], [], ls="", marker="s", ms=7, color=C_SER,
                          label=f"poor  {100*(grade==2).mean():.0f} %"),
                   Line2D([], [], color=INK, lw=1, label="walked path")],
          loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8.5, frameon=False)

ax = axs[1]
f_in = slab[fidx]
o = occ[slab[occ]]
low = P[o, 2] < floor + 0.3                          # floor very faint, walls and furniture grey
ax.scatter(P[o[low], 0], P[o[low], 1], s=ms * 0.6, c="#efeee9", marker="s", lw=0)
ax.scatter(P[o[~low], 0], P[o[~low], 1], s=ms * 0.6, c="#c3c2b7", marker="s", lw=0)
m = f_in & (fcat == 0)
ax.scatter(P[fidx[m], 0], P[fidx[m], 1], s=ms, c=C_VIOLET, marker="s", lw=0, alpha=0.18)
for k, col in [(1, C_ORANGE), (2, C_BLUE)]:
    m = f_in & (fcat == k)
    ax.scatter(P[fidx[m], 0], P[fidx[m], 1], s=ms * 2.4, c=col, marker="s", lw=0)
ax.plot(traj[:, 0], traj[:, 1], color=INK, lw=0.7, alpha=0.8)
ax.set_title("Edge of observed space: what would it take to see it?", loc="left",
             color=INK, fontsize=12)
ax.legend(handles=[Line2D([], [], ls="", marker="s", ms=7, color="#c3c2b7",
                          label="walls and furniture (for orientation)"),
                   Line2D([], [], ls="", marker="s", ms=7, color=C_VIOLET, alpha=0.35,
                          label=f"no clear view from the path  ({(fcat==0).sum():,})"),
                   Line2D([], [], ls="", marker="s", ms=7, color=C_ORANGE,
                          label=f"clear view only from > {r_res:.1f} m: go closer  ({(fcat==1).sum():,})"),
                   Line2D([], [], ls="", marker="s", ms=7, color=C_BLUE,
                          label=f"clear view within {r_res:.1f} m, still empty  ({(fcat==2).sum():,})"),
                   Line2D([], [], color=INK, lw=1, label="walked path")],
          loc="upper left", bbox_to_anchor=(0, -0.06), ncol=2, fontsize=8.5, frameon=False)
fig.text(0.01, 0.01, f"{scan}  |  top view of the layer from the floor to about 2.3 m  |  "
         f"{g:.2f} m voxels  |  right: counts for the whole scan; the {(fcat==-1).sum():,} "
         f"frontier voxels directly next to a surface are not drawn", color=MUTED, fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / f"map_{scan}.png", facecolor=SURF, bbox_inches="tight", pad_inches=0.25)
print("ok", (fcat == 0).sum(), (fcat == 1).sum(), (fcat == 2).sum())
