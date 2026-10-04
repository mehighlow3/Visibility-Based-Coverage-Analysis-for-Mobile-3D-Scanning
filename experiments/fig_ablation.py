"""Figure: what the tool calls the edge of a deleted stretch of the walk.

    python fig_ablation.py [scan ...]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

H = Path(__file__).parent
scans = sys.argv[1:] or ["institutGesamt"]
SURF, INK, INK2, MUTED, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
C_OK, C_OCC, C_UNS = "#2a78d6", "#a3a29c", "#d9d8d2"
KEY = "multi-strict"                       # the rule coverage.py uses
LABEL = {"institutGesamt": "Building scan, 24 min\n60 s of walking deleted ×4"}

rows = []
for s in scans:
    W = json.loads((H / f"abl_{s}" / "summary.json").read_text())["windows"]
    wt = np.array([w["frontier_in_removed"] for w in W], float)
    avg = lambda f: float((np.array([f(w) for w in W]) * wt).sum() / wt.sum())
    rows.append((LABEL.get(s, s), avg(lambda w: w[KEY]["recall"]),
                 avg(lambda w: w[KEY]["boundary_as_occluded"]),
                 avg(lambda w: w[KEY]["boundary_as_unsampleable"])))

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10.5})
fig, ax = plt.subplots(figsize=(9, 1.4 + 0.75 * len(rows)), dpi=150)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
gap = 0.004
for y, (lab, mis, occ, uns) in enumerate(rows):
    x = 0
    for val, col in [(mis, C_OK), (occ, C_OCC), (uns, C_UNS)]:
        if val > 0.0005:
            ax.barh(y, max(val - gap, 0.002), left=x, height=0.56, color=col, lw=0)
        x += val
    ax.text(mis + 0.012, y, f"missed {100*mis:.0f} %", va="center", ha="left",
            color=INK, fontsize=10, fontweight="bold")
    ax.text(0.985, y, f"occluded {100*occ:.0f} %", va="center", ha="right",
            color="#fcfcfb", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=INK2)
ax.set_ylim(len(rows) - 0.4, -0.6)
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["0", "25", "50", "75", "100 %"], color=MUTED)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="y", length=0)
ax.set_title("Part of the walk deleted on purpose: what does the tool call the forgotten area?",
             loc="left", color=INK, fontsize=12, pad=12)
ax.legend(handles=[Patch(color=C_OK, label="missed (correct answer: you didn't go there)"),
                   Patch(color=C_OCC, label="occluded (\"hidden behind something\")")],
          loc="upper left", bbox_to_anchor=(0, 0), borderaxespad=2.4, ncol=2, frameon=False,
          fontsize=9)
ax.annotate("Share of the forgotten area's edge, by label.", (0, 0), xycoords="axes fraction",
            xytext=(0, -50), textcoords="offset points", color=MUTED, fontsize=8, va="top")
fig.tight_layout()
name = "ablation.png" if len(scans) > 1 else f"ablation_{scans[0]}.png"
fig.savefig(H.parent / "figures" / name, facecolor=SURF, bbox_inches="tight", pad_inches=0.2)
print("ok", [(r[0].split(",")[0], round(100 * r[1], 1), round(100 * r[2], 1)) for r in rows])
