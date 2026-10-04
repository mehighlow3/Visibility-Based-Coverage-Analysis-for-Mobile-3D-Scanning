"""Figure: why 4.3 m. Top: side view of the laser lines spreading onto two walls.
Bottom: spacing between neighbouring lines vs distance, against the 20 cm voxel."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path(__file__).parent.parent / "figures"; OUT.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, BLUE_T, GREY_T, RED = "#2a78d6", "#e3eefc", "#f0efec", "#d03b3b"

BEAM = 2.66                    # deg between neighbouring rings, measured from the scan
VOX = 0.20                     # analysis voxel [m]
R_OK = VOX / np.tan(np.radians(BEAM))

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10})
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.6), dpi=150,
                             gridspec_kw={"height_ratios": [1.0, 1.35]})
fig.patch.set_facecolor(SURF)
for a in (a1, a2):
    a.set_facecolor(SURF)

# top: side view, lowest 5 of the 16 lines
near, far = 2.0, 8.0
angles = np.arange(0, 5) * BEAM
for th in angles:
    a1.plot([0, far], [0, far * np.tan(np.radians(th))], color=BLUE, lw=0.9, alpha=0.55)
for wx, lab in [(near, f"wall at {near:.0f} m"), (far, f"wall at {far:.0f} m")]:
    top = far * np.tan(np.radians(angles[-1])) + 0.15
    a1.plot([wx, wx], [-0.12, top if wx == far else 0.55], color=INK2, lw=2.2)
    hits = wx * np.tan(np.radians(angles))
    a1.plot([wx] * len(hits), hits, "o", ms=4.5, color=BLUE, mec=SURF, mew=1.2, zorder=5)
    a1.text(wx, -0.22, lab, ha="center", va="top", color=INK2, fontsize=9.5)
s_near = near * np.tan(np.radians(BEAM)); s_far = far * np.tan(np.radians(BEAM))


def bracket(x, y0, y1, text, col=INK):
    a1.plot([x, x], [y0, y1], color=col, lw=1.0)
    for y in (y0, y1):
        a1.plot([x - 0.06, x + 0.06], [y, y], color=col, lw=1.0)
    a1.text(x + 0.12, (y0 + y1) / 2, text, va="center", ha="left", color=col, fontsize=9.5)


bracket(near + 0.18, 3 * s_near, 4 * s_near, f"{100*s_near:.0f} cm")
bracket(far + 0.18, 3 * s_far, 4 * s_far, f"{100*s_far:.0f} cm")
# a 20 cm voxel on the far wall, between the 2nd and 3rd line
y0 = s_far + (s_far - VOX) / 2
a1.add_patch(Rectangle((far - VOX / 2, y0), VOX, VOX, fc="none", ec=RED, lw=1.6, zorder=6))
a1.text(far + 0.18, y0 + VOX / 2, "a 20 cm voxel fits between two lines:\nno point lands in it",
        va="center", ha="left", color=RED, fontsize=9.5)
a1.plot([0], [0], "o", ms=8, color=INK, zorder=6)
a1.text(0.12, -0.08, "sensor", va="top", color=INK, fontsize=9.5)
a1.set_xlim(-0.3, far + 3.2); a1.set_ylim(-0.45, 1.6)
a1.set_aspect("equal"); a1.axis("off")
a1.set_title("Side view: the 16 laser lines are 2.66° apart, so they spread out with distance "
             "(lowest 5 shown)", loc="left", color=INK, fontsize=11)

# bottom: spacing vs distance
d = np.linspace(0, 10, 300)
s = 100 * d * np.tan(np.radians(BEAM))
a2.axvspan(0, R_OK, color=BLUE_T, lw=0, zorder=0)
a2.axvspan(R_OK, 10, color=GREY_T, lw=0, zorder=0)
a2.plot(d, s, color=BLUE, lw=2.2, zorder=3)
a2.axhline(100 * VOX, color=INK2, lw=1.0, zorder=2)
a2.text(0.15, 100 * VOX + 1.2, "voxel size 20 cm", color=INK2, fontsize=9.5, va="bottom")
a2.plot([R_OK], [100 * VOX], "o", ms=8, color=BLUE, mec=SURF, mew=2, zorder=5)
a2.annotate(f"{R_OK:.1f} m", (R_OK, 100 * VOX), xytext=(R_OK + 0.25, 100 * VOX - 8),
            color=INK, fontsize=11, fontweight="bold")
a2.text(R_OK / 2, 38, "reliable\nevery voxel is hit\nby at least one line", ha="center",
        va="center", color=INK, fontsize=10)
a2.text((R_OK + 10) / 2, 9, "unreliable\na voxel can fall between lines:\n"
        "\"no points\" ≠ \"nothing there\"", ha="center", va="center", color=INK, fontsize=10)
slope = 100 * np.tan(np.radians(BEAM))      # cm per m
a2.text(6.2, slope * 6.2 + 1.8, "space between neighbouring lines", color=BLUE, fontsize=9.5,
        rotation=np.degrees(np.arctan(slope)), rotation_mode="anchor",
        transform_rotates_text=True, ha="left", va="bottom")
a2.set_xlim(0, 10); a2.set_ylim(0, 50)
a2.set_xlabel("distance from the sensor [m]", color=INK2)
a2.set_ylabel("space between lines [cm]", color=INK2)
for sp in ("top", "right"):
    a2.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    a2.spines[sp].set_color(AXIS)
a2.tick_params(colors=MUTED)
a2.set_title("Where that crosses the voxel size is the reliable range for a single view",
             loc="left", color=INK, fontsize=11)
fig.text(0.01, 0.005, "Smaller voxels are worse: 10 cm → 2.2 m, 5 cm → 1.1 m. "
         "Several views from different positions fill in the gaps, so this is the conservative limit.",
         color=MUTED, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(OUT / "sensor_range.png", facecolor=SURF)
print("ok", round(R_OK, 2))
