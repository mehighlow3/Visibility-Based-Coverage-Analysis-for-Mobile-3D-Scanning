"""What is physically at a reported gap? Top and front view of the vendor cloud
around it with the gap voxels overlaid, and the camera frames from when the sensor
was closest.

    python inspect_gap.py [scan] [rank1,rank2,...]
"""
import sys, json, io, tempfile
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.highlevel import AnyReader
from PIL import Image

SCAN = sys.argv[1] if len(sys.argv) > 1 else "institutGesamt"
RANK = [int(r) for r in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1]
ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / SCAN
OD = ROOT / "out" / SCAN
# camera frames show people: written to the temp folder, outside the repository
HERE = Path(tempfile.gettempdir()) / f"p1_gap_{SCAN}"
HERE.mkdir(exist_ok=True)

gaps = json.loads((OD / "gaps.json").read_text())
sel = [g for g in gaps if g["rank"] in RANK]
C = np.mean([g["center"] for g in sel], axis=0)
print("gap center (mean of ranks", RANK, "):", np.round(C, 2))


def read_ply_ascii(p):
    with open(p) as fh:
        n = 0
        while True:
            line = fh.readline()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            if line.startswith("end_header"):
                break
        a = np.loadtxt(fh, max_rows=n)
    return a[:, :3], a[:, 3:6].astype(int)


# voxel classes by their colour in coverage_voxels.ply
VX, VC = read_ply_ascii(OD / "coverage_voxels.ply")
red = (VC[:, 0] == 230) & (VC[:, 1] == 40)
blue = (VC[:, 0] == 90) & (VC[:, 1] == 110)
grey = (VC[:, 0] == 130) & (VC[:, 1] == 130)

pcd = o3d.io.read_point_cloud(str(SD / "recalculate_result.pcd"))
P = np.asarray(pcd.points)
H = 3.0
m = np.all(np.abs(P[:, :2] - C[:2]) < H, axis=1)
Q = P[m]
print(f"vendor cloud: {len(P):,} pts, {len(Q):,} within ±{H} m (xy) of gap")

T = np.loadtxt(SD / "recalculate_trj.txt")
tt, txyz = T[:, 0], T[:, 1:4]
d = np.linalg.norm(txyz[:, :2] - C[:2], axis=1)
i0 = int(np.argmin(d))
print(f"closest approach: {d[i0]:.2f} m at t+{tt[i0]-tt[0]:.1f} s")
d3 = np.linalg.norm(txyz - C, axis=1)
print(f"closest 3D approach: {d3.min():.2f} m; per-gap closest (3D):")
for g in sel:
    print(f"   rank {g['rank']}: {np.linalg.norm(txyz - np.array(g['center']), axis=1).min():.2f} m")
near = d < 4.0
print(f"time within 4 m of gap: {near.sum()/200:.1f} s")

fig, axs = plt.subplots(1, 2, figsize=(14, 6.5))
ax = axs[0]
zb = (Q[:, 2] > C[2] - 1.2) & (Q[:, 2] < C[2] + 1.2)
ax.scatter(Q[zb, 0], Q[zb, 1], s=0.05, c="k", alpha=0.3, label="vendor cloud (slice)")
mv = np.all(np.abs(VX[:, :2] - C[:2]) < H, axis=1)
for msk, col, lab in [(grey, "0.55", "occluded"), (blue, "royalblue", "sampling-limited"),
                      (red, "red", "candidate missed")]:
    k = msk & mv
    ax.scatter(VX[k, 0], VX[k, 1], s=6, c=col, label=lab, marker="s")
tn = np.all(np.abs(txyz[:, :2] - C[:2]) < H, axis=1)
ax.plot(txyz[tn, 0], txyz[tn, 1], ".", ms=1, c="green", label="sensor path")
ax.set_aspect("equal"); ax.set_title(f"{SCAN} — top view around gap ranks {RANK}")
ax.legend(loc="upper right", markerscale=4, fontsize=8)
ax.set_xlim(C[0]-H, C[0]+H); ax.set_ylim(C[1]-H, C[1]+H)

# front view: projected on the main horizontal axis of the gap
G = VX[red & mv]
if len(G) > 3:
    u, s, vt = np.linalg.svd(G[:, :2] - G[:, :2].mean(0))
    a = vt[0]
else:
    a = np.array([1.0, 0.0])
nrm = np.array([-a[1], a[0]])
ax = axs[1]
off = (Q[:, :2] - C[:2]) @ nrm
along = (Q[:, :2] - C[:2]) @ a
sl = np.abs(off) < 0.35
ax.scatter(along[sl], Q[sl, 2], s=0.1, c=off[sl], cmap="coolwarm", alpha=0.6)
galong = (VX[red & mv, :2] - C[:2]) @ a
ax.scatter(galong, VX[red & mv, 2], s=14, c="red", marker="s", label="candidate missed")
ax.set_aspect("equal"); ax.set_xlabel("along gap sheet [m]"); ax.set_ylabel("z [m]")
ax.set_title("front view: vendor points within ±0.35 m of the sheet plane")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "gap_views.png", dpi=110)
print("wrote", HERE / "gap_views.png")

t_star = tt[i0]
bag = sorted(SD.glob("IMAGE_*.bag"))
best = {}
with AnyReader(bag) as r:
    for conn, ts, raw in r.messages(start=int((t_star - 1.0) * 1e9),
                                    stop=int((t_star + 1.0) * 1e9)):
        msg = r.deserialize(raw, conn.msgtype)
        th = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        k = conn.topic
        if k not in best or abs(th - t_star) < best[k][0]:
            best[k] = (abs(th - t_star), msg.format, bytes(msg.data))
for k, (dt, fmt, data) in best.items():
    im = Image.open(io.BytesIO(data))
    im.thumbnail((1400, 1400))
    name = k.strip("/").split("/")[0] + ".jpg"
    im.save(HERE / name, quality=85)
    print(f"{k}: dt={dt:.3f}s format={fmt} -> {name} {im.size}")
