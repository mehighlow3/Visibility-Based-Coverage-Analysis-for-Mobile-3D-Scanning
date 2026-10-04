"""What are the 'clear view within reach, still empty' frontier voxels physically?
Lists the largest clusters and saves the camera frames from when the sensor was
closest to one of them.

    python inspect_blue.py [cluster_index]
"""
import sys, io, tempfile
from pathlib import Path
import numpy as np
from scipy import ndimage
from rosbags.highlevel import AnyReader
from PIL import Image

H = Path(__file__).parent
scan = "institutGesamt"
F = np.load(H / f"abl_{scan}" / "full.npz")
SD = H.resolve().parent / scan
g = float(F["grid"]); IJK = F["IJK"]; shape = tuple(F["shape"])
fidx = F["fidx"]; dist = F["pr_dist"].astype(float); reach = F["pr_reach"]
touch = F["touches"]; r_res = float(F["r_res"])
d_clear = np.where(reach, dist, np.inf).min(1)
blue = ~touch & np.isfinite(d_clear) & (d_clear <= r_res)
print(f"blue frontier voxels: {blue.sum()}  (of {len(fidx)})")

V = np.zeros(int(np.prod(shape)), bool); V[fidx[blue]] = True
lab, n = ndimage.label(V.reshape(shape), ndimage.generate_binary_structure(3, 3))
sizes = ndimage.sum(V.reshape(shape), lab, range(1, n + 1))
order = np.argsort(sizes)[::-1]
P = (IJK + 0.5) * g
T = np.loadtxt(SD / "recalculate_trj.txt")
tt, txyz = T[:, 0], T[:, 1:4]
zc = np.median(txyz[:, 2])
print(f"median sensor z {zc:.2f}")
print("largest clusters: size, center, extent, z-range rel. sensor, closest clear view, closest path")
for li in order[:8]:
    m = (lab.ravel() == li + 1)
    pts = P[m]
    c = pts.mean(0)
    ext = pts.max(0) - pts.min(0) + g
    dpath = np.linalg.norm(txyz - c, axis=1)
    sel = np.isin(fidx, np.flatnonzero(m))
    print(f"  {int(sizes[li]):4d}  {np.round(c,1)}  {np.round(ext,1)}  "
          f"z {pts[:,2].min()-zc:+.1f}..{pts[:,2].max()-zc:+.1f}  "
          f"clear {d_clear[sel].min():.1f} m  path {dpath.min():.1f} m @ t+{tt[dpath.argmin()]-tt[0]:.0f}s")

li = order[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
m = (lab.ravel() == li + 1)
c = P[m].mean(0)
i0 = int(np.linalg.norm(txyz - c, axis=1).argmin())
t_star = tt[i0]
# camera frames show people: written to the temp folder, outside the repository
out = Path(tempfile.gettempdir()) / "p1_blue_check"; out.mkdir(exist_ok=True)
with AnyReader(sorted(SD.glob("IMAGE_*.bag"))) as r:
    best = {}
    for conn, ts, raw in r.messages(start=int((t_star - 1) * 1e9), stop=int((t_star + 1) * 1e9)):
        msg = r.deserialize(raw, conn.msgtype)
        th = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if conn.topic not in best or abs(th - t_star) < best[conn.topic][0]:
            best[conn.topic] = (abs(th - t_star), bytes(msg.data))
for k, (dt, data) in best.items():
    im = Image.open(io.BytesIO(data)); im.thumbnail((1200, 1200))
    im.save(out / (k.strip("/").split("/")[0] + ".jpg"), quality=85)
print("cluster", np.round(c, 1), "-> frames at t+%.0fs" % (t_star - tt[0]), list(best))
