"""Is the operator's body in the scan? Close returns (0.4-1.2 m) by direction in the
sensor frame.

    python check_body.py [scan]
"""
import sys
import numpy as np
from rosbags.highlevel import AnyReader
import cov_lib as L

sd = L.SCANS / (sys.argv[1] if len(sys.argv) > 1 else "institutGesamt")
az_all, el_all, r_all = [], [], []
n = 0
with AnyReader(sorted(sd.glob("LIDAR_*.bag"))) as reader:
    conns = [c for c in reader.connections if c.topic == L.TOPIC]
    for conn, _, raw in reader.messages(connections=conns):
        n += 1
        if n % 5:
            continue
        a = L.pc2_to_numpy(reader.deserialize(raw, conn.msgtype))
        xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
        r = np.linalg.norm(xyz, axis=1)
        ok = np.isfinite(r) & (r > 0.05)
        xyz, r = xyz[ok], r[ok]
        az_all.append(np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])))
        el_all.append(np.degrees(np.arcsin(xyz[:, 2] / r)))
        r_all.append(r)
az = np.concatenate(az_all); el = np.concatenate(el_all); r = np.concatenate(r_all)
print(f"sweeps sampled {n//5}, points {len(r):,}")
for lo, hi in [(0.05, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, 3.0)]:
    m = (r >= lo) & (r < hi)
    print(f"range {lo:.2f}-{hi:.2f} m: {100*m.mean():5.2f} % of points")
near = (r >= 0.4) & (r < 1.2)
h, edges = np.histogram(az[near], bins=36, range=(-180, 180))
hall, _ = np.histogram(az, bins=36, range=(-180, 180))
frac = h / np.maximum(hall, 1)
print("azimuth sectors (10 deg) where >10 % of returns are 0.4-1.2 m away:")
for i in np.argsort(frac)[::-1][:8]:
    print(f"   {edges[i]:+5.0f}..{edges[i+1]:+5.0f} deg : {100*frac[i]:5.1f} % near  "
          f"(median elev {np.median(el[near & (az>=edges[i]) & (az<edges[i+1])]):+.0f} deg)")

band = (r >= 0.4) & (r < 0.7)
hb, _ = np.histogram(az[band], bins=36, range=(-180, 180))
fb = hb / np.maximum(hall, 1)
print(f"0.4-0.7 m band: median sector {100*np.median(fb):.1f} %, max sector {100*fb.max():.1f} % "
      f"at {edges[fb.argmax()]:+.0f} deg; sectors > 3x median: "
      f"{[int(edges[i]) for i in np.flatnonzero(fb > 3*np.median(fb))]}")
