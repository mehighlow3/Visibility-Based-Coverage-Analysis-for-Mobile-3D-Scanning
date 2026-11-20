"""Read a scan as a stream of motion-corrected sweeps.

A sweep is what the robot would receive every 100 ms: points in the world frame,
the sensor position each point was measured from, and its range. Everything
downstream consumes this stream and nothing else, so the offline tool and an
online node differ only in where the sweeps come from.
"""
import os
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [Path(os.environ.get("P1_SCANS", "_unset_")),
               _HERE.parent.parent / "ravenScans",
               _HERE.parent.parent.parent / "ravenScans"]
SCANS = next((p for p in _CANDIDATES if p.is_dir()), _CANDIDATES[1])
TOPIC = "/vanjee_722z"
PC2 = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
       5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pc2_to_numpy(msg):
    n, f, o = [], [], []
    for fl in msg.fields:
        n.append(fl.name); f.append(PC2[fl.datatype]); o.append(fl.offset)
    dt = np.dtype({"names": n, "formats": f, "offsets": o, "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def quats_to_R(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2*(y*y + z*z); R[:, 0, 1] = 2*(x*y - z*w); R[:, 0, 2] = 2*(x*z + y*w)
    R[:, 1, 0] = 2*(x*y + z*w);     R[:, 1, 1] = 1 - 2*(x*x + z*z); R[:, 1, 2] = 2*(y*z - x*w)
    R[:, 2, 0] = 2*(x*z - y*w);     R[:, 2, 1] = 2*(y*z + x*w); R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def slerp(q0, q1, w):
    d = np.sum(q0 * q1, axis=1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.abs(d).clip(-1.0, 1.0)
    th = np.arccos(d); st = np.sin(th)
    lin = st < 1e-6
    safe = np.where(lin, 1.0, st)
    s0 = np.where(lin, 1.0 - w, np.sin((1.0 - w) * th) / safe)
    s1 = np.where(lin, w, np.sin(w * th) / safe)
    q = s0[:, None] * q0 + s1[:, None] * q1
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def trajectory(scan):
    """(t, xyz, quat) of the vendor LiDAR-IMU trajectory, 200 Hz, TUM format."""
    T = np.loadtxt(SCANS / scan / "recalculate_trj.txt")
    return T[:, 0], T[:, 1:4], T[:, 4:8]


def sweeps(scan, traj=None, drop=(), stride=1, min_range=0.4, max_range=15.0, t_end=None):
    """Yield (t_rel, world[n,3], origin[n,3], range[n]) per sweep.

    drop: iterable of (t0, t1) windows in seconds from the start of the trajectory;
          sweeps inside are skipped (used to delete part of a walk on purpose).
    """
    from rosbags.highlevel import AnyReader
    t_traj, xyz_traj, q_traj = traj if traj is not None else trajectory(scan)
    t0 = t_traj[0]
    drop = [tuple(d) for d in drop]
    bags = sorted((SCANS / scan).rglob("LIDAR_*.bag"))
    with AnyReader(bags) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        for k, (conn, _, raw) in enumerate(reader.messages(connections=conns)):
            if k % stride:
                continue
            msg = reader.deserialize(raw, conn.msgtype)
            t_sw = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            trel = t_sw - t0
            if t_end is not None and trel > t_end:
                break
            if any(a <= trel < b for a, b in drop):
                continue
            a = pc2_to_numpy(msg)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
            r = np.linalg.norm(xyz, axis=1)
            ok = np.isfinite(xyz).all(1) & (r > min_range) & (r < max_range)
            if ok.sum() < 50:
                continue
            xyz, r = xyz[ok], r[ok]
            t_pt = t_sw + a["timestamp"].astype(np.float64)[ok]
            if t_pt.max() < t_traj[0] or t_pt.min() > t_traj[-1]:
                continue
            j1 = np.clip(np.searchsorted(t_traj, t_pt), 1, len(t_traj) - 1)
            j0 = j1 - 1
            span = t_traj[j1] - t_traj[j0]
            w = np.clip((t_pt - t_traj[j0]) / np.where(span > 0, span, 1.0), 0.0, 1.0)
            pos = xyz_traj[j0] + w[:, None] * (xyz_traj[j1] - xyz_traj[j0])
            Rp = quats_to_R(slerp(q_traj[j0], q_traj[j1], w))
            world = np.einsum("nij,nj->ni", Rp, xyz) + pos
            yield trel, world, pos, r


def insert_octree(tree, world, pos, chunks=5):
    """Ray-cast one sweep into the OctoMap, in time slices with their own origin."""
    for c in range(chunks):
        s0 = len(world) * c // chunks
        s1 = len(world) * (c + 1) // chunks
        if s1 - s0 >= 5:
            tree.insertPointCloud(pointcloud=np.ascontiguousarray(world[s0:s1]),
                                  origin=np.ascontiguousarray(pos[(s0 + s1) // 2]))
