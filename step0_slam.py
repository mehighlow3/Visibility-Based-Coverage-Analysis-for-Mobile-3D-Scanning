"""LiDAR-only trajectory baseline with KISS-ICP. The pipeline uses the RayStudio
LiDAR-IMU trajectory (recalculate_trj.txt) instead.

Output out\\<scan>\\: poses.npy (N,4,4), map.ply, traj.ply, step0.json

    python step0_slam.py --scan institutGesamt [--max-frames 300] [--max-range 30]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCANS_ROOT = HERE
OUT_ROOT = HERE / "out"
DEFAULT_SCAN = "institutGesamt"
TOPIC = "/vanjee_722z"

PC2_DTYPES = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
              5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}


def pc2_to_numpy(msg):
    names, formats, offsets = [], [], []
    for f in msg.fields:
        names.append(f.name)
        formats.append(PC2_DTYPES[f.datatype])
        offsets.append(f.offset)
    dt = np.dtype({"names": names, "formats": formats,
                   "offsets": offsets, "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)


def write_ply(path, xyz):
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(xyz)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        np.savetxt(fh, xyz, fmt="%.4f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=DEFAULT_SCAN)
    ap.add_argument("--root", default=str(SCANS_ROOT))
    ap.add_argument("--out", default=str(OUT_ROOT))
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--min-range", type=float, default=0.4,
                    help="drops returns from the device / the operator")
    ap.add_argument("--voxel-size", type=float, default=0.25)
    args = ap.parse_args()

    from rosbags.highlevel import AnyReader
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig

    scan_dir = Path(args.root) / args.scan
    bags = sorted(scan_dir.rglob("LIDAR_*.bag")) or sorted(scan_dir.rglob("*.bag"))
    if not bags:
        print(f"[FAIL] no bag files in {scan_dir}")
        return 1

    out_dir = Path(args.out) / args.scan
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"STEP 0 — SLAM   scan={args.scan}")
    print("=" * 70)
    print(f"bags        : {[b.name for b in bags]}")
    print(f"range       : {args.min_range} .. {args.max_range} m")
    print(f"voxel_size  : {args.voxel_size} m")

    cfg = KISSConfig()
    cfg.data.max_range = args.max_range
    cfg.data.min_range = args.min_range
    cfg.data.deskew = True
    cfg.mapping.voxel_size = args.voxel_size
    odom = KissICP(cfg)

    n_in = n_used = 0
    ts_kind = None
    pose_list = []
    t0 = time.time()

    with AnyReader(bags) as reader:
        conns = [c for c in reader.connections if c.topic == TOPIC]
        if not conns:
            print(f"[FAIL] no topic {TOPIC}")
            return 1
        total = sum(c.msgcount for c in conns)
        limit = args.max_frames or total
        print(f"sweeps      : {total}  (processing {min(limit, total)})\n")

        for conn, _, raw in reader.messages(connections=conns):
            if n_in >= limit:
                break
            n_in += 1
            a = pc2_to_numpy(reader.deserialize(raw, conn.msgtype))

            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
            ok = np.isfinite(xyz).all(1) & ~(np.abs(xyz) < 1e-6).all(1)
            xyz = xyz[ok]
            if len(xyz) < 100:
                continue

            if "timestamp" in a.dtype.names:
                ts = a["timestamp"][ok].astype(np.float64)
                span = ts.max() - ts.min()
                if ts_kind is None:
                    ts_kind = f"per-point, span {span:.6g}"
                ts = (ts - ts.min()) / span if span > 0 else np.zeros(len(ts))
            else:
                ts = np.zeros(len(xyz))
                ts_kind = ts_kind or "none (deskew disabled in effect)"

            odom.register_frame(xyz, ts)
            # KissICP keeps only the last pose
            pose_list.append(np.asarray(odom.last_pose, dtype=np.float64).copy())
            n_used += 1

            if n_used % 200 == 0:
                p = pose_list[-1][:3, 3]
                print(f"  {n_used:5d}/{min(limit,total)}   pos "
                      f"[{p[0]:7.2f} {p[1]:7.2f} {p[2]:6.2f}]   "
                      f"{time.time()-t0:5.1f}s")

    dt = time.time() - t0
    poses = np.asarray(pose_list)
    if len(poses) == 0:
        print("[FAIL] no poses — SLAM did not run")
        return 1

    xyz_traj = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(xyz_traj, axis=0), axis=1)
    path_len = float(steps.sum())
    net = float(np.linalg.norm(xyz_traj[-1] - xyz_traj[0]))

    cloud = np.asarray(odom.local_map.point_cloud())

    print(f"\n{'-'*70}")
    print(f"poses           : {len(poses)}   (from {n_used} registered sweeps)")
    print(f"timestamps      : {ts_kind}")
    print(f"path length     : {path_len:.2f} m")
    print(f"start -> end    : {net:.2f} m")
    print(f"step per sweep  : median {np.median(steps)*100:.1f} cm   max {steps.max()*100:.1f} cm")
    print(f"walking speed   : {path_len / (n_used/10.0):.2f} m/s")
    print(f"path extent     : x {xyz_traj[:,0].min():.1f}..{xyz_traj[:,0].max():.1f}   "
          f"y {xyz_traj[:,1].min():.1f}..{xyz_traj[:,1].max():.1f}   "
          f"z {xyz_traj[:,2].min():.1f}..{xyz_traj[:,2].max():.1f}")
    print(f"map             : {len(cloud):,} points")
    print(f"time            : {dt:.1f} s  ({n_used/dt:.1f} sweeps/s)")

    np.save(out_dir / "poses.npy", poses)
    write_ply(out_dir / "map.ply", cloud)
    write_ply(out_dir / "traj.ply", xyz_traj)
    meta = {
        "scan": args.scan,
        "frames_read": n_in, "frames_registered": n_used,
        "max_range": args.max_range, "min_range": args.min_range,
        "voxel_size": args.voxel_size, "deskew": True,
        "timestamps": ts_kind,
        "path_length_m": path_len, "net_displacement_m": net,
        "median_step_cm": float(np.median(steps) * 100),
        "map_points": int(len(cloud)), "runtime_s": dt,
        "kiss_icp": "1.3.0",
    }
    (out_dir / "step0.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nsaved to {out_dir}")
    for f in ("poses.npy", "map.ply", "traj.ply", "step0.json"):
        p = out_dir / f
        print(f"   {f:<12} {p.stat().st_size/1e6:8.2f} MB")
    print("\n" + "=" * 70)
    print("STEP 0 OK — baseline trajectory written.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
