"""Reads a Raven ROS1 bag without ROS: lists the topics, reads the first
PointCloud2 message and writes it to out\\<scan>_scan0.ply.

    python check_bag.py --scan institutGesamt [--topic /vanjee_722z]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCANS_ROOT = HERE
DEFAULT_SCAN = "institutGesamt"
DEFAULT_TOPIC = "/vanjee_722z"
OUT_DIR = HERE / "out"

PC2_DTYPES = {
    1: np.int8,    2: np.uint8,
    3: np.int16,   4: np.uint16,
    5: np.int32,   6: np.uint32,
    7: np.float32, 8: np.float64,
}


def pc2_to_numpy(msg):
    """PointCloud2 message -> numpy structured array."""
    names, formats, offsets = [], [], []
    for f in msg.fields:
        if f.datatype not in PC2_DTYPES:
            raise ValueError(f"unknown datatype {f.datatype} for field '{f.name}'")
        names.append(f.name)
        formats.append(PC2_DTYPES[f.datatype])
        offsets.append(f.offset)

    dt = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": msg.point_step,
    })
    n = msg.width * msg.height
    return np.frombuffer(bytes(msg.data), dtype=dt, count=n)


def write_ply(path, xyz, intensity=None):
    """ASCII .ply (CloudCompare, MeshLab, Open3D)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with open(path, "w", encoding="ascii") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {n}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        if intensity is not None:
            fh.write("property float intensity\n")
        fh.write("end_header\n")
        if intensity is None:
            for p in xyz:
                fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        else:
            for p, i in zip(xyz, intensity):
                fh.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {i:.2f}\n")


def find_bags(scan_dir):
    bags = sorted(scan_dir.rglob("*.bag"))
    if not bags:
        bags = sorted(scan_dir.rglob("*.bag.active"))
    return bags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default=DEFAULT_SCAN)
    ap.add_argument("--topic", default=DEFAULT_TOPIC)
    ap.add_argument("--root", default=str(SCANS_ROOT))
    args = ap.parse_args()

    print("=" * 70)
    print("P1 FEASIBILITY CHECK")
    print("=" * 70)

    try:
        from rosbags.highlevel import AnyReader
    except ImportError:
        print("\n[FAIL] the 'rosbags' library is not installed.")
        print("       run:  pip install rosbags numpy")
        return 1
    print("\n[1/5] rosbags library           OK")

    scan_dir = Path(args.root) / args.scan
    if not scan_dir.is_dir():
        print(f"\n[FAIL] folder does not exist: {scan_dir}")
        print("       check the path or pass --scan / --root")
        return 1

    bags = find_bags(scan_dir)
    if not bags:
        print(f"\n[FAIL] no .bag file in: {scan_dir}")
        print("       folder contents:")
        for p in sorted(scan_dir.iterdir())[:20]:
            print(f"         {p.name}")
        return 1

    total_gb = sum(b.stat().st_size for b in bags) / 1e9
    print(f"[2/5] found {len(bags)} bag file(s)  ({total_gb:.2f} GB)   OK")
    for b in bags:
        print(f"        {b.name}  ({b.stat().st_size / 1e9:.2f} GB)")

    try:
        reader_ctx = AnyReader(bags)
    except Exception as e:
        print(f"\n[FAIL] AnyReader does not accept the files: {e}")
        return 1

    with reader_ctx as reader:
        print(f"\n[3/5] bag opened                 OK")
        dur = (reader.end_time - reader.start_time) / 1e9
        print(f"        duration: {dur:.1f} s  ({dur/60:.1f} min)")
        print(f"\n        TOPICS:")
        print(f"        {'topic':<40} {'messages':>9}  type")
        print(f"        {'-'*40} {'-'*9}  {'-'*30}")
        for c in sorted(reader.connections, key=lambda c: c.topic):
            print(f"        {c.topic:<40} {c.msgcount:>9}  {c.msgtype}")

        conns = [c for c in reader.connections if c.topic == args.topic]
        if not conns:
            print(f"\n[FAIL] topic '{args.topic}' does not exist in this bag.")
            print("       pick one from the list above with --topic")
            return 1

        print(f"\n[4/5] reading the first message on {args.topic} ...")
        arr = None
        msg = None
        for conn, timestamp, raw in reader.messages(connections=conns):
            try:
                msg = reader.deserialize(raw, conn.msgtype)
                arr = pc2_to_numpy(msg)
            except Exception as e:
                print(f"\n[FAIL] deserialisation failed: {type(e).__name__}: {e}")
                return 1
            break

        if arr is None:
            print(f"\n[FAIL] no message on that topic.")
            return 1

        print(f"        fields         : {list(arr.dtype.names)}")
        print(f"        points         : {len(arr):,}")
        print(f"        point_step     : {msg.point_step} B")
        print(f"        frame_id       : {msg.header.frame_id}")

        for ax in ("x", "y", "z"):
            if ax not in arr.dtype.names:
                print(f"\n[FAIL] no field '{ax}' — this is not an XYZ cloud.")
                return 1

        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        nonzero = ~np.all(np.abs(xyz) < 1e-6, axis=1)
        xyz = xyz[nonzero]

        if len(xyz) == 0:
            print(f"\n[FAIL] all points are NaN or zero.")
            return 1

        rng = np.linalg.norm(xyz, axis=1)
        print(f"\n        valid points   : {len(xyz):,}")
        print(f"        x  [{xyz[:,0].min():8.2f} .. {xyz[:,0].max():8.2f}] m")
        print(f"        y  [{xyz[:,1].min():8.2f} .. {xyz[:,1].max():8.2f}] m")
        print(f"        z  [{xyz[:,2].min():8.2f} .. {xyz[:,2].max():8.2f}] m")
        print(f"        range  min {rng.min():.2f} m   max {rng.max():.2f} m   median {np.median(rng):.2f} m")

        inten = arr["intensity"][finite][nonzero].astype(np.float64) \
            if "intensity" in arr.dtype.names else None

        out = OUT_DIR / f"{args.scan}_scan0.ply"
        write_ply(out, xyz, inten)
        print(f"\n[5/5] saved: {out}")
        print(f"        ({len(xyz):,} points, {out.stat().st_size/1e6:.1f} MB)")

    print("\n" + "=" * 70)
    print("OK — the bags are readable and contain XYZ data.")
    print("=" * 70)
    print("\nNEXT: open the .ply and check that it looks like a room.")
    print("  - CloudCompare (free):  https://www.cloudcompare.org/")
    print("  - or:  pip install open3d")
    print("          python -c \"import open3d as o3d; o3d.visualization.draw_geometries(")
    print(f"                    [o3d.io.read_point_cloud(r'{out}')])\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
