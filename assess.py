"""Scan check: surface quality against a requirement, and what to do about it.

    python assess.py --scan institutGesamt            # 5 cm requirement
    python assess.py --scan institutGesamt --s 0.10   # coarser requirement
    python assess.py --scan institutGesamt --things   # ask the requirement of furniture and clutter too

Writes out/<scan>/scanq/: quality.json (for a program), scan_check.html (for a phone).
The first run builds the octree and the patch store (minutes); later runs reuse them.
"""
import argparse

from scanq import assess as A, report, session

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="institutGesamt")
    ap.add_argument("--s", type=float, default=0.05, help="required point spacing [m]: 0.025, 0.05 or 0.10")
    ap.add_argument("--range", type=float, default=None, help="override the range the requirement maps to [m]")
    ap.add_argument("--min-gain", type=float, default=1.0, help="smallest action worth listing [m2]")
    ap.add_argument("--things", action="store_true",
                    help="count furniture and clutter (default: building only; they are left out)")
    a = ap.parse_args()
    B = session.load_or_build(a.scan)
    R = A.assess(B, s_req=a.s, R=a.range, min_gain=a.min_gain, things=a.things)
    d = session.folder(a.scan, "full").parent
    report.to_json(R, d / "quality.json")
    report.to_html(R, d / "scan_check.html")
    print(f"    wrote {d / 'quality.json'} and {d / 'scan_check.html'}")
    for x in R["actions"][:12]:
        print(f"    {x['id']:3d} {x['kind']:8s} at ({x['xy'][0]:6.1f}, {x['xy'][1]:6.1f})  +{x['gain_m2']:6.1f} m2  walk {x['walk_m']}")
