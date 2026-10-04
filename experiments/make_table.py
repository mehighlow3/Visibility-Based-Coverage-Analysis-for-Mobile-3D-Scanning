"""Deletion-test table and key numbers for README.md (finding 3).

    python make_table.py [scan]
"""
import json, sys
from pathlib import Path
import numpy as np

H = Path(__file__).parent
scan = sys.argv[1] if len(sys.argv) > 1 else "institutGesamt"
KEY = "multi-strict"            # the rule coverage.py uses
G = 0.20

W = json.loads((H / f"abl_{scan}" / "summary.json").read_text())["windows"]
PH = json.loads((H / f"abl_{scan}" / "posthoc.json").read_text())

wlen = W[0]["window"][1] - W[0]["window"][0]
path = [w["path_m"] for w in W]
vol = [w["removed_vox"] * G ** 3 for w in W]
rec = np.array([w[KEY]["recall"] for w in W])
wt = np.array([w["frontier_in_removed"] for w in W], float)
gt = sum(w[KEY]["gaps_true"] for w in W)
ga = sum(w[KEY]["gaps"] for w in W)

print("| deleted | observed volume lost | edge called \"missed\" (correct) | gap clusters on the forgotten area |")
print("|---|---|---|---|")
print(f"| {len(W)} × {wlen:.0f} s ({min(path):.0f}–{max(path):.0f} m of walk) | "
      f"{min(vol):.0f}–{max(vol):.0f} m³ | {100*rec.min():.0f}–{100*rec.max():.0f} % "
      f"(mean {100*(rec*wt).sum()/wt.sum():.0f} %) | {gt} of {ga} |")
print()

span = lambda k: f"{min(p[k] for p in PH):.2f}–{max(p[k] for p in PH):.2f}"
other = [p[k] for p in PH for k in ("auc_d_clear", "auc_not_touching", "auc_few_clear")]
print(f"AUC distance to the path: {span('auc_d_view')}")
print(f"AUC clear-view distance, clear views, adjacency to a surface: {min(other):.2f}–{max(other):.2f}")
nf = [p["new_in_R"] / max(p["new_frontier"], 1) for p in PH]
print(f"new frontier on the forgotten area: {100*min(nf):.0f}–{100*max(nf):.0f} %")
nc = [p["categories"]["no clear view"]["n"] / p["frontier"] for p in PH]
print(f"frontier with no clear view from the path: {100*min(nc):.0f}–{100*max(nc):.0f} %")
