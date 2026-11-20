"""Build once, keep on disk: out/<scan>/scanq/<tag>/ {store.npz, octree.bt, meta.npz}."""
import json
from pathlib import Path

import numpy as np

from . import build as _build
from .store import PatchStore

ROOT = Path(__file__).resolve().parent.parent / "out"


def folder(scan, tag="full"):
    return ROOT / scan / "scanq" / tag


def load_or_build(scan, tag="full", drop=(), t_end=None, log=print, **kw):
    import octomap
    d = folder(scan, tag)
    if (d / "meta.npz").exists():
        z = np.load(d / "meta.npz")
        tree = octomap.OcTree(str(d / "octree.bt").encode())
        return dict(scan=scan, tree=tree, store=PatchStore.load(d / "store.npz"),
                    traj=z["traj"], traj_t=z["traj_t"], traj_full=z["traj_full"],
                    el_hist=z["el_hist"], res=float(z["res"]), max_range=float(z["max_range"]),
                    timing=json.loads(str(z["timing"])), n_sweep=int(z["n_sweep"]), n_pts=int(z["n_pts"]))
    d.mkdir(parents=True, exist_ok=True)
    B = _build.build(scan, drop=drop, t_end=t_end, log=log, **kw)
    B["store"].save(d / "store.npz")
    B["tree"].writeBinary(str(d / "octree.bt").encode())
    np.savez(d / "meta.npz", traj=B["traj"], traj_t=B["traj_t"], traj_full=B["traj_full"],
             el_hist=B["el_hist"], res=B["res"], max_range=B["max_range"],
             timing=json.dumps(B["timing"]), n_sweep=B["n_sweep"], n_pts=B["n_pts"],
             drop=np.array(drop, float).reshape(-1, 2))
    return B
