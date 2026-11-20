"""One pass over the sweeps: occupancy octree + PatchStore, with timings.

The loop body is the online update: one sweep in, both structures updated.
"""
import time

import numpy as np

from . import io
from .store import PatchStore


def lattice_bounds(xyz_traj, max_range=15.0, below=4.0, above=6.0):
    lo = xyz_traj.min(0) - np.array([max_range, max_range, below])
    hi = xyz_traj.max(0) + np.array([max_range, max_range, above])
    return lo, hi


def build(scan, drop=(), stride=1, res=0.10, g=0.20, nb=8, max_range=15.0, t_end=None,
          octree=True, on_sweep=None, T=None, into=None, log=print):
    """Returns dict(tree, store, traj..., timing). on_sweep(t_rel, state) is called
    after every sweep (used for the online replay).
    T:    4x4 rigid transform applied to this scan (a second session registered to the first).
    into: (tree, store) of an earlier session to continue filling; store may be None to
          get this session's own store on the same lattice (pass lattice via into[2])."""
    traj = io.trajectory(scan)
    t_traj, xyz_traj, _ = traj
    if T is not None:
        xyz_traj = xyz_traj @ T[:3, :3].T + T[:3, 3]
    tree = None
    if into is not None:
        tree, lat = into[0], into[1]
        store = PatchStore.__new__(PatchStore)
        store.g, store.nb, store.gl, store.shape = lat.g, lat.nb, lat.gl.copy(), lat.shape
        store.lut = np.full(store.shape, -1, np.int32); store.n_rows = 0; store.cap = 0
        store._alloc(1 << 16)
    else:
        lo, hi = lattice_bounds(xyz_traj, max_range)
        store = PatchStore(lo, hi, g, nb)
        if octree:
            import octomap
            tree = octomap.OcTree(res)
    t_oct = t_store = 0.0
    n_sweep = n_pts = 0
    el_hist = np.zeros(180, np.int64)            # world elevation of the rays, 1 deg bins
    t0 = time.time()
    for trel, world, pos, rng in io.sweeps(scan, traj, drop, stride, max_range=max_range, t_end=t_end):
        if T is not None:
            world = world @ T[:3, :3].T + T[:3, 3]
            pos = pos @ T[:3, :3].T + T[:3, 3]
        a = time.perf_counter()
        if tree is not None:
            io.insert_octree(tree, world, pos)
        b = time.perf_counter()
        store.add(world, pos, rng)
        c = time.perf_counter()
        t_oct += b - a; t_store += c - b
        el = np.degrees(np.arcsin(np.clip((world[:, 2] - pos[:, 2]) / rng, -1, 1)))
        el_hist += np.bincount(np.clip((el + 90).astype(int), 0, 179), minlength=180)
        n_sweep += 1; n_pts += len(world)
        if on_sweep is not None:
            on_sweep(trel, dict(scan=scan, tree=tree, store=store, res=res, max_range=max_range,
                                el_hist=el_hist, traj_all=xyz_traj, traj_t_all=t_traj - t_traj[0]))
    if tree is not None:
        tree.updateInnerOccupancy()
    wall = time.time() - t0
    log(f"    build {scan}: {n_sweep:,} sweeps, {n_pts:,} pts, {store.n_rows:,} patches, "
        f"{wall:.0f}s (octree {1e3*t_oct/max(n_sweep,1):.1f} ms/sweep, "
        f"store {1e3*t_store/max(n_sweep,1):.1f} ms/sweep)")
    trel = t_traj - t_traj[0]
    keep = np.ones(len(trel), bool)
    for a, b in drop:
        keep &= ~((trel >= a) & (trel < b))
    if t_end is not None:
        keep &= trel <= t_end
    return dict(scan=scan, tree=tree, store=store, res=res, max_range=max_range,
                traj_t=trel[keep][::20], traj=xyz_traj[keep][::20],
                traj_full=xyz_traj[::20], el_hist=el_hist, n_sweep=n_sweep, n_pts=n_pts,
                timing=dict(wall_s=wall, octree_ms_per_sweep=1e3 * t_oct / max(n_sweep, 1),
                            store_ms_per_sweep=1e3 * t_store / max(n_sweep, 1)))
