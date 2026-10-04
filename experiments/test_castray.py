"""What does OctoMap castRay(..., ignoreUnknownCells=False) return for a ray that
enters unknown space before reaching its target? It returns False both for a clear
ray and for one that stops in unknown space, which is why coverage.py also checks
where the ray ended."""
import numpy as np
import octomap

t = octomap.OcTree(0.1)
# free space from the origin to a wall at x = 2.0, along a thin bundle of beams
pts = np.array([[2.0, y, z] for y in np.linspace(-0.3, 0.3, 7)
                for z in np.linspace(-0.3, 0.3, 7)])
t.insertPointCloud(pts, np.array([0.0, 0.0, 0.0]))
t.updateInnerOccupancy()

def probe(origin, target, label):
    o = np.asarray(origin, float); p = np.asarray(target, float)
    d = np.linalg.norm(p - o); dirv = (p - o) / d
    end = np.zeros(3)
    hit = t.castRay(o, dirv, end, False, d)
    n = t.search(p)
    try:
        st = "occupied" if t.isNodeOccupied(n) else "free"
    except Exception:
        st = "unknown"
    print(f"{label:55s} return={hit!s:5s} end={np.round(end,2)}  "
          f"stop-dist={np.linalg.norm(end-o):.2f} of {d:.2f}  target={st}")

probe([0, 0, 0], [1.5, 0, 0], "1 free path -> free target")
probe([0, 0, 0], [2.5, 0, 0], "2 free path -> behind wall")
probe([0, 0, 0], [0.0, 3.0, 0], "3 path through UNKNOWN -> unknown target 3 m away")
probe([0, 0, 0], [1.5, 0.6, 0], "4 mostly free, exits bundle -> unknown target")
