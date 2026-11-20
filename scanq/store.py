"""PatchStore: what the scan knows about each surface patch, kept incrementally.

A patch is one cell of a fixed lattice (default 0.20 m) that has received points.
Per patch the store keeps only quantities that can be updated one sweep at a time
and combined across sessions without the points:

    n, sum x, sum x x^T      -> centroid, plane normal, thickness      (added)
    sum of view directions   -> which side the surface was seen from   (added)
    three bit masks, nb x nb -> which sub-cells of the patch, projected along each
                                world axis, have received a point      (OR-ed)
    min range per direction  -> the closest view from each of ND
                                directions on the sphere               (min-ed)

Add, OR and min are associative and commutative, so two stores built from
different sweeps (another session, another robot, a deleted part of the walk)
merge exactly: merge(A, B) == store built from A's and B's sweeps together.
"""
import numpy as np

ND = 64


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    ph = i * np.pi * (3 - 5 ** 0.5)
    s = np.sqrt(1 - z * z)
    return np.stack([s * np.cos(ph), s * np.sin(ph), z], 1)


DIRS = _fib_sphere(ND)          # unit vectors, direction sensor -> point


class PatchStore:
    def __init__(self, lo, hi, g=0.20, nb=8):
        self.g, self.nb = float(g), int(nb)
        self.gl = np.floor(np.asarray(lo, float) / g).astype(np.int64)
        self.shape = tuple((np.ceil(np.asarray(hi, float) / g).astype(np.int64) - self.gl).tolist())
        self.lut = np.full(self.shape, -1, np.int32)
        self.n_rows = 0
        self._alloc(1 << 16)

    # ---------------------------------------------------------------- storage
    def _alloc(self, cap):
        old = getattr(self, "cap", 0)
        def grow(name, shape, dtype, fill):
            a = np.full((cap,) + shape, fill, dtype)
            if old:
                a[:self.n_rows] = getattr(self, name)[:self.n_rows]
            setattr(self, name, a)
        grow("ijk", (3,), np.int32, 0)
        grow("M", (13,), np.float64, 0.0)        # n | x y z | xx xy xz yy yz zz | dx dy dz
        grow("mask", (3, self.nb * self.nb), bool, False)
        grow("vr", (ND,), np.float32, np.inf)
        grow("dirty", (), bool, False)
        self.cap = cap

    def _rows(self, world):
        ijk = np.floor(world / self.g).astype(np.int64) - self.gl
        ok = ((ijk >= 0) & (ijk < np.array(self.shape))).all(1)
        ijk = ijk[ok]
        flat = np.ravel_multi_index(ijk.T, self.shape)
        lut = self.lut.reshape(-1)
        rows = lut[flat]
        new = np.unique(flat[rows < 0])
        if len(new):
            if self.n_rows + len(new) > self.cap:
                self._alloc(max(self.cap * 2, self.n_rows + len(new)))
                lut = self.lut.reshape(-1)
            ids = np.arange(self.n_rows, self.n_rows + len(new), dtype=np.int32)
            lut[new] = ids
            self.ijk[ids] = np.stack(np.unravel_index(new, self.shape), 1)
            self.n_rows += len(new)
            rows = lut[flat]
        return rows, ok

    # ---------------------------------------------------------------- update
    def add(self, world, origin, rng):
        """Add one or more sweeps: points, the sensor position of each, its range."""
        rows, ok = self._rows(world)
        if not ok.all():
            world, origin, rng = world[ok], origin[ok], rng[ok]
        d = world - origin
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        x, y, z = world.T
        n = self.n_rows
        cols = (None, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z, d[:, 0], d[:, 1], d[:, 2])
        ur, inv = np.unique(rows, return_inverse=True)
        for c, w in enumerate(cols):
            self.M[ur, c] += np.bincount(inv, weights=w, minlength=len(ur))
        f = world / self.g
        b = np.minimum((self.nb * (f - np.floor(f))).astype(np.int64), self.nb - 1)
        self.mask[rows, 0, b[:, 1] * self.nb + b[:, 2]] = True      # seen along x: (y, z)
        self.mask[rows, 1, b[:, 0] * self.nb + b[:, 2]] = True      # seen along y: (x, z)
        self.mask[rows, 2, b[:, 0] * self.nb + b[:, 1]] = True      # seen along z: (x, y)
        k = np.argmax(d @ DIRS.T, axis=1)
        np.minimum.at(self.vr.reshape(-1), rows.astype(np.int64) * ND + k, rng.astype(np.float32))
        self.dirty[ur] = True
        return len(ur)

    def merge(self, other):
        """Fold another store on the same lattice into this one."""
        assert other.g == self.g and other.nb == self.nb and other.shape == self.shape \
            and (other.gl == self.gl).all(), "stores must share the lattice"
        m = other.n_rows
        flat = np.ravel_multi_index(other.ijk[:m].T.astype(np.int64), self.shape)
        lut = self.lut.reshape(-1)
        new = flat[lut[flat] < 0]
        if len(new):
            if self.n_rows + len(new) > self.cap:
                self._alloc(max(self.cap * 2, self.n_rows + len(new)))
                lut = self.lut.reshape(-1)
            ids = np.arange(self.n_rows, self.n_rows + len(new), dtype=np.int32)
            lut[new] = ids
            self.ijk[ids] = np.stack(np.unravel_index(new, self.shape), 1)
            self.n_rows += len(new)
        r = lut[flat]
        self.M[r] += other.M[:m]
        self.mask[r] |= other.mask[:m]
        self.vr[r] = np.minimum(self.vr[r], other.vr[:m])
        self.dirty[r] = True
        return self

    # ---------------------------------------------------------------- access
    def view(self):
        n = self.n_rows
        return dict(ijk=self.ijk[:n].astype(np.int64) + self.gl, M=self.M[:n],
                    mask=self.mask[:n], vr=self.vr[:n])

    def neighbours(self, offsets):
        """Row index of each patch's neighbour at the given lattice offsets (-1 if none)."""
        n = self.n_rows
        ijk = self.ijk[:n].astype(np.int64)
        out = np.full((n, len(offsets)), -1, np.int32)
        sh = np.array(self.shape)
        for c, off in enumerate(offsets):
            q = ijk + np.asarray(off)
            ok = ((q >= 0) & (q < sh)).all(1)
            out[ok, c] = self.lut[q[ok, 0], q[ok, 1], q[ok, 2]]
        return out

    def save(self, path):
        n = self.n_rows
        np.savez_compressed(path, g=self.g, nb=self.nb, gl=self.gl, shape=np.array(self.shape),
                            ijk=self.ijk[:n], M=self.M[:n],
                            mask=np.packbits(self.mask[:n], axis=-1), vr=self.vr[:n])

    @classmethod
    def load(cls, path):
        z = np.load(path)
        s = cls.__new__(cls)
        s.g, s.nb = float(z["g"]), int(z["nb"])
        s.gl, s.shape = z["gl"], tuple(z["shape"].tolist())
        n = len(z["ijk"])
        s.n_rows = 0
        s.lut = np.full(s.shape, -1, np.int32)
        s._alloc(max(n, 1))
        s.n_rows = n
        s.ijk[:n] = z["ijk"]; s.M[:n] = z["M"]; s.vr[:n] = z["vr"]
        s.mask[:n] = np.unpackbits(z["mask"], axis=-1)[..., :s.nb * s.nb].astype(bool)
        s.lut[s.ijk[:n, 0], s.ijk[:n, 1], s.ijk[:n, 2]] = np.arange(n, dtype=np.int32)
        return s
