from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
)


@dataclass
class PSLGVertex:
    id: int
    x: float
    y: float

    @property
    def xy(self):
        return (self.x, self.y)


@dataclass
class PSLGSegment:
    id: int
    v0: int
    v1: int

    def key(self):
        return (self.v0, self.v1)

    def undirected_key(self):
        return (self.v0, self.v1) if self.v0 < self.v1 else (self.v1, self.v0)


@dataclass
class PSLG:
    tol: float = 1e-9
    vertices: list[PSLGVertex] = field(default_factory=list)
    segments: list[PSLGSegment] = field(default_factory=list)

    _vertex_spatial_hash: dict = field(default_factory=dict, init=False)

    # -------------------------------------------------
    # Vertex handling
    # -------------------------------------------------

    def _hash_key(self, x, y):
        h = 1.0 / self.tol
        return (int(x * h), int(y * h))

    def add_vertex(self, x: float, y: float) -> int:
        key = self._hash_key(x, y)

        if key in self._vertex_spatial_hash:
            vid = self._vertex_spatial_hash[key]
            return vid

        vid = len(self.vertices)
        v = PSLGVertex(vid, x, y)
        self.vertices.append(v)
        self._vertex_spatial_hash[key] = vid
        return vid

    # -------------------------------------------------
    # Segment handling
    # -------------------------------------------------

    def add_segment(self, v0: int, v1: int) -> int:
        if v0 == v1:
            raise ValueError("Degenerate segment")

        sid = len(self.segments)
        s = PSLGSegment(sid, v0, v1)
        self.segments.append(s)
        return sid

    # -------------------------------------------------
    # Convenience builders
    # -------------------------------------------------

    def add_polyline(self, pts, closed=False):
        vids = [self.add_vertex(x, y) for x, y in pts]

        for i in range(len(vids) - 1):
            self.add_segment(vids[i], vids[i + 1])

        if closed:
            self.add_segment(vids[-1], vids[0])

        return vids

    def add_polygon(self, pts):
        return self.add_polyline(pts, closed=True)

    # -------------------------------------------------
    # Basic queries
    # -------------------------------------------------

    def num_vertices(self):
        return len(self.vertices)

    def num_segments(self):
        return len(self.segments)

    def summary(self):
        return f"PSLG(\n" f"  vertices={self.num_vertices()},\n" f"  segments={self.num_segments()}\n" f")"


if __name__ == "__main__":
    pslg = PSLG(tol=1e-8)

outer = [(0, 0), (4, 0), (4, 4), (0, 4)]

hole = [(1, 1), (3, 1), (3, 3), (1, 3)]

pslg.add_polygon(outer)
pslg.add_polygon(hole)

print(pslg.summary())
