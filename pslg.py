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

    def _orient(self, ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    def _on_segment(self, ax, ay, bx, by, cx, cy):
        return (
            min(ax, bx) - self.tol <= cx <= max(ax, bx) + self.tol
            and min(ay, by) - self.tol <= cy <= max(ay, by) + self.tol
        )

    def _orient_sign(self, a, b, c):
        ax, ay = a
        bx, by = b
        cx, cy = c

        val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

        if abs(val) < self.tol:
            return 0
        return 1 if val > 0 else -1

    def _proper_intersection(self, a, b, c, d):
        o1 = self._orient_sign(a, b, c)
        o2 = self._orient_sign(a, b, d)
        o3 = self._orient_sign(c, d, a)
        o4 = self._orient_sign(c, d, b)

        return o1 * o2 < 0 and o3 * o4 < 0

    def _endpoint_touch(self, a, b, c, d):
        def on_seg(p, q, r):
            return (
                min(p[0], q[0]) - self.tol <= r[0] <= max(p[0], q[0]) + self.tol
                and min(p[1], q[1]) - self.tol <= r[1] <= max(p[1], q[1]) + self.tol
            )

        if self._orient_sign(a, b, c) == 0 and on_seg(a, b, c):
            return True
        if self._orient_sign(a, b, d) == 0 and on_seg(a, b, d):
            return True
        if self._orient_sign(c, d, a) == 0 and on_seg(c, d, a):
            return True
        if self._orient_sign(c, d, b) == 0 and on_seg(c, d, b):
            return True

        return False

    def _colinear(self, a, b, c):
        return abs(self._orient_sign(a, b, c)) == 0

    def _interval_overlap(self, a0, a1, b0, b1):
        lo = max(min(a0, a1), min(b0, b1))
        hi = min(max(a0, a1), max(b0, b1))
        return hi - lo

    def _colinear_overlap_type(self, a, b, c, d):

        # choose projection axis (dominant direction)
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])

        if dx >= dy:
            overlap = self._interval_overlap(a[0], b[0], c[0], d[0])
        else:
            overlap = self._interval_overlap(a[1], b[1], c[1], d[1])

        if overlap < -self.tol:
            return None

        if abs(overlap) <= self.tol:
            return "touch"

        return "overlap"

    def _point_distance_sq(self, p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx * dx + dy * dy

    def _point_on_segment(self, p, a, b):
        """
        Return True if point p lies on segment ab within tolerance.
        Includes endpoints.
        """
        ax, ay = a
        bx, by = b
        px, py = p

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= self.tol * self.tol:
            return False

        # Perpendicular distance via cross product magnitude
        cross = abx * apy - aby * apx
        if abs(cross) > self.tol * math.sqrt(ab_len_sq):
            return False

        # Parametric coordinate on line
        t = (apx * abx + apy * aby) / ab_len_sq
        return -self.tol <= t <= 1.0 + self.tol

    def _point_on_segment_interior(self, p, a, b):
        """
        Return True if point p lies strictly in the interior of segment ab
        within tolerance, excluding endpoints.
        """
        ax, ay = a
        bx, by = b
        px, py = p

        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay

        ab_len_sq = abx * abx + aby * aby
        if ab_len_sq <= self.tol * self.tol:
            return False

        cross = abx * apy - aby * apx
        if abs(cross) > self.tol * math.sqrt(ab_len_sq):
            return False

        t = (apx * abx + apy * aby) / ab_len_sq
        return self.tol < t < 1.0 - self.tol

    def find_segment_intersections(self):

        issues = []

        n = len(self.segments)

        for i in range(n):
            s1 = self.segments[i]
            a = self.vertices[s1.v0].xy
            b = self.vertices[s1.v1].xy

            for j in range(i + 1, n):
                s2 = self.segments[j]
                c = self.vertices[s2.v0].xy
                d = self.vertices[s2.v1].xy

                # --- COLINEAR TEST FIRST (CRITICAL) ---
                if self._colinear(a, b, c) and self._colinear(a, b, d):

                    typ = self._colinear_overlap_type(a, b, c, d)

                    if typ == "overlap":
                        issues.append({"type": "overlap", "seg_a": s1.id, "seg_b": s2.id})
                    elif typ == "touch":
                        shared = s1.v0 == s2.v0 or s1.v0 == s2.v1 or s1.v1 == s2.v0 or s1.v1 == s2.v1
                        if not shared:
                            issues.append({"type": "t_junction", "seg_a": s1.id, "seg_b": s2.id})
                    continue

                # --- PROPER INTERSECTION ---
                if self._proper_intersection(a, b, c, d):
                    issues.append({"type": "proper", "seg_a": s1.id, "seg_b": s2.id})
                    continue

                # --- NON-COLINEAR TOUCH (T-JUNCTION) ---
                if self._endpoint_touch(a, b, c, d):
                    shared = s1.v0 == s2.v0 or s1.v0 == s2.v1 or s1.v1 == s2.v0 or s1.v1 == s2.v1
                    if not shared:
                        issues.append({"type": "t_junction", "seg_a": s1.id, "seg_b": s2.id})

        return issues

    def find_vertices_on_segments(self):
        """
        Detect vertices that lie on the interior of segments that do not
        explicitly use that vertex.

        Returns a list of diagnostics:
        {
            "type": "vertex_on_segment",
            "vertex": vertex_id,
            "segment": segment_id,
        }
        """
        issues = []

        for v in self.vertices:
            p = v.xy

            for s in self.segments:
                if v.id == s.v0 or v.id == s.v1:
                    continue

                a = self.vertices[s.v0].xy
                b = self.vertices[s.v1].xy

                if self._point_on_segment_interior(p, a, b):
                    issues.append(
                        {
                            "type": "vertex_on_segment",
                            "vertex": v.id,
                            "segment": s.id,
                        }
                    )

        return issues

    def summary(self):
        return f"PSLG(\n" f"  vertices={self.num_vertices()},\n" f"  segments={self.num_segments()}\n" f")"

pslg = PSLG()

a = pslg.add_vertex(0, 0)
b = pslg.add_vertex(4, 0)
c = pslg.add_vertex(2, 0)
d = pslg.add_vertex(2, 3)

pslg.add_segment(a, b)
pslg.add_segment(c, d)

print(pslg.find_segment_intersections())
print(pslg.find_vertices_on_segments())

pslg = PSLG()
pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])

print(pslg.find_vertices_on_segments())

pslg = PSLG()

pslg.add_polygon([(0, 0), (4, 0), (4, 4), (0, 4)])
v = pslg.add_vertex(2, 0)

print(pslg.find_vertices_on_segments())