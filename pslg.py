from __future__ import annotations
from dataclasses import dataclass, field
import math


# ============================================================
# REPORT TYPES
# ============================================================

@dataclass
class PSLGReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


@dataclass
class PSLGLoopReport:
    loops: list[list[int]]
    open_chains: list[list[int]]
    loop_areas: list[float]
    ccw_loops: list[list[int]]
    cw_loops: list[list[int]]


# ============================================================
# PRIMITIVES
# ============================================================

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

    def undirected_key(self):
        return (self.v0, self.v1) if self.v0 < self.v1 else (self.v1, self.v0)


# ============================================================
# PSLG
# ============================================================

@dataclass
class PSLG:
    tol: float = 1e-9
    vertices: list[PSLGVertex] = field(default_factory=list)
    segments: list[PSLGSegment] = field(default_factory=list)

    _hash: dict = field(default_factory=dict, init=False)

    # --------------------------------------------------------
    # Vertex management
    # --------------------------------------------------------

    def _hash_key(self, x, y):
        h = 1.0 / self.tol
        return (int(x * h), int(y * h))

    def add_vertex(self, x, y):
        key = self._hash_key(x, y)
        if key in self._hash:
            return self._hash[key]

        vid = len(self.vertices)
        self.vertices.append(PSLGVertex(vid, x, y))
        self._hash[key] = vid
        return vid

    # --------------------------------------------------------
    # Segment management
    # --------------------------------------------------------

    def add_segment(self, v0, v1):
        if v0 == v1:
            raise ValueError("degenerate segment")
        sid = len(self.segments)
        self.segments.append(PSLGSegment(sid, v0, v1))
        return sid

    def add_polygon(self, pts):
        vids = [self.add_vertex(*p) for p in pts]
        for i in range(len(vids)):
            self.add_segment(vids[i], vids[(i + 1) % len(vids)])

    # --------------------------------------------------------
    # Geometry utilities
    # --------------------------------------------------------

    def _orient(self, a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

    def _orient_sign(self, a, b, c):
        v = self._orient(a, b, c)
        if abs(v) <= self.tol:
            return 0
        return 1 if v > 0 else -1

    def _proper_intersection(self, a, b, c, d):
        o1 = self._orient_sign(a, b, c)
        o2 = self._orient_sign(a, b, d)
        o3 = self._orient_sign(c, d, a)
        o4 = self._orient_sign(c, d, b)
        return o1 * o2 < 0 and o3 * o4 < 0

    def _colinear(self, a, b, c):
        return abs(self._orient(a, b, c)) <= self.tol

    def _interval_overlap(self, a0, a1, b0, b1):
        lo = max(min(a0, a1), min(b0, b1))
        hi = min(max(a0, a1), max(b0, b1))
        return hi - lo

    def _colinear_overlap(self, a, b, c, d):
        dx = abs(b[0]-a[0])
        dy = abs(b[1]-a[1])
        if dx >= dy:
            overlap = self._interval_overlap(a[0], b[0], c[0], d[0])
        else:
            overlap = self._interval_overlap(a[1], b[1], c[1], d[1])

        if overlap < -self.tol:
            return None
        if abs(overlap) <= self.tol:
            return "touch"
        return "overlap"

    def _point_on_segment_interior(self, p, a, b):
        abx = b[0]-a[0]
        aby = b[1]-a[1]
        apx = p[0]-a[0]
        apy = p[1]-a[1]

        ab_len = abx*abx + aby*aby
        if ab_len <= self.tol:
            return False

        cross = abx*apy - aby*apx
        if abs(cross) > self.tol * math.sqrt(ab_len):
            return False

        t = (apx*abx + apy*aby)/ab_len
        return self.tol < t < 1 - self.tol

    # --------------------------------------------------------
    # INTERSECTION DETECTION
    # --------------------------------------------------------

    def find_segment_intersections(self):
        issues = []
        n = len(self.segments)

        for i in range(n):
            s1 = self.segments[i]
            a = self.vertices[s1.v0].xy
            b = self.vertices[s1.v1].xy

            for j in range(i+1, n):
                s2 = self.segments[j]
                c = self.vertices[s2.v0].xy
                d = self.vertices[s2.v1].xy

                if self._colinear(a, b, c) and self._colinear(a, b, d):
                    typ = self._colinear_overlap(a, b, c, d)
                    if typ == "overlap":
                        issues.append({"type": "overlap", "seg_a": s1.id, "seg_b": s2.id})
                    continue

                if self._proper_intersection(a, b, c, d):
                    issues.append({"type": "proper", "seg_a": s1.id, "seg_b": s2.id})

        return issues

    def find_vertices_on_segments(self):
        issues = []

        for v in self.vertices:
            p = v.xy
            for s in self.segments:
                if v.id in (s.v0, s.v1):
                    continue
                a = self.vertices[s.v0].xy
                b = self.vertices[s.v1].xy

                if self._point_on_segment_interior(p, a, b):
                    issues.append(
                        {"type": "vertex_on_segment", "vertex": v.id, "segment": s.id}
                    )
        return issues

    # --------------------------------------------------------
    # LOOP EXTRACTION
    # --------------------------------------------------------

    def _vertex_adjacency(self):
        adj = {v.id: [] for v in self.vertices}
        for s in self.segments:
            adj[s.v0].append((s.v1, s.id))
            adj[s.v1].append((s.v0, s.id))
        return adj

    def _loop_signed_area(self, vids):
        area = 0
        n = len(vids)
        for i in range(n):
            v0 = self.vertices[vids[i]]
            v1 = self.vertices[vids[(i+1)%n]]
            area += v0.x * v1.y - v1.x * v0.y
        return 0.5 * area

    def extract_loops(self):
        adj = self._vertex_adjacency()
        unused = {s.id for s in self.segments}
        seg_map = {s.id: s for s in self.segments}

        loops = []
        open_chains = []

        def trace(start, nxt, sid):
            chain = [start, nxt]
            unused.remove(sid)
            prev = start
            cur = nxt

            while True:
                candidates = [
                    (nbr, sid2)
                    for nbr, sid2 in adj[cur]
                    if sid2 in unused and nbr != prev
                ]
                if not candidates:
                    break
                nbr, sid2 = candidates[0]
                chain.append(nbr)
                unused.remove(sid2)
                prev, cur = cur, nbr
                if cur == start:
                    break
            return chain

        while unused:
            sid = next(iter(unused))
            s = seg_map[sid]
            chain = trace(s.v0, s.v1, sid)

            if chain[0] == chain[-1]:
                loops.append(chain[:-1])
            else:
                open_chains.append(chain)

        areas = [self._loop_signed_area(l) for l in loops]
        ccw = [l for l,a in zip(loops, areas) if a > 0]
        cw = [l for l,a in zip(loops, areas) if a <= 0]

        return PSLGLoopReport(loops, open_chains, areas, ccw, cw)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(self):
        errors = []
        warnings = []

        inter = self.find_segment_intersections()
        v_on_seg = self.find_vertices_on_segments()
        loops = self.extract_loops()

        if inter:
            for i in inter:
                errors.append(f"{i['type']} intersection between segments {i['seg_a']} and {i['seg_b']}")

        if v_on_seg:
            for i in v_on_seg:
                errors.append(f"vertex {i['vertex']} on segment {i['segment']}")

        if loops.open_chains:
            errors.append("PSLG has open chains")

        stats = {
            "num_vertices": len(self.vertices),
            "num_segments": len(self.segments),
            "num_loops": len(loops.loops),
            "num_open_chains": len(loops.open_chains),
        }

        return PSLGReport(len(errors)==0, errors, warnings, stats)