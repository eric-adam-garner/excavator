from __future__ import annotations

import math
from dataclasses import (
    dataclass,
    field,
)

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


@dataclass
class PSLGNestingReport:
    loops: list[list[int]]
    loop_areas: list[float]
    parents: list[int | None]
    children: dict[int, list[int]]
    depths: list[int]
    outer_loops: list[list[int]]
    hole_loops: list[list[int]]

    def __repr__(self) -> str:
        return (
            "PSLGNestingReport(\n"
            f"  parents={self.parents},\n"
            f"  depths={self.depths},\n"
            f"  outer_loops={self.outer_loops},\n"
            f"  hole_loops={self.hole_loops}\n"
            ")"
        )


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
        """
        Add segment if it is non‑degenerate.

        Degenerate segments can arise after:
            - vertex snapping
            - segment splitting
            - duplicate polyline points
        """
        if v0 == v1:
            return None

        # also guard against zero-length in geometry space
        a = self.vertices[v0]
        b = self.vertices[v1]
        dx = a.x - b.x
        dy = a.y - b.y
        if dx * dx + dy * dy <= self.tol * self.tol:
            return None

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
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

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

    def _point_on_segment_interior(self, p, a, b):
        abx = b[0] - a[0]
        aby = b[1] - a[1]
        apx = p[0] - a[0]
        apy = p[1] - a[1]

        ab_len = abx * abx + aby * aby
        if ab_len <= self.tol:
            return False

        cross = abx * apy - aby * apx
        if abs(cross) > self.tol * math.sqrt(ab_len):
            return False

        t = (apx * abx + apy * aby) / ab_len
        return self.tol < t < 1 - self.tol

    def _loop_coords_from_vertex_ids(self, vids: list[int]) -> list[tuple[float, float]]:
        return [self.vertices[vid].xy for vid in vids]

    def _point_in_polygon(self, p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
        """
        Ray casting test.
        Returns True for strictly interior points.
        Boundary handling is intentionally conservative and should be avoided
        by using a non-boundary probe point.
        """
        x, y = p
        inside = False
        n = len(poly)

        for i in range(n):
            x0, y0 = poly[i]
            x1, y1 = poly[(i + 1) % n]

            intersects = (y0 > y) != (y1 > y)
            if intersects:
                x_cross = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if x < x_cross:
                    inside = not inside

        return inside

    def _loop_centroid(self, vids: list[int]) -> tuple[float, float]:
        pts = self._loop_coords_from_vertex_ids(vids)
        cx = sum(x for x, _ in pts) / len(pts)
        cy = sum(y for _, y in pts) / len(pts)
        return (cx, cy)

    def _loop_probe_point(self, vids: list[int]) -> tuple[float, float]:
        """
        Return a point guaranteed (heuristically) to lie inside the loop.

        Strategy:
            Take first edge, move slightly inward using orientation.
        """
        if len(vids) < 3:
            return self.vertices[vids[0]].xy

        v0 = self.vertices[vids[0]]
        v1 = self.vertices[vids[1]]
        v2 = self.vertices[vids[2]]

        x0, y0 = v0.xy
        x1, y1 = v1.xy
        x2, y2 = v2.xy

        # edge vector
        ex = x1 - x0
        ey = y1 - y0

        # signed area to determine orientation
        area = self._orient((x0, y0), (x1, y1), (x2, y2))

        # inward normal depends on orientation
        if area > 0:  # CCW
            nx = -ey
            ny = ex
        else:  # CW
            nx = ey
            ny = -ex

        length = math.hypot(nx, ny)
        if length == 0:
            return (x0, y0)

        nx /= length
        ny /= length

        eps = self.tol * 10

        return (x0 + nx * eps, y0 + ny * eps)

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
                    issues.append({"type": "vertex_on_segment", "vertex": v.id, "segment": s.id})
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
            v1 = self.vertices[vids[(i + 1) % n]]
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
                candidates = [(nbr, sid2) for nbr, sid2 in adj[cur] if sid2 in unused and nbr != prev]
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
        ccw = [l for l, a in zip(loops, areas) if a > 0]
        cw = [l for l, a in zip(loops, areas) if a <= 0]

        return PSLGLoopReport(loops, open_chains, areas, ccw, cw)

    def classify_loops(self) -> PSLGNestingReport:
        """
        Classify extracted loops by nesting depth.

        Returns:
            PSLGNestingReport

        Interpretation:
            depth 0 -> outer
            depth 1 -> hole
            depth 2 -> island inside hole
            depth 3 -> hole inside island
            etc.

        For standard polygon-with-holes domains, you typically want:
            - one or more depth-0 loops
            - any depth-1 loops as holes
            - no deeper nesting unless explicitly supported
        """
        loop_report = self.extract_loops()
        loops = loop_report.loops
        loop_areas = loop_report.loop_areas

        n = len(loops)
        polys = [self._loop_coords_from_vertex_ids(loop) for loop in loops]
        probes = [self._loop_probe_point(loop) for loop in loops]

        parents: list[int | None] = [None] * n

        # -------------------------------------------------
        # Parent = smallest containing loop
        # -------------------------------------------------
        for i in range(n):
            containing = []

            for j in range(n):
                if i == j:
                    continue

                if self._point_in_polygon(probes[i], polys[j]):
                    area_j = abs(loop_areas[j])
                    containing.append((area_j, j))

            if containing:
                containing.sort(key=lambda t: t[0])
                parents[i] = containing[0][1]

        # -------------------------------------------------
        # Children map
        # -------------------------------------------------
        children: dict[int, list[int]] = {i: [] for i in range(n)}
        for i, parent in enumerate(parents):
            if parent is not None:
                children[parent].append(i)

        # -------------------------------------------------
        # Depths
        # -------------------------------------------------
        depths: list[int] = [0] * n
        visiting: set[int] = set()
        computed: dict[int, int] = {}
        cyclic_nodes: set[int] = set()

        def compute_depth(i: int) -> int:
            if i in computed:
                return computed[i]

            if i in visiting:
                cyclic_nodes.add(i)
                return 0

            visiting.add(i)

            parent = parents[i]
            if parent is None:
                depth = 0
            else:
                depth = compute_depth(parent) + 1

            visiting.remove(i)
            computed[i] = depth
            return depth

        for i in range(n):
            depths[i] = compute_depth(i)

        outer_loops = [loops[i] for i in range(n) if depths[i] % 2 == 0]
        hole_loops = [loops[i] for i in range(n) if depths[i] % 2 == 1]

        return PSLGNestingReport(
            loops=loops,
            loop_areas=loop_areas,
            parents=parents,
            children=children,
            depths=depths,
            outer_loops=outer_loops,
            hole_loops=hole_loops,
        )

    def _reverse_loop(self, vids: list[int]):
        """
        Reverse loop orientation in-place by reversing its segments.
        """
        n = len(vids)

        edges = [(vids[i], vids[(i + 1) % n]) for i in range(n)]
        rev_edges = [(b, a) for (a, b) in reversed(edges)]

        # remove old segments
        seg_set = set(edges)
        self.segments = [s for s in self.segments if (s.v0, s.v1) not in seg_set]

        # add reversed
        for v0, v1 in rev_edges:
            self.add_segment(v0, v1)

    def normalize_orientation(self):
        """
        Enforce:
            outer loops → CCW
            hole loops → CW

        Modifies PSLG in-place.
        """

        nesting = self.classify_loops()

        for i, loop in enumerate(nesting.loops):

            area = nesting.loop_areas[i]
            depth = nesting.depths[i]

            should_be_ccw = depth % 2 == 0

            if should_be_ccw and area < 0:
                self._reverse_loop(loop)

            if not should_be_ccw and area > 0:
                self._reverse_loop(loop)

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    def validate(self):
        errors = []
        warnings = []

        inter = self.find_segment_intersections()
        v_on_seg = self.find_vertices_on_segments()
        loops = self.extract_loops()
        nesting = self.classify_loops()

        max_depth = max(nesting.depths, default=0)

        if max_depth > 1:
            warnings.append(f"PSLG has nested depth {max_depth}; deeper-than-hole nesting is present")

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
            "num_outer_loops": len(nesting.outer_loops),
            "num_hole_loops": len(nesting.hole_loops),
            "loop_depths": nesting.depths,
            "loop_parents": nesting.parents,
            "outer_loops": nesting.outer_loops,
            "hole_loops": nesting.hole_loops,
        }

        return PSLGReport(len(errors) == 0, errors, warnings, stats)
