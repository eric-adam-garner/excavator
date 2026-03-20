from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import triangle as tr

import half_edge_mesh


@dataclass(frozen=True)
class TriangleInput:
    points: np.ndarray  # shape (N, 2), float64
    segments: np.ndarray  # shape (M, 2), int32
    regions: np.ndarray  # shape (K, 4), float64


@dataclass(frozen=True)
class TriangleMesh:
    vertices: list[Vertex]
    triangles: list[Triangle]
    triangle_region_ids: list[int]


@dataclass(frozen=True)
class Vertex:
    x: float
    y: float


@dataclass(frozen=True)
class Triangle:
    v0: int
    v1: int
    v2: int


class TrianglePrecheckReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}

    @property
    def is_valid(self):
        return len(self.errors) == 0

    def __repr__(self):
        return (
            f"TrianglePrecheckReport("
            f"is_valid={self.is_valid}, "
            f"errors={self.errors}, "
            f"warnings={self.warnings}, "
            f"stats={self.stats})"
        )


def validate_domain_segments(domain):
    """
    Returns:
        errors, warnings
    """
    errors = []
    warnings = []

    verts = domain.vertices
    edges = domain.edges
    tol = domain.tol

    segs = [(verts[i], verts[j], idx, i, j) for idx, (i, j) in enumerate(edges)]

    for i in range(len(segs)):
        a, b, idx_ab, ia0, ia1 = segs[i]

        if _is_degenerate_edge(a, b, tol):
            errors.append(f"degenerate edge {idx_ab}: {(ia0, ia1)}")
            continue

        for j in range(i + 1, len(segs)):
            c, d, idx_cd, ic0, ic1 = segs[j]

            # skip if they share a vertex; endpoint touch is allowed there
            shared = {ia0, ia1}.intersection({ic0, ic1})
            relation = _segment_relation(a, b, c, d, tol)

            if relation is None:
                continue

            if relation == "endpoint_touch":
                # allowed only if they actually share an indexed endpoint
                if not shared:
                    errors.append(
                        f"unexpected endpoint touch between edges {idx_ab}{(ia0, ia1)} and {idx_cd}{(ic0, ic1)}"
                    )

            elif relation == "proper_intersection":
                errors.append(f"proper intersection between edges {idx_ab}{(ia0, ia1)} and {idx_cd}{(ic0, ic1)}")

            elif relation == "t_junction":
                errors.append(f"t-junction between edges {idx_ab}{(ia0, ia1)} and {idx_cd}{(ic0, ic1)}")

            elif relation == "colinear_overlap":
                errors.append(f"colinear overlap between edges {idx_ab}{(ia0, ia1)} and {idx_cd}{(ic0, ic1)}")

    return errors, warnings


def validate_segment_graph(points, segments):
    """
    Basic graph-level checks for Triangle suitability.

    Returns:
        dict with:
            degree_map
            dangling_vertices
            isolated_vertices
            nonmanifold_vertices
    """
    degree = defaultdict(int)

    used_vertices = set()
    for a, b in segments:
        degree[a] += 1
        degree[b] += 1
        used_vertices.add(a)
        used_vertices.add(b)

    dangling_vertices = [v for v, d in degree.items() if d == 1]
    isolated_vertices = [i for i in range(len(points)) if i not in used_vertices]
    nonmanifold_vertices = [v for v, d in degree.items() if d > 2]

    return {
        "degree_map": dict(degree),
        "dangling_vertices": dangling_vertices,
        "isolated_vertices": isolated_vertices,
        "nonmanifold_vertices": nonmanifold_vertices,
    }


def validate_triangle_input_geometry(tri_in, tol):
    """
    Triangle-focused validation beyond simple duplicate/seed checks.
    """
    report = TrianglePrecheckReport()

    points = tri_in.points
    segments = tri_in.segments

    # 4. intersections / T-junctions / overlaps
    inter = find_segment_intersections(points, segments, tol)

    for i, j in inter["proper"]:
        report.errors.append(f"proper segment intersection between segments {i} and {j}")

    for i, j in inter["t_junction"]:
        report.errors.append(f"T-junction between segments {i} and {j}")

    for i, j in inter["overlap"]:
        report.errors.append(f"colinear overlap between segments {i} and {j}")

    # 5. graph enclosure / open chains
    graph = validate_segment_graph(points, segments)

    for v in graph["dangling_vertices"]:
        report.errors.append(f"dangling vertex {v}")

    for v in graph["nonmanifold_vertices"]:
        report.warnings.append(f"nonmanifold/high-degree vertex {v} with degree {graph['degree_map'][v]}")

    # isolated vertices are usually harmless to Triangle if unused, but suspicious
    for v in graph["isolated_vertices"]:
        report.warnings.append(f"isolated unused vertex {v}")

    report.stats = {
        "num_points": len(points),
        "num_segments": len(segments),
        "num_regions": len(tri_in.regions),
        "num_proper_intersections": len(inter["proper"]),
        "num_t_junctions": len(inter["t_junction"]),
        "num_overlaps": len(inter["overlap"]),
        "num_dangling_vertices": len(graph["dangling_vertices"]),
        "num_nonmanifold_vertices": len(graph["nonmanifold_vertices"]),
        "num_isolated_vertices": len(graph["isolated_vertices"]),
    }

    return report


def validate_triangle_input(tri_in: TriangleInput) -> None:
    if tri_in.points.size == 0:
        raise ValueError("Triangle input has no points.")

    if tri_in.segments.size == 0:
        raise ValueError("Triangle input has no segments.")

    if tri_in.points.ndim != 2 or tri_in.points.shape[1] != 2:
        raise ValueError(f"Invalid points shape: {tri_in.points.shape}")

    if tri_in.segments.ndim != 2 or tri_in.segments.shape[1] != 2:
        raise ValueError(f"Invalid segments shape: {tri_in.segments.shape}")

    if tri_in.regions.ndim != 2 or tri_in.regions.shape[1] != 4:
        raise ValueError(f"Invalid regions shape: {tri_in.regions.shape}")

    n = len(tri_in.points)

    seen_segments = set()
    for i, (a, b) in enumerate(tri_in.segments):
        a = int(a)
        b = int(b)

        if a == b:
            raise ValueError(f"Degenerate segment at index {i}: ({a}, {b})")

        if not (0 <= a < n and 0 <= b < n):
            raise ValueError(f"Out-of-range segment at index {i}: ({a}, {b})")

        key = (a, b) if a < b else (b, a)
        if key in seen_segments:
            raise ValueError(f"Duplicate segment detected: {key}")
        seen_segments.add(key)


def build_triangle_input(domain) -> TriangleInput:
    points = np.asarray(domain.vertices, dtype=np.float64)
    segments = np.asarray(domain.edges, dtype=np.int32)

    faces = domain.faces
    bench_ids = domain.face_bench_ids

    unique_bench_ids = sorted(set(bench_ids))
    region_id_map = {bench_id: i for i, bench_id in enumerate(unique_bench_ids)}

    regions = []

    for face, bench_id in zip(faces, bench_ids):
        coords = [tuple(points[v]) for v in face]
        sx, sy = _safe_face_seed(coords)
        rid = region_id_map[bench_id]

        # Triangle region row: [x, y, attribute, max_area]
        # max_area is ignored unless -a is passed, but the wrapper usually
        # still wants the 4th column to exist.
        regions.append((sx, sy, float(rid), 0.0))

    regions = np.asarray(regions, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points has invalid shape {points.shape}")

    if segments.ndim != 2 or segments.shape[1] != 2:
        raise ValueError(f"segments has invalid shape {segments.shape}")

    if regions.ndim != 2 or regions.shape[1] != 4:
        raise ValueError(f"regions has invalid shape {regions.shape}")

    return TriangleInput(
        points=np.ascontiguousarray(points),
        segments=np.ascontiguousarray(segments),
        regions=np.ascontiguousarray(regions),
    )


def triangulate_partition_domain(domain, triangle_flags: str = "pA"):
    tri_in = build_triangle_input(domain)
    validate_triangle_input(tri_in)

    pre = validate_triangle_input_geometry(tri_in, domain.tol)
    if not pre.is_valid:
        raise RuntimeError(pre)

    tri_dict = {
        "vertices": tri_in.points,
        "segments": tri_in.segments,
        "regions": tri_in.regions,
    }

    out = tr.triangulate(tri_dict, triangle_flags)

    if "vertices" not in out:
        raise RuntimeError("Triangle output missing vertices.")
    if "triangles" not in out:
        raise RuntimeError("Triangle output missing triangles.")

    vertices = [tuple(map(float, xy)) for xy in out["vertices"]]
    triangles = [tuple(map(int, tri)) for tri in out["triangles"]]

    if "triangle_attributes" in out:
        triangle_region_ids = [int(attr[0]) for attr in out["triangle_attributes"]]
    else:
        triangle_region_ids = [-1] * len(triangles)

    return TriangleMesh(
        vertices=vertices,
        triangles=triangles,
        triangle_region_ids=triangle_region_ids,
    )


def triangle_to_halfedge_mesh(tri_mesh: TriangleMesh):

    mesh = half_edge_mesh.Mesh()

    # -----------------------------
    # Create vertices
    # -----------------------------
    for idx, (x, y) in enumerate(tri_mesh.vertices):
        mesh.vertices.append(half_edge_mesh.Vertex(id=idx, x=x, y=y))

    # map for twin linking
    edge_map = {}

    # -----------------------------
    # Create faces + halfedges
    # -----------------------------
    half_edge_count = 0
    for tri_idx, (a, b, c) in enumerate(tri_mesh.triangles):

        face = half_edge_mesh.Face(id=tri_idx)
        face.region_id = tri_mesh.triangle_region_ids[tri_idx]

        he0 = half_edge_mesh.HalfEdge(id=half_edge_count, origin=mesh.vertices[a])
        half_edge_count += 1

        he1 = half_edge_mesh.HalfEdge(id=half_edge_count, origin=mesh.vertices[b])
        half_edge_count += 1

        he2 = half_edge_mesh.HalfEdge(id=half_edge_count, origin=mesh.vertices[c])
        half_edge_count += 1

        he0.next = he1
        he1.next = he2
        he2.next = he0

        he0.prev = he2
        he1.prev = he0
        he2.prev = he1

        he0.face = face
        he1.face = face
        he2.face = face

        face.halfedge = he0

        mesh.halfedges.extend([he0, he1, he2])
        mesh.faces.append(face)

        # store directed edges
        tri_edges = [(a, b, he0), (b, c, he1), (c, a, he2)]

        for u, v, he in tri_edges:
            key = (u, v)
            twin_key = (v, u)

            if twin_key in edge_map:
                twin = edge_map[twin_key]
                he.twin = twin
                twin.twin = he
            else:
                edge_map[key] = he

    # # -----------------------------
    # # Detect boundary edges
    # # -----------------------------
    # for he in mesh.halfedges:
    #     if he.twin is None:
    #         he.is_boundary = True
    #     else:
    #         he.is_boundary = False

    return mesh


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _orient_sign(a, b, c, tol):
    val = _orient(a, b, c)
    if abs(val) <= tol:
        return 0
    return 1 if val > 0 else -1


def _point_on_segment(p, a, b, tol):
    ax, ay = a
    bx, by = b
    px, py = p

    if px < min(ax, bx) - tol or px > max(ax, bx) + tol:
        return False
    if py < min(ay, by) - tol or py > max(ay, by) + tol:
        return False

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    cross = abx * apy - aby * apx
    seg_len = (abx * abx + aby * aby) ** 0.5
    if abs(cross) > tol * max(1.0, seg_len):
        return False

    dot = apx * abx + apy * aby
    if dot < -tol:
        return False
    if dot > abx * abx + aby * aby + tol:
        return False

    return True


def _proper_segment_intersection(a, b, c, d, tol):
    o1 = _orient_sign(a, b, c, tol)
    o2 = _orient_sign(a, b, d, tol)
    o3 = _orient_sign(c, d, a, tol)
    o4 = _orient_sign(c, d, b, tol)

    return o1 * o2 < 0 and o3 * o4 < 0


def _colinear_overlap_type(a, b, c, d, tol):
    """
    Returns:
        None      -> disjoint
        "touch"   -> colinear endpoint touch only
        "overlap" -> positive-length overlap
    """
    if not (_orient_sign(a, b, c, tol) == 0 and _orient_sign(a, b, d, tol) == 0):
        return None

    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])

    if dx >= dy:
        a0, a1 = sorted((a[0], b[0]))
        b0, b1 = sorted((c[0], d[0]))
    else:
        a0, a1 = sorted((a[1], b[1]))
        b0, b1 = sorted((c[1], d[1]))

    lo = max(a0, b0)
    hi = min(a1, b1)
    overlap = hi - lo

    if overlap < -tol:
        return None
    if abs(overlap) <= tol:
        return "touch"
    return "overlap"


def find_segment_intersections(points, segments, tol):
    """
    Detect Triangle-invalid segment interactions.

    Returns:
        dict with keys:
            proper
            t_junction
            overlap
    """
    issues = {
        "proper": [],
        "t_junction": [],
        "overlap": [],
    }

    n = len(segments)

    for i in range(n):
        ia, ib = segments[i]
        a = points[ia]
        b = points[ib]

        for j in range(i + 1, n):
            ic, id_ = segments[j]
            c = points[ic]
            d = points[id_]

            shared_vertices = len({ia, ib, ic, id_}) < 4

            # proper crossing
            if _proper_segment_intersection(a, b, c, d, tol):
                issues["proper"].append((i, j))
                continue

            # colinear overlap
            overlap_type = _colinear_overlap_type(a, b, c, d, tol)
            if overlap_type == "overlap":
                issues["overlap"].append((i, j))
                continue

            # T-junction / endpoint-on-interior
            if not shared_vertices:
                if _point_on_segment(c, a, b, tol) or _point_on_segment(d, a, b, tol):
                    issues["t_junction"].append((i, j))
                    continue
                if _point_on_segment(a, c, d, tol) or _point_on_segment(b, c, d, tol):
                    issues["t_junction"].append((i, j))
                    continue

    return issues


def _is_degenerate_edge(a, b, tol):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return dx * dx + dy * dy <= tol * tol


def _on_segment(a, b, p, tol):
    if (
        min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol
        and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol
    ):
        area = abs(_orient(a, b, p))
        edge_len = max(1.0, ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)
        return area <= tol * edge_len
    return False


def _segment_relation(a, b, c, d, tol):
    """
    Classify relation between segments ab and cd.

    Returns one of:
        None
        "proper_intersection"
        "endpoint_touch"
        "t_junction"
        "colinear_overlap"
    """
    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    def sgn(x):
        if abs(x) <= tol:
            return 0
        return 1 if x > 0 else -1

    s1, s2, s3, s4 = sgn(o1), sgn(o2), sgn(o3), sgn(o4)

    # Proper crossing
    if s1 * s2 < 0 and s3 * s4 < 0:
        return "proper_intersection"

    # Colinear case
    if s1 == 0 and s2 == 0 and s3 == 0 and s4 == 0:
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])

        if dx >= dy:
            a0, a1 = sorted([a[0], b[0]])
            c0, c1 = sorted([c[0], d[0]])
        else:
            a0, a1 = sorted([a[1], b[1]])
            c0, c1 = sorted([c[1], d[1]])

        lo = max(a0, c0)
        hi = min(a1, c1)

        if hi < lo - tol:
            return None
        if abs(hi - lo) <= tol:
            return "endpoint_touch"
        return "colinear_overlap"

    # Touch / T-junction cases
    touches = []
    if s1 == 0 and _on_segment(a, b, c, tol):
        touches.append(c)
    if s2 == 0 and _on_segment(a, b, d, tol):
        touches.append(d)
    if s3 == 0 and _on_segment(c, d, a, tol):
        touches.append(a)
    if s4 == 0 and _on_segment(c, d, b, tol):
        touches.append(b)

    if not touches:
        return None

    # Shared endpoint only
    shared_endpoints = {tuple(a), tuple(b)}.intersection({tuple(c), tuple(d)})

    if shared_endpoints and len(touches) == 1:
        return "endpoint_touch"

    return "t_junction"


def _point_in_polygon(pt: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    x, y = pt
    wn = 0
    n = len(poly)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]

        cross = (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)

        if y0 <= y:
            if y1 > y and cross > 0:
                wn += 1
        else:
            if y1 <= y and cross < 0:
                wn -= 1

    return wn != 0


def _safe_face_seed(face_coords: list[tuple[float, float]]) -> tuple[float, float]:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(face_coords)

    for i in range(n):
        x0, y0 = face_coords[i]
        x1, y1 = face_coords[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area = 0.5 * area2
    if abs(area) > 1e-16:
        cx /= 6.0 * area
        cy /= 6.0 * area
        if _point_in_polygon((cx, cy), face_coords):
            return (cx, cy)

    mx = sum(x for x, _ in face_coords) / len(face_coords)
    my = sum(y for _, y in face_coords) / len(face_coords)
    if _point_in_polygon((mx, my), face_coords):
        return (mx, my)

    x0, y0 = face_coords[0]
    x1, y1 = face_coords[1]
    ex = x1 - x0
    ey = y1 - y0
    nx = -ey
    ny = ex
    L = (nx * nx + ny * ny) ** 0.5
    if L > 0:
        nx /= L
        ny /= L

    return (0.5 * (x0 + x1) + 1e-6 * nx, 0.5 * (y0 + y1) + 1e-6 * ny)
