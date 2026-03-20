from __future__ import annotations

from collections import defaultdict
from math import atan2

from geometry.reconcile import (
    deduplicate_segments,
    snap_vertices,
    split_segments,
)
from geometry.utils import (
    face_centroid,
    point_in_polygon,
    qpoint,
)
from partition_domain import PartitionDomain
from polyline import clean_polyline
from pslg import PSLG


def assign_face_bench_ids(faces, benches, tol):

    bench_polys = [clean_polyline(b.to_2d(), tol) for b in benches]

    face_ids = []

    for face in faces:
        c = face_centroid(face)

        assigned = None
        for i, poly in enumerate(bench_polys):
            if point_in_polygon(c, poly):
                assigned = benches[i].id
                break

        if assigned is None:
            raise RuntimeError("Face could not be assigned to any bench")

        face_ids.append(assigned)

    return face_ids


def extract_faces_from_edges(edges, tol):
    """
    Extract face cycles from a planar embedded undirected edge set.

    Args:
        edges:
            list of ((x0, y0), (x1, y1)) undirected edges
        tol:
            quantization tolerance for vertex identity

    Returns:
        list[list[(x, y)]]
            Directed face cycles as ordered vertex coordinates.
            Includes the outer face; filter by signed area later.
    """

    def q(p):
        return (round(p[0] / tol), round(p[1] / tol))

    def signed_area(loop):
        area = 0.0
        n = len(loop)
        for i in range(n):
            x0, y0 = loop[i]
            x1, y1 = loop[(i + 1) % n]
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    # ----------------------------------
    # Canonical vertex geometry
    # ----------------------------------
    geom = {}
    undirected_edges = set()

    for a, b in edges:
        qa = q(a)
        qb = q(b)
        geom[qa] = a
        geom[qb] = b
        if qa != qb:
            key = (qa, qb) if qa <= qb else (qb, qa)
            undirected_edges.add(key)

    # ----------------------------------
    # Build directed half-edges
    # ----------------------------------
    outgoing = defaultdict(list)
    halfedges = set()

    for u, v in undirected_edges:
        halfedges.add((u, v))
        halfedges.add((v, u))

    for u, v in halfedges:
        ux, uy = geom[u]
        vx, vy = geom[v]
        ang = atan2(vy - uy, vx - ux)
        outgoing[u].append((ang, v))

    for u in outgoing:
        outgoing[u].sort(key=lambda t: t[0])

    # ----------------------------------
    # Successor map:
    # face on the left of each half-edge
    # ----------------------------------
    successor = {}

    for u, v in halfedges:
        # At vertex v, find outgoing edge v->u in angular order
        nbrs = outgoing[v]
        nbr_vertices = [w for _, w in nbrs]

        idx = nbr_vertices.index(u)

        # Choose previous in CCW-sorted order => clockwise turn
        # This traces faces consistently on one side.
        next_idx = (idx - 1) % len(nbrs)
        w = nbrs[next_idx][1]

        successor[(u, v)] = (v, w)

    # ----------------------------------
    # Trace faces
    # ----------------------------------
    visited = set()
    faces = []

    for he_start in halfedges:
        if he_start in visited:
            continue

        face = []
        he = he_start

        while he not in visited:
            visited.add(he)
            u, v = he
            face.append(geom[u])
            he = successor[he]

        if len(face) >= 3:
            faces.append(face)

    # Optional: remove duplicate cycles caused by degenerate graph situations
    # Keep as-is for now; filter by area outside.
    return faces


def benches_to_partition_edges(benches, tol):
    """
    Build canonical unique edge set representing the planar partition.
    Shared boundaries appear exactly once.

    Returns:
        list[((x0,y0),(x1,y1))]
    """

    polylines = []
    for b in benches:
        poly = clean_polyline(b.to_2d(), tol)
        if len(poly) >= 2:
            polylines.append(poly)

    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)

    # canonical mode keeps one copy of every undirected edge
    edges = deduplicate_segments(polylines, tol, mode="canonical")

    return edges


def benches_to_boundary_pslg(benches, tol):
    """
    Build PSLG of the OUTER UNION BOUNDARY only.
    Shared internal edges are removed.

    Returns:
        PSLG
    """

    polylines = []
    for b in benches:
        poly = clean_polyline(b.to_2d(), tol)
        if len(poly) >= 2:
            polylines.append(poly)

    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)

    # boundary mode removes shared partition edges
    boundary_edges = deduplicate_segments(polylines, tol, mode="boundary")

    # trace loops from boundary edges
    boundary_faces = extract_faces_from_edges(boundary_edges, tol)

    # filter outer face artifacts
    boundary_faces = filter_faces(boundary_faces, tol)

    p = PSLG(tol=tol)

    for face in boundary_faces:
        if len(face) >= 3:
            p.add_polygon(face)

    return p


def filter_faces(faces, tol, min_area=None):
    """
    Remove obvious spurious faces:
    - repeated-vertex cycles
    - tiny-area cycles
    - largest-magnitude outer face
    """
    if not faces:
        return []

    def signed_area(loop):
        area = 0.0
        n = len(loop)
        for i in range(n):
            x0, y0 = loop[i]
            x1, y1 = loop[(i + 1) % n]
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    def has_repeated_vertices(loop):
        seen = set()
        for x, y in loop:
            q = (round(x / tol), round(y / tol))
            if q in seen:
                return True
            seen.add(q)
        return False

    areas = [signed_area(face) for face in faces]

    if min_area is None:
        max_abs_area = max(abs(a) for a in areas) if areas else 0.0
        min_area = max_abs_area * 1e-8

    candidates = []
    for face, area in zip(faces, areas):
        if len(face) < 3:
            continue
        if has_repeated_vertices(face):
            continue
        if abs(area) <= min_area:
            continue
        candidates.append((face, area))

    if not candidates:
        return []

    # remove the outer face = largest absolute area
    idx_outer = max(range(len(candidates)), key=lambda i: abs(candidates[i][1]))

    filtered = [face for i, (face, area) in enumerate(candidates) if i != idx_outer]

    return filtered


def build_partition_domain(benches, tol: float) -> PartitionDomain:
    """
    Build a canonical partition domain from bench polygons.

    Pipeline:
        bench polylines
        -> clean
        -> snap
        -> split
        -> canonical partition edges
        -> extract faces
        -> filter valid bounded faces
        -> assign bench ids
        -> index vertices / edges / faces
    """
    polylines = []
    for b in benches:
        poly = clean_polyline(b.to_2d(), tol)
        if len(poly) >= 2:
            polylines.append(poly)

    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)

    # For planar partition semantics:
    # keep one copy of every undirected edge.
    edges_xy = deduplicate_segments(polylines, tol, mode="canonical")

    all_faces_xy = extract_faces_from_edges(edges_xy, tol)
    faces_xy = filter_faces(all_faces_xy, tol, min_area=None)

    face_bench_ids = assign_face_bench_ids(faces_xy, benches, tol)

    vertex_index: dict[tuple[int, int], int] = {}
    vertices: list[tuple[float, float]] = []

    def get_vid(p: tuple[float, float]) -> int:
        key = qpoint(p, tol)
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append(p)
        return vertex_index[key]

    edges: list[tuple[int, int]] = []
    seen_edges: set[tuple[int, int]] = set()

    for a, b in edges_xy:
        ia = get_vid(a)
        ib = get_vid(b)
        if ia == ib:
            continue
        key = (ia, ib) if ia < ib else (ib, ia)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(key)

    faces: list[list[int]] = []
    for face in faces_xy:
        vids = [get_vid(p) for p in face]

        # Drop repeated closing vertex if present.
        if len(vids) > 1 and vids[0] == vids[-1]:
            vids = vids[:-1]

        # Defensive cleanup of consecutive duplicates.
        cleaned_vids = [vids[0]]
        for v in vids[1:]:
            if v != cleaned_vids[-1]:
                cleaned_vids.append(v)

        if len(cleaned_vids) >= 3:
            faces.append(cleaned_vids)

    if len(faces) != len(face_bench_ids):
        raise RuntimeError(f"Face count / bench-id count mismatch: {len(faces)} vs {len(face_bench_ids)}")

    return PartitionDomain(
        vertices=vertices,
        vertex_index=vertex_index,
        edges=edges,
        faces=faces,
        face_bench_ids=face_bench_ids,
        tol=tol,
    )
