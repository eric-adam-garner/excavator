from collections import defaultdict
from math import atan2

import numpy as np

from bench_io import load_benches_json
from geometry_reconcile import (
    deduplicate_segments,
    merge_colinear_segments,
    snap_vertices,
    split_segments,
)
from plotter2d import (
    plot_directed_edges,
    plot_faces,
    plot_pslg,
)
from polyline import clean_polyline
from pslg import PSLG


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


def estimate_geometry_scale(benches):
    xs = []
    ys = []

    for b in benches:
        for x, y, _ in b.points3d:
            xs.append(x)
            ys.append(y)

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    return max(dx, dy)


def estimate_spacing_scale(benches):
    lengths = []

    for b in benches:
        pts = b.points3d
        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            lengths.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    return np.median(lengths)


def estimate_noise_scale(benches):
    lengths = []

    for b in benches:
        pts = b.points3d
        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            lengths.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    lengths = np.array(lengths)
    return np.percentile(lengths, 1)


def recommend_tol(benches):
    scale = estimate_geometry_scale(benches)
    spacing = estimate_spacing_scale(benches)
    noise = estimate_noise_scale(benches)

    tol = max(
        noise * 2,
        spacing * 1e-3,
        scale * 1e-12,
    )

    return tol


def benches_to_unique_segments(benches, tol=1e-6):
    p = PSLG(tol=tol)

    polylines = []
    for b in benches:
        polyline = clean_polyline(b.to_2d(), tol)
        polylines.append(polyline)

    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)
    # polylines = merge_colinear_segments(polylines, tol) # seems like this destroys the new vertices from split... logically should not be done now...
    segments = deduplicate_segments(polylines, tol, mode="canonical")  # this does not seem to behave as described

    return segments


def benches_to_pslg(benches, tol=1e-6):
    p = PSLG(tol=tol)

    polylines = []
    for b in benches:
        polyline = clean_polyline(b.to_2d(), tol)
        polylines.append(polyline)

    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)
    # polylines = merge_colinear_segments(polylines, tol) # seems like this destroys the new vertices from split... logically should not be done now...
    # polylines = deduplicate_segments(polylines, tol)  # this does not seem to behave as described

    for polyline in polylines:
        p.add_polygon(polyline)

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

    filtered = [
        face
        for i, (face, area) in enumerate(candidates)
        if i != idx_outer
    ]

    return filtered

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)[:2]
    tol = recommend_tol(benches)
    edges = benches_to_unique_segments(benches, tol=tol)
    
    faces = extract_faces_from_edges(edges, tol)
    
    faces = filter_faces(faces, tol, min_area=None)
    print(len(faces))
    # print(faces)
    # for face in faces:
    #     print(len(face))
    plot_faces(
        faces,
        show_face_ids=True,
        show_areas=False,
        show_edges=True,
        alpha=0.35,
    )

    plot_directed_edges(edges, tol, show_ids=True, show_degrees=True)

    # p = benches_to_pslg(benches, tol=tol)

    # report = p.validate()
    # print(report)

    # plot_pslg(p, show_vertex_ids=True, show_loop_ids=True, show_segment_ids=True)
