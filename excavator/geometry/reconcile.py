from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from excavator.geometry.utils import (
    colinear,
    point_on_segment,
)


def snap_vertices(polylines, tol):

    pts = np.array([p for poly in polylines for p in poly])
    tree = cKDTree(pts)

    visited = np.zeros(len(pts), dtype=bool)
    mapping = {}

    for i in range(len(pts)):
        if visited[i]:
            continue

        idx = tree.query_ball_point(pts[i], tol)
        cluster = pts[idx]
        centroid = cluster.mean(axis=0)

        for j in idx:
            mapping[j] = centroid
            visited[j] = True

    # rebuild polylines
    snapped = []
    k = 0
    for poly in polylines:
        new = []
        for _ in poly:
            new.append(tuple(mapping[k]))
            k += 1
        snapped.append(new)

    return snapped


def split_segments(polylines, tol):
    """
    Global segment splitter:
        - builds global segment list
        - splits segments at foreign vertices
        - returns new polylines with inserted split points
    """
    # collect all vertices
    vertices = [p for poly in polylines for p in poly]

    # build segment list
    segments = []
    for poly in polylines:
        n = len(poly)
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            segments.append((a, b))

    # for each segment, find interior points
    split_map = {}

    for si, (a, b) in enumerate(segments):
        split_pts = []

        for p in vertices:
            if p == a or p == b:
                continue
            if point_on_segment(p, a, b, tol):
                split_pts.append(p)

        if split_pts:
            # sort along segment parameter
            ax, ay = a
            bx, by = b
            length2 = (bx - ax) ** 2 + (by - ay) ** 2

            def t_param(p):
                px, py = p
                return ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / length2

            split_pts = sorted(split_pts, key=t_param)
            split_map[si] = split_pts

    # rebuild polylines with splits
    new_polys = []
    seg_idx = 0

    for poly in polylines:
        n = len(poly)
        new_poly = []

        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]

            new_poly.append(a)

            if seg_idx in split_map:
                new_poly.extend(split_map[seg_idx])

            seg_idx += 1

        new_polys.append(new_poly)

    return new_polys


def merge_colinear_segments(polylines, tol):
    """
    Merge consecutive colinear vertices in each closed polyline.

    This removes redundant intermediate vertices introduced by:
        - snapping
        - segment splitting
        - original CAD sampling differences

    Notes:
        - Treats each polyline as closed.
        - Preserves the cyclic order of the boundary.
        - Removes vertices b where a-b-c are colinear within tolerance.
    """
    merged = []

    for poly in polylines:
        if len(poly) < 3:
            merged.append(poly[:])
            continue

        pts = poly[:]
        changed = True

        while changed and len(pts) >= 3:
            changed = False
            new_pts = []
            n = len(pts)

            for i in range(n):
                a = pts[(i - 1) % n]
                b = pts[i]
                c = pts[(i + 1) % n]

                # Drop duplicate consecutive vertices and colinear middle vertices.
                if b == a:
                    changed = True
                    continue

                if colinear(a, b, c, tol):
                    changed = True
                    continue

                new_pts.append(b)

            pts = new_pts

        merged.append(pts)

    return merged


def deduplicate_segments(polylines, tol, mode="canonical"):
    """
    Convert polygon soup → canonical global edge set.

    Args:
        polylines:
            list[list[(x,y)]], assumed closed loops
        tol:
            snapping tolerance
        mode:
            "canonical" → keep one copy of every undirected edge
            "boundary"  → keep only edges with multiplicity == 1
                          (use when benches form a partition)

    Returns:
        list[((x0,y0),(x1,y1))] directed edges
    """

    def is_degenerate(a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        return dx * dx + dy * dy <= tol * tol

    def qpoint(p):
        return (round(p[0] / tol), round(p[1] / tol))

    edge_count = defaultdict(int)
    edge_geom = {}

    # -------------------------
    # Count undirected edges
    # -------------------------
    for poly in polylines:
        n = len(poly)
        if n < 2:
            continue

        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]

            if is_degenerate(a, b):
                continue

            qa = qpoint(a)
            qb = qpoint(b)

            if qa <= qb:
                key = (qa, qb)
                geom = (a, b)
            else:
                key = (qb, qa)
                geom = (b, a)

            edge_count[key] += 1
            if key not in edge_geom:
                edge_geom[key] = geom

    # -------------------------
    # Select edges
    # -------------------------
    edges = []

    for key, count in edge_count.items():

        if mode == "boundary":
            if count != 1:
                continue

        geom = edge_geom[key]
        edges.append(geom)

    return edges
