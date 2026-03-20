from __future__ import annotations

from geometry import Point2D
from geometry.utils import qpoint
from partition_domain import PartitionDomain
from polyline import clean_polyline


def build_shell_domain(super_loop, outer_loop, tol: float) -> PartitionDomain:
    """
    Build a shell domain matching PartitionDomain format.

    Represents:
        super_loop − outer_loop

    as a single-face planar partition with constrained inner boundary.
    """

    super_loop = clean_polyline(super_loop, tol)
    outer_loop = clean_polyline(outer_loop, tol)

    vertex_index: dict[tuple[int, int], int] = {}
    vertices: list[Point2D] = []

    def get_vid(p: Point2D) -> int:
        key = qpoint(p, tol)
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append(p)
        return vertex_index[key]

    # --- build face loop (super boundary)
    face_vids = [get_vid(p) for p in super_loop]

    if len(face_vids) > 1 and face_vids[0] == face_vids[-1]:
        face_vids = face_vids[:-1]

    cleaned_face = [face_vids[0]]
    for v in face_vids[1:]:
        if v != cleaned_face[-1]:
            cleaned_face.append(v)

    if len(cleaned_face) < 3:
        raise RuntimeError("Super loop degenerate after cleaning.")

    # --- build constraint edges (super + outer)
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_loop(loop):
        vids = [get_vid(p) for p in loop]
        if len(vids) > 1 and vids[0] == vids[-1]:
            vids = vids[:-1]

        n = len(vids)
        for i in range(n):
            a = vids[i]
            b = vids[(i + 1) % n]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                edges.append(key)

    add_loop(super_loop)
    add_loop(outer_loop)

    faces = [cleaned_face]
    face_region_ids = [None]

    return PartitionDomain(
        vertices=vertices,
        vertex_index=vertex_index,
        edges=edges,
        faces=faces,
        face_region_ids=face_region_ids,
        tol=tol,
    )
