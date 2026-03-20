from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExtrudedConnectivity:
    """
    Topology-only 3D extrusion connectivity.

    vertices_2d:
        Original 2D vertices from the partition domain.

    top_vertex_keys:
        Maps each top 3D vertex index to a pair:
            (base_vertex_id, region_id)

    top_triangles:
        Region-local top surface triangles, as indices into top_vertex_keys.

    wall_triangles_internal:
        Stitching triangles between adjacent regions.

    wall_triangles_outer:
        Exterior wall triangles from a region top edge to the base plane.

    base_vertices:
        Base plane vertices, one per original 2D vertex.

    base_triangles:
        Optional bottom surface triangles, if desired later.
    """

    vertices_2d: list[tuple[float, float]]
    top_vertex_keys: list[tuple[int, Any]]
    top_triangles: list[tuple[int, int, int]]
    top_triangle_regions: list[Any]
    wall_triangles_internal: list[tuple[int, int, int]]
    wall_triangles_outer: list[tuple[int, int, int]]
    base_vertices: list[int]
    base_triangles: list[tuple[int, int, int]]


def build_extruded_connectivity_from_mesh(mesh):
    """
    Build extrusion connectivity from a triangulated partition mesh.

    Returns
    -------
    ExtrudedConnectivity
    """

    # 2D base geometry
    vertices_2d = [(v.x, v.y) for v in mesh.vertices]

    # One base-plane vertex per original 2D vertex.
    # These are indexed after the top vertices later.
    base_vertices = list(range(len(vertices_2d)))

    # Region-local top vertices:
    # key = (base_vertex_id, region_id) -> top_vertex_index
    top_vertex_index: dict[tuple[int, object], int] = {}
    top_vertex_keys: list[tuple[int, object]] = []

    top_triangles: list[tuple[int, int, int]] = []
    top_triangle_regions: list[object] = []

    def get_top_vid(base_vid, region_id):
        key = (base_vid, region_id)
        if key not in top_vertex_index:
            top_vertex_index[key] = len(top_vertex_keys)
            top_vertex_keys.append(key)
        return top_vertex_index[key]

    # -------------------------
    # Top triangles
    # -------------------------
    for face, region_id in zip(mesh.faces, mesh.face_region_ids):

        # Adapt this accessor to your face API.
        a, b, c = face.vertex_ids()  # expected [a, b, c]

        ta = get_top_vid(a, region_id)
        tb = get_top_vid(b, region_id)
        tc = get_top_vid(c, region_id)

        top_triangles.append((ta, tb, tc))
        top_triangle_regions.append(region_id)

    # -------------------------
    # Internal wall triangles
    # -------------------------
    wall_triangles_internal: list[tuple[int, int, int]] = []

    # We need one wall quad per edge separating two different regions.
    # Best source is half-edge adjacency.
    seen_region_edges = set()

    for he in mesh.halfedges:
        if he.twin is None:
            continue

        f0 = he.face
        f1 = he.twin.face

        if f0 is None or f1 is None:
            continue

        r0 = f0.region_id
        r1 = f1.region_id

        if r0 == r1:
            continue

        a = he.origin.id
        b = he.dest.id

        # Canonical undirected + region pairing key
        edge_key = tuple(sorted((a, b)))
        region_key = tuple(sorted((r0, r1)))
        key = (edge_key, region_key)

        if key in seen_region_edges:
            continue
        seen_region_edges.add(key)

        a0 = get_top_vid(a, r0)
        b0 = get_top_vid(b, r0)
        a1 = get_top_vid(a, r1)
        b1 = get_top_vid(b, r1)

        # Two latent stitching triangles across the wall quad
        wall_triangles_internal.append((a0, b0, a1))
        wall_triangles_internal.append((b0, b1, a1))

    # -------------------------
    # Exterior wall triangles
    # -------------------------
    wall_triangles_outer: list[tuple[int, int, int]] = []

    # We will reference base-plane vertices using a separate index space later,
    # so for now just record local indices by offsetting after top vertices.
    base_offset = len(top_vertex_keys)

    for he in mesh.halfedges:
        if he.twin is not None:
            continue

        if he.face is None:
            continue

        region_id = he.face.region_id
        a = he.origin.id
        b = he.dest.id

        at = get_top_vid(a, region_id)
        bt = get_top_vid(b, region_id)

        ab = base_offset + a
        bb = base_offset + b

        wall_triangles_outer.append((at, bt, ab))
        wall_triangles_outer.append((bt, bb, ab))

    # -------------------------
    # Optional bottom surface
    # -------------------------
    base_triangles: list[tuple[int, int, int]] = []
    # Leave empty for now unless you want a closed solid immediately.

    return ExtrudedConnectivity(
        vertices_2d=vertices_2d,
        top_vertex_keys=top_vertex_keys,
        top_triangles=top_triangles,
        top_triangle_regions=top_triangle_regions,
        wall_triangles_internal=wall_triangles_internal,
        wall_triangles_outer=wall_triangles_outer,
        base_vertices=base_vertices,
        base_triangles=base_triangles,
    )


def realize_extruded_vertices(connectivity, region_z, base_z=0.0):
    """
    Create 3D vertex coordinates from fixed connectivity.
    """
    verts3d = []

    # top region-local vertices
    for base_vid, region_id in connectivity.top_vertex_keys:
        x, y = connectivity.vertices_2d[base_vid]
        z = region_z[region_id]
        verts3d.append((x, y, z))

    # base vertices
    for base_vid in connectivity.base_vertices:
        x, y = connectivity.vertices_2d[base_vid]
        verts3d.append((x, y, base_z))

    return verts3d
