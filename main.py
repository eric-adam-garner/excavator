from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from bench_io import (
    export_bench_slabs,
    load_benches_json,
)
from domain_builder import build_partition_domain
from domain_validator import validate_partition_domain
from extrusion import (
    build_extruded_connectivity_from_mesh,
    realize_extruded_vertices,
)
from geometry.tolerance import recommend_tol
from geometry.utils import point_in_polygon
from half_edge_mesh import (
    extract_outer_loop_from_mesh,
    extract_region_loops,
)
from plotter3d import plot_extrusion_vedo
from shell_builder import build_shell_domain
from triangle_backend import (
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
    triangulate_shell_domain,
    weld_triangle_meshes,
)

# TODO: add scrolling through levels in vedo plotter
# TODO: Log export event
# TODO: Add documentation
# TODO: Package


# TODO: retain all mesh faces above z for render in next excavation level
def generate_final_upper_mesh(benches, connectivity, triangle_region_ids, z_prev, z):

    faces = []

    for tri in connectivity.top_triangles:
        faces.append(tri)

    for tri in connectivity.wall_triangles_internal:
        faces.append(tri)

    for tri in connectivity.wall_triangles_outer:
        faces.append(tri)

    faces = np.asarray(faces, dtype=int)

    def build_region_z(bench, triangle_region_ids, z_prev, z):
        region_z = {}
        for bench_id in range(len(triangle_region_ids)):
            region_z[bench_id] = z if bench_id < bench else z_prev
        region_z[-1] = z_prev
        return region_z

    # ---------- initial geometry
    region_z = build_region_z(len(benches), triangle_region_ids, z_prev, z)
    verts3d = realize_extruded_vertices(connectivity, region_z)

    verts = np.asarray(verts3d, dtype=float)
    retained_faces = []
    for face in faces:
        cx, cy, cz = np.mean(verts[face], axis=0)
        if cz > z + tol:
            retained_faces.append(face)

    return trimesh.Trimesh(vertices=verts3d, faces=retained_faces)


def iterate_descending(d: dict[float, Any]):
    for k in sorted(d.keys(), reverse=True):
        yield k, d[k]


if __name__ == "__main__":

    bench_dir_path = Path("data/input")
    output_path = Path("data/output")

    level_heights_map = {}
    for filename in os.listdir("data/input"):
        path = bench_dir_path / filename
        benches = load_benches_json(path)
        z = benches[0].points3d[0][2]
        level_heights_map[z] = filename
        tol = recommend_tol(benches)

    z_prev = 100.0
    super_loop = None

    previous_level_meshes: list[trimesh.Trimesh] = []
    outer_tri_meshes = []

    zmin = min(level_heights_map.keys())

    level_id = -1
    level_map = {}
    for z, filename in iterate_descending(level_heights_map):

        level_map[level_id] = z

        path = bench_dir_path / filename
        benches = load_benches_json(path)

        domain = build_partition_domain(benches, tol)

        report = validate_partition_domain(domain)
        assert len(report.errors) == 0

        bench_tri_mesh = triangulate_partition_domain(domain, triangle_flags="pA")
        bench_half_edge_mesh = triangle_to_halfedge_mesh(bench_tri_mesh)

        if super_loop is None:
            xmin, ymin, xmax, ymax = bench_half_edge_mesh.bounding_box(padding=0.05)
            super_loop = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        assert max(bench_tri_mesh.triangle_region_ids) == len(benches) - 1

        outer_loop = extract_outer_loop_from_mesh(bench_half_edge_mesh)
        bench_loops = extract_region_loops(bench_half_edge_mesh)

        export_bench_slabs(bench_tri_mesh, z, z_prev, path=output_path)

        shell_domain = build_shell_domain(
            super_loop=super_loop,
            outer_loop=outer_loop,
            tol=tol,
        )

        # Check if outer loop is contained in super loop

        assert np.all(
            [point_in_polygon(pt, super_loop) for pt in outer_loop]
        ), "Outer loop boundary not contained in super loop"

        outer_tri_mesh = triangulate_shell_domain(
            shell_domain, outer_loop=super_loop, inner_loop=outer_loop, level_id=level_id
        )

        merged_outer_tri_mesh = outer_tri_mesh
        for msh in outer_tri_meshes:
            merged_outer_tri_mesh = weld_triangle_meshes(merged_outer_tri_mesh, msh, tol=tol)

        outer_tri_meshes.append(outer_tri_mesh)

        tri_mesh = weld_triangle_meshes(bench_tri_mesh, merged_outer_tri_mesh, tol=tol)
        half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)

        connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)

        if z == zmin:
            plot_extrusion_vedo(
                connectivity,
                triangle_region_ids=tri_mesh.triangle_region_ids,
                level_map=level_map,
                current_level=level_id,
                z_prev=z_prev,
                z=z,
                color_by_region=True,
                wireframe=False,
                static_meshes=[],
            )

        upper_mesh = generate_final_upper_mesh(benches, connectivity, tri_mesh.triangle_region_ids, z_prev, z)
        previous_level_meshes.append(upper_mesh)

        z_prev = z
        super_loop = outer_loop
        level_id -= 1
