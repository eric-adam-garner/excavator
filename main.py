from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from bench_io import (
    export_bench_slabs_obj,
    load_benches_json,
)
from domain_builder import build_partition_domain
from domain_validator import validate_partition_domain
from extrusion import build_extruded_connectivity_from_mesh
from geometry.tolerance import recommend_tol
from geometry.utils import point_in_polygon
from half_edge_mesh import (
    extract_outer_loop_from_mesh,
    extract_region_loops,
)
from plotter3d import plot_extrusion_vedo
from shell_builder import build_shell_domain
from triangle_backend import (
    TriangleMesh,
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
    triangulate_shell_domain,
    weld_triangle_meshes,
)

# TODO: add scrolling through levels in vedo plotter
# TODO: Log export event
# TODO: Add documentation
# TODO: Package

INPUT_BENCH_PATH = Path("data/input")
OUTPUT_SLAB_PATH = Path("data/output")

Z_INIT = 100.0


def iterate_descending(d: dict[float, Any]):
    for k in sorted(d.keys(), reverse=True):
        yield k, d[k]


if __name__ == "__main__":

    level_height_filename_map = {}
    for filename in os.listdir("data/input"):
        path = INPUT_BENCH_PATH / filename
        benches = load_benches_json(path)
        z = benches[0].points3d[0][2]
        level_height_filename_map[z] = filename

    level_id_height_map = {-(idx + 1): z for idx, (z, _) in enumerate(iterate_descending(level_height_filename_map))}

    zmin = min(level_height_filename_map.keys())
    zmax = max(level_height_filename_map.keys())

    assert zmax < Z_INIT

    z_prev = Z_INIT

    tol = recommend_tol(benches)

    super_loop = None

    level_outer_tri_meshes: list[TriangleMesh] = []
    level_inner_tri_meshes: list[TriangleMesh] = []

    for idx, (z, filename) in enumerate(iterate_descending(level_height_filename_map)):

        level_id = -(idx + 1)

        path = INPUT_BENCH_PATH / filename
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

        export_bench_slabs_obj(bench_tri_mesh, z, z_prev, path=OUTPUT_SLAB_PATH)

        shell_domain = build_shell_domain(
            super_loop=super_loop,
            outer_loop=outer_loop,
            tol=tol,
        )

        assert np.all(
            [point_in_polygon(pt, super_loop) for pt in outer_loop]
        ), "Outer loop boundary not contained in super loop"

        outer_tri_mesh = triangulate_shell_domain(
            shell_domain, outer_loop=super_loop, inner_loop=outer_loop, level_id=level_id
        )

        level_outer_tri_meshes.append(outer_tri_mesh)
        level_inner_tri_meshes.append(bench_tri_mesh)

        z_prev = z
        super_loop = outer_loop

    merged_outer_tri_mesh = outer_tri_mesh
    for msh in level_outer_tri_meshes:
        merged_outer_tri_mesh = weld_triangle_meshes(merged_outer_tri_mesh, msh, tol=tol)

    tri_mesh = weld_triangle_meshes(bench_tri_mesh, merged_outer_tri_mesh, tol=tol)
    half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)
    connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)

    plot_extrusion_vedo(
        connectivity=connectivity,
        triangle_region_ids=tri_mesh.triangle_region_ids,
        level_id_height_map=level_id_height_map,
        level_id=level_id,
        color_by_region=True,
        wireframe=False,
    )
