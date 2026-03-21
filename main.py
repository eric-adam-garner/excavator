from __future__ import annotations

from typing import Any

import numpy as np

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

# TODO: handle multiple levels
# TODO: animate excavation sequence


def iterate_descending(d: dict[float, Any]):
    for k in sorted(d.keys(), reverse=True):
        yield k, d[k]


if __name__ == "__main__":

    import os
    from pathlib import Path

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
    for z, filename in iterate_descending(level_heights_map):

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

        outer_tri_mesh = triangulate_shell_domain(shell_domain, outer_loop=super_loop, inner_loop=outer_loop)

        tri_mesh = weld_triangle_meshes(bench_tri_mesh, outer_tri_mesh, tol=tol)
        half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)

        connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)
        region_z = {id: np.random.choice([z, z_prev]) for id in set(tri_mesh.triangle_region_ids)}
        region_z[-1] = z_prev
        verts3d = realize_extruded_vertices(connectivity, region_z)

        plot_extrusion_vedo(connectivity, verts3d, color_by_region=True, wireframe=False)

        # TODO: retain all mesh faces above z for render in next excavation level
        z_prev = z
        super_loop = outer_loop
