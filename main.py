from __future__ import annotations

import numpy as np

from bench_io import load_benches_json
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
from plotter2d import (
    plot_half_edge_mesh,
    plot_mesh_edges,
    plot_partition_domain,
    plot_triangle_mesh,
)
from plotter3d import plot_extrusion_vedo
from shell_builder import build_shell_domain
from triangle_backend import (
    Triangle,
    TriangleMesh,
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
    triangulate_shell_domain,
    weld_triangle_meshes,
)

# TODO: merge outer tri mesh with tri mesh and then create new half edge mesh for extrusion

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)

    tol = recommend_tol(benches)

    domain = build_partition_domain(benches, tol)

    report = validate_partition_domain(domain)
    assert len(report.errors) == 0

    bench_tri_mesh = triangulate_partition_domain(domain, triangle_flags="pA")
    bench_half_edge_mesh = triangle_to_halfedge_mesh(bench_tri_mesh)

    xmin, ymin, xmax, ymax = bench_half_edge_mesh.bounding_box(padding=0.05)
    super_loop = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    assert max(bench_tri_mesh.triangle_region_ids) == len(benches) - 1

    outer_loop = extract_outer_loop_from_mesh(bench_half_edge_mesh)
    bench_loops = extract_region_loops(bench_half_edge_mesh)

    shell_domain = build_shell_domain(
        super_loop=super_loop,
        outer_loop=outer_loop,
        tol=tol,
    )

    outer_tri_mesh = triangulate_shell_domain(shell_domain, outer_loop=super_loop, inner_loop=outer_loop)

    tri_mesh = weld_triangle_meshes(bench_tri_mesh, outer_tri_mesh, tol=tol)
    half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)

    # plot_triangle_mesh(tri_mesh)
    # plot_triangle_mesh(tri_mesh)
    # plot_triangle_mesh(outer_tri_mesh)
    # plot_half_edge_mesh(half_edge_mesh)

    connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)
    region_z = np.random.choice([50, 100], len(benches))
    verts3d = realize_extruded_vertices(connectivity, region_z)

    plot_extrusion_vedo(connectivity, verts3d, color_by_region=True, wireframe=False)
