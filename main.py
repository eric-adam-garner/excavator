from bench_io import load_benches_json
from domain_builder import build_partition_domain
from domain_validator import validate_partition_domain
from geometry.tolerance import recommend_tol
from plotter2d import (
    plot_half_edge_mesh,
    plot_triangle_mesh,
)
from triangle_backend import (
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
)

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)

    tol = recommend_tol(benches)

    domain = build_partition_domain(benches, tol)

    report = validate_partition_domain(domain)
    print(report.errors)

    tri_mesh = triangulate_partition_domain(domain, triangle_flags="pA")
    half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)

    assert max(tri_mesh.triangle_region_ids) == len(benches) - 1

    plot_triangle_mesh(
        tri_mesh,
        show_triangle_ids=False,
        domain_edges=domain.edges,
    )

    plot_half_edge_mesh(half_edge_mesh)
