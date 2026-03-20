from bench_io import load_benches_json
from domain_builder import build_partition_domain
from domain_validator import validate_partition_domain
from plotter2d import (
    plot_partition_domain,
    plot_triangle_mesh,
)
from tolerance import recommend_tol
from triangle_backend import (
    build_triangle_input,
    triangulate_partition_domain,
)

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)[:5]

    tol = recommend_tol(benches)

    domain = build_partition_domain(benches, tol)

    print("vertices:", domain.num_vertices())
    print("edges:", domain.num_edges())
    print("faces:", domain.num_faces())
    print("bench ids:", domain.face_bench_ids[:10])

    report = validate_partition_domain(domain)
    print(report.errors)

    for i in range(min(5, domain.num_faces())):
        print(i, domain.face_centroid(i), domain.face_bench_ids[i])

    # plot_partition_domain(domain)

    domain = build_partition_domain(benches, tol)

    mesh = triangulate_partition_domain(domain, triangle_flags="pA")

    print("mesh vertices:", len(mesh.vertices))
    print("mesh triangles:", len(mesh.triangles))
    print("mesh regions:", len(set(mesh.triangle_region_ids)))

    plot_triangle_mesh(
        mesh,
        show_triangle_ids=False,
        domain_edges=domain.edges,
    )
