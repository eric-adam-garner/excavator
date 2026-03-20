from bench_io import load_benches_json
from domain_builder import (
    assign_face_bench_ids,
    benches_to_boundary_pslg,
    benches_to_partition_edges,
    build_partition_domain,
    extract_faces_from_edges,
    filter_faces,
)
from domain_validator import validate_partition_domain
from plotter2d import (
    plot_directed_edges,
    plot_faces,
    plot_partition_domain,
    plot_pslg,
)
from polyline import clean_polyline
from tolerance import recommend_tol

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)[:10]

    tol = recommend_tol(benches)

    # edges = benches_to_partition_edges(benches, tol=tol)
    # # plot_directed_edges(edges, tol, show_ids=True, show_degrees=True)

    # faces = extract_faces_from_edges(edges, tol)
    # faces = filter_faces(faces, tol, min_area=None)

    # face_ids = assign_face_bench_ids(faces, benches, tol=tol)

    # plot_faces(
    #     faces,
    #     face_ids,
    #     show_face_ids=True,
    #     show_areas=False,
    #     show_edges=True,
    #     alpha=0.35,
    # )

    # p = benches_to_boundary_pslg(benches, tol)
    # report = p.validate()
    # print(report)
    # plot_pslg(p, show_vertex_ids=True, show_loop_ids=True, show_segment_ids=True)

    domain = build_partition_domain(benches, tol)

    print("vertices:", domain.num_vertices())
    print("edges:", domain.num_edges())
    print("faces:", domain.num_faces())
    print("bench ids:", domain.face_bench_ids[:10])
    
    report = validate_partition_domain(domain)
    print(report.errors)
    
    for i in range(min(5, domain.num_faces())):
        print(i, domain.face_centroid(i), domain.face_bench_ids[i])

    plot_partition_domain(domain)