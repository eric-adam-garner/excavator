from bench_io import load_benches_json
from domain_builder import (
    benches_to_boundary_pslg,
    benches_to_partition_edges,
    extract_faces_from_edges,
    filter_faces,
)
from plotter2d import (
    plot_directed_edges,
    plot_faces,
    plot_pslg,
)
from tolerance import recommend_tol

if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)[:10]
    tol = recommend_tol(benches)
    edges = benches_to_partition_edges(benches, tol=tol)
    plot_directed_edges(edges, tol, show_ids=True, show_degrees=True)
    faces = extract_faces_from_edges(edges, tol)
    faces = filter_faces(faces, tol, min_area=None)

    plot_faces(
        faces,
        show_face_ids=True,
        show_areas=False,
        show_edges=True,
        alpha=0.35,
    )

    p = benches_to_boundary_pslg(benches, tol)
    report = p.validate()
    print(report)
    plot_pslg(p, show_vertex_ids=True, show_loop_ids=True, show_segment_ids=True)
