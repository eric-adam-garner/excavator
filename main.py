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
from polyline import clean_polyline
from tolerance import recommend_tol


def point_in_polygon(pt, poly):
    x, y = pt
    wn = 0
    n = len(poly)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]

        if y0 <= y:
            if y1 > y and (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) > 0:
                wn += 1
        else:
            if y1 <= y and (x1 - x0) * (y - y0) - (x - x0) * (y1 - y0) < 0:
                wn -= 1

    return wn != 0


def face_centroid(face):
    A = 0
    cx = 0
    cy = 0
    n = len(face)

    for i in range(n):
        x0, y0 = face[i]
        x1, y1 = face[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        A += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    A *= 0.5
    if abs(A) < 1e-16:
        return face[0]

    cx /= 6 * A
    cy /= 6 * A
    return (cx, cy)


def assign_face_bench_ids(faces, benches, tol):

    bench_polys = [clean_polyline(b.to_2d(), tol) for b in benches]

    face_ids = []

    for face in faces:
        c = face_centroid(face)

        assigned = None
        for i, poly in enumerate(bench_polys):
            if point_in_polygon(c, poly):
                assigned = benches[i].id
                break

        if assigned is None:
            raise RuntimeError("Face could not be assigned to any bench")

        face_ids.append(assigned)

    return face_ids


if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)[:10]

    tol = recommend_tol(benches)

    edges = benches_to_partition_edges(benches, tol=tol)
    # plot_directed_edges(edges, tol, show_ids=True, show_degrees=True)

    faces = extract_faces_from_edges(edges, tol)
    faces = filter_faces(faces, tol, min_area=None)

    face_ids = assign_face_bench_ids(faces, benches, tol=tol)

    plot_faces(
        faces,
        face_ids,
        show_face_ids=True,
        show_areas=False,
        show_edges=True,
        alpha=0.35,
    )

    p = benches_to_boundary_pslg(benches, tol)
    report = p.validate()
    print(report)
    plot_pslg(p, show_vertex_ids=True, show_loop_ids=True, show_segment_ids=True)
