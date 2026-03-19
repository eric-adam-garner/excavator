import numpy as np

from bench_io import load_benches_json
from geometry_reconcile import snap_vertices, split_segments, merge_colinear_segments
from plotter2d import plot_pslg
from polyline import clean_polyline
from pslg import PSLG


def estimate_geometry_scale(benches):
    xs = []
    ys = []

    for b in benches:
        for x, y, _ in b.points3d:
            xs.append(x)
            ys.append(y)

    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    return max(dx, dy)


def estimate_spacing_scale(benches):
    lengths = []

    for b in benches:
        pts = b.points3d
        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            lengths.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    return np.median(lengths)


def estimate_noise_scale(benches):
    lengths = []

    for b in benches:
        pts = b.points3d
        for i in range(len(pts) - 1):
            x0, y0, _ = pts[i]
            x1, y1, _ = pts[i + 1]
            lengths.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    lengths = np.array(lengths)
    return np.percentile(lengths, 1)


def recommend_tol(benches):
    scale = estimate_geometry_scale(benches)
    spacing = estimate_spacing_scale(benches)
    noise = estimate_noise_scale(benches)

    tol = max(
        noise * 2,
        spacing * 1e-3,
        scale * 1e-12,
    )

    return tol


def benches_to_pslg(benches, tol=1e-6):
    p = PSLG(tol=tol)

    polylines = []
    for b in benches:
        polyline = clean_polyline(b.to_2d(), tol)
        polylines.append(polyline)
        
    polylines = snap_vertices(polylines, tol)
    polylines = split_segments(polylines, tol)
    polylines = merge_colinear_segments(polylines, tol)
    
    for polyline in polylines:
        p.add_polygon(polyline)

    return p


if __name__ == "__main__":

    path = "data/take-home_360.json"
    benches = load_benches_json(path)
    tol = recommend_tol(benches)
    p = benches_to_pslg(benches, tol=tol)

    report = p.validate()
    print(report)

    plot_pslg(p, show_vertex_ids=False, show_loop_ids=True)
