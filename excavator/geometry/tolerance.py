import numpy as np


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
