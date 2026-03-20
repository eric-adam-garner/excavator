import numpy as np
from geometry import Point2D


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


def point_on_segment(p, a, b, tol):
    """
    Check if point p lies on segment a-b within tolerance.
    """
    ax, ay = a
    bx, by = b
    px, py = p

    # bounding box check
    if px < min(ax, bx) - tol or px > max(ax, bx) + tol:
        return False
    if py < min(ay, by) - tol or py > max(ay, by) + tol:
        return False

    # colinearity check via area
    area = abs((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    if area > tol * max(1.0, np.hypot(bx - ax, by - ay)):
        return False

    return True


def colinear(a, b, c, tol):
    """
    Return True if a, b, c are colinear within tolerance.
    """
    ax, ay = a
    bx, by = b
    cx, cy = c
    area = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    return area <= tol * max(1.0, np.hypot(bx - ax, by - ay))


def polygon_signed_area(poly: list[Point2D]) -> float:
    area = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return 0.5 * area


def polygon_centroid(poly: list[Point2D]) -> Point2D:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    n = len(poly)

    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross

    area = 0.5 * area2
    if abs(area) < 1e-16:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    cx /= 6.0 * area
    cy /= 6.0 * area
    return (cx, cy)


def qpoint(p: Point2D, tol: float) -> tuple[int, int]:
    return (round(p[0] / tol), round(p[1] / tol))
