from __future__ import annotations

from dataclasses import dataclass
from typing import Any


Point2D = tuple[float, float]
Edge2D = tuple[Point2D, Point2D]
Face2D = list[Point2D]


@dataclass(frozen=True)
class PartitionDomain:
    """
    Canonical mesher input for a planar partition.

    Attributes:
        vertices:
            Unique 2D vertices.
        vertex_index:
            Mapping from quantized vertex key to vertex index.
        edges:
            Constrained partition edges as pairs of vertex indices.
        faces:
            Faces as ordered vertex-index loops.
        face_bench_ids:
            Bench id for each face.
        tol:
            Quantization / reconciliation tolerance.
    """
    vertices: list[Point2D]
    vertex_index: dict[tuple[int, int], int]
    edges: list[tuple[int, int]]
    faces: list[list[int]]
    face_bench_ids: list[Any]
    tol: float

    def face_centroid(self, face_idx: int) -> Point2D:
        face = self.faces[face_idx]
        pts = [self.vertices[i] for i in face]
        return polygon_centroid(pts)

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_edges(self) -> int:
        return len(self.edges)

    def num_faces(self) -> int:
        return len(self.faces)


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

    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return (cx, cy)


def qpoint(p: Point2D, tol: float) -> tuple[int, int]:
    return (round(p[0] / tol), round(p[1] / tol))