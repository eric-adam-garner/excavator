from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from excavator.geometry import Point2D
from excavator.geometry.utils import polygon_centroid


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
        face_region_ids:
            Bench id for each face.
        tol:
            Quantization / reconciliation tolerance.
    """

    vertices: list[Point2D]
    vertex_index: dict[tuple[int, int], int]
    edges: list[tuple[int, int]]
    faces: list[list[int]]
    face_region_ids: list[Any]
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
