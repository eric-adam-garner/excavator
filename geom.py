from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import Iterable


@dataclass
class Vertex:
    id: int
    x: float
    y: float
    halfedge: HalfEdge | None = None

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def __repr__(self) -> str:
        return f"Vertex(id={self.id}, x={self.x:.6g}, y={self.y:.6g})"


@dataclass
class Face:
    id: int
    halfedge: HalfEdge | None = None

    def iter_halfedges(self) -> Iterable[HalfEdge]:
        """
        Iterate over halfedges forming this face boundary.
        """
        if self.halfedge is None:
            return

        start = self.halfedge
        current = start
        yield current

        current = current.next
        while current is not None and current is not start:
            yield current
            current = current.next

    def vertex_ids(self) -> list[int]:
        return [he.origin.id for he in self.iter_halfedges()]

    def __repr__(self) -> str:
        return f"Face(id={self.id}, vertex_ids={self.vertex_ids()})"


@dataclass
class HalfEdge:
    id: int
    origin: Vertex
    face: Face | None = None
    next: HalfEdge | None = None
    prev: HalfEdge | None = None
    twin: HalfEdge | None = None

    @property
    def dest(self) -> Vertex | None:
        if self.next is None:
            return None
        return self.next.origin

    @property
    def edge_key(self) -> tuple[int, int] | None:
        """
        Directed edge key (origin -> dest).
        """
        if self.dest is None:
            return None
        return (self.origin.id, self.dest.id)

    @property
    def undirected_edge_key(self) -> tuple[int, int] | None:
        """
        Undirected edge key for edge-pair matching.
        """
        if self.dest is None:
            return None
        a = self.origin.id
        b = self.dest.id
        return (a, b) if a < b else (b, a)

    @property
    def is_boundary(self) -> bool:
        return self.twin is None

    def __repr__(self) -> str:
        dest_id = self.dest.id if self.dest is not None else None
        face_id = self.face.id if self.face is not None else None
        twin_id = self.twin.id if self.twin is not None else None
        return (
            f"HalfEdge(id={self.id}, origin={self.origin.id}, dest={dest_id}, "
            f"face={face_id}, twin={twin_id})"
        )


@dataclass
class Mesh:
    vertices: list[Vertex] = field(default_factory=list)
    halfedges: list[HalfEdge] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)

    @classmethod
    def from_vertices_and_faces(
        cls,
        vertices_xy: list[tuple[float, float]],
        face_indices: list[list[int]],
    ) -> Mesh:
        """
        Build a half-edge mesh from vertex coordinates and polygon face indices.

        Args:
            vertices_xy:
                List of (x, y) coordinates.
            face_indices:
                List of polygon faces, where each face is a list of vertex indices.

        Returns:
            Mesh:
                Constructed half-edge mesh with next/prev/face/twin relations.

        Raises:
            ValueError:
                If a face is degenerate, contains repeated consecutive vertices,
                has fewer than 3 vertices, or if a non-manifold edge is detected.
        """
        mesh = cls()

        mesh.vertices = [
            Vertex(id=i, x=xy[0], y=xy[1])
            for i, xy in enumerate(vertices_xy)
        ]

        edge_map: dict[tuple[int, int], HalfEdge] = {}
        undirected_counts: dict[tuple[int, int], int] = {}

        halfedge_id = 0

        for face_id, face_vertex_ids in enumerate(face_indices):
            if len(face_vertex_ids) < 3:
                raise ValueError(
                    f"Face {face_id} has fewer than 3 vertices: {face_vertex_ids}"
                )

            n = len(face_vertex_ids)

            for i in range(n):
                a = face_vertex_ids[i]
                b = face_vertex_ids[(i + 1) % n]
                if a == b:
                    raise ValueError(
                        f"Face {face_id} has repeated consecutive vertex {a}"
                    )

            face = Face(id=face_id)
            mesh.faces.append(face)

            face_halfedges: list[HalfEdge] = []

            for vid in face_vertex_ids:
                if vid < 0 or vid >= len(mesh.vertices):
                    raise ValueError(
                        f"Face {face_id} references invalid vertex index {vid}"
                    )

                he = HalfEdge(
                    id=halfedge_id,
                    origin=mesh.vertices[vid],
                    face=face,
                )
                halfedge_id += 1
                face_halfedges.append(he)
                mesh.halfedges.append(he)

                if he.origin.halfedge is None:
                    he.origin.halfedge = he

            for i, he in enumerate(face_halfedges):
                he.next = face_halfedges[(i + 1) % n]
                he.prev = face_halfedges[(i - 1) % n]

            face.halfedge = face_halfedges[0]

            for he in face_halfedges:
                directed = he.edge_key
                reverse = None if directed is None else (directed[1], directed[0])
                undirected = he.undirected_edge_key

                if directed is None or undirected is None:
                    raise ValueError("Encountered halfedge with incomplete connectivity")

                if directed in edge_map:
                    raise ValueError(
                        f"Duplicate directed edge detected: {directed}. "
                        "This usually indicates duplicate faces or inconsistent input."
                    )

                if reverse in edge_map:
                    twin = edge_map[reverse]
                    if twin.twin is not None:
                        raise ValueError(
                            f"Non-manifold edge detected on undirected edge {undirected}"
                        )
                    he.twin = twin
                    twin.twin = he

                edge_map[directed] = he
                undirected_counts[undirected] = undirected_counts.get(undirected, 0) + 1

                if undirected_counts[undirected] > 2:
                    raise ValueError(
                        f"Non-manifold edge detected on undirected edge {undirected}"
                    )

        return mesh

    def boundary_halfedges(self) -> list[HalfEdge]:
        """
        Return all halfedges that do not have a twin.
        """
        return [he for he in self.halfedges if he.is_boundary]

    def interior_halfedges(self) -> list[HalfEdge]:
        """
        Return all halfedges that do have a twin.
        """
        return [he for he in self.halfedges if not he.is_boundary]

    def boundary_edges(self) -> list[tuple[int, int]]:
        """
        Return directed boundary edges as (origin, dest).
        """
        edges: list[tuple[int, int]] = []
        for he in self.boundary_halfedges():
            if he.dest is None:
                raise ValueError(f"Halfedge {he.id} has no destination")
            edges.append((he.origin.id, he.dest.id))
        return edges

    def face_vertex_lists(self) -> list[list[int]]:
        """
        Return each face as a vertex-id list.
        """
        return [face.vertex_ids() for face in self.faces]

    def summary(self) -> str:
        num_boundary_halfedges = len(self.boundary_halfedges())
        num_interior_halfedges = len(self.interior_halfedges())

        return (
            f"Mesh(\n"
            f"  vertices={len(self.vertices)},\n"
            f"  halfedges={len(self.halfedges)},\n"
            f"  faces={len(self.faces)},\n"
            f"  boundary_halfedges={num_boundary_halfedges},\n"
            f"  interior_halfedges={num_interior_halfedges}\n"
            f")"
        )