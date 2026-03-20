from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from typing import Iterable


@dataclass
class TopologyReport:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        lines = [f"TopologyReport(is_valid={self.is_valid})"]

        if self.errors:
            lines.append("Errors:")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("Warnings:")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        if self.stats:
            lines.append("Stats:")
            for key, value in self.stats.items():
                lines.append(f"  - {key}: {value}")

        return "\n".join(lines)


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
    region_id: int | None = None

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
        return f"HalfEdge(id={self.id}, origin={self.origin.id}, dest={dest_id}, " f"face={face_id}, twin={twin_id})"


@dataclass
class Mesh:
    vertices: list[Vertex] = field(default_factory=list)
    halfedges: list[HalfEdge] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    face_region_ids: list[int] | None = None

    @classmethod
    def from_vertices_and_faces(
        cls,
        vertices_xy: list[tuple[float, float]],
        triangles: list[list[int]],
        triangle_region_ids: list[int] | None = None,
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

        mesh.vertices = [Vertex(id=i, x=xy[0], y=xy[1]) for i, xy in enumerate(vertices_xy)]

        mesh.face_region_ids = triangle_region_ids

        edge_map: dict[tuple[int, int], HalfEdge] = {}
        undirected_counts: dict[tuple[int, int], int] = {}

        halfedge_id = 0

        for face_id, face_vertex_ids in enumerate(triangles):
            if len(face_vertex_ids) < 3:
                raise ValueError(f"Face {face_id} has fewer than 3 vertices: {face_vertex_ids}")

            n = len(face_vertex_ids)

            for i in range(n):
                a = face_vertex_ids[i]
                b = face_vertex_ids[(i + 1) % n]
                if a == b:
                    raise ValueError(f"Face {face_id} has repeated consecutive vertex {a}")

            face = Face(id=face_id)
            mesh.faces.append(face)

            face_halfedges: list[HalfEdge] = []

            for vid in face_vertex_ids:
                if vid < 0 or vid >= len(mesh.vertices):
                    raise ValueError(f"Face {face_id} references invalid vertex index {vid}")

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
                        raise ValueError(f"Non-manifold edge detected on undirected edge {undirected}")
                    he.twin = twin
                    twin.twin = he

                edge_map[directed] = he
                undirected_counts[undirected] = undirected_counts.get(undirected, 0) + 1

                if undirected_counts[undirected] > 2:
                    raise ValueError(f"Non-manifold edge detected on undirected edge {undirected}")

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
        return [face.vertex_ids() for face in self.faces]

    def boundary_successor(self, he: HalfEdge) -> HalfEdge:
        """
        Return the next boundary halfedge along the same boundary loop.

        This assumes `he` is a boundary halfedge. The successor is found by
        rotating around the destination vertex until the next boundary halfedge
        is reached.
        """
        if not he.is_boundary:
            raise ValueError(f"Halfedge {he.id} is not a boundary halfedge")

        candidate = he.next
        if candidate is None:
            raise ValueError(f"Halfedge {he.id} has no next pointer")

        guard = 0
        max_steps = max(1, len(self.halfedges))

        while candidate.twin is not None:
            candidate = candidate.twin.next
            if candidate is None:
                raise ValueError("Encountered broken topology while tracing boundary successor")

            guard += 1
            if guard > max_steps:
                raise ValueError(
                    f"Boundary successor walk exceeded {max_steps} steps. " "Topology is likely inconsistent."
                )

        return candidate

    def trace_boundary_loop_halfedges(self, start_he: HalfEdge) -> list[HalfEdge]:
        """
        Trace one complete boundary loop starting from a boundary halfedge.
        """
        if not start_he.is_boundary:
            raise ValueError(f"Halfedge {start_he.id} is not a boundary halfedge")

        loop: list[HalfEdge] = []
        current = start_he

        guard = 0
        max_steps = max(1, len(self.halfedges) + 1)

        while True:
            loop.append(current)
            current = self.boundary_successor(current)

            guard += 1
            if guard > max_steps:
                raise ValueError(
                    f"Boundary loop tracing exceeded {max_steps} steps. " "Topology is likely inconsistent."
                )

            if current is start_he:
                break

        return loop

    def trace_boundary_loops_halfedges(self) -> list[list[HalfEdge]]:
        """
        Trace all boundary loops as lists of halfedges.
        """
        loops: list[list[HalfEdge]] = []
        visited: set[int] = set()

        for he in self.boundary_halfedges():
            if he.id in visited:
                continue

            loop = self.trace_boundary_loop_halfedges(he)
            loops.append(loop)

            for loop_he in loop:
                visited.add(loop_he.id)

        return loops

    def trace_boundary_loops_vertex_ids(self) -> list[list[int]]:
        """
        Trace all boundary loops as ordered lists of vertex ids.
        """
        loops_halfedges = self.trace_boundary_loops_halfedges()
        return [[he.origin.id for he in loop] for loop in loops_halfedges]

    def trace_boundary_loops_coords(self) -> list[list[tuple[float, float]]]:
        """
        Trace all boundary loops as ordered lists of vertex coordinates.
        """
        loops_halfedges = self.trace_boundary_loops_halfedges()
        return [[he.origin.xy for he in loop] for loop in loops_halfedges]

    def _boundary_loop_signed_area(self, loop_halfedges):
        """
        Compute signed area of a boundary loop.

        Positive → CCW → outer boundary
        Negative → CW → hole
        """
        area = 0.0

        for he in loop_halfedges:
            x0, y0 = he.origin.xy
            x1, y1 = he.dest.xy
            area += x0 * y1 - x1 * y0

        return area * 0.5

    def classify_boundary_loops(self):
        """
        Classify boundary loops into outer loops and holes.

        Returns
        -------
        outer_loops : list[list[int]]
        hole_loops : list[list[int]]
        """

        loops_he = self.trace_boundary_loops_halfedges()

        outer_loops = []
        hole_loops = []

        for loop in loops_he:
            area = self._boundary_loop_signed_area(loop)

            verts = [he.origin.id for he in loop]

            if area > 0:
                outer_loops.append(verts)
            else:
                hole_loops.append(verts)

        return outer_loops, hole_loops

    def num_vertices(self) -> int:
        return len(self.vertices)

    def num_faces(self) -> int:
        return len(self.faces)

    def num_edges(self) -> int:
        """
        Count undirected edges.
        """
        edges = set()

        for he in self.halfedges:
            if he.dest is None:
                raise ValueError(f"Halfedge {he.id} has no destination")
            a = he.origin.id
            b = he.dest.id
            edges.add((a, b) if a < b else (b, a))

        return len(edges)

    def euler_characteristic(self) -> int:
        return self.num_vertices() - self.num_edges() + self.num_faces()

    def _face_signed_area(self, face: Face) -> float:
        halfedges = list(face.iter_halfedges())
        area = 0.0

        for he in halfedges:
            x0, y0 = he.origin.xy
            x1, y1 = he.dest.xy
            area += x0 * y1 - x1 * y0

        return 0.5 * area

    def _face_connected_components(self) -> list[list[int]]:
        """
        Connected components over faces via twin-adjacency.
        Returns lists of face ids.
        """
        face_to_neighbors: dict[int, set[int]] = {face.id: set() for face in self.faces}

        for he in self.halfedges:
            if he.face is None or he.twin is None or he.twin.face is None:
                continue
            if he.face.id != he.twin.face.id:
                face_to_neighbors[he.face.id].add(he.twin.face.id)

        visited: set[int] = set()
        components: list[list[int]] = []

        for face in self.faces:
            if face.id in visited:
                continue

            stack = [face.id]
            comp: list[int] = []

            while stack:
                fid = stack.pop()
                if fid in visited:
                    continue

                visited.add(fid)
                comp.append(fid)

                for nbr in face_to_neighbors[fid]:
                    if nbr not in visited:
                        stack.append(nbr)

            components.append(comp)

        return components

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

    def validate_topology(self, strict: bool = True) -> TopologyReport:
        errors: list[str] = []
        warnings: list[str] = []

        # -------------------------------------------------
        # Basic entity counts
        # -------------------------------------------------
        num_vertices = len(self.vertices)
        num_halfedges = len(self.halfedges)
        num_faces = len(self.faces)

        # -------------------------------------------------
        # Halfedge pointer consistency
        # -------------------------------------------------
        for he in self.halfedges:
            if he.origin is None:
                errors.append(f"Halfedge {he.id} has no origin")

            if he.next is None:
                errors.append(f"Halfedge {he.id} has no next pointer")

            if he.prev is None:
                errors.append(f"Halfedge {he.id} has no prev pointer")

            if he.face is None:
                errors.append(f"Halfedge {he.id} has no face")

            if he.next is not None and he.next.prev is not he:
                errors.append(f"Halfedge {he.id} violates next/prev consistency with halfedge {he.next.id}")

            if he.prev is not None and he.prev.next is not he:
                errors.append(f"Halfedge {he.id} violates prev/next consistency with halfedge {he.prev.id}")

            if he.twin is not None and he.twin.twin is not he:
                errors.append(f"Halfedge {he.id} violates twin symmetry with halfedge {he.twin.id}")

            if he.dest is None:
                errors.append(f"Halfedge {he.id} has no destination")

            if he.dest is not None and he.origin.id == he.dest.id:
                errors.append(f"Halfedge {he.id} is degenerate ({he.origin.id} -> {he.dest.id})")

        # -------------------------------------------------
        # Face cycle consistency
        # -------------------------------------------------
        seen_face_ids = set()

        for face in self.faces:
            if face.id in seen_face_ids:
                errors.append(f"Duplicate face id detected: {face.id}")
            seen_face_ids.add(face.id)

            if face.halfedge is None:
                errors.append(f"Face {face.id} has no representative halfedge")
                continue

            loop = list(face.iter_halfedges())

            if len(loop) < 3:
                errors.append(f"Face {face.id} has fewer than 3 halfedges")

            seen_in_loop = set()
            for he in loop:
                if he.id in seen_in_loop:
                    errors.append(f"Face {face.id} repeats halfedge {he.id} in its cycle")
                seen_in_loop.add(he.id)

                if he.face is not face:
                    errors.append(
                        f"Face {face.id} cycle includes halfedge {he.id} belonging to face "
                        f"{he.face.id if he.face else None}"
                    )

            area = self._face_signed_area(face) if len(loop) >= 3 else 0.0
            if area == 0.0:
                errors.append(f"Face {face.id} has zero signed area")
            elif area < 0:
                warnings.append(f"Face {face.id} has negative signed area (clockwise winding)")

        # -------------------------------------------------
        # Duplicate face detection (by cyclic vertex set/order normalized)
        # -------------------------------------------------
        normalized_faces: set[tuple[int, ...]] = set()

        for face in self.faces:
            vids = face.vertex_ids()

            if len(vids) >= 3:
                rotations = [tuple(vids[i:] + vids[:i]) for i in range(len(vids))]
                rev = list(reversed(vids))
                rev_rotations = [tuple(rev[i:] + rev[:i]) for i in range(len(rev))]
                canonical = min(rotations + rev_rotations)

                if canonical in normalized_faces:
                    errors.append(f"Duplicate face detected with vertices {vids}")
                normalized_faces.add(canonical)

        # -------------------------------------------------
        # Edge statistics
        # -------------------------------------------------
        undirected_edges: dict[tuple[int, int], list[int]] = {}

        for he in self.halfedges:
            if he.dest is None:
                continue

            a = he.origin.id
            b = he.dest.id
            key = (a, b) if a < b else (b, a)
            undirected_edges.setdefault(key, []).append(he.id)

        nonmanifold_edges = []
        boundary_edge_count = 0
        interior_edge_count = 0

        for key, he_ids in undirected_edges.items():
            n = len(he_ids)
            if n == 1:
                boundary_edge_count += 1
            elif n == 2:
                interior_edge_count += 1
            else:
                nonmanifold_edges.append((key, he_ids))

        for key, he_ids in nonmanifold_edges:
            errors.append(f"Non-manifold undirected edge {key} used by halfedges {he_ids}")

        # -------------------------------------------------
        # Boundary loop tracing and classification
        # -------------------------------------------------
        loops_halfedges: list[list[HalfEdge]] = []
        outer_loops: list[list[int]] = []
        hole_loops: list[list[int]] = []

        try:
            loops_halfedges = self.trace_boundary_loops_halfedges()

            visited_boundary = {he.id for loop in loops_halfedges for he in loop}
            actual_boundary = {he.id for he in self.boundary_halfedges()}

            if visited_boundary != actual_boundary:
                missing = sorted(actual_boundary - visited_boundary)
                extra = sorted(visited_boundary - actual_boundary)

                if missing:
                    errors.append(f"Boundary tracing missed boundary halfedges: {missing}")
                if extra:
                    errors.append(f"Boundary tracing visited non-boundary halfedges: {extra}")

            outer_loops, hole_loops = self.classify_boundary_loops()

            for i, loop in enumerate(loops_halfedges):
                area = self._boundary_loop_signed_area(loop)
                if area == 0.0:
                    errors.append(f"Boundary loop {i} has zero signed area")

        except Exception as exc:
            errors.append(f"Boundary loop tracing failed: {exc}")

        # -------------------------------------------------
        # Connected components of faces
        # -------------------------------------------------
        components: list[list[int]] = []
        try:
            components = self._face_connected_components()
        except Exception as exc:
            errors.append(f"Connected component computation failed: {exc}")

        if components and len(components) > 1:
            warnings.append(f"Mesh has {len(components)} face-connected components")

        # -------------------------------------------------
        # Euler consistency with hole count
        # For a planar connected region with h holes:
        # chi = 1 - h
        # More generally with c connected planar components:
        # chi = c - h
        # -------------------------------------------------
        euler = None
        num_edges = None

        try:
            num_edges = self.num_edges()
            euler = self.euler_characteristic()

            if components and loops_halfedges:
                c = len(components)
                h = len(hole_loops)
                expected_euler = c - h

                if euler != expected_euler:
                    errors.append(
                        f"Euler mismatch: chi={euler}, but connected-components minus holes "
                        f"gives {c} - {h} = {expected_euler}"
                    )

            if len(outer_loops) == 0 and len(self.boundary_halfedges()) > 0:
                errors.append("No outer boundary loop detected despite boundary edges existing")

            if strict and len(outer_loops) > len(components) and components:
                errors.append(
                    f"Detected {len(outer_loops)} outer loops but only {len(components)} " f"face-connected components"
                )

        except Exception as exc:
            errors.append(f"Euler/statistics computation failed: {exc}")

        # -------------------------------------------------
        # Vertex outgoing halfedge sanity
        # -------------------------------------------------
        for v in self.vertices:
            if v.halfedge is None:
                warnings.append(f"Vertex {v.id} has no representative halfedge")
            elif v.halfedge.origin is not v:
                errors.append(
                    f"Vertex {v.id} representative halfedge {v.halfedge.id} " f"does not originate from that vertex"
                )

        stats = {
            "num_vertices": num_vertices,
            "num_halfedges": num_halfedges,
            "num_edges": num_edges,
            "num_faces": num_faces,
            "num_boundary_halfedges": len(self.boundary_halfedges()),
            "num_boundary_loops": len(loops_halfedges),
            "num_outer_loops": len(outer_loops),
            "num_hole_loops": len(hole_loops),
            "num_face_components": len(components),
            "euler_characteristic": euler,
            "boundary_edge_count": boundary_edge_count,
            "interior_edge_count": interior_edge_count,
            "nonmanifold_edges": nonmanifold_edges,
            "components": components,
        }

        is_valid = len(errors) == 0
        return TopologyReport(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

    def bounding_box_raw(self):
        """
        Compute axis-aligned bounding rectangle of a HalfEdgeMesh.

        Returns
        -------
        xmin, ymin, xmax, ymax
        """

        if not self.vertices:
            raise ValueError("Mesh has no vertices")

        xmin = float("inf")
        ymin = float("inf")
        xmax = float("-inf")
        ymax = float("-inf")

        for v in self.vertices:
            x = v.x
            y = v.y

            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x > xmax:
                xmax = x
            if y > ymax:
                ymax = y

        return xmin, ymin, xmax, ymax

    def bounding_box(self, padding=0.0):
        xmin, ymin, xmax, ymax = self.bounding_box_raw()

        dx = xmax - xmin
        dy = ymax - ymin

        xmin -= padding * dx
        xmax += padding * dx
        ymin -= padding * dy
        ymax += padding * dy

        return xmin, ymin, xmax, ymax


def extract_region_loops(mesh):
    """
    Extract clean boundary loops for each region from a half-edge mesh.

    Returns
    -------
    dict[region_id, list[list[tuple[float, float]]]]
        Mapping:
            region_id -> list of coordinate loops
    """

    def is_region_boundary(he, region_id):
        return (
            he.face is not None
            and he.face.region_id == region_id
            and (he.twin is None or he.twin.face is None or he.twin.face.region_id != region_id)
        )

    region_boundary_halfedges = {}

    # -------------------------------------------------
    # Collect region-boundary halfedges
    # -------------------------------------------------
    for he in mesh.halfedges:
        if he.face is None:
            continue

        region_id = he.face.region_id

        if is_region_boundary(he, region_id):
            region_boundary_halfedges.setdefault(region_id, []).append(he)

    region_loops = {}

    # -------------------------------------------------
    # For each region, trace loops
    # -------------------------------------------------
    for region_id, boundary_hes in region_boundary_halfedges.items():
        boundary_ids = {he.id for he in boundary_hes}
        by_id = {he.id: he for he in boundary_hes}

        # Build successor map:
        # after boundary halfedge he, continue walking along the same region
        successor = {}

        for he in boundary_hes:
            cur = he.next
            if cur is None:
                raise RuntimeError(f"Half-edge {he.id} has no next pointer.")

            # Walk around the incident region until the next boundary halfedge
            # of the same region is found.
            while cur.id not in boundary_ids:
                if cur.twin is None or cur.twin.next is None:
                    raise RuntimeError(
                        f"Could not find successor for region boundary half-edge {he.id} " f"in region {region_id}."
                    )
                cur = cur.twin.next

            successor[he.id] = cur.id

        # Trace loops through successor map
        used = set()
        loops = []

        for start_he in boundary_hes:
            if start_he.id in used:
                continue

            loop = []
            curr_id = start_he.id

            while curr_id not in used:
                used.add(curr_id)

                he = by_id[curr_id]
                loop.append((he.origin.x, he.origin.y))

                curr_id = successor[curr_id]

            if len(loop) >= 3:
                loops.append(loop)

        region_loops[region_id] = loops

    return region_loops


def extract_outer_loop_from_mesh(mesh):
    """
    Extract the single outer boundary loop from a partition half-edge mesh.

    Returns
    -------
    outer_loop : list[(x, y)]
    """
    raw_loops_vids = mesh.trace_boundary_loops_vertex_ids()

    if not raw_loops_vids:
        raise RuntimeError("No boundary loops found in mesh.")

    if len(raw_loops_vids) != 1:
        raise RuntimeError(f"Expected exactly one outer boundary loop for Case A, got {len(raw_loops_vids)}")

    loop_vids = raw_loops_vids[0]
    return [(mesh.vertices[i].x, mesh.vertices[i].y) for i in loop_vids]
