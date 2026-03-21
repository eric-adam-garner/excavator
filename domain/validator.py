from collections import defaultdict


class DomainReport:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.stats = {}

    @property
    def is_valid(self):
        return len(self.errors) == 0


def validate_partition_domain(domain):

    report = DomainReport()

    verts = domain.vertices
    edges = domain.edges
    faces = domain.faces
    tol = domain.tol

    # ---------- edge length ----------
    for i, (a, b) in enumerate(edges):
        x0, y0 = verts[a]
        x1, y1 = verts[b]
        if (x1 - x0) ** 2 + (y1 - y0) ** 2 < tol * tol:
            report.errors.append(f"zero length edge {i}")

    # ---------- duplicate edges ----------
    seen = set()
    for i, e in enumerate(edges):
        key = tuple(sorted(e))
        if key in seen:
            report.errors.append(f"duplicate edge {e}")
        seen.add(key)

    # ---------- vertex degree ----------
    degree = defaultdict(int)
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    for v, d in degree.items():
        if d < 2:
            report.errors.append(f"dangling vertex {v}")

    # ---------- face validity ----------
    for i, face in enumerate(faces):
        if len(face) < 3:
            report.errors.append(f"degenerate face {i}")

    # ---------- face area ----------
    def signed_area(face):
        A = 0
        for i in range(len(face)):
            x0, y0 = verts[face[i]]
            x1, y1 = verts[face[(i + 1) % len(face)]]
            A += x0 * y1 - x1 * y0
        return 0.5 * A

    for i, face in enumerate(faces):
        if abs(signed_area(face)) < tol * tol:
            report.warnings.append(f"tiny face {i}")

    # ---------- manifold edge check ----------
    edge_count = defaultdict(int)
    for face in faces:
        for i in range(len(face)):
            a = face[i]
            b = face[(i + 1) % len(face)]
            key = tuple(sorted((a, b)))
            edge_count[key] += 1

    for e, c in edge_count.items():
        if c > 2:
            report.errors.append(f"non manifold edge {e}")

    report.stats = {
        "num_vertices": len(verts),
        "num_edges": len(edges),
        "num_faces": len(faces),
    }

    return report
