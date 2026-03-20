from collections import defaultdict

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def plot_mesh(
    mesh,
    report=None,
    show_vertices=True,
    show_faces=False,
    show_boundary=True,
    boundary_arrows=False,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    # -------------------------------------------------
    # Background tint if invalid
    # -------------------------------------------------
    if report is not None and not report.is_valid:
        ax.set_facecolor("#ffe6e6")  # light red

    # -------------------------------------------------
    # Component coloring
    # -------------------------------------------------
    component_colors = {}
    if report and "components" in report.stats:
        comps = report.stats["components"]
        cmap = plt.get_cmap("tab20")

        for i, comp in enumerate(comps):
            color = cmap(i % 20)
            for fid in comp:
                component_colors[fid] = color

    # -------------------------------------------------
    # Plot mesh edges
    # -------------------------------------------------
    for he in mesh.halfedges:
        v0 = he.origin
        v1 = he.dest
        if v1 is None:
            continue

        color = "lightgray"

        if report and "nonmanifold_edges" in report.stats:
            for edge, he_ids in report.stats["nonmanifold_edges"]:
                a, b = edge
                if {v0.id, v1.id} == {a, b}:
                    color = "magenta"

        ax.plot(
            [v0.x, v1.x],
            [v0.y, v1.y],
            color=color,
            linewidth=2 if color == "magenta" else 1,
            zorder=1,
        )

    # -------------------------------------------------
    # Plot faces colored by component
    # -------------------------------------------------
    if show_faces:
        for f in mesh.faces:
            verts = [he.origin for he in f.iter_halfedges()]
            xs = [v.x for v in verts]
            ys = [v.y for v in verts]

            color = component_colors.get(f.id, "#dddddd")

            ax.fill(xs, ys, color=color, alpha=0.3, zorder=0)

    # -------------------------------------------------
    # Plot boundary loops
    # -------------------------------------------------
    if show_boundary:
        loops = mesh.trace_boundary_loops_halfedges()
        cmap = plt.get_cmap("tab10")

        for i, loop in enumerate(loops):
            color = cmap(i % 10)

            xs = [he.origin.x for he in loop] + [loop[0].origin.x]
            ys = [he.origin.y for he in loop] + [loop[0].origin.y]

            ax.plot(xs, ys, color=color, linewidth=3, zorder=3)

            if boundary_arrows:
                for he in loop:
                    dx = he.dest.x - he.origin.x
                    dy = he.dest.y - he.origin.y

                    ax.arrow(
                        he.origin.x,
                        he.origin.y,
                        0.7 * dx,
                        0.7 * dy,
                        head_width=0.05 * np.linalg.norm([dx, dy]),
                        color=color,
                        length_includes_head=True,
                        zorder=4,
                    )

    # -------------------------------------------------
    # Highlight zero-area faces
    # -------------------------------------------------
    if report:
        for err in report.errors:
            if "zero signed area" in err:
                fid = int(err.split("Face ")[1].split()[0])
                f = mesh.faces[fid]

                verts = [he.origin for he in f.iter_halfedges()]
                cx = np.mean([v.x for v in verts])
                cy = np.mean([v.y for v in verts])

                ax.scatter(cx, cy, color="red", s=80, zorder=5)

    # -------------------------------------------------
    # Plot vertices
    # -------------------------------------------------
    if show_vertices:
        for v in mesh.vertices:
            ax.scatter(v.x, v.y, color="black", s=20, zorder=5)
            ax.text(
                v.x,
                v.y,
                str(v.id),
                fontsize=10,
                ha="right",
                va="bottom",
                zorder=6,
            )

    # -------------------------------------------------
    # Euler mismatch overlay
    # -------------------------------------------------
    if report and not report.is_valid:
        text = f"INVALID TOPOLOGY\nEuler = {report.stats.get('euler_characteristic')}"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=12,
            color="red",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.8),
        )

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_pslg(pslg, show_vertex_ids=False, show_loop_ids=False, show_segment_ids=False):
    nesting = pslg.classify_loops()
    depths = nesting.depths
    loops = nesting.loops

    cmap = cm.get_cmap("tab10")
    norm = mcolors.Normalize(vmin=min(depths) if depths else 0, vmax=max(depths) if depths else 1)
    fig, ax = plt.subplots()

    # background segments
    for s in pslg.segments:
        v0 = pslg.vertices[s.v0]
        v1 = pslg.vertices[s.v1]
        ax.plot([v0.x, v1.x], [v0.y, v1.y], color="lightgray", linewidth=1)

    # loops colored by depth
    for i, loop in enumerate(loops):
        xs = []
        ys = []
        for vid in loop:
            v = pslg.vertices[vid]
            xs.append(v.x)
            ys.append(v.y)

        xs.append(xs[0])
        ys.append(ys[0])

        color = cmap(depths[i] % 10)
        ax.plot(xs, ys, color=color, linewidth=2)

        if show_loop_ids:
            cx = sum(xs[:-1]) / len(xs[:-1])
            cy = sum(ys[:-1]) / len(ys[:-1])
            ax.text(cx, cy, f"L{i}", fontsize=10, color=color)

    if show_vertex_ids:
        for v in pslg.vertices:
            ax.text(v.x, v.y, str(v.id), fontsize=6)

    if show_segment_ids:
        for s in pslg.segments:
            v0 = pslg.vertices[s.v0]
            v1 = pslg.vertices[s.v1]
            cx = 0.5 * (v0.x + v1.x)
            cy = 0.5 * (v0.y + v1.y)
            ax.text(cx, cy, str(s.id), fontsize=8, color="red")

    ax.set_aspect("equal")
    ax.set_title("PSLG Visualization")
    import matplotlib.cm as mcm

    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Loop depth")
    plt.show()


def plot_directed_edges(edges, tol, show_ids=True, show_degrees=True):
    """
    Diagnostic plot for directed edge set.

    edges:
        list of ((x0,y0),(x1,y1)) directed edges
    """

    def qpoint(p):
        return (round(p[0] / tol), round(p[1] / tol))

    fig, ax = plt.subplots()

    # -------------------------
    # Build adjacency (degree)
    # -------------------------
    adj = defaultdict(list)

    for a, b in edges:
        qa = qpoint(a)
        qb = qpoint(b)
        adj[qa].append(qb)
        adj[qb].append(qa)

    # -------------------------
    # Plot edges
    # -------------------------
    for i, (a, b) in enumerate(edges):

        x0, y0 = a
        x1, y1 = b

        dx = x1 - x0
        dy = y1 - y0

        ax.arrow(
            x0, y0, dx, dy, length_includes_head=True, head_width=0.02 * np.hypot(dx, dy), color="black", alpha=0.7
        )

        if show_ids:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            ax.text(cx, cy, str(i), color="red", fontsize=8)

    # -------------------------
    # Plot vertex degrees
    # -------------------------
    if show_degrees:
        for q, nbrs in adj.items():
            x, y = q[0] * tol, q[1] * tol
            deg = len(nbrs)
            ax.text(x, y, f"d={deg}", color="blue", fontsize=9)

    ax.set_aspect("equal")
    ax.set_title("Directed Edge Graph Diagnostic")
    plt.show()


def plot_faces(
    faces,
    face_ids,
    show_face_ids=True,
    show_areas=False,
    show_edges=True,
    alpha=0.35,
):
    """
    Plot extracted face cycles.

    Args:
        faces:
            list of faces, where each face is a list of (x, y) points
            in traversal order, without repeated last point.
        show_face_ids:
            if True, label each face with its index.
        show_areas:
            if True, show signed area next to face id.
        show_edges:
            if True, draw face boundaries.
        alpha:
            fill transparency.
    """

    def signed_area(loop):
        area = 0.0
        n = len(loop)
        for i in range(n):
            x0, y0 = loop[i]
            x1, y1 = loop[(i + 1) % n]
            area += x0 * y1 - x1 * y0
        return 0.5 * area

    def centroid(loop):
        xs = [p[0] for p in loop]
        ys = [p[1] for p in loop]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")

    for i, face in enumerate(faces):
        if len(face) < 3:
            continue

        xs = [p[0] for p in face]
        ys = [p[1] for p in face]

        color = cmap(i % 20)

        ax.fill(xs, ys, color=color, alpha=alpha, zorder=1)

        if show_edges:
            ax.plot(xs + [xs[0]], ys + [ys[0]], color=color, linewidth=1.5, zorder=2)

        if show_face_ids or show_areas:
            cx, cy = centroid(face)
            label = []

            if show_face_ids:
                label.append(f"F{face_ids[i]}")

            if show_areas:
                a = signed_area(face)
                label.append(f"A={a:.3g}")

            ax.text(
                cx,
                cy,
                "\n".join(label),
                fontsize=9,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                zorder=3,
            )

    ax.set_aspect("equal")
    ax.set_title("Extracted Faces")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.show()


def plot_partition_domain(
    domain,
    show_vertex_ids=False,
    show_edge_ids=False,
    show_face_ids=True,
    show_centroids=True,
):
    """
    Visualize PartitionDomain for meshing inspection.
    """

    fig, ax = plt.subplots()

    vertices = domain.vertices
    edges = domain.edges
    faces = domain.faces
    bench_ids = domain.face_bench_ids

    # ---------- face coloring ----------
    unique_ids = list(sorted(set(bench_ids)))
    id_to_color_index = {bid: i for i, bid in enumerate(unique_ids)}

    cmap = cm.get_cmap("tab20")
    norm = mcolors.Normalize(vmin=0, vmax=max(len(unique_ids) - 1, 1))

    # ---------- draw faces ----------
    for i, face in enumerate(faces):
        xs = [vertices[v][0] for v in face]
        ys = [vertices[v][1] for v in face]

        xs.append(xs[0])
        ys.append(ys[0])

        color = cmap(norm(id_to_color_index[bench_ids[i]]))

        ax.fill(xs, ys, color=color, alpha=0.35, edgecolor="black")

        if show_face_ids:
            cx, cy = domain.face_centroid(i)
            ax.text(cx, cy, f"F{i}\nB{bench_ids[i]}", fontsize=8)

    # ---------- draw edges ----------
    for i, (v0, v1) in enumerate(edges):
        x0, y0 = vertices[v0]
        x1, y1 = vertices[v1]

        ax.plot([x0, x1], [y0, y1], color="black", linewidth=1)

        if show_edge_ids:
            cx = 0.5 * (x0 + x1)
            cy = 0.5 * (y0 + y1)
            ax.text(cx, cy, str(i), fontsize=6, color="red")

    # ---------- draw vertices ----------
    if show_vertex_ids:
        for i, (x, y) in enumerate(vertices):
            ax.text(x, y, str(i), fontsize=6, color="blue")

    # ---------- centroids ----------
    if show_centroids:
        for i in range(len(faces)):
            cx, cy = domain.face_centroid(i)
            ax.plot(cx, cy, "ko", markersize=3)

    ax.set_aspect("equal")
    ax.set_title("PartitionDomain inspection")
    plt.show()


def plot_triangle_mesh(
    mesh,
    show_triangle_ids=False,
    show_vertices=False,
    show_edges=False,
    domain_edges=None,
):
    """
    Plot Triangle mesh with region coloring.

    Parameters
    ----------
    mesh : TriangleMesh
    domain_edges : optional list[(i,j)]
        Used to overlay constraint edges
    """

    verts = mesh.vertices
    tris = mesh.triangles
    regions = mesh.triangle_region_ids

    unique_regions = sorted(set(regions))
    cmap = cm.get_cmap("tab20")
    norm = mcolors.Normalize(
        vmin=min(unique_regions),
        vmax=max(unique_regions) if len(unique_regions) > 1 else 1,
    )

    fig, ax = plt.subplots()

    # ---- triangles ----
    for i, tri in enumerate(tris):
        xs = [verts[v][0] for v in tri]
        ys = [verts[v][1] for v in tri]

        xs.append(xs[0])
        ys.append(ys[0])

        color = cmap(norm(regions[i]))

        ax.fill(xs, ys, color=color, alpha=0.6, edgecolor="black", linewidth=0.2)

        if show_triangle_ids:
            cx = sum(xs[:-1]) / 3
            cy = sum(ys[:-1]) / 3
            ax.text(cx, cy, str(i), fontsize=5)

    # ---- constrained edges overlay ----
    if domain_edges is not None:
        for a, b in domain_edges:
            x0, y0 = verts[a]
            x1, y1 = verts[b]
            ax.plot([x0, x1], [y0, y1], color="red", linewidth=1.5)

    # ---- vertices ----
    if show_vertices:
        for i, (x, y) in enumerate(verts):
            ax.text(x, y, str(i), fontsize=5, color="blue")

    ax.set_aspect("equal")
    ax.set_title("Triangle Mesh")
    plt.show()
