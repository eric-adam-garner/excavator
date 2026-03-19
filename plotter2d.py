import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


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


def plot_pslg(pslg, show_vertex_ids=False, show_loop_ids=False):
    nesting = pslg.classify_loops()
    depths = nesting.depths
    loops = nesting.loops

    cmap = cm.get_cmap("tab10")
    norm = mcolors.Normalize(vmin=min(depths) if depths else 0,
                             vmax=max(depths) if depths else 1)
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

    ax.set_aspect("equal")
    ax.set_title("PSLG Visualization")
    import matplotlib.cm as mcm
    sm = mcm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Loop depth")
    plt.show()
