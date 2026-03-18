import matplotlib.pyplot as plt
import numpy as np


def plot_mesh(
    mesh,
    show_vertices=True,
    show_faces=False,
    show_boundary=True,
    boundary_arrows=False,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots()

    # -------------------------------------------------
    # Plot all mesh edges (light gray)
    # -------------------------------------------------
    for he in mesh.halfedges:
        v0 = he.origin
        v1 = he.dest
        if v1 is None:
            continue

        ax.plot(
            [v0.x, v1.x],
            [v0.y, v1.y],
            color="lightgray",
            linewidth=1,
            zorder=1,
        )

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
    # Plot face ids (centroids)
    # -------------------------------------------------
    if show_faces:
        for f in mesh.faces:
            verts = [he.origin for he in f.iter_halfedges()]
            cx = np.mean([v.x for v in verts])
            cy = np.mean([v.y for v in verts])

            ax.text(
                cx,
                cy,
                f"F{f.id}",
                color="blue",
                fontsize=10,
                ha="center",
                va="center",
                zorder=6,
            )

    # -------------------------------------------------
    # Final formatting
    # -------------------------------------------------
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    plt.show()