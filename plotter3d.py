import numpy as np
from matplotlib import cm
from vedo import (
    Mesh,
    Plotter,
    show,
)

from extrusion import realize_extruded_vertices



def plot_extrusion_vedo(connectivity, triangle_region_ids, z_prev, z,
                        color_by_region=True, wireframe=False):

    faces = []
    colors = []

    plt = Plotter()

    # ---------- initial bench
    current_bench = 5

    # ---------- precompute region colors
    regions = sorted(set(connectivity.top_triangle_regions))
    cmap = cm.get_cmap("tab20")
    region_to_color = {
        r: [int(255 * c) for c in cmap(i % 20)[:3]]
        for i, r in enumerate(regions)
    }

    # ---------- static faces + colors
    for tri, region in zip(connectivity.top_triangles,
                           connectivity.top_triangle_regions):
        faces.append(tri)
        colors.append(region_to_color[region])

    for tri in connectivity.wall_triangles_internal:
        faces.append(tri)
        colors.append([200, 200, 200])

    for tri in connectivity.wall_triangles_outer:
        faces.append(tri)
        colors.append([150, 150, 150])

    faces = np.asarray(faces, dtype=int)
    colors = np.asarray(colors, dtype=np.uint8)

    # ---------- helper to build region_z
    def build_region_z(bench):
        region_z = {}
        for bench_id in range(len(triangle_region_ids)):
            region_z[bench_id] = z if bench_id < bench else z_prev
        region_z[-1] = z_prev
        return region_z

    # ---------- initial geometry
    region_z = build_region_z(current_bench)
    verts3d = realize_extruded_vertices(connectivity, region_z)
    verts = np.asarray(verts3d, dtype=float)

    mesh = Mesh([verts, faces])

    if color_by_region:
        mesh.cellcolors = colors

    if wireframe:
        mesh.wireframe()

    mesh.lighting("plastic")
    plt.add(mesh)

    # ---------- store mutable state
    state = {"bench": current_bench, "mesh": mesh}

    # ---------- regeneration function
    def regenerate():
        region_z = build_region_z(state["bench"])
        verts3d = realize_extruded_vertices(connectivity, region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        state["mesh"].points = new_pts
        state["mesh"].modified()
        plt.render()

    # ---------- slider callback
    def slider_callback(widget, event):
        rep = widget.GetRepresentation()
        val = int(round(rep.GetValue()))
        rep.SetValue(val)

        if val != state["bench"]:
            state["bench"] = val
            regenerate()

    xmax = max(connectivity.top_triangle_regions) + 1

    plt.add_slider(
        slider_callback,
        xmin=0,
        xmax=xmax,
        value=current_bench,
        title="bench"
    )

    plt.show(mesh, axes=1)