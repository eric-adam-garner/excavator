import numpy as np
from matplotlib import cm
from vedo import (
    Mesh,
    Plotter,
    show,
)

from extrusion import realize_extruded_vertices


def plot_extrusion_vedo(
    connectivity,
    level_id_height_map,
    level_id,
    triangle_region_ids,
    color_by_region=True,
    wireframe=False,
):

    z_prev = level_id_height_map[level_id + 1]
    z = level_id_height_map[level_id]

    faces = []
    colors = []

    plt = Plotter()

    # ---------- initial bench
    current_bench = 5

    # ---------- precompute region colors
    regions = sorted(set(connectivity.top_triangle_regions))
    cmap = cm.get_cmap("tab20")
    region_to_color = {r: [int(255 * c) for c in cmap(i % 20)[:3]] for i, r in enumerate(regions)}

    # ---------- static faces + colors
    for tri, region in zip(connectivity.top_triangles, connectivity.top_triangle_regions):
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
    def build_region_z(bench, level_map):
        region_z = {}

        for bench_id in range(len(triangle_region_ids)):
            region_z[bench_id] = z if bench_id < bench else z_prev

        for key, val in level_map.items():
            region_z[key] = val

        return region_z

    # ---------- initial geometry
    region_z = build_region_z(current_bench, level_id_height_map)
    verts3d = realize_extruded_vertices(connectivity, region_z)
    verts = np.asarray(verts3d, dtype=float)

    xmin, ymin, _ = verts.min(axis=0)
    xmax, ymax, _ = verts.max(axis=0)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2

    mesh = Mesh([verts, faces])

    if color_by_region:
        mesh.cellcolors = colors

    if wireframe:
        mesh.wireframe()

    mesh.lighting("plastic")
    plt.add(mesh)

    # ---------- store mutable state
    state = {"bench": current_bench, "level": level_id, "mesh": mesh}

    # ---------- regeneration function
    def regenerate():
        region_z = build_region_z(state["bench"], level_id_height_map)
        verts3d = realize_extruded_vertices(connectivity, region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        state["mesh"].points = new_pts
        state["mesh"].modified()
        plt.render()

    # ---------- slider callback
    def bench_slider_callback(widget, event):
        rep = widget.GetRepresentation()
        val = int(round(rep.GetValue()))
        rep.SetValue(val)

        if val != state["bench"]:
            state["bench"] = val
            regenerate()

    def level_slider_callback(widget, event):
        rep = widget.GetRepresentation()
        val = int(round(rep.GetValue()))
        rep.SetValue(val)

        if val != state["level"]:
            state["level"] = val
            pass

    xmax_bench_slider = max(connectivity.top_triangle_regions) + 1
    xmax_level_slider = -min(connectivity.top_triangle_regions)

    plt.add_slider(
        bench_slider_callback,
        xmin=0,
        xmax=xmax_bench_slider,
        value=current_bench,
        title="bench",
        pos=[(0.6, 0.05), (0.8, 0.05)],
    )
    plt.add_slider(
        level_slider_callback,
        xmin=0,
        xmax=xmax_level_slider,
        value=level_id,
        title="level",
        pos=[(0.2, 0.05), (0.4, 0.05)],
    )

    z_mid = 0.5 * (z_prev + z)
    pivot = (cx, cy, z_mid)

    rotation_state = {"on": True}

    def rotate_timer(evt):
        if rotation_state["on"]:
            plt.camera.Azimuth(0.3)
            plt.render()

    def keypress(evt):
        if evt.keypress == "space":
            rotation_state["on"] = not rotation_state["on"]

    plt.add_callback("timer", rotate_timer)
    plt.add_callback("key press", keypress)
    plt.timer_callback("start", dt=30)

    plt.camera.SetFocalPoint(*pivot)
    d = np.array([-1, -1, -1])
    plt.camera.SetPosition(pivot - d * 100)

    plt.camera.SetViewUp(0, 0, 1)
    plt.show(mesh, axes=0)
