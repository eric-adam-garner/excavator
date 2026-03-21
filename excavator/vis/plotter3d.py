from __future__ import annotations

import logging

import numpy as np
from matplotlib import cm
from vedo import Mesh, Plotter

from excavator.extrusion import build_extruded_connectivity_from_mesh, realize_extruded_vertices
from excavator.triangulation.triangle_backend import triangle_to_halfedge_mesh, weld_triangle_meshes

logger = logging.getLogger(__name__)


def plot_extrusion_vedo(
    outer_tri_meshes,
    bench_tri_meshes,
    level_id_height_map,
    tol,
    level_id,
    color_by_region=True,
    wireframe=False,
):

    z_prev = level_id_height_map[level_id + 1]
    z = level_id_height_map[level_id]

    plt = Plotter()

    # ---------- initial bench
    current_bench = 5

    def build_region_z(bench, level_map):
        region_z = {}

        for bench_id in range(len(state["bench_tri_mesh"].triangle_region_ids)):
            region_z[bench_id] = state["z"] if bench_id < bench else state["z_prev"]

        for key, val in level_map.items():
            region_z[key] = val

        logger.info("recomputed level bench vertex positions")

        return region_z

    def build_connectivity(level_id):

        merged_outer_tri_mesh = outer_tri_meshes[level_id]
        for level_id_, msh in outer_tri_meshes.items():
            if level_id_ > level_id:
                merged_outer_tri_mesh = weld_triangle_meshes(merged_outer_tri_mesh, msh, tol=tol)

        tri_mesh = weld_triangle_meshes(bench_tri_meshes[level_id], merged_outer_tri_mesh, tol=tol)
        half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)
        connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)

        bench_tri_mesh = bench_tri_meshes[level_id]

        # ---------- precompute region colors
        regions = sorted(set(connectivity.top_triangle_regions))
        cmap = cm.get_cmap("tab20")
        region_to_color = {r: [int(255 * c) for c in cmap(i % 20)[:3]] for i, r in enumerate(regions)}
        for r in regions:
            if r < 0:
                region_to_color[r] = [100, 100, 100]

        # ---------- static faces + colors
        faces = []
        colors = []
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

        logger.info("rebuilt level connectivity")

        return connectivity, bench_tri_mesh, faces, colors

    # ---------- initial geomety
    connectivity, bench_tri_mesh, faces, colors = build_connectivity(level_id)
    region_z = build_region_z(current_bench, level_id_height_map) if False else None
    # build_region_z uses state, so initialize verts manually for the first mesh
    region_z = {}
    for bench_id in range(len(bench_tri_mesh.triangle_region_ids)):
        region_z[bench_id] = z if bench_id < current_bench else z_prev
    for key, val in level_id_height_map.items():
        region_z[key] = val

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
    state = {
        "bench": current_bench,
        "level": level_id,
        "mesh": mesh,
        "connectivity": connectivity,
        "bench_tri_mesh": bench_tri_mesh,
        "faces": faces,
        "colors": colors,
        "z_prev": z_prev,
        "z": z,
    }

    # ---------- regeneration function
    def regenerate_bench():
        region_z = build_region_z(state["bench"], level_id_height_map)
        verts3d = realize_extruded_vertices(state["connectivity"], region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        state["mesh"].points = new_pts
        state["mesh"].modified()
        plt.render()

    def regenerate_level():
        connectivity, bench_tri_mesh, faces, colors = build_connectivity(state["level"])

        state["connectivity"] = connectivity
        state["bench_tri_mesh"] = bench_tri_mesh
        state["faces"] = faces
        state["colors"] = colors
        state["z_prev"] = level_id_height_map[state["level"] + 1]
        state["z"] = level_id_height_map[state["level"]]

        region_z = build_region_z(state["bench"], level_id_height_map)
        verts3d = realize_extruded_vertices(state["connectivity"], region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        # topology changed -> recreate mesh, do not only update points
        plt.remove(state["mesh"])
        new_mesh = Mesh([new_pts, state["faces"]])

        if color_by_region:
            new_mesh.cellcolors = state["colors"]

        if wireframe:
            new_mesh.wireframe()

        new_mesh.lighting("plastic")
        plt.add(new_mesh)

        state["mesh"] = new_mesh
        plt.render()

    # ---------- slider callback
    def bench_slider_callback(widget, event):
        rep = widget.GetRepresentation()
        val = int(round(rep.GetValue()))
        rep.SetValue(val)

        if val != state["bench"]:
            state["bench"] = val
            regenerate_bench()

    def level_slider_callback(widget, event):
        rep = widget.GetRepresentation()
        val = int(round(rep.GetValue()))
        rep.SetValue(val)

        if val != state["level"]:
            state["level"] = -(val + 1)
            regenerate_level()

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
        xmin=1,
        xmax=xmax_level_slider,
        value=xmax_level_slider,
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
    plt.show(mesh, axes=1)
