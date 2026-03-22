from __future__ import annotations

import logging

import numpy as np
from vedo import Light, Mesh, Plotter, Text2D

from excavator.extrusion import build_extruded_connectivity_from_mesh, realize_extruded_vertices
from excavator.triangulation.triangle_backend import triangle_to_halfedge_mesh, weld_triangle_meshes

GROUND_COLOR = [100, 100, 100]

EARTH_GRADIENT = np.array([
    [241, 233, 210],
    [217, 200, 161],
    [191, 161, 106],
    [155, 122, 69],
    [110, 86, 49],
    [74, 58, 35],
], dtype=np.uint8)

logger = logging.getLogger(__name__)


def sequence_color(i, n):
    t = i / max(1, n-1)
    idx = t * (len(EARTH_GRADIENT) - 1)
    lo = int(np.floor(idx))
    hi = int(np.ceil(idx))
    w = idx - lo
    return ((1-w)*EARTH_GRADIENT[lo] + w*EARTH_GRADIENT[hi]).astype(np.uint8)


def apply_excavation_material(mesh):
    mesh.lighting("plastic")

    prop = mesh.properties
    prop.SetSpecular(0.04)
    prop.SetSpecularPower(6)
    prop.SetAmbient(0.38)
    prop.SetDiffuse(0.82)
    
    
def print_controls():
    print("\n" + "=" * 60)
    print("EXCAVATOR VIEWER CONTROLS")
    print("=" * 60)
    print("Mouse: rotate / pan / zoom")
    print("Keyboard: SPACE → rotate")
    print("Sliders: bench / level")
    print("=" * 60 + "\n")


def setup_excavation_lighting(plt, mesh):

    bounds = mesh.bounds()
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diag = (dx*dx + dy*dy + dz*dz) ** 0.5

    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])

    try:
        plt.renderer.RemoveAllLights()
    except:
        pass

    # 🌞 main sun (soft top)
    sun = Light(
        pos=(cx + 0.4*diag, cy + 0.3*diag, cz + 1.6*diag),
        focal_point=(cx, cy, cz),
        intensity=1.1,
        c=(1.0, 0.98, 0.95),
    )

    # 🌤 grazing side light (terrace readability)
    side = Light(
        pos=(cx - 1.2*diag, cy + 0.8*diag, cz + 0.4*diag),
        focal_point=(cx, cy, cz),
        intensity=0.35,
        c=(0.8, 0.85, 1.0),
    )

    # 🌫 hemisphere fill
    fill = Light(
        pos=(cx, cy - 1.5*diag, cz + 0.8*diag),
        focal_point=(cx, cy, cz),
        intensity=0.25,
        c=(0.9, 0.92, 1.0),
    )

    plt.renderer.AddLight(sun)
    plt.renderer.AddLight(side)
    plt.renderer.AddLight(fill)
    
    
def slider_to_level(val):
    return -(val + 1)


def build_region_z(connectivity, bench, level_map, level):
    z = level_map[level]
    z_prev = level_map[level + 1]

    region_z = {}

    regions = sorted(r for r in set(connectivity.top_triangle_regions) if r >= 0)

    for region_id in regions:
        region_z[region_id] = z if region_id < bench else z_prev

    # explicit overrides (outer walls etc)
    for key, val in level_map.items():
        region_z[key] = val

    return region_z


def build_faces_and_colors(connectivity):
    regions = sorted(set(connectivity.top_triangle_regions))
    region_to_color = {r: sequence_color(i, len(regions)) for i, r in enumerate(regions)}

    for r in regions:
        if r < 0:
            region_to_color[r] = GROUND_COLOR

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

    return (
        np.asarray(faces, dtype=int),
        np.asarray(colors, dtype=np.uint8),
    )

    
def update_hud(hud, state):
    hud.text(
        f"Level: {-state['level'] - 1}\n"
        f"Bench: {state['bench']}\n"
        f"Triangles: {len(state['faces'])}"
    )


def apply_mesh_style(mesh, colors, color_by_region, wireframe):
    if color_by_region:
        mesh.cellcolors = colors

    if wireframe:
        mesh.wireframe()

    apply_excavation_material(mesh)


def configure_camera(plt, mesh):
    bounds = mesh.bounds()
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    diag = (dx*dx + dy*dy + dz*dz) ** 0.5

    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    
    plt.camera.SetFocalPoint(cx, cy, cz)
    plt.camera.SetPosition(cx + diag, cy + diag, cz + 0.6 * diag)
    plt.camera.SetViewUp(0, 0, 1)

        
def set_slider_max(slider, xmax):
    rep = slider.GetRepresentation()
    rep.SetMaximumValue(xmax)

    if rep.GetValue() > xmax:
        rep.SetValue(xmax)

    rep.Modified()
    
    
def plot_excavation(
    outer_tri_meshes,
    bench_tri_meshes,
    level_id_height_map,
    tol,
    level_id,
):
    current_bench = 0
    
    print_controls()

    plt = Plotter(bg=(0.1, 0.1, 0.1))

    try:
        plt.window.SetMultiSamples(8)
    except AttributeError:
        pass

    try:
        plt.renderer.UseFXAAOn()
    except AttributeError:
        pass
    
    def build_extrusion_connectivity(level_id):

        merged_outer_tri_mesh = outer_tri_meshes[level_id]
        for level_id_, msh in outer_tri_meshes.items():
            if level_id_ > level_id:
                merged_outer_tri_mesh = weld_triangle_meshes(merged_outer_tri_mesh, msh, tol=tol)

        tri_mesh = weld_triangle_meshes(bench_tri_meshes[level_id], merged_outer_tri_mesh, tol=tol)
        half_edge_mesh = triangle_to_halfedge_mesh(tri_mesh)
        connectivity = build_extruded_connectivity_from_mesh(half_edge_mesh)

        bench_tri_mesh = bench_tri_meshes[level_id]

        # ---------- precompute region colors
        
        return connectivity, bench_tri_mesh

    connectivity, bench_tri_mesh = build_extrusion_connectivity(level_id)
    faces, colors = build_faces_and_colors(connectivity)
    
    region_z = build_region_z(
        connectivity,
        current_bench,
        level_id_height_map,
        level_id,
        )

    verts3d = realize_extruded_vertices(connectivity, region_z)
    verts = np.asarray(verts3d, dtype=float)
    mesh = Mesh([verts, faces])
    mesh.cellcolors = colors

    apply_excavation_material(mesh)
        
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
    }

    def regenerate_bench():
        region_z = build_region_z(
            state["connectivity"],
            state["bench"],
            level_id_height_map,
            state["level"],
        )
        verts3d = realize_extruded_vertices(state["connectivity"], region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        update_hud(hud, state)

        state["mesh"].points = new_pts
        state["mesh"].modified()
        plt.render()

    def regenerate_level():
        connectivity, bench_tri_mesh = build_extrusion_connectivity(state["level"])
        faces, colors = build_faces_and_colors(connectivity)

        state["connectivity"] = connectivity
        state["bench_tri_mesh"] = bench_tri_mesh
        state["faces"] = faces
        state["colors"] = colors
        
        xmax_bench_slider = max(connectivity.top_triangle_regions) + 1
        set_slider_max(state["bench_slider"], xmax_bench_slider)
        
        region_z = build_region_z(
            state["connectivity"],
            state["bench"],
            level_id_height_map,
            state["level"],
        )
        # region_z = build_region_z(state["bench"], level_id_height_map)
        verts3d = realize_extruded_vertices(state["connectivity"], region_z)
        new_pts = np.asarray(verts3d, dtype=float)

        # topology changed -> recreate mesh, do not only update points
        plt.remove(state["mesh"])
        new_mesh = Mesh([new_pts, state["faces"]])
        new_mesh.cellcolors = state["colors"]

        apply_excavation_material(new_mesh)
        plt.add(new_mesh)

        update_hud(hud, state)
        
        state["mesh"] = new_mesh
        plt.render()

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
            state["level"] = slider_to_level(val)
            regenerate_level()

    xmax_bench_slider = max(connectivity.top_triangle_regions) + 1
    xmax_level_slider = -min(connectivity.top_triangle_regions)

    bench_slider = plt.add_slider(
        bench_slider_callback,
        xmin=0,
        xmax=xmax_bench_slider,
        value=current_bench,
        title="bench",
        pos=[(0.6, 0.05), (0.8, 0.05)],
    )
    
    level_slider = plt.add_slider(
        level_slider_callback,
        xmin=1,
        xmax=xmax_level_slider,
        value=xmax_level_slider,
        title="level",
        pos=[(0.2, 0.05), (0.4, 0.05)],
    )

    state["bench_slider"] = bench_slider
    state["level_slider"] = level_slider
    
    rotation_state = {"on": False}

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
    
    configure_camera(plt, mesh)

    setup_excavation_lighting(plt, mesh)
    
    hud = Text2D(
        "",
        pos="top-left",
        c=(0.9, 0.9, 0.9),
        bg=(0.15, 0.15, 0.15),   # semi panel effect
        alpha=0.8,
        s=1.6,
    )
    plt.add(hud)
    update_hud(hud, state)

    plt.show(mesh, axes=0)
