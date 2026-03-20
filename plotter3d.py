import numpy as np
from matplotlib import cm
from vedo import (
    Mesh,
    show,
)


def plot_extrusion_vedo(connectivity, verts3d, color_by_region=True, wireframe=False):
    faces = []
    colors = []

    regions = sorted(set(connectivity.top_triangle_regions))
    cmap = cm.get_cmap("tab20")

    region_to_color = {r: [int(255 * c) for c in cmap(i % 20)[:3]] for i, r in enumerate(regions)}

    # top
    for tri, region in zip(connectivity.top_triangles, connectivity.top_triangle_regions):
        faces.append(tri)
        colors.append(region_to_color[region])

    # internal walls
    for tri in connectivity.wall_triangles_internal:
        faces.append(tri)
        colors.append([200, 200, 200])

    # outer walls
    for tri in connectivity.wall_triangles_outer:
        faces.append(tri)
        colors.append([150, 150, 150])

    verts = np.asarray(verts3d, dtype=float)
    faces = np.asarray(faces, dtype=int)
    colors = np.asarray(colors, dtype=np.uint8)

    mesh = Mesh([verts, faces])

    if color_by_region:
        mesh.cellcolors = colors

    if wireframe:
        mesh.wireframe()

    mesh.lighting("plastic")
    show(mesh, axes=1)
