import logging

import trimesh

from extrusion import extrude_mesh_between_z

logger = logging.getLogger(__name__)


def export_bench_slabs_obj(bench_tri_mesh, level_id_height_map, level_id, level_file_id, path):

    z0 = level_id_height_map[level_id]
    z1 = level_id_height_map[level_id + 1]

    bench_faces = {key: [] for key in set(bench_tri_mesh.triangle_region_ids)}
    for idx, bench_id in enumerate(bench_tri_mesh.triangle_region_ids):
        bench_faces[bench_id].append(bench_tri_mesh.triangles[idx])

    for bench_id, faces in bench_faces.items():
        vertices, faces = extrude_mesh_between_z(
            vertices=bench_tri_mesh.vertices,
            faces=faces,
            z0=z0,
            z1=z1,
        )
        msh_tm = trimesh.Trimesh(vertices=vertices, faces=faces)
        msh_tm.fix_normals()
        msh_tm.export(path / f"{level_file_id}-{bench_id}.obj")

    logger.info(f"exported level: {level_file_id} slabs")
