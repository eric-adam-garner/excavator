import json
from dataclasses import dataclass
from pathlib import Path

import trimesh

from extrusion import extrude_mesh_between_z


@dataclass
class BenchPolyline:
    id: str
    points3d: list[tuple[float, float, float]]

    def to_2d(self):
        return [(x, y) for x, y, _ in self.points3d]


def load_benches_json(path: str | Path) -> list[BenchPolyline]:
    with open(path) as f:
        data = json.load(f)

    benches = []

    for b in data["benches"]:
        pts = b["polyline"]["points"]
        points3d = [(p["x"], p["y"], p["z"]) for p in pts]
        benches.append(BenchPolyline(b["id"], points3d))

    return benches


def export_bench_slabs_obj(bench_tri_mesh, z0, z1, path):
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
        msh_tm.export(path / f"{bench_id}.obj")
