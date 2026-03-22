from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

from excavator.domain.builder import build_partition_domain, build_shell_domain
from excavator.domain.validator import validate_partition_domain
from excavator.geometry.tolerance import recommend_tol
from excavator.geometry.utils import point_in_polygon
from excavator.io.export_slabs import export_bench_slabs_obj
from excavator.io.load_benches import load_benches_json
from excavator.logger import setup_logging
from excavator.mesh.half_edge_mesh import extract_outer_loop_from_mesh, extract_region_loops
from excavator.triangulation.triangle_backend import (
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
    triangulate_shell_domain,
)
from excavator.vis.plotter3d import plot_excavation

# TODO: cleanup
# TODO: Add documentation
# TODO: Package

INPUT_BENCH_PATH = Path("data/input")
OUTPUT_SLAB_PATH = Path("data/output")

Z_INIT = 100.0


def extract_layer_number(filename: str) -> int:
    m = re.search(r"_(\d+)\.json$", filename)
    if not m:
        raise ValueError(f"No number found in {filename}")
    return int(m.group(1))

def main():

    logger = setup_logging()
    logger = logging.getLogger(__name__)

    level_height_filename_map = {}
    for filename in os.listdir("data/input"):
        path = INPUT_BENCH_PATH / filename
        benches = load_benches_json(path)
        z = benches[0].points3d[0][2]
        level_height_filename_map[z] = filename

    # -1: ground level, -2: first bench height, etc
    level_id_height_map = {-(i + 2): z for i, z in enumerate(sorted(level_height_filename_map.keys(), reverse=True))}
    level_id_height_map[-1] = Z_INIT

    zmin = min(level_height_filename_map.keys())
    zmax = max(level_height_filename_map.keys())

    assert zmax < Z_INIT

    z_prev = Z_INIT

    tol = recommend_tol(benches)

    super_loop = None

    outer_tri_meshes = {}
    bench_tri_meshes = {}

    for level_id in sorted(level_id_height_map, reverse=True):

        z = level_id_height_map[level_id]

        if z == Z_INIT:
            continue

        logger.info(f"processing {filename}")

        filename = level_height_filename_map[z]

        level_file_id = extract_layer_number(filename)

        path = INPUT_BENCH_PATH / filename
        benches = load_benches_json(path)

        domain = build_partition_domain(benches, tol)

        report = validate_partition_domain(domain)
        assert len(report.errors) == 0

        bench_tri_mesh = triangulate_partition_domain(domain, triangle_flags="pA")
        bench_half_edge_mesh = triangle_to_halfedge_mesh(bench_tri_mesh)

        if super_loop is None:
            xmin, ymin, xmax, ymax = bench_half_edge_mesh.bounding_box(padding=0.05)
            super_loop = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        assert max(bench_tri_mesh.triangle_region_ids) == len(benches) - 1

        outer_loop = extract_outer_loop_from_mesh(bench_half_edge_mesh)
        bench_loops = extract_region_loops(bench_half_edge_mesh)

        export_bench_slabs_obj(
            bench_tri_mesh,
            level_id_height_map,
            level_id,
            level_file_id,
            path=OUTPUT_SLAB_PATH,
        )

        shell_domain = build_shell_domain(
            super_loop=super_loop,
            outer_loop=outer_loop,
            tol=tol,
        )

        assert np.all(
            [point_in_polygon(pt, super_loop) for pt in outer_loop]
        ), "Outer loop boundary not contained in super loop"

        outer_tri_mesh = triangulate_shell_domain(
            shell_domain, outer_loop=super_loop, inner_loop=outer_loop, level_id=level_id + 1
        )

        outer_tri_meshes[level_id] = outer_tri_mesh
        bench_tri_meshes[level_id] = bench_tri_mesh

        super_loop = outer_loop

    plot_excavation(
        outer_tri_meshes=outer_tri_meshes,
        bench_tri_meshes=bench_tri_meshes,
        level_id_height_map=level_id_height_map,
        tol=tol,
        level_id=level_id,
    )

if __name__ == "__main__":
    main()