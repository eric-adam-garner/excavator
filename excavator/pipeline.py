from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from excavator.domain.builder import build_partition_domain, build_shell_domain
from excavator.domain.validator import validate_partition_domain
from excavator.geometry.tolerance import recommend_tol
from excavator.geometry.utils import point_in_polygon
from excavator.io.export_slabs import export_bench_slabs_obj
from excavator.io.load_benches import load_benches_json
from excavator.mesh.half_edge_mesh import extract_outer_loop_from_mesh
from excavator.triangulation.triangle_backend import (
    triangle_to_halfedge_mesh,
    triangulate_partition_domain,
    triangulate_shell_domain,
)


@dataclass(frozen=True)
class PipelineConfig:
    input_bench_path: Path
    output_slab_path: Path
    z_init: float = 100.0
    triangle_flags: str = "pA"
    super_loop_padding: float = 0.05
    
    
@dataclass
class ExcavationArtifacts:
    level_id_height_map: dict[int, float]
    level_height_filename_map: dict[float, str]
    outer_tri_meshes: dict[int, object]
    bench_tri_meshes: dict[int, object]
    tol: float
    final_level_id: int | None


class ExcavationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger("excavator.pipeline")
        self.mesh_logger = logging.getLogger("excavator.mesh")

    def run(self) -> ExcavationArtifacts:
        self.logger.info("Pipeline started")

        level_height_filename_map, all_benches = self._discover_inputs()
        tol = recommend_tol(all_benches)
        self.logger.info("Tolerance = %.6f", tol)

        level_id_height_map = self._build_level_id_height_map(level_height_filename_map)
        levels = self._build_processing_levels(level_id_height_map)

        outer_tri_meshes = {}
        bench_tri_meshes = {}
        super_loop = None
        final_level_id = None

        for level_id in levels:
            final_level_id = level_id
            z = level_id_height_map[level_id]
            filename = level_height_filename_map[z]

            self.logger.info(
                "Processing level_id=%d file=%s z=%.3f",
                level_id,
                filename,
                z,
            )

            benches = load_benches_json(self.config.input_bench_path / filename)

            level_result = self._process_level(
                benches=benches,
                tol=tol,
                level_id=level_id,
                level_id_height_map=level_id_height_map,
                filename=filename,
                super_loop=super_loop,
            )

            outer_tri_meshes[level_id] = level_result["outer_tri_mesh"]
            bench_tri_meshes[level_id] = level_result["bench_tri_mesh"]
            super_loop = level_result["outer_loop"]

        self.logger.info("Pipeline finished")

        return ExcavationArtifacts(
            level_id_height_map=level_id_height_map,
            level_height_filename_map=level_height_filename_map,
            outer_tri_meshes=outer_tri_meshes,
            bench_tri_meshes=bench_tri_meshes,
            tol=tol,
            final_level_id=final_level_id,
        )

    def _discover_inputs(self) -> tuple[dict[float, str], list]:
        level_height_filename_map = {}
        all_benches = []

        for path in sorted(self.config.input_bench_path.glob("*.json")):
            benches = load_benches_json(path)
            if not benches:
                raise RuntimeError(f"No benches found in {path}")

            z = benches[0].points3d[0][2]
            level_height_filename_map[z] = path.name
            all_benches.extend(benches)

        if not level_height_filename_map:
            raise RuntimeError("No input bench files found")

        zmax = max(level_height_filename_map.keys())
        if zmax >= self.config.z_init:
            raise RuntimeError("Top input elevation must be below z_init")

        self.logger.info("Discovered %d input files", len(level_height_filename_map))
        return level_height_filename_map, all_benches

    def _build_level_id_height_map(
        self,
        level_height_filename_map: dict[float, str],
    ) -> dict[int, float]:
        level_id_height_map = {
            -(i + 2): z
            for i, z in enumerate(sorted(level_height_filename_map.keys(), reverse=True))
        }
        level_id_height_map[-1] = self.config.z_init
        return level_id_height_map

    def _build_processing_levels(self, level_id_height_map: dict[int, float]) -> list[int]:
        return sorted((lid for lid in level_id_height_map if lid != -1), reverse=True)

    def _process_level(
        self,
        benches,
        tol: float,
        level_id: int,
        level_id_height_map: dict[int, float],
        filename: str,
        super_loop,
    ) -> dict:
        domain = build_partition_domain(benches, tol)

        report = validate_partition_domain(domain)
        if report.errors:
            self.logger.error("Domain validation failed for %s", filename)
            raise RuntimeError(report.errors)

        import time

        self.mesh_logger.info(
            "Partition triangulation | level=%d benches=%d",
            level_id,
            len(benches),
        )

        t0 = time.perf_counter()
        bench_tri_mesh = triangulate_partition_domain(
            domain,
            triangle_flags=self.config.triangle_flags,
        )
        dt = time.perf_counter() - t0

        self.mesh_logger.debug(
            "Partition mesh built | tris=%d time=%.3fs",
            len(bench_tri_mesh.triangles),
            dt,
        )
        
        self.mesh_logger.debug("Building half-edge mesh")

        t0 = time.perf_counter()
        bench_half_edge_mesh = triangle_to_halfedge_mesh(bench_tri_mesh)
        dt = time.perf_counter() - t0

        self.mesh_logger.debug(
            "Half-edge mesh built | faces=%d edges=%d time=%.3fs",
            bench_half_edge_mesh.num_faces(),
            bench_half_edge_mesh.num_edges(),
            dt,
        )

        if super_loop is None:
            self.logger.info("Creating initial super loop")
            xmin, ymin, xmax, ymax = bench_half_edge_mesh.bounding_box(
                padding=self.config.super_loop_padding
            )
            super_loop = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        outer_loop = extract_outer_loop_from_mesh(bench_half_edge_mesh)

        export_bench_slabs_obj(
            bench_tri_mesh=bench_tri_mesh,
            level_id_height_map=level_id_height_map,
            level_id=level_id,
            level_file_id=self._extract_layer_number(filename),
            path=self.config.output_slab_path,
        )

        shell_domain = build_shell_domain(
            super_loop=super_loop,
            outer_loop=outer_loop,
            tol=tol,
        )

        if not np.all([point_in_polygon(pt, super_loop) for pt in outer_loop]):
            raise RuntimeError("Outer loop boundary not contained in super loop")

        outer_tri_mesh = triangulate_shell_domain(
            shell_domain,
            outer_loop=super_loop,
            inner_loop=outer_loop,
            level_id=level_id + 1,
        )

        return {
            "bench_tri_mesh": bench_tri_mesh,
            "outer_tri_mesh": outer_tri_mesh,
            "outer_loop": outer_loop,
        }

    @staticmethod
    def _extract_layer_number(filename: str) -> int:
        import re

        m = re.search(r"_(\d+)\.json$", filename)
        if not m:
            raise ValueError(f"No number found in {filename}")
        return int(m.group(1))