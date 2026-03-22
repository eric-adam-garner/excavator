import logging
from pathlib import Path

from excavator.logger import setup_logging
from excavator.pipeline import ExcavationPipeline, PipelineConfig
from excavator.vis.plotter3d import plot_excavation

INPUT_BENCH_PATH = Path("data/input")
OUTPUT_SLAB_PATH = Path("data/output")


def main():

    setup_logging(logging.DEBUG)
    logger = logging.getLogger("excavator.app")

    logger.info("Starting excavation application")

    config = PipelineConfig(
        input_bench_path=INPUT_BENCH_PATH,
        output_slab_path=OUTPUT_SLAB_PATH,
        z_init=100.0,
        triangle_flags="pA",
        super_loop_padding=0.05,
    )

    pipeline = ExcavationPipeline(config)
    artifacts = pipeline.run()

    logger.info(
        "Launching visualization | levels=%d",
        len(artifacts.level_id_height_map) - 1,
    )

    plot_excavation(
        outer_tri_meshes=artifacts.outer_tri_meshes,
        bench_tri_meshes=artifacts.bench_tri_meshes,
        level_id_height_map=artifacts.level_id_height_map,
        tol=artifacts.tol,
        level_id=artifacts.final_level_id,
    )