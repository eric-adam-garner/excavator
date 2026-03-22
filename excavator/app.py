from __future__ import annotations

import logging
from pathlib import Path

from excavator.pipeline import ExcavationPipeline, PipelineConfig


def run_app(
    input_bench_path: Path = Path("data/input"),
    output_slab_path: Path = Path("data/output"),
    *,
    z_init: float = 100.0,
    triangle_flags: str = "pA",
    super_loop_padding: float = 0.05,
    visualize: bool = True,
):
    """
    Run the excavation application.

    Builds excavation artifacts through the pipeline and optionally launches
    the visualization layer.

    Params:
    -------
    input_bench_path:
        Directory containing input bench JSON files.
    output_slab_path:
        Directory where slab exports are written.
    z_init:
        Initial ground elevation.
    triangle_flags:
        Flags forwarded to the triangulation backend.
    super_loop_padding:
        Padding applied when constructing the initial super loop.
    visualize:
        Whether to launch the 3D viewer after pipeline completion.

    Returns:
    --------
    ExcavationArtifacts
        Pipeline outputs including meshes, level map, and tolerance.
    """
    logger = logging.getLogger("excavator.app")

    logger.info("Starting excavation application")

    config = PipelineConfig(
        input_bench_path=input_bench_path,
        output_slab_path=output_slab_path,
        z_init=z_init,
        triangle_flags=triangle_flags,
        super_loop_padding=super_loop_padding,
    )

    pipeline = ExcavationPipeline(config)
    artifacts = pipeline.run()

    if visualize:
        logger.info(
            "Launching visualization | levels=%d",
            len(artifacts.level_id_height_map) - 1,
        )

        try:
            import vedo  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "\nVisualization dependencies are missing.\n\n"
                "Install with:\n"
                "    pip install -e .[viz]\n"
            ) from e

        from excavator.vis.plotter3d import plot_excavation

        logger.info("Visualization mode active (extras: viz)")

        plot_excavation(
            outer_tri_meshes=artifacts.outer_tri_meshes,
            bench_tri_meshes=artifacts.bench_tri_meshes,
            level_id_height_map=artifacts.level_id_height_map,
            tol=artifacts.tol,
            level_id=artifacts.final_level_id,
        )

    return artifacts