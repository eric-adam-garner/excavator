from __future__ import annotations

import argparse
import logging
from pathlib import Path

from excavator.app import run_app
from excavator.logger import setup_logging


def _ensure_viz_dependencies() -> None:
    """
    Ensure visualization dependencies are installed.

    This prevents confusing VTK / vedo import crashes on interview machines.
    """
    try:
        import vedo  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "\nVisualization dependencies are missing.\n\n"
            "Install with:\n"
            "    pip install excavator[viz]\n"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="excavator-demo",
        description="Excavator computational geometry demo",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input"),
        help="Input bench directory",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output"),
        help="Output slab directory",
    )

    parser.add_argument(
        "--z-init",
        type=float,
        default=100.0,
        help="Initial ground elevation",
    )

    parser.add_argument(
        "--triangle-flags",
        type=str,
        default="pA",
        help="Triangle meshing flags",
    )

    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Run pipeline without launching visualization",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    return parser


def main() -> None:
    
    _ensure_viz_dependencies()
    
    parser = build_parser()
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(log_level)

    logger = logging.getLogger("excavator.app")
    logger.info("CLI invocation received")

    run_app(
        input_bench_path=args.input,
        output_slab_path=args.output,
        z_init=args.z_init,
        triangle_flags=args.triangle_flags,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
    
