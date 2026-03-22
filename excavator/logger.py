import logging


def setup_logging(level=logging.INFO):

    root = logging.getLogger()

    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(level)

    # ---- suppress noisy libraries
    logging.getLogger("trimesh").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("vedo").setLevel(logging.WARNING)
    logging.getLogger("vtk").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # triangle often logs under root or generic names
    logging.getLogger("triangle").setLevel(logging.ERROR)