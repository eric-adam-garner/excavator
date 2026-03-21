import logging


def setup_logging(level=logging.INFO):
    root = logging.getLogger()

    if root.handlers:
        return  # already configured

    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%H:%M:%S"
    ))

    root.addHandler(handler)