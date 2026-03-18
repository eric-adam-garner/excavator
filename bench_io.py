import json

from models import Bench


def load_benches(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return[Bench.from_json(bench) for bench in data["benches"]]
