import json
from dataclasses import dataclass
from pathlib import Path


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
