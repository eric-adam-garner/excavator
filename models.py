
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Bench:
    id: int
    points: npt.NDArray[np.float64]

    @classmethod
    def from_json(cls, json_bench):

        point_dicts = json_bench["polyline"]["points"]
        if not point_dicts:
            raise ValueError("Bench polyline has no points")

        points = np.array(
            [[p["x"], p["y"], p["z"]] for p in point_dicts],
            dtype=float
        )

        if not np.allclose(points[:, 2], points[0, 2]):
            raise ValueError("Bench polyline must be horizontal")

        return cls(
            id=json_bench["id"],
            points=points
        )
        
