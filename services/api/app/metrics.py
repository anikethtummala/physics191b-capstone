from __future__ import annotations

import math


CONFIDENCE_LEVEL = 0.95
CONFIDENCE_METHOD = "wilson_score"


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 0.0

    phat = successes / trials
    denominator = 1 + (z * z) / trials
    center = (phat + (z * z) / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt((phat * (1 - phat)) / trials + (z * z) / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
