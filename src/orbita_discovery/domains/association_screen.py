"""orbita_discovery.domains.association_screen — DIRECTION 2: which features
really predict the outcome?

Give it a table: an outcome column and several candidate feature columns. It
proposes 'feature F predicts y' for each, fits a single-feature model, and lets
the falsifiers separate real drivers from chance correlations. With several pure-
noise features in the table, some will clear a single held-out split by luck
(the multiple-comparisons trap) -- and the cross-seed falsifier is what kills
them. Real drivers survive all three attacks.

Same FittableDomain interface as numeric_law, so the SAME engine, judges, and
falsifiers run unchanged. That is the point: plug in a direction, spine stays put.
"""
from __future__ import annotations

import random
from typing import Any

from ..core import Candidate
from .numeric_law import _lstsq, _r2


class AssociationScreenDomain:
    name = "association_screen"

    def __init__(self, n: int = 160, n_noise: int = 5, noise: float = 1.0, seed: int = 7):
        rng = random.Random(seed)
        self.cols: dict[str, list[float]] = {}
        x_strong = [rng.gauss(0, 1) for _ in range(n)]
        x_mod = [rng.gauss(0, 1) for _ in range(n)]
        self.cols["drv_strong"] = x_strong
        self.cols["drv_moderate"] = x_mod
        for k in range(n_noise):
            self.cols[f"noise_{k}"] = [rng.gauss(0, 1) for _ in range(n)]
        # outcome depends ONLY on the two real drivers
        self.y = [2.5 * a + 1.8 * b + rng.gauss(0, 0.6) for a, b in zip(x_strong, x_mod)]

    def propose(self):
        for name in self.cols:
            yield Candidate(id=f"assoc:{name}",
                            statement=f"feature '{name}' predicts the outcome",
                            payload={"feature": name})

    def evidence_for(self, c: Candidate) -> Any:
        return {"x": self.cols[c.payload["feature"]], "y": self.y}

    def splits(self, evidence, seed: int):
        idx = list(range(len(evidence["y"])))
        random.Random(seed).shuffle(idx)
        cut = int(0.7 * len(idx))
        pick = lambda I: {"x": [evidence["x"][i] for i in I], "y": [evidence["y"][i] for i in I]}
        return pick(idx[:cut]), pick(idx[cut:])

    def refit(self, c: Candidate, train) -> Any:
        X = [[1.0, xi] for xi in train["x"]]
        return _lstsq(X, train["y"])

    def score(self, c: Candidate, model, test) -> float:
        yhat = [model[0] + model[1] * xi for xi in test["x"]]
        return _r2(test["y"], yhat)

    def baseline_score(self, test) -> float:
        return 0.0
