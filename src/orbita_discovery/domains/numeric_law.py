"""orbita_discovery.domains.numeric_law — DIRECTION 1: symbolic-law discovery.

Hand it (x, y) data from some hidden process. It proposes a family of functional
forms (linear, quadratic, log, power, exponential, sinusoid), fits each in pure
Python, and lets the engine's falsifiers try to kill them. The law that beats the
baseline AND survives held-out AND survives cross-seed reshuffling is reported as
a *candidate* (never 'proved').

The critical test is the world='noise' case: a trustworthy discovery engine must
return NOTHING there. The naive judge will happily 'discover' a quadratic in pure
noise; the gated judge will not. That contrast is the demo.

Zero dependencies (no numpy) so it runs anywhere Python runs.
"""
from __future__ import annotations

import math
import random
from typing import Any

from ..core import Candidate

KINDS = ["constant", "linear", "quadratic", "log", "power", "exp", "sinusoid"]


# --------------------------------------------------------------------- tiny linalg
def _lstsq(X: list[list[float]], y: list[float]) -> list[float]:
    """Solve least squares via normal equations + Gaussian elimination (ridge-stabilized)."""
    k = len(X[0])
    A = [[0.0] * (k + 1) for _ in range(k)]
    for row, yi in zip(X, y):
        for i in range(k):
            A[i][k] += row[i] * yi
            for j in range(k):
                A[i][j] += row[i] * row[j]
    for i in range(k):
        A[i][i] += 1e-9
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        if abs(A[col][col]) < 1e-12:
            continue
        for r in range(k):
            if r == col:
                continue
            f = A[r][col] / A[col][col]
            for cc in range(col, k + 1):
                A[r][cc] -= f * A[col][cc]
    return [A[i][k] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(k)]


def _r2(y: list[float], yhat: list[float]) -> float:
    n = len(y)
    if n == 0:
        return 0.0
    mean = sum(y) / n
    ss_tot = sum((v - mean) ** 2 for v in y)
    ss_res = sum((v - h) ** 2 for v, h in zip(y, yhat))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------ per-kind fit / predict
def _features(kind: str, xs: list[float]):
    if kind == "constant":
        return [[1.0] for _ in xs]
    if kind == "linear":
        return [[1.0, x] for x in xs]
    if kind == "quadratic":
        return [[1.0, x, x * x] for x in xs]
    if kind == "log":
        return [[1.0, math.log(x)] for x in xs]      # requires x>0
    return None


def _fit(kind: str, xs: list[float], ys: list[float]) -> dict | None:
    try:
        if kind in ("constant", "linear", "quadratic", "log"):
            X = _features(kind, xs)
            return {"kind": kind, "beta": _lstsq(X, ys)}
        if kind == "power":                          # y = a * x^b  -> ln y = ln a + b ln x
            if any(x <= 0 or v <= 0 for x, v in zip(xs, ys)):
                return None
            X = [[1.0, math.log(x)] for x in xs]
            beta = _lstsq(X, [math.log(v) for v in ys])
            return {"kind": kind, "beta": beta}
        if kind == "exp":                            # y = a * e^(b x) -> ln y = ln a + b x
            if any(v <= 0 for v in ys):
                return None
            X = [[1.0, x] for x in xs]
            beta = _lstsq(X, [math.log(v) for v in ys])
            return {"kind": kind, "beta": beta}
        if kind == "sinusoid":                       # y = c0 + a sin(w x) + b cos(w x); grid w
            best = None
            for i in range(1, 80):
                w = 0.1 + i * (6.0 / 80)
                X = [[1.0, math.sin(w * x), math.cos(w * x)] for x in xs]
                beta = _lstsq(X, ys)
                yhat = [beta[0] + beta[1] * math.sin(w * x) + beta[2] * math.cos(w * x) for x in xs]
                r = _r2(ys, yhat)
                if best is None or r > best[0]:
                    best = (r, w, beta)
            return {"kind": kind, "w": best[1], "beta": best[2]}
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _predict(model: dict, xs: list[float]) -> list[float]:
    if model is None:
        return [0.0] * len(xs)
    k, beta = model["kind"], model.get("beta", [])
    if k == "constant":
        return [beta[0]] * len(xs)
    if k == "linear":
        return [beta[0] + beta[1] * x for x in xs]
    if k == "quadratic":
        return [beta[0] + beta[1] * x + beta[2] * x * x for x in xs]
    if k == "log":
        return [beta[0] + beta[1] * math.log(x) if x > 0 else 0.0 for x in xs]
    if k == "power":
        return [math.exp(beta[0]) * (x ** beta[1]) if x > 0 else 0.0 for x in xs]
    if k == "exp":
        return [math.exp(beta[0] + beta[1] * x) for x in xs]
    if k == "sinusoid":
        w = model["w"]
        return [beta[0] + beta[1] * math.sin(w * x) + beta[2] * math.cos(w * x) for x in xs]
    return [0.0] * len(xs)


# ------------------------------------------------------------------------ the domain
class NumericLawDomain:
    name = "numeric_law"

    def __init__(self, world: str = "sin", n: int = 120, noise: float = 0.15, seed: int = 7):
        self.world = world
        rng = random.Random(seed)
        xs = [round(0.5 + 6.0 * i / (n - 1), 5) for i in range(n)]
        rng.shuffle(xs)
        self.xs = xs
        self.ys = [self._world(world, x, rng, noise) for x in xs]

    @staticmethod
    def _world(world: str, x: float, rng: random.Random, noise: float) -> float:
        base = {
            "linear":    2.0 * x + 1.0,
            "power":     3.0 * (x ** 1.7),
            "sin":       4.0 * math.sin(1.5 * x) + 6.0,
            "log":       2.5 * math.log(x) + 3.0,
            "noise":     0.0,
        }[world]
        scale = 1.0 if world == "noise" else abs(base) + 1.0
        return base + rng.gauss(0.0, noise * scale if world != "noise" else 1.0)

    # ---- engine plugin interface
    def propose(self):
        for kind in KINDS:
            if kind == "constant":
                continue                  # constant is the baseline, not a candidate
            yield Candidate(id=f"law:{kind}", statement=f"y follows a {kind} law in x",
                            payload={"kind": kind})

    def evidence_for(self, c: Candidate) -> Any:
        return {"xs": self.xs, "ys": self.ys}

    # ---- FittableDomain interface (what the falsifiers + judges call)
    def splits(self, evidence, seed: int):
        idx = list(range(len(evidence["xs"])))
        random.Random(seed).shuffle(idx)
        cut = int(0.7 * len(idx))
        tr, te = idx[:cut], idx[cut:]
        pick = lambda I: {"xs": [evidence["xs"][i] for i in I], "ys": [evidence["ys"][i] for i in I]}
        return pick(tr), pick(te)

    def refit(self, c: Candidate, train) -> Any:
        return _fit(c.payload["kind"], train["xs"], train["ys"])

    def score(self, c: Candidate, model, test) -> float:
        yhat = _predict(model, test["xs"])
        return _r2(test["ys"], yhat)

    def baseline_score(self, test) -> float:
        return 0.0                         # R^2 of the mean predictor is 0 by definition
