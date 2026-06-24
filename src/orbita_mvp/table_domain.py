from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from orbita_discovery.core import Candidate


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _candidate_id(kind: str, *parts: str) -> str:
    raw = "|".join([kind, *parts])
    return f"{kind}:{_slug('_'.join(parts))}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]
    if len(y) < 3:
        return 0.0
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1e-15:
        return 0.0
    score = 1.0 - float(np.sum((y - pred) ** 2)) / denom
    return max(-1.0, min(1.0, score))


def _goal_columns(goal: str, columns: Iterable[str]) -> list[str]:
    normalized_goal = re.sub(r"[^a-z0-9]+", " ", goal.lower())
    found = []
    for column in columns:
        forms = {
            column.lower(),
            column.lower().replace("_", " "),
            re.sub(r"[^a-z0-9]+", " ", column.lower()).strip(),
        }
        if any(form and form in normalized_goal for form in forms):
            found.append(column)
    return found


def generate_table_candidates(
    df: pd.DataFrame,
    *,
    goal: str = "",
    max_candidates: int = 60,
    scout_fraction: float = 0.6,
    seed: int = 20260623,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(df) < 6:
        raise ValueError("At least 6 rows are required for discovery and held-out checking")
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    cut = max(3, min(len(indices) - 3, int(len(indices) * scout_fraction)))
    scout = df.iloc[indices[:cut]].copy()

    numeric_columns = []
    categorical_columns = []
    for column in df.columns:
        name = str(column)
        numeric_fraction = float(pd.to_numeric(df[column], errors="coerce").notna().mean())
        unique = int(df[column].nunique(dropna=True))
        if numeric_fraction >= 0.85 and unique >= 3:
            numeric_columns.append(name)
        elif 2 <= unique <= min(12, max(3, int(len(df) * 0.2))):
            categorical_columns.append(name)

    goal_columns = _goal_columns(goal, [str(c) for c in df.columns]) if goal.strip() else []
    scored: list[tuple[float, dict[str, Any]]] = []

    for i, x in enumerate(numeric_columns):
        for y in numeric_columns[i + 1 :]:
            if goal_columns and not ({x, y} & set(goal_columns)):
                continue
            pair = scout[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 5 or pair[x].nunique() < 3 or pair[y].nunique() < 3:
                continue
            r = float(pair[x].corr(pair[y]))
            if not math.isfinite(r) or abs(r) < 0.2:
                continue
            direction = "positive" if r >= 0 else "negative"
            # Prefer a goal-named column as outcome when possible; otherwise keep a stable order.
            if y in goal_columns and x not in goal_columns:
                predictor, outcome = x, y
            elif x in goal_columns and y not in goal_columns:
                predictor, outcome = y, x
            else:
                predictor, outcome = x, y
            spec = {
                "id": _candidate_id("linear", predictor, outcome),
                "statement": f"{predictor} and {outcome} show a stable {direction} linear association.",
                "kind": "linear_association",
                "predictor": predictor,
                "outcome": outcome,
                "expected_direction": direction,
                "scout_metric": {"pearson_r": r, "n": int(len(pair))},
                "parents": [],
            }
            scored.append((abs(r), spec))

    for group in categorical_columns:
        groups = scout[group].dropna().astype(str)
        if groups.nunique() < 2:
            continue
        for outcome in numeric_columns:
            if goal_columns and not ({group, outcome} & set(goal_columns)):
                continue
            temp = pd.DataFrame({"group": groups, "y": pd.to_numeric(scout.loc[groups.index, outcome], errors="coerce")}).dropna()
            if len(temp) < 6:
                continue
            overall = float(temp["y"].mean())
            total = float(((temp["y"] - overall) ** 2).sum())
            if total <= 1e-15:
                continue
            between = 0.0
            counts: dict[str, int] = {}
            means: dict[str, float] = {}
            for label, subset in temp.groupby("group"):
                counts[str(label)] = int(len(subset))
                means[str(label)] = float(subset["y"].mean())
                between += len(subset) * (float(subset["y"].mean()) - overall) ** 2
            eta2 = between / total
            if eta2 < 0.04:
                continue
            spec = {
                "id": _candidate_id("group", group, outcome),
                "statement": f"{outcome} differs systematically across levels of {group}.",
                "kind": "group_difference",
                "group": group,
                "outcome": outcome,
                "scout_metric": {"eta_squared": eta2, "group_counts": counts, "group_means": means, "n": int(len(temp))},
                "parents": [],
            }
            scored.append((float(eta2), spec))

    scored.sort(key=lambda item: (-item[0], item[1]["id"]))
    candidates = [spec for _, spec in scored[:max_candidates]]
    generation = {
        "strategy": "locked_scout_then_confirmation",
        "seed": seed,
        "scout_fraction": scout_fraction,
        "scout_rows": len(scout),
        "confirmation_rows": len(df) - len(scout),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "goal_columns": goal_columns,
        "generated_candidates": len(candidates),
        "candidate_budget": max_candidates,
    }
    return candidates, generation


class UploadedTableDomain:
    """Fittable domain for frozen candidates compiled from an uploaded table.

    Candidate generation sees only a deterministic scout partition. The judge
    and falsifiers score frozen candidates on the locked confirmation partition.
    Cross-seed checks bootstrap the confirmation partition rather than exposing
    it to candidate generation.
    """

    name = "uploaded_table"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        candidates: list[dict[str, Any]],
        *,
        scout_fraction: float = 0.6,
        seed: int = 20260623,
    ):
        if not candidates:
            raise ValueError("The approved plan contains no testable candidates")
        self.df = dataframe.reset_index(drop=True)
        self.specs = candidates
        indices = list(range(len(self.df)))
        random.Random(seed).shuffle(indices)
        cut = max(3, min(len(indices) - 3, int(len(indices) * scout_fraction)))
        self.scout = self.df.iloc[indices[:cut]].copy()
        self.confirmation = self.df.iloc[indices[cut:]].copy()

    def propose(self):
        for spec in self.specs:
            yield Candidate(
                id=str(spec["id"]),
                statement=str(spec["statement"]),
                payload={k: v for k, v in spec.items() if k not in {"id", "statement", "parents"}},
                parents=list(spec.get("parents", [])),
            )

    def evidence_for(self, c: Candidate) -> Any:
        return {"scout": self.scout, "confirmation": self.confirmation}

    def splits(self, evidence: Any, seed: int):
        train = evidence["scout"]
        confirmation = evidence["confirmation"]
        if seed in {0, 1} or len(confirmation) < 4:
            return train, confirmation
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(confirmation), size=len(confirmation))
        return train, confirmation.iloc[picks].copy()

    def refit(self, c: Candidate, train: pd.DataFrame) -> dict[str, Any]:
        kind = c.payload["kind"]
        if kind == "linear_association":
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            pair = train[[x_name, y_name]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                return {"kind": kind, "valid": False}
            x = pair[x_name].to_numpy(float)
            y = pair[y_name].to_numpy(float)
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return {"kind": kind, "valid": True, "intercept": float(beta[0]), "slope": float(beta[1])}
        if kind == "group_difference":
            group, outcome = c.payload["group"], c.payload["outcome"]
            temp = pd.DataFrame({"g": train[group].astype(str), "y": pd.to_numeric(train[outcome], errors="coerce")}).dropna()
            means = {str(label): float(part["y"].mean()) for label, part in temp.groupby("g")}
            return {"kind": kind, "valid": bool(means), "means": means, "overall": float(temp["y"].mean()) if len(temp) else 0.0}
        return {"kind": kind, "valid": False}

    def score(self, c: Candidate, model: dict[str, Any], test: pd.DataFrame) -> float:
        if not model.get("valid") or len(test) < 3:
            return 0.0
        kind = model["kind"]
        if kind == "linear_association":
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            pair = test[[x_name, y_name]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3:
                return 0.0
            x = pair[x_name].to_numpy(float)
            y = pair[y_name].to_numpy(float)
            slope = float(model["slope"])
            expected = c.payload.get("expected_direction")
            if expected == "positive" and slope <= 0:
                return -1.0
            if expected == "negative" and slope >= 0:
                return -1.0
            return _r2(y, model["intercept"] + slope * x)
        if kind == "group_difference":
            group, outcome = c.payload["group"], c.payload["outcome"]
            temp = pd.DataFrame({"g": test[group].astype(str), "y": pd.to_numeric(test[outcome], errors="coerce")}).dropna()
            if len(temp) < 3:
                return 0.0
            pred = np.array([model["means"].get(label, model["overall"]) for label in temp["g"]], dtype=float)
            return _r2(temp["y"].to_numpy(float), pred)
        return 0.0

    def baseline_score(self, test: Any) -> float:
        return 0.0
