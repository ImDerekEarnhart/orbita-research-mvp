from __future__ import annotations

import hashlib
import math
import random
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

from orbita_discovery.core import Candidate

from .artifacts import detect_structural_relations, structural_for
from .metrics import compute_metric, higher_is_better, validate_metric


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48]


def _candidate_id(kind: str, *parts: str) -> str:
    raw = "|".join([kind, *parts])
    return f"{kind}:{_slug('_'.join(parts))}:{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    """Internal R² — used by all model-fitness checks regardless of evaluation metric."""
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


# ---------------------------------------------------------------------------
# Nonlinear candidate families (quadratic / log-x / log-y / log-log power law).
#
# Every form is fit in its natural space but ALWAYS SCORED in the original
# outcome units (predictions inverted back before R²) so scores are directly
# comparable across forms of the same relationship — a log fit gets no unfair
# advantage from being evaluated in log space (see metrics.py note).
# ---------------------------------------------------------------------------

# Model complexity (free parameters) per form, for complexity-aware selection.
FORM_COMPLEXITY = {"linear": 2, "log_x": 2, "log_y": 2, "log_log": 2, "quadratic": 3}
_NONLINEAR_FORMS = ("quadratic", "log_x", "log_y", "log_log")


def _fit_form(x: np.ndarray, y: np.ndarray, form: str) -> dict[str, Any] | None:
    """Fit ``form`` and return ``{params, r2}`` with r2 in ORIGINAL y units.

    Returns None when the form is inapplicable (e.g. log of non-positive values)
    or degenerate.
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.ptp(x) <= 1e-12:
        return None
    if form == "quadratic":
        X = np.column_stack([np.ones(len(x)), x, x ** 2])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        return {"params": {"intercept": float(beta[0]), "b1": float(beta[1]), "b2": float(beta[2])},
                "r2": _r2(y, pred)}
    if form == "log_x":
        if np.any(x <= 0):
            return None
        lx = np.log(x)
        X = np.column_stack([np.ones(len(lx)), lx])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ beta
        return {"params": {"intercept": float(beta[0]), "slope": float(beta[1])}, "r2": _r2(y, pred)}
    if form == "log_y":
        if np.any(y <= 0):
            return None
        ly = np.log(y)
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        pred = np.exp(X @ beta)
        return {"params": {"intercept": float(beta[0]), "slope": float(beta[1])}, "r2": _r2(y, pred)}
    if form == "log_log":
        if np.any(x <= 0) or np.any(y <= 0):
            return None
        lx, ly = np.log(x), np.log(y)
        X = np.column_stack([np.ones(len(lx)), lx])
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        pred = np.exp(X @ beta)
        # slope in log-log space is the power-law exponent.
        return {"params": {"intercept": float(beta[0]), "slope": float(beta[1])},
                "r2": _r2(y, pred), "exponent": float(beta[1])}
    return None


def _predict_form(x: np.ndarray, form: str, params: dict[str, float]) -> np.ndarray | None:
    """Predict outcome values in ORIGINAL units for ``form`` given stored params."""
    with np.errstate(all="ignore"):
        if form == "quadratic":
            return params["intercept"] + params["b1"] * x + params["b2"] * x ** 2
        if form == "log_x":
            out = np.full_like(x, np.nan, dtype=float)
            ok = x > 0
            out[ok] = params["intercept"] + params["slope"] * np.log(x[ok])
            return out
        if form == "log_y":
            return np.exp(params["intercept"] + params["slope"] * x)
        if form == "log_log":
            out = np.full_like(x, np.nan, dtype=float)
            ok = x > 0
            out[ok] = np.exp(params["intercept"] + params["slope"] * np.log(x[ok]))
            return out
    return None


def _nonlinear_statement(predictor: str, outcome: str, form: str, params: dict[str, float]) -> str:
    if form == "quadratic":
        return f"{outcome} has a quadratic (single-peak/curved) relationship with {predictor}."
    if form == "log_x":
        return f"{outcome} varies with the logarithm of {predictor} (diminishing-returns relationship)."
    if form == "log_y":
        return f"{outcome} grows exponentially with {predictor} (log {outcome} is linear in {predictor})."
    if form == "log_log":
        exp = params.get("slope")
        exp_txt = f" with exponent ≈ {exp:.3g}" if isinstance(exp, (int, float)) else ""
        return f"{outcome} follows an approximate power law in {predictor} ({outcome} ≈ a·{predictor}^b){exp_txt}."
    return f"{outcome} has a nonlinear ({form}) relationship with {predictor}."


def _nonlinear_form_specs(
    scout: pd.DataFrame,
    predictor: str,
    outcome: str,
    *,
    screen_r2: float = 0.2,
) -> list[tuple[float, dict[str, Any]]]:
    """Fit each nonlinear form on the scout partition; emit specs that clear the screen."""
    x = pd.to_numeric(scout[predictor], errors="coerce").to_numpy(float)
    y = pd.to_numeric(scout[outcome], errors="coerce").to_numpy(float)
    out: list[tuple[float, dict[str, Any]]] = []
    for form in _NONLINEAR_FORMS:
        fit = _fit_form(x, y, form)
        if not fit:
            continue
        r2 = fit["r2"]
        if not math.isfinite(r2) or r2 < screen_r2:
            continue
        spec = {
            "id": _candidate_id("nlform", predictor, outcome, form),
            "statement": _nonlinear_statement(predictor, outcome, form, fit["params"]),
            "kind": "nonlinear_association",
            "predictor": predictor,
            "outcome": outcome,
            "form": form,
            "scout_metric": {
                "r2": round(r2, 6),
                "params": {k: round(v, 6) for k, v in fit["params"].items()},
                "complexity": FORM_COMPLEXITY.get(form),
                "n": int(np.isfinite(x).sum()),
            },
            "parents": [],
        }
        if "exponent" in fit:
            spec["scout_metric"]["exponent"] = round(fit["exponent"], 6)
        out.append((r2, spec))
    return out


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
    exclude_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(df) < 6:
        raise ValueError("At least 6 rows are required for discovery and held-out checking")
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} not found in dataset columns")
    _exclude: set[str] = set(exclude_columns or [])
    rng = random.Random(seed)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    cut = max(3, min(len(indices) - 3, int(len(indices) * scout_fraction)))
    scout = df.iloc[indices[:cut]].copy()

    # Predictor-eligible columns: exclude identifiers AND the target column.
    # The target column is allowed as an OUTCOME but never as a predictor.
    _exclude_as_predictor: set[str] = _exclude | ({target_column} if target_column else set())
    numeric_columns = []        # eligible as predictors
    categorical_columns = []    # eligible as predictors
    for column in df.columns:
        if str(column) in _exclude_as_predictor:
            continue
        name = str(column)
        numeric_fraction = float(pd.to_numeric(df[column], errors="coerce").notna().mean())
        unique = int(df[column].nunique(dropna=True))
        if numeric_fraction >= 0.85 and unique >= 3:
            numeric_columns.append(name)
        elif 2 <= unique <= min(12, max(3, int(len(df) * 0.2))):
            categorical_columns.append(name)

    goal_columns = _goal_columns(goal, [str(c) for c in df.columns]) if goal.strip() else []
    # When an explicit target_column is provided it is always the outcome.
    # goal_columns are still used to constrain which pairs are explored, but
    # the direction is always predictor → target_column.
    if target_column and target_column not in goal_columns:
        goal_columns = goal_columns + [target_column]
    scored: list[tuple[float, dict[str, Any]]] = []

    # Build numeric columns that CAN appear as outcomes.  When target_column is
    # specified only that column is a valid outcome; otherwise any numeric column is.
    _target_col_set: set[str] = {target_column} if target_column else set()
    _any_outcome = not bool(target_column)

    def _is_valid_outcome(col: str) -> bool:
        return _any_outcome or col in _target_col_set

    # The target column must also be included in correlation checks even though
    # it is excluded from numeric_columns (the predictor list).  Add it for
    # structural detection and pair scoring using the full df column list.
    target_numeric: list[str] = []
    if target_column:
        tc_series = pd.to_numeric(df[target_column], errors="coerce")
        if tc_series.notna().mean() >= 0.85 and tc_series.nunique() >= 3:
            target_numeric = [target_column]

    # structural relations check all numeric predictor columns pairwise.
    structural = detect_structural_relations(df, numeric_columns=numeric_columns)
    structural_relations: list[dict[str, Any]] = []
    seen_structural: set[str] = set()

    # Build the full set of numeric columns for outcome lookup.
    all_numeric = numeric_columns + target_numeric

    for i, x in enumerate(all_numeric):
        for y in all_numeric[i + 1 :]:
            # At least one of the pair must be an eligible outcome.
            if not (_is_valid_outcome(x) or _is_valid_outcome(y)):
                continue
            # If a goal_columns filter is active, respect it.
            if goal_columns and not ({x, y} & set(goal_columns)):
                continue
            # Enforce: target_column is never a predictor.
            # When x == target_column it would become the predictor in the else
            # branch below; skip entirely so direction logic below never sees it.
            if target_column and x == target_column:
                continue
            artifact = structural_for(structural, x, y)
            if artifact is not None:
                key = _candidate_id("structural", x, y)
                if key not in seen_structural:
                    seen_structural.add(key)
                    structural_relations.append({
                        "id": key,
                        "statement": (
                            f"{x} and {y} are structurally related "
                            f"({artifact['kind']}): {artifact.get('detail', '')}."
                        ),
                        "kind": "structural_relation",
                        "artifact_kind": artifact["kind"],
                        "detail": artifact.get("detail", ""),
                        "columns": [x, y],
                    })
                continue
            pair = scout[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 5 or pair[x].nunique() < 3 or pair[y].nunique() < 3:
                continue
            r = float(pair[x].corr(pair[y]))
            if not math.isfinite(r):
                continue
            # Orientation is independent of the linear-correlation strength, so
            # nonlinear forms can be screened even when |r| ≈ 0 (e.g. a
            # symmetric inverted-U has near-zero linear correlation).
            # Explicit target takes precedence over goal_columns direction.
            if target_column and y == target_column:
                predictor, outcome = x, y
            elif target_column and x == target_column:
                predictor, outcome = y, x
            elif y in goal_columns and x not in goal_columns:
                predictor, outcome = x, y
            elif x in goal_columns and y not in goal_columns:
                predictor, outcome = y, x
            else:
                predictor, outcome = x, y
            # Hard guard: target_column must not be a predictor.
            if target_column and predictor == target_column:
                continue

            # Nonlinear candidate family (quadratic / log-x / log-y / log-log).
            # Members of the same relationship family as the linear form; the
            # selection layer picks the preferred form after falsification.
            nonlinear_specs = _nonlinear_form_specs(scout, predictor, outcome)
            for nl_score, nl_spec in nonlinear_specs:
                scored.append((float(nl_score), nl_spec))

            # Emit the linear form when it clears its own correlation screen OR
            # when a nonlinear sibling exists — so a family always includes the
            # raw linear baseline. A weak linear form (e.g. the failed linear
            # fit of an inverted-U) is then killed and reclassified as
            # functional_form_rejected against the surviving curved sibling,
            # rather than the underlying relationship being called refuted.
            if abs(r) < 0.2 and not nonlinear_specs:
                continue
            direction = "positive" if r >= 0 else "negative"
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
        outcome_pool = all_numeric if target_numeric else numeric_columns
        for outcome in outcome_pool:
            # Categorical predictor: target_column can only be an outcome here.
            if target_column and outcome != target_column and not _any_outcome:
                continue
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

    # Post-generation leakage assertion: target_column must never be a predictor.
    if target_column:
        for cand in candidates:
            payload_predictor = cand.get("predictor")
            payload_predictors = cand.get("predictors", [])
            if payload_predictor == target_column or target_column in payload_predictors:
                raise ValueError(
                    f"Target leakage detected during candidate generation: "
                    f"target column {target_column!r} appears as a predictor in "
                    f"candidate {cand['id']!r}. This is a bug — please report it."
                )

    generation = {
        "strategy": "locked_scout_then_confirmation",
        "seed": seed,
        "scout_fraction": scout_fraction,
        "scout_rows": len(scout),
        "confirmation_rows": len(df) - len(scout),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "goal_columns": goal_columns,
        "target_column": target_column,
        "generated_candidates": len(candidates),
        "candidate_budget": max_candidates,
        "structural_relations": structural_relations,
        "structural_relation_count": len(structural_relations),
    }
    return candidates, generation


def _apply_transform(y: np.ndarray, transform: str | None) -> np.ndarray:
    if transform == "log1p":
        return np.log1p(np.clip(y, 0, None))
    return y


def _invert_transform(y: np.ndarray, transform: str | None) -> np.ndarray:
    if transform == "log1p":
        return np.expm1(y)
    return y


class UploadedTableDomain:
    """Fittable domain for frozen candidates compiled from an uploaded table.

    Partition layout
    ----------------
    The full dataset is split **once** using a deterministic shuffle:

    * ``scout``            (default 60 %) — candidate generation only.
    * ``selection``        (default 25 %) — all model-selection decisions:
                           held-out, cross-seed, ablation, improvement checks.
    * ``final_validation`` (default 15 %) — **never** accessed during candidate
                           generation or model selection.  Used by the service
                           layer to compute the unbiased final reported score.

    The engine and all falsifiers receive ``evidence["confirmation"] = selection``
    through ``evidence_for()``.  ``final_validation`` is exposed as an attribute
    but never injected into the evidence dict so the engine cannot reach it.

    Metric
    ------
    ``evaluation_metric`` governs how ``score_metric()`` evaluates predictions.
    The internal ``score()`` method always uses R² to give the ``GatedJudge``
    and threshold-based falsifiers a direction-consistent fitness signal
    regardless of the external evaluation metric.  ``ImprovementFalsifier``
    explicitly calls ``score_metric()`` for metric-direction-aware comparison.

    Parameters
    ----------
    target_transform:
        Monotone transform applied to numeric outcome columns before fitting and
        scoring (``"log1p"`` supported).  Predictions are returned in transformed
        space by ``score()``; callers are responsible for inverting with
        ``_invert_transform``.
    evaluation_metric:
        Metric used by ``score_metric()`` and the final validation scorer.
        Defaults to ``"r2"`` (backward-compatible).
    confirmation_fraction:
        Fraction of total rows reserved for the *selection* partition.
    final_validation_fraction:
        Fraction reserved for the *final validation* partition.  Must satisfy
        ``scout_fraction + confirmation_fraction + final_validation_fraction <= 1``.
    """

    name = "uploaded_table"

    def __init__(
        self,
        dataframe: pd.DataFrame,
        candidates: list[dict[str, Any]],
        *,
        scout_fraction: float = 0.6,
        confirmation_fraction: float = 0.25,
        final_validation_fraction: float = 0.15,
        seed: int = 20260623,
        target_transform: str | None = None,
        evaluation_metric: str = "r2",
    ):
        if not candidates:
            raise ValueError("The approved plan contains no testable candidates")
        validate_metric(evaluation_metric)
        total = scout_fraction + confirmation_fraction + final_validation_fraction
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"scout_fraction ({scout_fraction}) + confirmation_fraction "
                f"({confirmation_fraction}) + final_validation_fraction "
                f"({final_validation_fraction}) = {total:.3f} > 1.0"
            )
        self.df = dataframe.reset_index(drop=True)
        self.specs = candidates
        self.target_transform = target_transform
        self.evaluation_metric = evaluation_metric

        n = len(self.df)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        scout_cut = max(3, min(n - 3, int(n * scout_fraction)))
        sel_cut = scout_cut + max(1, int(n * confirmation_fraction))
        # Final validation: whatever remains after scout + selection.
        # Clamp so we always have at least 1 row in final_validation (if n allows).
        sel_cut = min(sel_cut, n - 1) if n > scout_cut + 1 else scout_cut + 1

        self.scout = self.df.iloc[indices[:scout_cut]].copy()
        self.selection = self.df.iloc[indices[scout_cut:sel_cut]].copy()
        self.final_validation = self.df.iloc[indices[sel_cut:]].copy()

        # Legacy alias used by tests that directly access .confirmation
        self.confirmation = self.selection

    def propose(self):
        for spec in self.specs:
            yield Candidate(
                id=str(spec["id"]),
                statement=str(spec["statement"]),
                payload={k: v for k, v in spec.items() if k not in {"id", "statement", "parents"}},
                parents=list(spec.get("parents", [])),
            )

    def evidence_for(self, c: Candidate) -> Any:
        # Expose scout and selection to the engine and falsifiers.
        # final_validation is intentionally NOT included here.
        return {"scout": self.scout, "confirmation": self.selection}

    def splits(self, evidence: Any, seed: int):
        """Fixed-model validation-resample splits.

        The training partition is ALWAYS the locked scout rows regardless of
        seed; only the validation rows are bootstrap-resampled. This is what
        ``validation_resample`` (formerly ``cross_seed``) uses. It intentionally
        does NOT repartition/refit — see :meth:`repeated_refit_split`.
        """
        train = evidence["scout"]
        confirmation = evidence["confirmation"]
        if seed in {0, 1} or len(confirmation) < 4:
            return train, confirmation
        rng = np.random.default_rng(seed)
        picks = rng.integers(0, len(confirmation), size=len(confirmation))
        return train, confirmation.iloc[picks].copy()

    def repeated_refit_split(self, seed: int, train_fraction: float = 0.7):
        """Fresh train/validation partition for genuine repeated refitting.

        Draws from the modelling pool (scout + selection) — never the reserved
        ``final_validation`` partition — and produces a new random split per
        seed so the candidate model is refit on independent training rows each
        time. Used by :class:`RepeatedRefitValidator`.
        """
        frames = [self.scout]
        if len(self.selection):
            frames.append(self.selection)
        pool = pd.concat(frames, ignore_index=True) if len(frames) > 1 else self.scout.reset_index(drop=True)
        n = len(pool)
        if n < 6:
            return pool.copy(), pool.copy()
        idx = np.arange(n)
        np.random.default_rng(1_000_003 + seed).shuffle(idx)
        cut = max(3, min(n - 3, int(n * train_fraction)))
        train = pool.iloc[idx[:cut]].copy()
        val = pool.iloc[idx[cut:]].copy()
        return train, val

    def _get_y(self, df: pd.DataFrame, col: str) -> np.ndarray:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        return _apply_transform(y, self.target_transform)

    def refit(self, c: Candidate, train: pd.DataFrame) -> dict[str, Any]:
        kind = c.payload["kind"]
        if kind == "linear_association":
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            x_s = pd.to_numeric(train[x_name], errors="coerce")
            y_raw = pd.to_numeric(train[y_name], errors="coerce")
            mask = x_s.notna() & y_raw.notna()
            if mask.sum() < 3:
                return {"kind": kind, "valid": False}
            x = x_s[mask].to_numpy(float)
            y = _apply_transform(y_raw[mask].to_numpy(float), self.target_transform)
            X = np.column_stack([np.ones(len(x)), x])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            return {
                "kind": kind, "valid": True,
                "intercept": float(beta[0]), "slope": float(beta[1]),
                "target_transform": self.target_transform,
            }
        if kind == "composite_linear":
            predictors = c.payload["predictors"]
            y_name = c.payload["outcome"]
            cols = predictors + [y_name]
            sub = train[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < len(predictors) + 2:
                return {"kind": kind, "valid": False}
            X = np.column_stack([np.ones(len(sub))] + [sub[p].to_numpy(float) for p in predictors])
            y = _apply_transform(sub[y_name].to_numpy(float), self.target_transform)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            # R² ablation contributions — report-only diagnostic.
            # AblationFalsifier uses its own metric-aware, selection-partition logic.
            ablation: dict[str, float] = {}
            full_pred = X @ beta
            full_r2 = _r2(y, full_pred)
            for i, p in enumerate(predictors):
                drop_cols = [j for j in range(len(predictors)) if j != i]
                X_drop = np.column_stack([np.ones(len(sub))] + [sub[predictors[j]].to_numpy(float) for j in drop_cols])
                b_drop, *_ = np.linalg.lstsq(X_drop, y, rcond=None)
                drop_r2 = _r2(y, X_drop @ b_drop)
                ablation[p] = round(full_r2 - drop_r2, 6)
            return {
                "kind": kind, "valid": True,
                "intercept": float(beta[0]),
                "coefficients": {p: float(beta[i + 1]) for i, p in enumerate(predictors)},
                "predictors": predictors,
                "ablation_contributions_r2_diagnostic": ablation,
                "target_transform": self.target_transform,
            }
        if kind == "group_difference":
            group, outcome = c.payload["group"], c.payload["outcome"]
            temp = pd.DataFrame({"g": train[group].astype(str), "y": pd.to_numeric(train[outcome], errors="coerce")}).dropna()
            means = {str(label): float(part["y"].mean()) for label, part in temp.groupby("g")}
            return {"kind": kind, "valid": bool(means), "means": means, "overall": float(temp["y"].mean()) if len(temp) else 0.0}
        if kind == "nonlinear_association":
            form = c.payload["form"]
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            x_s = pd.to_numeric(train[x_name], errors="coerce")
            y_s = pd.to_numeric(train[y_name], errors="coerce")
            mask = x_s.notna() & y_s.notna()
            if mask.sum() < 5:
                return {"kind": kind, "valid": False}
            fit = _fit_form(x_s[mask].to_numpy(float), y_s[mask].to_numpy(float), form)
            if fit is None:
                return {"kind": kind, "valid": False}
            params = fit["params"]
            return {
                "kind": kind, "valid": True, "form": form,
                "intercept": float(params.get("intercept", 0.0)),
                "params": params,
            }
        return {"kind": kind, "valid": False}

    def _predict_raw(self, c: Candidate, model: dict[str, Any], test: pd.DataFrame) -> np.ndarray | None:
        """Return predictions in *transformed* space (pre-invert), or None on failure."""
        kind = model["kind"]
        if kind == "linear_association":
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            x_s = pd.to_numeric(test[x_name], errors="coerce")
            y_raw = pd.to_numeric(test[y_name], errors="coerce")
            mask = x_s.notna() & y_raw.notna()
            if mask.sum() < 3:
                return None
            x = x_s[mask].to_numpy(float)
            slope = float(model["slope"])
            expected = c.payload.get("expected_direction")
            if expected == "positive" and slope <= 0:
                return None
            if expected == "negative" and slope >= 0:
                return None
            pred = model["intercept"] + slope * x
            y_t = _apply_transform(y_raw[mask].to_numpy(float), self.target_transform)
            return np.stack([y_t, pred])  # [true, pred] for downstream

        if kind == "composite_linear":
            predictors = model["predictors"]
            y_name = c.payload["outcome"]
            cols = predictors + [y_name]
            sub = test[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                return None
            X = np.column_stack([np.ones(len(sub))] + [sub[p].to_numpy(float) for p in predictors])
            y = _apply_transform(sub[y_name].to_numpy(float), self.target_transform)
            beta = np.array([model["intercept"]] + [model["coefficients"][p] for p in predictors])
            pred = X @ beta
            return np.stack([y, pred])

        if kind == "group_difference":
            group, outcome = c.payload["group"], c.payload["outcome"]
            temp = pd.DataFrame({"g": test[group].astype(str), "y": pd.to_numeric(test[outcome], errors="coerce")}).dropna()
            if len(temp) < 3:
                return None
            pred = np.array([model["means"].get(label, model["overall"]) for label in temp["g"]], dtype=float)
            return np.stack([temp["y"].to_numpy(float), pred])

        if kind == "nonlinear_association":
            x_name, y_name = c.payload["predictor"], c.payload["outcome"]
            x_s = pd.to_numeric(test[x_name], errors="coerce")
            y_s = pd.to_numeric(test[y_name], errors="coerce")
            mask = x_s.notna() & y_s.notna()
            if mask.sum() < 3:
                return None
            pred = _predict_form(x_s[mask].to_numpy(float), model["form"], model["params"])
            if pred is None:
                return None
            y_true = y_s[mask].to_numpy(float)
            # Nonlinear forms predict in ORIGINAL outcome units already; return
            # them directly (score() computes R² in that comparable space).
            valid = np.isfinite(pred) & np.isfinite(y_true)
            if valid.sum() < 3:
                return None
            return np.stack([y_true[valid], pred[valid]])
        return None

    def score(self, c: Candidate, model: dict[str, Any], test: pd.DataFrame) -> float:
        """R² score on *test* data (always R², used by GatedJudge and threshold falsifiers)."""
        if not model.get("valid") or len(test) < 3:
            return 0.0
        data = self._predict_raw(c, model, test)
        if data is None:
            return 0.0
        return _r2(data[0], data[1])

    def score_metric(self, c: Candidate, model: dict[str, Any], test: pd.DataFrame) -> float:
        """Score under ``self.evaluation_metric``.

        For ``"r2"`` this is identical to ``score()``.  For error metrics
        (rmsle, rmse, mae) predictions are inverted out of transform space
        before evaluation so the result is in the original target units.
        """
        if not model.get("valid") or len(test) < 3:
            from .metrics import NULL_SCORE
            return NULL_SCORE.get(self.evaluation_metric, float("nan"))
        data = self._predict_raw(c, model, test)
        if data is None:
            from .metrics import NULL_SCORE
            return NULL_SCORE.get(self.evaluation_metric, float("nan"))
        y_t_tf, pred_tf = data[0], data[1]
        if self.evaluation_metric == "r2":
            return _r2(y_t_tf, pred_tf)
        # For error metrics: invert transform so we compare original-scale values
        y_orig = _invert_transform(y_t_tf, self.target_transform)
        pred_orig = _invert_transform(pred_tf, self.target_transform)
        return compute_metric(self.evaluation_metric, y_orig, pred_orig)

    def baseline_score(self, test: Any) -> float:
        return 0.0
