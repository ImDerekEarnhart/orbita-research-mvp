from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from urllib.parse import unquote, urlparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .execution import (
    ContainerExecutionRuntime,
    ContainerExecutionSpec,
    ExecutionClaimTest,
    OutputObligation,
    ResourceLimits,
    StagedFile,
)
from .models import ActorRole, ObjectKind, SupportState

DISCOVERY_API_VERSION = "1.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _new_id(prefix: str) -> str:
    from .ledger import new_id

    return new_id(prefix)


@dataclass(frozen=True, slots=True)
class DiscoveryBudget:
    max_hypotheses: int = 12
    max_container_runs: int = 2
    max_human_approvals: int = 2
    max_wall_seconds_per_run: int = 120
    max_output_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if not 1 <= self.max_hypotheses <= 100:
            raise ValueError("max_hypotheses must be between 1 and 100")
        if not 1 <= self.max_container_runs <= 8:
            raise ValueError("max_container_runs must be between 1 and 8")
        if not 1 <= self.max_human_approvals <= 8:
            raise ValueError("max_human_approvals must be between 1 and 8")
        if not 5 <= self.max_wall_seconds_per_run <= 3600:
            raise ValueError("max_wall_seconds_per_run must be between 5 and 3600")
        if not 1024 <= self.max_output_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_output_bytes is outside the supported range")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DiscoveryBudget":
        return cls(**(value or {}))


@dataclass(frozen=True, slots=True)
class HypothesisSeed:
    x: str
    y: str
    direction: str
    claim_id: str | None = None
    rationale: str = ""
    origin: str = "human"

    def __post_init__(self) -> None:
        if self.direction not in {"positive", "negative"}:
            raise ValueError("direction must be positive or negative")
        if not self.x or not self.y or self.x == self.y:
            raise ValueError("Hypothesis seed requires two different columns")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HypothesisSeed":
        return cls(
            x=str(value["x"]),
            y=str(value["y"]),
            direction=str(value["direction"]),
            claim_id=value.get("claim_id"),
            rationale=str(value.get("rationale", "")),
            origin=str(value.get("origin", "human")),
        )


@dataclass(frozen=True, slots=True)
class DiscoverySpec:
    question: str
    dataset_path: str | Path
    image: str
    replication_dataset_path: str | Path | None = None
    candidate_hypotheses: tuple[HypothesisSeed, ...] = ()
    seed: int = 20260619
    discovery_fraction: float = 0.5
    min_rows: int = 12
    numeric_fraction: float = 0.8
    min_discovery_abs_r: float = 0.45
    min_confirmation_abs_r: float = 0.35
    robust_min_abs_r: float = 0.25
    null_abs_r: float = 0.10
    refute_abs_r: float = 0.35
    alpha: float = 0.05
    permutation_trials: int = 199
    bootstrap_trials: int = 200
    min_bootstrap_sign_stability: float = 0.80
    trim_fraction: float = 0.10
    budget: DiscoveryBudget = field(default_factory=DiscoveryBudget)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question is required")
        if not 0.2 <= self.discovery_fraction <= 0.8:
            raise ValueError("discovery_fraction must be between 0.2 and 0.8")
        if self.min_rows < 6:
            raise ValueError("min_rows must be at least 6")
        if not 0.5 <= self.numeric_fraction <= 1.0:
            raise ValueError("numeric_fraction must be between 0.5 and 1.0")
        for name in (
            "min_discovery_abs_r",
            "min_confirmation_abs_r",
            "robust_min_abs_r",
            "null_abs_r",
            "refute_abs_r",
            "min_bootstrap_sign_stability",
            "trim_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 19 <= self.permutation_trials <= 10000:
            raise ValueError("permutation_trials must be between 19 and 10000")
        if not 20 <= self.bootstrap_trials <= 10000:
            raise ValueError("bootstrap_trials must be between 20 and 10000")
        if not isinstance(self.budget, DiscoveryBudget):
            object.__setattr__(self, "budget", DiscoveryBudget.from_dict(self.budget))
        seeds = tuple(
            item if isinstance(item, HypothesisSeed) else HypothesisSeed.from_dict(item)
            for item in self.candidate_hypotheses
        )
        object.__setattr__(self, "candidate_hypotheses", seeds)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, base_dir: str | Path | None = None) -> "DiscoverySpec":
        base = Path(base_dir).resolve() if base_dir is not None else None

        def resolve(path_value: Any) -> Any:
            if path_value in {None, ""}:
                return None
            path = Path(str(path_value)).expanduser()
            if not path.is_absolute() and base is not None:
                path = base / path
            return path.resolve()

        return cls(
            question=str(value["question"]),
            dataset_path=resolve(value["dataset_path"]),
            image=str(value["image"]),
            replication_dataset_path=resolve(value.get("replication_dataset_path")),
            candidate_hypotheses=tuple(
                HypothesisSeed.from_dict(item) for item in value.get("candidate_hypotheses", [])
            ),
            seed=int(value.get("seed", 20260619)),
            discovery_fraction=float(value.get("discovery_fraction", 0.5)),
            min_rows=int(value.get("min_rows", 12)),
            numeric_fraction=float(value.get("numeric_fraction", 0.8)),
            min_discovery_abs_r=float(value.get("min_discovery_abs_r", 0.45)),
            min_confirmation_abs_r=float(value.get("min_confirmation_abs_r", 0.35)),
            robust_min_abs_r=float(value.get("robust_min_abs_r", 0.25)),
            null_abs_r=float(value.get("null_abs_r", 0.10)),
            refute_abs_r=float(value.get("refute_abs_r", 0.35)),
            alpha=float(value.get("alpha", 0.05)),
            permutation_trials=int(value.get("permutation_trials", 199)),
            bootstrap_trials=int(value.get("bootstrap_trials", 200)),
            min_bootstrap_sign_stability=float(value.get("min_bootstrap_sign_stability", 0.80)),
            trim_fraction=float(value.get("trim_fraction", 0.10)),
            budget=DiscoveryBudget.from_dict(value.get("budget")),
            metadata=dict(value.get("metadata", {})),
        )


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise DiscoveryError(f"Expected file URI, got: {uri}")
    path = unquote(parsed.path)
    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
        path = f"//{parsed.netloc}{path}"
    return Path(path)


class DiscoveryError(RuntimeError):
    pass


class GovernedDiscoveryRuntime:
    """Restart-safe, counterexample-first research orchestration.

    Candidate mining uses only a discovery split and creates no warrant. Exact
    hypothesis tests are pre-registered, bound into a container manifest, and
    executed only after human approval. A separate replication dataset, when
    supplied, receives a second manifest and independent evidence identity.
    """

    def __init__(self, ledger: "EpistemicLedger", workspace: str | Path | None = None):
        self.ledger = ledger
        self.workspace = (
            Path(workspace).expanduser().resolve()
            if workspace is not None
            else ledger.db.path.parent / "discovery_workspace"
        )
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.executions = ContainerExecutionRuntime(ledger, self.workspace / "executions")

    # ------------------------------------------------------------------
    # Campaign lifecycle
    # ------------------------------------------------------------------
    def create(
        self,
        spec: DiscoverySpec | dict[str, Any],
        *,
        actor: str = "user",
        actor_role: ActorRole = ActorRole.HUMAN,
    ) -> dict[str, Any]:
        if isinstance(spec, dict):
            spec = DiscoverySpec.from_dict(spec)
        dataset_path = Path(spec.dataset_path).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        replication_path = (
            Path(spec.replication_dataset_path).expanduser().resolve()
            if spec.replication_dataset_path is not None
            else None
        )
        if replication_path is not None and not replication_path.is_file():
            raise FileNotFoundError(replication_path)

        investigation_id = _new_id("inv")
        dataset_hash, dataset_size = _sha256_file(dataset_path)
        replication_hash = None
        replication_size = None
        if replication_path is not None:
            replication_hash, replication_size = _sha256_file(replication_path)
        rows, fieldnames = self._load_csv(dataset_path)
        profile = self._profile(rows, fieldnames, spec)
        discovery_indices, holdout_indices = self._split_indices(len(rows), spec)
        candidates = self._candidate_hypotheses(
            rows,
            fieldnames,
            discovery_indices,
            spec,
            investigation_id,
            dataset_hash,
            actor,
            actor_role,
        )
        now = _utcnow()
        status = "no_candidates" if not candidates else "preparing_confirmation"
        stored_spec = self._serialize_spec(spec, dataset_path, replication_path)
        budget = asdict(spec.budget)
        budget_used = {
            "hypotheses": len(candidates),
            "container_runs_prepared": 0,
            "human_approvals_consumed": 0,
        }
        cursor = {
            "phase": "candidate_generation",
            "confirmation_run_id": None,
            "replication_run_id": None,
            "discovery_indices": discovery_indices,
            "holdout_indices": holdout_indices,
        }
        self.ledger.db.conn.execute(
            """INSERT INTO discovery_investigations
               (id, question, status, current_phase, dataset_uri, dataset_hash,
                dataset_size_bytes, replication_dataset_uri, replication_dataset_hash,
                replication_dataset_size_bytes, spec_json, budget_json, budget_used_json,
                profile_json, resume_cursor_json, report_json, report_hash, created_at,
                updated_at, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', NULL, ?, ?, NULL)""",
            (
                investigation_id,
                spec.question.strip(),
                status,
                "candidate_generation",
                dataset_path.as_uri(),
                dataset_hash,
                dataset_size,
                replication_path.as_uri() if replication_path else None,
                replication_hash,
                replication_size,
                _stable_json(stored_spec),
                _stable_json(budget),
                _stable_json(budget_used),
                _stable_json(profile),
                _stable_json(cursor),
                now,
                now,
            ),
        )
        for position, candidate in enumerate(candidates):
            self.ledger.db.conn.execute(
                """INSERT INTO discovery_hypotheses
                   (id, investigation_id, position, claim_id, x_column, y_column,
                    direction, origin, rationale, discovery_metrics_json,
                    preregistration_json, status, confirmation_run_id,
                    replication_run_id, confirmation_result_json,
                    replication_result_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', NULL, NULL, '{}', '{}', ?, ?)""",
                (
                    candidate["id"],
                    investigation_id,
                    position,
                    candidate["claim_id"],
                    candidate["x"],
                    candidate["y"],
                    candidate["direction"],
                    candidate["origin"],
                    candidate["rationale"],
                    _stable_json(candidate["discovery_metrics"]),
                    _stable_json(candidate["preregistration"]),
                    now,
                    now,
                ),
            )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            "DISCOVERY_INVESTIGATION_CREATED",
            {
                "question": spec.question,
                "dataset_hash": dataset_hash,
                "replication_dataset_hash": replication_hash,
                "candidate_count": len(candidates),
                "budget": budget,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        if candidates:
            self._prepare_phase_run(investigation_id, phase="confirmation", actor=actor, actor_role=actor_role)
        else:
            self.compile_report(investigation_id)
        return self.get(investigation_id)

    def advance(
        self,
        investigation_id: str,
        *,
        engine=None,
        actor: str = "discovery_runtime",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> dict[str, Any]:
        inv = self.get(investigation_id)
        if inv["status"] in {"concluded", "failed", "budget_exhausted", "no_candidates"}:
            return inv

        cursor = inv["resume_cursor"]
        phase = inv["current_phase"]
        if phase == "confirmation":
            run_id = cursor.get("confirmation_run_id")
            if not run_id:
                raise DiscoveryError("Confirmation phase has no execution run")
            run = self.executions.get(run_id)
            if run["status"] == "waiting_approval":
                self._set_status(investigation_id, "awaiting_confirmation_approval", phase)
                return self.get(investigation_id)
            if run["status"] == "approved":
                # Current execution API retains waiting_approval after approval; this
                # branch exists for forwards compatibility.
                run = self.executions.execute(run_id, engine=engine)
            elif run["status"] == "running":
                self._set_status(investigation_id, "running_confirmation", phase)
                return self.get(investigation_id)
            elif run["status"] not in {"succeeded", "failed"}:
                approval = run.get("approval") or {}
                if approval.get("status") == "approved" and approval.get("consumed_at") is None:
                    run = self.executions.execute(run_id, engine=engine)
                else:
                    self._set_status(investigation_id, "awaiting_confirmation_approval", phase)
                    return self.get(investigation_id)
            if run["status"] == "failed":
                self._fail(investigation_id, f"Confirmation execution failed: {run.get('error')}")
                self.compile_report(investigation_id)
                return self.get(investigation_id)
            self._consume_phase_result(investigation_id, "confirmation", run)
            inv = self.get(investigation_id)
            survivors = [h for h in inv["hypotheses"] if h["status"] == "confirmed_holdout"]
            if survivors and inv["replication_dataset_uri"]:
                if inv["budget_used"]["container_runs_prepared"] >= inv["budget"]["max_container_runs"]:
                    self._set_status(investigation_id, "budget_exhausted", "replication")
                    self.compile_report(investigation_id)
                    return self.get(investigation_id)
                self._prepare_phase_run(investigation_id, phase="replication", actor=actor, actor_role=actor_role)
                return self.get(investigation_id)
            self._conclude(investigation_id)
            return self.get(investigation_id)

        if phase == "replication":
            run_id = cursor.get("replication_run_id")
            if not run_id:
                raise DiscoveryError("Replication phase has no execution run")
            run = self.executions.get(run_id)
            approval = run.get("approval") or {}
            if run["status"] == "rejected":
                self._fail(investigation_id, "Replication execution manifest was rejected")
                self.compile_report(investigation_id)
                return self.get(investigation_id)
            if run["status"] == "waiting_approval" and not (
                approval.get("status") == "approved" and approval.get("consumed_at") is None
            ):
                self._set_status(investigation_id, "awaiting_replication_approval", phase)
                return self.get(investigation_id)
            if run["status"] not in {"succeeded", "failed"}:
                run = self.executions.execute(run_id, engine=engine)
            if run["status"] == "failed":
                self._fail(investigation_id, f"Replication execution failed: {run.get('error')}")
                self.compile_report(investigation_id)
                return self.get(investigation_id)
            self._consume_phase_result(investigation_id, "replication", run)
            self._conclude(investigation_id)
            return self.get(investigation_id)

        raise DiscoveryError(f"Unknown current phase: {phase}")

    def get(self, investigation_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM discovery_investigations WHERE id = ?", (investigation_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown discovery investigation: {investigation_id}")
        result = dict(row)
        for key in (
            "spec_json",
            "budget_json",
            "budget_used_json",
            "profile_json",
            "resume_cursor_json",
            "report_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "{}")
        result["hypotheses"] = [
            self._hypothesis_row(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM discovery_hypotheses WHERE investigation_id = ? ORDER BY position",
                (investigation_id,),
            ).fetchall()
        ]
        result["artifacts"] = [
            dict(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM discovery_artifacts WHERE investigation_id = ? ORDER BY role",
                (investigation_id,),
            ).fetchall()
        ]
        result["report_integrity_valid"] = self.verify_report(investigation_id)
        return result

    def list(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM discovery_investigations ORDER BY created_at"
        ).fetchall()
        return [self.get(row["id"]) for row in rows]

    # ------------------------------------------------------------------
    # Report compiler
    # ------------------------------------------------------------------
    def compile_report(self, investigation_id: str) -> dict[str, Any]:
        inv = self.get(investigation_id)
        from .support import SupportEngine

        support = SupportEngine(self.ledger)
        hypotheses = []
        for item in inv["hypotheses"]:
            report = support.evaluate(item["claim_id"]).as_dict()
            hypotheses.append(
                {
                    "hypothesis_id": item["id"],
                    "claim_id": item["claim_id"],
                    "claim": self.ledger.get_claim(item["claim_id"])["canonical_text"],
                    "direction": item["direction"],
                    "origin": item["origin"],
                    "status": item["status"],
                    "discovery_metrics": item["discovery_metrics"],
                    "preregistration": item["preregistration"],
                    "confirmation": item["confirmation_result"],
                    "replication": item["replication_result"],
                    "current_support": report,
                }
            )
        summary = {
            "candidates": len(hypotheses),
            "confirmed_holdout": sum(h["status"] == "confirmed_holdout" for h in hypotheses),
            "replicated": sum(h["status"] == "replicated" for h in hypotheses),
            "falsified": sum("falsified" in h["status"] for h in hypotheses),
            "inconclusive": sum("inconclusive" in h["status"] for h in hypotheses),
        }
        report = {
            "schema_version": DISCOVERY_API_VERSION,
            "investigation_id": investigation_id,
            "question": inv["question"],
            "status": inv["status"],
            "dataset": {
                "uri": inv["dataset_uri"],
                "sha256": inv["dataset_hash"],
                "size_bytes": inv["dataset_size_bytes"],
            },
            "replication_dataset": (
                {
                    "uri": inv["replication_dataset_uri"],
                    "sha256": inv["replication_dataset_hash"],
                    "size_bytes": inv["replication_dataset_size_bytes"],
                }
                if inv["replication_dataset_uri"]
                else None
            ),
            "design": {
                "discovery_fraction": inv["spec"]["discovery_fraction"],
                "discovery_is_non_warranting": True,
                "confirmation_is_holdout": True,
                "multiple_testing": "Bonferroni across preregistered candidates",
                "falsification_gates": [
                    "effect direction and minimum magnitude",
                    "outlier-trimmed robustness",
                    "permutation p-value",
                    "bootstrap sign stability",
                ],
                "independent_replication_required_for_replicated_label": True,
            },
            "budget": inv["budget"],
            "budget_used": inv["budget_used"],
            "profile": inv["profile"],
            "summary": summary,
            "hypotheses": hypotheses,
            "limitations": [
                "Association claims are not causal claims.",
                "Internal holdout confirmation is not independent replication.",
                "Automated thresholds are preregistered policy choices, not universal scientific laws.",
                "Human review remains required before durable commitment or consequential action.",
            ],
            "compiled_at": _utcnow(),
        }
        payload = {k: v for k, v in report.items() if k != "compiled_at"}
        report_hash = _sha256_bytes(_stable_json(payload).encode("utf-8"))
        report["report_hash"] = report_hash
        root = self.workspace / "reports" / investigation_id
        root.mkdir(parents=True, exist_ok=True)
        json_path = root / "report.json"
        md_path = root / "report.md"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(self._markdown_report(report), encoding="utf-8")
        self.ledger.db.conn.execute(
            "UPDATE discovery_investigations SET report_json = ?, report_hash = ?, updated_at = ? WHERE id = ?",
            (_stable_json(report), report_hash, _utcnow(), investigation_id),
        )
        self.ledger.db.conn.execute(
            "DELETE FROM discovery_artifacts WHERE investigation_id = ?", (investigation_id,)
        )
        for role, path, media in (
            ("report_json", json_path, "application/json"),
            ("report_markdown", md_path, "text/markdown"),
        ):
            digest, size = _sha256_file(path)
            self.ledger.db.conn.execute(
                """INSERT INTO discovery_artifacts
                   (id, investigation_id, role, path, content_hash, size_bytes, media_type, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (_new_id("dar"), investigation_id, role, str(path), digest, size, media, _utcnow()),
            )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            "DISCOVERY_REPORT_COMPILED",
            {"report_hash": report_hash, "summary": summary},
            "discovery_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get(investigation_id)["report"]

    def verify_report(self, investigation_id: str) -> bool | None:
        row = self.ledger.db.conn.execute(
            "SELECT report_json, report_hash FROM discovery_investigations WHERE id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise KeyError(investigation_id)
        if not row["report_hash"]:
            return None
        report = json.loads(row["report_json"])
        stored = report.pop("report_hash", None)
        report.pop("compiled_at", None)
        expected = _sha256_bytes(_stable_json(report).encode("utf-8"))
        if stored != row["report_hash"] or expected != row["report_hash"]:
            return False
        artifacts = self.ledger.db.conn.execute(
            "SELECT * FROM discovery_artifacts WHERE investigation_id = ?", (investigation_id,)
        ).fetchall()
        for item in artifacts:
            path = Path(item["path"])
            if not path.is_file():
                return False
            digest, size = _sha256_file(path)
            if digest != item["content_hash"] or size != item["size_bytes"]:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal orchestration
    # ------------------------------------------------------------------
    def _prepare_phase_run(
        self,
        investigation_id: str,
        *,
        phase: str,
        actor: str,
        actor_role: ActorRole,
    ) -> None:
        inv = self.get(investigation_id)
        if phase not in {"confirmation", "replication"}:
            raise ValueError(phase)
        candidates = inv["hypotheses"]
        if phase == "replication":
            candidates = [item for item in candidates if item["status"] == "confirmed_holdout"]
        if not candidates:
            self._conclude(investigation_id)
            return
        if inv["budget_used"]["container_runs_prepared"] >= inv["budget"]["max_container_runs"]:
            self._set_status(investigation_id, "budget_exhausted", phase)
            self.compile_report(investigation_id)
            return
        spec = inv["spec"]
        if phase == "confirmation":
            source_path = _path_from_file_uri(inv["dataset_uri"])
            row_indices = inv["resume_cursor"]["holdout_indices"]
        else:
            source_path = _path_from_file_uri(inv["replication_dataset_uri"])
            rows, _ = self._load_csv(source_path)
            row_indices = list(range(len(rows)))
        adjusted_alpha = float(spec["alpha"]) / max(1, len(candidates))
        plan = {
            "schema_version": DISCOVERY_API_VERSION,
            "investigation_id": investigation_id,
            "phase": phase,
            "seed": int(spec["seed"]) + (100000 if phase == "replication" else 0),
            "row_indices": row_indices,
            "min_rows": int(spec["min_rows"]),
            "min_confirmation_abs_r": float(spec["min_confirmation_abs_r"]),
            "robust_min_abs_r": float(spec["robust_min_abs_r"]),
            "null_abs_r": float(spec["null_abs_r"]),
            "refute_abs_r": float(spec["refute_abs_r"]),
            "adjusted_alpha": adjusted_alpha,
            "permutation_trials": int(spec["permutation_trials"]),
            "bootstrap_trials": int(spec["bootstrap_trials"]),
            "min_bootstrap_sign_stability": float(spec["min_bootstrap_sign_stability"]),
            "trim_fraction": float(spec["trim_fraction"]),
            "hypotheses": [
                {
                    "id": item["id"],
                    "claim_id": item["claim_id"],
                    "x": item["x_column"],
                    "y": item["y_column"],
                    "direction": item["direction"],
                }
                for item in candidates
            ],
        }
        plan_text = json.dumps(plan, indent=2, ensure_ascii=False)
        output_schema = {
            "type": "object",
            "required": ["schema_version", "investigation_id", "phase", "hypotheses"],
            "properties": {
                "schema_version": {"type": "string"},
                "investigation_id": {"type": "string"},
                "phase": {"type": "string"},
                "hypotheses": {"type": "object"},
            },
            "additionalProperties": True,
        }
        claim_tests = tuple(
            ExecutionClaimTest(
                claim_id=item["claim_id"],
                output_path="result.json",
                metric_path=f"hypotheses.{item['id']}.decision_code",
                support_condition={"operator": ">=", "value": 1},
                refute_condition={"operator": "<=", "value": -1},
                confidence=1.0,
                rationale=(
                    f"Pre-registered {phase} gate: direction, effect size, robust trim, "
                    "permutation significance, and bootstrap stability"
                ),
            )
            for item in candidates
        )
        run_spec = ContainerExecutionSpec(
            name=f"Orbita discovery {phase}: {inv['question'][:80]}",
            image=str(spec["image"]),
            command=("python", "/workspace/code/discovery_worker.py"),
            code_files=(
                StagedFile(
                    "discovery_worker.py",
                    text=self.worker_source(),
                    media_type="text/x-python",
                ),
            ),
            input_files=(
                StagedFile("dataset.csv", source=source_path, media_type="text/csv"),
                StagedFile("plan.json", text=plan_text, media_type="application/json"),
            ),
            outputs=(
                OutputObligation(
                    "result.json",
                    media_type="application/json",
                    max_bytes=int(inv["budget"]["max_output_bytes"]),
                    json_schema=output_schema,
                ),
            ),
            limits=ResourceLimits(
                timeout_seconds=int(inv["budget"]["max_wall_seconds_per_run"]),
                memory_mb=256,
                cpus=1.0,
                pids=32,
                tmpfs_mb=64,
                stdout_bytes=128 * 1024,
                stderr_bytes=128 * 1024,
            ),
            claim_tests=claim_tests,
            metadata={
                "discovery_investigation_id": investigation_id,
                "phase": phase,
                "counterexample_first": True,
                "discovery_split_non_warranting": True,
                "adjusted_alpha": adjusted_alpha,
            },
        )
        run = self.executions.submit(run_spec, actor=actor, actor_role=actor_role)
        cursor = inv["resume_cursor"]
        cursor[f"{phase}_run_id"] = run["id"]
        budget_used = inv["budget_used"]
        budget_used["container_runs_prepared"] += 1
        now = _utcnow()
        self.ledger.db.conn.execute(
            """UPDATE discovery_investigations
               SET status = ?, current_phase = ?, resume_cursor_json = ?,
                   budget_used_json = ?, updated_at = ? WHERE id = ?""",
            (
                f"awaiting_{phase}_approval",
                phase,
                _stable_json(cursor),
                _stable_json(budget_used),
                now,
                investigation_id,
            ),
        )
        self.ledger.db.conn.execute(
            f"UPDATE discovery_hypotheses SET {phase}_run_id = ?, updated_at = ? WHERE investigation_id = ? AND id IN ({','.join('?' for _ in candidates)})",
            (run["id"], now, investigation_id, *[item["id"] for item in candidates]),
        )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            f"DISCOVERY_{phase.upper()}_MANIFEST_STAGED",
            {
                "run_id": run["id"],
                "manifest_hash": run["manifest_hash"],
                "hypothesis_count": len(candidates),
                "adjusted_alpha": adjusted_alpha,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()

    def _consume_phase_result(self, investigation_id: str, phase: str, run: dict[str, Any]) -> None:
        if run["status"] != "succeeded":
            raise DiscoveryError("Cannot consume a non-successful execution")
        if run["receipt_integrity_valid"] is not True or run["artifact_integrity_valid"] is not True:
            raise DiscoveryError("Execution receipt or artifacts failed integrity verification")
        artifact = next(
            (item for item in run["artifacts"] if item["role"] == "output" and item["relative_path"] == "result.json"),
            None,
        )
        if artifact is None:
            raise DiscoveryError("Verified execution is missing result.json artifact")
        output = json.loads(Path(artifact["path"]).read_text(encoding="utf-8"))
        results = output.get("hypotheses", {})
        now = _utcnow()
        for item in self.get(investigation_id)["hypotheses"]:
            if item["id"] not in results:
                continue
            result = results[item["id"]]
            decision = result.get("decision")
            if phase == "confirmation":
                status = {
                    "support": "confirmed_holdout",
                    "refute": "falsified_confirmation",
                    "inconclusive": "inconclusive_confirmation",
                }.get(decision, "inconclusive_confirmation")
            else:
                status = {
                    "support": "replicated",
                    "refute": "falsified_replication",
                    "inconclusive": "inconclusive_replication",
                }.get(decision, "inconclusive_replication")
            self.ledger.db.conn.execute(
                f"""UPDATE discovery_hypotheses
                    SET status = ?, {phase}_result_json = ?, updated_at = ? WHERE id = ?""",
                (status, _stable_json(result), now, item["id"]),
            )
        inv = self.get(investigation_id)
        budget_used = inv["budget_used"]
        budget_used["human_approvals_consumed"] = min(
            inv["budget"]["max_human_approvals"],
            budget_used.get("human_approvals_consumed", 0) + 1,
        )
        self.ledger.db.conn.execute(
            "UPDATE discovery_investigations SET budget_used_json = ?, updated_at = ? WHERE id = ?",
            (_stable_json(budget_used), now, investigation_id),
        )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            f"DISCOVERY_{phase.upper()}_RESULT_CONSUMED",
            {
                "run_id": run["id"],
                "receipt_hash": run["receipt_hash"],
                "result_count": len(results),
            },
            "discovery_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()

    def _conclude(self, investigation_id: str) -> None:
        now = _utcnow()
        self.ledger.db.conn.execute(
            """UPDATE discovery_investigations
               SET status = 'concluded', current_phase = 'complete', updated_at = ?, completed_at = ?
               WHERE id = ?""",
            (now, now, investigation_id),
        )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            "DISCOVERY_INVESTIGATION_CONCLUDED",
            {},
            "discovery_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        self.compile_report(investigation_id)

    def _fail(self, investigation_id: str, error: str) -> None:
        inv = self.get(investigation_id)
        cursor = inv["resume_cursor"]
        cursor["error"] = error
        now = _utcnow()
        self.ledger.db.conn.execute(
            """UPDATE discovery_investigations
               SET status = 'failed', resume_cursor_json = ?, updated_at = ?, completed_at = ?
               WHERE id = ?""",
            (_stable_json(cursor), now, now, investigation_id),
        )
        self.ledger._event(
            "discovery_investigation",
            investigation_id,
            "DISCOVERY_INVESTIGATION_FAILED",
            {"error": error},
            "discovery_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()

    def _set_status(self, investigation_id: str, status: str, phase: str) -> None:
        self.ledger.db.conn.execute(
            "UPDATE discovery_investigations SET status = ?, current_phase = ?, updated_at = ? WHERE id = ?",
            (status, phase, _utcnow(), investigation_id),
        )
        self.ledger.db.conn.commit()

    # ------------------------------------------------------------------
    # Candidate mining (non-warranting)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DiscoveryError("CSV has no header")
            rows = [dict(row) for row in reader]
        if not rows:
            raise DiscoveryError("CSV contains no data rows")
        return rows, list(reader.fieldnames)

    def _profile(
        self,
        rows: list[dict[str, str]],
        fieldnames: list[str],
        spec: DiscoverySpec,
    ) -> dict[str, Any]:
        columns = {}
        numeric = []
        for name in fieldnames:
            usable = 0
            numeric_count = 0
            distinct: set[str] = set()
            for row in rows:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    continue
                usable += 1
                distinct.add(raw)
                try:
                    value = float(raw)
                    if math.isfinite(value):
                        numeric_count += 1
                except ValueError:
                    pass
            fraction = numeric_count / usable if usable else 0.0
            is_numeric = numeric_count >= spec.min_rows and fraction >= spec.numeric_fraction
            if is_numeric:
                numeric.append(name)
            columns[name] = {
                "usable": usable,
                "numeric": numeric_count,
                "numeric_fraction": fraction,
                "distinct": len(distinct),
                "classified_as": "numeric" if is_numeric else "non_numeric",
            }
        return {
            "rows": len(rows),
            "columns": len(fieldnames),
            "numeric_columns": numeric,
            "column_profile": columns,
        }

    @staticmethod
    def _split_indices(n_rows: int, spec: DiscoverySpec) -> tuple[list[int], list[int]]:
        if n_rows < spec.min_rows * 2:
            raise DiscoveryError(
                f"Need at least {spec.min_rows * 2} rows for discovery/holdout separation; found {n_rows}"
            )
        indices = list(range(n_rows))
        random.Random(spec.seed).shuffle(indices)
        cut = int(round(n_rows * spec.discovery_fraction))
        cut = max(spec.min_rows, min(n_rows - spec.min_rows, cut))
        return sorted(indices[:cut]), sorted(indices[cut:])

    def _candidate_hypotheses(
        self,
        rows: list[dict[str, str]],
        fieldnames: list[str],
        discovery_indices: list[int],
        spec: DiscoverySpec,
        investigation_id: str,
        dataset_hash: str,
        actor: str,
        actor_role: ActorRole,
    ) -> list[dict[str, Any]]:
        profile = self._profile(rows, fieldnames, spec)
        numeric = profile["numeric_columns"]
        candidates: list[dict[str, Any]] = []
        seeds = list(spec.candidate_hypotheses)
        if seeds:
            pairs = [(seed.x, seed.y, seed.direction, seed) for seed in seeds]
        else:
            pairs = []
            for i, x in enumerate(numeric):
                for y in numeric[i + 1 :]:
                    pair = self._pairs(rows, discovery_indices, x, y)
                    if len(pair) < spec.min_rows:
                        continue
                    r = self._pearson([a for a, _ in pair], [b for _, b in pair])
                    if r is None or abs(r) < spec.min_discovery_abs_r:
                        continue
                    direction = "positive" if r > 0 else "negative"
                    pairs.append((x, y, direction, None))
            pairs.sort(
                key=lambda item: -abs(
                    self._pearson(
                        [a for a, _ in self._pairs(rows, discovery_indices, item[0], item[1])],
                        [b for _, b in self._pairs(rows, discovery_indices, item[0], item[1])],
                    )
                    or 0.0
                )
            )
        seen: set[tuple[str, str, str]] = set()
        for x, y, direction, seed in pairs:
            if x not in fieldnames or y not in fieldnames:
                raise DiscoveryError(f"Unknown candidate columns: {x!r}, {y!r}")
            key = (x, y, direction)
            if key in seen:
                continue
            seen.add(key)
            pair = self._pairs(rows, discovery_indices, x, y)
            if len(pair) < spec.min_rows:
                continue
            xs = [a for a, _ in pair]
            ys = [b for _, b in pair]
            r = self._pearson(xs, ys)
            if r is None:
                continue
            if seed is None and abs(r) < spec.min_discovery_abs_r:
                continue
            expected_direction = direction
            if seed is None:
                expected_direction = "positive" if r > 0 else "negative"
            claim_id = seed.claim_id if seed else None
            predicate = (
                "positively_correlates_with"
                if expected_direction == "positive"
                else "negatively_correlates_with"
            )
            qualifiers = {
                "investigation_id": investigation_id,
                "dataset_sha256": dataset_hash,
                "scope": "same-row association in the registered dataset population",
                "phase": "preregistered_after_discovery_split",
                "non_causal": True,
            }
            if claim_id is None:
                claim_id = self.ledger.add_relation_claim(
                    x,
                    predicate,
                    y,
                    subject_type="measurement",
                    object_kind=ObjectKind.ENTITY,
                    object_type="measurement",
                    qualifiers=qualifiers,
                    scope={"dataset_sha256": dataset_hash, "investigation_id": investigation_id},
                    metadata={
                        "discovery_origin": seed.origin if seed else "deterministic_miner",
                        "discovery_split_non_warranting": True,
                        "requires_independent_replication": True,
                        "replication_min_independent_sources": 2,
                    },
                    actor=actor,
                    actor_role=actor_role,
                )
            else:
                self.ledger._require_claim(claim_id)
            hid = _new_id("hyp")
            candidates.append(
                {
                    "id": hid,
                    "claim_id": claim_id,
                    "x": x,
                    "y": y,
                    "direction": expected_direction,
                    "origin": seed.origin if seed else "deterministic_miner",
                    "rationale": seed.rationale if seed else "Ranked by discovery-split absolute correlation",
                    "discovery_metrics": {
                        "n": len(pair),
                        "pearson_r": r,
                        "abs_r": abs(r),
                        "split_role": "discovery_only_non_warranting",
                    },
                    "preregistration": {
                        "direction": expected_direction,
                        "min_abs_r": spec.min_confirmation_abs_r,
                        "robust_min_abs_r": spec.robust_min_abs_r,
                        "null_abs_r": spec.null_abs_r,
                        "refute_abs_r": spec.refute_abs_r,
                        "alpha_familywise": spec.alpha,
                        "permutation_trials": spec.permutation_trials,
                        "bootstrap_trials": spec.bootstrap_trials,
                        "min_bootstrap_sign_stability": spec.min_bootstrap_sign_stability,
                        "trim_fraction": spec.trim_fraction,
                    },
                }
            )
            if len(candidates) >= spec.budget.max_hypotheses:
                break
        return candidates

    @staticmethod
    def _pairs(
        rows: list[dict[str, str]], indices: Iterable[int], x: str, y: str
    ) -> list[tuple[float, float]]:
        result = []
        for index in indices:
            try:
                a = float((rows[index].get(x) or "").strip())
                b = float((rows[index].get(y) or "").strip())
            except (ValueError, AttributeError):
                continue
            if math.isfinite(a) and math.isfinite(b):
                result.append((a, b))
        return result

    @staticmethod
    def _pearson(xs: list[float], ys: list[float]) -> float | None:
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        mx = math.fsum(xs) / len(xs)
        my = math.fsum(ys) / len(ys)
        sxx = math.fsum((x - mx) ** 2 for x in xs)
        syy = math.fsum((y - my) ** 2 for y in ys)
        if sxx == 0 or syy == 0:
            return None
        value = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(sxx * syy)
        return max(-1.0, min(1.0, value))

    # ------------------------------------------------------------------
    # Serialization and report rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _serialize_spec(spec: DiscoverySpec, dataset_path: Path, replication_path: Path | None) -> dict[str, Any]:
        return {
            "question": spec.question,
            "dataset_path": str(dataset_path),
            "image": spec.image,
            "replication_dataset_path": str(replication_path) if replication_path else None,
            "candidate_hypotheses": [asdict(item) for item in spec.candidate_hypotheses],
            "seed": spec.seed,
            "discovery_fraction": spec.discovery_fraction,
            "min_rows": spec.min_rows,
            "numeric_fraction": spec.numeric_fraction,
            "min_discovery_abs_r": spec.min_discovery_abs_r,
            "min_confirmation_abs_r": spec.min_confirmation_abs_r,
            "robust_min_abs_r": spec.robust_min_abs_r,
            "null_abs_r": spec.null_abs_r,
            "refute_abs_r": spec.refute_abs_r,
            "alpha": spec.alpha,
            "permutation_trials": spec.permutation_trials,
            "bootstrap_trials": spec.bootstrap_trials,
            "min_bootstrap_sign_stability": spec.min_bootstrap_sign_stability,
            "trim_fraction": spec.trim_fraction,
            "budget": asdict(spec.budget),
            "metadata": spec.metadata,
        }

    @staticmethod
    def _hypothesis_row(row) -> dict[str, Any]:
        result = dict(row)
        for key in (
            "discovery_metrics_json",
            "preregistration_json",
            "confirmation_result_json",
            "replication_result_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key) or "{}")
        return result

    @staticmethod
    def _markdown_report(report: dict[str, Any]) -> str:
        lines = [
            f"# Governed discovery report: {report['question']}",
            "",
            f"- Investigation: `{report['investigation_id']}`",
            f"- Status: **{report['status']}**",
            f"- Dataset SHA-256: `{report['dataset']['sha256']}`",
            f"- Report SHA-256: `{report['report_hash']}`",
            "",
            "## Design",
            "",
            "The candidate-mining split was used only to propose hypotheses. It created no warrant. "
            "Every reported confirmation was pre-registered and evaluated on withheld rows through a "
            "manifest-bound execution receipt.",
            "",
            "## Summary",
            "",
        ]
        for key, value in report["summary"].items():
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")
        lines.extend(["", "## Hypotheses", ""])
        for item in report["hypotheses"]:
            lines.extend(
                [
                    f"### {item['claim']}",
                    "",
                    f"- Claim ID: `{item['claim_id']}`",
                    f"- Status: **{item['status']}**",
                    f"- Current support: **{item['current_support']['state']}**",
                    f"- Discovery r: `{item['discovery_metrics'].get('pearson_r')}` (non-warranting)",
                    f"- Confirmation: `{json.dumps(item['confirmation'], sort_keys=True)}`",
                    f"- Replication: `{json.dumps(item['replication'], sort_keys=True)}`",
                    "",
                ]
            )
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {item}" for item in report["limitations"])
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Container worker
    # ------------------------------------------------------------------
    @staticmethod
    def worker_source() -> str:
        """Return the exact standard-library worker staged into the OCI manifest."""
        return _DISCOVERY_WORKER_SOURCE


_DISCOVERY_WORKER_SOURCE = r'''from __future__ import annotations
import csv, json, math, os, random, statistics
from pathlib import Path

INPUT = Path(os.environ.get("ORBITA_INPUT_DIR", "/workspace/input"))
OUTPUT = Path(os.environ.get("ORBITA_OUTPUT_DIR", "/workspace/output"))


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx = math.fsum(xs) / len(xs); my = math.fsum(ys) / len(ys)
    sxx = math.fsum((x-mx)**2 for x in xs); syy = math.fsum((y-my)**2 for y in ys)
    if sxx == 0 or syy == 0: return None
    r = math.fsum((x-mx)*(y-my) for x,y in zip(xs,ys)) / math.sqrt(sxx*syy)
    return max(-1.0, min(1.0, r))


def pairs(rows, indices, x, y):
    out=[]
    for i in indices:
        if i < 0 or i >= len(rows): continue
        try: a=float((rows[i].get(x) or '').strip()); b=float((rows[i].get(y) or '').strip())
        except (ValueError, AttributeError): continue
        if math.isfinite(a) and math.isfinite(b): out.append((a,b))
    return out


def trimmed_r(data, fraction):
    if len(data) < 5: return None
    xs=[a for a,_ in data]; ys=[b for _,b in data]
    medx=statistics.median(xs); medy=statistics.median(ys)
    madx=statistics.median([abs(x-medx) for x in xs]) or 1.0
    mady=statistics.median([abs(y-medy) for y in ys]) or 1.0
    ranked=sorted(data, key=lambda p: ((p[0]-medx)/madx)**2 + ((p[1]-medy)/mady)**2)
    keep=max(3, len(ranked)-int(math.floor(len(ranked)*fraction)))
    kept=ranked[:keep]
    return pearson([a for a,_ in kept],[b for _,b in kept])


def permutation_p(data, observed, trials, rng):
    xs=[a for a,_ in data]; ys=[b for _,b in data]; extreme=0
    for _ in range(trials):
        shuffled=list(ys); rng.shuffle(shuffled)
        value=pearson(xs, shuffled)
        if value is not None and abs(value) >= abs(observed)-1e-15: extreme += 1
    return (extreme+1)/(trials+1)


def bootstrap_stability(data, expected_sign, trials, rng):
    values=[]; n=len(data)
    for _ in range(trials):
        sample=[data[rng.randrange(n)] for _ in range(n)]
        value=pearson([a for a,_ in sample],[b for _,b in sample])
        if value is not None: values.append(value)
    if not values: return 0.0, None
    same=sum((v>0 if expected_sign>0 else v<0) for v in values)/len(values)
    return same, statistics.median(values)


def evaluate(h, rows, plan, position):
    data=pairs(rows, plan['row_indices'], h['x'], h['y'])
    result={'x':h['x'],'y':h['y'],'direction':h['direction'],'n':len(data)}
    if len(data) < plan['min_rows']:
        result.update(decision='inconclusive',decision_code=0,reasons=['insufficient_complete_rows'])
        return result
    r=pearson([a for a,_ in data],[b for _,b in data])
    if r is None:
        result.update(decision='inconclusive',decision_code=0,reasons=['undefined_correlation'])
        return result
    expected=1 if h['direction']=='positive' else -1
    robust=trimmed_r(data, plan['trim_fraction'])
    rng=random.Random(plan['seed'] + position*100003)
    p=permutation_p(data, r, plan['permutation_trials'], rng)
    stability, bootstrap_median=bootstrap_stability(data, expected, plan['bootstrap_trials'], rng)
    sign_match=(r*expected)>0
    robust_match=(robust is not None and robust*expected>0 and abs(robust)>=plan['robust_min_abs_r'])
    gates={
      'direction':sign_match,
      'effect_size':abs(r)>=plan['min_confirmation_abs_r'],
      'robust_trim':robust_match,
      'permutation':p<=plan['adjusted_alpha'],
      'bootstrap_sign_stability':stability>=plan['min_bootstrap_sign_stability'],
    }
    if all(gates.values()):
        decision='support'; code=1; reasons=['all_preregistered_gates_passed']
    elif ((not sign_match) and abs(r)>=plan['refute_abs_r']) or (abs(r)<=plan['null_abs_r'] and p>plan['adjusted_alpha']):
        decision='refute'; code=-1; reasons=['predeclared_refutation_region_reached']
    else:
        decision='inconclusive'; code=0; reasons=[k for k,v in gates.items() if not v]
    result.update(
      pearson_r=r, abs_r=abs(r), robust_trimmed_r=robust,
      permutation_p=p, adjusted_alpha=plan['adjusted_alpha'],
      bootstrap_sign_stability=stability, bootstrap_median_r=bootstrap_median,
      gates=gates, decision=decision, decision_code=code, reasons=reasons,
    )
    return result


def main():
    plan=json.loads((INPUT/'plan.json').read_text(encoding='utf-8'))
    with (INPUT/'dataset.csv').open('r',encoding='utf-8-sig',newline='') as handle:
        rows=[dict(row) for row in csv.DictReader(handle)]
    results={}
    for position,h in enumerate(plan['hypotheses']): results[h['id']]=evaluate(h,rows,plan,position)
    output={
      'schema_version':plan['schema_version'],
      'investigation_id':plan['investigation_id'],
      'phase':plan['phase'],
      'rows_total':len(rows),
      'row_indices_tested':len(plan['row_indices']),
      'hypotheses':results,
    }
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'result.json').write_text(json.dumps(output,indent=2,sort_keys=True),encoding='utf-8')


if __name__=='__main__': main()
'''
