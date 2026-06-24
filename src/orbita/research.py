from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
import statistics
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

from .evaluation import EVALUATION_RESPONSE_SCHEMA, RESPONSE_SCHEMA_VERSION
from .models import ActorRole

RESEARCH_API_VERSION = "1.0"
REVIEW_LABELS = {"pass", "fail", "uncertain"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
class StudyArmSpec:
    key: str
    name: str
    system_kind: str
    provider: str | None = None
    model_version: str | None = None
    system_prompt: str = ""
    user_prompt_template: str = ""
    temperature: float = 0.0
    retrieval_config: dict[str, Any] = field(default_factory=dict)
    verifier_config: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    max_cost_usd_per_run: float | None = None

    def __post_init__(self) -> None:
        if not self.key.strip() or not self.name.strip():
            raise ValueError("Study arm key and name are required")
        if self.system_kind not in {
            "base_llm",
            "rag",
            "final_answer_verifier",
            "orbita",
            "custom",
        }:
            raise ValueError(f"Unsupported system kind: {self.system_kind}")
        if not 0 <= float(self.temperature) <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if self.max_cost_usd_per_run is not None and self.max_cost_usd_per_run < 0:
            raise ValueError("max_cost_usd_per_run cannot be negative")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StudyArmSpec":
        return cls(
            key=str(value["key"]),
            name=str(value["name"]),
            system_kind=str(value["system_kind"]),
            provider=value.get("provider"),
            model_version=value.get("model_version"),
            system_prompt=str(value.get("system_prompt", "")),
            user_prompt_template=str(value.get("user_prompt_template", "")),
            temperature=float(value.get("temperature", 0.0)),
            retrieval_config=dict(value.get("retrieval_config", {})),
            verifier_config=dict(value.get("verifier_config", {})),
            config=dict(value.get("config", {})),
            max_cost_usd_per_run=(
                None
                if value.get("max_cost_usd_per_run") is None
                else float(value["max_cost_usd_per_run"])
            ),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalStudySpec:
    title: str
    suite_id: str
    arms: tuple[StudyArmSpec, ...]
    repetitions: int = 3
    private_fraction: float = 0.5
    partition_seed: int = 20260619
    reviewers_per_item: int = 2
    primary_metric: str = "mean_private_task_score"
    alpha: float = 0.05
    target_power: float = 0.8
    expected_baseline: float = 0.60
    expected_improvement: float = 0.15
    assumed_discordance: float = 0.30
    max_total_cost_usd: float | None = None
    stopping_rules: dict[str, Any] = field(default_factory=dict)
    hypotheses: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.title.strip() or not self.suite_id.strip():
            raise ValueError("Study title and suite_id are required")
        if len(self.arms) < 2:
            raise ValueError("At least two study arms are required")
        keys = [arm.key for arm in self.arms]
        if len(keys) != len(set(keys)):
            raise ValueError("Study arm keys must be unique")
        if self.repetitions < 1:
            raise ValueError("repetitions must be at least 1")
        if not 0 < self.private_fraction < 1:
            raise ValueError("private_fraction must be between 0 and 1")
        if self.reviewers_per_item < 1:
            raise ValueError("reviewers_per_item must be at least 1")
        if not 0 < self.alpha < 1 or not 0 < self.target_power < 1:
            raise ValueError("alpha and target_power must be between 0 and 1")
        if not 0 <= self.expected_baseline <= 1:
            raise ValueError("expected_baseline must be between 0 and 1")
        if not 0 < self.expected_improvement < 1:
            raise ValueError("expected_improvement must be between 0 and 1")
        if not self.expected_improvement <= self.assumed_discordance <= 1:
            raise ValueError("assumed_discordance must be >= expected_improvement and <= 1")
        if self.max_total_cost_usd is not None and self.max_total_cost_usd < 0:
            raise ValueError("max_total_cost_usd cannot be negative")
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "hypotheses", tuple(str(x) for x in self.hypotheses))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmpiricalStudySpec":
        return cls(
            title=str(value["title"]),
            suite_id=str(value["suite_id"]),
            arms=tuple(StudyArmSpec.from_dict(item) for item in value["arms"]),
            repetitions=int(value.get("repetitions", 3)),
            private_fraction=float(value.get("private_fraction", 0.5)),
            partition_seed=int(value.get("partition_seed", 20260619)),
            reviewers_per_item=int(value.get("reviewers_per_item", 2)),
            primary_metric=str(value.get("primary_metric", "mean_private_task_score")),
            alpha=float(value.get("alpha", 0.05)),
            target_power=float(value.get("target_power", 0.8)),
            expected_baseline=float(value.get("expected_baseline", 0.60)),
            expected_improvement=float(value.get("expected_improvement", 0.15)),
            assumed_discordance=float(value.get("assumed_discordance", 0.30)),
            max_total_cost_usd=(
                None
                if value.get("max_total_cost_usd") is None
                else float(value["max_total_cost_usd"])
            ),
            stopping_rules=dict(value.get("stopping_rules", {})),
            hypotheses=tuple(value.get("hypotheses", [])),
            metadata=dict(value.get("metadata", {})),
        )


class ResearchError(RuntimeError):
    pass


class EmpiricalResearchRuntime:
    def __init__(self, ledger: Any, workspace: str | Path | None = None):
        self.ledger = ledger
        base = Path(workspace) if workspace is not None else Path(ledger.db.path).parent / "research_workspace"
        self.workspace = base.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Study creation and preregistration
    # ------------------------------------------------------------------
    def create_study(self, spec: EmpiricalStudySpec) -> dict[str, Any]:
        suite = self.ledger.evaluations.get_suite(spec.suite_id)
        if not suite["integrity_valid"]:
            raise ResearchError("Evaluation suite integrity check failed")
        if len(suite["tasks"]) < 2:
            raise ResearchError("A study requires at least two tasks")

        ordered = sorted(
            suite["tasks"],
            key=lambda item: _sha256_text(f"{spec.partition_seed}:{item['id']}"),
        )
        private_count = max(1, min(len(ordered) - 1, round(len(ordered) * spec.private_fraction)))
        private_ids = [item["id"] for item in ordered[:private_count]]
        development_ids = [item["id"] for item in ordered[private_count:]]
        partition = {
            "algorithm": "sha256(seed:task_id) stable ordering",
            "seed": spec.partition_seed,
            "development_task_ids": development_ids,
            "private_task_ids": private_ids,
        }
        partition_hash = _sha256_text(_stable_json(partition))
        power_plan = self.power_plan(
            expected_improvement=spec.expected_improvement,
            assumed_discordance=spec.assumed_discordance,
            alpha=spec.alpha,
            target_power=spec.target_power,
        )
        spec_payload = asdict(spec)
        preregistration = {
            "api_version": RESEARCH_API_VERSION,
            "title": spec.title,
            "suite_id": spec.suite_id,
            "suite_hash": suite["suite_hash"],
            "partition_hash": partition_hash,
            "partition_counts": {
                "development": len(development_ids),
                "private": len(private_ids),
            },
            "arms": [self._arm_public_preregistration(arm) for arm in spec.arms],
            "repetitions": spec.repetitions,
            "reviewers_per_item": spec.reviewers_per_item,
            "primary_metric": spec.primary_metric,
            "alpha": spec.alpha,
            "target_power": spec.target_power,
            "power_plan": power_plan,
            "max_total_cost_usd": spec.max_total_cost_usd,
            "stopping_rules": spec.stopping_rules,
            "hypotheses": list(spec.hypotheses),
            "interpretation_boundary": (
                "This registration freezes the protocol and analysis plan. It does not establish "
                "that any system outperforms another until empirical runs and blinded review are complete."
            ),
        }
        spec_hash = _sha256_text(_stable_json(spec_payload))
        prereg_hash = _sha256_text(_stable_json(preregistration))
        duplicate = self.ledger.db.conn.execute(
            "SELECT id FROM research_studies WHERE spec_hash = ?", (spec_hash,)
        ).fetchone()
        if duplicate:
            return self.get_study(duplicate["id"])

        study_id = _new_id("rst")
        now = _utcnow()
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """INSERT INTO research_studies
                   (id, title, suite_id, status, spec_json, spec_hash, suite_hash,
                    partition_json, partition_hash, preregistration_json,
                    preregistration_hash, report_json, report_hash, created_at, updated_at)
                   VALUES (?, ?, ?, 'sealed', ?, ?, ?, ?, ?, ?, ?, '{}', NULL, ?, ?)""",
                (
                    study_id,
                    spec.title,
                    spec.suite_id,
                    _stable_json(spec_payload),
                    spec_hash,
                    suite["suite_hash"],
                    _stable_json(partition),
                    partition_hash,
                    _stable_json(preregistration),
                    prereg_hash,
                    now,
                    now,
                ),
            )
            for arm in spec.arms:
                arm_payload = asdict(arm)
                self.ledger.db.conn.execute(
                    """INSERT INTO research_arms
                       (id, study_id, arm_key, name, system_kind, config_json, config_hash, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _new_id("rsa"),
                        study_id,
                        arm.key,
                        arm.name,
                        arm.system_kind,
                        _stable_json(arm_payload),
                        _sha256_text(_stable_json(arm_payload)),
                        now,
                    ),
                )
        self.ledger._event(
            "research_study",
            study_id,
            "RESEARCH_STUDY_PREREGISTERED",
            {
                "suite_id": spec.suite_id,
                "preregistration_hash": prereg_hash,
                "partition_hash": partition_hash,
                "arms": len(spec.arms),
                "repetitions": spec.repetitions,
            },
            "research_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_study(study_id)

    @staticmethod
    def _arm_public_preregistration(arm: StudyArmSpec) -> dict[str, Any]:
        payload = asdict(arm)
        payload["system_prompt_hash"] = _sha256_text(arm.system_prompt)
        payload["user_prompt_template_hash"] = _sha256_text(arm.user_prompt_template)
        payload.pop("system_prompt")
        payload.pop("user_prompt_template")
        return payload

    def list_studies(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM research_studies ORDER BY created_at DESC"
        ).fetchall()
        return [self._study_row(row, include_details=False) for row in rows]

    def get_study(self, study_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM research_studies WHERE id = ?", (study_id,)
        ).fetchone()
        if row is None:
            raise ResearchError(f"Unknown research study: {study_id}")
        return self._study_row(row, include_details=True)

    def _study_row(self, row: Any, *, include_details: bool) -> dict[str, Any]:
        value = {
            "id": row["id"],
            "title": row["title"],
            "suite_id": row["suite_id"],
            "status": row["status"],
            "spec_hash": row["spec_hash"],
            "suite_hash": row["suite_hash"],
            "partition_hash": row["partition_hash"],
            "preregistration_hash": row["preregistration_hash"],
            "report_hash": row["report_hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "integrity_valid": self.verify_study(row["id"]),
        }
        if include_details:
            value["spec"] = json.loads(row["spec_json"])
            value["partition"] = json.loads(row["partition_json"])
            value["preregistration"] = json.loads(row["preregistration_json"])
            value["report"] = json.loads(row["report_json"])
            value["arms"] = self.list_arms(row["id"])
            value["runs"] = self.list_runs(row["id"])
            value["amendments"] = self.list_amendments(row["id"])
            value["review_agreement"] = self.review_agreement(row["id"])
        return value

    def verify_study(self, study_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM research_studies WHERE id = ?", (study_id,)
        ).fetchone()
        if row is None:
            return False
        try:
            spec = json.loads(row["spec_json"])
            partition = json.loads(row["partition_json"])
            prereg = json.loads(row["preregistration_json"])
        except json.JSONDecodeError:
            return False
        if _sha256_text(_stable_json(spec)) != row["spec_hash"]:
            return False
        if _sha256_text(_stable_json(partition)) != row["partition_hash"]:
            return False
        if _sha256_text(_stable_json(prereg)) != row["preregistration_hash"]:
            return False
        suite = self.ledger.evaluations.get_suite(row["suite_id"], include_gold=False)
        if suite["suite_hash"] != row["suite_hash"] or not suite["integrity_valid"]:
            return False
        arms = self.ledger.db.conn.execute(
            "SELECT * FROM research_arms WHERE study_id = ?", (study_id,)
        ).fetchall()
        if len(arms) != len(spec.get("arms", [])):
            return False
        return all(
            _sha256_text(row_arm["config_json"]) == row_arm["config_hash"] for row_arm in arms
        )

    # ------------------------------------------------------------------
    # Arms, run packs, and empirical run import
    # ------------------------------------------------------------------
    def list_arms(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM research_arms WHERE study_id = ? ORDER BY arm_key", (study_id,)
        ).fetchall()
        return [
            {
                "id": row["id"],
                "study_id": row["study_id"],
                "key": row["arm_key"],
                "name": row["name"],
                "system_kind": row["system_kind"],
                "config": json.loads(row["config_json"]),
                "config_hash": row["config_hash"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _get_arm(self, study_id: str, arm_key: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM research_arms WHERE study_id = ? AND arm_key = ?",
            (study_id, arm_key),
        ).fetchone()
        if row is None:
            raise ResearchError(f"Unknown study arm: {arm_key}")
        return {
            "id": row["id"],
            "key": row["arm_key"],
            "name": row["name"],
            "system_kind": row["system_kind"],
            "config": json.loads(row["config_json"]),
            "config_hash": row["config_hash"],
        }

    def export_run_pack(
        self,
        study_id: str,
        arm_key: str,
        repetition: int,
        *,
        partition_name: str = "private",
        out_path: str | Path | None = None,
    ) -> dict[str, Any]:
        study = self.get_study(study_id)
        if not study["integrity_valid"]:
            raise ResearchError("Study integrity check failed")
        if partition_name not in {"development", "private"}:
            raise ValueError("partition_name must be development or private")
        if not 0 <= repetition < int(study["spec"]["repetitions"]):
            raise ValueError("repetition is outside the preregistered range")
        arm = self._get_arm(study_id, arm_key)
        existing = self.ledger.db.conn.execute(
            """SELECT * FROM research_runs
               WHERE study_id = ? AND arm_id = ? AND repetition = ? AND partition_name = ?""",
            (study_id, arm["id"], repetition, partition_name),
        ).fetchone()
        if existing:
            pack = json.loads(existing["pack_json"])
            if self._pack_hash(pack) != existing["pack_hash"]:
                raise ResearchError("Frozen run pack integrity check failed")
            if out_path is not None:
                path = Path(out_path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
            return {
                "research_run_id": existing["id"],
                "pack": pack,
                "pack_hash": existing["pack_hash"],
            }

        task_ids = set(study["partition"][f"{partition_name}_task_ids"])
        suite = self.ledger.evaluations.get_suite(study["suite_id"], include_gold=False)
        tasks = [
            {
                key: task[key]
                for key in ("id", "category", "prompt", "context", "sequence", "metadata")
            }
            for task in suite["tasks"]
            if task["id"] in task_ids
        ]
        seed = int(_sha256_text(f"{study_id}:{arm_key}:{repetition}:{partition_name}")[:8], 16)
        run_id = _new_id("rsr")
        blind_code = f"B-{_sha256_text(run_id)[:10].upper()}"
        pack = {
            "api_version": RESEARCH_API_VERSION,
            "study_id": study_id,
            "study_title": study["title"],
            "preregistration_hash": study["preregistration_hash"],
            "partition_hash": study["partition_hash"],
            "partition": partition_name,
            "arm": {
                "key": arm["key"],
                "name": arm["name"],
                "system_kind": arm["system_kind"],
                "config": arm["config"],
                "config_hash": arm["config_hash"],
            },
            "repetition": repetition,
            "seed": seed,
            "blind_code": blind_code,
            "tasks": tasks,
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "response_schema": EVALUATION_RESPONSE_SCHEMA,
            "submission_requirements": {
                "evaluation_mode": "empirical",
                "metadata_fields": {
                    "study_id": study_id,
                    "study_arm_key": arm_key,
                    "study_repetition": repetition,
                    "study_partition": partition_name,
                },
            },
        }
        pack_hash = self._pack_hash(pack)
        pack["pack_hash"] = pack_hash
        now = _utcnow()
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """INSERT INTO research_runs
                   (id, study_id, arm_id, repetition, partition_name, seed, blind_code,
                    status, pack_json, pack_hash, evaluation_run_id, response_hash,
                    cost_usd, latency_ms, token_usage_json, metadata_json, created_at, completed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'assigned', ?, ?, NULL, NULL, 0, 0, '{}', '{}', ?, NULL)""",
                (
                    run_id,
                    study_id,
                    arm["id"],
                    repetition,
                    partition_name,
                    seed,
                    blind_code,
                    _stable_json(pack),
                    pack_hash,
                    now,
                ),
            )
        if out_path is not None:
            path = Path(out_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"research_run_id": run_id, "pack": pack, "pack_hash": pack_hash}

    @staticmethod
    def _pack_hash(pack: dict[str, Any]) -> str:
        body = dict(pack)
        body.pop("pack_hash", None)
        return _sha256_text(_stable_json(body))

    def import_run(
        self,
        study_id: str,
        arm_key: str,
        repetition: int,
        payload: dict[str, Any],
        *,
        run_origin: str,
        partition_name: str = "private",
        cost_usd: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if run_origin not in {"live_model", "human", "replay", "validation_fixture"}:
            raise ValueError("run_origin must be live_model, human, replay, or validation_fixture")
        if cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        run_pack = self.export_run_pack(
            study_id, arm_key, repetition, partition_name=partition_name
        )
        run_id = run_pack["research_run_id"]
        pack = run_pack["pack"]
        study = self.get_study(study_id)
        arm = self._get_arm(study_id, arm_key)
        row = self.ledger.db.conn.execute(
            "SELECT * FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row["status"] == "completed":
            return self.get_run(run_id)

        result_ids = {item.get("task_id") for item in payload.get("results", [])}
        expected_ids = {item["id"] for item in pack["tasks"]}
        if result_ids != expected_ids:
            missing = sorted(expected_ids - result_ids)
            extra = sorted(result_ids - expected_ids)
            raise ResearchError(f"Response task set mismatch; missing={missing}, extra={extra}")
        system = payload.get("system", {})
        if system.get("evaluation_mode") != "empirical":
            raise ResearchError("Empirical study responses must use evaluation_mode='empirical'")
        if system.get("kind") != arm["system_kind"]:
            raise ResearchError("Response system kind does not match preregistered arm")
        response_meta = payload.get("metadata", {})
        required_meta = {
            "study_id": study_id,
            "study_arm_key": arm_key,
            "study_repetition": repetition,
            "study_partition": partition_name,
            "study_pack_hash": run_pack["pack_hash"],
        }
        for key, expected in required_meta.items():
            if response_meta.get(key) != expected:
                raise ResearchError(f"Response metadata {key!r} does not match frozen run pack")

        arm_limit = arm["config"].get("max_cost_usd_per_run")
        if arm_limit is not None and cost_usd > float(arm_limit):
            raise ResearchError("Run cost exceeds preregistered per-run budget")
        spent = self.ledger.db.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM research_runs WHERE study_id = ?",
            (study_id,),
        ).fetchone()["total"]
        total_limit = study["spec"].get("max_total_cost_usd")
        if total_limit is not None and float(spent) + cost_usd > float(total_limit):
            raise ResearchError("Run cost would exceed preregistered total study budget")

        evaluation_run = self.ledger.evaluations.import_run(study["suite_id"], payload)
        latency_ms = sum(float(item.get("latency_ms") or 0) for item in payload["results"])
        token_totals: dict[str, float] = {}
        for item in payload["results"]:
            for key, value in (item.get("token_usage") or {}).items():
                if isinstance(value, (int, float)):
                    token_totals[key] = token_totals.get(key, 0.0) + float(value)
        now = _utcnow()
        response_hash = _sha256_text(_stable_json(payload))
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """UPDATE research_runs
                   SET status='completed', evaluation_run_id=?, response_hash=?, cost_usd=?,
                       latency_ms=?, token_usage_json=?, metadata_json=?, completed_at=?
                   WHERE id=?""",
                (
                    evaluation_run["id"],
                    response_hash,
                    cost_usd,
                    latency_ms,
                    _stable_json(token_totals),
                    _stable_json({"run_origin": run_origin, **(metadata or {})}),
                    now,
                    run_id,
                ),
            )
        self.ledger._event(
            "research_run",
            run_id,
            "EMPIRICAL_RUN_IMPORTED",
            {
                "study_id": study_id,
                "arm_key": arm_key,
                "repetition": repetition,
                "partition": partition_name,
                "evaluation_run_id": evaluation_run["id"],
                "cost_usd": cost_usd,
                "latency_ms": latency_ms,
                "run_origin": run_origin,
            },
            "research_runtime",
            ActorRole.TOOL,
        )
        self.ledger.db.conn.commit()
        return self.get_run(run_id)

    def list_runs(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            """SELECT r.*, a.arm_key, a.name AS arm_name
               FROM research_runs r JOIN research_arms a ON a.id = r.arm_id
               WHERE r.study_id = ? ORDER BY a.arm_key, r.repetition, r.partition_name""",
            (study_id,),
        ).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            """SELECT r.*, a.arm_key, a.name AS arm_name
               FROM research_runs r JOIN research_arms a ON a.id = r.arm_id
               WHERE r.id = ?""",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ResearchError(f"Unknown research run: {run_id}")
        return self._run_row(row)

    def _run_row(self, row: Any) -> dict[str, Any]:
        value = {
            "id": row["id"],
            "study_id": row["study_id"],
            "arm_key": row["arm_key"],
            "arm_name": row["arm_name"],
            "repetition": row["repetition"],
            "partition": row["partition_name"],
            "seed": row["seed"],
            "blind_code": row["blind_code"],
            "status": row["status"],
            "pack_hash": row["pack_hash"],
            "evaluation_run_id": row["evaluation_run_id"],
            "response_hash": row["response_hash"],
            "cost_usd": row["cost_usd"],
            "latency_ms": row["latency_ms"],
            "token_usage": json.loads(row["token_usage_json"]),
            "metadata": json.loads(row["metadata_json"]),
            "run_origin": json.loads(row["metadata_json"]).get("run_origin"),
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
        }
        if row["evaluation_run_id"]:
            evaluation_run = self.ledger.evaluations.get_run(row["evaluation_run_id"])
            value["evaluation_metrics"] = evaluation_run["metrics"]
            value["integrity_valid"] = self.verify_run(row["id"])
        else:
            value["evaluation_metrics"] = None
            value["integrity_valid"] = _sha256_text(row["pack_json"]) == row["pack_hash"]
        return value

    def verify_run(self, run_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM research_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return False
        try:
            pack = json.loads(row["pack_json"])
        except json.JSONDecodeError:
            return False
        if self._pack_hash(pack) != row["pack_hash"]:
            return False
        if row["status"] == "completed":
            if not row["evaluation_run_id"] or not row["response_hash"]:
                return False
            evaluation_run = self.ledger.evaluations.get_run(row["evaluation_run_id"])
            return evaluation_run["response_hash"] == row["response_hash"] and evaluation_run["integrity_valid"]
        return True

    # ------------------------------------------------------------------
    # Protocol amendments
    # ------------------------------------------------------------------
    def add_amendment(
        self,
        study_id: str,
        amendment: dict[str, Any],
        *,
        reason: str,
        actor: str,
    ) -> dict[str, Any]:
        if not reason.strip() or not actor.strip():
            raise ValueError("reason and actor are required")
        self.get_study(study_id)
        row = self.ledger.db.conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS seq FROM research_amendments WHERE study_id = ?",
            (study_id,),
        ).fetchone()
        sequence = int(row["seq"]) + 1
        payload = {
            "sequence": sequence,
            "amendment": amendment,
            "reason": reason,
            "actor": actor,
            "created_at": _utcnow(),
        }
        amendment_hash = _sha256_text(_stable_json(payload))
        amendment_id = _new_id("ram")
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """INSERT INTO research_amendments
                   (id, study_id, sequence, amendment_json, amendment_hash, reason, actor, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    amendment_id,
                    study_id,
                    sequence,
                    _stable_json(payload),
                    amendment_hash,
                    reason,
                    actor,
                    payload["created_at"],
                ),
            )
        return {
            "id": amendment_id,
            "study_id": study_id,
            "sequence": sequence,
            "payload": payload,
            "amendment_hash": amendment_hash,
        }

    def list_amendments(self, study_id: str) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT * FROM research_amendments WHERE study_id = ? ORDER BY sequence",
            (study_id,),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "sequence": row["sequence"],
                "payload": json.loads(row["amendment_json"]),
                "amendment_hash": row["amendment_hash"],
                "reason": row["reason"],
                "actor": row["actor"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Blinded human review
    # ------------------------------------------------------------------
    def assign_reviews(self, study_id: str, reviewers: list[str]) -> dict[str, Any]:
        study = self.get_study(study_id)
        required = int(study["spec"]["reviewers_per_item"])
        clean = sorted({item.strip() for item in reviewers if item.strip()})
        if len(clean) < required:
            raise ResearchError(f"At least {required} distinct reviewers are required")
        runs = [run for run in self.list_runs(study_id) if run["status"] == "completed" and run["partition"] == "private"]
        created = 0
        now = _utcnow()
        rng = random.Random(int(study["spec"]["partition_seed"]))
        for run in runs:
            result_rows = self.ledger.db.conn.execute(
                """SELECT etr.*, et.task_key, et.public_json
                   FROM evaluation_task_results etr
                   JOIN evaluation_tasks et ON et.id = etr.task_id
                   WHERE etr.run_id = ? AND et.task_key IN ({})
                   ORDER BY et.position""".format(
                    ",".join("?" for _ in study["partition"]["private_task_ids"])
                ),
                (run["evaluation_run_id"], *study["partition"]["private_task_ids"]),
            ).fetchall()
            for result_row in result_rows:
                chosen = clean.copy()
                rng.shuffle(chosen)
                chosen = chosen[:required]
                response = json.loads(result_row["response_json"])
                public_task = json.loads(result_row["public_json"])
                for reviewer in chosen:
                    payload = {
                        "api_version": RESEARCH_API_VERSION,
                        "study_id": study_id,
                        "blind_run_code": run["blind_code"],
                        "task_id": result_row["task_key"],
                        "task": public_task,
                        "response": response,
                        "review_schema": {
                            "label": sorted(REVIEW_LABELS),
                            "confidence": "0.0 to 1.0",
                            "rationale": "required free text",
                        },
                    }
                    assignment_hash = _sha256_text(_stable_json(payload))
                    existing = self.ledger.db.conn.execute(
                        """SELECT id FROM research_review_assignments
                           WHERE research_run_id=? AND task_id=? AND reviewer=?""",
                        (run["id"], result_row["task_id"], reviewer),
                    ).fetchone()
                    if existing:
                        continue
                    with self.ledger.db.conn:
                        self.ledger.db.conn.execute(
                            """INSERT INTO research_review_assignments
                               (id, study_id, research_run_id, task_id, reviewer, blind_code,
                                payload_json, payload_hash, status, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                            (
                                _new_id("rra"),
                                study_id,
                                run["id"],
                                result_row["task_id"],
                                reviewer,
                                run["blind_code"],
                                _stable_json(payload),
                                assignment_hash,
                                now,
                            ),
                        )
                    created += 1
        return {"study_id": study_id, "assignments_created": created, "reviewers": clean}

    def export_review_bundle(
        self, study_id: str, reviewer: str, out_path: str | Path | None = None
    ) -> dict[str, Any]:
        rows = self.ledger.db.conn.execute(
            """SELECT * FROM research_review_assignments
               WHERE study_id=? AND reviewer=? ORDER BY blind_code, created_at""",
            (study_id, reviewer),
        ).fetchall()
        assignments = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            assignments.append(
                {
                    "assignment_id": row["id"],
                    "status": row["status"],
                    "payload": payload,
                    "payload_hash": row["payload_hash"],
                }
            )
        bundle = {
            "api_version": RESEARCH_API_VERSION,
            "study_id": study_id,
            "reviewer": reviewer,
            "assignments": assignments,
        }
        bundle["bundle_hash"] = _sha256_text(_stable_json(bundle))
        if out_path is not None:
            path = Path(out_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        return bundle

    def submit_review(
        self,
        assignment_id: str,
        *,
        reviewer: str,
        label: str,
        confidence: float,
        rationale: str,
    ) -> dict[str, Any]:
        if label not in REVIEW_LABELS:
            raise ValueError(f"label must be one of {sorted(REVIEW_LABELS)}")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not rationale.strip():
            raise ValueError("rationale is required")
        assignment = self.ledger.db.conn.execute(
            "SELECT * FROM research_review_assignments WHERE id=?", (assignment_id,)
        ).fetchone()
        if assignment is None:
            raise ResearchError(f"Unknown review assignment: {assignment_id}")
        if assignment["reviewer"] != reviewer:
            raise ResearchError("Reviewer does not own this blinded assignment")
        if _sha256_text(assignment["payload_json"]) != assignment["payload_hash"]:
            raise ResearchError("Review assignment integrity check failed")
        payload = {
            "assignment_id": assignment_id,
            "reviewer": reviewer,
            "label": label,
            "confidence": confidence,
            "rationale": rationale,
            "created_at": _utcnow(),
        }
        review_hash = _sha256_text(_stable_json(payload))
        existing = self.ledger.db.conn.execute(
            "SELECT id FROM research_reviews WHERE assignment_id=?", (assignment_id,)
        ).fetchone()
        if existing:
            raise ResearchError("Assignment already reviewed")
        review_id = _new_id("rrv")
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """INSERT INTO research_reviews
                   (id, assignment_id, reviewer, label, confidence, rationale, review_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    assignment_id,
                    reviewer,
                    label,
                    confidence,
                    rationale,
                    review_hash,
                    payload["created_at"],
                ),
            )
            self.ledger.db.conn.execute(
                "UPDATE research_review_assignments SET status='completed' WHERE id=?",
                (assignment_id,),
            )
        return {"id": review_id, **payload, "review_hash": review_hash}

    def adjudicate(
        self,
        study_id: str,
        research_run_id: str,
        task_key: str,
        *,
        adjudicator: str,
        label: str,
        rationale: str,
    ) -> dict[str, Any]:
        if label not in REVIEW_LABELS:
            raise ValueError(f"label must be one of {sorted(REVIEW_LABELS)}")
        if not adjudicator.strip() or not rationale.strip():
            raise ValueError("adjudicator and rationale are required")
        task = self.ledger.db.conn.execute(
            """SELECT et.id FROM evaluation_tasks et
               JOIN research_studies rs ON rs.suite_id=et.suite_id
               WHERE rs.id=? AND et.task_key=?""",
            (study_id, task_key),
        ).fetchone()
        if task is None:
            raise ResearchError("Unknown task for study")
        payload = {
            "study_id": study_id,
            "research_run_id": research_run_id,
            "task_id": task["id"],
            "adjudicator": adjudicator,
            "label": label,
            "rationale": rationale,
            "created_at": _utcnow(),
        }
        item_hash = _sha256_text(_stable_json(payload))
        adjudication_id = _new_id("rad")
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                """INSERT INTO research_adjudications
                   (id, study_id, research_run_id, task_id, adjudicator, label,
                    rationale, adjudication_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(research_run_id, task_id) DO UPDATE SET
                    adjudicator=excluded.adjudicator, label=excluded.label,
                    rationale=excluded.rationale, adjudication_hash=excluded.adjudication_hash,
                    created_at=excluded.created_at""",
                (
                    adjudication_id,
                    study_id,
                    research_run_id,
                    task["id"],
                    adjudicator,
                    label,
                    rationale,
                    item_hash,
                    payload["created_at"],
                ),
            )
        return {"id": adjudication_id, **payload, "adjudication_hash": item_hash}

    def review_agreement(self, study_id: str) -> dict[str, Any]:
        rows = self.ledger.db.conn.execute(
            """SELECT a.research_run_id, a.task_id, r.label
               FROM research_review_assignments a
               JOIN research_reviews r ON r.assignment_id=a.id
               WHERE a.study_id=? ORDER BY a.research_run_id, a.task_id""",
            (study_id,),
        ).fetchall()
        grouped: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            grouped.setdefault((row["research_run_id"], row["task_id"]), []).append(row["label"])
        items = [labels for labels in grouped.values() if len(labels) >= 2]
        if not items:
            return {
                "rated_items": 0,
                "ratings": len(rows),
                "pairwise_percent_agreement": None,
                "fleiss_kappa": None,
                "disagreements": 0,
            }
        total_pairs = 0
        agreeing_pairs = 0
        counts_global = {label: 0 for label in REVIEW_LABELS}
        p_items = []
        disagreements = 0
        for labels in items:
            n = len(labels)
            local = {label: labels.count(label) for label in REVIEW_LABELS}
            for label, count in local.items():
                counts_global[label] += count
                agreeing_pairs += count * (count - 1) // 2
            pairs = n * (n - 1) // 2
            total_pairs += pairs
            p_items.append(sum(count * (count - 1) for count in local.values()) / (n * (n - 1)))
            if max(local.values()) != n:
                disagreements += 1
        total_ratings = sum(counts_global.values())
        proportions = {label: count / total_ratings for label, count in counts_global.items()}
        p_bar = statistics.fmean(p_items)
        p_expected = sum(value * value for value in proportions.values())
        kappa = None if math.isclose(1 - p_expected, 0.0) else (p_bar - p_expected) / (1 - p_expected)
        return {
            "rated_items": len(items),
            "ratings": total_ratings,
            "pairwise_percent_agreement": agreeing_pairs / total_pairs if total_pairs else None,
            "fleiss_kappa": kappa,
            "label_prevalence": proportions,
            "disagreements": disagreements,
        }

    # ------------------------------------------------------------------
    # Power planning, reports, release bundle, and backup
    # ------------------------------------------------------------------
    @staticmethod
    def power_plan(
        *,
        expected_improvement: float,
        assumed_discordance: float,
        alpha: float,
        target_power: float,
    ) -> dict[str, Any]:
        if not 0 < expected_improvement <= assumed_discordance <= 1:
            raise ValueError("expected_improvement must be positive and <= assumed_discordance <= 1")
        z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
        z_power = NormalDist().inv_cdf(target_power)
        delta = expected_improvement
        discordance = assumed_discordance
        numerator = z_alpha * math.sqrt(discordance) + z_power * math.sqrt(
            max(discordance - delta * delta, 0)
        )
        n = math.ceil((numerator / delta) ** 2)
        return {
            "method": "normal approximation for paired binary McNemar design",
            "expected_improvement": expected_improvement,
            "assumed_discordance": assumed_discordance,
            "alpha_two_sided": alpha,
            "target_power": target_power,
            "recommended_paired_items": n,
            "warning": "Planning estimate only; revise using pilot discordance and task-cluster effects.",
        }

    def compile_report(self, study_id: str) -> dict[str, Any]:
        study = self.get_study(study_id)
        private_ids = set(study["partition"]["private_task_ids"])
        runs = [run for run in study["runs"] if run["partition"] == "private"]
        completed = [run for run in runs if run["status"] == "completed"]
        empirical_runs = [run for run in completed if run.get("run_origin") in {"live_model", "human"}]
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for run in completed:
            evaluation_run = self.ledger.evaluations.get_run(run["evaluation_run_id"])
            private_scores = [
                result["score"]["task_score"]
                for result in evaluation_run["results"]
                if result["task_id"] in private_ids
            ]
            by_arm.setdefault(run["arm_key"], []).append(
                {
                    "research_run_id": run["id"],
                    "blind_code": run["blind_code"],
                    "repetition": run["repetition"],
                    "mean_private_task_score": (
                        statistics.fmean(private_scores) if private_scores else 0.0
                    ),
                    "private_task_scores": {
                        result["task_id"]: result["score"]["task_score"]
                        for result in evaluation_run["results"]
                        if result["task_id"] in private_ids
                    },
                    "cost_usd": run["cost_usd"],
                    "latency_ms": run["latency_ms"],
                    "token_usage": run["token_usage"],
                    "integrity_valid": run["integrity_valid"],
                }
            )
        arm_summaries = []
        for arm in study["arms"]:
            arm_runs = by_arm.get(arm["key"], [])
            scores = [item["mean_private_task_score"] for item in arm_runs]
            arm_summaries.append(
                {
                    "arm_key": arm["key"],
                    "arm_name": arm["name"],
                    "completed_runs": len(arm_runs),
                    "expected_runs": int(study["spec"]["repetitions"]),
                    "mean_private_task_score": statistics.fmean(scores) if scores else None,
                    "score_stdev": statistics.stdev(scores) if len(scores) > 1 else None,
                    "total_cost_usd": sum(item["cost_usd"] for item in arm_runs),
                    "median_latency_ms": (
                        statistics.median(item["latency_ms"] for item in arm_runs)
                        if arm_runs
                        else None
                    ),
                    "runs": arm_runs,
                }
            )
        arm_summaries.sort(
            key=lambda item: -1 if item["mean_private_task_score"] is None else item["mean_private_task_score"],
            reverse=True,
        )
        comparisons = self._paired_arm_comparisons(arm_summaries, private_ids)
        agreement = self.review_agreement(study_id)
        expected_run_count = len(study["arms"]) * int(study["spec"]["repetitions"])
        empirical_complete = len(empirical_runs) == expected_run_count
        validation_only = bool(completed) and not empirical_runs
        report = {
            "api_version": RESEARCH_API_VERSION,
            "study": {
                "id": study_id,
                "title": study["title"],
                "suite_id": study["suite_id"],
                "suite_hash": study["suite_hash"],
                "preregistration_hash": study["preregistration_hash"],
                "partition_hash": study["partition_hash"],
                "private_tasks": len(private_ids),
                "expected_runs": expected_run_count,
                "completed_runs": len(completed),
                "empirical_runs": len(empirical_runs),
            },
            "status": "complete" if empirical_complete else "validation_only" if validation_only else "incomplete",
            "interpretation_boundary": (
                "Results are empirical only when all imported runs are real system outputs. "
                "Synthetic fixtures, missing repetitions, unblinded review, or incomplete adjudication "
                "must not be described as evidence of superiority."
            ),
            "arms": arm_summaries,
            "paired_private_task_comparisons": comparisons,
            "blinded_review_agreement": agreement,
            "amendments": study["amendments"],
            "run_origins": {origin: sum(1 for run in completed if run.get("run_origin") == origin) for origin in sorted({run.get("run_origin") for run in completed if run.get("run_origin")})},
            "total_cost_usd": sum(run["cost_usd"] for run in completed),
            "generated_at": _utcnow(),
        }
        report_hash = _sha256_text(_stable_json(report))
        out_dir = self.workspace / "reports" / study_id
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "report.json"
        md_path = out_dir / "report.md"
        prereg_path = out_dir / "preregistration.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(self._report_markdown(report), encoding="utf-8")
        prereg_path.write_text(
            json.dumps(study["preregistration"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        now = _utcnow()
        with self.ledger.db.conn:
            self.ledger.db.conn.execute(
                "UPDATE research_studies SET report_json=?, report_hash=?, updated_at=? WHERE id=?",
                (_stable_json(report), report_hash, now, study_id),
            )
            self.ledger.db.conn.execute(
                "DELETE FROM research_artifacts WHERE study_id=?", (study_id,)
            )
            for role, path, media_type in (
                ("report_json", json_path, "application/json"),
                ("report_markdown", md_path, "text/markdown"),
                ("preregistration", prereg_path, "application/json"),
            ):
                content_hash, size = _sha256_file(path)
                self.ledger.db.conn.execute(
                    """INSERT INTO research_artifacts
                       (id, study_id, role, path, content_hash, size_bytes, media_type, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        _new_id("raf"),
                        study_id,
                        role,
                        str(path),
                        content_hash,
                        size,
                        media_type,
                        now,
                    ),
                )
        return {"report": report, "report_hash": report_hash, "report_dir": str(out_dir)}

    @staticmethod
    def _paired_arm_comparisons(
        arm_summaries: list[dict[str, Any]], private_ids: set[str]
    ) -> list[dict[str, Any]]:
        output = []
        rng = random.Random(20260619)
        for index, left in enumerate(arm_summaries):
            for right in arm_summaries[index + 1 :]:
                left_by_rep = {item["repetition"]: item for item in left["runs"]}
                right_by_rep = {item["repetition"]: item for item in right["runs"]}
                diffs = []
                for repetition in sorted(set(left_by_rep).intersection(right_by_rep)):
                    left_scores = left_by_rep[repetition]["private_task_scores"]
                    right_scores = right_by_rep[repetition]["private_task_scores"]
                    for task_id in sorted(private_ids.intersection(left_scores).intersection(right_scores)):
                        diffs.append(float(left_scores[task_id]) - float(right_scores[task_id]))
                if not diffs:
                    continue
                observed = statistics.fmean(diffs)
                bootstrap = []
                for _ in range(2000):
                    sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
                    bootstrap.append(statistics.fmean(sample))
                bootstrap.sort()
                lo = bootstrap[math.floor(0.025 * len(bootstrap))]
                hi = bootstrap[math.ceil(0.975 * len(bootstrap)) - 1]
                output.append(
                    {
                        "left_arm": left["arm_key"],
                        "right_arm": right["arm_key"],
                        "paired_observations": len(diffs),
                        "mean_score_difference": observed,
                        "bootstrap_95_ci": [lo, hi],
                        "direction": (
                            "left_better" if observed > 0 else "right_better" if observed < 0 else "tie"
                        ),
                    }
                )
        return output

    @staticmethod
    def _report_markdown(report: dict[str, Any]) -> str:
        lines = [
            f"# Empirical Research Report — {report['study']['title']}",
            "",
            f"Study ID: `{report['study']['id']}`  ",
            f"Preregistration hash: `{report['study']['preregistration_hash']}`  ",
            f"Private tasks: {report['study']['private_tasks']}  ",
            f"Runs: {report['study']['completed_runs']} / {report['study']['expected_runs']}  ",
            f"Status: **{report['status']}**",
            "",
            f"> {report['interpretation_boundary']}",
            "",
            "## Arm summary",
            "",
            "| Arm | Completed | Mean private score | SD | Cost | Median latency |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm in report["arms"]:
            mean = "—" if arm["mean_private_task_score"] is None else f"{arm['mean_private_task_score']:.3f}"
            sd = "—" if arm["score_stdev"] is None else f"{arm['score_stdev']:.3f}"
            latency = "—" if arm["median_latency_ms"] is None else f"{arm['median_latency_ms']:.0f} ms"
            lines.append(
                f"| {arm['arm_name']} | {arm['completed_runs']} / {arm['expected_runs']} | {mean} | {sd} | ${arm['total_cost_usd']:.4f} | {latency} |"
            )
        lines.extend(["", "## Paired comparisons", ""])
        for item in report["paired_private_task_comparisons"]:
            lo, hi = item["bootstrap_95_ci"]
            lines.append(
                f"- **{item['left_arm']} vs {item['right_arm']}**: {item['mean_score_difference']:+.3f}; "
                f"95% bootstrap CI [{lo:+.3f}, {hi:+.3f}] across {item['paired_observations']} paired observations."
            )
        agreement = report["blinded_review_agreement"]
        lines.extend(
            [
                "",
                "## Blinded review",
                "",
                f"- Rated items: {agreement['rated_items']}",
                f"- Pairwise agreement: {agreement['pairwise_percent_agreement']}",
                f"- Fleiss kappa: {agreement['fleiss_kappa']}",
                f"- Items with disagreement: {agreement['disagreements']}",
                "",
                f"Total recorded API cost: **${report['total_cost_usd']:.4f}**",
                "",
            ]
        )
        return "\n".join(lines)

    def verify_report(self, study_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT report_json, report_hash FROM research_studies WHERE id=?", (study_id,)
        ).fetchone()
        if row is None or not row["report_hash"]:
            return False
        try:
            report = json.loads(row["report_json"])
        except json.JSONDecodeError:
            return False
        if _sha256_text(_stable_json(report)) != row["report_hash"]:
            return False
        artifacts = self.ledger.db.conn.execute(
            "SELECT * FROM research_artifacts WHERE study_id=?", (study_id,)
        ).fetchall()
        if len(artifacts) < 3:
            return False
        for artifact in artifacts:
            path = Path(artifact["path"])
            if not path.is_file():
                return False
            content_hash, size = _sha256_file(path)
            if content_hash != artifact["content_hash"] or size != artifact["size_bytes"]:
                return False
        return True

    def export_release_bundle(self, study_id: str, out_path: str | Path) -> dict[str, Any]:
        if not self.verify_report(study_id):
            raise ResearchError("Compile and verify the study report before export")
        study = self.get_study(study_id)
        out = Path(out_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        bundle_dir = self.workspace / "bundles" / study_id
        bundle_dir.mkdir(parents=True, exist_ok=True)
        files: list[Path] = []
        for artifact in self.ledger.db.conn.execute(
            "SELECT * FROM research_artifacts WHERE study_id=? ORDER BY role", (study_id,)
        ).fetchall():
            files.append(Path(artifact["path"]))
        amendments_path = bundle_dir / "amendments.json"
        amendments_path.write_text(
            json.dumps(study["amendments"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        files.append(amendments_path)
        agreement_path = bundle_dir / "review_agreement.json"
        agreement_path.write_text(
            json.dumps(study["review_agreement"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        files.append(agreement_path)
        manifest = {"study_id": study_id, "files": []}
        for path in files:
            digest, size = _sha256_file(path)
            manifest["files"].append(
                {"name": path.name, "sha256": digest, "size_bytes": size}
            )
        manifest["manifest_hash"] = _sha256_text(_stable_json(manifest))
        manifest_path = bundle_dir / "MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files + [manifest_path]:
                archive.write(path, arcname=path.name)
        bundle_hash, size = _sha256_file(out)
        return {
            "study_id": study_id,
            "bundle_path": str(out),
            "bundle_sha256": bundle_hash,
            "size_bytes": size,
            "manifest": manifest,
        }

    def backup_database(self, destination: str | Path) -> dict[str, Any]:
        path = self.ledger.db.backup(destination)
        digest, size = _sha256_file(path)
        return {"path": str(path), "sha256": digest, "size_bytes": size}

    @staticmethod
    def restore_database(source: str | Path, destination: str | Path) -> dict[str, Any]:
        source_path = Path(source).resolve()
        destination_path = Path(destination).resolve()
        if not source_path.is_file():
            raise ResearchError(f"Backup does not exist: {source_path}")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            raise ResearchError("Restore destination already exists; refusing to overwrite")
        with sqlite3.connect(source_path) as source_db:
            integrity = source_db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise ResearchError(f"Backup integrity check failed: {integrity}")
            with sqlite3.connect(destination_path) as destination_db:
                source_db.backup(destination_db)
        digest, size = _sha256_file(destination_path)
        return {
            "source": str(source_path),
            "destination": str(destination_path),
            "sha256": digest,
            "size_bytes": size,
            "integrity_check": "ok",
        }
