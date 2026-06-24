from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import platform
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from .models import (
    ActorRole,
    AnalysisOutcome,
    AnalysisStatus,
    EvidenceKind,
    Stance,
)

if TYPE_CHECKING:  # pragma: no cover
    from .ledger import EpistemicLedger


ANALYZER_API_VERSION = "1"
_ALLOWED_OPERATORS = {">", ">=", "<", "<=", "==", "!="}


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
class MetricCondition:
    operator: str
    value: float | int | str | bool

    def __post_init__(self) -> None:
        if self.operator not in _ALLOWED_OPERATORS:
            raise ValueError(f"Unsupported comparison operator: {self.operator}")

    @classmethod
    def from_value(cls, value: "MetricCondition | dict[str, Any]") -> "MetricCondition":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError("A metric condition must be a MetricCondition or object")
        return cls(operator=str(value["operator"]), value=value["value"])

    def evaluate(self, observed: Any) -> bool:
        if observed is None:
            return False
        if self.operator == ">":
            return observed > self.value
        if self.operator == ">=":
            return observed >= self.value
        if self.operator == "<":
            return observed < self.value
        if self.operator == "<=":
            return observed <= self.value
        if self.operator == "==":
            return observed == self.value
        if self.operator == "!=":
            return observed != self.value
        raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class AnalysisClaimTest:
    """A receipt-bound test connecting one computed metric to one exact claim.

    The support and refute regions must not both match the same observed value.
    Values outside both regions are recorded as inconclusive and create no
    warranting attestation.
    """

    claim_id: str
    metric_path: str
    support_condition: MetricCondition | dict[str, Any]
    refute_condition: MetricCondition | dict[str, Any] | None = None
    confidence: float = 1.0
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.claim_id:
            raise ValueError("claim_id is required")
        if not self.metric_path:
            raise ValueError("metric_path is required")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")
        object.__setattr__(self, "support_condition", MetricCondition.from_value(self.support_condition))
        if self.refute_condition is not None:
            object.__setattr__(
                self,
                "refute_condition",
                MetricCondition.from_value(self.refute_condition),
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AnalysisClaimTest":
        return cls(
            claim_id=value["claim_id"],
            metric_path=value["metric_path"],
            support_condition=value["support_condition"],
            refute_condition=value.get("refute_condition"),
            confidence=float(value.get("confidence", 1.0)),
            rationale=str(value.get("rationale", "")),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "metric_path": self.metric_path,
            "support_condition": asdict(self.support_condition),
            "refute_condition": (
                asdict(self.refute_condition) if self.refute_condition is not None else None
            ),
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class DatasetAnalysisSpec:
    dataset_path: str | Path
    analysis_type: str
    parameters: dict[str, Any]
    preprocessing: dict[str, Any] = field(default_factory=dict)
    claim_tests: tuple[AnalysisClaimTest, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DatasetAnalysisSpec":
        tests = tuple(
            item if isinstance(item, AnalysisClaimTest) else AnalysisClaimTest.from_dict(item)
            for item in value.get("claim_tests", [])
        )
        return cls(
            dataset_path=value["dataset_path"],
            analysis_type=value["analysis_type"],
            parameters=dict(value.get("parameters", {})),
            preprocessing=dict(value.get("preprocessing", {})),
            claim_tests=tests,
            metadata=dict(value.get("metadata", {})),
        )


class AnalysisError(RuntimeError):
    pass


class DatasetAnalysisRuntime:
    """Hash-bound, deterministic receipts for a small safe analysis vocabulary.

    This layer intentionally does not execute arbitrary user code. Containerized
    code execution is a later phase. v0.3 provides built-in analyzers whose code
    identity is captured, whose inputs are hashed, and whose outputs can be
    replayed and compared.
    """

    def __init__(self, ledger: "EpistemicLedger", artifact_root: str | Path | None = None):
        self.ledger = ledger
        self.artifact_root = (
            Path(artifact_root)
            if artifact_root is not None
            else ledger.db.path.parent / "analysis_artifacts"
        )
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._analyzers: dict[str, Callable[[list[dict[str, str]], dict[str, Any], dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]] = {
            "pearson_correlation": self._pearson_correlation,
            "group_mean_difference": self._group_mean_difference,
            "column_summary": self._column_summary,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        spec: DatasetAnalysisSpec | dict[str, Any],
        *,
        actor: str = "analysis_runtime",
        actor_role: ActorRole = ActorRole.TOOL,
    ) -> dict[str, Any]:
        if isinstance(spec, dict):
            spec = DatasetAnalysisSpec.from_dict(spec)
        dataset_path = Path(spec.dataset_path).expanduser().resolve()
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        if spec.analysis_type not in self._analyzers:
            raise ValueError(
                f"Unknown analysis_type {spec.analysis_type!r}; "
                f"available: {sorted(self._analyzers)}"
            )
        for test in spec.claim_tests:
            self.ledger._require_claim(test.claim_id)

        receipt_id = _new_id("anr")
        started_at = _utcnow()
        dataset_hash, dataset_size = _sha256_file(dataset_path)
        preprocessing = self._normalized_preprocessing(spec.preprocessing)
        code_hash = self._code_hash()
        code_identity = f"orbita.builtin:{spec.analysis_type}:api-{ANALYZER_API_VERSION}"
        environment = self._environment()

        rows: list[dict[str, str]] = []
        schema: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        status = AnalysisStatus.COMPLETED
        failure: str | None = None
        assessments: list[dict[str, Any]] = []
        try:
            rows, schema = self._load_csv(dataset_path, preprocessing)
            outputs, diagnostics = self._analyzers[spec.analysis_type](
                rows,
                dict(spec.parameters),
                preprocessing,
            )
            assessments = self._assess_claims(spec.claim_tests, outputs)
        except Exception as exc:  # receipt failure is explicit and durable
            status = AnalysisStatus.FAILED
            failure = f"{type(exc).__name__}: {exc}"
            diagnostics = {**diagnostics, "failure": failure}
            assessments = []
        completed_at = _utcnow()
        payload = self._receipt_payload(
            receipt_id=receipt_id,
            analysis_type=spec.analysis_type,
            status=status,
            dataset_uri=dataset_path.as_uri(),
            dataset_hash=dataset_hash,
            dataset_size_bytes=dataset_size,
            code_hash=code_hash,
            code_identity=code_identity,
            environment=environment,
            schema=schema,
            preprocessing=preprocessing,
            parameters=dict(spec.parameters),
            outputs=outputs,
            diagnostics=diagnostics,
            metadata=dict(spec.metadata),
            parent_receipt_id=None,
            comparison={},
            assessments=assessments,
            started_at=started_at,
            completed_at=completed_at,
        )
        receipt_hash = _sha256_bytes(_stable_json(payload).encode("utf-8"))
        self._insert_receipt(payload, receipt_hash)
        artifact = self._write_receipt_artifact(receipt_id, payload, receipt_hash)
        self._insert_artifact(receipt_id, artifact)

        evidence_id = None
        if status == AnalysisStatus.COMPLETED:
            evidence_id = self._create_receipt_evidence(
                payload,
                receipt_hash,
                dataset_hash,
                actor,
                actor_role,
            )
            self.ledger.db.conn.execute(
                "UPDATE analysis_receipts SET evidence_id = ? WHERE id = ?",
                (evidence_id, receipt_id),
            )
        self._insert_assessments(receipt_id, assessments, evidence_id, actor, actor_role)
        self.ledger._event(
            "analysis_receipt",
            receipt_id,
            "ANALYSIS_RECEIPT_FINALIZED",
            {
                "status": status.value,
                "receipt_hash": receipt_hash,
                "dataset_hash": dataset_hash,
                "code_hash": code_hash,
                "failure": failure,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return self.get(receipt_id)

    def reproduce(
        self,
        receipt_id: str,
        *,
        dataset_path: str | Path | None = None,
        actor: str = "analysis_runtime",
        actor_role: ActorRole = ActorRole.TOOL,
        rel_tol: float = 1e-9,
        abs_tol: float = 1e-12,
    ) -> dict[str, Any]:
        original = self.get(receipt_id)
        if original["status"] not in {
            AnalysisStatus.COMPLETED.value,
            AnalysisStatus.REPRODUCED.value,
        }:
            raise ValueError("Only successful receipts can be reproduced")

        if dataset_path is None:
            uri = original["dataset_uri"]
            if not uri.startswith("file://"):
                raise ValueError("A local dataset_path is required for non-file receipts")
            from urllib.parse import unquote, urlparse

            dataset_path = Path(unquote(urlparse(uri).path))
        path = Path(dataset_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        current_hash, current_size = _sha256_file(path)
        current_code_hash = self._code_hash()
        current_environment = self._environment()
        started_at = _utcnow()
        reproduction_id = _new_id("anr")
        comparison: dict[str, Any] = {
            "expected_dataset_hash": original["dataset_hash"],
            "observed_dataset_hash": current_hash,
            "dataset_hash_match": current_hash == original["dataset_hash"],
            "expected_code_hash": original["code_hash"],
            "observed_code_hash": current_code_hash,
            "code_hash_match": current_code_hash == original["code_hash"],
            "rel_tol": rel_tol,
            "abs_tol": abs_tol,
        }
        preprocessing = dict(original["preprocessing"])
        parameters = dict(original["parameters"])
        schema: dict[str, Any] = {}
        outputs: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        assessments: list[dict[str, Any]] = []

        if current_hash != original["dataset_hash"]:
            status = AnalysisStatus.INPUT_MISMATCH
            diagnostics = {"failure": "Dataset hash does not match the original receipt"}
        elif current_code_hash != original["code_hash"]:
            status = AnalysisStatus.CODE_MISMATCH
            diagnostics = {"failure": "Analyzer code hash does not match the original receipt"}
        else:
            try:
                rows, schema = self._load_csv(path, preprocessing)
                outputs, diagnostics = self._analyzers[original["analysis_type"]](
                    rows,
                    parameters,
                    preprocessing,
                )
                outputs_match, differences = self._compare_values(
                    original["outputs"], outputs, rel_tol=rel_tol, abs_tol=abs_tol
                )
                comparison["outputs_match"] = outputs_match
                comparison["differences"] = differences
                status = AnalysisStatus.REPRODUCED if outputs_match else AnalysisStatus.DIVERGED
                claim_tests = [
                    AnalysisClaimTest(
                        claim_id=item["claim_id"],
                        metric_path=item["metric_path"],
                        support_condition=item["support_condition"],
                        refute_condition=item["refute_condition"],
                        confidence=item["confidence"],
                        rationale=item["rationale"],
                    )
                    for item in original["assessments"]
                ]
                assessments = self._assess_claims(claim_tests, outputs)
            except Exception as exc:
                status = AnalysisStatus.FAILED
                diagnostics = {"failure": f"{type(exc).__name__}: {exc}"}

        completed_at = _utcnow()
        payload = self._receipt_payload(
            receipt_id=reproduction_id,
            analysis_type=original["analysis_type"],
            status=status,
            dataset_uri=path.as_uri(),
            dataset_hash=current_hash,
            dataset_size_bytes=current_size,
            code_hash=current_code_hash,
            code_identity=original["code_identity"],
            environment=current_environment,
            schema=schema,
            preprocessing=preprocessing,
            parameters=parameters,
            outputs=outputs,
            diagnostics=diagnostics,
            metadata={"reproduction_of": receipt_id},
            parent_receipt_id=receipt_id,
            comparison=comparison,
            assessments=assessments,
            started_at=started_at,
            completed_at=completed_at,
        )
        receipt_hash = _sha256_bytes(_stable_json(payload).encode("utf-8"))
        self._insert_receipt(payload, receipt_hash)
        artifact = self._write_receipt_artifact(reproduction_id, payload, receipt_hash)
        self._insert_artifact(reproduction_id, artifact)

        evidence_id = None
        if status == AnalysisStatus.REPRODUCED:
            evidence_id = self._create_receipt_evidence(
                payload,
                receipt_hash,
                current_hash,
                actor,
                actor_role,
            )
            self.ledger.db.conn.execute(
                "UPDATE analysis_receipts SET evidence_id = ? WHERE id = ?",
                (evidence_id, reproduction_id),
            )
        self._insert_assessments(reproduction_id, assessments, evidence_id, actor, actor_role)
        self.ledger._event(
            "analysis_receipt",
            reproduction_id,
            "ANALYSIS_REPRODUCTION_FINALIZED",
            {
                "parent_receipt_id": receipt_id,
                "status": status.value,
                "receipt_hash": receipt_hash,
                "comparison": comparison,
            },
            actor,
            actor_role,
        )
        self.ledger.db.conn.commit()
        return self.get(reproduction_id)

    def get(self, receipt_id: str) -> dict[str, Any]:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM analysis_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown analysis receipt: {receipt_id}")
        result = dict(row)
        for key in (
            "environment_json",
            "schema_json",
            "preprocessing_json",
            "parameters_json",
            "outputs_json",
            "diagnostics_json",
            "metadata_json",
            "comparison_json",
        ):
            result[key.removesuffix("_json")] = json.loads(result.pop(key))
        result["artifacts"] = [
            self._artifact_row(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM analysis_artifacts WHERE receipt_id = ? ORDER BY created_at",
                (receipt_id,),
            ).fetchall()
        ]
        result["assessments"] = [
            self._assessment_row(item)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM analysis_claim_assessments WHERE receipt_id = ? ORDER BY position",
                (receipt_id,),
            ).fetchall()
        ]
        result["integrity_valid"] = self.verify_integrity(receipt_id)
        result["artifact_integrity_valid"] = self.verify_artifacts(receipt_id)
        result["evidence_binding_valid"] = self.verify_evidence_binding(receipt_id)
        if result.get("evidence_id"):
            evidence_row = self.ledger.db.conn.execute(
                "SELECT active FROM evidence WHERE id = ?", (result["evidence_id"],)
            ).fetchone()
            result["evidence_active"] = bool(evidence_row["active"]) if evidence_row else False
        else:
            result["evidence_active"] = None
        return result

    def list(self) -> list[dict[str, Any]]:
        rows = self.ledger.db.conn.execute(
            "SELECT id FROM analysis_receipts ORDER BY created_at"
        ).fetchall()
        return [self.get(row["id"]) for row in rows]

    def verify_integrity(self, receipt_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT * FROM analysis_receipts WHERE id = ?", (receipt_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown analysis receipt: {receipt_id}")
        assessments = [
            self._assessment_row(item, include_runtime_fields=False)
            for item in self.ledger.db.conn.execute(
                "SELECT * FROM analysis_claim_assessments WHERE receipt_id = ? ORDER BY position",
                (receipt_id,),
            ).fetchall()
        ]
        payload = self._receipt_payload(
            receipt_id=row["id"],
            analysis_type=row["analysis_type"],
            status=AnalysisStatus(row["status"]),
            dataset_uri=row["dataset_uri"],
            dataset_hash=row["dataset_hash"],
            dataset_size_bytes=row["dataset_size_bytes"],
            code_hash=row["code_hash"],
            code_identity=row["code_identity"],
            environment=json.loads(row["environment_json"]),
            schema=json.loads(row["schema_json"]),
            preprocessing=json.loads(row["preprocessing_json"]),
            parameters=json.loads(row["parameters_json"]),
            outputs=json.loads(row["outputs_json"]),
            diagnostics=json.loads(row["diagnostics_json"]),
            metadata=json.loads(row["metadata_json"]),
            parent_receipt_id=row["parent_receipt_id"],
            comparison=json.loads(row["comparison_json"]),
            assessments=assessments,
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )
        observed = _sha256_bytes(_stable_json(payload).encode("utf-8"))
        return observed == row["receipt_hash"]

    def verify_artifacts(self, receipt_id: str) -> bool:
        rows = self.ledger.db.conn.execute(
            "SELECT path, content_hash, size_bytes FROM analysis_artifacts WHERE receipt_id = ?",
            (receipt_id,),
        ).fetchall()
        if not rows:
            return False
        for row in rows:
            path = Path(row["path"])
            if not path.is_file():
                return False
            digest, size = _sha256_file(path)
            if digest != row["content_hash"] or size != row["size_bytes"]:
                return False
        return True

    def verify_evidence_binding(self, receipt_id: str) -> bool:
        row = self.ledger.db.conn.execute(
            "SELECT status, dataset_hash, receipt_hash, evidence_id FROM analysis_receipts WHERE id = ?",
            (receipt_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown analysis receipt: {receipt_id}")
        expects_evidence = row["status"] in {
            AnalysisStatus.COMPLETED.value,
            AnalysisStatus.REPRODUCED.value,
        }
        if not expects_evidence:
            return row["evidence_id"] is None
        if row["evidence_id"] is None:
            return False
        evidence = self.ledger.db.conn.execute(
            "SELECT source_uri, source_kind, content_hash, independence_key, active "
            "FROM evidence WHERE id = ?",
            (row["evidence_id"],),
        ).fetchone()
        if evidence is None:
            return False
        return (
            evidence["source_uri"] == f"analysis://{receipt_id}"
            and evidence["source_kind"] == EvidenceKind.DATASET_ANALYSIS_RECEIPT.value
            and evidence["content_hash"] == row["receipt_hash"]
            and evidence["independence_key"] == f"dataset:{row['dataset_hash']}"
        )

    # ------------------------------------------------------------------
    # CSV and analyzers
    # ------------------------------------------------------------------
    def _normalized_preprocessing(self, value: dict[str, Any]) -> dict[str, Any]:
        missing = value.get("missing_values", ["", "NA", "N/A", "null", "None"])
        if not isinstance(missing, list):
            raise TypeError("missing_values must be a list")
        return {
            "encoding": str(value.get("encoding", "utf-8")),
            "delimiter": str(value.get("delimiter", ",")),
            "strip_whitespace": bool(value.get("strip_whitespace", True)),
            "missing_values": [str(item) for item in missing],
            "missing_policy": "pairwise_drop",
        }

    def _load_csv(
        self, path: Path, preprocessing: dict[str, Any]
    ) -> tuple[list[dict[str, str]], dict[str, Any]]:
        with path.open("r", encoding=preprocessing["encoding"], newline="") as handle:
            reader = csv.DictReader(handle, delimiter=preprocessing["delimiter"])
            if not reader.fieldnames:
                raise AnalysisError("CSV has no header row")
            columns = [name.strip() if preprocessing["strip_whitespace"] else name for name in reader.fieldnames]
            if len(set(columns)) != len(columns):
                raise AnalysisError("CSV has duplicate column names after normalization")
            rows: list[dict[str, str]] = []
            for raw in reader:
                cleaned: dict[str, str] = {}
                for original, column in zip(reader.fieldnames, columns):
                    value = raw.get(original, "")
                    cleaned[column] = value.strip() if preprocessing["strip_whitespace"] else value
                rows.append(cleaned)
        if not rows:
            raise AnalysisError("CSV contains no data rows")
        schema = self._infer_schema(rows, columns, set(preprocessing["missing_values"]))
        schema["row_count"] = len(rows)
        schema["column_count"] = len(columns)
        schema["columns_ordered"] = columns
        return rows, schema

    def _infer_schema(
        self,
        rows: list[dict[str, str]],
        columns: list[str],
        missing_values: set[str],
    ) -> dict[str, Any]:
        report: dict[str, Any] = {"columns": {}}
        for column in columns:
            values = [row[column] for row in rows]
            present = [value for value in values if value not in missing_values]
            numeric = 0
            integers = 0
            for value in present:
                try:
                    number = float(value)
                    if math.isfinite(number):
                        numeric += 1
                        if number.is_integer():
                            integers += 1
                except ValueError:
                    pass
            if present and numeric == len(present):
                inferred = "integer" if integers == len(present) else "float"
            elif numeric:
                inferred = "mixed"
            else:
                inferred = "string"
            report["columns"][column] = {
                "inferred_type": inferred,
                "non_missing_count": len(present),
                "missing_count": len(values) - len(present),
                "unique_count": len(set(present)),
            }
        return report

    def _numeric_pairs(
        self,
        rows: list[dict[str, str]],
        x: str,
        y: str,
        missing_values: set[str],
    ) -> tuple[list[float], list[float], int]:
        if not rows or x not in rows[0] or y not in rows[0]:
            raise AnalysisError(f"Unknown numeric column pair: {x!r}, {y!r}")
        xs: list[float] = []
        ys: list[float] = []
        dropped = 0
        for row in rows:
            xv, yv = row[x], row[y]
            if xv in missing_values or yv in missing_values:
                dropped += 1
                continue
            try:
                xnum, ynum = float(xv), float(yv)
            except ValueError as exc:
                raise AnalysisError(f"Non-numeric value in {x!r} or {y!r}") from exc
            if not (math.isfinite(xnum) and math.isfinite(ynum)):
                dropped += 1
                continue
            xs.append(xnum)
            ys.append(ynum)
        return xs, ys, dropped

    def _pearson_correlation(
        self,
        rows: list[dict[str, str]],
        parameters: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        x = str(parameters["x"])
        y = str(parameters["y"])
        min_rows = int(parameters.get("min_rows", 3))
        xs, ys, dropped = self._numeric_pairs(
            rows, x, y, set(preprocessing["missing_values"])
        )
        if len(xs) < min_rows:
            raise AnalysisError(f"Need at least {min_rows} complete rows; found {len(xs)}")
        x_mean = math.fsum(xs) / len(xs)
        y_mean = math.fsum(ys) / len(ys)
        sxx = math.fsum((value - x_mean) ** 2 for value in xs)
        syy = math.fsum((value - y_mean) ** 2 for value in ys)
        sxy = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(xs, ys))
        if sxx == 0 or syy == 0:
            raise AnalysisError("Pearson correlation is undefined for a constant column")
        r = sxy / math.sqrt(sxx * syy)
        r = max(-1.0, min(1.0, r))
        slope = sxy / sxx
        intercept = y_mean - slope * x_mean
        outputs = {
            "analysis": "pearson_correlation",
            "x": x,
            "y": y,
            "n": len(xs),
            "pearson_r": r,
            "abs_r": abs(r),
            "direction": "positive" if r > 0 else "negative" if r < 0 else "zero",
            "slope_y_on_x": slope,
            "intercept_y_on_x": intercept,
        }
        diagnostics = {
            "rows_total": len(rows),
            "rows_used": len(xs),
            "rows_dropped_missing_or_nonfinite": dropped,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_sum_squared_deviation": sxx,
            "y_sum_squared_deviation": syy,
        }
        return outputs, diagnostics

    def _column_summary(
        self,
        rows: list[dict[str, str]],
        parameters: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        column = str(parameters["column"])
        if column not in rows[0]:
            raise AnalysisError(f"Unknown column: {column}")
        missing_values = set(preprocessing["missing_values"])
        values: list[float] = []
        dropped = 0
        for row in rows:
            raw = row[column]
            if raw in missing_values:
                dropped += 1
                continue
            try:
                number = float(raw)
            except ValueError as exc:
                raise AnalysisError(f"Non-numeric value in {column!r}") from exc
            if not math.isfinite(number):
                dropped += 1
                continue
            values.append(number)
        if not values:
            raise AnalysisError("No usable numeric observations")
        outputs = {
            "analysis": "column_summary",
            "column": column,
            "n": len(values),
            "mean": math.fsum(values) / len(values),
            "minimum": min(values),
            "maximum": max(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        return outputs, {
            "rows_total": len(rows),
            "rows_used": len(values),
            "rows_dropped_missing_or_nonfinite": dropped,
        }

    def _group_mean_difference(
        self,
        rows: list[dict[str, str]],
        parameters: dict[str, Any],
        preprocessing: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        group_column = str(parameters["group"])
        outcome_column = str(parameters["outcome"])
        if group_column not in rows[0] or outcome_column not in rows[0]:
            raise AnalysisError("Unknown group or outcome column")
        missing_values = set(preprocessing["missing_values"])
        groups: dict[str, list[float]] = {}
        dropped = 0
        for row in rows:
            group = row[group_column]
            raw = row[outcome_column]
            if group in missing_values or raw in missing_values:
                dropped += 1
                continue
            try:
                number = float(raw)
            except ValueError as exc:
                raise AnalysisError(f"Non-numeric outcome in {outcome_column!r}") from exc
            if not math.isfinite(number):
                dropped += 1
                continue
            groups.setdefault(group, []).append(number)
        group_a = str(parameters.get("group_a", "")) or None
        group_b = str(parameters.get("group_b", "")) or None
        if group_a is None or group_b is None:
            names = sorted(groups)
            if len(names) != 2:
                raise AnalysisError(
                    "group_a and group_b are required unless exactly two groups are present"
                )
            group_a, group_b = names
        if group_a not in groups or group_b not in groups:
            raise AnalysisError("Requested groups are absent from the usable rows")
        a, b = groups[group_a], groups[group_b]
        if not a or not b:
            raise AnalysisError("Both groups need at least one usable observation")
        mean_a = math.fsum(a) / len(a)
        mean_b = math.fsum(b) / len(b)
        variance_a = statistics.variance(a) if len(a) > 1 else 0.0
        variance_b = statistics.variance(b) if len(b) > 1 else 0.0
        pooled_denominator = len(a) + len(b) - 2
        pooled_variance = (
            ((len(a) - 1) * variance_a + (len(b) - 1) * variance_b) / pooled_denominator
            if pooled_denominator > 0
            else 0.0
        )
        pooled_sd = math.sqrt(max(0.0, pooled_variance))
        difference = mean_a - mean_b
        standardized = difference / pooled_sd if pooled_sd > 0 else None
        outputs = {
            "analysis": "group_mean_difference",
            "group_column": group_column,
            "outcome_column": outcome_column,
            "group_a": group_a,
            "group_b": group_b,
            "n_a": len(a),
            "n_b": len(b),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "difference_a_minus_b": difference,
            "pooled_standard_deviation": pooled_sd,
            "standardized_difference": standardized,
        }
        diagnostics = {
            "rows_total": len(rows),
            "rows_used": len(a) + len(b),
            "rows_dropped_missing_or_nonfinite": dropped,
            "observed_groups": sorted(groups),
            "variance_a": variance_a,
            "variance_b": variance_b,
        }
        return outputs, diagnostics

    # ------------------------------------------------------------------
    # Assessments, persistence, and integrity
    # ------------------------------------------------------------------
    def _assess_claims(
        self,
        tests: Iterable[AnalysisClaimTest],
        outputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        assessments: list[dict[str, Any]] = []
        for test in tests:
            observed = self._metric_value(outputs, test.metric_path)
            support = test.support_condition.evaluate(observed)
            refute = (
                test.refute_condition.evaluate(observed)
                if test.refute_condition is not None
                else False
            )
            if support and refute:
                raise AnalysisError(
                    f"Support and refute conditions both matched for {test.claim_id}"
                )
            outcome = (
                AnalysisOutcome.SUPPORT
                if support
                else AnalysisOutcome.REFUTE
                if refute
                else AnalysisOutcome.INCONCLUSIVE
            )
            assessments.append(
                {
                    "claim_id": test.claim_id,
                    "metric_path": test.metric_path,
                    "metric_value": observed,
                    "outcome": outcome.value,
                    "support_condition": asdict(test.support_condition),
                    "refute_condition": (
                        asdict(test.refute_condition)
                        if test.refute_condition is not None
                        else None
                    ),
                    "confidence": test.confidence,
                    "rationale": test.rationale,
                }
            )
        return assessments

    def _metric_value(self, value: dict[str, Any], path: str) -> Any:
        current: Any = value
        for component in path.split("."):
            if not isinstance(current, dict) or component not in current:
                raise AnalysisError(f"Unknown metric path: {path}")
            current = current[component]
        return current

    def _receipt_payload(self, **values: Any) -> dict[str, Any]:
        status = values["status"]
        if isinstance(status, AnalysisStatus):
            status = status.value
        return {
            "receipt_id": values["receipt_id"],
            "analysis_type": values["analysis_type"],
            "status": status,
            "dataset_uri": values["dataset_uri"],
            "dataset_hash": values["dataset_hash"],
            "dataset_size_bytes": values["dataset_size_bytes"],
            "code_hash": values["code_hash"],
            "code_identity": values["code_identity"],
            "environment": values["environment"],
            "schema": values["schema"],
            "preprocessing": values["preprocessing"],
            "parameters": values["parameters"],
            "outputs": values["outputs"],
            "diagnostics": values["diagnostics"],
            "metadata": values["metadata"],
            "parent_receipt_id": values["parent_receipt_id"],
            "comparison": values["comparison"],
            "assessments": values["assessments"],
            "started_at": values["started_at"],
            "completed_at": values["completed_at"],
        }

    def _insert_receipt(self, payload: dict[str, Any], receipt_hash: str) -> None:
        self.ledger.db.conn.execute(
            """INSERT INTO analysis_receipts
               (id, analysis_type, status, dataset_uri, dataset_hash, dataset_size_bytes,
                code_hash, code_identity, environment_json, schema_json,
                preprocessing_json, parameters_json, outputs_json, diagnostics_json,
                metadata_json, parent_receipt_id, comparison_json, receipt_hash,
                started_at, completed_at, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                payload["receipt_id"],
                payload["analysis_type"],
                payload["status"],
                payload["dataset_uri"],
                payload["dataset_hash"],
                payload["dataset_size_bytes"],
                payload["code_hash"],
                payload["code_identity"],
                _stable_json(payload["environment"]),
                _stable_json(payload["schema"]),
                _stable_json(payload["preprocessing"]),
                _stable_json(payload["parameters"]),
                _stable_json(payload["outputs"]),
                _stable_json(payload["diagnostics"]),
                _stable_json(payload["metadata"]),
                payload["parent_receipt_id"],
                _stable_json(payload["comparison"]),
                receipt_hash,
                payload["started_at"],
                payload["completed_at"],
                _utcnow(),
            ),
        )

    def _write_receipt_artifact(
        self, receipt_id: str, payload: dict[str, Any], receipt_hash: str
    ) -> dict[str, Any]:
        directory = self.artifact_root / receipt_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "receipt.json"
        content = json.dumps(
            {"receipt_hash": receipt_hash, "receipt": payload},
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
        path.write_text(content, encoding="utf-8")
        digest, size = _sha256_file(path)
        return {
            "id": _new_id("art"),
            "role": "receipt_document",
            "path": str(path),
            "content_hash": digest,
            "size_bytes": size,
            "media_type": "application/json",
            "metadata": {},
        }

    def _insert_artifact(self, receipt_id: str, artifact: dict[str, Any]) -> None:
        self.ledger.db.conn.execute(
            """INSERT INTO analysis_artifacts
               (id, receipt_id, role, path, content_hash, size_bytes, media_type,
                metadata_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                artifact["id"],
                receipt_id,
                artifact["role"],
                artifact["path"],
                artifact["content_hash"],
                artifact["size_bytes"],
                artifact["media_type"],
                _stable_json(artifact["metadata"]),
                _utcnow(),
            ),
        )

    def _create_receipt_evidence(
        self,
        payload: dict[str, Any],
        receipt_hash: str,
        dataset_hash: str,
        actor: str,
        actor_role: ActorRole,
    ) -> str:
        outcomes = [item["outcome"] for item in payload["assessments"]]
        summary = (
            f"{payload['analysis_type']} completed on dataset sha256:{dataset_hash}; "
            f"claim outcomes={outcomes or ['none']}"
        )
        return self.ledger.add_evidence(
            f"analysis://{payload['receipt_id']}",
            summary,
            source_kind=EvidenceKind.DATASET_ANALYSIS_RECEIPT,
            independence_key=f"dataset:{dataset_hash}",
            content=_stable_json(payload),
            metadata={
                "receipt_id": payload["receipt_id"],
                "receipt_hash": receipt_hash,
                "analysis_type": payload["analysis_type"],
                "code_hash": payload["code_hash"],
                "dataset_hash": dataset_hash,
            },
            actor=actor,
            actor_role=actor_role,
        )

    def _insert_assessments(
        self,
        receipt_id: str,
        assessments: list[dict[str, Any]],
        evidence_id: str | None,
        actor: str,
        actor_role: ActorRole,
    ) -> None:
        for position, item in enumerate(assessments):
            assessment_id = _new_id("asm")
            linked_evidence = evidence_id if item["outcome"] != AnalysisOutcome.INCONCLUSIVE.value else None
            self.ledger.db.conn.execute(
                """INSERT INTO analysis_claim_assessments
                   (id, receipt_id, position, claim_id, metric_path, metric_value_json, outcome,
                    support_condition_json, refute_condition_json, confidence,
                    rationale, evidence_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    assessment_id,
                    receipt_id,
                    position,
                    item["claim_id"],
                    item["metric_path"],
                    _stable_json(item["metric_value"]),
                    item["outcome"],
                    _stable_json(item["support_condition"]),
                    _stable_json(item["refute_condition"]),
                    item["confidence"],
                    item["rationale"],
                    linked_evidence,
                    _utcnow(),
                ),
            )
            if linked_evidence is not None:
                stance = (
                    Stance.SUPPORT
                    if item["outcome"] == AnalysisOutcome.SUPPORT.value
                    else Stance.REFUTE
                )
                self.ledger.attest(
                    item["claim_id"],
                    linked_evidence,
                    stance,
                    confidence=item["confidence"],
                    actor=actor,
                    actor_role=actor_role,
                )
            self.ledger._event(
                "claim",
                item["claim_id"],
                "ANALYSIS_CLAIM_ASSESSED",
                {
                    "assessment_id": assessment_id,
                    "receipt_id": receipt_id,
                    "outcome": item["outcome"],
                    "metric_path": item["metric_path"],
                    "metric_value": item["metric_value"],
                },
                actor,
                actor_role,
            )

    def _assessment_row(
        self, row: Any, *, include_runtime_fields: bool = True
    ) -> dict[str, Any]:
        result = dict(row)
        result["metric_value"] = json.loads(result.pop("metric_value_json"))
        result["support_condition"] = json.loads(result.pop("support_condition_json"))
        result["refute_condition"] = json.loads(result.pop("refute_condition_json"))
        if not include_runtime_fields:
            result.pop("id", None)
            result.pop("receipt_id", None)
            result.pop("position", None)
            result.pop("evidence_id", None)
            result.pop("created_at", None)
        return result

    def _artifact_row(self, row: Any) -> dict[str, Any]:
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def _environment(self) -> dict[str, Any]:
        return {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "orbita_version": "0.5.0",
            "analyzer_api_version": ANALYZER_API_VERSION,
        }

    def _code_hash(self) -> str:
        # Hash the complete module rather than only one analyzer function so
        # shared parsing and comparison logic cannot change invisibly.
        source_path = Path(inspect.getsourcefile(DatasetAnalysisRuntime) or __file__)
        return _sha256_file(source_path)[0]

    def _compare_values(
        self,
        expected: Any,
        observed: Any,
        *,
        rel_tol: float,
        abs_tol: float,
        path: str = "$",
    ) -> tuple[bool, list[dict[str, Any]]]:
        differences: list[dict[str, Any]] = []
        if isinstance(expected, bool) or isinstance(observed, bool):
            if expected != observed:
                differences.append({"path": path, "expected": expected, "observed": observed})
            return not differences, differences
        if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
            if not math.isclose(float(expected), float(observed), rel_tol=rel_tol, abs_tol=abs_tol):
                differences.append({"path": path, "expected": expected, "observed": observed})
            return not differences, differences
        if isinstance(expected, dict) and isinstance(observed, dict):
            for key in sorted(set(expected) | set(observed)):
                child = f"{path}.{key}"
                if key not in expected or key not in observed:
                    differences.append(
                        {
                            "path": child,
                            "expected": expected.get(key, "<missing>"),
                            "observed": observed.get(key, "<missing>"),
                        }
                    )
                    continue
                _, child_differences = self._compare_values(
                    expected[key], observed[key], rel_tol=rel_tol, abs_tol=abs_tol, path=child
                )
                differences.extend(child_differences)
            return not differences, differences
        if isinstance(expected, list) and isinstance(observed, list):
            if len(expected) != len(observed):
                differences.append(
                    {"path": path, "expected_length": len(expected), "observed_length": len(observed)}
                )
            for index, (left, right) in enumerate(zip(expected, observed)):
                _, child_differences = self._compare_values(
                    left,
                    right,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    path=f"{path}[{index}]",
                )
                differences.extend(child_differences)
            return not differences, differences
        if expected != observed:
            differences.append({"path": path, "expected": expected, "observed": observed})
        return not differences, differences
