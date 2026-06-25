from __future__ import annotations

import base64
import csv
import io
import logging
import os
import secrets
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse

logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

from orbita import EvidenceKind, Stance

from .graph_ui import GRAPH_HTML, build_events, build_graph_data
from .metrics import higher_is_better, validate_metric
from .service import ResearchMVP


_data_dir = Path("/data") if Path("/data").exists() else Path(".")
DB_PATH = Path(os.getenv("ORBITA_MVP_DB", str(_data_dir / "orbita_mvp.db")))
WORKSPACE = Path(os.getenv("ORBITA_MVP_WORKSPACE", str(_data_dir / "orbita_workspace")))
# Prefer an explicit build-time SHA; fall back to Railway's injected variable.
_GIT_COMMIT = os.getenv("GIT_COMMIT_SHA", os.getenv("RAILWAY_GIT_COMMIT_SHA", "unknown"))
# Test endpoints are disabled by default. Set ORBITA_ENABLE_TEST_ENDPOINTS=true on staging only.
# Production must omit this variable or set it to any value other than "true".
_TEST_ENDPOINTS_ENABLED = os.getenv("ORBITA_ENABLE_TEST_ENDPOINTS", "").lower() == "true"
service = ResearchMVP(DB_PATH, WORKSPACE)

@asynccontextmanager
async def _lifespan(app):
    if _TEST_ENDPOINTS_ENABLED:
        logger.warning("Test endpoints ENABLED (ORBITA_ENABLE_TEST_ENDPOINTS=true) — staging only")
    else:
        logger.info("Test endpoints disabled (production-safe)")
    yield


app = FastAPI(
    title="Orbita Research MVP",
    version="0.2.1",
    description=(
        "Upload research material, compile an explicit governed plan, run frozen "
        "discovery candidates, persist findings in an epistemic graph, and produce "
        "an expert-readable dossier."
    ),
    lifespan=_lifespan,
)

_DEMO_USER = os.getenv("ORBITA_DEMO_USER", "")
_DEMO_PASS = os.getenv("ORBITA_DEMO_PASS", "")


@app.middleware("http")
async def basic_auth(request: Request, call_next):
    if not (_DEMO_USER and _DEMO_PASS):
        return await call_next(request)
    if request.url.path in ("/health", "/healthz"):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            user, _, pw = decoded.partition(":")
            if secrets.compare_digest(user, _DEMO_USER) and secrets.compare_digest(pw, _DEMO_PASS):
                return await call_next(request)
        except Exception:
            pass
    return Response(
        content="Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": 'Basic realm="Orbita Demo"'},
    )


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "Orbita Research MVP", "version": "0.1.0", "docs": "/docs"}


class CaseCreate(BaseModel):
    name: str = "Untitled research case"
    goal: str = ""
    domain_hint: str | None = None


class CompileRequest(BaseModel):
    max_candidates: int = Field(default=60, ge=1, le=500)
    target_transform: str | None = Field(default=None, description="Monotone transform for numeric outcomes before fitting: 'log1p' or null")
    outcome_domain: str | None = Field(default=None, description="Domain constraint for predictions: 'nonneg' clips output to [0, inf), or null")
    evaluation_metric: str = Field(default="r2", description="Metric used for model selection and final validation: r2, rmse, mae, rmsle")
    confirmation_fraction: float = Field(default=0.25, ge=0.05, le=0.5, description="Fraction of rows reserved for model-selection (selection partition)")
    final_validation_fraction: float = Field(default=0.15, ge=0.05, le=0.4, description="Fraction of rows reserved for final unbiased validation (never touched during model selection)")


class ApproveRequest(BaseModel):
    reviewer: str = "local-user"


class RunRequest(BaseModel):
    plan_id: str | None = None
    auto_approve: bool = False


class ExternalPlanRequest(BaseModel):
    plan: dict[str, Any]
    compiler: str = "external-ai"


class PlanRevisionRequest(BaseModel):
    plan: dict[str, Any]
    compiler: str = "human-review"


class SupersedeRequest(BaseModel):
    new_statement: str
    rationale: str


class RevokeEvidenceRequest(BaseModel):
    rationale: str


class ResolveRequest(BaseModel):
    resolution: dict[str, Any]
    actor: str = "local-user"


class ContradictionRequest(BaseModel):
    claim_a: str
    claim_b: str
    rationale: str


class EvidenceRequest(BaseModel):
    source_uri: str
    excerpt: str
    source_kind: str = "dataset"
    independence_key: str
    stance: str = "support"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DerivationRequest(BaseModel):
    premise_claim_ids: list[str]
    rule: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _guard(callable_, *args, **kwargs):
    try:
        return callable_(*args, **kwargs)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": "0.2.1",
        "git_commit": _GIT_COMMIT,
        "plan_schema": "orbita-research-plan/0.3",
        "db": str(DB_PATH.resolve()),
        "workspace": str(WORKSPACE.resolve()),
    }


@app.post("/cases")
def create_case(request: CaseCreate) -> dict[str, Any]:
    return _guard(service.create_case, name=request.name, goal=request.goal, domain_hint=request.domain_hint)


@app.get("/cases")
def list_cases() -> dict[str, Any]:
    return {"cases": service.store.list_cases()}


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    return _guard(service.store.get_case, case_id)


class CaseUpdate(BaseModel):
    name: str


@app.patch("/cases/{case_id}")
def update_case(case_id: str, request: CaseUpdate) -> dict[str, Any]:
    _guard(service.store.get_case, case_id)
    service.store.ledger.db.conn.execute(
        "UPDATE research_cases SET name = ? WHERE id = ?", (request.name.strip(), case_id)
    )
    service.store.ledger.db.conn.commit()
    return _guard(service.store.get_case, case_id)


class FileUpdate(BaseModel):
    original_name: str


@app.patch("/cases/{case_id}/files/{file_id}")
def update_file(case_id: str, file_id: str, request: FileUpdate) -> dict[str, Any]:
    _guard(service.store.get_case, case_id)
    service.store.ledger.db.conn.execute(
        "UPDATE case_files SET original_name = ? WHERE id = ? AND case_id = ?",
        (request.original_name.strip(), file_id, case_id),
    )
    service.store.ledger.db.conn.commit()
    return _guard(service.store.get_case, case_id)


@app.post("/cases/{case_id}/files")
def upload_file(case_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(file.filename or "upload.bin").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = Path(temp.name)
    try:
        # Preserve the user-facing filename for the ingestor.
        named = temp_path.with_name(Path(file.filename or temp_path.name).name)
        if named.exists():
            named.unlink()
        temp_path.rename(named)
        return _guard(service.add_file, case_id, named)
    finally:
        for path in {temp_path, locals().get("named", temp_path)}:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass


@app.post("/cases/{case_id}/compile")
def compile_case(case_id: str, request: CompileRequest) -> dict[str, Any]:
    return _guard(
        service.compile_case,
        case_id,
        max_candidates=request.max_candidates,
        target_transform=request.target_transform,
        outcome_domain=request.outcome_domain,
        evaluation_metric=request.evaluation_metric,
        confirmation_fraction=request.confirmation_fraction,
        final_validation_fraction=request.final_validation_fraction,
    )


@app.post("/cases/{case_id}/plans")
def submit_external_plan(case_id: str, request: ExternalPlanRequest) -> dict[str, Any]:
    return _guard(service.submit_external_plan, case_id, request.plan, compiler=request.compiler)


@app.post("/plans/{plan_id}/approve")
def approve_plan(plan_id: str, request: ApproveRequest) -> dict[str, Any]:
    return _guard(service.approve_plan, plan_id, reviewer=request.reviewer)


@app.post("/plans/{plan_id}/revise")
def revise_plan(plan_id: str, request: PlanRevisionRequest) -> dict[str, Any]:
    return _guard(service.revise_plan, plan_id, request.plan, compiler=request.compiler)


@app.post("/cases/{case_id}/run")
def run_case(case_id: str, request: RunRequest) -> dict[str, Any]:
    return _guard(service.run_case, case_id, plan_id=request.plan_id, auto_approve=request.auto_approve)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    return _guard(service.store.get_run, run_id)


@app.get("/cases/{case_id}/context")
def ai_context(case_id: str) -> dict[str, Any]:
    case = _guard(service.store.get_case, case_id)
    return {
        "case_id": case_id,
        "goal": case.get("goal", ""),
        "mode": case.get("mode"),
        "files": [
            {
                "id": item["id"],
                "name": item["original_name"],
                "artifact_kind": item["artifact_kind"],
                "parse_status": item["parse_status"],
                "profile": item["profile"],
                "sha256": item["sha256"],
            }
            for item in case.get("files", [])
        ],
        "instructions": (
            "An external AI may propose an explicit plan through POST /cases/{case_id}/plans. "
            "Do not invent columns. Mark associations as non-causal. Include frozen candidates "
            "and thresholds for held-out and cross-seed checks."
        ),
    }


@app.get("/cases/{case_id}/report")
def latest_report(case_id: str):
    case = _guard(service.store.get_case, case_id)
    for run in case.get("runs", []):
        artifact = run.get("result", {}).get("reports", {}).get("html")
        if artifact and Path(artifact["path"]).exists():
            return FileResponse(artifact["path"], media_type="text/html", filename=f"{case_id}_research_dossier.html")
    raise HTTPException(status_code=404, detail="No completed report exists for this case")


@app.get("/cases/{case_id}/download/{role}")
def download_report_artifact(case_id: str, role: str):
    case = _guard(service.store.get_case, case_id)
    for run in case.get("runs", []):
        artifact = run.get("result", {}).get("reports", {}).get(role)
        if artifact and Path(artifact["path"]).exists():
            media = {"html": "text/html", "markdown": "text/markdown", "json": "application/json"}.get(role, "application/octet-stream")
            return FileResponse(artifact["path"], media_type=media, filename=Path(artifact["path"]).name)
    raise HTTPException(status_code=404, detail=f"No {role} report artifact exists for this case")


@app.get("/runs/{run_id}/ledger")
def download_ledger(run_id: str):
    """Download the hash-chained JSONL discovery ledger for a run."""
    run = _guard(service.store.get_run, run_id)
    ledger_path = run.get("result", {}).get("ledger_path") or run.get("ledger_path")
    if not ledger_path or not Path(ledger_path).exists():
        raise HTTPException(status_code=404, detail="Ledger file not found for this run")
    return FileResponse(ledger_path, media_type="application/x-ndjson", filename=f"{run_id}_ledger.jsonl")


@app.post("/runs/{run_id}/predict")
async def predict(
    run_id: str,
    file: UploadFile = File(...),
    target_column: str = Query(..., description="Name of the outcome column to predict (must match training data)"),
    identifier_column: str = Query("row_id", description="Column to use as row identifier in output (not used as a feature)"),
) -> StreamingResponse:
    """Apply committed survivors from a run to a new dataset; returns row_id,predicted_y CSV.

    Re-reads the original training CSV and refits each surviving linear model on the full
    training set to produce final coefficients. For group-difference survivors, uses the
    group means from the full training set. The top-scoring survivor predicting target_column
    is used. Row identifier is preserved in output; it is never used as a predictor.
    """
    run = _guard(service.store.get_run, run_id)
    result = run.get("result", {})
    findings = result.get("findings", [])
    if not findings:
        raise HTTPException(status_code=422, detail="Run has no findings (run may have failed)")

    # Identify survivors whose outcome is the requested target
    accepted = {"supported", "challenged", "provisional"}
    target_survivors = [
        f for f in findings
        if f["final_status"] in accepted
        and not any(a["killed"] for a in f["falsifications"])
        and f["candidate"]["payload"].get("outcome") == target_column
    ]
    if not target_survivors:
        raise HTTPException(
            status_code=422,
            detail=f"No surviving candidate predicts '{target_column}'. "
                   f"Available outcomes in survivors: "
                   f"{sorted({f['candidate']['payload'].get('outcome') for f in findings if f['final_status'] in accepted and not any(a['killed'] for a in f['falsifications'])})}"
        )

    # Re-read training data plan so we can read the frozen evaluation metric.
    plan_record = service.store.get_plan(run["plan_id"]) if run.get("plan_id") else None
    if not plan_record:
        case_id_run = run.get("case_id")
        if case_id_run:
            case_obj = service.store.get_case(case_id_run)
            if case_obj.get("plans"):
                plan_record = case_obj["plans"][0]
    if not plan_record:
        raise HTTPException(status_code=422, detail="Cannot locate training plan for this run")

    plan_body = plan_record["plan"]

    # Load the pre-frozen selected_model_id.
    # final_validation_metric_score is report-only and must not influence
    # which model is served — that decision was frozen before fvs was computed.
    selected_models = result.get("selected_models", {})
    selected_info = selected_models.get(target_column)

    if selected_info:
        # Modern runs: read the frozen selection winner directly.
        selected_model_id = selected_info["selected_model_id"]
        best = next(
            (f for f in target_survivors if f["candidate"]["id"] == selected_model_id),
            None,
        )
        if best is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Frozen selected_model_id '{selected_model_id}' is not present "
                    f"among committed survivors for outcome '{target_column}'. "
                    "The run result may be corrupted."
                ),
            )
    else:
        # Legacy runs without selected_models: rank by selection_metric_score,
        # then verdict score.  Never use final_validation_metric_score.
        evaluation_metric = plan_body.get("evaluation_metric") or "r2"
        try:
            validate_metric(evaluation_metric)
        except ValueError:
            evaluation_metric = "r2"
        hib_flag = higher_is_better(evaluation_metric)

        def _legacy_sort_key(f):
            sc = f.get("selection_metric_score")
            if sc is None or not isinstance(sc, (int, float)):
                sc = float(f["verdict"]["score"])
            return (sc if hib_flag else -sc, f["candidate"]["id"])

        best = max(target_survivors, key=_legacy_sort_key)

    payload = best["candidate"]["payload"]
    kind = payload["kind"]

    target_transform = plan_body.get("target_transform") or None
    outcome_domain = plan_body.get("outcome_domain") or None

    from .table_domain import _invert_transform

    # Load the frozen model artifact — never refit (lstsq) at inference time.
    model_artifacts = result.get("model_artifacts", {})
    artifact_info = model_artifacts.get(target_column, {})
    artifact_path_str = artifact_info.get("model_artifact_path")
    model: dict
    if artifact_path_str and Path(artifact_path_str).exists():
        from .model_artifact import load_model_artifact, model_from_artifact
        try:
            artifact = load_model_artifact(artifact_path_str)
        except Exception as _art_err:
            raise HTTPException(
                status_code=500,
                detail=f"Model artifact integrity check failed: {_art_err}"
            )
        model = model_from_artifact(artifact, payload)
        if not model.get("valid"):
            raise HTTPException(
                status_code=422,
                detail=f"Frozen artifact for '{target_column}' could not be reconstructed (kind={kind!r})"
            )
    else:
        # No artifact — this run predates artifact serialization or artifact was lost.
        # Refuse to silently refit; require the artifact to be present.
        if artifact_path_str:
            artifact_missing_hint = (
                f"Model artifact expected at {artifact_path_str!r} but file is missing. "
                "The volume may have been remounted or the artifact was deleted."
            )
        else:
            artifact_missing_hint = (
                "This historical run predates serialized deployment artifacts and cannot "
                "be used for new inference. Create a new case and run under plan schema 0.3."
            )
        raise HTTPException(status_code=422, detail=artifact_missing_hint)

    # Load test CSV
    suffix = Path(file.filename or "test.csv").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)
    try:
        test_df = pd.read_csv(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if identifier_column not in test_df.columns:
        raise HTTPException(status_code=422, detail=f"identifier_column '{identifier_column}' not found in test file")

    # Generate predictions row by row
    row_ids = test_df[identifier_column].tolist()
    raw_preds: np.ndarray | None = None

    if kind == "linear_association":
        predictor = payload["predictor"]
        if predictor not in test_df.columns:
            raise HTTPException(status_code=422, detail=f"Predictor column '{predictor}' not found in test file")
        xs = pd.to_numeric(test_df[predictor], errors="coerce").to_numpy(float)
        raw_preds = model["intercept"] + model["slope"] * xs

    elif kind == "composite_linear":
        predictors = model["predictors"]
        missing = [p for p in predictors if p not in test_df.columns]
        if missing:
            raise HTTPException(status_code=422, detail=f"Composite predictor columns missing from test file: {missing}")
        X = np.column_stack([np.ones(len(test_df))] + [
            pd.to_numeric(test_df[p], errors="coerce").to_numpy(float) for p in predictors
        ])
        beta = np.array([model["intercept"]] + [model["coefficients"][p] for p in predictors])
        raw_preds = X @ beta

    elif kind == "group_difference":
        group_col = payload["group"]
        if group_col not in test_df.columns:
            raise HTTPException(status_code=422, detail=f"Group column '{group_col}' not found in test file")
        raw_preds = np.array([
            model["means"].get(str(g), model["overall"])
            for g in test_df[group_col].astype(str)
        ], dtype=float)
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported survivor kind for prediction: {kind}")

    # Invert transform and apply domain constraint
    final_preds = _invert_transform(raw_preds, target_transform)
    if outcome_domain == "nonneg":
        final_preds = np.clip(final_preds, 0, None)

    predictions: list[float | None] = [
        None if not np.isfinite(v) else round(float(v), 8) for v in final_preds
    ]

    # Build CSV output
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([identifier_column, f"predicted_{target_column}"])
    for rid, pred in zip(row_ids, predictions):
        writer.writerow([rid, pred if pred is not None else ""])
    output.seek(0)

    provenance_header = (
        f"run_id={run_id}; "
        f"candidate_id={best['candidate']['id']}; "
        f"kind={kind}; "
        f"verdict_score={best['verdict']['score']:.6f}; "
        f"candidate_sha256={best['sha256']}"
    )
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{run_id}_predictions.csv"',
            "X-Orbita-Provenance": provenance_header,
        },
    )


@app.get("/claims/{claim_id}/history")
def claim_history(claim_id: str) -> dict[str, Any]:
    return _guard(service.claim_history, claim_id)


@app.get("/cases/{case_id}/claims")
def case_claims(case_id: str) -> dict[str, Any]:
    _guard(service.store.get_case, case_id)
    return {
        "claims": service.store.case_claims(case_id),
        "counts": service.store.case_claim_counts(case_id),
    }


@app.get("/claims/{claim_id}/impact")
def claim_impact(claim_id: str) -> dict[str, Any]:
    return _guard(service.memory.impact_view, claim_id)


@app.post("/claims/{claim_id}/evidence")
def add_claim_evidence(claim_id: str, request: EvidenceRequest) -> dict[str, Any]:
    try:
        source_kind = EvidenceKind(request.source_kind)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EvidenceKind)
        raise HTTPException(status_code=400, detail=f"Unknown source_kind. Allowed: {allowed}") from exc
    try:
        stance = Stance(request.stance)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in Stance)
        raise HTTPException(status_code=400, detail=f"Unknown stance. Allowed: {allowed}") from exc
    return _guard(
        service.memory.add_evidence,
        claim_id,
        source_uri=request.source_uri,
        excerpt=request.excerpt,
        source_kind=source_kind,
        independence_key=request.independence_key,
        stance=stance,
        confidence=request.confidence,
        content=request.content,
        metadata=request.metadata,
    )


@app.post("/claims/{claim_id}/derive")
def add_derivation(claim_id: str, request: DerivationRequest) -> dict[str, Any]:
    return _guard(
        service.memory.add_derivation,
        claim_id,
        request.premise_claim_ids,
        rule=request.rule,
        metadata=request.metadata,
    )


@app.post("/contradictions")
def add_contradiction(request: ContradictionRequest) -> dict[str, Any]:
    return _guard(
        service.add_contradiction,
        request.claim_a,
        request.claim_b,
        rationale=request.rationale,
    )


@app.post("/claims/{claim_id}/supersede")
def supersede_claim(claim_id: str, request: SupersedeRequest) -> dict[str, Any]:
    return _guard(service.supersede_claim, claim_id, new_statement=request.new_statement, rationale=request.rationale)


@app.post("/evidence/{evidence_id}/revoke")
def revoke_evidence(evidence_id: str, request: RevokeEvidenceRequest) -> dict[str, Any]:
    return _guard(service.revoke_evidence, evidence_id, rationale=request.rationale)


@app.get("/reexamination")
def reexamination() -> dict[str, Any]:
    return {"items": service.reexamination_queue()}


@app.post("/reexamination/{queue_id}/resolve")
def resolve_reexamination(queue_id: str, request: ResolveRequest) -> dict[str, Any]:
    return _guard(service.memory.resolve_reexamination, queue_id, resolution=request.resolution, actor=request.actor)


@app.get("/graph", response_class=HTMLResponse)
def graph_page() -> str:
    return GRAPH_HTML


@app.get("/cases/{case_id}/graph")
def case_graph(case_id: str) -> dict[str, Any]:
    _guard(service.store.get_case, case_id)
    return build_graph_data(case_id, service.ledger.db.conn)


@app.get("/cases/{case_id}/events")
def case_events(case_id: str, since: str = "") -> dict[str, Any]:
    _guard(service.store.get_case, case_id)
    return build_events(case_id, service.ledger.db.conn, since)


@app.post("/admin/test/tamper-artifact")
def tamper_artifact_for_test(
    run_id: str = Query(..., description="Run ID whose deployment artifact to corrupt"),
    target_column: str = Query(..., description="Outcome column"),
    x_test_token: str = Query(None, alias="test_token"),
) -> dict[str, Any]:
    """Test-only endpoint: corrupt a deployment artifact to verify integrity checks.
    Only active when ORBITA_ENABLE_TEST_ENDPOINTS=true. Requires test_token matching
    ORBITA_TEST_TOKEN env var. Returns 404 when test endpoints are disabled.
    """
    if not _TEST_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")
    _test_tok = os.getenv("ORBITA_TEST_TOKEN", "")
    if not _test_tok or x_test_token != _test_tok:
        raise HTTPException(status_code=403, detail="Forbidden: requires valid test_token")
    run = _guard(service.store.get_run, run_id)
    result = run.get("result", {})
    art_info = result.get("model_artifacts", {}).get(target_column, {})
    art_path_str = art_info.get("model_artifact_path")
    if not art_path_str or not Path(art_path_str).exists():
        raise HTTPException(status_code=404, detail=f"Deployment artifact not found at {art_path_str!r}")
    art_path = Path(art_path_str)
    import json as _json
    original_bytes = art_path.read_bytes()
    artifact = _json.loads(original_bytes)
    backup_path = art_path.with_suffix(".json.backup")
    backup_path.write_bytes(original_bytes)
    coefs = artifact.get("coefficients", {})
    if coefs:
        first_key = next(iter(coefs))
        original_val = coefs[first_key]
        artifact["coefficients"][first_key] = original_val + 9999.0
        corrupted_field = first_key
    else:
        original_val = artifact.get("intercept", 0.0)
        artifact["intercept"] = original_val + 9999.0
        corrupted_field = "intercept"
    art_path.write_text(_json.dumps(artifact, indent=2), encoding="utf-8")
    return {
        "status": "corrupted",
        "artifact_path": str(art_path),
        "backup_path": str(backup_path),
        "expected_sha256": art_info.get("model_artifact_sha256"),
        "corrupted_field": corrupted_field,
        "original_value": original_val,
        "corrupted_value": original_val + 9999.0,
    }


@app.post("/admin/test/restore-artifact")
def restore_artifact_for_test(
    run_id: str = Query(...),
    target_column: str = Query(...),
    x_test_token: str = Query(None, alias="test_token"),
) -> dict[str, Any]:
    """Test-only endpoint: restore a previously tampered deployment artifact from backup.
    Returns 404 when test endpoints are disabled.
    """
    if not _TEST_ENDPOINTS_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")
    _test_tok = os.getenv("ORBITA_TEST_TOKEN", "")
    if not _test_tok or x_test_token != _test_tok:
        raise HTTPException(status_code=403, detail="Forbidden")
    run = _guard(service.store.get_run, run_id)
    result = run.get("result", {})
    art_info = result.get("model_artifacts", {}).get(target_column, {})
    art_path_str = art_info.get("model_artifact_path")
    if not art_path_str:
        raise HTTPException(status_code=404, detail="Deployment artifact path not found")
    art_path = Path(art_path_str)
    backup_path = art_path.with_suffix(".json.backup")
    if not backup_path.exists():
        raise HTTPException(status_code=404, detail="Backup not found — artifact was not previously tampered via this endpoint")
    art_path.write_bytes(backup_path.read_bytes())
    backup_path.unlink()
    return {
        "status": "restored",
        "artifact_path": str(art_path),
        "restored_sha256": art_info.get("model_artifact_sha256"),
    }


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("ORBITA_MVP_PORT", "8010")))
    uvicorn.run("orbita_mvp.api:app", host="0.0.0.0", port=port, reload=False)
