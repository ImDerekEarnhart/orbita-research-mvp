from __future__ import annotations

import base64
import csv
import io
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from orbita import EvidenceKind, Stance

from .graph_ui import GRAPH_HTML, build_events, build_graph_data
from .service import ResearchMVP


_data_dir = Path("/data") if Path("/data").exists() else Path(".")
DB_PATH = Path(os.getenv("ORBITA_MVP_DB", str(_data_dir / "orbita_mvp.db")))
WORKSPACE = Path(os.getenv("ORBITA_MVP_WORKSPACE", str(_data_dir / "orbita_workspace")))
service = ResearchMVP(DB_PATH, WORKSPACE)

app = FastAPI(
    title="Orbita Research MVP",
    version="0.1.0",
    description=(
        "Upload research material, compile an explicit governed plan, run frozen "
        "discovery candidates, persist findings in an epistemic graph, and produce "
        "an expert-readable dossier."
    ),
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
    return {"status": "ok", "version": "0.1.0", "db": str(DB_PATH.resolve()), "workspace": str(WORKSPACE.resolve())}


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
    return _guard(service.compile_case, case_id, max_candidates=request.max_candidates)


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

    # Pick best survivor (highest verdict score)
    best = max(target_survivors, key=lambda f: f["verdict"]["score"])
    payload = best["candidate"]["payload"]
    kind = payload["kind"]

    # Re-read training data and refit on full training set
    plan_record = service.store.get_plan(run["plan_id"]) if run.get("plan_id") else None
    if not plan_record:
        # fall back: find plan via case
        case_id = run.get("case_id")
        if case_id:
            case = service.store.get_case(case_id)
            if case.get("plans"):
                plan_record = case["plans"][0]
    if not plan_record:
        raise HTTPException(status_code=422, detail="Cannot locate training plan for this run")

    train_path = plan_record["plan"].get("selected_dataset", {}).get("normalized_path")
    if not train_path or not Path(train_path).exists():
        raise HTTPException(status_code=422, detail="Training CSV is no longer available on this server")

    train_df = pd.read_csv(train_path)

    # Refit on full training set (not just scout partition)
    from .table_domain import UploadedTableDomain
    from orbita_discovery.core import Candidate
    c = Candidate(id=best["candidate"]["id"], statement=best["candidate"]["statement"], payload=payload)

    domain = UploadedTableDomain(train_df, [best["candidate"]])
    model = domain.refit(c, train_df)
    if not model.get("valid"):
        raise HTTPException(status_code=422, detail="Refitting the survivor on training data produced an invalid model")

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
    predictions: list[float | None] = []

    if kind == "linear_association":
        predictor = payload["predictor"]
        if predictor not in test_df.columns:
            raise HTTPException(status_code=422, detail=f"Predictor column '{predictor}' not found in test file")
        xs = pd.to_numeric(test_df[predictor], errors="coerce")
        preds = model["intercept"] + model["slope"] * xs
        predictions = [None if not np.isfinite(v) else round(float(v), 8) for v in preds]

    elif kind == "group_difference":
        group_col = payload["group"]
        if group_col not in test_df.columns:
            raise HTTPException(status_code=422, detail=f"Group column '{group_col}' not found in test file")
        predictions = [
            model["means"].get(str(g), model["overall"])
            for g in test_df[group_col].astype(str)
        ]
    else:
        raise HTTPException(status_code=422, detail=f"Unsupported survivor kind for prediction: {kind}")

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


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", os.getenv("ORBITA_MVP_PORT", "8010")))
    uvicorn.run("orbita_mvp.api:app", host="0.0.0.0", port=port, reload=False)
