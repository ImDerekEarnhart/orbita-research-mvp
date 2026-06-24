from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
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
