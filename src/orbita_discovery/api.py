"""Local REST API for Orbita Discovery Kit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .registry import DOMAIN_INFO
from .service import execute_run

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Install API dependencies: pip install 'orbita-discovery-kit[api]'") from exc


class RunRequest(BaseModel):
    domain: str
    config: dict[str, Any] = Field(default_factory=dict)
    judge: str = "governed"
    commit_at: float = 0.5
    baseline_margin: float = 0.05
    falsifiers: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(
    title="Orbita Discovery API",
    version="0.2.0",
    description="Governed propose → judge → falsify → hash-ledger service.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.2.0"}


@app.get("/domains")
def domains() -> dict[str, Any]:
    return {"domains": DOMAIN_INFO}


@app.post("/runs")
def run_discovery(request: RunRequest) -> dict[str, Any]:
    try:
        return execute_run(
            domain_name=request.domain,
            domain_config=request.config,
            judge_name=request.judge,
            commit_at=request.commit_at,
            baseline_margin=request.baseline_margin,
            falsifier_config=request.falsifiers,
            output_dir=Path(os.getenv("ORBITA_RUN_DIR", "runs")),
        )
    except (ValueError, RuntimeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install API dependencies: pip install 'orbita-discovery-kit[api]'") from exc
    uvicorn.run("orbita_discovery.api:app", host="127.0.0.1", port=8000, reload=False)
