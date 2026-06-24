"""Command-line interface for Orbita Discovery Kit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import verify_ledger
from .registry import DOMAIN_INFO
from .service import execute_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="orbita")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("domains", help="List built-in domains")

    run = sub.add_parser("run", help="Run from a JSON configuration file")
    run.add_argument("config", type=Path)
    run.add_argument("--out-dir", default="runs")

    verify = sub.add_parser("verify-ledger", help="Verify a hash-chained JSONL ledger")
    verify.add_argument("ledger", type=Path)

    serve = sub.add_parser("serve", help="Start the local REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "domains":
        for name, description in DOMAIN_INFO.items():
            print(f"{name:<24} {description}")
        return
    if args.command == "verify-ledger":
        ok, errors = verify_ledger(args.ledger)
        print("VALID" if ok else "INVALID")
        for error in errors:
            print(" -", error)
        raise SystemExit(0 if ok else 1)
    if args.command == "serve":
        try:
            import uvicorn
        except ImportError as exc:
            raise SystemExit("Install API dependencies: pip install -e '.[api]'") from exc
        uvicorn.run("orbita_discovery.api:app", host=args.host, port=args.port)
        return

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    result = execute_run(
        domain_name=cfg["domain"],
        domain_config=cfg.get("config"),
        judge_name=cfg.get("judge", "governed"),
        commit_at=float(cfg.get("commit_at", 0.5)),
        baseline_margin=float(cfg.get("baseline_margin", 0.05)),
        falsifier_config=cfg.get("falsifiers"),
        output_dir=args.out_dir,
        progress=print,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "findings"}, indent=2))
