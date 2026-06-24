from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .service import ResearchMVP


def main() -> None:
    parser = argparse.ArgumentParser(prog="orbita-mvp", description="Orbita Research MVP")
    parser.add_argument("--db", default=os.getenv("ORBITA_MVP_DB", "orbita_mvp.db"))
    parser.add_argument("--workspace", default=os.getenv("ORBITA_MVP_WORKSPACE", "orbita_workspace"))
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start the local web service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8010)

    demo = sub.add_parser("demo", help="Run an end-to-end case from a table")
    demo.add_argument("table")
    demo.add_argument("--name", default="Orbita demo")
    demo.add_argument("--goal", default="")

    history = sub.add_parser("history", help="Print reconstructed claim history")
    history.add_argument("claim_id")

    sub.add_parser("queue", help="List open re-examination tasks")

    args = parser.parse_args()
    if args.command == "serve":
        os.environ["ORBITA_MVP_DB"] = str(args.db)
        os.environ["ORBITA_MVP_WORKSPACE"] = str(args.workspace)
        import uvicorn

        uvicorn.run("orbita_mvp.api:app", host=args.host, port=args.port, reload=False)
        return

    with ResearchMVP(args.db, args.workspace) as service:
        if args.command == "demo":
            case = service.create_case(name=args.name, goal=args.goal)
            service.add_file(case["id"], Path(args.table))
            plan = service.compile_case(case["id"])
            service.approve_plan(plan["id"], reviewer="cli-user")
            run = service.run_case(case["id"], plan_id=plan["id"])
            print(json.dumps(run, indent=2, default=str))
        elif args.command == "history":
            print(json.dumps(service.claim_history(args.claim_id), indent=2, default=str))
        elif args.command == "queue":
            print(json.dumps(service.reexamination_queue(), indent=2, default=str))


if __name__ == "__main__":
    main()
