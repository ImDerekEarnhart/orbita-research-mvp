from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .analysis import DatasetAnalysisSpec
from .agent_os import AutonomyMode, ComputerAgentRuntime
from .adaptive import AdaptiveWorkflowStatus, DesktopActionSpec
from .coding import CodingRuntime, CodingTestSpec, PatchProposal
from .discovery import DiscoverySpec, GovernedDiscoveryRuntime
from .demo import run_demo
from .ledger import EpistemicLedger
from .execution import CliOCIEngine, ContainerExecutionSpec
from .evaluation import EvaluationSuiteSpec, ComparativeEvaluationRuntime, default_adversarial_suite
from .research import EmpiricalResearchRuntime, EmpiricalStudySpec
from .models import LiteralDatatype, ObjectKind, ReviewDecision, TypedLiteral
from .proposals import ModelIdentity, PROPOSAL_SCHEMA, ProposalRequest
from .support import SupportEngine
from .integrations import OpenClawBridge
from .ui import UIConfig, serve_ui


def _literal_from_cli(raw: str, datatype: str, unit: str | None) -> TypedLiteral:
    dt = LiteralDatatype(datatype)
    value: Any = raw
    if dt == LiteralDatatype.INTEGER:
        value = int(raw)
    elif dt == LiteralDatatype.FLOAT:
        value = float(raw)
    elif dt == LiteralDatatype.BOOLEAN:
        lowered = raw.strip().casefold()
        if lowered not in {"true", "false"}:
            raise ValueError("Boolean literals must be true or false")
        value = lowered == "true"
    elif dt == LiteralDatatype.JSON:
        value = json.loads(raw)
    return TypedLiteral(value, dt, unit)


def _load_analysis_spec(path: Path) -> DatasetAnalysisSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Analysis spec must be a JSON object")
    dataset_path = Path(raw["dataset_path"])
    if not dataset_path.is_absolute():
        raw["dataset_path"] = str((path.parent / dataset_path).resolve())
    return DatasetAnalysisSpec.from_dict(raw)




def _load_discovery_spec(path: Path) -> DiscoverySpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Discovery spec must be a JSON object")
    return DiscoverySpec.from_dict(raw, base_dir=path.parent)


def _load_execution_spec(path: Path) -> ContainerExecutionSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Execution spec must be a JSON object")
    return ContainerExecutionSpec.from_dict(raw, base_dir=path.parent)




def _load_coding_test_spec(path: Path) -> CodingTestSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Coding test spec must be a JSON object")
    return CodingTestSpec.from_dict(raw)


def _load_research_spec(path: Path) -> EmpiricalStudySpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Research study spec must be a JSON object")
    return EmpiricalStudySpec.from_dict(raw)


def _load_evaluation_spec(path: Path) -> EvaluationSuiteSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Evaluation suite spec must be a JSON object")
    return EvaluationSuiteSpec.from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(prog="orbita", description="Orbita Epistemic Runtime")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run the end-to-end proof-carrying AI demo")
    demo.add_argument("--root", type=Path, default=None)

    ui = sub.add_parser("ui", help="Launch the secure local Orbita web interface")
    ui.add_argument("db", type=Path, help="SQLite ledger path")
    ui.add_argument("--workspace", type=Path, default=None)
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--allow-remote", action="store_true")
    ui.add_argument("--no-browser", action="store_true")
    ui.add_argument("--max-upload-mb", type=int, default=5)

    report = sub.add_parser("report", help="Show the current support report for a claim")
    report.add_argument("db", type=Path)
    report.add_argument("claim_id")

    history = sub.add_parser("history", help="Show append-only history for a claim")
    history.add_argument("db", type=Path)
    history.add_argument("claim_id")

    relation_add = sub.add_parser("relation-add", help="Add a typed subject-predicate-object claim")
    relation_add.add_argument("db", type=Path)
    relation_add.add_argument("subject")
    relation_add.add_argument("predicate")
    relation_add.add_argument("object")
    relation_add.add_argument("--subject-type", default="thing")
    relation_add.add_argument("--object-type", default="thing")
    relation_add.add_argument("--literal-datatype", choices=[d.value for d in LiteralDatatype])
    relation_add.add_argument("--unit")
    relation_add.add_argument("--negative", action="store_true")
    relation_add.add_argument("--valid-from")
    relation_add.add_argument("--valid-to")
    relation_add.add_argument("--qualifiers-json", default="{}")

    relation_list = sub.add_parser("relation-list", help="Query typed relation claims")
    relation_list.add_argument("db", type=Path)
    relation_list.add_argument("--subject")
    relation_list.add_argument("--subject-type")
    relation_list.add_argument("--predicate")
    relation_list.add_argument("--object")
    relation_list.add_argument("--object-type")
    relation_list.add_argument("--literal-datatype", choices=[d.value for d in LiteralDatatype])
    relation_list.add_argument("--unit")
    relation_list.add_argument("--negative", action="store_true")
    relation_list.add_argument("--positive", action="store_true")
    relation_list.add_argument("--valid-at")

    language_ask = sub.add_parser(
        "language-ask", help="Answer through the meaning-first warranted language engine"
    )
    language_ask.add_argument("db", type=Path)
    language_ask.add_argument("utterance")

    language_interpret = sub.add_parser(
        "language-interpret", help="Parse an utterance into a semantic frame without answering"
    )
    language_interpret.add_argument("db", type=Path)
    language_interpret.add_argument("utterance")

    language_show = sub.add_parser(
        "language-show", help="Show one persisted sentence-warrant response"
    )
    language_show.add_argument("db", type=Path)
    language_show.add_argument("response_id")

    language_verify = sub.add_parser(
        "language-verify", help="Verify one language response and its sentence receipts"
    )
    language_verify.add_argument("db", type=Path)
    language_verify.add_argument("response_id")

    language_ground = sub.add_parser(
        "language-ground", help="Inspect recursive meaning grounding for an entity"
    )
    language_ground.add_argument("db", type=Path)
    language_ground.add_argument("reference")
    language_ground.add_argument("--max-depth", type=int, default=12)

    language_alias = sub.add_parser(
        "language-alias", help="Register a phrase alias for a typed predicate"
    )
    language_alias.add_argument("db", type=Path)
    language_alias.add_argument("alias")
    language_alias.add_argument("predicate")

    language_extract = sub.add_parser(
        "language-extract", help="Extract conservative relation proposals without committing evidence"
    )
    language_extract.add_argument("db", type=Path)
    language_extract.add_argument("text")

    language_ingest = sub.add_parser(
        "language-ingest", help="Create provisional typed claims from controlled text"
    )
    language_ingest.add_argument("db", type=Path)
    language_ingest.add_argument("text")
    language_ingest.add_argument("--source-uri")
    language_ingest.add_argument("--independence-key")
    language_ingest.add_argument("--create-evidence", action="store_true")

    language_plan = sub.add_parser(
        "language-plan", help="Build and persist a meaning-level discourse plan without emitting an answer"
    )
    language_plan.add_argument("db", type=Path)
    language_plan.add_argument("utterance")

    language_policy_train = sub.add_parser(
        "language-policy-train", help="Train the compact semantic-action ranker from pairwise examples"
    )
    language_policy_train.add_argument("db", type=Path)
    language_policy_train.add_argument("examples", type=Path)
    language_policy_train.add_argument("--epochs", type=int, default=5)
    language_policy_train.add_argument("--rate", type=float, default=0.1)

    agent_compile = sub.add_parser("agent-compile", help="Compile a computer task into a typed goal")
    agent_compile.add_argument("db", type=Path)
    agent_compile.add_argument("utterance")
    agent_compile.add_argument("--workspace", type=Path)

    agent_create = sub.add_parser("agent-create", help="Create and persist a governed computer goal")
    agent_create.add_argument("db", type=Path)
    agent_create.add_argument("utterance")
    agent_create.add_argument("--workspace", type=Path)
    agent_create.add_argument("--mode", choices=[m.value for m in AutonomyMode], default=AutonomyMode.VERIFIED.value)

    agent_plan = sub.add_parser("agent-plan", help="Compile a persisted goal into a durable skill plan")
    agent_plan.add_argument("db", type=Path)
    agent_plan.add_argument("goal_id")
    agent_plan.add_argument("--workspace", type=Path)

    agent_run = sub.add_parser("agent-run", help="Execute safe steps until the plan succeeds or blocks")
    agent_run.add_argument("db", type=Path)
    agent_run.add_argument("plan_id")
    agent_run.add_argument("--workspace", type=Path)

    agent_goal_show = sub.add_parser("agent-goal-show", help="Show one computer goal")
    agent_goal_show.add_argument("db", type=Path)
    agent_goal_show.add_argument("goal_id")
    agent_goal_show.add_argument("--workspace", type=Path)

    agent_plan_show = sub.add_parser("agent-plan-show", help="Show one computer plan and receipts")
    agent_plan_show.add_argument("db", type=Path)
    agent_plan_show.add_argument("plan_id")
    agent_plan_show.add_argument("--workspace", type=Path)

    agent_approve = sub.add_parser("agent-approve", help="Approve an exact computer step argument hash")
    agent_approve.add_argument("db", type=Path)
    agent_approve.add_argument("approval_id")
    agent_approve.add_argument("--reviewer", required=True)
    agent_approve.add_argument("--rationale", required=True)
    agent_approve.add_argument("--workspace", type=Path)

    agent_skills = sub.add_parser("agent-skills", help="List available computer skill contracts")
    agent_skills.add_argument("db", type=Path)
    agent_skills.add_argument("--workspace", type=Path)

    agent_state = sub.add_parser("agent-state", help="Capture a hash-bound computer workspace state")
    agent_state.add_argument("db", type=Path)
    agent_state.add_argument("--workspace", type=Path)
    agent_state.add_argument("--goal-id")

    agent_verify = sub.add_parser("agent-verify", help="Verify a computer plan or action receipt hash")
    agent_verify.add_argument("db", type=Path)
    agent_verify.add_argument("--plan-id")
    agent_verify.add_argument("--receipt-id")
    agent_verify.add_argument("--workspace", type=Path)

    adaptive_status = sub.add_parser("adaptive-status", help="Show desktop-perception and learned-workflow status")
    adaptive_status.add_argument("db", type=Path)

    desktop_observe = sub.add_parser("desktop-observe", help="Capture a signed desktop accessibility/screenshot observation")
    desktop_observe.add_argument("db", type=Path)
    desktop_observe.add_argument("--label", default="desktop observation")
    desktop_observe.add_argument("--endpoint", default="http://127.0.0.1:18793")
    desktop_observe.add_argument("--secret-env", default="ORBITA_OPENCLAW_SECRET")
    desktop_observe.add_argument("--allow-remote", action="store_true")
    desktop_observe.add_argument("--no-screenshot", action="store_true")

    desktop_list = sub.add_parser("desktop-list", help="List desktop observations")
    desktop_list.add_argument("db", type=Path)
    desktop_list.add_argument("--limit", type=int, default=100)

    desktop_show = sub.add_parser("desktop-show", help="Show one desktop observation")
    desktop_show.add_argument("db", type=Path)
    desktop_show.add_argument("observation_id")

    desktop_diff = sub.add_parser("desktop-diff", help="Compare two desktop observations for interface drift")
    desktop_diff.add_argument("db", type=Path)
    desktop_diff.add_argument("before_id")
    desktop_diff.add_argument("after_id")

    desktop_action_propose = sub.add_parser("desktop-action-propose", help="Propose an exact desktop action from JSON")
    desktop_action_propose.add_argument("db", type=Path)
    desktop_action_propose.add_argument("spec", type=Path)

    desktop_action_request = sub.add_parser("desktop-action-request-approval", help="Request approval for an exact desktop action")
    desktop_action_request.add_argument("db", type=Path)
    desktop_action_request.add_argument("action_id")

    desktop_action_decide = sub.add_parser("desktop-action-decide", help="Approve or reject an exact desktop action")
    desktop_action_decide.add_argument("db", type=Path)
    desktop_action_decide.add_argument("approval_id")
    desktop_action_decide.add_argument("decision", choices=["approve", "reject"])
    desktop_action_decide.add_argument("--reviewer", required=True)
    desktop_action_decide.add_argument("--rationale", required=True)

    desktop_action_run = sub.add_parser("desktop-action-run", help="Execute an approved desktop action through the local broker")
    desktop_action_run.add_argument("db", type=Path)
    desktop_action_run.add_argument("action_id")
    desktop_action_run.add_argument("--endpoint", default="http://127.0.0.1:18793")
    desktop_action_run.add_argument("--secret-env", default="ORBITA_OPENCLAW_SECRET")
    desktop_action_run.add_argument("--allow-remote", action="store_true")

    workflow_learn = sub.add_parser("workflow-learn", help="Propose a reusable workflow from a verified successful plan")
    workflow_learn.add_argument("db", type=Path)
    workflow_learn.add_argument("plan_id")
    workflow_learn.add_argument("--name", required=True)
    workflow_learn.add_argument("--description", required=True)
    workflow_learn.add_argument("--parameter-map-json", default="{}")
    workflow_learn.add_argument("--required-observation-id")

    workflow_list = sub.add_parser("workflow-list", help="List adaptive workflows")
    workflow_list.add_argument("db", type=Path)
    workflow_list.add_argument("--status", choices=[s.value for s in AdaptiveWorkflowStatus])

    workflow_show = sub.add_parser("workflow-show", help="Show one adaptive workflow")
    workflow_show.add_argument("db", type=Path)
    workflow_show.add_argument("workflow_id")

    workflow_review = sub.add_parser("workflow-review", help="Activate or reject a proposed adaptive workflow")
    workflow_review.add_argument("db", type=Path)
    workflow_review.add_argument("workflow_id")
    workflow_review.add_argument("decision", choices=["approve", "reject"])
    workflow_review.add_argument("--reviewer", required=True)
    workflow_review.add_argument("--rationale", required=True)

    workflow_instantiate = sub.add_parser("workflow-instantiate", help="Materialize a reviewed workflow into a durable computer plan")
    workflow_instantiate.add_argument("db", type=Path)
    workflow_instantiate.add_argument("workflow_id")
    workflow_instantiate.add_argument("--parameters-json", default="{}")
    workflow_instantiate.add_argument("--workspace", type=Path)
    workflow_instantiate.add_argument("--mode", choices=[m.value for m in AutonomyMode], default=AutonomyMode.VERIFIED.value)
    workflow_instantiate.add_argument("--current-observation-id")

    integration_status = sub.add_parser("integration-status", help="Show governed integration status")
    integration_status.add_argument("db", type=Path)

    email_draft = sub.add_parser("email-draft-create", help="Create a local email draft without sending")
    email_draft.add_argument("db", type=Path)
    email_draft.add_argument("--to", required=True)
    email_draft.add_argument("--subject", required=True)
    email_draft.add_argument("--body", required=True)
    email_draft.add_argument("--cc")
    email_draft.add_argument("--bcc")

    calendar_draft = sub.add_parser("calendar-draft-create", help="Create a local calendar-event draft")
    calendar_draft.add_argument("db", type=Path)
    calendar_draft.add_argument("--title", required=True)
    calendar_draft.add_argument("--start", required=True)
    calendar_draft.add_argument("--end", required=True)
    calendar_draft.add_argument("--timezone", required=True)
    calendar_draft.add_argument("--attendee", action="append", default=[])
    calendar_draft.add_argument("--location")
    calendar_draft.add_argument("--description")

    draft_list = sub.add_parser("integration-draft-list", help="List email and calendar drafts")
    draft_list.add_argument("db", type=Path)

    draft_show = sub.add_parser("integration-draft-show", help="Show one integration draft")
    draft_show.add_argument("db", type=Path)
    draft_show.add_argument("draft_id")

    draft_request = sub.add_parser("integration-draft-request-approval", help="Request exact-payload approval")
    draft_request.add_argument("db", type=Path)
    draft_request.add_argument("draft_id")

    draft_decide = sub.add_parser("integration-draft-decide", help="Approve or reject an exact draft")
    draft_decide.add_argument("db", type=Path)
    draft_decide.add_argument("approval_id")
    draft_decide.add_argument("--decision", choices=["approve", "reject"], required=True)
    draft_decide.add_argument("--reviewer", required=True)
    draft_decide.add_argument("--rationale", required=True)

    draft_execute = sub.add_parser("integration-draft-execute", help="Execute an approved draft through OpenClaw")
    draft_execute.add_argument("db", type=Path)
    draft_execute.add_argument("draft_id")
    draft_execute.add_argument("--endpoint", default="http://127.0.0.1:7331")
    draft_execute.add_argument("--secret-env", default="ORBITA_OPENCLAW_SECRET")
    draft_execute.add_argument("--allow-remote", action="store_true")

    browser_verify = sub.add_parser("browser-verify", help="Navigate through OpenClaw and verify returned HTML")
    browser_verify.add_argument("db", type=Path)
    browser_verify.add_argument("url")
    browser_verify.add_argument("--contains", action="append", default=[])
    browser_verify.add_argument("--forbid", action="append", default=[])
    browser_verify.add_argument("--title-contains")
    browser_verify.add_argument("--expected-url-prefix")
    browser_verify.add_argument("--endpoint", default="http://127.0.0.1:7331")
    browser_verify.add_argument("--secret-env", default="ORBITA_OPENCLAW_SECRET")
    browser_verify.add_argument("--allow-remote", action="store_true")

    windows_register = sub.add_parser("windows-app-register", help="Register an allowlisted Windows application")
    windows_register.add_argument("db", type=Path)
    windows_register.add_argument("app_id")
    windows_register.add_argument("--name", required=True)
    windows_register.add_argument("--executable-hint", required=True)
    windows_register.add_argument("--allow-arg", action="append", default=[])

    windows_launch = sub.add_parser("windows-app-launch", help="Launch an allowlisted Windows app through OpenClaw")
    windows_launch.add_argument("db", type=Path)
    windows_launch.add_argument("app_id")
    windows_launch.add_argument("--argument", action="append", default=[])
    windows_launch.add_argument("--endpoint", default="http://127.0.0.1:7331")
    windows_launch.add_argument("--secret-env", default="ORBITA_OPENCLAW_SECRET")
    windows_launch.add_argument("--allow-remote", action="store_true")

    schedule_once = sub.add_parser("schedule-once", help="Create a reboot-safe one-time computer task")
    schedule_once.add_argument("db", type=Path)
    schedule_once.add_argument("--name", required=True)
    schedule_once.add_argument("--run-at", required=True)
    schedule_once.add_argument("--goal", required=True)
    schedule_once.add_argument("--workspace", type=Path)
    schedule_once.add_argument("--mode", choices=[m.value for m in AutonomyMode], default=AutonomyMode.VERIFIED.value)

    schedule_interval = sub.add_parser("schedule-interval", help="Create a reboot-safe interval task")
    schedule_interval.add_argument("db", type=Path)
    schedule_interval.add_argument("--name", required=True)
    schedule_interval.add_argument("--every-seconds", type=int, required=True)
    schedule_interval.add_argument("--goal", required=True)
    schedule_interval.add_argument("--first-run-at")
    schedule_interval.add_argument("--workspace", type=Path)
    schedule_interval.add_argument("--mode", choices=[m.value for m in AutonomyMode], default=AutonomyMode.VERIFIED.value)
    schedule_interval.add_argument("--max-runs", type=int)

    schedule_list = sub.add_parser("schedule-list", help="List scheduled computer tasks")
    schedule_list.add_argument("db", type=Path)

    schedule_show = sub.add_parser("schedule-show", help="Show one scheduled computer task")
    schedule_show.add_argument("db", type=Path)
    schedule_show.add_argument("schedule_id")

    schedule_pause = sub.add_parser("schedule-pause", help="Pause a scheduled computer task")
    schedule_pause.add_argument("db", type=Path)
    schedule_pause.add_argument("schedule_id")

    schedule_cancel = sub.add_parser("schedule-cancel", help="Cancel a scheduled computer task")
    schedule_cancel.add_argument("db", type=Path)
    schedule_cancel.add_argument("schedule_id")

    schedule_enable = sub.add_parser("schedule-enable", help="Reactivate a paused or failed schedule")
    schedule_enable.add_argument("db", type=Path)
    schedule_enable.add_argument("schedule_id")

    schedule_tick = sub.add_parser("schedule-tick", help="Run due scheduled tasks once")
    schedule_tick.add_argument("db", type=Path)
    schedule_tick.add_argument("--worker", default="orbita-cli")
    schedule_tick.add_argument("--max-jobs", type=int, default=10)
    schedule_tick.add_argument("--now")

    schedule_resume = sub.add_parser("schedule-resume", help="Resume a blocked scheduled task")
    schedule_resume.add_argument("db", type=Path)
    schedule_resume.add_argument("schedule_id")
    schedule_resume.add_argument("--worker", default="orbita-cli")

    schedule_worker = sub.add_parser("schedule-worker", help="Run the durable scheduler worker loop")
    schedule_worker.add_argument("db", type=Path)
    schedule_worker.add_argument("--worker", default="orbita-worker")
    schedule_worker.add_argument("--poll-seconds", type=float, default=30.0)
    schedule_worker.add_argument("--max-jobs", type=int, default=10)
    schedule_worker.add_argument("--once", action="store_true")


    coding_start = sub.add_parser("coding-start", help="Create a governed native coding session")
    coding_start.add_argument("db", type=Path)
    coding_start.add_argument("repository", type=Path)
    coding_start.add_argument("goal")
    coding_start.add_argument("--workspace", type=Path)
    coding_start.add_argument("--test-spec", type=Path)
    coding_start.add_argument("--allowed-path", action="append", default=[])
    coding_start.add_argument("--max-candidates", type=int, default=4)

    coding_list = sub.add_parser("coding-list", help="List native coding sessions")
    coding_list.add_argument("db", type=Path)
    coding_list.add_argument("--workspace", type=Path)

    coding_show = sub.add_parser("coding-show", help="Show one native coding session")
    coding_show.add_argument("db", type=Path)
    coding_show.add_argument("session_id")
    coding_show.add_argument("--workspace", type=Path)

    coding_add = sub.add_parser("coding-add-candidate", help="Register one unified-diff patch candidate")
    coding_add.add_argument("db", type=Path)
    coding_add.add_argument("session_id")
    coding_add.add_argument("patch", type=Path)
    coding_add.add_argument("--provider", default="human")
    coding_add.add_argument("--rationale", required=True)
    coding_add.add_argument("--expected-effect", default="")
    coding_add.add_argument("--workspace", type=Path)

    coding_prepare = sub.add_parser("coding-prepare", help="Apply a candidate in an isolated Git worktree and run static checks")
    coding_prepare.add_argument("db", type=Path)
    coding_prepare.add_argument("candidate_id")
    coding_prepare.add_argument("--workspace", type=Path)

    coding_baseline = sub.add_parser("coding-test-baseline", help="Submit the immutable baseline container test")
    coding_baseline.add_argument("db", type=Path)
    coding_baseline.add_argument("session_id")
    coding_baseline.add_argument("--workspace", type=Path)

    coding_test = sub.add_parser("coding-test-candidate", help="Submit one prepared candidate for containerized testing")
    coding_test.add_argument("db", type=Path)
    coding_test.add_argument("candidate_id")
    coding_test.add_argument("--workspace", type=Path)

    coding_test_run = sub.add_parser("coding-test-run", help="Approve, execute, and finalize a coding test")
    coding_test_run.add_argument("db", type=Path)
    coding_test_run.add_argument("coding_test_id")
    coding_test_run.add_argument("--reviewer", required=True)
    coding_test_run.add_argument("--rationale", required=True)
    coding_test_run.add_argument("--engine", choices=["auto", "docker", "podman"], default="auto")
    coding_test_run.add_argument("--workspace", type=Path)

    coding_rank = sub.add_parser("coding-rank", help="Rank tested patch candidates")
    coding_rank.add_argument("db", type=Path)
    coding_rank.add_argument("session_id")
    coding_rank.add_argument("--workspace", type=Path)

    coding_select = sub.add_parser("coding-select", help="Select the best verified passing candidate")
    coding_select.add_argument("db", type=Path)
    coding_select.add_argument("session_id")
    coding_select.add_argument("--candidate-id")
    coding_select.add_argument("--workspace", type=Path)

    coding_promotion_request = sub.add_parser("coding-promotion-request", help="Create an exact candidate-promotion approval")
    coding_promotion_request.add_argument("db", type=Path)
    coding_promotion_request.add_argument("candidate_id")
    coding_promotion_request.add_argument("--workspace", type=Path)

    coding_approve = sub.add_parser("coding-approve", help="Approve an exact coding promotion or rollback")
    coding_approve.add_argument("db", type=Path)
    coding_approve.add_argument("approval_id")
    coding_approve.add_argument("--reviewer", required=True)
    coding_approve.add_argument("--rationale", required=True)
    coding_approve.add_argument("--workspace", type=Path)

    coding_promote = sub.add_parser("coding-promote", help="Promote the exact tested candidate into the original working tree")
    coding_promote.add_argument("db", type=Path)
    coding_promote.add_argument("approval_id")
    coding_promote.add_argument("--workspace", type=Path)

    coding_rollback_request = sub.add_parser("coding-rollback-request", help="Request exact rollback approval for a promotion")
    coding_rollback_request.add_argument("db", type=Path)
    coding_rollback_request.add_argument("promotion_id")
    coding_rollback_request.add_argument("--workspace", type=Path)

    coding_rollback = sub.add_parser("coding-rollback", help="Execute an approved exact rollback")
    coding_rollback.add_argument("db", type=Path)
    coding_rollback.add_argument("approval_id")
    coding_rollback.add_argument("--workspace", type=Path)

    proposal_schema = sub.add_parser(
        "proposal-schema", help="Print the strict JSON Schema accepted from language models"
    )

    proposal_prompt = sub.add_parser(
        "proposal-prompt", help="Build the exact prompts for a model proposal request"
    )
    proposal_prompt.add_argument("task")
    proposal_prompt.add_argument("--context-json", default="{}")
    proposal_prompt.add_argument("--allowed-predicate", action="append", default=[])
    proposal_prompt.add_argument("--max-proposals", type=int, default=25)

    proposal_ingest = sub.add_parser(
        "proposal-ingest", help="Validate and ingest one schema-constrained model response"
    )
    proposal_ingest.add_argument("db", type=Path)
    proposal_ingest.add_argument("response", type=Path)
    proposal_ingest.add_argument("--provider", required=True)
    proposal_ingest.add_argument("--model", required=True)
    proposal_ingest.add_argument("--model-version")
    proposal_ingest.add_argument("--system-prompt-file", type=Path)
    proposal_ingest.add_argument("--user-prompt-file", type=Path)
    proposal_ingest.add_argument("--generation-json", default="{}")
    proposal_ingest.add_argument("--response-id")

    proposal_show = sub.add_parser("proposal-show", help="Show one proposal batch")
    proposal_show.add_argument("db", type=Path)
    proposal_show.add_argument("batch_id")

    proposal_list = sub.add_parser("proposal-list", help="List model proposal batches")
    proposal_list.add_argument("db", type=Path)
    proposal_list.add_argument("--status", choices=["processing", "applied", "needs_review", "rejected"])

    proposal_review = sub.add_parser(
        "proposal-review", help="Approve or reject one quarantined proposal item"
    )
    proposal_review.add_argument("db", type=Path)
    proposal_review.add_argument("item_id")
    proposal_review.add_argument("decision", choices=[d.value for d in ReviewDecision])
    proposal_review.add_argument("--reviewer", required=True)
    proposal_review.add_argument("--rationale", required=True)

    proposal_retry = sub.add_parser(
        "proposal-retry", help="Retry non-human-gated items after dependencies are resolved"
    )
    proposal_retry.add_argument("db", type=Path)
    proposal_retry.add_argument("batch_id")

    analysis_run = sub.add_parser(
        "analysis-run",
        help="Run a hash-bound built-in dataset analysis from a JSON specification",
    )
    analysis_run.add_argument("db", type=Path)
    analysis_run.add_argument("spec", type=Path)

    analysis_show = sub.add_parser("analysis-show", help="Show one dataset-analysis receipt")
    analysis_show.add_argument("db", type=Path)
    analysis_show.add_argument("receipt_id")

    analysis_list = sub.add_parser("analysis-list", help="List dataset-analysis receipts")
    analysis_list.add_argument("db", type=Path)

    analysis_reproduce = sub.add_parser(
        "analysis-reproduce",
        help="Replay a successful receipt and compare its hash-bound result",
    )
    analysis_reproduce.add_argument("db", type=Path)
    analysis_reproduce.add_argument("receipt_id")
    analysis_reproduce.add_argument("--dataset-path", type=Path)

    analysis_verify = sub.add_parser(
        "analysis-verify",
        help="Recompute and verify a receipt's integrity hash",
    )
    analysis_verify.add_argument("db", type=Path)
    analysis_verify.add_argument("receipt_id")

    execution_status = sub.add_parser(
        "execution-status", help="Show available OCI engines and execution policy"
    )
    execution_status.add_argument("db", type=Path)
    execution_status.add_argument("--workspace", type=Path, default=None)

    execution_submit = sub.add_parser(
        "execution-submit", help="Stage a digest-pinned container execution manifest"
    )
    execution_submit.add_argument("db", type=Path)
    execution_submit.add_argument("spec", type=Path)
    execution_submit.add_argument("--workspace", type=Path, default=None)

    execution_approve = sub.add_parser(
        "execution-approve", help="Approve the exact hash-bound execution manifest"
    )
    execution_approve.add_argument("db", type=Path)
    execution_approve.add_argument("run_id")
    execution_approve.add_argument("--reviewer", required=True)
    execution_approve.add_argument("--rationale", required=True)
    execution_approve.add_argument("--workspace", type=Path, default=None)

    execution_reject = sub.add_parser(
        "execution-reject", help="Reject a pending container execution"
    )
    execution_reject.add_argument("db", type=Path)
    execution_reject.add_argument("run_id")
    execution_reject.add_argument("--reviewer", required=True)
    execution_reject.add_argument("--rationale", required=True)
    execution_reject.add_argument("--workspace", type=Path, default=None)

    execution_run = sub.add_parser(
        "execution-run", help="Execute one approved manifest through Docker or Podman"
    )
    execution_run.add_argument("db", type=Path)
    execution_run.add_argument("run_id")
    execution_run.add_argument("--engine", choices=["docker", "podman"])
    execution_run.add_argument("--workspace", type=Path, default=None)

    execution_show = sub.add_parser("execution-show", help="Show one execution receipt")
    execution_show.add_argument("db", type=Path)
    execution_show.add_argument("run_id")
    execution_show.add_argument("--workspace", type=Path, default=None)

    execution_list = sub.add_parser("execution-list", help="List container executions")
    execution_list.add_argument("db", type=Path)
    execution_list.add_argument("--workspace", type=Path, default=None)

    execution_verify = sub.add_parser(
        "execution-verify", help="Verify manifest, artifacts, and receipt integrity"
    )
    execution_verify.add_argument("db", type=Path)
    execution_verify.add_argument("run_id")
    execution_verify.add_argument("--workspace", type=Path, default=None)

    execution_reproduce = sub.add_parser(
        "execution-reproduce-prepare",
        help="Stage an exact reproduction that requires a new human approval",
    )
    execution_reproduce.add_argument("db", type=Path)
    execution_reproduce.add_argument("run_id")
    execution_reproduce.add_argument("--workspace", type=Path, default=None)

    discovery_create = sub.add_parser(
        "discovery-create",
        help="Create a restart-safe governed discovery investigation from a JSON specification",
    )
    discovery_create.add_argument("db", type=Path)
    discovery_create.add_argument("spec", type=Path)
    discovery_create.add_argument("--workspace", type=Path, default=None)

    discovery_list = sub.add_parser("discovery-list", help="List governed discovery investigations")
    discovery_list.add_argument("db", type=Path)
    discovery_list.add_argument("--workspace", type=Path, default=None)

    discovery_show = sub.add_parser("discovery-show", help="Show one discovery investigation")
    discovery_show.add_argument("db", type=Path)
    discovery_show.add_argument("investigation_id")
    discovery_show.add_argument("--workspace", type=Path, default=None)

    discovery_approve = sub.add_parser(
        "discovery-approve", help="Approve the current exact confirmation or replication manifest"
    )
    discovery_approve.add_argument("db", type=Path)
    discovery_approve.add_argument("investigation_id")
    discovery_approve.add_argument("--reviewer", required=True)
    discovery_approve.add_argument("--rationale", required=True)
    discovery_approve.add_argument("--workspace", type=Path, default=None)

    discovery_advance = sub.add_parser(
        "discovery-advance", help="Resume the current approved discovery phase"
    )
    discovery_advance.add_argument("db", type=Path)
    discovery_advance.add_argument("investigation_id")
    discovery_advance.add_argument("--engine", choices=["docker", "podman"])
    discovery_advance.add_argument("--workspace", type=Path, default=None)

    discovery_report = sub.add_parser(
        "discovery-report", help="Compile the deterministic investigation report"
    )
    discovery_report.add_argument("db", type=Path)
    discovery_report.add_argument("investigation_id")
    discovery_report.add_argument("--workspace", type=Path, default=None)

    discovery_verify = sub.add_parser(
        "discovery-verify", help="Verify the investigation report and artifact hashes"
    )
    discovery_verify.add_argument("db", type=Path)
    discovery_verify.add_argument("investigation_id")
    discovery_verify.add_argument("--workspace", type=Path, default=None)


    research_create = sub.add_parser(
        "research-create", help="Create and seal a preregistered empirical study"
    )
    research_create.add_argument("db", type=Path)
    research_create.add_argument("spec", type=Path)
    research_create.add_argument("--workspace", type=Path, default=None)

    research_list = sub.add_parser("research-list", help="List empirical research studies")
    research_list.add_argument("db", type=Path)
    research_list.add_argument("--workspace", type=Path, default=None)

    research_show = sub.add_parser("research-show", help="Show one empirical research study")
    research_show.add_argument("db", type=Path)
    research_show.add_argument("study_id")
    research_show.add_argument("--workspace", type=Path, default=None)

    research_pack = sub.add_parser("research-pack", help="Export a frozen gold-free run pack")
    research_pack.add_argument("db", type=Path)
    research_pack.add_argument("study_id")
    research_pack.add_argument("arm_key")
    research_pack.add_argument("repetition", type=int)
    research_pack.add_argument("--partition", choices=["development", "private"], default="private")
    research_pack.add_argument("--out", type=Path, required=True)
    research_pack.add_argument("--workspace", type=Path, default=None)

    research_import = sub.add_parser("research-import", help="Import a real empirical system response")
    research_import.add_argument("db", type=Path)
    research_import.add_argument("study_id")
    research_import.add_argument("arm_key")
    research_import.add_argument("repetition", type=int)
    research_import.add_argument("response", type=Path)
    research_import.add_argument("--partition", choices=["development", "private"], default="private")
    research_import.add_argument("--origin", choices=["live_model", "human", "replay", "validation_fixture"], required=True)
    research_import.add_argument("--cost-usd", type=float, default=0.0)
    research_import.add_argument("--workspace", type=Path, default=None)

    research_assign = sub.add_parser("research-assign-reviews", help="Create blinded review assignments")
    research_assign.add_argument("db", type=Path)
    research_assign.add_argument("study_id")
    research_assign.add_argument("--reviewer", action="append", required=True)
    research_assign.add_argument("--workspace", type=Path, default=None)

    research_review_export = sub.add_parser("research-review-export", help="Export one reviewer's blinded bundle")
    research_review_export.add_argument("db", type=Path)
    research_review_export.add_argument("study_id")
    research_review_export.add_argument("reviewer")
    research_review_export.add_argument("--out", type=Path, required=True)
    research_review_export.add_argument("--workspace", type=Path, default=None)

    research_review_submit = sub.add_parser("research-review-submit", help="Submit a blinded review")
    research_review_submit.add_argument("db", type=Path)
    research_review_submit.add_argument("assignment_id")
    research_review_submit.add_argument("--reviewer", required=True)
    research_review_submit.add_argument("--label", choices=["pass", "fail", "uncertain"], required=True)
    research_review_submit.add_argument("--confidence", type=float, required=True)
    research_review_submit.add_argument("--rationale", required=True)
    research_review_submit.add_argument("--workspace", type=Path, default=None)

    research_adjudicate = sub.add_parser("research-adjudicate", help="Record a blinded disagreement adjudication")
    research_adjudicate.add_argument("db", type=Path)
    research_adjudicate.add_argument("study_id")
    research_adjudicate.add_argument("research_run_id")
    research_adjudicate.add_argument("task_id")
    research_adjudicate.add_argument("--adjudicator", required=True)
    research_adjudicate.add_argument("--label", choices=["pass", "fail", "uncertain"], required=True)
    research_adjudicate.add_argument("--rationale", required=True)
    research_adjudicate.add_argument("--workspace", type=Path, default=None)

    research_amend = sub.add_parser("research-amend", help="Append a visible protocol amendment")
    research_amend.add_argument("db", type=Path)
    research_amend.add_argument("study_id")
    research_amend.add_argument("amendment", type=Path)
    research_amend.add_argument("--reason", required=True)
    research_amend.add_argument("--actor", required=True)
    research_amend.add_argument("--workspace", type=Path, default=None)

    research_report = sub.add_parser("research-report", help="Compile the empirical study report")
    research_report.add_argument("db", type=Path)
    research_report.add_argument("study_id")
    research_report.add_argument("--workspace", type=Path, default=None)

    research_verify = sub.add_parser("research-verify", help="Verify study and report integrity")
    research_verify.add_argument("db", type=Path)
    research_verify.add_argument("study_id")
    research_verify.add_argument("--workspace", type=Path, default=None)

    research_bundle = sub.add_parser("research-bundle", help="Export a hash-manifested public release bundle")
    research_bundle.add_argument("db", type=Path)
    research_bundle.add_argument("study_id")
    research_bundle.add_argument("--out", type=Path, required=True)
    research_bundle.add_argument("--workspace", type=Path, default=None)

    research_backup = sub.add_parser("research-backup", help="Create a consistent SQLite study backup")
    research_backup.add_argument("db", type=Path)
    research_backup.add_argument("--out", type=Path, required=True)
    research_backup.add_argument("--workspace", type=Path, default=None)

    research_restore = sub.add_parser("research-restore", help="Restore a verified SQLite backup to a new path")
    research_restore.add_argument("source", type=Path)
    research_restore.add_argument("destination", type=Path)

    evaluation_default = sub.add_parser(
        "evaluation-default", help="Create the sealed Orbita adversarial evaluation suite"
    )
    evaluation_default.add_argument("db", type=Path)
    evaluation_default.add_argument("--workspace", type=Path, default=None)

    evaluation_create = sub.add_parser(
        "evaluation-create", help="Create a sealed comparative evaluation suite from JSON"
    )
    evaluation_create.add_argument("db", type=Path)
    evaluation_create.add_argument("spec", type=Path)
    evaluation_create.add_argument("--workspace", type=Path, default=None)

    evaluation_list = sub.add_parser("evaluation-list", help="List comparative evaluation suites")
    evaluation_list.add_argument("db", type=Path)
    evaluation_list.add_argument("--workspace", type=Path, default=None)

    evaluation_show = sub.add_parser("evaluation-show", help="Show one evaluation suite and its runs")
    evaluation_show.add_argument("db", type=Path)
    evaluation_show.add_argument("suite_id")
    evaluation_show.add_argument("--workspace", type=Path, default=None)

    evaluation_export = sub.add_parser(
        "evaluation-export", help="Export the public task bundle and response schema without gold labels"
    )
    evaluation_export.add_argument("db", type=Path)
    evaluation_export.add_argument("suite_id")
    evaluation_export.add_argument("--out", type=Path, required=True)
    evaluation_export.add_argument("--workspace", type=Path, default=None)

    evaluation_import = sub.add_parser(
        "evaluation-import", help="Import and score a schema-constrained system response"
    )
    evaluation_import.add_argument("db", type=Path)
    evaluation_import.add_argument("suite_id")
    evaluation_import.add_argument("response", type=Path)
    evaluation_import.add_argument("--workspace", type=Path, default=None)

    evaluation_fixture = sub.add_parser(
        "evaluation-fixture", help="Create a clearly labeled synthetic fixture run for harness validation"
    )
    evaluation_fixture.add_argument("db", type=Path)
    evaluation_fixture.add_argument("suite_id")
    evaluation_fixture.add_argument(
        "profile", choices=["base_llm", "rag", "final_answer_verifier", "orbita"]
    )
    evaluation_fixture.add_argument("--workspace", type=Path, default=None)

    evaluation_run_show = sub.add_parser("evaluation-run-show", help="Show one scored evaluation run")
    evaluation_run_show.add_argument("db", type=Path)
    evaluation_run_show.add_argument("run_id")
    evaluation_run_show.add_argument("--workspace", type=Path, default=None)

    evaluation_report = sub.add_parser(
        "evaluation-report", help="Compile a hash-verifiable comparative report"
    )
    evaluation_report.add_argument("db", type=Path)
    evaluation_report.add_argument("suite_id")
    evaluation_report.add_argument("--workspace", type=Path, default=None)

    evaluation_verify = sub.add_parser(
        "evaluation-verify", help="Verify suite, run, report, and artifact integrity"
    )
    evaluation_verify.add_argument("db", type=Path)
    evaluation_verify.add_argument("suite_id")
    evaluation_verify.add_argument("--run-id")
    evaluation_verify.add_argument("--workspace", type=Path, default=None)

    evaluation_audit_start = sub.add_parser(
        "evaluation-audit-start", help="Start a human audit timer for one task result"
    )
    evaluation_audit_start.add_argument("db", type=Path)
    evaluation_audit_start.add_argument("run_id")
    evaluation_audit_start.add_argument("task_id")
    evaluation_audit_start.add_argument("--auditor", required=True)
    evaluation_audit_start.add_argument("--workspace", type=Path, default=None)

    evaluation_audit_stop = sub.add_parser(
        "evaluation-audit-stop", help="Stop a human audit timer"
    )
    evaluation_audit_stop.add_argument("db", type=Path)
    evaluation_audit_stop.add_argument("audit_id")
    evaluation_audit_stop.add_argument("--notes", default="")
    evaluation_audit_stop.add_argument("--elapsed-seconds", type=float)
    evaluation_audit_stop.add_argument("--workspace", type=Path, default=None)

    graph_snapshot = sub.add_parser(
        "graph-snapshot",
        help="Capture and optionally render a deterministic epistemic graph",
    )
    graph_snapshot.add_argument("db", type=Path)
    graph_snapshot.add_argument("--root", action="append", default=[])
    graph_snapshot.add_argument("--include-descendants", action="store_true")
    graph_snapshot.add_argument("--name", default="epistemic snapshot")
    graph_snapshot.add_argument("--out", type=Path)
    graph_snapshot.add_argument(
        "--format", action="append", choices=["json", "dot", "svg", "html"], default=[]
    )

    graph_list = sub.add_parser("graph-list", help="List persisted graph snapshots")
    graph_list.add_argument("db", type=Path)

    graph_show = sub.add_parser("graph-show", help="Show one persisted graph snapshot")
    graph_show.add_argument("db", type=Path)
    graph_show.add_argument("snapshot_id")

    graph_diff = sub.add_parser(
        "graph-diff", help="Compare two graph snapshots and render a collapse diff"
    )
    graph_diff.add_argument("db", type=Path)
    graph_diff.add_argument("before_snapshot_id")
    graph_diff.add_argument("after_snapshot_id")
    graph_diff.add_argument("--name", default="epistemic collapse diff")
    graph_diff.add_argument("--out", type=Path)
    graph_diff.add_argument(
        "--format", action="append", choices=["json", "dot", "svg", "html"], default=[]
    )

    graph_diff_list = sub.add_parser("graph-diff-list", help="List persisted graph diffs")
    graph_diff_list.add_argument("db", type=Path)

    graph_diff_show = sub.add_parser("graph-diff-show", help="Show one persisted graph diff")
    graph_diff_show.add_argument("db", type=Path)
    graph_diff_show.add_argument("diff_id")

    graph_verify = sub.add_parser(
        "graph-verify", help="Verify snapshot/diff hashes and rendered artifacts"
    )
    graph_verify.add_argument("db", type=Path)
    graph_verify.add_argument("--snapshot-id")
    graph_verify.add_argument("--diff-id")

    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(run_demo(args.root), indent=2))
        return
    if args.command == "ui":
        serve_ui(
            UIConfig(
                db_path=args.db,
                workspace=args.workspace,
                host=args.host,
                port=args.port,
                allow_remote=args.allow_remote,
                open_browser=not args.no_browser,
                max_upload_bytes=args.max_upload_mb * 1024 * 1024,
            )
        )
        return
    if args.command == "research-restore":
        print(json.dumps(EmpiricalResearchRuntime.restore_database(args.source, args.destination), indent=2))
        return
    if args.command == "proposal-schema":
        print(json.dumps(PROPOSAL_SCHEMA, indent=2))
        return
    if args.command == "proposal-prompt":
        context = json.loads(args.context_json)
        if not isinstance(context, dict):
            raise ValueError("--context-json must decode to an object")
        request = ProposalRequest(
            args.task,
            context=context,
            allowed_predicates=tuple(args.allowed_predicate),
            max_proposals=args.max_proposals,
        )
        # A temporary ledger is not needed to build prompts; use the static prompt logic
        # through a minimal in-memory runtime.
        with EpistemicLedger(":memory:") as prompt_ledger:
            system_prompt, user_prompt = prompt_ledger.proposals.build_prompts(request)
        print(json.dumps({"system_prompt": system_prompt, "user_prompt": user_prompt}, indent=2))
        return

    with EpistemicLedger(args.db) as ledger:
        if args.command == "adaptive-status":
            print(json.dumps(ledger.adaptive.status(), indent=2))
            return
        if args.command in {"desktop-observe", "desktop-action-run"}:
            import os
            secret = os.environ.get(args.secret_env)
            if not secret:
                raise ValueError(f"Environment variable {args.secret_env} is required")
            ledger.integrations.bind_provider(OpenClawBridge(
                args.endpoint, secret, allow_remote=args.allow_remote
            ))
            if args.command == "desktop-observe":
                print(json.dumps(ledger.adaptive.capture_desktop(
                    label=args.label, include_screenshot=not args.no_screenshot
                ), indent=2))
            else:
                print(json.dumps(ledger.adaptive.execute_action(args.action_id), indent=2))
            return
        if args.command == "desktop-list":
            print(json.dumps(ledger.adaptive.list_observations(args.limit), indent=2))
            return
        if args.command == "desktop-show":
            print(json.dumps(ledger.adaptive.get_observation(args.observation_id), indent=2))
            return
        if args.command == "desktop-diff":
            print(json.dumps(ledger.adaptive.compare_observations(args.before_id, args.after_id), indent=2))
            return
        if args.command == "desktop-action-propose":
            raw = json.loads(args.spec.read_text(encoding="utf-8"))
            print(json.dumps(ledger.adaptive.propose_action(DesktopActionSpec.from_dict(raw)), indent=2))
            return
        if args.command == "desktop-action-request-approval":
            print(json.dumps(ledger.adaptive.request_action_approval(args.action_id), indent=2))
            return
        if args.command == "desktop-action-decide":
            print(json.dumps(ledger.adaptive.decide_action_approval(
                args.approval_id, approved=args.decision == "approve",
                reviewer=args.reviewer, rationale=args.rationale
            ), indent=2))
            return
        if args.command == "workflow-learn":
            parameter_map = json.loads(args.parameter_map_json)
            if not isinstance(parameter_map, dict):
                raise ValueError("--parameter-map-json must decode to an object")
            print(json.dumps(ledger.adaptive.learn_workflow_from_plan(
                args.plan_id, name=args.name, description=args.description,
                parameter_map=parameter_map,
                required_observation_id=args.required_observation_id
            ), indent=2))
            return
        if args.command == "workflow-list":
            print(json.dumps(ledger.adaptive.list_workflows(args.status), indent=2))
            return
        if args.command == "workflow-show":
            print(json.dumps(ledger.adaptive.get_workflow(args.workflow_id), indent=2))
            return
        if args.command == "workflow-review":
            print(json.dumps(ledger.adaptive.review_workflow(
                args.workflow_id, approved=args.decision == "approve",
                reviewer=args.reviewer, rationale=args.rationale
            ), indent=2))
            return
        if args.command == "workflow-instantiate":
            parameters = json.loads(args.parameters_json)
            if not isinstance(parameters, dict):
                raise ValueError("--parameters-json must decode to an object")
            print(json.dumps(ledger.adaptive.instantiate_workflow(
                args.workflow_id, parameters, workspace=args.workspace,
                autonomy_mode=args.mode, current_observation_id=args.current_observation_id
            ), indent=2))
            return
        if args.command == "integration-status":
            print(json.dumps(ledger.integrations.status(), indent=2))
            return
        if args.command == "email-draft-create":
            print(json.dumps(ledger.integrations.create_email_draft(
                to=args.to, subject=args.subject, body=args.body, cc=args.cc, bcc=args.bcc
            ), indent=2))
            return
        if args.command == "calendar-draft-create":
            print(json.dumps(ledger.integrations.create_calendar_draft(
                title=args.title, start=args.start, end=args.end, timezone_name=args.timezone,
                attendees=args.attendee, location=args.location, description=args.description
            ), indent=2))
            return
        if args.command == "integration-draft-list":
            print(json.dumps(ledger.integrations.list_drafts(), indent=2))
            return
        if args.command == "integration-draft-show":
            print(json.dumps(ledger.integrations.get_draft(args.draft_id), indent=2))
            return
        if args.command == "integration-draft-request-approval":
            print(json.dumps(ledger.integrations.request_draft_approval(args.draft_id), indent=2))
            return
        if args.command == "integration-draft-decide":
            print(json.dumps(ledger.integrations.decide_draft_approval(
                args.approval_id, approved=args.decision == "approve",
                reviewer=args.reviewer, rationale=args.rationale
            ), indent=2))
            return
        if args.command in {"integration-draft-execute", "browser-verify", "windows-app-launch"}:
            import os
            secret = os.environ.get(args.secret_env)
            if not secret:
                raise ValueError(f"Environment variable {args.secret_env} is required")
            ledger.integrations.bind_provider(OpenClawBridge(
                args.endpoint, secret, allow_remote=args.allow_remote
            ))
            if args.command == "integration-draft-execute":
                print(json.dumps(ledger.integrations.execute_draft(args.draft_id), indent=2))
            elif args.command == "browser-verify":
                verification = {
                    "expected_url_prefix": args.expected_url_prefix or args.url,
                    "title_contains": args.title_contains,
                    "required_text": args.contains,
                    "forbidden_text": args.forbid,
                }
                print(json.dumps(ledger.integrations.navigate_verified(args.url, verification), indent=2))
            else:
                print(json.dumps(ledger.integrations.launch_windows_app(args.app_id, args.argument), indent=2))
            return
        if args.command == "windows-app-register":
            print(json.dumps(ledger.integrations.register_windows_app(
                args.app_id, display_name=args.name, executable_hint=args.executable_hint,
                allowed_argument_patterns=args.allow_arg
            ), indent=2))
            return
        if args.command == "schedule-once":
            print(json.dumps(ledger.scheduler.create_once(
                name=args.name, run_at=args.run_at, goal_utterance=args.goal,
                workspace=args.workspace, autonomy_mode=args.mode
            ), indent=2))
            return
        if args.command == "schedule-interval":
            print(json.dumps(ledger.scheduler.create_interval(
                name=args.name, every_seconds=args.every_seconds, goal_utterance=args.goal,
                first_run_at=args.first_run_at, workspace=args.workspace,
                autonomy_mode=args.mode, max_runs=args.max_runs
            ), indent=2))
            return
        if args.command == "schedule-list":
            print(json.dumps(ledger.scheduler.list(), indent=2))
            return
        if args.command == "schedule-show":
            print(json.dumps(ledger.scheduler.get(args.schedule_id), indent=2))
            return
        if args.command == "schedule-pause":
            print(json.dumps(ledger.scheduler.pause(args.schedule_id), indent=2))
            return
        if args.command == "schedule-cancel":
            print(json.dumps(ledger.scheduler.cancel(args.schedule_id), indent=2))
            return
        if args.command == "schedule-enable":
            print(json.dumps(ledger.scheduler.activate(args.schedule_id), indent=2))
            return
        if args.command == "schedule-tick":
            print(json.dumps(ledger.scheduler.tick(args.worker, max_jobs=args.max_jobs, now=args.now), indent=2))
            return
        if args.command == "schedule-resume":
            print(json.dumps(ledger.scheduler.resume(args.schedule_id, args.worker), indent=2))
            return
        if args.command == "schedule-worker":
            if args.poll_seconds < 1:
                raise ValueError("--poll-seconds must be at least 1")
            try:
                while True:
                    results = ledger.scheduler.tick(args.worker, max_jobs=args.max_jobs)
                    for result in results:
                        print(json.dumps(result, sort_keys=True), flush=True)
                    if args.once:
                        break
                    time.sleep(args.poll_seconds)
            except KeyboardInterrupt:
                pass
            return
        if args.command.startswith("agent-"):
            runtime = ComputerAgentRuntime(ledger, args.workspace) if getattr(args, "workspace", None) else ledger.agent
            if args.command == "agent-compile":
                print(json.dumps(runtime.compile_goal(args.utterance), indent=2))
            elif args.command == "agent-create":
                print(json.dumps(runtime.create_goal(args.utterance, autonomy_mode=args.mode), indent=2))
            elif args.command == "agent-plan":
                print(json.dumps(runtime.plan_goal(args.goal_id), indent=2))
            elif args.command == "agent-run":
                print(json.dumps(runtime.run_until_blocked(args.plan_id), indent=2))
            elif args.command == "agent-goal-show":
                print(json.dumps(runtime.get_goal(args.goal_id), indent=2))
            elif args.command == "agent-plan-show":
                print(json.dumps(runtime.get_plan(args.plan_id), indent=2))
            elif args.command == "agent-approve":
                print(json.dumps(runtime.approve(args.approval_id, reviewer=args.reviewer, rationale=args.rationale), indent=2))
            elif args.command == "agent-skills":
                print(json.dumps(runtime.registry.list_contracts(), indent=2))
            elif args.command == "agent-state":
                print(json.dumps(runtime.machine_state(goal_id=args.goal_id), indent=2))
            elif args.command == "agent-verify":
                if not args.plan_id and not args.receipt_id:
                    raise ValueError("Provide --plan-id or --receipt-id")
                result = {}
                if args.plan_id:
                    result["plan_id"] = args.plan_id
                    result["plan_integrity_valid"] = runtime.verify_plan(args.plan_id)
                if args.receipt_id:
                    result["receipt_id"] = args.receipt_id
                    result["receipt_integrity_valid"] = runtime.verify_receipt(args.receipt_id)
                print(json.dumps(result, indent=2))
            return
        if args.command.startswith("coding-"):
            runtime = CodingRuntime(ledger, args.workspace) if getattr(args, "workspace", None) else ledger.coding
            if args.command == "coding-start":
                spec = _load_coding_test_spec(args.test_spec) if args.test_spec else None
                print(json.dumps(runtime.start_session(
                    args.repository, args.goal, test_spec=spec,
                    allowed_paths=args.allowed_path or ["."], max_candidates=args.max_candidates
                ), indent=2))
            elif args.command == "coding-list":
                print(json.dumps(runtime.list_sessions(), indent=2))
            elif args.command == "coding-show":
                print(json.dumps(runtime.get_session(args.session_id), indent=2))
            elif args.command == "coding-add-candidate":
                proposal = PatchProposal(
                    args.patch.read_text(encoding="utf-8"), args.rationale,
                    provider=args.provider, expected_effect=args.expected_effect
                )
                print(json.dumps(runtime.add_candidate(args.session_id, proposal), indent=2))
            elif args.command == "coding-prepare":
                print(json.dumps(runtime.prepare_candidate(args.candidate_id), indent=2))
            elif args.command == "coding-test-baseline":
                print(json.dumps(runtime.submit_baseline_test(args.session_id), indent=2))
            elif args.command == "coding-test-candidate":
                print(json.dumps(runtime.submit_candidate_test(args.candidate_id), indent=2))
            elif args.command == "coding-test-run":
                engine = None if args.engine == "auto" else CliOCIEngine(args.engine)
                print(json.dumps(runtime.approve_and_execute_test(
                    args.coding_test_id, reviewer=args.reviewer, rationale=args.rationale, engine=engine
                ), indent=2))
            elif args.command == "coding-rank":
                print(json.dumps(runtime.rank_candidates(args.session_id), indent=2))
            elif args.command == "coding-select":
                print(json.dumps(runtime.select_candidate(args.session_id, args.candidate_id), indent=2))
            elif args.command == "coding-promotion-request":
                print(json.dumps(runtime.request_promotion(args.candidate_id), indent=2))
            elif args.command == "coding-approve":
                print(json.dumps(runtime.approve(args.approval_id, reviewer=args.reviewer, rationale=args.rationale), indent=2))
            elif args.command == "coding-promote":
                print(json.dumps(runtime.promote(args.approval_id), indent=2))
            elif args.command == "coding-rollback-request":
                print(json.dumps(runtime.request_rollback(args.promotion_id), indent=2))
            elif args.command == "coding-rollback":
                print(json.dumps(runtime.rollback(args.approval_id), indent=2))
            return
        if args.command.startswith("language-"):
            if args.command == "language-ask":
                print(json.dumps(ledger.language.ask(args.utterance), indent=2))
            elif args.command == "language-interpret":
                print(json.dumps(ledger.language.interpret(args.utterance), indent=2))
            elif args.command == "language-show":
                print(json.dumps(ledger.language.get_response(args.response_id), indent=2))
            elif args.command == "language-verify":
                print(json.dumps({"response_id": args.response_id, "integrity_valid": ledger.language.verify_response(args.response_id)}, indent=2))
            elif args.command == "language-ground":
                print(json.dumps(ledger.language.ground_reference(args.reference, max_depth=args.max_depth).as_dict(), indent=2))
            elif args.command == "language-alias":
                ledger.language.register_predicate_alias(args.alias, args.predicate)
                print(json.dumps({"alias": args.alias, "predicate": args.predicate, "registered": True}, indent=2))
            elif args.command == "language-extract":
                print(json.dumps(ledger.language.extract_relation_proposals(args.text), indent=2))
            elif args.command == "language-ingest":
                print(json.dumps(ledger.language.ingest_controlled_text(
                    args.text,
                    source_uri=args.source_uri,
                    independence_key=args.independence_key,
                    create_evidence=args.create_evidence,
                ), indent=2))
            elif args.command == "language-plan":
                print(json.dumps(ledger.language.plan_discourse(args.utterance), indent=2))
            elif args.command == "language-policy-train":
                examples = json.loads(args.examples.read_text(encoding="utf-8"))
                if not isinstance(examples, list):
                    raise ValueError("Training examples must be a JSON array")
                print(json.dumps(ledger.language.train_semantic_policy(examples, epochs=args.epochs, rate=args.rate), indent=2))
            return
        if args.command.startswith("research-"):
            runtime = EmpiricalResearchRuntime(ledger, args.workspace) if getattr(args, "workspace", None) else ledger.research
            if args.command == "research-create":
                print(json.dumps(runtime.create_study(_load_research_spec(args.spec)), indent=2))
            elif args.command == "research-list":
                print(json.dumps(runtime.list_studies(), indent=2))
            elif args.command == "research-show":
                print(json.dumps(runtime.get_study(args.study_id), indent=2))
            elif args.command == "research-pack":
                print(json.dumps(runtime.export_run_pack(args.study_id, args.arm_key, args.repetition, partition_name=args.partition, out_path=args.out), indent=2))
            elif args.command == "research-import":
                payload = json.loads(args.response.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Research response must be a JSON object")
                print(json.dumps(runtime.import_run(args.study_id, args.arm_key, args.repetition, payload, run_origin=args.origin, partition_name=args.partition, cost_usd=args.cost_usd), indent=2))
            elif args.command == "research-assign-reviews":
                print(json.dumps(runtime.assign_reviews(args.study_id, args.reviewer), indent=2))
            elif args.command == "research-review-export":
                print(json.dumps(runtime.export_review_bundle(args.study_id, args.reviewer, args.out), indent=2))
            elif args.command == "research-review-submit":
                print(json.dumps(runtime.submit_review(args.assignment_id, reviewer=args.reviewer, label=args.label, confidence=args.confidence, rationale=args.rationale), indent=2))
            elif args.command == "research-adjudicate":
                print(json.dumps(runtime.adjudicate(args.study_id, args.research_run_id, args.task_id, adjudicator=args.adjudicator, label=args.label, rationale=args.rationale), indent=2))
            elif args.command == "research-amend":
                amendment = json.loads(args.amendment.read_text(encoding="utf-8"))
                if not isinstance(amendment, dict):
                    raise ValueError("Amendment must be a JSON object")
                print(json.dumps(runtime.add_amendment(args.study_id, amendment, reason=args.reason, actor=args.actor), indent=2))
            elif args.command == "research-report":
                print(json.dumps(runtime.compile_report(args.study_id), indent=2))
            elif args.command == "research-verify":
                print(json.dumps({"study_id": args.study_id, "study_integrity_valid": runtime.verify_study(args.study_id), "report_integrity_valid": runtime.verify_report(args.study_id)}, indent=2))
            elif args.command == "research-bundle":
                print(json.dumps(runtime.export_release_bundle(args.study_id, args.out), indent=2))
            elif args.command == "research-backup":
                print(json.dumps(runtime.backup_database(args.out), indent=2))
            return
        if args.command.startswith("evaluation-"):
            runtime = ComparativeEvaluationRuntime(ledger, args.workspace) if getattr(args, "workspace", None) else ledger.evaluations
            if args.command == "evaluation-default":
                print(json.dumps(runtime.create_suite(default_adversarial_suite()), indent=2))
            elif args.command == "evaluation-create":
                print(json.dumps(runtime.create_suite(_load_evaluation_spec(args.spec)), indent=2))
            elif args.command == "evaluation-list":
                print(json.dumps(runtime.list_suites(), indent=2))
            elif args.command == "evaluation-show":
                print(json.dumps(runtime.get_suite(args.suite_id), indent=2))
            elif args.command == "evaluation-export":
                print(json.dumps(runtime.export_public_suite(args.suite_id, args.out), indent=2))
            elif args.command == "evaluation-import":
                payload = json.loads(args.response.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Evaluation response must be a JSON object")
                print(json.dumps(runtime.import_run(args.suite_id, payload), indent=2))
            elif args.command == "evaluation-fixture":
                print(json.dumps(runtime.create_fixture_run(args.suite_id, args.profile), indent=2))
            elif args.command == "evaluation-run-show":
                print(json.dumps(runtime.get_run(args.run_id), indent=2))
            elif args.command == "evaluation-report":
                print(json.dumps(runtime.compile_report(args.suite_id), indent=2))
            elif args.command == "evaluation-verify":
                result = {
                    "suite_id": args.suite_id,
                    "suite_integrity_valid": runtime.verify_suite(args.suite_id),
                    "report_integrity_valid": runtime.verify_report(args.suite_id),
                }
                if args.run_id:
                    result["run_id"] = args.run_id
                    result["run_integrity_valid"] = runtime.verify_run(args.run_id)
                print(json.dumps(result, indent=2))
            elif args.command == "evaluation-audit-start":
                print(json.dumps(runtime.start_audit(args.run_id, args.task_id, args.auditor), indent=2))
            elif args.command == "evaluation-audit-stop":
                print(json.dumps(runtime.stop_audit(args.audit_id, notes=args.notes, elapsed_seconds=args.elapsed_seconds), indent=2))
            return
        if args.command.startswith("discovery-"):
            runtime = GovernedDiscoveryRuntime(ledger, args.workspace) if getattr(args, "workspace", None) else ledger.discovery
            if args.command == "discovery-create":
                print(json.dumps(runtime.create(_load_discovery_spec(args.spec)), indent=2))
            elif args.command == "discovery-list":
                print(json.dumps(runtime.list(), indent=2))
            elif args.command == "discovery-show":
                print(json.dumps(runtime.get(args.investigation_id), indent=2))
            elif args.command == "discovery-approve":
                inv = runtime.get(args.investigation_id)
                run_id = inv["resume_cursor"].get(f"{inv['current_phase']}_run_id")
                if not run_id:
                    raise ValueError("Investigation has no current execution manifest")
                print(json.dumps(runtime.executions.approve(run_id, reviewer=args.reviewer, rationale=args.rationale), indent=2))
            elif args.command == "discovery-advance":
                engine = CliOCIEngine(args.engine) if args.engine else None
                print(json.dumps(runtime.advance(args.investigation_id, engine=engine), indent=2))
            elif args.command == "discovery-report":
                print(json.dumps(runtime.compile_report(args.investigation_id), indent=2))
            elif args.command == "discovery-verify":
                print(json.dumps({"investigation_id": args.investigation_id, "report_integrity_valid": runtime.verify_report(args.investigation_id)}, indent=2))
            return
        if args.command.startswith("execution-"):
            runtime = ledger.executions
            if getattr(args, "workspace", None) is not None:
                from .execution import ContainerExecutionRuntime
                runtime = ContainerExecutionRuntime(ledger, args.workspace)
            if args.command == "execution-status":
                print(json.dumps(runtime.runtime_status(), indent=2))
            elif args.command == "execution-submit":
                print(json.dumps(runtime.submit(_load_execution_spec(args.spec)), indent=2))
            elif args.command == "execution-approve":
                print(json.dumps(runtime.approve(args.run_id, reviewer=args.reviewer, rationale=args.rationale), indent=2))
            elif args.command == "execution-reject":
                print(json.dumps(runtime.reject(args.run_id, reviewer=args.reviewer, rationale=args.rationale), indent=2))
            elif args.command == "execution-run":
                engine = CliOCIEngine(args.engine) if args.engine else None
                print(json.dumps(runtime.execute(args.run_id, engine=engine), indent=2))
            elif args.command == "execution-show":
                print(json.dumps(runtime.get(args.run_id), indent=2))
            elif args.command == "execution-list":
                print(json.dumps(runtime.list(), indent=2))
            elif args.command == "execution-verify":
                print(json.dumps({
                    "run_id": args.run_id,
                    "manifest_integrity_valid": runtime.verify_manifest(args.run_id),
                    "artifact_integrity_valid": runtime.verify_artifacts(args.run_id),
                    "receipt_integrity_valid": runtime.verify_receipt(args.run_id),
                }, indent=2))
            elif args.command == "execution-reproduce-prepare":
                print(json.dumps(runtime.prepare_reproduction(args.run_id), indent=2))
            return
        if args.command == "proposal-ingest":
            generation = json.loads(args.generation_json)
            if not isinstance(generation, dict):
                raise ValueError("--generation-json must decode to an object")
            system_prompt = (
                args.system_prompt_file.read_text(encoding="utf-8")
                if args.system_prompt_file
                else "External model proposal ingestion"
            )
            user_prompt = (
                args.user_prompt_file.read_text(encoding="utf-8")
                if args.user_prompt_file
                else "Response imported from a model provider"
            )
            batch = ledger.ingest_model_response(
                args.response.read_text(encoding="utf-8"),
                identity=ModelIdentity(args.provider, args.model, args.model_version),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_parameters=generation,
                response_id=args.response_id,
            )
            print(json.dumps(batch, indent=2))
        elif args.command == "proposal-show":
            print(json.dumps(ledger.get_proposal_batch(args.batch_id), indent=2))
        elif args.command == "proposal-list":
            print(json.dumps(ledger.list_proposal_batches(status=args.status), indent=2))
        elif args.command == "proposal-review":
            print(
                json.dumps(
                    ledger.review_proposal_item(
                        args.item_id,
                        args.decision,
                        reviewer=args.reviewer,
                        rationale=args.rationale,
                    ),
                    indent=2,
                )
            )
        elif args.command == "proposal-retry":
            print(json.dumps(ledger.proposals.retry_ready_items(args.batch_id), indent=2))
        elif args.command == "report":
            print(json.dumps(SupportEngine(ledger).evaluate(args.claim_id).as_dict(), indent=2))
        elif args.command == "history":
            print(json.dumps(ledger.history("claim", args.claim_id), indent=2))
        elif args.command == "relation-add":
            qualifiers = json.loads(args.qualifiers_json)
            if not isinstance(qualifiers, dict):
                raise ValueError("--qualifiers-json must decode to an object")
            object_value: Any = args.object
            object_kind: ObjectKind | None = None
            if args.literal_datatype:
                object_value = _literal_from_cli(args.object, args.literal_datatype, args.unit)
                object_kind = ObjectKind.LITERAL
            claim_id = ledger.add_relation_claim(
                args.subject,
                args.predicate,
                object_value,
                subject_type=args.subject_type,
                object_type=args.object_type,
                object_kind=object_kind,
                polarity=not args.negative,
                valid_from=args.valid_from,
                valid_to=args.valid_to,
                qualifiers=qualifiers,
            )
            print(json.dumps(ledger.get_claim(claim_id), indent=2))
        elif args.command == "relation-list":
            if args.negative and args.positive:
                raise ValueError("Choose at most one of --negative or --positive")
            polarity = False if args.negative else True if args.positive else None
            query_object: Any = args.object
            query_kind: ObjectKind | None = None
            if args.object is not None and args.literal_datatype:
                query_object = _literal_from_cli(args.object, args.literal_datatype, args.unit)
                query_kind = ObjectKind.LITERAL
            print(
                json.dumps(
                    ledger.find_relation_claims(
                        subject=args.subject,
                        subject_type=args.subject_type,
                        predicate=args.predicate,
                        object_value=query_object,
                        object_kind=query_kind,
                        object_type=args.object_type,
                        polarity=polarity,
                        valid_at=args.valid_at,
                    ),
                    indent=2,
                )
            )
        elif args.command == "analysis-run":
            print(json.dumps(ledger.run_analysis(_load_analysis_spec(args.spec)), indent=2))
        elif args.command == "analysis-show":
            print(json.dumps(ledger.get_analysis_receipt(args.receipt_id), indent=2))
        elif args.command == "analysis-list":
            print(json.dumps(ledger.list_analysis_receipts(), indent=2))
        elif args.command == "analysis-reproduce":
            print(
                json.dumps(
                    ledger.reproduce_analysis(
                        args.receipt_id,
                        dataset_path=args.dataset_path,
                    ),
                    indent=2,
                )
            )
        elif args.command == "analysis-verify":
            print(
                json.dumps(
                    {
                        "receipt_id": args.receipt_id,
                        "integrity_valid": ledger.analyses.verify_integrity(args.receipt_id),
                    },
                    indent=2,
                )
            )
        elif args.command == "graph-snapshot":
            snapshot = ledger.capture_graph(
                name=args.name,
                root_claim_ids=args.root,
                include_descendants=args.include_descendants,
            )
            formats = tuple(args.format) or ("json", "dot", "svg", "html")
            artifacts = ledger.graphs.render_snapshot(
                snapshot, output_dir=args.out, formats=formats
            )
            print(json.dumps({"snapshot": snapshot, "artifacts": artifacts}, indent=2))
        elif args.command == "graph-list":
            print(json.dumps(ledger.graphs.list_snapshots(), indent=2))
        elif args.command == "graph-show":
            print(json.dumps(ledger.get_graph_snapshot(args.snapshot_id), indent=2))
        elif args.command == "graph-diff":
            diff = ledger.compare_graphs(
                args.before_snapshot_id, args.after_snapshot_id, name=args.name
            )
            formats = tuple(args.format) or ("json", "dot", "svg", "html")
            artifacts = ledger.graphs.render_diff(diff, output_dir=args.out, formats=formats)
            print(json.dumps({"diff": diff, "artifacts": artifacts}, indent=2))
        elif args.command == "graph-diff-list":
            print(json.dumps(ledger.graphs.list_diffs(), indent=2))
        elif args.command == "graph-diff-show":
            print(json.dumps(ledger.get_graph_diff(args.diff_id), indent=2))
        elif args.command == "graph-verify":
            if bool(args.snapshot_id) == bool(args.diff_id):
                raise ValueError("Provide exactly one of --snapshot-id or --diff-id")
            if args.snapshot_id:
                result = {
                    "snapshot_id": args.snapshot_id,
                    "integrity_valid": ledger.graphs.verify_snapshot(args.snapshot_id),
                    "artifacts_valid": ledger.graphs.verify_artifacts(snapshot_id=args.snapshot_id),
                }
            else:
                result = {
                    "diff_id": args.diff_id,
                    "integrity_valid": ledger.graphs.verify_diff(args.diff_id),
                    "artifacts_valid": ledger.graphs.verify_artifacts(diff_id=args.diff_id),
                }
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
