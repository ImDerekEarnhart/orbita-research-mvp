from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    status TEXT NOT NULL,
    scope_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias_normalized TEXT NOT NULL,
    alias_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    created_at TEXT NOT NULL,
    PRIMARY KEY(alias_normalized, entity_type)
);

CREATE TABLE IF NOT EXISTS predicates (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL UNIQUE,
    domain_type TEXT,
    range_kind TEXT NOT NULL,
    range_type TEXT,
    inverse_predicate_id TEXT REFERENCES predicates(id),
    symmetric INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS relation_claims (
    claim_id TEXT PRIMARY KEY REFERENCES claims(id),
    subject_entity_id TEXT NOT NULL REFERENCES entities(id),
    predicate_id TEXT NOT NULL REFERENCES predicates(id),
    object_kind TEXT NOT NULL,
    object_entity_id TEXT REFERENCES entities(id),
    literal_json TEXT,
    literal_datatype TEXT,
    literal_unit TEXT,
    polarity INTEGER NOT NULL,
    valid_from TEXT,
    valid_to TEXT,
    qualifiers_json TEXT NOT NULL,
    canonical_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    CHECK (polarity IN (0, 1)),
    CHECK (
        (object_kind = 'entity' AND object_entity_id IS NOT NULL AND literal_json IS NULL)
        OR
        (object_kind = 'literal' AND object_entity_id IS NULL AND literal_json IS NOT NULL)
    )
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    excerpt TEXT NOT NULL,
    independence_key TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attestations (
    id TEXT PRIMARY KEY,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    evidence_id TEXT NOT NULL REFERENCES evidence(id),
    stance TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(claim_id, evidence_id, stance)
);

CREATE TABLE IF NOT EXISTS proofs (
    id TEXT PRIMARY KEY,
    conclusion_claim_id TEXT NOT NULL REFERENCES claims(id),
    rule TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS proof_premises (
    proof_id TEXT NOT NULL REFERENCES proofs(id),
    premise_claim_id TEXT NOT NULL REFERENCES claims(id),
    position INTEGER NOT NULL,
    PRIMARY KEY(proof_id, premise_claim_id)
);

CREATE TABLE IF NOT EXISTS contradictions (
    id TEXT PRIMARY KEY,
    claim_a TEXT NOT NULL REFERENCES claims(id),
    claim_b TEXT NOT NULL REFERENCES claims(id),
    rationale TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    actor TEXT NOT NULL,
    actor_role TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    sequence INTEGER NOT NULL,
    intent TEXT NOT NULL,
    action_type TEXT NOT NULL,
    args_json TEXT NOT NULL,
    required_claims_json TEXT NOT NULL,
    obligations_json TEXT NOT NULL,
    risk TEXT NOT NULL,
    status TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(plan_id, sequence)
);

CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES steps(id),
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    approved_by TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT
);

CREATE TABLE IF NOT EXISTS action_receipts (
    id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES steps(id),
    ok INTEGER NOT NULL,
    outputs_json TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS analysis_receipts (
    id TEXT PRIMARY KEY,
    analysis_type TEXT NOT NULL,
    status TEXT NOT NULL,
    dataset_uri TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    dataset_size_bytes INTEGER NOT NULL,
    code_hash TEXT NOT NULL,
    code_identity TEXT NOT NULL,
    environment_json TEXT NOT NULL,
    schema_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    outputs_json TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    parent_receipt_id TEXT REFERENCES analysis_receipts(id),
    comparison_json TEXT NOT NULL,
    receipt_hash TEXT NOT NULL UNIQUE,
    evidence_id TEXT REFERENCES evidence(id),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_artifacts (
    id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL REFERENCES analysis_receipts(id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS proposal_batches (
    id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    response_id TEXT,
    status TEXT NOT NULL,
    system_prompt TEXT NOT NULL,
    user_prompt TEXT NOT NULL,
    system_prompt_hash TEXT NOT NULL,
    user_prompt_hash TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    raw_response TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    generation_parameters_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    errors_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS proposal_items (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES proposal_batches(id),
    position INTEGER NOT NULL,
    local_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    rationale TEXT NOT NULL,
    requires_human_review INTEGER NOT NULL DEFAULT 0,
    review_reason TEXT,
    durable_entity_type TEXT,
    durable_entity_id TEXT,
    error_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewed_at TEXT,
    reviewed_by TEXT,
    UNIQUE(batch_id, local_id),
    UNIQUE(batch_id, position)
);

CREATE TABLE IF NOT EXISTS proposal_dependencies (
    item_id TEXT NOT NULL REFERENCES proposal_items(id),
    depends_on_item_id TEXT NOT NULL REFERENCES proposal_items(id),
    dependency_kind TEXT NOT NULL,
    PRIMARY KEY(item_id, depends_on_item_id, dependency_kind)
);

CREATE TABLE IF NOT EXISTS analysis_claim_assessments (
    id TEXT PRIMARY KEY,
    receipt_id TEXT NOT NULL REFERENCES analysis_receipts(id),
    position INTEGER NOT NULL,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    metric_path TEXT NOT NULL,
    metric_value_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    support_condition_json TEXT NOT NULL,
    refute_condition_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    evidence_id TEXT REFERENCES evidence(id),
    created_at TEXT NOT NULL,
    UNIQUE(receipt_id, position)
);

CREATE TABLE IF NOT EXISTS graph_snapshots (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    root_claim_ids_json TEXT NOT NULL,
    include_descendants INTEGER NOT NULL DEFAULT 0,
    graph_json TEXT NOT NULL,
    graph_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_diffs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    before_snapshot_id TEXT NOT NULL REFERENCES graph_snapshots(id),
    after_snapshot_id TEXT NOT NULL REFERENCES graph_snapshots(id),
    diff_json TEXT NOT NULL,
    diff_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_artifacts (
    id TEXT PRIMARY KEY,
    snapshot_id TEXT REFERENCES graph_snapshots(id),
    diff_id TEXT REFERENCES graph_diffs(id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK ((snapshot_id IS NOT NULL AND diff_id IS NULL)
        OR (snapshot_id IS NULL AND diff_id IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS execution_runs (
    id TEXT PRIMARY KEY,
    parent_run_id TEXT REFERENCES execution_runs(id),
    name TEXT NOT NULL,
    status TEXT NOT NULL,
    image_ref TEXT NOT NULL,
    image_digest TEXT NOT NULL,
    command_json TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL UNIQUE,
    run_root TEXT NOT NULL,
    required_claims_json TEXT NOT NULL,
    output_obligations_json TEXT NOT NULL,
    claim_tests_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    engine_used TEXT,
    exit_code INTEGER,
    timed_out INTEGER NOT NULL DEFAULT 0,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    checks_json TEXT NOT NULL DEFAULT '[]',
    receipt_json TEXT NOT NULL DEFAULT '{}',
    receipt_hash TEXT,
    evidence_id TEXT REFERENCES evidence(id),
    comparison_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES execution_runs(id),
    manifest_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS execution_artifacts (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES execution_runs(id),
    role TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, role, relative_path)
);

CREATE TABLE IF NOT EXISTS execution_claim_assessments (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES execution_runs(id),
    position INTEGER NOT NULL,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    output_path TEXT NOT NULL,
    metric_path TEXT NOT NULL,
    metric_value_json TEXT NOT NULL,
    outcome TEXT NOT NULL,
    support_condition_json TEXT NOT NULL,
    refute_condition_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    evidence_id TEXT REFERENCES evidence(id),
    attestation_id TEXT REFERENCES attestations(id),
    created_at TEXT NOT NULL,
    UNIQUE(run_id, position)
);

CREATE TABLE IF NOT EXISTS discovery_investigations (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,
    current_phase TEXT NOT NULL,
    dataset_uri TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    dataset_size_bytes INTEGER NOT NULL,
    replication_dataset_uri TEXT,
    replication_dataset_hash TEXT,
    replication_dataset_size_bytes INTEGER,
    spec_json TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    budget_used_json TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    resume_cursor_json TEXT NOT NULL,
    report_json TEXT NOT NULL DEFAULT '{}',
    report_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS discovery_hypotheses (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES discovery_investigations(id),
    position INTEGER NOT NULL,
    claim_id TEXT NOT NULL REFERENCES claims(id),
    x_column TEXT NOT NULL,
    y_column TEXT NOT NULL,
    direction TEXT NOT NULL,
    origin TEXT NOT NULL,
    rationale TEXT NOT NULL,
    discovery_metrics_json TEXT NOT NULL,
    preregistration_json TEXT NOT NULL,
    status TEXT NOT NULL,
    confirmation_run_id TEXT REFERENCES execution_runs(id),
    replication_run_id TEXT REFERENCES execution_runs(id),
    confirmation_result_json TEXT NOT NULL DEFAULT '{}',
    replication_result_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(investigation_id, position)
);

CREATE TABLE IF NOT EXISTS discovery_artifacts (
    id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL REFERENCES discovery_investigations(id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(investigation_id, role)
);

CREATE TABLE IF NOT EXISTS evaluation_suites (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,
    seed INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    suite_hash TEXT NOT NULL UNIQUE,
    metadata_json TEXT NOT NULL,
    report_json TEXT NOT NULL DEFAULT '{}',
    report_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_tasks (
    id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(id),
    task_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    category TEXT NOT NULL,
    prompt TEXT NOT NULL,
    public_json TEXT NOT NULL,
    gold_json TEXT NOT NULL,
    task_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(suite_id, task_key),
    UNIQUE(suite_id, position)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(id),
    system_kind TEXT NOT NULL,
    system_name TEXT NOT NULL,
    system_version TEXT,
    provider TEXT,
    evaluation_mode TEXT NOT NULL,
    config_json TEXT NOT NULL,
    system_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    response_json TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    UNIQUE(suite_id, response_hash)
);

CREATE TABLE IF NOT EXISTS evaluation_task_results (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES evaluation_runs(id),
    task_id TEXT NOT NULL REFERENCES evaluation_tasks(id),
    response_json TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    score_json TEXT NOT NULL,
    result_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, task_id)
);

CREATE TABLE IF NOT EXISTS evaluation_audits (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES evaluation_runs(id),
    task_id TEXT NOT NULL REFERENCES evaluation_tasks(id),
    auditor TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    elapsed_seconds REAL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_artifacts (
    id TEXT PRIMARY KEY,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(suite_id, role)
);



CREATE TABLE IF NOT EXISTS research_studies (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    suite_id TEXT NOT NULL REFERENCES evaluation_suites(id),
    status TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    spec_hash TEXT NOT NULL UNIQUE,
    suite_hash TEXT NOT NULL,
    partition_json TEXT NOT NULL,
    partition_hash TEXT NOT NULL,
    preregistration_json TEXT NOT NULL,
    preregistration_hash TEXT NOT NULL,
    report_json TEXT NOT NULL,
    report_hash TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_arms (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    arm_key TEXT NOT NULL,
    name TEXT NOT NULL,
    system_kind TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(study_id, arm_key)
);

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    arm_id TEXT NOT NULL REFERENCES research_arms(id),
    repetition INTEGER NOT NULL,
    partition_name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    blind_code TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    pack_json TEXT NOT NULL,
    pack_hash TEXT NOT NULL,
    evaluation_run_id TEXT REFERENCES evaluation_runs(id),
    response_hash TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    token_usage_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(study_id, arm_id, repetition, partition_name)
);

CREATE TABLE IF NOT EXISTS research_amendments (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    sequence INTEGER NOT NULL,
    amendment_json TEXT NOT NULL,
    amendment_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(study_id, sequence)
);

CREATE TABLE IF NOT EXISTS research_review_assignments (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    research_run_id TEXT NOT NULL REFERENCES research_runs(id),
    task_id TEXT NOT NULL REFERENCES evaluation_tasks(id),
    reviewer TEXT NOT NULL,
    blind_code TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(research_run_id, task_id, reviewer)
);

CREATE TABLE IF NOT EXISTS research_reviews (
    id TEXT PRIMARY KEY,
    assignment_id TEXT NOT NULL UNIQUE REFERENCES research_review_assignments(id),
    reviewer TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    review_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS research_adjudications (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    research_run_id TEXT NOT NULL REFERENCES research_runs(id),
    task_id TEXT NOT NULL REFERENCES evaluation_tasks(id),
    adjudicator TEXT NOT NULL,
    label TEXT NOT NULL,
    rationale TEXT NOT NULL,
    adjudication_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(research_run_id, task_id)
);

CREATE TABLE IF NOT EXISTS research_artifacts (
    id TEXT PRIMARY KEY,
    study_id TEXT NOT NULL REFERENCES research_studies(id),
    role TEXT NOT NULL,
    path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(study_id, role)
);


CREATE TABLE IF NOT EXISTS language_predicate_aliases (
    alias_normalized TEXT PRIMARY KEY,
    alias_text TEXT NOT NULL,
    predicate_id TEXT NOT NULL REFERENCES predicates(id),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS language_responses (
    id TEXT PRIMARY KEY,
    utterance TEXT NOT NULL,
    intent TEXT NOT NULL,
    frame_json TEXT NOT NULL,
    grounding_json TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    status TEXT NOT NULL,
    response_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sentence_warrants (
    id TEXT PRIMARY KEY,
    response_id TEXT NOT NULL REFERENCES language_responses(id),
    position INTEGER NOT NULL,
    sentence TEXT NOT NULL,
    semantic_act TEXT NOT NULL,
    claim_ids_json TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    proof_ids_json TEXT NOT NULL,
    support_state TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    sentence_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(response_id, position)
);


CREATE TABLE IF NOT EXISTS discourse_plans (
    id TEXT PRIMARY KEY,
    response_id TEXT REFERENCES language_responses(id),
    utterance TEXT NOT NULL,
    frame_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    candidate_moves_json TEXT NOT NULL,
    selected_moves_json TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_trace_json TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS semantic_policy_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    feature_schema_json TEXT NOT NULL,
    weights_json TEXT NOT NULL,
    training_metadata_json TEXT NOT NULL,
    model_hash TEXT NOT NULL UNIQUE,
    active INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS computer_goals (
    id TEXT PRIMARY KEY,
    utterance TEXT NOT NULL,
    goal_type TEXT NOT NULL,
    structured_json TEXT NOT NULL,
    workspace TEXT NOT NULL,
    autonomy_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS computer_plans (
    id TEXT PRIMARY KEY,
    goal_id TEXT NOT NULL REFERENCES computer_goals(id),
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    plan_hash TEXT NOT NULL,
    rationale_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(goal_id, revision)
);

CREATE TABLE IF NOT EXISTS computer_steps (
    id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL REFERENCES computer_plans(id),
    step_key TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    skill_name TEXT NOT NULL,
    args_json TEXT NOT NULL,
    args_hash TEXT NOT NULL,
    depends_on_json TEXT NOT NULL,
    obligations_json TEXT NOT NULL,
    required_claims_json TEXT NOT NULL,
    risk TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(plan_id, step_key),
    UNIQUE(plan_id, sequence)
);

CREATE TABLE IF NOT EXISTS computer_approvals (
    id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES computer_steps(id),
    args_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS computer_receipts (
    id TEXT PRIMARY KEY,
    step_id TEXT NOT NULL REFERENCES computer_steps(id),
    ok INTEGER NOT NULL,
    outputs_json TEXT NOT NULL,
    checks_json TEXT NOT NULL,
    error TEXT,
    receipt_hash TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS computer_state_snapshots (
    id TEXT PRIMARY KEY,
    goal_id TEXT REFERENCES computer_goals(id),
    workspace TEXT NOT NULL,
    state_json TEXT NOT NULL,
    state_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS coding_sessions (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    repository_path TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    base_branch TEXT NOT NULL,
    allowed_paths_json TEXT NOT NULL,
    test_spec_json TEXT NOT NULL,
    max_candidates INTEGER NOT NULL,
    status TEXT NOT NULL,
    session_hash TEXT NOT NULL,
    initial_status_json TEXT NOT NULL,
    selected_candidate_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coding_candidates (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES coding_sessions(id),
    position INTEGER NOT NULL,
    provider TEXT NOT NULL,
    rationale TEXT NOT NULL,
    expected_effect TEXT NOT NULL,
    patch_text TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    changed_paths_json TEXT NOT NULL,
    status TEXT NOT NULL,
    worktree_path TEXT,
    applied_diff_text TEXT NOT NULL DEFAULT '',
    applied_diff_hash TEXT,
    static_checks_json TEXT NOT NULL,
    diff_stats_json TEXT NOT NULL,
    score_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(session_id, position),
    UNIQUE(session_id, patch_hash)
);

CREATE TABLE IF NOT EXISTS coding_tests (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES coding_sessions(id),
    candidate_id TEXT REFERENCES coding_candidates(id),
    phase TEXT NOT NULL,
    execution_run_id TEXT NOT NULL REFERENCES execution_runs(id),
    status TEXT NOT NULL,
    diagnostics_json TEXT NOT NULL,
    receipt_hash TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS coding_approvals (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES coding_sessions(id),
    candidate_id TEXT REFERENCES coding_candidates(id),
    action TEXT NOT NULL,
    binding_json TEXT NOT NULL,
    binding_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS coding_promotions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES coding_sessions(id),
    candidate_id TEXT NOT NULL REFERENCES coding_candidates(id),
    approval_id TEXT NOT NULL REFERENCES coding_approvals(id),
    repository_path TEXT NOT NULL,
    base_commit TEXT NOT NULL,
    patch_text TEXT NOT NULL,
    patch_hash TEXT NOT NULL,
    post_diff_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    receipt_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    rolled_back_at TEXT,
    rollback_receipt_hash TEXT
);

CREATE INDEX IF NOT EXISTS idx_coding_sessions_status ON coding_sessions(status, created_at);
CREATE INDEX IF NOT EXISTS idx_coding_candidates_session ON coding_candidates(session_id, position);
CREATE INDEX IF NOT EXISTS idx_coding_candidates_status ON coding_candidates(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_coding_tests_session ON coding_tests(session_id, phase, created_at);
CREATE INDEX IF NOT EXISTS idx_coding_tests_candidate ON coding_tests(candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_coding_approvals_session ON coding_approvals(session_id, action, status);
CREATE INDEX IF NOT EXISTS idx_coding_promotions_session ON coding_promotions(session_id, created_at);



CREATE TABLE IF NOT EXISTS integration_drafts (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS integration_approvals (
    id TEXT PRIMARY KEY,
    draft_id TEXT NOT NULL REFERENCES integration_drafts(id),
    payload_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS integration_receipts (
    id TEXT PRIMARY KEY,
    draft_id TEXT REFERENCES integration_drafts(id),
    action_kind TEXT NOT NULL,
    provider TEXT NOT NULL,
    capability TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    response_json TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    receipt_hash TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS windows_app_registry (
    app_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    executable_hint TEXT NOT NULL,
    allowed_argument_patterns_json TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule_kind TEXT NOT NULL,
    goal_utterance TEXT NOT NULL,
    workspace TEXT,
    autonomy_mode TEXT NOT NULL,
    next_run_at TEXT NOT NULL,
    interval_seconds INTEGER,
    max_runs INTEGER,
    run_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    schedule_hash TEXT NOT NULL,
    active_run_id TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT,
    last_error TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduled_job_runs (
    id TEXT PRIMARY KEY,
    schedule_id TEXT NOT NULL REFERENCES scheduled_jobs(id),
    scheduled_for TEXT NOT NULL,
    goal_id TEXT NOT NULL REFERENCES computer_goals(id),
    plan_id TEXT NOT NULL REFERENCES computer_plans(id),
    status TEXT NOT NULL,
    result_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    run_hash TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_integration_drafts_status ON integration_drafts(status, created_at);
CREATE INDEX IF NOT EXISTS idx_integration_approvals_draft ON integration_approvals(draft_id, status);
CREATE INDEX IF NOT EXISTS idx_integration_receipts_draft ON integration_receipts(draft_id, created_at);
CREATE INDEX IF NOT EXISTS idx_scheduled_jobs_due ON scheduled_jobs(status, next_run_at, lease_expires_at);
CREATE INDEX IF NOT EXISTS idx_scheduled_runs_schedule ON scheduled_job_runs(schedule_id, created_at);



CREATE TABLE IF NOT EXISTS desktop_observations (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    provider TEXT NOT NULL,
    active_app TEXT NOT NULL,
    window_title TEXT NOT NULL,
    screen_json TEXT NOT NULL,
    elements_json TEXT NOT NULL,
    accessibility_fingerprint TEXT NOT NULL,
    visual_fingerprint TEXT,
    state_fingerprint TEXT NOT NULL,
    screenshot_path TEXT,
    screenshot_hash TEXT,
    payload_json TEXT NOT NULL,
    observation_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS desktop_observation_diffs (
    id TEXT PRIMARY KEY,
    before_id TEXT NOT NULL REFERENCES desktop_observations(id),
    after_id TEXT NOT NULL REFERENCES desktop_observations(id),
    diff_json TEXT NOT NULL,
    drift_score REAL NOT NULL,
    diff_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS desktop_actions (
    id TEXT PRIMARY KEY,
    observation_id TEXT NOT NULL REFERENCES desktop_observations(id),
    action_kind TEXT NOT NULL,
    selector_json TEXT NOT NULL,
    target_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    risk TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS desktop_action_approvals (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES desktop_actions(id),
    action_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer TEXT,
    rationale TEXT,
    created_at TEXT NOT NULL,
    decided_at TEXT,
    consumed_at TEXT
);

CREATE TABLE IF NOT EXISTS desktop_action_receipts (
    id TEXT PRIMARY KEY,
    action_id TEXT NOT NULL REFERENCES desktop_actions(id),
    provider TEXT NOT NULL,
    response_json TEXT NOT NULL,
    post_observation_id TEXT REFERENCES desktop_observations(id),
    checks_json TEXT NOT NULL,
    ok INTEGER NOT NULL,
    error TEXT,
    receipt_hash TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adaptive_workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    status TEXT NOT NULL,
    source_plan_id TEXT REFERENCES computer_plans(id),
    definition_json TEXT NOT NULL,
    workflow_hash TEXT NOT NULL UNIQUE,
    required_state_fingerprint TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adaptive_workflow_reviews (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES adaptive_workflows(id),
    workflow_hash TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS adaptive_workflow_instances (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL REFERENCES adaptive_workflows(id),
    workflow_hash TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    goal_id TEXT NOT NULL REFERENCES computer_goals(id),
    plan_id TEXT NOT NULL REFERENCES computer_plans(id),
    binding_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_desktop_observations_created ON desktop_observations(created_at);
CREATE INDEX IF NOT EXISTS idx_desktop_diffs_pair ON desktop_observation_diffs(before_id, after_id);
CREATE INDEX IF NOT EXISTS idx_desktop_actions_status ON desktop_actions(status, created_at);
CREATE INDEX IF NOT EXISTS idx_desktop_action_approvals ON desktop_action_approvals(action_id, status);
CREATE INDEX IF NOT EXISTS idx_adaptive_workflows_status ON adaptive_workflows(status, created_at);
CREATE INDEX IF NOT EXISTS idx_adaptive_instances_workflow ON adaptive_workflow_instances(workflow_id, created_at);

CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);



CREATE INDEX IF NOT EXISTS idx_discourse_plans_response ON discourse_plans(response_id, created_at);
CREATE INDEX IF NOT EXISTS idx_semantic_policy_active ON semantic_policy_models(active, created_at);
CREATE INDEX IF NOT EXISTS idx_computer_goals_status ON computer_goals(status, created_at);
CREATE INDEX IF NOT EXISTS idx_computer_plans_goal ON computer_plans(goal_id, revision);
CREATE INDEX IF NOT EXISTS idx_computer_steps_plan ON computer_steps(plan_id, sequence);
CREATE INDEX IF NOT EXISTS idx_computer_steps_status ON computer_steps(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_computer_approvals_step ON computer_approvals(step_id, status);
CREATE INDEX IF NOT EXISTS idx_computer_receipts_step ON computer_receipts(step_id, created_at);

CREATE INDEX IF NOT EXISTS idx_language_alias_predicate ON language_predicate_aliases(predicate_id);
CREATE INDEX IF NOT EXISTS idx_language_responses_created ON language_responses(created_at);
CREATE INDEX IF NOT EXISTS idx_sentence_warrants_response ON sentence_warrants(response_id, position);

CREATE INDEX IF NOT EXISTS idx_evaluation_tasks_suite ON evaluation_tasks(suite_id, position);
CREATE INDEX IF NOT EXISTS idx_evaluation_runs_suite ON evaluation_runs(suite_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evaluation_results_run ON evaluation_task_results(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_audits_run ON evaluation_audits(run_id, task_id);
CREATE INDEX IF NOT EXISTS idx_evaluation_artifacts_suite ON evaluation_artifacts(suite_id, role);


CREATE INDEX IF NOT EXISTS idx_research_studies_suite ON research_studies(suite_id, created_at);
CREATE INDEX IF NOT EXISTS idx_research_arms_study ON research_arms(study_id, arm_key);
CREATE INDEX IF NOT EXISTS idx_research_runs_study ON research_runs(study_id, arm_id, repetition);
CREATE INDEX IF NOT EXISTS idx_research_assignments_reviewer ON research_review_assignments(study_id, reviewer, status);
CREATE INDEX IF NOT EXISTS idx_research_reviews_assignment ON research_reviews(assignment_id);
CREATE INDEX IF NOT EXISTS idx_research_artifacts_study ON research_artifacts(study_id, role);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(normalized_name, entity_type);
CREATE INDEX IF NOT EXISTS idx_aliases_entity ON entity_aliases(entity_id);
CREATE INDEX IF NOT EXISTS idx_predicates_name ON predicates(normalized_name);
CREATE INDEX IF NOT EXISTS idx_relation_sp ON relation_claims(subject_entity_id, predicate_id);
CREATE INDEX IF NOT EXISTS idx_relation_object_entity ON relation_claims(object_entity_id);
CREATE INDEX IF NOT EXISTS idx_relation_polarity ON relation_claims(polarity);
CREATE INDEX IF NOT EXISTS idx_attestations_claim ON attestations(claim_id);
CREATE INDEX IF NOT EXISTS idx_proofs_conclusion ON proofs(conclusion_claim_id);
CREATE INDEX IF NOT EXISTS idx_premises_claim ON proof_premises(premise_claim_id);
CREATE INDEX IF NOT EXISTS idx_events_entity ON events(entity_type, entity_id, id);

CREATE INDEX IF NOT EXISTS idx_proposal_batches_status ON proposal_batches(status, created_at);
CREATE INDEX IF NOT EXISTS idx_proposal_items_batch ON proposal_items(batch_id, position);
CREATE INDEX IF NOT EXISTS idx_proposal_items_status ON proposal_items(status, item_type);
CREATE INDEX IF NOT EXISTS idx_proposal_dependencies_item ON proposal_dependencies(item_id);
CREATE INDEX IF NOT EXISTS idx_analysis_dataset_hash ON analysis_receipts(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_analysis_parent ON analysis_receipts(parent_receipt_id);
CREATE INDEX IF NOT EXISTS idx_analysis_claim ON analysis_claim_assessments(claim_id);
CREATE INDEX IF NOT EXISTS idx_analysis_artifact_receipt ON analysis_artifacts(receipt_id);
CREATE INDEX IF NOT EXISTS idx_graph_snapshot_created ON graph_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_graph_diff_before_after ON graph_diffs(before_snapshot_id, after_snapshot_id);
CREATE INDEX IF NOT EXISTS idx_graph_artifact_snapshot ON graph_artifacts(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_graph_artifact_diff ON graph_artifacts(diff_id);
CREATE INDEX IF NOT EXISTS idx_execution_runs_status ON execution_runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_execution_runs_parent ON execution_runs(parent_run_id);
CREATE INDEX IF NOT EXISTS idx_execution_approvals_run ON execution_approvals(run_id, status);
CREATE INDEX IF NOT EXISTS idx_execution_artifacts_run ON execution_artifacts(run_id, role);
CREATE INDEX IF NOT EXISTS idx_execution_assessments_claim ON execution_claim_assessments(claim_id);
CREATE INDEX IF NOT EXISTS idx_discovery_status ON discovery_investigations(status, created_at);
CREATE INDEX IF NOT EXISTS idx_discovery_hypotheses_investigation ON discovery_hypotheses(investigation_id, position);
CREATE INDEX IF NOT EXISTS idx_discovery_hypotheses_claim ON discovery_hypotheses(claim_id);
CREATE INDEX IF NOT EXISTS idx_discovery_artifacts_investigation ON discovery_artifacts(investigation_id, role);
"""


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI executes ordinary synchronous endpoint functions in a worker
        # thread. The local MVP intentionally shares one ledger connection, so
        # thread affinity must be disabled. SQLite itself is compiled in
        # serialized mode in supported CPython builds; WAL and a busy timeout
        # reduce lock contention for the single-user local service.
        self.conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 30000")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT INTO schema_meta(key, value, updated_at) VALUES ('schema_version', '1.5.0', datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at"
        )
        self.conn.commit()

    def backup(self, destination: str | Path) -> Path:
        target = Path(destination).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as out:
            self.conn.backup(out)
        return target

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
