# AI operator guide

Orbita Research MVP can be controlled by a person through the browser or by an external AI through the REST API. The external AI is a **research compiler and interpreter**, not the final judge.

## Operating contract

The AI may:

- inspect case metadata and deterministic file profiles;
- propose candidate relationships and assumptions;
- ask high-value clarification questions;
- submit an explicit analysis plan;
- summarize engine results and belief history.

The AI must not:

- invent fields that are absent from the case context;
- silently reinterpret identifiers, units, groups, or missing-value codes;
- describe association as causation;
- change a frozen plan after seeing confirmation results;
- treat a finite-data survivor as universal proof;
- delete superseded or weakened beliefs.

## Agent workflow

1. `POST /cases`
2. `POST /cases/{case_id}/files`
3. `GET /cases/{case_id}/context`
4. Either:
   - `POST /cases/{case_id}/compile` to use Orbita's deterministic compiler, or
   - `POST /cases/{case_id}/plans` to submit an AI-authored plan.
5. Review the returned plan and resolve blocking questions.
6. `POST /plans/{plan_id}/approve`
7. `POST /cases/{case_id}/run`
8. Read:
   - `GET /runs/{run_id}`
   - `GET /cases/{case_id}/claims`
   - `GET /claims/{claim_id}/history`
   - `GET /claims/{claim_id}/impact`
   - `GET /cases/{case_id}/report`
9. When new evidence arrives, use evidence, contradiction, supersession, or derivation endpoints rather than rewriting history.

## Current candidate schema

The v0.1 uploaded-table engine accepts two candidate kinds.

### Linear association

```json
{
  "id": "linear:marker_a:response",
  "statement": "marker_a and response show a stable positive linear association.",
  "kind": "linear_association",
  "predictor": "marker_a",
  "outcome": "response",
  "expected_direction": "positive",
  "parents": []
}
```

### Group difference

```json
{
  "id": "group:treatment:response",
  "statement": "response differs systematically across levels of treatment.",
  "kind": "group_difference",
  "group": "treatment",
  "outcome": "response",
  "parents": []
}
```

`parents` contains candidate IDs in the same plan. All parents within one recorded derivation are required together; multiple separate derivations act as alternative support routes.

## Prompt for an external AI

```text
You are operating Orbita Research MVP as a cautious research compiler.

Read the case context and create an explicit plan using only fields that appear in the uploaded data profile. Preserve the user's original goal. When the goal is blank, use Open Discovery: propose a bounded set of high-value candidates rather than exhaustive pattern mining.

For every candidate include:
- stable ID
- plain-language statement
- supported candidate kind
- exact input fields
- expected direction when applicable
- parent candidate IDs only when a real derivation exists

Also include:
- assumptions requiring human review
- data-quality concerns
- blocking clarification questions
- held-out and cross-seed thresholds
- report modules

Treat all generated relations as non-causal unless the approved design supports causal inference. Do not inspect confirmation results before the candidate list is frozen. Return a complete plan object for POST /cases/{case_id}/plans.
```

## Interpreting results

A held-out score is an R²-style predictive score for the current table route. It is not a p-value or posterior probability.

- `supported`: crossed the judge threshold and survived configured attacks.
- `provisional` or `challenged`: retained some support but has limitations or tension.
- `refuted`: at least one configured attack killed the candidate.
- `unknown`: insufficient evidence.

The AI's final explanation should distinguish:

1. what the user supplied;
2. what the compiler inferred;
3. what the approved plan froze;
4. what the engine measured;
5. what remains speculative.
