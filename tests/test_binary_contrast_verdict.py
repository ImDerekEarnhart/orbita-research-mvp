from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orbita_discovery.core import Candidate
from orbita_mvp.contrast import analyze_predeclared_contrast
from orbita_mvp.ingestion import profile_dataframe
from orbita_mvp.semantics import derive_finding_record, verdict_presentation
from orbita_mvp.service import ResearchMVP
from orbita_mvp.table_domain import UploadedTableDomain, generate_table_candidates


FIXTURE = Path(__file__).parent / "fixtures" / "t63_binary_contrast.csv"
OUTCOME = "participation_ratio_to_generic_mean"


def contrast_config(**overrides):
    config = {
        "outcome_column": OUTCOME,
        "contrast_column": "is_t63",
        "positive_level": "1",
        "reference_level": "0",
        "block_column": "condition_id",
        "direction": "positive_less_than_reference",
        "primary_effect": "mean_difference",
        "validation_method": "paired_permutation_exact",
    }
    config.update(overrides)
    return config


def test_binary_numeric_profile_and_auto_candidate_are_not_lost():
    df = pd.read_csv(FIXTURE)
    profile = profile_dataframe(df)
    is_t63 = next(row for row in profile["column_profiles"] if row["name"] == "is_t63")
    assert is_t63["kind"] == "numeric"

    auto, metadata = generate_table_candidates(
        df, target_column=OUTCOME, predictor_interpretation="auto", max_candidates=100
    )
    binary = [row for row in auto if row.get("predictor") == "is_t63"]
    assert len(binary) == 1
    assert binary[0]["kind"] == "binary_indicator"
    assert "is_t63" in metadata["binary_indicator_columns"]

    numeric, _ = generate_table_candidates(
        df, target_column=OUTCOME, predictor_interpretation="numeric", max_candidates=100
    )
    numeric_relation = [row for row in numeric if row.get("predictor") == "is_t63"]
    assert len(numeric_relation) == 1
    assert numeric_relation[0]["kind"] == "linear_association"

    categorical, _ = generate_table_candidates(
        df, target_column=OUTCOME, predictor_interpretation="categorical", max_candidates=100
    )
    group_relation = [row for row in categorical if row.get("group") == "is_t63"]
    assert len(group_relation) == 1
    assert group_relation[0]["kind"] == "group_difference"


def test_predeclared_contrast_matches_indicator_regression_and_effect():
    df = pd.read_csv(FIXTURE)
    result = analyze_predeclared_contrast(df, contrast_config())
    assert result["group_counts"] == {"positive": 16, "reference": 16}
    assert result["group_means"]["positive"] == pytest.approx(18.10, abs=1e-5)
    assert result["group_means"]["reference"] == pytest.approx(46.63, abs=1e-5)
    assert result["mean_difference"] == pytest.approx(-28.53, abs=1e-5)
    assert result["ratio"] == pytest.approx(0.38816, abs=2e-5)
    assert result["percentage_change"] == pytest.approx(-61.183787, abs=2e-4)
    assert result["indicator_regression"]["r2"] == pytest.approx(0.596, abs=1e-5)
    assert result["indicator_regression"]["coefficient_equals_mean_difference"] is True
    assert result["matched_pairs"]["direction_consistency_count"] == 16
    assert result["matched_pairs"]["direction_consistency_total"] == 16
    assert result["matched_pairs"]["dropped_incomplete_pairs"] == 0
    assert result["validation_status"] == "validated_in_dataset"
    assert result["interpretation_scope"].startswith("Simulation contrast only")
    assert any("Reachable-state count differs" in value for value in result["interpretation_cautions"])


def test_matched_partitions_and_resamples_never_split_a_condition():
    df = pd.read_csv(FIXTURE)
    candidates, _ = generate_table_candidates(
        df,
        target_column=OUTCOME,
        predictor_interpretation="predeclared_contrast",
        contrast_config=contrast_config(),
    )
    domain = UploadedTableDomain(df, candidates, block_column="condition_id", seed=41)
    partitions = [
        set(domain.scout["condition_id"]),
        set(domain.selection["condition_id"]),
        set(domain.final_validation["condition_id"]),
    ]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert set.union(*partitions) == set(df["condition_id"])
    assert all(len(part) == 2 * part["condition_id"].nunique() for part in (
        domain.scout, domain.selection, domain.final_validation
    ))

    candidate = Candidate(
        id=candidates[0]["id"],
        statement=candidates[0]["statement"],
        payload={key: value for key, value in candidates[0].items() if key not in {"id", "statement", "parents"}},
    )
    train, bootstrap = domain.splits(domain.evidence_for(candidate), seed=9)
    assert set(train["condition_id"]).isdisjoint(set(bootstrap["condition_id"]))
    counts = bootstrap["condition_id"].value_counts()
    assert all(count % 2 == 0 for count in counts)
    for seed in range(8):
        refit_train, refit_val = domain.repeated_refit_split(seed)
        assert set(refit_train["condition_id"]).isdisjoint(set(refit_val["condition_id"]))


def test_contrast_edge_cases_are_explicit_and_conservative():
    blocks = np.repeat([f"b{i}" for i in range(8)], 2)
    levels = np.tile(["generic", "exceptional"], 8)
    no_effect = pd.DataFrame({"block": blocks, "kind": levels, "y": np.repeat(np.arange(8), 2)})
    base = {
        "outcome_column": "y", "contrast_column": "kind",
        "positive_level": "exceptional", "reference_level": "generic",
        "block_column": "block", "direction": "two_sided",
    }
    assert analyze_predeclared_contrast(no_effect, base)["validation_status"] == "not_supported"

    reversed_effect = no_effect.copy()
    reversed_effect.loc[reversed_effect["kind"] == "exceptional", "y"] += 2
    reversed = analyze_predeclared_contrast(
        reversed_effect, {**base, "direction": "positive_less_than_reference"}
    )
    assert reversed["validation_status"] == "direction_contradicted"

    incomplete = reversed_effect.drop(index=[1]).reset_index(drop=True)
    incomplete_result = analyze_predeclared_contrast(incomplete, base)
    assert incomplete_result["matched_pairs"]["complete_pairs"] == 7
    assert incomplete_result["matched_pairs"]["dropped_incomplete_pairs"] == 1

    imbalanced = pd.DataFrame({"kind": ["rare"] * 2 + ["rest"] * 20, "y": [5, 6] + list(range(20))})
    imbalanced_result = analyze_predeclared_contrast(imbalanced, {
        "outcome_column": "y", "contrast_column": "kind",
        "positive_level": "rare", "reference_level": "rest",
    })
    assert any("severely imbalanced" in value for value in imbalanced_result["interpretation_cautions"])

    one_vs_rest = pd.DataFrame({"kind": ["rare", "a", "b", "a", "b", "a"], "y": [-3, -1, 0, 1, 2, 3]})
    rest_result = analyze_predeclared_contrast(one_vs_rest, {
        "outcome_column": "y", "contrast_column": "kind",
        "positive_level": "rare", "reference_level": "__rest__",
    })
    assert rest_result["group_counts"] == {"positive": 1, "reference": 5}
    assert rest_result["mean_difference"] < 0


def test_service_preserves_scores_verdict_copy_and_provisional_claim(tmp_path: Path):
    with ResearchMVP(tmp_path / "orbita.db", tmp_path / "workspace") as service:
        case = service.create_case(name="T63 simulation contrast", goal="")
        service.add_file(case["id"], FIXTURE)
        plan_record = service.compile_case(
            case["id"],
            evaluation_metric="r2",
            investigation_mode="predeclared_contrast",
            predictor_interpretation="predeclared_contrast",
            contrast_config=contrast_config(),
        )
        plan = plan_record["plan"]
        assert plan["candidate_generation"]["analysis_config"]["contrast"]["block_column"] == "condition_id"
        run = service.run_case(case["id"], plan_id=plan_record["id"], auto_approve=True)
        finding = run["result"]["findings"][0]
        assert finding["selection_metric_score"] is not None
        assert finding["final_validation_metric_score"] is not None
        assert finding["public_verdict"] == "provisional"
        assert "surviv" not in finding["verdict_presentation"]["summary"].lower()
        assert finding["finding_detail"]["contrast_analysis"]["matched_pairs"]["complete_pairs"] == 16
        claim_id = run["result"]["belief_import"]["candidate_to_claim"][finding["candidate"]["id"]]
        assert service.ledger.get_claim(claim_id)["status"] == "provisional"


def test_null_scores_round_trip_and_refuted_copy_never_claims_survival():
    finding = {
        "candidate": {"id": "group:x:y", "statement": "Candidate group difference", "payload": {"kind": "group_difference"}},
        "verdict": {"score": None},
        "falsifications": [{"name": "held_out", "killed": True, "detail": {"score": None}}],
        "final_status": "refuted",
        "selection_metric": "r2",
        "selection_metric_score": None,
        "final_validation_metric_score": None,
    }
    record = derive_finding_record(finding, "falsified_candidate")
    encoded = json.loads(json.dumps(record))
    assert encoded["selection_metric_score"] is None
    assert encoded["final_validation_metric_score"] is None
    assert encoded["verdict"] == "rejected"
    assert "surviv" not in encoded["verdict_presentation"]["summary"].lower()
    for state in ("rejected", "not_supported", "inconclusive", "unresolved"):
        assert verdict_presentation(state)["survivor_language"] is False
