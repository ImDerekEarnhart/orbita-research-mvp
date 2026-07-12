from __future__ import annotations

from fastapi.testclient import TestClient

from orbita_mvp.finite_affine import (
    AffineOperator,
    analyze_commutator_family,
    generate_compatible_lift_families,
    group_orbits,
    infer_affine_operators,
)


def _observations(operator: AffineOperator) -> list[tuple[int, int]]:
    return [(value, operator.apply(value)) for value in range(operator.modulus)]


def test_tzolkin_raw_maps_recover_exact_operators_and_harmonic_quads():
    antipode = AffineOperator(260, 1, 130, "antipode")
    occult = AffineOperator(260, -1, 259, "occult")
    analog = AffineOperator(260, 79, 117, "analog")

    for expected in (antipode, occult, analog):
        inferred = infer_affine_operators(_observations(expected), 260)
        assert len(inferred) == 1
        assert inferred[0].multiplier == expected.multiplier
        assert inferred[0].translation == expected.translation
        assert inferred[0].order() == 2

    quads = group_orbits([antipode, occult])
    assert len(quads) == 65
    assert all(len(orbit) == 4 for orbit in quads)
    assert quads[0] == (0, 129, 130, 259)

    assert antipode.commutator(occult) == AffineOperator.identity(260)
    commutator = occult.commutator(analog)
    assert commutator.is_translation
    assert commutator.translation == 104
    assert commutator.order() == 5


def test_calendar_round_lifts_find_unique_exception_and_exact_condition():
    base = AffineOperator(260, 79, 117, "base analog")
    occult_lift = AffineOperator(18_980, -1, -1, "occult lift")
    families = generate_compatible_lift_families(
        base,
        18_980,
        multiplier_constraints=[(-1, 365)],
    )

    assert len(families) == 1
    family = families[0]
    assert family.lifted_multiplier == 10_219
    assert len(family.members) == 73
    assert family.members[0].operator.translation == 117
    assert family.members[63].operator.translation == (117 + 260 * 63) % 18_980

    result = analyze_commutator_family(occult_lift, family)
    assert result["result_type"] == "exact_finite_classification"
    assert result["epistemic_status"] == "verified_within_finite_scope"
    assert result["commutator_formula"]["translation_intercept"] == 9_984
    assert result["commutator_formula"]["translation_slope"] == -520
    assert result["order_formula"] == {
        "numerator": 365,
        "gcd_modulus": 365,
        "intercept": 192,
        "slope": -10,
        "common_factor_removed": 52,
        "rendered": "365/gcd(365, 192 -10*t)",
    }

    generic = next(regime for regime in result["regimes"] if regime["kind"] == "generic")
    exceptional = next(
        regime for regime in result["regimes"] if regime["kind"] == "exceptional"
    )
    assert generic == {"kind": "generic", "commutator_order": 365, "count": 72}
    assert exceptional["commutator_order"] == 5
    assert exceptional["parameters"] == [63]
    assert exceptional["condition"] == "t = 63 (mod 73)"
    assert exceptional["condition_matches"] is True

    factors = {item["factor"]: item for item in result["prime_factor_conditions"]}
    assert factors[5]["solutions"] == []
    assert factors[73]["solutions"] == [63]
    assert result["proof_certificate"]["status"] == "passed"
    assert all(
        obligation["passed"]
        for obligation in result["proof_certificate"]["obligations"]
    )
    assert [item["classification"] for item in result["provenance"]] == [
        "exact_identity",
        "derived_theorem",
        "exhaustive_finite_verification",
    ]


def test_affine_analysis_api_is_authenticated_and_returns_scoped_certificate(monkeypatch):
    import orbita_mvp.api as api_module

    monkeypatch.setenv("APP_ENV", "staging")
    monkeypatch.setattr(api_module, "_DEMO_USER", "affine-test-user")
    monkeypatch.setattr(api_module, "_DEMO_PASS", "affine-test-password")
    request = {
        "base_operator": {
            "modulus": 260,
            "multiplier": 79,
            "translation": 117,
            "label": "base analog",
        },
        "extension_modulus": 18_980,
        "reference_operator": {
            "modulus": 18_980,
            "multiplier": -1,
            "translation": -1,
            "label": "occult lift",
        },
        "multiplier_constraints": [{"residue": -1, "modulus": 365}],
    }

    with TestClient(api_module.app) as client:
        assert client.post("/analysis/finite-affine/lifts", json=request).status_code == 401
        assert client.post(
            "/analysis/finite-affine/lifts",
            json=request,
            auth=("affine-test-user", "wrong-password"),
        ).status_code == 401
        response = client.post(
            "/analysis/finite-affine/lifts",
            json=request,
            auth=("affine-test-user", "affine-test-password"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["family_count"] == 1
    assert body["member_count"] == 73
    analysis = body["analyses"][0]
    assert analysis["scope"]["parameter_space"] == "Z_73"
    assert analysis["proof_certificate"]["status"] == "passed"
    assert analysis["proof_certificate"]["counterexamples"] == 0


def test_incompatible_or_oversized_requests_fail_closed():
    base = AffineOperator(260, 79, 117)

    try:
        generate_compatible_lift_families(
            base,
            18_980,
            multiplier_constraints=[(0, 365)],
        )
    except ValueError as exc:
        assert "incompatible" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("incompatible congruences must be rejected")

    try:
        AffineOperator(100_001, 1, 0)
    except ValueError as exc:
        assert "modulus must be between" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("oversized modulus must be rejected")
