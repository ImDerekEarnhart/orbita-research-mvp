"""Exact analysis for affine operators on finite cyclic spaces.

This module is deliberately separate from Orbita's statistical discovery path.
Every conclusion is derived with integer modular arithmetic and is scoped to the
finite operator family that was exhaustively checked.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import gcd
from typing import Iterable, Sequence


MAX_AFFINE_MODULUS = 100_000
MAX_AFFINE_FAMILY_MEMBERS = 10_000


def _require_modulus(modulus: int) -> int:
    modulus = int(modulus)
    if not 2 <= modulus <= MAX_AFFINE_MODULUS:
        raise ValueError(
            f"modulus must be between 2 and {MAX_AFFINE_MODULUS}"
        )
    return modulus


def _signed_residue(value: int, modulus: int) -> int:
    value %= modulus
    return value - modulus if value > modulus // 2 else value


def _solve_linear_congruence(
    coefficient: int,
    target: int,
    modulus: int,
) -> list[int]:
    """Return every x modulo ``modulus`` satisfying coefficient*x = target."""
    modulus = _require_modulus(modulus)
    coefficient %= modulus
    target %= modulus
    divisor = gcd(coefficient, modulus)
    if target % divisor:
        return []
    reduced_modulus = modulus // divisor
    if reduced_modulus == 1:
        root = 0
    else:
        root = (
            (target // divisor)
            * pow(coefficient // divisor, -1, reduced_modulus)
        ) % reduced_modulus
    return sorted(root + offset * reduced_modulus for offset in range(divisor))


def _combine_congruences(
    congruences: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Combine compatible, possibly non-coprime congruences with generalized CRT."""
    if not congruences:
        raise ValueError("at least one congruence is required")
    residue, period = congruences[0]
    period = _require_modulus(period)
    residue %= period
    for next_residue, next_period in congruences[1:]:
        next_period = _require_modulus(next_period)
        next_residue %= next_period
        divisor = gcd(period, next_period)
        difference = next_residue - residue
        if difference % divisor:
            raise ValueError("multiplier congruences are incompatible")
        reduced_next = next_period // divisor
        if reduced_next == 1:
            step = 0
        else:
            step = (
                (difference // divisor)
                * pow(period // divisor, -1, reduced_next)
            ) % reduced_next
        combined_period = period * reduced_next
        residue = (residue + period * step) % combined_period
        period = combined_period
    return residue, period


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1 if divisor == 2 else 2
    if value > 1:
        factors.append(value)
    return factors


@dataclass(frozen=True)
class AffineOperator:
    """The bijection candidate x -> multiplier*x + translation (mod modulus)."""

    modulus: int
    multiplier: int
    translation: int
    label: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        modulus = _require_modulus(self.modulus)
        object.__setattr__(self, "modulus", modulus)
        object.__setattr__(self, "multiplier", int(self.multiplier) % modulus)
        object.__setattr__(self, "translation", int(self.translation) % modulus)

    @classmethod
    def identity(cls, modulus: int) -> "AffineOperator":
        return cls(modulus, 1, 0, "identity")

    @property
    def is_bijection(self) -> bool:
        return gcd(self.multiplier, self.modulus) == 1

    @property
    def is_translation(self) -> bool:
        return self.multiplier == 1

    def apply(self, value: int) -> int:
        return (self.multiplier * int(value) + self.translation) % self.modulus

    def compose(self, other: "AffineOperator") -> "AffineOperator":
        """Return self after other, i.e. ``self(other(x))``."""
        if self.modulus != other.modulus:
            raise ValueError("operators must use the same modulus")
        return AffineOperator(
            self.modulus,
            self.multiplier * other.multiplier,
            self.multiplier * other.translation + self.translation,
        )

    def inverse(self) -> "AffineOperator":
        if not self.is_bijection:
            raise ValueError("operator is not invertible")
        inverse_multiplier = pow(self.multiplier, -1, self.modulus)
        return AffineOperator(
            self.modulus,
            inverse_multiplier,
            -inverse_multiplier * self.translation,
        )

    def power(self, exponent: int) -> "AffineOperator":
        exponent = int(exponent)
        if exponent < 0:
            return self.inverse().power(-exponent)
        result = AffineOperator.identity(self.modulus)
        factor = self
        while exponent:
            if exponent & 1:
                result = result.compose(factor)
            factor = factor.compose(factor)
            exponent >>= 1
        return result

    def order(self) -> int:
        """Return the exact group order of this affine bijection."""
        if not self.is_bijection:
            raise ValueError("operator order requires an invertible multiplier")
        multiplier_power = 1
        multiplier_order = 0
        for exponent in range(1, self.modulus + 1):
            multiplier_power = (multiplier_power * self.multiplier) % self.modulus
            if multiplier_power == 1:
                multiplier_order = exponent
                break
        if not multiplier_order:  # pragma: no cover - Euler's theorem guarantees it
            raise RuntimeError("could not determine multiplier order")
        translation = self.power(multiplier_order)
        translation_order = self.modulus // gcd(
            self.modulus,
            translation.translation,
        )
        return multiplier_order * translation_order

    def commutator(self, other: "AffineOperator") -> "AffineOperator":
        """Return self*other*self^-1*other^-1 under function composition."""
        return (
            self.compose(other)
            .compose(self.inverse())
            .compose(other.inverse())
        )

    def fixed_points(self) -> list[int]:
        return _solve_linear_congruence(
            self.multiplier - 1,
            -self.translation,
            self.modulus,
        )

    def to_dict(self) -> dict[str, int | str | bool]:
        return {
            "modulus": self.modulus,
            "multiplier": self.multiplier,
            "translation": self.translation,
            "label": self.label,
            "is_bijection": self.is_bijection,
        }


@dataclass(frozen=True)
class ParameterizedOperator:
    parameter: int
    operator: AffineOperator


@dataclass(frozen=True)
class AffineLiftFamily:
    base_operator: AffineOperator
    extension_modulus: int
    lifted_multiplier: int
    multiplier_constraints: tuple[tuple[int, int], ...]
    members: tuple[ParameterizedOperator, ...]


def infer_affine_operators(
    observations: Iterable[tuple[int, int]],
    modulus: int,
    *,
    require_bijection: bool = True,
) -> list[AffineOperator]:
    """Infer every exact affine map consistent with the observed input/output pairs."""
    modulus = _require_modulus(modulus)
    pairs = [(int(x) % modulus, int(y) % modulus) for x, y in observations]
    if len(pairs) < 2:
        raise ValueError("at least two observations are required")
    x0, y0 = pairs[0]
    matches: list[AffineOperator] = []
    for multiplier in range(modulus):
        if require_bijection and gcd(multiplier, modulus) != 1:
            continue
        translation = (y0 - multiplier * x0) % modulus
        if all(
            (multiplier * x + translation) % modulus == y
            for x, y in pairs
        ):
            matches.append(AffineOperator(modulus, multiplier, translation))
    return matches


def group_orbits(generators: Sequence[AffineOperator]) -> list[tuple[int, ...]]:
    """Partition Z_n into orbits under the group generated by ``generators``."""
    if not generators:
        raise ValueError("at least one generator is required")
    modulus = generators[0].modulus
    if any(operator.modulus != modulus for operator in generators):
        raise ValueError("all generators must use the same modulus")
    if any(not operator.is_bijection for operator in generators):
        raise ValueError("group generators must be invertible")
    actions = list(generators) + [operator.inverse() for operator in generators]
    visited: set[int] = set()
    orbits: list[tuple[int, ...]] = []
    for start in range(modulus):
        if start in visited:
            continue
        orbit = {start}
        frontier = [start]
        while frontier:
            current = frontier.pop()
            for action in actions:
                image = action.apply(current)
                if image not in orbit:
                    orbit.add(image)
                    frontier.append(image)
        visited.update(orbit)
        orbits.append(tuple(sorted(orbit)))
    return sorted(orbits, key=lambda values: values[0])


def generate_compatible_lift_families(
    base_operator: AffineOperator,
    extension_modulus: int,
    *,
    multiplier_constraints: Sequence[tuple[int, int]] = (),
) -> list[AffineLiftFamily]:
    """Generate every affine lift, grouped by compatible lifted multiplier."""
    extension_modulus = _require_modulus(extension_modulus)
    if extension_modulus % base_operator.modulus:
        raise ValueError("extension modulus must be a multiple of the base modulus")
    constraints = [(base_operator.multiplier, base_operator.modulus)]
    for residue, modulus in multiplier_constraints:
        modulus = _require_modulus(modulus)
        if extension_modulus % modulus:
            raise ValueError("each multiplier constraint modulus must divide the extension")
        constraints.append((int(residue), modulus))
    residue, period = _combine_congruences(constraints)
    if extension_modulus % period:
        raise ValueError("combined multiplier period must divide the extension modulus")
    multipliers = [
        residue + branch * period
        for branch in range(extension_modulus // period)
        if gcd(residue + branch * period, extension_modulus) == 1
    ]
    if not multipliers:
        raise ValueError("no invertible multiplier satisfies the lift constraints")

    translation_count = extension_modulus // base_operator.modulus
    total_members = len(multipliers) * translation_count
    if total_members > MAX_AFFINE_FAMILY_MEMBERS:
        raise ValueError(
            f"lift family exceeds the {MAX_AFFINE_FAMILY_MEMBERS}-member safety limit"
        )
    frozen_constraints = tuple((int(r), int(m)) for r, m in multiplier_constraints)
    families: list[AffineLiftFamily] = []
    for multiplier in multipliers:
        members = tuple(
            ParameterizedOperator(
                parameter,
                AffineOperator(
                    extension_modulus,
                    multiplier,
                    base_operator.translation + base_operator.modulus * parameter,
                ),
            )
            for parameter in range(translation_count)
        )
        families.append(
            AffineLiftFamily(
                base_operator=base_operator,
                extension_modulus=extension_modulus,
                lifted_multiplier=multiplier,
                multiplier_constraints=frozen_constraints,
                members=members,
            )
        )
    return families


def analyze_commutator_family(
    reference_operator: AffineOperator,
    family: AffineLiftFamily,
) -> dict:
    """Derive and exhaustively verify a commutator-order classification."""
    if reference_operator.modulus != family.extension_modulus:
        raise ValueError("reference operator must use the extension modulus")
    parameters = [member.parameter for member in family.members]
    if parameters != list(range(len(parameters))):
        raise ValueError("family parameters must be the consecutive range starting at zero")
    if not family.members:
        raise ValueError("family must contain at least one member")

    commutators = [
        reference_operator.commutator(member.operator)
        for member in family.members
    ]
    all_translations = all(operator.is_translation for operator in commutators)
    if not all_translations:
        raise ValueError("commutator-family analysis currently requires translations")

    modulus = family.extension_modulus
    steps = [operator.translation for operator in commutators]
    intercept = steps[0]
    slope_mod = (steps[1] - steps[0]) % modulus if len(steps) > 1 else 0
    slope = _signed_residue(slope_mod, modulus)
    formula_matches = all(
        step == (intercept + slope * parameter) % modulus
        for parameter, step in zip(parameters, steps)
    )

    orders = [operator.order() for operator in commutators]
    formula_orders = [modulus // gcd(modulus, step) for step in steps]
    order_formula_matches = orders == formula_orders
    counts = Counter(orders)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ValueError("family has no unique generic order regime")
    generic_order, generic_count = ranked[0]

    common_factor = gcd(modulus, intercept, abs(slope))
    reduced_modulus = modulus // common_factor
    reduced_intercept = intercept // common_factor
    reduced_slope = slope // common_factor
    reduced_orders = [
        reduced_modulus
        // gcd(reduced_modulus, reduced_intercept + reduced_slope * parameter)
        for parameter in parameters
    ]
    reduced_formula_matches = reduced_orders == orders

    constraints_pass = all(
        member.operator.multiplier % family.base_operator.modulus
        == family.base_operator.multiplier
        and member.operator.translation % family.base_operator.modulus
        == family.base_operator.translation
        and all(
            member.operator.multiplier % constraint_modulus
            == constraint_residue % constraint_modulus
            for constraint_residue, constraint_modulus in family.multiplier_constraints
        )
        for member in family.members
    )

    prime_conditions = []
    for factor in _prime_factors(reduced_modulus):
        solutions = _solve_linear_congruence(
            reduced_slope,
            -reduced_intercept,
            factor,
        )
        prime_conditions.append(
            {
                "factor": factor,
                "solutions": solutions,
                "condition": (
                    "no parameter satisfies the congruence"
                    if not solutions
                    else " or ".join(f"t = {value} (mod {factor})" for value in solutions)
                ),
            }
        )

    grouped_parameters: dict[int, list[int]] = defaultdict(list)
    for parameter, order in zip(parameters, orders):
        grouped_parameters[order].append(parameter)
    regimes = [
        {
            "kind": "generic",
            "commutator_order": generic_order,
            "count": generic_count,
        }
    ]
    generic_gcd = reduced_modulus // generic_order
    for order in sorted(grouped_parameters):
        if order == generic_order:
            continue
        exceptional_parameters = grouped_parameters[order]
        exceptional_gcd = reduced_modulus // order
        new_factor = exceptional_gcd // gcd(exceptional_gcd, generic_gcd)
        solutions = (
            _solve_linear_congruence(
                reduced_slope,
                -reduced_intercept,
                new_factor,
            )
            if new_factor > 1
            else []
        )
        condition = (
            " or ".join(f"t = {value} (mod {new_factor})" for value in solutions)
            if solutions
            else "enumerated exceptional set"
        )
        parameters_matching_condition = {
            parameter
            for parameter in parameters
            if new_factor > 1
            and (reduced_intercept + reduced_slope * parameter) % new_factor == 0
        }
        condition_matches = (
            bool(solutions)
            and parameters_matching_condition == set(exceptional_parameters)
        )
        regimes.append(
            {
                "kind": "exceptional",
                "commutator_order": order,
                "count": len(exceptional_parameters),
                "parameters": exceptional_parameters,
                "new_gcd_factor": new_factor,
                "condition": condition,
                "condition_matches": condition_matches,
            }
        )

    exceptional_conditions_match = all(
        regime.get("condition_matches", True)
        for regime in regimes
    )

    obligations = [
        {
            "name": "compatible_lift_constraints",
            "passed": constraints_pass,
            "checked": len(family.members),
        },
        {
            "name": "commutators_are_translations",
            "passed": all_translations,
            "checked": len(commutators),
        },
        {
            "name": "affine_commutator_formula",
            "passed": formula_matches,
            "checked": len(steps),
        },
        {
            "name": "translation_order_gcd_formula",
            "passed": order_formula_matches and reduced_formula_matches,
            "checked": len(orders),
        },
        {
            "name": "exceptional_congruence_is_exact",
            "passed": exceptional_conditions_match,
            "checked": sum(
                regime["count"]
                for regime in regimes
                if regime["kind"] == "exceptional"
            ),
        },
        {
            "name": "finite_family_exhausted",
            "passed": len(parameters) == modulus // family.base_operator.modulus,
            "checked": len(parameters),
        },
    ]
    certificate_passed = all(item["passed"] for item in obligations)

    return {
        "analysis_kind": "finite_affine_commutator_family",
        "result_type": "exact_finite_classification" if certificate_passed else "incomplete",
        "epistemic_status": (
            "verified_within_finite_scope" if certificate_passed else "review_required"
        ),
        "scope": {
            "base_modulus": family.base_operator.modulus,
            "extension_modulus": modulus,
            "parameter_space": f"Z_{len(parameters)}",
            "family_size": len(parameters),
            "lifted_multiplier": family.lifted_multiplier,
        },
        "reference_operator": reference_operator.to_dict(),
        "commutator_formula": {
            "multiplier": 1,
            "translation_intercept": intercept,
            "translation_slope": slope,
            "modulus": modulus,
            "rendered": f"T_({intercept} {slope:+d}*t) mod {modulus}",
        },
        "order_formula": {
            "numerator": reduced_modulus,
            "gcd_modulus": reduced_modulus,
            "intercept": reduced_intercept,
            "slope": reduced_slope,
            "common_factor_removed": common_factor,
            "rendered": (
                f"{reduced_modulus}/gcd({reduced_modulus}, "
                f"{reduced_intercept} {reduced_slope:+d}*t)"
            ),
        },
        "prime_factor_conditions": prime_conditions,
        "regimes": regimes,
        "orders_by_parameter": [
            {
                "parameter": parameter,
                "commutator_step": step,
                "commutator_order": order,
            }
            for parameter, step, order in zip(parameters, steps, orders)
        ],
        "proof_certificate": {
            "status": "passed" if certificate_passed else "failed",
            "obligations": obligations,
            "counterexamples": 0 if certificate_passed else None,
        },
        "provenance": [
            {
                "relationship": "affine composition and commutator formula",
                "classification": "exact_identity",
                "basis": "integer modular arithmetic",
            },
            {
                "relationship": "translation order formula",
                "classification": "derived_theorem",
                "basis": "order(T_s) = n/gcd(n, s)",
            },
            {
                "relationship": "generic and exceptional regimes",
                "classification": "exhaustive_finite_verification",
                "checked": len(parameters),
            },
        ],
    }


def analyze_compatible_lift_families(
    *,
    base_operator: AffineOperator,
    extension_modulus: int,
    reference_operator: AffineOperator,
    multiplier_constraints: Sequence[tuple[int, int]] = (),
) -> dict:
    families = generate_compatible_lift_families(
        base_operator,
        extension_modulus,
        multiplier_constraints=multiplier_constraints,
    )
    return {
        "analysis_kind": "finite_affine_compatible_lifts",
        "family_count": len(families),
        "member_count": sum(len(family.members) for family in families),
        "analyses": [
            analyze_commutator_family(reference_operator, family)
            for family in families
        ],
    }
