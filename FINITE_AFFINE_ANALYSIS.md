# Exact finite affine analysis

Orbita now has a deterministic analysis path for affine operators on finite
cyclic spaces. It is separate from the statistical discovery pipeline: results
are classified as exact identities, derived theorems, and exhaustive finite
verification rather than predictive associations.

## Authenticated API

`POST /analysis/finite-affine/lifts`

```json
{
  "base_operator": {
    "modulus": 260,
    "multiplier": 79,
    "translation": 117,
    "label": "base analog"
  },
  "extension_modulus": 18980,
  "reference_operator": {
    "modulus": 18980,
    "multiplier": -1,
    "translation": -1,
    "label": "occult lift"
  },
  "multiplier_constraints": [
    {"residue": -1, "modulus": 365}
  ]
}
```

The backend's existing fail-closed Basic Auth middleware protects this route.
The request is stateless: it does not read or mutate cases, claims, graphs, or
user data. Moduli and generated family sizes are bounded to limit computation.

## Current exact capabilities

- affine composition, inverse, powers, order, commutator, and fixed points;
- inference of an exact affine map from raw finite input/output pairs;
- group-orbit decomposition from affine generators;
- generalized CRT constraints and compatible lift-family generation;
- generic/exceptional commutator-order classification;
- compact congruence conditions for exceptional parameters;
- proof obligations and exhaustive finite verification receipts;
- provenance labels that distinguish identities from empirical findings.

## Benchmark result

The tests recover 65 four-state Tzolkin orbits, the order-5 commutator
translation, all 73 compatible Calendar Round lifts, and the unique exceptional
parameter `t = 63 (mod 73)`. The exact order formula is
`365/gcd(365, 192 - 10*t)`.

This result is scoped to the supplied finite operator family. It is not a claim
about a physical theory, causality, or behavior outside that family.
