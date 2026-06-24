"""Built-in domain registry and run factory."""
from __future__ import annotations

from typing import Any

from .domains.association_screen import AssociationScreenDomain
from .domains.numeric_law import NumericLawDomain
from .domains.tabular_predicate import TabularPredicateDomain

DOMAIN_INFO = {
    "numeric_law": "Fit and falsify candidate numeric functional forms.",
    "association_screen": "Separate predictive features from chance correlations.",
    "tabular_predicate": "Test declarative Boolean hypotheses over JSON records.",
    "quantum_circuit_audit": "Audit random-phase/QFT circuit claims with exact statevectors.",
}


def create_domain(name: str, config: dict[str, Any] | None = None):
    config = dict(config or {})
    if name == "numeric_law":
        return NumericLawDomain(**config)
    if name == "association_screen":
        return AssociationScreenDomain(**config)
    if name == "tabular_predicate":
        return TabularPredicateDomain(**config)
    if name == "quantum_circuit_audit":
        try:
            from .domains.quantum_circuit_audit import QuantumCircuitAuditDomain
        except ImportError as exc:
            raise RuntimeError("Install the quantum extra: pip install 'orbita-discovery-kit[quantum]'") from exc
        return QuantumCircuitAuditDomain(**config)
    raise ValueError(f"Unknown domain: {name}")
