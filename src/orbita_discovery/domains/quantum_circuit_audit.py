"""Orbita domain for auditing the QFT/random-phase circuits in Circuitcolab.

The domain generates exact statevector data with NumPy.  It does not require
Qiskit and does not treat simulator output as evidence of quantum advantage.
"""
from __future__ import annotations

import math
import random
from typing import Any, Callable

import numpy as np

from ..core import Candidate

H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)


def _apply_1q(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    shaped = state.reshape([2] * n)
    # q=0 is least-significant qubit; tensor axis is reversed.
    axis = n - 1 - q
    moved = np.moveaxis(shaped, axis, 0).reshape(2, -1)
    moved = gate @ moved
    return np.moveaxis(moved.reshape([2] + [2] * (n - 1)), 0, axis).reshape(-1)


def _apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    out = np.zeros_like(state)
    for i, amp in enumerate(state):
        j = i ^ (1 << target) if ((i >> control) & 1) else i
        out[j] += amp
    return out


def _apply_cp(state: np.ndarray, angle: float, control: int, target: int) -> np.ndarray:
    out = state.copy()
    phase = np.exp(1j * angle)
    for i in range(len(out)):
        if ((i >> control) & 1) and ((i >> target) & 1):
            out[i] *= phase
    return out


def _apply_rz(state: np.ndarray, angle: float, q: int, n: int) -> np.ndarray:
    gate = np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])
    return _apply_1q(state, gate, q, n)


def _qft(state: np.ndarray) -> np.ndarray:
    """Mathematical QFT, normalized, using the statevector index basis."""
    N = len(state)
    j = np.arange(N)
    mat = np.exp(2j * np.pi * np.outer(j, j) / N) / math.sqrt(N)
    return mat @ state


def _initial_plus_with_phases(n: int, angles: list[float]) -> np.ndarray:
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for q in range(n):
        state = _apply_1q(state, H, q, n)
    for q, a in enumerate(angles):
        state = _apply_rz(state, a, q, n)
    return state


def _probs(state: np.ndarray) -> np.ndarray:
    p = np.abs(state) ** 2
    p /= p.sum()
    return p


def _entropy(p: np.ndarray) -> float:
    nz = p[p > 1e-15]
    return float(-(nz * np.log2(nz)).sum())


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def _single_qubit_entropy(state: np.ndarray, q: int, n: int) -> float:
    shaped = state.reshape([2] * n)
    axis = n - 1 - q
    psi = np.moveaxis(shaped, axis, 0).reshape(2, -1)
    rho = psi @ psi.conj().T
    vals = np.linalg.eigvalsh(rho).real
    vals = vals[vals > 1e-14]
    return float(-(vals * np.log2(vals)).sum())


def _mean_1q_entropy(state: np.ndarray, n: int) -> float:
    return sum(_single_qubit_entropy(state, q, n) for q in range(n)) / n


def _make_case(n: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    angles = [rng.uniform(0, 2 * math.pi) for _ in range(n)]

    plus_phase = _initial_plus_with_phases(n, angles)
    uniform = np.full(2**n, 1 / (2**n))

    # Document's diagonal-only "QFT-like" variant: after H, only phase gates
    # and swaps occur. Swaps only permute equal probabilities.
    diagonal = plus_phase.copy()
    for q in range(1, n):
        diagonal = _apply_1q(diagonal, np.diag([1, np.exp(1j * math.pi / (2**q))]), q, n)
    p_diag = _probs(diagonal)

    # Complete mathematical QFT after random local phases.
    p_qft = _probs(_qft(plus_phase))

    # The early notebook calls this an "entanglement" chain: H on all qubits,
    # then CNOT(0,1), CNOT(1,2), ... before the RZ phases.
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for q in range(n):
        state = _apply_1q(state, H, q, n)
    after_h = state.copy()
    for q in range(n - 1):
        state = _apply_cnot(state, q, q + 1, n)
    after_cnot = state.copy()
    cnot_ent = _mean_1q_entropy(after_cnot, n)
    for q, a in enumerate(angles):
        state = _apply_rz(state, a, q, n)
    p_cnot_qft = _probs(_qft(state))

    # A genuinely entangling CP control for contrast.
    cp_state = after_h.copy()
    if n >= 2:
        cp_state = _apply_cp(cp_state, math.pi / 3, 0, 1)
    cp_ent = _mean_1q_entropy(cp_state, n)

    return {
        "n": n,
        "seed": seed,
        "diag_tv_uniform": _tv(p_diag, uniform),
        "diag_entropy": _entropy(p_diag),
        "qft_tv_uniform": _tv(p_qft, uniform),
        "qft_entropy": _entropy(p_qft),
        "qft_max_prob": float(p_qft.max()),
        "cnot_qft_tv_vs_qft": _tv(p_cnot_qft, p_qft),
        "cnot_entanglement": cnot_ent,
        "cp_entanglement": cp_ent,
        "qft_argmax": int(np.argmax(p_qft)),
        "qft_probs": p_qft.tolist(),
    }


PREDICATES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "diagonal_uniform": lambda r: r["diag_tv_uniform"] < 1e-12,
    "cnot_entangles_plus": lambda r: r["cnot_entanglement"] > 1e-6,
    "cnot_changes_qft": lambda r: r["cnot_qft_tv_vs_qft"] > 1e-8,
    "full_qft_nonuniform": lambda r: r["qft_tv_uniform"] > 0.02,
    "full_qft_lowers_entropy": lambda r: r["qft_entropy"] < r["n"] - 0.02,
    "cp_can_entangle": lambda r: r["cp_entanglement"] > 1e-4,
}


class QuantumCircuitAuditDomain:
    name = "quantum_circuit_audit"

    def __init__(self, seeds_per_n: int = 32):
        self.rows = [
            _make_case(n, seed=1000 * n + s)
            for n in range(3, 9)
            for s in range(seeds_per_n)
        ]

    def propose(self):
        specs = [
            (
                "qc:diagonal_uniform",
                "After H on every qubit, phase-only gates and swaps leave computational-basis probabilities uniform",
                "diagonal_uniform",
            ),
            (
                "qc:cnot_entangles_plus",
                "A nearest-neighbor CNOT chain applied immediately after H on every qubit creates entanglement",
                "cnot_entangles_plus",
            ),
            (
                "qc:cnot_changes_qft",
                "That CNOT chain changes the later random-phase-plus-QFT output distribution",
                "cnot_changes_qft",
            ),
            (
                "qc:full_qft_nonuniform",
                "A complete QFT after independent random RZ phases usually produces a non-uniform output distribution",
                "full_qft_nonuniform",
            ),
            (
                "qc:full_qft_lowers_entropy",
                "A complete QFT after independent random RZ phases lowers measurement entropy below the uniform maximum",
                "full_qft_lowers_entropy",
            ),
            (
                "qc:cp_can_entangle",
                "A nontrivial controlled-phase gate on a plus-state input can create entanglement",
                "cp_can_entangle",
            ),
        ]
        for cid, statement, key in specs:
            yield Candidate(cid, statement, payload={"predicate": key})

    def evidence_for(self, c: Candidate) -> Any:
        return self.rows

    def splits(self, evidence, seed: int):
        idx = list(range(len(evidence)))
        random.Random(seed).shuffle(idx)
        cut = int(0.7 * len(idx))
        return [evidence[i] for i in idx[:cut]], [evidence[i] for i in idx[cut:]]

    def refit(self, c: Candidate, train) -> Any:
        return c.payload["predicate"]

    def score(self, c: Candidate, model, test) -> float:
        pred = PREDICATES[model]
        return sum(bool(pred(r)) for r in test) / max(1, len(test))

    def baseline_score(self, test) -> float:
        return 0.5
