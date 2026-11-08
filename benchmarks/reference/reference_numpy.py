"""Tiny dense reference, independent of native lowering and cone analysis."""

import numpy as np

MATRICES = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
}


def matrix_on_wire(matrix, wire, n):
    result = np.ones((1, 1), dtype=complex)
    for q in reversed(range(n)):
        result = np.kron(result, matrix if q == wire else MATRICES["I"])
    return result


def statevector(n, operations):
    if n > 8:
        raise ValueError("Dense reference is restricted to at most eight qubits")
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1
    for name, wires, angle in operations:
        if name == "CX":
            matrix = np.zeros((2**n, 2**n), dtype=complex)
            control, target = wires
            for i in range(2**n):
                j = i ^ (1 << target) if i & (1 << control) else i
                matrix[j, i] = 1
        elif name in ("CZ", "RZZ"):
            # Diagonal gates, written from their definitions: CZ = diag(1, 1, 1, -1) and
            # RZZ(theta) = exp(-i theta Z Z / 2), whose phase depends on the parity of the pair.
            a, b = wires
            diagonal = np.ones(2**n, dtype=complex)
            for i in range(2**n):
                bits = (i >> a) & 1, (i >> b) & 1
                if name == "CZ":
                    diagonal[i] = -1 if bits == (1, 1) else 1
                else:
                    parity = -1 if bits[0] != bits[1] else 1  # eigenvalue of Z (x) Z
                    diagonal[i] = np.exp(-0.5j * angle * parity)
            matrix = np.diag(diagonal)
        else:
            local = (
                (np.cos(angle / 2) * MATRICES["I"] - 1j * np.sin(angle / 2) * MATRICES[name[1]])
                if name.startswith("R")
                else MATRICES[name]
            )
            matrix = matrix_on_wire(local, wires[0], n)
        state = matrix @ state
    return state


def expectation(state, factors):
    n = int(np.log2(len(state)))
    factors = dict(factors)
    operator = np.ones((1, 1), dtype=complex)
    for q in reversed(range(n)):
        operator = np.kron(operator, MATRICES[factors.get(q, "I")])
    value = np.vdot(state, operator @ state)
    assert abs(value.imag) < 1e-12
    return float(value.real)
