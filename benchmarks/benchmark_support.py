"""Helpers shared by the benchmark runners: provenance, statistics, workloads and oracles.

Kept free of LazyQC imports at module level so runners can pin native threads first.
"""

import argparse
import hashlib
import math
import os
import platform
import random
import sys
from pathlib import Path

from core_checkout import REFERENCE, label, package_files

ROOT = Path(__file__).resolve().parents[1]


def pin_single_thread():
    """Force one native thread before Qulacs, OpenMP or BLAS initialize."""
    from lazyqc.cost import THREAD_KEYS

    for key in THREAD_KEYS:
        os.environ[key] = "1"


def positive_int(text):
    """Parse a positive integer command-line argument."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return value


# --- provenance -------------------------------------------------------------


def cpu_name():
    """Return the CPU model name, falling back to the platform processor string."""
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def source_hash(runner):
    """Hash the package sources, the runner, this module and the lockfile."""
    h = hashlib.sha256()
    paths = package_files() + [
        Path(runner).resolve(),
        Path(__file__).resolve(),
        ROOT / "uv.lock",
    ]
    for path in paths:
        h.update(label(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


# --- statistics -------------------------------------------------------------


def percentile(values, fraction):
    """Linearly interpolated percentile of `values` at `fraction` in [0, 1]."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def geomean(values):
    """Geometric mean of positive values."""
    return math.exp(sum(map(math.log, values)) / len(values))


def bootstrap_interval(ratios, seed=9701):
    """Seeded 2000-sample bootstrap 95% interval of the geometric mean."""
    rng = random.Random(seed)
    logs = [math.log(r) for r in ratios]
    samples = sorted(math.exp(sum(rng.choices(logs, k=len(logs))) / len(logs)) for _ in range(2000))
    return [samples[49], samples[1949]]


# --- workloads --------------------------------------------------------------


def pair_case(m, prefix="independent_pairs"):
    """Fixture case of m RY/CX pairs with four closed-form observables per pair."""
    if not 1 <= m <= 12:
        raise ValueError("Pair sweep supports 1 through 12 pairs")
    operations, requests = [], []
    for j in range(m):
        a, b = 2 * j, 2 * j + 1
        theta = (j + 1) * math.pi / (2 * m + 1)
        operations.extend(
            [
                {"gate": "RY", "qubits": [a], "angle_radians": theta},
                {"gate": "CX", "qubits": [a, b]},
            ]
        )
        for suffix, obs, expected in [
            ("za", [(a, "Z")], math.cos(theta)),
            ("zb", [(b, "Z")], math.cos(theta)),
            ("zz", [(a, "Z"), (b, "Z")], 1.0),
            ("xx", [(a, "X"), (b, "X")], math.sin(theta)),
        ]:
            requests.append(
                {
                    "id": f"pair{j}_{suffix}",
                    "expected": expected,
                    "observable": [{"qubit": q, "pauli": p} for q, p in obs],
                }
            )
    return {
        "name": f"{prefix}_{2 * m}q",
        "qubits": 2 * m,
        "initial_state": "zero",
        "operations": operations,
        "requests": requests,
    }


def build_ansatz(n, layers, rng):
    """Hardware-efficient ansatz: RY layers plus a linear CX chain, as plain tuples."""
    gates = []  # ("ry", q, theta) | ("cx", a, b)
    for _ in range(layers):
        for q in range(n):
            gates.append(("ry", q, rng.uniform(-math.pi, math.pi)))
        for a in range(n - 1):
            gates.append(("cx", a, a + 1))
    return gates


def overlapping_terms(n, count, rng):
    """Draw `count` distinct local Pauli terms with radius 0..2 around random centres."""
    terms, seen = [], set()
    while len(terms) < count:
        center = rng.randrange(n)
        radius = rng.choice((0, 0, 1, 1, 2))
        support = sorted(q for q in range(max(0, center - radius), min(n, center + radius + 1)))
        obs = {q: rng.choice("XYZ") for q in support}
        key = tuple(sorted(obs.items()))
        if key in seen:
            continue
        seen.add(key)
        terms.append(obs)
    return terms


# --- independent oracles ----------------------------------------------------


def dense_expectations(requests):
    """Dense NumPy reference values (benchmarks/reference/reference_numpy), one state per program."""
    sys.path.insert(0, str(REFERENCE))
    from reference_numpy import expectation, statevector

    states = {}
    for r in requests:
        p = r.program
        if p not in states:
            states[p] = statevector(
                p.n_qubits, [(g.name, g.qubits, g.angle) for g in p.resolved_operations()]
            )
    return tuple(expectation(states[r.program], r.observable.factors) for r in requests)


def _apply_ry(state, n, q, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    step = 1 << q
    for base in range(0, 1 << n, 2 * step):
        for i in range(base, base + step):
            a, b = state[i], state[i + step]
            state[i] = c * a - s * b
            state[i + step] = s * a + c * b


def _apply_cx(state, n, a, b):
    for i in range(1 << n):
        if (i >> a) & 1 and not (i >> b) & 1:
            j = i | (1 << b)
            state[i], state[j] = state[j], state[i]


def ansatz_expectations(n, gates, terms):
    """Loop-based statevector oracle for `build_ansatz` gates; no Qulacs, no matrices."""
    import numpy as np

    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    for g in gates:
        if g[0] == "ry":
            _apply_ry(state, n, g[1], g[2])
        else:
            _apply_cx(state, n, g[1], g[2])
    values = []
    for obs in terms:
        v = state.copy()
        for q, p in obs.items():
            step = 1 << q
            if p == "X":
                for base in range(0, 1 << n, 2 * step):
                    for i in range(base, base + step):
                        v[i], v[i + step] = v[i + step], v[i]
            elif p == "Z":
                for base in range(0, 1 << n, 2 * step):
                    for i in range(base + step, base + 2 * step):
                        v[i] = -v[i]
            else:  # Y
                for base in range(0, 1 << n, 2 * step):
                    for i in range(base, base + step):
                        a, b = v[i], v[i + step]
                        v[i] = -1j * b
                        v[i + step] = 1j * a
        values.append(float(np.vdot(state, v).real))
    return values
