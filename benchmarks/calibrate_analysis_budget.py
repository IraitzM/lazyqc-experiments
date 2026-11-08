"""Fit the analysis-budget constants used by heuristic ``auto`` planning.

The fit uses seeded random circuits that are unrelated to the QML evaluation
families, so the budget is never tuned on the workloads it is evaluated on.
Dense cost is LazyQC's forced full-state plan plus collection; analysis cost is
the extra time ``auto`` spends building cone candidates beyond the full one.
"""

import argparse
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

os.environ.update(
    QULACS_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)

import numpy as np  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from lazyqc import Gate, Pauli, Program, plan_all  # noqa: E402
from lazyqc.planner import _analyze  # noqa: E402


def random_batch(rng, n, gates, outputs, programs):
    """Return requests on `programs` random circuits sharing no structure by design."""
    requests = []
    for _ in range(programs):
        operations = []
        for _ in range(gates):
            if n > 1 and rng.random() < 0.3:
                a, b = rng.sample(range(n), 2)
                operations.append(Gate("CX", (a, b)))
            else:
                name = rng.choice(("RX", "RY", "RZ", "H"))
                angle = rng.uniform(-3, 3) if name != "H" else None
                operations.append(Gate(name, (rng.randrange(n),), angle))
        program = Program(n, tuple(operations))
        for _ in range(outputs):
            weight = min(n, rng.choice((1, 1, 2)))
            wires = rng.sample(range(n), weight)
            requests.append(
                program.expectation(Pauli(tuple((q, rng.choice("XYZ")) for q in sorted(wires))))
            )
    return tuple(requests)


def median_ns(function, repeats):
    """Return the median wall time of `function` after one warm call."""
    function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/analysis_budget/calibration.json")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = []
    for n in range(2, 15):
        for gates in (8, 32, 128, 320):
            for outputs in (1, n, 3 * n):
                for programs in (1, 3):
                    batch = random_batch(rng, n, gates, outputs, programs)
                    dense = median_ns(
                        lambda: plan_all(batch, strategy="full").collect(), args.repeats
                    )
                    full = median_ns(lambda: _analyze(batch, "full"), args.repeats)
                    auto = median_ns(lambda: _analyze(batch, "auto"), args.repeats)
                    rows.append(
                        {
                            "qubits": n,
                            "gates": gates,
                            "outputs_per_program": outputs,
                            "programs": programs,
                            "request_gates": gates * outputs * programs,
                            "dense_ns": dense,
                            "analysis_ns": max(auto - full, 0.0),
                        }
                    )
    # Relative least squares: each row is weighted by its own magnitude.
    dense_x = np.array(
        [
            [
                r["programs"],
                r["programs"] * r["gates"],
                r["programs"] * r["gates"] * 2 ** r["qubits"],
                r["programs"] * r["outputs_per_program"] * 2 ** r["qubits"],
            ]
            for r in rows
        ],
        dtype=float,
    )
    dense_y = np.array([r["dense_ns"] for r in rows])
    dense_fit, _ = nnls(dense_x / dense_y[:, None], np.ones_like(dense_y))
    analysis_x = np.array([[1.0, r["request_gates"]] for r in rows])
    analysis_y = np.array([max(r["analysis_ns"], 1.0) for r in rows])
    analysis_fit, _ = nnls(analysis_x / analysis_y[:, None], np.ones_like(analysis_y))
    dense_pred, analysis_pred = dense_x @ dense_fit, analysis_x @ analysis_fit
    summary = {
        "dense_ns": [float(v) for v in dense_fit],
        "dense_terms": ["per program", "per gate", "per amplitude-gate", "per amplitude-output"],
        "analysis_ns": [float(v) for v in analysis_fit],
        "analysis_terms": ["per call", "per request gate"],
        "dense_median_abs_relative_error": float(np.median(np.abs(dense_pred / dense_y - 1))),
        "analysis_median_abs_relative_error": float(
            np.median(np.abs(analysis_pred / analysis_y - 1))
        ),
        "rows": len(rows),
    }
    host = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "threads": os.environ["QULACS_NUM_THREADS"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"host": host, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
