"""Fit the per-task constant of the work score (issue #2).

Every candidate plan is timed on seeded random circuits unrelated to the evaluation
families (calibrate_analysis_budget.random_batch). The collection time is regressed on
the current work score W and the task count T by relative non-negative least squares,

    time = a * W + b * T + c,        c0 = b / a  (work units per task),

and, because lowering is paid per gate as well as per task, against the gate count G too:

    time = a * W + g * G + c,        c1 = g / a  (work units per gate).

Ranking only needs the ratio, since W' = W + c0 * T (or W + c1 * G) for every candidate. The evaluation manifest
is hashed into the output, so the fit provably follows its freezing.

    uv run --locked python benchmarks/calibrate_task_cost.py
"""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

os.environ.update(
    QULACS_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import numpy as np  # noqa: E402
from calibrate_analysis_budget import random_batch  # noqa: E402
from scipy.optimize import nnls  # noqa: E402

from lazyqc import plan_all  # noqa: E402

MANIFEST = HERE / "task_cost_eval_manifest_v2.json"


def raw_work(tasks):
    """Work score without any fixed cost, so the fit does not depend on the library constants."""
    return sum((1 << t.width) * (1 + len(t.operations) + len(t.outputs)) for t in tasks)


def collect_ns(plan, repeats):
    """Median execution time of `collect` after one warm call."""
    plan.collect()
    return statistics.median(plan.collect().stats.execution_ns for _ in range(repeats))


def fit(rows, columns):
    """Relative NNLS: minimise sum ((x.beta - y) / y)^2 with beta >= 0."""
    x = np.array([[r[c] for c in columns] for r in rows], dtype=float)
    y = np.array([r["time_ns"] for r in rows], dtype=float)
    beta, _ = nnls(x / y[:, None], np.ones_like(y))
    return beta, x @ beta, y


def ranking(batches, c0, c1):
    """Share of batches whose argmin of W + c0 T + c1 G is the fastest candidate, and mean ratio."""
    hits, ratios = 0, []
    for rows in batches:
        pick = min(rows, key=lambda r: (r["W"] + c0 * r["T"] + c1 * r["G"], r["T"]))
        best = min(r["time_ns"] for r in rows)
        hits += pick["time_ns"] == best
        ratios.append(pick["time_ns"] / best)
    return hits / len(batches), float(np.exp(np.mean(np.log(ratios)))), float(np.max(ratios))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/task_cost/calibration.json")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    rng = random.Random(manifest["calibration"]["seed"])
    batches = []
    for n in range(2, 15):
        for gates in (8, 32, 96, 200):
            for outputs in (1, n, 3 * n):
                for programs in (1, 3):
                    batch = random_batch(rng, n, gates, outputs, programs)
                    rows = []
                    for strategy in ("full", "separate", "grouped"):
                        plan = plan_all(batch, strategy=strategy, analysis="always")
                        rows.append(
                            {
                                "qubits": n,
                                "gates": gates,
                                "outputs": outputs,
                                "programs": programs,
                                "strategy": strategy,
                                "W": raw_work(plan.selected.tasks),
                                "T": len(plan.selected.tasks),
                                "G": sum(len(t.operations) for t in plan.selected.tasks),
                                "time_ns": collect_ns(plan, args.repeats),
                            }
                        )
                    batches.append(rows)
    rows = [r for b in batches for r in b]
    models = {
        "task": ("W", "T", "one"),
        "gate": ("W", "G", "one"),
        "task_and_gate": ("W", "T", "G", "one"),
        "none": ("W", "one"),
    }
    order = list(range(len(batches)))
    random.Random(1).shuffle(order)
    folds = [order[i::5] for i in range(5)]
    fits = {}
    for name, columns in models.items():
        beta = fit([r | {"one": 1.0} for r in rows], columns)[0]
        errors, ratios = [], []
        for k in range(5):
            train = [r | {"one": 1.0} for i in order if i not in folds[k] for r in batches[i]]
            test = [r | {"one": 1.0} for i in folds[k] for r in batches[i]]
            b, _, _ = fit(train, columns)
            x = np.array([[r[c] for c in columns] for r in test])
            t = np.array([r["time_ns"] for r in test])
            errors.append(float(np.median(np.abs(x @ b / t - 1))))
            if "T" in columns or "G" in columns:
                ratios.append({c: float(b[j] / b[0]) for j, c in enumerate(columns) if c in "TG"})
        fits[name] = {
            "columns": columns,
            "beta": [float(v) for v in beta],
            "work_units": {c: float(beta[j] / beta[0]) for j, c in enumerate(columns) if c in "TG"},
            "cv_median_abs_relative_error": errors,
            "cv_mean": float(np.mean(errors)),
            "cv_work_units": ratios,
        }
    c0 = fits["task"]["work_units"]["T"]
    c1 = fits["gate"]["work_units"]["G"]
    summary = {
        "fits": fits,
        "c0_work_units": c0,
        "c1_work_units_per_gate": c1,
        "batches": len(batches),
        "rows": len(rows),
        "calibration_ranking": {
            "current": ranking(batches, 0.0, 0.0),
            "per task": ranking(batches, c0, 0.0),
            "per gate": ranking(batches, 0.0, c1),
        },
        "ranking_fields": ["share of batches picking the fastest", "geomean ratio", "max ratio"],
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    }
    host = {"platform": platform.platform(), "python": platform.python_version()}
    out = HERE.parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": host, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
