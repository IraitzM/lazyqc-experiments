#!/usr/bin/env python3
"""Analysis budget on Hamiltonian averaging: many Pauli terms on small circuits.

Each batch requests many overlapping local Pauli terms of one bound
hardware-efficient ansatz, the shape of a small-molecule energy evaluation.
The linear CX chains saturate every cone, so cone analysis cannot pay off.
We compare auto with the budget, auto without it and the forced dense plan,
interleaved, single-threaded and validated against a NumPy oracle.

    uv run --locked python benchmarks/run_analysis_budget.py \
        --output results/analysis_budget/hamiltonian.json
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import (
    ansatz_expectations,
    build_ansatz,
    overlapping_terms,
    pin_single_thread,
    source_hash,
)

ROOT = Path(__file__).resolve().parents[1]
WIDTHS = {6: 2, 8: 2, 10: 3, 12: 3}
TERMS = (16, 64, 128)
MODES = ("auto_budgeted", "auto_always", "full")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pin_single_thread()

    import numpy as np

    from lazyqc import Gate, Program, plan_all

    plan_all([Program(1).h(0).expectation({0: "Z"})], strategy="full").collect()
    rng = random.Random(args.seed)
    order_rng = random.Random(args.seed + 1)
    rows, cases = [], []
    for n, layers in WIDTHS.items():
        gates = build_ansatz(n, layers, rng)
        program = Program(
            n,
            tuple(
                Gate("RY", (g[1],), g[2]) if g[0] == "ry" else Gate("CX", (g[1], g[2]))
                for g in gates
            ),
        )
        for count in TERMS:
            terms = overlapping_terms(n, count, rng)
            requests = [program.expectation(t) for t in terms]
            expected = np.asarray(ansatz_expectations(n, gates, terms))
            case = {
                "case_id": f"ansatz_{n}q_l{layers}_{count}terms",
                "qubits": n,
                "layers": layers,
                "terms": count,
                "gates": len(program.operations),
            }
            budget = plan_all(requests).explain()["analysis_budget"]
            cases.append(case | {"analysis_budget": budget})
            options = {
                "auto_budgeted": {},
                "auto_always": {"analysis": "always"},
                "full": {"strategy": "full"},
            }
            for repetition in range(-1, args.repeats):
                modes = list(MODES)
                order_rng.shuffle(modes)
                for mode in modes:
                    start = perf_counter_ns()
                    plan = plan_all(requests, **options[mode])
                    planned = perf_counter_ns()
                    values = plan.collect().values
                    finished = perf_counter_ns()
                    error = float(np.max(np.abs(np.asarray(values) - expected)))
                    if error > 1e-10:
                        raise AssertionError(f"{case['case_id']} {mode}: error {error}")
                    if repetition >= 0:
                        rows.append(
                            case
                            | {
                                "mode": mode,
                                "repetition": repetition,
                                "planning_ns": planned - start,
                                "collection_ns": finished - planned,
                                "total_ns": finished - start,
                                "selected": plan.selected.strategy,
                                "tasks": len(plan.selected.tasks),
                                "max_abs_error": error,
                            }
                        )
            medians = {
                mode: median(
                    r["total_ns"]
                    for r in rows
                    if r["case_id"] == case["case_id"] and r["mode"] == mode
                )
                / 1e6
                for mode in MODES
            }
            print(
                f"{case['case_id']:26s} skipped={budget['skipped']!s:5s} "
                + " ".join(f"{m}={v:.3f}ms" for m, v in medians.items()),
                flush=True,
            )
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hash(__file__),
            "command": sys.argv,
            "repeats": args.repeats,
        },
        "cases": cases,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
