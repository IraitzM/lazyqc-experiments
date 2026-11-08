#!/usr/bin/env python3
"""Frozen QML-inspired batch workload comparison for observable sharing."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import cpu_name, pin_single_thread, source_hash
from qml_workload_support import make_batch, numpy_oracle_expectations, workload_cases

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "qml_workload_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--analysis",
        choices=("budgeted", "always"),
        help="plan_all analysis mode; omitted, the planner default applies",
    )
    args = parser.parse_args()
    analysis_option = {"analysis": args.analysis} if args.analysis else {}

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    repeats = args.repeats or manifest["comparison"]["default_repeats"]
    if repeats < 1:
        parser.error("--repeats must be positive")

    pin_single_thread()
    import numpy as np

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    strategies = manifest["comparison"]["strategies"]
    rng = random.Random(manifest["comparison"]["randomized_strategy_order_seed"])
    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()

    rows, failures = [], []
    cases = workload_cases(manifest, smoke=args.smoke)
    for case in cases:
        requests = make_batch(
            case["family"],
            case["qubits"],
            case["depth"],
            case["pattern"],
            case["seed"],
        )
        expected = np.asarray(numpy_oracle_expectations(requests))
        unique_program_count = len({request.program for request in requests})
        for repetition in range(repeats):
            order = list(strategies)
            rng.shuffle(order)
            for strategy in order:
                backend = QulacsBackend()
                start = perf_counter_ns()
                plan = plan_all(
                    requests,
                    **analysis_option,
                    strategy=strategy,
                    optimization=manifest["comparison"]["optimization"],
                    max_state_qubits=manifest["comparison"]["max_state_qubits"],
                    max_state_bytes=manifest["comparison"]["max_state_bytes"],
                )
                planned = perf_counter_ns()
                result = plan.collect(backend=backend)
                finished = perf_counter_ns()
                error = float(np.max(np.abs(np.asarray(result.values) - expected)))
                if not np.isfinite(error) or error > manifest["comparison"]["error_tolerance"]:
                    failures.append(
                        {
                            "case_id": case["case_id"],
                            "strategy": strategy,
                            "error": error,
                        }
                    )
                    continue
                rows.append(
                    {
                        **case,
                        "repetition": repetition,
                        "strategy": strategy,
                        "selected_strategy": result.selected_strategy,
                        "planning_ns": planned - start,
                        "collection_ns": finished - planned,
                        "total_ns": finished - start,
                        "request_count": len(requests),
                        "unique_program_count": unique_program_count,
                        "unique_evolutions": result.stats.evolutions,
                        "task_count": result.stats.allocations,
                        "max_state_qubits": result.stats.max_state_qubits,
                        "estimated_state_bytes": result.stats.estimated_state_bytes,
                        "max_abs_error": error,
                    }
                )

    summary = []
    for case_id in sorted({row["case_id"] for row in rows}):
        for strategy in strategies:
            selected = [
                row for row in rows if row["case_id"] == case_id and row["strategy"] == strategy
            ]
            if not selected:
                continue
            summary.append(
                {
                    "case_id": case_id,
                    "strategy": strategy,
                    "median_total_ms": median(row["total_ns"] for row in selected) / 1e6,
                    "median_planning_ms": median(row["planning_ns"] for row in selected) / 1e6,
                    "median_collection_ms": median(row["collection_ns"] for row in selected) / 1e6,
                    "unique_evolutions": selected[0]["unique_evolutions"],
                    "task_count": selected[0]["task_count"],
                    "max_state_qubits": selected[0]["max_state_qubits"],
                    "max_abs_error": max(row["max_abs_error"] for row in selected),
                    "n": len(selected),
                }
            )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": source_hash(__file__),
            "workload_support_sha256": hashlib.sha256(
                (ROOT / "benchmarks" / "qml_workload_support.py").read_bytes()
            ).hexdigest(),
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "python": sys.version,
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("lazyqc", "qulacs", "numpy")
            },
            "command": sys.argv,
            "repeats": repeats,
            "smoke": args.smoke,
            "case_count": len(cases),
            "failures": failures,
        },
        "summary": summary,
        "runs": rows,
    }
    output = args.output or ROOT / "results/qml_workloads/v1/evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")

    print("case  strategy  total / planning / collection ms  evolutions  maxq  error")
    for row in summary:
        print(
            f"{row['case_id']}  {row['strategy']}  "
            f"{row['median_total_ms']:.3f} / {row['median_planning_ms']:.3f} / "
            f"{row['median_collection_ms']:.3f}  {row['unique_evolutions']}  "
            f"{row['max_state_qubits']}  {row['max_abs_error']:.1e}"
        )
    print(f"Saved {len(rows)} timed rows to {output}")
    if failures:
        raise SystemExit(f"{len(failures)} numerical validation failures")


if __name__ == "__main__":
    main()
