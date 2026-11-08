#!/usr/bin/env python3
"""Scale shared-cone QML execution to 20 qubits on the local Qulacs backend."""

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

import numpy as np
from benchmark_support import cpu_name, pin_single_thread, source_hash
from qml_workload_support import make_batch, numpy_oracle_expectations

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "qml_workload_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest["large_cone_probe"]
    repeats = args.repeats or config["repetitions"]
    if repeats < 1:
        parser.error("--repeats must be positive")
    pin_single_thread()

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()
    qubits = config["qubits"][-1:] if args.smoke else config["qubits"]
    families = config["families"][:1] if args.smoke else config["families"]
    offsets = config["seed_offsets"][:1] if args.smoke else config["seed_offsets"]
    strategies = config["strategies"]
    rng = random.Random(manifest["comparison"]["randomized_strategy_order_seed"] + 811)
    rows, summaries, failures = [], [], []

    for family in families:
        for n in qubits:
            for offset in offsets:
                seed = manifest["seed"] + offset
                requests = make_batch(family, n, config["depth"], config["pattern"], seed)
                expected = np.asarray(numpy_oracle_expectations(requests))
                case = {
                    "case_id": f"{family}_{n}q_d{config['depth']}_{config['pattern']}_s{offset}",
                    "family": family,
                    "qubits": n,
                    "depth": config["depth"],
                    "pattern": config["pattern"],
                    "seed": seed,
                    "seed_offset": offset,
                }
                plans, planning_times = {}, {}
                for strategy in strategies:
                    start = perf_counter_ns()
                    plans[strategy] = plan_all(
                        requests,
                        strategy=strategy,
                        optimization="none",
                        max_state_qubits=config["max_state_qubits"],
                        max_state_bytes=config["max_state_bytes"],
                    )
                    planning_times[strategy] = perf_counter_ns() - start

                samples = {strategy: [] for strategy in strategies}
                for repetition in range(repeats):
                    order = list(strategies)
                    rng.shuffle(order)
                    for strategy in order:
                        plan = plans[strategy]
                        backend = QulacsBackend()
                        start = perf_counter_ns()
                        result = plan.collect(backend=backend)
                        collection_ns = perf_counter_ns() - start
                        error = float(np.max(np.abs(np.asarray(result.values) - expected)))
                        if not np.isfinite(error) or error > 1e-10:
                            failures.append(
                                {"case_id": case["case_id"], "strategy": strategy, "error": error}
                            )
                            continue
                        row = {
                            **case,
                            "strategy": strategy,
                            "selected_strategy": result.selected_strategy,
                            "repetition": repetition,
                            "planning_ns": planning_times[strategy],
                            "collection_ns": collection_ns,
                            "total_ns": planning_times[strategy] + collection_ns,
                            "evolutions": result.stats.evolutions,
                            "task_count": len(plan.selected.tasks),
                            "task_widths": [task.width for task in plan.selected.tasks],
                            "max_state_qubits": result.stats.max_state_qubits,
                            "estimated_state_bytes": result.stats.estimated_state_bytes,
                            "max_abs_error": error,
                        }
                        rows.append(row)
                        samples[strategy].append(row)

                for strategy, strategy_rows in samples.items():
                    if not strategy_rows:
                        continue
                    summaries.append(
                        {
                            **case,
                            "strategy": strategy,
                            "median_planning_ms": median(r["planning_ns"] for r in strategy_rows)
                            / 1e6,
                            "median_collection_ms": median(
                                r["collection_ns"] for r in strategy_rows
                            )
                            / 1e6,
                            "median_total_ms": median(r["total_ns"] for r in strategy_rows)
                            / 1e6,
                            "evolutions": strategy_rows[0]["evolutions"],
                            "task_count": strategy_rows[0]["task_count"],
                            "task_widths": strategy_rows[0]["task_widths"],
                            "max_state_qubits": strategy_rows[0]["max_state_qubits"],
                            "max_abs_error": max(r["max_abs_error"] for r in strategy_rows),
                            "n": len(strategy_rows),
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
            "case_count": len(summaries) // len(strategies),
            "failures": failures,
        },
        "summary": summaries,
        "runs": rows,
    }
    output = args.output or ROOT / "results/qml_workloads/large_cone_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"cases={payload['metadata']['case_count']} rows={len(rows)} "
        f"failures={len(failures)} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} local oracle mismatches")


if __name__ == "__main__":
    main()
