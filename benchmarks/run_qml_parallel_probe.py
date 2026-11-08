#!/usr/bin/env python3
"""Compare serial and local process execution on fixed QML cone graphs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from dataclasses import replace
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
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    probe = manifest["parallel_probe"]
    repeats = args.repeats or probe["repetitions"]
    if repeats < 1:
        parser.error("--repeats must be positive")
    pin_single_thread()

    from lazyqc import CutOptions, ExecutionSession, Program, plan_all

    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()
    offsets = probe["seed_offsets"][:1] if args.smoke else probe["seed_offsets"]
    patterns = probe["patterns"][:2] if args.smoke else probe["patterns"]
    cases = []
    for family in manifest["families"]:
        for offset in offsets:
            for pattern in patterns:
                cases.append(
                    {
                        "case_id": (
                            f"{family}_{probe['qubits']}q_d{probe['depth']}_"
                            f"{pattern}_s{offset}"
                        ),
                        "family": family,
                        "qubits": probe["qubits"],
                        "depth": probe["depth"],
                        "pattern": pattern,
                        "seed": manifest["seed"] + offset,
                        "seed_offset": offset,
                    }
                )

    graphs = []
    for case in cases:
        requests = make_batch(
            case["family"], case["qubits"], case["depth"], case["pattern"], case["seed"]
        )
        expected = np.asarray(numpy_oracle_expectations(requests))
        for strategy in probe["graph_strategies"]:
            start = perf_counter_ns()
            plan = plan_all(
                requests,
                strategy=strategy,
                workers=1,
                max_state_qubits=24,
                max_state_bytes=256 * 2**20,
                cutting=CutOptions(max_fragment_qubits=24, max_cuts=0, search="independent"),
            )
            planning_ns = perf_counter_ns() - start
            task_payload = max((task.state_bytes for task in plan.selected.tasks), default=16)
            graphs.append((case, requests, expected, strategy, plan, planning_ns, task_payload))

    workers_to_run = [1, 2, 4]
    if args.smoke:
        workers_to_run = [1, 2]
    sessions, setup_ns, profiles = {}, {}, {}
    for workers in workers_to_run:
        if workers == 1:
            continue
        session = ExecutionSession(workers, policy="parallel")
        start = perf_counter_ns()
        profiles[workers] = session.warmup()
        setup_ns[workers] = perf_counter_ns() - start
        sessions[workers] = session

    rng = random.Random(manifest["comparison"]["randomized_strategy_order_seed"] + 37)
    rows, failures = [], []
    try:
        for case, requests, expected, strategy, base_plan, planning_ns, task_payload in graphs:
            for repetition in range(repeats):
                order = list(workers_to_run)
                rng.shuffle(order)
                for workers in order:
                    payload_budget = min(
                        base_plan.max_state_bytes, workers * task_payload
                    )
                    execution_plan = replace(
                        base_plan, workers=workers, total_state_bytes=payload_budget
                    )
                    start = perf_counter_ns()
                    result = execution_plan.collect(session=sessions.get(workers))
                    collection_ns = perf_counter_ns() - start
                    error = float(np.max(np.abs(np.asarray(result.values) - expected)))
                    if not np.isfinite(error) or error > 1e-10:
                        failures.append(
                            {
                                "case_id": case["case_id"],
                                "graph_strategy": strategy,
                                "workers": workers,
                                "error": error,
                            }
                        )
                        continue
                    rows.append(
                        {
                            **case,
                            "graph_strategy": strategy,
                            "workers": workers,
                            "repetition": repetition,
                            "planning_ns": planning_ns,
                            "collection_ns": collection_ns,
                            "task_count": len(base_plan.selected.tasks),
                            "evolutions": result.stats.evolutions,
                            "peak_active_state_bytes": result.stats.peak_active_state_bytes,
                            "worker_peak_rss_bytes": result.stats.worker_peak_rss_bytes,
                            "max_abs_error": error,
                        }
                    )
    finally:
        shutdown_ns = {}
        for workers, session in sessions.items():
            start = perf_counter_ns()
            session.close()
            shutdown_ns[workers] = perf_counter_ns() - start

    per_case = {}
    for case in cases:
        for strategy in probe["graph_strategies"]:
            medians = {}
            for workers in workers_to_run:
                selected = [
                    row["collection_ns"]
                    for row in rows
                    if row["case_id"] == case["case_id"]
                    and row["graph_strategy"] == strategy
                    and row["workers"] == workers
                ]
                if selected:
                    medians[str(workers)] = median(selected)
            per_case[f"{case['case_id']}:{strategy}"] = medians

    aggregate = {}
    serial_ns = sum(row["collection_ns"] for row in rows if row["workers"] == 1)
    for workers in workers_to_run:
        worker_ns = sum(row["collection_ns"] for row in rows if row["workers"] == workers)
        lifecycle_ns = worker_ns + setup_ns.get(workers, 0) + shutdown_ns.get(workers, 0)
        aggregate[str(workers)] = {
            "collection_sum_ns": worker_ns,
            "setup_ns": setup_ns.get(workers, 0),
            "shutdown_ns": shutdown_ns.get(workers, 0),
            "suite_lifecycle_over_serial": lifecycle_ns / serial_ns,
        }

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
            "workers": workers_to_run,
            "repeats": repeats,
            "smoke": args.smoke,
            "case_count": len(cases),
            "session_profiles": {
                str(worker): {
                    "startup_ns": profile.startup_ns,
                    "roundtrip_ns": profile.roundtrip_ns,
                    "workers": profile.workers,
                }
                for worker, profile in profiles.items()
            },
            "session_setup_ns": {str(k): v for k, v in setup_ns.items()},
            "session_shutdown_ns": {str(k): v for k, v in shutdown_ns.items()},
            "suite_aggregate": aggregate,
            "failures": failures,
        },
        "per_case_collection_medians_ns": per_case,
        "runs": rows,
    }
    output = args.output or ROOT / "results/qml_workloads/parallel_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"cases={len(cases)} rows={len(rows)} failures={len(failures)} output={output}")
    print(json.dumps(aggregate, indent=2))
    if failures:
        raise SystemExit(f"{len(failures)} numerical validation failures")


if __name__ == "__main__":
    main()
