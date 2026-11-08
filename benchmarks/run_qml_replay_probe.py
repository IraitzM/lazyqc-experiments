#!/usr/bin/env python3
"""Measure charged structural-plan replay across QML parameter trajectories."""

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
from qml_workload_support import numpy_oracle_expectations, trajectory_batch

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
    config = manifest["replay_probe"]
    repeats = args.repeats or config["repetitions_per_binding"]
    if repeats < 1:
        parser.error("--repeats must be positive")
    pin_single_thread()

    from lazyqc import CutOptions, Program
    from lazyqc.backends.qulacs import QulacsBackend
    from lazyqc.batch_planner import plan_batch

    warm = Program(1).h(0)
    plan_batch([warm.expectation({0: "Z"})], cutting=CutOptions(max_cuts=0)).collect()
    options = CutOptions(
        max_fragment_qubits=24,
        max_cuts=0,
        max_group_size=4,
        beam_width=3,
        max_search_states=64,
        soft_timeout_ms=1000,
        search="joint",
    )
    offsets = config["seed_offsets"][:1] if args.smoke else config["seed_offsets"]
    sizes = config["qubits"][:1] if args.smoke else config["qubits"]
    families = config["families"][:1] if args.smoke else config["families"]
    patterns = config["patterns"][:1] if args.smoke else config["patterns"]
    cases = []
    for family in families:
        for n in sizes:
            for offset in offsets:
                for pattern in patterns:
                    cases.append(
                        {
                            "case_id": f"{family}_{n}q_d{config['depth']}_{pattern}_s{offset}",
                            "family": family,
                            "qubits": n,
                            "depth": config["depth"],
                            "seed_offset": offset,
                            "seed": manifest["seed"] + offset,
                            "pattern": pattern,
                        }
                    )

    rng = random.Random(manifest["comparison"]["randomized_strategy_order_seed"] + 79)
    rows, case_summaries, failures = [], [], []
    theta_values = tuple(-0.55 + 0.22 * index for index in range(config["bindings_per_trajectory"]))
    for case in cases:
        shifted_pair = case["pattern"] == "single_occurrence_shift_pair_trajectory"
        batches = [
            trajectory_batch(
                case["family"],
                case["qubits"],
                case["depth"],
                case["seed"],
                theta,
                shifted_pair=shifted_pair,
            )
            for theta in theta_values
        ]
        expected = [np.asarray(numpy_oracle_expectations(batch)) for batch in batches]

        # Template acquisition is the actual first trajectory step and is charged once.
        template_backend = QulacsBackend()
        start = perf_counter_ns()
        template_plan = plan_batch(
            batches[0],
            cutting=options,
            strategy="auto",
            optimization="none",
            max_state_qubits=24,
            max_state_bytes=256 * 2**20,
        )
        template_planning_ns = perf_counter_ns() - start
        start = perf_counter_ns()
        template_result = template_plan.collect(backend=template_backend)
        template_collection_ns = perf_counter_ns() - start
        template_error = float(
            np.max(np.abs(np.asarray(template_result.values) - expected[0]))
        )
        if template_error > 1e-10:
            failures.append(
                {
                    "case_id": case["case_id"],
                    "mode": "template",
                    "error": template_error,
                }
            )
            continue
        template = template_plan.explain()["search"]["structural_template"]
        case_fresh = []
        case_replay = []

        rows.append(
            {
                **case,
                "binding_index": 0,
                "theta": theta_values[0],
                "mode": "template_acquisition",
                "repetition": 0,
                "planning_ns": template_planning_ns,
                "collection_ns": template_collection_ns,
                "total_ns": template_planning_ns + template_collection_ns,
                "structural_hit": 0,
                "task_count": len(template_plan.selected.tasks),
                "request_count": len(batches[0]),
                "max_abs_error": template_error,
            }
        )

        for binding_index in range(len(theta_values)):
            if binding_index == 0:
                fresh_values = []
                for repetition in range(repeats):
                    backend = QulacsBackend()
                    start = perf_counter_ns()
                    plan = plan_batch(
                        batches[0],
                        cutting=options,
                        strategy="auto",
                        optimization="none",
                        max_state_qubits=24,
                        max_state_bytes=256 * 2**20,
                    )
                    planning_ns = perf_counter_ns() - start
                    start = perf_counter_ns()
                    result = plan.collect(backend=backend)
                    collection_ns = perf_counter_ns() - start
                    error = float(np.max(np.abs(np.asarray(result.values) - expected[0])))
                    fresh_values.append((planning_ns, collection_ns))
                    rows.append(
                        {
                            **case,
                            "binding_index": binding_index,
                            "theta": theta_values[binding_index],
                            "mode": "fresh",
                            "repetition": repetition,
                            "planning_ns": planning_ns,
                            "collection_ns": collection_ns,
                            "total_ns": planning_ns + collection_ns,
                            "structural_hit": 0,
                            "task_count": len(plan.selected.tasks),
                            "request_count": len(batches[binding_index]),
                            "max_abs_error": error,
                        }
                    )
                    if error > 1e-10:
                        failures.append(
                            {"case_id": case["case_id"], "mode": "fresh", "error": error}
                        )
                case_fresh.append(fresh_values)
                case_replay.append([(template_planning_ns, template_collection_ns)])
                continue

            modes = ["fresh", "replay"]
            fresh_values, replay_values = [], []
            for repetition in range(repeats):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    backend = QulacsBackend()
                    start = perf_counter_ns()
                    plan = plan_batch(
                        batches[binding_index],
                        cutting=options,
                        strategy="auto",
                        optimization="none",
                        max_state_qubits=24,
                        max_state_bytes=256 * 2**20,
                        _structural_hints=template if mode == "replay" else None,
                    )
                    planning_ns = perf_counter_ns() - start
                    start = perf_counter_ns()
                    result = plan.collect(backend=backend)
                    collection_ns = perf_counter_ns() - start
                    error = float(
                        np.max(np.abs(np.asarray(result.values) - expected[binding_index]))
                    )
                    hit = dict(plan.analysis_stats).get("structural_hit", 0)
                    rows.append(
                        {
                            **case,
                            "binding_index": binding_index,
                            "theta": theta_values[binding_index],
                            "mode": mode,
                            "repetition": repetition,
                            "planning_ns": planning_ns,
                            "collection_ns": collection_ns,
                            "total_ns": planning_ns + collection_ns,
                            "structural_hit": hit,
                            "task_count": len(plan.selected.tasks),
                            "request_count": len(batches[binding_index]),
                            "max_abs_error": error,
                        }
                    )
                    if error > 1e-10 or (mode == "replay" and hit != 1):
                        failures.append(
                            {
                                "case_id": case["case_id"],
                                "mode": mode,
                                "binding_index": binding_index,
                                "error": error,
                                "structural_hit": hit,
                            }
                        )
                    (fresh_values if mode == "fresh" else replay_values).append(
                        (planning_ns, collection_ns)
                    )
            case_fresh.append(fresh_values)
            case_replay.append(replay_values)

        def totals(per_binding):
            return sum(
                median(sample[0] for sample in binding)
                + median(sample[1] for sample in binding)
                for binding in per_binding
            )

        fresh_lifetime = totals(case_fresh)
        replay_lifetime = totals(case_replay)
        case_summaries.append(
            {
                **case,
                "bindings": len(theta_values),
                "template_planning_ns": template_planning_ns,
                "template_collection_ns": template_collection_ns,
                "fresh_lifetime_ns": fresh_lifetime,
                "replay_lifetime_ns": replay_lifetime,
                "fresh_over_replay_lifetime": fresh_lifetime / replay_lifetime,
                "median_fresh_planning_ns": median(
                    row["planning_ns"]
                    for row in rows
                    if row["case_id"] == case["case_id"] and row["mode"] == "fresh"
                ),
                "median_replay_planning_ns": median(
                    row["planning_ns"]
                    for row in rows
                    if row["case_id"] == case["case_id"] and row["mode"] == "replay"
                ),
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
            "repetitions_per_binding": repeats,
            "case_count": len(cases),
            "failures": failures,
            "max_abs_error": max(
                row["max_abs_error"] for row in rows
            ),
        },
        "cases": case_summaries,
        "runs": rows,
    }
    output = args.output or ROOT / "results/qml_workloads/replay_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"cases={len(cases)} rows={len(rows)} failures={len(failures)} "
        f"max_error={payload['metadata']['max_abs_error']:.2e} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} replay or numerical failures")


if __name__ == "__main__":
    main()
