#!/usr/bin/env python3
"""Measure a finite exhaustive reference for small same-state cone groupings."""

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


def partitions(items):
    """Yield each canonical set partition of an ordered index tuple once."""
    if not items:
        yield ()
        return
    head, *tail = items
    for partition in partitions(tuple(tail)):
        yield ((head,), *partition)
        for block_index in range(len(partition)):
            merged = list(partition)
            merged[block_index] = (head, *merged[block_index])
            yield tuple(merged)


def task_signature(task):
    operations = tuple((gate.name, gate.qubits, gate.angle) for gate in task.operations)
    outputs = tuple((index, observable.factors) for index, observable in task.outputs)
    return task.wires, operations, outputs


def plan_signature(plan):
    return tuple(sorted((task_signature(task) for task in plan.selected.tasks), key=repr))


def exhaustive_plans(requests):
    """Enumerate complete plans induced by every partition of active outputs."""
    from lazyqc.ir import Pauli
    from lazyqc.passes import causal_cone
    from lazyqc.plan import Candidate, Plan, Task

    active = tuple(i for i, request in enumerate(requests) if request.observable.factors)
    constants = tuple(
        (i, 1.0) for i, request in enumerate(requests) if not request.observable.factors
    )
    program = requests[active[0]].program
    operations = program.resolved_operations()
    recipes = {}
    for partition in partitions(active):
        tasks = []
        for block in partition:
            support = sorted(
                {q for i in block for q, _ in requests[i].observable.factors}
            )
            probe = Pauli(tuple((q, "Z") for q in support))
            wires, cone = causal_cone(operations, probe)
            outputs = tuple((i, requests[i].observable) for i in block)
            tasks.append(Task(wires, cone, outputs))
        tasks = tuple(tasks)
        if {index for task in tasks for index, _ in task.outputs} != set(active):
            raise ArithmeticError("Exhaustive partition produced incomplete coverage")
        signature = tuple(sorted((task_signature(task) for task in tasks), key=repr))
        if signature not in recipes:
            recipes[signature] = tasks

    full_task = Task(
        tuple(range(program.n_qubits)),
        operations,
        tuple((index, requests[index].observable) for index in active),
    )
    full_signature = (task_signature(full_task),)
    recipes.setdefault(full_signature, (full_task,))

    plans = []
    for candidate_index, tasks in enumerate(recipes.values()):
        candidate = Candidate(f"partition-{candidate_index}", tasks)
        plans.append(
            Plan(
                candidate,
                (candidate,),
                len(requests),
                constants,
                "enumerated-partition",
                24,
                256 * 2**20,
                0,
            )
        )
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest["exhaustive_oracle"]
    repeats = args.repeats or config["repetitions"]
    collections_per_plan = config["collections_per_plan"]
    if repeats < 1:
        parser.error("--repeats must be positive")
    pin_single_thread()

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()

    cases = []
    offsets = config["seed_offsets"][:1] if args.smoke else config["seed_offsets"]
    for family in config["families"]:
        for n in config["qubits"]:
            for depth in config["depths"]:
                for offset in offsets:
                    case = {
                        "case_id": f"{family}_{n}q_d{depth}_s{offset}",
                        "family": family,
                        "qubits": n,
                        "depth": depth,
                        "seed_offset": offset,
                        "seed": manifest["seed"] + offset,
                    }
                    requests = make_batch(
                        family,
                        n,
                        depth,
                        "same_state_local_outputs",
                        case["seed"],
                    )[: config["active_observables_per_case"]]
                    if len({request.program for request in requests}) != 1:
                        raise ValueError(
                            "Exhaustive oracle requires one bound preparation per batch"
                        )
                    oracle = np.asarray(numpy_oracle_expectations(requests))
                    start = perf_counter_ns()
                    candidates = exhaustive_plans(requests)
                    enumeration_ns = perf_counter_ns() - start
                    cases.append((case, requests, oracle, candidates, enumeration_ns))
    if args.smoke:
        cases = cases[:2]

    rng = random.Random(manifest["comparison"]["randomized_strategy_order_seed"] + 101)
    rows, failures = [], []
    for case, requests, oracle, candidates, enumeration_ns in cases:
        candidate_signatures = {
            plan_signature(plan): plan.selected.label for plan in candidates
        }
        jobs = [("enumerated", candidate) for candidate in candidates]
        jobs.extend((strategy, None) for strategy in ("full", "separate", "grouped", "auto"))
        for repetition in range(repeats):
            ordered = list(jobs)
            rng.shuffle(ordered)
            for label, fixed_plan in ordered:
                backend = QulacsBackend()
                planning_ns = 0
                if fixed_plan is None:
                    start = perf_counter_ns()
                    plan = plan_all(
                        requests,
                        strategy=label,
                        max_state_qubits=24,
                        max_state_bytes=256 * 2**20,
                    )
                    planning_ns = perf_counter_ns() - start
                else:
                    plan = fixed_plan
                collection_times = []
                error = 0.0
                result = None
                for _ in range(collections_per_plan):
                    start = perf_counter_ns()
                    result = plan.collect(backend=backend)
                    collection_times.append(perf_counter_ns() - start)
                    error = max(
                        error,
                        float(np.max(np.abs(np.asarray(result.values) - oracle))),
                    )
                collection_ns = sum(collection_times)
                matched_candidate = candidate_signatures.get(plan_signature(plan))
                if not np.isfinite(error) or error > 1e-10:
                    failures.append(
                        {"case_id": case["case_id"], "policy": label, "error": error}
                    )
                    continue
                rows.append(
                    {
                        **case,
                        "repetition": repetition,
                        "policy": label,
                        "candidate_id": (
                            plan.selected.label
                        ),
                        "planning_ns": planning_ns,
                        "collection_ns": collection_ns,
                        "collection_per_call_ns": collection_ns / collections_per_plan,
                        "total_ns": planning_ns + collection_ns,
                        "candidate_count": len(candidates),
                        "partition_covered": matched_candidate is not None,
                        "matched_candidate_id": matched_candidate,
                        "task_count": len(plan.selected.tasks),
                        "evolutions": result.stats.evolutions,
                        "max_state_qubits": result.stats.max_state_qubits,
                        "max_abs_error": error,
                    }
                )

    case_summaries = []
    for case, _, _, candidates, enumeration_ns in cases:
        selected = [row for row in rows if row["case_id"] == case["case_id"]]
        policy_medians = {}
        for policy in ("full", "separate", "grouped", "auto"):
            policy_rows = [row for row in selected if row["policy"] == policy]
            policy_medians[policy] = {
                "planning_ns": median(row["planning_ns"] for row in policy_rows),
                "collection_ns": median(
                    row["collection_per_call_ns"] for row in policy_rows
                ),
                "total_ns": median(row["total_ns"] for row in policy_rows),
            }
        candidate_medians = {
            candidate.selected.label: median(
                row["collection_per_call_ns"]
                for row in selected
                if row["policy"] == "enumerated"
                and row["candidate_id"] == candidate.selected.label
            )
            for candidate in candidates
        }
        best_candidate_id, best_enum = min(candidate_medians.items(), key=lambda item: item[1])
        auto_selected_candidate_id = next(
            row["matched_candidate_id"]
            for row in selected
            if row["policy"] == "auto"
        )
        auto_plan_ns = policy_medians["auto"]["planning_ns"]
        auto_per_call_ns = policy_medians["auto"]["collection_ns"]
        case_summaries.append(
            {
                **case,
                "candidate_count": len(candidates),
                "enumeration_ns": enumeration_ns,
                "best_enumerated_collection_ns": best_enum,
                "best_enumerated_candidate_id": best_candidate_id,
                "auto_selected_candidate_id": auto_selected_candidate_id,
                "policy_medians_ns": policy_medians,
                "policy_graph_coverage": {
                    policy: next(
                        row["partition_covered"]
                        for row in selected
                        if row["policy"] == policy
                    )
                    for policy in ("full", "separate", "grouped", "auto")
                },
                "auto_collection_regret": policy_medians["auto"]["collection_ns"] / best_enum,
                "auto_total_regret_at_R1": (auto_plan_ns + auto_per_call_ns) / best_enum,
                "auto_total_regret_at_R5": (
                    auto_plan_ns + collections_per_plan * auto_per_call_ns
                )
                / (collections_per_plan * best_enum),
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
            "collections_per_plan": collections_per_plan,
            "smoke": args.smoke,
            "case_count": len(cases),
            "raw_row_count": len(rows),
            "maximum_absolute_error": max(row["max_abs_error"] for row in rows),
            "failures": failures,
            "oracle_scope": config["scope"],
            "enumeration_cost_treatment": (
                "reported separately; exhaustive execution minimum is a finite-portfolio "
                "reference, not a zero-cost policy"
            ),
        },
        "cases": case_summaries,
        "runs": rows,
    }
    stem = manifest.get("output_stem", "grouping_oracle")
    output = args.output or ROOT / "results/qml_workloads" / f"{stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"cases={len(cases)} rows={len(rows)} failures={len(failures)} "
        f"max_error={payload['metadata']['maximum_absolute_error']:.2e} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} numerical validation failures")


if __name__ == "__main__":
    main()
