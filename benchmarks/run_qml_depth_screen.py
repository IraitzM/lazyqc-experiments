#!/usr/bin/env python3
"""Run the exploratory local QML depth screen against Qulacs."""

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
MANIFEST = ROOT / "benchmarks" / "qml_depth_screen_manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run one small case")
    parser.add_argument("--repeats", type=int, help="override collection count per plan")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--session", type=int, help="fresh-interpreter session index; reseeds method order"
    )
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    repeats = args.repeats or manifest["collections_per_plan"]
    if repeats < 1:
        parser.error("--repeats must be positive")
    pin_single_thread()

    from direct_support import qulacs_direct

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    plan_all([Program(1).h(0).expectation({0: "Z"})], strategy="full").collect(
        backend=QulacsBackend()
    )
    families = manifest["families"][:1] if args.smoke else manifest["families"]
    qubits = manifest["qubits"][:1] if args.smoke else manifest["qubits"]
    depths = manifest["depths"][:1] if args.smoke else manifest["depths"]
    patterns = manifest["patterns"][:1] if args.smoke else manifest["patterns"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    strategies = manifest["strategies"]
    direct = manifest.get("direct_baseline", False)
    methods = [*strategies, "direct"] if direct else list(strategies)
    order_rng = random.Random(manifest["randomized_order_seed"] + 7919 * (args.session or 0))
    rows, summaries, failures = [], [], []

    for family in families:
        for n in qubits:
            for depth in depths:
                for pattern in patterns:
                    for offset in offsets:
                        seed = manifest["base_seed"] + offset
                        requests = make_batch(family, n, depth, pattern, seed)
                        expected = np.asarray(numpy_oracle_expectations(requests))
                        case = {
                            "case_id": f"{family}_{n}q_d{depth}_{pattern}_s{offset}",
                            "family": family,
                            "qubits": n,
                            "depth": depth,
                            "pattern": pattern,
                            "seed": seed,
                            "seed_offset": offset,
                            "request_count": len(requests),
                        }
                        plans, planning_ns = {}, {}
                        try:
                            for strategy in strategies:
                                start = perf_counter_ns()
                                plans[strategy] = plan_all(
                                    requests,
                                    strategy=strategy,
                                    optimization="none",
                                    max_state_qubits=manifest["max_state_qubits"],
                                    max_state_bytes=manifest["max_state_bytes"],
                                )
                                planning_ns[strategy] = perf_counter_ns() - start
                        except Exception as exc:
                            failures.append({**case, "stage": "planning", "error": repr(exc)})
                            continue
                        if direct:
                            # Structural strata use only planning output, fixed before timing.
                            grouped = plans["grouped"].selected
                            case.update(
                                grouped_max_width=max(t.width for t in grouped.tasks),
                                grouped_tasks=len(grouped.tasks),
                                separate_tasks=len(plans["separate"].selected.tasks),
                                auto_strategy=plans["auto"].selected.strategy,
                            )
                            start = perf_counter_ns()
                            direct_run = qulacs_direct(
                                requests[0].program, [r.observable for r in requests]
                            )
                            planning_ns["direct"] = perf_counter_ns() - start
                            if len({r.program for r in requests}) != 1:
                                raise ValueError("direct baseline needs one shared program")

                        samples = {strategy: [] for strategy in methods}
                        for repetition in range(repeats):
                            execution_order = list(methods)
                            order_rng.shuffle(execution_order)
                            for strategy in execution_order:
                                try:
                                    if strategy == "direct":
                                        start = perf_counter_ns()
                                        values = direct_run()
                                        elapsed = perf_counter_ns() - start
                                        error = float(
                                            np.max(np.abs(np.asarray(values) - expected))
                                        )
                                        if not np.isfinite(error) or error > 1e-10:
                                            failures.append(
                                                {
                                                    **case,
                                                    "strategy": strategy,
                                                    "repetition": repetition,
                                                    "stage": "oracle",
                                                    "error": error,
                                                }
                                            )
                                            continue
                                        record = {
                                            **case,
                                            "strategy": strategy,
                                            "selected_strategy": "direct",
                                            "session": args.session,
                                            "repetition": repetition,
                                            "planning_ns": planning_ns[strategy],
                                            "collection_ns": elapsed,
                                            "one_call_lifetime_ns": planning_ns[strategy]
                                            + elapsed,
                                            "evolutions": 1,
                                            "task_count": 1,
                                            "task_widths": [case["qubits"]],
                                            "max_state_qubits": case["qubits"],
                                            "estimated_state_bytes": 16 << case["qubits"],
                                            "max_abs_error": error,
                                        }
                                        rows.append(record)
                                        samples[strategy].append(record)
                                        continue
                                    plan = plans[strategy]
                                    start = perf_counter_ns()
                                    result = plan.collect(backend=QulacsBackend())
                                    elapsed = perf_counter_ns() - start
                                    error = float(
                                        np.max(np.abs(np.asarray(result.values) - expected))
                                    )
                                    if not np.isfinite(error) or error > 1e-10:
                                        failures.append(
                                            {
                                                **case,
                                                "strategy": strategy,
                                                "repetition": repetition,
                                                "stage": "oracle",
                                                "error": error,
                                            }
                                        )
                                        continue
                                    record = {
                                        **case,
                                        "strategy": strategy,
                                        "selected_strategy": result.selected_strategy,
                                        "session": args.session,
                                        "repetition": repetition,
                                        "planning_ns": planning_ns[strategy],
                                        "collection_ns": elapsed,
                                        "one_call_lifetime_ns": planning_ns[strategy] + elapsed,
                                        "evolutions": result.stats.evolutions,
                                        "task_count": len(plan.selected.tasks),
                                        "task_widths": [t.width for t in plan.selected.tasks],
                                        "max_state_qubits": result.stats.max_state_qubits,
                                        "estimated_state_bytes": result.stats.estimated_state_bytes,
                                        "max_abs_error": error,
                                    }
                                    rows.append(record)
                                    samples[strategy].append(record)
                                except Exception as exc:
                                    failures.append(
                                        {
                                            **case,
                                            "strategy": strategy,
                                            "repetition": repetition,
                                            "stage": "collection",
                                            "error": repr(exc),
                                        }
                                    )

                        for strategy, records in samples.items():
                            if not records:
                                continue
                            collections = [r["collection_ns"] for r in records]
                            summaries.append(
                                {
                                    **case,
                                    "strategy": strategy,
                                    "median_planning_ms": median(
                                        r["planning_ns"] for r in records
                                    ) / 1e6,
                                    "median_collection_ms": median(collections) / 1e6,
                                    "two_collection_lifetime_ms": (
                                        planning_ns[strategy] + sum(collections)
                                    ) / 1e6,
                                    "evolutions": records[0]["evolutions"],
                                    "task_count": records[0]["task_count"],
                                    "task_widths": records[0]["task_widths"],
                                    "max_state_qubits": records[0]["max_state_qubits"],
                                    "max_abs_error": max(r["max_abs_error"] for r in records),
                                    "successful_collections": len(records),
                                }
                            )
                        print(
                            f"finished {case['case_id']} "
                            f"({len(summaries)} strategy summaries, {len(failures)} failures)",
                            flush=True,
                        )

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "exploratory local screen; not confirmatory",
            "backend": "QulacsBackend, single native thread",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": source_hash(__file__),
            "workload_support_sha256": hashlib.sha256(
                (ROOT / "benchmarks" / "qml_workload_support.py").read_bytes()
            ).hexdigest(),
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")
            },
            "command": sys.argv,
            "repeats": repeats,
            "session": args.session,
            "smoke": args.smoke,
            "case_count": len({r["case_id"] for r in summaries}),
            "failures": failures,
        },
        "summary": summaries,
        "runs": rows,
    }
    stem = manifest_path.stem.removesuffix("_manifest")
    if args.session is not None:
        stem += f"_session{args.session}"
    default_output = ROOT / "results/qml_workloads" / f"{stem}.json"
    output = args.output or default_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"cases={payload['metadata']['case_count']} rows={len(rows)} "
        f"failures={len(failures)} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} cases/collections failed; inspect recorded failures")


if __name__ == "__main__":
    main()
