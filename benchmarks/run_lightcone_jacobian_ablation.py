#!/usr/bin/env python3
"""Cross-shift Qiskit LightCone + Aer deduplication versus LazyQC planning.

Uses the frozen ``lightcone_jacobian_manifest.json`` and the same optional GPU
stack as ``run_lightcone_ablation.py``. The compared object is the complete raw
expectation batch needed for a full Jacobian; shift-rule assembly is excluded.
"""

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from core_checkout import label, package_files

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "benchmarks")]
MANIFEST = ROOT / "benchmarks" / "lightcone_jacobian_manifest.json"


def source_hash():
    digest = hashlib.sha256()
    paths = package_files() + [
        Path(__file__).resolve(),
        ROOT / "benchmarks" / "run_lightcone_ablation.py",
        ROOT / "benchmarks" / "direct_support.py",
        ROOT / "benchmarks" / "qml_workload_support.py",
        MANIFEST,
    ]
    for path in paths:
        digest.update(label(path).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from direct_support import (
        aer_simulator,
        atomic_write,
        environment,
        jacobian_requests,
        pin_threads,
        shifted_programs,
        timed,
        trainable_indices,
    )

    pin_threads(1)
    import numpy as np
    from qiskit.converters import circuit_to_dag
    from qml_workload_support import make_batch
    from run_lightcone_ablation import (
        execute_aer_tasks,
        lightcone_batch_tasks,
        qiskit_circuit,
    )

    from lazyqc import plan_all

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    widths = manifest["qubits"][:1] if args.smoke else manifest["qubits"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    limits = {"max_state_qubits": 28, "max_state_bytes": 4 * 2**30}
    simulator = aer_simulator("GPU")
    rng = random.Random(manifest["seed"] + 2909)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_hash(),
        "command": sys.argv,
        "environment": environment(),
        "smoke": args.smoke,
    }
    rows, cases, failures = [], [], []

    for n in widths:
        for offset in offsets:
            seed = manifest["seed"] + offset
            base_requests = make_batch(
                manifest["family"], n, manifest["depth"], "same_state_local_outputs", seed
            )
            program = base_requests[0].program
            observables = [request.observable for request in base_requests]
            parameters = trainable_indices(program)
            programs = shifted_programs(program, parameters)
            requests = jacobian_requests(programs, observables)
            dags = [circuit_to_dag(qiskit_circuit(p)) for p in programs]
            jobs = [
                (dag, wire, program_index * n + wire)
                for program_index, dag in enumerate(dags)
                for wire in range(n)
            ]
            if len(jobs) != len(requests):
                raise AssertionError("Qiskit and LazyQC request orders differ")

            def lazyqc_run():
                plan, analysis_ns = timed(lambda: plan_all(requests, **limits))
                result, execution_ns = timed(plan.collect)
                lazyqc_run.detail = {
                    "analysis_ns": analysis_ns,
                    "execution_ns": execution_ns,
                    "tasks": len(plan.selected.tasks),
                    "min_width": min(t.width for t in plan.selected.tasks),
                    "max_width": max(t.width for t in plan.selected.tasks),
                    "selected_strategy": plan.selected.strategy,
                }
                return list(result.values)

            def aer_run(deduplicate):
                tasks, analysis_ns = timed(lambda: lightcone_batch_tasks(jobs, deduplicate))
                values, execution_ns = execute_aer_tasks(simulator, tasks, len(requests))
                aer_run.detail = {
                    "analysis_ns": analysis_ns,
                    "execution_ns": execution_ns,
                    "tasks": len(tasks),
                    "min_width": min(task[0].num_qubits for task in tasks),
                    "max_width": max(task[0].num_qubits for task in tasks),
                    "selected_strategy": (
                        "lightcone_dedup" if deduplicate else "lightcone_separate"
                    ),
                }
                return values

            reference = np.asarray(lazyqc_run())
            case = {
                "case_id": f"hardware_efficient_local_{n}q_d4_jacobian_s{offset}",
                "family": manifest["family"],
                "qubits": n,
                "depth": manifest["depth"],
                "seed_offset": offset,
                "parameters": len(parameters),
                "programs": len(programs),
                "outputs": n,
                "requests": len(requests),
            }
            repetitions = 1 if args.smoke else manifest["repetitions"][str(n)]
            methods = [
                ("lightcone_separate_aer", lambda: aer_run(False)),
                ("lightcone_dedup_aer", lambda: aer_run(True)),
                ("lazyqc", lazyqc_run),
            ]
            rng.shuffle(methods)
            for method, run in methods:
                try:
                    warm_values = np.asarray(run())
                    warm_error = float(np.max(np.abs(warm_values - reference)))
                    if warm_error > 1e-9:
                        failures.append(
                            {"case_id": case["case_id"], "method": method, "error": warm_error}
                        )
                    for repetition in range(repetitions):
                        values, total_ns = timed(run)
                        error = float(np.max(np.abs(np.asarray(values) - reference)))
                        if error > 1e-9:
                            failures.append(
                                {"case_id": case["case_id"], "method": method, "error": error}
                            )
                        detail = lazyqc_run.detail if method == "lazyqc" else aer_run.detail
                        rows.append(
                            case
                            | {
                                "method": method,
                                "repetition": repetition,
                                "analysis_ns": detail["analysis_ns"],
                                "execution_ns": detail["execution_ns"],
                                "total_ns": total_ns,
                                "tasks": detail["tasks"],
                                "min_width": detail["min_width"],
                                "max_width": detail["max_width"],
                                "selected_strategy": detail["selected_strategy"],
                                "max_abs_error": error,
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - recorded in the artifact
                    failures.append(
                        {"case_id": case["case_id"], "method": method, "exception": repr(exc)}
                    )
            cases.append(case)
            summary = {
                method: {
                    "tasks": next(
                        row["tasks"]
                        for row in rows
                        if row["case_id"] == case["case_id"] and row["method"] == method
                    ),
                    "best_ms": min(
                        row["total_ns"]
                        for row in rows
                        if row["case_id"] == case["case_id"] and row["method"] == method
                    )
                    / 1e6,
                }
                for method, _ in methods
                if any(
                    row["case_id"] == case["case_id"] and row["method"] == method for row in rows
                )
            }
            print(f"{case['case_id']:52s} {summary}", flush=True)
            atomic_write(
                args.output,
                json.dumps(
                    {"metadata": metadata, "failures": failures, "cases": cases, "rows": rows},
                    indent=1,
                ),
            )
    atomic_write(
        args.output,
        json.dumps(
            {"metadata": metadata, "failures": failures, "cases": cases, "rows": rows}, indent=1
        ),
    )
    print(f"cases {len(cases)} rows {len(rows)} failures {len(failures)}")
    if failures:
        raise SystemExit(f"{len(failures)} LightCone Jacobian-ablation failures")


if __name__ == "__main__":
    main()
