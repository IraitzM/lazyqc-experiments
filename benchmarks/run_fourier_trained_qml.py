#!/usr/bin/env python3
"""Replay trained Fourier-study angle classifiers as LazyQC inference batches.

Needs the archives written by ``export_fourier_trained_models.py``. Each model is
rebuilt gate by gate: per upload, RY(angle_i) on every wire, then one
StronglyEntanglingLayers layer (Rot(phi, theta, omega) = RZ(phi), RY(theta),
RZ(omega) on each wire, then CX(i, i + 1 mod 8)). The read-out is Z_i and
Z_i Z_{i+1 mod 8}, followed by the trained linear head.
"""

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

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "fourier_trained_qml_manifest.json"


def build_program(angles, weights):
    from lazyqc import Gate, Program

    n = len(angles)
    operations = []
    for upload in weights:
        operations.extend(Gate("RY", (q,), float(angles[q])) for q in range(n))
        for layer in upload:
            for q, (phi, theta, omega) in enumerate(layer):
                operations.extend(
                    (
                        Gate("RZ", (q,), float(phi)),
                        Gate("RY", (q,), float(theta)),
                        Gate("RZ", (q,), float(omega)),
                    )
                )
            if n > 1:  # as in PennyLane, one wire has no entangling ring
                operations.extend(Gate("CX", (q, (q + 1) % n)) for q in range(n))
    return Program(n, operations)


def readout(n):
    from lazyqc import Pauli

    singles = [Pauli(((q, "Z"),)) for q in range(n)]
    pairs = [Pauli(tuple(sorted(((q, "Z"), ((q + 1) % n, "Z"))))) for q in range(n)]
    return singles + pairs


def balanced_accuracy(labels, predictions):
    return float(np.mean([np.mean(predictions[labels == c] == c) for c in np.unique(labels)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one model, batch size 1")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    index_path = ROOT / manifest["source_index"]
    index = json.loads(index_path.read_text())
    pin_single_thread()

    from direct_support import qulacs_direct
    from qml_workload_support import numpy_oracle_expectations

    from lazyqc import plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    runs = index["runs"][:1] if args.smoke else index["runs"]
    batch_sizes = manifest["batch_sizes"][:1] if args.smoke else manifest["batch_sizes"]
    strategies = manifest["strategies"]
    methods = [*strategies, "direct"] if manifest["direct_baseline"] else list(strategies)
    order_rng = random.Random(manifest["randomized_order_seed"])
    plan_all([build_program([0.0], [[[[0.0, 0.0, 0.0]]]]).expectation({0: "Z"})]).collect(
        backend=QulacsBackend()
    )
    validations, rows, failures = [], [], []

    for run in runs:
        data = np.load(index_path.parent / run["file"])
        angles, weights = data["angles"], data["weights"]
        n = angles.shape[1]
        observables = readout(n)
        programs = [build_program(row, weights) for row in angles]
        requests = [p.expectation(o) for p in programs for o in observables]
        values = np.asarray(
            plan_all(requests, strategy="full", optimization="none")
            .collect(backend=QulacsBackend())
            .values
        )
        oracle = np.asarray(numpy_oracle_expectations(requests))
        error = float(np.max(np.abs(values - oracle)))
        features = values.reshape(len(programs), len(observables))
        logits = features @ data["readout_weight"].T + data["readout_bias"]
        predictions = logits.argmax(axis=1)
        mismatched = predictions != data["study_predictions"]
        margins = np.abs(logits[:, 1] - logits[:, 0]) if logits.shape[1] == 2 else None
        validation = {
            "dataset": run["dataset"],
            "seed": run["seed"],
            "test_examples": len(programs),
            "max_abs_oracle_error": error,
            "prediction_mismatches": int(mismatched.sum()),
            "largest_mismatch_margin": (
                float(margins[mismatched].max()) if margins is not None and mismatched.any()
                else None
            ),
            "balanced_accuracy": balanced_accuracy(data["labels"], predictions),
            "study_balanced_accuracy": balanced_accuracy(
                data["labels"], data["study_predictions"]
            ),
        }
        validations.append(validation)
        if error > 1e-10:
            failures.append({**validation, "stage": "oracle"})
        print(
            f"validated {run['dataset']}-{run['seed']}: error={error:.1e} "
            f"mismatches={validation['prediction_mismatches']}",
            flush=True,
        )

        for batch_size in batch_sizes:
            batch = programs[:batch_size]
            batch_requests = [p.expectation(o) for p in batch for o in observables]
            expected = oracle[: len(batch_requests)]
            case = {
                "case_id": f"{run['dataset']}-{run['seed']}_b{batch_size}",
                "dataset": run["dataset"],
                "seed": run["seed"],
                "batch_size": batch_size,
                "request_count": len(batch_requests),
            }
            # New data arrive with every inference batch, so each repetition
            # pays setup (planning or Qulacs construction) and collection.
            for repetition in range(manifest["repeats"]):
                order = list(methods)
                order_rng.shuffle(order)
                for method in order:
                    start = perf_counter_ns()
                    if method == "direct":
                        runners = [qulacs_direct(p, observables) for p in batch]
                    else:
                        plan = plan_all(
                            batch_requests,
                            strategy=method,
                            optimization="none",
                            max_state_qubits=manifest["max_state_qubits"],
                            max_state_bytes=manifest["max_state_bytes"],
                        )
                    setup_ns = perf_counter_ns() - start
                    start = perf_counter_ns()
                    if method == "direct":
                        got = [v for runner in runners for v in runner()]
                        stats = {"evolutions": len(batch), "task_widths": [n] * len(batch)}
                        selected = "direct"
                    else:
                        result = plan.collect(backend=QulacsBackend())
                        got = result.values
                        selected = result.selected_strategy
                        stats = {
                            "evolutions": result.stats.evolutions,
                            "task_widths": [t.width for t in plan.selected.tasks],
                        }
                    elapsed = perf_counter_ns() - start
                    err = float(np.max(np.abs(np.asarray(got) - expected)))
                    if err > 1e-10:
                        failures.append({**case, "method": method, "stage": "collection",
                                         "error": err})
                        continue
                    rows.append(
                        {
                            **case,
                            "method": method,
                            "selected_strategy": selected,
                            "repetition": repetition,
                            "setup_ns": setup_ns,
                            "collection_ns": elapsed,
                            "task_count": len(stats["task_widths"]),
                            "max_task_width": max(stats["task_widths"]),
                            "evolutions": stats["evolutions"],
                            "max_abs_error": err,
                        }
                    )
            print(f"timed {case['case_id']}", flush=True)

    summary = []
    for batch_size in batch_sizes:
        cases = sorted({r["case_id"] for r in rows if r["batch_size"] == batch_size})
        lives = {m: [] for m in methods}
        choices = {}
        for case_id in cases:
            for method in methods:
                selected = [r for r in rows if r["case_id"] == case_id and r["method"] == method]
                lives[method].append(
                    median(r["setup_ns"] + r["collection_ns"] for r in selected)
                )
                if method == "auto":
                    choice = selected[0]["selected_strategy"]
                    choices[choice] = choices.get(choice, 0) + 1
        summary.append(
            {
                "batch_size": batch_size,
                "cases": len(cases),
                "auto_choices": choices,
                "median_lifetime_ms": {m: median(v) / 1e6 for m, v in lives.items()},
                "median_auto_over_direct": median(
                    a / d for a, d in zip(lives["auto"], lives["direct"], strict=True)
                ),
                "median_auto_over_best_forced": median(
                    a / min(f, s, g)
                    for a, f, s, g in zip(
                        lives["auto"], lives["full"], lives["separate"], lives["grouped"],
                        strict=True,
                    )
                ),
            }
        )
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "exploratory trained-model replay; not confirmatory",
            "backend": "QulacsBackend and direct Qulacs, single native thread",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "source_commit": index["source_commit"],
            "source_sha256": source_hash(__file__),
            "platform": platform.platform(),
            "cpu": cpu_name(),
            "python": sys.version,
            "packages": {p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")},
            "command": sys.argv,
            "smoke": args.smoke,
            "failures": failures,
        },
        "validation": validations,
        "summary": summary,
        "runs": rows,
    }
    output = args.output or ROOT / "results/qml_workloads/fourier_trained_qml.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"models={len(validations)} rows={len(rows)} failures={len(failures)} output={output}")
    if failures:
        raise SystemExit(f"{len(failures)} numerical failures; inspect artifact")


if __name__ == "__main__":
    main()
