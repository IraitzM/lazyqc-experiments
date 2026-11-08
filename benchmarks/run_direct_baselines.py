#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "lazyqc==0.5.0",
#   "numpy==2.5.4",
#   "scipy>=1.14,<2",
#   "qulacs==0.6.14",
#   "qiskit==2.6.1",
#   "qiskit-aer-gpu==0.18.0",
#   "pennylane==0.45.1",
#   "pennylane-lightning==0.45.0",
#   "pennylane-lightning-gpu==0.45.0",
#   "quimb==1.15.0",
# ]
# ///
"""Planner versus direct dense execution on CPU and GPU across circuit width.

Run once per thread setting; methods outside the manifest's limits are skipped:

    uv run --script benchmarks/run_direct_baselines.py --threads 1 \
        --methods lazyqc,lazyqc_gpu,qulacs_1t --output results/direct/scaling_1t.json
    uv run --script benchmarks/run_direct_baselines.py --threads 20 \
        --methods aer_cpu,aer_gpu,lightning_gpu --output results/direct/scaling_mt.json
"""

import argparse
import gc
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from core_checkout import label, package_files

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "benchmarks")]
MANIFEST = ROOT / "benchmarks" / "direct_baseline_manifest.json"
METHODS = (
    "lazyqc",
    "lazyqc_gpu",
    "qulacs_1t",
    "aer_cpu",
    "aer_gpu",
    "lightning_gpu",
    "quimb",
    "lazyqc_aer_gpu",
    "lazyqc_lightning_gpu",
    "lazyqc_quimb",
)
GPU_METHODS = ("lazyqc_gpu", "aer_gpu", "lightning_gpu", "lazyqc_aer_gpu", "lazyqc_lightning_gpu")


def source_hash():
    """Hash the package, this runner, its support modules and the manifest."""
    digest = hashlib.sha256()
    paths = package_files() + [
        Path(__file__).resolve(),
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
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--qubits", help="comma-separated subset of the manifest widths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    methods = tuple(args.methods.split(","))
    if unknown := set(methods) - set(METHODS):
        parser.error(f"unknown methods {sorted(unknown)}")

    from direct_support import pin_threads

    pin_threads(args.threads)

    import numpy as np
    from direct_support import (
        aer_direct,
        aer_simulator,
        atomic_write,
        environment,
        lightning_device,
        lightning_direct,
        qulacs_direct,
        timed,
    )
    from qml_workload_support import make_batch, numpy_oracle_expectations

    from lazyqc import plan_all

    manifest = json.loads(MANIFEST.read_text())
    config = manifest["scaling"]
    limits = manifest["planner_limits"]
    wide_from = manifest["placement"]["wide_from"]
    qubits = config["qubits"]
    if args.qubits:
        qubits = [int(q) for q in args.qubits.split(",")]
    if args.smoke:
        qubits = qubits[:1]
    families = config["families"]
    patterns = config["patterns"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    rng = random.Random(manifest["seed"] + 4049 + args.threads)

    simulator = aer_simulator("GPU") if set(GPU_METHODS) & set(methods) else None
    cpu_simulator = aer_simulator("CPU") if "aer_cpu" in methods else None
    wide_backend = None
    if {"lazyqc_gpu", "lazyqc_aer_gpu"} & set(methods):
        from lazyqc.backends.aer import AerBackend

        wide_backend = AerBackend(device="GPU")
    if "lazyqc_lightning_gpu" in methods:
        from lazyqc.backends.lightning import LightningBackend
    if {"lazyqc_quimb", "quimb"} & set(methods):
        from direct_support import quimb_direct

        from lazyqc.backends.tensor import QuimbBackend

    def lazyqc_runner(requests, collect_options=None, strategy="auto"):
        options = collect_options or {}

        def run():
            plan, planning = timed(
                lambda: plan_all(
                    requests,
                    strategy=strategy,
                    max_state_qubits=limits["max_state_qubits"],
                    max_state_bytes=limits["max_state_bytes"],
                )
            )
            result, collection = timed(lambda: plan.collect(**options))
            run.detail = {
                "planning_ns": planning,
                "collection_ns": collection,
                "selected_strategy": plan.selected.strategy,
                "tasks": len(plan.selected.tasks),
                "max_width": plan.selected.max_width,
            }
            return list(result.values)

        run.detail = {}
        return run

    def eligible(method, family, n):
        settings = config["methods"][method]
        if n > settings.get("max_qubits", 99):
            return False
        if family == "data_reuploading" and n > settings.get("max_qubits_saturated", 99):
            return False
        return True

    rows, failures = [], []
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_sha256": source_hash(),
        "command": sys.argv,
        "threads": args.threads,
        "methods": methods,
        "environment": environment(),
        "smoke": args.smoke,
    }
    for n in qubits:
        repeats = config["repetitions"][str(n)]
        for family in families:
            for pattern in patterns:
                for offset in offsets:
                    seed = manifest["seed"] + offset
                    requests = make_batch(family, n, manifest["depth"], pattern, seed)
                    program = requests[0].program
                    observables = [r.observable for r in requests]
                    case = {
                        "case_id": f"{family}_{n}q_d{manifest['depth']}_{pattern}_s{offset}",
                        "family": family,
                        "qubits": n,
                        "depth": manifest["depth"],
                        "pattern": pattern,
                        "seed_offset": offset,
                        "outputs": len(observables),
                        "gates": len(program.operations),
                    }
                    # One reference per case: narrow cones on Qulacs, else Aer GPU.
                    probe = plan_all(
                        requests,
                        strategy="grouped",
                        max_state_qubits=28,
                        max_state_bytes=limits["max_state_bytes"],
                    )
                    if probe.selected.max_width <= 20 or simulator is None:
                        reference, reference_name = list(probe.collect().values), "lazyqc_grouped"
                    else:
                        reference = aer_direct(program, observables, simulator)()
                        reference_name = "aer_gpu"
                    reference = np.asarray(reference)
                    oracle_error = None
                    if n <= 12:
                        oracle = np.asarray(numpy_oracle_expectations(requests))
                        oracle_error = float(np.max(np.abs(reference - oracle)))
                    order = [m for m in methods if eligible(m, family, n)]
                    rng.shuffle(order)
                    for method in order:
                        if method == "lazyqc":
                            run = lazyqc_runner(requests)
                        elif method == "lazyqc_gpu":
                            placement = {"wide_backend": wide_backend, "wide_from": wide_from}
                            run = lazyqc_runner(requests, placement)
                        elif method == "lazyqc_aer_gpu":
                            run = lazyqc_runner(requests, {"backend": wide_backend})
                        elif method == "lazyqc_lightning_gpu":
                            lightning = LightningBackend("lightning.gpu")
                            run = lazyqc_runner(requests, {"backend": lightning})
                        elif method == "lazyqc_quimb":
                            # Dense width does not drive contraction cost, so the
                            # tensor-network backend always receives the cone plan.
                            run = lazyqc_runner(requests, {"backend": QuimbBackend()}, "grouped")
                        elif method == "quimb":
                            run = quimb_direct(program, observables)
                        elif method == "qulacs_1t":
                            run = qulacs_direct(program, observables)
                        elif method == "aer_cpu":
                            run = aer_direct(program, observables, cpu_simulator)
                        elif method == "aer_gpu":
                            run = aer_direct(program, observables, simulator)
                        else:
                            device, _ = lightning_device("lightning.gpu", n)
                            run = lightning_direct(program, observables, device)
                        try:
                            run()
                            for repetition in range(repeats):
                                values, elapsed = timed(run)
                                error = float(np.max(np.abs(np.asarray(values) - reference)))
                                row = (
                                    case
                                    | {
                                        "method": method,
                                        "threads": args.threads,
                                        "repetition": repetition,
                                        "total_ns": elapsed,
                                        "max_abs_error": error,
                                        "reference": reference_name,
                                        "reference_oracle_error": oracle_error,
                                    }
                                    | getattr(run, "detail", {})
                                )
                                rows.append(row)
                                if error > 1e-9:
                                    failures.append(
                                        {
                                            "case_id": case["case_id"],
                                            "method": method,
                                            "error": error,
                                        }
                                    )
                        except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                            failures.append(
                                {
                                    "case_id": case["case_id"],
                                    "method": method,
                                    "exception": repr(exc),
                                }
                            )
                        finally:
                            run = None
                            if method == "lightning_gpu":
                                del device
                            if method == "lazyqc_lightning_gpu":
                                lightning.close()
                                lightning = None
                            gc.collect()
                        done = [
                            r
                            for r in rows
                            if r["case_id"] == case["case_id"] and r["method"] == method
                        ]
                        if done:
                            best = min(r["total_ns"] for r in done) / 1e6
                            print(f"{case['case_id']:58s} {method:14s} {best:12.3f} ms", flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write(
                        args.output,
                        json.dumps(
                            {"metadata": metadata, "failures": failures, "rows": rows}, indent=1
                        ),
                    )
    print(f"rows {len(rows)} failures {len(failures)}")


if __name__ == "__main__":
    main()
