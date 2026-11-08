#!/usr/bin/env python3
"""Planner beyond device memory, compared with exact MPS and quimb light cones.

Direct dense execution needs 16 * 2**n bytes, which no CPU or GPU here can hold.
The planner either finds narrow/intermediate cone plans or refuses with a structured reason.
We also time a global exact MPS baseline (quimb CircuitMPS with cutoff=0.0, shared across
all requested outputs of the batch) and a reused-circuit quimb light-cone contraction baseline,
plus intermediate-cone cases at 48 qubits (depths 6 and 8) with CPU and GPU placement.

    uv run --locked --all-extras python benchmarks/run_memory_wall.py \
        --output results/direct/memory_wall.json
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
MANIFEST = ROOT / "benchmarks" / "direct_baseline_manifest.json"


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


def quimb_expectations(requests):
    """Reused-circuit light-cone tensor-network values for single-program batches."""
    import quimb as qu
    import quimb.tensor as qtn

    program = requests[0].program
    circuit = qtn.Circuit(program.n_qubits)
    for gate in program.resolved_operations():
        if gate.name in ("RX", "RY", "RZ"):
            # quimb's rotations are exp(-i theta P / 2), the LazyQC convention.
            circuit.apply_gate(gate.name, gate.angle, *gate.qubits)
        else:
            circuit.apply_gate(gate.name, *gate.qubits)
    values = []
    for request in requests:
        factors = request.observable.factors
        operator = qu.kron(*(qu.pauli(p) for _, p in factors))
        where = tuple(q for q, _ in factors)
        values.append(float(circuit.local_expectation(operator, where, optimize="greedy").real))
    return values


def mps_expectations(requests):
    """Global exact MPS (cutoff=0.0) prepared once per batch and queried for all outputs."""
    import quimb as qu
    import quimb.tensor as qtn

    program = requests[0].program
    mps = qtn.CircuitMPS(program.n_qubits, cutoff=0.0)
    for gate in program.resolved_operations():
        if gate.name in ("RX", "RY", "RZ"):
            mps.apply_gate(gate.name, gate.angle, *gate.qubits)
        else:
            mps.apply_gate(gate.name, *gate.qubits)
    values = []
    for request in requests:
        factors = request.observable.factors
        operator = qu.kron(*(qu.pauli(p) for _, p in factors))
        where = tuple(q for q, _ in factors)
        values.append(float(mps.local_expectation(operator, where).real))
    return values, int(mps.psi.max_bond())


def aer_gpu_backend():
    """Return Aer GPU only when an actual simulator run confirms device support."""
    try:
        from qiskit import QuantumCircuit
        from lazyqc.backends.aer import AerBackend

        backend = AerBackend(device="GPU")
    except ImportError as exc:
        print(f"Skipping Aer GPU cases: {exc}", flush=True)
        return None

    try:
        backend.simulator.run(QuantumCircuit(1)).result()
    except RuntimeError as exc:
        if "not supported on this system" not in str(exc):
            raise
        print(f"Skipping Aer GPU cases: {exc}", flush=True)
        return None
    return backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from direct_support import atomic_write, environment, pin_threads, timed

    pin_threads(1)
    import numpy as np
    from qml_workload_support import make_batch

    from lazyqc import NoFeasiblePlan, plan_all

    wide_backend = aer_gpu_backend()

    manifest = json.loads(MANIFEST.read_text())
    config = manifest["memory_wall"]
    limits = manifest["planner_limits"]
    wide_from = manifest["placement"]["wide_from"]
    qubits = config["qubits"][:1] if args.smoke else config["qubits"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    repeats = 1 if args.smoke else config["repetitions"]
    mps_repeats = 1 if args.smoke else config.get("mps_repetitions", 3)
    rng = random.Random(manifest["seed"] + 7331)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_sha256": source_hash(),
        "command": sys.argv,
        "environment": environment(),
        "smoke": args.smoke,
    }
    rows, mps_rows, tn_rows, extra_rows, cases, failures = [], [], [], [], [], []

    def plan_and_collect(requests, collect_options=None):
        options = collect_options or {}
        plan, planning = timed(
            lambda: plan_all(
                requests,
                max_state_qubits=limits["max_state_qubits"],
                max_state_bytes=limits["max_state_bytes"],
            )
        )
        result, collection = timed(lambda: plan.collect(**options))
        return plan, result, planning, collection

    jobs = [
        (family, n, manifest["depth"], pattern, offset, False)
        for family in config["families"]
        for n in qubits
        for pattern in config["patterns"]
        for offset in offsets
    ]
    extra_configs = config.get("extra_cases", [])
    if args.smoke:
        extra_configs = extra_configs[:1]
    for extra in extra_configs:
        for offset in offsets:
            jobs.append(
                (extra["family"], extra["qubits"], extra["depth"], extra["pattern"], offset, True)
            )
    rng.shuffle(jobs)
    for family, n, depth, pattern, offset, is_extra in jobs:
        seed = manifest["seed"] + offset
        requests = make_batch(family, n, depth, pattern, seed)
        case = {
            "case_id": f"{family}_{n}q_d{depth}_{pattern}_s{offset}",
            "family": family,
            "qubits": n,
            "depth": depth,
            "pattern": pattern,
            "seed_offset": offset,
            "outputs": len(requests),
            "gates": len(requests[0].program.operations),
            "dense_state_bytes": 16 * 2**n,
            "extra": is_extra,
        }

        # Global exact MPS (cutoff=0.0)
        mps_vals, max_bond = None, None
        actual_mps_repeats = mps_repeats
        for rep in range(actual_mps_repeats):
            (mps_vals, max_bond), elapsed_mps = timed(lambda: mps_expectations(requests))
            mps_rows.append(
                case
                | {
                    "repetition": rep,
                    "total_ns": elapsed_mps,
                    "max_bond": max_bond,
                }
            )
        mps_reference = np.asarray(mps_vals)

        try:
            plan, result, _, _ = plan_and_collect(requests)
        except NoFeasiblePlan as exc:
            _, refusal = timed(lambda: _refuse(plan_all, requests, limits, NoFeasiblePlan))
            # Validate MPS against quimb local_expectation on the first 3 outputs
            sample_tn = np.asarray(quimb_expectations(requests[:3]))
            mps_error = float(np.max(np.abs(mps_reference[:3] - sample_tn)))
            if mps_error > 1e-9:
                failures.append({"case_id": case["case_id"], "method": "mps", "error": mps_error})
            cases.append(
                case
                | {
                    "status": "refused",
                    "refusal_ns": refusal,
                    "reason": str(exc),
                    "mps_max_bond": max_bond,
                    "mps_max_abs_error_vs_quimb": mps_error,
                }
            )
            best_mps = min(r["total_ns"] for r in mps_rows if r["case_id"] == case["case_id"]) / 1e6
            print(
                f"{case['case_id']:58s} refused in {refusal / 1e6:.2f} ms "
                f"| MPS {best_mps:8.2f} ms (bond {max_bond})",
                flush=True,
            )
            continue

        # Planned case: run quimb light-cone reference & timing
        tn_values, tn_first_ns = timed(lambda: quimb_expectations(requests))
        reference = np.asarray(tn_values)
        tn_rows.append(case | {"repetition": 0, "total_ns": tn_first_ns})
        if not is_extra and pattern == "same_state_local_outputs":
            for rep in range(1, mps_repeats):
                _, elapsed_tn = timed(lambda: quimb_expectations(requests))
                tn_rows.append(case | {"repetition": rep, "total_ns": elapsed_tn})

        mps_error = float(np.max(np.abs(mps_reference - reference)))
        if mps_error > 1e-9:
            failures.append({"case_id": case["case_id"], "method": "mps", "error": mps_error})
        error = float(np.max(np.abs(np.asarray(result.values) - reference)))
        if error > 1e-9:
            failures.append({"case_id": case["case_id"], "error": error})

        target_rows = extra_rows if is_extra else rows
        for repetition in range(repeats):
            plan, result, planning, collection = plan_and_collect(requests)
            widths = [t.width for t in plan.selected.tasks]
            target_rows.append(
                case
                | {
                    "method": "lazyqc",
                    "repetition": repetition,
                    "planning_ns": planning,
                    "collection_ns": collection,
                    "total_ns": planning + collection,
                    "tasks": len(plan.selected.tasks),
                    "min_width": min(widths),
                    "median_width": sorted(widths)[len(widths) // 2],
                    "max_width": plan.selected.max_width,
                    "task_state_bytes": plan.selected.state_bytes,
                    "selected_strategy": plan.selected.strategy,
                }
            )
        if is_extra and wide_backend is not None:
            placement = {"wide_backend": wide_backend, "wide_from": wide_from}
            plan_and_collect(requests, placement)  # warm up GPU backend
            for repetition in range(repeats):
                plan_g, res_g, plan_ns_g, col_ns_g = plan_and_collect(requests, placement)
                err_g = float(np.max(np.abs(np.asarray(res_g.values) - reference)))
                if err_g > 1e-9:
                    failures.append(
                        {"case_id": case["case_id"], "method": "lazyqc_gpu", "error": err_g}
                    )
                widths = [t.width for t in plan_g.selected.tasks]
                extra_rows.append(
                    case
                    | {
                        "method": "lazyqc_gpu",
                        "repetition": repetition,
                        "planning_ns": plan_ns_g,
                        "collection_ns": col_ns_g,
                        "total_ns": plan_ns_g + col_ns_g,
                        "tasks": len(plan_g.selected.tasks),
                        "min_width": min(widths),
                        "median_width": sorted(widths)[len(widths) // 2],
                        "max_width": plan_g.selected.max_width,
                        "task_state_bytes": plan_g.selected.state_bytes,
                        "selected_strategy": plan_g.selected.strategy,
                    }
                )

        cases.append(
            case
            | {
                "status": "planned",
                "max_abs_error_vs_quimb": error,
                "mps_max_bond": max_bond,
                "mps_max_abs_error_vs_quimb": mps_error,
            }
        )
        best = (
            min(
                r["total_ns"]
                for r in target_rows
                if r["case_id"] == case["case_id"] and r.get("method", "lazyqc") == "lazyqc"
            )
            / 1e6
        )
        best_mps = min(r["total_ns"] for r in mps_rows if r["case_id"] == case["case_id"]) / 1e6
        print(
            f"{case['case_id']:58s} {best:9.2f} ms width {plan.selected.max_width:2d} "
            f"tasks {len(plan.selected.tasks):3d} | MPS {best_mps:8.2f} ms (bond {max_bond}) "
            f"err {error:.1e}",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(
            args.output,
            json.dumps(
                {
                    "metadata": metadata,
                    "failures": failures,
                    "cases": cases,
                    "rows": rows,
                    "mps_rows": mps_rows,
                    "tn_rows": tn_rows,
                    "extra_rows": extra_rows,
                },
                indent=1,
            ),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        args.output,
        json.dumps(
            {
                "metadata": metadata,
                "failures": failures,
                "cases": cases,
                "rows": rows,
                "mps_rows": mps_rows,
                "tn_rows": tn_rows,
                "extra_rows": extra_rows,
            },
            indent=1,
        ),
    )
    print(
        f"cases {len(cases)} rows {len(rows)} mps_rows {len(mps_rows)} "
        f"extra_rows {len(extra_rows)} failures {len(failures)}"
    )


def _refuse(plan_all, requests, limits, error):
    """Time the planner's structured refusal."""
    try:
        plan_all(
            requests,
            max_state_qubits=limits["max_state_qubits"],
            max_state_bytes=limits["max_state_bytes"],
        )
    except error:
        return None
    raise AssertionError("expected a refusal")


if __name__ == "__main__":
    main()
