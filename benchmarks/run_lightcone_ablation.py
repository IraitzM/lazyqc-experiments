#!/usr/bin/env python3
"""Qiskit LightCone + Aer versus structure-keyed LazyQC batch planning.

Run with the same optional simulator stack as the direct study, for example:

    uv run --with qiskit-aer-gpu==0.18.0 --with qiskit==2.6.1 \
      --with qulacs==0.6.14 --with quimb==1.15.0 --with numpy==2.5.4 \
      python benchmarks/run_lightcone_ablation.py \
      --output results/direct/lightcone_ablation.json
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
MANIFEST = ROOT / "benchmarks" / "lightcone_ablation_manifest.json"


def source_hash():
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


def qiskit_circuit(program):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(program.n_qubits)
    for gate in program.resolved_operations():
        if gate.name == "CX":
            circuit.cx(*gate.qubits)
        else:
            getattr(circuit, gate.name.lower())(gate.angle, gate.qubits[0])
    return circuit


def compact_circuit(circuit):
    """Compact the wires used by a LightCone result and return a circuit signature."""
    used = sorted({circuit.find_bit(q).index for op in circuit.data for q in op.qubits})
    mapping = {original: local for local, original in enumerate(used)}
    operations = []
    for instruction in circuit.data:
        wires = tuple(mapping[circuit.find_bit(q).index] for q in instruction.qubits)
        operations.append((instruction.operation, wires))
    signature = tuple(
        (operation.name, tuple(float(p) for p in operation.params), wires)
        for operation, wires in operations
    )
    return len(used), operations, signature, mapping


def lightcone_batch_tasks(jobs, deduplicate):
    """Reduce ``(DAG, observed wire, global output index)`` jobs for Aer."""
    from qiskit import QuantumCircuit
    from qiskit.converters import dag_to_circuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.transpiler.passes import LightCone

    groups = {}
    for dag, observed_wire, output in jobs:
        reduced = dag_to_circuit(LightCone(bit_terms="Z", indices=[observed_wire]).run(dag))
        width, operations, signature, mapping = compact_circuit(reduced)
        key = signature if deduplicate else (output, signature)
        if key not in groups:
            groups[key] = [width, operations, mapping, []]
        groups[key][3].append((output, mapping[observed_wire]))

    tasks = []
    for width, operations, _, group_outputs in groups.values():
        circuit = QuantumCircuit(width)
        for operation, wires in operations:
            circuit.append(operation, wires)
        for output, local in group_outputs:
            operator = SparsePauliOp.from_sparse_list([("Z", [local], 1.0)], width)
            circuit.save_expectation_value(operator, range(width), label=f"o{output}")
        tasks.append((circuit, tuple(group_outputs)))
    return tasks


def lightcone_tasks(dag, outputs, deduplicate):
    """Apply one LightCone pass per output and prepare Aer circuits."""
    return lightcone_batch_tasks([(dag, output, output) for output in outputs], deduplicate)


def execute_aer_tasks(simulator, tasks, output_count):
    """Execute all reduced circuits as one Aer job and restore request order."""
    from direct_support import timed

    result, execution_ns = timed(lambda: simulator.run([task[0] for task in tasks]).result())
    values = [None] * output_count
    for index, (_, group_outputs) in enumerate(tasks):
        data = result.data(index)
        for output, _ in group_outputs:
            values[output] = float(complex(data[f"o{output}"]).real)
    return values, execution_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from direct_support import (
        aer_simulator,
        atomic_write,
        environment,
        pin_threads,
        timed,
    )

    pin_threads(1)
    import numpy as np
    from qiskit.converters import circuit_to_dag
    from qml_workload_support import make_batch

    from lazyqc import plan_all

    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    widths = manifest["qubits"][:1] if args.smoke else manifest["qubits"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    repetitions = 1 if args.smoke else manifest["repetitions"]
    limits = {"max_state_qubits": 28, "max_state_bytes": 4 * 2**30}
    simulator = aer_simulator("GPU")
    rng = random.Random(manifest["seed"] + 1907)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_hash(),
        "command": sys.argv,
        "environment": environment(),
        "smoke": args.smoke,
    }
    rows, cases, failures = [], [], []

    def lazyqc_run(requests):
        plan, planning_ns = timed(lambda: plan_all(requests, **limits))
        result, collection_ns = timed(plan.collect)
        lazyqc_run.detail = {
            "analysis_ns": planning_ns,
            "execution_ns": collection_ns,
            "tasks": len(plan.selected.tasks),
            "min_width": min(t.width for t in plan.selected.tasks),
            "max_width": max(t.width for t in plan.selected.tasks),
            "selected_strategy": plan.selected.strategy,
        }
        return list(result.values)

    def aer_run(dag, outputs, deduplicate):
        tasks, analysis_ns = timed(lambda: lightcone_tasks(dag, outputs, deduplicate))
        circuits = [task[0] for task in tasks]
        values, execution_ns = execute_aer_tasks(simulator, tasks, len(outputs))
        aer_run.detail = {
            "analysis_ns": analysis_ns,
            "execution_ns": execution_ns,
            "tasks": len(tasks),
            "min_width": min(c.num_qubits for c in circuits),
            "max_width": max(c.num_qubits for c in circuits),
            "selected_strategy": "lightcone_dedup" if deduplicate else "lightcone_separate",
        }
        return values

    for n in widths:
        for offset in offsets:
            seed = manifest["seed"] + offset
            requests = make_batch(
                manifest["family"], n, manifest["depth"], manifest["pattern"], seed
            )
            program = requests[0].program
            outputs = list(range(len(requests)))
            dag = circuit_to_dag(qiskit_circuit(program))
            reference = np.asarray(lazyqc_run(requests))
            case = {
                "case_id": f"hardware_efficient_local_{n}q_d4_Z_all_s{offset}",
                "family": manifest["family"],
                "qubits": n,
                "depth": manifest["depth"],
                "seed_offset": offset,
                "outputs": len(outputs),
                "reference_values": reference.tolist(),
            }
            methods = [
                ("lightcone_separate_aer", lambda: aer_run(dag, outputs, False)),
                ("lightcone_dedup_aer", lambda: aer_run(dag, outputs, True)),
                ("lazyqc", lambda: lazyqc_run(requests)),
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
            cases.append({k: v for k, v in case.items() if k != "reference_values"})
            best = {
                method: min(
                    row["total_ns"]
                    for row in rows
                    if row["case_id"] == case["case_id"] and row["method"] == method
                )
                / 1e6
                for method, _ in methods
                if any(
                    row["case_id"] == case["case_id"] and row["method"] == method for row in rows
                )
            }
            print(f"{case['case_id']:52s} {best}", flush=True)
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
        raise SystemExit(f"{len(failures)} LightCone-ablation failures")


if __name__ == "__main__":
    main()
