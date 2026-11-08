#!/usr/bin/env python3
"""Train a small binary Iris QML classifier under LazyQC and direct Qulacs."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from benchmark_support import cpu_name, pin_single_thread, source_hash

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "trained_iris_qml_manifest.json"


def build_program(features, angles, n, depth, ansatz="ring"):
    """Angle-encode features, then `depth` trainable RY layers with an entangler.

    ``ring`` applies a CX ring every layer; ``brickwork`` applies one
    alternating nearest-neighbour CX sublayer, so output cones stay local.
    """
    from lazyqc import Gate, Program

    encoded = np.resize(features, n)
    operations = [Gate("RY", (q,), float(encoded[q])) for q in range(n)]
    trainable = []
    parameter = 0
    for layer in range(depth):
        for q in range(n):
            trainable.append(len(operations))
            operations.append(Gate("RY", (q,), float(angles[parameter])))
            parameter += 1
        if ansatz == "ring":
            edges = [(q, (q + 1) % n) for q in range(n)]
        elif ansatz == "brickwork":
            edges = [(q, q + 1) for q in range(layer % 2, n - 1, 2)]
        else:
            raise ValueError(f"Unknown Iris ansatz: {ansatz}")
        operations.extend(Gate("CX", edge) for edge in edges)
    return Program(n, operations), tuple(trainable)


def shifted_variants(program, indices):
    from lazyqc import Gate, Program

    base = list(program.operations)
    variants = [program]
    for index in indices:
        gate = base[index]
        for sign in (1, -1):
            operations = list(base)
            operations[index] = Gate(gate.name, gate.qubits, gate.angle + sign * np.pi / 2)
            variants.append(Program(program.n_qubits, operations))
    return variants


def assemble(values, batch_size, parameters, outputs):
    table = np.asarray(values, dtype=float).reshape(batch_size, 1 + 2 * parameters, outputs)
    forward = table[:, 0, :]
    jacobian = 0.5 * (table[:, 1::2, :] - table[:, 2::2, :]).transpose(0, 2, 1)
    return forward, jacobian


def sigmoid(x):
    x = np.clip(x, -40, 40)
    return 1 / (1 + np.exp(-x))


def model_loss(logits, labels):
    return float(np.mean(np.logaddexp(0, logits) - labels * logits))


def dataset(split_seed):
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    mask = iris.target < 2
    features = iris.data[mask].astype(np.float64)
    labels = iris.target[mask].astype(np.float64)
    train_i, test_i = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=split_seed, stratify=labels
    )
    mean = features[train_i].mean(axis=0)
    scale = features[train_i].std(axis=0)
    scale[scale == 0] = 1
    transformed = np.pi * np.tanh((features - mean) / scale / 2)
    return transformed[train_i], labels[train_i], transformed[test_i], labels[test_i]


def make_requests(features, angles, n, depth, observables, ansatz="ring"):
    from lazyqc import Program

    programs, all_variants = [], []
    indices = None
    for row in features:
        program, indices = build_program(row, angles, n, depth, ansatz)
        variants = shifted_variants(program, indices)
        programs.append(program)
        all_variants.extend(variants)
    requests = [program.expectation(o) for program in all_variants for o in observables]
    return programs, indices, all_variants, requests


def evaluate_direct(features, angles, weights, bias, n, depth, observables, ansatz="ring"):
    from direct_support import qulacs_direct

    values = []
    programs, indices, variants, _ = make_requests(
        features, angles, n, depth, observables, ansatz
    )
    runners = [qulacs_direct(program, observables) for program in variants]
    for runner in runners:
        values.extend(runner())
    forward, _ = assemble(values, len(features), len(indices), len(observables))
    logits = forward @ weights + bias
    return forward, logits


def write(path, metadata, steps, models, failures):
    payload = {"metadata": metadata, "steps": steps, "models": models, "failures": failures}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="one 4-qubit, one-seed trajectory")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest_bytes = manifest_path.read_bytes()
    config = json.loads(manifest_bytes)
    pin_single_thread()

    from lazyqc import Pauli, plan_all
    from lazyqc.backends.qulacs import QulacsBackend
    from qml_workload_support import numpy_oracle_expectations
    from direct_support import qulacs_direct

    # Warm native libraries and establish the run's reproducibility metadata.
    warm, _ = build_program(np.zeros(4), np.zeros(4), 4, 1)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect(
        backend=QulacsBackend()
    )
    split_seeds = config.get("split_seeds", [config.get("split_seed")])
    ansatze = config.get("ansatze", ["ring"])
    depths = config.get("ansatz_depths", [config.get("ansatz_depth")])
    datasets = {split_seed: dataset(split_seed) for split_seed in split_seeds}
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory trained-model screen; not confirmatory",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_hash(__file__),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "python": sys.version,
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("lazyqc", "qulacs", "numpy", "scikit-learn")
        },
        "command": sys.argv,
        "split_seeds": split_seeds,
        "splits": {
            str(split_seed): {
                "train_examples": len(train_y),
                "test_examples": len(test_y),
                "train_class_counts": np.bincount(train_y.astype(int)).tolist(),
                "test_class_counts": np.bincount(test_y.astype(int)).tolist(),
            }
            for split_seed, (_, train_y, _, test_y) in datasets.items()
        },
        "backend": "QulacsBackend and direct Qulacs, single native thread",
        "smoke": args.smoke,
    }
    default_name = manifest_path.stem.removesuffix("_manifest")
    output = args.output or ROOT / "results/qml_workloads" / f"{default_name}.json"
    widths = config["qubits"][:1] if args.smoke else config["qubits"]
    offsets = config["seed_offsets"][:1] if args.smoke else config["seed_offsets"]
    if args.smoke:
        split_seeds, ansatze, depths = split_seeds[:1], ansatze[:1], depths[:1]
    methods = config["strategies"]
    steps_per_run = 2 if args.smoke else config["training_steps"]
    batch_size = config["train_batch_size"]
    rng_order = random.Random(config["randomized_strategy_order_seed"])
    steps, models, failures = [], [], []

    for split_seed, ansatz, depth, n in product(split_seeds, ansatze, depths, widths):
        train_x, train_y, test_x, test_y = datasets[split_seed]
        observables = [
            Pauli(((q, pauli),))
            for pauli in config.get("readout_paulis", ["Z"])
            for q in range(n)
        ]
        for offset in offsets:
            seed = config["base_seed"] + offset
            init_rng = np.random.default_rng(seed + n)
            initial = {
                "angles": init_rng.uniform(-0.3, 0.3, n * depth),
                "weights": init_rng.normal(0, 0.05, len(observables)),
                "bias": 0.0,
            }
            shuffle_rng = np.random.default_rng(seed + 97531)
            batches = []
            while len(batches) < steps_per_run:
                permutation = shuffle_rng.permutation(len(train_y))
                batches.extend(
                    permutation[i : i + batch_size]
                    for i in range(0, len(permutation), batch_size)
                )
            batches = batches[:steps_per_run]
            states = {
                method: {
                    "angles": initial["angles"].copy(),
                    "weights": initial["weights"].copy(),
                    "bias": initial["bias"],
                }
                for method in methods
            }
            case_id = f"iris_binary_{n}q_d{depth}_s{offset}"
            if len(split_seeds) > 1 or ansatze != ["ring"]:
                case_id = f"iris_binary_{ansatz}_{n}q_d{depth}_split{split_seed}_s{offset}"
            for step_index, batch_indices in enumerate(batches):
                batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
                order = list(methods)
                rng_order.shuffle(order)
                same_step_outputs = {}
                case_records = []
                for method in order:
                    state = states[method]
                    build_start = perf_counter_ns()
                    _, parameter_indices, variants, requests = make_requests(
                        batch_x, state["angles"], n, depth, observables, ansatz
                    )
                    direct_runners = (
                        [qulacs_direct(program, observables) for program in variants]
                        if method == "direct_qulacs"
                        else None
                    )
                    build_ns = perf_counter_ns() - build_start
                    plan_ns = 0
                    if method == "direct_qulacs":
                        collection_start = perf_counter_ns()
                        values = [v for runner in direct_runners for v in runner()]
                        collection_ns = perf_counter_ns() - collection_start
                        selected = "direct_qulacs"
                        evolutions = len(variants)
                        task_count = len(variants)
                        task_widths = [n] * task_count
                    else:
                        plan_start = perf_counter_ns()
                        plan = plan_all(
                            requests,
                            strategy=method,
                            optimization="none",
                            max_state_qubits=config["max_state_qubits"],
                            max_state_bytes=config["max_state_bytes"],
                        )
                        plan_ns = perf_counter_ns() - plan_start
                        collection_start = perf_counter_ns()
                        result = plan.collect(backend=QulacsBackend())
                        collection_ns = perf_counter_ns() - collection_start
                        values = list(result.values)
                        selected = result.selected_strategy
                        evolutions = result.stats.evolutions
                        task_count = len(plan.selected.tasks)
                        task_widths = [task.width for task in plan.selected.tasks]
                    same_step_outputs[method] = np.asarray(values)
                    forward, jacobian = assemble(
                        values, len(batch_x), len(parameter_indices), len(observables)
                    )
                    logits = forward @ state["weights"] + state["bias"]
                    probability = sigmoid(logits)
                    loss = model_loss(logits, batch_y)
                    residual = (probability - batch_y) / len(batch_y)
                    grad_weights = forward.T @ residual
                    grad_bias = float(np.sum(residual))
                    dlogit = np.einsum("boi,o->bi", jacobian, state["weights"])
                    grad_angles = np.einsum("b,bi->i", residual, dlogit)
                    update_start = perf_counter_ns()
                    state["angles"] -= config["learning_rate"] * grad_angles
                    state["weights"] -= config["learning_rate"] * grad_weights
                    state["bias"] -= config["learning_rate"] * grad_bias
                    update_ns = perf_counter_ns() - update_start
                    row = {
                        "case_id": case_id,
                        "split_seed": split_seed,
                        "ansatz": ansatz,
                        "qubits": n,
                        "depth": depth,
                        "seed_offset": offset,
                        "step": step_index,
                        "batch_indices": batch_indices.tolist(),
                        "method": method,
                        "selected_strategy": selected,
                        "loss_before_update": loss,
                        "build_ns": build_ns,
                        "planning_ns": plan_ns,
                        "collection_ns": collection_ns,
                        "classical_update_ns": update_ns,
                        "step_lifetime_ns": build_ns + plan_ns + collection_ns + update_ns,
                        "parameters": len(parameter_indices),
                        "request_count": len(requests),
                        "task_count": task_count,
                        "task_widths": task_widths,
                        "evolutions": evolutions,
                    }
                    case_records.append(row)

                # On the initial state, check the whole shift batch against an
                # independent statevector oracle. Later methods must agree with
                # direct Qulacs to numerical precision on each common step.
                reference_method = "direct_qulacs"
                reference_values = same_step_outputs[reference_method]
                if step_index == 0:
                    oracle = np.asarray(numpy_oracle_expectations(requests))
                    for method, values in same_step_outputs.items():
                        err = float(np.max(np.abs(values - oracle)))
                        if err > config["oracle_tolerance"]:
                            failures.append(
                                {"case_id": case_id, "step": step_index,
                                 "method": method, "stage": "oracle",
                                 "max_abs_error": err}
                            )
                for method, values in same_step_outputs.items():
                    err = float(np.max(np.abs(values - reference_values)))
                    if err > config["oracle_tolerance"]:
                        failures.append(
                            {"case_id": case_id, "step": step_index,
                             "method": method, "stage": "direct-agreement",
                             "max_abs_error": err}
                        )
                steps.extend(case_records)
                write(output, metadata, steps, models, failures)
                print(f"{case_id} step {step_index + 1}/{steps_per_run} complete", flush=True)

            for method, state in states.items():
                final_forward, final_logits = evaluate_direct(
                    test_x, state["angles"], state["weights"], state["bias"],
                    n, depth, observables, ansatz
                )
                predictions = (final_logits >= 0).astype(int)
                models.append(
                    {
                        "case_id": case_id,
                        "split_seed": split_seed,
                        "ansatz": ansatz,
                        "qubits": n,
                        "depth": depth,
                        "seed_offset": offset,
                        "method": method,
                        "test_loss": model_loss(final_logits, test_y),
                        "test_accuracy": float(np.mean(predictions == test_y)),
                        "train_loss_final_batch": next(
                            row["loss_before_update"]
                            for row in reversed(steps)
                            if row["case_id"] == case_id and row["method"] == method
                        ),
                        "final_angles": state["angles"].tolist(),
                        "final_head_weights": state["weights"].tolist(),
                        "final_head_bias": state["bias"],
                    }
                )
            write(output, metadata, steps, models, failures)

    metadata["case_count"] = len({x["case_id"] for x in models})
    metadata["steps_per_case"] = steps_per_run
    write(output, metadata, steps, models, failures)
    print(
        f"cases={metadata['case_count']} step_rows={len(steps)} "
        f"models={len(models)} failures={len(failures)} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} numerical failures; inspect artifact")


if __name__ == "__main__":
    main()
