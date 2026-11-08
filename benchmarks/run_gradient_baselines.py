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
# ]
# ///
"""Parameter-shift Jacobians and training steps: planner versus direct CPU and GPU.

The planner evaluates the base program and every +-pi/2 occurrence shift as one
batch. A shift outside an output's cone leaves that output's cone circuit
unchanged, so grouping evaluates it once and the derivative is exactly zero.
Direct baselines are a GPU parameter-shift batch and Lightning adjoint
differentiation on GPU and CPU.

    uv run --script benchmarks/run_gradient_baselines.py --study jacobian --threads 1 \
        --methods lazyqc,lazyqc_gpu --output results/direct/jacobian_1t.json
"""

import argparse
import gc
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "benchmarks")]
MANIFEST = ROOT / "benchmarks" / "direct_baseline_manifest.json"
JACOBIAN_METHODS = (
    "lazyqc",
    "lazyqc_gpu",
    "aer_gpu_shift",
    "lightning_gpu_adjoint",
    "lightning_cpu_adjoint",
)
TRAINING_METHODS = ("lazyqc", "lightning_gpu_vjp", "aer_gpu_shift")
OBSERVABLE_CHUNK = 8


def structural_zeros(program, observables, indices):
    """Count (output, parameter) pairs whose parameter lies outside the output's cone."""
    from lazyqc.passes import cone_indices

    qubits = tuple(g.qubits for g in program.operations)
    trainable, zeros = set(indices), 0
    for pauli in observables:
        _, retained = cone_indices(qubits, (q for q, _ in pauli.factors))
        zeros += len(trainable - set(retained))
    return zeros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("jacobian", "training"), required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--qubits", help="comma-separated subset of the manifest widths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    methods = tuple(args.methods.split(","))
    allowed = JACOBIAN_METHODS if args.study == "jacobian" else TRAINING_METHODS
    if unknown := set(methods) - set(allowed):
        parser.error(f"unknown methods {sorted(unknown)}")

    from direct_support import pin_threads

    pin_threads(args.threads)
    manifest = json.loads(MANIFEST.read_text())
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "command": sys.argv,
        "threads": args.threads,
        "methods": methods,
        "study": args.study,
        "smoke": args.smoke,
    }
    from direct_support import environment

    metadata["environment"] = environment()
    study = jacobian_study if args.study == "jacobian" else training_study
    study(manifest, methods, args, metadata)


def jacobian_study(manifest, methods, args, metadata):
    """Time one full Jacobian per repetition for every method and case."""
    import numpy as np
    from direct_support import (
        aer_batch,
        aer_simulator,
        assemble_jacobian,
        jacobian_requests,
        lightning_device,
        pennylane_tape,
        shifted_programs,
        timed,
        trainable_indices,
    )
    from qml_workload_support import make_batch

    from lazyqc import Pauli, plan_all

    config = manifest["jacobian"]
    limits = manifest["planner_limits"]
    wide_from = manifest["placement"]["wide_from"]
    qubits = [int(q) for q in args.qubits.split(",")] if args.qubits else config["qubits"]
    if args.smoke:
        qubits = qubits[:1]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    rng = random.Random(manifest["seed"] + 9157 + args.threads)
    gpu = aer_simulator("GPU") if {"lazyqc_gpu", "aer_gpu_shift"} & set(methods) else None
    wide_backend = None
    if "lazyqc_gpu" in methods:
        from lazyqc.backends.aer import AerBackend

        wide_backend = AerBackend(device="GPU")
    rows, failures, jacobians = [], [], {}
    for n in qubits:
        repeats = config["repetitions"][str(n)]
        for family in config["families"]:
            if family == "data_reuploading" and n > config["max_qubits_saturated"]:
                continue
            for offset in offsets:
                seed = manifest["seed"] + offset
                base = make_batch(family, n, manifest["depth"], "same_state_local_outputs", seed)
                program = base[0].program
                observables = [Pauli(((q, "Z"),)) for q in range(n)]
                indices = trainable_indices(program)
                programs = shifted_programs(program, indices)
                requests = jacobian_requests(programs, observables)
                case = {
                    "case_id": f"{family}_{n}q_d{manifest['depth']}_jacobian_s{offset}",
                    "family": family,
                    "qubits": n,
                    "seed_offset": offset,
                    "outputs": n,
                    "parameters": len(indices),
                    "programs": len(programs),
                    "requests": len(requests),
                    "structural_zeros": structural_zeros(program, observables, indices),
                }
                order = [m for m in methods if n <= config["methods"][m].get("max_qubits", 99)]
                rng.shuffle(order)
                for method in order:
                    detail, device = {}, None
                    if method in ("lazyqc", "lazyqc_gpu"):

                        def run(placed=method == "lazyqc_gpu"):
                            plan, planning = timed(
                                lambda: plan_all(
                                    requests,
                                    max_state_qubits=limits["max_state_qubits"],
                                    max_state_bytes=limits["max_state_bytes"],
                                )
                            )
                            options = {"wide_backend": wide_backend, "wide_from": wide_from}
                            result, collection = timed(
                                lambda: plan.collect(**options) if placed else plan.collect()
                            )
                            detail.update(
                                planning_ns=planning,
                                collection_ns=collection,
                                tasks=len(plan.selected.tasks),
                                max_width=plan.selected.max_width,
                                selected_strategy=plan.selected.strategy,
                            )
                            return assemble_jacobian(result.values, len(indices), n)

                    elif method == "aer_gpu_shift":
                        batch = aer_batch(programs, observables, gpu)

                        def run(batch=batch):
                            return assemble_jacobian(batch(), len(indices), n)

                    else:
                        name = "lightning.gpu" if "gpu" in method else "lightning.qubit"
                        device, config_ = lightning_device(name, n)
                        # The adjoint method keeps one state per observable; when the whole
                        # batch does not fit in device memory, observables run in chunks.
                        tapes = [pennylane_tape(program, observables, trainable=indices)]

                        def run(device=device, tapes=tapes, config_=config_):
                            values, rows_ = [], []
                            for part in tapes:
                                v, j = device.execute_and_compute_derivatives(part, config_)
                                values.extend(np.atleast_1d(v))
                                rows_.extend(np.atleast_2d(j))
                            return np.asarray(values), np.asarray(rows_)

                    try:
                        try:
                            forward, jacobian = run()
                        except RuntimeError as exc:
                            if method.startswith("lightning") and "out of memory" in str(exc):
                                failures.append(
                                    {
                                        "case_id": case["case_id"],
                                        "method": method,
                                        "note": "observable batch out of memory; "
                                        f"retried in chunks of {OBSERVABLE_CHUNK}",
                                    }
                                )
                                tapes[:] = [
                                    pennylane_tape(
                                        program,
                                        observables[i : i + OBSERVABLE_CHUNK],
                                        trainable=indices,
                                    )
                                    for i in range(0, len(observables), OBSERVABLE_CHUNK)
                                ]
                                detail["observable_chunk"] = OBSERVABLE_CHUNK
                                gc.collect()
                                forward, jacobian = run()
                            else:
                                raise
                        jacobians[f"{case['case_id']}|{method}"] = {
                            "forward": np.asarray(forward).tolist(),
                            "jacobian": np.asarray(jacobian).tolist(),
                        }
                        for repetition in range(repeats):
                            _, elapsed = timed(run)
                            rows.append(
                                case
                                | {
                                    "method": method,
                                    "threads": args.threads,
                                    "repetition": repetition,
                                    "total_ns": elapsed,
                                }
                                | detail
                            )
                    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                        failures.append(
                            {"case_id": case["case_id"], "method": method, "exception": repr(exc)}
                        )
                    finally:
                        run = device = None
                        gc.collect()
                    done = [
                        r["total_ns"]
                        for r in rows
                        if r["case_id"] == case["case_id"] and r["method"] == method
                    ]
                    if done:
                        print(
                            f"{case['case_id']:52s} {method:22s} {min(done) / 1e6:11.2f} ms",
                            flush=True,
                        )
                write(args.output, metadata, rows, failures, {"values": jacobians})
    write(args.output, metadata, rows, failures, {"values": jacobians})
    print(f"rows {len(rows)} failures {len(failures)}")


def training_study(manifest, methods, args, metadata):
    """Run measured gradient-descent loops; every step pays its full cost."""
    import numpy as np
    from direct_support import (
        aer_batch,
        aer_simulator,
        assemble_jacobian,
        jacobian_requests,
        lightning_device,
        pennylane_tape,
        shifted_programs,
        timed,
        trainable_indices,
        with_angles,
    )
    from qml_workload_support import make_batch

    from lazyqc import Pauli, plan_all

    config = manifest["training"]
    limits = manifest["planner_limits"]
    qubits = [int(q) for q in args.qubits.split(",")] if args.qubits else config["qubits"]
    steps = 2 if args.smoke else config["steps"]
    rate = config["learning_rate"]
    gpu = aer_simulator("GPU") if "aer_gpu_shift" in methods else None
    loops = []
    for n in qubits:
        seed = manifest["seed"] + config["seed_offset"]
        base = make_batch(config["family"], n, manifest["depth"], "same_state_local_outputs", seed)
        program = base[0].program
        observables = [Pauli(((q, "Z"),)) for q in range(n)]
        indices = trainable_indices(program)
        targets = np.random.default_rng(manifest["seed"]).uniform(-0.5, 0.5, n)
        start_angles = np.array([program.operations[i].angle for i in indices])
        for method in methods:
            if n > config["methods"][method].get("max_qubits", 99):
                continue
            device = None
            if method == "lightning_gpu_vjp":
                device, config_ = lightning_device("lightning.gpu", n)

            def step(angles):
                current = with_angles(program, indices, angles)
                if method == "lazyqc":
                    requests = jacobian_requests(shifted_programs(current, indices), observables)
                    plan = plan_all(
                        requests,
                        max_state_qubits=limits["max_state_qubits"],
                        max_state_bytes=limits["max_state_bytes"],
                    )
                    forward, jacobian = assemble_jacobian(plan.collect().values, len(indices), n)
                    gradient = jacobian.T @ (2 * (forward - targets) / n)
                elif method == "aer_gpu_shift":
                    values = aer_batch(shifted_programs(current, indices), observables, gpu)()
                    forward, jacobian = assemble_jacobian(values, len(indices), n)
                    gradient = jacobian.T @ (2 * (forward - targets) / n)
                else:
                    tape = pennylane_tape(current, observables, trainable=indices)
                    forward = np.asarray(device.execute(tape))
                    cotangent = 2 * (forward - targets) / n
                    gradient = np.asarray(device.compute_vjp(tape, cotangent, config_))
                loss = float(np.mean((forward - targets) ** 2))
                return angles - rate * gradient, loss

            step(start_angles)  # warm caches, devices and kernels once
            angles, losses, times = start_angles.copy(), [], []
            for _ in range(steps):
                (angles, loss), elapsed = timed(lambda angles=angles: step(angles))
                losses.append(loss)
                times.append(elapsed)
            loops.append(
                {
                    "family": config["family"],
                    "qubits": n,
                    "method": method,
                    "threads": args.threads,
                    "parameters": len(indices),
                    "step_ns": times,
                    "loss": losses,
                }
            )
            print(
                f"{n:3d}q {method:20s} total {sum(times) / 1e6:10.1f} ms "
                f"loss {losses[0]:.6f} -> {losses[-1]:.6f}",
                flush=True,
            )
            device = None
            gc.collect()
            write(args.output, metadata, [], [], {"loops": loops})
    write(args.output, metadata, [], [], {"loops": loops})


def write(path, metadata, rows, failures, extra):
    """Persist the current results atomically enough for long runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "failures": failures, "rows": rows} | extra
    from direct_support import atomic_write

    atomic_write(path, json.dumps(payload, indent=1))


if __name__ == "__main__":
    main()
