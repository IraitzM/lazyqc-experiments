#!/usr/bin/env python3
"""Fixed-DAG local strong/weak scaling, setup repayment, and resource constraints."""

import argparse
import hashlib
import json
import math
import os
import pickle
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from local_parallel_support import ResourceSampler, numpy_expectation, numpy_state
from core_checkout import label, package_files

ROOT = Path(__file__).resolve().parents[1]
THREADS = ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def manifest(smoke=False):
    if smoke:
        return [dict(name="smoke", width=4, depths=[2, 2], seed=6200)]
    return [
        dict(name="tiny_4q", width=4, depths=[4] * 8, seed=6200),
        dict(name="medium_10q", width=10, depths=[20] * 8, seed=6200),
        *[
            dict(name=f"balanced_14q_{n}tasks", width=14, depths=[40] * n, seed=6200)
            for n in (2, 4, 8)
        ],
        dict(name="wide_16q", width=16, depths=[40] * 8, seed=6200),
        dict(name="shallow_14q", width=14, depths=[4] * 8, seed=6200),
        dict(name="deep_14q", width=14, depths=[80] * 8, seed=6200),
        dict(name="imbalanced_14q", width=14, depths=[4] * 7 + [80], seed=6200),
        dict(name="bridge_cut_8q", width=8, depths=[20, 20], seed=6200, bridge=True),
    ]


def build(case):
    from lazyqc import Program
    from lazyqc.ir import Gate

    width = case["width"]
    components, operations, queries = [], [], []
    for block, depth in enumerate(case["depths"]):
        rng = random.Random(case["seed"] + block)
        local = []
        for _ in range(depth):
            for q in range(width):
                local.extend(
                    (Gate("RY", (q,), rng.uniform(-1, 1)), Gate("RZ", (q,), rng.uniform(-1, 1)))
                )
            local.extend(Gate("CX", (q, q + 1)) for q in range(width - 1))
        components.append(tuple(local))
        operations.extend(
            replace(g, qubits=tuple(q + block * width for q in g.qubits)) for g in local
        )
    if case.get("bridge"):
        operations.extend(
            (
                Gate("CX", (width - 1, width)),
                Gate("RY", (width - 1,), 0.37),
                Gate("RY", (width,), -0.29),
            )
        )
    p = Program(width * len(components), tuple(operations))
    if case.get("bridge"):
        queries.append(p.expectation({width - 1: "X", width: "Z", 2 * width - 1: "Z"}))
    else:
        queries.extend(p.expectation({(b + 1) * width - 1: "Z"}) for b in range(len(components)))
    return tuple(queries), components


def oracle(case, requests, components):
    if case.get("bridge"):
        state = numpy_state(requests[0].program.n_qubits, requests[0].program.operations)
        return tuple(numpy_expectation(state, r.observable.factors) for r in requests)
    return tuple(
        numpy_expectation(numpy_state(case["width"], ops), ((case["width"] - 1, "Z"),))
        for ops in components
    )


def check(values, expected):
    if len(values) != len(expected):
        raise AssertionError("Missing results")
    error = max((abs(a - b) for a, b in zip(values, expected, strict=True)), default=0)
    if error > 1e-10 or any(not math.isfinite(x) for x in values):
        raise AssertionError(f"Oracle mismatch: {error}")
    return error


def collect(plan, expected, session=None):
    mark = perf_counter_ns()
    result = plan.collect(session=session)
    total = perf_counter_ns() - mark
    assert result.stats.peak_active_state_bytes <= plan.total_state_bytes
    assert result.stats.allocations == len(plan.selected.tasks)
    return {
        "total_ns": total,
        "max_abs_error": check(result.values, expected),
        "stats": asdict(result.stats),
    }


def measured_resources(plan, expected, session=None):
    # Read-only private pool access isolates exactly the processes under test.
    pids = [os.getpid()] + (list(session._pool._processes) if session else [])
    with ResourceSampler(pids) as sampler:
        row = collect(plan, expected, session)
    return {"collection": row, **sampler.report()}


def benchmark_case(case, model, repeats, session_id):
    from lazyqc import CutOptions, ExecutionSession, plan_all
    from lazyqc.batch_executor import _one

    mark = perf_counter_ns()
    requests, components = build(case)
    construction = perf_counter_ns() - mark
    expected = oracle(case, requests, components)
    width = case["width"]
    options = CutOptions(
        max_fragment_qubits=width,
        max_cuts=int(case.get("bridge", False)),
        search="independent",
        soft_timeout_ms=30000,
    )
    if case.get("bridge"):
        options = replace(options, partition=(tuple(range(width)), tuple(range(width, 2 * width))))
    mark = perf_counter_ns()
    plan = plan_all(
        requests,
        strategy="separate",
        cutting=options,
        workers=1,
        max_state_qubits=width,
        max_state_bytes=16 * 2**width,
        total_state_bytes=16 * 2**width,
        cost_model=model,
    )
    planning = perf_counter_ns() - mark
    dag = {
        "tasks": [asdict(t) for t in plan.selected.tasks],
        "reconstruction": [asdict(r) for r in plan.selected.reconstruction],
    }
    dag_hash = hashlib.sha256(json.dumps(dag, sort_keys=True).encode()).hexdigest()
    profiles = []
    for task in plan.selected.tasks:
        encoded = pickle.dumps(task, protocol=5)
        marks = []
        for _ in range(3):
            start = perf_counter_ns()
            pickle.loads(pickle.dumps(task, protocol=5))
            marks.append(perf_counter_ns() - start)
        service = _one(task)
        profiles.append(
            {
                "width": task.width,
                "state_bytes": task.state_bytes,
                "gates": len(task.operations),
                "outputs": len(task.outputs),
                "serialized_task_bytes": len(encoded),
                "serialized_result_bytes": len(pickle.dumps(service, protocol=5)),
                "pickle_roundtrip_median_ns": median(marks),
                "phase_ns": service[1:4],
            }
        )
    configs = [(1, 1), (2, 2), (4, 4)]
    if case["name"] in ("wide_16q", "smoke"):
        configs += [(4, 1), (4, 2)]
    rng = random.Random(6400 + session_id)
    rng.shuffle(configs)
    outputs = []
    for workers, slots in configs:
        parallel_plan = replace(plan, workers=workers, total_state_bytes=slots * 16 * 2**width)
        setup = shutdown = 0
        rows = []
        pool = None
        profile = None
        if workers > 1:
            pool = ExecutionSession(workers, policy="parallel")
            start = perf_counter_ns()
            profile = asdict(pool.warmup())
            setup = perf_counter_ns() - start
        try:
            # First numerical call is retained; subsequent trials are warm.
            first = collect(parallel_plan, expected, pool)
            first_serial = collect(plan, expected) if workers > 1 else first
            for trial in range(repeats):
                modes = ["serial", "parallel", "auto"] if workers > 1 else ["serial"]
                rng.shuffle(modes)
                for mode in modes:
                    if mode == "serial":
                        row = collect(plan, expected)
                    else:
                        pool.policy = mode
                        row = collect(parallel_plan, expected, pool)
                    rows.append({"trial": trial, "mode": mode, **row})
            if pool:
                pool.policy = "parallel"
            resources = measured_resources(parallel_plan, expected, pool)
        finally:
            if pool:
                mark = perf_counter_ns()
                pool.close()
                shutdown = perf_counter_ns() - mark
        outputs.append(
            {
                "workers": workers,
                "payload_slots": slots,
                "total_state_bytes": parallel_plan.total_state_bytes,
                "warmup_ns": setup,
                "shutdown_ns": shutdown,
                "transport_profile": profile,
                "first": first,
                "first_serial": first_serial,
                "runs": rows,
                "resources": resources,
            }
        )
    return {
        "case": case,
        "construction_ns": construction,
        "planning_ns": planning,
        "reference_values": expected,
        "oracle": (
            "independent NumPy vector operations; factorized by declared independent "
            "components, full 16-qubit oracle for cut bridge"
        ),
        "dag_sha256": dag_hash,
        "dag": dag,
        "plan": plan.explain(),
        "task_profiles": profiles,
        "configurations": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/local_parallel/session0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    for key in THREADS:
        os.environ[key] = "1"
    from lazyqc import calibrate_batch
    from lazyqc.cost import environment

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "evaluation.json"
    if target.exists():
        raise FileExistsError(f"Preserve existing results: {target}")
    cases = manifest(args.smoke)
    widths = (1, 2, 4) if args.smoke else (1, 2, 4, 8, 10, 14, 16)
    mark = perf_counter_ns()
    model, calibration = calibrate_batch(widths=widths, repeats=3 if args.smoke else 7)
    calibration_total = perf_counter_ns() - mark
    model.save(args.output_dir / "profile.json")
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    metadata = {
        "session": args.session,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "logical_cpus": os.cpu_count(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "loadavg_start": os.getloadavg(),
        "repeats": args.repeats,
        "calibration_wall_ns": calibration_total,
        "profile_id": model.profile_id,
        "manifest_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "source_sha256": {
            label(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package_files()
        },
        "benchmark_sha256": {
            label(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "benchmarks/local_parallel_support.py")
        },
        "scope": (
            "fixed DAG; construction, cold planning, calibration, full pool warmup, first "
            "collection, warm trials and shutdown separate; paired randomized modes; "
            "resource probes excluded from timing samples; no remote execution"
        ),
    }
    result = {"metadata": metadata, "manifest": cases, "cases": []}
    # Same predefined workload set, different execution order across interpreters.
    order = list(cases)
    random.Random(6300 + args.session).shuffle(order)
    for case in order:
        print(f"Session {args.session}: {case['name']}", flush=True)
        one = benchmark_case(case, model, args.repeats, args.session)
        result["cases"].append(one)
        (args.output_dir / "partial.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        for config in one["configurations"]:
            means = {
                m: round(median(r["total_ns"] for r in config["runs"] if r["mode"] == m) / 1e6, 3)
                for m in {r["mode"] for r in config["runs"]}
            }
            print(
                f"  workers={config['workers']} slots={config['payload_slots']}: {means} ms",
                flush=True,
            )
    result["metadata"]["loadavg_end"] = os.getloadavg()
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.output_dir / "partial.json").unlink()
    print(f"Saved {target}", flush=True)


if __name__ == "__main__":
    main()
