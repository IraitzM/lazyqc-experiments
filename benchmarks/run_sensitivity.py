"""Reproducible accuracy/precision ablations and charged local-worker timings."""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np  # noqa: E402
from local_parallel_support import ResourceSampler, numpy_expectation, numpy_state  # noqa: E402

from lazyqc import ExecutionSession, Gate, Parameter, Program, plan_all  # noqa: E402
from lazyqc.sensitivity import plan_approximate, profile_sensitivity  # noqa: E402

from core_checkout import label, package_files

ROOT = Path(__file__).resolve().parents[1]
ALLOWANCE = 1e-10


def reference(program, observables):
    state = numpy_state(program.n_qubits, program.resolved_operations())
    return np.array([numpy_expectation(state, tuple(o.items())) for o in observables])


def changed(program, indices, deltas):
    ops = list(program.resolved_operations())
    for i, delta in zip(indices, deltas, strict=True):
        ops[i] = replace(ops[i], angle=ops[i].angle + float(delta))
    return Program(program.qubits, tuple(ops))


def accuracy_study(count):
    rows = []
    for seed in range(count):
        rng = np.random.default_rng(17000 + seed)
        width = 2 + seed % 4
        gates = []
        for _ in range(4):
            for q in range(width):
                gates.append(Gate("R" + rng.choice(list("XYZ")), (q,), rng.uniform(-2, 2)))
            for q in range(width - 1):
                gates.append(Gate("CX", (q, q + 1)))
        program = Program(width, tuple(gates))
        indices = [i for i, g in enumerate(gates) if g.name.startswith("R")][:6]
        observables = [{0: "X"}, {width - 1: "Z"}, {0: "Y", width - 1: "X"}]
        profile = profile_sensitivity(program, observables, indices, evaluation_error=ALLOWANCE)
        baseline = reference(program, observables)
        native = plan_all([program.expectation(o) for o in observables]).collect().values
        reference_error = float(np.max(np.abs(np.array(native) - baseline)))
        fd_errors = []
        for j in range(len(indices)):
            ds = np.zeros(len(indices))
            ds[j] = 1e-5
            fd = (
                reference(changed(program, indices, ds), observables)
                - reference(changed(program, indices, -ds), observables)
            ) / 2e-5
            fd_errors.append(float(np.max(np.abs(fd - np.array(profile.jacobian)[:, j]))))
        for scale in (1e-6, 1e-3, 0.1, 1.0):
            ds = rng.uniform(-scale, scale, len(indices))
            actual = np.abs(reference(changed(program, indices, ds), observables) - baseline)
            linear = np.abs(np.array(profile.jacobian) @ ds)
            bound = np.array(profile.bounds(ds))
            assert np.all(actual <= bound + 1e-13)
            rows.append(
                dict(
                    seed=seed,
                    width=width,
                    delta_scale=scale,
                    max_reference_error=reference_error,
                    max_fd_error=max(fd_errors),
                    actual=actual.tolist(),
                    linear=linear.tolist(),
                    bound=bound.tolist(),
                )
            )
    # Analytic stationary and shared-parameter counterexamples, plus step-size sweep.
    p = Program(1).ry(0, math.pi)
    stationary = profile_sensitivity(p, [{0: "Z"}], [0], evaluation_error=ALLOWANCE)
    stationary_plan = plan_approximate(stationary, error_budget=0.01)
    shared = Program(1).ry(0, Parameter("t")).ry(0, Parameter("t")).bind(t=0.3)
    shared_profile = profile_sensitivity(shared, [{0: "Z"}], [0, 1], evaluation_error=ALLOWANCE)
    theta = 0.37

    def value(t):
        return plan_all([Program(1).ry(0, t).expectation({0: "Z"})]).collect().values[0]

    precision = [
        dict(
            step=h,
            finite_difference=(value(theta + h) - value(theta - h)) / (2 * h),
            absolute_error=abs((value(theta + h) - value(theta - h)) / (2 * h) + math.sin(theta)),
        )
        for h in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16)
    ]
    shift = profile_sensitivity(
        Program(1).ry(0, theta), [{0: "Z"}], [0], evaluation_error=ALLOWANCE
    )
    return dict(
        random=rows,
        stationary=dict(
            gradient=stationary.jacobian[0][0],
            actual_zeroing_error=2.0,
            bound=stationary.bounds([-math.pi])[0],
            accepted=bool(stationary_plan.zeroed_indices),
        ),
        shared_parameter=dict(
            gradient=shared_profile.parameter_gradient("t")[0],
            analytic=-2 * math.sin(0.6),
            wrong_global_shift=(
                reference(shared.bind(t=0.3 + math.pi / 2), [{0: "Z"}])[0]
                - reference(shared.bind(t=0.3 - math.pi / 2), [{0: "Z"}])[0]
            )
            / 2,
        ),
        finite_difference=precision,
        parameter_shift_error=abs(shift.jacobian[0][0] + math.sin(theta)),
    )


def fixture(block_width, blocks, seed, control=False):
    rng = np.random.default_rng(22000 + seed)
    width = block_width * blocks
    gates = []

    def layers(depth):
        for _ in range(depth):
            for q in range(width):
                for axis in ("Y", "Z"):
                    gates.append(Gate("R" + axis, (q,), rng.uniform(-2, 2)))
            for block in range(blocks):
                start = block * block_width
                for q in range(start, start + block_width - 1):
                    gates.append(Gate("CX", (q, q + 1)))

    layers(20)
    indices = []
    for block in range(blocks - 1):
        a, b = (block + 1) * block_width - 1, (block + 1) * block_width
        gates.append(Gate("CX", (a, b)))
        indices.append(len(gates))
        gates.append(Gate("RZ", (b,), 0.003 * (block + 1)))
        gates.append(Gate("CX", (a, b)))
        if control:
            gates.append(Gate("CX", (a, b)))
    layers(4)
    observables = [{block * block_width: axis} for block in range(blocks) for axis in ("X", "Z")]
    return Program(width, tuple(gates)), observables, indices


def collect(plan, session):
    start = perf_counter_ns()
    result = plan.collect(**({"session": session} if session is not None else {}))
    return perf_counter_ns() - start, np.array(result.values)


def timing_case(spec, workers, repeats):
    block_width, blocks, seed, control = spec
    program, observables, indices = fixture(*spec)
    baseline_reference = reference(program, observables)
    options = dict(workers=workers, optimization="local", strategy="grouped")
    session = ExecutionSession(workers, policy="parallel") if workers > 1 else None
    startup = 0
    if session:
        start = perf_counter_ns()
        session.warmup()
        startup = perf_counter_ns() - start
    try:
        mark = perf_counter_ns()
        exact = plan_all([program.expectation(o) for o in observables], **options)
        exact_planning = perf_counter_ns() - mark
        profile = profile_sensitivity(
            program, observables, indices, evaluation_error=ALLOWANCE, session=session, **options
        )
        approximation = plan_approximate(profile, error_budget=0.01, **options)
        target_reference = reference(approximation.program, observables)
        actual = np.abs(target_reference - baseline_reference)
        assert np.all(actual <= np.array(approximation.error_bounds) + 1e-13)
        runs = {"exact": [], "bounded": []}
        first = {}
        numerical_error = 0.0
        for name, plan, ref in (
            ("exact", exact, baseline_reference),
            ("bounded", approximation.plan, target_reference),
        ):
            ns, values = collect(plan, session)
            first[name] = ns
            numerical_error = max(numerical_error, float(np.max(np.abs(values - ref))))
        for rep in range(repeats):
            order = ("exact", "bounded") if rep % 2 == 0 else ("bounded", "exact")
            for name in order:
                plan = exact if name == "exact" else approximation.plan
                ref = baseline_reference if name == "exact" else target_reference
                ns, values = collect(plan, session)
                runs[name].append(ns)
                numerical_error = max(numerical_error, float(np.max(np.abs(values - ref))))
        assert numerical_error < ALLOWANCE
        # RSS probes are separate from timing samples and include the same live pool.
        pids = [os.getpid()]
        if session:
            pids += list(session._pool._processes)
        resources = {}
        for name, plan in (("exact", exact), ("bounded", approximation.plan)):
            with ResourceSampler(pids) as sampler:
                collect(plan, session)
            resources[name] = sampler.report()
        sweeps = []
        # Accuracy policies all use the same eligible occurrences. Uniform rounds to
        # a 0.02-radian grid (both bridge angles become zero), without an error test.
        ds = [-program.operations[i].angle for i in indices]
        uniform = changed(program, indices, ds)
        uniform_error = np.abs(reference(uniform, observables) - baseline_reference)
        linear = np.abs(np.array(profile.jacobian) @ ds)
        for budget in (1e-6, 1e-4, 1e-2):
            bounded = plan_approximate(profile, error_budget=budget, **options)
            observed = np.abs(reference(bounded.program, observables) - baseline_reference)
            assert np.all(observed <= budget + 1e-13)
            sweeps.append(
                dict(
                    budget=budget,
                    bounded_zeroed=list(bounded.zeroed_indices),
                    bounded_errors=observed.tolist(),
                    bounds=list(bounded.error_bounds),
                    width=bounded.plan.selected.max_width,
                    uniform_errors=uniform_error.tolist(),
                    uniform_violates=bool(np.any(uniform_error > budget)),
                    gradient_accepts=bool(np.all(linear <= budget)),
                    gradient_would_violate=bool(
                        np.all(linear <= budget) and np.any(uniform_error > budget)
                    ),
                )
            )
    finally:
        start = perf_counter_ns()
        if session:
            session.close()
        shutdown = perf_counter_ns() - start
    baseline_total = startup + exact_planning + shutdown + first["exact"] + sum(runs["exact"])
    approximate_total = (
        startup
        + profile.elapsed_ns
        + approximation.planning_ns
        + shutdown
        + first["bounded"]
        + sum(runs["bounded"])
    )
    baseline_cumulative = (
        np.cumsum([first["exact"], *runs["exact"]]) + startup + exact_planning + shutdown
    )
    approximate_cumulative = (
        np.cumsum([first["bounded"], *runs["bounded"]])
        + startup
        + profile.elapsed_ns
        + approximation.planning_ns
        + shutdown
    )
    break_even = next(
        (
            i + 1
            for i in range(len(baseline_cumulative))
            if np.all(approximate_cumulative[i:] < baseline_cumulative[i:])
        ),
        None,
    )
    return dict(
        name=f"{'control' if control else 'bridge'}_{block_width * blocks}q_{blocks}b_s{seed}",
        seed=seed,
        workers=workers,
        width=program.n_qubits,
        gate_count=len(program.operations),
        eligible_indices=indices,
        observables=observables,
        startup_ns=startup,
        shutdown_ns=shutdown,
        exact_planning_ns=exact_planning,
        profile_ns=profile.elapsed_ns,
        approximation_planning_ns=approximation.planning_ns,
        shifted_requests=profile.shifted_requests,
        jacobian=profile.jacobian,
        zeroed_indices=approximation.zeroed_indices,
        error_bounds=approximation.error_bounds,
        actual_errors=actual.tolist(),
        max_numerical_error=numerical_error,
        exact=exact.explain(),
        bounded=approximation.plan.explain(),
        exact_width=exact.selected.max_width,
        bounded_width=approximation.plan.selected.max_width,
        exact_tasks=len(exact.selected.tasks),
        bounded_tasks=len(approximation.plan.selected.tasks),
        exact_payload=exact.selected.state_bytes,
        bounded_payload=approximation.plan.selected.state_bytes,
        resources=resources,
        sweeps=sweeps,
        first_ns=first,
        warm_ns=runs,
        warm_speedup=median(runs["exact"]) / median(runs["bounded"]),
        lifetime_speedup=baseline_total / approximate_total,
        observed_break_even=break_even,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    specs = (
        [(4, 2, 0, False)]
        if args.smoke
        else [
            (4, 2, 0, False),
            (6, 2, 0, False),
            *((8, 2, seed, False) for seed in range(3)),
            *((5, 3, seed, False) for seed in range(3)),
            (8, 2, 0, True),
        ]
    )
    sources = [
        *package_files(),
        Path(__file__),
        ROOT / "benchmarks/local_parallel_support.py",
    ]
    result = dict(
        schema=1,
        metadata=dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            python=sys.version,
            platform=platform.platform(),
            logical_cpus=os.cpu_count(),
            native_threads=1,
            evaluation_error_assumption=ALLOWANCE,
            source_sha256={
                label(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sources
            },
        ),
        accuracy=accuracy_study(4 if args.smoke else 96),
        cases=[],
    )
    for spec in specs:
        for workers in (1, 2, 4):
            row = timing_case(spec, workers, args.repeats)
            result["cases"].append(row)
            print(
                f"{row['name']} w={workers}: {row['warm_speedup']:.2f}x warm, "
                f"{row['lifetime_speedup']:.2f}x lifetime; "
                f"width {row['exact_width']}->{row['bounded_width']}",
                flush=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
