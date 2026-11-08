#!/usr/bin/env python3
"""Measure setup-inclusive reuse, unique bindings and persistent-worker dispatch."""

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import dense_expectations, pin_single_thread
from core_checkout import FIXTURES, label, package_files

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def generated(seed, width):
    from lazyqc import Program

    rng = random.Random(seed)
    p = Program(width)
    for q in range(width):
        p = p.ry(q, rng.uniform(-1, 1)).rz(q, rng.uniform(-1, 1))
    for q in range(width - 1):
        p = p.cx(q, q + 1).ry(q + 1, rng.uniform(-1, 1))
    return tuple(p.expectation({q: axis}) for q in range(width) for axis in "XZ")


def check(values, expected):
    error = max((abs(a - b) for a, b in zip(values, expected, strict=True)), default=0)
    if error > 1e-10 or any(not math.isfinite(v) for v in values):
        raise AssertionError(f"Incorrect result: {error}")
    return error


def worker_benchmark(model, trials):
    from lazyqc import CutOptions, ExecutionSession, Program, plan_all

    fixtures = {}
    bell = Program(2).h(0).cx(0, 1)
    fixtures["bell"] = (tuple(bell.expectation({0: a, 1: a}) for a in "XYZ"), 1, 1)
    for width in (12, 16):
        p = Program(2 * width)
        # Keep both the modest and more compute-intensive native workloads.
        for base in (0, width):
            for layer in range(80):
                for q in range(base, base + width):
                    p = p.ry(q, 0.2 + 0.001 * layer + 0.002 * q)
                for q in range(base, base + width - 1):
                    p = p.cx(q, q + 1)
        fixtures[f"two_{width}q_components"] = (
            (p.expectation({width - 1: "Z"}), p.expectation({2 * width - 1: "Z"})),
            width,
            0,
        )
    result = {}
    for name, (requests, width, cuts) in fixtures.items():
        plan = plan_all(
            requests,
            strategy="separate",
            workers=2,
            cost_model=model,
            cutting=CutOptions(max_fragment_qubits=width, max_cuts=cuts, search="independent"),
            total_state_bytes=32 * 2**width,
        )
        serial = replace(plan, workers=1)
        expected = serial.collect().values
        modes = ("serial_same_dag", "fresh_processes", "persistent_parallel", "persistent_auto")
        rows = []
        with ExecutionSession(2, policy="parallel") as parallel, ExecutionSession(2) as auto:
            profiles = {"parallel": asdict(parallel.warmup()), "auto": asdict(auto.warmup())}
            rng = random.Random(571)
            for trial in range(trials):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    start = perf_counter_ns()
                    if mode == "serial_same_dag":
                        output = serial.collect()
                    else:
                        session = (
                            parallel
                            if mode == "persistent_parallel"
                            else auto
                            if mode == "persistent_auto"
                            else None
                        )
                        output = plan.collect(session=session)
                    elapsed = perf_counter_ns() - start
                    rows.append(
                        {
                            "trial": trial,
                            "mode": mode,
                            "total_ns": elapsed,
                            "max_abs_error": check(output.values, expected),
                            **asdict(output.stats),
                        }
                    )
        result[name] = {
            "runs": rows,
            "transport_profiles": profiles,
            "unique_experiments": len(plan.selected.tasks),
            "oracle": "same-DAG serial Qulacs execution consistency",
            "median_ns": {
                mode: median(r["total_ns"] for r in rows if r["mode"] == mode) for mode in modes
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/priorities/session0")
    parser.add_argument("--collections", type=positive, default=512)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--seeds", type=positive, default=3)
    parser.add_argument("--worker-trials", type=positive, default=7)
    parser.add_argument("--unique-bindings", type=positive, default=16)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["qaoa_path", "qaoa_star", "local_pauli_batch16", "overlapping_uncut_batch16"],
    )
    args = parser.parse_args()
    pin_single_thread()
    from lazyqc import CutOptions, PlanningCache, Program, calibrate_batch, plan_all
    from lazyqc.cost import environment
    from lazyqc.fixtures import build_requests

    frozen = json.loads((FIXTURES / "p4_p5.json").read_text())
    cases = []
    for case in frozen["cases"]:
        if case["name"] in args.cases:
            cases.append(
                (
                    case["name"],
                    build_requests(case),
                    CutOptions(
                        max_fragment_qubits=case["fragment_width"],
                        max_cuts=case["max_cuts"],
                        **frozen["cut_limits"],
                    ),
                )
            )
    if len(cases) != len(args.cases):
        raise ValueError("Unknown or duplicate case")
    # Fixed generator/seeds/widths, unchanged across independent sessions.
    for seed in range(6100, 6100 + args.seeds):
        for width in (4, 6, 8):
            cases.append(
                (
                    f"seed{seed}_{width}q",
                    generated(seed, width),
                    CutOptions(max_fragment_qubits=width, max_cuts=0),
                )
            )
    manifest = [
        {"name": name, "requests": [asdict(r) for r in requests], "cutting": asdict(opts)}
        for name, requests, opts in cases
    ]
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    print(
        f"Session {args.session}: manifest frozen ({manifest_hash[:12]}), calibrating...",
        flush=True,
    )
    model, calibration = calibrate_batch(widths=(1, 2, 3, 4, 6, 8, 10, 12, 16), repeats=7)
    import pyzx  # noqa: F401

    from lazyqc.cutting import cx_branches

    cx_branches()
    rng = random.Random(5700 + args.session)
    rows, summaries, plans = [], [], {}
    for name, requests, options in cases:
        expected = dense_expectations(requests)
        configs = {
            "independent": replace(options, search="independent"),
            "joint": options,
            "joint_budget": replace(options, expected_collections=args.collections),
        }
        caches = {mode: PlanningCache(4, max_bytes=32 * 2**20) for mode in configs}
        cumulative = {mode: [] for mode in configs}
        first_planning = {}
        for index in range(args.collections):
            order = list(configs)
            rng.shuffle(order)
            for mode in order:
                start = perf_counter_ns()
                plan = plan_all(
                    requests,
                    cutting=configs[mode],
                    optimization="zx",
                    cost_model=model,
                    cache=caches[mode],
                )
                planned = perf_counter_ns()
                output = plan.collect()
                finish = perf_counter_ns()
                elapsed = finish - start
                cumulative[mode].append(elapsed + (cumulative[mode][-1] if index else 0))
                row = {
                    "case": name,
                    "mode": mode,
                    "collection": index + 1,
                    "total_ns": elapsed,
                    "planning_ns": planned - start,
                    "execution_ns": finish - planned,
                    "cache_status": plan.cache_status,
                    "unique_experiments": len(plan.selected.tasks),
                    "max_abs_error": check(output.values, expected),
                }
                rows.append(row)
                if index == 0:
                    first_planning[mode] = row["planning_ns"]
                    plans[f"{name}/{mode}"] = plan.explain()
        comparisons = {}
        for mode in ("joint", "joint_budget"):
            wins = [
                a <= b for a, b in zip(cumulative[mode], cumulative["independent"], strict=True)
            ]
            stable = next((i + 1 for i in range(len(wins)) if all(wins[i:])), None)
            comparisons[mode] = {
                "first_observed_break_even": next(
                    (i + 1 for i, win in enumerate(wins) if win), None
                ),
                "break_even_sustained_to_end": stable,
                "setup_inclusive_speedup": cumulative["independent"][-1] / cumulative[mode][-1],
            }
        one = perf_counter_ns()
        one_plan = plan_all(
            requests,
            cutting=replace(options, expected_collections=1),
            optimization="zx",
            cost_model=model,
        )
        one_result = one_plan.collect()
        one_ns = perf_counter_ns() - one
        check(one_result.values, expected)
        summaries.append(
            {
                "case": name,
                "comparisons": comparisons,
                "cumulative_ns": cumulative,
                "first_planning_ns": first_planning,
                "one_shot_budget_total_ns": one_ns,
                "one_shot_search_states": one_plan.search_states,
                "cache_info": {m: asdict(c.info()) for m, c in caches.items()},
            }
        )
        print(f"Validated {name}: {comparisons}", flush=True)
    # Every parameter binding below is unique: cache hits would be incorrect.
    unique = []
    for mode in ("independent", "joint", "joint_budget"):
        cache = PlanningCache(4, max_bytes=2**20)
        for i in range(args.unique_bindings):
            p = Program(4).ry(0, 0.13 + 0.019 * i).cx(0, 1).ry(2, 0.29).cx(2, 3)
            requests = tuple(p.expectation({q: "Z"}) for q in range(4))
            expected = dense_expectations(requests)
            start = perf_counter_ns()
            plan = plan_all(
                requests,
                optimization="zx",
                cache=cache,
                cost_model=model,
                cutting=CutOptions(
                    max_fragment_qubits=2,
                    max_cuts=0,
                    search="independent" if mode == "independent" else "joint",
                    expected_collections=1 if mode == "joint_budget" else None,
                ),
            )
            result = plan.collect()
            elapsed = perf_counter_ns() - start
            if plan.cache_status != "miss":
                raise AssertionError("Unique binding unexpectedly reused a plan")
            unique.append(
                {
                    "mode": mode,
                    "binding": i,
                    "total_ns": elapsed,
                    "max_abs_error": check(result.values, expected),
                    "cache_info": asdict(cache.info()),
                }
            )
    workers = worker_benchmark(model, args.worker_trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "profile.json")
    (args.output_dir / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    output = {
        "metadata": {
            "session": args.session,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment(),
            "manifest_sha256": manifest_hash,
            "collections": args.collections,
            "profile_id": model.profile_id,
            "source_sha256": {
                label(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in package_files()
            },
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": (
                "same request descriptions; plan lookup/search + fresh execution; "
                "first cache miss included"
            ),
            "worker_scope": (
                "same DAG; pool warmup measured separately; no persistent numerical results"
            ),
        },
        "manifest": manifest,
        "runs": rows,
        "summary": summaries,
        "plans": plans,
        "unique_bindings": unique,
        "workers": workers,
    }
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(f"Saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
