#!/usr/bin/env python3
"""Weeks 1-2 manifest runner, family C (v2): B3 independent vs B5 joint.

Covers manifest cases qaoa_path / qaoa_star from benchmarks/reference/fixtures/p4_p5.json.
Each repeat runs cold (fresh plan) and warm (shared PlanningCache) pairs so
both halves of proposal preliminary result 3 are tested: cold joint planning
is expected to lose; warm sharing is expected to win on the path.
B5_budget1 modes add a one-use reuse budget (expected_collections=1), an
early probe of the Weeks 3-5 setup-aware search policy: cold planning cost
should fall while warm plan quality is preserved.

Writes results/workload_manifest/v2/evaluation.json.
Requires extras: uv run --locked --all-extras python benchmarks/run_workload_manifest_v2.py
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import (
    bootstrap_interval,
    cpu_name,
    dense_expectations,
    pin_single_thread,
    source_hash,
)
from core_checkout import FIXTURES, label

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "workload_manifest.json"
FIXTURE = FIXTURES / "p4_p5.json"
MODES_BASE = (
    "B3_independent",
    "B5_joint",
    "B3_cached",
    "B5_cached",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument(
        "--budgets",
        default="1",
        help="Comma-separated expected_collections values; each adds cold+cached modes",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/workload_manifest/v2/evaluation.json",
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    fam_c = manifest["circuit_families"]["C_qaoa_graphs"]
    if args.cases is None:
        args.cases = sorted(fam_c["cases"])
    if args.repeats is None:
        args.repeats = fam_c["repeats"]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    if any(b < 1 for b in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("Budgets must be distinct positive integers")
    modes_all = list(MODES_BASE)
    for b in budgets:
        modes_all += [f"B5_budget{b}", f"B5_budget{b}_cached"]
    pin_single_thread()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/lazyqc-numba")

    from lazyqc import CutOptions, NoFeasiblePlan, PlanningCache, calibrate_batch, plan_all
    from lazyqc.cutting import cx_branches
    from lazyqc.fixtures import build_requests, load_fixture

    # Prime the optional-dependency import (~300 ms qiskit tax) outside timed planning.
    cx_branches()
    fixture, cases = load_fixture(FIXTURE)
    cases = [c for c in cases if c["name"] in args.cases]
    if set(args.cases) != {c["name"] for c in cases}:
        raise ValueError("Unknown case requested")
    tolerance = fixture["absolute_tolerance"]

    print("Calibrating instrument execution profile...", flush=True)
    model, training = calibrate_batch(widths=(1, 2, 3, 4, 6, 8), repeats=5)

    rng = random.Random(manifest["seeds"]["timing_order_seed"])
    rows, failures, explanations = [], [], {}
    for case in cases:
        fw = case["fragment_width"]
        opts = CutOptions(
            max_fragment_qubits=fw,
            max_cuts=case["max_cuts"],
            **fixture["cut_limits"],
        )
        expected = dense_expectations(build_requests(case))
        caches = {mode: PlanningCache(8) for mode in modes_all if "cached" in mode}
        base_kwargs = {
            "optimization": "zx",
            "strategy": "auto",
            "cost_model": model,
            "workers": 1,
            "max_state_qubits": fw,
            "max_state_bytes": 16 * 2**fw,
            "total_state_bytes": 32 * 2**fw,
        }
        for rep in range(-1, args.repeats):
            modes = list(modes_all)
            rng.shuffle(modes)
            for mode in modes:
                if mode.startswith("B3"):
                    search, budget = "independent", None
                elif "budget" in mode:
                    search, budget = "joint", int(mode.split("budget")[1].split("_")[0])
                else:
                    search, budget = "joint", None
                kwargs = {
                    **base_kwargs,
                    "cutting": replace(opts, search=search, expected_collections=budget),
                }
                if "cached" in mode:
                    kwargs["cache"] = caches[mode]
                start = perf_counter_ns()
                row = {"case": case["name"], "mode": mode, "repetition": rep}
                try:
                    requests = build_requests(case)
                    built = perf_counter_ns()
                    plan = plan_all(requests, **kwargs)
                    planned = perf_counter_ns()
                    result = plan.collect()
                    values = result.values
                    finished = perf_counter_ns()
                    error = max(abs(a - b) for a, b in zip(values, expected, strict=True))
                    if not all(math.isfinite(v) for v in values) or error > tolerance:
                        raise AssertionError(f"Incorrect {case['name']}/{mode}: {error}")
                    row.update(
                        {
                            "status": "ok",
                            "max_abs_error": error,
                            "unique_experiments": len(plan.selected.tasks),
                            "selected_candidate": plan.selected.label,
                            "search_states": plan.search_states,
                            "search_exhausted": plan.search_exhausted,
                            "lowering_ns": result.stats.lowering_ns,
                            "evolution_ns": result.stats.evolution_ns,
                            "expectation_ns": result.stats.expectation_ns,
                            "reconstruction_ns": result.stats.reconstruction_ns,
                            "scheduling_ns": result.stats.scheduling_ns,
                            "collect_execution_ns": result.stats.execution_ns,
                            "planning_ns": planned - built,
                            "execution_ns": finished - planned,
                            "total_ns": finished - start,
                        }
                    )
                    if rep == -1:
                        explanations[f"{case['name']}/{mode}"] = plan.explain()
                except NoFeasiblePlan as exc:
                    row.update({"status": "infeasible", "reason": str(exc)[:200]})
                    failures.append({"case": case["name"], "mode": mode, "rep": rep})
                rows.append(row)

    summary = []
    comparisons = [
        ("B3_independent", "B5_joint"),
        ("B3_cached", "B5_cached"),
    ]
    for b in budgets:
        comparisons += [
            ("B3_independent", f"B5_budget{b}"),
            ("B3_cached", f"B5_budget{b}_cached"),
            ("B5_joint", f"B5_budget{b}"),
            ("B5_cached", f"B5_budget{b}_cached"),
        ]
    for case in cases:
        ok = [r for r in rows if r["case"] == case["name"] and r["status"] == "ok"]
        entry = {"case": case["name"]}
        for mode in modes_all:
            sub = [r for r in ok if r["mode"] == mode and r["repetition"] >= 0]
            if sub:
                entry[mode] = {
                    "n": len(sub),
                    "median_total_ms": median(r["total_ns"] for r in sub) / 1e6,
                    "median_planning_ms": median(r["planning_ns"] for r in sub) / 1e6,
                    "unique_experiments": sorted({r["unique_experiments"] for r in sub}),
                    "search_states": sorted({r["search_states"] for r in sub}),
                    "exhausted_repeats": sum(r["search_exhausted"] for r in sub),
                    "max_abs_error": max(r["max_abs_error"] for r in sub),
                }
        for base, proposed in comparisons:
            a = {r["repetition"]: r for r in ok if r["mode"] == base}
            b = {r["repetition"]: r for r in ok if r["mode"] == proposed}
            paired = sorted(set(a) & set(b))
            if paired:
                ratios = [a[i]["total_ns"] / b[i]["total_ns"] for i in paired]
                entry[f"{base}/{proposed}"] = {
                    "paired_speedups": ratios,
                    "geomean": math.exp(sum(map(math.log, ratios)) / len(ratios)),
                    "bootstrap_95pct": bootstrap_interval(ratios),
                    "regression_fraction": sum(r < 1 for r in ratios) / len(ratios),
                }
        summary.append(entry)

    import platform

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest["manifest_version"],
        "manifest_sha256": hashlib.sha256(MANIFEST.read_text().encode()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "source_sha256": source_hash(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "packages": {p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")},
        "calibration_training": training,
        "command": sys.argv,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"metadata": metadata, "summary": summary, "runs": rows, "plans": explanations},
        indent=2,
    )
    args.output.write_text(payload + "\n")

    for entry in summary:
        print(f"== {entry['case']} ==")
        for mode in modes_all:
            if mode in entry:
                m = entry[mode]
                print(
                    f"  {mode}: {m['median_total_ms']:.2f} ms "
                    f"(plan {m['median_planning_ms']:.2f}), "
                    f"tasks {m['unique_experiments']}, err {m['max_abs_error']:.1e}"
                )
        for comp in [f"{a}/{b}" for a, b in comparisons]:
            if comp in entry:
                c = entry[comp]
                lo, hi = c["bootstrap_95pct"]
                print(f"  {comp}: geomean {c['geomean']:.2f}x [{lo:.2f}-{hi:.2f}], "
                      f"regressions {c['regression_fraction']:.0%}")
    print(f"Saved {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
