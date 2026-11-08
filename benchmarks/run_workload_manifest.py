#!/usr/bin/env python3
"""Weeks 1-2 frozen workload manifest runner (v1): families A + B.

Reads benchmarks/workload_manifest.json, materializes deterministic cases,
and runs strategies full/separate/grouped/auto with interleaved order
(seed 0), threads=1. Family C (QAOA cutting) is listed in the manifest
but deferred to the v2 runner / benchmarks/run_p4_p5.py.

Writes results/workload_manifest/v1/evaluation.json.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import (
    ansatz_expectations,
    build_ansatz,
    cpu_name,
    overlapping_terms,
    pair_case,
    percentile,
    pin_single_thread,
    source_hash,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "workload_manifest.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats-pairs", type=int, default=20)
    ap.add_argument("--repeats-connected", type=int, default=10)
    ap.add_argument("--strategies", default="full,separate,grouped,auto")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "results/workload_manifest/v1/evaluation.json"
    )
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    pin_single_thread()

    import numpy as np

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend
    from lazyqc.fixtures import build_requests
    from lazyqc.planner import STRATEGIES

    strategies = args.strategies.split(",")
    assert len(set(strategies)) == len(strategies) and all(s in STRATEGIES for s in strategies)

    # --- materialize cases ---
    fam_a = manifest["circuit_families"]["A_pair_controls"]
    pair_cases = [pair_case(m, prefix="pairs") for m in fam_a["sizes_pairs"]]

    fam_b = manifest["circuit_families"]["B_connected_local"]
    seed = manifest["seeds"]["connected_ansatz"]
    layers_of = {"8": 2, "12": 3}
    connected = []  # (case_dict, expected_list)
    for n in fam_b["widths"]:
        rng = random.Random(seed + n)
        gates = build_ansatz(n, layers_of[str(n)], rng)
        ops = [
            {"gate": "RY", "qubits": [g[1]], "angle_radians": g[2]} if g[0] == "ry"
            else {"gate": "CX", "qubits": [g[1], g[2]]}
            for g in gates
        ]
        batches = {f"overlap_{c}": overlapping_terms(n, c, rng) for c in (1, 4, 16, 64)}
        batches["disjoint"] = [{q: "Z"} for q in range(n)]
        for name, terms in batches.items():
            case = {
                "name": f"connected_{n}q_{name}",
                "qubits": n,
                "initial_state": "zero",
                "operations": ops,
                "requests": [
                    {
                        "id": str(i),
                        "observable": [{"qubit": q, "pauli": p} for q, p in obs.items()],
                    }
                    for i, obs in enumerate(terms)
                ],
            }
            connected.append((case, ansatz_expectations(n, gates, terms)))

    # --- run ---
    rng = random.Random(manifest["seeds"]["timing_order_seed"])
    # one-qubit warmup, untimed
    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()

    rows = []
    failures = []
    for case in pair_cases:
        expected = np.array([r["expected"] for r in case["requests"]])
        for rep in list(range(args.repeats_pairs)):
            order = list(strategies)
            rng.shuffle(order)
            for s in order:
                requests = build_requests(case)
                backend = QulacsBackend()
                t0 = perf_counter_ns()
                plan = plan_all(requests, strategy=s)
                res = plan.collect(backend=backend)
                dt = perf_counter_ns() - t0
                err = float(np.max(np.abs(np.array(res.values) - expected)))
                if not math.isfinite(err) or err > 1e-10:
                    failures.append({"case": case["name"], "strategy": s, "error": err})
                    continue
                rows.append(
                    {
                        "case": case["name"],
                        "strategy": s,
                        "selected": res.selected_strategy,
                        "repetition": rep,
                        "total_ns": dt,
                        "evolutions": res.stats.evolutions,
                        "max_state_qubits": res.stats.max_state_qubits,
                        "estimated_state_bytes": res.stats.estimated_state_bytes,
                        "max_abs_error": err,
                    }
                )
    for case, expected in connected:
        expected = np.array(expected)
        for rep in list(range(args.repeats_connected)):
            order = list(strategies)
            rng.shuffle(order)
            for s in order:
                requests = build_requests(case)
                backend = QulacsBackend()
                t0 = perf_counter_ns()
                plan = plan_all(requests, strategy=s)
                res = plan.collect(backend=backend)
                dt = perf_counter_ns() - t0
                err = float(np.max(np.abs(np.array(res.values) - expected)))
                if not math.isfinite(err) or err > 1e-10:
                    failures.append({"case": case["name"], "strategy": s, "error": err})
                    continue
                rows.append(
                    {
                        "case": case["name"],
                        "strategy": s,
                        "selected": res.selected_strategy,
                        "repetition": rep,
                        "total_ns": dt,
                        "evolutions": res.stats.evolutions,
                        "max_state_qubits": res.stats.max_state_qubits,
                        "estimated_state_bytes": res.stats.estimated_state_bytes,
                        "max_abs_error": err,
                    }
                )

    summary = []
    for case_name in sorted({r["case"] for r in rows}):
        for s in strategies:
            sub = [r for r in rows if r["case"] == case_name and r["strategy"] == s]
            if not sub:
                continue
            totals = [r["total_ns"] for r in sub]
            summary.append(
                {
                    "case": case_name,
                    "strategy": s,
                    "selected": sub[0]["selected"],
                    "evolutions": sub[0]["evolutions"],
                    "max_state_qubits": sub[0]["max_state_qubits"],
                    "estimated_state_bytes": sub[0]["estimated_state_bytes"],
                    "median_ms": median(totals) / 1e6,
                    "q25_ms": percentile(totals, 0.25) / 1e6,
                    "q75_ms": percentile(totals, 0.75) / 1e6,
                    "max_abs_error": max(r["max_abs_error"] for r in sub),
                    "n": len(sub),
                }
            )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest["manifest_version"],
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source_sha256": source_hash(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "packages": {p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")},
        "command": sys.argv,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"metadata": metadata, "summary": summary, "runs": rows}, indent=2)
    args.output.write_text(payload + "\n")

    print("case  strategy(selected)  median [IQR] ms  evolutions  maxq  err")
    for s in summary:
        print(
            f"{s['case']}  {s['strategy']}({s['selected']})  "
            f"{s['median_ms']:.3f} [{s['q25_ms']:.3f}, {s['q75_ms']:.3f}]  "
            f"{s['evolutions']}  {s['max_state_qubits']}  {s['max_abs_error']:.1e}"
        )
    if failures:
        print(f"FAILURES: {len(failures)}")
    print(f"Saved {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
