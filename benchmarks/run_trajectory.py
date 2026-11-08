#!/usr/bin/env python3
"""RQ3 trajectory validation: structural-hint replay across VQE bindings.

Seed binding 0 plans fully; later bindings plan both fully (fidelity
reference) and hinted (template replay). Checks oracle values, decision
fidelity (hinted tasks/scores == full), hit rate, and planning speedup.
v1 scope: optimization="none" (topology stable by construction).

Writes results/workload_manifest/v2/trajectory.json.
"""

import argparse
import importlib.metadata
import json
import math
import platform
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from benchmark_support import cpu_name, pin_single_thread, source_hash
from core_checkout import FIXTURES, REFERENCE

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", nargs="+", default=["qaoa_path", "qaoa_star"])
    ap.add_argument("--bindings", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--optimization", default="none")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "results/workload_manifest/v2/trajectory.json"
    )
    args = ap.parse_args()

    pin_single_thread()

    from lazyqc import CutOptions, Parameter, Program, calibrate_batch
    from lazyqc.batch_planner import plan_batch
    from lazyqc.fixtures import load_fixture
    from lazyqc.ir import Gate

    sys.path.insert(0, str(REFERENCE))
    from reference_numpy import expectation, statevector

    from lazyqc.cutting import cx_branches

    cx_branches()

    fixture, cases = load_fixture(FIXTURES / "p4_p5.json")
    cases = [c for c in cases if c["name"] in args.cases]
    if set(args.cases) != {c["name"] for c in cases}:
        raise ValueError("Unknown case requested")
    tolerance = fixture["absolute_tolerance"]

    print("Calibrating...", flush=True)
    model, training = calibrate_batch(widths=(1, 2, 3, 4, 6, 8), repeats=5)

    summary, failures = [], []
    for case in cases:
        fw = case["fragment_width"]
        opts = CutOptions(
            max_fragment_qubits=fw, max_cuts=case["max_cuts"], **fixture["cut_limits"]
        )
        kwargs = {
            "strategy": "auto",
            "optimization": args.optimization,
            "cost_model": model,
            "workers": 1,
            "max_state_qubits": fw,
            "max_state_bytes": 16 * 2**fw,
            "total_state_bytes": 32 * 2**fw,
        }
        ops, pidx = [], 0
        for op in case["operations"]:
            if "angle_radians" in op:
                ops.append(Gate(op["gate"], tuple(op["qubits"]), Parameter(f"g{pidx}")))
                pidx += 1
            else:
                ops.append(Gate(op["gate"], tuple(op["qubits"])))
        base = Program(case["qubits"], tuple(ops))
        obs = [[(f["qubit"], f["pauli"]) for f in r["observable"]] for r in case["requests"]]
        rng = random.Random(args.seed + fw)
        binds = [{f"g{i}": rng.uniform(-1, 1) for i in range(pidx)} for _ in range(args.bindings)]
        progs = [base.bind(**b) for b in binds]

        case_rows = []
        template = None
        for bi, prog in enumerate(progs):
            requests = [prog.expectation(dict(o)) for o in obs]
            state = statevector(
                prog.n_qubits, [(g.name, g.qubits, g.angle) for g in prog.resolved_operations()]
            )
            expected = tuple(expectation(state, r.observable.factors) for r in requests)
            t0 = perf_counter_ns()
            full = plan_batch(requests, cutting=replace(opts, search="joint"), **kwargs)
            full_ms = (perf_counter_ns() - t0) / 1e6
            if template is None:
                template = full.explain()["search"]["structural_template"]
            t0 = perf_counter_ns()
            hinted = plan_batch(
                requests,
                cutting=replace(opts, search="joint"),
                _structural_hints=template,
                **kwargs,
            )
            hint_ms = (perf_counter_ns() - t0) / 1e6
            analysis = dict(hinted.analysis_stats)
            values = hinted.collect().values
            error = max(abs(a - b) for a, b in zip(values, expected, strict=True))
            if not math.isfinite(error) or error > tolerance:
                failures.append({"case": case["name"], "binding": bi, "error": error})
                continue
            case_rows.append(
                {
                    "binding": bi,
                    "full_tasks": len(full.selected.tasks),
                    "hint_tasks": len(hinted.selected.tasks),
                    "scores_equal": full.selected.score == hinted.selected.score,
                    "full_ms": full_ms,
                    "hint_ms": hint_ms,
                    "hit": analysis.get("structural_hit", 0),
                    "miss": analysis.get("structural_miss", 0),
                    "max_abs_error": error,
                }
            )
            print(
                f"{case['name']} b{bi}: full={len(full.selected.tasks)}t/{full_ms:.0f}ms "
                f"hint={len(hinted.selected.tasks)}t/{hint_ms:.1f}ms "
                f"hit={analysis.get('structural_hit', 0)} err={error:.1e}"
            )
        fidel = all(
            r["full_tasks"] == r["hint_tasks"] and r["scores_equal"] for r in case_rows
        )
        full_sorted = sorted(r["full_ms"] for r in case_rows)
        hint_sorted = sorted(r["hint_ms"] for r in case_rows)
        summary.append(
            {
                "case": case["name"],
                "bindings": len(case_rows),
                "hit_rate": sum(r["hit"] for r in case_rows) / max(1, len(case_rows)),
                "decision_fidelity": fidel,
                "median_full_ms": full_sorted[len(case_rows) // 2] if case_rows else None,
                "median_hint_ms": hint_sorted[len(case_rows) // 2] if case_rows else None,
                "max_abs_error": max((r["max_abs_error"] for r in case_rows), default=None),
            }
        )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": source_hash(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "packages": {p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")},
        "command": sys.argv,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "summary": summary}, indent=2) + "\n")
    print("summary:", json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
