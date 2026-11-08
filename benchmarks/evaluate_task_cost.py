"""Rank full, separate and grouped plans with and without fixed costs per task and per gate (issue #2).

Reads task_cost_eval_manifest.json (frozen before the constant was fitted), times the three
forced plans of every fresh instance in two randomized sessions, and compares the plan each
rule selects with the fastest one. A rule is a function of the candidates' work W and task
count T only, so one set of timings evaluates them all. Regret is cross-validated: the fastest
plan is chosen on one session and the rule is timed on the other, averaged both ways.

    uv run --locked python benchmarks/evaluate_task_cost.py
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from pathlib import Path

os.environ.update(
    QULACS_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
)
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]

import numpy as np  # noqa: E402
from qml_workload_support import make_batch  # noqa: E402

from lazyqc import plan_all  # noqa: E402
from lazyqc.plan import GATE_COST  # noqa: E402

MANIFEST = HERE / "task_cost_eval_manifest_v2.json"
FORCED = ("full", "separate", "grouped")


def timed(plan, repeats):
    plan.collect()
    return statistics.median(plan.collect().stats.execution_ns for _ in range(repeats))


def work(tasks):
    """Raw work score (no fixed costs), the task count and the gate count."""
    return (
        sum((1 << t.width) * (1 + len(t.operations) + len(t.outputs)) for t in tasks),
        len(tasks),
        sum(len(t.operations) for t in tasks),
    )


def rules(fits):
    """Selection rules as functions of (W, T, G) per strategy; the fits come from calibration."""
    c_task = fits["task"]["work_units"]["T"]
    c_gate = fits["gate"]["work_units"]["G"]
    both = fits["task_and_gate"]["work_units"]

    def by(c0, c1):
        return lambda c, n, w: min(
            FORCED, key=lambda s: (c[s][0] + c0 * c[s][1] + c1 * c[s][2], c[s][1])
        )

    return {
        "current": by(0.0, 0.0),
        "task_cost": by(c_task, 0.0),
        "gate_cost": by(0.0, c_gate),
        "task_and_gate_cost": by(both["T"], both["G"]),
        "library": by(0.0, GATE_COST),
        "width_n": lambda c, n, w: "grouped" if w < n else "full",
        "width_half": lambda c, n, w: "grouped" if 2 * w <= n else "full",
        "width_10": lambda c, n, w: "grouped" if w <= 10 else "full",
    }


def summarize(values):
    values = sorted(values)
    p90 = values[min(len(values) - 1, int(0.9 * len(values)))]
    return {
        "median": statistics.median(values),
        "p90": p90,
        "geomean": float(np.exp(np.mean(np.log(values)))),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/task_cost/evaluation.json")
    parser.add_argument("--calibration", default="results/task_cost/calibration.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    calibration = json.loads((HERE.parent / args.calibration).read_text())["summary"]
    if calibration["manifest_sha256"] != hashlib.sha256(MANIFEST.read_bytes()).hexdigest():
        raise SystemExit("The calibration was fitted against a different manifest")
    rule_set = rules(calibration["fits"])
    order_rng = random.Random(manifest["randomized_order_seed"])
    repeats = manifest["collections_per_plan"]
    offsets = manifest["seed_offsets"][:1] if args.smoke else manifest["seed_offsets"]
    rows = []
    for group, families in manifest["families"].items():
        for family in families:
            for n in manifest["qubits"]:
                for depth in manifest["depths"]:
                    for pattern in manifest["patterns"]:
                        for offset in offsets:
                            batch = make_batch(family, n, depth, pattern, manifest["base_seed"] + offset)
                            plans = {s: plan_all(batch, strategy=s, analysis="always") for s in FORCED}
                            cand = {s: work(plans[s].selected.tasks) for s in FORCED}
                            width = plans["grouped"].selected.max_width
                            times = {s: [] for s in FORCED}
                            for _ in manifest["sessions"]:
                                for s in order_rng.sample(FORCED, len(FORCED)):
                                    times[s].append(timed(plans[s], repeats))
                            rows.append(
                                {
                                    "group": group,
                                    "family": family,
                                    "qubits": n,
                                    "depth": depth,
                                    "pattern": pattern,
                                    "seed_offset": offset,
                                    "grouped_width": width,
                                    "cone_ratio": width / n,
                                    "work": {s: cand[s][0] for s in FORCED},
                                    "tasks": {s: cand[s][1] for s in FORCED},
                                    "gates": {s: cand[s][2] for s in FORCED},
                                    "time_ns": times,
                                }
                            )
        print(group, "done", len(rows), flush=True)

    def regret(row, pick):
        """Cross-validated ratio: the pick's time over the best plan, each session in turn."""
        ratios = []
        for chosen, measured in ((0, 1), (1, 0)):
            best = min(FORCED, key=lambda s: row["time_ns"][s][chosen])
            ratios.append(row["time_ns"][pick][measured] / row["time_ns"][best][measured])
        return sum(ratios) / 2

    summary = {}
    strata = {
        "small_tree, 6-8 qubits": lambda r: r["group"] == "small_tree" and r["qubits"] <= 8,
        "small_tree, all": lambda r: r["group"] == "small_tree",
        "narrow_regime (cone ratio < 0.5)": lambda r: r["cone_ratio"] < 0.5,
        "saturated (cone ratio >= 0.9)": lambda r: r["cone_ratio"] >= 0.9,
        "all": lambda r: True,
    }
    picks = {}
    for name, rule in rule_set.items():
        picks[name] = [
            rule(
                {s: (r["work"][s], r["tasks"][s], r["gates"][s]) for s in FORCED},
                r["qubits"],
                r["grouped_width"],
            )
            for r in rows
        ]
    for stratum, keep in strata.items():
        summary[stratum] = {}
        for name in rule_set:
            values = [regret(r, p) for r, p in zip(rows, picks[name]) if keep(r)]
            if values:
                summary[stratum][name] = summarize(values)
    reduction = {
        name: {
            "cone_plan_share_when_narrow": float(
                np.mean([p != "full" for r, p in zip(rows, picks[name]) if r["cone_ratio"] < 0.5])
            )
        }
        for name in rule_set
    }
    out = HERE.parent / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                "calibration_fits": calibration["fits"],
                "library_gate_cost": GATE_COST,
                "summary": summary,
                "reduction": reduction,
                "rows": rows,
                "picks": picks,
            },
            indent=1,
        )
    )
    for stratum, table in summary.items():
        print(f"\n{stratum}")
        for name, s in table.items():
            print(f"  {name:18s} median {s['median']:.3f}  p90 {s['p90']:.3f}  geomean {s['geomean']:.3f}  (n={s['n']})")
    print("\nshare of narrow-cone instances that keep a cone plan")
    for name, s in reduction.items():
        print(f"  {name:18s} {s['cone_plan_share_when_narrow']:.3f}")


if __name__ == "__main__":
    main()
