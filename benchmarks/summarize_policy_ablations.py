#!/usr/bin/env python3
"""Compare LazyQC's decisions with simple policies on the frozen held-out screen.

This is a post-hoc policy ablation: it reuses the two existing session files and
performs no new timing. Simple structural selectors are charged the same analysis
cost as ``auto`` plus the measured collection time of the plan they choose. The
reference is selected on one session and evaluated on the other, exactly as in
the frozen held-out-family analysis. Width thresholds are illustrative policies,
not coefficients tuned on this screen.

    uv run --locked python benchmarks/summarize_policy_ablations.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmark_support import percentile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "qml_workloads"
OUT = ROOT / "paper" / "sections" / "generated"
FORCED = ("full", "separate", "grouped")


def geomean(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


def analyze():
    payloads = [
        json.loads((RESULTS / f"qml_heldout_family_session{s}.json").read_text()) for s in (0, 1)
    ]
    rows = [row for payload in payloads for row in payload["runs"]]
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    if len(by_case) != 480:
        raise ValueError(f"Expected the frozen 480-case screen, found {len(by_case)} cases")

    policies = (
        "LazyQC auto",
        "Always full (dense)",
        "Always separate",
        "Always grouped",
        "Reduce iff w < n",
        "Reduce iff w <= n/2",
        "Reduce iff w <= 10",
    )
    results = {name: [] for name in policies}
    counts = {name: {"false_positive": 0, "false_negative": 0} for name in policies}
    best_counts = {"full": 0, "cone": 0}

    for case_id, case_rows in sorted(by_case.items()):
        case = case_rows[0]
        n, w = case["qubits"], case["grouped_max_width"]
        width_ratio = w / n
        stratum = (
            "narrow" if width_ratio < 0.5 else "intermediate" if width_ratio < 0.9 else "saturated"
        )

        def setup_and_collection(strategy, session):
            selected = [
                r for r in case_rows if r["strategy"] == strategy and r["session"] == session
            ]
            if not selected:
                raise ValueError(f"Missing {case_id}/{strategy}/session{session}")
            return median(r["planning_ns"] for r in selected), median(
                r["collection_ns"] for r in selected
            )

        def lifetime(strategy, session):
            setup, collection = setup_and_collection(strategy, session)
            return setup + collection

        def collection(strategy, session):
            return setup_and_collection(strategy, session)[1]

        def analysis(session):
            return setup_and_collection("auto", session)[0]

        def cost(name, pick, session):
            if name == "LazyQC auto":
                return lifetime("auto", session)
            if name.startswith("Always "):
                return lifetime(pick, session)
            # Same cone-analysis charge as auto; only the selection rule changes.
            return analysis(session) + collection(pick, session)

        pooled = {
            strategy: geomean([lifetime(strategy, session) for session in (0, 1)])
            for strategy in FORCED
        }
        best = min(FORCED, key=pooled.get)
        best_counts["full" if best == "full" else "cone"] += 1
        picks = {
            "LazyQC auto": case["auto_strategy"],
            "Always full (dense)": "full",
            "Always separate": "separate",
            "Always grouped": "grouped",
            "Reduce iff w < n": "grouped" if w < n else "full",
            "Reduce iff w <= n/2": "grouped" if 2 * w <= n else "full",
            "Reduce iff w <= 10": "grouped" if w <= 10 else "full",
        }
        for name, pick in picks.items():
            cross = []
            direct_ratios = []
            for train, test in ((0, 1), (1, 0)):
                reference = min(FORCED, key=lambda s: lifetime(s, train))
                cross.append(cost(name, pick, test) / lifetime(reference, test))
                direct_ratios.append(cost(name, pick, test) / lifetime("direct", test))
            results[name].append(
                {
                    "case_id": case_id,
                    "stratum": stratum,
                    "qubits": n,
                    "regret": geomean(cross),
                    "direct_ratio": geomean(direct_ratios),
                }
            )
            if best == "full" and pick != "full":
                counts[name]["false_positive"] += 1
            if best != "full" and pick == "full":
                counts[name]["false_negative"] += 1

    table = []
    for name in policies:
        entries = results[name]
        table.append(
            {
                "policy": name,
                "median_regret": median(e["regret"] for e in entries),
                "p90_regret": percentile([e["regret"] for e in entries], 0.9),
                "max_regret": max(e["regret"] for e in entries),
                "narrow_direct": median(
                    e["direct_ratio"] for e in entries if e["stratum"] == "narrow"
                ),
                "saturated_direct": median(
                    e["direct_ratio"] for e in entries if e["stratum"] == "saturated"
                ),
                **counts[name],
            }
        )
    payload = {
        "protocol": {
            "source": [
                "results/qml_workloads/qml_heldout_family_session0.json",
                "results/qml_workloads/qml_heldout_family_session1.json",
            ],
            "cases": len(by_case),
            "timing": "No new timings; medians of setup and collection per strategy and session.",
            "cross_validation": (
                "Best forced plan chosen on one session and charged on the other; "
                "geometric mean of both directions."
            ),
            "simple_selector_cost": "Auto analysis time plus selected candidate collection time.",
            "reference": (
                "Best of full, separate and grouped; errors use the pooled best forced plan."
            ),
            "false_positive": (
                "Selected separate/grouped when full was the pooled best forced plan."
            ),
            "false_negative": (
                "Selected full when separate/grouped was the pooled best forced plan."
            ),
            "best_full_cases": best_counts["full"],
            "best_cone_cases": best_counts["cone"],
            "threshold_status": "Illustrative fixed rules, not fitted on these held-out instances.",
        },
        "policies": table,
    }
    (RESULTS / "policy_ablations.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def latex(payload):
    labels = {
        "LazyQC auto": "LazyQC \\texttt{auto}",
        "Always full (dense)": "Always full",
        "Always separate": "Always separate (per-output cone)",
        "Always grouped": "Always grouped",
        "Reduce iff w < n": "Width rule $w<n$",
        "Reduce iff w <= n/2": "Width rule $2w\\leq n$",
        "Reduce iff w <= 10": "Width rule $w\\leq10$",
    }
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Policy & Med. & p90 & Narrow & Saturated & FP & FN\\",
        r"\midrule",
    ]
    for row in payload["policies"]:
        lines.append(
            f"{labels[row['policy']]} & {row['median_regret']:.3f} & "
            f"{row['p90_regret']:.3f} & {row['narrow_direct']:.3f} & "
            f"{row['saturated_direct']:.3f} & {row['false_positive']} & "
            f"{row['false_negative']}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "policy_ablations.tex").write_text(text)
    return text


def markdown(payload):
    protocol = payload["protocol"]
    lines = [
        "# Held-out decision-policy ablation",
        "",
        f"Cases: {protocol['cases']}; full is best in {protocol['best_full_cases']} and "
        f"a cone plan in {protocol['best_cone_cases']}. No new timings were collected.",
        "",
        protocol["simple_selector_cost"],
        "",
        "| Policy | Median CV regret | p90 | Max | Narrow policy/direct | "
        "Saturated policy/direct | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["policies"]:
        lines.append(
            f"| {row['policy']} | {row['median_regret']:.3f} | {row['p90_regret']:.3f} | "
            f"{row['max_regret']:.3f} | {row['narrow_direct']:.3f} | "
            f"{row['saturated_direct']:.3f} | {row['false_positive']} | {row['false_negative']} |"
        )
    lines += [
        "",
        "FP selects a cone plan when full is the pooled best forced plan; "
        "FN selects full when a cone plan is best.",
        f"The denominators are respectively {protocol['best_full_cases']} and "
        f"{protocol['best_cone_cases']}. Structural thresholds "
        "were not fitted on this screen.",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "policy_ablations.md").write_text(text)
    return text


def main():
    payload = analyze()
    print(latex(payload))
    print(markdown(payload))


if __name__ == "__main__":
    main()
