#!/usr/bin/env python3
"""Summarize frozen QML workload measurements by request pattern and seed split."""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def geomean(values):
    values = tuple(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_ratio_intervals(case_ids, grouped, samples=5000, seed=20261026):
    """Paired two-level bootstrap: resample cases, then paired timing blocks."""
    rng = random.Random(seed)
    distributions = {"grouped_over_separate": [], "grouped_over_full": []}
    for _ in range(samples):
        selected_ids = [rng.choice(case_ids) for _ in case_ids]
        case_ratios = {name: [] for name in distributions}
        for case_id in selected_ids:
            policies = grouped[case_id]
            repetition_ids = sorted(row["repetition"] for row in policies["full"])
            sampled_repetitions = [rng.choice(repetition_ids) for _ in repetition_ids]
            medians = {
                strategy: median(
                    next(row["total_ns"] for row in policies[strategy] if row["repetition"] == rep)
                    for rep in sampled_repetitions
                )
                for strategy in ("full", "separate", "grouped")
            }
            case_ratios["grouped_over_separate"].append(
                medians["grouped"] / medians["separate"]
            )
            case_ratios["grouped_over_full"].append(medians["grouped"] / medians["full"])
        for name, ratios in case_ratios.items():
            distributions[name].append(geomean(ratios))

    intervals = {}
    for name, values in distributions.items():
        ordered = sorted(values)
        intervals[name] = [ordered[int(0.025 * samples)], ordered[int(0.975 * samples)]]
    return intervals


def summarize(payload):
    strategies = ("full", "separate", "grouped", "auto")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in payload["runs"]:
        grouped[row["case_id"]][row["strategy"]].append(row)

    cases = []
    for case_id, policies in sorted(grouped.items()):
        representative = policies["full"][0]
        medians = {
            strategy: median(row["total_ns"] for row in policies[strategy])
            for strategy in strategies
        }
        selected = policies["auto"][0]["selected_strategy"]
        best_forced = min(medians[s] for s in strategies[:3])
        cases.append(
            {
                "case_id": case_id,
                "family": representative["family"],
                "qubits": representative["qubits"],
                "depth": representative["depth"],
                "pattern": representative["pattern"],
                "seed_offset": representative["seed_offset"],
                "split": (
                    "heldout"
                    if representative["seed_offset"] in (202, 303)
                    else "development"
                ),
                "median_total_ns": medians,
                "grouped_over_separate": medians["grouped"] / medians["separate"],
                "grouped_over_full": medians["grouped"] / medians["full"],
                "auto_over_best_forced": medians["auto"] / best_forced,
                "auto_choice_proxy_over_best": medians[selected] / best_forced,
                "auto_selected_strategy": selected,
            }
        )

    aggregates = []
    for split in ("development", "heldout"):
        for pattern in sorted({case["pattern"] for case in cases}):
            selected_cases = [
                case for case in cases if case["split"] == split and case["pattern"] == pattern
            ]
            if not selected_cases:
                continue
            aggregates.append(
                {
                    "split": split,
                    "pattern": pattern,
                    "case_count": len(selected_cases),
                    "geomean_grouped_over_separate": geomean(
                        case["grouped_over_separate"] for case in selected_cases
                    ),
                    "grouped_beats_separate_cases": sum(
                        case["grouped_over_separate"] < 1 for case in selected_cases
                    ),
                    "geomean_grouped_over_full": geomean(
                        case["grouped_over_full"] for case in selected_cases
                    ),
                    "grouped_beats_full_cases": sum(
                        case["grouped_over_full"] < 1 for case in selected_cases
                    ),
                    "geomean_auto_over_best_forced": geomean(
                        case["auto_over_best_forced"] for case in selected_cases
                    ),
                    "auto_beats_best_forced_cases": sum(
                        case["auto_over_best_forced"] <= 1 for case in selected_cases
                    ),
                    "geomean_auto_choice_proxy_over_best": geomean(
                        case["auto_choice_proxy_over_best"] for case in selected_cases
                    ),
                    "auto_selected_strategy_counts": dict(
                        Counter(case["auto_selected_strategy"] for case in selected_cases)
                    ),
                    "bootstrap_ci95": bootstrap_ratio_intervals(
                        [case["case_id"] for case in selected_cases], grouped
                    ),
                }
            )

    return {
        "source_artifact": "results/qml_workloads/v1/evaluation.json",
        "interpretation": (
            "Descriptive ratios of per-case medians. Seed offsets 202/303 are held-out "
            "circuit instances; timing repetitions are not independent workloads. "
            "The best-forced reference is the minimum measured median among full, "
            "separate, and grouped on the same case."
        ),
        "raw_run_count": len(payload["runs"]),
        "validation_failure_count": len(payload["metadata"]["failures"]),
        "max_abs_error": max(row["max_abs_error"] for row in payload["runs"]),
        "cases": cases,
        "aggregates": aggregates,
    }


def render_markdown(summary):
    lines = [
        "# QML Workload Sharing Results",
        "",
        (
            f"Raw timed rows: {summary['raw_run_count']}; validation failures: "
            f"{summary['validation_failure_count']}; maximum absolute oracle error: "
            f"{summary['max_abs_error']:.3e}."
        ),
        "",
        (
            "Ratios use each case's median end-to-end time (planning plus collection), "
            "then take a geometric mean across cases. A ratio below 1 favors the "
            "numerator policy. The forced-policy reference is the best measured "
            "median among full, separate, and grouped for that same case; it is "
            "descriptive, not an independent prospective oracle."
        ),
        "",
        (
            "Seed offsets 0/101 are development instances; 202/303 are held out. "
            "These are random instances from the same generators, not held-out "
            "circuit families. Repetitions estimate timing variability within a case."
        ),
        "",
        (
            "| Split | Request pattern | Cases | Grouped / separate | Grouped wins | "
            "Grouped/separate 95% CI | Grouped / full | Grouped wins | "
            "Auto / best forced | Auto wins | Auto choice proxy / best |"
        ),
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {row['split']} | {row['pattern']} | {row['case_count']} | "
            f"{row['geomean_grouped_over_separate']:.3f} | "
            f"{row['grouped_beats_separate_cases']} | "
            f"[{row['bootstrap_ci95']['grouped_over_separate'][0]:.3f}, "
            f"{row['bootstrap_ci95']['grouped_over_separate'][1]:.3f}] | "
            f"{row['geomean_grouped_over_full']:.3f} | "
            f"{row['grouped_beats_full_cases']} | "
            f"{row['geomean_auto_over_best_forced']:.3f} | "
            f"{row['auto_beats_best_forced_cases']} | "
            f"{row['geomean_auto_choice_proxy_over_best']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "`Auto choice proxy / best` compares the forced policy selected by auto "
                "with the best forced median, excluding auto's planning time. "
                "`Auto / best forced` includes auto planning and collection. Neither "
                "comparison establishes a family-general selector."
            ),
        (
            "The 95% interval resamples held-out cases and paired timing repetitions "
            "within each case. It is descriptive for these generated instances, not "
            "evidence of transfer to new QML families."
        ),
            "",
            "See `evaluation.json` for every raw repetition and environment/source fingerprints.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/qml_workloads/v1/evaluation.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qml_workloads/v1")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    result = summarize(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "SUMMARY.md").write_text(render_markdown(result))
    print(render_markdown(result))


if __name__ == "__main__":
    main()
