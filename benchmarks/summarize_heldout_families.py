#!/usr/bin/env python3
"""Apply the frozen analysis plan of the held-out-family QML screen.

Reads the per-session runner outputs, recomputes every ratio per circuit
instance and writes a JSON summary plus Markdown tables. All definitions come
from ``analysis_plan`` in ``benchmarks/qml_heldout_family_manifest.json``.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import median

from benchmark_support import percentile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "qml_heldout_family_manifest.json"
FORCED = ("full", "separate", "grouped")


def bootstrap_median(values, seed=9701, draws=10000):
    """Percentile bootstrap 95% interval of the median, resampling instances."""
    rng = random.Random(seed)
    samples = sorted(median(rng.choices(values, k=len(values))) for _ in range(draws))
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def stratum(case):
    ratio = case["grouped_max_width"] / case["qubits"]
    width = "narrow" if ratio < 0.5 else "intermediate" if ratio < 0.9 else "saturated"
    overlap = case["grouped_tasks"] <= 0.5 * case["separate_tasks"]
    return width, overlap


def lifetime(setup, collection, horizon):
    return setup + horizon * collection


def instance_metrics(case_rows, horizons):
    """Per-instance lifetimes, regrets and ratios from all rows of one case."""
    by_method = defaultdict(list)
    for row in case_rows:
        by_method[row["strategy"]].append(row)
    sessions = sorted({row["session"] for row in case_rows})

    def setup_and_collection(method, session=None):
        rows = [r for r in by_method[method] if session is None or r["session"] == session]
        setups = {r["session"]: r["planning_ns"] for r in rows}
        return median(setups.values()), median(r["collection_ns"] for r in rows)

    pooled = {m: setup_and_collection(m) for m in (*FORCED, "auto", "direct")}
    metrics = {}
    for horizon in horizons:
        lives = {m: lifetime(*pooled[m], horizon) for m in pooled}
        best = min(FORCED, key=lives.get)
        cross = []
        for train, test in ((sessions[0], sessions[-1]), (sessions[-1], sessions[0])):
            fit = {m: lifetime(*setup_and_collection(m, train), horizon) for m in FORCED}
            pick = min(FORCED, key=fit.get)
            held = {m: lifetime(*setup_and_collection(m, test), horizon) for m in (pick, "auto")}
            cross.append(held["auto"] / held[pick])
        session_direct = [
            lifetime(*setup_and_collection("auto", s), horizon)
            / lifetime(*setup_and_collection("direct", s), horizon)
            for s in sessions
        ]
        metrics[horizon] = {
            "lifetimes_ns": lives,
            "best_forced": best,
            "selection_regret": lives["auto"] / lives[best],
            "cross_validated_regret": math.sqrt(cross[0] * cross[1]),
            "direct_ratio": lives["auto"] / lives["direct"],
            "grouped_over_separate": lives["grouped"] / lives["separate"],
            "grouped_over_full": lives["grouped"] / lives["full"],
            "session_direct_ratios": session_direct,
        }
    return metrics


def describe(values):
    return {
        "median": median(values),
        "interval": bootstrap_median(values) if len(values) > 1 else None,
        "p90": percentile(values, 0.9),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    plan = manifest["analysis_plan"]
    horizons = plan["reuse_horizons"]
    stem = args.manifest.stem.removesuffix("_manifest")
    inputs = args.inputs or [
        ROOT / "results/qml_workloads" / f"{stem}_session{s}.json" for s in manifest["sessions"]
    ]
    payloads = [json.loads(path.read_text()) for path in inputs]
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    for path, payload in zip(inputs, payloads, strict=True):
        if payload["metadata"]["manifest_sha256"] != digest:
            raise SystemExit(f"{path} was produced from a different manifest")
    failures = [f for p in payloads for f in p["metadata"]["failures"]]
    rows = [row for payload in payloads for row in payload["runs"]]

    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    instances = []
    for case_id, case_rows in sorted(cases.items()):
        case = case_rows[0]
        width, overlap = stratum(case)
        instances.append(
            {
                "case_id": case_id,
                **{
                    k: case[k]
                    for k in (
                        "family",
                        "qubits",
                        "depth",
                        "pattern",
                        "seed_offset",
                        "grouped_max_width",
                        "grouped_tasks",
                        "separate_tasks",
                        "auto_strategy",
                    )
                },
                "cone_stratum": width,
                "high_overlap": overlap,
                "max_abs_error": max(r["max_abs_error"] for r in case_rows),
                "metrics": instance_metrics(case_rows, horizons),
            }
        )

    def group(selector):
        return [i for i in instances if selector(i)]

    def block(selected):
        out = {"instances": len(selected)}
        if not selected:
            return out
        out["auto_choices"] = dict(
            sorted(
                {
                    s: sum(i["auto_strategy"] == s for i in selected)
                    for s in {i["auto_strategy"] for i in selected}
                }.items()
            )
        )
        for horizon in horizons:
            values = [i["metrics"][horizon] for i in selected]
            out[f"R{horizon}"] = {
                key: describe([v[key] for v in values])
                for key in (
                    "selection_regret",
                    "cross_validated_regret",
                    "direct_ratio",
                    "grouped_over_separate",
                    "grouped_over_full",
                )
            }
        return out

    strata = {}
    for family in manifest["families"]:
        for width in ("narrow", "intermediate", "saturated"):
            for overlap in (True, False):
                selected = group(
                    lambda i: i["family"] == family
                    and i["cone_stratum"] == width
                    and i["high_overlap"] == overlap
                )
                if selected:
                    strata[f"{family}|{width}|{'high' if overlap else 'low'}"] = block(selected)
    patterns = {}
    for family in manifest["families"]:
        for pattern in manifest["patterns"]:
            selected = group(lambda i: i["family"] == family and i["pattern"] == pattern)
            if selected:
                patterns[f"{family}|{pattern}"] = block(selected)
    overall = block(instances)
    primary = block(group(lambda i: i["cone_stratum"] == "narrow" and i["high_overlap"]))
    saturated = block(group(lambda i: i["cone_stratum"] == "saturated"))

    def criterion(selected, key, threshold, upper_below_one=False):
        stat = selected.get("R1", {}).get(key)
        passed = bool(stat and stat["median"] <= threshold)
        if passed and upper_below_one:
            passed = stat["interval"] is not None and stat["interval"][1] < 1
        return {
            "instances": selected["instances"],
            "median": stat and stat["median"],
            "interval": stat and stat["interval"],
            "pass": passed,
        }

    criteria = {
        "C1_direct_improvement": criterion(primary, "direct_ratio", 0.90, upper_below_one=True),
        "C2_selection_regret": criterion(overall, "cross_validated_regret", 1.10),
        "C3_saturation_guard": criterion(saturated, "direct_ratio", 1.10),
    }
    session_spread = [
        abs(math.log(s[0] / s[1]))
        for i in instances
        for s in [i["metrics"][1]["session_direct_ratios"]]
        if len(s) == 2
    ]
    summary = {
        "metadata": {
            "manifest_sha256": digest,
            "inputs": [str(p.resolve().relative_to(ROOT, walk_up=True)) for p in inputs],
            "sessions": [p["metadata"]["session"] for p in payloads],
            "runner_sha256": sorted({p["metadata"]["source_sha256"] for p in payloads}),
            "instances": len(instances),
            "rows": len(rows),
            "failures": failures,
            "max_abs_error": max(i["max_abs_error"] for i in instances),
            "median_abs_log_session_difference_direct_ratio_R1": median(session_spread),
        },
        "criteria": criteria,
        "overall": overall,
        "primary_stratum": primary,
        "saturated_stratum": saturated,
        "strata": strata,
        "patterns": patterns,
        "instances": instances,
    }
    output = args.output or ROOT / "results/qml_workloads" / f"{stem}_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    output.with_suffix(".md").write_text(tables(summary, horizons))
    for name, result in criteria.items():
        print(name, result)
    print(f"instances={len(instances)} failures={len(failures)} output={output}")


def fmt(stat, digits=3):
    if stat["interval"] is None:
        return f"{stat['median']:.{digits}f}"
    low, high = stat["interval"]
    return f"{stat['median']:.{digits}f} ({low:.{digits}f}–{high:.{digits}f})"


def tables(summary, horizons):
    lines = ["# Held-out family screen tables", ""]
    lines += [
        "| Criterion | Instances | Median | 95% interval | Pass |",
        "|---|---:|---:|---|---|",
    ]
    for name, c in summary["criteria"].items():
        interval = c["interval"] and f"{c['interval'][0]:.3f}–{c['interval'][1]:.3f}"
        median_text = "n/a" if c["median"] is None else f"{c['median']:.3f}"
        lines.append(
            f"| {name} | {c['instances']} | {median_text} | {interval} | {c['pass']} |"
        )
    lines += [
        "",
        "## Strata (R=1 unless stated)",
        "",
        "| Family | Cone | Overlap | Instances | Auto choices | CV regret | Auto / direct R=1 "
        "| Auto / direct R=32 | Grouped / separate | Grouped / full |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for key, b in summary["strata"].items():
        family, width, overlap = key.split("|")
        choices = ", ".join(f"{k} {v}" for k, v in b["auto_choices"].items())
        r1, r32 = b["R1"], b[f"R{horizons[-1]}"]
        lines.append(
            f"| {family} | {width} | {overlap} | {b['instances']} | {choices} | "
            f"{fmt(r1['cross_validated_regret'])} | {fmt(r1['direct_ratio'])} | "
            f"{fmt(r32['direct_ratio'])} | {fmt(r1['grouped_over_separate'])} | "
            f"{fmt(r1['grouped_over_full'])} |"
        )
    lines += [
        "",
        "## Tails over all instances",
        "",
        "| Horizon | Regret median | Regret p90 | Regret max | CV regret median "
        "| Direct ratio median | Direct p90 | Direct max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon in horizons:
        o = summary["overall"][f"R{horizon}"]
        s, c, d = o["selection_regret"], o["cross_validated_regret"], o["direct_ratio"]
        lines.append(
            f"| {horizon} | {s['median']:.3f} | {s['p90']:.3f} | {s['max']:.3f} | "
            f"{c['median']:.3f} | {d['median']:.3f} | {d['p90']:.3f} | {d['max']:.3f} |"
        )
    lines += [
        "",
        "## Output patterns (R=1)",
        "",
        "| Family | Pattern | Instances | Auto choices | CV regret | Auto / direct |",
        "|---|---|---:|---|---:|---:|",
    ]
    for key, b in summary["patterns"].items():
        family, pattern = key.split("|")
        choices = ", ".join(f"{k} {v}" for k, v in b["auto_choices"].items())
        lines.append(
            f"| {family} | {pattern} | {b['instances']} | {choices} | "
            f"{fmt(b['R1']['cross_validated_regret'])} | {fmt(b['R1']['direct_ratio'])} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
