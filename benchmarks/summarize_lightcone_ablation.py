#!/usr/bin/env python3
"""Regenerate LightCone ablation tables from the frozen raw timing artifacts."""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "direct"
OUT = ROOT / "paper" / "sections" / "generated"
METHODS = ("lightcone_separate_aer", "lightcone_dedup_aer", "lazyqc")


def geomean(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


def summarize(filename):
    payload = json.loads((RESULTS / filename).read_text())
    grouped = defaultdict(list)
    for row in payload["rows"]:
        grouped[(row["qubits"], row["method"], row["case_id"])].append(row)
    configs = defaultdict(lambda: defaultdict(list))
    for (n, method, _), rows in grouped.items():
        configs[(n, method)]["analysis_ms"].append(median(r["analysis_ns"] for r in rows) / 1e6)
        configs[(n, method)]["execution_ms"].append(median(r["execution_ns"] for r in rows) / 1e6)
        configs[(n, method)]["total_ms"].append(median(r["total_ns"] for r in rows) / 1e6)
        configs[(n, method)]["tasks"].extend(r["tasks"] for r in rows)
        configs[(n, method)]["min_width"].extend(r["min_width"] for r in rows)
        configs[(n, method)]["max_width"].extend(r["max_width"] for r in rows)
    table = []
    for (n, method), values in sorted(configs.items()):
        table.append(
            {
                "qubits": n,
                "method": method,
                "analysis_ms": geomean(values["analysis_ms"]),
                "execution_ms": geomean(values["execution_ms"]),
                "total_ms": geomean(values["total_ms"]),
                "tasks": max(values["tasks"]),
                "min_width": min(values["min_width"]),
                "max_width": max(values["max_width"]),
            }
        )
    summary = {
        "source": f"results/direct/{filename}",
        "source_sha256": payload["metadata"]["source_sha256"],
        "manifest_sha256": payload["metadata"]["manifest_sha256"],
        "failures": payload["failures"],
        "max_abs_error": max(row["max_abs_error"] for row in payload["rows"]),
        "cases": payload["cases"],
        "configs": table,
    }
    if payload["failures"]:
        raise ValueError(f"Cannot summarize failed benchmark: {payload['failures']}")
    return summary


def config(summary, n, method):
    return next(c for c in summary["configs"] if c["qubits"] == n and c["method"] == method)


def latex(forward, jacobian):
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Workload & Requests & LC tasks S/D & Lazy tasks & LC analysis & "
        r"Lazy analysis & LC total & Lazy total\\",
        r"\midrule",
    ]
    selected = [("Forward", forward, n) for n in (12, 28, 128)] + [
        ("Jacobian", jacobian, n) for n in (12, 20)
    ]
    for label, summary, n in selected:
        separate = config(summary, n, "lightcone_separate_aer")
        dedup = config(summary, n, "lightcone_dedup_aer")
        lazy = config(summary, n, "lazyqc")
        requests = (
            next(c["requests"] for c in summary["cases"] if c["qubits"] == n)
            if label == "Jacobian"
            else n
        )
        name = f"{label}, $n={n}$"
        lines.append(
            f"{name} & {requests} & {separate['tasks']}/{dedup['tasks']} & "
            f"{lazy['tasks']} & {dedup['analysis_ms']:.1f} & {lazy['analysis_ms']:.2f} & "
            f"{dedup['total_ms']:.1f} & {lazy['total_ms']:.2f}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lightcone_ablation.tex").write_text(text)
    return text


def markdown(forward, jacobian):
    lines = [
        "# Qiskit LightCone + Aer ablation",
        "",
        "All times are milliseconds. LC S/D means per-output LightCone circuits "
        "without/with exact reduced-circuit deduplication. LazyQC uses auto planning "
        "and single-thread Qulacs; LC uses one Aer GPU job per call.",
        "",
        "| Workload | Qubits | Method | Tasks | Analysis | Execution | Total | "
        "Min width | Max width |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("Forward", forward), ("Jacobian", jacobian)):
        for row in summary["configs"]:
            lines.append(
                f"| {label} | {row['qubits']} | {row['method']} | {row['tasks']} | "
                f"{row['analysis_ms']:.3f} | {row['execution_ms']:.3f} | "
                f"{row['total_ms']:.3f} | {row['min_width']} | {row['max_width']} |"
            )
    lines += [
        "",
        f"Maximum absolute errors: forward {forward['max_abs_error']:.2e}; "
        f"Jacobian {jacobian['max_abs_error']:.2e}. No failures.",
        "",
        "Qiskit's pass is commutation-aware; LazyQC uses conservative wire-support "
        "cones. Their reduced circuits are therefore not asserted to be identical.",
    ]
    text = "\n".join(lines) + "\n"
    (RESULTS / "lightcone_ablation_SUMMARY.md").write_text(text)
    return text


def main():
    forward = summarize("lightcone_ablation.json")
    jacobian = summarize("lightcone_jacobian_ablation.json")
    (RESULTS / "lightcone_ablation_summary.json").write_text(
        json.dumps({"forward": forward, "jacobian": jacobian}, indent=2) + "\n"
    )
    print(latex(forward, jacobian))
    print(markdown(forward, jacobian))


if __name__ == "__main__":
    main()
