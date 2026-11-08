#!/usr/bin/env python3
"""Generate the paper's generalization tables from saved results.

Reads the held-out-family summary, both grouping-oracle runs, the Iris v3
screen and the Fourier trained-model replay, and writes LaTeX tabulars to
``paper/sections/generated/``: ``heldout_families.tex``,
``regret_decomposition.tex`` and ``trained_models.tex``. Standard library only.
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qml_workloads"
OUT = ROOT / "paper/sections/generated"
FORCED = ("full", "separate", "grouped")


def geomean(values):
    values = list(values)
    return math.exp(sum(map(math.log, values)) / len(values))


def num(value, digits=2):
    return f"{value:.{digits + 1 if abs(value) < 0.1 else digits}f}"


def interval(stat, digits=2):
    low, high = stat["interval"]
    return f"{stat['median']:.{digits}f} ({low:.{digits}f}--{high:.{digits}f})"


ABBREVIATION = {"full": "F", "separate": "S", "grouped": "G"}


def choices(counts):
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{ABBREVIATION[k]}\\,{v}" for k, v in ordered)


def heldout_table():
    summary = json.loads((RESULTS / "qml_heldout_family_summary.json").read_text())
    instances = summary["instances"]

    def r1(instance):
        return instance["metrics"]["1"]

    def row(label, selected):
        counts = defaultdict(int)
        for i in selected:
            counts[i["auto_strategy"]] += 1
        direct = [r1(i)["direct_ratio"] for i in selected]
        regret = [r1(i)["cross_validated_regret"] for i in selected]
        return (
            f"{label} & {len(selected)} & {choices(counts)} & "
            f"{num(median(direct))} & {num(median(regret))}\\\\"
        )

    lines = [
        r"\begin{tabular}{lrlrr}",
        r"\toprule",
        r"Stratum & Cases & Auto choice & Auto/direct & CV regret\\",
        r"\midrule",
    ]
    names = {"tree_mera_like": "Tree", "strongly_entangling": "Strongly ent."}
    lines.append(r"\multicolumn{5}{l}{\emph{Disjoint family/cone strata (480 cases)}}\\")
    for family, width, overlap, label in (
        ("tree_mera_like", "narrow", True, "narrow, shared"),
        ("tree_mera_like", "narrow", False, "narrow, distinct"),
        ("tree_mera_like", "intermediate", None, "intermediate"),
        ("tree_mera_like", "saturated", None, "saturated"),
        ("strongly_entangling", "saturated", None, "saturated"),
    ):
        selected = [
            i
            for i in instances
            if i["family"] == family
            and i["cone_stratum"] == width
            and (overlap is None or i["high_overlap"] == overlap)
        ]
        lines.append(row(f"{names[family]}, {label}", selected))
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{5}{l}{\emph{Width breakdown of the 405 saturated cases "
        r"(overlapping rows)}}\\"
    )
    for qubits in (8, 12, 16, 20):
        selected = [
            i for i in instances if i["cone_stratum"] == "saturated" and i["qubits"] == qubits
        ]
        lines.append(row(f"Saturated, {qubits} qubits", selected))
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\emph{All 480 cases (each counted once)}}\\")
    lines.append(row("All held-out cases", instances))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def regret_rows(path, split_by):
    payload = json.loads(path.read_text())
    groups = defaultdict(list)
    for case in payload["cases"]:
        if split_by == "seed":
            key = "Development seeds" if case["seed_offset"] == 0 else "Held-out seeds"
        else:
            key = {
                "tree_mera_like": "Held-out family: tree",
                "strongly_entangling": "Held-out family: strongly ent.",
            }[case["family"]]
        groups[key].append(case)
    rows = []
    for key, cases in groups.items():

        def generated(case):
            medians = case["policy_medians_ns"]
            return min(medians[s]["collection_ns"] for s in FORCED)

        execution = geomean(c["auto_collection_regret"] for c in cases)
        ranking = geomean(
            c["policy_medians_ns"]["auto"]["collection_ns"] / generated(c) for c in cases
        )
        coverage = geomean(generated(c) / c["best_enumerated_collection_ns"] for c in cases)
        total = geomean(c["auto_total_regret_at_R1"] for c in cases)
        rows.append(
            f"{key} & {len(cases)} & {num(execution)} & {num(ranking)} & "
            f"{num(coverage)} & {num(total)}\\\\"
        )
    return rows


def regret_table():
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Split & Cases & Execution & Ranking & Coverage & Total, $R{=}1$\\",
        r"\midrule",
        *regret_rows(RESULTS / "grouping_oracle.json", "seed"),
        *regret_rows(RESULTS / "grouping_oracle_heldout.json", "family"),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def iris_rows():
    payload = json.loads((RESULTS / "trained_iris_qml_v3.json").read_text())
    lifetimes = defaultdict(lambda: defaultdict(float))
    meta, auto = {}, defaultdict(set)
    for step in payload["steps"]:
        lifetimes[step["case_id"]][step["method"]] += step["step_lifetime_ns"]
        meta[step["case_id"]] = (step["ansatz"], step["qubits"])
        if step["method"] == "auto":
            auto[(step["ansatz"], step["qubits"])].add(step["selected_strategy"])
    groups = defaultdict(list)
    for case_id, values in lifetimes.items():
        groups[meta[case_id]].append(values)
    rows = []
    for ansatz, qubits in sorted(groups, key=lambda k: (k[0] != "brickwork", k[1])):
        cases = groups[(ansatz, qubits)]
        label = f"Iris, {ansatz}, {qubits} qubits"
        rows.append(
            f"{label} & {len(cases)} & "
            f"{', '.join(ABBREVIATION[s] for s in sorted(auto[(ansatz, qubits)]))} & "
            f"{num(median(c['auto'] / c['direct_qulacs'] for c in cases))} & "
            f"{num(median(c['grouped'] / c['direct_qulacs'] for c in cases))}\\\\"
        )
    return rows


def fourier_rows():
    payload = json.loads((RESULTS / "fourier_trained_qml.json").read_text())
    rows = defaultdict(list)
    for row in payload["runs"]:
        if row["batch_size"] == 64:
            rows[(row["dataset"], row["seed"], row["method"])].append(
                row["setup_ns"] + row["collection_ns"]
            )
    per_model = defaultdict(dict)
    choice = defaultdict(set)
    for (dataset, seed, method), values in rows.items():
        per_model[(dataset, seed)][method] = median(values)
    for row in payload["runs"]:
        if row["method"] == "auto":
            choice[row["dataset"]].add(row["selected_strategy"])
    lines = []
    for dataset in ("MNIST1D", "EEG", "PneumoniaMNIST"):
        models = [v for (d, _), v in per_model.items() if d == dataset]
        if not models:
            continue
        lines.append(
            f"{dataset}, 8 qubits & {len(models)} & "
            f"{', '.join(ABBREVIATION[s] for s in sorted(choice[dataset]))} & "
            f"{num(median(m['auto'] / m['direct'] for m in models))} & "
            f"{num(median(m['grouped'] / m['direct'] for m in models))}\\\\"
        )
    return lines


def trained_table():
    lines = [
        r"\begin{tabular}{lrlrr}",
        r"\toprule",
        r"Models & $N$ & Auto choice & Auto/direct & Grouped/direct\\",
        r"\midrule",
        *iris_rows(),
        r"\midrule",
        *fourier_rows(),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in (
        ("heldout_families.tex", heldout_table()),
        ("regret_decomposition.tex", regret_table()),
        ("trained_models.tex", trained_table()),
    ):
        (OUT / name).write_text(text)
        print(f"== {name}\n{text}")


if __name__ == "__main__":
    main()
