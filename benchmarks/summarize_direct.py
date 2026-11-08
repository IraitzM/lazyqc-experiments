#!/usr/bin/env python3
"""Summarize the planner-versus-direct study and draw its figures.

Reads whatever of results/direct/*.json exists and writes SUMMARY.md,
summary.json and the paper figures. Ratios are descriptive: medians per case,
geometric means over the two held-out seeds.

    uv run --locked --group notebooks python benchmarks/summarize_direct.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "direct"
FIGURES = ROOT / "paper" / "figures"
FAMILIES = ("hardware_efficient_local", "qcnn_like_local", "data_reuploading")
FAMILY_LABELS = {
    "hardware_efficient_local": "Hardware-efficient local",
    "qcnn_like_local": "QCNN-like local",
    "data_reuploading": "Data re-uploading (saturated)",
}
PATTERN_LABELS = {
    "same_state_local_outputs": "Z on every wire",
    "same_state_shared_cone": "X, Y, Z on wire 0",
}
# Fixed categorical order, validated with the dataviz palette checker (light mode).
STYLE = {
    "lazyqc": ("LazyQC (1 CPU thread)", "#2a78d6", "o", "-"),
    "lazyqc_gpu": ("LazyQC + GPU placement", "#eb6834", "s", "--"),
    "qulacs_1t": ("Qulacs, 1 thread", "#1baf7a", "^", "-"),
    "aer_cpu": ("Aer CPU, 20 threads", "#eda100", "D", "-"),
    "aer_gpu": ("Aer GPU", "#e87ba4", "v", "-"),
    "lightning_gpu": ("Lightning GPU", "#008300", "X", "-"),
    "aer_gpu_shift": ("Aer GPU, parameter shift", "#e87ba4", "v", "-"),
    "lightning_gpu_adjoint": ("Lightning GPU, adjoint", "#008300", "X", "-"),
    "lightning_cpu_adjoint": ("Lightning CPU, adjoint", "#eda100", "D", "-"),
    "lightning_gpu_vjp": ("Lightning GPU, adjoint VJP", "#008300", "X", "-"),
}
DIRECT = ("qulacs_1t", "aer_cpu", "aer_gpu", "lightning_gpu")


def load(name, directory=RESULTS):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def geomean(values):
    values = [v for v in values if v is not None]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def per_config(rows, keys):
    """Median ms per case and method, then geometric mean over seeds."""
    cases = defaultdict(list)
    for row in rows:
        cases[(row["case_id"], row["method"])].append(row)
    configs = defaultdict(list)
    for (_, method), group in cases.items():
        key = tuple(group[0][k] for k in keys) + (method,)
        configs[key].append(median(r["total_ns"] for r in group) / 1e6)
    return {key: geomean(values) for key, values in configs.items()}


def scaling_summary(directory=RESULTS, names=("scaling_1t.json", "scaling_mt.json")):
    parts = [load(name, directory) for name in names]
    parts = [p for p in parts if p]
    if not parts:
        return None
    rows = [r for p in parts for r in p["rows"]]
    failures = [f for p in parts for f in p["failures"]]
    table = per_config(rows, ("family", "pattern", "qubits"))
    out = {
        "failures": failures,
        "max_abs_error": max(r["max_abs_error"] for r in rows),
        "rows": len(rows),
        "configs": [],
    }
    for (family, pattern, n, method), ms in sorted(table.items()):
        out["configs"].append(
            {"family": family, "pattern": pattern, "qubits": n, "method": method, "ms": ms}
        )
    planner_detail = defaultdict(list)
    for r in rows:
        if r["method"] in ("lazyqc", "lazyqc_gpu"):
            planner_detail[(r["family"], r["pattern"], r["qubits"], r["method"])].append(r)
    out["planner"] = [
        {
            "family": k[0],
            "pattern": k[1],
            "qubits": k[2],
            "method": k[3],
            "planning_ms": median(r["planning_ns"] for r in v) / 1e6,
            "collection_ms": median(r["collection_ns"] for r in v) / 1e6,
            "tasks": max(r["tasks"] for r in v),
            "max_width": max(r["max_width"] for r in v),
            "strategy": sorted({r["selected_strategy"] for r in v}),
        }
        for k, v in sorted(planner_detail.items())
    ]
    return out


def best_direct(configs, family, pattern, n):
    candidates = [
        (c["ms"], c["method"])
        for c in configs
        if (c["family"], c["pattern"], c["qubits"]) == (family, pattern, n)
        and c["method"] in DIRECT
    ]
    return min(candidates) if candidates else (None, None)


def value(configs, family, pattern, n, method):
    for c in configs:
        if (c["family"], c["pattern"], c["qubits"], c["method"]) == (family, pattern, n, method):
            return c["ms"]
    return None


def plot_scaling(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = summary["configs"]
    patterns = list(PATTERN_LABELS)
    fig, axes = plt.subplots(len(patterns), 3, figsize=(7.16, 4.6), sharex=True, sharey="row")
    for i, pattern in enumerate(patterns):
        for j, family in enumerate(FAMILIES):
            ax = axes[i][j]
            for method in (
                "qulacs_1t",
                "aer_cpu",
                "lightning_gpu",
                "aer_gpu",
                "lazyqc",
                "lazyqc_gpu",
            ):
                points = sorted(
                    (c["qubits"], c["ms"])
                    for c in configs
                    if c["family"] == family and c["pattern"] == pattern and c["method"] == method
                )
                if not points:
                    continue
                label, color, marker, dash = STYLE[method]
                xs, ys = zip(*points)
                ax.plot(
                    xs,
                    ys,
                    dash,
                    color=color,
                    marker=marker,
                    markersize=4.2,
                    linewidth=1.4,
                    label=label,
                    markeredgecolor="white",
                    markeredgewidth=0.6,
                    zorder=3 if method.startswith("lazyqc") else 2,
                )
            ax.set_yscale("log")
            ax.axvline(28.5, color="#9a9994", linewidth=0.8)
            if i == 0:
                ax.set_title(FAMILY_LABELS[family], fontsize=8, color="#0b0b0b")
            if j == 0:
                ax.set_ylabel(
                    f"{PATTERN_LABELS[pattern]}\ntime (ms)", fontsize=7.5, color="#52514e"
                )
            if i == len(patterns) - 1:
                ax.set_xlabel("qubits", fontsize=7.5, color="#52514e")
            ax.set_xticks([12, 16, 20, 24, 28])
            ax.tick_params(labelsize=7, colors="#52514e")
            ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
            for spine in ax.spines.values():
                spine.set_color("#c3c2b7")
                spine.set_linewidth(0.6)
    handles, labels = axes[0][2].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"direct_scaling.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def memory_wall_summary():
    data = load("memory_wall.json")
    if not data:
        return None
    groups = defaultdict(list)
    for r in data["rows"]:
        groups[(r["family"], r["pattern"], r["qubits"])].append(r)
    planned = []
    for (family, pattern, n), rows in sorted(groups.items()):
        cases = defaultdict(list)
        for r in rows:
            cases[r["case_id"]].append(r)
        planned.append(
            {
                "family": family,
                "pattern": pattern,
                "qubits": n,
                "total_ms": geomean(
                    [median(x["total_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
                "planning_ms": geomean(
                    [median(x["planning_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
                "min_width": min(r.get("min_width", r["max_width"]) for r in rows),
                "median_width": int(median(r.get("median_width", r["max_width"]) for r in rows)),
                "max_width": max(r["max_width"] for r in rows),
                "tasks": max(r["tasks"] for r in rows),
                "dense_bytes": rows[0]["dense_state_bytes"],
            }
        )
    mps_groups = defaultdict(list)
    for r in data.get("mps_rows", []):
        mps_groups[(r["family"], r["pattern"], r["qubits"], r.get("depth", 4))].append(r)
    mps = []
    for (family, pattern, n, depth), rows in sorted(mps_groups.items()):
        cases = defaultdict(list)
        for r in rows:
            cases[r["case_id"]].append(r)
        mps.append(
            {
                "family": family,
                "pattern": pattern,
                "qubits": n,
                "depth": depth,
                "total_ms": geomean(
                    [median(x["total_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
                "max_bond": max(r["max_bond"] for r in rows),
            }
        )
    tn_groups = defaultdict(list)
    for r in data.get("tn_rows", []):
        tn_groups[(r["family"], r["pattern"], r["qubits"], r.get("depth", 4))].append(r)
    tn = []
    for (family, pattern, n, depth), rows in sorted(tn_groups.items()):
        cases = defaultdict(list)
        for r in rows:
            cases[r["case_id"]].append(r)
        tn.append(
            {
                "family": family,
                "pattern": pattern,
                "qubits": n,
                "depth": depth,
                "total_ms": geomean(
                    [median(x["total_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
            }
        )
    extra_groups = defaultdict(list)
    for r in data.get("extra_rows", []):
        extra_groups[
            (r["family"], r["pattern"], r["qubits"], r.get("depth", 4), r.get("method", "lazyqc"))
        ].append(r)
    extra = []
    for (family, pattern, n, depth, method), rows in sorted(extra_groups.items()):
        cases = defaultdict(list)
        for r in rows:
            cases[r["case_id"]].append(r)
        extra.append(
            {
                "family": family,
                "pattern": pattern,
                "qubits": n,
                "depth": depth,
                "method": method,
                "total_ms": geomean(
                    [median(x["total_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
                "planning_ms": geomean(
                    [median(x["planning_ns"] for x in v) / 1e6 for v in cases.values()]
                ),
                "min_width": min(r.get("min_width", r["max_width"]) for r in rows),
                "median_width": int(median(r.get("median_width", r["max_width"]) for r in rows)),
                "max_width": max(r["max_width"] for r in rows),
                "tasks": max(r["tasks"] for r in rows),
            }
        )
    errors = [c["max_abs_error_vs_quimb"] for c in data["cases"] if c["status"] == "planned"]
    mps_errors = [
        c["mps_max_abs_error_vs_quimb"]
        for c in data["cases"]
        if c.get("mps_max_abs_error_vs_quimb") is not None
    ]
    refused = [c for c in data["cases"] if c["status"] == "refused"]
    return {
        "planned": planned,
        "mps": mps,
        "tn": tn,
        "extra": extra,
        "max_error_vs_quimb": max(errors) if errors else None,
        "max_mps_error_vs_quimb": max(mps_errors) if mps_errors else None,
        "refused": [
            {
                "case_id": c["case_id"],
                "family": c["family"],
                "pattern": c["pattern"],
                "qubits": c["qubits"],
                "ms": c["refusal_ns"] / 1e6,
            }
            for c in refused
        ],
        "failures": data["failures"],
    }


def jacobian_summary(directory=RESULTS):
    parts = [json.loads(path.read_text()) for path in sorted(directory.glob("jacobian_*.json"))]
    if not parts:
        return None
    rows = [r for p in parts for r in p["rows"]]
    values = {}
    for p in parts:
        values.update(p.get("values", {}))
    table = per_config(rows, ("family", "qubits"))
    errors = []
    for key, entry in values.items():
        case, method = key.split("|")
        reference = values.get(f"{case}|lightning_gpu_adjoint")
        if reference and method != "lightning_gpu_adjoint":
            a = [x for row in entry["jacobian"] for x in row]
            b = [x for row in reference["jacobian"] for x in row]
            errors.append(
                {
                    "case": case,
                    "method": method,
                    "max_abs_error": max(abs(x - y) for x, y in zip(a, b, strict=True)),
                }
            )
    info = {}
    for r in rows:
        info[(r["family"], r["qubits"])] = {
            k: r[k] for k in ("parameters", "programs", "requests", "structural_zeros", "outputs")
        }
    tasks = defaultdict(list)
    for r in rows:
        if r["method"] == "lazyqc":
            tasks[(r["family"], r["qubits"])].append(r)
    return {
        "configs": [
            {"family": k[0], "qubits": k[1], "method": k[2], "ms": v}
            for k, v in sorted(table.items())
        ],
        "cases": [
            {"family": k[0], "qubits": k[1]}
            | v
            | (
                {
                    "planner_tasks": max(r["tasks"] for r in tasks[k]),
                    "planning_ms": median(r["planning_ns"] for r in tasks[k]) / 1e6,
                    "collection_ms": median(r["collection_ns"] for r in tasks[k]) / 1e6,
                }
                if tasks.get(k)
                else {}
            )
            for k, v in sorted(info.items())
        ],
        "validation": errors,
        "failures": [f for p in parts for f in p["failures"]],
    }


def plot_jacobian(summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = summary["configs"]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), sharey=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        for method in (
            "lightning_cpu_adjoint",
            "aer_gpu_shift",
            "lightning_gpu_adjoint",
            "lazyqc",
            "lazyqc_gpu",
        ):
            points = sorted(
                (c["qubits"], c["ms"])
                for c in configs
                if c["family"] == family and c["method"] == method
            )
            if not points:
                continue
            label, color, marker, dash = STYLE[method]
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                dash,
                color=color,
                marker=marker,
                markersize=4.2,
                linewidth=1.4,
                label=label,
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        ax.set_yscale("log")
        ax.set_title(FAMILY_LABELS[family], fontsize=8, color="#0b0b0b")
        ax.set_xlabel("qubits", fontsize=7.5, color="#52514e")
        ax.set_xticks([12, 16, 20, 24])
        ax.tick_params(labelsize=7, colors="#52514e")
        ax.grid(True, which="major", color="#e4e3df", linewidth=0.6)
        for spine in ax.spines.values():
            spine.set_color("#c3c2b7")
            spine.set_linewidth(0.6)
    axes[0].set_ylabel("full Jacobian time (ms)", fontsize=7.5, color="#52514e")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[1:]:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"gradient_scaling.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def training_summary(directory=RESULTS):
    parts = [load("training_1t.json", directory), load("training_mt.json", directory)]
    loops = [loop for p in parts if p for loop in p.get("loops", [])]
    if not loops:
        return None
    out = []
    for loop in loops:
        reference = next(
            (
                x
                for x in loops
                if x["qubits"] == loop["qubits"] and x["method"] == "lightning_gpu_vjp"
            ),
            None,
        )
        agreement = None
        if reference and reference is not loop:
            agreement = max(
                abs(a - b) for a, b in zip(loop["loss"], reference["loss"], strict=True)
            )
        out.append(
            {
                "qubits": loop["qubits"],
                "method": loop["method"],
                "parameters": loop["parameters"],
                "total_ms": sum(loop["step_ns"]) / 1e6,
                "median_step_ms": median(loop["step_ns"]) / 1e6,
                "first_loss": loop["loss"][0],
                "last_loss": loop["loss"][-1],
                "max_loss_difference_vs_lightning": agreement,
            }
        )
    return sorted(out, key=lambda x: (x["qubits"], x["method"]))


def markdown(scaling, wall, jacobian, training):
    lines = ["# Planner versus direct dense execution", ""]
    if scaling:
        lines += [
            f"Scaling rows: {scaling['rows']}; failures: {len(scaling['failures'])}; "
            f"maximum absolute error vs reference: {scaling['max_abs_error']:.2e}.",
            "",
            "| Family | Pattern | Qubits | LazyQC ms | LazyQC+GPU ms | Best direct | "
            "Best direct ms | Best direct / LazyQC+GPU |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
        configs = scaling["configs"]
        keys = sorted({(c["family"], c["pattern"], c["qubits"]) for c in configs})
        for family, pattern, n in keys:
            planner = [value(configs, family, pattern, n, m) for m in ("lazyqc", "lazyqc_gpu")]
            direct_ms, direct = best_direct(configs, family, pattern, n)
            # The planner with placement enabled is the reported configuration throughout.
            ratio = direct_ms / planner[1] if direct_ms and planner[1] else None
            fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"  # noqa: E731
            lines.append(
                f"| {family} | {pattern} | {n} | {fmt(planner[0])} | {fmt(planner[1])} "
                f"| {direct or 'n/a'} | {fmt(direct_ms)} | {fmt(ratio)} |"
            )
        lines.append("")
    if wall:
        lines += [
            "## Beyond device memory",
            "",
            f"Maximum error vs quimb: {wall['max_error_vs_quimb']}; "
            f"maximum MPS error vs quimb: {wall.get('max_mps_error_vs_quimb')}; refusals: "
            f"{len(wall['refused'])}; failures: {len(wall['failures'])}.",
            "",
            "| Family | Pattern | Qubits | Planner total ms | Planning ms | Width (min–max) | "
            "Tasks | Exact MPS ms | Max bond | Dense payload (GiB) |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|",
        ]
        for p in wall["planned"]:
            mps_entry = next(
                (
                    m
                    for m in wall.get("mps", [])
                    if (m["family"], m["pattern"], m["qubits"], m.get("depth", 4))
                    == (p["family"], p["pattern"], p["qubits"], 4)
                ),
                None,
            )
            mps_text = f"{mps_entry['total_ms']:.2f}" if mps_entry else "n/a"
            bond_text = str(mps_entry["max_bond"]) if mps_entry else "n/a"
            w_min = p.get("min_width", p["max_width"])
            w_text = str(p["max_width"]) if w_min == p["max_width"] else f"{w_min}–{p['max_width']}"
            lines.append(
                f"| {p['family']} | {p['pattern']} | {p['qubits']} | "
                f"{p['total_ms']:.2f} | {p['planning_ms']:.2f} | {w_text} | "
                f"{p['tasks']} | {mps_text} | {bond_text} | {p['dense_bytes'] / 2**30:.3g} |"
            )
        if wall.get("extra"):
            lines += [
                "",
                "### Intermediate-cone cases (48 qubits, depths 6 and 8)",
                "",
                "| Family | Qubits | Depth | Method | Total ms | Planning ms | "
                "Width (min–max) | Tasks | Exact MPS ms | Reused quimb TN ms |",
                "|---|---:|---:|---|---:|---:|---|---:|---:|---:|",
            ]
            for e in wall["extra"]:
                mps_entry = next(
                    (
                        m
                        for m in wall.get("mps", [])
                        if (m["family"], m["pattern"], m["qubits"], m.get("depth", 4))
                        == (e["family"], e["pattern"], e["qubits"], e["depth"])
                    ),
                    None,
                )
                tn_entry = next(
                    (
                        t
                        for t in wall.get("tn", [])
                        if (t["family"], t["pattern"], t["qubits"], t.get("depth", 4))
                        == (e["family"], e["pattern"], e["qubits"], e["depth"])
                    ),
                    None,
                )
                w_min = e.get("min_width", e["max_width"])
                w_text = (
                    str(e["max_width"]) if w_min == e["max_width"] else f"{w_min}–{e['max_width']}"
                )
                mps_text = f"{mps_entry['total_ms']:.2f}" if mps_entry else "n/a"
                tn_text = f"{tn_entry['total_ms']:.2f}" if tn_entry else "n/a"
                lines.append(
                    f"| {e['family']} | {e['qubits']} | {e['depth']} | {e['method']} | "
                    f"{e['total_ms']:.2f} | {e['planning_ms']:.2f} | {w_text} | {e['tasks']} | "
                    f"{mps_text} | {tn_text} |"
                )
        lines.append("")
    if jacobian:
        lines += [
            "## Parameter-shift Jacobians",
            "",
            "| Family | Qubits | Method | ms |",
            "|---|---:|---|---:|",
        ]
        for c in jacobian["configs"]:
            lines.append(f"| {c['family']} | {c['qubits']} | {c['method']} | {c['ms']:.2f} |")
        lines += [
            "",
            "| Family | Qubits | Parameters | Requests | Structural zeros | Planner tasks |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for c in jacobian["cases"]:
            lines.append(
                f"| {c['family']} | {c['qubits']} | {c['parameters']} | "
                f"{c['requests']} | {c['structural_zeros']} | "
                f"{c.get('planner_tasks', 'n/a')} |"
            )
        worst = max((e["max_abs_error"] for e in jacobian["validation"]), default=None)
        lines += ["", f"Maximum Jacobian difference vs Lightning adjoint: {worst}.", ""]
    if training:
        lines += [
            "## Training loops",
            "",
            "| Qubits | Method | Total ms | Median step ms | "
            "Loss first -> last | Max loss difference |",
            "|---:|---|---:|---:|---|---:|",
        ]
        for t in training:
            diff = t["max_loss_difference_vs_lightning"]
            lines.append(
                f"| {t['qubits']} | {t['method']} | {t['total_ms']:.1f} | "
                f"{t['median_step_ms']:.1f} | {t['first_loss']:.5f} -> "
                f"{t['last_loss']:.5f} | {diff if diff is not None else '-'} |"
            )
    return "\n".join(lines) + "\n"


def ratio_text(value):
    """Compact ratio: two decimals below 10, one below 100, integers above."""
    if value is None:
        return "n/a"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:,.0f}".replace(",", "{,}")


def latex_tables(scaling, wall, jacobian, training):
    """Write the paper's result tables from the summarized data."""
    out = ROOT / "paper" / "sections" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    short = {
        "hardware_efficient_local": "Hardware-eff.",
        "qcnn_like_local": "QCNN-like",
        "data_reuploading": "Re-uploading",
    }
    outputs = {"same_state_local_outputs": "$Z$, all", "same_state_shared_cone": "$X,Y,Z$, one"}
    if scaling:
        configs = scaling["configs"]
        widths = sorted({c["qubits"] for c in configs})
        lines = []
        for family in FAMILIES:
            for pattern in PATTERN_LABELS:
                cells = []
                for n in widths:
                    direct_ms, _ = best_direct(configs, family, pattern, n)
                    planner = value(configs, family, pattern, n, "lazyqc_gpu")
                    cells.append(ratio_text(direct_ms / planner if direct_ms and planner else None))
                lines.append(f"{short[family]} & {outputs[pattern]} & " + " & ".join(cells) + r"\\")
        header = " & ".join(str(n) for n in widths)
        (out / "rq1_ratios.tex").write_text(
            "\\begin{tabular}{ll" + "r" * len(widths) + "}\n\\toprule\n"
            f"Family & Outputs & {header}\\\\\n\\midrule\n"
            + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
    if wall:
        widths = sorted({p["qubits"] for p in wall["planned"]})
        lines = []
        for family in FAMILIES:
            for pattern in PATTERN_LABELS:
                cells = []
                w_range = "n/a"
                for n in widths:
                    entry = next(
                        (
                            p
                            for p in wall["planned"]
                            if (p["family"], p["pattern"], p["qubits"]) == (family, pattern, n)
                        ),
                        None,
                    )
                    if entry:
                        w_min = entry.get("min_width", entry["max_width"])
                        w_max = entry["max_width"]
                        w_range = str(w_max) if w_min == w_max else f"{w_min}--{w_max}"
                        if w_max == 0:  # no state-vector task: the planner handed off to exact MPS
                            w_range = "MPS"
                    cells.append(f"{entry['total_ms']:.2f}" if entry else "n/a")
                lines.append(
                    f"LazyQC & {short[family]} & {outputs[pattern]} & {w_range} & "
                    + " & ".join(cells)
                    + r"\\"
                )
        if wall.get("mps"):
            lines.append("\\midrule")
            for family in FAMILIES:
                for pattern in ("same_state_local_outputs",):
                    cells = []
                    bond = "n/a"
                    for n in widths:
                        entry = next(
                            (
                                m
                                for m in wall["mps"]
                                if (m["family"], m["pattern"], m["qubits"], m.get("depth", 4))
                                == (family, pattern, n, 4)
                            ),
                            None,
                        )
                        if entry:
                            bond = f"$\\chi{{\\le}}{entry['max_bond']}$"
                        cells.append(f"{entry['total_ms']:.1f}" if entry else "n/a")
                    lines.append(
                        f"Exact MPS & {short[family]} & {outputs[pattern]} & {bond} & "
                        + " & ".join(cells)
                        + r"\\"
                    )
        dense = " & ".join(f"$2^{{{n + 4}}}$" for n in widths)
        (out / "memory_wall.tex").write_text(
            "\\begin{tabular}{lllr" + "r" * len(widths) + "}\n\\toprule\n"
            "Method & Family & Outputs & Width / $\\chi$ & "
            + " & ".join(str(n) for n in widths)
            + "\\\\\n\\midrule\n"
            + "\n".join(lines)
            + "\n\\midrule\n"
            f"\\multicolumn{{4}}{{l}}{{Dense state (bytes)}} & {dense}\\\\\n"
            "\\bottomrule\n\\end{tabular}\n"
        )
    if jacobian:
        lines = []
        for case in jacobian["cases"]:
            family, n = case["family"], case["qubits"]
            ms = {
                c["method"]: c["ms"]
                for c in jacobian["configs"]
                if (c["family"], c["qubits"]) == (family, n)
            }
            zeros = 100 * case["structural_zeros"] / (case["parameters"] * case["outputs"])

            def cell(method, ms=ms):
                if method not in ms:
                    return "n/a"
                text = f"{ms[method]:.1f}" if ms[method] < 10 else f"{ms[method]:,.0f}"
                return text.replace(",", "{,}")

            methods = ("lazyqc_gpu", "aer_gpu_shift", "lightning_gpu_adjoint")
            timings = " & ".join(cell(m) for m in methods + ("lightning_cpu_adjoint",))
            lines.append(
                f"{short[family]} & {n} & {case['parameters']} & "
                f"{case.get('planner_tasks', 'n/a')} & {zeros:.0f} & {timings}\\\\"
            )
        (out / "jacobian.tex").write_text(
            "\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
            "Family & $n$ & $P$ & Tasks & Zeros (\\%) & LazyQC & Aer shift & L.\\ GPU adj. & "
            "L.\\ CPU adj.\\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n"
        )
    if training:
        names = {
            "lazyqc": "LazyQC, shift rule",
            "lightning_gpu_vjp": "Lightning GPU, adjoint",
            "aer_gpu_shift": "Aer GPU, shift rule",
        }
        lines = []
        for t in training:
            lines.append(
                f"{t['qubits']} & {names[t['method']]} & {t['total_ms'] / 1e3:.2f} & "
                f"{t['median_step_ms']:.0f} & {t['last_loss']:.4f}\\\\"
            )
        (out / "training.tex").write_text(
            "\\begin{tabular}{rlrrr}\n\\toprule\n"
            "$n$ & Method & Total (s) & Step (ms) & Final loss\\\\\n\\midrule\n"
            + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )


def intermediate_cone_latex(wall):
    """Generate the frozen 48-qubit intermediate-cone comparison."""
    if not wall or not wall.get("extra"):
        return
    out = ROOT / "paper" / "sections" / "generated"
    out.mkdir(parents=True, exist_ok=True)

    def entry(method, depth):
        return next(e for e in wall["extra"] if e["method"] == method and e["depth"] == depth)

    def baseline(kind, depth):
        return next(e for e in wall[kind] if e["depth"] == depth)

    lazy6, lazy8 = entry("lazyqc", 6), entry("lazyqc", 8)
    gpu6, gpu8 = entry("lazyqc_gpu", 6), entry("lazyqc_gpu", 8)
    mps6, mps8 = baseline("mps", 6), baseline("mps", 8)
    tn6, tn8 = baseline("tn", 6), baseline("tn", 8)
    widths = (
        f"{lazy6['min_width']}--{lazy6['max_width']} / {lazy8['min_width']}--{lazy8['max_width']}"
    )
    rows = [
        f"LazyQC, CPU & {widths} & {lazy6['total_ms'] / 1000:.3f} & "
        f"{lazy8['total_ms'] / 1000:.3f}\\\\",
        f"LazyQC, GPU placement & {widths} & {gpu6['total_ms'] / 1000:.3f} & "
        f"{gpu8['total_ms'] / 1000:.3f}\\\\",
        f"Exact MPS & $\\chi\\leq{mps6['max_bond']}$ / $\\chi\\leq{mps8['max_bond']}$ & "
        f"{mps6['total_ms'] / 1000:.3f} & {mps8['total_ms'] / 1000:.3f}\\\\",
        f"quimb light-cone TN & -- & {tn6['total_ms'] / 1000:.3f} & "
        f"{tn8['total_ms'] / 1000:.3f}\\\\",
    ]
    (out / "intermediate_cones.tex").write_text(
        "\\begin{tabular}{llrr}\n\\toprule\n"
        "Method & Width / $\\chi$, $d=6/8$ & $d=6$ (s) & $d=8$ (s)\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    )


def ratios(summary):
    """Fastest direct method's time over the planner with placement, per configuration."""
    configs = summary["configs"]
    out = {}
    for family, pattern, n in sorted({(c["family"], c["pattern"], c["qubits"]) for c in configs}):
        direct_ms, _ = best_direct(configs, family, pattern, n)
        planner = value(configs, family, pattern, n, "lazyqc_gpu")
        if direct_ms and planner:
            out[(family, pattern, n)] = direct_ms / planner
    return out


def confirmation(directory, main_summary):
    """Compare a later rerun of the 12- and 16-qubit cases with the main pass."""
    later = scaling_summary(directory, ("confirm_1t.json", "confirm_mt.json"))
    if not later or not main_summary:
        return None
    first, second = ratios(main_summary), ratios(later)
    deviations = [abs(second[k] / first[k] - 1) for k in second if k in first]
    if not deviations:
        return None
    return {
        "configurations": len(deviations),
        "median_relative_deviation": median(deviations),
        "max_relative_deviation": max(deviations),
    }


BACKEND_PAIRS = (
    ("Qulacs, 1 thread", "qulacs_1t", "lazyqc"),
    ("Aer GPU", "aer_gpu", "lazyqc_aer_gpu"),
    ("Lightning GPU", "lightning_gpu", "lazyqc_lightning_gpu"),
    ("quimb (tensor network)", "quimb", "lazyqc_quimb"),
)


def backend_matrix(directory, scaling, tag=None):
    """Direct execution on each backend over LazyQC running on that backend."""
    data = load("backend_matrix.json", directory)
    if not data or not scaling:
        return ""
    configs = scaling["configs"] + [
        {"family": k[0], "pattern": k[1], "qubits": k[2], "method": k[3], "ms": v}
        for k, v in per_config(data["rows"], ("family", "pattern", "qubits")).items()
    ]
    widths = sorted({r["qubits"] for r in data["rows"]})
    errors = max(r["max_abs_error"] for r in data["rows"])
    lines = [
        f"## Same-backend comparison{f' ({tag})' if tag else ''}",
        "",
        f"Rows: {len(data['rows'])}; failures: {len(data['failures'])}; maximum error "
        f"{errors:.1e}. Direct time over LazyQC time on the same backend (above one "
        "favours LazyQC); the quimb pair uses the grouped cone plan.",
        "",
        "| Backend | Family | Pattern | " + " | ".join(str(n) for n in widths) + " |",
        "|---|---|---|" + "---:|" * len(widths),
    ]
    short = {
        "hardware_efficient_local": "Hardware-eff.",
        "qcnn_like_local": "QCNN-like",
        "data_reuploading": "Re-uploading",
    }
    tex = []
    for label, direct, planner in BACKEND_PAIRS:
        for family in FAMILIES:
            for pattern in PATTERN_LABELS:
                cells = []
                for n in widths:
                    a = value(configs, family, pattern, n, direct)
                    b = value(configs, family, pattern, n, planner)
                    cells.append(ratio_text(a / b) if a and b else "n/a")
                lines.append(
                    f"| {label} | {family} | {pattern} | "
                    + " | ".join(c.replace("{,}", ",") for c in cells)
                    + " |"
                )
                if pattern == "same_state_local_outputs":
                    tex.append(
                        f"{label.split(' (')[0]} & {short[family]} & " + " & ".join(cells) + r"\\"
                    )
    out = ROOT / "paper" / "sections" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    name = f"backend_matrix_{tag}.tex" if tag else "backend_matrix.tex"
    (out / name).write_text(
        "\\begin{tabular}{ll" + "r" * len(widths) + "}\n\\toprule\n"
        "Backend & Family & "
        + " & ".join(str(n) for n in widths)
        + "\\\\\n\\midrule\n"
        + "\n".join(tex)
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    return "\n".join(lines) + "\n"


def remote_summaries():
    """Summarize every results/direct/remote/<tag>/ produced by scripts/sky_gpu.sh."""
    remote = {}
    for directory in sorted((RESULTS / "remote").glob("*/")):
        scaling = scaling_summary(directory)
        if not scaling:
            continue
        jacobian, training = jacobian_summary(directory), training_summary(directory)
        gpu = load("scaling_mt.json", directory) or {}
        remote[directory.name] = {
            "gpu": gpu.get("metadata", {}).get("environment", {}).get("gpu"),
            "scaling": scaling,
            "jacobian": jacobian,
            "training": training,
            "confirmation": confirmation(directory, scaling),
        }
        (directory / "SUMMARY.md").write_text(markdown(scaling, None, jacobian, training))
    return remote


def remote_comparison(local, remote):
    """Markdown and LaTeX tables of laptop versus remote ratios."""
    if not local or not remote:
        return ""
    widths = (16, 20, 24, 28)
    base = ratios(local)
    short = {
        "hardware_efficient_local": "Hardware-eff.",
        "qcnn_like_local": "QCNN-like",
        "data_reuploading": "Re-uploading",
    }
    outputs = {"same_state_local_outputs": "$Z$, all", "same_state_shared_cone": "$X,Y,Z$, one"}
    lines = [
        "## Laptop versus remote GPUs",
        "",
        "Fastest direct method over LazyQC+GPU on each host (laptop / remote).",
        "",
    ]
    header = "| Family | Pattern | " + " | ".join(f"{n} qubits" for n in widths) + " |"
    for tag, data in remote.items():
        other = ratios(data["scaling"])
        lines += [f"### {tag} ({data['gpu']})", "", header, "|---|---|" + "---:|" * len(widths)]
        rows_tex = []
        for family in FAMILIES:
            for pattern in PATTERN_LABELS:
                cells, tex = [], []
                for n in widths:
                    a, b = base.get((family, pattern, n)), other.get((family, pattern, n))
                    cells.append(f"{ratio_text(a)} / {ratio_text(b)}".replace("{,}", ","))
                    tex += [ratio_text(a), ratio_text(b)]
                lines.append(f"| {family} | {pattern} | " + " | ".join(cells) + " |")
                rows_tex.append(
                    f"{short[family]} & {outputs[pattern]} & " + " & ".join(tex) + r"\\"
                )
        if data["confirmation"]:
            c = data["confirmation"]
            lines += [
                "",
                f"Drift check on {tag}: ratios of a later rerun differ by a median of "
                f"{100 * c['median_relative_deviation']:.1f}% (max "
                f"{100 * c['max_relative_deviation']:.1f}%).",
            ]
        lines.append("")
        columns = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for n in widths)
        rules = "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(widths)))
        hosts = " & ".join("Lap. & Rem." for _ in widths)
        out = ROOT / "paper" / "sections" / "generated"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"remote_ratios_{tag}.tex").write_text(
            "\\begin{tabular}{ll" + "rr" * len(widths) + "}\n\\toprule\n"
            f" & & {columns}\\\\\n{rules}\nFamily & Outputs & {hosts}\\\\\n\\midrule\n"
            + "\n".join(rows_tex)
            + "\n\\bottomrule\n\\end{tabular}\n"
        )
    return "\n".join(lines) + "\n"


def main():
    scaling, wall = scaling_summary(), memory_wall_summary()
    jacobian, training = jacobian_summary(), training_summary()
    if scaling:
        plot_scaling(scaling)
    if jacobian:
        plot_jacobian(jacobian)
    latex_tables(scaling, wall, jacobian, training)
    intermediate_cone_latex(wall)
    remote = remote_summaries()
    local_check = confirmation(RESULTS, scaling)
    summary = {
        "scaling": scaling,
        "memory_wall": wall,
        "jacobian": jacobian,
        "training": training,
        "confirmation": local_check,
        "remote": {tag: {k: v for k, v in data.items()} for tag, data in remote.items()},
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    text = markdown(scaling, wall, jacobian, training)
    if local_check:
        text += (
            f"\nDrift check on the laptop: a rerun of the 12- and 16-qubit cases one hour "
            f"later differs by a median of {100 * local_check['median_relative_deviation']:.1f}% "
            f"(max {100 * local_check['max_relative_deviation']:.1f}%) in the "
            "direct/planner ratios.\n"
        )
    text += remote_comparison(scaling, remote)
    text += backend_matrix(RESULTS, scaling)
    for tag, data in remote.items():
        text += backend_matrix(RESULTS / "remote" / tag, data["scaling"], tag)
    (RESULTS / "SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
