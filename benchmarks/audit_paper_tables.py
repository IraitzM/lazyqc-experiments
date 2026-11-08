#!/usr/bin/env python3
"""Regenerate every generated LaTeX table and verify byte-identical provenance.

Run with ``uv run --locked --group notebooks python benchmarks/audit_paper_tables.py``.
The audit executes analysis scripts only; it never runs a measurement or changes raw data.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper" / "sections" / "generated"
OUT = ROOT / "paper" / "analysis"

DIRECT = "benchmarks/summarize_direct.py"
GENERALIZATION = "benchmarks/paper_generalization_tables.py"
POLICY = "benchmarks/summarize_policy_ablations.py"
LIGHTCONE = "benchmarks/summarize_lightcone_ablation.py"

TABLES = [
    (
        "tab:rq1",
        "rq1_ratios.tex",
        DIRECT,
        ["results/direct/scaling_1t.json", "results/direct/scaling_mt.json"],
    ),
    (
        "tab:backends",
        "backend_matrix.tex",
        DIRECT,
        ["results/direct/backend_matrix.json", "results/direct/scaling_1t.json"],
    ),
    (
        "artifact-only:remote-backend",
        "backend_matrix_nvidia-h100-pcie.tex",
        DIRECT,
        ["results/direct/remote/nvidia-h100-pcie/backend_matrix.json"],
    ),
    ("tab:memory-wall", "memory_wall.tex", DIRECT, ["results/direct/memory_wall.json"]),
    (
        "tab:intermediate-cones",
        "intermediate_cones.tex",
        DIRECT,
        ["results/direct/memory_wall.json"],
    ),
    (
        "tab:jacobian",
        "jacobian.tex",
        DIRECT,
        [
            "results/direct/jacobian_1t.json",
            "results/direct/jacobian_mt.json",
            "results/direct/jacobian_mt_chunked24.json",
        ],
    ),
    (
        "tab:training",
        "training.tex",
        DIRECT,
        ["results/direct/training_1t.json", "results/direct/training_mt.json"],
    ),
    (
        "tab:remote",
        "remote_ratios_nvidia-h100-pcie.tex",
        DIRECT,
        [
            "results/direct/scaling_1t.json",
            "results/direct/scaling_mt.json",
            "results/direct/remote/nvidia-h100-pcie/scaling_1t.json",
            "results/direct/remote/nvidia-h100-pcie/scaling_mt.json",
        ],
    ),
    (
        "tab:heldout",
        "heldout_families.tex",
        GENERALIZATION,
        ["results/qml_workloads/qml_heldout_family_summary.json"],
    ),
    (
        "tab:regret",
        "regret_decomposition.tex",
        GENERALIZATION,
        [
            "results/qml_workloads/grouping_oracle.json",
            "results/qml_workloads/grouping_oracle_heldout.json",
        ],
    ),
    (
        "tab:trained",
        "trained_models.tex",
        GENERALIZATION,
        [
            "results/qml_workloads/trained_iris_qml_v3.json",
            "results/qml_workloads/fourier_trained_qml.json",
        ],
    ),
    (
        "tab:policy-ablations",
        "policy_ablations.tex",
        POLICY,
        [
            "results/qml_workloads/qml_heldout_family_session0.json",
            "results/qml_workloads/qml_heldout_family_session1.json",
        ],
    ),
    (
        "tab:lightcone-ablation",
        "lightcone_ablation.tex",
        LIGHTCONE,
        [
            "results/direct/lightcone_ablation.json",
            "results/direct/lightcone_jacobian_ablation.json",
        ],
    ),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def check_raw():
    checks = []

    def record(name, condition, detail):
        checks.append({"check": name, "pass": bool(condition), "detail": detail})
        if not condition:
            raise ValueError(f"Raw-artifact check failed: {name}: {detail}")

    memory = json.loads((ROOT / "results/direct/memory_wall.json").read_text())
    record("memory-wall cases", len(memory["cases"]) == 64, len(memory["cases"]))
    record("memory-wall MPS rows", len(memory["mps_rows"]) == 192, len(memory["mps_rows"]))
    record("memory-wall extra rows", len(memory["extra_rows"]) == 40, len(memory["extra_rows"]))
    record("memory-wall failures", not memory["failures"], memory["failures"])
    forward = json.loads((ROOT / "results/direct/lightcone_ablation.json").read_text())
    record("LightCone forward cases", len(forward["cases"]) == 16, len(forward["cases"]))
    record("LightCone forward rows", len(forward["rows"]) == 144, len(forward["rows"]))
    record("LightCone forward failures", not forward["failures"], forward["failures"])
    jacobian = json.loads((ROOT / "results/direct/lightcone_jacobian_ablation.json").read_text())
    record("LightCone Jacobian cases", len(jacobian["cases"]) == 4, len(jacobian["cases"]))
    record("LightCone Jacobian rows", len(jacobian["rows"]) == 30, len(jacobian["rows"]))
    record("LightCone Jacobian failures", not jacobian["failures"], jacobian["failures"])
    policy = json.loads((ROOT / "results/qml_workloads/policy_ablations.json").read_text())
    record("policy cases", policy["protocol"]["cases"] == 480, policy["protocol"]["cases"])
    record("policy count", len(policy["policies"]) == 7, len(policy["policies"]))
    return checks


def main():
    raw_checks = check_raw()
    expected = {filename for _, filename, _, _ in TABLES}
    actual = {path.name for path in GENERATED.glob("*.tex")}
    if expected != actual:
        raise ValueError(f"Generated-table inventory differs: expected {expected}, found {actual}")

    before = {filename: sha256(GENERATED / filename) for _, filename, _, _ in TABLES}
    commands = sorted({generator for _, _, generator, _ in TABLES})
    for command in commands:
        completed = subprocess.run(
            [sys.executable, command],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"{command} failed with {completed.returncode}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
    rows = []
    for label, filename, generator, sources in TABLES:
        path = GENERATED / filename
        after = sha256(path)
        rows.append(
            {
                "label": label,
                "generated_file": relative(path),
                "generator": generator,
                "sha256_before": before[filename],
                "sha256_after": after,
                "byte_identical": before[filename] == after,
                "sources": [
                    {
                        "path": source,
                        "sha256": sha256(ROOT / source),
                    }
                    for source in sources
                ],
            }
        )
    mismatches = [row["generated_file"] for row in rows if not row["byte_identical"]]
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not mismatches else "fail",
        "scope": (
            "All LaTeX files in paper/sections/generated; analysis-only regeneration, "
            "no measurements."
        ),
        "commands": commands,
        "raw_checks": raw_checks,
        "mismatches": mismatches,
        "tables": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "generated_table_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Generated-table provenance audit",
        "",
        f"Status: **{payload['status'].upper()}**. {len(rows)} generated tables regenerated "
        "and compared byte for byte. No benchmark was rerun and no raw result was edited.",
        "",
        "| Label | Generated file | Generator | Raw source(s) | Byte-identical |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        sources = "<br>".join(f"`{source['path']}`" for source in row["sources"])
        lines.append(
            f"| `{row['label']}` | `{row['generated_file']}` | `{row['generator']}` | "
            f"{sources} | {row['byte_identical']} |"
        )
    lines += [
        "",
        "The JSON companion records SHA-256 hashes for every generated table and raw source, "
        "the regeneration commands, and the raw-artifact count/failure checks.",
    ]
    (OUT / "generated_table_audit.md").write_text("\n".join(lines) + "\n")
    print(
        f"{payload['status'].upper()}: {len(rows)} generated tables; {len(raw_checks)} raw checks"
    )
    if mismatches:
        raise SystemExit(f"Non-reproducible generated tables: {mismatches}")


if __name__ == "__main__":
    main()
