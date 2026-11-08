#!/usr/bin/env python3
"""Weeks 3-5 gate prototype: reuse-policy selection error vs fastest-forced.

Operates on a recorded deterministic evaluation (default: v2 r7_fixed).
Lifetime model per policy: L(R) = H + R*T with H = median cold planning ns,
T = median warm cached total ns. Oracle = argmin over forced modes by
measured lifetime. Rules evaluated over an R grid on train repeats, with
selection error and regret reported on held-out repeats.

Writes results/workload_manifest/v2/policy_evaluation.json.
"""

import argparse
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/workload_manifest/v2/evaluation_r7_fixed.json"
R_GRID = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def lifetimes(rows, train):
    """Return per-mode (H, T) medians from train repeats."""
    out = {}
    for mode, cold, warm in (
        ("B3", "B3_independent", "B3_cached"),
        ("B5", "B5_joint", "B5_cached"),
    ):
        h = median(
            r["planning_ns"]
            for r in rows
            if r["mode"] == cold and r["repetition"] in train and r["status"] == "ok"
        )
        t = median(
            r["total_ns"]
            for r in rows
            if r["mode"] == warm and r["repetition"] in train and r["status"] == "ok"
        )
        out[mode] = (h, t)
    return out


def breakeven(h_inc, t_inc, h_joint, t_joint):
    """First R in grid where joint lifetime beats independent, else None."""
    for r in R_GRID:
        if h_joint + r * t_joint < h_inc + r * t_inc:
            return r
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--train", default="0,1,2,3")
    ap.add_argument("--test", default="4,5,6")
    ap.add_argument(
        "--output", type=Path, default=ROOT / "results/workload_manifest/v2/policy_evaluation.json"
    )
    args = ap.parse_args()
    train = [int(i) for i in args.train.split(",")]
    test = [int(i) for i in args.test.split(",")]

    data = json.loads(args.input.read_text())
    rows = data["runs"]
    report = {"input": str(args.input), "train": train, "test": test, "cases": []}

    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        for mode in ("B3_independent", "B5_joint", "B3_cached", "B5_cached"):
            tasks = {r["unique_experiments"] for r in sub if r["mode"] == mode}
            states = {r["search_states"] for r in sub if r["mode"] == mode}
            assert len(tasks) == 1 and len(states) == 1, f"{case}/{mode} not deterministic"
        life_train = lifetimes(sub, train)
        life_test = lifetimes(sub, test)
        (h_inc_tr, t_inc_tr), (h_joint_tr, t_joint_tr) = life_train["B3"], life_train["B5"]
        (h_inc_te, t_inc_te), (h_joint_te, t_joint_te) = life_test["B3"], life_test["B5"]
        r_star = breakeven(h_inc_tr, t_inc_tr, h_joint_tr, t_joint_tr)

        # Score-predicted selection uses rep-1 explanations (deterministic plans).
        plans = data["plans"]
        joint_alts = plans[f"{case}/B5_joint"]["alternatives"]
        s_inc = min(a["score"] for a in joint_alts if a["candidate"].split("/")[0] != "joint")
        s_joint = plans[f"{case}/B5_joint"]["estimated_cost"]

        case_rep = {"case": case, "H_ns": life_train, "H_test_ns": life_test,
                    "R_star_train": r_star,
                    "S_inc_pred_ns": s_inc, "S_joint_pred_ns": s_joint, "R": []}
        both_pick = "B5" if s_joint <= s_inc else "B3"
        for r in R_GRID:
            oracle = "B5" if h_joint_te + r * t_joint_te < h_inc_te + r * t_inc_te else "B3"
            l_oracle = min(h_inc_te + r * t_inc_te, h_joint_te + r * t_joint_te)
            # Deployable R-threshold rule: joint iff declared R reaches train break-even.
            thresh = "B5" if r_star is not None and r >= r_star else "B3"
            # Score-predicted rule (ex post): min predicted total with train H.
            score_pick = (
                "B5" if h_joint_tr + r * s_joint < h_inc_tr + r * s_inc else "B3"
            )
            # Shipped selector (both searches + honest double-H accounting).
            h_both = h_inc_te + h_joint_te
            t_both = t_joint_te if both_pick == "B5" else t_inc_te
            entry = {"R": r, "oracle": oracle, "both_pick": both_pick}
            pick_lifetime = {
                "B3": h_inc_te + r * t_inc_te,
                "B5": h_joint_te + r * t_joint_te,
            }
            for name, pick in (
                ("always_B3", "B3"),
                ("always_B5", "B5"),
                ("R_threshold", thresh),
                ("score_predicted", score_pick),
            ):
                entry[name] = {
                    "pick": pick,
                    "error": pick != oracle,
                    "regret_ns": pick_lifetime[pick] - l_oracle,
                }
            entry["both_search"] = {
                "pick": both_pick,
                "error": both_pick != oracle,
                "regret_ns": h_both + r * t_both - l_oracle,
            }
            case_rep["R"].append(entry)
        report["cases"].append(case_rep)

    summary = {}
    for name in ("always_B3", "always_B5", "R_threshold", "score_predicted", "both_search"):
        errs, regrets = [], []
        for c in report["cases"]:
            for e in c["R"]:
                errs.append(e[name]["error"])
                regrets.append(e[name]["regret_ns"])
        summary[name] = {
            "selection_error_rate": sum(errs) / len(errs),
            "mean_regret_ms": (sum(regrets) / len(regrets)) / 1e6,
            "max_regret_ms": max(regrets) / 1e6,
        }
    report["summary"] = summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    for c in report["cases"]:
        print(f"== {c['case']} (R* train = {c['R_star_train']}) ==")
        hb3, tb3 = c["H_ns"]["B3"]
        hb5, tb5 = c["H_ns"]["B5"]
        print(
            f"  H_B3={hb3/1e6:.1f}ms T_B3={tb3/1e6:.2f}ms | "
            f"H_B5={hb5/1e6:.1f}ms T_B5={tb5/1e6:.2f}ms | "
            f"pred saving={(c['S_inc_pred_ns']-c['S_joint_pred_ns'])/1e6:.2f}ms"
        )
        rules = ("always_B3", "always_B5", "R_threshold", "score_predicted", "both_search")
        for e in c["R"]:
            flags = "".join("!" if e[n]["error"] else "." for n in rules)
            regrets = ",".join(f"{e[n]['regret_ns']/1e6:.2f}" for n in rules)
            print(f"  R={e['R']:>3}: oracle={e['oracle']} [{flags}] regret_ms={regrets}")
    print("summary:", json.dumps(summary, indent=2))
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
