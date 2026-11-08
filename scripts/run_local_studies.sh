#!/usr/bin/env bash
# Reruns the CPU-only studies on the lazyqc release pinned in uv.lock (0.5.0 on
# PyPI): QML workload matrix, held-out families, grouping oracle, trained
# classifiers, analysis budget, probes, local workers, reuse priorities,
# sensitivity and the task-cost rule. Pilots on external hardware (IQM, IonQ)
# are not part of it. Run on an idle machine: every step is a timing study.
#
# Each step leaves results/.local_done/NAME, so a relaunch resumes. The previous
# results are copied once to results/archive/pre-0.5.0-local/ before anything is
# overwritten. FORCE=name reruns one step; STEPS="a b" restricts the run.
set -uo pipefail
cd "$(dirname "$0")/.."

done_dir=results/.local_done
log_dir=results/local_logs
archive=results/archive/pre-0.5.0-local
mkdir -p "$done_dir" "$log_dir"

if [ ! -d "$archive" ]; then
  mkdir -p "$archive"
  for d in qml_workloads local_parallel priorities sensitivity analysis_budget task_cost; do
    [ -d "results/$d" ] && cp -a "results/$d" "$archive/$d"
  done
fi

run() {
  # run NAME COMMAND...
  local name=$1
  shift
  if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $name "* ]]; then return; fi
  if [ -e "$done_dir/$name" ] && [ "${FORCE:-}" != "$name" ]; then
    echo "$(date -u +%FT%TZ) skip $name (done)"
    return
  fi
  echo "$(date -u +%FT%TZ) start $name"
  if "$@" > "$log_dir/$name.log" 2>&1; then
    touch "$done_dir/$name"
    echo "$(date -u +%FT%TZ) done  $name"
  else
    echo "$(date -u +%FT%TZ) FAILED $name (see $log_dir/$name.log)"
  fi
}

py() { uv run --locked python "$@"; }

run heldout_s0 py benchmarks/run_qml_depth_screen.py --manifest benchmarks/qml_heldout_family_manifest.json --session 0
run heldout_s1 py benchmarks/run_qml_depth_screen.py --manifest benchmarks/qml_heldout_family_manifest.json --session 1
run heldout_summary py benchmarks/summarize_heldout_families.py
run qml_workloads py benchmarks/run_qml_workloads.py
run qml_workloads_summary py benchmarks/summarize_qml_workloads.py
run grouping_oracle py benchmarks/run_grouping_oracle.py
run grouping_oracle_heldout py benchmarks/run_grouping_oracle.py --manifest benchmarks/grouping_oracle_heldout_manifest.json
run trained_iris_v3 py benchmarks/run_trained_iris_qml.py --manifest benchmarks/trained_iris_qml_v3_manifest.json
run fourier_trained py benchmarks/run_fourier_trained_qml.py
run analysis_budget_calibration py benchmarks/calibrate_analysis_budget.py
run analysis_budget_hamiltonian py benchmarks/run_analysis_budget.py --output results/analysis_budget/hamiltonian.json
run parallel_probe py benchmarks/run_qml_parallel_probe.py
run replay_probe py benchmarks/run_qml_replay_probe.py
run large_cone_probe py benchmarks/run_qml_large_cone_probe.py
for s in 0 1 2; do
  run local_parallel_s$s py benchmarks/run_local_parallel.py --session $s --repeats 9 --output-dir results/local_parallel/session$s
done
for s in 0 1 2; do
  run priorities_s$s py benchmarks/run_priorities.py --session $s --output-dir results/priorities/session$s
done
for s in 0 1 2; do
  run sensitivity_s$s py benchmarks/run_sensitivity.py --output results/sensitivity/session$s/evaluation.json
done
run task_cost_calibration py benchmarks/calibrate_task_cost.py --repeats 9
run task_cost_evaluation py benchmarks/evaluate_task_cost.py --output results/task_cost/evaluation_v2.json
echo "$(date -u +%FT%TZ) all steps visited"
