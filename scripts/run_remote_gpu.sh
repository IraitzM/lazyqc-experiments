#!/usr/bin/env bash
# Runs on the SkyPilot cluster (cloud/gpu-baselines.yaml): the GPU-dependent
# part of the planner-versus-direct study, with the single-thread and the
# all-threads passes on the same host so that every planner/direct ratio stays
# within one machine. The 12- and 16-qubit cases run again at the end as a
# drift check. Each step leaves a .done marker, so a relaunch resumes.
#
# REMOTE_TAG names the results folder (default: derived from the GPU name).
# SMOKE=1 runs every step on its smallest case, for testing this script.
# STEPS="a b" restricts the run to the named steps (default: all of them).
set -euo pipefail
cd "$(dirname "$0")/.."

gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
tag=${REMOTE_TAG:-$(echo "$gpu" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//')}
out=${REMOTE_OUT:-results/direct/remote/$tag}
threads=$(nproc)
smoke=()
[ "${SMOKE:-0}" = 1 ] && smoke=(--smoke)
mkdir -p "$out"

# Resume after an abrupt end: restore what was collected from an earlier
# cluster with the same GPU model, without overwriting anything here.
collected=${COLLECTED:-$HOME/collected}
if [ -d "$collected/$tag" ]; then
  rsync -a --ignore-existing --exclude SUMMARY.md "$collected/$tag/" "$out/"
  echo "$(date -u +%FT%TZ) restored $(find "$out" -name '*.done' | wc -l) completed steps"
fi

nvidia-smi > "$out/nvidia-smi.txt"
lscpu > "$out/lscpu.txt"
{ uname -a; free -g; uv --version; } > "$out/host.txt"

step() {
  # step NAME RUNNER ARGS...: run once, log, and mark as done.
  local name=$1
  shift
  if [ -n "${STEPS:-}" ] && [[ " $STEPS " != *" $name "* ]]; then
    echo "$(date -u +%FT%TZ) skip $name (not in STEPS)"
    return
  fi
  if [ -f "$out/$name.done" ]; then
    echo "$(date -u +%FT%TZ) skip $name (done)"
    return
  fi
  echo "$(date -u +%FT%TZ) start $name"
  uv run --locked --script "$@" "${smoke[@]}" 2>&1 | tee -a "$out/$name.log"
  date -u +%FT%TZ > "$out/$name.done"
}

scaling=benchmarks/run_direct_baselines.py
gradients=benchmarks/run_gradient_baselines.py
step scaling_1t "$scaling" --threads 1 --methods lazyqc,lazyqc_gpu,qulacs_1t \
  --output "$out/scaling_1t.json"
step scaling_mt "$scaling" --threads "$threads" --methods aer_cpu,aer_gpu,lightning_gpu \
  --output "$out/scaling_mt.json"
step jacobian_1t "$gradients" --study jacobian --threads 1 --methods lazyqc,lazyqc_gpu \
  --output "$out/jacobian_1t.json"
step jacobian_mt "$gradients" --study jacobian --threads "$threads" \
  --methods aer_gpu_shift,lightning_gpu_adjoint,lightning_cpu_adjoint --output "$out/jacobian_mt.json"
step training_1t "$gradients" --study training --threads 1 --methods lazyqc \
  --output "$out/training_1t.json"
step training_mt "$gradients" --study training --threads "$threads" \
  --methods lightning_gpu_vjp,aer_gpu_shift --output "$out/training_mt.json"
# Same-backend comparison: each backend directly against LazyQC running on it.
step backend_matrix "$scaling" --threads "$threads" \
  --methods aer_gpu,lazyqc_aer_gpu,lightning_gpu,lazyqc_lightning_gpu,quimb,lazyqc_quimb \
  --qubits 12,16,20,24,28 --output "$out/backend_matrix.json"
step confirm_1t "$scaling" --threads 1 --methods lazyqc,lazyqc_gpu,qulacs_1t --qubits 12,16 \
  --output "$out/confirm_1t.json"
step confirm_mt "$scaling" --threads "$threads" --methods aer_cpu,aer_gpu,lightning_gpu \
  --qubits 12,16 --output "$out/confirm_mt.json"
[ -z "${STEPS:-}" ] && date -u +%FT%TZ > "$out/complete.txt"
echo "All steps complete in $out"
