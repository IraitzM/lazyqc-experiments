#!/usr/bin/env bash
# Library: the lazyqc release pinned in uv.lock and in the script lockfiles (0.5.0 on PyPI).
# Laptop counterpart of scripts/run_remote_gpu.sh: reruns the GPU-dependent steps
# inside the locked script environments (uv run --locked --script), which the
# 3 Nov 2026 rerun missed, so lazyqc_gpu failed there. Results go to results/direct/.
set -uo pipefail
cd "$(dirname "$0")/.."
threads=$(nproc); out=results/direct
scaling=benchmarks/run_direct_baselines.py; gradients=benchmarks/run_gradient_baselines.py
step() { n=$1; shift; [ -f "$out/.$n.done" ] && { echo "skip $n"; return; }
  echo "$(date -u +%FT%TZ) start $n"; uv run --locked --script "$@" 2>&1 | tee -a "$out/$n.log" && date -u +%FT%TZ > "$out/.$n.done"; }
step scaling_1t "$scaling" --threads 1 --methods lazyqc,lazyqc_gpu,qulacs_1t --output $out/scaling_1t.json
step jacobian_1t "$gradients" --study jacobian --threads 1 --methods lazyqc,lazyqc_gpu --output $out/jacobian_1t.json
step training_1t "$gradients" --study training --threads 1 --methods lazyqc --output $out/training_1t.json
step scaling_mt "$scaling" --threads "$threads" --methods aer_cpu,aer_gpu,lightning_gpu --output $out/scaling_mt.json
step jacobian_mt "$gradients" --study jacobian --threads "$threads" --methods aer_gpu_shift,lightning_gpu_adjoint,lightning_cpu_adjoint --output $out/jacobian_mt.json
step training_mt "$gradients" --study training --threads "$threads" --methods lightning_gpu_vjp,aer_gpu_shift --output $out/training_mt.json
step backend_matrix "$scaling" --threads "$threads" --methods aer_gpu,lazyqc_aer_gpu,lightning_gpu,lazyqc_lightning_gpu,quimb,lazyqc_quimb --qubits 12,16,20,24,28 --output $out/backend_matrix.json
stack=(--with qiskit-aer-gpu==0.18.0 --with qiskit==2.6.1 --with qulacs==0.6.14 --with quimb==1.15.0 --with numpy==2.5.4)
run() { n=$1; shift; [ -f "$out/.$n.done" ] && { echo "skip $n"; return; }
  echo "$(date -u +%FT%TZ) start $n"; "$@" 2>&1 | tee -a "$out/$n.log" && date -u +%FT%TZ > "$out/.$n.done"; }
run memory_wall uv run --locked --all-extras --with qiskit-aer-gpu==0.18.0 --with qiskit==2.6.1 python benchmarks/run_memory_wall.py --output $out/memory_wall.json
run lightcone_ablation uv run "${stack[@]}" python benchmarks/run_lightcone_ablation.py --output $out/lightcone_ablation.json
run lightcone_jacobian_ablation uv run "${stack[@]}" python benchmarks/run_lightcone_jacobian_ablation.py --output $out/lightcone_jacobian_ablation.json
step confirm_1t "$scaling" --threads 1 --methods lazyqc,lazyqc_gpu,qulacs_1t --qubits 12,16 --output $out/confirm_1t.json
step confirm_mt "$scaling" --threads "$threads" --methods aer_cpu,aer_gpu,lightning_gpu --qubits 12,16 --output $out/confirm_mt.json
step jacobian_mt_chunked24 "$gradients" --study jacobian --threads "$threads" --methods lightning_gpu_adjoint --qubits 24 --output $out/jacobian_mt_chunked24.json
echo "$(date -u +%FT%TZ) laptop run complete"
