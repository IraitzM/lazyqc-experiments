# Planner versus direct dense execution

Scaling rows: 1576; failures: 0; maximum absolute error vs reference: 2.84e-13.

| Family | Pattern | Qubits | LazyQC ms | LazyQC+GPU ms | Best direct | Best direct ms | Best direct / LazyQC+GPU |
|---|---|---:|---:|---:|---|---:|---:|
| data_reuploading | same_state_local_outputs | 12 | 2.055 | 2.322 | qulacs_1t | 1.166 | 0.502 |
| data_reuploading | same_state_local_outputs | 16 | 31.796 | 8.002 | lightning_gpu | 3.195 | 0.399 |
| data_reuploading | same_state_local_outputs | 20 | 611.536 | 31.377 | lightning_gpu | 23.542 | 0.750 |
| data_reuploading | same_state_local_outputs | 24 | 11739.630 | 413.374 | aer_gpu | 410.929 | 0.994 |
| data_reuploading | same_state_local_outputs | 26 | n/a | 1777.348 | aer_gpu | 1778.259 | 1.001 |
| data_reuploading | same_state_local_outputs | 28 | n/a | 7740.986 | aer_gpu | 7758.790 | 1.002 |
| data_reuploading | same_state_shared_cone | 12 | 1.197 | 1.295 | qulacs_1t | 0.701 | 0.542 |
| data_reuploading | same_state_shared_cone | 16 | 17.326 | 5.684 | lightning_gpu | 2.756 | 0.485 |
| data_reuploading | same_state_shared_cone | 20 | 343.770 | 26.726 | lightning_gpu | 21.080 | 0.789 |
| data_reuploading | same_state_shared_cone | 24 | 6796.846 | 373.233 | aer_gpu | 373.254 | 1.000 |
| data_reuploading | same_state_shared_cone | 26 | n/a | 1616.458 | aer_gpu | 1615.221 | 0.999 |
| data_reuploading | same_state_shared_cone | 28 | n/a | 7037.727 | aer_gpu | 7036.684 | 1.000 |
| hardware_efficient_local | same_state_local_outputs | 12 | 1.670 | 1.291 | qulacs_1t | 1.186 | 0.919 |
| hardware_efficient_local | same_state_local_outputs | 16 | 2.055 | 2.040 | lightning_gpu | 3.139 | 1.539 |
| hardware_efficient_local | same_state_local_outputs | 20 | 2.849 | 2.856 | aer_gpu | 16.091 | 5.634 |
| hardware_efficient_local | same_state_local_outputs | 24 | 3.436 | 3.397 | aer_gpu | 234.833 | 69.131 |
| hardware_efficient_local | same_state_local_outputs | 26 | 3.723 | 3.690 | aer_gpu | 1001.733 | 271.436 |
| hardware_efficient_local | same_state_local_outputs | 28 | 3.769 | 3.733 | aer_gpu | 4346.827 | 1164.510 |
| hardware_efficient_local | same_state_shared_cone | 12 | 0.229 | 0.254 | qulacs_1t | 0.791 | 3.114 |
| hardware_efficient_local | same_state_shared_cone | 16 | 0.235 | 0.237 | lightning_gpu | 2.770 | 11.703 |
| hardware_efficient_local | same_state_shared_cone | 20 | 0.237 | 0.247 | aer_gpu | 13.231 | 53.645 |
| hardware_efficient_local | same_state_shared_cone | 24 | 0.266 | 0.264 | aer_gpu | 197.443 | 747.162 |
| hardware_efficient_local | same_state_shared_cone | 26 | 0.297 | 0.279 | aer_gpu | 839.400 | 3013.516 |
| hardware_efficient_local | same_state_shared_cone | 28 | 0.266 | 0.317 | aer_gpu | 3650.664 | 11511.821 |
| qcnn_like_local | same_state_local_outputs | 12 | 0.523 | 0.517 | qulacs_1t | 0.524 | 1.013 |
| qcnn_like_local | same_state_local_outputs | 16 | 0.744 | 0.778 | lightning_gpu | 1.632 | 2.099 |
| qcnn_like_local | same_state_local_outputs | 20 | 0.869 | 0.921 | aer_gpu | 8.731 | 9.482 |
| qcnn_like_local | same_state_local_outputs | 24 | 1.081 | 1.145 | aer_gpu | 126.002 | 110.073 |
| qcnn_like_local | same_state_local_outputs | 26 | 1.193 | 1.230 | aer_gpu | 521.983 | 424.510 |
| qcnn_like_local | same_state_local_outputs | 28 | 1.387 | 1.293 | aer_gpu | 2309.981 | 1786.124 |
| qcnn_like_local | same_state_shared_cone | 12 | 0.193 | 0.188 | qulacs_1t | 0.295 | 1.564 |
| qcnn_like_local | same_state_shared_cone | 16 | 0.193 | 0.204 | lightning_gpu | 1.410 | 6.928 |
| qcnn_like_local | same_state_shared_cone | 20 | 0.209 | 0.212 | aer_gpu | 6.832 | 32.155 |
| qcnn_like_local | same_state_shared_cone | 24 | 0.238 | 0.238 | aer_gpu | 88.551 | 371.796 |
| qcnn_like_local | same_state_shared_cone | 26 | 0.230 | 0.231 | aer_gpu | 359.660 | 1559.847 |
| qcnn_like_local | same_state_shared_cone | 28 | 0.277 | 0.294 | aer_gpu | 1601.008 | 5453.979 |

## Beyond device memory

Maximum error vs quimb: 1.6827428339638573e-11; maximum MPS error vs quimb: 1.6827428339638573e-11; refusals: 0; failures: 0.

| Family | Pattern | Qubits | Planner total ms | Planning ms | Width (min–max) | Tasks | Exact MPS ms | Max bond | Dense payload (GiB) |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| data_reuploading | same_state_local_outputs | 32 | 47.28 | 2.98 | 32–0 | 1 | 44.84 | 16 | 64 |
| data_reuploading | same_state_local_outputs | 48 | 73.17 | 6.53 | 48–0 | 1 | 67.37 | 16 | 4.19e+06 |
| data_reuploading | same_state_local_outputs | 64 | 101.24 | 11.93 | 64–0 | 1 | 89.16 | 16 | 2.75e+11 |
| data_reuploading | same_state_local_outputs | 100 | 169.62 | 29.90 | 100–0 | 1 | 143.44 | 16 | 1.89e+22 |
| data_reuploading | same_state_local_outputs | 128 | 226.64 | 48.05 | 128–0 | 1 | 179.70 | 16 | 5.07e+30 |
| data_reuploading | same_state_shared_cone | 32 | 38.42 | 0.30 | 32–0 | 1 | 38.03 | 16 | 64 |
| data_reuploading | same_state_shared_cone | 48 | 57.77 | 0.42 | 48–0 | 1 | 56.89 | 16 | 4.19e+06 |
| data_reuploading | same_state_shared_cone | 64 | 78.51 | 0.58 | 64–0 | 1 | 76.59 | 16 | 2.75e+11 |
| data_reuploading | same_state_shared_cone | 100 | 121.12 | 0.86 | 100–0 | 1 | 119.21 | 16 | 1.89e+22 |
| data_reuploading | same_state_shared_cone | 128 | 152.42 | 1.10 | 128–0 | 1 | 151.95 | 16 | 5.07e+30 |
| hardware_efficient_local | same_state_local_outputs | 32 | 4.61 | 1.76 | 6–10 | 16 | 65.66 | 32 | 64 |
| hardware_efficient_local | same_state_local_outputs | 48 | 8.32 | 3.59 | 6–10 | 24 | 100.37 | 32 | 4.19e+06 |
| hardware_efficient_local | same_state_local_outputs | 64 | 12.44 | 5.98 | 6–10 | 32 | 133.03 | 32 | 2.75e+11 |
| hardware_efficient_local | same_state_local_outputs | 100 | 24.12 | 13.95 | 6–10 | 50 | 214.43 | 32 | 1.89e+22 |
| hardware_efficient_local | same_state_local_outputs | 128 | 34.94 | 21.99 | 6–10 | 64 | 274.28 | 32 | 5.07e+30 |
| hardware_efficient_local | same_state_shared_cone | 32 | 0.25 | 0.14 | 6 | 1 | 60.10 | 32 | 64 |
| hardware_efficient_local | same_state_shared_cone | 48 | 0.27 | 0.17 | 6 | 1 | 89.93 | 32 | 4.19e+06 |
| hardware_efficient_local | same_state_shared_cone | 64 | 0.34 | 0.23 | 6 | 1 | 120.32 | 32 | 2.75e+11 |
| hardware_efficient_local | same_state_shared_cone | 100 | 0.44 | 0.33 | 6 | 1 | 190.47 | 32 | 1.89e+22 |
| hardware_efficient_local | same_state_shared_cone | 128 | 0.46 | 0.35 | 6 | 1 | 243.88 | 32 | 5.07e+30 |
| qcnn_like_local | same_state_local_outputs | 32 | 1.63 | 0.79 | 2–8 | 28 | 51.53 | 8 | 64 |
| qcnn_like_local | same_state_local_outputs | 48 | 2.71 | 1.50 | 2–8 | 42 | 78.56 | 8 | 4.19e+06 |
| qcnn_like_local | same_state_local_outputs | 64 | 4.05 | 2.47 | 2–8 | 56 | 101.85 | 8 | 2.75e+11 |
| qcnn_like_local | same_state_local_outputs | 100 | 8.11 | 5.64 | 2–8 | 87 | 158.03 | 8 | 1.89e+22 |
| qcnn_like_local | same_state_local_outputs | 128 | 12.24 | 9.05 | 2–8 | 112 | 205.58 | 8 | 5.07e+30 |
| qcnn_like_local | same_state_shared_cone | 32 | 0.23 | 0.09 | 8 | 1 | 50.19 | 8 | 64 |
| qcnn_like_local | same_state_shared_cone | 48 | 0.21 | 0.10 | 8 | 1 | 67.90 | 8 | 4.19e+06 |
| qcnn_like_local | same_state_shared_cone | 64 | 0.23 | 0.12 | 8 | 1 | 88.85 | 8 | 2.75e+11 |
| qcnn_like_local | same_state_shared_cone | 100 | 0.25 | 0.15 | 8 | 1 | 139.19 | 8 | 1.89e+22 |
| qcnn_like_local | same_state_shared_cone | 128 | 0.29 | 0.18 | 8 | 1 | 181.29 | 8 | 5.07e+30 |

### Intermediate-cone cases (48 qubits, depths 6 and 8)

| Family | Qubits | Depth | Method | Total ms | Planning ms | Width (min–max) | Tasks | Exact MPS ms | Reused quimb TN ms |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| hardware_efficient_local | 48 | 6 | lazyqc | 71.44 | 5.33 | 8–14 | 24 | 262.39 | 3095.30 |
| hardware_efficient_local | 48 | 6 | lazyqc_gpu | 73.38 | 5.55 | 8–14 | 24 | 262.39 | 3095.30 |
| hardware_efficient_local | 48 | 8 | lazyqc | 1412.29 | 7.46 | 10–18 | 24 | 4638.42 | 5173.30 |
| hardware_efficient_local | 48 | 8 | lazyqc_gpu | 167.24 | 7.43 | 10–18 | 24 | 4638.42 | 5173.30 |

## Parameter-shift Jacobians

| Family | Qubits | Method | ms |
|---|---:|---|---:|
| data_reuploading | 12 | aer_gpu_shift | 261.41 |
| data_reuploading | 12 | lazyqc | 157.78 |
| data_reuploading | 12 | lazyqc_gpu | 157.97 |
| data_reuploading | 12 | lightning_cpu_adjoint | 4.87 |
| data_reuploading | 12 | lightning_gpu_adjoint | 24.33 |
| data_reuploading | 16 | aer_gpu_shift | 623.23 |
| data_reuploading | 16 | lazyqc | 3795.16 |
| data_reuploading | 16 | lazyqc_gpu | 936.10 |
| data_reuploading | 16 | lightning_cpu_adjoint | 1476.67 |
| data_reuploading | 16 | lightning_gpu_adjoint | 61.96 |
| hardware_efficient_local | 12 | aer_gpu_shift | 254.02 |
| hardware_efficient_local | 12 | lazyqc | 54.65 |
| hardware_efficient_local | 12 | lazyqc_gpu | 54.30 |
| hardware_efficient_local | 12 | lightning_cpu_adjoint | 5.44 |
| hardware_efficient_local | 12 | lightning_gpu_adjoint | 24.80 |
| hardware_efficient_local | 16 | aer_gpu_shift | 529.36 |
| hardware_efficient_local | 16 | lazyqc | 94.59 |
| hardware_efficient_local | 16 | lazyqc_gpu | 93.82 |
| hardware_efficient_local | 16 | lightning_cpu_adjoint | 1033.90 |
| hardware_efficient_local | 16 | lightning_gpu_adjoint | 59.88 |
| hardware_efficient_local | 20 | aer_gpu_shift | 2492.16 |
| hardware_efficient_local | 20 | lazyqc | 143.01 |
| hardware_efficient_local | 20 | lazyqc_gpu | 145.10 |
| hardware_efficient_local | 20 | lightning_cpu_adjoint | 3330.85 |
| hardware_efficient_local | 20 | lightning_gpu_adjoint | 846.64 |
| hardware_efficient_local | 24 | lazyqc | 193.85 |
| hardware_efficient_local | 24 | lazyqc_gpu | 187.68 |
| hardware_efficient_local | 24 | lightning_gpu_adjoint | 23875.73 |
| qcnn_like_local | 12 | aer_gpu_shift | 59.31 |
| qcnn_like_local | 12 | lazyqc | 8.26 |
| qcnn_like_local | 12 | lazyqc_gpu | 8.31 |
| qcnn_like_local | 12 | lightning_cpu_adjoint | 2.43 |
| qcnn_like_local | 12 | lightning_gpu_adjoint | 13.42 |
| qcnn_like_local | 16 | aer_gpu_shift | 133.36 |
| qcnn_like_local | 16 | lazyqc | 16.04 |
| qcnn_like_local | 16 | lazyqc_gpu | 15.80 |
| qcnn_like_local | 16 | lightning_cpu_adjoint | 42.85 |
| qcnn_like_local | 16 | lightning_gpu_adjoint | 32.58 |
| qcnn_like_local | 20 | aer_gpu_shift | 630.43 |
| qcnn_like_local | 20 | lazyqc | 19.43 |
| qcnn_like_local | 20 | lazyqc_gpu | 19.39 |
| qcnn_like_local | 20 | lightning_cpu_adjoint | 1426.59 |
| qcnn_like_local | 20 | lightning_gpu_adjoint | 364.97 |
| qcnn_like_local | 24 | lazyqc | 28.99 |
| qcnn_like_local | 24 | lazyqc_gpu | 29.14 |
| qcnn_like_local | 24 | lightning_gpu_adjoint | 10540.18 |

| Family | Qubits | Parameters | Requests | Structural zeros | Planner tasks |
|---|---:|---:|---:|---:|---:|
| data_reuploading | 12 | 48 | 1164 | 55 | 97 |
| data_reuploading | 16 | 64 | 2064 | 105 | 129 |
| hardware_efficient_local | 12 | 48 | 1164 | 292 | 290 |
| hardware_efficient_local | 16 | 64 | 2064 | 628 | 404 |
| hardware_efficient_local | 20 | 80 | 3220 | 1092 | 518 |
| hardware_efficient_local | 24 | 96 | 4632 | 1684 | 632 |
| qcnn_like_local | 12 | 22 | 540 | 196 | 102 |
| qcnn_like_local | 16 | 32 | 1040 | 408 | 158 |
| qcnn_like_local | 20 | 38 | 1540 | 640 | 181 |
| qcnn_like_local | 24 | 48 | 2328 | 996 | 237 |

Maximum Jacobian difference vs Lightning adjoint: 5.020428517354958e-13.

## Training loops

| Qubits | Method | Total ms | Median step ms | Loss first -> last | Max loss difference |
|---:|---|---:|---:|---|---:|
| 16 | aer_gpu_shift | 7253.6 | 718.8 | 0.09327 -> 0.08408 | 4.163336342344337e-17 |
| 16 | lazyqc | 957.2 | 95.3 | 0.09327 -> 0.08408 | 5.551115123125783e-17 |
| 16 | lightning_gpu_vjp | 180.5 | 18.0 | 0.09327 -> 0.08408 | - |
| 20 | lazyqc | 1397.4 | 140.5 | 0.08319 -> 0.07600 | 4.163336342344337e-17 |
| 20 | lightning_gpu_vjp | 1257.9 | 125.5 | 0.08319 -> 0.07600 | - |
| 24 | lazyqc | 1933.9 | 193.0 | 0.09803 -> 0.09088 | 9.71445146547012e-17 |
| 24 | lightning_gpu_vjp | 32157.9 | 3215.9 | 0.09803 -> 0.09088 | - |

Drift check on the laptop: a rerun of the 12- and 16-qubit cases one hour later differs by a median of 7.9% (max 23.3%) in the direct/planner ratios.
## Laptop versus remote GPUs

Fastest direct method over LazyQC+GPU on each host (laptop / remote).

### nvidia-h100-pcie (NVIDIA H100 PCIe, 81559 MiB, 570.148.08, 1755 MHz, 350.00 W)

| Family | Pattern | 16 qubits | 20 qubits | 24 qubits | 28 qubits |
|---|---|---:|---:|---:|---:|
| hardware_efficient_local | same_state_local_outputs | 1.54 / 1.15 | 5.63 / 1.40 | 69.1 / 6.71 | 1,165 / 51.2 |
| hardware_efficient_local | same_state_shared_cone | 11.7 / 7.51 | 53.6 / 13.2 | 747 / 73.4 | 11,512 / 775 |
| qcnn_like_local | same_state_local_outputs | 2.10 / 1.77 | 9.48 / 2.13 | 110 / 9.88 | 1,786 / 94.8 |
| qcnn_like_local | same_state_shared_cone | 6.93 / 5.05 | 32.2 / 6.80 | 372 / 47.2 | 5,454 / 300 |
| data_reuploading | same_state_local_outputs | 0.40 / 0.21 | 0.75 / 0.27 | 0.99 / 0.64 | 1.00 / 0.92 |
| data_reuploading | same_state_shared_cone | 0.48 / 0.25 | 0.79 / 0.34 | 1.00 / 0.82 | 1.00 / 0.94 |

Drift check on nvidia-h100-pcie: ratios of a later rerun differ by a median of 5.6% (max 12.7%).

## Same-backend comparison

Rows: 1440; failures: 0; maximum error 2.4e-13. Direct time over LazyQC time on the same backend (above one favours LazyQC); the quimb pair uses the grouped cone plan.

| Backend | Family | Pattern | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---:|---:|---:|---:|---:|
| Qulacs, 1 thread | hardware_efficient_local | same_state_local_outputs | 0.71 | 14.5 | 209 | 3,406 | n/a |
| Qulacs, 1 thread | hardware_efficient_local | same_state_shared_cone | 3.46 | 71.5 | 1,408 | 25,132 | n/a |
| Qulacs, 1 thread | qcnn_like_local | same_state_local_outputs | 1.00 | 21.8 | 364 | 5,696 | n/a |
| Qulacs, 1 thread | qcnn_like_local | same_state_shared_cone | 1.52 | 37.1 | 628 | 11,577 | n/a |
| Qulacs, 1 thread | data_reuploading | same_state_local_outputs | 0.57 | 0.94 | 0.98 | 0.98 | n/a |
| Qulacs, 1 thread | data_reuploading | same_state_shared_cone | 0.59 | 0.94 | 0.96 | 0.97 | n/a |
| Aer GPU | hardware_efficient_local | same_state_local_outputs | 0.24 | 0.28 | 0.81 | 10.1 | 154 |
| Aer GPU | hardware_efficient_local | same_state_shared_cone | 1.48 | 1.55 | 7.34 | 103 | 1,848 |
| Aer GPU | qcnn_like_local | same_state_local_outputs | 0.20 | 0.20 | 0.61 | 6.72 | 114 |
| Aer GPU | qcnn_like_local | same_state_shared_cone | 0.67 | 1.19 | 4.20 | 58.7 | 811 |
| Aer GPU | data_reuploading | same_state_local_outputs | 0.62 | 0.68 | 0.95 | 0.99 | 1.00 |
| Aer GPU | data_reuploading | same_state_shared_cone | 0.77 | 0.85 | 0.96 | 1.00 | 1.00 |
| Lightning GPU | hardware_efficient_local | same_state_local_outputs | 0.22 | 0.31 | 1.83 | 38.3 | 600 |
| Lightning GPU | hardware_efficient_local | same_state_shared_cone | 1.31 | 2.59 | 18.4 | 518 | 9,260 |
| Lightning GPU | qcnn_like_local | same_state_local_outputs | 0.36 | 0.30 | 1.56 | 31.9 | 486 |
| Lightning GPU | qcnn_like_local | same_state_shared_cone | 0.92 | 1.44 | 7.78 | 216 | 3,818 |
| Lightning GPU | data_reuploading | same_state_local_outputs | 0.52 | 0.62 | 0.89 | 0.99 | 1.00 |
| Lightning GPU | data_reuploading | same_state_shared_cone | 0.56 | 0.68 | 0.94 | 1.00 | 1.00 |
| quimb (tensor network) | hardware_efficient_local | same_state_local_outputs | 1.02 | 0.93 | 1.46 | 1.06 | 1.10 |
| quimb (tensor network) | hardware_efficient_local | same_state_shared_cone | 1.46 | 1.76 | 1.97 | 2.31 | 2.35 |
| quimb (tensor network) | qcnn_like_local | same_state_local_outputs | 0.93 | 0.92 | 0.93 | 0.92 | 0.96 |
| quimb (tensor network) | qcnn_like_local | same_state_shared_cone | 1.06 | 1.24 | 1.28 | 1.42 | 1.49 |
| quimb (tensor network) | data_reuploading | same_state_local_outputs | 0.82 | 0.83 | 0.87 | 0.83 | 0.86 |
| quimb (tensor network) | data_reuploading | same_state_shared_cone | 1.07 | 0.89 | 0.86 | 0.87 | 1.41 |
## Same-backend comparison (nvidia-h100-pcie)

Rows: 1440; failures: 0; maximum error 2.3e-13. Direct time over LazyQC time on the same backend (above one favours LazyQC); the quimb pair uses the grouped cone plan.

| Backend | Family | Pattern | 12 | 16 | 20 | 24 | 28 |
|---|---|---|---:|---:|---:|---:|---:|
| Qulacs, 1 thread | hardware_efficient_local | same_state_local_outputs | 0.88 | 14.1 | 222 | 3,948 | n/a |
| Qulacs, 1 thread | hardware_efficient_local | same_state_shared_cone | 3.22 | 59.8 | 1,254 | 25,465 | n/a |
| Qulacs, 1 thread | qcnn_like_local | same_state_local_outputs | 0.98 | 16.9 | 279 | 5,156 | n/a |
| Qulacs, 1 thread | qcnn_like_local | same_state_shared_cone | 1.40 | 29.8 | 574 | 12,378 | n/a |
| Qulacs, 1 thread | data_reuploading | same_state_local_outputs | 0.58 | 0.92 | 0.97 | 0.96 | n/a |
| Qulacs, 1 thread | data_reuploading | same_state_shared_cone | 0.68 | 0.92 | 0.97 | 0.98 | n/a |
| Aer GPU | hardware_efficient_local | same_state_local_outputs | 0.23 | 0.18 | 0.14 | 0.26 | 3.06 |
| Aer GPU | hardware_efficient_local | same_state_shared_cone | 1.83 | 1.27 | 2.29 | 4.67 | 57.9 |
| Aer GPU | qcnn_like_local | same_state_local_outputs | 0.20 | 0.15 | 0.16 | 0.23 | 2.22 |
| Aer GPU | qcnn_like_local | same_state_shared_cone | 1.15 | 0.95 | 1.70 | 2.13 | 23.3 |
| Aer GPU | data_reuploading | same_state_local_outputs | 0.73 | 0.35 | 0.41 | 0.59 | 0.92 |
| Aer GPU | data_reuploading | same_state_shared_cone | 0.85 | 0.59 | 0.81 | 0.77 | 0.95 |
| Lightning GPU | hardware_efficient_local | same_state_local_outputs | 0.32 | 0.26 | 0.40 | 3.37 | 40.8 |
| Lightning GPU | hardware_efficient_local | same_state_shared_cone | 1.90 | 2.49 | 4.55 | 43.3 | 603 |
| Lightning GPU | qcnn_like_local | same_state_local_outputs | 0.20 | 0.28 | 0.31 | 2.37 | 28.7 |
| Lightning GPU | qcnn_like_local | same_state_shared_cone | 0.82 | 1.38 | 1.68 | 14.5 | 229 |
| Lightning GPU | data_reuploading | same_state_local_outputs | 0.78 | 1.03 | 0.94 | 0.93 | 1.00 |
| Lightning GPU | data_reuploading | same_state_shared_cone | 0.92 | 1.07 | 1.12 | 0.94 | 1.00 |
| quimb (tensor network) | hardware_efficient_local | same_state_local_outputs | 1.01 | 1.02 | 1.28 | 1.13 | 1.05 |
| quimb (tensor network) | hardware_efficient_local | same_state_shared_cone | 1.43 | 1.73 | 1.95 | 2.24 | 1.95 |
| quimb (tensor network) | qcnn_like_local | same_state_local_outputs | 0.91 | 0.90 | 1.14 | 1.16 | 0.95 |
| quimb (tensor network) | qcnn_like_local | same_state_shared_cone | 1.07 | 1.21 | 1.32 | 1.45 | 1.22 |
| quimb (tensor network) | data_reuploading | same_state_local_outputs | 0.87 | 0.83 | 0.84 | 0.81 | 0.87 |
| quimb (tensor network) | data_reuploading | same_state_shared_cone | 0.99 | 0.99 | 1.00 | 0.98 | 0.83 |
