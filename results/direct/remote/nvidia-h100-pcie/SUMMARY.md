# Planner versus direct dense execution

Scaling rows: 1576; failures: 0; maximum absolute error vs reference: 1.33e-12.

| Family | Pattern | Qubits | LazyQC ms | LazyQC+GPU ms | Best direct | Best direct ms | Best direct / LazyQC+GPU |
|---|---|---:|---:|---:|---|---:|---:|
| data_reuploading | same_state_local_outputs | 12 | 4.261 | 4.308 | qulacs_1t | 2.455 | 0.570 |
| data_reuploading | same_state_local_outputs | 16 | 65.476 | 30.806 | lightning_gpu | 6.514 | 0.211 |
| data_reuploading | same_state_local_outputs | 20 | 1055.996 | 33.054 | lightning_gpu | 8.920 | 0.270 |
| data_reuploading | same_state_local_outputs | 24 | 17702.534 | 58.987 | aer_gpu | 37.805 | 0.641 |
| data_reuploading | same_state_local_outputs | 26 | n/a | 146.744 | aer_gpu | 109.769 | 0.748 |
| data_reuploading | same_state_local_outputs | 28 | n/a | 492.818 | aer_gpu | 451.895 | 0.917 |
| data_reuploading | same_state_shared_cone | 12 | 2.475 | 2.477 | qulacs_1t | 1.692 | 0.683 |
| data_reuploading | same_state_shared_cone | 16 | 36.709 | 20.507 | lightning_gpu | 5.138 | 0.251 |
| data_reuploading | same_state_shared_cone | 20 | 466.779 | 24.051 | lightning_gpu | 8.074 | 0.336 |
| data_reuploading | same_state_shared_cone | 24 | 10555.809 | 41.140 | aer_gpu | 33.623 | 0.817 |
| data_reuploading | same_state_shared_cone | 26 | n/a | 106.707 | aer_gpu | 92.816 | 0.870 |
| data_reuploading | same_state_shared_cone | 28 | n/a | 408.584 | aer_gpu | 385.339 | 0.943 |
| hardware_efficient_local | same_state_local_outputs | 12 | 2.837 | 2.844 | qulacs_1t | 2.503 | 0.880 |
| hardware_efficient_local | same_state_local_outputs | 16 | 4.223 | 4.225 | lightning_gpu | 4.867 | 1.152 |
| hardware_efficient_local | same_state_local_outputs | 20 | 5.717 | 5.721 | lightning_gpu | 8.012 | 1.400 |
| hardware_efficient_local | same_state_local_outputs | 24 | 4.729 | 4.641 | aer_gpu | 31.128 | 6.707 |
| hardware_efficient_local | same_state_local_outputs | 26 | 5.124 | 5.068 | aer_gpu | 98.207 | 19.378 |
| hardware_efficient_local | same_state_local_outputs | 28 | 5.666 | 7.589 | aer_gpu | 388.589 | 51.207 |
| hardware_efficient_local | same_state_shared_cone | 12 | 0.525 | 0.546 | qulacs_1t | 1.689 | 3.093 |
| hardware_efficient_local | same_state_shared_cone | 16 | 0.564 | 0.573 | lightning_gpu | 4.305 | 7.511 |
| hardware_efficient_local | same_state_shared_cone | 20 | 0.582 | 0.599 | lightning_gpu | 7.894 | 13.184 |
| hardware_efficient_local | same_state_shared_cone | 24 | 0.412 | 0.399 | aer_gpu | 29.307 | 73.449 |
| hardware_efficient_local | same_state_shared_cone | 26 | 0.394 | 0.384 | aer_gpu | 78.823 | 205.135 |
| hardware_efficient_local | same_state_shared_cone | 28 | 0.414 | 0.410 | aer_gpu | 317.538 | 775.042 |
| qcnn_like_local | same_state_local_outputs | 12 | 1.354 | 1.348 | qulacs_1t | 1.325 | 0.983 |
| qcnn_like_local | same_state_local_outputs | 16 | 1.913 | 1.917 | lightning_gpu | 3.387 | 1.766 |
| qcnn_like_local | same_state_local_outputs | 20 | 2.354 | 2.353 | lightning_gpu | 5.002 | 2.125 |
| qcnn_like_local | same_state_local_outputs | 24 | 1.924 | 1.937 | aer_gpu | 19.149 | 9.884 |
| qcnn_like_local | same_state_local_outputs | 26 | 1.968 | 1.963 | aer_gpu | 51.878 | 26.426 |
| qcnn_like_local | same_state_local_outputs | 28 | 2.125 | 2.136 | aer_gpu | 202.513 | 94.811 |
| qcnn_like_local | same_state_shared_cone | 12 | 0.489 | 0.497 | qulacs_1t | 0.683 | 1.375 |
| qcnn_like_local | same_state_shared_cone | 16 | 0.479 | 0.502 | lightning_gpu | 2.533 | 5.046 |
| qcnn_like_local | same_state_shared_cone | 20 | 0.508 | 0.540 | lightning_gpu | 3.673 | 6.795 |
| qcnn_like_local | same_state_shared_cone | 24 | 0.350 | 0.386 | aer_gpu | 18.195 | 47.183 |
| qcnn_like_local | same_state_shared_cone | 26 | 0.353 | 0.358 | aer_gpu | 35.748 | 99.792 |
| qcnn_like_local | same_state_shared_cone | 28 | 0.358 | 0.448 | aer_gpu | 134.260 | 299.558 |

## Parameter-shift Jacobians

| Family | Qubits | Method | ms |
|---|---:|---|---:|
| data_reuploading | 12 | aer_gpu_shift | 396.58 |
| data_reuploading | 12 | lazyqc | 272.70 |
| data_reuploading | 12 | lazyqc_gpu | 277.65 |
| data_reuploading | 12 | lightning_cpu_adjoint | 10.69 |
| data_reuploading | 12 | lightning_gpu_adjoint | 46.50 |
| data_reuploading | 16 | aer_gpu_shift | 1030.88 |
| data_reuploading | 16 | lazyqc | 5819.56 |
| data_reuploading | 16 | lazyqc_gpu | 4333.02 |
| data_reuploading | 16 | lightning_cpu_adjoint | 293.89 |
| data_reuploading | 16 | lightning_gpu_adjoint | 77.27 |
| hardware_efficient_local | 12 | aer_gpu_shift | 402.58 |
| hardware_efficient_local | 12 | lazyqc | 126.51 |
| hardware_efficient_local | 12 | lazyqc_gpu | 127.15 |
| hardware_efficient_local | 12 | lightning_cpu_adjoint | 9.87 |
| hardware_efficient_local | 12 | lightning_gpu_adjoint | 43.76 |
| hardware_efficient_local | 16 | aer_gpu_shift | 747.60 |
| hardware_efficient_local | 16 | lazyqc | 139.17 |
| hardware_efficient_local | 16 | lazyqc_gpu | 141.09 |
| hardware_efficient_local | 16 | lightning_cpu_adjoint | 292.92 |
| hardware_efficient_local | 16 | lightning_gpu_adjoint | 78.43 |
| hardware_efficient_local | 20 | aer_gpu_shift | 1271.61 |
| hardware_efficient_local | 20 | lazyqc | 281.37 |
| hardware_efficient_local | 20 | lazyqc_gpu | 270.74 |
| hardware_efficient_local | 20 | lightning_cpu_adjoint | 9341.45 |
| hardware_efficient_local | 20 | lightning_gpu_adjoint | 223.77 |
| hardware_efficient_local | 24 | lazyqc | 372.75 |
| hardware_efficient_local | 24 | lazyqc_gpu | 301.29 |
| hardware_efficient_local | 24 | lightning_gpu_adjoint | 2743.38 |
| qcnn_like_local | 12 | aer_gpu_shift | 97.30 |
| qcnn_like_local | 12 | lazyqc | 22.60 |
| qcnn_like_local | 12 | lazyqc_gpu | 22.57 |
| qcnn_like_local | 12 | lightning_cpu_adjoint | 5.14 |
| qcnn_like_local | 12 | lightning_gpu_adjoint | 26.10 |
| qcnn_like_local | 16 | aer_gpu_shift | 209.08 |
| qcnn_like_local | 16 | lazyqc | 26.68 |
| qcnn_like_local | 16 | lazyqc_gpu | 26.86 |
| qcnn_like_local | 16 | lightning_cpu_adjoint | 141.18 |
| qcnn_like_local | 16 | lightning_gpu_adjoint | 42.29 |
| qcnn_like_local | 20 | aer_gpu_shift | 334.99 |
| qcnn_like_local | 20 | lazyqc | 35.93 |
| qcnn_like_local | 20 | lazyqc_gpu | 44.48 |
| qcnn_like_local | 20 | lightning_cpu_adjoint | 3953.29 |
| qcnn_like_local | 20 | lightning_gpu_adjoint | 106.92 |
| qcnn_like_local | 24 | lazyqc | 81.97 |
| qcnn_like_local | 24 | lazyqc_gpu | 82.44 |
| qcnn_like_local | 24 | lightning_gpu_adjoint | 1218.89 |

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

Maximum Jacobian difference vs Lightning adjoint: 7.172040739078511e-14.

## Training loops

| Qubits | Method | Total ms | Median step ms | Loss first -> last | Max loss difference |
|---:|---|---:|---:|---|---:|
| 16 | aer_gpu_shift | 14876.2 | 1504.4 | 0.09327 -> 0.08408 | 4.163336342344337e-17 |
| 16 | lazyqc | 2250.0 | 221.8 | 0.09327 -> 0.08408 | 5.551115123125783e-17 |
| 16 | lightning_gpu_vjp | 256.8 | 25.6 | 0.09327 -> 0.08408 | - |
| 20 | lazyqc | 3308.6 | 333.6 | 0.08319 -> 0.07600 | 4.163336342344337e-17 |
| 20 | lightning_gpu_vjp | 425.6 | 42.6 | 0.08319 -> 0.07600 | - |
| 24 | lazyqc | 2952.9 | 296.1 | 0.09803 -> 0.09088 | 6.938893903907228e-17 |
| 24 | lightning_gpu_vjp | 4303.6 | 429.0 | 0.09803 -> 0.09088 | - |
