# Qiskit LightCone + Aer ablation

All times are milliseconds. LC S/D means per-output LightCone circuits without/with exact reduced-circuit deduplication. LazyQC uses auto planning and single-thread Qulacs; LC uses one Aer GPU job per call.

| Workload | Qubits | Method | Tasks | Analysis | Execution | Total | Min width | Max width |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Forward | 12 | lazyqc | 6 | 0.374 | 0.741 | 1.129 | 6 | 10 |
| Forward | 12 | lightcone_dedup_aer | 12 | 20.975 | 8.743 | 29.775 | 4 | 10 |
| Forward | 12 | lightcone_separate_aer | 12 | 20.410 | 8.846 | 29.265 | 4 | 10 |
| Forward | 16 | lazyqc | 8 | 0.573 | 1.203 | 1.762 | 6 | 10 |
| Forward | 16 | lightcone_dedup_aer | 16 | 31.311 | 12.335 | 43.551 | 4 | 10 |
| Forward | 16 | lightcone_separate_aer | 16 | 31.196 | 12.107 | 43.282 | 4 | 10 |
| Forward | 20 | lazyqc | 10 | 0.892 | 1.697 | 2.628 | 6 | 10 |
| Forward | 20 | lightcone_dedup_aer | 20 | 42.791 | 16.086 | 58.847 | 4 | 10 |
| Forward | 20 | lightcone_separate_aer | 20 | 40.950 | 15.563 | 56.661 | 4 | 10 |
| Forward | 24 | lazyqc | 12 | 1.116 | 2.050 | 3.158 | 6 | 10 |
| Forward | 24 | lightcone_dedup_aer | 24 | 50.930 | 18.766 | 69.712 | 4 | 10 |
| Forward | 24 | lightcone_separate_aer | 24 | 50.930 | 18.529 | 69.405 | 4 | 10 |
| Forward | 28 | lazyqc | 14 | 1.445 | 2.454 | 3.900 | 6 | 10 |
| Forward | 28 | lightcone_dedup_aer | 28 | 61.983 | 21.888 | 83.861 | 4 | 10 |
| Forward | 28 | lightcone_separate_aer | 28 | 62.558 | 21.941 | 84.534 | 4 | 10 |
| Forward | 32 | lazyqc | 16 | 1.768 | 2.860 | 4.689 | 6 | 10 |
| Forward | 32 | lightcone_dedup_aer | 32 | 73.576 | 25.071 | 98.758 | 4 | 10 |
| Forward | 32 | lightcone_separate_aer | 32 | 74.145 | 25.481 | 99.708 | 4 | 10 |
| Forward | 64 | lazyqc | 32 | 6.004 | 6.361 | 12.413 | 6 | 10 |
| Forward | 64 | lightcone_dedup_aer | 64 | 195.458 | 51.113 | 246.431 | 4 | 10 |
| Forward | 64 | lightcone_separate_aer | 64 | 196.006 | 52.361 | 249.080 | 4 | 10 |
| Forward | 128 | lazyqc | 64 | 22.241 | 12.939 | 35.177 | 6 | 10 |
| Forward | 128 | lightcone_dedup_aer | 128 | 686.157 | 103.189 | 789.375 | 4 | 10 |
| Forward | 128 | lightcone_separate_aer | 128 | 689.374 | 104.073 | 793.618 | 4 | 10 |
| Jacobian | 12 | lazyqc | 290 | 8.517 | 43.093 | 52.016 | 6 | 10 |
| Jacobian | 12 | lightcone_dedup_aer | 504 | 1945.532 | 383.045 | 2329.603 | 4 | 10 |
| Jacobian | 12 | lightcone_separate_aer | 1164 | 2137.532 | 816.297 | 2953.568 | 4 | 10 |
| Jacobian | 20 | lazyqc | 518 | 22.780 | 104.347 | 127.676 | 6 | 10 |
| Jacobian | 20 | lightcone_dedup_aer | 896 | 6221.207 | 758.484 | 6982.585 | 4 | 10 |
| Jacobian | 20 | lightcone_separate_aer | 3220 | 7053.109 | 2433.325 | 9493.914 | 4 | 10 |

Maximum absolute errors: forward 1.11e-15; Jacobian 1.22e-15. No failures.

Qiskit's pass is commutation-aware; LazyQC uses conservative wire-support cones. Their reduced circuits are therefore not asserted to be identical.
