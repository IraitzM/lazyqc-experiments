# Analysis budget on Hamiltonian averaging

| Case | Skipped | auto (budget) ms | auto (always) ms | full ms | budget / full | always / full |
|---|---|---:|---:|---:|---:|---:|
| ansatz_6q_l2_16terms | True | 0.196 | 0.372 | 0.211 | 0.93 | 1.76 |
| ansatz_6q_l2_64terms | True | 0.519 | 0.780 | 0.529 | 0.98 | 1.47 |
| ansatz_6q_l2_128terms | True | 1.127 | 1.496 | 1.174 | 0.96 | 1.27 |
| ansatz_8q_l2_16terms | True | 0.277 | 0.483 | 0.251 | 1.10 | 1.92 |
| ansatz_8q_l2_64terms | True | 0.643 | 1.107 | 0.673 | 0.96 | 1.64 |
| ansatz_8q_l2_128terms | True | 1.265 | 1.801 | 1.044 | 1.21 | 1.73 |
| ansatz_10q_l3_16terms | False | 0.712 | 0.706 | 0.398 | 1.79 | 1.77 |
| ansatz_10q_l3_64terms | True | 0.762 | 1.350 | 0.758 | 1.01 | 1.78 |
| ansatz_10q_l3_128terms | True | 1.481 | 2.184 | 1.368 | 1.08 | 1.60 |
| ansatz_12q_l3_16terms | False | 1.069 | 1.032 | 0.696 | 1.54 | 1.48 |
| ansatz_12q_l3_64terms | False | 2.485 | 2.700 | 1.725 | 1.44 | 1.57 |
| ansatz_12q_l3_128terms | False | 4.142 | 4.049 | 2.651 | 1.56 | 1.53 |

Skipped=True: 8 cases; budget/full geomean 1.025 (range 0.93-1.21); always/full geomean 1.636 (range 1.27-1.92)

Skipped=False: 4 cases; budget/full geomean 1.577 (range 1.44-1.79); always/full geomean 1.584 (range 1.48-1.77)

Rows: 540; maximum absolute error vs NumPy oracle: 1.55e-15
