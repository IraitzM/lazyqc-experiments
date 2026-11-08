# Exhaustive Small-Instance Grouping Reference

Raw rows: 10784; numerical failures: 0; maximum absolute error: 6.106e-16.

The finite portfolio contains every set partition of five outputs, with one causal-cone task per partition block, plus an explicit full-state plan. It excludes cuts, rewrites, approximation and parallel workers. Ratios compare the current auto plan with the fastest measured candidate median in this declared finite set; the reference is not a global simulator optimum.

Each measured plan is collected five times per timing block, eight blocks per case. `R` amortizes the auto planning median over repeated collections of the same bound batch. Exhaustive enumeration time is reported separately; it is not treated as a zero-cost selection algorithm.

| Split | Cases | Candidates/case | Enumeration ms median | Auto graph covered | Auto execution / best | Auto total R=1 | R=5 | R=32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| development | 12 | 52.0 | 5.85 | 12/12 | 1.239 | 2.915 | 1.576 | 1.292 |
| heldout | 12 | 52.0 | 5.84 | 12/12 | 1.262 | 3.001 | 1.613 | 1.317 |

The results measure a narrow no-cut same-state grouping space on generated circuits. Seed holdout is not family holdout. The best measured candidate is subject to minimum-selection timing bias across 52–53 candidates.

See `grouping_oracle.json` for raw repetitions, candidate IDs, and per-case results.
