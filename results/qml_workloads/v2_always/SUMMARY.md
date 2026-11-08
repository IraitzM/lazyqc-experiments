# QML Workload Sharing Results

Raw timed rows: 7680; validation failures: 0; maximum absolute oracle error: 4.108e-15.

Ratios use each case's median end-to-end time (planning plus collection), then take a geometric mean across cases. A ratio below 1 favors the numerator policy. The forced-policy reference is the best measured median among full, separate, and grouped for that same case; it is descriptive, not an independent prospective oracle.

Seed offsets 0/101 are development instances; 202/303 are held out. These are random instances from the same generators, not held-out circuit families. Repetitions estimate timing variability within a case.

| Split | Request pattern | Cases | Grouped / separate | Grouped wins | Grouped/separate 95% CI | Grouped / full | Grouped wins | Auto / best forced | Auto wins | Auto choice proxy / best |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| development | different_examples | 24 | 0.865 | 22 | [0.803, 0.932] | 1.118 | 8 | 1.269 | 3 | 1.115 |
| development | parameter_shift_pair | 24 | 0.860 | 20 | [0.810, 0.910] | 1.134 | 8 | 1.251 | 5 | 1.092 |
| development | parameter_trajectory | 24 | 0.839 | 23 | [0.790, 0.900] | 1.077 | 8 | 1.232 | 2 | 1.114 |
| development | same_state_local_outputs | 24 | 0.815 | 23 | [0.765, 0.865] | 1.770 | 8 | 1.511 | 0 | 1.172 |
| development | same_state_shared_cone | 24 | 0.546 | 24 | [0.507, 0.593] | 0.588 | 16 | 1.143 | 2 | 1.008 |
| heldout | different_examples | 24 | 0.862 | 19 | [0.795, 0.924] | 1.105 | 8 | 1.234 | 1 | 1.108 |
| heldout | parameter_shift_pair | 24 | 0.866 | 21 | [0.816, 0.914] | 1.150 | 8 | 1.261 | 3 | 1.094 |
| heldout | parameter_trajectory | 24 | 0.856 | 23 | [0.796, 0.914] | 1.112 | 8 | 1.245 | 1 | 1.124 |
| heldout | same_state_local_outputs | 24 | 0.788 | 24 | [0.747, 0.849] | 1.773 | 8 | 1.461 | 1 | 1.181 |
| heldout | same_state_shared_cone | 24 | 0.543 | 24 | [0.496, 0.590] | 0.581 | 16 | 1.149 | 0 | 1.008 |

`Auto choice proxy / best` compares the forced policy selected by auto with the best forced median, excluding auto's planning time. `Auto / best forced` includes auto planning and collection. Neither comparison establishes a family-general selector.
The 95% interval resamples held-out cases and paired timing repetitions within each case. It is descriptive for these generated instances, not evidence of transfer to new QML families.

See `evaluation.json` for every raw repetition and environment/source fingerprints.
