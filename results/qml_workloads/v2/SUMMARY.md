# QML Workload Sharing Results

Raw timed rows: 7680; validation failures: 0; maximum absolute oracle error: 4.108e-15.

Ratios use each case's median end-to-end time (planning plus collection), then take a geometric mean across cases. A ratio below 1 favors the numerator policy. The forced-policy reference is the best measured median among full, separate, and grouped for that same case; it is descriptive, not an independent prospective oracle.

Seed offsets 0/101 are development instances; 202/303 are held out. These are random instances from the same generators, not held-out circuit families. Repetitions estimate timing variability within a case.

| Split | Request pattern | Cases | Grouped / separate | Grouped wins | Grouped/separate 95% CI | Grouped / full | Grouped wins | Auto / best forced | Auto wins | Auto choice proxy / best |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| development | different_examples | 24 | 0.846 | 20 | [0.780, 0.913] | 1.074 | 8 | 1.270 | 1 | 1.114 |
| development | parameter_shift_pair | 24 | 0.865 | 20 | [0.809, 0.921] | 1.173 | 8 | 1.309 | 3 | 1.104 |
| development | parameter_trajectory | 24 | 0.831 | 23 | [0.780, 0.888] | 1.072 | 8 | 1.242 | 1 | 1.120 |
| development | same_state_local_outputs | 24 | 0.828 | 23 | [0.776, 0.877] | 1.790 | 8 | 1.515 | 3 | 1.167 |
| development | same_state_shared_cone | 24 | 0.544 | 24 | [0.505, 0.585] | 0.600 | 14 | 1.212 | 0 | 1.000 |
| heldout | different_examples | 24 | 0.867 | 18 | [0.809, 0.929] | 1.112 | 8 | 1.262 | 3 | 1.109 |
| heldout | parameter_shift_pair | 24 | 0.863 | 23 | [0.813, 0.912] | 1.154 | 8 | 1.270 | 2 | 1.098 |
| heldout | parameter_trajectory | 24 | 0.828 | 23 | [0.780, 0.890] | 1.076 | 8 | 1.275 | 0 | 1.112 |
| heldout | same_state_local_outputs | 24 | 0.806 | 24 | [0.759, 0.857] | 1.782 | 8 | 1.571 | 1 | 1.173 |
| heldout | same_state_shared_cone | 24 | 0.541 | 24 | [0.495, 0.590] | 0.583 | 15 | 1.206 | 0 | 1.005 |

`Auto choice proxy / best` compares the forced policy selected by auto with the best forced median, excluding auto's planning time. `Auto / best forced` includes auto planning and collection. Neither comparison establishes a family-general selector.
The 95% interval resamples held-out cases and paired timing repetitions within each case. It is descriptive for these generated instances, not evidence of transfer to new QML families.

See `evaluation.json` for every raw repetition and environment/source fingerprints.
