# QML Workload Sharing Results

Raw timed rows: 7680; validation failures: 0; maximum absolute oracle error: 4.108e-15.

Ratios use each case's median end-to-end time (planning plus collection), then take a geometric mean across cases. A ratio below 1 favors the numerator policy. The forced-policy reference is the best measured median among full, separate, and grouped for that same case; it is descriptive, not an independent prospective oracle.

Seed offsets 0/101 are development instances; 202/303 are held out. These are random instances from the same generators, not held-out circuit families. Repetitions estimate timing variability within a case.

| Split | Request pattern | Cases | Grouped / separate | Grouped wins | Grouped/separate 95% CI | Grouped / full | Grouped wins | Auto / best forced | Auto wins | Auto choice proxy / best |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| development | different_examples | 24 | 0.861 | 23 | [0.811, 0.914] | 1.122 | 8 | 1.228 | 0 | 1.038 |
| development | parameter_shift_pair | 24 | 0.844 | 24 | [0.802, 0.892] | 1.151 | 8 | 1.257 | 0 | 1.032 |
| development | parameter_trajectory | 24 | 0.851 | 23 | [0.798, 0.908] | 1.107 | 8 | 1.204 | 2 | 1.032 |
| development | same_state_local_outputs | 24 | 0.816 | 23 | [0.770, 0.861] | 1.797 | 8 | 1.408 | 0 | 1.000 |
| development | same_state_shared_cone | 24 | 0.555 | 24 | [0.515, 0.597] | 0.611 | 14 | 1.171 | 0 | 1.000 |
| heldout | different_examples | 24 | 0.852 | 23 | [0.798, 0.905] | 1.097 | 8 | 1.220 | 0 | 1.035 |
| heldout | parameter_shift_pair | 24 | 0.840 | 23 | [0.797, 0.889] | 1.161 | 8 | 1.239 | 2 | 1.048 |
| heldout | parameter_trajectory | 24 | 0.862 | 20 | [0.806, 0.918] | 1.119 | 8 | 1.213 | 1 | 1.040 |
| heldout | same_state_local_outputs | 24 | 0.816 | 24 | [0.769, 0.852] | 1.792 | 8 | 1.414 | 0 | 1.000 |
| heldout | same_state_shared_cone | 24 | 0.556 | 24 | [0.514, 0.603] | 0.613 | 16 | 1.157 | 0 | 1.002 |

`Auto choice proxy / best` compares the forced policy selected by auto with the best forced median, excluding auto's planning time. `Auto / best forced` includes auto planning and collection. Neither comparison establishes a family-general selector.
The 95% interval resamples held-out cases and paired timing repetitions within each case. It is descriptive for these generated instances, not evidence of transfer to new QML families.

See `evaluation.json` for every raw repetition and environment/source fingerprints.
