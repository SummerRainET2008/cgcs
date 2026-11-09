# S3 independent archive audit

This administrative audit is independent of the frozen analyzer. Structural completeness does not imply that a promotion gate passed.

## Artifact completeness

- Training summaries: 81 (burn-in 3, K1 6, search 72).
- Raw-K1 evaluations: 39.
- Composite evaluations: 36.
- Examples per evaluation: 200.
- Frozen question, reference, and order audit: PASS.
- Actor adapter, Value head, and training-log audit: PASS.

## Preregistered stability evidence

- Sequence-ratio clip violations (threshold < 0.20): 72/72; maximum 0.368889.
- Mean-KL violations (threshold < 0.02): 0/72; maximum 0.003597.
- E3 ESS exceeds E1 in all 18 comparisons: True.
- Raw answer-rate regression gate: False.
- Frozen analyzer stability gate: False.
- Independently corrected stability gate: False.
- Frozen analyzer overall gate: False.
- Independently corrected overall gate: False.
- Zero-support truthiness defect present in frozen analyzer: True.
- Overall verdict invariant to that defect: True.

## Within-epoch ratio diagnostic

- First accumulation block: 2304 states, clip fraction 0.000000, Actor surrogate saturation fraction 0.000000.
- Later blocks: 13708 states, clip fraction 0.325212, Actor surrogate saturation fraction 0.174059, outside-band corrective-gradient fraction 0.151153.
- First-block ratios all reconstruct one within 1e-7: True.
- E4 has the largest later-block clip fraction in 6/6 round-by-family cells and the largest Actor-platform fraction in 6/6 cells.
- Interpretation boundary: sequence_ratio_clip_fraction is the fraction of state-action sequence ratios outside [0.8, 1.2]. It is a two-sided, sequence-level diagnostic, not a token-drop rate and not the fraction of examples with zero total gradient. PPO actor-branch saturation additionally depends on the advantage sign. The audit reports that plateau fraction separately, as well as ratios outside the band that retain a corrective Actor gradient; Value and KL terms can still contribute gradients in either case.

| Estimator | First clip | First plateau | Later clip | Later plateau | Later corrective outside | Later KL | Later abs-log-ratio p50 | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_standard_actor_ratio | 0.000000 | 0.000000 | 0.296177 | 0.163992 | 0.132186 | 0.001820 | 0.084841 | 0.445059 | 1.018253 | 2.377634 |
| E1_trajectory_propensity | 0.000000 | 0.000000 | 0.310536 | 0.166182 | 0.144354 | 0.002059 | 0.091168 | 0.472415 | 1.064726 | 2.172157 |
| E3_step_cumulative_lambda_0.75 | 0.000000 | 0.000000 | 0.310184 | 0.165159 | 0.145025 | 0.001916 | 0.086058 | 0.488097 | 0.980739 | 2.547188 |
| E4_extended_candidate_set | 0.000000 | 0.000000 | 0.384143 | 0.200995 | 0.183148 | 0.002124 | 0.117869 | 0.690904 | 1.738375 | 5.295100 |

### Later-block ratio diagnostics by round and search family

| Round | Family | Estimator | States | Lower outside | Upper outside | All outside | Actor plateau | Corrective outside |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | best_of_n | E0_standard_actor_ratio | 577 | 0.129983 | 0.171577 | 0.301560 | 0.164645 | 0.136915 |
| 1 | best_of_n | E1_trajectory_propensity | 577 | 0.138648 | 0.154246 | 0.292894 | 0.145581 | 0.147314 |
| 1 | best_of_n | E3_step_cumulative_lambda_0.75 | 577 | 0.150780 | 0.152513 | 0.303293 | 0.143847 | 0.159445 |
| 1 | best_of_n | E4_extended_candidate_set | 577 | 0.188908 | 0.195841 | 0.384749 | 0.209705 | 0.175043 |
| 1 | entropy_value | E0_standard_actor_ratio | 574 | 0.114983 | 0.177700 | 0.292683 | 0.181185 | 0.111498 |
| 1 | entropy_value | E1_trajectory_propensity | 574 | 0.127178 | 0.162021 | 0.289199 | 0.155052 | 0.134146 |
| 1 | entropy_value | E3_step_cumulative_lambda_0.75 | 574 | 0.132404 | 0.168990 | 0.301394 | 0.167247 | 0.134146 |
| 1 | entropy_value | E4_extended_candidate_set | 574 | 0.193380 | 0.212544 | 0.405923 | 0.202091 | 0.203833 |
| 1 | value_value | E0_standard_actor_ratio | 568 | 0.140845 | 0.158451 | 0.299296 | 0.176056 | 0.123239 |
| 1 | value_value | E1_trajectory_propensity | 568 | 0.146127 | 0.179577 | 0.325704 | 0.200704 | 0.125000 |
| 1 | value_value | E3_step_cumulative_lambda_0.75 | 568 | 0.140845 | 0.188380 | 0.329225 | 0.198944 | 0.130282 |
| 1 | value_value | E4_extended_candidate_set | 568 | 0.181338 | 0.186620 | 0.367958 | 0.204225 | 0.163732 |
| 2 | best_of_n | E0_standard_actor_ratio | 579 | 0.138169 | 0.155440 | 0.293610 | 0.158895 | 0.134715 |
| 2 | best_of_n | E1_trajectory_propensity | 579 | 0.153713 | 0.146805 | 0.300518 | 0.169257 | 0.131261 |
| 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 574 | 0.146341 | 0.162021 | 0.308362 | 0.163763 | 0.144599 |
| 2 | best_of_n | E4_extended_candidate_set | 574 | 0.200348 | 0.196864 | 0.397213 | 0.209059 | 0.188153 |
| 2 | entropy_value | E0_standard_actor_ratio | 567 | 0.126984 | 0.169312 | 0.296296 | 0.146384 | 0.149912 |
| 2 | entropy_value | E1_trajectory_propensity | 567 | 0.158730 | 0.181658 | 0.340388 | 0.172840 | 0.167549 |
| 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 566 | 0.127208 | 0.178445 | 0.305654 | 0.167845 | 0.137809 |
| 2 | entropy_value | E4_extended_candidate_set | 563 | 0.175844 | 0.211368 | 0.387211 | 0.181172 | 0.206039 |
| 2 | value_value | E0_standard_actor_ratio | 562 | 0.120996 | 0.172598 | 0.293594 | 0.156584 | 0.137011 |
| 2 | value_value | E1_trajectory_propensity | 571 | 0.162872 | 0.152364 | 0.315236 | 0.154116 | 0.161121 |
| 2 | value_value | E3_step_cumulative_lambda_0.75 | 568 | 0.149648 | 0.163732 | 0.313380 | 0.149648 | 0.163732 |
| 2 | value_value | E4_extended_candidate_set | 562 | 0.177936 | 0.183274 | 0.361210 | 0.199288 | 0.161922 |

## Per-seed evaluation metrics

| Mode | Seed | Family | Estimator | Accuracy | Answer rate |
|---|---:|---|---|---:|---:|
| composite | 99801 | best_of_n | E0_standard_actor_ratio | 34.00% | 42.50% |
| composite | 99801 | best_of_n | E1_trajectory_propensity | 36.00% | 42.50% |
| composite | 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 35.50% | 41.50% |
| composite | 99801 | best_of_n | E4_extended_candidate_set | 35.50% | 43.00% |
| composite | 99801 | entropy_value | E0_standard_actor_ratio | 38.00% | 45.00% |
| composite | 99801 | entropy_value | E1_trajectory_propensity | 36.00% | 42.50% |
| composite | 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 37.00% | 44.00% |
| composite | 99801 | entropy_value | E4_extended_candidate_set | 36.00% | 42.50% |
| composite | 99801 | value_value | E0_standard_actor_ratio | 33.50% | 40.50% |
| composite | 99801 | value_value | E1_trajectory_propensity | 35.00% | 40.50% |
| composite | 99801 | value_value | E3_step_cumulative_lambda_0.75 | 33.50% | 41.00% |
| composite | 99801 | value_value | E4_extended_candidate_set | 34.00% | 40.00% |
| composite | 99802 | best_of_n | E0_standard_actor_ratio | 37.00% | 44.50% |
| composite | 99802 | best_of_n | E1_trajectory_propensity | 34.50% | 42.00% |
| composite | 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 32.00% | 38.50% |
| composite | 99802 | best_of_n | E4_extended_candidate_set | 36.00% | 43.00% |
| composite | 99802 | entropy_value | E0_standard_actor_ratio | 36.50% | 43.50% |
| composite | 99802 | entropy_value | E1_trajectory_propensity | 36.50% | 44.00% |
| composite | 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 36.50% | 43.00% |
| composite | 99802 | entropy_value | E4_extended_candidate_set | 37.00% | 44.50% |
| composite | 99802 | value_value | E0_standard_actor_ratio | 34.00% | 41.50% |
| composite | 99802 | value_value | E1_trajectory_propensity | 35.00% | 40.50% |
| composite | 99802 | value_value | E3_step_cumulative_lambda_0.75 | 33.50% | 38.00% |
| composite | 99802 | value_value | E4_extended_candidate_set | 36.00% | 42.00% |
| composite | 99803 | best_of_n | E0_standard_actor_ratio | 33.50% | 40.50% |
| composite | 99803 | best_of_n | E1_trajectory_propensity | 34.50% | 43.00% |
| composite | 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 32.50% | 40.50% |
| composite | 99803 | best_of_n | E4_extended_candidate_set | 33.50% | 41.00% |
| composite | 99803 | entropy_value | E0_standard_actor_ratio | 36.00% | 44.00% |
| composite | 99803 | entropy_value | E1_trajectory_propensity | 35.00% | 42.50% |
| composite | 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 35.50% | 43.00% |
| composite | 99803 | entropy_value | E4_extended_candidate_set | 35.00% | 41.50% |
| composite | 99803 | value_value | E0_standard_actor_ratio | 34.50% | 41.00% |
| composite | 99803 | value_value | E1_trajectory_propensity | 33.00% | 38.50% |
| composite | 99803 | value_value | E3_step_cumulative_lambda_0.75 | 34.00% | 39.00% |
| composite | 99803 | value_value | E4_extended_candidate_set | 35.00% | 42.00% |
| raw_k1 | 99801 | best_of_n | E0_standard_actor_ratio | 34.50% | 41.00% |
| raw_k1 | 99801 | best_of_n | E1_trajectory_propensity | 33.50% | 39.50% |
| raw_k1 | 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 33.50% | 39.00% |
| raw_k1 | 99801 | best_of_n | E4_extended_candidate_set | 34.50% | 41.00% |
| raw_k1 | 99801 | entropy_value | E0_standard_actor_ratio | 31.50% | 37.00% |
| raw_k1 | 99801 | entropy_value | E1_trajectory_propensity | 31.50% | 37.00% |
| raw_k1 | 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 33.00% | 38.00% |
| raw_k1 | 99801 | entropy_value | E4_extended_candidate_set | 33.00% | 38.50% |
| raw_k1 | 99801 | k1 | E0_standard_actor_ratio | 32.50% | 38.50% |
| raw_k1 | 99801 | value_value | E0_standard_actor_ratio | 34.00% | 39.00% |
| raw_k1 | 99801 | value_value | E1_trajectory_propensity | 33.00% | 38.00% |
| raw_k1 | 99801 | value_value | E3_step_cumulative_lambda_0.75 | 32.50% | 37.50% |
| raw_k1 | 99801 | value_value | E4_extended_candidate_set | 32.50% | 37.50% |
| raw_k1 | 99802 | best_of_n | E0_standard_actor_ratio | 34.00% | 38.50% |
| raw_k1 | 99802 | best_of_n | E1_trajectory_propensity | 32.00% | 37.00% |
| raw_k1 | 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 34.00% | 39.50% |
| raw_k1 | 99802 | best_of_n | E4_extended_candidate_set | 32.50% | 39.00% |
| raw_k1 | 99802 | entropy_value | E0_standard_actor_ratio | 34.00% | 39.00% |
| raw_k1 | 99802 | entropy_value | E1_trajectory_propensity | 32.00% | 37.50% |
| raw_k1 | 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 34.50% | 39.50% |
| raw_k1 | 99802 | entropy_value | E4_extended_candidate_set | 34.00% | 39.50% |
| raw_k1 | 99802 | k1 | E0_standard_actor_ratio | 34.00% | 39.50% |
| raw_k1 | 99802 | value_value | E0_standard_actor_ratio | 34.00% | 39.00% |
| raw_k1 | 99802 | value_value | E1_trajectory_propensity | 33.50% | 39.00% |
| raw_k1 | 99802 | value_value | E3_step_cumulative_lambda_0.75 | 33.00% | 38.00% |
| raw_k1 | 99802 | value_value | E4_extended_candidate_set | 33.50% | 40.00% |
| raw_k1 | 99803 | best_of_n | E0_standard_actor_ratio | 32.00% | 37.00% |
| raw_k1 | 99803 | best_of_n | E1_trajectory_propensity | 33.50% | 38.50% |
| raw_k1 | 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 32.00% | 36.50% |
| raw_k1 | 99803 | best_of_n | E4_extended_candidate_set | 33.50% | 38.00% |
| raw_k1 | 99803 | entropy_value | E0_standard_actor_ratio | 34.00% | 39.00% |
| raw_k1 | 99803 | entropy_value | E1_trajectory_propensity | 33.50% | 38.00% |
| raw_k1 | 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 33.00% | 38.50% |
| raw_k1 | 99803 | entropy_value | E4_extended_candidate_set | 33.00% | 37.50% |
| raw_k1 | 99803 | k1 | E0_standard_actor_ratio | 36.00% | 41.50% |
| raw_k1 | 99803 | value_value | E0_standard_actor_ratio | 32.50% | 38.50% |
| raw_k1 | 99803 | value_value | E1_trajectory_propensity | 34.00% | 39.00% |
| raw_k1 | 99803 | value_value | E3_step_cumulative_lambda_0.75 | 34.00% | 39.50% |
| raw_k1 | 99803 | value_value | E4_extended_candidate_set | 33.50% | 38.50% |

### Valid-final and finalization diagnostics

Answer rate is the fraction of examples with a scalar answer extracted from the frozen strict `FINAL:` protocol. It is not an explicit abstention probability. Every unanswered example must exhaust all frozen generation steps; the final two columns separate transcripts with no final-like wording from those with noncanonical final-like wording.

| Mode | Seed | Family | Estimator | Valid finals | Unanswered | Accuracy given valid final | No final-like marker | Noncanonical final-like marker | Mean steps | Answered mean steps | Unanswered mean steps |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| composite | 99801 | best_of_n | E0_standard_actor_ratio | 85 | 115 | 80.00% | 112 | 3 | 6.815 | 5.212 | 8.000 |
| composite | 99801 | best_of_n | E1_trajectory_propensity | 85 | 115 | 84.71% | 114 | 1 | 6.875 | 5.353 | 8.000 |
| composite | 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 83 | 117 | 85.54% | 115 | 2 | 6.795 | 5.096 | 8.000 |
| composite | 99801 | best_of_n | E4_extended_candidate_set | 86 | 114 | 82.56% | 112 | 2 | 6.810 | 5.233 | 8.000 |
| composite | 99801 | entropy_value | E0_standard_actor_ratio | 90 | 110 | 84.44% | 110 | 0 | 6.855 | 5.456 | 8.000 |
| composite | 99801 | entropy_value | E1_trajectory_propensity | 85 | 115 | 84.71% | 114 | 1 | 6.835 | 5.259 | 8.000 |
| composite | 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 88 | 112 | 84.09% | 112 | 0 | 6.840 | 5.364 | 8.000 |
| composite | 99801 | entropy_value | E4_extended_candidate_set | 85 | 115 | 84.71% | 115 | 0 | 6.855 | 5.306 | 8.000 |
| composite | 99801 | value_value | E0_standard_actor_ratio | 81 | 119 | 82.72% | 118 | 1 | 6.875 | 5.222 | 8.000 |
| composite | 99801 | value_value | E1_trajectory_propensity | 81 | 119 | 86.42% | 118 | 1 | 6.920 | 5.333 | 8.000 |
| composite | 99801 | value_value | E3_step_cumulative_lambda_0.75 | 82 | 118 | 81.71% | 118 | 0 | 6.930 | 5.390 | 8.000 |
| composite | 99801 | value_value | E4_extended_candidate_set | 80 | 120 | 85.00% | 119 | 1 | 6.955 | 5.388 | 8.000 |
| composite | 99802 | best_of_n | E0_standard_actor_ratio | 89 | 111 | 83.15% | 110 | 1 | 6.825 | 5.360 | 8.000 |
| composite | 99802 | best_of_n | E1_trajectory_propensity | 84 | 116 | 82.14% | 116 | 0 | 6.875 | 5.321 | 8.000 |
| composite | 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 77 | 123 | 83.12% | 122 | 1 | 6.910 | 5.169 | 8.000 |
| composite | 99802 | best_of_n | E4_extended_candidate_set | 86 | 114 | 83.72% | 114 | 0 | 6.850 | 5.326 | 8.000 |
| composite | 99802 | entropy_value | E0_standard_actor_ratio | 87 | 113 | 83.91% | 113 | 0 | 6.885 | 5.437 | 8.000 |
| composite | 99802 | entropy_value | E1_trajectory_propensity | 88 | 112 | 82.95% | 111 | 1 | 6.875 | 5.443 | 8.000 |
| composite | 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 86 | 114 | 84.88% | 114 | 0 | 6.855 | 5.337 | 8.000 |
| composite | 99802 | entropy_value | E4_extended_candidate_set | 89 | 111 | 83.15% | 111 | 0 | 6.870 | 5.461 | 8.000 |
| composite | 99802 | value_value | E0_standard_actor_ratio | 83 | 117 | 81.93% | 117 | 0 | 6.920 | 5.398 | 8.000 |
| composite | 99802 | value_value | E1_trajectory_propensity | 81 | 119 | 86.42% | 118 | 1 | 6.955 | 5.420 | 8.000 |
| composite | 99802 | value_value | E3_step_cumulative_lambda_0.75 | 76 | 124 | 88.16% | 123 | 1 | 6.960 | 5.263 | 8.000 |
| composite | 99802 | value_value | E4_extended_candidate_set | 84 | 116 | 85.71% | 116 | 0 | 6.925 | 5.440 | 8.000 |
| composite | 99803 | best_of_n | E0_standard_actor_ratio | 81 | 119 | 82.72% | 119 | 0 | 6.885 | 5.247 | 8.000 |
| composite | 99803 | best_of_n | E1_trajectory_propensity | 86 | 114 | 80.23% | 114 | 0 | 6.900 | 5.442 | 8.000 |
| composite | 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 81 | 119 | 80.25% | 119 | 0 | 6.905 | 5.296 | 8.000 |
| composite | 99803 | best_of_n | E4_extended_candidate_set | 82 | 118 | 81.71% | 117 | 1 | 6.870 | 5.244 | 8.000 |
| composite | 99803 | entropy_value | E0_standard_actor_ratio | 88 | 112 | 81.82% | 112 | 0 | 6.875 | 5.443 | 8.000 |
| composite | 99803 | entropy_value | E1_trajectory_propensity | 85 | 115 | 82.35% | 115 | 0 | 6.855 | 5.306 | 8.000 |
| composite | 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 86 | 114 | 82.56% | 114 | 0 | 6.915 | 5.477 | 8.000 |
| composite | 99803 | entropy_value | E4_extended_candidate_set | 83 | 117 | 84.34% | 117 | 0 | 6.890 | 5.325 | 8.000 |
| composite | 99803 | value_value | E0_standard_actor_ratio | 82 | 118 | 84.15% | 118 | 0 | 6.940 | 5.415 | 8.000 |
| composite | 99803 | value_value | E1_trajectory_propensity | 77 | 123 | 85.71% | 122 | 1 | 6.990 | 5.377 | 8.000 |
| composite | 99803 | value_value | E3_step_cumulative_lambda_0.75 | 78 | 122 | 87.18% | 120 | 2 | 6.925 | 5.244 | 8.000 |
| composite | 99803 | value_value | E4_extended_candidate_set | 84 | 116 | 83.33% | 116 | 0 | 6.935 | 5.464 | 8.000 |
| raw_k1 | 99801 | best_of_n | E0_standard_actor_ratio | 82 | 118 | 84.15% | 117 | 1 | 6.800 | 5.073 | 8.000 |
| raw_k1 | 99801 | best_of_n | E1_trajectory_propensity | 79 | 121 | 84.81% | 121 | 0 | 6.840 | 5.063 | 8.000 |
| raw_k1 | 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 78 | 122 | 85.90% | 122 | 0 | 6.860 | 5.077 | 8.000 |
| raw_k1 | 99801 | best_of_n | E4_extended_candidate_set | 82 | 118 | 84.15% | 118 | 0 | 6.765 | 4.988 | 8.000 |
| raw_k1 | 99801 | entropy_value | E0_standard_actor_ratio | 74 | 126 | 85.14% | 126 | 0 | 6.960 | 5.189 | 8.000 |
| raw_k1 | 99801 | entropy_value | E1_trajectory_propensity | 74 | 126 | 85.14% | 126 | 0 | 6.930 | 5.108 | 8.000 |
| raw_k1 | 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 76 | 124 | 86.84% | 124 | 0 | 6.935 | 5.197 | 8.000 |
| raw_k1 | 99801 | entropy_value | E4_extended_candidate_set | 77 | 123 | 85.71% | 123 | 0 | 6.900 | 5.143 | 8.000 |
| raw_k1 | 99801 | k1 | E0_standard_actor_ratio | 77 | 123 | 84.42% | 123 | 0 | 6.950 | 5.273 | 8.000 |
| raw_k1 | 99801 | value_value | E0_standard_actor_ratio | 78 | 122 | 87.18% | 122 | 0 | 6.840 | 5.026 | 8.000 |
| raw_k1 | 99801 | value_value | E1_trajectory_propensity | 76 | 124 | 86.84% | 124 | 0 | 6.895 | 5.092 | 8.000 |
| raw_k1 | 99801 | value_value | E3_step_cumulative_lambda_0.75 | 75 | 125 | 86.67% | 125 | 0 | 6.845 | 4.920 | 8.000 |
| raw_k1 | 99801 | value_value | E4_extended_candidate_set | 75 | 125 | 86.67% | 125 | 0 | 6.955 | 5.213 | 8.000 |
| raw_k1 | 99802 | best_of_n | E0_standard_actor_ratio | 77 | 123 | 88.31% | 123 | 0 | 6.905 | 5.156 | 8.000 |
| raw_k1 | 99802 | best_of_n | E1_trajectory_propensity | 74 | 126 | 86.49% | 126 | 0 | 6.930 | 5.108 | 8.000 |
| raw_k1 | 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 79 | 121 | 86.08% | 121 | 0 | 6.945 | 5.329 | 8.000 |
| raw_k1 | 99802 | best_of_n | E4_extended_candidate_set | 78 | 122 | 83.33% | 122 | 0 | 6.880 | 5.128 | 8.000 |
| raw_k1 | 99802 | entropy_value | E0_standard_actor_ratio | 78 | 122 | 87.18% | 122 | 0 | 6.885 | 5.141 | 8.000 |
| raw_k1 | 99802 | entropy_value | E1_trajectory_propensity | 75 | 125 | 85.33% | 125 | 0 | 6.895 | 5.053 | 8.000 |
| raw_k1 | 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 79 | 121 | 87.34% | 121 | 0 | 6.870 | 5.139 | 8.000 |
| raw_k1 | 99802 | entropy_value | E4_extended_candidate_set | 79 | 121 | 86.08% | 120 | 1 | 6.920 | 5.266 | 8.000 |
| raw_k1 | 99802 | k1 | E0_standard_actor_ratio | 79 | 121 | 86.08% | 121 | 0 | 6.935 | 5.304 | 8.000 |
| raw_k1 | 99802 | value_value | E0_standard_actor_ratio | 78 | 122 | 87.18% | 121 | 1 | 6.915 | 5.218 | 8.000 |
| raw_k1 | 99802 | value_value | E1_trajectory_propensity | 78 | 122 | 85.90% | 122 | 0 | 6.900 | 5.179 | 8.000 |
| raw_k1 | 99802 | value_value | E3_step_cumulative_lambda_0.75 | 76 | 124 | 86.84% | 123 | 1 | 6.915 | 5.145 | 8.000 |
| raw_k1 | 99802 | value_value | E4_extended_candidate_set | 80 | 120 | 83.75% | 120 | 0 | 6.890 | 5.225 | 8.000 |
| raw_k1 | 99803 | best_of_n | E0_standard_actor_ratio | 74 | 126 | 86.49% | 126 | 0 | 6.890 | 5.000 | 8.000 |
| raw_k1 | 99803 | best_of_n | E1_trajectory_propensity | 77 | 123 | 87.01% | 123 | 0 | 6.905 | 5.156 | 8.000 |
| raw_k1 | 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 73 | 127 | 87.67% | 127 | 0 | 6.960 | 5.151 | 8.000 |
| raw_k1 | 99803 | best_of_n | E4_extended_candidate_set | 76 | 124 | 88.16% | 124 | 0 | 6.865 | 5.013 | 8.000 |
| raw_k1 | 99803 | entropy_value | E0_standard_actor_ratio | 78 | 122 | 87.18% | 122 | 0 | 6.825 | 4.987 | 8.000 |
| raw_k1 | 99803 | entropy_value | E1_trajectory_propensity | 76 | 124 | 88.16% | 124 | 0 | 6.910 | 5.132 | 8.000 |
| raw_k1 | 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 77 | 123 | 85.71% | 123 | 0 | 6.895 | 5.130 | 8.000 |
| raw_k1 | 99803 | entropy_value | E4_extended_candidate_set | 75 | 125 | 88.00% | 125 | 0 | 6.905 | 5.080 | 8.000 |
| raw_k1 | 99803 | k1 | E0_standard_actor_ratio | 83 | 117 | 86.75% | 117 | 0 | 6.825 | 5.169 | 8.000 |
| raw_k1 | 99803 | value_value | E0_standard_actor_ratio | 77 | 123 | 84.42% | 123 | 0 | 6.920 | 5.195 | 8.000 |
| raw_k1 | 99803 | value_value | E1_trajectory_propensity | 78 | 122 | 87.18% | 122 | 0 | 6.890 | 5.154 | 8.000 |
| raw_k1 | 99803 | value_value | E3_step_cumulative_lambda_0.75 | 79 | 121 | 86.08% | 121 | 0 | 6.865 | 5.127 | 8.000 |
| raw_k1 | 99803 | value_value | E4_extended_candidate_set | 77 | 123 | 87.01% | 123 | 0 | 6.875 | 5.078 | 8.000 |

## Retained-composite expansion and diversity

Exact-text uniqueness and pairwise token-prefix divergence measure surface-form branching only. They do not establish semantic or reasoning-path diversity because candidate-level parsed answers and counterfactual terminal rewards were not logged. Source rollout is the frozen round-two rollout that supplied the evaluation threshold; evaluation-minus-source shifts therefore isolate deployment-state shift from target-to-source threshold calibration error.

| Seed | Family | Estimator | States | Target expansion | Source-rollout expansion | Evaluation expansion | Eval-source expansion shift | Target candidates/state | Source-rollout candidates/state | Evaluation candidates/state | Eval-source candidate shift | Candidates | Candidate tokens | Selected tokens | Candidate-token multiplier | Mean unique when expanded | Fully identical | Candidate pairs | Exact token pairs | Mean token-prefix divergence |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99801 | best_of_n | E0_standard_actor_ratio | 1363 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2726 | 155563 | 77434 | 2.0090 | 1.5664 | 43.36% | 1363 | 29.13% | 56.56% |
| 99801 | best_of_n | E1_trajectory_propensity | 1375 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2750 | 154240 | 76903 | 2.0056 | 1.5636 | 43.64% | 1375 | 28.36% | 56.43% |
| 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 1359 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2718 | 154557 | 76849 | 2.0112 | 1.5467 | 45.33% | 1359 | 29.58% | 55.54% |
| 99801 | best_of_n | E4_extended_candidate_set | 1362 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2724 | 154355 | 77161 | 2.0004 | 1.5485 | 45.15% | 1362 | 29.07% | 55.90% |
| 99801 | entropy_value | E0_standard_actor_ratio | 1371 | 33.33% | 37.21% | 50.77% | +13.56 pp | 2.0000 | 2.1163 | 2.5230 | +0.4067 | 3459 | 208107 | 76946 | 2.7046 | 2.9684 | 13.07% | 4176 | 14.32% | 71.74% |
| 99801 | entropy_value | E1_trajectory_propensity | 1367 | 33.33% | 34.74% | 51.65% | +16.90 pp | 2.0000 | 2.0423 | 2.5494 | +0.5071 | 3485 | 210440 | 77065 | 2.7307 | 2.9093 | 14.73% | 4236 | 14.16% | 71.24% |
| 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 1368 | 33.33% | 33.49% | 50.37% | +16.88 pp | 2.0000 | 2.0047 | 2.5110 | +0.5063 | 3435 | 205410 | 77029 | 2.6667 | 2.9086 | 15.09% | 4134 | 14.54% | 71.26% |
| 99801 | entropy_value | E4_extended_candidate_set | 1371 | 33.33% | 34.11% | 50.55% | +16.43 pp | 2.0000 | 2.0234 | 2.5164 | +0.4930 | 3450 | 208272 | 78436 | 2.6553 | 2.9509 | 12.70% | 4158 | 13.83% | 72.14% |
| 99801 | value_value | E0_standard_actor_ratio | 1375 | 33.33% | 36.45% | 0.00% | -36.45 pp | 2.0000 | 2.0935 | 1.0000 | -1.0935 | 1375 | 76677 | 76677 | 1.0000 | N/A | N/A | 0 | N/A | N/A |
| 99801 | value_value | E1_trajectory_propensity | 1384 | 33.33% | 36.57% | 0.14% | -36.43 pp | 2.0000 | 2.0972 | 1.0043 | -1.0929 | 1390 | 77654 | 77239 | 1.0054 | 4.0000 | 0.00% | 12 | 0.00% | 95.10% |
| 99801 | value_value | E3_step_cumulative_lambda_0.75 | 1386 | 33.33% | 35.29% | 0.14% | -35.15 pp | 2.0000 | 2.0588 | 1.0043 | -1.0545 | 1392 | 79544 | 79127 | 1.0053 | 4.0000 | 0.00% | 12 | 0.00% | 98.08% |
| 99801 | value_value | E4_extended_candidate_set | 1391 | 33.33% | 34.10% | 0.14% | -33.96 pp | 2.0000 | 2.0230 | 1.0043 | -1.0187 | 1397 | 78507 | 78090 | 1.0053 | 4.0000 | 0.00% | 12 | 0.00% | 95.23% |
| 99802 | best_of_n | E0_standard_actor_ratio | 1365 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2730 | 152379 | 75886 | 2.0080 | 1.5407 | 45.93% | 1365 | 30.26% | 55.07% |
| 99802 | best_of_n | E1_trajectory_propensity | 1375 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2750 | 152996 | 76412 | 2.0023 | 1.5469 | 45.31% | 1375 | 29.96% | 55.66% |
| 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 1382 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2764 | 155113 | 77542 | 2.0004 | 1.5456 | 45.44% | 1382 | 29.81% | 55.70% |
| 99802 | best_of_n | E4_extended_candidate_set | 1370 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2740 | 150365 | 75133 | 2.0013 | 1.5401 | 45.99% | 1370 | 31.02% | 55.06% |
| 99802 | entropy_value | E0_standard_actor_ratio | 1377 | 33.33% | 36.65% | 47.93% | +11.28 pp | 2.0000 | 2.0995 | 2.4379 | +0.3384 | 3357 | 202574 | 78038 | 2.5958 | 2.9970 | 12.42% | 3960 | 12.93% | 72.49% |
| 99802 | entropy_value | E1_trajectory_propensity | 1375 | 33.33% | 36.77% | 48.95% | +12.17 pp | 2.0000 | 2.1031 | 2.4684 | +0.3652 | 3394 | 207031 | 78975 | 2.6215 | 2.9643 | 13.22% | 4038 | 13.45% | 73.02% |
| 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 1371 | 33.33% | 37.39% | 47.85% | +10.46 pp | 2.0000 | 2.1216 | 2.4354 | +0.3138 | 3339 | 204369 | 78971 | 2.5879 | 2.9695 | 14.79% | 3936 | 12.60% | 72.69% |
| 99802 | entropy_value | E4_extended_candidate_set | 1374 | 33.33% | 37.50% | 48.40% | +10.90 pp | 2.0000 | 2.1250 | 2.4520 | +0.3270 | 3369 | 201077 | 77065 | 2.6092 | 2.9579 | 13.53% | 3990 | 13.86% | 71.83% |
| 99802 | value_value | E0_standard_actor_ratio | 1384 | 33.33% | 34.09% | 0.00% | -34.09 pp | 2.0000 | 2.0227 | 1.0000 | -1.0227 | 1384 | 77446 | 77446 | 1.0000 | N/A | N/A | 0 | N/A | N/A |
| 99802 | value_value | E1_trajectory_propensity | 1391 | 33.33% | 35.59% | 0.00% | -35.59 pp | 2.0000 | 2.0676 | 1.0000 | -1.0676 | 1391 | 79187 | 79187 | 1.0000 | N/A | N/A | 0 | N/A | N/A |
| 99802 | value_value | E3_step_cumulative_lambda_0.75 | 1392 | 33.33% | 38.25% | 0.00% | -38.25 pp | 2.0000 | 2.1475 | 1.0000 | -1.1475 | 1392 | 79395 | 79395 | 1.0000 | N/A | N/A | 0 | N/A | N/A |
| 99802 | value_value | E4_extended_candidate_set | 1385 | 33.33% | 35.32% | 0.07% | -35.25 pp | 2.0000 | 2.0596 | 1.0022 | -1.0575 | 1388 | 78691 | 78403 | 1.0037 | 4.0000 | 0.00% | 6 | 0.00% | 99.48% |
| 99803 | best_of_n | E0_standard_actor_ratio | 1377 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2754 | 154703 | 77204 | 2.0038 | 1.5454 | 45.46% | 1377 | 29.70% | 55.46% |
| 99803 | best_of_n | E1_trajectory_propensity | 1380 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2760 | 154504 | 76957 | 2.0077 | 1.5413 | 45.87% | 1380 | 28.84% | 55.56% |
| 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 1381 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2762 | 155533 | 77638 | 2.0033 | 1.5619 | 43.81% | 1381 | 29.33% | 56.78% |
| 99803 | best_of_n | E4_extended_candidate_set | 1374 | 100.00% | 100.00% | 100.00% | +0.00 pp | 2.0000 | 2.0000 | 2.0000 | +0.0000 | 2748 | 152613 | 76237 | 2.0018 | 1.5531 | 44.69% | 1374 | 29.11% | 55.44% |
| 99803 | entropy_value | E0_standard_actor_ratio | 1375 | 33.33% | 29.52% | 47.42% | +17.90 pp | 2.0000 | 1.8855 | 2.4225 | +0.5371 | 3331 | 199799 | 75844 | 2.6343 | 2.9601 | 13.19% | 3912 | 12.99% | 72.70% |
| 99803 | entropy_value | E1_trajectory_propensity | 1371 | 33.33% | 31.72% | 49.09% | +17.37 pp | 2.0000 | 1.9515 | 2.4726 | +0.5211 | 3390 | 207690 | 78595 | 2.6425 | 2.9643 | 13.08% | 4038 | 13.40% | 72.38% |
| 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 1383 | 33.33% | 33.33% | 46.71% | +13.38 pp | 2.0000 | 2.0000 | 2.4013 | +0.4013 | 3321 | 201030 | 77762 | 2.5852 | 2.9706 | 12.54% | 3876 | 13.18% | 72.39% |
| 99803 | entropy_value | E4_extended_candidate_set | 1378 | 33.33% | 31.88% | 48.19% | +16.31 pp | 2.0000 | 1.9563 | 2.4456 | +0.4892 | 3370 | 205950 | 78681 | 2.6175 | 2.9533 | 12.20% | 3984 | 13.18% | 72.45% |
| 99803 | value_value | E0_standard_actor_ratio | 1388 | 33.33% | 37.50% | 0.07% | -37.43 pp | 2.0000 | 2.1250 | 1.0022 | -1.1228 | 1391 | 79344 | 79056 | 1.0036 | 4.0000 | 0.00% | 6 | 0.00% | 92.36% |
| 99803 | value_value | E1_trajectory_propensity | 1398 | 33.33% | 37.99% | 0.14% | -37.85 pp | 2.0000 | 2.1397 | 1.0043 | -1.1354 | 1404 | 79403 | 78849 | 1.0070 | 4.0000 | 0.00% | 12 | 0.00% | 97.31% |
| 99803 | value_value | E3_step_cumulative_lambda_0.75 | 1385 | 33.33% | 35.84% | 0.07% | -35.77 pp | 2.0000 | 2.0752 | 1.0022 | -1.0731 | 1388 | 78195 | 77907 | 1.0037 | 4.0000 | 0.00% | 6 | 0.00% | 99.83% |
| 99803 | value_value | E4_extended_candidate_set | 1387 | 33.33% | 36.77% | 0.14% | -36.63 pp | 2.0000 | 2.1031 | 1.0043 | -1.0988 | 1393 | 78167 | 77591 | 1.0074 | 4.0000 | 0.00% | 12 | 0.00% | 89.24% |

## Training-rollout mechanism metrics

| Seed | Round | Family | Estimator | Accuracy | Answer rate | States | Expansion | Candidates | Mean unique | Fully identical | Candidate pairs | Exact token pairs | Token-prefix divergence |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99801 | 0 | burnin | shared | 89.58% | 95.83% | 206 | 0.00% | 206 | n/a | n/a | 0 | n/a | n/a |
| 99801 | 1 | k1 | E0_standard_actor_ratio | 85.42% | 87.50% | 218 | 0.00% | 218 | n/a | n/a | 0 | n/a | n/a |
| 99801 | 2 | k1 | E0_standard_actor_ratio | 85.42% | 91.67% | 217 | 0.00% | 217 | n/a | n/a | 0 | n/a | n/a |
| 99801 | 1 | best_of_n | shared | 87.50% | 89.58% | 226 | 100.00% | 452 | 1.4204 | 57.96% | 226 | 42.48% | 42.54% |
| 99801 | 2 | best_of_n | E0_standard_actor_ratio | 93.75% | 95.83% | 219 | 100.00% | 438 | 1.4018 | 59.82% | 219 | 45.21% | 41.54% |
| 99801 | 2 | best_of_n | E1_trajectory_propensity | 91.67% | 93.75% | 223 | 100.00% | 446 | 1.3812 | 61.88% | 223 | 43.95% | 41.33% |
| 99801 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 93.75% | 95.83% | 217 | 100.00% | 434 | 1.4055 | 59.45% | 217 | 41.94% | 43.25% |
| 99801 | 2 | best_of_n | E4_extended_candidate_set | 91.67% | 95.83% | 217 | 100.00% | 434 | 1.4055 | 59.45% | 217 | 44.24% | 41.06% |
| 99801 | 1 | entropy_value | shared | 85.42% | 89.58% | 225 | 40.44% | 498 | 2.7253 | 13.19% | 546 | 15.75% | 66.87% |
| 99801 | 2 | entropy_value | E0_standard_actor_ratio | 87.50% | 93.75% | 215 | 37.21% | 455 | 2.7375 | 18.75% | 480 | 12.29% | 68.90% |
| 99801 | 2 | entropy_value | E1_trajectory_propensity | 85.42% | 91.67% | 213 | 34.74% | 435 | 2.8108 | 17.57% | 444 | 9.46% | 71.93% |
| 99801 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 91.67% | 95.83% | 215 | 33.49% | 431 | 2.7361 | 16.67% | 432 | 11.81% | 69.35% |
| 99801 | 2 | entropy_value | E4_extended_candidate_set | 83.33% | 91.67% | 214 | 34.11% | 433 | 2.6849 | 20.55% | 438 | 11.87% | 68.27% |
| 99801 | 1 | value_value | shared | 87.50% | 89.58% | 220 | 35.45% | 454 | 1.6410 | 60.26% | 468 | 62.18% | 30.29% |
| 99801 | 2 | value_value | E0_standard_actor_ratio | 85.42% | 93.75% | 214 | 36.45% | 448 | 1.4872 | 70.51% | 468 | 62.39% | 27.42% |
| 99801 | 2 | value_value | E1_trajectory_propensity | 83.33% | 91.67% | 216 | 36.57% | 453 | 1.5063 | 68.35% | 474 | 61.39% | 26.38% |
| 99801 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 85.42% | 91.67% | 221 | 35.29% | 455 | 1.5513 | 66.67% | 468 | 60.47% | 28.20% |
| 99801 | 2 | value_value | E4_extended_candidate_set | 81.25% | 87.50% | 217 | 34.10% | 439 | 1.5270 | 64.86% | 444 | 60.36% | 27.79% |
| 99802 | 0 | burnin | shared | 91.67% | 97.92% | 200 | 0.00% | 200 | n/a | n/a | 0 | n/a | n/a |
| 99802 | 1 | k1 | E0_standard_actor_ratio | 85.42% | 87.50% | 220 | 0.00% | 220 | n/a | n/a | 0 | n/a | n/a |
| 99802 | 2 | k1 | E0_standard_actor_ratio | 87.50% | 91.67% | 217 | 0.00% | 217 | n/a | n/a | 0 | n/a | n/a |
| 99802 | 1 | best_of_n | shared | 85.42% | 89.58% | 226 | 100.00% | 452 | 1.4027 | 59.73% | 226 | 41.15% | 44.35% |
| 99802 | 2 | best_of_n | E0_standard_actor_ratio | 79.17% | 87.50% | 231 | 100.00% | 462 | 1.3896 | 61.04% | 231 | 45.45% | 41.89% |
| 99802 | 2 | best_of_n | E1_trajectory_propensity | 79.17% | 87.50% | 229 | 100.00% | 458 | 1.3843 | 61.57% | 229 | 44.54% | 42.18% |
| 99802 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 77.08% | 87.50% | 231 | 100.00% | 462 | 1.3680 | 63.20% | 231 | 44.16% | 40.66% |
| 99802 | 2 | best_of_n | E4_extended_candidate_set | 77.08% | 85.42% | 230 | 100.00% | 460 | 1.3783 | 62.17% | 230 | 41.74% | 43.07% |
| 99802 | 1 | entropy_value | shared | 87.50% | 91.67% | 220 | 42.27% | 499 | 2.6667 | 13.98% | 558 | 13.80% | 68.18% |
| 99802 | 2 | entropy_value | E0_standard_actor_ratio | 83.33% | 89.58% | 221 | 36.65% | 464 | 2.8395 | 12.35% | 486 | 13.37% | 71.42% |
| 99802 | 2 | entropy_value | E1_trajectory_propensity | 79.17% | 85.42% | 223 | 36.77% | 469 | 2.9268 | 10.98% | 492 | 10.98% | 73.78% |
| 99802 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 83.33% | 89.58% | 222 | 37.39% | 471 | 2.8313 | 14.46% | 498 | 11.85% | 71.88% |
| 99802 | 2 | entropy_value | E4_extended_candidate_set | 81.25% | 87.50% | 216 | 37.50% | 459 | 2.9383 | 11.11% | 486 | 11.52% | 73.00% |
| 99802 | 1 | value_value | shared | 87.50% | 89.58% | 218 | 35.78% | 452 | 1.5769 | 64.10% | 468 | 65.60% | 27.65% |
| 99802 | 2 | value_value | E0_standard_actor_ratio | 89.58% | 93.75% | 220 | 34.09% | 445 | 1.4667 | 73.33% | 450 | 69.11% | 23.40% |
| 99802 | 2 | value_value | E1_trajectory_propensity | 85.42% | 91.67% | 222 | 35.59% | 459 | 1.5316 | 69.62% | 474 | 65.19% | 25.75% |
| 99802 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 87.50% | 91.67% | 217 | 38.25% | 466 | 1.5783 | 69.88% | 498 | 63.86% | 27.59% |
| 99802 | 2 | value_value | E4_extended_candidate_set | 87.50% | 93.75% | 218 | 35.32% | 449 | 1.6234 | 67.53% | 462 | 64.07% | 28.12% |
| 99803 | 0 | burnin | shared | 91.67% | 95.83% | 195 | 0.00% | 195 | n/a | n/a | 0 | n/a | n/a |
| 99803 | 1 | k1 | E0_standard_actor_ratio | 91.67% | 93.75% | 223 | 0.00% | 223 | n/a | n/a | 0 | n/a | n/a |
| 99803 | 2 | k1 | E0_standard_actor_ratio | 79.17% | 85.42% | 224 | 0.00% | 224 | n/a | n/a | 0 | n/a | n/a |
| 99803 | 1 | best_of_n | shared | 85.42% | 87.50% | 221 | 100.00% | 442 | 1.4434 | 55.66% | 221 | 42.08% | 44.36% |
| 99803 | 2 | best_of_n | E0_standard_actor_ratio | 81.25% | 87.50% | 225 | 100.00% | 450 | 1.3511 | 64.89% | 225 | 43.11% | 41.82% |
| 99803 | 2 | best_of_n | E1_trajectory_propensity | 81.25% | 87.50% | 223 | 100.00% | 446 | 1.3318 | 66.82% | 223 | 41.26% | 43.01% |
| 99803 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 83.33% | 87.50% | 222 | 100.00% | 444 | 1.3649 | 63.51% | 222 | 42.79% | 43.32% |
| 99803 | 2 | best_of_n | E4_extended_candidate_set | 83.33% | 87.50% | 223 | 100.00% | 446 | 1.3587 | 64.13% | 223 | 39.01% | 45.13% |
| 99803 | 1 | entropy_value | shared | 89.58% | 91.67% | 225 | 36.44% | 471 | 2.8293 | 10.98% | 492 | 14.43% | 67.98% |
| 99803 | 2 | entropy_value | E0_standard_actor_ratio | 87.50% | 93.75% | 227 | 29.52% | 428 | 2.7910 | 16.42% | 402 | 10.45% | 72.60% |
| 99803 | 2 | entropy_value | E1_trajectory_propensity | 87.50% | 93.75% | 227 | 31.72% | 443 | 2.8472 | 16.67% | 432 | 9.95% | 74.07% |
| 99803 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 89.58% | 95.83% | 225 | 33.33% | 450 | 2.8400 | 14.67% | 450 | 9.33% | 73.57% |
| 99803 | 2 | entropy_value | E4_extended_candidate_set | 89.58% | 91.67% | 229 | 31.88% | 448 | 2.8356 | 15.07% | 438 | 9.36% | 73.80% |
| 99803 | 1 | value_value | shared | 89.58% | 91.67% | 226 | 34.96% | 463 | 1.5443 | 68.35% | 474 | 69.41% | 23.62% |
| 99803 | 2 | value_value | E0_standard_actor_ratio | 81.25% | 91.67% | 224 | 37.50% | 476 | 1.4762 | 67.86% | 504 | 66.07% | 25.73% |
| 99803 | 2 | value_value | E1_trajectory_propensity | 87.50% | 91.67% | 229 | 37.99% | 490 | 1.5402 | 70.11% | 522 | 65.71% | 26.22% |
| 99803 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 81.25% | 89.58% | 226 | 35.84% | 469 | 1.6049 | 69.14% | 486 | 64.20% | 29.14% |
| 99803 | 2 | value_value | E4_extended_candidate_set | 83.33% | 91.67% | 223 | 36.77% | 469 | 1.5610 | 68.29% | 492 | 64.63% | 27.31% |

## Training-rollout Value calibration and ranking

These are descriptive diagnostics, not additional promotion gates. Raw Brier uses the unconstrained scalar head; ECE10 clips predictions to `[0,1]`; AUC measures ranking. They are state-weighted, so correlated states and trajectory length preclude independent-sample inference.

| Seed | Round | Family | Estimator | Mean reward | Pre mean | Pre Brier | Pre ECE10 | Pre AUC | Post Brier | Post ECE10 | Post AUC | Out-of-unit pre/post |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99801 | 0 | burnin | shared | 0.8689 | 0.0000 | 0.8689 | 0.8689 | 0.5000 | 0.8689 | 0.8689 | 0.5000 | 0.0000/0.0000 |
| 99801 | 1 | k1 | E0_standard_actor_ratio | 0.7890 | 0.1056 | 0.6323 | 0.6834 | 0.6178 | 0.6330 | 0.6835 | 0.5533 | 0.0000/0.0000 |
| 99801 | 2 | k1 | E0_standard_actor_ratio | 0.8065 | 0.2115 | 0.5098 | 0.5950 | 0.5332 | 0.5113 | 0.5959 | 0.4920 | 0.0000/0.0000 |
| 99801 | 1 | best_of_n | shared | 0.8230 | 0.1056 | 0.6593 | 0.7174 | 0.6176 | 0.6596 | 0.7175 | 0.5839 | 0.0000/0.0000 |
| 99801 | 2 | best_of_n | E0_standard_actor_ratio | 0.9269 | 0.2257 | 0.5595 | 0.7013 | 0.5409 | 0.5623 | 0.7031 | 0.5037 | 0.0000/0.0000 |
| 99801 | 2 | best_of_n | E1_trajectory_propensity | 0.8969 | 0.2255 | 0.5440 | 0.6714 | 0.4674 | 0.5466 | 0.6729 | 0.4111 | 0.0000/0.0000 |
| 99801 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.9263 | 0.2250 | 0.5602 | 0.7013 | 0.5317 | 0.5624 | 0.7027 | 0.4900 | 0.0000/0.0000 |
| 99801 | 2 | best_of_n | E4_extended_candidate_set | 0.9171 | 0.2245 | 0.5560 | 0.6926 | 0.5159 | 0.5586 | 0.6943 | 0.4821 | 0.0000/0.0000 |
| 99801 | 1 | entropy_value | shared | 0.8044 | 0.1059 | 0.6444 | 0.6985 | 0.5962 | 0.6450 | 0.6987 | 0.5561 | 0.0000/0.0000 |
| 99801 | 2 | entropy_value | E0_standard_actor_ratio | 0.8558 | 0.2261 | 0.5205 | 0.6297 | 0.4646 | 0.5237 | 0.6318 | 0.4272 | 0.0000/0.0000 |
| 99801 | 2 | entropy_value | E1_trajectory_propensity | 0.8028 | 0.2257 | 0.4903 | 0.5771 | 0.5809 | 0.4930 | 0.5791 | 0.5322 | 0.0000/0.0000 |
| 99801 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.9023 | 0.2253 | 0.5464 | 0.6770 | 0.5214 | 0.5491 | 0.6786 | 0.4588 | 0.0000/0.0000 |
| 99801 | 2 | entropy_value | E4_extended_candidate_set | 0.7944 | 0.2207 | 0.4922 | 0.5737 | 0.5313 | 0.4953 | 0.5751 | 0.4385 | 0.0000/0.0000 |
| 99801 | 1 | value_value | shared | 0.8227 | 0.1061 | 0.6585 | 0.7166 | 0.6058 | 0.6592 | 0.7173 | 0.6070 | 0.0000/0.0000 |
| 99801 | 2 | value_value | E0_standard_actor_ratio | 0.8271 | 0.2122 | 0.5202 | 0.6149 | 0.5612 | 0.5224 | 0.6166 | 0.5398 | 0.0000/0.0000 |
| 99801 | 2 | value_value | E1_trajectory_propensity | 0.7917 | 0.2122 | 0.5002 | 0.5795 | 0.5384 | 0.5026 | 0.5812 | 0.4998 | 0.0000/0.0000 |
| 99801 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 0.8281 | 0.2119 | 0.5226 | 0.6161 | 0.4630 | 0.5253 | 0.6177 | 0.4111 | 0.0000/0.0000 |
| 99801 | 2 | value_value | E4_extended_candidate_set | 0.7604 | 0.2115 | 0.4838 | 0.5489 | 0.4903 | 0.4865 | 0.5504 | 0.4305 | 0.0000/0.0000 |
| 99802 | 0 | burnin | shared | 0.8950 | 0.0000 | 0.8950 | 0.8950 | 0.5000 | 0.8950 | 0.8950 | 0.5000 | 0.0000/0.0000 |
| 99802 | 1 | k1 | E0_standard_actor_ratio | 0.7864 | 0.1064 | 0.6296 | 0.6800 | 0.5779 | 0.6312 | 0.6805 | 0.4887 | 0.0000/0.0000 |
| 99802 | 2 | k1 | E0_standard_actor_ratio | 0.8433 | 0.2121 | 0.5297 | 0.6312 | 0.5683 | 0.5333 | 0.6333 | 0.5023 | 0.0000/0.0000 |
| 99802 | 1 | best_of_n | shared | 0.7920 | 0.1063 | 0.6343 | 0.6857 | 0.5667 | 0.6361 | 0.6864 | 0.4719 | 0.0000/0.0000 |
| 99802 | 2 | best_of_n | E0_standard_actor_ratio | 0.7359 | 0.2255 | 0.4534 | 0.5104 | 0.5851 | 0.4569 | 0.5116 | 0.4717 | 0.0000/0.0000 |
| 99802 | 2 | best_of_n | E1_trajectory_propensity | 0.7336 | 0.2257 | 0.4525 | 0.5079 | 0.5594 | 0.4560 | 0.5092 | 0.4496 | 0.0000/0.0000 |
| 99802 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.7186 | 0.2232 | 0.4472 | 0.4954 | 0.5328 | 0.4503 | 0.4966 | 0.4473 | 0.0000/0.0000 |
| 99802 | 2 | best_of_n | E4_extended_candidate_set | 0.7043 | 0.2231 | 0.4387 | 0.4813 | 0.5680 | 0.4422 | 0.4826 | 0.4591 | 0.0000/0.0000 |
| 99802 | 1 | entropy_value | shared | 0.8227 | 0.1063 | 0.6587 | 0.7164 | 0.5487 | 0.6605 | 0.7171 | 0.4547 | 0.0000/0.0000 |
| 99802 | 2 | entropy_value | E0_standard_actor_ratio | 0.7692 | 0.2131 | 0.4861 | 0.5562 | 0.5404 | 0.4899 | 0.5582 | 0.4629 | 0.0000/0.0000 |
| 99802 | 2 | entropy_value | E1_trajectory_propensity | 0.7085 | 0.2124 | 0.4504 | 0.4962 | 0.5986 | 0.4538 | 0.4983 | 0.5311 | 0.0000/0.0000 |
| 99802 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.7748 | 0.2122 | 0.4886 | 0.5626 | 0.6319 | 0.4923 | 0.5646 | 0.5469 | 0.0000/0.0000 |
| 99802 | 2 | entropy_value | E4_extended_candidate_set | 0.7361 | 0.2063 | 0.4729 | 0.5298 | 0.5914 | 0.4766 | 0.5312 | 0.4941 | 0.0000/0.0000 |
| 99802 | 1 | value_value | shared | 0.8165 | 0.1065 | 0.6538 | 0.7100 | 0.5312 | 0.6557 | 0.7109 | 0.4515 | 0.0000/0.0000 |
| 99802 | 2 | value_value | E0_standard_actor_ratio | 0.8591 | 0.2129 | 0.5372 | 0.6462 | 0.6015 | 0.5405 | 0.6480 | 0.5291 | 0.0000/0.0000 |
| 99802 | 2 | value_value | E1_trajectory_propensity | 0.8198 | 0.2126 | 0.5150 | 0.6073 | 0.5875 | 0.5178 | 0.6087 | 0.5176 | 0.0000/0.0000 |
| 99802 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 0.8249 | 0.2087 | 0.5221 | 0.6162 | 0.6373 | 0.5256 | 0.6185 | 0.5676 | 0.0000/0.0000 |
| 99802 | 2 | value_value | E4_extended_candidate_set | 0.8578 | 0.2127 | 0.5379 | 0.6451 | 0.5204 | 0.5414 | 0.6473 | 0.4682 | 0.0000/0.0000 |
| 99803 | 0 | burnin | shared | 0.8821 | 0.0000 | 0.8821 | 0.8821 | 0.5000 | 0.8821 | 0.8821 | 0.5000 | 0.0000/0.0000 |
| 99803 | 1 | k1 | E0_standard_actor_ratio | 0.8879 | 0.1031 | 0.7154 | 0.7848 | 0.5459 | 0.7170 | 0.7856 | 0.4622 | 0.0000/0.0000 |
| 99803 | 2 | k1 | E0_standard_actor_ratio | 0.7277 | 0.2071 | 0.4669 | 0.5206 | 0.6115 | 0.4714 | 0.5228 | 0.5073 | 0.0000/0.0000 |
| 99803 | 1 | best_of_n | shared | 0.7919 | 0.1033 | 0.6381 | 0.6886 | 0.6104 | 0.6395 | 0.6891 | 0.5293 | 0.0000/0.0000 |
| 99803 | 2 | best_of_n | E0_standard_actor_ratio | 0.7422 | 0.2071 | 0.4755 | 0.5351 | 0.6188 | 0.4793 | 0.5378 | 0.5619 | 0.0000/0.0000 |
| 99803 | 2 | best_of_n | E1_trajectory_propensity | 0.7578 | 0.2075 | 0.4848 | 0.5504 | 0.5989 | 0.4884 | 0.5529 | 0.5476 | 0.0000/0.0000 |
| 99803 | 2 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.7748 | 0.2070 | 0.4954 | 0.5678 | 0.5910 | 0.4997 | 0.5709 | 0.5421 | 0.0000/0.0000 |
| 99803 | 2 | best_of_n | E4_extended_candidate_set | 0.7713 | 0.2073 | 0.4928 | 0.5640 | 0.5920 | 0.4963 | 0.5663 | 0.5432 | 0.0000/0.0000 |
| 99803 | 1 | entropy_value | shared | 0.8578 | 0.1036 | 0.6903 | 0.7542 | 0.5803 | 0.6918 | 0.7550 | 0.5387 | 0.0000/0.0000 |
| 99803 | 2 | entropy_value | E0_standard_actor_ratio | 0.8458 | 0.2213 | 0.5188 | 0.6245 | 0.6412 | 0.5227 | 0.6269 | 0.5598 | 0.0000/0.0000 |
| 99803 | 2 | entropy_value | E1_trajectory_propensity | 0.8458 | 0.2218 | 0.5189 | 0.6240 | 0.5847 | 0.5229 | 0.6263 | 0.4985 | 0.0000/0.0000 |
| 99803 | 2 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.8889 | 0.2214 | 0.5444 | 0.6675 | 0.5142 | 0.5488 | 0.6699 | 0.4100 | 0.0000/0.0000 |
| 99803 | 2 | entropy_value | E4_extended_candidate_set | 0.8646 | 0.2183 | 0.5336 | 0.6463 | 0.6212 | 0.5379 | 0.6491 | 0.5474 | 0.0000/0.0000 |
| 99803 | 1 | value_value | shared | 0.8584 | 0.1035 | 0.6906 | 0.7549 | 0.6142 | 0.6926 | 0.7558 | 0.5298 | 0.0000/0.0000 |
| 99803 | 2 | value_value | E0_standard_actor_ratio | 0.7723 | 0.2228 | 0.4787 | 0.5496 | 0.4766 | 0.4829 | 0.5522 | 0.4177 | 0.0000/0.0000 |
| 99803 | 2 | value_value | E1_trajectory_propensity | 0.8253 | 0.2235 | 0.5061 | 0.6019 | 0.5459 | 0.5113 | 0.6051 | 0.4628 | 0.0000/0.0000 |
| 99803 | 2 | value_value | E3_step_cumulative_lambda_0.75 | 0.7611 | 0.2225 | 0.4707 | 0.5386 | 0.5807 | 0.4753 | 0.5411 | 0.4778 | 0.0000/0.0000 |
| 99803 | 2 | value_value | E4_extended_candidate_set | 0.7848 | 0.2234 | 0.4846 | 0.5614 | 0.4912 | 0.4890 | 0.5640 | 0.4199 | 0.0000/0.0000 |

## Matched-compute evidence

| Family | Estimator | States | Updates | Candidate forwards | Candidate tokens | Selected tokens | Forward ratio vs E0 | Token ratio vs E0 | Invocation-time ratio vs E0 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| best_of_n | E0_standard_actor_ratio | 1348 | 46 | 2696 | 119373 | 59564 | 1.0000 | 1.0000 | 1.0000 |
| best_of_n | E1_trajectory_propensity | 1348 | 45 | 2696 | 118777 | 59236 | 1.0000 | 0.9950 | 0.9979 |
| best_of_n | E3_step_cumulative_lambda_0.75 | 1343 | 45 | 2686 | 118720 | 59203 | 0.9963 | 0.9945 | 0.9855 |
| best_of_n | E4_extended_candidate_set | 1343 | 45 | 2686 | 117950 | 58871 | 0.9963 | 0.9881 | 0.9996 |
| entropy_value | E0_standard_actor_ratio | 1333 | 45 | 2815 | 139124 | 57743 | 1.0000 | 1.0000 | 1.0000 |
| entropy_value | E1_trajectory_propensity | 1333 | 45 | 2815 | 140305 | 58229 | 1.0000 | 1.0085 | 0.9917 |
| entropy_value | E3_step_cumulative_lambda_0.75 | 1332 | 45 | 2820 | 139972 | 58053 | 1.0018 | 1.0061 | 0.9732 |
| entropy_value | E4_extended_candidate_set | 1329 | 45 | 2808 | 139253 | 57903 | 0.9975 | 1.0009 | 0.9544 |
| value_value | E0_standard_actor_ratio | 1322 | 43 | 2738 | 101373 | 58004 | 1.0000 | 1.0000 | 1.0000 |
| value_value | E1_trajectory_propensity | 1331 | 44 | 2771 | 102205 | 58040 | 1.0121 | 1.0082 | 0.9980 |
| value_value | E3_step_cumulative_lambda_0.75 | 1328 | 44 | 2759 | 102462 | 58184 | 1.0077 | 1.0107 | 0.9989 |
| value_value | E4_extended_candidate_set | 1322 | 43 | 2726 | 100801 | 57795 | 0.9956 | 0.9944 | 0.9896 |

## Per-seed paired evidence

The frozen JSON stores the protocol-required per-seed paired differences even though its Markdown report shows pooled W/L/T only.
The 10,000-resample clustered interval resamples the 200 problem columns while retaining all three seed observations for each problem. It therefore quantifies confirmation-problem uncertainty conditional on these three training seeds; it is not a precise estimate of training-seed superpopulation uncertainty.

| Contrast | Family | Seed | Difference | W/L/T |
|---|---|---:|---:|---:|
| raw_E3_minus_E0 | best_of_n | 99801 | -1.00 pp | 1/3/196 |
| raw_E3_minus_E0 | best_of_n | 99802 | +0.00 pp | 5/5/190 |
| raw_E3_minus_E0 | best_of_n | 99803 | +0.00 pp | 3/3/194 |
| raw_E3_minus_E0 | entropy_value | 99801 | +1.50 pp | 4/1/195 |
| raw_E3_minus_E0 | entropy_value | 99802 | +0.50 pp | 5/4/191 |
| raw_E3_minus_E0 | entropy_value | 99803 | -1.00 pp | 0/2/198 |
| raw_E3_minus_E0 | value_value | 99801 | -1.50 pp | 2/5/193 |
| raw_E3_minus_E0 | value_value | 99802 | -1.00 pp | 3/5/192 |
| raw_E3_minus_E0 | value_value | 99803 | +1.50 pp | 5/2/193 |
| raw_E1_minus_E0 | best_of_n | 99801 | -1.00 pp | 3/5/192 |
| raw_E1_minus_E0 | best_of_n | 99802 | -2.00 pp | 0/4/196 |
| raw_E1_minus_E0 | best_of_n | 99803 | +1.50 pp | 5/2/193 |
| raw_E1_minus_E0 | entropy_value | 99801 | +0.00 pp | 3/3/194 |
| raw_E1_minus_E0 | entropy_value | 99802 | -2.00 pp | 4/8/188 |
| raw_E1_minus_E0 | entropy_value | 99803 | -0.50 pp | 2/3/195 |
| raw_E1_minus_E0 | value_value | 99801 | -1.00 pp | 3/5/192 |
| raw_E1_minus_E0 | value_value | 99802 | -0.50 pp | 4/5/191 |
| raw_E1_minus_E0 | value_value | 99803 | +1.50 pp | 6/3/191 |
| composite_E4_minus_E0 | best_of_n | 99801 | +1.50 pp | 6/3/191 |
| composite_E4_minus_E0 | best_of_n | 99802 | -1.00 pp | 2/4/194 |
| composite_E4_minus_E0 | best_of_n | 99803 | +0.00 pp | 6/6/188 |
| composite_E4_minus_E0 | entropy_value | 99801 | -2.00 pp | 3/7/190 |
| composite_E4_minus_E0 | entropy_value | 99802 | +0.50 pp | 3/2/195 |
| composite_E4_minus_E0 | entropy_value | 99803 | -1.00 pp | 5/7/188 |
| composite_E4_minus_E0 | value_value | 99801 | +0.50 pp | 3/2/195 |
| composite_E4_minus_E0 | value_value | 99802 | +2.00 pp | 6/2/192 |
| composite_E4_minus_E0 | value_value | 99803 | +0.50 pp | 4/3/193 |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99801 | +1.00 pp | 5/3/192 |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99802 | +0.00 pp | 4/4/192 |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99803 | -4.00 pp | 1/9/190 |

### Accuracy and finalization decomposition

Each registered overall-accuracy contrast is decomposed into its answer-rate difference and descriptive correctness difference conditional on producing a parseable final answer. A difference driven by finalization remains an observed deployment-budget effect; it is not evidence of improved mathematical correctness conditional on answering. This diagnostic does not replace the registered overall-accuracy estimand.

| Contrast | Family | Seed | Accuracy difference | Answer-rate difference | LHS conditional accuracy | RHS conditional accuracy | Conditional-accuracy difference |
|---|---|---:|---:|---:|---:|---:|---:|
| raw_E3_minus_E0 | best_of_n | 99801 | -1.00 pp | -2.00 pp | 85.90% | 84.15% | +1.75 pp |
| raw_E3_minus_E0 | best_of_n | 99802 | +0.00 pp | +1.00 pp | 86.08% | 88.31% | -2.24 pp |
| raw_E3_minus_E0 | best_of_n | 99803 | +0.00 pp | -0.50 pp | 87.67% | 86.49% | +1.18 pp |
| raw_E3_minus_E0 | entropy_value | 99801 | +1.50 pp | +1.00 pp | 86.84% | 85.14% | +1.71 pp |
| raw_E3_minus_E0 | entropy_value | 99802 | +0.50 pp | +0.50 pp | 87.34% | 87.18% | +0.16 pp |
| raw_E3_minus_E0 | entropy_value | 99803 | -1.00 pp | -0.50 pp | 85.71% | 87.18% | -1.47 pp |
| raw_E3_minus_E0 | value_value | 99801 | -1.50 pp | -1.50 pp | 86.67% | 87.18% | -0.51 pp |
| raw_E3_minus_E0 | value_value | 99802 | -1.00 pp | -1.00 pp | 86.84% | 87.18% | -0.34 pp |
| raw_E3_minus_E0 | value_value | 99803 | +1.50 pp | +1.00 pp | 86.08% | 84.42% | +1.66 pp |
| raw_E1_minus_E0 | best_of_n | 99801 | -1.00 pp | -1.50 pp | 84.81% | 84.15% | +0.66 pp |
| raw_E1_minus_E0 | best_of_n | 99802 | -2.00 pp | -1.50 pp | 86.49% | 88.31% | -1.83 pp |
| raw_E1_minus_E0 | best_of_n | 99803 | +1.50 pp | +1.50 pp | 87.01% | 86.49% | +0.53 pp |
| raw_E1_minus_E0 | entropy_value | 99801 | +0.00 pp | +0.00 pp | 85.14% | 85.14% | +0.00 pp |
| raw_E1_minus_E0 | entropy_value | 99802 | -2.00 pp | -1.50 pp | 85.33% | 87.18% | -1.85 pp |
| raw_E1_minus_E0 | entropy_value | 99803 | -0.50 pp | -1.00 pp | 88.16% | 87.18% | +0.98 pp |
| raw_E1_minus_E0 | value_value | 99801 | -1.00 pp | -1.00 pp | 86.84% | 87.18% | -0.34 pp |
| raw_E1_minus_E0 | value_value | 99802 | -0.50 pp | +0.00 pp | 85.90% | 87.18% | -1.28 pp |
| raw_E1_minus_E0 | value_value | 99803 | +1.50 pp | +0.50 pp | 87.18% | 84.42% | +2.76 pp |
| composite_E4_minus_E0 | best_of_n | 99801 | +1.50 pp | +0.50 pp | 82.56% | 80.00% | +2.56 pp |
| composite_E4_minus_E0 | best_of_n | 99802 | -1.00 pp | -1.50 pp | 83.72% | 83.15% | +0.57 pp |
| composite_E4_minus_E0 | best_of_n | 99803 | +0.00 pp | +0.50 pp | 81.71% | 82.72% | -1.01 pp |
| composite_E4_minus_E0 | entropy_value | 99801 | -2.00 pp | -2.50 pp | 84.71% | 84.44% | +0.26 pp |
| composite_E4_minus_E0 | entropy_value | 99802 | +0.50 pp | +1.00 pp | 83.15% | 83.91% | -0.76 pp |
| composite_E4_minus_E0 | entropy_value | 99803 | -1.00 pp | -2.50 pp | 84.34% | 81.82% | +2.52 pp |
| composite_E4_minus_E0 | value_value | 99801 | +0.50 pp | -0.50 pp | 85.00% | 82.72% | +2.28 pp |
| composite_E4_minus_E0 | value_value | 99802 | +2.00 pp | +0.50 pp | 85.71% | 81.93% | +3.79 pp |
| composite_E4_minus_E0 | value_value | 99803 | +0.50 pp | +1.00 pp | 83.33% | 84.15% | -0.81 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99801 | +1.00 pp | +0.50 pp | 85.90% | 84.42% | +1.48 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99802 | +0.00 pp | +0.00 pp | 86.08% | 86.08% | +0.00 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99803 | -4.00 pp | -5.00 pp | 87.67% | 86.75% | +0.92 pp |

## GPU and session audit

- Successful sessions: 1.
- Multiple runner sessions: False.
- Reserved GPU-hours: 108.585833.
- GPU UUID: `GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48`.
- GPU samples: 78145.

## Remote post-run Torch receipt

- Tests: 18.
- Failures/errors/skips: 0/0/0.
- Log SHA-256: `341d5dca33250b815baddbafb6001e174c20bf1716bb74f3d612e77777db8938`.
- JUnit SHA-256: `795ed366b86eece4ac3f59cbabbdc6f22cb30f3483afb4170f0497a2a0cafe75`.

## Archive SHA-256 receipt

This receipt is generated only after both terminal completion markers and the full artifact matrix pass the independent audit. The frozen analysis JSON contains the per-artifact reproducibility manifest; the table below binds its inputs, generator, outputs, and this administrative audit implementation.

| Artifact role | SHA-256 |
|---|---|
| `administrative_archive_audit_source` | `be637e6de32c2689c80b38a2fe2d7fe0e83f9710fd1ffe6ad79ecec131b425c9` |
| `evaluation_data` | `471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7` |
| `evaluation_manifest` | `ec96c0b63eb0dadb6132e12961c2efee85a11b9df1e92ca7a29a9933d560008d` |
| `frozen_analysis_json` | `c97074306680e740862827f526fc23bb0ef5fc58938a935e12861d2c2eb3905d` |
| `frozen_analysis_source` | `dc30d59b790bae341d97786817c90463bbb6a7dc77315ec424392604b26f2c27` |
| `frozen_protocol` | `be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978` |
| `frozen_report` | `b0a24c4405ff3db3ef8b9c2ed191e5841536166dc14d942cae9890f84ec89001` |
| `frozen_runner` | `466dd3608434f2244abd8861d55a554fee6d4c34aa2d17319423187acaf966a1` |
| `iclr_evidence_table_renderer_source` | `631738c8a92e61265fe805bcac4312379ce1f416e298a4ae132a9cfcd39f6c55` |
| `iclr_figure_renderer_source` | `de0c8fc0f94e8417e0159ce52190407012a969634f76404a7c37d9101bfe5736` |
| `iclr_manuscript_review_recorder_source` | `22b398c38ca61de4983e833b2528ba63a2530b7be823d070432f409d6479313f` |
| `iclr_terminal_results_renderer_source` | `f9df8c39c2cedf668962640159c25125a8fa409ac543023979a80347b6124d97` |
| `manual_compute_review_preparer_source` | `fe609534b44916a902d09df7cd453612fa1bd19532691bb91381483718388bd9` |
| `manual_compute_review_receipt` | `999f205e1b2e2e933d9efb809da8a419b9427c1921c72efaf87d4d075a1e0dd4` |
| `manual_compute_review_recorder_source` | `39ca61901f2e71e61c6a3943724324f10cab4251c98d10b2b30fa41aa7665221` |
| `remote_postrun_test_junit` | `795ed366b86eece4ac3f59cbabbdc6f22cb30f3483afb4170f0497a2a0cafe75` |
| `remote_postrun_test_log` | `341d5dca33250b815baddbafb6001e174c20bf1716bb74f3d612e77777db8938` |
| `remote_postrun_test_source` | `7c03595abbaf13b33e389953a6108eb637004d4c96ad3735e51c0de4fd09cf75` |
| `s3_final_archive_manifest_builder_source` | `1c00bb9b6227a0d26dbccc411b0a2c6665d8c0bcf6d9dd7f97305aecad910826` |

## Decision boundary

Manual compute-competitiveness review: **PASS**.
A structurally complete archive with any failed frozen gate remains a mixed/negative S3 result and does not authorize a next-stage run.
