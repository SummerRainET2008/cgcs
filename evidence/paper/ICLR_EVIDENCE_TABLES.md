# ICLR evidence tables

Status: **evidence complete**. Generated from terminal validated artifacts; not a protocol amendment.

## Table 1 — estimator definitions

| Estimator | Implementation | Target | Correction | Main finite-sample issue |
|---|---|---|---|---|
| E0 | standard Actor ratio | search-distilled surrogate | none for search allocation/retention | selection and occupancy bias |
| E1 | trajectory propensity | raw-Actor objective | full candidate-selection trajectory | product weights and low ESS |
| E2 | executed-Action marginal | raw-Actor objective | Rao--Blackwellized executed Action | marginal probability computation |
| E3 | partial cumulative correction | raw-Actor objective with controlled bias | development-frozen lambda | sample-size-dependent bias--variance tuning |
| E4 | extended retained candidate set | retained composite-search objective | scores the latent retained candidate set | different estimand and greater compute |

## Table 2 — cross-stage evidence

### Joint Actor–Value mechanism replication

| Arm | Position | n | AUC | Brier (clipped) | ECE-10 (clipped) |
|---|---|---:|---:|---:|---:|
| k1 | pre_action | 1323 | 0.5134 | 0.5311 | 0.5872 |
| k1 | selected_post_action | 1323 | 0.4813 | 0.5336 | 0.5881 |
| random_uniform | pre_action | 1308 | 0.5059 | 0.5304 | 0.5834 |
| random_uniform | selected_post_action | 1308 | 0.4755 | 0.5326 | 0.5840 |
| entropy_uniform | pre_action | 1308 | 0.4831 | 0.5424 | 0.6001 |
| entropy_uniform | selected_post_action | 1308 | 0.4522 | 0.5447 | 0.6009 |
| value_gate_uniform | pre_action | 1331 | 0.4671 | 0.5247 | 0.5710 |
| value_gate_uniform | selected_post_action | 1331 | 0.4318 | 0.5269 | 0.5715 |
| random_value | pre_action | 1333 | 0.4781 | 0.5322 | 0.5844 |
| random_value | selected_post_action | 1333 | 0.4466 | 0.5345 | 0.5851 |
| value_guided | pre_action | 1324 | 0.4719 | 0.5265 | 0.5754 |
| value_guided | selected_post_action | 1324 | 0.4392 | 0.5288 | 0.5759 |

### Exact-tree gate calibration and diversity

| Condition | Gate | Allocation AUC | Allocation Brier | Return gain | Mean unique | Identical prob. |
|---|---|---:|---:|---:|---:|---:|
| K2_B1.5 | random_uniform | 0.5000 | 0.2398 | 0.0000 | 1.7520 | 0.2480 |
| K2_B1.5 | entropy_uniform | 0.5361 | 0.2703 | 0.0000 | 1.7745 | 0.2255 |
| K2_B1.5 | value_uniform | 0.4969 | 0.2849 | -0.0000 | 1.7533 | 0.2467 |
| K2_B1.5 | value_and_entropy_uniform | 0.5147 | 0.2695 | 0.0000 | 1.7673 | 0.2327 |
| K2_B1.5 | learned_uniform | 0.5150 | 0.2398 | 0.0000 | 1.7522 | 0.2478 |
| K2_B1.5 | oracle_gain_uniform | 0.7138 | 0.2046 | 0.0000 | 1.7591 | 0.2409 |
| K2_B2 | random_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K2_B2 | entropy_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K2_B2 | value_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K2_B2 | value_and_entropy_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K2_B2 | learned_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K2_B2 | oracle_gain_uniform | 0.5000 | 0.0000 | 0.0000 | 1.7520 | 0.2480 |
| K4_B1.5 | random_uniform | 0.5000 | 0.1370 | 0.0000 | 2.8063 | 0.0330 |
| K4_B1.5 | entropy_uniform | 0.5330 | 0.1450 | 0.0000 | 2.9117 | 0.0178 |
| K4_B1.5 | value_uniform | 0.5259 | 0.1475 | 0.0000 | 2.8081 | 0.0333 |
| K4_B1.5 | value_and_entropy_uniform | 0.5265 | 0.1445 | 0.0000 | 2.8812 | 0.0211 |
| K4_B1.5 | learned_uniform | 0.5073 | 0.1369 | 0.0000 | 2.8081 | 0.0327 |
| K4_B1.5 | oracle_gain_uniform | 0.5538 | 0.1476 | -0.0000 | 2.8411 | 0.0269 |
| K4_B2 | random_uniform | 0.5000 | 0.2160 | 0.0000 | 2.8063 | 0.0330 |
| K4_B2 | entropy_uniform | 0.5153 | 0.2423 | 0.0000 | 2.8969 | 0.0194 |
| K4_B2 | value_uniform | 0.5121 | 0.2445 | 0.0000 | 2.8098 | 0.0329 |
| K4_B2 | value_and_entropy_uniform | 0.5011 | 0.2404 | 0.0000 | 2.8686 | 0.0228 |
| K4_B2 | learned_uniform | 0.5095 | 0.2159 | 0.0000 | 2.8077 | 0.0328 |
| K4_B2 | oracle_gain_uniform | 0.6210 | 0.2160 | 0.0000 | 2.8357 | 0.0279 |

### Exact-tree Critic ranking diagnostics

| Condition | Gate | n | Ranking vs gain | Ranking vs TV | Ranking vs ESS | Ranking vs selector entropy |
|---|---|---:|---:|---:|---:|---:|
| K2_B1.5 | entropy | 270 | 0.9310 | 0.0254 | 0.1817 | -0.0738 |
| K2_B1.5 | random | 270 | 0.9304 | 0.0412 | 0.0917 | -0.0755 |
| K4_B1.5 | entropy | 270 | 0.9209 | 0.0114 | -0.1896 | -0.3070 |
| K4_B1.5 | random | 270 | 0.9318 | 0.0406 | -0.2809 | -0.2747 |
| K4_B2 | entropy | 270 | 0.9273 | 0.0104 | -0.1535 | -0.3029 |
| K4_B2 | random | 270 | 0.9311 | 0.0485 | -0.2321 | -0.2736 |

### Exact-tree estimator bias--variance evidence

| Family | Estimator | Relative bias | MSE@256 | MSE@1024 | ESS | Raw-Actor change | Composite change |
|---|---|---:|---:|---:|---:|---:|---:|
| best_of_n | E0_standard_selected_action | 0.3691 | 0.2480 | 0.1716 | 1.0000 | 0.3774 | 0.4005 |
| best_of_n | E1_trajectory | 0.0686 | 0.8400 | 0.2138 | 0.1515 | 0.3845 | 0.3688 |
| best_of_n | E2_cumulative_marginal | 0.0371 | 0.2347 | 0.0597 | 0.5066 | 0.3847 | 0.3689 |
| best_of_n | E3_step_cumulative_lambda_0.75 | 0.0958 | 0.1764 | 0.0515 | 0.6742 | 0.3863 | 0.3800 |
| best_of_n | E4_extended_candidate_set | 0.0338 | 0.1730 | 0.0441 | 1.0000 | 0.3799 | 0.4027 |
| entropy_value | E0_standard_selected_action | 0.2494 | 0.1735 | 0.0932 | 1.0000 | 0.3850 | 0.3788 |
| entropy_value | E1_trajectory | 0.0419 | 0.3392 | 0.0862 | 0.3176 | 0.3846 | 0.3631 |
| entropy_value | E2_cumulative_marginal | 0.0280 | 0.1298 | 0.0331 | 0.7786 | 0.3845 | 0.3630 |
| entropy_value | E3_step_cumulative_lambda_0.75 | 0.0687 | 0.1220 | 0.0342 | 0.8676 | 0.3862 | 0.3685 |
| entropy_value | E4_extended_candidate_set | 0.0350 | 0.2017 | 0.0514 | 1.0000 | 0.3842 | 0.3803 |
| value_value | E0_standard_selected_action | 0.2341 | 0.1681 | 0.0852 | 1.0000 | 0.3886 | 0.3765 |
| value_value | E1_trajectory | 0.0455 | 0.3117 | 0.0796 | 0.3199 | 0.3841 | 0.3584 |
| value_value | E2_cumulative_marginal | 0.0259 | 0.1263 | 0.0321 | 0.7730 | 0.3845 | 0.3586 |
| value_value | E3_step_cumulative_lambda_0.75 | 0.0672 | 0.1199 | 0.0335 | 0.8663 | 0.3870 | 0.3646 |
| value_value | E4_extended_candidate_set | 0.0316 | 0.1944 | 0.0494 | 1.0000 | 0.3869 | 0.3778 |

### Exact-tree search-shift conditions

| Critic condition | Family | MDPs | Ranking concordance | Search gain | Occupancy TV | Headroom to exact |
|---|---|---:|---:|---:|---:|---:|
| exact | best_of_n | 30 | 1.0000 | 0.7237 | 0.1605 | 0.0000 |
| exact | entropy_value | 30 | 1.0000 | 0.4186 | 0.0941 | 0.0000 |
| exact | value_value | 30 | 1.0000 | 0.4036 | 0.0962 | 0.0000 |
| inverted_diagnostic | best_of_n | 30 | 0.0000 | -0.7106 | 0.1587 | 1.4343 |
| inverted_diagnostic | entropy_value | 30 | 0.0000 | -0.4084 | 0.0947 | 0.8270 |
| inverted_diagnostic | value_value | 30 | 0.0000 | -0.3972 | 0.0969 | 0.8009 |
| lagged_kl_0.01 | best_of_n | 30 | 0.9735 | 0.7136 | 0.1584 | 0.0101 |
| lagged_kl_0.01 | entropy_value | 30 | 0.9735 | 0.4141 | 0.0931 | 0.0045 |
| lagged_kl_0.01 | value_value | 30 | 0.9735 | 0.3998 | 0.0953 | 0.0038 |
| lagged_kl_0.04 | best_of_n | 30 | 0.9516 | 0.6977 | 0.1566 | 0.0260 |
| lagged_kl_0.04 | entropy_value | 30 | 0.9516 | 0.4061 | 0.0924 | 0.0125 |
| lagged_kl_0.04 | value_value | 30 | 0.9516 | 0.3936 | 0.0942 | 0.0101 |
| learned_1024 | best_of_n | 30 | 0.6997 | 0.6146 | 0.1504 | 0.1091 |
| learned_1024 | entropy_value | 30 | 0.6997 | 0.3545 | 0.0876 | 0.0641 |
| learned_1024 | value_value | 30 | 0.6997 | 0.3411 | 0.0906 | 0.0625 |
| learned_256 | best_of_n | 30 | 0.6510 | 0.5502 | 0.1457 | 0.1734 |
| learned_256 | entropy_value | 30 | 0.6510 | 0.3160 | 0.0849 | 0.1026 |
| learned_256 | value_value | 30 | 0.6510 | 0.3086 | 0.0879 | 0.0950 |
| noise_0.5 | best_of_n | 30 | 0.8803 | 0.6686 | 0.1565 | 0.0551 |
| noise_0.5 | entropy_value | 30 | 0.8803 | 0.3869 | 0.0928 | 0.0317 |
| noise_0.5 | value_value | 30 | 0.8803 | 0.3731 | 0.0954 | 0.0305 |
| noise_1 | best_of_n | 30 | 0.7851 | 0.5413 | 0.1489 | 0.1823 |
| noise_1 | entropy_value | 30 | 0.7851 | 0.3110 | 0.0898 | 0.1076 |
| noise_1 | value_value | 30 | 0.7851 | 0.2980 | 0.0912 | 0.1056 |
| noise_2 | best_of_n | 30 | 0.6719 | 0.3402 | 0.1427 | 0.3835 |
| noise_2 | entropy_value | 30 | 0.6719 | 0.1915 | 0.0855 | 0.2271 |
| noise_2 | value_value | 30 | 0.6719 | 0.1816 | 0.0863 | 0.2220 |

### S3 Value diagnostics (descriptive aggregation)

| Family | Estimator | Position | Rollouts | States | AUC | Raw Brier | Clipped ECE-10 | Out-of-unit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| best_of_n | E0_standard_actor_ratio | candidate_zero_post_state | 3 | 675 | 0.5122 | 0.4986 | 0.5825 | 0.0000 |
| best_of_n | E0_standard_actor_ratio | pre_state | 3 | 675 | 0.5820 | 0.4952 | 0.5806 | 0.0000 |
| best_of_n | E1_trajectory_propensity | candidate_zero_post_state | 3 | 675 | 0.4693 | 0.4966 | 0.5778 | 0.0000 |
| best_of_n | E1_trajectory_propensity | pre_state | 3 | 675 | 0.5421 | 0.4934 | 0.5759 | 0.0000 |
| best_of_n | E3_step_cumulative_lambda_0.75 | candidate_zero_post_state | 3 | 670 | 0.4925 | 0.5030 | 0.5880 | 0.0000 |
| best_of_n | E3_step_cumulative_lambda_0.75 | pre_state | 3 | 670 | 0.5518 | 0.4997 | 0.5861 | 0.0000 |
| best_of_n | E4_extended_candidate_set | candidate_zero_post_state | 3 | 670 | 0.4945 | 0.4979 | 0.5790 | 0.0000 |
| best_of_n | E4_extended_candidate_set | pre_state | 3 | 670 | 0.5591 | 0.4947 | 0.5772 | 0.0000 |
| best_of_n | shared | candidate_zero_post_state | 3 | 673 | 0.5284 | 0.6451 | 0.6977 | 0.0000 |
| best_of_n | shared | pre_state | 3 | 673 | 0.5982 | 0.6439 | 0.6973 | 0.0000 |
| burnin | shared | candidate_zero_post_state | 3 | 601 | 0.5000 | 0.8819 | 0.8819 | 0.0000 |
| burnin | shared | pre_state | 3 | 601 | 0.5000 | 0.8819 | 0.8819 | 0.0000 |
| entropy_value | E0_standard_actor_ratio | candidate_zero_post_state | 3 | 663 | 0.4845 | 0.5121 | 0.6056 | 0.0000 |
| entropy_value | E0_standard_actor_ratio | pre_state | 3 | 663 | 0.5503 | 0.5085 | 0.6034 | 0.0000 |
| entropy_value | E1_trajectory_propensity | candidate_zero_post_state | 3 | 663 | 0.5203 | 0.4900 | 0.5681 | 0.0000 |
| entropy_value | E1_trajectory_propensity | pre_state | 3 | 663 | 0.5882 | 0.4867 | 0.5660 | 0.0000 |
| entropy_value | E3_step_cumulative_lambda_0.75 | candidate_zero_post_state | 3 | 662 | 0.4717 | 0.5300 | 0.6374 | 0.0000 |
| entropy_value | E3_step_cumulative_lambda_0.75 | pre_state | 3 | 662 | 0.5560 | 0.5263 | 0.6354 | 0.0000 |
| entropy_value | E4_extended_candidate_set | candidate_zero_post_state | 3 | 659 | 0.4946 | 0.5040 | 0.5865 | 0.0000 |
| entropy_value | E4_extended_candidate_set | pre_state | 3 | 659 | 0.5822 | 0.5002 | 0.5846 | 0.0000 |
| entropy_value | shared | candidate_zero_post_state | 3 | 670 | 0.5170 | 0.6658 | 0.7237 | 0.0000 |
| entropy_value | shared | pre_state | 3 | 670 | 0.5752 | 0.6645 | 0.7231 | 0.0000 |
| k1 | E0_standard_actor_ratio | candidate_zero_post_state | 6 | 1319 | 0.5008 | 0.5830 | 0.6503 | 0.0000 |
| k1 | E0_standard_actor_ratio | pre_state | 6 | 1319 | 0.5758 | 0.5808 | 0.6492 | 0.0000 |
| value_value | E0_standard_actor_ratio | candidate_zero_post_state | 3 | 658 | 0.4946 | 0.5150 | 0.6052 | 0.0000 |
| value_value | E0_standard_actor_ratio | pre_state | 3 | 658 | 0.5458 | 0.5118 | 0.6031 | 0.0000 |
| value_value | E1_trajectory_propensity | candidate_zero_post_state | 3 | 667 | 0.4930 | 0.5106 | 0.5985 | 0.0000 |
| value_value | E1_trajectory_propensity | pre_state | 3 | 667 | 0.5573 | 0.5071 | 0.5964 | 0.0000 |
| value_value | E3_step_cumulative_lambda_0.75 | candidate_zero_post_state | 3 | 664 | 0.4850 | 0.5084 | 0.5919 | 0.0000 |
| value_value | E3_step_cumulative_lambda_0.75 | pre_state | 3 | 664 | 0.5601 | 0.5048 | 0.5898 | 0.0000 |
| value_value | E4_extended_candidate_set | candidate_zero_post_state | 3 | 658 | 0.4394 | 0.5055 | 0.5871 | 0.0000 |
| value_value | E4_extended_candidate_set | pre_state | 3 | 658 | 0.5006 | 0.5020 | 0.5850 | 0.0000 |
| value_value | shared | candidate_zero_post_state | 3 | 664 | 0.5297 | 0.6694 | 0.7283 | 0.0000 |
| value_value | shared | pre_state | 3 | 664 | 0.5841 | 0.6679 | 0.7275 | 0.0000 |

### S3 paired confirmation contrasts

| Contrast | Family | Difference | Clustered 95% CI | W/L/T | Holm p |
|---|---|---:|---:|---:|---:|
| raw_E3_minus_E0 | best_of_n | -0.33 pp | [-1.67, +1.00] | 9/11/580 | 1 |
| raw_E3_minus_E0 | entropy_value | +0.33 pp | [-1.00, +1.67] | 9/7/584 | 1 |
| raw_E3_minus_E0 | value_value | -0.33 pp | [-1.33, +0.67] | 10/12/578 | 1 |
| raw_E1_minus_E0 | best_of_n | -0.50 pp | [-2.00, +1.00] | 8/11/581 | 1 |
| raw_E1_minus_E0 | entropy_value | -0.83 pp | [-2.50, +0.83] | 9/14/577 | 1 |
| raw_E1_minus_E0 | value_value | +0.00 pp | [-1.33, +1.33] | 13/13/574 | 1 |
| composite_E4_minus_E0 | best_of_n | +0.17 pp | [-1.67, +2.00] | 14/13/573 | 0.8533 |
| composite_E4_minus_E0 | entropy_value | -0.83 pp | [-2.50, +0.83] | 11/16/573 | 0.8291 |
| composite_E4_minus_E0 | value_value | +1.00 pp | [-0.33, +2.50] | 13/7/580 | 0.6116 |
| raw_best_of_n_E3_minus_k1 | best_of_n | -1.00 pp | [-3.00, +0.83] | 10/16/574 | 0.3887 |

#### Per-seed accuracy/finalization decomposition

The registered overall-accuracy difference remains primary. The answer-rate and conditional-accuracy differences diagnose whether the observed change is associated with completion/finalization, correctness among valid finals, or both.

| Contrast | Family | Seed | Accuracy difference | Answer-rate difference | Conditional-accuracy difference |
|---|---|---:|---:|---:|---:|
| raw_E3_minus_E0 | best_of_n | 99801 | -1.00 pp | -2.00 pp | +1.75 pp |
| raw_E3_minus_E0 | best_of_n | 99802 | +0.00 pp | +1.00 pp | -2.24 pp |
| raw_E3_minus_E0 | best_of_n | 99803 | +0.00 pp | -0.50 pp | +1.18 pp |
| raw_E3_minus_E0 | entropy_value | 99801 | +1.50 pp | +1.00 pp | +1.71 pp |
| raw_E3_minus_E0 | entropy_value | 99802 | +0.50 pp | +0.50 pp | +0.16 pp |
| raw_E3_minus_E0 | entropy_value | 99803 | -1.00 pp | -0.50 pp | -1.47 pp |
| raw_E3_minus_E0 | value_value | 99801 | -1.50 pp | -1.50 pp | -0.51 pp |
| raw_E3_minus_E0 | value_value | 99802 | -1.00 pp | -1.00 pp | -0.34 pp |
| raw_E3_minus_E0 | value_value | 99803 | +1.50 pp | +1.00 pp | +1.66 pp |
| raw_E1_minus_E0 | best_of_n | 99801 | -1.00 pp | -1.50 pp | +0.66 pp |
| raw_E1_minus_E0 | best_of_n | 99802 | -2.00 pp | -1.50 pp | -1.83 pp |
| raw_E1_minus_E0 | best_of_n | 99803 | +1.50 pp | +1.50 pp | +0.53 pp |
| raw_E1_minus_E0 | entropy_value | 99801 | +0.00 pp | +0.00 pp | +0.00 pp |
| raw_E1_minus_E0 | entropy_value | 99802 | -2.00 pp | -1.50 pp | -1.85 pp |
| raw_E1_minus_E0 | entropy_value | 99803 | -0.50 pp | -1.00 pp | +0.98 pp |
| raw_E1_minus_E0 | value_value | 99801 | -1.00 pp | -1.00 pp | -0.34 pp |
| raw_E1_minus_E0 | value_value | 99802 | -0.50 pp | +0.00 pp | -1.28 pp |
| raw_E1_minus_E0 | value_value | 99803 | +1.50 pp | +0.50 pp | +2.76 pp |
| composite_E4_minus_E0 | best_of_n | 99801 | +1.50 pp | +0.50 pp | +2.56 pp |
| composite_E4_minus_E0 | best_of_n | 99802 | -1.00 pp | -1.50 pp | +0.57 pp |
| composite_E4_minus_E0 | best_of_n | 99803 | +0.00 pp | +0.50 pp | -1.01 pp |
| composite_E4_minus_E0 | entropy_value | 99801 | -2.00 pp | -2.50 pp | +0.26 pp |
| composite_E4_minus_E0 | entropy_value | 99802 | +0.50 pp | +1.00 pp | -0.76 pp |
| composite_E4_minus_E0 | entropy_value | 99803 | -1.00 pp | -2.50 pp | +2.52 pp |
| composite_E4_minus_E0 | value_value | 99801 | +0.50 pp | -0.50 pp | +2.28 pp |
| composite_E4_minus_E0 | value_value | 99802 | +2.00 pp | +0.50 pp | +3.79 pp |
| composite_E4_minus_E0 | value_value | 99803 | +0.50 pp | +1.00 pp | -0.81 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99801 | +1.00 pp | +0.50 pp | +1.48 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99802 | +0.00 pp | +0.00 pp | +0.00 pp |
| raw_best_of_n_E3_minus_k1 | best_of_n | 99803 | -4.00 pp | -5.00 pp | +0.92 pp |

### S3 matched-compute evidence

| Family | Estimator | States | Updates | Candidate forwards | Candidate tokens | Selected tokens | Token ratio vs E0 | Time ratio vs E0 | Min ESS | Max clip | Max KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| best_of_n | E0_standard_actor_ratio | 1348 | 46 | 2696 | 119373 | 59564 | 1.0000 | 1.0000 | 1.0000 | 0.2743 | 0.0025 |
| best_of_n | E1_trajectory_propensity | 1348 | 45 | 2696 | 118777 | 59236 | 0.9950 | 0.9979 | 0.9677 | 0.2743 | 0.0019 |
| best_of_n | E3_step_cumulative_lambda_0.75 | 1343 | 45 | 2686 | 118720 | 59203 | 0.9945 | 0.9855 | 0.9815 | 0.2832 | 0.0020 |
| best_of_n | E4_extended_candidate_set | 1343 | 45 | 2686 | 117950 | 58871 | 0.9881 | 0.9996 | 1.0000 | 0.3677 | 0.0036 |
| entropy_value | E0_standard_actor_ratio | 1333 | 45 | 2815 | 139124 | 57743 | 1.0000 | 1.0000 | 1.0000 | 0.2896 | 0.0025 |
| entropy_value | E1_trajectory_propensity | 1333 | 45 | 2815 | 140305 | 58229 | 1.0085 | 0.9917 | 0.9484 | 0.3099 | 0.0032 |
| entropy_value | E3_step_cumulative_lambda_0.75 | 1332 | 45 | 2820 | 139972 | 58053 | 1.0061 | 0.9732 | 0.9732 | 0.2889 | 0.0019 |
| entropy_value | E4_extended_candidate_set | 1329 | 45 | 2808 | 139253 | 57903 | 1.0009 | 0.9544 | 1.0000 | 0.3689 | 0.0022 |
| value_value | E0_standard_actor_ratio | 1322 | 43 | 2738 | 101373 | 58004 | 1.0000 | 1.0000 | 1.0000 | 0.2727 | 0.0021 |
| value_value | E1_trajectory_propensity | 1331 | 44 | 2771 | 102205 | 58040 | 1.0082 | 0.9980 | 0.9528 | 0.3584 | 0.0027 |
| value_value | E3_step_cumulative_lambda_0.75 | 1328 | 44 | 2759 | 102462 | 58184 | 1.0107 | 0.9989 | 0.9953 | 0.3363 | 0.0028 |
| value_value | E4_extended_candidate_set | 1322 | 43 | 2726 | 100801 | 57795 | 0.9944 | 0.9896 | 1.0000 | 0.3394 | 0.0020 |

### S3 retained-composite diversity

Answer rate denotes strict scalar `FINAL:` extraction rather than explicit abstention. Conditional accuracy is accuracy among those valid finals; unanswered failures are split by retained-text evidence.

| Seed | Family | Estimator | Accuracy | Answer rate | Conditional accuracy | No final-like | Noncanonical final-like | Target expansion | Source expansion | Evaluation expansion | Eval-source expansion shift | Target candidates/state | Source candidates/state | Evaluation candidates/state | Eval-source candidate shift | Candidates | Candidate tokens | Selected tokens | Token multiplier | Mean unique | Identical | Exact-token pairs | Prefix divergence |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99801 | best_of_n | E0_standard_actor_ratio | 0.3400 | 0.4250 | 0.8000 | 112 | 3 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2726 | 155563 | 77434 | 2.0090 | 1.5664 | 0.4336 | 0.2913 | 0.5656 |
| 99801 | best_of_n | E1_trajectory_propensity | 0.3600 | 0.4250 | 0.8471 | 114 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2750 | 154240 | 76903 | 2.0056 | 1.5636 | 0.4364 | 0.2836 | 0.5643 |
| 99801 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.3550 | 0.4150 | 0.8554 | 115 | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2718 | 154557 | 76849 | 2.0112 | 1.5467 | 0.4533 | 0.2958 | 0.5554 |
| 99801 | best_of_n | E4_extended_candidate_set | 0.3550 | 0.4300 | 0.8256 | 112 | 2 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2724 | 154355 | 77161 | 2.0004 | 1.5485 | 0.4515 | 0.2907 | 0.5590 |
| 99801 | entropy_value | E0_standard_actor_ratio | 0.3800 | 0.4500 | 0.8444 | 110 | 0 | 0.3333 | 0.3721 | 0.5077 | 0.1356 | 2.0000 | 2.1163 | 2.5230 | 0.4067 | 3459 | 208107 | 76946 | 2.7046 | 2.9684 | 0.1307 | 0.1432 | 0.7174 |
| 99801 | entropy_value | E1_trajectory_propensity | 0.3600 | 0.4250 | 0.8471 | 114 | 1 | 0.3333 | 0.3474 | 0.5165 | 0.1690 | 2.0000 | 2.0423 | 2.5494 | 0.5071 | 3485 | 210440 | 77065 | 2.7307 | 2.9093 | 0.1473 | 0.1416 | 0.7124 |
| 99801 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.3700 | 0.4400 | 0.8409 | 112 | 0 | 0.3333 | 0.3349 | 0.5037 | 0.1688 | 2.0000 | 2.0047 | 2.5110 | 0.5063 | 3435 | 205410 | 77029 | 2.6667 | 2.9086 | 0.1509 | 0.1454 | 0.7126 |
| 99801 | entropy_value | E4_extended_candidate_set | 0.3600 | 0.4250 | 0.8471 | 115 | 0 | 0.3333 | 0.3411 | 0.5055 | 0.1643 | 2.0000 | 2.0234 | 2.5164 | 0.4930 | 3450 | 208272 | 78436 | 2.6553 | 2.9509 | 0.1270 | 0.1383 | 0.7214 |
| 99801 | value_value | E0_standard_actor_ratio | 0.3350 | 0.4050 | 0.8272 | 118 | 1 | 0.3333 | 0.3645 | 0.0000 | -0.3645 | 2.0000 | 2.0935 | 1.0000 | -1.0935 | 1375 | 76677 | 76677 | 1.0000 | NA | NA | NA | NA |
| 99801 | value_value | E1_trajectory_propensity | 0.3500 | 0.4050 | 0.8642 | 118 | 1 | 0.3333 | 0.3657 | 0.0014 | -0.3643 | 2.0000 | 2.0972 | 1.0043 | -1.0929 | 1390 | 77654 | 77239 | 1.0054 | 4.0000 | 0.0000 | 0.0000 | 0.9510 |
| 99801 | value_value | E3_step_cumulative_lambda_0.75 | 0.3350 | 0.4100 | 0.8171 | 118 | 0 | 0.3333 | 0.3529 | 0.0014 | -0.3515 | 2.0000 | 2.0588 | 1.0043 | -1.0545 | 1392 | 79544 | 79127 | 1.0053 | 4.0000 | 0.0000 | 0.0000 | 0.9808 |
| 99801 | value_value | E4_extended_candidate_set | 0.3400 | 0.4000 | 0.8500 | 119 | 1 | 0.3333 | 0.3410 | 0.0014 | -0.3396 | 2.0000 | 2.0230 | 1.0043 | -1.0187 | 1397 | 78507 | 78090 | 1.0053 | 4.0000 | 0.0000 | 0.0000 | 0.9523 |
| 99802 | best_of_n | E0_standard_actor_ratio | 0.3700 | 0.4450 | 0.8315 | 110 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2730 | 152379 | 75886 | 2.0080 | 1.5407 | 0.4593 | 0.3026 | 0.5507 |
| 99802 | best_of_n | E1_trajectory_propensity | 0.3450 | 0.4200 | 0.8214 | 116 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2750 | 152996 | 76412 | 2.0023 | 1.5469 | 0.4531 | 0.2996 | 0.5566 |
| 99802 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.3200 | 0.3850 | 0.8312 | 122 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2764 | 155113 | 77542 | 2.0004 | 1.5456 | 0.4544 | 0.2981 | 0.5570 |
| 99802 | best_of_n | E4_extended_candidate_set | 0.3600 | 0.4300 | 0.8372 | 114 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2740 | 150365 | 75133 | 2.0013 | 1.5401 | 0.4599 | 0.3102 | 0.5506 |
| 99802 | entropy_value | E0_standard_actor_ratio | 0.3650 | 0.4350 | 0.8391 | 113 | 0 | 0.3333 | 0.3665 | 0.4793 | 0.1128 | 2.0000 | 2.0995 | 2.4379 | 0.3384 | 3357 | 202574 | 78038 | 2.5958 | 2.9970 | 0.1242 | 0.1293 | 0.7249 |
| 99802 | entropy_value | E1_trajectory_propensity | 0.3650 | 0.4400 | 0.8295 | 111 | 1 | 0.3333 | 0.3677 | 0.4895 | 0.1217 | 2.0000 | 2.1031 | 2.4684 | 0.3652 | 3394 | 207031 | 78975 | 2.6215 | 2.9643 | 0.1322 | 0.1345 | 0.7302 |
| 99802 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.3650 | 0.4300 | 0.8488 | 114 | 0 | 0.3333 | 0.3739 | 0.4785 | 0.1046 | 2.0000 | 2.1216 | 2.4354 | 0.3138 | 3339 | 204369 | 78971 | 2.5879 | 2.9695 | 0.1479 | 0.1260 | 0.7269 |
| 99802 | entropy_value | E4_extended_candidate_set | 0.3700 | 0.4450 | 0.8315 | 111 | 0 | 0.3333 | 0.3750 | 0.4840 | 0.1090 | 2.0000 | 2.1250 | 2.4520 | 0.3270 | 3369 | 201077 | 77065 | 2.6092 | 2.9579 | 0.1353 | 0.1386 | 0.7183 |
| 99802 | value_value | E0_standard_actor_ratio | 0.3400 | 0.4150 | 0.8193 | 117 | 0 | 0.3333 | 0.3409 | 0.0000 | -0.3409 | 2.0000 | 2.0227 | 1.0000 | -1.0227 | 1384 | 77446 | 77446 | 1.0000 | NA | NA | NA | NA |
| 99802 | value_value | E1_trajectory_propensity | 0.3500 | 0.4050 | 0.8642 | 118 | 1 | 0.3333 | 0.3559 | 0.0000 | -0.3559 | 2.0000 | 2.0676 | 1.0000 | -1.0676 | 1391 | 79187 | 79187 | 1.0000 | NA | NA | NA | NA |
| 99802 | value_value | E3_step_cumulative_lambda_0.75 | 0.3350 | 0.3800 | 0.8816 | 123 | 1 | 0.3333 | 0.3825 | 0.0000 | -0.3825 | 2.0000 | 2.1475 | 1.0000 | -1.1475 | 1392 | 79395 | 79395 | 1.0000 | NA | NA | NA | NA |
| 99802 | value_value | E4_extended_candidate_set | 0.3600 | 0.4200 | 0.8571 | 116 | 0 | 0.3333 | 0.3532 | 0.0007 | -0.3525 | 2.0000 | 2.0596 | 1.0022 | -1.0575 | 1388 | 78691 | 78403 | 1.0037 | 4.0000 | 0.0000 | 0.0000 | 0.9948 |
| 99803 | best_of_n | E0_standard_actor_ratio | 0.3350 | 0.4050 | 0.8272 | 119 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2754 | 154703 | 77204 | 2.0038 | 1.5454 | 0.4546 | 0.2970 | 0.5546 |
| 99803 | best_of_n | E1_trajectory_propensity | 0.3450 | 0.4300 | 0.8023 | 114 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2760 | 154504 | 76957 | 2.0077 | 1.5413 | 0.4587 | 0.2884 | 0.5556 |
| 99803 | best_of_n | E3_step_cumulative_lambda_0.75 | 0.3250 | 0.4050 | 0.8025 | 119 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2762 | 155533 | 77638 | 2.0033 | 1.5619 | 0.4381 | 0.2933 | 0.5678 |
| 99803 | best_of_n | E4_extended_candidate_set | 0.3350 | 0.4100 | 0.8171 | 117 | 1 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | 0.0000 | 2748 | 152613 | 76237 | 2.0018 | 1.5531 | 0.4469 | 0.2911 | 0.5544 |
| 99803 | entropy_value | E0_standard_actor_ratio | 0.3600 | 0.4400 | 0.8182 | 112 | 0 | 0.3333 | 0.2952 | 0.4742 | 0.1790 | 2.0000 | 1.8855 | 2.4225 | 0.5371 | 3331 | 199799 | 75844 | 2.6343 | 2.9601 | 0.1319 | 0.1299 | 0.7270 |
| 99803 | entropy_value | E1_trajectory_propensity | 0.3500 | 0.4250 | 0.8235 | 115 | 0 | 0.3333 | 0.3172 | 0.4909 | 0.1737 | 2.0000 | 1.9515 | 2.4726 | 0.5211 | 3390 | 207690 | 78595 | 2.6425 | 2.9643 | 0.1308 | 0.1340 | 0.7238 |
| 99803 | entropy_value | E3_step_cumulative_lambda_0.75 | 0.3550 | 0.4300 | 0.8256 | 114 | 0 | 0.3333 | 0.3333 | 0.4671 | 0.1338 | 2.0000 | 2.0000 | 2.4013 | 0.4013 | 3321 | 201030 | 77762 | 2.5852 | 2.9706 | 0.1254 | 0.1318 | 0.7239 |
| 99803 | entropy_value | E4_extended_candidate_set | 0.3500 | 0.4150 | 0.8434 | 117 | 0 | 0.3333 | 0.3188 | 0.4819 | 0.1631 | 2.0000 | 1.9563 | 2.4456 | 0.4892 | 3370 | 205950 | 78681 | 2.6175 | 2.9533 | 0.1220 | 0.1318 | 0.7245 |
| 99803 | value_value | E0_standard_actor_ratio | 0.3450 | 0.4100 | 0.8415 | 118 | 0 | 0.3333 | 0.3750 | 0.0007 | -0.3743 | 2.0000 | 2.1250 | 1.0022 | -1.1228 | 1391 | 79344 | 79056 | 1.0036 | 4.0000 | 0.0000 | 0.0000 | 0.9236 |
| 99803 | value_value | E1_trajectory_propensity | 0.3300 | 0.3850 | 0.8571 | 122 | 1 | 0.3333 | 0.3799 | 0.0014 | -0.3785 | 2.0000 | 2.1397 | 1.0043 | -1.1354 | 1404 | 79403 | 78849 | 1.0070 | 4.0000 | 0.0000 | 0.0000 | 0.9731 |
| 99803 | value_value | E3_step_cumulative_lambda_0.75 | 0.3400 | 0.3900 | 0.8718 | 120 | 2 | 0.3333 | 0.3584 | 0.0007 | -0.3577 | 2.0000 | 2.0752 | 1.0022 | -1.0731 | 1388 | 78195 | 77907 | 1.0037 | 4.0000 | 0.0000 | 0.0000 | 0.9983 |
| 99803 | value_value | E4_extended_candidate_set | 0.3500 | 0.4200 | 0.8333 | 116 | 0 | 0.3333 | 0.3677 | 0.0014 | -0.3663 | 2.0000 | 2.1031 | 1.0043 | -1.0988 | 1393 | 78167 | 77591 | 1.0074 | 4.0000 | 0.0000 | 0.0000 | 0.8924 |

S3 Value aggregates are state-count-weighted descriptive summaries of per-rollout metrics, not pooled calibration estimates or independent-sample inference.

## Table 3 — frozen gates and reproducibility

```json
{
  "archive_sha256": {
    "administrative_archive_audit_source": "be637e6de32c2689c80b38a2fe2d7fe0e83f9710fd1ffe6ad79ecec131b425c9",
    "evaluation_data": "471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7",
    "evaluation_manifest": "ec96c0b63eb0dadb6132e12961c2efee85a11b9df1e92ca7a29a9933d560008d",
    "frozen_analysis_json": "c97074306680e740862827f526fc23bb0ef5fc58938a935e12861d2c2eb3905d",
    "frozen_analysis_source": "dc30d59b790bae341d97786817c90463bbb6a7dc77315ec424392604b26f2c27",
    "frozen_protocol": "be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978",
    "frozen_report": "b0a24c4405ff3db3ef8b9c2ed191e5841536166dc14d942cae9890f84ec89001",
    "frozen_runner": "466dd3608434f2244abd8861d55a554fee6d4c34aa2d17319423187acaf966a1",
    "iclr_evidence_table_renderer_source": "631738c8a92e61265fe805bcac4312379ce1f416e298a4ae132a9cfcd39f6c55",
    "iclr_figure_renderer_source": "de0c8fc0f94e8417e0159ce52190407012a969634f76404a7c37d9101bfe5736",
    "iclr_manuscript_review_recorder_source": "22b398c38ca61de4983e833b2528ba63a2530b7be823d070432f409d6479313f",
    "iclr_terminal_results_renderer_source": "f9df8c39c2cedf668962640159c25125a8fa409ac543023979a80347b6124d97",
    "manual_compute_review_preparer_source": "fe609534b44916a902d09df7cd453612fa1bd19532691bb91381483718388bd9",
    "manual_compute_review_receipt": "999f205e1b2e2e933d9efb809da8a419b9427c1921c72efaf87d4d075a1e0dd4",
    "manual_compute_review_recorder_source": "39ca61901f2e71e61c6a3943724324f10cab4251c98d10b2b30fa41aa7665221",
    "remote_postrun_test_junit": "795ed366b86eece4ac3f59cbabbdc6f22cb30f3483afb4170f0497a2a0cafe75",
    "remote_postrun_test_log": "341d5dca33250b815baddbafb6001e174c20bf1716bb74f3d612e77777db8938",
    "remote_postrun_test_source": "7c03595abbaf13b33e389953a6108eb637004d4c96ad3735e51c0de4fd09cf75",
    "s3_final_archive_manifest_builder_source": "1c00bb9b6227a0d26dbccc411b0a2c6665d8c0bcf6d9dd7f97305aecad910826"
  },
  "artifact_counts": {
    "all_training": 81,
    "burnin_training": 3,
    "composite_evaluations": 36,
    "k1_adaptive_training": 6,
    "raw_k1_evaluations": 39,
    "search_adaptive_training": 72
  },
  "frozen_gate_logic_audit": {
    "answer_rate_gate": false,
    "frozen_analyzer_expression": "all(completeness_safety.values())",
    "frozen_analyzer_overall_gate": false,
    "frozen_analyzer_stability_gate": false,
    "independent_corrected_overall_gate": false,
    "independent_corrected_stability_gate": false,
    "interpretation": "The frozen analyzer treats integer zero support violations as false inside all(completeness_safety.values()). The independent audit recomputes the intended condition as support_violations == 0 without modifying the frozen analyzer.",
    "overall_verdict_invariant_to_defect": true,
    "support_violations": 0,
    "zero_support_truthiness_defect_present": true
  },
  "gpu_audit": {
    "all_status_zero": true,
    "expected_gpu_uuid": "GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48",
    "gpu_reserved_hours": 108.58583333333333,
    "gpu_usage_files": [
      "gpu_usage_1793225144_554765.csv"
    ],
    "matches_frozen_analyzer": true,
    "mean_sampled_utilization_percent": 30.42474886429074,
    "multiple_run_sessions": false,
    "peak_memory_mib": 17580.0,
    "sessions": [
      {
        "end": 1793616053,
        "pid": 554765,
        "start": 1793225144,
        "status": 0
      }
    ],
    "successful_session_count": 1,
    "total_reserved_seconds": 390909,
    "usage_sample_count": 78145
  },
  "manual_compute_gate": {
    "analysis_sha256": "c97074306680e740862827f526fc23bb0ef5fc58938a935e12861d2c2eb3905d",
    "compute_evidence_sha256": "e3c85546b0f19324cffbc10d5c469435eedd43c2e370e46636b0def22213959a",
    "preparer_sha256": "fe609534b44916a902d09df7cd453612fa1bd19532691bb91381483718388bd9",
    "rationale": "Within each search family, the four estimators are compute-matched: candidate-token ratios to E0 span 0.9881\u20131.0107, candidate-forward ratios span 0.9956\u20131.0121, and summed invocation-time ratios span 0.9544\u20130.9996. The independent audit also reconstructs one successful GPU1 session totaling 108.586 reserved hours. This PASS supports only the fairness and credibility of within-family estimator comparisons; it does not override the failed stability or positive-transfer gates, does not equate the deliberately different search-family budgets, and does not establish a winning estimator.",
    "receipt_sha256": "999f205e1b2e2e933d9efb809da8a419b9427c1921c72efaf87d4d075a1e0dd4",
    "recorder_sha256": "39ca61901f2e71e61c6a3943724324f10cab4251c98d10b2b30fa41aa7665221",
    "report_sha256": "b0a24c4405ff3db3ef8b9c2ed191e5841536166dc14d942cae9890f84ec89001",
    "required": true,
    "verdict": "PASS"
  },
  "remote_postrun_tests": {
    "errors": 0,
    "failures": 0,
    "junit_sha256": "795ed366b86eece4ac3f59cbabbdc6f22cb30f3483afb4170f0497a2a0cafe75",
    "log_sha256": "341d5dca33250b815baddbafb6001e174c20bf1716bb74f3d612e77777db8938",
    "skipped": 0,
    "tests": 18
  },
  "stability_gate_audit": {
    "all_clip_fractions_below_0_20": false,
    "all_mean_kl_proxies_below_0_02": true,
    "clip_threshold_strictly_below": 0.2,
    "clip_violation_count": 72,
    "e3_ess_exceeds_e1_in_all_18_comparisons": true,
    "e3_vs_e1_ess_comparisons": [
      {
        "e1_ess": 0.9906787926164471,
        "e3_ess": 0.9946839544728396,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 1,
        "seed": 99801
      },
      {
        "e1_ess": 0.9980031062704894,
        "e3_ess": 0.9988816169052032,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 1,
        "seed": 99801
      },
      {
        "e1_ess": 0.9973935456228655,
        "e3_ess": 0.9985207620053469,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 1,
        "seed": 99801
      },
      {
        "e1_ess": 0.9676508422850353,
        "e3_ess": 0.9814641055110048,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 2,
        "seed": 99801
      },
      {
        "e1_ess": 0.9841918053230879,
        "e3_ess": 0.9909785589238905,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 2,
        "seed": 99801
      },
      {
        "e1_ess": 0.9973927646737336,
        "e3_ess": 0.9979227653331513,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 2,
        "seed": 99801
      },
      {
        "e1_ess": 0.9973154287772437,
        "e3_ess": 0.9984843032059623,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 1,
        "seed": 99802
      },
      {
        "e1_ess": 0.9976837078412258,
        "e3_ess": 0.9987002246323814,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 1,
        "seed": 99802
      },
      {
        "e1_ess": 0.9984938676327911,
        "e3_ess": 0.9991556360971122,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 1,
        "seed": 99802
      },
      {
        "e1_ess": 0.9883836498976003,
        "e3_ess": 0.9937709564728625,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 2,
        "seed": 99802
      },
      {
        "e1_ess": 0.9873505461138045,
        "e3_ess": 0.9919080456650263,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 2,
        "seed": 99802
      },
      {
        "e1_ess": 0.9914634484991155,
        "e3_ess": 0.9952786719799361,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 2,
        "seed": 99802
      },
      {
        "e1_ess": 0.9958432569090312,
        "e3_ess": 0.9976354475892042,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 1,
        "seed": 99803
      },
      {
        "e1_ess": 0.9976831102779222,
        "e3_ess": 0.9986930521087098,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 1,
        "seed": 99803
      },
      {
        "e1_ess": 0.999404121024703,
        "e3_ess": 0.9996652743855327,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 1,
        "seed": 99803
      },
      {
        "e1_ess": 0.9903981992449662,
        "e3_ess": 0.99497693374153,
        "e3_exceeds_e1": true,
        "family": "best_of_n",
        "round": 2,
        "seed": 99803
      },
      {
        "e1_ess": 0.9484161849628512,
        "e3_ess": 0.9732099797686333,
        "e3_exceeds_e1": true,
        "family": "entropy_value",
        "round": 2,
        "seed": 99803
      },
      {
        "e1_ess": 0.9527854468903613,
        "e3_ess": 0.9973628655543286,
        "e3_exceeds_e1": true,
        "family": "value_value",
        "round": 2,
        "seed": 99803
      }
    ],
    "kl_threshold_strictly_below": 0.02,
    "kl_violation_count": 0,
    "maximum_clip_fraction": 0.3688888888888889,
    "maximum_kl_proxy": 0.0035974720328008775
  }
}
```


## Deterministic provenance

| Input | SHA-256 |
|---|---|
| renderer | `631738c8a92e61265fe805bcac4312379ce1f416e298a4ae132a9cfcd39f6c55` |
| critic_confirmation | `87e1f3c3062a44de8d9651be1f305a5aea64305ef53642f9457175da40daa093` |
| exact_analysis | `c82d8513c1b561b9c5fe57f3532844a501ea144c87ca023c478690bd1b3311a5` |
| exact_confirmation | `6e9655cebe49c8b76dd4dd309aed41d70fb71a58c99343b71af5f3f7e02028b0` |
| gate_confirmation | `1018ef234ea1b3b2010cb60b1046f4b2ee4593ad1b0af7eebfadf90e76b578c7` |
| mechanism_analysis | `1dc5efe692c14a43ff51a86a9cb69baef4744839d77a90f22949c558f3dff994` |
| s3_analysis | `c97074306680e740862827f526fc23bb0ef5fc58938a935e12861d2c2eb3905d` |
| s3_audit | `053eb5bad18b43a230600557a572147e6f892af2b4fc8cd0af9afe38f7fc496f` |
