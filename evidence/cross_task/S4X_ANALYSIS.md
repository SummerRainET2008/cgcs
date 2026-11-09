# S4-X ARC-Challenge Analysis

Validated 24/24 cells and 28,128 evaluations.

## Registered accuracy contrasts

| Horizon | Baseline | E4 difference | 95% CI | Holm p |
|---|---|---:|---:|---:|
| H8 | best_of_n_E0_standard_actor_ratio | 0.0020 | [-0.0023, 0.0063] | 1 |
| H8 | best_of_n_E3_step_cumulative_lambda_0.75 | 0.0014 | [-0.0028, 0.0057] | 1 |
| H8 | k1_E0_standard_actor_ratio | 0.0028 | [-0.0014, 0.0074] | 1 |
| H16 | best_of_n_E0_standard_actor_ratio | 0.0014 | [-0.0026, 0.0057] | 1 |
| H16 | best_of_n_E3_step_cumulative_lambda_0.75 | 0.0020 | [-0.0017, 0.0060] | 1 |
| H16 | k1_E0_standard_actor_ratio | 0.0017 | [-0.0026, 0.0060] | 1 |
| H8+H16 | best_of_n_E0_standard_actor_ratio | 0.0017 | [-0.0023, 0.0057] | 1 |
| H8+H16 | best_of_n_E3_step_cumulative_lambda_0.75 | 0.0017 | [-0.0021, 0.0055] | 1 |
| H8+H16 | k1_E0_standard_actor_ratio | 0.0023 | [-0.0020, 0.0065] | 1 |

Answer-rate, conditional-accuracy, per-seed, transition, token, and termination audits are retained in the JSON artifact.
