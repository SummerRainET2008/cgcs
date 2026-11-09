# Exact-truth study pre-freeze checklist

This checklist is fail-closed. The confirmation seed generator and confirmation
runner must not execute until every required item is evidenced.

| Requirement | Current state | Required evidence |
|---|---|---|
| Current three-seed mechanism replication complete | satisfied | root `COMPLETE`; 3/3 burn-ins, 36/36 adaptive summaries, and 18/18 evaluations with ordered IDs 800--999 passed strict analysis |
| Mechanism report archived and reviewed | satisfied | deterministic JSON/Markdown; 256/256 SHA-256 entries verified remotely and locally; 17 focused tests passed; README entry added |
| Development-only evidence isolated | satisfied | seeds 1000--1029 documented and excluded |
| Raw-Actor and composite-search estimands fixed | satisfied | frozen protocol text |
| E0--E4 estimator definitions fixed | satisfied | frozen protocol text and exact reference tests |
| Search families and compute matching fixed | satisfied | frozen protocol text |
| Gate model/feature sufficiency rules fixed | satisfied | frozen protocol plus development notes |
| Candidate budgets and selector floors fixed | satisfied | B={1.5,2.0}, K={2,4}, beta=1, epsilon=0.10 |
| Critic-quality conditions fixed | satisfied | exact, noise {0.5,1,2}, lag KL {0.01,0.04}, learned {256,1024} |
| V-trace and partial-correction grids fixed | satisfied | truncations {(1,1),(2,1)}, lambda grid and tie rule |
| Numerical tolerances and mismatch gate fixed | satisfied | S0 tolerances and S1 quantitative rule |
| Minimal LLM matrix fixed conditionally on S0--S2 | satisfied | four search rows, four estimator columns, ≥3 seeds |
| Environment generator satisfies required pathologies | satisfied | construction tests plus 1,000,000-trajectory smoke summaries for dev seeds 1000--1002 |
| Exact gradients agree with finite differences | satisfied | raw and composite gradient relative L2 error tests at tolerance 1e-5 |
| Conditional, marginal and composite ratios recover named targets | satisfied | exact-enumeration tests plus E1/E2 cumulative smoke recovery within 0.01 and 3SE |
| Candidate-set ratio includes every candidate likelihood | satisfied | complete E4 identity passes; deliberate selected-only removal differs by >1e-3 |
| Confirmation seed generator tested | satisfied | deterministic/exclusion/overwrite tests |
| Protocol copied to repository and marked frozen | satisfied | `results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md`; exact digest recorded in seed manifest |
| Confirmation seed manifest generated once | satisfied | `CONFIRMATION_SEEDS.json`, 30 unique MDPs x 10 shared streams, protocol digest match |
| Confirmation outcomes remained unseen through freeze/commitment | satisfied | first instantiation occurred only after frozen protocol, analysis schema, and seed manifest were archived |

Only after all rows are satisfied may Gate S0 confirmation execution begin.
