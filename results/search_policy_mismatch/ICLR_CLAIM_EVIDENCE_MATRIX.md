# ICLR claim--evidence matrix

Status: living administrative synthesis; **not a protocol amendment**.

This matrix separates claims already supported by frozen exact-truth evidence
from claims that still depend on the running S3 LLM confirmation. It is intended
to prevent intermediate LLM results from leaking into the paper narrative.

## Supported claims

| Candidate claim | Current evidence | Allowed scope |
|---|---|---|
| Search-selected trajectories are not ordinary raw-Actor on-policy data. | In the committed 30-MDP confirmation, E0 relative gradient bias is 36.91% for best-of-2, 24.94% for entropy-gated K4, and 23.41% for low-Value-gated K4. S1's mismatch condition passes on 30/30 MDPs. | Exact synthetic search families under the frozen protocol. |
| Exact correction is not a single finite-sample solution. | E1 trajectory propensity and E2 executed-action marginal correction are both exact in expectation, but E1 has materially higher batch-256 gradient MSE and lower ESS. | Conditional-propensity versus executed-marginal corrections in the exact tree study. |
| Partial correction exposes a bias--variance frontier. | Development-frozen E3 `lambda=0.75` beats both E0 and full E2 in batch-256 gradient MSE for all 27 Critic-by-family conditions; at batch 1,024 the optimum moves toward full correction in the K4 families. | Finite-data conclusion, explicitly batch dependent. |
| Search mismatch can affect the no-search Actor. | At KL-matched update size, exact-tree best-of-2 E3 improves K1 return over E0 by +0.0102 with clustered CI [+0.0061,+0.0145]. The two gated K4 intervals cross zero. | Best-of-2 exact-tree transfer; not yet an LLM claim. |
| Gate allocation quality and selector quality are distinct mechanisms. | Entropy is the only simple deployable gate with positive confirmation intervals in all three non-degenerate compute settings. Exact and inverted Critics have similar TV/ESS but nearly opposite search returns. | Exact-tree mechanism identification. |
| Scalar Value and Actor entropy are insufficient statistics for oracle expansion gain. | The frozen four-feature Value--entropy gate is approximately random, whereas Critic-shape diagnostics reach development-test Pearson 0.9942/0.9867 for K2/K4 and reveal substantial oracle headroom. | Diagnostic sufficiency result; Critic-shape features are not free deployable signals. |
| Critic directional quality determines whether search-induced shift helps. | Across the 30-MDP x 9-Critic confirmation, ranking concordance correlates above 0.92 with search gain but below 0.05 with maximum-depth state--Action TV. | Exact-tree Critic axis. |
| The exact composite objective must account for latent candidate generation. | Executed-action marginal E4 and full candidate-set E4 recover the exact composite gradient; marginalization has lower variance. The selected-Action-only proxy remains labelled biased. | Exact-tree composite-search objective. |

## Claims awaiting completed S3 evaluation

All 81 training summaries are complete.  Their preregistered stability result is
already decisive: 72/72 search branches exceed the sequence-ratio clip-fraction
ceiling of 0.20.  S3 therefore cannot pass its full gate or authorize a larger
PPO/GRPO stage, irrespective of the remaining accuracy results.  This does not
permit outcome-based early stopping: all 39 raw-K1 and 36 retained-composite
evaluations must still finish, followed by the frozen analyzer and independent
archive audit.

The following performance statements remain explicitly unresolved until that
evaluation and archival work is complete:

1. Partial correction improves a no-search Qwen3-4B Actor relative to E0.
2. Full candidate-set E4 improves retained composite-search performance over E0.
3. Any S3 contrast is consistent across all three training seeds.
4. The positive contrast survives paired problem-clustered uncertainty,
   answer-rate, ESS, clip, KL, candidate-token, and GPU-hour checks.

The former candidate claim that S3 passes and authorizes a separately frozen
larger-scale PPO/GRPO comparison is now contradicted by the frozen clip gate. It
must be reported as a failed stage gate, not left pending or reframed after the
evaluation results are observed.

## Claims not licensed by the current program

- No estimator or correction strength is universally optimal.
- Entropy gating is not claimed to be optimal; the oracle gap remains large.
- The intuitive low-Value/high-entropy quadrant is not itself a sufficient gate.
- Search-time composite accuracy is not interchangeable with raw K1 Actor
  accuracy.
- S3 does not measure per-update gradient norms; exact gradients, autograd
  coverage tests, and runtime stability diagnostics are separate evidence types.
- S3 answer rate is the fraction of examples yielding a scalar answer under the
  frozen strict `FINAL:` extractor and generation budget.  It is not a learned
  abstention or refusal probability.  Accuracy conditional on a valid final and
  transcript-level finalization failures must be reported separately.
- S3 alone does not license a claim that estimator rankings are invariant to
  the eight-step limit, the 96-token-per-step limit, the finalization prompt, or
  the strict extractor.  A separately frozen post-S3 sensitivity analysis may
  diagnose that robustness, but it cannot replace the registered S3 estimand or
  repair a failed S3 promotion gate.
- Exact-text uniqueness and pairwise token-prefix divergence in S3 measure
  surface-form branching only. Candidate-level parsed answers and
  counterfactual terminal rewards were not logged, so S3 cannot convert those
  diagnostics into a claim of semantic or mathematical-strategy diversity.
- Results on code execution or delayed-reward tool use remain future work until
  their datasets and protocols are separately frozen.

## Related-work boundary conditions

- BPO proves an unbiased sibling-advantage construction when branching from an
  on-policy prefix and retaining the sibling samples.  That boundary case is
  not a proof that winner-only best-of-N, similarity-pruned LATR groups, or
  PRM-retained paths follow the raw Actor occupancy.  The manuscript must
  distinguish conditional advantage correctness from full occupancy weighting.
- PaTR explicitly acknowledges controlled, localized rollout-group bias while
  using ordinary Actor ratios.  This rules out any claim that the field is
  universally unaware of search-induced mismatch; our distinct contribution is
  the target-matched exact and finite-sample comparison of leaving versus
  correcting that bias.
- TSR explicitly names a search distribution distinct from the Actor and
  interprets its update as policy-proximal expert iteration.  TS-LLM and
  ReST-MCTS* likewise use search traces as a training teacher.  These are valid
  intentional distillation objectives and should be compared to E0/E4, not
  criticized as failed attempts to estimate the raw-Actor gradient.
- LATR and EPIG-Tree are allocation/diversity and gradient-efficiency
  competitors, respectively.  Their empirical gains do not answer the separate
  question of which density ratio is correct for a declared raw-Actor or
  composite-search target.  Allocation and estimator must remain orthogonal
  experimental axes.
- VinePPO and TRePO address response-level-to-token-level credit assignment by
  estimating continuation Value or advantages from prefix-conditioned Monte
  Carlo rollouts.  TRePO's theorem assumes trajectories and continuations are
  sampled on-policy from the current Actor.  Neither result, by itself,
  establishes that best-of-N winners, adaptively expanded branches, or
  PRM-retained trajectories have the raw Actor's occupancy distribution.
- Accordingly, the manuscript may use TRePO as theoretical support for the
  continuation-based credit-assignment axis, but must not cite it as an
  existing solution to search-selection propensity correction.  The two axes
  should be compared empirically where they interact and kept distinct in the
  formal estimands.
- EPIG-Tree is a direct competitor on branch allocation and local gradient
  estimation.  The manuscript must credit its occupancy/score/Value-uncertainty
  allocation law and restrict our distinct claim to target-aware correction of
  search-generated samples plus exact-truth and raw-K1 transfer evidence.
- General off-policy policy-gradient theory already supplies consistency and
  coverage results, while Top-K recommender correction already marginalizes a
  set-selection mechanism.  Neither is an adaptive LLM tree-search estimator
  matrix, but both rule out claims that off-policy or marginal selection
  correction is novel in itself.

## Manuscript narrative spine

This section fixes the intended organization of the ICLR manuscript without
promoting any claim that still awaits S3.  Search-augmented RL should be
presented as three separable interventions on the data-generating process:

1. **Allocation:** a gate decides where additional candidate compute is spent.
   Entropy, Value, and learned Value--entropy gates belong to this axis.
2. **Retention:** a selector or pruning rule decides which generated Actions
   and trajectories reach the update.  Best-of-N, PRM pruning, Value selection,
   and tree-search backup belong to this axis and can change both Action
   propensities and downstream state occupancy.
3. **Credit assignment:** a return, Value, or advantage estimator decides how
   the retained trajectory is scored across time.  VinePPO and TRePO primarily
   belong to this axis when their prefix continuations are sampled on-policy.

The paper's central distinction is that correct credit on a retained
trajectory does not make that trajectory an ordinary raw-Actor on-policy
sample.  Conversely, correcting search-selection propensity does not solve
poor temporal credit assignment.  The formal section should therefore define
two estimands before introducing estimators: the no-search raw-Actor objective
and the retained composite-search objective.  E0--E3 are then compared against
the former, while E4 targets the latter.  No numerical result may compare raw
K1 and retained-composite accuracy as if they were the same estimand.

The evidence should be ordered as a falsifiable chain rather than as a method
catalogue:

1. exact trees establish that search changes the behavior policy and occupancy;
2. E1/E2 establish exactness and expose representation-dependent variance;
3. E3 establishes the finite-sample bias--variance frontier;
4. the gate and Critic studies identify when extra search can produce useful
   alternatives rather than duplicate candidates;
5. S3 tests whether those estimator conclusions transfer to a no-search LLM
   Actor and, separately, to a retained-search composite system.

This framing supports a broad claim about search-generated training data while
keeping best-of-N, entropy branching, Value/PRM rollout, and credit-assignment
methods as controlled instances of the same data-generation decomposition.
Any stronger cross-family or large-model claim remains conditional on the
frozen S3 analysis and archive audit.

## Precommitted paper display plan

The main-paper displays are fixed by evidence role rather than by which method
wins.  Negative and mixed S3 contrasts remain in the same displays.

1. **Figure 1 -- estimands and data generation.** Diagram the raw Actor,
   allocation gate, candidate generator, retention rule, and credit estimator;
   mark the raw-Actor and retained-composite objectives separately.  Place
   VinePPO/TRePO on the credit axis rather than depicting them as propensity
   corrections.
2. **Figure 2 -- exact bias--variance frontier.** Show E0, E1, E2, and the
   development-frozen E3 correction strengths for all three search families at
   batch 256 and 1,024.  Include bias, MSE, and ESS views; do not show only the
   batch size or family that favors E3.
3. **Figure 3 -- when search shift helps.** Plot Critic ranking concordance,
   search gain, occupancy shift, and oracle search-gain headroom (the
   exact-Critic gain minus each condition's gain).  Include exact,
   learned, noisy, and inverted Critic conditions so the figure distinguishes
   shift magnitude from shift direction.
4. **Figure 4 -- no-search LLM transfer.** For each family, show every seed and
   the problem-clustered interval for E3-minus-E0 and E1-minus-E0, plus the
   preregistered best-of-N E3-minus-K1 contrast.  Include accuracy and answer
   rate; do not pool families based on their observed outcome.
5. **Figure 5 -- retained-search objective and cost.** Show E4-minus-E0
   retained-composite contrasts beside candidate tokens, realized expansion,
   exact-text uniqueness, token-prefix divergence, identical-candidate
   fraction, and GPU-hours. Label the diversity measures as surface-form
   diagnostics rather than semantic-path evidence.
6. **Table 1 -- estimator definitions.** State each estimator's target
   objective, logged information, exactness conditions, expected variance, and
   support requirement.  E4 is not ranked as though it estimated the raw-Actor
   objective.
7. **Table 2 -- cross-stage evidence.** Report mechanism-replication
   AUC/Brier/ECE and diversity, exact-tree bias/MSE/ESS/calibration, and S3
   accuracy/answer-rate/clip/KL/compute in separate columns with explicit data
   provenance.
8. **Table 3 -- frozen gates and reproducibility.** List every pass/fail gate,
   artifact count, protocol/data/code digest, test result, and archive-audit
   result.  A failed safety or compute gate remains visible even when an
   accuracy contrast is positive. Tables 1--3 must be rendered by the
   fail-closed generator from the terminal analysis and independent audit,
   retain the seven input SHA-256 values (including both the exact estimator
   confirmation artifact and its source-bound analysis), and reproduce
   byte-for-byte on a second invocation. Figures 1--5 must then be rendered only from that terminal table
   JSON, in PNG and PDF, with a SHA-256 manifest and a second byte-identical
   render. No observed winner may change which panels are emitted.

Appendix displays must retain the full per-seed wins/losses/ties, all
problem-clustered intervals and Holm adjustments, weight quantiles and support
checks, per-depth occupancy diagnostics, gate calibration curves, candidate
examples, and the complete matched-compute accounting.  This display plan is
administrative and does not alter any frozen experimental decision.

## Reproducibility anchors

- Exact-truth estimator matrix artifact SHA-256:
  `6e9655cebe49c8b76dd4dd309aed41d70fb71a58c99343b71af5f3f7e02028b0`.
- Exact-truth Critic artifact SHA-256:
  `87e1f3c3062a44de8d9651be1f305a5aea64305ef53642f9457175da40daa093`.
- Frozen S3 protocol SHA-256:
  `be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978`.
- Frozen MATH-500 scalar data SHA-256:
  `471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7`.
- Frozen S3 runner SHA-256:
  `466dd3608434f2244abd8861d55a554fee6d4c34aa2d17319423187acaf966a1`.

The final manuscript may promote an awaiting claim only by citing the completed
S3 analysis and archive audit. A mixed or negative S3 result is retained as such
and must not be rewritten as a positive large-model result.

Structural completeness is not sufficient for a compute-competitive claim.
The terminal workflow first generates a byte-reproducible preliminary archive
audit, then requires an explicit PASS/FAIL manual compute review bound to the
recorder, preliminary-audit preparer, terminal analysis, report, 12
family-by-estimator compute rows, and run-session/GPU-hour evidence.  The final
table and TeX renderers reject `UNSET`, a short or absent rationale, a
cross-version review toolchain, and any stale or malformed SHA-256 field.

The manuscript's related-work assertions are additionally anchored to the
living primary-source audit in `RELATED_WORK_AUDIT.md`.  As of 2026-10-29, its
paper-level entries include BPO (`2607.14171`), PaTR (`2607.15610`), EPIG-Tree
(`2609.20004`), TSR (`2602.11767`), and TRePO (`2506.02553`); its code-level
entries pin LATR, VinePPO, TS-LLM, MCTS-DPO, and the released ReST-MCTS* search
pipeline.  A paper-only audit must never be described as an implementation
audit.
