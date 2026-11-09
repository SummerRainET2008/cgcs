# Exact-truth stage-gate status

| Gate | Status | Evidence |
|---|---|---|
| S0 mathematical/software validity | PASS | 11 required identities/checks, 1M-trajectory recovery on three development smoke MDPs, focused tests |
| S1 identifiable search-induced mismatch | PASS | 30/30 committed MDPs versus threshold 20/30; exact E2 pooled bias 0 and cosine 1 |
| S2 complete bias--variance finding | PASS | 2 ratio bases x 3 constructions x 5 lambdas on all 30 MDPs; development selection replicated under clustered confirmation |
| S3 controlled LLM transfer | COMPLETE; STABILITY GATE FAILED | Frozen SHA `be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978`; all 81 training summaries, 39 raw-K1 evaluations, and 36 retained-composite evaluations are complete. 72/72 search branches exceed the preregistered sequence-ratio clip-fraction ceiling of 0.20; the frozen positive-transfer gate is FAIL/MIXED, so S3 cannot authorize expansion. |

The S3 clip failure is a preregistered safety-gate result, not an interim
accuracy stopping decision.  The frozen job completed unchanged, and all 39
raw-K1 and 36 retained-composite evaluations, paired contrasts, clustered
intervals, compute accounting, and diagnostics are retained.  The
independent audit additionally separates ratio excursions from actual
Actor-surrogate saturation; the frozen clip gate remains failed under its
original definition.  No next-stage PPO/GRPO experiment is authorized.

This terminal row is immutable: the negative/mixed final result remains in the
archive, including the failed stability and positive-transfer gates.  No larger
LLM experiment is authorized by S3, regardless of any favorable terminal
accuracy contrast.

Passing S2 is a completeness result, not a claim that every correction improves return. The retained finding is batch-dependent: lambda 0.75 minimizes gradient MSE at batch 256, while lambda 1 is selected at batch 1024.

The preregistered Critic-quality axis is also complete on all 30 committed
confirmation MDPs.  Under matched selector scale and compute, pooled Critic
ranking concordance correlates above 0.92 with search gain, while its correlation
with maximum-depth state-action TV is below 0.05.  Exact and inverted Critics
produce almost equal shift magnitude but opposite return effects.  This passes
the mechanism-identification purpose of the axis: mismatch magnitude and Critic
directional quality are distinct, and both must be reported.  See
`CRITIC_QUALITY_REPORT.md` and `CRITIC_CONFIRMATION.json`.

The canonical cross-family E0--E4 confirmation is complete for all 30 committed
MDPs, nine Critic conditions, and three search families.  Two full runs are
byte-identical.  Exact E1/E2 and E4 identities hold; the development-frozen
lambda 0.75 has lower batch-256 gradient MSE than E0 and full E2 in all 27
Critic-by-family conditions.  At batch 1024, full correction catches or exceeds
partial correction in the two K4 families, confirming the preregistered
batch-dependent bias--variance interpretation.  E3 improves K1 transfer over E0
for best-of-2, while the two K4 intervals cross zero.  The executed-action E4
marginal is the lower-variance exact composite reference; the selected-action
proxy remains labelled biased.  See `ESTIMATOR_MATRIX_CONFIRMATION_REPORT.md`.
