# Exact-truth E0--E4 estimator matrix: confirmation report

## Status and reproducibility

The frozen confirmation matrix is complete.  It contains all 810 unique
`(MDP, Critic condition, search family)` rows: 30 committed MDPs, nine Critic
conditions, and three canonical search families.  Every row uses the ten
manifested common-random-number streams with 4,096 trajectories per stream
(40,960 trajectories per row), expected-candidate budget 2.0, and ten retained
estimators.  All values are finite and every exact target identity passes.

The complete matrix was generated twice from the frozen protocol and seed
manifest.  The two files are byte-identical with SHA-256
`6e9655cebe49c8b76dd4dd309aed41d70fb71a58c99343b71af5f3f7e02028b0`.
The archived analysis is bound to that digest and contains 260 paired,
MDP-clustered contrasts using 10,000 bootstrap resamples.  Critic conditions are
repeated measurements within an MDP; they are never treated as independent
problems.

## Primary result

Search-selected data are not ordinary raw-Actor on-policy data.  Under the exact
Critic, treating selected Actions as ordinary samples (E0) has mean relative
gradient bias 36.91% for always-expanded best-of-2, 24.94% for entropy-gated K4,
and 23.41% for low-Value-gated K4.  The bias accompanies measurable
state--Action distribution shift; it is not only a selector-level artifact.

Both trajectory-level conditional propensity correction (E1-trajectory) and
exact executed-action marginal correction (E2-cumulative) recover the raw-Actor
gradient exactly in expectation.  Their finite-sample behavior differs sharply.
E1 multiplies conditional selector propensities and collapses effective sample
size, while E2 first marginalizes the executed behavior policy.  Pooled over all
nine Critic conditions, E1 has higher batch-256 relative gradient MSE than E2 by:

| Search family | E1 minus E2 MSE | 95% MDP-clustered CI |
|---|---:|---:|
| Best-of-2 | +0.6185 | [+0.5299, +0.7152] |
| Entropy-gated K4 | +0.2037 | [+0.1880, +0.2200] |
| Low-Value-gated K4 | +0.1828 | [+0.1687, +0.1983] |

Thus unbiasedness alone is insufficient: how the correction represents the
behavior policy determines its usable sample efficiency.

## Partial correction and the bias--variance frontier

The development-frozen step-cumulative partial correction with `lambda=0.75`
has lower batch-256 gradient MSE than both E0 and full E2 in every one of the 27
Critic-by-family conditions.  Pooled paired differences are:

| Search family | E3(0.75) minus E0 MSE256 | E3(0.75) minus E2 MSE256 |
|---|---:|---:|
| Best-of-2 | -0.0488 `[-0.0739,-0.0291]` | -0.0610 `[-0.0698,-0.0528]` |
| Entropy-gated K4 | -0.0410 `[-0.0507,-0.0323]` | -0.0089 `[-0.0106,-0.0072]` |
| Low-Value-gated K4 | -0.0406 `[-0.0502,-0.0318]` | -0.0075 `[-0.0093,-0.0059]` |

This is the strongest confirmation result: finite-data training benefits from a
controlled amount of residual search-distillation bias because full correction
pays a larger variance cost.

The result is batch-dependent, as preregistered.  At batch 1,024, partial
correction still beats full E2 for best-of-2 by -0.0100
`[-0.0125,-0.0075]`.  For entropy-gated K4 the difference is +0.0004 with an
interval crossing zero, while for low-Value-gated K4 full E2 wins by 0.0008
`[0.00004,0.00153]`.  The correct interpretation is therefore not that
`lambda=0.75` is universally optimal.  Correction strength should increase as
effective sample size grows and the variance penalty becomes affordable.

## Transfer to the no-search Actor

Every estimator update is normalized to the same raw-Actor KL step, so return
changes are not caused by different update magnitudes.  Pooled over Critic
conditions, E3(0.75) improves the post-update K1 return over E0 for best-of-2 by
+0.0102 `[+0.0061,+0.0145]`.  The corresponding differences for entropy-gated
K4 (+0.0021 `[-0.0006,+0.0049]`) and low-Value-gated K4
(+0.0006 `[-0.0017,+0.0030]`) are not statistically resolved.

Therefore the matrix confirms a robust estimator-MSE benefit and a clear K1
transfer benefit in the strongest-mismatch best-of-2 setting.  It does not yet
establish that partial correction improves K1 return for every gated K4 search.
That boundary is carried forward to the controlled LLM stage.

## Composite-search objective and E4

For a deployed search controller, both the executed-action marginal E4 estimator
and the full logged-candidate-set E4 estimator recover the exact composite-policy
gradient.  Marginalizing latent candidate sets is a Rao--Blackwellization and has
substantially lower batch-256 MSE:

| Search family | E4 marginal minus E4 extended-set MSE | 95% CI |
|---|---:|---:|
| Best-of-2 | -0.0772 | [-0.0817,-0.0728] |
| Entropy-gated K4 | -0.0995 | [-0.1078,-0.0917] |
| Low-Value-gated K4 | -0.0924 | [-0.0988,-0.0862] |

The selected-Action-only E4 proxy is not exact because it omits generated but
unselected candidates.  It illustrates the opposite side of the same tradeoff:
for best-of-2 its lower variance yields slightly lower MSE than exact marginal E4
(-0.0021 `[-0.0041,-0.0002]`) despite systematic bias; for both K4 families the
exact marginal estimator has lower MSE.  A cheap proxy can be useful, but it must
not be described as the exact joint policy ratio.

## Search-family context

With the exact Critic, mean search gain and maximum-depth state--Action TV are:

| Search family | Search gain vs raw Actor | Maximum-depth TV |
|---|---:|---:|
| Always-expanded best-of-2 | +0.7237 | 0.1605 |
| Entropy-gated K4 | +0.4186 | 0.0941 |
| Low-Value-gated K4 | +0.4036 | 0.0962 |

Best-of-2 exceeds entropy-gated K4 search gain by +0.3051
`[+0.2786,+0.3326]` under the exact Critic and remains higher under every
non-inverted Critic condition.  Entropy-gated and low-Value-gated K4 differ by
only +0.0150 `[-0.0063,+0.0371]` under the exact Critic.  Greater search gain in
best-of-2 comes with greater distribution shift and a more severe correction
variance problem; it is not a free dominance result at matched expected compute.

## Decision

The exact-truth estimator/search-family confirmation gate passes.

1. Search induces a real behavior-policy and occupancy mismatch.
2. Ordinary selected-Action Actor ratios are biased for the raw-Actor objective.
3. Exact executed-action marginal correction is materially more sample-efficient
   than conditional trajectory propensity correction.
4. Partial correction gives the best finite-data frontier at batch 256, while
   larger batches move the optimum toward full correction.
5. For the composite-search objective, executed-action marginalization is the
   preferred exact E4 reference; selected-Action-only E4 remains a labelled
   biased proxy.

This gate authorizes freezing a minimal controlled LLM transfer experiment.  It
does not authorize a broad claim that any search family or correction is
universally superior, and it does not yet establish improvement at large-model
scale.
