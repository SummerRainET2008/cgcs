# Exact-truth Critic-quality confirmation

## Scope and frozen design

The Critic-quality axis was executed on all 30 committed confirmation MDPs under
the frozen exact-truth protocol.  It covers exact Q, three shared-noise levels,
an explicitly labelled inverted diagnostic, two lagged-Actor snapshots at exact
state-weighted KL 0.01 and 0.04, and tabular Q estimates learned from 256 or 1024
raw-Actor trajectories.  Selector scores are standardized within state and use
the same beta and epsilon, so score scale cannot masquerade as Critic quality.
All conditions use identical candidate counts, exact realized expansion budgets,
MDPs, gates, and raw Actors.  The saved artifact is deterministic byte-for-byte;
its SHA-256 is `87e1f3c3062a44de8d9651be1f305a5aea64305ef53642f9457175da40daa093`.

## Primary finding

For the K=4, expected-candidate budget 2.0, random-gate condition:

| Critic condition | Pairwise ranking concordance | Search gain vs raw Actor | Max-depth SA TV | One-step marginal ESS | Selector entropy |
|---|---:|---:|---:|---:|---:|
| Exact | 1.0000 | +0.4028 | 0.0880 | 0.9702 | 1.1847 |
| Noise 0.5 | 0.8803 | +0.3720 | 0.0865 | 0.9700 | 1.1846 |
| Noise 1.0 | 0.7851 | +0.2992 | 0.0828 | 0.9702 | 1.1870 |
| Noise 2.0 | 0.6719 | +0.1871 | 0.0797 | 0.9704 | 1.1888 |
| Inverted diagnostic | 0.0000 | -0.3909 | 0.0872 | 0.9718 | 1.1991 |
| Lagged KL 0.01 | 0.9735 | +0.3984 | 0.0869 | 0.9703 | 1.1854 |
| Lagged KL 0.04 | 0.9516 | +0.3909 | 0.0860 | 0.9703 | 1.1856 |
| Learned, 256 trajectories | 0.6510 | +0.3029 | 0.0806 | 0.9738 | 1.2105 |
| Learned, 1024 trajectories | 0.6997 | +0.3393 | 0.0828 | 0.9734 | 1.2083 |

Across the complete 30-MDP x 9-condition axis, ranking concordance has Pearson
correlation 0.931 with search gain for the K=4/B=2 random gate and 0.927 for the
entropy gate.  The analogous correlations with maximum-depth TV are only 0.049
and 0.010.  The exact and inverted Critics are the decisive directional control:
they induce almost the same TV and ESS, but their gains have nearly equal and
opposite signs.  Thus selector strength and support determine how much the
executed policy shifts; Critic ranking quality determines whether that shift is
useful or destructive.  ESS and TV diagnose mismatch magnitude, not its utility.

## Paired confirmation against exact Q

With the entropy gate at K=4/B=2, every imperfect Critic is below exact Q in
paired problem-clustered confirmation:

| Condition | Search gain | Difference from exact | 95% clustered CI | Wins/losses |
|---|---:|---:|---:|---:|
| Noise 0.5 | +0.3869 | -0.0317 | [-0.0369, -0.0269] | 0/30 |
| Noise 1.0 | +0.3110 | -0.1076 | [-0.1208, -0.0954] | 0/30 |
| Noise 2.0 | +0.1915 | -0.2271 | [-0.2502, -0.2059] | 0/30 |
| Inverted diagnostic | -0.4084 | -0.8270 | [-0.8775, -0.7773] | 0/30 |
| Lagged KL 0.01 | +0.4141 | -0.0045 | [-0.0066, -0.0028] | 2/28 |
| Lagged KL 0.04 | +0.4061 | -0.0125 | [-0.0188, -0.0076] | 0/30 |
| Learned, 256 trajectories | +0.3160 | -0.1026 | [-0.1114, -0.0945] | 0/30 |
| Learned, 1024 trajectories | +0.3545 | -0.0641 | [-0.0693, -0.0588] | 0/30 |

The finite-data comparison directly answers the insufficient-training
hypothesis: more data improves ranking concordance from 0.6510 to 0.6997 and
search gain from 0.3160 to 0.3545, so Critic training quality matters.  It does
not explain away the earlier gate result, however.  Even exact Q leaves the
simple low-Value/high-entropy gate below the oracle allocation gate, because
state Value and Actor entropy do not encode the candidate-specific value-of-
computation structure.

## Interaction with the gate finding

Entropy allocation remains beneficial over matched random allocation when the
Critic has positive ranking skill, but its absolute payoff falls with Critic
quality.  At K=4/B=2, entropy-minus-random gain is +0.0158 with exact Q, +0.0118
with noise 1.0, +0.0044 with noise 2.0, and -0.0175 under the inverted diagnostic.
This separates two mechanisms that must not be conflated:

1. the gate allocates candidate-generation compute to states;
2. the Critic ranks the candidates produced at those states.

A good gate cannot rescue a directionally wrong selector, and a good selector
does not make low Value plus high entropy a sufficient value-of-computation
statistic.  The planned search-policy-mismatch experiments must therefore report
gate allocation quality, Critic ranking quality, and correction-estimator quality
as separate axes.

## Reproducibility evidence

- Artifact: `CRITIC_CONFIRMATION.json` (30 committed MDPs, 9 Critics, 4
  compute settings, 6 gates).
- Exact budgets are verified under each condition's induced occupancy.
- Lagged snapshots hit KL 0.01 and 0.04 to numerical precision on every MDP.
- Learned Critic fitting streams are deterministic and use exactly 256 or 1024
  raw-Actor trajectories.
- Paired differences use 10,000 problem-clustered bootstrap resamples.
- Unit and artifact tests verify frozen condition completeness, manifest and
  protocol binding, compute matching, directional controls, and pooled
  diagnostics.

This stage completes the preregistered Critic-quality axis.  It does not by
itself authorize LLM scaling; the remaining exact-truth estimator/search-family
matrix and K1-transfer gates remain required.
