# CGCS Search-Induced Policy Mismatch: Exact-Truth Study

Status: **frozen**

Frozen on 2026-10-28 after the preregistered mechanism replication archive and
all fail-closed pre-freeze construction/S0 checks passed. The completed replication passed only
one of five mechanism gates: low-Value expansion produced 1.730 unique candidate
texts on average, 57.17% of expanded sets were completely identical, selected
post-Action Value AUC was 0.4392, and the full method did not improve held-out K=1
accuracy.  Those results motivate the exact-truth diagnostic but did not alter this
protocol's primary estimands, estimator set, search families, thresholds, or promotion
gates. Construction and S0 evidence is recorded in the companion pre-freeze checklist
and development-only smoke artifact. Any content change creates a new protocol version
and invalidates the seed commitment derived from these exact bytes.

## 1. Primary scientific question

When an Actor generates a candidate set and a search controller chooses which
candidate is executed, the executed behavior policy is generally not the raw Actor
policy.  The study asks which correction, if any, best transfers search-collected
experience into a strong no-search Actor.

The study separates two legitimate but different objectives:

1. **Raw-Actor objective**

   `J_actor(theta) = E[tau ~ pi_theta][R(tau)]`.

   This is the target when the final deliverable is a K=1 Actor with search removed.

2. **Composite-search objective**

   `J_search(theta, phi) = E[tau ~ mu_{theta,phi}][R(tau)]`.

   This is the target when trigger, candidate generation, and selector remain part
   of the deployed policy.

An estimator is not described as "more correct" without naming which of these two
objectives it targets.

## 2. Exact behavior distribution

At state `s`, let `g(s)` be the probability of expansion, let the raw Actor be
`pi(a|s)`, let `C=(a_1,...,a_K) ~ pi(.|s)^K`, and let
`q(i|s,C)` be the selector distribution.  The executed-action marginal is

```
mu(a|s) = (1-g(s)) pi(a|s)
        + g(s) sum_C p(C|s) sum_i q(i|s,C) 1[a_i=a].
```

For every finite synthetic problem, `mu` is enumerated exactly.  Trigger and
selector exploration floors are positive in all correction experiments so support
violations are measurable rather than silently hidden.

## 3. Environment and exact truth

- Primary finite layered delayed-reward MDP is fixed at `H=6`, `S=7`, `A=6`,
  `gamma=0.97`.  No confirmation-scale sensitivity environment is added without
  a new protocol version and new confirmation commitment.
- Use the repository's `LayeredMDP` and exact backward dynamic programming as the
  implementation base.  If it cannot represent every required construction
  property, this draft is not freezable; changing the environment requires a new
  protocol version before any confirmation seed is generated.
- Thirty committed confirmation MDP seeds.  Ten independent Monte Carlo streams
  per MDP are shared across every estimator and search family.  Development seeds
  1000, 1001, and 1002 are the fixed smoke subset used only for tests.
- Exact state values, action values, state occupancies, `J_actor`, and `J_search`
  are computed by dynamic programming and enumeration.
- The true gradient is obtained by differentiating or finite-differencing the exact
  objective.  Finite-difference agreement is a required implementation test, not a
  reported scientific result.

The generated MDP suite must include delayed credit assignment, misleading local
reward, low-value/low-entropy states, low-value/high-entropy states, and cases in
which the Critic is imperfect.  These properties are fixed from environment
construction, not selected after estimator results are seen.

### Confirmation seed commitment

Development MDP seeds 1000--1029 are permanently excluded.  After this protocol
is reviewed, copied into the repository, and marked `Status: **frozen**`, its exact
bytes are SHA-256 hashed.  A versioned deterministic generator derives thirty
unique confirmation MDP seeds from that digest and a fixed domain separator.  It
then derives ten Monte Carlo stream seeds per MDP without using any estimator or
search-family label.  Thus every method uses common random numbers within an MDP.

The generated manifest records the protocol digest and refuses to overwrite an
existing commitment.  No confirmation environment may be instantiated or
inspected before both the protocol and seed manifest are archived.  Editing the
protocol changes its digest and necessarily creates a new study version rather
than silently retaining favorable confirmation seeds.

## 4. Search families

The same environment, Actor snapshots, compute budget, and random-number streams
are shared where pairing is possible.

1. **Best-of-N:** always expand; selector is epsilon-soft value/PRM ranking.
2. **Entropy branching:** expand from an entropy gate; uniform and value selectors
   are both evaluated so gating is not confounded with selection.
3. **Value/PRM tree rollout:** expand from a low-value gate; uniform and value
   selectors are both evaluated.
4. **Value-and-entropy gate:** expand when correction need and branching
   opportunity are jointly present.  Evaluate both the direct AND-style heuristic
   and a gate fitted on development MDPs from Value, entropy, and their interaction.
   This is a mechanistic hypothesis, not assumed to win.
5. **Oracle expansion-gain gate:** rank states by the exact incremental value of
   expanding versus K=1.  This is an undeployable upper bound and supplies the
   ground-truth target for gate AUC/calibration.

Primary candidate counts are `K in {2,4}`.  `K=1` is the raw-Actor control.  The
two primary compute budgets are expected candidates `B in {1.5, 2.0}` per visited
state.  For a K-candidate expansion, the occupancy-weighted target expansion is
`g=(B-1)/(K-1)`; infeasible `(K,B)` pairs are omitted rather than silently given a
different budget.  Gates are calibrated by exact occupancy to this target, not by
matching a nominal threshold.  Non-uniform selectors use standardized Critic
scores with `beta=1.0` and an `epsilon=0.10` uniform floor.  Soft gates use sigmoid
temperature 1.0.  Uniform selectors remain exactly `1/K`.

The gate's fundamental target is value of computation,
`Delta_expand(s) = V_after_search(s) - V_K1(s)`, not low Value or high entropy by
itself.  Low Value represents possible correction need; high entropy represents
possible branching opportunity.  The exact study tests whether either signal or
their interaction predicts `Delta_expand`.  Any learned gate is fitted only on
development MDP seeds, then frozen for disjoint confirmation seeds.

The development split is fixed as follows: MDP seeds 1000--1019 are available for
gate fitting and seeds 1020--1029 are a disjoint development test.  All thirty are
permanently excluded from the confirmation study.  The confirmation seeds are
generated and frozen only after the implementation tests and analysis schema are
complete, and before any confirmation outcome is inspected.

The primary learned proxy is ridge regression with penalty `1e-3` and fixed
features `[1, low_value, entropy, low_value * entropy]`.  Its result is retained
even if negative.  The development-only feature-sufficiency diagnostic contains
exactly four additional fixed groups: Value--entropy monomials through degree 3;
the primary group plus top Actor probability, top-two probability margin, and
collision probability; the primary group plus policy-weighted Q standard
deviation, best-Q minus mean-Q, and Actor-log-prob/Q covariance; and the union of
those features plus `entropy * Q_std` and `low_value * best_Q_gap`.  Every group
uses the same ridge penalty.  No other feature, model class, or hyperparameter may
be chosen using confirmation MDPs.  This separates three possible findings: the
VOC target lacks useful headroom, the available signals do not identify it, or the
gate model lacks capacity.

The richer diagnostic must label full per-Action Critic moments as diagnostic
rather than free deployment features.  Any LLM version that estimates return
dispersion through extra candidates, an uncertainty head, or an ensemble charges
that estimation cost to its compute budget.  A sequential `K=1 -> K=2 -> K=4`
controller is allowed only as a separately named search family with exact logging
of each stopping probability; it must not be folded into the original gate after
results are known.

## 5. Critic-quality axis

Each search family is evaluated with:

- exact `Q^pi` (oracle signal, diagnostic only);
- calibrated noisy Q obtained after within-state standardization by adding
  independent standard-normal noise with `sigma in {0.5, 1.0, 2.0}`; negative
  exact Q is retained as a labelled inverted diagnostic, not a primary condition;
- lagged/off-policy Q from Actor snapshots at exact mean state-weighted KL
  distances `{0.01, 0.04}` from the current Actor; within each MDP, the lagged
  snapshot is produced deterministically by moving opposite the exact raw-Actor
  gradient and solving the step size to the declared KL, with no condition chosen
  from its measured search result;
- learned finite-data Q fitted on `{256, 1024}` raw-Actor trajectories by one-hot
  tabular ridge regression with penalty `1e-3` and discounted terminal-return
  targets, with disjoint fitting and evaluation Monte Carlo streams.

For the oracle/noisy/inverted comparisons, Critic scores are standardized within
state and selector beta/epsilon are fixed.  This matches selection pressure while
changing ranking quality.  Report pairwise Q-ranking concordance, selector entropy,
search return, and occupancy divergence separately.  Otherwise a poorer Critic can
appear to induce less mismatch merely because its score scale is smaller.

This axis determines whether a failure is caused by the correction estimator,
the search mechanism, or insufficient Critic quality.  It also tests the specific
hypothesis that selector strength controls the *magnitude* of search-induced shift,
whereas Critic ranking quality controls whether that shift is beneficial.

## 6. Estimators and unambiguous names

### E0. Standard selected-action Actor ratio

Use the selected action as ordinary on-policy data and apply only
`pi_theta(a|s) / pi_old(a|s)`.  This is the common search-distillation baseline.
It is expected to target neither `J_actor` nor the full composite objective exactly
when search changes the behavior distribution; that claim is measured, not assumed.

### E1. Conditional selector propensity correction

For an expanded candidate set, use the Horvitz--Thompson factor
`1 / (K q_old(i|s,C))` outside PPO clipping.  For an unexpanded state the factor is
one.  This maps the conditional selector expectation back to uniform selection
within a logged candidate set.  Two variants are kept separate:

1. **E1-local:** apply only the current state's factor to the current training
   sample, as in a conventional weighted PPO minibatch.  This corrects the
   conditional selected-Action expectation but does not correct the state
   distribution inherited from earlier search decisions.
2. **E1-trajectory:** use the cumulative product of conditional factors up to
   each reward/advantage term.  With full support and no clipping, this also
   corrects the preceding search-induced state-occupancy shift, but can have much
   higher variance.

The two variants must never be reported under one undifferentiated
"propensity-corrected" label.

### E2. Exact executed-action marginal correction

For the raw-Actor target, use exact per-decision ratios based on
`pi_old(a|s) / mu_old(a|s)`, together with the new/old Actor ratio where required.
Use cumulative ratios for the exact trajectory reference.  Also report the
local-only marginal ratio as a diagnostic and V-trace truncation pairs
`(rho_bar,c_bar) in {(1,1),(2,1)}` as bias--variance alternatives.  This is the
exact finite-environment reference for both selected-Action and state-distribution
correction.

### E3. Partial correction

Partial correction is evaluated under three explicitly named constructions:

1. **E3-local:** for the current sample weight `w_t`, use

   `w_t(lambda) = (1-lambda) + lambda w_t`.

   This matches the practical existing selector-weight ablation, but it remains
   local-only and therefore does not claim state-occupancy correction.
2. **E3-step-cumulative (primary occupancy-aware partial estimator):** apply the
   same interpolation at every step, then use the cumulative product
   `prod_{u<=t} w_u(lambda)` for the depth-`t` term.
3. **E3-trajectory-blend (diagnostic):** first form the exact cumulative ratio
   `W_t=prod_{u<=t}w_u`, then use `(1-lambda)+lambda W_t`.  This is a direct convex
   interpolation between behavior and corrected estimators, but need not correspond
   to a Markov behavior policy, so it is not the primary deployable method.

All use `lambda in {0, 0.25, 0.5, 0.75, 1}`.  The primary analysis is the
gradient/return bias--variance--MSE frontier at preregistered effective batch sizes
`{256, 1024}`, together with ESS, rather than whichever lambda maximizes one return.
No lambda is chosen from the final MDP seeds.  The minimal LLM stage uses only
definitions selected and frozen from development MDP seeds; it does not carry all
three constructions forward automatically.

### E4. Composite-policy ratios

Two quantities must not be conflated:

1. **Executed-action marginal ratio:** `mu_theta(a|s) / mu_old(a|s)`.
2. **Extended-action ratio:** for logged `(C,i)`, the exact ratio of the full
   candidate-generation and selector probability.  Because every candidate in
   `C` is generated by the Actor, the numerator includes the likelihood ratio for
   the complete candidate set, not only the selected action, followed by the
   selector ratio `q_theta/q_old`.

These estimators target `J_search`, not `J_actor`.  A selected-action-times-selector
proxy is included as an ablation because it is cheaper, but is not labelled exact
unless enumeration proves equality in the tested special case.

## 7. Primary measurements

For every estimator, search family, and Critic-quality condition, report:

- gradient bias norm and componentwise sign error;
- gradient variance trace and mean-squared error;
- cosine similarity to the named exact target gradient;
- effective sample size, maximum weight, weight quantiles, and support violations;
- state-occupancy total variation/KL by depth and return-estimator bias by depth;
- gate AUC/calibration against exact `Delta_expand`, allocation gain per expansion,
  and paired return versus random gating at exactly matched expansion;
- gate-model capacity and feature-sufficiency on the development split, with the
  fixed linear proxy always reported and no confirmation-set model selection;
- exact return change after scaling each estimated logit-gradient update to exact
  mean state-weighted KL 0.01 from the old policy (or zero movement for a zero
  gradient); both raw-Actor and composite-search return are then recomputed;
- KL from the old Actor and from the old composite behavior;
- expected generated candidates and expected environment steps;
- no-search K=1 Actor return and composite-search return separately.

Confidence intervals are clustered by MDP seed.  Monte Carlo seeds are repeated
measurements inside an MDP cluster, not independent problem instances.

## 8. Required implementation checks

1. `K=1` recovers the raw Actor distribution and objective.
2. A uniform selector recovers the raw Actor executed-action marginal.
3. Enumerated behavior probabilities sum to one and have the declared support.
4. Exact marginal importance sampling recovers raw-Actor expectations.
5. Exact composite ratios recover composite-policy expectations.
6. Conditional propensity correction recovers uniform selection conditional on C.
7. With unit ratios, per-decision IS and V-trace reduce to their on-policy forms.
8. Exact dynamic-programming gradients agree with central finite differences.
9. Candidate-set extended ratios include all candidate likelihood terms; a test
   must fail if non-selected candidate terms are deliberately removed.
10. Paired random streams produce identical raw candidate pools across estimators.
11. In the delayed-reward MDP, cumulative conditional and exact-marginal ratios
    recover the raw-Actor return on smoke seeds using 1,000,000 trajectories: the
    absolute error must be at most 0.01 and the exact target must lie within three
    estimated standard errors.  Local-only weights are reported separately and
    are not accepted as proof of occupancy correction.

## 9. Stage gates

### Gate S0: mathematical and software validity

All eleven implementation checks pass.  Probability normalization and exact
expectation identities use absolute tolerance `1e-10`; central finite-difference
gradients use relative L2 error at most `1e-5`; the large Monte Carlo recovery test
uses the rule in check 11.  Failure stops expansion and triggers diagnosis.

### Gate S1: identifiable search-induced mismatch

The primary mismatch setting is best-of-4 with exact Q, `epsilon=0.10`, and
expected-candidate budget 2.0.  S1 requires at least 20 of 30 MDPs to have both
maximum-depth state-Action occupancy TV at least 0.05 and either E0 relative
gradient bias at least 0.10 or E0 cosine similarity at most 0.98.  Exact cumulative
E2 must have relative gradient bias at most 0.02 and cosine at least 0.995 in the
pooled exact expectation.  E1-local and E1-trajectory are compared explicitly. If
this gate fails, the paper cannot claim the mismatch matters in this environment;
environment changes require a new protocol and fresh seeds.

### Gate S2: bias--variance finding

The study must estimate the complete lambda frontier and establish whether partial
correction improves gradient MSE or downstream K=1 return relative to both
endpoints.  S2 is a completeness gate, not a positive-outcome gate: a negative
result is retained.  A lambda may advance only if selected by mean development-MDP
gradient MSE at its declared batch size before confirmation; ties choose the
larger lambda.  No LLM expansion is authorized merely because one lambda wins one
synthetic seed.

### Gate S3: transfer to controlled LLM experiments

Only after S0--S2 are archived may the repository freeze the minimal LLM matrix.
Its fixed search rows are K1; always-expand best-of-2; entropy-gated K4; and
Value/PRM-gated K4, with the latter two calibrated to expected-candidate budget
2.0.  Its estimator columns are E0, E1-trajectory, the single E3-step-cumulative
lambda selected on development MDPs at the matching batch size, and E4 extended
candidate-set ratio for the composite target.  At least three training seeds use
shared data order and candidate random streams.  LLM studies compare the final
search-free K1 Actor and the retained composite policy separately at matched
generated tokens, accepted tokens, optimizer updates, and GPU-hours.

## 10. Planned repository artifacts after review

- frozen protocol under `results/search_policy_mismatch/`;
- exact-enumeration module extending `candidate_policy.py` and `layered_mdp.py`;
- estimator module with explicit target names;
- unit and Monte Carlo regression tests;
- machine-readable per-seed summaries;
- bias--variance, ESS, and compute--return tables;
- final report linking every claim to a frozen contrast and artifact.

## 11. Non-goals for this stage

- no new GPU job before the current mechanism replication is validated and
  archived;
- no outcome-driven threshold or lambda selection;
- no claim that Value--entropy gating is superior before matched tests;
- no automatic expansion to MCTS, more model sizes, or more datasets before the
  exact-truth and minimal LLM stage gates pass.
