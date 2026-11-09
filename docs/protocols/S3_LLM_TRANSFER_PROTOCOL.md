# S3 controlled LLM transfer protocol

Status: **frozen**

This protocol implements Gate S3 of `EXACT_TRUTH_PROTOCOL.md`.  It is a minimal
controlled transfer of the exact-truth estimator findings, not the full PPO/GRPO
publication matrix.  No confirmation output may be opened until this file is
marked frozen, its digest is embedded in the only runner, and all pre-freeze
checks pass.

## 1. Questions and named objectives

1. Does search-selected LLM data exhibit the same finite-sample correction
   tradeoff as the exact environment?
2. Does partial selector correction improve the final search-free K1 Actor over
   ordinary selected-Action training at matched search data and update budget?
3. Does the extended candidate-set score improve the retained composite search
   policy when its controller is treated as fixed during each update?

Raw-Actor results and retained-search results are separate estimands.  No method
is called more correct without naming its target.

## 2. Model, shared Value, and task

- Base: `Qwen/Qwen3-4B-Instruct-2507`, BF16, SDPA.
- Actor: attention LoRA rank 32, alpha 64.  Value: one scalar head sharing the
  Actor backbone; no separately pretrained Critic or PRM.
- Response-level correctness is the only reward; intermediate reward is zero.
- Training source: GSM8K train offsets 6500--6643, already exposed to an earlier
  selector-data construction but never used as S3 confirmation.  It is training
  data, not evidence of independent generalization.
- Burn-in: offset 6500/count 48.  Adaptive rounds: 6548/count 48 and
  6596/count 48.  Maximum seven Actions and 80 tokens per Action.
- Sampling/optimizer seeds: 99801, 99802, 99803.
- Training temperature 1.3, top-p 1.0.  Stored and recomputed token
  probabilities both use logits divided by 1.3.

## 3. Independent evaluation and data seal

The independent evaluation is the scalar-answer subset of
`HuggingFaceH4/MATH-500` at immutable revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.  The deterministic preparation
script accepts only integers, decimals, and exact integer fractions and rejects
commas rather than confusing tuples with thousands separators.  The resulting
368-row file has SHA-256
`471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7`.

A fixed SHA-256 rank of source unique IDs with domain
`cgcs-search-policy-mismatch-s3-math500-v1` creates:

- open development: offset 0/count 48;
- sealed S3 confirmation: offset 48/count 200;
- untouched reserve: offset 248/count 120.

The confirmation partition is not evaluated until every seed and training arm
has completed both adaptive rounds.  The reserve partition is not read in S3.
Raw K1 evaluation is greedy, maximum eight Actions, 96 tokens per Action.  The
retained composite evaluation uses temperature 1.3 and one fixed common sampling
seed 100901 for pairing.  Both use maximum prompt length 768 and head retention
192.  All 200 problem IDs must appear in the frozen manifest order.

## 4. Search rows and compute budget

All search rows use beta 1.0, selector epsilon 0.10, Value scale floor 0.05,
and exact logged selector probabilities.

1. `k1`: one Actor Action, no search.
2. `best_of_n`: always-expanded K=2 with Value selection; expected candidates 2.
3. `entropy_value`: K=4, high-entropy gate calibrated from the preceding rollout
   to expansion probability 1/3, followed by Value selection.
4. `value_value`: K=4, low post-Action Value gate calibrated from the preceding
   rollout to expansion probability 1/3, followed by Value selection.

The K4 rows therefore target expected candidates 2.  Ties may cause a measured
realized-budget deviation; thresholds are never adjusted from evaluation output.
Candidate streams, data order, selector draws, and starting checkpoint are shared
where policies have not yet diverged.  Round-one search rollouts are collected
once per seed/family and reused by all estimator columns.

## 5. Estimator columns

The K1 control uses only E0.  Each search row uses all four columns below.
Every generated candidate is forwarded in every column so candidate-token
training accounting is comparable within a search row.

### E0 standard Actor ratio

Use the selected complete Action sequence ratio
`pi_new(a|s)/pi_old(a|s)` and no selector correction.  This is the practical
search-distillation baseline and is not claimed to be raw-Actor on-policy.

### E1 trajectory propensity

For an expanded state use `w_t=1/(K q_old(i|s,C))`; otherwise use one.  Multiply
these factors cumulatively along the problem trajectory.  The cumulative weight
is outside PPO clipping.  Positive epsilon gives full selector support.

### E3 step-cumulative partial correction

At each step use `(1-lambda)+lambda*w_t`, multiply along the trajectory, and
place the cumulative weight outside PPO clipping.  `lambda=0.75` is transferred
unchanged from the batch-256 exact-truth selection; S3 does not retune it.

The open-ended language executed-action marginal is not exactly enumerable.
Therefore E1 and E3 use the logged conditional-selector basis, explicitly not
the synthetic E2 marginal basis.  The exact study predicts that this is a
high-variance stress test; ESS and maximum weights are mandatory diagnostics.

### E4 extended candidate set

Use the product of generation likelihood ratios of every logged candidate in
the set.  Gate and selector gradients are stopped during the update, matching
the fixed-controller exact-truth E4 definition.  The shared Value Head still
receives its response-return regression loss.  Consequently this column targets
the candidate-generation policy under the logged controller and is named a
**frozen-controller composite surrogate**; it is not represented as the exact
gradient of a simultaneously changing hard gate and Value selector.

All four columns use sequence-level PPO clipping 0.2.  Non-finite sequence
ratios fail closed.  One epoch, learning rate 5e-6, accumulation 32, KL
coefficient 0.02, Value coefficient 0.5, and maximum gradient norm 1.0 are fixed.

## 6. Execution graph

Each seed first produces one common K1 burn-in checkpoint.  The K1 branch then
runs E0 for two rounds.  Each search family collects one shared round-one rollout
from burn-in, trains the four estimator branches, then collects and trains its
own round-two rollout from each diverged branch.  This yields:

- 3 burn-in summaries;
- 6 K1 adaptive summaries;
- 72 search-estimator adaptive summaries (3 seeds x 3 families x 4 estimators x
  2 rounds);
- 39 final raw-K1 evaluations (3 seeds x 13 branches);
- 36 retained-composite evaluations (3 seeds x 12 search branches).

Only `scripts/run_search_policy_mismatch_s3_rsaturn.sh` may run the frozen job.
It is resumable and skips only artifacts that pass structural validation.

## 7. Measurements and inference

For every rollout and update retain accuracy, answer rate, states, Actions,
candidate count/tokens, accepted tokens, realized expansion, candidate diversity,
selector probabilities, collection/training wall time, Actor/Value loss, sequence
ratio, clip fraction, KL proxy, external-weight ESS/maximum/quantiles, optimizer
updates, peak memory when available, and GPU-hours.

Primary paired contrasts are:

1. raw K1: E3 minus E0 within each search family;
2. raw K1: E1 minus E0, diagnostic for full propensity correction;
3. retained composite: E4 minus E0 within each search family;
4. best-of-2 E3 minus the K1 control, secondary quality--compute context.

Report per-seed wins/losses/ties and differences.  The primary interval is a
10,000-resample problem-clustered bootstrap in which all seed observations for a
problem remain in one cluster.  Family-level intervals are reported separately;
no outcome-selected family pooling.  Holm correction is reported for the three
family instances of each primary contrast.  Accuracy, answer rate, generated
tokens, candidate tokens, wall time, and GPU-hours are all retained; optimizer
updates alone do not establish compute matching.

## 8. Gates and stop rules

Completeness and safety require every planned artifact, finite losses/weights,
zero support violations, sequence-ratio clip fraction below 20%, mean selected
KL proxy below 0.02, and no answer-rate regression exceeding two percentage
points versus the K1 control.  Failure triggers diagnosis, not silent exclusion.

S3 establishes positive transfer only if all of the following hold:

1. E3 has positive raw-K1 mean difference versus E0 in every training seed for
   at least one preregistered search family, and that family's clustered 95%
   interval has nonnegative lower bound;
2. E3 has higher external-weight ESS than E1 in every seed/family;
3. E4 has positive retained-composite mean difference versus E0 in every seed
   for at least one family, with nonnegative clustered lower bound;
4. the winning contrast has no material answer-rate or stability regression and
   remains competitive after total candidate-token and GPU-hour accounting.

S3 is otherwise retained as a negative or mixed transfer result.  No intermediate
accuracy may stop, extend, or retune the run.  Passing S3 authorizes a separately
frozen full PPO/GRPO comparison; it does not automatically launch it.

## 9. Compute isolation

Use only r-saturn physical GPU1 UUID
`GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48`, an RTX PRO 6000 Blackwell with
97,887 MiB.  GPU0 and every Ollama process are out of scope and must never be
stopped or disturbed.  GPU1 may host at most one project GPU process.  Start or
resume only when the S3 main process and children are absent and GPU1 has no
other compute process.  A collision causes safe waiting, never interference.

## 10. Freeze receipt

Frozen on 2026-10-28 only after all of the following passed:

- deterministic MATH-500 scalar preparation and partition audit;
- local exact-truth/search/S3 regression suite: 123 passed, one CUDA-only test
  skipped on the CPU-only local host;
- r-saturn preflight suite: 27 passed;
- Qwen3-4B GPU1 collection and all four estimator save/load paths on two open
  development examples;
- a separate non-evidence mixed-reward implementation fixture exercising
  nonzero Actor and Value gradients, all-candidate E4 gradients, and the
  expected `ESS(E3) > ESS(E1)` ordering;
- runner shell parsing, analysis-module compilation, and local
  `git diff --check`.

The two-example smoke used only development offsets 0--1.  Its artificial
mixed-reward copy is an implementation test, not scientific evidence.  Neither
the 200-example confirmation partition nor the 120-example reserve was opened
during protocol development.  The smoke result did not alter any frozen
hyperparameter or gate.
