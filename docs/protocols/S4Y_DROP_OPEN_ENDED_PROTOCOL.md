# S4-Y DROP Open-Ended Cross-Task Confirmation Protocol

Status: preregistered before any sealed-validation model run or outcome inspection.

## Purpose and reporting commitment

S4-Y tests whether the S4 finding transfers from multiple-choice reasoning to
open-ended reading and discrete numerical reasoning.  It is triggered by the
pre-registered S4-X ceiling rule and does not replace S4-X.  S4, S4-X, and S4-Y
will all be reported regardless of sign or significance.  No third-party
inference API, model judge, task substitution, outcome-dependent extension, or
early stopping is allowed.

## Data and exposure boundary

- Dataset: official AllenAI DROP release from
  `https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip`.
- Source ZIP SHA-256:
  `39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6`.
- Official release counts: 77,409 train questions and 9,536 validation questions.
- Deterministic order domain: `cgcs-s4y-drop-v1`.
- The first 512 stable-hash-ordered official-train questions form open
  development; the other 76,897 form training.
- All 9,536 official validation questions form the sealed evaluation.  Their
  labels may be transformed mechanically into the frozen JSONL but no model
  evaluation, aggregate outcome, or per-item outcome may be inspected before
  every registered training branch is complete.
- Prepared hashes:
  - train: `86ff3c72fb7d1d22dda2f9a9ebf3883105e47c152a6471fe8ba8a96165aa902e`
  - open development: `fab647e3a708c18702a7ddcfceed4a3cc58568d0ffaeaba23015b3a3dc458010`
  - sealed validation: `3a71f2a647687a75bfdc308aa34d22af18565147ccd346f160cbf31c402306e0`
  - preparation manifest: `b0a54aeed2a31aa20bfb7623a6b2db64fb72aeb932eb2e3fa182477b61343ef3`

The pre-run exposure audit found no prior DROP dataset, cached task output, or
DROP result in the project.  The only pre-freeze model runs are the registered
24-question open-development smoke calibrations; `sealed_validation_touched` is
false in their receipt.

## Frozen answer interface and scoring

The model receives the passage and question and emits one concise reasoning step
per turn.  A valid answer is a single line `FINAL: <answer>`.  Multiple disjoint
spans use the exact separator ` || `.  No reference answer is present in the
prompt.

Exact match and token F1 use the official AllenAI DROP normalization, number
matching, and maximum one-to-one multi-span alignment.  The local scorer was
cross-checked against the official evaluator on 5,000 randomized cases with
exact agreement.  Training reward remains terminal exact match, preserving the
binary terminal-reward semantics used by S4 and S4-X.  Evaluation reports exact
match, F1, answer rate, conditional exact match/F1, steps, tokens, termination
reasons, and item-level answer-state transitions.

## Model, training, and compute matching

- Model: `Qwen/Qwen3-4B-Instruct-2507`, fixed cached revision used by S4/S4-X.
- Arithmetic: BF16; tokenizer, adapter architecture, optimizer, and value head
  are unchanged from S4-X.
- Seeds: 99801, 99802, 99803.
- Training arms per seed:
  - K1 with `E0_standard_actor_ratio`;
  - best-of-N with `E0_standard_actor_ratio`;
  - best-of-N with `E3_step_cumulative_lambda_0.75`;
  - best-of-N with `E4_extended_candidate_set`.
- Burn-in uses 128 training items at offset 0.  Round 1 uses 128 at offset 128;
  round 2 uses 128 at offset 256.  All best-of-N estimators share the round-1
  rollout, and each estimator uses its own round-2 rollout from its round-1
  policy.  This gives the same 27 training summaries and matched candidate
  budgets as S4-X.
- Collection: at most 7 steps, 80 tokens per step, 1,536 prompt tokens, 256
  preserved head tokens, sampling temperature 1.3.  Best-of-N has 2 candidates,
  epsilon 0.10, beta 1.0, and expansion rate 1.0.
- Training: learning rate 5e-6, gradient accumulation 32, LoRA rank 32 and alpha
  64, PPO clip 0.2, KL coefficient 0.02, value coefficient 0.5, partial lambda
  0.75, and no controller gradient.

## Frozen evaluation matrix

- Actors: K1/E0 and best-of-N/E0/E3/E4.
- Horizons: H8 and H16.
- Seeds: 99801, 99802, 99803.
- Matrix: 3 x 4 x 2 = 24 cells.
- Each cell evaluates all 9,536 official validation questions in the identical
  stable order, for 228,864 greedy evaluations total.
- Evaluation uses 96 tokens per step, no separate completion-token cap, 1,536
  prompt tokens, 256 preserved head tokens, and batch size 24.

The open-development smoke receipt SHA-256 is
`084a4d785bdad7e0fe8b64738e43deda3d7e406d34cb13c482ad158e8ca3b6c8`.
At H16/1,536 prompt tokens, batch 24 completed all 24 items in 49.73 model
seconds, reached 12,433 MiB peak memory, and was the fastest tested safe batch.
Open-development prompt lengths had median 370, p99 1,080, and only 1/512 above
1,536 tokens; a 2,048-item training audit had only 4 items above 1,536.

## Hardware, launcher, and stopping

The four RTX 4080 cards are bound only by these UUIDs:

1. `GPU-acdf26d2-ac77-ac48-43b3-1d99f496e89b`
2. `GPU-44417620-dcd2-43cc-31b3-402e558c8af1`
3. `GPU-10f2aae3-89f2-e258-4f2e-112f834f9970`
4. `GPU-08af3168-b02e-c10d-91d6-36f0b9287299`

Each card must have no compute PID and at least 14,000 MiB free at launch.  One
project process is allowed per card.  Evaluation cells are assigned by canonical
cell index modulo four.  The independent result root is
`results/search_policy_mismatch/s4y_drop_full`.

Only complete, schema-valid checkpoints may be resumed.  A failed cell is never
counted as evidence.  The run ends only after 27 training summaries, all 24
evaluation cells, 9,536 unique ordered source IDs per cell, and 228,864 total
evaluations pass the frozen integrity checks.  No outcome-based stopping applies.

## Frozen analysis

The primary metric is official DROP F1; exact match is the co-primary benchmark
metric.  E4 is compared with best-of-N/E0, best-of-N/E3, and K1/E0 at H8, H16,
and pooled H8+H16.  For each metric, 10,000 problem-clustered bootstrap samples
resample source IDs jointly across seeds/horizons.  Two-sided bootstrap p-values
are Holm-adjusted across the nine registered contrasts separately for F1 and
exact match.  Answer-rate, conditional metrics, per-seed effects, state
transitions, steps, tokens, and termination reasons are mandatory audits.

All artifacts and conclusions are reported whether positive, null, or negative.
Any post-freeze deviation is a protocol violation and must be disclosed rather
than silently repaired.
