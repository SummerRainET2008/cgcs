# S4 frozen inference-horizon and answer-rate sensitivity protocol

Status: **FROZEN FOR ONE INFERENCE-ONLY RUN**

Freeze date: 2026-11-02 (America/Los_Angeles).  This stage is a post-S3
robustness diagnostic.  It cannot modify the terminal S3 result, repair the
failed S3 stability gate, select a training estimator, or authorize larger
PPO/GRPO training.

## Authorization and prerequisites

The user explicitly authorized execution after reviewing the proposed
8/12/16-step sensitivity plan.  S3 is terminally archived and its final manifest
has SHA-256
`8e9d89e31f07153857e434c56d10d47a86f8b4fc71997251ab7d8f393362b154`.
The run may use only r-saturn physical GPU1 UUID
`GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48`, after verifying that it has no
compute process.  GPU0 is out of scope for this frozen run because the freeze
preflight observed an Ollama compute process there.  No Ollama process may be
stopped, signalled, or otherwise disturbed.

## Question and estimand

S3 evaluates no-search Actors with at most eight reasoning Actions, each with a
96-token ceiling, and a strict scalar `FINAL:` extractor.  S4 asks:

1. Is the low answer rate caused by the eight-Action horizon?
2. Does a longer horizon improve overall correctness, or merely produce more
   parseable final answers?
3. Does any improvement survive a matched total completion-token ceiling?

The decomposition

\[
P(\text{correct}) = P(\text{valid final})
P(\text{correct}\mid\text{valid final})
\]

is mandatory.  A finalization gain is not automatically a reasoning-quality
gain.

## Immutable models and data

- Evaluate all 39 frozen S3 raw-K1 Actors: three K1 controls and all 36
  family-by-estimator-by-seed Actors.
- No training, adapter change, checkpoint selection, threshold fitting, or
  search is performed.
- Base model: `Qwen/Qwen3-4B-Instruct-2507`, locally cached and offline.
- Dataset: `data/math500_scalar.jsonl`, SHA-256
  `471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7`.
- Dataset manifest: `data/math500_scalar_manifest.json`, SHA-256
  `ec96c0b63eb0dadb6132e12961c2efee85a11b9df1e92ca7a29a9933d560008d`.
- S4 uses reserve offset 248, count 100, ordered IDs 248--347.  These are
  disjoint from development IDs 0--47 and S3 confirmation IDs 48--247.  Reserve
  IDs 348--367 remain unused.
- Greedy K1 decoding, prompt format, maximum prompt tokens 768, retained head
  tokens 192, fixed seed 101901, and strict S3 scalar verifier remain unchanged.

Retained-composite systems are not regenerated in S4.  They are a different
estimand and would multiply search compute.  A composite-horizon study, if
needed, must be separately frozen after S4 without changing this raw-Actor
diagnostic.

## Frozen generation matrix

Every Actor runs on the same 100 ordered problems under all five conditions:

| Condition | `max_steps` | `step_tokens` | total completion-token cap | Role |
|---|---:|---:|---:|---|
| H8 | 8 | 96 | none | Fresh-partition S3-horizon reference |
| H12 | 12 | 96 | none | Natural-compute 12-step horizon |
| H16 | 16 | 96 | none | Natural-compute 16-step horizon |
| C12 | 12 | 96 | 768 | Twelve-step matched-output-budget diagnostic |
| C16 | 16 | 96 | 768 | Sixteen-step matched-output-budget diagnostic |

H8 has the same theoretical 768-token ceiling as C12/C16.  The matched
conditions permit more, shorter Actions but cannot generate more than 768
completion tokens.  Attention/prompt FLOPs are reported and are not claimed to
be exactly equal merely because output-token ceilings match.

The complete matrix contains 39 Actors x 5 conditions x 100 problems = 19,500
greedy evaluations.  Conditions, Actors, or examples may not be removed based
on interim results.  Each output is resumable only by exact Actor, condition,
data, seed, and generation configuration.

## Offline extractor audit

Without regenerating tokens, reparse every transcript from all 75 terminal S3
evaluations.  Preserve the strict result and add one secondary anchored grammar:

- inspect only the last nonempty line;
- accept a case-insensitive prefix in the closed set `FINAL`, `FINAL ANSWER`,
  or `ANSWER`, followed by `:` or `=`;
- require one terminal scalar expression and reuse frozen scalar
  normalization/correctness;
- reject prose-only, multiple-number, intermediate-equation, and unanchored
  matches.

The secondary parser is descriptive.  It never replaces S3 or S4 strict
accuracy.

## Registered outputs and inference

For every Actor and condition report overall accuracy, strict answer rate,
accuracy conditional on an answer, step count, completion tokens, termination
reason, wall time, and GPU-hours.  For H12/H16/C12/C16 versus H8 report paired
wins/losses/ties and transitions among correct, incorrect-with-valid-final, and
unanswered.

Use 10,000 problem-clustered bootstrap resamples over the 100 common problem
IDs, keeping all seed observations for one problem together.  Report 95%
intervals for changes in accuracy, answer rate, conditional accuracy, steps,
and completion tokens.  Apply Holm correction separately within each metric and
contrast across the 12 family-by-estimator cells.  K1-control contrasts are
descriptive references.  All seeds remain visible.

## Horizon selection rule

H12 is preferred over H8 only if its pooled raw-Actor comparison has:

1. answer-rate improvement at least +5 percentage points with 95% interval
   lower bound above zero;
2. accuracy interval lower bound at least zero;
3. conditional-accuracy interval lower bound at least -2 percentage points.

If H12 fails, apply the same rule to H16.  Choose the shortest passing natural
horizon; otherwise retain H8.  C12/C16 diagnose whether a gain survives a
matched output-token ceiling and never override this rule.  No result from this
100-problem reserve may be used to tune another S4 condition.

## Interpretation and stage boundary

- Higher answer rate with unchanged conditional accuracy is a
  completion/finalization result.
- Higher answer rate with worse conditional accuracy is a coverage--quality
  tradeoff, not an unqualified improvement.
- A gain only in H12/H16 but not C12/C16 is compute-dependent.
- Stable estimator ordering supports robustness only on this model and reserve
  slice.
- S4 can select a frozen inference horizon for a future OPD/OSPD-style transfer
  protocol.  It cannot promote any existing S3 estimator or launch training.

The unique launch script, protocol digest, code digests, run manifest, GPU
session receipt, complete analysis, and archive manifest are required before S4
can be declared complete.
