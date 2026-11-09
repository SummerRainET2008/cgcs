# Search-induced policy mismatch data-exposure ledger

Status: administrative reproducibility record.  This file summarizes frozen
protocols and completed/running artifacts; it does not amend a protocol, open a
sealed split, select a method, or authorize a later stage.

## Exact-truth synthetic study

| Partition | Identity | Allowed use | Exposure status |
| --- | --- | --- | --- |
| Smoke development | MDP seeds 1000--1002 | Mathematical/software recovery checks only | Exposed before confirmation |
| Gate/model development | MDP seeds 1000--1019 | Fit gates and select preregistered estimator settings | Exposed before confirmation |
| Disjoint development test | MDP seeds 1020--1029 | Test development-frozen features and settings | Exposed before confirmation |
| Frozen confirmation | 30 SHA-derived committed MDP seeds, with ten common-random-number streams per MDP | S1/S2, gate/search-family, Critic-quality, and E0--E4 confirmation | Opened only after protocol and seed commitment; complete |

Development seeds 1000--1029 are permanently excluded from the confirmation
set.  Monte Carlo streams are repeated measurements within an MDP and are not
treated as independent problems.  Confirmation intervals therefore cluster by
MDP.

The frozen `EXACT_TRUTH_PROTOCOL.md` SHA-256 is
`5bfa19bd76380764bb83e241cd7495e6e599e053fd2ff9f5afa2689a7b4711c3`,
which exactly matches `CONFIRMATION_SEEDS.json`.  The manifest contains 30
unique MDP seeds, 30 stream-map entries, and exactly ten streams per entry.  It
also records that common random numbers are shared across methods and that no
estimator or search-family label entered seed derivation.

Authoritative sources:

- `EXACT_TRUTH_PROTOCOL.md`
- `CONFIRMATION_SEEDS.json`
- `FREEZE_RECEIPT.md`
- `S1_PRIMARY_REPORT.md`
- `S2_BIAS_VARIANCE_REPORT.md`
- `ESTIMATOR_MATRIX_CONFIRMATION_REPORT.md`

## S3 controlled LLM transfer

### Training source

All three training seeds use the same GSM8K training-source ranges.  These
problems train the Actor--Value branches and are not independent generalization
evidence.

| Role | GSM8K train range | Count | Exposure status |
| --- | --- | ---: | --- |
| Common burn-in | offset 6500 | 48 | Training input |
| Adaptive round 1 | offset 6548 | 48 | Training input |
| Adaptive round 2 | offset 6596 | 48 | Training input |

Sampling/optimizer seeds are 99801, 99802, and 99803.  Repeating the same
problem ranges across seeds measures training/randomness variation; it does not
create 432 independent training problems.

### Independent evaluation source

The evaluation source is the deterministic scalar-answer subset of
`HuggingFaceH4/MATH-500` revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`.  The prepared 368-row artifact is
`data/math500_scalar.jsonl`, SHA-256
`471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7`.
A fixed SHA-256 rank of source unique IDs defines the following immutable split.

| Partition | Prepared offset/count | Allowed use | Exposure status |
| --- | --- | --- | --- |
| Open development | 0/48 | Implementation checks and development smoke only | Open |
| Sealed S3 confirmation | 48/200 | Frozen raw-K1 and retained-composite comparisons | Opened only after all training arms completed |
| Reserved follow-up pool | 248/120 | Future independently frozen follow-up only | Manifest and boundary-row source metadata checked; question/answer content and model outcomes remain unexposed |

The two-example implementation smoke used only open-development offsets 0--1.
Every S3 confirmation evaluation must contain the same 200 problem IDs in the
frozen manifest order.  Seed observations on one problem are paired repeated
measurements, so the primary bootstrap clusters by problem rather than counting
each seed as a new problem.

### Reserve metadata receipt

On 2026-10-29, while drafting a non-authorizing post-S3 answer-rate sensitivity
design, the prepared-artifact length, manifest partition counts, and the
`source` metadata at reserve boundary offsets 248, 347, 348, and 367 were
checked.  The commands projected only JSON keys and `source` metadata; no
reserve `question` or `answer` field was displayed, evaluated, summarized, or
used to select a method or condition, and no model was run on the reserve.
This metadata receipt corrects the misleading implication that a file named
MATH-500 necessarily contains 500 scalar-answer rows: the frozen scalar filter
retains exactly 368 rows, of which 120 are reserve.

The draft at
`S3_POSTRUN_ANSWER_RATE_SENSITIVITY_DESIGN.md` proposes offsets 248--347 for a
future 100-problem inference-only diagnostic and leaves offsets 348--367
unused.  It is not frozen and does not open either subset.  Any later use must
first bind the data and manifest hashes in a separate protocol and add a new
exposure receipt; the current S3 estimand and gate remain unchanged.

### In-run seal-order receipt

A read-only remote filesystem audit on 2026-10-29 found all 81 required
training summaries before inspecting the first confirmation artifact.  The
latest training summary was
`s99803/round2/value_value/E4_extended_candidate_set/train/summary.json` with
mtime-ns `1793280718441392770`.  The earliest confirmation file was
`s99801/eval/raw_k1/k1_E0_standard_actor_ratio.json` with mtime-ns
`1793284041266903649`, strictly later than the final training summary.  At that
snapshot there were three raw-K1 files and two retained-composite files, one of
which was still partial.  This directly confirms the preregistered ordering
that the sealed 200-row partition was opened only after every training branch
completed; final archival must recheck the ordering and exact 39/36 evaluation
counts.

At a later in-run snapshot, the frozen `load_gsm8k` loader was used to compare
all 818 generated evaluation records across four complete files and one partial
file against the prepared artifact.  Every example ID, question, normalized
scalar reference answer, and row position matched, and every referenced source
row carried `s3_partition=confirmation`.  There were zero identity or partition
errors.  This partial audit must be repeated over all 75 completed evaluation
files during final archival.

Authoritative sources:

- `S3_LLM_TRANSFER_PROTOCOL.md`
- `S3_POSTRUN_ARCHIVE_CHECKLIST.md`
- `data/math500_scalar.jsonl`
- the completed S3 confirmation manifest and analysis (after the running stage
  finishes)

## Claim boundary

- GSM8K offsets 6500--6643 support training comparisons, not an independent
  generalization claim.
- The 48 open MATH-500 rows support implementation/development checks, not a
  sealed confirmation claim.
- The 200 sealed rows support the preregistered S3 comparisons only after the
  complete matrix and clustered analysis pass validation.
- The 120 reserve rows are not evidence for S3.  Their question/answer content
  and any model outcomes must not be inspected or used for tuning unless a
  separate protocol is frozen and recorded here first.
- Code-generation and delayed-reward tool/agent benchmarks remain future Stage D
  extensions.  No dataset for either extension is frozen by this ledger.
