# Full GPU reproduction

The paper can be audited and its figures rebuilt without GPUs. This document
describes the substantially more expensive end-to-end reruns.

## S3 original matrix

Run `scripts/run_search_policy_mismatch_s3_rsaturn.sh` after preparing the
frozen GSM8K training partitions and MATH-500 scalar evaluation partition.
S3 uses three seeds, best-of-2, entropy–Value branching, and Value/PRM rollout,
with E0/E1/E3/E4 where computable. The registered sequence-product gate fails;
that failure is an archived result, not an invitation to retune.

## S5 low-update confirmation

S5 is **best-of-2 only**, with two candidates, temperature 1.3, epsilon 0.10,
beta 1.0, and full expansion. It compares K1, E0, E1, and E3 at learning rate
1e-6 over seeds 99802 and 99803, then evaluates raw Actors at H8-W96 and
H16-W96. It does not contain E4, entropy–Value, or Value/PRM conditions.

Run calibration first, freeze its selection receipt, then run confirmation:

```bash
bash scripts/run_s5_stable_calibration.sh
bash scripts/run_s5_stable_confirmation.sh
```

The historical wrappers contain machine-specific staging paths. Before a new
rerun, change only storage/model-cache locations; do not change frozen model,
data, seed, estimator, or search parameters.

## Cross-task studies

- ARC-Challenge: `scripts/run_s4x_arc_challenge_4x4080.sh`
- DROP H8-W96: `scripts/run_s4y_drop_4x4080.sh`
- H16-W96 DROP is provenance-only under the released exclusion amendment and
  must not be promoted into paper results.

## Hardware

The historical S5 confirmation ran on eight 20-GiB GPUs with one independent
worker per GPU. Lower-memory or fewer-GPU systems can serialize workers without
changing scientific configuration, at a longer wall-clock time.
