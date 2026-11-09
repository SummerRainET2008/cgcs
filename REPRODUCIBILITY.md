# Reproducibility map

Canonical repository:
[SummerRainET2008/cgcs](https://github.com/SummerRainET2008/cgcs).
The released code is licensed under MIT.

## Environment

CPU analyses require Python 3.10+ and NumPy. Figure generation additionally
requires Matplotlib. The frozen LLM runs use Qwen3-4B-Instruct-2507 at model
revision `cdbee75f17c01a7cc42f958dc650907174af0554` and the dependencies in the
`llm` and `train` extras.

## Claim-to-command map

| Paper evidence | Released input | Reproduction command | Expected output |
|---|---|---|---|
| Exact E0–E4 bias/variance/ESS matrix | `evidence/exact/ESTIMATOR_MATRIX_CONFIRMATION.json` | `python scripts/analyze_exact_truth_estimator_matrix.py ...` | `reproduced/exact/analysis.json` |
| S5 two-seed best-of-2 H8/H16 contrasts | `evidence/s5/input/` | `python scripts/analyze_s5_stable_confirmation.py ...` | `reproduced/s5/analysis.json` |
| Five manuscript figures | `evidence/paper/ICLR_EVIDENCE_TABLES.json` | figure renderers in `scripts/reproduce_cpu.sh` | `reproduced/figures/` |
| Original S3 terminal claims | compact audited analysis and archive audit in `evidence/s3/` | schema/hash verification | released JSON and manifest |
| ARC, DROP, horizon diagnostics | `evidence/cross_task/` | task analyzers or direct audit | released derived JSON |

## Statistical units

- S3 resamples problems and retains all three fixed-seed observations for each
  sampled problem; registered family-wise tests use Holm adjustment.
- S5 resamples problems and retains both fixed-seed observations. The four
  E1/E3-versus-E0 H8/H16 intervals are descriptive and unadjusted for
  multiplicity.
- The release analyzers derive bootstrap seeds from SHA-256 names rather than
  Python's randomized process hash.

## Determinism

The exact-tree and S5 analyses should be byte-identical to the expected files
under `evidence/*/expected/`. Matplotlib output can vary at the PDF metadata or
font-subsetting level across library versions; scientific values are validated
from the evidence-table JSON before rendering.

## Non-bundled material

Model weights, LoRA adapters, optimizer states, and third-party datasets are
not redistributed. The full-run protocols record model revisions, seeds,
dataset partitions, and hashes. Users must obtain datasets and models under
their original licenses.
