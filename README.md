# When Search Is Not On-Policy

Reproducibility package for **“When Search Is Not On-Policy: Correcting
Search-Induced Policy Mismatch in Language Model RL.”**

Authors: Cyrus Tian Xia (PAII) and Shenghua He (Amazon).

This repository supports a methodology and negative-results paper. It does not
claim a universally better training recipe. The code separates the no-search
raw-Actor target from the retained composite-search target and audits joint,
marginal, partial, and retained-candidate estimators.

## What can be reproduced

The release has three levels:

1. **Quick tests:** estimator identities, exact finite-MDP behavior, schema and
   deterministic-seed checks. No GPU is required.
2. **CPU reproduction:** regenerate the exact-tree estimator analysis, rerun
   the S5 paired bootstrap from the released closed evaluation files, and
   rebuild the five paper figures from the audited terminal evidence table.
3. **Full LLM rerun:** launch the frozen Qwen3-4B S3/S5, ARC-Challenge, and DROP
   studies from public models and datasets. This tier requires GPUs and does
   not redistribute model weights or third-party datasets.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,paper]'
python scripts/verify_release.py
pytest
bash scripts/reproduce_cpu.sh
```

The CPU reproduction writes only to `reproduced/`. It never edits the released
evidence. See `REPRODUCIBILITY.md` for the claim-to-command map and
`docs/GPU_REPRODUCTION.md` for the full model runs.

## Repository layout

- `src/acbpo/`: estimator, exact-tree, task, and LLM experiment code.
- `scripts/`: frozen runners, analyzers, figure renderers, and release checks.
- `tests/`: focused estimator, exact-tree, analyzer, and determinism tests.
- `evidence/`: compact closed inputs and audited derived outputs needed for CPU
  reproduction. Large checkpoints and third-party datasets are not bundled.
- `docs/protocols/`: prospectively frozen protocols and exclusion amendments.
- `paper/`: self-contained arXiv LaTeX source.

## Integrity and scope

`MANIFEST.sha256` binds every released file except the manifest itself. Run
`python scripts/verify_release.py` before analysis. The public release preserves
failed gates and negative results. H16-W96 DROP was excluded by protocol and is
not paper evidence; its provenance remains documented in the amendment.

The S5 code uses SHA-256-derived named bootstrap seeds. This avoids Python's
process-randomized `hash()` and makes the CPU reanalysis identical across
processes and machines.

## Public release status

The canonical public repository is
[SummerRainET2008/071.open-source.p011.cgcs](https://github.com/SummerRainET2008/071.open-source.p011.cgcs).
The code is released under the MIT License. The manifest, compact evidence,
protocols, and negative or unresolved results are part of the public record.

## Citation

See `CITATION.cff`. The arXiv identifier should be added after the first public
submission.
