#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python3}
export PYTHONPATH="$root/src"
out="$root/reproduced"
mkdir -p "$out/exact" "$out/s5" "$out/figures"

"$python_bin" "$root/scripts/verify_release.py"

"$python_bin" "$root/scripts/analyze_exact_truth_estimator_matrix.py" \
  --input "$root/evidence/exact/ESTIMATOR_MATRIX_CONFIRMATION.json" \
  --output "$out/exact/analysis.json"
cmp "$out/exact/analysis.json" "$root/evidence/exact/expected/ESTIMATOR_MATRIX_CONFIRMATION_ANALYSIS.json"

"$python_bin" "$root/scripts/analyze_s5_stable_confirmation.py" \
  --root "$root/evidence/s5/input/confirmation" \
  --output-json "$out/s5/analysis.json" \
  --output-report "$out/s5/analysis.md"
cmp "$out/s5/analysis.json" "$root/evidence/s5/expected/S5_STABLE_CONFIRMATION_ANALYSIS.json"

"$python_bin" "$root/scripts/render_search_policy_mismatch_iclr_figures.py" \
  --evidence-tables "$root/evidence/paper/ICLR_EVIDENCE_TABLES.json" \
  --output-dir "$out/figures" \
  --manifest "$out/figures/MANIFEST.base.json"
"$python_bin" "$root/scripts/render_search_policy_mismatch_iclr_v4_figures.py" \
  --evidence-tables "$root/evidence/paper/ICLR_EVIDENCE_TABLES.json" \
  --output-dir "$out/figures" \
  --manifest "$out/figures/MANIFEST.v4.json"

echo "CPU_REPRODUCTION_PASS"
