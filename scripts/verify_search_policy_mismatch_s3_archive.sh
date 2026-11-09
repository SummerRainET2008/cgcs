#!/usr/bin/env bash
set -euo pipefail

# Post-run verification only. This script never starts, resumes, or modifies the
# frozen S3 experiment. Run it locally after the complete remote result tree has
# been synchronized into the repository.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:-$project_root/results/search_policy_mismatch/llm_s3}"
cache_root="${TMPDIR:-/tmp}/uv-cache-cgcs-s3-archive"
pycache_root="${TMPDIR:-/tmp}/cgcs-s3-archive-pycache"
matplotlib_root="${TMPDIR:-/tmp}/cgcs-s3-matplotlib"
reanalyze_root="$(mktemp -d "${TMPDIR:-/tmp}/cgcs-s3-reanalysis.XXXXXX")"

cleanup() {
  rm -rf -- "$reanalyze_root"
}
trap cleanup EXIT

cd "$project_root"

required=(
  "$root/EVIDENCE_COMPLETE"
  "$root/COMPLETE"
  "$root/REMOTE_POSTRUN_TESTS_COMPLETE"
  "$root/REMOTE_POSTRUN_TESTS.log"
  "$root/REMOTE_POSTRUN_TESTS.xml"
  "$root/S3_LLM_TRANSFER_ANALYSIS.json"
  "$root/S3_LLM_TRANSFER_REPORT.md"
  "$root/S3_MANUAL_COMPUTE_REVIEW.json"
)
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "S3 archive is incomplete; missing: $path" >&2
    exit 2
  fi
done

# Recompute the frozen analysis into a temporary directory and require exact
# byte-for-byte agreement with the synchronized remote outputs. This proves
# that the report is reproducible from the archived artifacts without
# modifying either the frozen result tree or the running experiment.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/analyze_search_policy_mismatch_s3.py \
  --root "$root" \
  --project-root "$project_root" \
  --output-json "$reanalyze_root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --output-report "$reanalyze_root/S3_LLM_TRANSFER_REPORT.md"
cmp -s \
  "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  "$reanalyze_root/S3_LLM_TRANSFER_ANALYSIS.json" || {
  echo "S3 frozen analysis is not reproducible from the synchronized artifacts." >&2
  exit 3
}
cmp -s \
  "$root/S3_LLM_TRANSFER_REPORT.md" \
  "$reanalyze_root/S3_LLM_TRANSFER_REPORT.md" || {
  echo "S3 frozen report is not reproducible from the synchronized artifacts." >&2
  exit 3
}

# Keep the preregistered analyzer byte-identical, then add level/subject
# diagnostics as an explicitly descriptive and independently reproducible
# post-run artifact.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/analyze_search_policy_mismatch_s3_strata.py \
  --root "$root" \
  --project-root "$project_root" \
  --output-json "$root/S3_MATH500_STRATA.json" \
  --output-markdown "$root/S3_MATH500_STRATA.md"
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/analyze_search_policy_mismatch_s3_strata.py \
  --root "$root" \
  --project-root "$project_root" \
  --output-json "$reanalyze_root/S3_MATH500_STRATA.json" \
  --output-markdown "$reanalyze_root/S3_MATH500_STRATA.md"
cmp -s \
  "$root/S3_MATH500_STRATA.json" \
  "$reanalyze_root/S3_MATH500_STRATA.json" || {
  echo "S3 MATH-500 stratified JSON is not deterministically reproducible." >&2
  exit 3
}
cmp -s \
  "$root/S3_MATH500_STRATA.md" \
  "$reanalyze_root/S3_MATH500_STRATA.md" || {
  echo "S3 MATH-500 stratified Markdown is not deterministically reproducible." >&2
  exit 3
}

UV_CACHE_DIR="$cache_root" uv run python \
  scripts/audit_search_policy_mismatch_s3_archive.py \
  --root "$root" \
  --analysis-json "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --report "$root/S3_LLM_TRANSFER_REPORT.md" \
  --project-root "$project_root" \
  --compute-review "$root/S3_MANUAL_COMPUTE_REVIEW.json" \
  --output "$root/S3_ARCHIVE_AUDIT.json" \
  --output-markdown "$root/S3_ARCHIVE_AUDIT.md"

# The administrative audit is itself evidence-bearing: it contains the
# independent 81/39/36 completeness matrix, GPU-session receipt, recomputed
# evaluation metrics, paired contrasts, and archive hashes.  Re-run it against
# the identical synchronized inputs and require byte identity before any paper
# renderer may consume it.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/audit_search_policy_mismatch_s3_archive.py \
  --root "$root" \
  --analysis-json "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --report "$root/S3_LLM_TRANSFER_REPORT.md" \
  --project-root "$project_root" \
  --compute-review "$root/S3_MANUAL_COMPUTE_REVIEW.json" \
  --output "$reanalyze_root/S3_ARCHIVE_AUDIT.json" \
  --output-markdown "$reanalyze_root/S3_ARCHIVE_AUDIT.md"
cmp -s \
  "$root/S3_ARCHIVE_AUDIT.json" \
  "$reanalyze_root/S3_ARCHIVE_AUDIT.json" || {
  echo "S3 independent archive-audit JSON is not deterministically reproducible." >&2
  exit 3
}
cmp -s \
  "$root/S3_ARCHIVE_AUDIT.md" \
  "$reanalyze_root/S3_ARCHIVE_AUDIT.md" || {
  echo "S3 independent archive-audit Markdown is not deterministically reproducible." >&2
  exit 3
}

# Render the three evidence-locked manuscript tables only after both the frozen
# analyzer and the independent archive audit have passed. The renderer rejects
# partial S3 matrices and never reads an in-progress result tree.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/render_search_policy_mismatch_iclr_tables.py \
  --mechanism-analysis \
    results/math_value_ppo/joint_value_mechanism_replication/analysis.json \
  --gate-confirmation results/search_policy_mismatch/GATE_CONFIRMATION.json \
  --critic-confirmation results/search_policy_mismatch/CRITIC_CONFIRMATION.json \
  --exact-confirmation \
    results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION.json \
  --exact-analysis \
    results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION_ANALYSIS.json \
  --s3-analysis "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --s3-audit "$root/S3_ARCHIVE_AUDIT.json" \
  --output-json "$root/ICLR_EVIDENCE_TABLES.json" \
  --output-markdown "$root/ICLR_EVIDENCE_TABLES.md"

UV_CACHE_DIR="$cache_root" uv run python \
  scripts/render_search_policy_mismatch_iclr_tables.py \
  --mechanism-analysis \
    results/math_value_ppo/joint_value_mechanism_replication/analysis.json \
  --gate-confirmation results/search_policy_mismatch/GATE_CONFIRMATION.json \
  --critic-confirmation results/search_policy_mismatch/CRITIC_CONFIRMATION.json \
  --exact-confirmation \
    results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION.json \
  --exact-analysis \
    results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION_ANALYSIS.json \
  --s3-analysis "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --s3-audit "$root/S3_ARCHIVE_AUDIT.json" \
  --output-json "$reanalyze_root/ICLR_EVIDENCE_TABLES.json" \
  --output-markdown "$reanalyze_root/ICLR_EVIDENCE_TABLES.md"
cmp -s \
  "$root/ICLR_EVIDENCE_TABLES.json" \
  "$reanalyze_root/ICLR_EVIDENCE_TABLES.json" || {
  echo "ICLR evidence-table JSON is not deterministically reproducible." >&2
  exit 3
}
cmp -s \
  "$root/ICLR_EVIDENCE_TABLES.md" \
  "$reanalyze_root/ICLR_EVIDENCE_TABLES.md" || {
  echo "ICLR evidence-table Markdown is not deterministically reproducible." >&2
  exit 3
}

# Render all five preregistered main-paper figures from the terminal evidence
# tables. PNG and vector PDF outputs, plus their SHA-256 manifest, must be
# byte-identical across two independent renders.
MPLCONFIGDIR="$matplotlib_root" UV_CACHE_DIR="$cache_root" \
  uv run --extra paper python \
  scripts/render_search_policy_mismatch_iclr_figures.py \
  --evidence-tables "$root/ICLR_EVIDENCE_TABLES.json" \
  --output-dir "$root/ICLR_FIGURES" \
  --manifest "$root/ICLR_FIGURES_MANIFEST.json"

MPLCONFIGDIR="$matplotlib_root" UV_CACHE_DIR="$cache_root" \
  uv run --extra paper python \
  scripts/render_search_policy_mismatch_iclr_figures.py \
  --evidence-tables "$root/ICLR_EVIDENCE_TABLES.json" \
  --output-dir "$reanalyze_root/ICLR_FIGURES" \
  --manifest "$reanalyze_root/ICLR_FIGURES_MANIFEST.json"
cmp -s \
  "$root/ICLR_FIGURES_MANIFEST.json" \
  "$reanalyze_root/ICLR_FIGURES_MANIFEST.json" || {
  echo "ICLR figure manifest is not deterministically reproducible." >&2
  exit 3
}
for stem in \
  figure_1_estimands_and_data_generation \
  figure_2_exact_bias_variance_frontier \
  figure_3_when_search_shift_helps \
  figure_4_no_search_llm_transfer \
  figure_5_retained_search_objective_and_cost; do
  for extension in png pdf; do
    cmp -s \
      "$root/ICLR_FIGURES/$stem.$extension" \
      "$reanalyze_root/ICLR_FIGURES/$stem.$extension" || {
      echo "ICLR figure is not deterministically reproducible: $stem.$extension" >&2
      exit 3
    }
  done
done

# Render the manuscript's terminal numerical block from the same evidence table
# twice and require byte identity. The paper inputs this file directly from the
# synchronized result root, so no hand-copied accuracy can enter the manuscript.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/render_search_policy_mismatch_iclr_results_tex.py \
  --evidence-tables "$root/ICLR_EVIDENCE_TABLES.json" \
  --output-tex "$root/ICLR_S3_TERMINAL_RESULTS.tex" \
  --manifest "$root/ICLR_S3_TERMINAL_RESULTS_MANIFEST.json"

UV_CACHE_DIR="$cache_root" uv run python \
  scripts/render_search_policy_mismatch_iclr_results_tex.py \
  --evidence-tables "$root/ICLR_EVIDENCE_TABLES.json" \
  --output-tex "$reanalyze_root/ICLR_S3_TERMINAL_RESULTS.tex" \
  --manifest "$reanalyze_root/ICLR_S3_TERMINAL_RESULTS_MANIFEST.json"
cmp -s \
  "$root/ICLR_S3_TERMINAL_RESULTS.tex" \
  "$reanalyze_root/ICLR_S3_TERMINAL_RESULTS.tex" || {
  echo "ICLR terminal-results TeX is not deterministically reproducible." >&2
  exit 3
}
cmp -s \
  "$root/ICLR_S3_TERMINAL_RESULTS_MANIFEST.json" \
  "$reanalyze_root/ICLR_S3_TERMINAL_RESULTS_MANIFEST.json" || {
  echo "ICLR terminal-results manifest is not deterministically reproducible." >&2
  exit 3
}

# The terminal numerical block and figures must be inspected in the complete
# compiled manuscript.  This administrative receipt is deliberately checked
# only after both manifests have been reproduced above, so a first verifier run
# can create the deterministic inputs; after compilation and visual review, a
# second verifier run must validate the hash-bound PASS receipt.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/record_search_policy_mismatch_iclr_manuscript_review.py verify \
  --main-tex paper/search_policy_mismatch_iclr2027/main.tex \
  --compiled-pdf paper/search_policy_mismatch_iclr2027/main.pdf \
  --terminal-results-manifest "$root/ICLR_S3_TERMINAL_RESULTS_MANIFEST.json" \
  --figure-manifest "$root/ICLR_FIGURES_MANIFEST.json" \
  --receipt "$root/ICLR_MANUSCRIPT_REVIEW.json"

# Seal every evidence-bearing terminal artifact, its generators, the complete
# manuscript source/PDF, and the two human-review receipts into one final
# deterministic manifest.  The manifest intentionally excludes itself.
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/build_search_policy_mismatch_s3_final_manifest.py \
  --project-root "$project_root" \
  --root "$root" \
  --output "$root/S3_FINAL_ARCHIVE_MANIFEST.json"
UV_CACHE_DIR="$cache_root" uv run python \
  scripts/build_search_policy_mismatch_s3_final_manifest.py \
  --project-root "$project_root" \
  --root "$root" \
  --output "$reanalyze_root/S3_FINAL_ARCHIVE_MANIFEST.json"
cmp -s \
  "$root/S3_FINAL_ARCHIVE_MANIFEST.json" \
  "$reanalyze_root/S3_FINAL_ARCHIVE_MANIFEST.json" || {
  echo "S3 final archive manifest is not deterministically reproducible." >&2
  exit 3
}

MPLCONFIGDIR="$matplotlib_root" UV_CACHE_DIR="$cache_root" \
  uv run --extra dev --extra paper pytest -q \
  tests/test_analyze_search_policy_mismatch_s3.py \
  tests/test_analyze_search_policy_mismatch_s3_strata.py \
  tests/test_audit_search_policy_mismatch_s3_archive.py \
  tests/test_render_search_policy_mismatch_iclr_tables.py \
  tests/test_render_search_policy_mismatch_iclr_figures.py \
  tests/test_render_search_policy_mismatch_iclr_results_tex.py \
  tests/test_search_policy_mismatch_iclr_manuscript.py
MPLCONFIGDIR="$matplotlib_root" UV_CACHE_DIR="$cache_root" \
  uv run --extra dev --extra paper pytest -q

bash -n scripts/run_search_policy_mismatch_s3_rsaturn.sh
bash -n scripts/verify_search_policy_mismatch_s3_archive.sh
PYTHONPYCACHEPREFIX="$pycache_root" UV_CACHE_DIR="$cache_root" \
  uv run python -m compileall -q src scripts tests
git diff --check

echo "S3 local archive verification passed."
echo "The independent audit report contains the final SHA-256 receipt."
echo "MATH-500 level/subject diagnostics were reproduced byte-for-byte."
echo "The independent audit JSON and Markdown were reproduced byte-for-byte."
echo "The manual compute review was explicit, hash-bound, and incorporated."
echo "The evidence-locked ICLR tables were rendered from terminal artifacts."
echo "All five preregistered ICLR figures were rendered byte-reproducibly."
echo "The ICLR terminal-results TeX was rendered byte-reproducibly."
echo "The complete compiled manuscript passed a hash-bound visual review."
echo "The complete terminal evidence archive was sealed deterministically."
echo "README/STAGE_GATES terminal interpretation was validated and sealed."
