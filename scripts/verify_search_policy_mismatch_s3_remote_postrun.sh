#!/usr/bin/env bash
set -euo pipefail

# Administrative terminal-only verification for the Torch-dependent tests that
# are skipped on the local macOS host. This script never starts, resumes, or
# modifies the frozen S3 experiment.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="${1:-$project_root/results/search_policy_mismatch/llm_s3}"
cache_root="${TMPDIR:-/tmp}/uv-cache-cgcs-s3-remote-postrun"
pycache_root="${TMPDIR:-/tmp}/cgcs-s3-remote-postrun-pycache"

cd "$project_root"

for marker in "$root/EVIDENCE_COMPLETE" "$root/COMPLETE"; do
  if [[ ! -f "$marker" ]]; then
    echo "S3 terminal marker is missing: $marker" >&2
    exit 2
  fi
done

if [[ -f "$root/pid" ]]; then
  main_pid="$(tr -d '[:space:]' < "$root/pid")"
  if [[ "$main_pid" =~ ^[0-9]+$ ]] && kill -0 "$main_pid" 2>/dev/null; then
    echo "S3 runner PID $main_pid is still live; post-run tests are forbidden." >&2
    exit 2
  fi
fi
if pgrep -f \
  '[r]un_search_policy_mismatch_s3_rsaturn\.sh|[a]cbpo\.math_(search_policy_mismatch|joint_value_ppo)' \
  >/dev/null; then
  echo "An S3 runner or project child process is still live; post-run tests are forbidden." >&2
  exit 2
fi

log="$root/REMOTE_POSTRUN_TESTS.log"
junit="$root/REMOTE_POSTRUN_TESTS.xml"

CUDA_VISIBLE_DEVICES="" \
PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONPYCACHEPREFIX="$pycache_root" \
UV_CACHE_DIR="$cache_root" \
  .venv/bin/python -m pytest -q --junitxml="$junit" \
  tests/test_math_search_policy_mismatch.py \
  tests/test_cgcs_ppo.py \
  tests/test_math_critic_experiment.py | tee "$log"

.venv/bin/python -c '
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag.endswith("testsuite") else list(root.findall(".//testsuite"))
if not suites:
    raise SystemExit("JUnit receipt contains no test suites")
totals = {
    key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
    for key in ("tests", "failures", "errors", "skipped")
}
if totals["tests"] <= 0 or any(totals[key] for key in ("failures", "errors", "skipped")):
    raise SystemExit(f"remote post-run Torch tests did not fully pass: {totals}")
' "$junit"

touch "$root/REMOTE_POSTRUN_TESTS_COMPLETE"
echo "S3 remote post-run Torch verification passed without failures or skips."
