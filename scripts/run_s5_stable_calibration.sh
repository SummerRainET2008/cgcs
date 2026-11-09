#!/usr/bin/env bash
set -euo pipefail
p=/workspace/p011.cgcs; py="$p/codes/.venv/bin/python"; r="$p/experiments/results/search_policy_mismatch/s5_stable_confirmation"; m=Qwen/Qwen3-4B-Instruct-2507
prior="$p/experiments/results/search_policy_mismatch/llm_s3/s99801/burnin/train"; k="$p/experiments/results/search_policy_mismatch/llm_s3/s99801/round1/k1/E0_standard_actor_ratio/rollout.json"; s="$p/experiments/results/search_policy_mismatch/llm_s3/s99801/round1/best_of_n/shared_rollout.json"
export PYTHONPATH="$p/codes/src" HF_HOME=/workspace/hf_cache TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$r/calibration/logs"
arms=(k1 search_e0 search_e1 search_e3); est=(E0_standard_actor_ratio E0_standard_actor_ratio E1_trajectory_propensity E3_step_cumulative_lambda_0.75); rolls=("$k" "$s" "$s" "$s"); lrs=(1e-6 1e-6 1e-6 1e-6 2e-6 2e-6 2e-6 2e-6); pids=()
for g in 0 1 2 3 4 5 6 7; do i=$((g%4)); lr="${lrs[$g]}"; a="${arms[$i]}"; e="${est[$i]}"; ro="${rolls[$i]}"; o="$r/calibration/lr_${lr}/${a}/train"; mkdir -p "$o"; (export CUDA_VISIBLE_DEVICES="$g"; "$py" -m acbpo.math_search_policy_mismatch train --rollout "$ro" --output-dir "$o" --model "$m" --initial-adapter "$prior/adapter" --initial-value-head "$prior/value_head.pt" --estimator "$e" --partial-lambda .75 --seed 99801 --learning-rate "$lr" --gradient-accumulation 32 --lora-rank 32 --lora-alpha 64 --ppo-clip .2 --kl-coefficient .02 --value-coefficient .5 --max-grad-norm 1) >"$r/calibration/logs/gpu${g}_${a}_lr${lr}.log" 2>&1 & pids+=("$!"); done
st=0; for x in "${pids[@]}"; do wait "$x" || st=1; done
"$py" "$p/codes/scripts/select_s5_stable_lr.py" --root "$r/calibration" --output "$r/calibration/SELECTION.json"; exit "$st"
