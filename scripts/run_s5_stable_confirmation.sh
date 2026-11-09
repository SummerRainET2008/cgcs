#!/usr/bin/env bash
set -euo pipefail
p=/workspace/p011.cgcs; py="$p/codes/.venv/bin/python"; r="$p/experiments/results/search_policy_mismatch/s5_stable_confirmation"; m=Qwen/Qwen3-4B-Instruct-2507
lr=$("$py" -c 'import json;print(json.load(open("/workspace/p011.cgcs/experiments/results/search_policy_mismatch/s5_stable_confirmation/calibration/SELECTION.json"))["selected_learning_rate"])')
[[ "$lr" == "1e-06" ]] || { echo "unexpected selected LR $lr" >&2; exit 1; }
export PYTHONPATH="$p/codes/src" HF_HOME=/workspace/hf_cache TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
arms=(k1 search_e0 search_e1 search_e3); est=(E0_standard_actor_ratio E0_standard_actor_ratio E1_trajectory_propensity E3_step_cumulative_lambda_0.75); seeds=(99802 99802 99802 99802 99803 99803 99803 99803); pids=()
worker(){
 g=$1; seed=$2; i=$3; a="${arms[$i]}"; e="${est[$i]}"; prior="$p/experiments/results/search_policy_mismatch/llm_s3/s${seed}/burnin/train"; base="$r/confirmation/s${seed}/${a}"; mkdir -p "$base"
 if [[ "$a" == k1 ]]; then ro="$p/experiments/results/search_policy_mismatch/llm_s3/s${seed}/round1/k1/E0_standard_actor_ratio/rollout.json"; fam=k1; else ro="$p/experiments/results/search_policy_mismatch/llm_s3/s${seed}/round1/best_of_n/shared_rollout.json"; fam=best_of_n; fi
 t1="$base/round1/train"; mkdir -p "$t1"; "$py" -m acbpo.math_search_policy_mismatch train --rollout "$ro" --output-dir "$t1" --model "$m" --initial-adapter "$prior/adapter" --initial-value-head "$prior/value_head.pt" --estimator "$e" --partial-lambda .75 --seed "$seed" --learning-rate "$lr" --gradient-accumulation 32 --lora-rank 32 --lora-alpha 64 --ppo-clip .2 --kl-coefficient .02 --value-coefficient .5 --max-grad-norm 1
 ro2="$base/round2/rollout.json"; mkdir -p "$base/round2"
 if [[ "$fam" == k1 ]]; then "$py" -m acbpo.math_joint_value_ppo collect --data "$p/data/gsm8k/train.jsonl" --output "$ro2" --model "$m" --actor-adapter "$t1/adapter" --value-head "$t1/value_head.pt" --arm k1 --offset 6596 --count 48 --seed "$seed" --max-steps 7 --step-tokens 80 --sampling-temperature 1.3; else "$py" -m acbpo.math_joint_value_ppo collect --data "$p/data/gsm8k/train.jsonl" --output "$ro2" --model "$m" --actor-adapter "$t1/adapter" --value-head "$t1/value_head.pt" --arm best_of_n --offset 6596 --count 48 --seed "$seed" --max-steps 7 --step-tokens 80 --candidates 2 --sampling-temperature 1.3 --epsilon .10 --beta 1 --expansion-rate 1; fi
 t2="$base/round2/train"; mkdir -p "$t2"; "$py" -m acbpo.math_search_policy_mismatch train --rollout "$ro2" --output-dir "$t2" --model "$m" --initial-adapter "$t1/adapter" --initial-value-head "$t1/value_head.pt" --estimator "$e" --partial-lambda .75 --seed "$seed" --learning-rate "$lr" --gradient-accumulation 32 --lora-rank 32 --lora-alpha 64 --ppo-clip .2 --kl-coefficient .02 --value-coefficient .5 --max-grad-norm 1
 mkdir -p "$base/eval"
 for h in 8 16; do "$py" -m acbpo.math_joint_value_ppo evaluate --data "$p/data/math500_scalar.jsonl" --output "$base/eval/H${h}.json" --model "$m" --actor-adapter "$t2/adapter" --offset 48 --count 200 --seed 100901 --max-steps "$h" --step-tokens 96 --max-prompt-tokens 768 --head-tokens 192 --batch-size 24; done
 touch "$base/COMPLETE"
}
mkdir -p "$r/confirmation/logs"
for g in 0 1 2 3 4 5 6 7; do i=$((g%4)); seed="${seeds[$g]}"; (export CUDA_VISIBLE_DEVICES="$g";worker "$g" "$seed" "$i") >"$r/confirmation/logs/gpu${g}_s${seed}_${arms[$i]}.log" 2>&1 & pids+=("$!"); done
st=0;for x in "${pids[@]}";do wait "$x"||st=1;done
[[ "$st" == 0 ]] && touch "$r/confirmation/EVIDENCE_COMPLETE";exit "$st"
