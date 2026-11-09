#!/usr/bin/env bash
set -euo pipefail

project_root=/ws/user/txia/p011.cgcs-on-policy
root="$project_root/results/search_policy_mismatch/llm_s3"
protocol="$project_root/results/search_policy_mismatch/S3_LLM_TRANSFER_PROTOCOL.md"
expected_protocol_sha=be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978
data="$project_root/data/math500_scalar.jsonl"
expected_data_sha=471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7
gpu=1
expected_uuid=GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48
model=Qwen/Qwen3-4B-Instruct-2507
seeds=(99801 99802 99803)
families=(best_of_n entropy_value value_value)
estimators=(E0_standard_actor_ratio E1_trajectory_propensity E3_step_cumulative_lambda_0.75 E4_extended_candidate_set)

sha256_file() { sha256sum "$1" | awk '{print $1}'; }

[[ -f "$protocol" ]] || { echo "missing frozen protocol: $protocol" >&2; exit 1; }
[[ "$(sha256_file "$protocol")" == "$expected_protocol_sha" ]] || {
  echo "protocol digest mismatch" >&2; exit 1;
}
[[ -f "$data" ]] || { echo "missing MATH-500 scalar data" >&2; exit 1; }
[[ "$(sha256_file "$data")" == "$expected_data_sha" ]] || {
  echo "MATH-500 scalar digest mismatch" >&2; exit 1;
}

mkdir -p "$root"
echo $$ >"$root/pid"
observed_uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
[[ "$observed_uuid" == "$expected_uuid" ]] || {
  echo "GPU $gpu UUID mismatch: $observed_uuid" >&2; exit 1;
}
compute_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)
[[ -z "$compute_pids" ]] || {
  echo "refusing to start: GPU1 has compute PID(s): $compute_pids" >&2; exit 1;
}

session_start=$(date +%s)
session_log="$root/run_sessions.tsv"
usage_log="$root/gpu_usage_${session_start}_$$.csv"
printf 'START\t%s\t%s\n' "$$" "$session_start" >>"$session_log"
nvidia-smi -i "$gpu" \
  --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits -l 5 >"$usage_log" &
monitor_pid=$!
session_closed=0
close_session() {
  local status="$1"
  [[ "$session_closed" == 0 ]] || return 0
  local session_end
  session_end=$(date +%s)
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  printf 'END\t%s\t%s\t%s\n' "$$" "$session_end" "$status" >>"$session_log"
  session_closed=1
}
record_session() {
  local status=$?
  close_session "$status"
  trap - EXIT
  exit "$status"
}
trap record_session EXIT

export CUDA_VISIBLE_DEVICES="$gpu"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="$project_root/src"

rollout_complete() {
  local path="$1" expected_count="$2" expected_arm="$3"
  [[ -f "$path" ]] || return 1
  "$project_root/.venv/bin/python" - "$path" "$expected_count" "$expected_arm" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); count=int(sys.argv[2]); arm=sys.argv[3]
c=d.get('config',{}); examples=d.get('examples',[])
ok=(len(examples)==count and d.get('summary',{}).get('examples')==count
    and c.get('count')==count and c.get('arm')==arm)
raise SystemExit(0 if ok else 1)
PY
}

train_complete() {
  local directory="$1" expected_estimator="$2"
  [[ -f "$directory/summary.json" && -f "$directory/value_head.pt" && -d "$directory/adapter" ]] || return 1
  "$project_root/.venv/bin/python" - "$directory/summary.json" "$expected_estimator" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1])); estimator=sys.argv[2]; c=d.get('config',{})
ok=(d.get('schema')=='cgcs-search-policy-mismatch-llm-train-v1'
    and c.get('estimator')==estimator and d.get('controller_gradient_stopped') is True
    and d.get('states',0)>0 and d.get('optimizer_updates',0)>0
    and math.isfinite(float(d.get('mean_actor_loss',float('nan'))))
    and math.isfinite(float(d.get('mean_value_loss',float('nan')))))
raise SystemExit(0 if ok else 1)
PY
}

eval_complete() {
  local path="$1" expected_count="$2" expected_offset="$3"
  [[ -f "$path" ]] || return 1
  "$project_root/.venv/bin/python" - "$path" "$expected_count" "$expected_offset" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); count=int(sys.argv[2]); offset=int(sys.argv[3]); c=d.get('config',{})
ok=(len(d.get('examples',[]))==count and d.get('summary',{}).get('examples')==count
    and c.get('offset')==offset and c.get('count')==count)
raise SystemExit(0 if ok else 1)
PY
}

threshold_arguments() {
  local family="$1" rollout="$2"
  "$project_root/.venv/bin/python" - "$family" "$rollout" <<'PY'
import json, sys
family=sys.argv[1]; c=json.load(open(sys.argv[2]))['config']
if family == 'entropy_value': print('--entropy-threshold', c['entropy_threshold'])
elif family == 'value_value': print('--value-threshold', c['value_threshold'])
PY
}

train_estimator() {
  local rollout="$1" output="$2" prior="$3" estimator="$4" seed="$5"
  if ! train_complete "$output" "$estimator"; then
    .venv/bin/python -m acbpo.math_search_policy_mismatch train \
      --rollout "$rollout" --output-dir "$output" --model "$model" \
      --initial-adapter "$prior/adapter" --initial-value-head "$prior/value_head.pt" \
      --estimator "$estimator" --partial-lambda 0.75 --seed "$seed" \
      --learning-rate 5e-6 --gradient-accumulation 32 --lora-rank 32 \
      --lora-alpha 64 --ppo-clip 0.2 --kl-coefficient 0.02 --value-coefficient 0.5
  fi
}

cd "$project_root"

for seed in "${seeds[@]}"; do
  seed_root="$root/s${seed}"
  burnin_rollout="$seed_root/burnin/rollout.json"
  burnin_train="$seed_root/burnin/train"
  mkdir -p "$seed_root/burnin"
  if ! rollout_complete "$burnin_rollout" 48 burnin; then
    .venv/bin/python -m acbpo.math_joint_value_ppo collect \
      --data data/gsm8k/train.jsonl --output "$burnin_rollout" --model "$model" \
      --arm burnin --offset 6500 --count 48 --seed "$seed" --max-steps 7 \
      --step-tokens 80 --sampling-temperature 1.3
  fi
  if ! train_complete "$burnin_train" E0_standard_actor_ratio; then
    .venv/bin/python -m acbpo.math_search_policy_mismatch train \
      --rollout "$burnin_rollout" --output-dir "$burnin_train" --model "$model" \
      --estimator E0_standard_actor_ratio --seed "$seed" --learning-rate 5e-6 \
      --gradient-accumulation 32 --lora-rank 32 --lora-alpha 64 \
      --ppo-clip 0.2 --kl-coefficient 0.02 --value-coefficient 0.5
  fi
  touch "$seed_root/burnin/COMPLETE"

  prior_train="$burnin_train"
  for round in 1 2; do
    if [[ "$round" == 1 ]]; then offset=6548; else offset=6596; fi
    branch="$seed_root/round${round}/k1/E0_standard_actor_ratio"
    mkdir -p "$branch"
    rollout="$branch/rollout.json"
    if ! rollout_complete "$rollout" 48 k1; then
      .venv/bin/python -m acbpo.math_joint_value_ppo collect \
        --data data/gsm8k/train.jsonl --output "$rollout" --model "$model" \
        --actor-adapter "$prior_train/adapter" --value-head "$prior_train/value_head.pt" \
        --arm k1 --offset "$offset" --count 48 --seed "$seed" --max-steps 7 \
        --step-tokens 80 --sampling-temperature 1.3
    fi
    train_estimator "$rollout" "$branch/train" "$prior_train" E0_standard_actor_ratio "$seed"
    prior_train="$branch/train"
    touch "$branch/COMPLETE"
  done

  for family in "${families[@]}"; do
    if [[ "$family" == best_of_n ]]; then candidates=2; expansion_rate=1.0
    else candidates=4; expansion_rate=0.3333333333333333; fi
    shared="$seed_root/round1/$family/shared_rollout.json"
    mkdir -p "$(dirname "$shared")"
    if ! rollout_complete "$shared" 48 "$family"; then
      threshold_args=()
      if [[ "$family" != best_of_n ]]; then threshold_args=(--threshold-source "$burnin_rollout"); fi
      .venv/bin/python -m acbpo.math_joint_value_ppo collect \
        --data data/gsm8k/train.jsonl --output "$shared" --model "$model" \
        --actor-adapter "$burnin_train/adapter" --value-head "$burnin_train/value_head.pt" \
        --arm "$family" --offset 6548 --count 48 --seed "$seed" --max-steps 7 \
        --step-tokens 80 --candidates "$candidates" --sampling-temperature 1.3 \
        --epsilon 0.10 --beta 1.0 --expansion-rate "$expansion_rate" "${threshold_args[@]}"
    fi

    for estimator in "${estimators[@]}"; do
      round1="$seed_root/round1/$family/$estimator"
      mkdir -p "$round1"
      train_estimator "$shared" "$round1/train" "$burnin_train" "$estimator" "$seed"
      touch "$round1/COMPLETE"

      round2="$seed_root/round2/$family/$estimator"
      rollout="$round2/rollout.json"
      mkdir -p "$round2"
      if ! rollout_complete "$rollout" 48 "$family"; then
        threshold_args=()
        if [[ "$family" != best_of_n ]]; then threshold_args=(--threshold-source "$shared"); fi
        .venv/bin/python -m acbpo.math_joint_value_ppo collect \
          --data data/gsm8k/train.jsonl --output "$rollout" --model "$model" \
          --actor-adapter "$round1/train/adapter" --value-head "$round1/train/value_head.pt" \
          --arm "$family" --offset 6596 --count 48 --seed "$seed" --max-steps 7 \
          --step-tokens 80 --candidates "$candidates" --sampling-temperature 1.3 \
          --epsilon 0.10 --beta 1.0 --expansion-rate "$expansion_rate" "${threshold_args[@]}"
      fi
      train_estimator "$rollout" "$round2/train" "$round1/train" "$estimator" "$seed"
      touch "$round2/COMPLETE"
    done
  done
  touch "$seed_root/TRAIN_COMPLETE"
done

# The sealed confirmation is opened only after every training branch is complete.
for seed in "${seeds[@]}"; do
  seed_root="$root/s${seed}"
  eval_root="$seed_root/eval"
  mkdir -p "$eval_root/raw_k1" "$eval_root/composite"
  k1_train="$seed_root/round2/k1/E0_standard_actor_ratio/train"
  k1_output="$eval_root/raw_k1/k1_E0_standard_actor_ratio.json"
  if ! eval_complete "$k1_output" 200 48; then
    .venv/bin/python -m acbpo.math_joint_value_ppo evaluate \
      --data data/math500_scalar.jsonl --output "$k1_output" --model "$model" \
      --actor-adapter "$k1_train/adapter" --offset 48 --count 200 --seed 100901 \
      --max-steps 8 --step-tokens 96 --max-prompt-tokens 768 --head-tokens 192
  fi

  for family in "${families[@]}"; do
    if [[ "$family" == best_of_n ]]; then candidates=2; expansion_rate=1.0
    else candidates=4; expansion_rate=0.3333333333333333; fi
    for estimator in "${estimators[@]}"; do
      branch="$seed_root/round2/$family/$estimator"
      raw="$eval_root/raw_k1/${family}_${estimator}.json"
      if ! eval_complete "$raw" 200 48; then
        .venv/bin/python -m acbpo.math_joint_value_ppo evaluate \
          --data data/math500_scalar.jsonl --output "$raw" --model "$model" \
          --actor-adapter "$branch/train/adapter" --offset 48 --count 200 \
          --seed 100901 --max-steps 8 --step-tokens 96 \
          --max-prompt-tokens 768 --head-tokens 192
      fi

      composite="$eval_root/composite/${family}_${estimator}.json"
      if ! eval_complete "$composite" 200 48; then
        threshold_args=()
        if [[ "$family" != best_of_n ]]; then
          read -r -a threshold_args <<<"$(threshold_arguments "$family" "$branch/rollout.json")"
        fi
        .venv/bin/python -m acbpo.math_joint_value_ppo collect \
          --data data/math500_scalar.jsonl --output "$composite" --model "$model" \
          --actor-adapter "$branch/train/adapter" --value-head "$branch/train/value_head.pt" \
          --arm "$family" --offset 48 --count 200 --seed 100901 --max-steps 8 \
          --step-tokens 96 --max-prompt-tokens 768 --head-tokens 192 \
          --candidates "$candidates" --sampling-temperature 1.3 --epsilon 0.10 \
          --beta 1.0 --expansion-rate "$expansion_rate" "${threshold_args[@]}"
      fi
    done
  done
  touch "$seed_root/COMPLETE"
done

touch "$root/EVIDENCE_COMPLETE"
close_session 0
trap - EXIT
.venv/bin/python scripts/analyze_search_policy_mismatch_s3.py \
  --root "$root" --project-root "$project_root" \
  --output-json "$root/S3_LLM_TRANSFER_ANALYSIS.json" \
  --output-report "$root/S3_LLM_TRANSFER_REPORT.md"
touch "$root/COMPLETE"
