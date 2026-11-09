#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$project_root/results/search_policy_mismatch/s4x_arc_challenge_full"
data_root="$project_root/data/s4x_arc_challenge"
train_data="$data_root/arc_challenge_train.jsonl"
validation_data="$data_root/arc_challenge_validation.jsonl"
test_data="$data_root/arc_challenge_test.jsonl"
data_manifest="$data_root/ARC_CHALLENGE_PREPARATION_MANIFEST.json"
protocol="$project_root/results/search_policy_mismatch/S4X_ARC_CHALLENGE_PROTOCOL.md"
exposure_receipt="$project_root/results/search_policy_mismatch/S4X_ARC_DATA_EXPOSURE_RECEIPT.json"
evaluator="$project_root/src/acbpo/math_joint_value_ppo.py"
arc_task="$project_root/src/acbpo/arc_task.py"
trainer="$project_root/src/acbpo/math_search_policy_mismatch.py"
model=Qwen/Qwen3-4B-Instruct-2507

expected_protocol_sha=e8b1061a52d3bb85f9578a0b4a5e5eac98de98f5aa4963bdff1ca73bf939d9ec
expected_exposure_sha=83fd54d85dff1dbdb7714b5ee2f2bbbd215fcc7c7e732bdd9b73763a3aeecf2b
expected_train_sha=9c60e9b8e7430620f14e970ac782c4151ac131012fd4d4282dd1bf1ea7219302
expected_validation_sha=15e53441074696ad15d589fbd93b89bb1091a0a212159d5c02de84a8acde7b68
expected_test_sha=b05ab02613ce66525668e69183e388c0603f858fda4bd563d0d0709837efd63d
expected_manifest_sha=4782ca836ae167f932e1d0be8ca56cd0c973cbdf016c9cf6a47474d9cc4b0eba
expected_evaluator_sha=29b18f0056dcd42d7acbea76b97306cd82575b73fbcc6e518c9d3ec56f3fdc58
expected_arc_task_sha=ea8e9543dff8b4dfd987dc5b4bdd5c5059c50e3f82320b9d1e5b30d976052f34
expected_trainer_sha=ec46661585035500780d1bc5801a5ecc8bd338186a5c3df2720b5b223a8a3f37
expected_smoke_collect_sha=1f4b9315737a04b02964721d159a594c21cf41085cdc6ffbf96572769ddf902c
expected_smoke_train_sha=0bc7b054ccf0c9ecb228625a028d8ab9128c335f59154e0ade1ebad841173e86

seeds=(99801 99802 99803)
estimators=(E0_standard_actor_ratio E3_step_cumulative_lambda_0.75 E4_extended_candidate_set)
gpu_uuids=(
  GPU-acdf26d2-ac77-ac48-43b3-1d99f496e89b
  GPU-44417620-dcd2-43cc-31b3-402e558c8af1
  GPU-10f2aae3-89f2-e258-4f2e-112f834f9970
  GPU-08af3168-b02e-c10d-91d6-36f0b9287299
)

sha256_file() { sha256sum "$1" | awk '{print $1}'; }
verify_sha() {
  local path="$1" expected="$2" label="$3"
  [[ -f "$path" ]] || { echo "missing $label: $path" >&2; exit 1; }
  [[ "$(sha256_file "$path")" == "$expected" ]] || {
    echo "$label digest mismatch" >&2; exit 1;
  }
}

verify_sha "$protocol" "$expected_protocol_sha" protocol
verify_sha "$exposure_receipt" "$expected_exposure_sha" exposure-receipt
verify_sha "$train_data" "$expected_train_sha" train-data
verify_sha "$validation_data" "$expected_validation_sha" validation-data
verify_sha "$test_data" "$expected_test_sha" sealed-test-data
verify_sha "$data_manifest" "$expected_manifest_sha" data-manifest
verify_sha "$evaluator" "$expected_evaluator_sha" evaluator
verify_sha "$arc_task" "$expected_arc_task_sha" arc-task
verify_sha "$trainer" "$expected_trainer_sha" trainer

mkdir -p "$root"
exec 9>"$root/launcher.lock"
flock -n 9 || { echo "another S4-X launcher holds the lock" >&2; exit 1; }
echo $$ >"$root/pid"

for uuid in "${gpu_uuids[@]}"; do
  name=$(nvidia-smi -i "$uuid" --query-gpu=name --format=csv,noheader)
  [[ "$name" == "NVIDIA GeForce RTX 4080" ]] || {
    echo "$uuid model mismatch: $name" >&2; exit 1;
  }
  pids=$(nvidia-smi -i "$uuid" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)
  [[ -z "$pids" ]] || { echo "$uuid has compute PID(s): $pids" >&2; exit 1; }
  free_mib=$(nvidia-smi -i "$uuid" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  (( free_mib >= 14000 )) || { echo "$uuid has only ${free_mib} MiB free" >&2; exit 1; }
done

smoke="$project_root/results/search_policy_mismatch/s4x_arc_challenge/dev_smoke"
[[ -f "$smoke/collect4.json" && -f "$smoke/train4/summary.json" ]] || {
  echo "missing validation-only smoke evidence" >&2; exit 1;
}
verify_sha "$smoke/collect4.json" "$expected_smoke_collect_sha" smoke-collect
verify_sha "$smoke/train4/summary.json" "$expected_smoke_train_sha" smoke-train
"$project_root/.venv/bin/python" - "$smoke/collect4.json" "$smoke/train4/summary.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1])); t=json.load(open(sys.argv[2]))
assert r["config"]["task"] == "arc_challenge"
assert "validation" in r["config"]["data"] and len(r["examples"]) == 4
assert all(row.get("source_id") for row in r["examples"])
assert t["states"] == 8 and t["optimizer_updates"] == 1
PY

export HF_HOME=/workspace/.hf_home
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root/src"

rollout_complete() {
  local path="$1" count="$2" offset="$3" arm="$4"
  [[ -f "$path" ]] || return 1
  "$project_root/.venv/bin/python" - "$path" "$count" "$offset" "$arm" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2]); off=int(sys.argv[3]); arm=sys.argv[4]
c=d.get("config",{}); rows=d.get("examples",[])
ok=(c.get("task")=="arc_challenge" and c.get("count")==n and c.get("offset")==off
    and c.get("arm")==arm and len(rows)==n
    and [int(x["example_id"]) for x in rows]==list(range(off,off+n))
    and all(x.get("source_id") for x in rows))
raise SystemExit(0 if ok else 1)
PY
}

train_complete() {
  local dir="$1" estimator="$2"
  [[ -f "$dir/summary.json" && -f "$dir/value_head.pt" && -d "$dir/adapter" ]] || return 1
  "$project_root/.venv/bin/python" - "$dir/summary.json" "$estimator" <<'PY'
import json, math, sys
d=json.load(open(sys.argv[1])); e=sys.argv[2]; c=d.get("config",{})
ok=(d.get("schema")=="cgcs-search-policy-mismatch-llm-train-v1"
    and c.get("estimator")==e and d.get("controller_gradient_stopped") is True
    and d.get("states",0)>0 and d.get("optimizer_updates",0)>0
    and math.isfinite(float(d.get("mean_actor_loss",float("nan"))))
    and math.isfinite(float(d.get("mean_value_loss",float("nan")))))
raise SystemExit(0 if ok else 1)
PY
}

collect_rollout() {
  local uuid="$1" output="$2" arm="$3" offset="$4" seed="$5" adapter="$6" head="$7"
  local -a prior_args candidate_args
  prior_args=(); candidate_args=()
  [[ -z "$adapter" ]] || prior_args=(--actor-adapter "$adapter" --value-head "$head")
  [[ "$arm" != best_of_n ]] || candidate_args=(--candidates 2 --epsilon 0.10 --beta 1.0 --expansion-rate 1.0)
  if ! rollout_complete "$output" 128 "$offset" "$arm"; then
    mkdir -p "$(dirname "$output")"
    CUDA_VISIBLE_DEVICES="$uuid" "$project_root/.venv/bin/python" -m acbpo.math_joint_value_ppo collect \
      --task arc_challenge --data "$train_data" --output "$output" --model "$model" \
      --arm "$arm" --offset "$offset" --count 128 --seed "$seed" --max-steps 7 \
      --step-tokens 80 --max-prompt-tokens 768 --head-tokens 192 \
      --sampling-temperature 1.3 "${prior_args[@]}" "${candidate_args[@]}"
  fi
}

train_actor() {
  local uuid="$1" rollout="$2" output="$3" prior="$4" estimator="$5" seed="$6"
  local -a prior_args
  prior_args=()
  [[ -z "$prior" ]] || prior_args=(--initial-adapter "$prior/adapter" --initial-value-head "$prior/value_head.pt")
  if ! train_complete "$output" "$estimator"; then
    CUDA_VISIBLE_DEVICES="$uuid" "$project_root/.venv/bin/python" -m acbpo.math_search_policy_mismatch train \
      --rollout "$rollout" --output-dir "$output" --model "$model" \
      --estimator "$estimator" --partial-lambda 0.75 --seed "$seed" \
      --learning-rate 5e-6 --gradient-accumulation 32 --lora-rank 32 \
      --lora-alpha 64 --ppo-clip 0.2 --kl-coefficient 0.02 \
      --value-coefficient 0.5 "${prior_args[@]}"
  fi
}

run_seed() {
  local uuid="$1" seed="$2" seed_root
  seed_root="$root/training/s${seed}"
  local burn_roll="$seed_root/burnin/rollout.json" burn_train="$seed_root/burnin/train"
  collect_rollout "$uuid" "$burn_roll" burnin 0 "$seed" "" ""
  train_actor "$uuid" "$burn_roll" "$burn_train" "" E0_standard_actor_ratio "$seed"

  local prior="$burn_train" round offset branch rollout
  for round in 1 2; do
    [[ "$round" == 1 ]] && offset=128 || offset=256
    branch="$seed_root/round${round}/k1/E0_standard_actor_ratio"
    rollout="$branch/rollout.json"
    collect_rollout "$uuid" "$rollout" k1 "$offset" "$seed" "$prior/adapter" "$prior/value_head.pt"
    train_actor "$uuid" "$rollout" "$branch/train" "$prior" E0_standard_actor_ratio "$seed"
    prior="$branch/train"
  done

  local shared="$seed_root/round1/best_of_n/shared_rollout.json"
  collect_rollout "$uuid" "$shared" best_of_n 128 "$seed" "$burn_train/adapter" "$burn_train/value_head.pt"
  local estimator r1 r2
  for estimator in "${estimators[@]}"; do
    r1="$seed_root/round1/best_of_n/$estimator"
    train_actor "$uuid" "$shared" "$r1/train" "$burn_train" "$estimator" "$seed"
    r2="$seed_root/round2/best_of_n/$estimator"
    collect_rollout "$uuid" "$r2/rollout.json" best_of_n 256 "$seed" "$r1/train/adapter" "$r1/train/value_head.pt"
    train_actor "$uuid" "$r2/rollout.json" "$r2/train" "$r1/train" "$estimator" "$seed"
  done
  touch "$seed_root/TRAIN_COMPLETE"
}

session_start=$(date +%s)
printf 'START\t%s\t%s\n' "$$" "$session_start" >>"$root/run_sessions.tsv"

worker_pids=()
for index in 0 1 2; do
  run_seed "${gpu_uuids[$index]}" "${seeds[$index]}" >"$root/train_worker${index}.log" 2>&1 &
  worker_pids+=("$!")
done
train_status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || train_status=1; done
(( train_status == 0 )) || { echo "S4-X training worker failed" >&2; exit 1; }
for seed in "${seeds[@]}"; do [[ -f "$root/training/s${seed}/TRAIN_COMPLETE" ]] || exit 1; done
touch "$root/TRAINING_COMPLETE"

eval_complete() {
  local path="$1" horizon="$2" adapter="$3" seed="$4"
  [[ -f "$path" ]] || return 1
  "$project_root/.venv/bin/python" - "$path" "$horizon" "$adapter" "$seed" "$test_data" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); h=int(sys.argv[2]); adapter=pathlib.Path(sys.argv[3]); seed=int(sys.argv[4]); data=sys.argv[5]
d=json.loads(p.read_text()); c=d.get("config",{}); rows=d.get("examples",[])
ok=(c.get("task")=="arc_challenge" and c.get("data")==data and c.get("offset")==0
    and c.get("count")==1172 and c.get("seed")==seed and c.get("max_steps")==h
    and c.get("step_tokens")==96 and c.get("max_completion_tokens") is None
    and c.get("max_prompt_tokens")==768 and c.get("head_tokens")==192
    and c.get("batch_size")==24 and pathlib.Path(c.get("actor_adapter",""))==adapter
    and len(rows)==1172 and [int(x["example_id"]) for x in rows]==list(range(1172))
    and all(x.get("source_id") for x in rows))
raise SystemExit(0 if ok else 1)
PY
}

actor_adapter() {
  local seed="$1" actor="$2"
  if [[ "$actor" == k1_E0_standard_actor_ratio ]]; then
    echo "$root/training/s${seed}/round2/k1/E0_standard_actor_ratio/train/adapter"
  else
    echo "$root/training/s${seed}/round2/best_of_n/${actor#best_of_n_}/train/adapter"
  fi
}

run_eval_worker() {
  local worker="$1" uuid="${gpu_uuids[$1]}" index=0 seed actor horizon adapter output
  local actors=(k1_E0_standard_actor_ratio best_of_n_E0_standard_actor_ratio best_of_n_E3_step_cumulative_lambda_0.75 best_of_n_E4_extended_candidate_set)
  for seed in "${seeds[@]}"; do
    for actor in "${actors[@]}"; do
      adapter=$(actor_adapter "$seed" "$actor")
      for horizon in 8 16; do
        if (( index % 4 == worker )); then
          output="$root/eval/s${seed}/H${horizon}/${actor}.json"
          if ! eval_complete "$output" "$horizon" "$adapter" "$seed"; then
            mkdir -p "$(dirname "$output")"
            CUDA_VISIBLE_DEVICES="$uuid" "$project_root/.venv/bin/python" -m acbpo.math_joint_value_ppo evaluate \
              --task arc_challenge --data "$test_data" --output "$output" --model "$model" \
              --actor-adapter "$adapter" --offset 0 --count 1172 --seed "$seed" \
              --max-steps "$horizon" --step-tokens 96 --max-prompt-tokens 768 \
              --head-tokens 192 --batch-size 24
          fi
        fi
        index=$((index + 1))
      done
    done
  done
}

worker_pids=()
for index in 0 1 2 3; do
  run_eval_worker "$index" >"$root/eval_worker${index}.log" 2>&1 & worker_pids+=("$!")
done
eval_status=0
for pid in "${worker_pids[@]}"; do wait "$pid" || eval_status=1; done
(( eval_status == 0 )) || { echo "S4-X evaluation worker failed" >&2; exit 1; }

"$project_root/.venv/bin/python" - "$root" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); files=sorted((root/"eval").glob("s*/H*/*.json"))
assert len(files)==24, len(files)
total=0
for path in files:
    rows=json.loads(path.read_text())["examples"]
    assert len(rows)==1172
    assert [int(x["example_id"]) for x in rows]==list(range(1172))
    assert all(x.get("source_id") for x in rows)
    total += len(rows)
assert total==28128, total
PY
touch "$root/EVIDENCE_COMPLETE"
printf 'END\t%s\t%s\t0\n' "$$" "$(date +%s)" >>"$root/run_sessions.tsv"
