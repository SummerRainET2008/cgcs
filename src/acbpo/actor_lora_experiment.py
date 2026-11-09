from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .step_critic_experiment import constrained_action_mask
from .tau_retail import DirectRetailEnv


@dataclass
class PolicyExample:
    messages: list[dict[str, str]]
    action_names: list[str]
    allowed_mask: list[bool]
    selected_action: int
    joint_ratio: float
    return_to_go: float
    critic_baseline: float
    advantage: float
    task_id: int
    seed: int
    step: int
    candidate_action_ids: list[int] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    semantic_action_ids: list[int] = field(default_factory=list)


def discounted_returns(rewards: Iterable[float], gamma: float) -> list[float]:
    rewards = list(rewards)
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def build_choice_messages(
    messages: list[dict[str, str]], action_names: list[str]
) -> list[dict[str, str]]:
    labels = [chr(ord("A") + index) for index in range(len(action_names))]
    choices = "\n".join(
        f"{label}: {name}" for label, name in zip(labels, action_names)
    )
    return messages + [
        {
            "role": "user",
            "content": (
                "Choose the single best next action name from this list. "
                "Return only its letter.\n" + choices + "\nAnswer:"
            ),
        }
    ]


def middle_truncate(token_ids: list[int], max_length: int, head_tokens: int) -> list[int]:
    if max_length < 2:
        raise ValueError("max_length must be at least two")
    if not 0 <= head_tokens < max_length:
        raise ValueError("head_tokens must be in [0, max_length)")
    if len(token_ids) <= max_length:
        return token_ids
    tail_tokens = max_length - head_tokens
    return token_ids[:head_tokens] + token_ids[-tail_tokens:]


def remaining_semantic_action_ids(
    action_names: list[str],
    allowed_mask: list[bool] | np.ndarray,
    gold_action_keys: list[tuple[str, str]],
    completed_gold_keys: set[tuple[str, str]],
) -> list[int]:
    """Return allowed action-name targets for unfinished official actions."""
    remaining_names = {
        name
        for name, arguments in gold_action_keys
        if (name, arguments) not in completed_gold_keys
    }
    return [
        index
        for index, name in enumerate(action_names)
        if bool(allowed_mask[index]) and name in remaining_names
    ]


def _load_critic_episodes(paths: list[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for path in paths:
        payload = json.loads(path.read_text())
        for episode in payload["evaluation_tasks"]["critic"]:
            key = (int(episode["seed"]), int(episode["task_id"]))
            if key in seen:
                raise ValueError(f"duplicate critic episode {key} in {path}")
            seen.add(key)
            episodes.append(episode)
    return episodes


def build_policy_examples(
    *,
    input_paths: list[Path],
    tau_root: Path,
    gamma: float,
    critic_score_scale: float,
) -> list[PolicyExample]:
    examples: list[PolicyExample] = []
    for episode in _load_critic_episodes(input_paths):
        env = DirectRetailEnv(tau_root, "train", int(episode["task_id"]))
        action_names = env.action_names
        messages = [
            {
                "role": "system",
                "content": env.wiki + "\n\nThe complete customer request follows.",
            },
            {"role": "user", "content": env.instruction},
        ]
        trajectory = episode["trajectory"]
        gold_action_keys = [
            (action.name, json.dumps(action.kwargs, sort_keys=True))
            for action in env.task.actions
            if action.name != "transfer_to_human_agents"
        ]
        completed_gold_keys: set[tuple[str, str]] = set()
        returns = discounted_returns(
            [float(step["shaped_reward"]) for step in trajectory], gamma
        )
        prefix: list[dict[str, Any]] = []
        for step, return_to_go in zip(trajectory, returns):
            mask = constrained_action_mask(action_names, env.instruction, prefix)
            selected_name = str(step["selected"])
            selected_action = action_names.index(selected_name)
            if not bool(mask[selected_action]):
                raise RuntimeError(
                    f"logged action {selected_name} is masked for task {episode['task_id']}"
                )
            selected_position = int(step["selected_position"])
            normalized_q = float(step["candidate_scores"][selected_position])
            baseline = normalized_q * critic_score_scale
            examples.append(
                PolicyExample(
                    messages=[dict(message) for message in messages],
                    action_names=list(action_names),
                    allowed_mask=mask.astype(bool).tolist(),
                    selected_action=selected_action,
                    joint_ratio=float(step["joint_ratio"]),
                    return_to_go=float(return_to_go),
                    critic_baseline=float(baseline),
                    advantage=float(return_to_go - baseline),
                    task_id=int(episode["task_id"]),
                    seed=int(episode["seed"]),
                    step=int(step["step"]),
                    candidate_action_ids=[
                        action_names.index(str(name))
                        for name in step.get("candidate_actions", [])
                    ],
                    candidate_scores=[
                        float(score) for score in step.get("candidate_scores", [])
                    ],
                    semantic_action_ids=remaining_semantic_action_ids(
                        action_names,
                        mask,
                        gold_action_keys,
                        completed_gold_keys,
                    ),
                )
            )
            result = env.step(selected_name, dict(step["arguments"]))
            if bool(result.error) != bool(step["tool_error"]):
                raise RuntimeError(
                    f"non-deterministic replay on task {episode['task_id']} step {step['step']}"
                )
            executed_key = (
                selected_name,
                json.dumps(dict(step["arguments"]), sort_keys=True),
            )
            if not result.error and executed_key in gold_action_keys:
                completed_gold_keys.add(executed_key)
            prefix.append(step)
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {"name": selected_name, "arguments": step["arguments"]},
                            ensure_ascii=False,
                        ),
                    },
                    {"role": "user", "content": f"Tool result: {result.observation}"},
                ]
            )
    return examples


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prepared-data",
        type=Path,
        help="Cache replayed policy examples as JSON for repeatable training runs.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--rho-clip", type=float, default=10.0)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_length <= args.head_tokens:
        raise ValueError("max-length must exceed head-tokens")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = np.load(args.critic_checkpoint)
    critic_score_scale = float(checkpoint["score_scale"])
    if args.prepared_data is not None and args.prepared_data.exists():
        examples = [
            PolicyExample(**item)
            for item in json.loads(args.prepared_data.read_text())["examples"]
        ]
    else:
        examples = build_policy_examples(
            input_paths=args.inputs,
            tau_root=args.tau_root,
            gamma=args.gamma,
            critic_score_scale=critic_score_scale,
        )
        if args.prepared_data is not None:
            args.prepared_data.parent.mkdir(parents=True, exist_ok=True)
            args.prepared_data.write_text(
                json.dumps(
                    {
                        "inputs": [str(path) for path in args.inputs],
                        "gamma": args.gamma,
                        "critic_score_scale": critic_score_scale,
                        "examples": [asdict(example) for example in examples],
                    }
                )
                + "\n"
            )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "examples": len(examples),
                    "task_seed_pairs": len(
                        {(example.seed, example.task_id) for example in examples}
                    ),
                    "mean_joint_ratio": float(
                        np.mean([example.joint_ratio for example in examples])
                    ),
                    "max_joint_ratio": float(
                        max(example.joint_ratio for example in examples)
                    ),
                    "advantage_mean": float(
                        np.mean([example.advantage for example in examples])
                    ),
                    "advantage_std": float(
                        np.std([example.advantage for example in examples])
                    ),
                },
                indent=2,
            )
        )
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.max_examples > 0:
        examples = examples[: args.max_examples]
    if not examples:
        raise RuntimeError("no policy examples were built")

    raw_advantages = np.asarray([x.advantage for x in examples], dtype=np.float64)
    advantage_mean = float(raw_advantages.mean())
    advantage_std = float(raw_advantages.std())
    advantage_std = max(advantage_std, 1e-6)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    label_ids = [
        tokenizer(chr(ord("A") + index), add_special_tokens=False).input_ids
        for index in range(16)
    ]
    if any(len(ids) != 1 for ids in label_ids):
        raise RuntimeError("action labels are not single tokens")
    label_ids_tensor = torch.tensor([ids[0] for ids in label_ids], device="cuda")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.train()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    updates = 0
    samples_seen = 0
    truncations = 0
    losses: list[float] = []
    peak_memory = 0
    with log_path.open("w") as log_file:
        for epoch in range(args.epochs):
            if epoch:
                rng.shuffle(examples)
            for example_index, example in enumerate(examples):
                prompt = tokenizer.apply_chat_template(
                    build_choice_messages(example.messages, example.action_names),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                token_ids = tokenizer(
                    prompt, add_special_tokens=False
                ).input_ids
                if len(token_ids) > args.max_length:
                    truncations += 1
                token_ids = middle_truncate(
                    token_ids, args.max_length, args.head_tokens
                )
                input_ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                next_logits = output.logits[0, -1, label_ids_tensor].float()
                allowed = torch.tensor(
                    example.allowed_mask, device="cuda", dtype=torch.bool
                )
                masked_logits = (next_logits / args.temperature).masked_fill(
                    ~allowed, -torch.inf
                )
                log_probabilities = torch.log_softmax(masked_logits, dim=0)
                normalized_advantage = (example.advantage - advantage_mean) / advantage_std
                normalized_advantage = float(
                    np.clip(
                        normalized_advantage,
                        -args.advantage_clip,
                        args.advantage_clip,
                    )
                )
                ratio = min(example.joint_ratio, args.rho_clip)
                loss = (
                    -ratio
                    * normalized_advantage
                    * log_probabilities[example.selected_action]
                    / args.gradient_accumulation
                )
                loss.backward()
                losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
                samples_seen += 1
                is_update = samples_seen % args.gradient_accumulation == 0
                is_last = epoch == args.epochs - 1 and example_index == len(examples) - 1
                if is_update or is_last:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
                if samples_seen % args.log_every == 0 or is_last:
                    record = {
                        "samples_seen": samples_seen,
                        "updates": updates,
                        "epoch": epoch,
                        "recent_mean_loss": float(
                            np.mean(losses[-args.log_every :])
                        ),
                        "peak_memory_gib": peak_memory / (1024**3),
                    }
                    print(json.dumps(record), flush=True)
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()

    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    summary = {
        "config": {
            key: [str(item) for item in value]
            if isinstance(value, list)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        },
        "examples": len(examples),
        "updates": updates,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "mean_joint_ratio": float(np.mean([x.joint_ratio for x in examples])),
        "max_joint_ratio": float(max(x.joint_ratio for x in examples)),
        "truncated_examples": truncations,
        "mean_loss": float(np.mean(losses)),
        "peak_memory_gib": peak_memory / (1024**3),
        "finite": bool(math.isfinite(float(np.mean(losses)))),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
