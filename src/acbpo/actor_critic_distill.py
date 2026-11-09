from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .actor_lora_experiment import (
    PolicyExample,
    build_choice_messages,
    build_policy_examples,
    middle_truncate,
)
from .qwen_tool_policy import selector_probabilities


def aggregate_teacher_distribution(
    candidate_action_ids: list[int],
    candidate_scores: list[float],
    *,
    beta: float,
    epsilon: float,
) -> tuple[list[int], np.ndarray]:
    if not candidate_action_ids or len(candidate_action_ids) != len(candidate_scores):
        raise ValueError("candidate actions and scores must be non-empty and aligned")
    candidates = np.asarray(candidate_action_ids, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    position_probabilities = selector_probabilities(
        candidates, scores, beta=beta, epsilon=epsilon
    )
    unique_actions = list(dict.fromkeys(candidate_action_ids))
    action_probabilities = np.asarray(
        [position_probabilities[candidates == action].sum() for action in unique_actions],
        dtype=np.float64,
    )
    action_probabilities /= action_probabilities.sum()
    return unique_actions, action_probabilities


def split_examples_by_pair(
    examples: list[PolicyExample], *, train_pairs: int, seed: int
) -> tuple[list[PolicyExample], list[PolicyExample], list[tuple[int, int]]]:
    pairs = sorted({(example.seed, example.task_id) for example in examples})
    random.Random(seed).shuffle(pairs)
    if not 0 < train_pairs < len(pairs):
        raise ValueError("train-pairs must leave non-empty train and validation sets")
    train_keys = set(pairs[:train_pairs])
    train = [x for x in examples if (x.seed, x.task_id) in train_keys]
    validation = [x for x in examples if (x.seed, x.task_id) not in train_keys]
    return train, validation, pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--critic-checkpoint", type=Path, required=True)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-pairs", type=int, default=100)
    parser.add_argument("--max-train-examples", type=int, default=128)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--base-kl-weight", type=float, default=0.0)
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=68001)
    parser.add_argument("--log-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint = np.load(args.critic_checkpoint)
    critic_score_scale = float(checkpoint["score_scale"])
    if args.prepared_data.exists():
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
        args.prepared_data.parent.mkdir(parents=True, exist_ok=True)
        args.prepared_data.write_text(
            json.dumps({"examples": [asdict(example) for example in examples]}) + "\n"
        )
    missing = [x for x in examples if not x.candidate_action_ids]
    if missing:
        raise RuntimeError(
            f"{len(missing)} examples lack logged candidates; rebuild prepared data"
        )
    if args.semantic_weight > 0 and not any(x.semantic_action_ids for x in examples):
        raise RuntimeError(
            "semantic anchor requested but prepared data has no semantic targets; "
            "rebuild with a new --prepared-data path"
        )
    training, validation, pairs = split_examples_by_pair(
        examples, train_pairs=args.train_pairs, seed=args.seed
    )
    if args.prepare_only:
        print(
            json.dumps(
                {
                    "examples": len(examples),
                    "pairs": len(pairs),
                    "training_pairs": args.train_pairs,
                    "validation_pairs": len(pairs) - args.train_pairs,
                    "training_examples": len(training),
                    "validation_examples": len(validation),
                    "semantic_examples": sum(bool(x.semantic_action_ids) for x in examples),
                },
                indent=2,
            )
        )
        return

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed + 1)
    rng.shuffle(training)
    rng.shuffle(validation)
    if args.max_train_examples > 0:
        training = training[: args.max_train_examples]
    validation = validation[: args.validation_examples]

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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    def tensors(example: PolicyExample):
        prompt = tokenizer.apply_chat_template(
            build_choice_messages(example.messages, example.action_names),
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        truncated = len(token_ids) > args.max_length
        token_ids = middle_truncate(token_ids, args.max_length, args.head_tokens)
        input_ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
        unique, teacher = aggregate_teacher_distribution(
            example.candidate_action_ids,
            example.candidate_scores,
            beta=args.beta,
            epsilon=args.epsilon,
        )
        indices = torch.tensor(unique, device="cuda", dtype=torch.long)
        target = torch.tensor(teacher, device="cuda", dtype=torch.float32)
        allowed = torch.tensor(
            np.flatnonzero(np.asarray(example.allowed_mask, dtype=bool)),
            device="cuda",
            dtype=torch.long,
        )
        semantic_ids = set(example.semantic_action_ids)
        semantic_positions = torch.tensor(
            [
                position
                for position, action_id in enumerate(allowed.tolist())
                if action_id in semantic_ids
            ],
            device="cuda",
            dtype=torch.long,
        )
        return input_ids, indices, target, allowed, semantic_positions, truncated

    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    critic_losses: list[float] = []
    base_kls: list[float] = []
    semantic_losses: list[float] = []
    truncations = 0
    updates = 0
    model.train()
    with (args.output_dir / "train_log.jsonl").open("w") as log_file:
        for epoch in range(args.epochs):
            for index, example in enumerate(training, start=1):
                (
                    input_ids,
                    candidate_indices,
                    teacher,
                    allowed_indices,
                    semantic_positions,
                    truncated,
                ) = tensors(example)
                truncations += int(truncated)
                attention_mask = torch.ones_like(input_ids)
                if args.base_kl_weight > 0:
                    with model.disable_adapter(), torch.no_grad():
                        base_logits = model(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).logits[0, -1, label_ids_tensor].float()
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[
                    0, -1, label_ids_tensor
                ].float()
                student_logp = torch.log_softmax(
                    logits[candidate_indices] / args.temperature, dim=0
                )
                critic_loss = -(teacher * student_logp).sum()
                raw_loss = critic_loss
                base_kl = logits.new_zeros(())
                if args.base_kl_weight > 0:
                    base_allowed_logp = torch.log_softmax(
                        base_logits[allowed_indices] / args.temperature, dim=0
                    )
                    student_allowed_logp = torch.log_softmax(
                        logits[allowed_indices] / args.temperature, dim=0
                    )
                    base_kl = torch.sum(
                        base_allowed_logp.exp()
                        * (base_allowed_logp - student_allowed_logp)
                    )
                    raw_loss = raw_loss + args.base_kl_weight * base_kl
                semantic_loss = logits.new_zeros(())
                if args.semantic_weight > 0 and semantic_positions.numel() > 0:
                    student_allowed_logp = torch.log_softmax(
                        logits[allowed_indices] / args.temperature, dim=0
                    )
                    semantic_loss = -student_allowed_logp[semantic_positions].mean()
                    raw_loss = raw_loss + args.semantic_weight * semantic_loss
                (raw_loss / args.gradient_accumulation).backward()
                losses.append(float(raw_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                base_kls.append(float(base_kl.detach().cpu()))
                if semantic_positions.numel() > 0:
                    semantic_losses.append(float(semantic_loss.detach().cpu()))
                update = index % args.gradient_accumulation == 0 or index == len(training)
                if update:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                if index % args.log_every == 0 or index == len(training):
                    record = {
                        "samples_seen": index,
                        "updates": updates,
                        "recent_mean_cross_entropy": float(
                            np.mean(critic_losses[-args.log_every :])
                        ),
                        "recent_mean_total_loss": float(
                            np.mean(losses[-args.log_every :])
                        ),
                        "recent_mean_base_kl": float(
                            np.mean(base_kls[-args.log_every :])
                        ),
                        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    }
                    print(json.dumps(record), flush=True)
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()

    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    model.eval()
    validation_records: list[dict[str, float | bool]] = []
    with torch.inference_mode():
        for index, example in enumerate(validation, start=1):
            (
                input_ids,
                candidate_indices,
                teacher,
                allowed_indices,
                semantic_positions,
                _,
            ) = tensors(example)
            attention_mask = torch.ones_like(input_ids)
            with model.disable_adapter():
                base_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[0, -1, label_ids_tensor].float()
            new_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[0, -1, label_ids_tensor].float()
            base_logp = torch.log_softmax(
                base_logits[candidate_indices] / args.temperature, dim=0
            )
            new_logp = torch.log_softmax(
                new_logits[candidate_indices] / args.temperature, dim=0
            )
            base_allowed_logp = torch.log_softmax(
                base_logits[allowed_indices] / args.temperature, dim=0
            )
            new_allowed_logp = torch.log_softmax(
                new_logits[allowed_indices] / args.temperature, dim=0
            )
            semantic_base_nll = float("nan")
            semantic_new_nll = float("nan")
            if semantic_positions.numel() > 0:
                semantic_base_nll = float(
                    (-base_allowed_logp[semantic_positions].mean()).cpu()
                )
                semantic_new_nll = float(
                    (-new_allowed_logp[semantic_positions].mean()).cpu()
                )
            validation_records.append(
                {
                    "base_cross_entropy": float((-(teacher * base_logp).sum()).cpu()),
                    "new_cross_entropy": float((-(teacher * new_logp).sum()).cpu()),
                    "base_top_agreement": bool(torch.argmax(base_logp) == torch.argmax(teacher)),
                    "new_top_agreement": bool(torch.argmax(new_logp) == torch.argmax(teacher)),
                    "new_to_base_kl": float(
                        torch.sum(new_logp.exp() * (new_logp - base_logp)).cpu()
                    ),
                    "base_to_new_allowed_kl": float(
                        torch.sum(
                            base_allowed_logp.exp()
                            * (base_allowed_logp - new_allowed_logp)
                        ).cpu()
                    ),
                    "semantic_base_nll": semantic_base_nll,
                    "semantic_new_nll": semantic_new_nll,
                }
            )
            if index % args.log_every == 0 or index == len(validation):
                print(json.dumps({"validation_scored": index}), flush=True)

    def average(key: str) -> float:
        return float(np.mean([float(record[key]) for record in validation_records]))

    base_ce = average("base_cross_entropy")
    new_ce = average("new_cross_entropy")
    semantic_validation = [
        record
        for record in validation_records
        if math.isfinite(float(record["semantic_base_nll"]))
    ]
    summary: dict[str, Any] = {
        "config": {
            key: [str(item) for item in value] if isinstance(value, list)
            else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "all_pairs": len(pairs),
        "training_pairs": args.train_pairs,
        "validation_pairs": len(pairs) - args.train_pairs,
        "training_examples": len(training),
        "validation_examples": len(validation_records),
        "updates": updates,
        "truncated_training_examples": truncations,
        "mean_training_cross_entropy": float(np.mean(losses)),
        "mean_training_critic_cross_entropy": float(np.mean(critic_losses)),
        "mean_training_base_kl": float(np.mean(base_kls)),
        "semantic_training_examples": len(semantic_losses),
        "mean_training_semantic_nll": (
            float(np.mean(semantic_losses)) if semantic_losses else None
        ),
        "base_validation_cross_entropy": base_ce,
        "new_validation_cross_entropy": new_ce,
        "validation_cross_entropy_improvement": base_ce - new_ce,
        "base_top_teacher_agreement": average("base_top_agreement"),
        "new_top_teacher_agreement": average("new_top_agreement"),
        "mean_new_to_base_kl": average("new_to_base_kl"),
        "mean_base_to_new_allowed_kl": average("base_to_new_allowed_kl"),
        "semantic_validation_examples": len(semantic_validation),
        "base_semantic_validation_nll": (
            float(np.mean([x["semantic_base_nll"] for x in semantic_validation]))
            if semantic_validation else None
        ),
        "new_semantic_validation_nll": (
            float(np.mean([x["semantic_new_nll"] for x in semantic_validation]))
            if semantic_validation else None
        ),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "finite": bool(math.isfinite(new_ce)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
