from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import extract_final_answer, load_gsm8k
from .math_step_action_pilot import step_messages, terminal_result


ARMS = ("actor_zero", "gated_random", "gated_ensemble")
EXPECTED_SEEDS = (97002, 97003, 97004)


@dataclass(frozen=True)
class StepTrainingExample:
    actor_seed: int
    example_id: int
    step_index: int
    question: str
    prefix: list[str]
    target: str


def load_matched_training_examples(
    result_paths: Sequence[Path], arm: str
) -> list[StepTrainingExample]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    documents = [json.loads(path.read_text()) for path in result_paths]
    observed_seeds = tuple(sorted(int(document["config"]["seed"]) for document in documents))
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(f"expected actor seeds {EXPECTED_SEEDS}, got {observed_seeds}")

    arm_maps: dict[str, dict[tuple[int, int, int], StepTrainingExample]] = {
        name: {} for name in ARMS
    }
    for document in documents:
        config = document["config"]
        if config["offset"] != 7350 or config["count"] != 120:
            raise ValueError("replication data slice does not match the frozen protocol")
        actor_seed = int(config["seed"])
        if len(document["examples"]) != 120:
            raise ValueError("each replication seed must contain exactly 120 examples")
        for row in document["examples"]:
            for name in ARMS:
                for trace in row["arms"][name]["trace"]:
                    key = (actor_seed, int(row["example_id"]), int(trace["step_index"]))
                    choice = int(trace["choice"])
                    arm_maps[name][key] = StepTrainingExample(
                        actor_seed=actor_seed,
                        example_id=int(row["example_id"]),
                        step_index=int(trace["step_index"]),
                        question=str(row["question"]),
                        prefix=[str(value) for value in trace["prefix"]],
                        target=str(trace["candidates"][choice]),
                    )

    matched_keys = sorted(set.intersection(*(set(values) for values in arm_maps.values())))
    if len(matched_keys) != 1688:
        raise ValueError(f"expected 1688 matched states, got {len(matched_keys)}")
    return [arm_maps[arm][key] for key in matched_keys]


def _middle_truncate(values: list[int], length: int, head: int) -> list[int]:
    if len(values) <= length:
        return values
    return values[:head] + values[-(length - head) :]


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_matched_training_examples(args.inputs, args.arm)
    random.Random(args.seed).shuffle(examples)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    records: list[dict[str, float | int]] = []
    updates = 0
    started = time.time()

    for step_number, example in enumerate(examples, 1):
        prompt = qwen_chat_prompt(step_messages(example.question, example.prefix))
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = tokenizer(
            example.target + "<|im_end|>\n", add_special_tokens=False
        ).input_ids
        prompt_budget = args.max_length - len(target_ids)
        if prompt_budget < 32:
            raise ValueError("target leaves insufficient prompt context")
        prompt_ids = _middle_truncate(
            prompt_ids, prompt_budget, min(args.head_tokens, prompt_budget - 1)
        )
        input_ids = torch.tensor(
            [prompt_ids + target_ids], device="cuda", dtype=torch.long
        )
        labels = torch.tensor(
            [[-100] * len(prompt_ids) + target_ids], device="cuda", dtype=torch.long
        )
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        logits = output.logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            ignore_index=-100,
        )
        (loss / args.gradient_accumulation).backward()
        update_now = (
            step_number % args.gradient_accumulation == 0
            or step_number == len(examples)
        )
        if update_now:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        records.append({"step": step_number, "loss": float(loss.detach().cpu())})
        if update_now:
            print(
                json.dumps(
                    {
                        "arm": args.arm,
                        "seed": args.seed,
                        "examples": step_number,
                        "updates": updates,
                        "loss": records[-1]["loss"],
                    }
                ),
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    summary = {
        "config": {
            key: [str(item) for item in value]
            if isinstance(value, list) and value and isinstance(value[0], Path)
            else str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
            if key != "func"
        },
        "matched_examples": len(examples),
        "optimizer_updates": updates,
        "mean_loss": mean(float(row["loss"]) for row in records),
        "final_100_mean_loss": mean(float(row["loss"]) for row in records[-100:]),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = (
        PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
        if args.adapter is not None
        else base
    )
    model.eval()
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        frozen = (args.offset, args.count, args.seed, str(args.adapter or "base"))
        observed = (
            previous["config"]["offset"],
            previous["config"]["count"],
            previous["config"]["seed"],
            previous["config"]["adapter"],
        )
        if observed != frozen:
            raise ValueError(f"evaluation resume mismatch: {observed} != {frozen}")
        existing = previous["examples"]
    completed = {int(row["example_id"]) for row in existing}
    started = time.time()
    for example in examples:
        if example.example_id in completed:
            continue
        steps: list[str] = []
        for step_index in range(args.max_steps):
            prompt = qwen_chat_prompt(step_messages(example.question, steps))
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=args.step_tokens,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = output[0, encoded["input_ids"].shape[1] :]
            text = tokenizer.decode(generated, skip_special_tokens=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            step = lines[0] if lines else "(empty step)"
            steps.append(step)
            if extract_final_answer(step) is not None:
                break
        existing.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "steps": steps,
                "result": terminal_result(steps, example.reference_answer),
            }
        )
        summary = {
            "examples": len(existing),
            "accuracy": mean(float(row["result"]["correct"]) for row in existing),
            "answered_fraction": mean(float(row["result"]["answered"]) for row in existing),
            "mean_steps": mean(len(row["steps"]) for row in existing),
        }
        payload = {
            "config": {
                "data": str(args.data),
                "model": args.model,
                "adapter": str(args.adapter or "base"),
                "offset": args.offset,
                "count": args.count,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "step_tokens": args.step_tokens,
                "decoding": "greedy",
            },
            "summary": summary,
            "elapsed_seconds": time.time() - started,
            "examples": existing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"example_id": example.example_id, **summary}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched step-action one-update study")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--head-tokens", type=int, default=192)
    train_parser.add_argument("--learning-rate", type=float, default=5e-6)
    train_parser.add_argument("--gradient-accumulation", type=int, default=32)
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--lora-alpha", type=int, default=64)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--data", type=Path, required=True)
    evaluate_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    evaluate_parser.add_argument("--adapter", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--offset", type=int, required=True)
    evaluate_parser.add_argument("--count", type=int, required=True)
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--max-steps", type=int, default=7)
    evaluate_parser.add_argument("--step-tokens", type=int, default=80)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
