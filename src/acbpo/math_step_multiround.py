from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import extract_final_answer, load_gsm8k
from .math_step_action_pilot import step_messages, terminal_result
from .math_step_closed_loop_replication import post_json, select_replication_actions
from .math_step_matched_update import StepTrainingExample, _middle_truncate


ARMS = ("actor_zero", "gated_random", "gated_ensemble")


def collect(args: argparse.Namespace) -> None:
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        frozen = (args.arm, args.offset, args.count, args.seed)
        observed = tuple(prior["config"][key] for key in ("arm", "offset", "count", "seed"))
        if observed != frozen:
            raise ValueError(f"collection resume mismatch: {observed} != {frozen}")
        existing = prior["examples"]
    completed = {int(row["example_id"]) for row in existing}
    started = time.time()

    def generate(question: str, prefix: Sequence[str], seed: int) -> str:
        row = post_json(
            args.server_url,
            "/completion",
            {
                "prompt": qwen_chat_prompt(step_messages(question, prefix)),
                "seed": seed,
                "n_predict": args.step_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": 40,
                "repeat_penalty": 1.05,
            },
        )
        text = str(row.get("content", "")).replace("<|im_end|>", "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0] if lines else "(empty step)"

    for example in examples:
        if example.example_id in completed:
            continue
        steps: list[str] = []
        trace: list[dict[str, Any]] = []
        seed_base = args.seed + example.example_id * 100_003
        for step_index in range(args.max_steps):
            seeds = [seed_base + step_index * 10_007 + index for index in range(4)]
            with ThreadPoolExecutor(max_workers=4) as executor:
                candidates = list(
                    executor.map(lambda value: generate(example.question, steps, value), seeds)
                )
            scored = post_json(
                args.server_url,
                "/score_all",
                {"problem": example.question, "prefix": steps, "candidates": candidates},
            )
            random_index = random.Random(seed_base + 19_000_033 + step_index * 2_003).choice((1, 2, 3))
            choices, diagnostics = select_replication_actions(
                scored["scores"], threshold=args.threshold, random_index=0,
                gated_random_index=random_index,
            )
            choice = choices[args.arm]
            prefix = list(steps)
            steps.append(candidates[choice])
            trace.append(
                {
                    "step_index": step_index,
                    "prefix": prefix,
                    "candidates": candidates,
                    "critic_scores": scored["scores"],
                    "choice": choice,
                    **diagnostics,
                }
            )
            if extract_final_answer(steps[-1]) is not None:
                break
        existing.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "steps": steps,
                "trace": trace,
                "result": terminal_result(steps, example.reference_answer),
            }
        )
        payload = {
            "config": {
                "arm": args.arm,
                "data": str(args.data),
                "offset": args.offset,
                "count": args.count,
                "seed": args.seed,
                "candidates": 4,
                "max_steps": args.max_steps,
                "step_tokens": args.step_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "threshold": args.threshold,
            },
            "summary": {
                "examples": len(existing),
                "accuracy": mean(float(row["result"]["correct"]) for row in existing),
                "answered_fraction": mean(float(row["result"]["answered"]) for row in existing),
                "states": sum(len(row["trace"]) for row in existing),
            },
            "elapsed_seconds": time.time() - started,
            "examples": existing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"example_id": example.example_id, **payload["summary"]}), flush=True)


def load_round_examples(paths: Sequence[Path], arm: str, limit: int) -> list[StepTrainingExample]:
    rows: list[StepTrainingExample] = []
    seeds = set()
    for path in paths:
        document = json.loads(path.read_text())
        config = document["config"]
        if config["arm"] != arm or config["offset"] != 7350 or config["count"] != 120:
            raise ValueError(f"collection mismatch in {path}")
        seeds.add(int(config["seed"]))
        if len(document["examples"]) != 120:
            raise ValueError(f"incomplete collection in {path}")
        for example in document["examples"]:
            for state in example["trace"]:
                choice = int(state["choice"])
                rows.append(
                    StepTrainingExample(
                        actor_seed=int(config["seed"]),
                        example_id=int(example["example_id"]),
                        step_index=int(state["step_index"]),
                        question=str(example["question"]),
                        prefix=[str(value) for value in state["prefix"]],
                        target=str(state["candidates"][choice]),
                    )
                )
    if len(seeds) != len(paths) or len(rows) < limit:
        raise ValueError(f"need {limit} states from distinct seeds, found {len(rows)}")
    return rows[:limit]


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_round_examples(args.inputs, args.arm, args.examples)
    random.Random(args.seed).shuffle(examples)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        args.model, device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.initial_adapter, is_trainable=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    records = []
    updates = 0
    started = time.time()
    for number, example in enumerate(examples, 1):
        prompt_ids = tokenizer(
            qwen_chat_prompt(step_messages(example.question, example.prefix)),
            add_special_tokens=False,
        ).input_ids
        target_ids = tokenizer(example.target + "<|im_end|>\n", add_special_tokens=False).input_ids
        budget = args.max_length - len(target_ids)
        prompt_ids = _middle_truncate(prompt_ids, budget, min(args.head_tokens, budget - 1))
        input_ids = torch.tensor([prompt_ids + target_ids], device="cuda", dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device="cuda", dtype=torch.long)
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        logits = output.logits[:, :-1].float()
        shifted = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100
        )
        (loss / args.gradient_accumulation).backward()
        update_now = number % args.gradient_accumulation == 0 or number == len(examples)
        if update_now:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        records.append(float(loss.detach().cpu()))
        if update_now:
            print(json.dumps({"arm": args.arm, "seed": args.seed, "examples": number, "updates": updates, "loss": records[-1]}), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    summary = {
        "config": {key: [str(x) for x in value] if isinstance(value, list) else str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "func"},
        "matched_examples": len(examples),
        "optimizer_updates": updates,
        "mean_loss": mean(records),
        "final_100_mean_loss": mean(records[-100:]),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(required=True)
    collect_parser = subs.add_parser("collect")
    collect_parser.add_argument("--data", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--server-url", required=True)
    collect_parser.add_argument("--arm", choices=ARMS, required=True)
    collect_parser.add_argument("--offset", type=int, required=True)
    collect_parser.add_argument("--count", type=int, required=True)
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--max-steps", type=int, default=7)
    collect_parser.add_argument("--step-tokens", type=int, default=80)
    collect_parser.add_argument("--temperature", type=float, default=0.8)
    collect_parser.add_argument("--top-p", type=float, default=0.95)
    collect_parser.add_argument("--threshold", type=float, default=0.0)
    collect_parser.set_defaults(func=collect)
    train_parser = subs.add_parser("train")
    train_parser.add_argument("--inputs", nargs=3, type=Path, required=True)
    train_parser.add_argument("--arm", choices=ARMS, required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    train_parser.add_argument("--initial-adapter", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--examples", type=int, default=1688)
    train_parser.add_argument("--max-length", type=int, default=512)
    train_parser.add_argument("--head-tokens", type=int, default=192)
    train_parser.add_argument("--learning-rate", type=float, default=5e-6)
    train_parser.add_argument("--gradient-accumulation", type=int, default=32)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.set_defaults(func=train)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
