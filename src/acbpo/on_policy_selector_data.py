from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from .actor_lora_experiment import middle_truncate
from .math_cgcs_ppo import NUMERIC_ANSWER, load_unique_prm_problems
from .math_rollout_experiment import answers_match, extract_final_answer, load_gsm8k


@dataclass(frozen=True)
class SelectorExample:
    example_id: str
    question: str
    reference_answer: str


def load_examples(
    path: Path, *, data_format: str, offset: int, count: int, numeric_only: bool = False
) -> list[SelectorExample]:
    if data_format == "gsm8k":
        source = load_gsm8k(path, offset=offset, count=count)
        return [
            SelectorExample(f"gsm8k-{item.example_id}", item.question, item.reference_answer)
            for item in source
        ]
    if data_format == "prm":
        source = load_unique_prm_problems(path)
        if numeric_only:
            source = [
                item
                for item in source
                if NUMERIC_ANSWER.fullmatch(item.answer.replace(r"\$", "$"))
            ]
        source = source[offset : offset + count]
        return [
            SelectorExample(
                item.problem_id,
                item.question,
                item.answer.replace(r"\$", "$"),
            )
            for item in source
        ]
    raise ValueError(f"unknown data format: {data_format}")


def solution_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly two lines and fewer than 80 tokens total. Do not restate "
                "the problem. Line 1 must be a compact calculation. Line 2 must be "
                "FINAL: <numeric answer>."
            ),
        },
        {"role": "user", "content": question},
    ]


def generation_batch_sizes(total: int, maximum: int) -> list[int]:
    if total < 1 or maximum < 1:
        raise ValueError("generation batch sizes must be positive")
    return [min(maximum, total - start) for start in range(0, total, maximum)]


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    rewards = [
        float(candidate["reward"])
        for record in records
        for candidate in record["candidates"]
    ]
    mixed = [
        record
        for record in records
        if len({int(candidate["reward"]) for candidate in record["candidates"]}) > 1
    ]
    all_correct = [
        record
        for record in records
        if record["candidates"]
        and all(int(candidate["reward"]) == 1 for candidate in record["candidates"])
    ]
    all_wrong = [
        record
        for record in records
        if record["candidates"]
        and all(int(candidate["reward"]) == 0 for candidate in record["candidates"])
    ]
    unique_counts = [
        len({" ".join(str(candidate["text"]).split()) for candidate in record["candidates"]})
        for record in records
    ]
    answered = [
        candidate
        for record in records
        for candidate in record["candidates"]
        if candidate.get("prediction") is not None
    ]
    fully_answered = [
        record
        for record in records
        if record["candidates"]
        and all(candidate.get("prediction") is not None for candidate in record["candidates"])
    ]
    return {
        "states": len(records),
        "candidates": len(rewards),
        "candidate_accuracy": mean(rewards) if rewards else math.nan,
        "pass_at_k": (
            mean(any(int(candidate["reward"]) == 1 for candidate in row["candidates"]) for row in records)
            if records
            else math.nan
        ),
        "mixed_states": len(mixed),
        "mixed_state_fraction": len(mixed) / len(records) if records else math.nan,
        "all_correct_states": len(all_correct),
        "all_wrong_states": len(all_wrong),
        "mean_unique_candidates": mean(unique_counts) if unique_counts else math.nan,
        "answered_candidates": len(answered),
        "answer_rate": len(answered) / len(rewards) if rewards else math.nan,
        "fully_answered_states": len(fully_answered),
        "fully_answered_state_fraction": (
            len(fully_answered) / len(records) if records else math.nan
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen-actor candidate sets with verifier rewards"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-format", choices=("gsm8k", "prm"), default="gsm8k")
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=0,
        help="candidate sequences per generate call; zero uses all candidates",
    )
    parser.add_argument(
        "--problem-batch-size",
        type=int,
        default=1,
        help="problem prompts per generate call",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    parser.add_argument("--head-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=73001)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.count < 1 or args.candidates < 2:
        raise ValueError("count must be positive and candidates must be at least two")
    if args.generation_batch_size < 0 or args.problem_batch_size < 1:
        raise ValueError("generation batch must be non-negative and problem batch positive")
    if args.max_prompt_tokens <= args.head_tokens:
        raise ValueError("max-prompt-tokens must exceed head-tokens")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_examples(
        args.data,
        data_format=args.data_format,
        offset=args.offset,
        count=args.count,
        numeric_only=args.numeric_only,
    )
    if len(examples) != args.count:
        raise RuntimeError(f"requested {args.count} examples but found {len(examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if args.output.exists():
        with args.output.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    completed_ids = {str(row["example_id"]) for row in records}

    def write_summary() -> None:
        payload = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            **summarize_records(records),
        }
        args.summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for problem_start in range(0, len(examples), args.problem_batch_size):
        example_batch = examples[problem_start : problem_start + args.problem_batch_size]
        if all(example.example_id in completed_ids for example in example_batch):
            continue
        prompt_rows = []
        for example in example_batch:
            prompt_text = tokenizer.apply_chat_template(
                solution_messages(example.question),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = middle_truncate(
                tokenizer(prompt_text, add_special_tokens=False).input_ids,
                args.max_prompt_tokens,
                args.head_tokens,
            )
            prompt_rows.append({"input_ids": prompt_ids})
        padded = tokenizer.pad(prompt_rows, padding=True, return_tensors="pt")
        inputs = padded["input_ids"].to("cuda")
        attention_mask = padded["attention_mask"].to("cuda")
        candidates_by_example: list[list[dict[str, Any]]] = [
            [] for _ in example_batch
        ]
        generation_batch = args.generation_batch_size or args.candidates
        generated_so_far = 0
        for candidates_per_problem in generation_batch_sizes(
            args.candidates, generation_batch
        ):
            # Fixed source-index batches retain deterministic resume behavior.
            # A partially written batch is regenerated in full and only missing
            # records are appended below.
            if len(example_batch) == 1:
                batch_key = example_batch[0].example_id
            else:
                batch_key = "|".join(example.example_id for example in example_batch)
            seed_key = (
                batch_key
                if len(example_batch) == 1 and generated_so_far == 0
                else f"{batch_key}:{generated_so_far}"
            )
            batch_seed = int(
                hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16
            )
            torch.manual_seed(args.seed + batch_seed)
            with torch.no_grad():
                output = model.generate(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    num_return_sequences=candidates_per_problem,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = output.sequences[:, inputs.shape[1] :]
            transition = model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )
            for example_index, example in enumerate(example_batch):
                row_start = example_index * candidates_per_problem
                row_end = row_start + candidates_per_problem
                for token_row, score_row in zip(
                    generated[row_start:row_end], transition[row_start:row_end]
                ):
                    mask = token_row != tokenizer.pad_token_id
                    if not bool(mask.any()):
                        mask = torch.ones_like(token_row, dtype=torch.bool)
                    text = tokenizer.decode(
                        token_row[mask], skip_special_tokens=True
                    ).strip()
                    prediction = extract_final_answer(text)
                    candidates_by_example[example_index].append(
                        {
                            "text": text,
                            "prediction": prediction,
                            "reward": int(
                                answers_match(prediction, example.reference_answer)
                            ),
                            "actor_logprob": float(
                                score_row[mask].float().mean().cpu()
                            ),
                            "token_count": int(mask.sum().item()),
                        }
                    )
            generated_so_far += candidates_per_problem
        previous_count = len(records)
        with args.output.open("a", encoding="utf-8") as handle:
            for example, candidates in zip(example_batch, candidates_by_example):
                if example.example_id in completed_ids:
                    continue
                record = {
                    "example_id": example.example_id,
                    "problem_id": example.example_id,
                    "question": example.question,
                    "reference_answer": example.reference_answer,
                    "max_new_tokens": args.max_new_tokens,
                    "candidates": candidates,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                completed_ids.add(example.example_id)
        if len(records) // args.log_every > previous_count // args.log_every:
            write_summary()
            print(json.dumps(summarize_records(records)), flush=True)

    write_summary()
    print(json.dumps(summarize_records(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
