from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from .actor_lora_experiment import middle_truncate


FINAL_PATTERN = re.compile(r"(?i)\bFINAL\s*:\s*([^\n]+)")
NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")


@dataclass(frozen=True)
class GSM8KExample:
    example_id: int
    question: str
    reference_answer: str


@dataclass(frozen=True)
class CandidateStep:
    text: str
    actor_logprob: float
    critic_score: float
    critic_probabilities: tuple[float, float, float]


def reference_answer(text: str) -> str:
    if "####" not in text:
        raise ValueError("GSM8K reference answer is missing the #### delimiter")
    return text.rsplit("####", 1)[1].strip()


def extract_final_answer(text: str) -> str | None:
    matches = FINAL_PATTERN.findall(text)
    if not matches:
        return None
    value = matches[-1].strip().rstrip(".。")
    numeric = NUMBER_PATTERN.findall(value)
    return numeric[-1] if numeric else value


def _number(value: str) -> Decimal | None:
    cleaned = value.strip().replace(",", "").replace("$", "")
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", 1)
            return Decimal(numerator.strip()) / Decimal(denominator.strip())
        return Decimal(cleaned)
    except (InvalidOperation, ZeroDivisionError):
        return None


def answers_match(prediction: str | None, reference: str) -> bool:
    if prediction is None:
        return False
    predicted_number = _number(prediction)
    reference_number = _number(reference)
    if predicted_number is not None and reference_number is not None:
        return predicted_number == reference_number
    normalize = lambda value: " ".join(value.lower().strip().split())
    return normalize(prediction) == normalize(reference)


def combined_scores(
    actor_logprobs: Sequence[float], critic_scores: Sequence[float], beta: float
) -> list[float]:
    if len(actor_logprobs) != len(critic_scores):
        raise ValueError("actor and critic score counts differ")
    if not actor_logprobs:
        raise ValueError("candidate set is empty")
    critic = np.asarray(critic_scores, dtype=np.float64)
    scale = float(critic.std())
    normalized = np.zeros_like(critic) if scale < 1e-8 else (critic - critic.mean()) / scale
    return [float(logp + beta * q) for logp, q in zip(actor_logprobs, normalized)]


def select_candidate(
    candidates: Sequence[CandidateStep], strategy: str, beta: float
) -> int:
    if not candidates:
        raise ValueError("candidate set is empty")
    if strategy == "single":
        return 0
    if strategy == "actor":
        scores = [item.actor_logprob for item in candidates]
    elif strategy == "critic":
        scores = [item.critic_score for item in candidates]
    elif strategy == "cgcs":
        scores = combined_scores(
            [item.actor_logprob for item in candidates],
            [item.critic_score for item in candidates],
            beta,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    return max(range(len(scores)), key=scores.__getitem__)


def load_gsm8k(path: Path, *, offset: int, count: int) -> list[GSM8KExample]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(GSM8KExample(index, item["question"].strip(), reference_answer(item["answer"])))
    if offset < 0 or offset > len(rows):
        raise ValueError("offset is outside the dataset")
    return rows[offset : offset + count if count > 0 else None]


def actor_messages(question: str, steps: Sequence[str]) -> list[dict[str, str]]:
    prefix = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
    if not prefix:
        prefix = "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Solve the math problem one step at a time. Produce exactly one concise next "
                "reasoning step. Do not repeat previous steps. When the solution is complete, "
                "output exactly FINAL: <numeric answer>."
            ),
        },
        {
            "role": "user",
            "content": f"Problem:\n{question}\n\nSolution so far:\n{prefix}\n\nNext step:",
        },
    ]


def critic_messages(question: str, steps: Sequence[str], candidate: str) -> list[dict[str, str]]:
    prefix = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
    if not prefix:
        prefix = "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Judge one proposed next step in a math solution. Output exactly one letter: "
                "A = incorrect, B = neutral or no progress, C = correct and useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\nSolution so far:\n{prefix}\n\n"
                f"Proposed next step:\n{candidate}\n\nRating:"
            ),
        },
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-critic GSM8K step rollout")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--strategy", choices=("single", "actor", "critic", "cgcs", "all"), required=True
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=108001)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.candidates < 1 or args.max_steps < 1:
        raise ValueError("candidates and max-steps must be positive")
    if args.max_length <= args.head_tokens:
        raise ValueError("max-length must exceed head-tokens")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    label_ids = [tokenizer(x, add_special_tokens=False).input_ids for x in ("A", "B", "C")]
    if any(len(ids) != 1 for ids in label_ids):
        raise RuntimeError(f"critic labels are not single tokens: {label_ids}")
    label_ids_tensor = torch.tensor([ids[0] for ids in label_ids], device="cuda")

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

    def prompt_ids(messages: list[dict[str, str]]) -> list[int]:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        return middle_truncate(ids, args.max_length, args.head_tokens)

    def generate_candidates(question: str, steps: Sequence[str]) -> tuple[list[str], list[float]]:
        inputs = torch.tensor([prompt_ids(actor_messages(question, steps))], device="cuda")
        attention_mask = torch.ones_like(inputs)
        with torch.no_grad():
            output = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.candidates,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output.sequences[:, inputs.shape[1] :]
        transition = model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        texts: list[str] = []
        logprobs: list[float] = []
        for token_row, score_row in zip(generated, transition):
            mask = token_row != tokenizer.pad_token_id
            if not bool(mask.any()):
                mask = torch.ones_like(token_row, dtype=torch.bool)
            text = tokenizer.decode(token_row[mask], skip_special_tokens=True).strip()
            texts.append(text or "(empty step)")
            logprobs.append(float(score_row[mask].float().mean().cpu()))
        return texts, logprobs

    def score_candidates(question: str, steps: Sequence[str], texts: Sequence[str]):
        encoded = [{"input_ids": prompt_ids(critic_messages(question, steps, text))} for text in texts]
        batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        batch = {key: value.to("cuda") for key, value in batch.items()}
        with torch.no_grad():
            logits = model(**batch, logits_to_keep=1).logits[:, -1, label_ids_tensor].float()
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
        scores = [-float(row[0]) + float(row[2]) for row in probabilities]
        return scores, probabilities

    args.output.parent.mkdir(parents=True, exist_ok=True)
    strategy_names = (
        ("single", "actor", "critic", "cgcs") if args.strategy == "all" else (args.strategy,)
    )
    completed: dict[str, list[dict[str, Any]]] = {name: [] for name in strategy_names}

    def result_item(
        example: GSM8KExample, steps: list[str], trace: list[dict[str, Any]]
    ) -> dict[str, Any]:
        prediction = next(
            (answer for answer in (extract_final_answer(step) for step in reversed(steps)) if answer),
            None,
        )
        return {
            "example_id": example.example_id,
            "question": example.question,
            "reference_answer": example.reference_answer,
            "prediction": prediction,
            "correct": answers_match(prediction, example.reference_answer),
            "steps": steps,
            "trace": trace,
        }

    def write_results() -> None:
        config = vars(args) | {"data": str(args.data), "output": str(args.output)}
        summaries = {
            name: {
                "accuracy": mean(float(row["correct"]) for row in rows) if rows else math.nan,
                "answered": sum(row["prediction"] is not None for row in rows),
                "examples": rows,
            }
            for name, rows in completed.items()
        }
        payload = (
            {"config": config, "strategies": summaries}
            if args.strategy == "all"
            else {"config": config, **summaries[args.strategy]}
        )
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    for example in examples:
        tracks = {
            name: {"steps": [], "trace": [], "done": False} for name in strategy_names
        }
        for step_index in range(args.max_steps):
            groups: dict[tuple[str, ...], list[str]] = {}
            for name, track in tracks.items():
                if not track["done"]:
                    groups.setdefault(tuple(track["steps"]), []).append(name)
            if not groups:
                break
            for group_index, (prefix, names) in enumerate(groups.items()):
                torch.manual_seed(
                    args.seed + example.example_id * 1009 + step_index * 97 + group_index
                )
                texts, actor_logprobs = generate_candidates(example.question, prefix)
                critic_scores, probabilities = score_candidates(example.question, prefix, texts)
                candidates = [
                    CandidateStep(text, logp, score, tuple(float(x) for x in probs))
                    for text, logp, score, probs in zip(
                        texts, actor_logprobs, critic_scores, probabilities
                    )
                ]
                serialized = [asdict(item) for item in candidates]
                for name in names:
                    selected = select_candidate(candidates, name, args.beta)
                    track = tracks[name]
                    track["trace"].append(
                        {"step": step_index, "candidates": serialized, "selected": selected}
                    )
                    track["steps"].append(candidates[selected].text)
                    track["done"] = extract_final_answer(candidates[selected].text) is not None

        status = {}
        for name, track in tracks.items():
            item = result_item(example, track["steps"], track["trace"])
            completed[name].append(item)
            status[name] = {
                "correct": item["correct"],
                "prediction": item["prediction"],
                "steps": len(item["steps"]),
            }
        write_results()
        print(
            json.dumps(
                {
                    "example_id": example.example_id,
                    "reference": example.reference_answer,
                    "strategies": status,
                }
            ),
            flush=True,
        )

    print(
        json.dumps(
            {
                name: {
                    "examples": len(rows),
                    "accuracy": mean(float(item["correct"]) for item in rows)
                    if rows
                    else math.nan,
                }
                for name, rows in completed.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
