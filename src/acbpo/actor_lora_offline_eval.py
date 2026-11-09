from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .actor_lora_experiment import PolicyExample, build_choice_messages, middle_truncate


def summarize_records(records: list[dict[str, float | bool]]) -> dict[str, float]:
    if not records:
        raise ValueError("records must not be empty")

    def mean(key: str) -> float:
        return float(np.mean([float(record[key]) for record in records]))

    positive = [record for record in records if float(record["advantage"]) >= 0.0]
    negative = [record for record in records if float(record["advantage"]) < 0.0]
    before = mean("surrogate_before")
    after = mean("surrogate_after")
    return {
        "examples": float(len(records)),
        "surrogate_before": before,
        "surrogate_after": after,
        "surrogate_improvement": after - before,
        "mean_kl_new_to_base": mean("kl_new_to_base"),
        "mean_total_variation": mean("total_variation"),
        "top_action_change_rate": mean("top_action_changed"),
        "positive_advantage_probability_change": float(
            np.mean([float(x["selected_probability_change"]) for x in positive])
        )
        if positive
        else float("nan"),
        "negative_advantage_probability_change": float(
            np.mean([float(x["selected_probability_change"]) for x in negative])
        )
        if negative
        else float("nan"),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-examples", type=int, default=128)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--rho-clip", type=float, default=10.0)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument("--log-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_length <= args.head_tokens:
        raise ValueError("max-length must exceed head-tokens")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    examples = [
        PolicyExample(**item)
        for item in json.loads(args.prepared_data.read_text())["examples"]
    ]
    random.Random(args.seed).shuffle(examples)
    training = examples[: args.training_examples]
    evaluation = examples[
        args.training_examples : args.training_examples + args.eval_examples
    ]
    if not training or not evaluation:
        raise RuntimeError("training and evaluation slices must both be non-empty")

    training_advantages = np.asarray(
        [example.advantage for example in training], dtype=np.float64
    )
    advantage_mean = float(training_advantages.mean())
    advantage_std = max(float(training_advantages.std()), 1e-6)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
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
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    records: list[dict[str, float | bool]] = []
    with torch.inference_mode():
        for index, example in enumerate(evaluation, start=1):
            prompt = tokenizer.apply_chat_template(
                build_choice_messages(example.messages, example.action_names),
                tokenize=False,
                add_generation_prompt=True,
            )
            token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            token_ids = middle_truncate(token_ids, args.max_length, args.head_tokens)
            input_ids = torch.tensor([token_ids], device="cuda", dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            allowed = torch.tensor(example.allowed_mask, device="cuda", dtype=torch.bool)

            with model.disable_adapter():
                base_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[0, -1, label_ids_tensor].float()
            new_logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits[0, -1, label_ids_tensor].float()
            base_logp = torch.log_softmax(
                (base_logits / args.temperature).masked_fill(~allowed, -torch.inf),
                dim=0,
            )
            new_logp = torch.log_softmax(
                (new_logits / args.temperature).masked_fill(~allowed, -torch.inf),
                dim=0,
            )
            base_probability = base_logp.exp()
            new_probability = new_logp.exp()
            normalized_advantage = float(
                np.clip(
                    (example.advantage - advantage_mean) / advantage_std,
                    -args.advantage_clip,
                    args.advantage_clip,
                )
            )
            weight = min(example.joint_ratio, args.rho_clip) * normalized_advantage
            selected = example.selected_action
            records.append(
                {
                    "advantage": normalized_advantage,
                    "surrogate_before": weight * float(base_logp[selected].cpu()),
                    "surrogate_after": weight * float(new_logp[selected].cpu()),
                    "selected_probability_change": float(
                        (new_probability[selected] - base_probability[selected]).cpu()
                    ),
                    "kl_new_to_base": float(
                        torch.sum(
                            new_probability[allowed]
                            * (new_logp[allowed] - base_logp[allowed])
                        ).cpu()
                    ),
                    "total_variation": float(
                        (0.5 * torch.sum(torch.abs(new_probability - base_probability))).cpu()
                    ),
                    "top_action_changed": bool(
                        torch.argmax(new_probability) != torch.argmax(base_probability)
                    ),
                }
            )
            if index % args.log_every == 0 or index == len(evaluation):
                print(json.dumps({"scored": index, **summarize_records(records)}), flush=True)

    summary: dict[str, Any] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "advantage_mean_from_training_slice": advantage_mean,
        "advantage_std_from_training_slice": advantage_std,
        **summarize_records(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
