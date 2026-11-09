from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean, median
from typing import Any, Sequence

import numpy as np

from .arc_task import (
    arc_step_messages,
    arc_terminal_result,
    extract_arc_final_answer,
    load_arc_challenge,
)
from .drop_task import (
    drop_step_messages,
    drop_terminal_result,
    extract_drop_final_answer,
    load_drop,
)
from .actor_lora_experiment import middle_truncate
from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import extract_final_answer, load_gsm8k
from .math_step_action_pilot import clean_step, step_messages, terminal_result


ARMS = (
    "burnin",
    "k1",
    "random_uniform",
    "entropy_uniform",
    "value_gate_uniform",
    "random_value",
    "value_guided",
    "best_of_n",
    "entropy_value",
    "value_value",
)

SEARCH_MISMATCH_ARMS = ("best_of_n", "entropy_value", "value_value")
TASKS = ("math", "arc_challenge", "drop")


def _load_problems(args: argparse.Namespace):
    if args.task == "math":
        return load_gsm8k(args.data, offset=args.offset, count=args.count)
    if args.task == "arc_challenge":
        return load_arc_challenge(args.data, offset=args.offset, count=args.count)
    if args.task == "drop":
        return load_drop(args.data, offset=args.offset, count=args.count)
    raise ValueError(f"unknown task: {args.task}")


def _task_messages(task: str, question: str, prefix: Sequence[str]):
    if task == "math":
        return step_messages(question, prefix)
    if task == "arc_challenge":
        return arc_step_messages(question, prefix)
    if task == "drop":
        return drop_step_messages(question, prefix)
    raise ValueError(f"unknown task: {task}")


def _task_final_answer(task: str, text: str) -> str | None:
    if task == "math":
        return extract_final_answer(text)
    if task == "arc_challenge":
        return extract_arc_final_answer(text)
    if task == "drop":
        return extract_drop_final_answer(text)
    raise ValueError(f"unknown task: {task}")


def _task_terminal_result(
    task: str, steps: Sequence[str], reference: str
) -> dict[str, Any]:
    if task == "math":
        return terminal_result(steps, reference)
    if task == "arc_challenge":
        return arc_terminal_result(steps, reference)
    if task == "drop":
        return drop_terminal_result(steps, reference)
    raise ValueError(f"unknown task: {task}")


def next_generation_token_limit(
    *, step_tokens: int, used_tokens: int, max_completion_tokens: int | None
) -> int | None:
    """Return the next per-step token ceiling, or ``None`` when exhausted."""
    if step_tokens <= 0 or used_tokens < 0:
        raise ValueError("step-tokens must be positive and used-tokens non-negative")
    if max_completion_tokens is None:
        return step_tokens
    if max_completion_tokens <= 0:
        raise ValueError("max-completion-tokens must be positive when provided")
    remaining = max_completion_tokens - used_tokens
    return None if remaining <= 0 else min(step_tokens, remaining)


def collection_frozen_keys(arm: str) -> tuple[str, ...]:
    keys = [
        "task",
        "arm",
        "offset",
        "count",
        "seed",
        "epsilon",
        "entropy_threshold",
        "value_threshold",
        "threshold_source",
        "sampling_temperature",
    ]
    if arm in SEARCH_MISMATCH_ARMS:
        keys.extend(
            [
                "model",
                "actor_adapter",
                "value_head",
                "max_steps",
                "step_tokens",
                "max_prompt_tokens",
                "head_tokens",
                "candidates",
                "beta",
                "value_scale_floor",
                "expansion_rate",
            ]
        )
    return tuple(keys)


def value_selector_distribution(
    values: Sequence[float], *, epsilon: float, beta: float = 1.0,
    scale_floor: float = 0.05,
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("values must be a non-empty vector")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    if beta < 0.0 or scale_floor <= 0.0:
        raise ValueError("beta must be non-negative and scale-floor positive")
    normalized = (scores - scores.mean()) / max(float(scores.std()), scale_floor)
    shifted = beta * normalized
    shifted -= shifted.max()
    softmax = np.exp(shifted)
    softmax /= softmax.sum()
    return (1.0 - epsilon) * softmax + epsilon / len(scores)


def matched_threshold(values: Sequence[float], *, low: bool) -> float:
    if not values:
        raise ValueError("cannot fit a threshold without observations")
    # The median is locked by the protocol. `low` is explicit at call sites and
    # retained here to make threshold direction auditable in metadata.
    _ = low
    return float(median(float(value) for value in values))


def matched_expansion_threshold(
    values: Sequence[float], *, low: bool, expansion_rate: float
) -> float:
    """Return the deterministic threshold for a requested expansion fraction.

    The threshold is fitted only on a preceding rollout.  Linear interpolation
    is deterministic and ties are retained as a measured realized-budget
    deviation rather than broken using evaluation outcomes.
    """
    if not values:
        raise ValueError("cannot fit a threshold without observations")
    if not 0.0 < expansion_rate < 1.0:
        raise ValueError("expansion_rate must be strictly between zero and one")
    quantile = expansion_rate if low else 1.0 - expansion_rate
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def should_expand(
    arm: str, *, entropy: float, post_value: float, threshold: float | None,
    random_draw: float, expansion_rate: float,
) -> bool:
    if arm in {"burnin", "k1"}:
        return False
    if arm == "best_of_n":
        return True
    if arm in {"random_uniform", "random_value"}:
        return random_draw < expansion_rate
    if threshold is None:
        raise ValueError(f"{arm} requires a fitted threshold")
    if arm in {"entropy_uniform", "entropy_value"}:
        return entropy >= threshold
    if arm in {"value_gate_uniform", "value_guided", "value_value"}:
        return post_value <= threshold
    raise ValueError(f"unknown arm: {arm}")


def fit_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    states = [state for row in payload["examples"] for state in row["states"]]
    if not states:
        raise ValueError("rollout contains no states")
    return {
        "entropy": matched_threshold(
            [float(row["candidate_zero_entropy"]) for row in states], low=False
        ),
        "value": matched_threshold(
            [float(row["candidate_zero_post_value"]) for row in states], low=True
        ),
    }


def _model_and_head(
    model_name: str,
    *,
    adapter: Path | None,
    value_head_path: Path | None,
    trainable: bool,
    lora_rank: int = 32,
    lora_alpha: int = 64,
):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if adapter is None:
        if trainable:
            model = get_peft_model(
                base,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=0.0,
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
        else:
            model = base
    else:
        model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
    head = torch.nn.Linear(
        int(model.config.hidden_size), 1, device="cuda", dtype=torch.float32
    )
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    if value_head_path is not None:
        head.load_state_dict(torch.load(value_head_path, map_location="cuda"))
    return tokenizer, model, head


def _prompt_ids(tokenizer, question: str, prefix: Sequence[str], args) -> list[int]:
    ids = tokenizer(
        qwen_chat_prompt(_task_messages(args.task, question, prefix)),
        add_special_tokens=False,
    ).input_ids
    return middle_truncate(ids, args.max_prompt_tokens, args.head_tokens)


def _state_value(model, head, input_ids: list[int]) -> float:
    import torch

    tensor = torch.tensor([input_ids], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        output = model(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            output_hidden_states=True,
            use_cache=False,
        )
        value = head(output.hidden_states[-1][0, -1].float()).squeeze()
    return float(value.cpu())


def _completion_stats(
    model,
    prompt: list[int],
    completion: list[int],
    *,
    sampling_temperature: float = 1.0,
) -> tuple[list[float], float]:
    import torch

    full = torch.tensor([prompt + completion], device="cuda", dtype=torch.long)
    with torch.inference_mode():
        logits = model(
            input_ids=full, attention_mask=torch.ones_like(full), use_cache=False
        ).logits[0].float()
        positions = torch.arange(
            len(prompt) - 1, len(prompt) + len(completion) - 1, device="cuda"
        )
        selected_logits = logits[positions] / sampling_temperature
        log_probabilities = torch.log_softmax(selected_logits, dim=-1)
        tokens = torch.tensor(completion, device="cuda", dtype=torch.long)
        token_logprobs = log_probabilities[
            torch.arange(len(completion), device="cuda"), tokens
        ]
        probabilities = torch.softmax(selected_logits, dim=-1)
        entropy = -(probabilities * log_probabilities).sum(dim=-1).mean()
    return token_logprobs.cpu().tolist(), float(entropy.cpu())


def _generate_candidate(model, tokenizer, prompt: list[int], *, seed: int, args) -> dict[str, Any]:
    import torch

    tensor = torch.tensor([prompt], device="cuda", dtype=torch.long)
    newline = tokenizer("\n", add_special_tokens=False).input_ids
    eos = [tokenizer.eos_token_id]
    if len(newline) == 1:
        eos.append(newline[0])
    torch.manual_seed(seed)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=tensor,
            attention_mask=torch.ones_like(tensor),
            do_sample=True,
            temperature=args.sampling_temperature,
            top_p=1.0,
            max_new_tokens=args.step_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos,
        )[0, len(prompt) :]
    completion = generated.tolist()
    if not completion:
        completion = [tokenizer.eos_token_id]
    text = clean_step(tokenizer.decode(completion, skip_special_tokens=True))
    logprobs, entropy = _completion_stats(
        model,
        prompt,
        completion,
        sampling_temperature=args.sampling_temperature,
    )
    return {
        "text": text,
        "completion_ids": completion,
        "old_token_logprobs": logprobs,
        "mean_token_entropy": entropy,
    }


def _fit_current_thresholds(model, head, tokenizer, path: Path, args) -> dict[str, float]:
    payload = json.loads(path.read_text())
    entropies: list[float] = []
    values: list[float] = []
    with_candidates = 0
    for example in payload["examples"]:
        question = str(example["question"])
        for state in example["states"]:
            candidates = state.get("candidates", [])
            if not candidates:
                continue
            candidate = candidates[0]
            prompt = list(state["prompt_ids"])
            _, entropy = _completion_stats(
                model,
                prompt,
                list(candidate["completion_ids"]),
                sampling_temperature=args.sampling_temperature,
            )
            post_prompt = _prompt_ids(
                tokenizer,
                question,
                [*state["prefix"], str(candidate["text"])],
                args,
            )
            entropies.append(entropy)
            values.append(_state_value(model, head, post_prompt))
            with_candidates += 1
    if not with_candidates:
        raise ValueError(f"threshold source {path} has no candidate states")
    if args.arm in {"entropy_value", "value_value"}:
        return {
            "entropy": matched_expansion_threshold(
                entropies, low=False, expansion_rate=args.expansion_rate
            ),
            "value": matched_expansion_threshold(
                values, low=True, expansion_rate=args.expansion_rate
            ),
        }
    return {
        "entropy": matched_threshold(entropies, low=False),
        "value": matched_threshold(values, low=True),
    }


def collect(args: argparse.Namespace) -> None:
    import torch

    if args.arm not in ARMS:
        raise ValueError(f"unknown arm: {args.arm}")
    if args.sampling_temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, model, head = _model_and_head(
        args.model,
        adapter=args.actor_adapter,
        value_head_path=args.value_head,
        trainable=False,
    )
    model.eval()
    head.eval()
    if args.threshold_source is not None:
        fitted = _fit_current_thresholds(
            model, head, tokenizer, args.threshold_source, args
        )
        if args.arm in {"entropy_uniform", "entropy_value"}:
            args.entropy_threshold = fitted["entropy"]
        elif args.arm in {"value_gate_uniform", "value_guided", "value_value"}:
            args.value_threshold = fitted["value"]
        print(json.dumps({"threshold_source": str(args.threshold_source), **fitted}), flush=True)
    problems = _load_problems(args)
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        observed = previous["config"]
        frozen_keys = collection_frozen_keys(args.arm)

        def normalized_argument(key: str):
            value = getattr(args, key)
            return str(value) if isinstance(value, Path) else value

        frozen = tuple(normalized_argument(key) for key in frozen_keys)
        prior = tuple(
            observed.get(key, 1.0 if key == "sampling_temperature" else None)
            for key in frozen_keys
        )
        if prior != frozen:
            raise ValueError(f"rollout resume mismatch: {prior} != {frozen}")
        existing = previous["examples"]
    completed = {int(row["example_id"]) for row in existing}
    started = time.time()

    for problem in problems:
        if problem.example_id in completed:
            continue
        prefix: list[str] = []
        states: list[dict[str, Any]] = []
        for step_index in range(args.max_steps):
            prompt = _prompt_ids(tokenizer, problem.question, prefix, args)
            pre_value = _state_value(model, head, prompt)
            seed_base = args.seed + problem.example_id * 100_003 + step_index * 10_007
            candidate_zero = _generate_candidate(
                model, tokenizer, prompt, seed=seed_base, args=args
            )
            zero_post_prompt = _prompt_ids(
                tokenizer, problem.question, [*prefix, candidate_zero["text"]], args
            )
            candidate_zero["post_value"] = _state_value(model, head, zero_post_prompt)
            draw = random.Random(seed_base + 7_000_019).random()
            gate_threshold = (
                args.entropy_threshold
                if args.arm in {"entropy_uniform", "entropy_value"}
                else args.value_threshold
            )
            expanded = should_expand(
                args.arm,
                entropy=float(candidate_zero["mean_token_entropy"]),
                post_value=float(candidate_zero["post_value"]),
                threshold=gate_threshold,
                random_draw=draw,
                expansion_rate=args.expansion_rate,
            )
            candidates = [candidate_zero]
            if expanded:
                for candidate_index in range(1, args.candidates):
                    candidate = _generate_candidate(
                        model,
                        tokenizer,
                        prompt,
                        seed=seed_base + candidate_index,
                        args=args,
                    )
                    post_prompt = _prompt_ids(
                        tokenizer, problem.question, [*prefix, candidate["text"]], args
                    )
                    candidate["post_value"] = _state_value(model, head, post_prompt)
                    candidates.append(candidate)
            if args.arm in {
                "random_value",
                "value_guided",
                "best_of_n",
                "entropy_value",
                "value_value",
            } and expanded:
                probabilities = value_selector_distribution(
                    [float(row["post_value"]) for row in candidates],
                    epsilon=args.epsilon,
                    beta=args.beta,
                    scale_floor=args.value_scale_floor,
                )
                choice = int(
                    np.random.default_rng(seed_base + 9_000_031).choice(
                        len(candidates), p=probabilities
                    )
                )
            elif expanded:
                probabilities = np.full(len(candidates), 1.0 / len(candidates))
                choice = random.Random(seed_base + 9_000_031).randrange(len(candidates))
            else:
                probabilities = np.ones(1)
                choice = 0
            state = {
                "step_index": step_index,
                "prefix": list(prefix),
                "prompt_ids": prompt,
                "pre_value": pre_value,
                "candidate_zero_entropy": float(candidate_zero["mean_token_entropy"]),
                "candidate_zero_post_value": float(candidate_zero["post_value"]),
                "expanded": expanded,
                "candidates": candidates,
                "selector_probabilities": probabilities.tolist(),
                "choice": choice,
                "selection_probability": float(probabilities[choice]),
            }
            states.append(state)
            prefix.append(str(candidates[choice]["text"]))
            if _task_final_answer(args.task, prefix[-1]) is not None:
                break
        result = _task_terminal_result(args.task, prefix, problem.reference_answer)
        for state in states:
            state["reward"] = float(result["correct"])
        existing.append(
            {
                "example_id": problem.example_id,
                "source_id": getattr(problem, "source_id", None),
                "question": problem.question,
                "reference_answer": problem.reference_answer,
                "steps": prefix,
                "states": states,
                "result": result,
            }
        )
        flat = [state for row in existing for state in row["states"]]
        expanded_states = [row for row in flat if row["expanded"]]
        unique_candidate_counts = [
            len({candidate["text"] for candidate in row["candidates"]})
            for row in expanded_states
        ]
        payload = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "func"
            },
            "summary": {
                "examples": len(existing),
                "accuracy": mean(float(row["result"]["correct"]) for row in existing),
                "answered_fraction": mean(float(row["result"]["answered"]) for row in existing),
                "states": len(flat),
                "expansion_rate": mean(float(row["expanded"]) for row in flat),
                "generated_candidates": sum(len(row["candidates"]) for row in flat),
                "mean_unique_candidates_when_expanded": (
                    mean(unique_candidate_counts) if unique_candidate_counts else None
                ),
                "identical_expanded_fraction": (
                    mean(count == 1 for count in unique_candidate_counts)
                    if unique_candidate_counts
                    else None
                ),
                **(
                    {
                        "mean_f1": mean(
                            float(row["result"]["f1"]) for row in existing
                        )
                    }
                    if args.task == "drop"
                    else {}
                ),
            },
            "elapsed_seconds": time.time() - started,
            "examples": existing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"example_id": problem.example_id, **payload["summary"]}), flush=True)


def train(args: argparse.Namespace) -> None:
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = json.loads(args.rollout.read_text())
    sampling_temperature = float(
        payload.get("config", {}).get("sampling_temperature", 1.0)
    )
    if sampling_temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")
    states = [state for row in payload["examples"] for state in row["states"]]
    if not states:
        raise ValueError("rollout has no states")
    random.Random(args.seed).shuffle(states)
    tokenizer, model, head = _model_and_head(
        args.model,
        adapter=args.initial_adapter,
        value_head_path=args.initial_value_head,
        trainable=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    head.train()
    parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    rewards = np.asarray([float(row["reward"]) for row in states], dtype=np.float64)
    old_pre = np.asarray([float(row["pre_value"]) for row in states], dtype=np.float64)
    advantages = rewards - old_pre
    advantage_mean = float(advantages.mean())
    advantage_std = max(float(advantages.std()), 0.1)
    records: list[dict[str, float | int]] = []
    updates = 0
    started = time.time()

    for step_number, state in enumerate(states, 1):
        candidate = state["candidates"][int(state["choice"])]
        prompt = list(state["prompt_ids"])
        completion = list(candidate["completion_ids"])
        old_logprob = torch.tensor(
            candidate["old_token_logprobs"], device="cuda", dtype=torch.float32
        )
        full = torch.tensor([prompt + completion], device="cuda", dtype=torch.long)
        output = model(
            input_ids=full,
            attention_mask=torch.ones_like(full),
            output_hidden_states=True,
            use_cache=False,
        )
        logits = output.logits[0].float()
        positions = torch.arange(
            len(prompt) - 1, len(prompt) + len(completion) - 1, device="cuda"
        )
        tokens = torch.tensor(completion, device="cuda", dtype=torch.long)
        new_logprob = torch.log_softmax(
            logits[positions] / sampling_temperature, dim=-1
        )[
            torch.arange(len(completion), device="cuda"), tokens
        ]
        ratio = torch.exp(new_logprob - old_logprob)
        normalized_advantage = (
            (float(state["reward"]) - float(state["pre_value"]) - advantage_mean)
            / advantage_std
        )
        advantage_tensor = torch.full_like(ratio, normalized_advantage)
        unclipped = ratio * advantage_tensor
        clipped = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * advantage_tensor
        actor_loss = -torch.minimum(unclipped, clipped).mean()
        pre_hidden = output.hidden_states[-1][0, len(prompt) - 1].float()
        post_hidden = output.hidden_states[-1][0, len(prompt) + len(completion) - 1].float()
        pre_prediction = head(pre_hidden).squeeze()
        post_prediction = head(post_hidden).squeeze()
        target = torch.tensor(float(state["reward"]), device="cuda", dtype=torch.float32)
        value_loss = 0.5 * (
            torch.square(pre_prediction - target) + torch.square(post_prediction - target)
        )
        kl_proxy = torch.square(new_logprob - old_logprob).mean()
        total = (
            actor_loss
            + args.value_coefficient * value_loss
            + args.kl_coefficient * kl_proxy
        ) / args.gradient_accumulation
        total.backward()
        update_now = (
            step_number % args.gradient_accumulation == 0
            or step_number == len(states)
        )
        if update_now:
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
        record = {
            "step": step_number,
            "reward": float(state["reward"]),
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "pre_value": float(pre_prediction.detach().cpu()),
            "post_value": float(post_prediction.detach().cpu()),
            "kl_proxy": float(kl_proxy.detach().cpu()),
            "ratio_mean": float(ratio.detach().mean().cpu()),
            "ratio_clip_fraction": float(
                ((ratio < 1 - args.ppo_clip) | (ratio > 1 + args.ppo_clip))
                .float()
                .mean()
                .cpu()
            ),
        }
        records.append(record)
        if update_now:
            print(json.dumps({**record, "updates": updates}), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    torch.save(head.state_dict(), args.output_dir / "value_head.pt")
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "func"
        },
        "states": len(states),
        "optimizer_updates": updates,
        "reward_mean": float(rewards.mean()),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "mean_actor_loss": mean(float(row["actor_loss"]) for row in records),
        "mean_value_loss": mean(float(row["value_loss"]) for row in records),
        "final_ratio_clip_fraction": mean(
            float(row["ratio_clip_fraction"]) for row in records
        ),
        "final_kl_proxy": mean(float(row["kl_proxy"]) for row in records),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import torch

    if args.max_steps <= 0 or args.step_tokens <= 0:
        raise ValueError("max-steps and step-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.max_completion_tokens is not None and args.max_completion_tokens <= 0:
        raise ValueError("max-completion-tokens must be positive when provided")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, model, _ = _model_and_head(
        args.model, adapter=args.actor_adapter, value_head_path=None, trainable=False
    )
    model.eval()
    problems = _load_problems(args)
    existing: list[dict[str, Any]] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text())["examples"]
    completed = {int(row["example_id"]) for row in existing}
    started = time.time()
    newline = tokenizer("\n", add_special_tokens=False).input_ids
    eos = [tokenizer.eos_token_id] + (newline if len(newline) == 1 else [])
    remaining = [problem for problem in problems if problem.example_id not in completed]
    for batch_start in range(0, len(remaining), args.batch_size):
        batch = remaining[batch_start : batch_start + args.batch_size]
        states = [
            {
                "problem": problem,
                "steps": [],
                "step_token_counts": [],
                "finished": False,
            }
            for problem in batch
        ]
        for _ in range(args.max_steps):
            groups: dict[int, list[dict[str, Any]]] = {}
            for state in states:
                if state["finished"]:
                    continue
                generation_limit = next_generation_token_limit(
                    step_tokens=args.step_tokens,
                    used_tokens=sum(state["step_token_counts"]),
                    max_completion_tokens=args.max_completion_tokens,
                )
                if generation_limit is None:
                    state["finished"] = True
                    continue
                groups.setdefault(generation_limit, []).append(state)
            if not groups:
                break
            for generation_limit, group in groups.items():
                prompts = [
                    _prompt_ids(
                        tokenizer,
                        state["problem"].question,
                        state["steps"],
                        args,
                    )
                    for state in group
                ]
                width = max(len(prompt) for prompt in prompts)
                inputs = torch.full(
                    (len(prompts), width),
                    tokenizer.pad_token_id,
                    device="cuda",
                    dtype=torch.long,
                )
                attention_mask = torch.zeros_like(inputs)
                for row, prompt in enumerate(prompts):
                    inputs[row, -len(prompt) :] = torch.tensor(
                        prompt, device="cuda", dtype=torch.long
                    )
                    attention_mask[row, -len(prompt) :] = 1
                with torch.inference_mode():
                    generated_batch = model.generate(
                        input_ids=inputs,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=generation_limit,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos,
                    )[:, width:]
                eos_set = set(eos)
                for state, generated in zip(group, generated_batch, strict=True):
                    completion = generated.tolist()
                    for index, token in enumerate(completion):
                        if token in eos_set:
                            completion = completion[: index + 1]
                            break
                    if not completion:
                        completion = [tokenizer.eos_token_id]
                    state["step_token_counts"].append(len(completion))
                    state["steps"].append(
                        clean_step(tokenizer.decode(completion, skip_special_tokens=True))
                    )
                    if _task_final_answer(args.task, state["steps"][-1]) is not None:
                        state["finished"] = True
        for state in states:
            problem = state["problem"]
            steps = state["steps"]
            step_token_counts = state["step_token_counts"]
            result = _task_terminal_result(args.task, steps, problem.reference_answer)
            completion_tokens = sum(step_token_counts)
            if result["answered"]:
                termination_reason = "valid_final"
            elif (
                args.max_completion_tokens is not None
                and completion_tokens >= args.max_completion_tokens
            ):
                termination_reason = "max_completion_tokens"
            else:
                termination_reason = "max_steps"
            existing.append(
                {
                    "example_id": problem.example_id,
                    "source_id": getattr(problem, "source_id", None),
                    "question": problem.question,
                    "reference_answer": problem.reference_answer,
                    "steps": steps,
                    "step_token_counts": step_token_counts,
                    "completion_tokens": completion_tokens,
                    "termination_reason": termination_reason,
                    "result": result,
                }
            )
        payload = {
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
                if key != "func"
            },
            "summary": {
                "examples": len(existing),
                "accuracy": mean(float(row["result"]["correct"]) for row in existing),
                "answered_fraction": mean(float(row["result"]["answered"]) for row in existing),
                "mean_steps": mean(len(row["steps"]) for row in existing),
                "total_completion_tokens": sum(
                    int(row["completion_tokens"]) for row in existing
                ),
                "mean_completion_tokens": mean(
                    int(row["completion_tokens"]) for row in existing
                ),
            },
            "elapsed_seconds": time.time() - started,
            "examples": existing,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
        print(
            json.dumps({"example_id": batch[-1].example_id, **payload["summary"]}),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint Actor-Value step-action PPO pilot")
    subparsers = parser.add_subparsers(required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--task", choices=TASKS, default="math")
    collect_parser.add_argument("--data", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    collect_parser.add_argument("--actor-adapter", type=Path)
    collect_parser.add_argument("--value-head", type=Path)
    collect_parser.add_argument("--arm", choices=ARMS, required=True)
    collect_parser.add_argument("--offset", type=int, required=True)
    collect_parser.add_argument("--count", type=int, required=True)
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--max-steps", type=int, default=7)
    collect_parser.add_argument("--step-tokens", type=int, default=80)
    collect_parser.add_argument("--max-prompt-tokens", type=int, default=384)
    collect_parser.add_argument("--head-tokens", type=int, default=128)
    collect_parser.add_argument("--candidates", type=int, default=4)
    collect_parser.add_argument("--sampling-temperature", type=float, default=1.0)
    collect_parser.add_argument("--epsilon", type=float, default=1.0)
    collect_parser.add_argument("--beta", type=float, default=1.0)
    collect_parser.add_argument("--value-scale-floor", type=float, default=0.05)
    collect_parser.add_argument("--expansion-rate", type=float, default=0.5)
    collect_parser.add_argument("--entropy-threshold", type=float)
    collect_parser.add_argument("--value-threshold", type=float)
    collect_parser.add_argument("--threshold-source", type=Path)
    collect_parser.set_defaults(func=collect)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--rollout", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    train_parser.add_argument("--initial-adapter", type=Path)
    train_parser.add_argument("--initial-value-head", type=Path)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--learning-rate", type=float, default=5e-6)
    train_parser.add_argument("--gradient-accumulation", type=int, default=32)
    train_parser.add_argument("--lora-rank", type=int, default=32)
    train_parser.add_argument("--lora-alpha", type=int, default=64)
    train_parser.add_argument("--ppo-clip", type=float, default=0.2)
    train_parser.add_argument("--kl-coefficient", type=float, default=0.02)
    train_parser.add_argument("--value-coefficient", type=float, default=0.5)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--task", choices=TASKS, default="math")
    eval_parser.add_argument("--data", type=Path, required=True)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    eval_parser.add_argument("--actor-adapter", type=Path, required=True)
    eval_parser.add_argument("--offset", type=int, required=True)
    eval_parser.add_argument("--count", type=int, required=True)
    eval_parser.add_argument("--seed", type=int, required=True)
    eval_parser.add_argument("--max-steps", type=int, default=7)
    eval_parser.add_argument("--step-tokens", type=int, default=80)
    eval_parser.add_argument("--max-completion-tokens", type=int)
    eval_parser.add_argument("--max-prompt-tokens", type=int, default=384)
    eval_parser.add_argument("--head-tokens", type=int, default=128)
    eval_parser.add_argument("--batch-size", type=int, default=1)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
