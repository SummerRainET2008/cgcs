"""LLM training estimators for the controlled search-policy mismatch transfer.

Collection remains in :mod:`acbpo.math_joint_value_ppo`.  This module consumes
one logged rollout and makes the estimator target explicit.  Gate and selector
parameters are frozen during an update; E4 therefore targets the composite
candidate-generation policy under the logged controller.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np

from .math_joint_value_ppo import _model_and_head


ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)


def conditional_selector_factor(state: dict[str, Any]) -> float:
    """Map a selected candidate back to a uniform Actor draw.

    Candidate positions are IID Actor samples.  Conditional on a logged
    expanded set, ``1 / (K q_i)`` is the Horvitz--Thompson correction from the
    selector distribution to uniform choice.  Unexpanded states already contain
    one Actor draw and therefore have factor one.
    """
    candidates = state.get("candidates", [])
    if not candidates:
        raise ValueError("state has no candidates")
    if not bool(state.get("expanded", False)):
        if len(candidates) != 1:
            raise ValueError("unexpanded state must contain exactly one candidate")
        return 1.0
    choice = int(state["choice"])
    probabilities = [float(value) for value in state["selector_probabilities"]]
    if len(probabilities) != len(candidates) or not 0 <= choice < len(candidates):
        raise ValueError("candidate/selector metadata mismatch")
    probability = probabilities[choice]
    if not math.isfinite(probability) or probability <= 0.0:
        raise ValueError("selected candidate must have positive finite propensity")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("selector probabilities must sum to one")
    return 1.0 / (len(candidates) * probability)


def external_trajectory_weights(
    payload: dict[str, Any],
    *,
    estimator: str,
    partial_lambda: float = 0.75,
) -> list[float]:
    """Return one outside-clipping correction weight per logged state."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator: {estimator}")
    if not 0.0 <= partial_lambda <= 1.0:
        raise ValueError("partial_lambda must be in [0, 1]")
    result: list[float] = []
    for example in payload.get("examples", []):
        cumulative = 1.0
        for state in example.get("states", []):
            factor = conditional_selector_factor(state)
            if estimator == "E1_trajectory_propensity":
                cumulative *= factor
            elif estimator == "E3_step_cumulative_lambda_0.75":
                cumulative *= (1.0 - partial_lambda) + partial_lambda * factor
            elif estimator in {
                "E0_standard_actor_ratio",
                "E4_extended_candidate_set",
            }:
                cumulative = 1.0
            if not math.isfinite(cumulative) or cumulative <= 0.0:
                raise ValueError("non-positive or non-finite trajectory weight")
            result.append(cumulative)
    return result


def normalized_ess(weights: Iterable[float]) -> float:
    array = np.asarray(list(weights), dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(array < 0.0):
        raise ValueError("weights must be a non-empty non-negative vector")
    denominator = float(np.square(array).sum())
    if denominator == 0.0:
        return 0.0
    return float(array.sum() ** 2 / (len(array) * denominator))


def policy_sequence_log_ratio(log_ratios, *, choice: int, estimator: str):
    """Compose candidate sequence ratios for the estimator's named target."""
    import torch

    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator: {estimator}")
    if not log_ratios or not 0 <= choice < len(log_ratios):
        raise ValueError("invalid candidate log-ratio vector or choice")
    if estimator == "E4_extended_candidate_set":
        return torch.stack(list(log_ratios)).sum()
    return log_ratios[choice] + sum(
        (0.0 * value for index, value in enumerate(log_ratios) if index != choice),
        torch.zeros_like(log_ratios[choice]),
    )


def _candidate_log_ratio(model, prompt: list[int], candidate: dict[str, Any], temperature: float):
    import torch

    completion = list(candidate["completion_ids"])
    old = torch.tensor(
        candidate["old_token_logprobs"], device="cuda", dtype=torch.float32
    )
    if len(completion) != int(old.numel()):
        raise ValueError("completion and old log-prob lengths differ")
    full = torch.tensor([prompt + completion], device="cuda", dtype=torch.long)
    output = model(
        input_ids=full,
        attention_mask=torch.ones_like(full),
        output_hidden_states=True,
        use_cache=False,
    )
    positions = torch.arange(
        len(prompt) - 1, len(prompt) + len(completion) - 1, device="cuda"
    )
    tokens = torch.tensor(completion, device="cuda", dtype=torch.long)
    new = torch.log_softmax(output.logits[0].float()[positions] / temperature, dim=-1)[
        torch.arange(len(completion), device="cuda"), tokens
    ]
    token_delta = new - old
    return token_delta.sum(), token_delta, output.hidden_states[-1][0]


def train(args: argparse.Namespace) -> None:
    import torch

    if args.estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator: {args.estimator}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    payload = json.loads(args.rollout.read_text())
    temperature = float(payload.get("config", {}).get("sampling_temperature", 1.0))
    if temperature <= 0.0:
        raise ValueError("sampling temperature must be positive")

    ordered_states = [
        state for example in payload.get("examples", []) for state in example.get("states", [])
    ]
    weights = external_trajectory_weights(
        payload, estimator=args.estimator, partial_lambda=args.partial_lambda
    )
    if not ordered_states or len(ordered_states) != len(weights):
        raise ValueError("rollout has no states or weight alignment failed")
    items = list(zip(ordered_states, weights, strict=True))
    random.Random(args.seed).shuffle(items)

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

    rewards = np.asarray([float(state["reward"]) for state in ordered_states])
    old_pre = np.asarray([float(state["pre_value"]) for state in ordered_states])
    advantages = rewards - old_pre
    advantage_mean = float(advantages.mean())
    advantage_std = max(float(advantages.std()), 0.1)
    records: list[dict[str, float | int]] = []
    updates = 0
    candidate_forward_count = 0
    selected_token_count = 0
    candidate_token_count = 0
    started = time.time()

    for step_number, (state, outside_weight) in enumerate(items, 1):
        prompt = list(state["prompt_ids"])
        candidates = list(state["candidates"])
        choice = int(state["choice"])
        if not 0 <= choice < len(candidates):
            raise ValueError("selected candidate index is out of bounds")

        log_ratios = []
        token_deltas = []
        hidden_states = []
        for candidate in candidates:
            log_ratio, deltas, hidden = _candidate_log_ratio(
                model, prompt, candidate, temperature
            )
            log_ratios.append(log_ratio)
            token_deltas.append(deltas)
            hidden_states.append(hidden)
            candidate_forward_count += 1
            candidate_token_count += len(candidate["completion_ids"])

        # Retain all candidate graphs so estimator columns have the same
        # candidate-token forward/backward accounting within a search row.
        policy_log_ratio = policy_sequence_log_ratio(
            log_ratios, choice=choice, estimator=args.estimator
        )
        if not bool(torch.isfinite(policy_log_ratio).detach().cpu()):
            raise FloatingPointError("non-finite sequence log ratio")
        ratio = torch.exp(policy_log_ratio)
        if not bool(torch.isfinite(ratio).detach().cpu()):
            raise FloatingPointError("non-finite sequence importance ratio")

        normalized_advantage = (
            (float(state["reward"]) - float(state["pre_value"]) - advantage_mean)
            / advantage_std
        )
        advantage = torch.tensor(
            normalized_advantage, device="cuda", dtype=torch.float32
        )
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip) * advantage
        actor_loss = -float(outside_weight) * torch.minimum(unclipped, clipped)

        # Token-level drift is defined on the selected raw-Actor completion.
        # This remains directly comparable across estimator columns even when
        # the estimator's sequence ratio marginalizes over retained candidates.
        selected_token_ratios = torch.exp(token_deltas[choice].detach().float())
        selected_token_clipped = (
            (selected_token_ratios < 1 - args.ppo_clip)
            | (selected_token_ratios > 1 + args.ppo_clip)
        )
        if normalized_advantage >= 0.0:
            sequence_ratio_saturated = ratio > 1 + args.ppo_clip
            selected_token_saturated = selected_token_ratios > 1 + args.ppo_clip
        else:
            sequence_ratio_saturated = ratio < 1 - args.ppo_clip
            selected_token_saturated = selected_token_ratios < 1 - args.ppo_clip

        selected_hidden = hidden_states[choice]
        completion_length = len(candidates[choice]["completion_ids"])
        selected_token_count += completion_length
        pre_hidden = selected_hidden[len(prompt) - 1].float()
        post_hidden = selected_hidden[len(prompt) + completion_length - 1].float()
        pre_prediction = head(pre_hidden).squeeze()
        post_prediction = head(post_hidden).squeeze()
        target = torch.tensor(float(state["reward"]), device="cuda", dtype=torch.float32)
        value_loss = 0.5 * (
            torch.square(pre_prediction - target) + torch.square(post_prediction - target)
        )
        selected_kl_proxy = torch.square(token_deltas[choice]).mean()
        total = (
            actor_loss
            + args.value_coefficient * value_loss
            + args.kl_coefficient * selected_kl_proxy
        ) / args.gradient_accumulation
        total.backward()
        update_now = (
            step_number % args.gradient_accumulation == 0 or step_number == len(items)
        )
        if update_now:
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

        record = {
            "step": step_number,
            "reward": float(state["reward"]),
            "outside_weight": float(outside_weight),
            "sequence_log_ratio": float(policy_log_ratio.detach().cpu()),
            "sequence_ratio": float(ratio.detach().cpu()),
            "sequence_ratio_clipped": int(
                bool(((ratio < 1 - args.ppo_clip) | (ratio > 1 + args.ppo_clip)).detach().cpu())
            ),
            "sequence_ratio_saturated": int(
                bool(sequence_ratio_saturated.detach().cpu())
            ),
            "selected_actor_token_count": int(selected_token_ratios.numel()),
            "selected_actor_token_ratio_clipped_count": int(
                selected_token_clipped.sum().cpu()
            ),
            "selected_actor_token_saturated_count": int(
                selected_token_saturated.sum().cpu()
            ),
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "pre_value": float(pre_prediction.detach().cpu()),
            "post_value": float(post_prediction.detach().cpu()),
            "selected_kl_proxy": float(selected_kl_proxy.detach().cpu()),
            "candidate_count": len(candidates),
        }
        records.append(record)
        if update_now:
            print(json.dumps({**record, "updates": updates}), flush=True)

    weight_array = np.asarray(weights, dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    torch.save(head.state_dict(), args.output_dir / "value_head.pt")
    summary = {
        "schema": "cgcs-search-policy-mismatch-llm-train-v1",
        "controller_gradient_stopped": True,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "func"
        },
        "states": len(items),
        "optimizer_updates": updates,
        "reward_mean": float(rewards.mean()),
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "external_weight_ess": normalized_ess(weights),
        "external_weight_maximum": float(weight_array.max()),
        "external_weight_quantiles_0_50_90_95_99_100": np.quantile(
            weight_array, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]
        ).tolist(),
        "candidate_forwards": candidate_forward_count,
        "selected_tokens": selected_token_count,
        "candidate_tokens": candidate_token_count,
        "mean_actor_loss": mean(float(row["actor_loss"]) for row in records),
        "mean_value_loss": mean(float(row["value_loss"]) for row in records),
        "mean_sequence_ratio": mean(float(row["sequence_ratio"]) for row in records),
        "sequence_ratio_clip_fraction": mean(
            int(row["sequence_ratio_clipped"]) for row in records
        ),
        "sequence_ratio_saturation_fraction": mean(
            int(row["sequence_ratio_saturated"]) for row in records
        ),
        "selected_actor_token_ratio_clip_fraction": (
            sum(int(row["selected_actor_token_ratio_clipped_count"]) for row in records)
            / sum(int(row["selected_actor_token_count"]) for row in records)
        ),
        "selected_actor_token_saturation_fraction": (
            sum(int(row["selected_actor_token_saturated_count"]) for row in records)
            / sum(int(row["selected_actor_token_count"]) for row in records)
        ),
        "mean_selected_kl_proxy": mean(
            float(row["selected_kl_proxy"]) for row in records
        ),
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Controlled LLM search-policy mismatch estimators"
    )
    subparsers = parser.add_subparsers(required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--rollout", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    train_parser.add_argument("--initial-adapter", type=Path)
    train_parser.add_argument("--initial-value-head", type=Path)
    train_parser.add_argument("--estimator", choices=ESTIMATORS, required=True)
    train_parser.add_argument("--partial-lambda", type=float, default=0.75)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
