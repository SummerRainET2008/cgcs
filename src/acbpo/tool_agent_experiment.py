from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .qwen_tool_policy import (
    QwenToolPolicy,
    exact_selected_action_probability,
    selector_probabilities,
)
from .tau_retail import DirectRetailEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--selector",
        choices=["uniform", "policy", "self_critic", "gold_action"],
        default="policy",
    )
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _score_actions(
    selector: str,
    probabilities: np.ndarray,
    action_names: list[str],
    gold_actions: list[str],
    step: int,
) -> np.ndarray:
    if selector == "uniform":
        return np.zeros_like(probabilities)
    if selector == "policy":
        return np.log(np.maximum(probabilities, 1e-12))
    if selector == "self_critic":
        raise ValueError("self_critic scores require the model and sampled candidates")
    # Diagnostic only: upstream gold actions omit many necessary read-only
    # calls, so this must not be interpreted as an oracle policy.
    expected = gold_actions[min(step, len(gold_actions) - 1)]
    return np.asarray([float(name == expected) for name in action_names])


def run_task(
    policy: QwenToolPolicy,
    args: argparse.Namespace,
    task_id: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    env = DirectRetailEnv(args.tau_root, args.split, task_id)
    action_names = env.action_names
    system = env.wiki + "\n\nThe complete customer request follows."
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": env.instruction},
    ]
    trajectory: list[dict[str, Any]] = []
    invalid_json = 0
    tool_errors = 0
    reward = 0.0
    completed_naturally = False
    for step in range(args.max_steps):
        probabilities = policy.choice_probabilities(
            messages,
            action_names,
            temperature=args.choice_temperature,
        )
        candidates = rng.choice(
            len(action_names), size=args.candidate_count, p=probabilities
        )
        if args.selector == "self_critic":
            candidate_names = [action_names[index] for index in candidates]
            value_map = policy.action_values(messages, candidate_names)
            action_scores = np.asarray(
                [value_map.get(name, 0.0) for name in action_names],
                dtype=np.float64,
            )
        else:
            action_scores = _score_actions(
                args.selector,
                probabilities,
                action_names,
                env.gold_action_names(),
                step,
            )
        selector = selector_probabilities(
            candidates,
            action_scores[candidates],
            beta=args.beta,
            epsilon=args.epsilon,
        )
        selected_position = int(rng.choice(args.candidate_count, p=selector))
        action_id = int(candidates[selected_position])
        action_name = action_names[action_id]
        selected_marginal = exact_selected_action_probability(
            action_id,
            probabilities,
            action_scores,
            candidate_count=args.candidate_count,
            beta=args.beta,
            epsilon=args.epsilon,
        )
        joint_ratio = 1.0 / (args.candidate_count * selector[selected_position])
        marginal_ratio = probabilities[action_id] / max(selected_marginal, 1e-12)
        generated = policy.generate_arguments(
            messages, action_name, env.schema_for(action_name)
        )
        invalid_json += int(not generated.valid_json)
        result = env.step(action_name, generated.arguments)
        tool_errors += int(result.error)
        reward = result.reward
        trajectory.append(
            {
                "step": step,
                "candidates": [action_names[index] for index in candidates],
                "selector_probabilities": selector.tolist(),
                "candidate_values": [float(action_scores[index]) for index in candidates],
                "selected": action_name,
                "actor_entropy": float(
                    -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
                ),
                "actor_top_choices": [
                    {
                        "name": action_names[index],
                        "probability": float(probabilities[index]),
                    }
                    for index in np.argsort(probabilities)[-5:][::-1]
                ],
                "actor_probability": float(probabilities[action_id]),
                "joint_ratio": float(joint_ratio),
                "exact_marginal_probability": selected_marginal,
                "exact_marginal_ratio": float(marginal_ratio),
                "arguments": generated.arguments,
                "raw_arguments": generated.raw_text,
                "valid_json": generated.valid_json,
                "observation": result.observation,
                "error": result.error,
            }
        )
        if result.done:
            completed_naturally = True
            break
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"name": action_name, "arguments": generated.arguments},
                        ensure_ascii=False,
                    ),
                },
                {"role": "user", "content": f"Tool result: {result.observation}"},
            ]
        )
    if not env.done:
        result = env.finish("I could not complete the request.")
        reward = result.reward
    return {
        "task_id": task_id,
        "reward": reward,
        "invalid_json": invalid_json,
        "tool_errors": tool_errors,
        "completed_naturally": completed_naturally,
        "steps": len(trajectory),
        "gold_actions": env.gold_action_names(),
        "trajectory": trajectory,
    }


def main() -> None:
    args = build_parser().parse_args()
    policy = QwenToolPolicy(args.model)
    rng = np.random.default_rng(args.seed)
    tasks = [run_task(policy, args, task_id, rng) for task_id in args.task_ids]
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": {
            "tasks": len(tasks),
            "success_rate": float(np.mean([task["reward"] for task in tasks])),
            "invalid_json_rate_per_step": float(
                sum(task["invalid_json"] for task in tasks)
                / max(1, sum(task["steps"] for task in tasks))
            ),
            "tool_error_rate_per_step": float(
                sum(task["tool_errors"] for task in tasks)
                / max(1, sum(task["steps"] for task in tasks))
            ),
            "mean_steps": float(np.mean([task["steps"] for task in tasks])),
            "natural_completion_rate": float(
                np.mean([task["completed_naturally"] for task in tasks])
            ),
            "mean_actor_entropy": float(
                np.mean(
                    [
                        step["actor_entropy"]
                        for task in tasks
                        for step in task["trajectory"]
                    ]
                )
            ),
            "mean_joint_ratio": float(
                np.mean(
                    [
                        step["joint_ratio"]
                        for task in tasks
                        for step in task["trajectory"]
                    ]
                )
            ),
            "mean_exact_marginal_ratio": float(
                np.mean(
                    [
                        step["exact_marginal_ratio"]
                        for task in tasks
                        for step in task["trajectory"]
                    ]
                )
            ),
        },
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
