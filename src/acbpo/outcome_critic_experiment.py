from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .qwen_tool_policy import (
    QwenToolPolicy,
    exact_selected_action_probability,
    selector_probabilities,
)
from .tau_retail import DirectRetailEnv


@dataclass
class Episode:
    result: dict[str, Any]
    features: list[np.ndarray]
    action_ids: list[int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train-tasks", type=int, default=20)
    parser.add_argument("--eval-task-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--critic-epochs", type=int, default=200)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-rank", type=int, default=64)
    parser.add_argument("--critic-weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--critic-score", choices=["logit", "probability"], default="logit"
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-in", type=Path)
    return parser


def _run_episode(
    *,
    policy: QwenToolPolicy,
    critic: Any | None,
    tau_root: Path,
    split: str,
    task_id: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Episode:
    torch = policy.torch
    env = DirectRetailEnv(tau_root, split, task_id)
    action_names = env.action_names
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": env.wiki + "\n\nThe complete customer request follows.",
        },
        {"role": "user", "content": env.instruction},
    ]
    features: list[np.ndarray] = []
    action_ids: list[int] = []
    trajectory: list[dict[str, Any]] = []
    reward = 0.0
    completed_naturally = False
    for step in range(args.max_steps):
        distribution = policy.choice_distribution(
            messages, action_names, temperature=args.choice_temperature
        )
        probabilities = distribution.probabilities
        if critic is None:
            action_scores = np.zeros(len(action_names), dtype=np.float64)
        else:
            critic.eval()
            feature_tensor = torch.from_numpy(distribution.state_features).to(
                policy.model.device
            )
            with torch.inference_mode():
                critic_logits = critic(feature_tensor).float()
                if args.critic_score == "probability":
                    critic_logits = torch.sigmoid(critic_logits)
                action_scores = critic_logits.cpu().numpy()
        candidates = rng.choice(
            len(action_names), size=args.candidate_count, p=probabilities
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
        marginal = exact_selected_action_probability(
            action_id,
            probabilities,
            action_scores,
            candidate_count=args.candidate_count,
            beta=args.beta,
            epsilon=args.epsilon,
        )
        generated = policy.generate_arguments(
            messages, action_name, env.schema_for(action_name)
        )
        step_result = env.step(action_name, generated.arguments)
        reward = step_result.reward
        features.append(distribution.state_features.astype(np.float16))
        action_ids.append(action_id)
        trajectory.append(
            {
                "step": step,
                "selected": action_name,
                "actor_entropy": float(
                    -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
                ),
                "candidate_count_unique": len(set(candidates.tolist())),
                "joint_ratio": float(
                    1.0 / (args.candidate_count * selector[selected_position])
                ),
                "exact_marginal_ratio": float(
                    probabilities[action_id] / max(marginal, 1e-12)
                ),
                "critic_value": float(action_scores[action_id]),
                "candidate_value_spread": float(
                    action_scores[candidates].max() - action_scores[candidates].min()
                ),
                "valid_json": generated.valid_json,
                "tool_error": step_result.error,
            }
        )
        if step_result.done:
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
                {"role": "user", "content": f"Tool result: {step_result.observation}"},
            ]
        )
    if not env.done:
        reward = env.finish("I could not complete the request.").reward
    return Episode(
        result={
            "task_id": task_id,
            "reward": reward,
            "completed_naturally": completed_naturally,
            "steps": len(trajectory),
            "trajectory": trajectory,
        },
        features=features,
        action_ids=action_ids,
    )


def _summarize(episodes: list[Episode]) -> dict[str, float]:
    steps = [step for episode in episodes for step in episode.result["trajectory"]]
    return {
        "tasks": len(episodes),
        "success_rate": float(
            np.mean([episode.result["reward"] for episode in episodes])
        ),
        "natural_completion_rate": float(
            np.mean(
                [episode.result["completed_naturally"] for episode in episodes]
            )
        ),
        "mean_steps": float(
            np.mean([episode.result["steps"] for episode in episodes])
        ),
        "candidate_diversity_rate": float(
            np.mean([step["candidate_count_unique"] > 1 for step in steps])
        ),
        "tool_error_rate": float(np.mean([step["tool_error"] for step in steps])),
        "invalid_json_rate": float(
            np.mean([not step["valid_json"] for step in steps])
        ),
        "mean_candidate_value_spread": float(
            np.mean([step["candidate_value_spread"] for step in steps])
        ),
        "mean_joint_ratio": float(np.mean([step["joint_ratio"] for step in steps])),
        "mean_exact_marginal_ratio": float(
            np.mean([step["exact_marginal_ratio"] for step in steps])
        ),
    }


def _make_critic(
    torch: Any,
    input_size: int,
    action_count: int,
    rank: int,
    seed: int,
) -> Any:
    nn = torch.nn

    class RandomProjectionCritic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            projection = torch.randn(
                input_size, rank, generator=generator, dtype=torch.float32
            ) / np.sqrt(rank)
            self.register_buffer("projection", projection)
            self.head = nn.Linear(rank, action_count)

        def forward(self, features: Any) -> Any:
            projected = features @ self.projection
            projected = nn.functional.layer_norm(projected, (rank,))
            return self.head(projected)

    return RandomProjectionCritic()


def _fit_critic(policy: QwenToolPolicy, episodes: list[Episode], args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    torch = policy.torch
    nn = torch.nn
    all_features = np.stack(
        [feature for episode in episodes for feature in episode.features]
    ).astype(np.float32)
    all_actions = np.asarray(
        [action for episode in episodes for action in episode.action_ids],
        dtype=np.int64,
    )
    all_labels = np.asarray(
        [
            episode.result["reward"]
            for episode in episodes
            for _ in episode.action_ids
        ],
        dtype=np.float32,
    )
    action_count = len(
        DirectRetailEnv(args.tau_root, "train", 0).action_names
    )
    critic = _make_critic(
        torch,
        all_features.shape[1],
        action_count,
        args.critic_rank,
        args.seed,
    ).to(policy.model.device)
    features = torch.from_numpy(all_features).to(policy.model.device)
    actions = torch.from_numpy(all_actions).to(policy.model.device)
    labels = torch.from_numpy(all_labels).to(policy.model.device)
    positives = max(1.0, float(all_labels.sum()))
    negatives = max(1.0, float(len(all_labels) - all_labels.sum()))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=policy.model.device)
    )
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.critic_weight_decay,
    )
    critic.train()
    final_loss = 0.0
    for _ in range(args.critic_epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = critic(features).gather(1, actions[:, None]).squeeze(1)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    critic.eval()
    with torch.inference_mode():
        selected_logits = critic(features).gather(1, actions[:, None]).squeeze(1)
        predictions = torch.sigmoid(selected_logits)
    diagnostics = {
        "samples": len(all_labels),
        "positive_rate": float(all_labels.mean()),
        "final_loss": final_loss,
        "positive_mean_value": float(
            predictions[labels > 0.5].mean().cpu()
        )
        if bool((labels > 0.5).any())
        else float("nan"),
        "negative_mean_value": float(
            predictions[labels <= 0.5].mean().cpu()
        ),
    }
    return critic, diagnostics


def main() -> None:
    args = build_parser().parse_args()
    policy = QwenToolPolicy(args.model)
    training: list[Episode] = []
    if args.checkpoint_in is None:
        train_rng = np.random.default_rng(args.seed)
        for task_id in range(args.train_tasks):
            training.append(
                _run_episode(
                    policy=policy,
                    critic=None,
                    tau_root=args.tau_root,
                    split="train",
                    task_id=task_id,
                    args=args,
                    rng=train_rng,
                )
            )
            if (task_id + 1) % 5 == 0 or task_id + 1 == args.train_tasks:
                print(
                    f"collected train tasks: {task_id + 1}/{args.train_tasks}",
                    flush=True,
                )
        critic, critic_fit = _fit_critic(policy, training, args)
    else:
        checkpoint = policy.torch.load(
            args.checkpoint_in, map_location=policy.model.device
        )
        state_dict = checkpoint["state_dict"]
        input_size, rank = state_dict["projection"].shape
        action_count = state_dict["head.weight"].shape[0]
        critic = _make_critic(
            policy.torch, input_size, action_count, rank, args.seed
        ).to(policy.model.device)
        critic.load_state_dict(state_dict)
        critic_fit = checkpoint.get("critic_fit", {})
    uniform_rng = np.random.default_rng(args.seed + 10_000)
    uniform = [
        _run_episode(
            policy=policy,
            critic=None,
            tau_root=args.tau_root,
            split="dev",
            task_id=task_id,
            args=args,
            rng=uniform_rng,
        )
        for task_id in args.eval_task_ids
    ]
    critic_rng = np.random.default_rng(args.seed + 10_000)
    guided = [
        _run_episode(
            policy=policy,
            critic=critic,
            tau_root=args.tau_root,
            split="dev",
            task_id=task_id,
            args=args,
            rng=critic_rng,
        )
        for task_id in args.eval_task_ids
    ]
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "training": {
            "summary": _summarize(training) if training else None,
            "critic_fit": critic_fit,
            "checkpoint_in": str(args.checkpoint_in) if args.checkpoint_in else None,
        },
        "evaluation": {
            "uniform": _summarize(uniform),
            "outcome_critic": _summarize(guided),
        },
        "uniform_tasks": [episode.result for episode in uniform],
        "outcome_critic_tasks": [episode.result for episode in guided],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.torch.save(
            {
                "state_dict": critic.state_dict(),
                "model": args.model,
                "action_names": DirectRetailEnv(
                    args.tau_root, "train", 0
                ).action_names,
                "critic_fit": critic_fit,
            },
            args.checkpoint,
        )
    print(json.dumps(result["evaluation"], indent=2))


if __name__ == "__main__":
    main()
