from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .outcome_critic_experiment import _make_critic
from .qwen_tool_policy import (
    QwenToolPolicy,
    exact_selected_action_probability,
    selector_probabilities,
)
from .tau_retail import DirectRetailEnv


@dataclass
class Transition:
    feature: np.ndarray
    action_id: int
    reward: float
    done: bool


@dataclass
class Episode:
    result: dict[str, Any]
    transitions: list[Transition]


_MUTATION_ACTIONS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
_FIND_ACTIONS = {"find_user_id_by_email", "find_user_id_by_name_zip"}
_READ_ACTIONS = _FIND_ACTIONS | {
    "calculate",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "list_all_product_types",
}
_MUTATION_CUE = re.compile(
    r"\b(cancel|return|refund|exchange|change|modify|update|replace|switch)\w*\b",
    re.IGNORECASE,
)
_ORDER_ID = re.compile(r"#W\d+", re.IGNORECASE)
_PAYMENT_ID = re.compile(
    r"\b(?:paypal|credit_card|gift_card)_\d+\b", re.IGNORECASE
)


def request_requires_mutation(instruction: str) -> bool:
    """Classify action requests from user-visible text without gold task data."""
    return _MUTATION_CUE.search(instruction) is not None


def requested_mutation_actions(instruction: str) -> set[str]:
    """Infer explicitly requested Retail mutations from user-visible text.

    The direct-user benchmark uses stable natural-language templates, but a
    request may contain several different mutations.  Return every observed
    intent rather than forcing the request into one class.  An empty result is
    deliberately treated as unknown by ``constrained_action_mask``.
    """
    requested: set[str] = set()
    patterns = {
        "cancel_pending_order": r"\bcancel\s+order\b",
        "return_delivered_order_items": r"\breturn\s+#W\d+\b",
        "exchange_delivered_order_items": r"\bexchange\b",
        "modify_pending_order_payment": r"\bchange\s+payment\b",
        "modify_pending_order_address": r"\bchange\s+address\s+to\b",
        "modify_pending_order_items": r"\bmodify\s+[^.;]+?\s+to\b",
        "modify_user_address": (
            r"\b(?:change|modify|update)\s+(?:my\s+|the\s+)?"
            r"(?:default\s+|account\s+|user\s+)?address\b"
        ),
    }
    for action_name, pattern in patterns.items():
        if re.search(pattern, instruction, flags=re.IGNORECASE):
            requested.add(action_name)
    # The pending-order template also matches the generic address expression.
    if "modify_pending_order_address" in requested:
        requested.discard("modify_user_address")
    return requested


def constrained_action_mask(
    action_names: list[str],
    instruction: str,
    trajectory: list[dict[str, Any]],
    *,
    strict_intent: bool = False,
) -> np.ndarray:
    """Return conservative action availability from observable tool history."""
    successful = {
        step["selected"] for step in trajectory if not step["tool_error"]
    }
    has_find = bool(successful & _FIND_ACTIONS)
    has_user = "get_user_details" in successful
    has_order = "get_order_details" in successful
    has_product = "get_product_details" in successful
    has_mutation = bool(successful & _MUTATION_ACTIONS)
    has_read = bool(successful & _READ_ACTIONS)
    order_id_visible = _ORDER_ID.search(instruction) is not None
    payment_id_visible = _PAYMENT_ID.search(instruction) is not None
    mutation_request = request_requires_mutation(instruction)

    allowed = {name: True for name in action_names}
    allowed["get_user_details"] = has_find
    allowed["get_order_details"] = order_id_visible or has_user
    allowed["get_product_details"] = has_order or "list_all_product_types" in successful
    allowed["cancel_pending_order"] = has_order
    allowed["return_delivered_order_items"] = has_order
    allowed["modify_pending_order_address"] = has_order
    allowed["exchange_delivered_order_items"] = has_order and has_product
    allowed["modify_pending_order_items"] = has_order and has_product
    allowed["modify_pending_order_payment"] = has_order and (
        has_user or payment_id_visible
    )
    allowed["modify_user_address"] = has_user
    allowed["respond"] = has_mutation if mutation_request else has_read
    if not mutation_request:
        for name in _MUTATION_ACTIONS:
            allowed[name] = False
    elif strict_intent:
        requested = requested_mutation_actions(instruction)
        if requested:
            for name in _MUTATION_ACTIONS - requested:
                allowed[name] = False
    mask = np.asarray([allowed.get(name, True) for name in action_names], dtype=bool)
    if not mask.any():
        raise RuntimeError("retail constraints masked every action")
    return mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--train-tasks", type=int, default=30)
    parser.add_argument("--train-task-ids", type=int, nargs="+")
    parser.add_argument("--train-rollouts-per-task", type=int, default=2)
    parser.add_argument(
        "--eval-split", choices=["train", "dev", "test"], default="dev"
    )
    parser.add_argument("--eval-task-ids", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--critic-epochs", type=int, default=150)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--critic-rank", type=int, default=32)
    parser.add_argument("--critic-weight-decay", type=float, default=1e-2)
    parser.add_argument("--target-update", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--progress-scale", type=float, default=2.0)
    parser.add_argument("--error-penalty", type=float, default=0.5)
    parser.add_argument("--repeat-penalty", type=float, default=0.2)
    parser.add_argument("--failed-respond-penalty", type=float, default=0.5)
    parser.add_argument("--failed-timeout-penalty", type=float, default=0.25)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--constraints", action="store_true")
    parser.add_argument("--seed", type=int, default=61)
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
    trajectory: list[dict[str, Any]] = []
    transitions: list[Transition] = []
    terminal_reward = 0.0
    completed_naturally = False

    for step in range(args.max_steps):
        distribution = policy.choice_distribution(
            messages, action_names, temperature=args.choice_temperature
        )
        actor_probabilities = distribution.probabilities.copy()
        if args.constraints:
            allowed_mask = constrained_action_mask(
                action_names, env.instruction, trajectory
            )
            actor_probabilities *= allowed_mask
            actor_probabilities /= actor_probabilities.sum()
        else:
            allowed_mask = np.ones(len(action_names), dtype=bool)
        if critic is None:
            action_scores = np.zeros(len(action_names), dtype=np.float64)
        else:
            critic.eval()
            features = torch.from_numpy(distribution.state_features).to(
                policy.model.device
            )
            with torch.inference_mode():
                action_scores = critic(features).float().cpu().numpy()

        candidates = rng.choice(
            len(action_names), size=args.candidate_count, p=actor_probabilities
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
            actor_probabilities,
            action_scores,
            candidate_count=args.candidate_count,
            beta=args.beta,
            epsilon=args.epsilon,
        )
        generated = policy.generate_arguments(
            messages, action_name, env.schema_for(action_name)
        )
        repeated = any(
            previous_name == action_name and previous_arguments == generated.arguments
            for previous_name, previous_arguments in env.actions
        )
        progress_before = env.database_progress()
        result = env.step(action_name, generated.arguments)
        progress_after = env.database_progress()
        shaped_reward = args.progress_scale * (progress_after - progress_before)
        shaped_reward -= args.error_penalty * float(result.error)
        shaped_reward -= args.repeat_penalty * float(repeated)
        if result.done:
            terminal_reward = result.reward
            shaped_reward += (
                args.success_bonus
                if result.reward > 0.5
                else -args.failed_respond_penalty
            )

        transitions.append(
            Transition(
                feature=distribution.state_features.astype(np.float16),
                action_id=action_id,
                reward=float(shaped_reward),
                done=result.done,
            )
        )
        trajectory.append(
            {
                "step": step,
                "selected": action_name,
                "arguments": generated.arguments,
                "actor_entropy": float(
                    -np.sum(
                        actor_probabilities
                        * np.log(np.maximum(actor_probabilities, 1e-12))
                    )
                ),
                "candidate_count_unique": len(set(candidates.tolist())),
                "allowed_action_count": int(allowed_mask.sum()),
                "respond_allowed": bool(allowed_mask[action_names.index("respond")]),
                "joint_ratio": float(
                    1.0 / (args.candidate_count * selector[selected_position])
                ),
                "exact_marginal_ratio": float(
                    actor_probabilities[action_id] / max(marginal, 1e-12)
                ),
                "critic_value": float(action_scores[action_id]),
                "candidate_value_spread": float(
                    action_scores[candidates].max() - action_scores[candidates].min()
                ),
                "valid_json": generated.valid_json,
                "tool_error": result.error,
                "repeated_action": repeated,
                "progress_before": progress_before,
                "progress_after": progress_after,
                "shaped_reward": float(shaped_reward),
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
        terminal_reward = env.finish("I could not complete the request.").reward
        timeout_reward = (
            args.success_bonus
            if terminal_reward > 0.5
            else -args.failed_timeout_penalty
        )
        if transitions:
            transitions[-1].reward += timeout_reward
            transitions[-1].done = True
            trajectory[-1]["shaped_reward"] = transitions[-1].reward

    return Episode(
        result={
            "task_id": task_id,
            "reward": terminal_reward,
            "completed_naturally": completed_naturally,
            "steps": len(trajectory),
            "database_progress": env.database_progress(),
            "shaped_return": float(sum(item.reward for item in transitions)),
            "trajectory": trajectory,
        },
        transitions=transitions,
    )


def _fit_td_critic(
    policy: QwenToolPolicy,
    episodes: list[Episode],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float]]:
    torch = policy.torch
    samples: list[tuple[Transition, Transition | None, bool]] = []
    for episode in episodes:
        for index, transition in enumerate(episode.transitions):
            has_next = not transition.done and index + 1 < len(episode.transitions)
            samples.append(
                (
                    transition,
                    episode.transitions[index + 1] if has_next else None,
                    has_next,
                )
            )
    if not samples:
        raise RuntimeError("no critic training transitions were collected")

    feature_array = np.stack([item[0].feature for item in samples]).astype(np.float32)
    action_array = np.asarray([item[0].action_id for item in samples], dtype=np.int64)
    reward_array = np.asarray([item[0].reward for item in samples], dtype=np.float32)
    nonterminal_array = np.asarray([item[2] for item in samples], dtype=np.float32)
    next_feature_array = np.stack(
        [item[1].feature if item[1] is not None else item[0].feature for item in samples]
    ).astype(np.float32)
    next_action_array = np.asarray(
        [item[1].action_id if item[1] is not None else item[0].action_id for item in samples],
        dtype=np.int64,
    )
    action_count = len(DirectRetailEnv(args.tau_root, "train", 0).action_names)
    critic = _make_critic(
        torch, feature_array.shape[1], action_count, args.critic_rank, args.seed
    ).to(policy.model.device)
    target = deepcopy(critic).to(policy.model.device)
    features = torch.from_numpy(feature_array).to(policy.model.device)
    actions = torch.from_numpy(action_array).to(policy.model.device)
    rewards = torch.from_numpy(reward_array).to(policy.model.device)
    nonterminal = torch.from_numpy(nonterminal_array).to(policy.model.device)
    next_features = torch.from_numpy(next_feature_array).to(policy.model.device)
    next_actions = torch.from_numpy(next_action_array).to(policy.model.device)
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=args.critic_lr,
        weight_decay=args.critic_weight_decay,
    )
    loss_function = torch.nn.SmoothL1Loss()
    final_loss = 0.0
    for epoch in range(args.critic_epochs):
        critic.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = critic(features).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = target(next_features).gather(
                1, next_actions[:, None]
            ).squeeze(1)
            td_targets = rewards + args.gamma * nonterminal * next_values
        loss = loss_function(predictions, td_targets)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        if (epoch + 1) % args.target_update == 0:
            target.load_state_dict(critic.state_dict())
    critic.eval()
    with torch.inference_mode():
        selected_values = critic(features).gather(1, actions[:, None]).squeeze(1).cpu()
    success_mask = torch.tensor(
        [
            episode.result["reward"] > 0.5
            for episode in episodes
            for _ in episode.transitions
        ],
        dtype=torch.bool,
    )
    diagnostics = {
        "episodes": len(episodes),
        "samples": len(samples),
        "train_success_rate": float(
            np.mean([episode.result["reward"] for episode in episodes])
        ),
        "mean_shaped_return": float(
            np.mean([episode.result["shaped_return"] for episode in episodes])
        ),
        "nonzero_reward_rate": float(np.mean(np.abs(reward_array) > 1e-9)),
        "final_loss": final_loss,
        "success_mean_value": float(selected_values[success_mask].mean())
        if bool(success_mask.any())
        else float("nan"),
        "failure_mean_value": float(selected_values[~success_mask].mean())
        if bool((~success_mask).any())
        else float("nan"),
    }
    return critic, diagnostics


def _summarize(episodes: list[Episode]) -> dict[str, float]:
    steps = [step for episode in episodes for step in episode.result["trajectory"]]
    return {
        "tasks": len(episodes),
        "success_rate": float(np.mean([episode.result["reward"] for episode in episodes])),
        "natural_completion_rate": float(
            np.mean([episode.result["completed_naturally"] for episode in episodes])
        ),
        "mean_steps": float(np.mean([episode.result["steps"] for episode in episodes])),
        "mean_database_progress": float(
            np.mean([episode.result["database_progress"] for episode in episodes])
        ),
        "mean_shaped_return": float(
            np.mean([episode.result["shaped_return"] for episode in episodes])
        ),
        "candidate_diversity_rate": float(
            np.mean([step["candidate_count_unique"] > 1 for step in steps])
        ),
        "tool_error_rate": float(np.mean([step["tool_error"] for step in steps])),
        "repeat_rate": float(np.mean([step["repeated_action"] for step in steps])),
        "positive_progress_rate": float(
            np.mean([step["progress_after"] > step["progress_before"] for step in steps])
        ),
        "mean_allowed_action_count": float(
            np.mean([step["allowed_action_count"] for step in steps])
        ),
        "respond_allowed_rate": float(
            np.mean([step["respond_allowed"] for step in steps])
        ),
        "invalid_json_rate": float(np.mean([not step["valid_json"] for step in steps])),
        "mean_candidate_value_spread": float(
            np.mean([step["candidate_value_spread"] for step in steps])
        ),
        "mean_joint_ratio": float(np.mean([step["joint_ratio"] for step in steps])),
        "mean_exact_marginal_ratio": float(
            np.mean([step["exact_marginal_ratio"] for step in steps])
        ),
    }


def main() -> None:
    args = build_parser().parse_args()
    policy = QwenToolPolicy(args.model)
    training: list[Episode] = []
    if args.checkpoint_in is None:
        train_rng = np.random.default_rng(args.seed)
        train_task_ids = (
            args.train_task_ids
            if args.train_task_ids is not None
            else list(range(args.train_tasks))
        )
        total_rollouts = len(train_task_ids) * args.train_rollouts_per_task
        for rollout in range(args.train_rollouts_per_task):
            for task_id in train_task_ids:
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
                print(
                    f"collected train rollout: {len(training)}/{total_rollouts}",
                    flush=True,
                )
        critic, critic_fit = _fit_td_critic(policy, training, args)
        train_summary = _summarize(training)
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
        train_summary = checkpoint.get("train_summary", {})

    uniform_rng = np.random.default_rng(args.seed + 10_000)
    guided_rng = np.random.default_rng(args.seed + 10_000)
    uniform: list[Episode] = []
    guided: list[Episode] = []
    for task_id in args.eval_task_ids:
        uniform.append(
            _run_episode(
                policy=policy,
                critic=None,
                tau_root=args.tau_root,
                split=args.eval_split,
                task_id=task_id,
                args=args,
                rng=uniform_rng,
            )
        )
        guided.append(
            _run_episode(
                policy=policy,
                critic=critic,
                tau_root=args.tau_root,
                split=args.eval_split,
                task_id=task_id,
                args=args,
                rng=guided_rng,
            )
        )
        print(f"evaluated {args.eval_split} task: {task_id}", flush=True)

    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "critic_fit": critic_fit,
        "train_summary": train_summary,
        "uniform_summary": _summarize(uniform),
        "step_critic_summary": _summarize(guided),
        "uniform_tasks": [episode.result for episode in uniform],
        "step_critic_tasks": [episode.result for episode in guided],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        policy.torch.save(
            {
                "state_dict": critic.state_dict(),
                "critic_fit": critic_fit,
                "train_summary": train_summary,
            },
            args.checkpoint,
        )
    print(
        json.dumps(
            {
                "critic_fit": critic_fit,
                "uniform": result["uniform_summary"],
                "step_critic": result["step_critic_summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
