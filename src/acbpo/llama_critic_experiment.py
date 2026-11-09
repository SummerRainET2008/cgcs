from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .llama_cpp_policy import LlamaCppEndpoints, LlamaCppToolPolicy
from .qwen_tool_policy import (
    exact_selected_action_probability,
    selector_probabilities,
)
from .step_critic_experiment import constrained_action_mask
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


@dataclass
class RidgeActionCritic:
    projection: np.ndarray
    weights: np.ndarray
    score_scale: float

    def scores(self, feature: np.ndarray) -> np.ndarray:
        projected = feature.astype(np.float32) @ self.projection
        design = np.concatenate([projected, np.ones(1, dtype=np.float32)])
        return (self.weights @ design / self.score_scale).astype(np.float64)


def _gold_observable_actions(env: DirectRetailEnv) -> list[tuple[str, dict[str, Any]]]:
    """Build training-only lookup prefixes that expose gold mutation arguments."""
    task_actions = [
        action
        for action in env.task.actions
        if action.name != "transfer_to_human_agents"
    ]
    result: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()

    def append(name: str, arguments: dict[str, Any]) -> None:
        key = (name, json.dumps(arguments, sort_keys=True))
        if key not in seen:
            seen.add(key)
            result.append((name, deepcopy(arguments)))

    order_ids = [
        action.kwargs["order_id"]
        for action in task_actions
        if "order_id" in action.kwargs
    ]
    for order_id in dict.fromkeys(order_ids):
        append("get_order_details", {"order_id": order_id})

    item_mutations = {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
    }
    for action in task_actions:
        if action.name not in item_mutations:
            continue
        order = env.data["orders"][action.kwargs["order_id"]]
        requested_item_ids = set(action.kwargs.get("item_ids", []))
        product_ids = [
            item["product_id"]
            for item in order["items"]
            if item["item_id"] in requested_item_ids
        ]
        if not product_ids:
            raise RuntimeError(
                f"could not map gold item IDs to products for task {env.task_index}"
            )
        for product_id in dict.fromkeys(product_ids):
            append("get_product_details", {"product_id": product_id})

    for action in task_actions:
        append(action.name, action.kwargs)
    append(
        "respond",
        {"content": " ".join(str(item) for item in env.task.outputs)},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--actor-urls", nargs="+", required=True)
    parser.add_argument("--embedding-urls", nargs="+", required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--train-rollouts", type=int, default=2)
    parser.add_argument(
        "--gold-train-count",
        type=int,
        default=0,
        help="Add successful gold-prefix episodes from the first N training tasks.",
    )
    parser.add_argument(
        "--eval-split", choices=["train", "dev"], default="train"
    )
    parser.add_argument("--eval-start", type=int, default=400)
    parser.add_argument("--eval-count", type=int, default=40)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[17001, 17002])
    parser.add_argument(
        "--eval-selectors",
        nargs="+",
        choices=["actor", "mask", "logprob", "critic"],
        default=["actor", "mask", "logprob", "critic"],
    )
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--disable-action-mask", action="store_true",
                        help="Keep every exposed tool available, including premature respond.")
    parser.add_argument("--selector-objective", choices=["return", "absolute_advantage"],
                        default="return")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--progress-scale", type=float, default=2.0)
    parser.add_argument("--error-penalty", type=float, default=0.5)
    parser.add_argument("--repeat-penalty", type=float, default=0.2)
    parser.add_argument("--failed-respond-penalty", type=float, default=0.5)
    parser.add_argument("--failed-timeout-penalty", type=float, default=0.25)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--critic-rank", type=int, default=128)
    parser.add_argument("--critic-ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        help="Load a fitted critic and skip all critic-training data collection.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".run/episode-cache"))
    return parser


def _rng_for(seed: int, split: str, task_id: int, rollout: int, group: str) -> np.random.Generator:
    # All evaluation selectors receive the same random-number stream. When
    # their state and actor distribution agree, they therefore see identical
    # candidate sets and differ only in the selection rule.
    group_code = 1 if group == "train" else 2
    split_code = {"train": 11, "dev": 13}[split]
    return np.random.default_rng(
        np.random.SeedSequence([seed, split_code, task_id, rollout, group_code])
    )


def _run_episode(
    *,
    policy: LlamaCppToolPolicy,
    critic: RidgeActionCritic | None,
    selector_name: str,
    tau_root: Path,
    split: str,
    task_id: int,
    rollout: int,
    seed: int,
    args: argparse.Namespace,
) -> Episode:
    rng_group = "train" if selector_name == "train" else selector_name
    rng = _rng_for(seed, split, task_id, rollout, rng_group)
    env = DirectRetailEnv(tau_root, split, task_id)
    action_names = env.action_names
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": env.wiki + "\n\nThe complete customer request follows.",
        },
        {"role": "user", "content": env.instruction},
    ]
    use_constraints = selector_name != "actor" and not getattr(args, "disable_action_mask", False)
    need_features = selector_name in {"train", "critic"}
    trajectory: list[dict[str, Any]] = []
    transitions: list[Transition] = []
    terminal_reward = 0.0
    completed_naturally = False
    prefix_joint_ratio = 1.0
    prefix_marginal_ratio = 1.0

    for step in range(args.max_steps):
        step_started = time.perf_counter()
        distribution = policy.choice_distribution(
            messages,
            action_names,
            temperature=args.choice_temperature,
            with_features=need_features,
        )
        actor_probabilities = distribution.probabilities.copy()
        if use_constraints:
            allowed_mask = constrained_action_mask(
                action_names,
                env.instruction,
                trajectory,
                strict_intent=bool(getattr(args, "strict_intent_mask", False)),
            )
            actor_probabilities *= allowed_mask
            actor_probabilities /= actor_probabilities.sum()
        else:
            allowed_mask = np.ones(len(action_names), dtype=bool)

        if selector_name == "logprob":
            action_scores = np.log(np.maximum(actor_probabilities, 1e-30))
        elif selector_name == "critic":
            if critic is None:
                raise RuntimeError("critic selector requires a fitted critic")
            action_scores = critic.scores(distribution.state_features)
        else:
            action_scores = np.zeros(len(action_names), dtype=np.float64)

        selector_scores = action_scores.copy()
        if selector_name == "critic" and getattr(args, "selector_objective", "return") == "absolute_advantage":
            selector_scores = np.abs(action_scores - np.dot(actor_probabilities, action_scores))

        deterministic_actions = bool(
            getattr(args, "deterministic_actions", False)
        )
        if deterministic_actions:
            action_id = int(np.argmax(actor_probabilities))
            candidates = np.asarray([action_id], dtype=np.int64)
            selector = np.asarray([1.0], dtype=np.float64)
            selected_position = 0
        else:
            candidates = rng.choice(
                len(action_names), size=args.candidate_count, p=actor_probabilities
            )
            selector = selector_probabilities(
                candidates,
                selector_scores[candidates],
                beta=args.beta,
                epsilon=args.epsilon,
            )
            selected_position = int(rng.choice(args.candidate_count, p=selector))
            action_id = int(candidates[selected_position])
        action_name = action_names[action_id]
        marginal = (
            1.0
            if deterministic_actions
            else exact_selected_action_probability(
                action_id,
                actor_probabilities,
                selector_scores,
                candidate_count=args.candidate_count,
                beta=args.beta,
                epsilon=args.epsilon,
            )
        )
        joint_ratio = float(1.0 / (len(candidates) * selector[selected_position]))
        marginal_ratio = float(actor_probabilities[action_id] / max(marginal, 1e-12))
        prefix_joint_ratio *= joint_ratio
        prefix_marginal_ratio *= marginal_ratio
        generated = policy.generate_arguments(
            messages,
            action_name,
            env.schema_for(action_name),
            max_new_tokens=getattr(args, "argument_max_new_tokens", 256),
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
        if need_features:
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
                "actor_probability": float(actor_probabilities[action_id]),
                "actor_entropy": float(
                    -np.sum(
                        actor_probabilities
                        * np.log(np.maximum(actor_probabilities, 1e-30))
                    )
                ),
                # Preserve the actual candidate set so candidate recall and
                # selector regret can be measured offline.  The original Q4
                # scaling run recorded only its number of unique actions,
                # which is insufficient to reconstruct oracle@K afterward.
                "candidate_actions": [action_names[int(item)] for item in candidates],
                "candidate_actor_probabilities": [
                    float(actor_probabilities[int(item)]) for item in candidates
                ],
                "candidate_scores": [float(action_scores[int(item)]) for item in candidates],
                "candidate_selector_scores": [float(selector_scores[int(item)]) for item in candidates],
                "selector_probabilities": selector.tolist(),
                "action_mask_enabled": use_constraints,
                "step_wall_seconds": time.perf_counter() - step_started,
                "selected_position": selected_position,
                "candidate_count_unique": len(set(candidates.tolist())),
                "allowed_action_count": int(allowed_mask.sum()),
                "respond_allowed": bool(allowed_mask[action_names.index("respond")]),
                "joint_ratio": joint_ratio,
                "exact_marginal_ratio": marginal_ratio,
                "prefix_joint_ratio": prefix_joint_ratio,
                "prefix_exact_marginal_ratio": prefix_marginal_ratio,
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
        if trajectory:
            trajectory[-1]["shaped_reward"] += timeout_reward

    return Episode(
        result={
            "task_id": task_id,
            "rollout": rollout,
            "seed": seed,
            "reward": terminal_reward,
            "completed_naturally": completed_naturally,
            "steps": len(trajectory),
            "database_progress": env.database_progress(),
            "shaped_return": float(sum(item["shaped_reward"] for item in trajectory)),
            "trajectory": trajectory,
        },
        transitions=transitions,
    )


def _run_gold_episode(
    *,
    policy: LlamaCppToolPolicy,
    tau_root: Path,
    split: str,
    task_id: int,
    rollout: int,
    seed: int,
    args: argparse.Namespace,
) -> Episode:
    """Collect a successful gold-prefix trajectory with model state features.

    Gold actions and arguments are used only on the critic-training split. The
    actor still supplies the state representation at each prefix; evaluation
    remains fully actor-generated on disjoint tasks.
    """
    env = DirectRetailEnv(tau_root, split, task_id)
    action_names = env.action_names
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": env.wiki + "\n\nThe complete customer request follows.",
        },
        {"role": "user", "content": env.instruction},
    ]
    gold_actions = _gold_observable_actions(env)
    official_gold_keys = {
        (action.name, json.dumps(action.kwargs, sort_keys=True))
        for action in env.task.actions
        if action.name != "transfer_to_human_agents"
    }
    trajectory: list[dict[str, Any]] = []
    transitions: list[Transition] = []
    terminal_reward = 0.0

    for step, (action_name, arguments) in enumerate(gold_actions):
        distribution = policy.choice_distribution(
            messages,
            action_names,
            temperature=args.choice_temperature,
            with_features=True,
        )
        action_id = action_names.index(action_name)
        allowed_mask = constrained_action_mask(
            action_names, env.instruction, trajectory
        )
        if not allowed_mask[action_id]:
            raise RuntimeError(
                f"gold-prefix action is masked for train task {task_id}: {action_name}"
            )
        repeated = any(
            previous_name == action_name and previous_arguments == arguments
            for previous_name, previous_arguments in env.actions
        )
        progress_before = env.database_progress()
        result = env.step(action_name, arguments)
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
                "arguments": arguments,
                "actor_probability": float(distribution.probabilities[action_id]),
                "actor_entropy": float(
                    -np.sum(
                        distribution.probabilities
                        * np.log(np.maximum(distribution.probabilities, 1e-30))
                    )
                ),
                "candidate_actions": [action_name],
                "candidate_actor_probabilities": [
                    float(distribution.probabilities[action_id])
                ],
                "candidate_scores": [0.0],
                "selected_position": 0,
                "candidate_count_unique": 1,
                "allowed_action_count": int(allowed_mask.sum()),
                "respond_allowed": bool(
                    allowed_mask[action_names.index("respond")]
                ),
                "gold_action_allowed": bool(allowed_mask[action_id]),
                "joint_ratio": 1.0,
                "exact_marginal_ratio": 1.0,
                "candidate_value_spread": 0.0,
                "valid_json": True,
                "tool_error": result.error,
                "repeated_action": repeated,
                "progress_before": progress_before,
                "progress_after": progress_after,
                "shaped_reward": float(shaped_reward),
            }
        )
        action_key = (action_name, json.dumps(arguments, sort_keys=True))
        if result.error and action_key not in official_gold_keys:
            raise RuntimeError(
                f"gold-prefix lookup failed for train task {task_id}: "
                f"{action_name}({arguments}) -> {result.observation}"
            )
        if result.done:
            break
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"name": action_name, "arguments": arguments},
                        ensure_ascii=False,
                    ),
                },
                {"role": "user", "content": f"Tool result: {result.observation}"},
            ]
        )

    if terminal_reward <= 0.5:
        raise RuntimeError(f"gold trajectory did not solve train task {task_id}")
    return Episode(
        result={
            "task_id": task_id,
            "rollout": rollout,
            "seed": seed,
            "reward": terminal_reward,
            "completed_naturally": True,
            "steps": len(trajectory),
            "database_progress": env.database_progress(),
            "shaped_return": float(sum(item["shaped_reward"] for item in trajectory)),
            "trajectory": trajectory,
            "source": "gold_prefix",
        },
        transitions=transitions,
    )


def fit_ridge_critic(
    episodes: list[Episode],
    *,
    action_count: int,
    rank: int,
    ridge: float,
    gamma: float,
    seed: int,
) -> tuple[RidgeActionCritic, dict[str, Any]]:
    features: list[np.ndarray] = []
    actions: list[int] = []
    targets: list[float] = []
    for episode in episodes:
        return_to_go = 0.0
        episode_targets: list[float] = []
        for transition in reversed(episode.transitions):
            return_to_go = transition.reward + gamma * return_to_go
            episode_targets.append(return_to_go)
        episode_targets.reverse()
        for transition, target in zip(episode.transitions, episode_targets):
            features.append(transition.feature.astype(np.float32))
            actions.append(transition.action_id)
            targets.append(target)
    if not features:
        raise RuntimeError("no critic training transitions were collected")
    feature_array = np.stack(features)
    action_array = np.asarray(actions, dtype=np.int64)
    target_array = np.asarray(targets, dtype=np.float32)
    rng = np.random.default_rng(seed)
    projection = (
        rng.normal(size=(feature_array.shape[1], rank)).astype(np.float32)
        / np.sqrt(rank)
    )
    projected = feature_array @ projection
    design = np.column_stack([projected, np.ones(len(projected), dtype=np.float32)])
    weights = np.empty((action_count, rank + 1), dtype=np.float32)
    global_mean = float(target_array.mean())
    counts: list[int] = []
    for action_id in range(action_count):
        mask = action_array == action_id
        counts.append(int(mask.sum()))
        if not mask.any():
            weights[action_id] = 0.0
            weights[action_id, -1] = global_mean
            continue
        x = design[mask].astype(np.float64)
        y = target_array[mask].astype(np.float64)
        regularizer = np.eye(rank + 1, dtype=np.float64) * ridge
        regularizer[-1, -1] = 0.0
        weights[action_id] = np.linalg.solve(
            x.T @ x + regularizer, x.T @ y
        ).astype(np.float32)
    raw_predictions = np.sum(weights[action_array] * design, axis=1)
    score_scale = max(float(np.std(raw_predictions)), 0.25)
    critic = RidgeActionCritic(projection, weights, score_scale)
    residual = target_array - raw_predictions
    denominator = float(np.sum((target_array - target_array.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / max(denominator, 1e-12)
    diagnostics = {
        "episodes": len(episodes),
        "samples": len(target_array),
        "feature_size": int(feature_array.shape[1]),
        "rank": rank,
        "ridge": ridge,
        "action_sample_counts": counts,
        "target_mean": float(target_array.mean()),
        "target_std": float(target_array.std()),
        "prediction_std": float(np.std(raw_predictions)),
        "score_scale": score_scale,
        "train_r_squared": r_squared,
        "train_success_rate": float(
            np.mean([episode.result["reward"] for episode in episodes])
        ),
    }
    return critic, diagnostics


def load_ridge_critic(path: Path) -> RidgeActionCritic:
    checkpoint = np.load(path)
    return RidgeActionCritic(
        projection=checkpoint["projection"].astype(np.float32),
        weights=checkpoint["weights"].astype(np.float32),
        score_scale=float(checkpoint["score_scale"]),
    )


def _summarize(episodes: list[Episode]) -> dict[str, float]:
    steps = [step for episode in episodes for step in episode.result["trajectory"]]
    return {
        "episodes": len(episodes),
        "unique_tasks": len({episode.result["task_id"] for episode in episodes}),
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
        "tool_error_rate": float(np.mean([step["tool_error"] for step in steps])),
        "repeat_rate": float(np.mean([step["repeated_action"] for step in steps])),
        "invalid_json_rate": float(np.mean([not step["valid_json"] for step in steps])),
        "positive_progress_rate": float(
            np.mean([step["progress_after"] > step["progress_before"] for step in steps])
        ),
        "candidate_diversity_rate": float(
            np.mean([step["candidate_count_unique"] > 1 for step in steps])
        ),
        "mean_candidate_value_spread": float(
            np.mean([step["candidate_value_spread"] for step in steps])
        ),
        "mean_exact_marginal_ratio": float(
            np.mean([step["exact_marginal_ratio"] for step in steps])
        ),
    }


def paired_bootstrap(
    baseline: list[Episode],
    treatment: list[Episode],
    *,
    seed: int,
    samples: int = 20_000,
) -> dict[str, float]:
    baseline_map = {
        (x.result["seed"], x.result["task_id"]): x.result["reward"] for x in baseline
    }
    treatment_map = {
        (x.result["seed"], x.result["task_id"]): x.result["reward"] for x in treatment
    }
    keys = sorted(set(baseline_map) & set(treatment_map))
    differences = np.asarray(
        [treatment_map[key] - baseline_map[key] for key in keys], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "pairs": len(differences),
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "wins": int(np.sum(differences > 0)),
        "losses": int(np.sum(differences < 0)),
        "ties": int(np.sum(differences == 0)),
    }


def _parallel_map(
    policies: list[LlamaCppToolPolicy],
    jobs: list[dict[str, Any]],
    function: Callable[[LlamaCppToolPolicy, dict[str, Any]], Episode],
    *,
    cache_dir: Path,
    fingerprint: str,
) -> list[Episode]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    shards = [jobs[index :: len(policies)] for index in range(len(policies))]

    def cache_path(job: dict[str, Any]) -> Path:
        return cache_dir / (
            f"{job['selector_name']}-{job['split']}-task{job['task_id']}-"
            f"rollout{job['rollout']}-seed{job['seed']}.pkl"
        )

    def run_shard(policy: LlamaCppToolPolicy, shard: Iterable[dict[str, Any]]) -> list[Episode]:
        results: list[Episode] = []
        for job in shard:
            path = cache_path(job)
            if path.exists():
                with path.open("rb") as handle:
                    cached = pickle.load(handle)
                if cached.get("fingerprint") == fingerprint:
                    results.append(cached["episode"])
                    print(
                        f"cached {job['selector_name']} {job['split']} task "
                        f"{job['task_id']} seed {job['seed']}",
                        flush=True,
                    )
                    continue
            episode = function(policy, job)
            results.append(episode)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                pickle.dump(
                    {"fingerprint": fingerprint, "episode": episode},
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary, path)
            print(
                f"completed {job['selector_name']} {job['split']} task "
                f"{job['task_id']} seed {job['seed']}",
                flush=True,
            )
        return results

    with ThreadPoolExecutor(max_workers=len(policies)) as executor:
        futures = [
            executor.submit(run_shard, policy, shard)
            for policy, shard in zip(policies, shards)
        ]
        return [episode for future in futures for episode in future.result()]


def main() -> None:
    args = build_parser().parse_args()
    if len(args.embedding_urls) == 1 and len(args.actor_urls) > 1:
        args.embedding_urls = args.embedding_urls * len(args.actor_urls)
    if len(args.actor_urls) != len(args.embedding_urls):
        raise ValueError(
            "embedding URL count must be one or match the actor URL count"
        )
    policies = [
        LlamaCppToolPolicy(LlamaCppEndpoints(actor, embedding))
        for actor, embedding in zip(args.actor_urls, args.embedding_urls)
    ]

    def execute(policy: LlamaCppToolPolicy, job: dict[str, Any]) -> Episode:
        if job["selector_name"] == "gold_train":
            gold_job = dict(job)
            gold_job.pop("selector_name")
            gold_job.pop("critic")
            return _run_gold_episode(policy=policy, args=args, **gold_job)
        return _run_episode(policy=policy, args=args, **job)

    fingerprint_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output", "checkpoint", "cache_dir", "actor_urls", "embedding_urls"}
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    training_jobs = [
        {
            "critic": None,
            "selector_name": "train",
            "tau_root": args.tau_root,
            "split": "train",
            "task_id": task_id,
            "rollout": rollout,
            "seed": args.seed,
        }
        for rollout in range(args.train_rollouts)
        for task_id in range(args.train_start, args.train_start + args.train_count)
    ]
    behavior_training = (
        _parallel_map(
            policies,
            training_jobs,
            execute,
            cache_dir=args.cache_dir / "training",
            fingerprint=fingerprint,
        )
        if args.checkpoint_in is None
        else []
    )
    gold_training_jobs = [
        {
            "critic": None,
            "selector_name": "gold_train",
            "tau_root": args.tau_root,
            "split": "train",
            "task_id": task_id,
            "rollout": 0,
            "seed": args.seed,
        }
        for task_id in range(
            args.train_start,
            args.train_start + args.gold_train_count,
        )
    ]
    gold_training = (
        _parallel_map(
            policies,
            gold_training_jobs,
            execute,
            cache_dir=args.cache_dir / "gold-training",
            fingerprint=fingerprint,
        )
        if gold_training_jobs and args.checkpoint_in is None
        else []
    )
    training = behavior_training + gold_training
    action_count = len(DirectRetailEnv(args.tau_root, "train", 0).action_names)
    if args.checkpoint_in is not None:
        critic = load_ridge_critic(args.checkpoint_in)
        if critic.weights.shape[0] != action_count:
            raise ValueError("loaded critic action count does not match environment")
        critic_fit: dict[str, Any] = {
            "loaded_checkpoint": str(args.checkpoint_in),
            "feature_size": int(critic.projection.shape[0]),
            "rank": int(critic.projection.shape[1]),
            "action_count": int(critic.weights.shape[0]),
            "score_scale": critic.score_scale,
        }
    else:
        critic, critic_fit = fit_ridge_critic(
            training,
            action_count=action_count,
            rank=args.critic_rank,
            ridge=args.critic_ridge,
            gamma=args.gamma,
            seed=args.seed,
        )
    if args.checkpoint is not None and args.checkpoint_in is None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.checkpoint,
            projection=critic.projection,
            weights=critic.weights,
            score_scale=np.asarray(critic.score_scale),
        )

    evaluation: dict[str, list[Episode]] = {}
    for selector_name in dict.fromkeys(args.eval_selectors):
        jobs = [
            {
                "critic": critic if selector_name == "critic" else None,
                "selector_name": selector_name,
                "tau_root": args.tau_root,
                "split": args.eval_split,
                "task_id": task_id,
                "rollout": seed_index,
                "seed": eval_seed,
            }
            for seed_index, eval_seed in enumerate(args.eval_seeds)
            for task_id in range(args.eval_start, args.eval_start + args.eval_count)
        ]
        evaluation[selector_name] = _parallel_map(
            policies,
            jobs,
            execute,
            cache_dir=args.cache_dir / "evaluation" / selector_name,
            fingerprint=fingerprint,
        )

    comparisons = {}
    if {"actor", "mask"} <= evaluation.keys():
        comparisons["mask_minus_actor"] = paired_bootstrap(
            evaluation["actor"], evaluation["mask"], seed=args.seed + 1
        )
    if {"mask", "logprob"} <= evaluation.keys():
        comparisons["logprob_minus_mask"] = paired_bootstrap(
            evaluation["mask"], evaluation["logprob"], seed=args.seed + 2
        )
    if {"mask", "critic"} <= evaluation.keys():
        comparisons["critic_minus_mask"] = paired_bootstrap(
            evaluation["mask"], evaluation["critic"], seed=args.seed + 3
        )
    summaries = {name: _summarize(value) for name, value in evaluation.items()}
    critic_comparison = comparisons.get("critic_minus_mask")
    gate = {
        "critic_promising": bool(
            critic_comparison is not None
            and critic_comparison["difference"] > 0.0
            and critic_comparison["ci95_low"] >= -0.05
            and summaries["critic"]["tool_error_rate"]
            <= summaries["mask"]["tool_error_rate"] + 0.02
        ),
        "test_ready": bool(
            critic_comparison is not None and critic_comparison["ci95_low"] > 0.0
        ),
        "rule": (
            "promising requires positive success delta, paired CI lower bound >= -5pp, "
            "and tool-error increase <=2pp; test requires paired CI lower bound >0"
        ),
    }
    if args.disable_action_mask:
        gate = {
            "status": "development_only_no_confirmatory_gate",
            "reason": "The unmasked extension uses a separate mechanism protocol; legacy masked gates do not apply.",
        }
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "critic_fit": critic_fit,
        "training_summary": _summarize(training) if training else None,
        "behavior_training_summary": (
            _summarize(behavior_training) if behavior_training else None
        ),
        "gold_training_summary": (
            _summarize(gold_training) if gold_training else None
        ),
        "summaries": summaries,
        "comparisons": comparisons,
        "gate": gate,
        "training_tasks": [episode.result for episode in training],
        "behavior_training_tasks": [
            episode.result for episode in behavior_training
        ],
        "gold_training_tasks": [episode.result for episode in gold_training],
        "evaluation_tasks": {
            name: [episode.result for episode in episodes]
            for name, episodes in evaluation.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "critic_fit": critic_fit,
                "summaries": summaries,
                "comparisons": comparisons,
                "gate": gate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
