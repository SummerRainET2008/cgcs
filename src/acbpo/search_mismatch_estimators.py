"""Finite-sample estimators for the frozen search-policy-mismatch S2 study."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .search_policy_mismatch import (
    SearchMDPPolicy,
    candidate_pools_from_uniforms,
    exact_values,
    forward_occupancy,
    gradient_diagnostics,
    raw_actor_gradient,
    sample_categorical,
    softmax,
)
from .layered_mdp import LayeredMDP


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
BATCH_SIZES = (256, 1024)


@dataclass(frozen=True)
class SampledSearchTrajectories:
    states: IntArray
    actions: IntArray
    rewards: FloatArray
    conditional_ratios: FloatArray
    marginal_ratios: FloatArray

    @property
    def sample_count(self) -> int:
        return int(self.states.shape[0])


def sample_search_trajectories(
    controller: SearchMDPPolicy,
    *,
    samples: int,
    seed: int,
    initial_state: int = 0,
) -> SampledSearchTrajectories:
    """Sample once; every estimator reuses these exact trajectories."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    mdp = controller.mdp
    rng = np.random.default_rng(seed)
    states = np.empty((samples, mdp.horizon), dtype=np.int64)
    actions = np.empty_like(states)
    rewards = np.empty((samples, mdp.horizon), dtype=np.float64)
    conditional_ratios = np.ones_like(rewards)
    marginal_ratios = np.ones_like(rewards)
    current_states = np.full(samples, initial_state, dtype=np.int64)
    rows = np.arange(samples)

    for t in range(mdp.horizon):
        states[:, t] = current_states
        proposals = mdp.actor[t, current_states]
        gates = controller.expansion_probabilities[t, current_states]
        expanded = rng.random(samples) < gates
        current_actions = sample_categorical(proposals, rng.random(samples))
        expanded_rows = np.flatnonzero(expanded)
        if expanded_rows.size:
            candidates = candidate_pools_from_uniforms(
                proposals[expanded_rows],
                rng.random((expanded_rows.size, controller.candidate_count)),
            )
            scores = controller.critic_scores[t, current_states[expanded_rows]][
                np.arange(expanded_rows.size)[:, None], candidates
            ]
            guided = softmax(controller.beta * scores, axis=-1)
            selector = (
                (1.0 - controller.epsilon) * guided
                + controller.epsilon / controller.candidate_count
            )
            selected = sample_categorical(
                selector, rng.random(expanded_rows.size)
            )
            current_actions[expanded_rows] = candidates[
                np.arange(expanded_rows.size), selected
            ]
            selected_probability = selector[
                np.arange(expanded_rows.size), selected
            ]
            conditional_ratios[expanded_rows, t] = 1.0 / (
                controller.candidate_count * selected_probability
            )

        actions[:, t] = current_actions
        rewards[:, t] = mdp.rewards[t, current_states, current_actions]
        marginal_ratios[:, t] = proposals[rows, current_actions] / (
            controller.executed_policy[t, current_states, current_actions]
        )
        if t + 1 < mdp.horizon:
            transition = mdp.transitions[t, current_states, current_actions]
            current_states = sample_categorical(transition, rng.random(samples))

    return SampledSearchTrajectories(
        states=states,
        actions=actions,
        rewards=rewards,
        conditional_ratios=conditional_ratios,
        marginal_ratios=marginal_ratios,
    )


def partial_weight_matrix(
    ratios: FloatArray, *, method: str, strength: float
) -> FloatArray:
    """Return E3 weights for every trajectory and depth."""
    if ratios.ndim != 2:
        raise ValueError("ratios must have shape [samples, horizon]")
    if strength not in LAMBDAS:
        raise ValueError(f"strength must be one of {LAMBDAS}")
    interpolated = (1.0 - strength) + strength * ratios
    if method == "local":
        return interpolated
    if method == "step_cumulative":
        return np.cumprod(interpolated, axis=1)
    if method == "trajectory_blend":
        exact_cumulative = np.cumprod(ratios, axis=1)
        return (1.0 - strength) + strength * exact_cumulative
    raise ValueError("unknown partial-correction method")


def _mean_gradient_and_second_norm(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
    weights: FloatArray,
) -> tuple[FloatArray, float]:
    mdp = controller.mdp
    sample_count = trajectories.sample_count
    raw_values, raw_q = exact_values(mdp, mdp.actor)
    advantages = raw_q - raw_values[:-1, :, None]
    gradient = np.zeros_like(mdp.actor)
    second_norm_sum = 0.0
    rows = np.arange(sample_count)
    for t in range(mdp.horizon):
        state = trajectories.states[:, t]
        action = trajectories.actions[:, t]
        advantage = advantages[t, state, action]
        scalar = (mdp.gamma**t) * weights[:, t] * advantage
        proposal = mdp.actor[t, state]
        score = -proposal.copy()
        score[rows, action] += 1.0
        contributions = scalar[:, None] * score
        for current_state in range(mdp.state_count):
            selected = state == current_state
            if np.any(selected):
                gradient[t, current_state] = contributions[selected].sum(axis=0)
        second_norm_sum += float(np.square(contributions).sum())
    gradient /= sample_count
    mean_second_norm = second_norm_sum / sample_count
    return gradient, mean_second_norm


def _normalized_ess(weights: FloatArray) -> float:
    denominator = weights.size * float(np.square(weights).sum())
    if denominator == 0.0:
        return 0.0
    return float(np.square(weights.sum()) / denominator)


def _estimate_one_weighting_with_gradient(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
    weights: FloatArray,
) -> tuple[dict[str, object], FloatArray]:
    mdp = controller.mdp
    target_gradient = raw_actor_gradient(mdp)
    mean_gradient, mean_second_norm = _mean_gradient_and_second_norm(
        controller, trajectories, weights
    )
    diagnostics = gradient_diagnostics(mean_gradient, target_gradient)
    target_norm_squared = float(np.square(target_gradient).sum())
    variance_trace = max(
        0.0, mean_second_norm - float(np.square(mean_gradient).sum())
    )
    gradient_bias_squared = float(np.square(mean_gradient - target_gradient).sum())

    discount = mdp.gamma ** np.arange(mdp.horizon)
    return_samples = np.sum(
        discount[None, :] * weights * trajectories.rewards, axis=1
    )
    raw_truth = float(exact_values(mdp, mdp.actor)[0][0, 0])
    return_bias = float(return_samples.mean() - raw_truth)
    return_variance = float(return_samples.var(ddof=1))
    final_weights = weights[:, -1]
    quantiles = np.quantile(
        final_weights, [0.0, 0.50, 0.90, 0.95, 0.99, 1.0]
    )

    output: dict[str, object] = {
        **diagnostics,
        "gradient_bias_squared": gradient_bias_squared,
        "gradient_variance_trace": variance_trace,
        "normalized_final_weight_ess": _normalized_ess(final_weights),
        "maximum_final_weight": float(final_weights.max()),
        "support_violations": 0,
        "final_weight_quantiles_0_50_90_95_99_100": quantiles.tolist(),
        "return_mean": float(return_samples.mean()),
        "return_truth": raw_truth,
        "return_bias": return_bias,
        "return_variance": return_variance,
    }
    for batch_size in BATCH_SIZES:
        gradient_mse = gradient_bias_squared + variance_trace / batch_size
        output[f"gradient_mse_batch_{batch_size}"] = gradient_mse
        output[f"gradient_relative_mse_batch_{batch_size}"] = (
            gradient_mse / target_norm_squared
        )
        output[f"return_mse_batch_{batch_size}"] = (
            return_bias * return_bias + return_variance / batch_size
        )
    return output, mean_gradient


def estimate_one_weighting(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
    weights: FloatArray,
) -> dict[str, object]:
    return _estimate_one_weighting_with_gradient(
        controller, trajectories, weights
    )[0]


def complete_partial_frontier(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
) -> dict[str, dict[str, object]]:
    """Evaluate both ratio bases, all three E3 constructions, and all lambdas."""
    output: dict[str, dict[str, object]] = {}
    for ratio_name, ratios in (
        ("conditional", trajectories.conditional_ratios),
        ("marginal", trajectories.marginal_ratios),
    ):
        for method in ("local", "step_cumulative", "trajectory_blend"):
            for strength in LAMBDAS:
                name = f"{ratio_name}_{method}_lambda_{strength:g}"
                weights = partial_weight_matrix(
                    ratios, method=method, strength=strength
                )
                output[name] = estimate_one_weighting(
                    controller, trajectories, weights
                )
    return output


def complete_partial_frontier_with_gradients(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
) -> tuple[dict[str, dict[str, object]], dict[str, FloatArray]]:
    metrics: dict[str, dict[str, object]] = {}
    gradients: dict[str, FloatArray] = {}
    for ratio_name, ratios in (
        ("conditional", trajectories.conditional_ratios),
        ("marginal", trajectories.marginal_ratios),
    ):
        for method in ("local", "step_cumulative", "trajectory_blend"):
            for strength in LAMBDAS:
                name = f"{ratio_name}_{method}_lambda_{strength:g}"
                weights = partial_weight_matrix(
                    ratios, method=method, strength=strength
                )
                metrics[name], gradients[name] = _estimate_one_weighting_with_gradient(
                    controller, trajectories, weights
                )
    return metrics, gradients


def concatenate_trajectories(
    trajectories: list[SampledSearchTrajectories],
) -> SampledSearchTrajectories:
    if not trajectories:
        raise ValueError("at least one trajectory batch is required")
    return SampledSearchTrajectories(
        states=np.concatenate([row.states for row in trajectories], axis=0),
        actions=np.concatenate([row.actions for row in trajectories], axis=0),
        rewards=np.concatenate([row.rewards for row in trajectories], axis=0),
        conditional_ratios=np.concatenate(
            [row.conditional_ratios for row in trajectories], axis=0
        ),
        marginal_ratios=np.concatenate(
            [row.marginal_ratios for row in trajectories], axis=0
        ),
    )


def _state_weighted_policy_kl(
    occupancy: FloatArray, old_policy: FloatArray, new_policy: FloatArray
) -> float:
    per_state = np.sum(
        old_policy
        * np.log(
            np.maximum(old_policy, 1e-300) / np.maximum(new_policy, 1e-300)
        ),
        axis=-1,
    )
    return float(np.sum(occupancy * per_state) / occupancy.shape[0])


def scaled_logit_update(
    mdp: LayeredMDP,
    gradient: FloatArray,
    *,
    target_kl: float = 0.01,
    initial_state: int = 0,
) -> tuple[LayeredMDP, float, float]:
    """Scale an update direction to the frozen exact mean Actor KL."""
    gradient = np.asarray(gradient, dtype=np.float64)
    if gradient.shape != mdp.actor.shape:
        raise ValueError("gradient has the wrong shape")
    centered = gradient - gradient.mean(axis=-1, keepdims=True)
    if float(np.linalg.norm(centered)) == 0.0:
        return mdp, 0.0, 0.0
    occupancy = forward_occupancy(mdp, mdp.actor, initial_state)
    logits = np.log(mdp.actor)

    def policy_and_kl(step: float) -> tuple[FloatArray, float]:
        candidate = softmax(logits + step * centered, axis=-1)
        return candidate, _state_weighted_policy_kl(
            occupancy, mdp.actor, candidate
        )

    high = 1.0
    _, high_kl = policy_and_kl(high)
    while high_kl < target_kl:
        high *= 2.0
        _, high_kl = policy_and_kl(high)
        if high > 1e12:
            raise RuntimeError("could not reach target KL")
    low = 0.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        _, middle_kl = policy_and_kl(middle)
        if middle_kl < target_kl:
            low = middle
        else:
            high = middle
    step = 0.5 * (low + high)
    actor, achieved_kl = policy_and_kl(step)
    updated = LayeredMDP(
        rewards=mdp.rewards,
        transitions=mdp.transitions,
        actor=actor,
        gamma=mdp.gamma,
    )
    return updated, achieved_kl, step


def post_update_metrics(
    controller: SearchMDPPolicy,
    gradient: FloatArray,
    *,
    target_kl: float = 0.01,
    initial_state: int = 0,
) -> dict[str, float]:
    """Recompute both named objectives after the KL-matched Actor update."""
    old_mdp = controller.mdp
    updated_mdp, actor_kl, step = scaled_logit_update(
        old_mdp, gradient, target_kl=target_kl, initial_state=initial_state
    )
    old_raw = float(exact_values(old_mdp, old_mdp.actor)[0][0, initial_state])
    new_raw = float(
        exact_values(updated_mdp, updated_mdp.actor)[0][0, initial_state]
    )
    updated_controller = SearchMDPPolicy(
        mdp=updated_mdp,
        critic_scores=controller.critic_scores,
        candidate_count=controller.candidate_count,
        beta=controller.beta,
        epsilon=controller.epsilon,
        expansion_probabilities=controller.expansion_probabilities,
    )
    old_search = float(controller.values()[0][0, initial_state])
    new_search = float(updated_controller.values()[0][0, initial_state])
    search_occupancy = controller.occupancy(initial_state)
    composite_kl = _state_weighted_policy_kl(
        search_occupancy,
        controller.executed_policy,
        updated_controller.executed_policy,
    )
    return {
        "logit_step_scale": step,
        "post_update_actor_kl": actor_kl,
        "post_update_composite_kl": composite_kl,
        "raw_actor_return_before": old_raw,
        "raw_actor_return_after": new_raw,
        "raw_actor_return_change": new_raw - old_raw,
        "composite_search_return_before": old_search,
        "composite_search_return_after": new_search,
        "composite_search_return_change": new_search - old_search,
    }


def selected_raw_actor_estimators(
    controller: SearchMDPPolicy,
    trajectories: SampledSearchTrajectories,
) -> tuple[dict[str, dict[str, object]], dict[str, FloatArray]]:
    """Evaluate the frozen cross-family E0--E3 matrix on common data.

    S2 selected marginal step-cumulative lambda 0.75 at effective batch 256
    and lambda 1.0 at batch 1024.  Both are retained under explicit names; the
    latter is mathematically identical to exact cumulative E2 and is not counted
    as an independent method.
    """
    weight_specs = {
        "E0_standard_selected_action": np.ones_like(trajectories.rewards),
        "E1_local": partial_weight_matrix(
            trajectories.conditional_ratios, method="local", strength=1.0
        ),
        "E1_trajectory": partial_weight_matrix(
            trajectories.conditional_ratios,
            method="step_cumulative",
            strength=1.0,
        ),
        "E2_local_marginal": partial_weight_matrix(
            trajectories.marginal_ratios, method="local", strength=1.0
        ),
        "E2_cumulative_marginal": partial_weight_matrix(
            trajectories.marginal_ratios,
            method="step_cumulative",
            strength=1.0,
        ),
        "E3_step_cumulative_lambda_0.75": partial_weight_matrix(
            trajectories.marginal_ratios,
            method="step_cumulative",
            strength=0.75,
        ),
        "E3_step_cumulative_lambda_1": partial_weight_matrix(
            trajectories.marginal_ratios,
            method="step_cumulative",
            strength=1.0,
        ),
    }
    metrics: dict[str, dict[str, object]] = {}
    gradients: dict[str, FloatArray] = {}
    for name, weights in weight_specs.items():
        metrics[name], gradients[name] = _estimate_one_weighting_with_gradient(
            controller, trajectories, weights
        )
        metrics[name].update(post_update_metrics(controller, gradients[name]))
    return metrics, gradients
