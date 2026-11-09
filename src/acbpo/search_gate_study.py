"""Exact compute-matched gate and search-family comparisons."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .layered_mdp import LayeredMDP
from .search_policy_mismatch import (
    SearchMDPPolicy,
    exact_values,
    forward_occupancy,
)


FloatArray = NDArray[np.float64]


def standardize_per_layer(values: FloatArray) -> FloatArray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=1, keepdims=True)
    scale = np.maximum(values.std(axis=1, keepdims=True), 1e-12)
    return centered / scale


def sigmoid(values: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def mixed_policy(
    actor: FloatArray, expanded_policy: FloatArray, gate: FloatArray
) -> FloatArray:
    return (1.0 - gate[:, :, None]) * actor + gate[:, :, None] * expanded_policy


def realized_expansion_rate(
    mdp: LayeredMDP,
    expanded_policy: FloatArray,
    gate: FloatArray,
    *,
    initial_state: int = 0,
) -> tuple[float, FloatArray, FloatArray]:
    policy = mixed_policy(mdp.actor, expanded_policy, gate)
    occupancy = forward_occupancy(mdp, policy, initial_state)
    rate = float(np.sum(occupancy * gate) / mdp.horizon)
    return rate, policy, occupancy


def calibrate_gate(
    mdp: LayeredMDP,
    expanded_policy: FloatArray,
    score: FloatArray,
    *,
    target_expansion_rate: float,
    temperature: float = 1.0,
) -> FloatArray:
    """Solve the gate intercept against exact induced state occupancy."""
    if not 0.0 < target_expansion_rate < 1.0:
        if target_expansion_rate == 0.0:
            return np.zeros_like(score)
        if target_expansion_rate == 1.0:
            return np.ones_like(score)
        raise ValueError("target_expansion_rate must be in [0,1]")
    low, high = -50.0, 50.0
    for _ in range(150):
        middle = 0.5 * (low + high)
        gate = sigmoid(score / temperature + middle)
        rate, _, _ = realized_expansion_rate(mdp, expanded_policy, gate)
        if rate < target_expansion_rate:
            low = middle
        else:
            high = middle
    gate = sigmoid(score / temperature + 0.5 * (low + high))
    rate, _, _ = realized_expansion_rate(mdp, expanded_policy, gate)
    if abs(rate - target_expansion_rate) > 1e-10:
        raise AssertionError("exact occupancy calibration failed")
    return gate


def exact_expansion_gain(
    mdp: LayeredMDP, expanded_policy: FloatArray, q_pi: FloatArray
) -> FloatArray:
    return np.sum((expanded_policy - mdp.actor) * q_pi, axis=-1)


def gate_features(mdp: LayeredMDP) -> tuple[FloatArray, dict[str, FloatArray]]:
    values, _ = mdp.values()
    entropy = -np.sum(
        mdp.actor * np.log(np.maximum(mdp.actor, 1e-300)), axis=-1
    )
    low_value = standardize_per_layer(-values[:-1])
    entropy_score = standardize_per_layer(entropy)
    features = np.stack(
        [
            np.ones_like(low_value),
            low_value,
            entropy_score,
            low_value * entropy_score,
        ],
        axis=-1,
    )
    return features, {
        "random": np.zeros_like(low_value),
        "entropy": entropy_score,
        "value": low_value,
        "value_and_entropy": np.minimum(low_value, entropy_score),
    }


def fixed_feature_sufficiency_groups(
    mdp: LayeredMDP, q_values: FloatArray
) -> dict[str, FloatArray]:
    """The four frozen development-only gate feature groups."""
    primary, _ = gate_features(mdp)
    low_value = primary[..., 1]
    entropy = primary[..., 2]
    actor_sorted = np.sort(mdp.actor, axis=-1)
    top_probability = actor_sorted[..., -1]
    top_two_margin = actor_sorted[..., -1] - actor_sorted[..., -2]
    collision_probability = np.sum(np.square(mdp.actor), axis=-1)
    q_mean = np.sum(mdp.actor * q_values, axis=-1)
    q_centered = q_values - q_mean[..., None]
    q_std = np.sqrt(np.sum(mdp.actor * np.square(q_centered), axis=-1))
    best_q_gap = np.max(q_values, axis=-1) - q_mean
    log_probability = np.log(np.maximum(mdp.actor, 1e-300))
    mean_log_probability = np.sum(mdp.actor * log_probability, axis=-1)
    log_q_covariance = np.sum(
        mdp.actor
        * (log_probability - mean_log_probability[..., None])
        * q_centered,
        axis=-1,
    )
    degree_three = np.stack(
        [
            np.ones_like(low_value),
            low_value,
            entropy,
            np.square(low_value),
            low_value * entropy,
            np.square(entropy),
            np.power(low_value, 3),
            np.square(low_value) * entropy,
            low_value * np.square(entropy),
            np.power(entropy, 3),
        ],
        axis=-1,
    )
    actor_confidence = np.concatenate(
        [
            primary,
            np.stack(
                [top_probability, top_two_margin, collision_probability], axis=-1
            ),
        ],
        axis=-1,
    )
    critic_shape = np.concatenate(
        [
            primary,
            np.stack([q_std, best_q_gap, log_q_covariance], axis=-1),
        ],
        axis=-1,
    )
    union = np.concatenate(
        [
            primary,
            np.stack(
                [
                    top_probability,
                    top_two_margin,
                    collision_probability,
                    q_std,
                    best_q_gap,
                    log_q_covariance,
                    entropy * q_std,
                    low_value * best_q_gap,
                ],
                axis=-1,
            ),
        ],
        axis=-1,
    )
    return {
        "value_entropy_degree_3": degree_three,
        "actor_confidence": actor_confidence,
        "critic_shape": critic_shape,
        "union_with_interactions": union,
    }


def fit_ridge_gate(
    feature_rows: FloatArray,
    targets: FloatArray,
    *,
    penalty: float = 1e-3,
) -> FloatArray:
    feature_rows = np.asarray(feature_rows, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if feature_rows.ndim != 2 or targets.shape != (feature_rows.shape[0],):
        raise ValueError("feature/target shapes do not match")
    gram = feature_rows.T @ feature_rows
    regularizer = penalty * np.eye(feature_rows.shape[1])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(gram + regularizer, feature_rows.T @ targets)


def weighted_oracle_labels(
    gain: FloatArray, occupancy: FloatArray, target_rate: float
) -> FloatArray:
    """Fractional top-gain labels with exactly matched occupancy mass."""
    flat_gain = gain.ravel()
    flat_weight = occupancy.ravel() / occupancy.shape[0]
    order = np.argsort(-flat_gain, kind="stable")
    labels = np.zeros_like(flat_gain)
    remaining = target_rate
    for index in order:
        weight = flat_weight[index]
        if remaining <= 0.0:
            break
        if weight <= remaining:
            labels[index] = 1.0
            remaining -= weight
        elif weight > 0.0:
            labels[index] = remaining / weight
            remaining = 0.0
    if abs(float(np.dot(flat_weight, labels)) - target_rate) > 1e-10:
        raise AssertionError("oracle label mass does not match target")
    return labels.reshape(gain.shape)


def weighted_auc(
    scores: FloatArray, labels: FloatArray, weights: FloatArray
) -> float:
    scores = scores.ravel()
    labels = labels.ravel()
    weights = weights.ravel()
    positive = weights * labels
    negative = weights * (1.0 - labels)
    denominator = float(positive.sum() * negative.sum())
    if denominator == 0.0:
        return 0.5
    comparison = scores[:, None] - scores[None, :]
    credit = (comparison > 0.0).astype(np.float64) + 0.5 * (
        comparison == 0.0
    )
    return float(np.sum(positive[:, None] * negative[None, :] * credit) / denominator)


@dataclass(frozen=True)
class GateEvaluation:
    metrics: dict[str, float | list[float]]
    gate: FloatArray
    policy: FloatArray


def evaluate_gate(
    mdp: LayeredMDP,
    expanded_policy: FloatArray,
    gate: FloatArray,
    *,
    oracle_gain: FloatArray,
    target_expansion_rate: float,
    candidate_count: int,
    initial_state: int = 0,
) -> GateEvaluation:
    rate, policy, occupancy = realized_expansion_rate(
        mdp, expanded_policy, gate, initial_state=initial_state
    )
    raw_values, _ = exact_values(mdp, mdp.actor)
    values, _ = exact_values(mdp, policy)
    raw_occupancy = forward_occupancy(mdp, mdp.actor, initial_state)
    oracle_labels = weighted_oracle_labels(
        oracle_gain, raw_occupancy, target_expansion_rate
    )
    allocation_mass = float(np.sum(occupancy * gate))
    allocated_gain = float(
        np.sum(occupancy * gate * oracle_gain) / max(allocation_mass, 1e-300)
    )
    state_action_tv = 0.5 * np.abs(
        occupancy[:, :, None] * policy
        - raw_occupancy[:, :, None] * mdp.actor
    ).sum(axis=(1, 2))
    brier = float(
        np.sum(raw_occupancy * np.square(gate - oracle_labels)) / mdp.horizon
    )
    marginal_second_moment = np.sum(
        np.square(mdp.actor) / np.maximum(policy, 1e-300), axis=-1
    )
    one_step_marginal_ess = float(
        1.0
        / (np.sum(occupancy * marginal_second_moment) / mdp.horizon)
    )
    expected_unique_by_state = np.sum(
        1.0 - np.power(1.0 - mdp.actor, candidate_count), axis=-1
    )
    identical_by_state = np.sum(np.power(mdp.actor, candidate_count), axis=-1)
    mean_unique = float(
        np.sum(occupancy * gate * expected_unique_by_state)
        / max(allocation_mass, 1e-300)
    )
    identical_fraction = float(
        np.sum(occupancy * gate * identical_by_state)
        / max(allocation_mass, 1e-300)
    )
    return GateEvaluation(
        metrics={
            "realized_expansion_rate": rate,
            "expected_generated_candidates": 1.0
            + (candidate_count - 1.0) * rate,
            "raw_actor_return": float(raw_values[0, initial_state]),
            "composite_search_return": float(values[0, initial_state]),
            "return_gain_vs_raw": float(
                values[0, initial_state] - raw_values[0, initial_state]
            ),
            "allocation_gain_per_expansion": allocated_gain,
            "gate_auc_vs_oracle_allocation": weighted_auc(
                gate, oracle_labels, raw_occupancy
            ),
            "gate_brier_vs_oracle_allocation": brier,
            "one_step_exact_marginal_ess_proxy": one_step_marginal_ess,
            "mean_unique_candidates_when_expanded": mean_unique,
            "all_candidates_identical_probability": identical_fraction,
            "state_action_tv_by_depth": state_action_tv.tolist(),
            "maximum_depth_state_action_tv": float(state_action_tv[-1]),
        },
        gate=gate,
        policy=policy,
    )


def expected_selector_entropy_by_state(
    actor: FloatArray,
    critic_scores: FloatArray,
    *,
    candidate_count: int,
    beta: float,
    epsilon: float,
) -> FloatArray:
    """Enumerate E_C[H(q(.|s,C))] for every layered state."""
    action_count = actor.shape[-1]
    tuples = np.asarray(
        list(product(range(action_count), repeat=candidate_count)), dtype=np.int64
    )
    output = np.zeros(actor.shape[:-1], dtype=np.float64)
    for index in np.ndindex(actor.shape[:-1]):
        proposal = actor[index]
        candidate_probability = np.prod(proposal[tuples], axis=-1)
        logits = beta * critic_scores[index][tuples]
        logits -= logits.max(axis=-1, keepdims=True)
        guided = np.exp(logits)
        guided /= guided.sum(axis=-1, keepdims=True)
        selector = (1.0 - epsilon) * guided + epsilon / candidate_count
        entropy = -np.sum(
            selector * np.log(np.maximum(selector, 1e-300)), axis=-1
        )
        output[index] = float(np.dot(candidate_probability, entropy))
    return output


def full_expansion_policy(
    mdp: LayeredMDP,
    critic_scores: FloatArray,
    *,
    candidate_count: int,
    beta: float,
    epsilon: float = 0.10,
) -> FloatArray:
    return SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic_scores,
        candidate_count=candidate_count,
        beta=beta,
        epsilon=epsilon,
        expansion_probabilities=np.ones((mdp.horizon, mdp.state_count)),
    ).executed_policy
