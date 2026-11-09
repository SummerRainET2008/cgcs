"""Frozen Critic-quality conditions for the exact-truth study."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .layered_mdp import LayeredMDP
from .search_mismatch_estimators import scaled_logit_update
from .search_policy_mismatch import (
    exact_values,
    raw_actor_gradient,
    sample_categorical,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def standardize_q(values: FloatArray) -> FloatArray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def ranking_concordance(signal: FloatArray, truth: FloatArray) -> float:
    favorable = 0.0
    comparisons = 0
    for left in range(truth.shape[-1]):
        for right in range(left + 1, truth.shape[-1]):
            truth_delta = truth[..., left] - truth[..., right]
            signal_delta = signal[..., left] - signal[..., right]
            valid = truth_delta != 0.0
            favorable += float(np.sum((truth_delta * signal_delta > 0.0) & valid))
            favorable += 0.5 * float(np.sum((signal_delta == 0.0) & valid))
            comparisons += int(np.sum(valid))
    return favorable / comparisons


def deterministic_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(b"cgcs-critic-quality-v1:" + payload).digest()
    return int.from_bytes(digest[:8], "big") % 2_000_000_000


def lagged_q_at_kl(
    mdp: LayeredMDP, *, target_kl: float
) -> tuple[FloatArray, float]:
    direction = -raw_actor_gradient(mdp)
    lagged_mdp, achieved_kl, _ = scaled_logit_update(
        mdp, direction, target_kl=target_kl
    )
    _, q_values = exact_values(lagged_mdp, lagged_mdp.actor)
    return q_values, achieved_kl


def sample_raw_actor_trajectories(
    mdp: LayeredMDP, *, samples: int, seed: int, initial_state: int = 0
) -> tuple[IntArray, IntArray, FloatArray]:
    rng = np.random.default_rng(seed)
    states = np.empty((samples, mdp.horizon), dtype=np.int64)
    actions = np.empty_like(states)
    rewards = np.empty((samples, mdp.horizon), dtype=np.float64)
    current_state = np.full(samples, initial_state, dtype=np.int64)
    rows = np.arange(samples)
    for t in range(mdp.horizon):
        states[:, t] = current_state
        probabilities = mdp.actor[t, current_state]
        action = sample_categorical(probabilities, rng.random(samples))
        actions[:, t] = action
        rewards[:, t] = mdp.rewards[t, current_state, action]
        if t + 1 < mdp.horizon:
            transition = mdp.transitions[t, current_state, action]
            current_state = sample_categorical(transition, rng.random(samples))
    return states, actions, rewards


def fit_tabular_ridge_q(
    mdp: LayeredMDP,
    *,
    trajectories: int,
    seed: int,
    penalty: float = 1e-3,
) -> tuple[FloatArray, dict[str, float]]:
    states, actions, rewards = sample_raw_actor_trajectories(
        mdp, samples=trajectories, seed=seed
    )
    returns = np.zeros_like(rewards)
    continuation = np.zeros(trajectories, dtype=np.float64)
    for t in range(mdp.horizon - 1, -1, -1):
        continuation = rewards[:, t] + mdp.gamma * continuation
        returns[:, t] = continuation
    sums = np.zeros_like(mdp.rewards)
    counts = np.zeros_like(mdp.rewards)
    for t in range(mdp.horizon):
        np.add.at(sums[t], (states[:, t], actions[:, t]), returns[:, t])
        np.add.at(counts[t], (states[:, t], actions[:, t]), 1.0)
    estimate = sums / (counts + penalty)
    return estimate, {
        "trajectories": trajectories,
        "visited_cells": int(np.sum(counts > 0.0)),
        "total_cells": int(counts.size),
        "minimum_nonzero_count": float(counts[counts > 0.0].min()),
        "maximum_count": float(counts.max()),
    }


@dataclass(frozen=True)
class CriticCondition:
    name: str
    raw_q: FloatArray
    selector_scores: FloatArray
    ranking_concordance: float
    metadata: dict[str, float | int]


def build_critic_conditions(mdp: LayeredMDP, *, mdp_seed: int) -> list[CriticCondition]:
    _, exact_q = mdp.values()
    exact_scores = standardize_q(exact_q)
    conditions = [
        CriticCondition(
            name="exact",
            raw_q=exact_q,
            selector_scores=exact_scores,
            ranking_concordance=1.0,
            metadata={},
        )
    ]
    noise = np.random.default_rng(
        deterministic_seed(mdp_seed, "shared_noise")
    ).normal(size=exact_q.shape)
    for sigma in (0.5, 1.0, 2.0):
        raw = exact_scores + sigma * noise
        score = standardize_q(raw)
        conditions.append(
            CriticCondition(
                name=f"noise_{sigma:g}",
                raw_q=raw,
                selector_scores=score,
                ranking_concordance=ranking_concordance(score, exact_q),
                metadata={"sigma": sigma},
            )
        )
    conditions.append(
        CriticCondition(
            name="inverted_diagnostic",
            raw_q=-exact_q,
            selector_scores=-exact_scores,
            ranking_concordance=0.0,
            metadata={},
        )
    )
    for target_kl in (0.01, 0.04):
        raw, achieved_kl = lagged_q_at_kl(mdp, target_kl=target_kl)
        score = standardize_q(raw)
        conditions.append(
            CriticCondition(
                name=f"lagged_kl_{target_kl:.2f}",
                raw_q=raw,
                selector_scores=score,
                ranking_concordance=ranking_concordance(score, exact_q),
                metadata={"target_kl": target_kl, "achieved_kl": achieved_kl},
            )
        )
    for trajectories in (256, 1024):
        raw, metadata = fit_tabular_ridge_q(
            mdp,
            trajectories=trajectories,
            seed=deterministic_seed(mdp_seed, "learned", trajectories),
            penalty=1e-3,
        )
        score = standardize_q(raw)
        conditions.append(
            CriticCondition(
                name=f"learned_{trajectories}",
                raw_q=raw,
                selector_scores=score,
                ranking_concordance=ranking_concordance(score, exact_q),
                metadata=metadata,
            )
        )
    return conditions
