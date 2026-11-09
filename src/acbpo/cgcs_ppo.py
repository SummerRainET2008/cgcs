from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SelectorDiagnostics:
    entropy: float
    effective_sample_size: float
    maximum_probability: float
    selection_weight: float


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum()


def standardized(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("values must be a non-empty vector")
    scale = float(array.std())
    if scale < 1e-8:
        return np.zeros_like(array)
    return (array - array.mean()) / scale


def selector_distribution(
    actor_logprobs: Sequence[float],
    critic_scores: Sequence[float],
    *,
    beta: float,
    temperature: float = 1.0,
    epsilon: float = 0.05,
    critic_scale_floor: float = 0.05,
) -> np.ndarray:
    """Return the conditional CGCS behavior distribution q(i | s, C)."""
    actor = np.asarray(actor_logprobs, dtype=np.float64)
    critic = np.asarray(critic_scores, dtype=np.float64)
    if actor.ndim != 1 or not len(actor) or actor.shape != critic.shape:
        raise ValueError("actor and critic scores must be same-size non-empty vectors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("epsilon must be in [0, 1)")
    if critic_scale_floor <= 0.0:
        raise ValueError("critic_scale_floor must be positive")
    centered_critic = critic - critic.mean()
    normalized_critic = centered_critic / max(float(critic.std()), critic_scale_floor)
    shaped = _softmax((actor + beta * normalized_critic) / temperature)
    return (1.0 - epsilon) * shaped + epsilon / len(actor)


def uniform_selection_weight(
    selected_probability: float, candidate_count: int, *, cap: float | None = None
) -> float:
    """Correct critic selection back to uniform selection conditional on C."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not 0.0 < selected_probability <= 1.0:
        raise ValueError("selected_probability must be in (0, 1]")
    weight = 1.0 / (candidate_count * selected_probability)
    return min(weight, cap) if cap is not None else weight


def effective_sample_size(weights: Sequence[float]) -> float:
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or not len(array) or np.any(array < 0.0):
        raise ValueError("weights must be a non-empty non-negative vector")
    denominator = float(np.square(array).sum())
    return 0.0 if denominator == 0.0 else float(array.sum() ** 2 / denominator)


def selector_diagnostics(probabilities: Sequence[float], selected: int) -> SelectorDiagnostics:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or not len(probabilities):
        raise ValueError("probabilities must be a non-empty vector")
    if selected < 0 or selected >= len(probabilities):
        raise ValueError("selected index is outside the candidate set")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30))))
    weight = uniform_selection_weight(float(probabilities[selected]), len(probabilities))
    return SelectorDiagnostics(
        entropy=entropy,
        effective_sample_size=effective_sample_size(
            [uniform_selection_weight(float(p), len(probabilities)) for p in probabilities]
        ),
        maximum_probability=float(probabilities.max()),
        selection_weight=weight,
    )


def weighted_ppo_loss(
    new_logprob,
    old_logprob,
    advantage,
    selection_weight,
    *,
    clip_epsilon: float,
):
    """PPO loss with selection IS outside the actor-ratio clipping operation.

    Arguments are torch tensors. Keeping torch as a runtime import lets the
    probability utilities remain usable in the lightweight numpy-only install.
    """
    import torch

    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1)")
    actor_ratio = torch.exp(new_logprob - old_logprob)
    clipped_ratio = torch.clamp(actor_ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(actor_ratio * advantage, clipped_ratio * advantage)
    return -(selection_weight * surrogate).mean(), actor_ratio
