from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidate_policy import CandidatePolicy


FloatArray = NDArray[np.float64]


def softmax_score(action: int, proposal: FloatArray) -> FloatArray:
    """Gradient of log softmax probability with respect to actor logits."""
    score = -proposal.copy()
    score[action] += 1.0
    return score


def exact_policy_gradient(proposal: FloatArray, rewards: FloatArray) -> FloatArray:
    gradient = np.zeros_like(proposal)
    for action, probability in enumerate(proposal):
        gradient += probability * rewards[action] * softmax_score(action, proposal)
    return gradient


@dataclass(frozen=True)
class EstimatorSummary:
    mean: FloatArray
    bias_norm: float
    variance_trace: float
    cosine: float


def summarize(samples: FloatArray, truth: FloatArray) -> EstimatorSummary:
    mean = samples.mean(axis=0)
    centered = samples - mean
    variance_trace = float(np.mean(np.sum(centered * centered, axis=1)))
    bias_norm = float(np.linalg.norm(mean - truth))
    denominator = float(np.linalg.norm(mean) * np.linalg.norm(truth))
    cosine = float(np.dot(mean, truth) / denominator) if denominator > 0.0 else 1.0
    return EstimatorSummary(mean, bias_norm, variance_trace, cosine)


def collect_gradient_samples(
    policy: CandidatePolicy,
    rewards: FloatArray,
    sample_count: int,
    phantom_sets: int,
    seed: int,
) -> dict[str, FloatArray]:
    """Collect per-sample gradients under several correction schemes."""
    rng = np.random.default_rng(seed)
    exact_marginal = policy.exact_marginal()
    output = {
        name: np.zeros((sample_count, policy.action_count), dtype=np.float64)
        for name in ("uncorrected", "joint", "marginal", "mc_marginal")
    }

    for row in range(sample_count):
        _, selector, index, action = policy.sample(rng)
        base = rewards[action] * softmax_score(action, policy.proposal)
        joint_weight = 1.0 / (policy.candidate_count * selector[index])
        marginal_weight = policy.proposal[action] / exact_marginal[action]
        estimated_marginal = policy.estimate_marginal_for_action(
            action, phantom_sets, rng
        )
        mc_weight = policy.proposal[action] / max(estimated_marginal, 1e-12)

        output["uncorrected"][row] = base
        output["joint"][row] = joint_weight * base
        output["marginal"][row] = marginal_weight * base
        output["mc_marginal"][row] = mc_weight * base
    return output
