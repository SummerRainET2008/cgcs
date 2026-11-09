from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _validate_probability_vector(probabilities: FloatArray) -> FloatArray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError("proposal must be a one-dimensional vector with >=2 actions")
    if np.any(probabilities <= 0.0):
        raise ValueError("proposal probabilities must have full support")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("proposal probabilities must sum to one")
    return probabilities


@dataclass(frozen=True)
class CandidatePolicy:
    """IID candidate generator followed by a critic-guided selector.

    Candidates are ordered and sampled with replacement. The selector uses a
    defensive uniform mixture so every candidate index has non-zero support.
    """

    proposal: FloatArray
    q_values: FloatArray
    candidate_count: int
    beta: float
    epsilon: float = 0.05

    def __post_init__(self) -> None:
        proposal = _validate_probability_vector(self.proposal)
        q_values = np.asarray(self.q_values, dtype=np.float64)
        if q_values.shape != proposal.shape:
            raise ValueError("q_values must have the same shape as proposal")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "q_values", q_values)

    @property
    def action_count(self) -> int:
        return int(self.proposal.size)

    def selector(self, candidates: IntArray) -> FloatArray:
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.shape != (self.candidate_count,):
            raise ValueError("candidates have the wrong shape")
        logits = self.beta * self.q_values[candidates]
        logits -= logits.max()
        guided = np.exp(logits)
        guided /= guided.sum()
        uniform = np.full(self.candidate_count, 1.0 / self.candidate_count)
        return (1.0 - self.epsilon) * guided + self.epsilon * uniform

    def sample(
        self, rng: np.random.Generator
    ) -> tuple[IntArray, FloatArray, int, int]:
        candidates = rng.choice(
            self.action_count,
            size=self.candidate_count,
            replace=True,
            p=self.proposal,
        ).astype(np.int64)
        selector = self.selector(candidates)
        selected_index = int(rng.choice(self.candidate_count, p=selector))
        return candidates, selector, selected_index, int(candidates[selected_index])

    def exact_marginal(self) -> FloatArray:
        """Enumerate the true executed-action behavior distribution."""
        marginal = np.zeros(self.action_count, dtype=np.float64)
        action_ids = range(self.action_count)
        for candidate_tuple in product(action_ids, repeat=self.candidate_count):
            candidates = np.asarray(candidate_tuple, dtype=np.int64)
            candidate_probability = float(np.prod(self.proposal[candidates]))
            selector = self.selector(candidates)
            for index, action in enumerate(candidates):
                marginal[action] += candidate_probability * selector[index]
        return marginal

    def estimate_marginal_for_action(
        self,
        action: int,
        phantom_sets: int,
        rng: np.random.Generator,
    ) -> float:
        """Estimate mu_K(a) using candidate sets conditioned on A_1=a."""
        if not 0 <= action < self.action_count:
            raise ValueError("invalid action")
        if phantom_sets < 1:
            raise ValueError("phantom_sets must be positive")
        if self.candidate_count == 1:
            return float(self.proposal[action])
        candidates = np.empty(
            (phantom_sets, self.candidate_count), dtype=np.int64
        )
        candidates[:, 0] = action
        candidates[:, 1:] = rng.choice(
            self.action_count,
            size=(phantom_sets, self.candidate_count - 1),
            replace=True,
            p=self.proposal,
        )
        logits = self.beta * self.q_values[candidates]
        logits -= logits.max(axis=1, keepdims=True)
        exponentials = np.exp(logits)
        guided_first_mass = exponentials[:, 0] / exponentials.sum(axis=1)
        selector_first_mass = (
            (1.0 - self.epsilon) * guided_first_mass
            + self.epsilon / self.candidate_count
        )
        conditional_mass = float(selector_first_mass.mean())
        return self.candidate_count * self.proposal[action] * conditional_mass
