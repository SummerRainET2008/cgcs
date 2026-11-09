from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidate_policy import CandidatePolicy


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _softmax(logits: FloatArray, axis: int = -1) -> FloatArray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


@dataclass(frozen=True)
class LayeredMDP:
    """A finite-horizon MDP with a fixed number of states in each layer."""

    rewards: FloatArray
    transitions: FloatArray
    actor: FloatArray
    gamma: float

    @property
    def horizon(self) -> int:
        return int(self.rewards.shape[0])

    @property
    def state_count(self) -> int:
        return int(self.rewards.shape[1])

    @property
    def action_count(self) -> int:
        return int(self.rewards.shape[2])

    @classmethod
    def random(
        cls,
        *,
        seed: int,
        horizon: int = 6,
        state_count: int = 7,
        action_count: int = 6,
        gamma: float = 0.97,
    ) -> "LayeredMDP":
        """Create a reproducible long-horizon problem with delayed rewards."""
        rng = np.random.default_rng(seed)
        actor_logits = rng.normal(scale=0.8, size=(horizon, state_count, action_count))
        actor = _softmax(actor_logits)

        transitions = np.empty(
            (horizon, state_count, action_count, state_count), dtype=np.float64
        )
        for t in range(horizon):
            for state in range(state_count):
                for action in range(action_count):
                    transitions[t, state, action] = rng.dirichlet(
                        np.full(state_count, 0.25)
                    )

        rewards = rng.normal(
            loc=-0.03, scale=0.12, size=(horizon, state_count, action_count)
        )
        terminal_quality = rng.normal(scale=1.8, size=(state_count, action_count))
        rewards[-1] += terminal_quality

        # Make action 0 superficially attractive early while its transitions
        # tend to reach states with poor terminal prospects. This creates a
        # state-distribution mismatch that single-step experiments cannot show.
        rewards[:-1, :, 0] += 0.30
        poor_states = np.argsort(terminal_quality.max(axis=1))[: max(2, state_count // 3)]
        poor_distribution = np.full(state_count, 0.03 / (state_count - len(poor_states)))
        poor_distribution[poor_states] = 0.97 / len(poor_states)
        for t in range(horizon - 1):
            transitions[t, :, 0, :] = poor_distribution

        return cls(rewards=rewards, transitions=transitions, actor=actor, gamma=gamma)

    def values(self) -> tuple[FloatArray, FloatArray]:
        """Return exact V^pi and Q^pi by backward dynamic programming."""
        values = np.zeros((self.horizon + 1, self.state_count), dtype=np.float64)
        q_values = np.zeros_like(self.rewards)
        for t in range(self.horizon - 1, -1, -1):
            continuation = np.einsum(
                "san,n->sa", self.transitions[t], values[t + 1]
            )
            q_values[t] = self.rewards[t] + self.gamma * continuation
            values[t] = np.sum(self.actor[t] * q_values[t], axis=1)
        return values, q_values


@dataclass(frozen=True)
class Trajectory:
    states: IntArray
    actions: IntArray
    rewards: FloatArray
    joint_ratios: FloatArray
    marginal_ratios: FloatArray
    mc_marginal_ratios: FloatArray


@dataclass
class CandidateMDPBehavior:
    mdp: LayeredMDP
    critic_q: FloatArray
    candidate_count: int
    beta: float
    epsilon: float

    def __post_init__(self) -> None:
        if self.critic_q.shape != self.mdp.rewards.shape:
            raise ValueError("critic_q has the wrong shape")
        self._candidate_policies = [
            [
                CandidatePolicy(
                    proposal=self.mdp.actor[t, state],
                    q_values=self.critic_q[t, state],
                    candidate_count=self.candidate_count,
                    beta=self.beta,
                    epsilon=self.epsilon,
                )
                for state in range(self.mdp.state_count)
            ]
            for t in range(self.mdp.horizon)
        ]
        self.exact_marginals = np.asarray(
            [
                [policy.exact_marginal() for policy in layer]
                for layer in self._candidate_policies
            ]
        )

    def sample_trajectory(
        self,
        rng: np.random.Generator,
        *,
        initial_state: int,
        phantom_sets: int,
        phantom_rng: np.random.Generator | None = None,
    ) -> Trajectory:
        if phantom_rng is None:
            phantom_rng = rng
        states = np.empty(self.mdp.horizon, dtype=np.int64)
        actions = np.empty(self.mdp.horizon, dtype=np.int64)
        rewards = np.empty(self.mdp.horizon, dtype=np.float64)
        joint_ratios = np.empty(self.mdp.horizon, dtype=np.float64)
        marginal_ratios = np.empty(self.mdp.horizon, dtype=np.float64)
        mc_marginal_ratios = np.empty(self.mdp.horizon, dtype=np.float64)

        state = initial_state
        for t in range(self.mdp.horizon):
            states[t] = state
            policy = self._candidate_policies[t][state]
            _, selector, selected_index, action = policy.sample(rng)
            actions[t] = action
            rewards[t] = self.mdp.rewards[t, state, action]
            joint_ratios[t] = 1.0 / (
                self.candidate_count * selector[selected_index]
            )
            marginal_ratios[t] = (
                self.mdp.actor[t, state, action]
                / self.exact_marginals[t, state, action]
            )
            estimated_marginal = policy.estimate_marginal_for_action(
                action, phantom_sets, phantom_rng
            )
            mc_marginal_ratios[t] = (
                self.mdp.actor[t, state, action] / max(estimated_marginal, 1e-12)
            )
            if t + 1 < self.mdp.horizon:
                state = int(
                    rng.choice(
                        self.mdp.state_count,
                        p=self.mdp.transitions[t, state, action],
                    )
                )
        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards,
            joint_ratios=joint_ratios,
            marginal_ratios=marginal_ratios,
            mc_marginal_ratios=mc_marginal_ratios,
        )
