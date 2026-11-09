"""Exact finite-MDP utilities for search-induced policy mismatch.

The module keeps two targets explicit:

* the raw Actor policy, which is the deployment target when search is removed;
* the composite search policy, whose extended action is either one raw action or
  an ordered candidate set together with the selected candidate index.

All candidate sets are enumerated.  The code is intentionally CPU-only and is
small enough to serve as a mathematical reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from .layered_mdp import LayeredMDP


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def softmax(logits: FloatArray, axis: int = -1) -> FloatArray:
    shifted = np.asarray(logits, dtype=np.float64)
    shifted = shifted - shifted.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


def _validate_probabilities(probabilities: FloatArray, name: str) -> FloatArray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 1 or probabilities.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with >=2 entries")
    if np.any(probabilities <= 0.0):
        raise ValueError(f"{name} must have full support")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-12):
        raise ValueError(f"{name} must sum to one")
    return probabilities


@dataclass(frozen=True)
class ExtendedOutcome:
    """One outcome in the composite policy's extended action space."""

    expanded: bool
    candidates: tuple[int, ...]
    selected_index: int
    action: int
    probability: float


@dataclass(frozen=True)
class SearchStatePolicy:
    """Exact trigger + IID candidate generator + selector at one state."""

    proposal: FloatArray
    critic_scores: FloatArray
    candidate_count: int
    beta: float
    epsilon: float
    expansion_probability: float

    def __post_init__(self) -> None:
        proposal = _validate_probabilities(self.proposal, "proposal")
        scores = np.asarray(self.critic_scores, dtype=np.float64)
        if scores.shape != proposal.shape:
            raise ValueError("critic_scores must have the same shape as proposal")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if not 0.0 <= self.expansion_probability <= 1.0:
            raise ValueError("expansion_probability must be in [0, 1]")
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "critic_scores", scores)

    @property
    def action_count(self) -> int:
        return int(self.proposal.size)

    def selector(self, candidates: tuple[int, ...] | IntArray) -> FloatArray:
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.shape != (self.candidate_count,):
            raise ValueError("candidates have the wrong shape")
        guided = softmax(self.beta * self.critic_scores[candidates])
        return (1.0 - self.epsilon) * guided + self.epsilon / self.candidate_count

    def outcomes(self) -> Iterator[ExtendedOutcome]:
        unexpanded_mass = 1.0 - self.expansion_probability
        if unexpanded_mass > 0.0:
            for action, probability in enumerate(self.proposal):
                yield ExtendedOutcome(
                    expanded=False,
                    candidates=(action,),
                    selected_index=0,
                    action=action,
                    probability=float(unexpanded_mass * probability),
                )

        expanded_mass = self.expansion_probability
        if expanded_mass == 0.0:
            return
        for candidate_tuple in product(
            range(self.action_count), repeat=self.candidate_count
        ):
            candidate_probability = float(
                np.prod(self.proposal[np.asarray(candidate_tuple, dtype=np.int64)])
            )
            selector = self.selector(candidate_tuple)
            for index, action in enumerate(candidate_tuple):
                yield ExtendedOutcome(
                    expanded=True,
                    candidates=candidate_tuple,
                    selected_index=index,
                    action=action,
                    probability=float(
                        expanded_mass * candidate_probability * selector[index]
                    ),
                )

    def outcome_probability(self, outcome: ExtendedOutcome) -> float:
        if not outcome.expanded:
            return float(
                (1.0 - self.expansion_probability) * self.proposal[outcome.action]
            )
        candidates = np.asarray(outcome.candidates, dtype=np.int64)
        candidate_probability = float(np.prod(self.proposal[candidates]))
        selector = self.selector(candidates)
        return float(
            self.expansion_probability
            * candidate_probability
            * selector[outcome.selected_index]
        )

    def executed_marginal(self) -> FloatArray:
        marginal = (1.0 - self.expansion_probability) * self.proposal.copy()
        if self.expansion_probability == 0.0:
            return marginal
        tuples = np.asarray(
            list(product(range(self.action_count), repeat=self.candidate_count)),
            dtype=np.int64,
        )
        candidate_probability = np.prod(self.proposal[tuples], axis=1)
        guided = softmax(self.beta * self.critic_scores[tuples], axis=-1)
        selector = (
            (1.0 - self.epsilon) * guided
            + self.epsilon / self.candidate_count
        )
        for index in range(self.candidate_count):
            np.add.at(
                marginal,
                tuples[:, index],
                self.expansion_probability
                * candidate_probability
                * selector[:, index],
            )
        return marginal

    def conditional_propensity_weight(self, outcome: ExtendedOutcome) -> float:
        """E1 weight mapping selection within a logged set to uniform selection."""
        if not outcome.expanded:
            return 1.0
        selector = self.selector(outcome.candidates)
        return float(
            1.0 / (self.candidate_count * selector[outcome.selected_index])
        )

    def raw_actor_marginal_weight(self, outcome: ExtendedOutcome) -> float:
        """E2 local ratio pi(a|s)/mu(a|s)."""
        marginal = self.executed_marginal()
        return float(self.proposal[outcome.action] / marginal[outcome.action])

    def executed_marginal_ratio_to(
        self, target: "SearchStatePolicy", outcome: ExtendedOutcome
    ) -> float:
        """E4 executed-action marginal ratio mu_target(a)/mu_source(a)."""
        source = self.executed_marginal()
        target_marginal = target.executed_marginal()
        return float(target_marginal[outcome.action] / source[outcome.action])

    def extended_ratio_to(
        self,
        target: "SearchStatePolicy",
        outcome: ExtendedOutcome,
        *,
        include_all_candidates: bool = True,
    ) -> float:
        """E4 ratio from this composite policy to ``target``.

        ``include_all_candidates=False`` is a deliberately wrong ablation used
        by a required regression test.  It retains only the selected candidate's
        Actor likelihood ratio.
        """
        if self.candidate_count != target.candidate_count:
            raise ValueError("source and target candidate_count must match")
        source_probability = self.outcome_probability(outcome)
        if source_probability <= 0.0:
            raise ZeroDivisionError("source outcome has zero support")
        if not outcome.expanded or include_all_candidates:
            return target.outcome_probability(outcome) / source_probability

        candidates = np.asarray(outcome.candidates, dtype=np.int64)
        selected_action = int(candidates[outcome.selected_index])
        proposal_ratio = (
            target.proposal[selected_action] / self.proposal[selected_action]
        )
        source_selector = self.selector(candidates)[outcome.selected_index]
        target_selector = target.selector(candidates)[outcome.selected_index]
        trigger_ratio = target.expansion_probability / self.expansion_probability
        return float(
            trigger_ratio
            * proposal_ratio
            * target_selector
            / source_selector
        )


def exact_values(
    mdp: LayeredMDP, policy: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return exact values for an arbitrary Markov policy in ``mdp``."""
    policy = np.asarray(policy, dtype=np.float64)
    if policy.shape != mdp.actor.shape:
        raise ValueError("policy has the wrong shape")
    if np.any(policy < 0.0) or not np.allclose(policy.sum(axis=-1), 1.0):
        raise ValueError("policy must contain probability distributions")
    values = np.zeros((mdp.horizon + 1, mdp.state_count), dtype=np.float64)
    q_values = np.zeros_like(mdp.rewards)
    for t in range(mdp.horizon - 1, -1, -1):
        continuation = np.einsum(
            "san,n->sa", mdp.transitions[t], values[t + 1]
        )
        q_values[t] = mdp.rewards[t] + mdp.gamma * continuation
        values[t] = np.sum(policy[t] * q_values[t], axis=-1)
    return values, q_values


def forward_occupancy(
    mdp: LayeredMDP, policy: FloatArray, initial_state: int = 0
) -> FloatArray:
    occupancy = np.zeros((mdp.horizon, mdp.state_count), dtype=np.float64)
    occupancy[0, initial_state] = 1.0
    for t in range(mdp.horizon - 1):
        occupancy[t + 1] = np.einsum(
            "s,sa,san->n", occupancy[t], policy[t], mdp.transitions[t]
        )
    return occupancy


@dataclass
class SearchMDPPolicy:
    """Composite policy over every state of a ``LayeredMDP``."""

    mdp: LayeredMDP
    critic_scores: FloatArray
    candidate_count: int
    beta: float
    epsilon: float
    expansion_probabilities: FloatArray

    def __post_init__(self) -> None:
        self.critic_scores = np.asarray(self.critic_scores, dtype=np.float64)
        self.expansion_probabilities = np.asarray(
            self.expansion_probabilities, dtype=np.float64
        )
        if self.critic_scores.shape != self.mdp.actor.shape:
            raise ValueError("critic_scores has the wrong shape")
        expected_gate_shape = (self.mdp.horizon, self.mdp.state_count)
        if self.expansion_probabilities.shape != expected_gate_shape:
            raise ValueError("expansion_probabilities has the wrong shape")
        self.state_policies = [
            [
                SearchStatePolicy(
                    proposal=self.mdp.actor[t, state],
                    critic_scores=self.critic_scores[t, state],
                    candidate_count=self.candidate_count,
                    beta=self.beta,
                    epsilon=self.epsilon,
                    expansion_probability=float(
                        self.expansion_probabilities[t, state]
                    ),
                )
                for state in range(self.mdp.state_count)
            ]
            for t in range(self.mdp.horizon)
        ]
        self.executed_policy = np.asarray(
            [
                [policy.executed_marginal() for policy in layer]
                for layer in self.state_policies
            ],
            dtype=np.float64,
        )

    def values(self) -> tuple[FloatArray, FloatArray]:
        return exact_values(self.mdp, self.executed_policy)

    def occupancy(self, initial_state: int = 0) -> FloatArray:
        return forward_occupancy(self.mdp, self.executed_policy, initial_state)


def raw_actor_gradient(mdp: LayeredMDP, initial_state: int = 0) -> FloatArray:
    """Exact gradient of J_actor with respect to tabular Actor logits."""
    values, q_values = exact_values(mdp, mdp.actor)
    occupancy = forward_occupancy(mdp, mdp.actor, initial_state)
    advantages = q_values - values[:-1, :, None]
    discount = mdp.gamma ** np.arange(mdp.horizon)
    return (
        discount[:, None, None]
        * occupancy[:, :, None]
        * mdp.actor
        * advantages
    )


def composite_search_gradient(
    controller: SearchMDPPolicy, initial_state: int = 0
) -> FloatArray:
    """Exact gradient of J_search for fixed trigger and selector parameters.

    Expanded outcomes include the score of every generated candidate.  The
    selector is treated as fixed with respect to Actor logits, matching the
    protocol's fixed-Critic reference condition.
    """
    mdp = controller.mdp
    _, q_values = controller.values()
    occupancy = controller.occupancy(initial_state)
    gradient = np.zeros_like(mdp.actor)
    discount = mdp.gamma ** np.arange(mdp.horizon)
    for t in range(mdp.horizon):
        for state in range(mdp.state_count):
            proposal = mdp.actor[t, state]
            state_gradient = np.zeros(mdp.action_count, dtype=np.float64)
            for outcome in controller.state_policies[t][state].outcomes():
                counts = np.zeros(mdp.action_count, dtype=np.float64)
                if outcome.expanded:
                    np.add.at(counts, np.asarray(outcome.candidates), 1.0)
                    score = counts - controller.candidate_count * proposal
                else:
                    score = -proposal.copy()
                    score[outcome.action] += 1.0
                state_gradient += (
                    outcome.probability * q_values[t, state, outcome.action] * score
                )
            gradient[t, state] = (
                discount[t] * occupancy[t, state] * state_gradient
            )
    return gradient


def actor_score_gradient(
    mdp: LayeredMDP,
    *,
    occupancy: FloatArray,
    action_measure: FloatArray,
    advantages: FloatArray,
) -> FloatArray:
    """Expected raw-Actor score update under a named state/action measure."""
    expected_shapes = (
        (mdp.horizon, mdp.state_count),
        mdp.actor.shape,
        mdp.actor.shape,
    )
    if (
        occupancy.shape,
        action_measure.shape,
        advantages.shape,
    ) != expected_shapes:
        raise ValueError("occupancy, action_measure, or advantages has wrong shape")
    terms = action_measure * advantages
    centered = terms - mdp.actor * terms.sum(axis=-1, keepdims=True)
    discount = mdp.gamma ** np.arange(mdp.horizon)
    return discount[:, None, None] * occupancy[:, :, None] * centered


def estimator_expected_gradients(
    controller: SearchMDPPolicy, initial_state: int = 0
) -> dict[str, FloatArray]:
    """Exact expected gradients for the primary raw-Actor correction contrast.

    Exact A^pi is used for every raw-Actor-target estimator so the contrast
    isolates action and occupancy mismatch rather than Critic error.
    """
    mdp = controller.mdp
    raw_values, raw_q = exact_values(mdp, mdp.actor)
    raw_advantage = raw_q - raw_values[:-1, :, None]
    raw_occupancy = forward_occupancy(mdp, mdp.actor, initial_state)
    search_occupancy = controller.occupancy(initial_state)
    raw = actor_score_gradient(
        mdp,
        occupancy=raw_occupancy,
        action_measure=mdp.actor,
        advantages=raw_advantage,
    )
    return {
        "raw_actor_target": raw,
        "E0_standard_selected_action": actor_score_gradient(
            mdp,
            occupancy=search_occupancy,
            action_measure=controller.executed_policy,
            advantages=raw_advantage,
        ),
        "E1_local": actor_score_gradient(
            mdp,
            occupancy=search_occupancy,
            action_measure=mdp.actor,
            advantages=raw_advantage,
        ),
        "E1_trajectory": raw.copy(),
        "E2_cumulative_marginal": raw.copy(),
    }


def gradient_diagnostics(estimate: FloatArray, target: FloatArray) -> dict[str, float]:
    difference = np.asarray(estimate) - np.asarray(target)
    target_norm = float(np.linalg.norm(target))
    estimate_norm = float(np.linalg.norm(estimate))
    relative_bias = (
        float(np.linalg.norm(difference) / target_norm)
        if target_norm > 0.0
        else float("inf")
    )
    cosine = (
        float(np.vdot(estimate, target) / (estimate_norm * target_norm))
        if estimate_norm > 0.0 and target_norm > 0.0
        else 0.0
    )
    nonzero = np.abs(target) > 1e-14
    sign_error = float(
        np.mean(np.sign(estimate[nonzero]) != np.sign(target[nonzero]))
    )
    return {
        "relative_bias": relative_bias,
        "cosine": cosine,
        "component_sign_error": sign_error,
    }


def occupancy_diagnostics(
    mdp: LayeredMDP, policy: FloatArray, initial_state: int = 0
) -> dict[str, list[float] | float]:
    raw_occupancy = forward_occupancy(mdp, mdp.actor, initial_state)
    behavior_occupancy = forward_occupancy(mdp, policy, initial_state)
    raw_state_action = raw_occupancy[:, :, None] * mdp.actor
    behavior_state_action = behavior_occupancy[:, :, None] * policy
    state_tv = 0.5 * np.abs(behavior_occupancy - raw_occupancy).sum(axis=-1)
    state_action_tv = 0.5 * np.abs(
        behavior_state_action - raw_state_action
    ).sum(axis=(1, 2))
    state_kl = np.sum(
        behavior_occupancy
        * np.log(
            np.maximum(behavior_occupancy, 1e-300)
            / np.maximum(raw_occupancy, 1e-300)
        ),
        axis=-1,
    )
    return {
        "state_tv_by_depth": state_tv.tolist(),
        "state_action_tv_by_depth": state_action_tv.tolist(),
        "state_kl_search_to_actor_by_depth": state_kl.tolist(),
        "maximum_depth_state_action_tv": float(state_action_tv[-1]),
    }


def mdp_with_actor_logits(mdp: LayeredMDP, logits: FloatArray) -> LayeredMDP:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != mdp.actor.shape:
        raise ValueError("logits have the wrong shape")
    return LayeredMDP(
        rewards=mdp.rewards,
        transitions=mdp.transitions,
        actor=softmax(logits, axis=-1),
        gamma=mdp.gamma,
    )


def central_finite_difference_gradient(
    objective,
    logits: FloatArray,
    *,
    step: float = 1e-6,
) -> FloatArray:
    """Central finite-difference gradient for a scalar objective on logits."""
    logits = np.asarray(logits, dtype=np.float64)
    gradient = np.zeros_like(logits)
    for index in np.ndindex(logits.shape):
        plus = logits.copy()
        minus = logits.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def construction_diagnostics(mdp: LayeredMDP) -> dict[str, float | int | bool]:
    """Auditable checks for the preregistered environment pathologies."""
    values, q_values = mdp.values()
    entropy = -np.sum(
        mdp.actor * np.log(np.maximum(mdp.actor, 1e-300)), axis=-1
    )
    state_values = values[:-1]
    low_value = state_values <= np.median(state_values)
    low_entropy = entropy <= np.median(entropy)
    high_entropy = entropy > np.median(entropy)
    immediate_best = np.argmax(mdp.rewards, axis=-1)
    long_term_best = np.argmax(q_values, axis=-1)
    misleading = immediate_best != long_term_best
    terminal_abs = float(np.abs(mdp.rewards[-1]).sum())
    early_abs = float(np.abs(mdp.rewards[:-1]).sum())
    return {
        "delayed_terminal_reward_dominates_per_layer": bool(
            terminal_abs > early_abs / max(1, mdp.horizon - 1)
        ),
        "misleading_local_reward_states": int(misleading.sum()),
        "low_value_low_entropy_states": int((low_value & low_entropy).sum()),
        "low_value_high_entropy_states": int((low_value & high_entropy).sum()),
        "terminal_absolute_reward": terminal_abs,
        "mean_early_layer_absolute_reward": early_abs
        / max(1, mdp.horizon - 1),
    }


def sample_categorical(probabilities: FloatArray, uniforms: FloatArray) -> IntArray:
    """Sample rows using supplied uniforms, enabling exact common randomness."""
    cumulative = np.cumsum(probabilities, axis=-1)
    return np.sum(uniforms[:, None] > cumulative, axis=-1).astype(np.int64)


def candidate_pools_from_uniforms(
    proposals: FloatArray, uniforms: FloatArray
) -> IntArray:
    """Generate ordered candidate pools independently of any selector."""
    proposals = np.asarray(proposals, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if proposals.ndim != 2 or uniforms.ndim != 2:
        raise ValueError("proposals and uniforms must both be matrices")
    if proposals.shape[0] != uniforms.shape[0]:
        raise ValueError("proposals and uniforms must share the row dimension")
    output = np.empty(uniforms.shape, dtype=np.int64)
    for position in range(uniforms.shape[1]):
        output[:, position] = sample_categorical(
            proposals, uniforms[:, position]
        )
    return output


def monte_carlo_cumulative_recovery(
    controller: SearchMDPPolicy,
    *,
    samples: int,
    seed: int,
    initial_state: int = 0,
) -> dict[str, dict[str, float]]:
    """Vectorized S0 recovery check for E1 and E2 cumulative ratios."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    mdp = controller.mdp
    rng = np.random.default_rng(seed)
    states = np.full(samples, initial_state, dtype=np.int64)
    behavior_return = np.zeros(samples, dtype=np.float64)
    local_conditional_return = np.zeros(samples, dtype=np.float64)
    cumulative_conditional_return = np.zeros(samples, dtype=np.float64)
    cumulative_marginal_return = np.zeros(samples, dtype=np.float64)
    cumulative_conditional = np.ones(samples, dtype=np.float64)
    cumulative_marginal = np.ones(samples, dtype=np.float64)

    for t in range(mdp.horizon):
        row = np.arange(samples)
        proposals = mdp.actor[t, states]
        gates = controller.expansion_probabilities[t, states]
        expanded = rng.random(samples) < gates
        actions = sample_categorical(proposals, rng.random(samples))
        conditional_ratio = np.ones(samples, dtype=np.float64)

        expanded_rows = np.flatnonzero(expanded)
        if expanded_rows.size:
            expanded_proposals = proposals[expanded_rows]
            candidates = candidate_pools_from_uniforms(
                expanded_proposals,
                rng.random((expanded_rows.size, controller.candidate_count)),
            )
            scores = controller.critic_scores[t, states[expanded_rows]][
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
            actions[expanded_rows] = candidates[
                np.arange(expanded_rows.size), selected
            ]
            selected_probability = selector[
                np.arange(expanded_rows.size), selected
            ]
            conditional_ratio[expanded_rows] = 1.0 / (
                controller.candidate_count * selected_probability
            )

        executed = controller.executed_policy[t, states, actions]
        marginal_ratio = proposals[row, actions] / executed
        rewards = mdp.rewards[t, states, actions]
        discount = mdp.gamma**t
        cumulative_conditional *= conditional_ratio
        cumulative_marginal *= marginal_ratio
        behavior_return += discount * rewards
        local_conditional_return += discount * conditional_ratio * rewards
        cumulative_conditional_return += (
            discount * cumulative_conditional * rewards
        )
        cumulative_marginal_return += discount * cumulative_marginal * rewards

        if t + 1 < mdp.horizon:
            transition = mdp.transitions[t, states, actions]
            states = sample_categorical(transition, rng.random(samples))

    def summarize(values: FloatArray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(samples)),
            "variance": float(values.var(ddof=1)),
        }

    return {
        "behavior": summarize(behavior_return),
        "e1_local": summarize(local_conditional_return),
        "e1_trajectory": summarize(cumulative_conditional_return),
        "e2_cumulative": summarize(cumulative_marginal_return),
    }
