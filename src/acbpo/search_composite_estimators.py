"""Finite-sample E4 estimators for the composite search-policy target."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .search_mismatch_estimators import (
    BATCH_SIZES,
    SampledSearchTrajectories,
    post_update_metrics,
)
from .search_policy_mismatch import (
    SearchMDPPolicy,
    actor_score_gradient,
    candidate_pools_from_uniforms,
    composite_search_gradient,
    gradient_diagnostics,
    sample_categorical,
    softmax,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class SampledCompositeTrajectories:
    states: IntArray
    actions: IntArray
    rewards: FloatArray
    expanded: BoolArray
    candidates: IntArray
    selected_indices: IntArray

    @property
    def sample_count(self) -> int:
        return int(self.states.shape[0])


def concatenate_composite_trajectories(
    trajectories: list[SampledCompositeTrajectories],
) -> SampledCompositeTrajectories:
    if not trajectories:
        raise ValueError("at least one trajectory batch is required")
    return SampledCompositeTrajectories(
        states=np.concatenate([row.states for row in trajectories], axis=0),
        actions=np.concatenate([row.actions for row in trajectories], axis=0),
        rewards=np.concatenate([row.rewards for row in trajectories], axis=0),
        expanded=np.concatenate([row.expanded for row in trajectories], axis=0),
        candidates=np.concatenate([row.candidates for row in trajectories], axis=0),
        selected_indices=np.concatenate(
            [row.selected_indices for row in trajectories], axis=0
        ),
    )


def sample_composite_trajectories(
    controller: SearchMDPPolicy,
    *,
    samples: int,
    seed: int,
    initial_state: int = 0,
) -> SampledCompositeTrajectories:
    """Sample extended outcomes once for all three E4 estimators."""
    if samples < 2:
        raise ValueError("samples must be at least two")
    mdp = controller.mdp
    rng = np.random.default_rng(seed)
    states = np.empty((samples, mdp.horizon), dtype=np.int64)
    actions = np.empty_like(states)
    rewards = np.empty((samples, mdp.horizon), dtype=np.float64)
    expanded = np.zeros((samples, mdp.horizon), dtype=np.bool_)
    candidates = np.full(
        (samples, mdp.horizon, controller.candidate_count),
        -1,
        dtype=np.int64,
    )
    selected_indices = np.zeros_like(states)
    current_states = np.full(samples, initial_state, dtype=np.int64)

    for t in range(mdp.horizon):
        states[:, t] = current_states
        proposals = mdp.actor[t, current_states]
        gates = controller.expansion_probabilities[t, current_states]
        # Draw a fixed block regardless of gate decisions or K.  K=2 uses the
        # first two candidate columns from the same K=4 block, preserving common
        # random numbers across every primary search family.
        gate_uniform = rng.random(samples)
        actor_uniform = rng.random(samples)
        candidate_uniform = rng.random(
            (samples, max(4, controller.candidate_count))
        )
        selector_uniform = rng.random(samples)
        transition_uniform = rng.random(samples)
        is_expanded = gate_uniform < gates
        expanded[:, t] = is_expanded
        current_actions = sample_categorical(proposals, actor_uniform)
        candidates[:, t, 0] = current_actions

        expanded_rows = np.flatnonzero(is_expanded)
        if expanded_rows.size:
            candidate_set = candidate_pools_from_uniforms(
                proposals[expanded_rows],
                candidate_uniform[
                    expanded_rows, : controller.candidate_count
                ],
            )
            candidates[expanded_rows, t] = candidate_set
            scores = controller.critic_scores[t, current_states[expanded_rows]][
                np.arange(expanded_rows.size)[:, None], candidate_set
            ]
            guided = softmax(controller.beta * scores, axis=-1)
            selector = (
                (1.0 - controller.epsilon) * guided
                + controller.epsilon / controller.candidate_count
            )
            selected = sample_categorical(
                selector, selector_uniform[expanded_rows]
            )
            selected_indices[expanded_rows, t] = selected
            current_actions[expanded_rows] = candidate_set[
                np.arange(expanded_rows.size), selected
            ]

        actions[:, t] = current_actions
        rewards[:, t] = mdp.rewards[t, current_states, current_actions]
        if t + 1 < mdp.horizon:
            transition = mdp.transitions[t, current_states, current_actions]
            current_states = sample_categorical(
                transition, transition_uniform
            )

    return SampledCompositeTrajectories(
        states=states,
        actions=actions,
        rewards=rewards,
        expanded=expanded,
        candidates=candidates,
        selected_indices=selected_indices,
    )


def as_raw_actor_trajectories(
    controller: SearchMDPPolicy,
    trajectories: SampledCompositeTrajectories,
) -> SampledSearchTrajectories:
    """Project the same logged extended outcomes onto E0--E3 fields."""
    mdp = controller.mdp
    conditional = np.ones_like(trajectories.rewards)
    marginal = np.ones_like(trajectories.rewards)
    rows = np.arange(trajectories.sample_count)
    for t in range(mdp.horizon):
        state = trajectories.states[:, t]
        action = trajectories.actions[:, t]
        proposal = mdp.actor[t, state]
        marginal[:, t] = proposal[rows, action] / (
            controller.executed_policy[t, state, action]
        )
        expanded_rows = np.flatnonzero(trajectories.expanded[:, t])
        if expanded_rows.size:
            candidate_set = trajectories.candidates[expanded_rows, t]
            scores = controller.critic_scores[t, state[expanded_rows]][
                np.arange(expanded_rows.size)[:, None], candidate_set
            ]
            guided = softmax(controller.beta * scores, axis=-1)
            selector = (
                (1.0 - controller.epsilon) * guided
                + controller.epsilon / controller.candidate_count
            )
            selected = trajectories.selected_indices[expanded_rows, t]
            conditional[expanded_rows, t] = 1.0 / (
                controller.candidate_count
                * selector[np.arange(expanded_rows.size), selected]
            )
    return SampledSearchTrajectories(
        states=trajectories.states,
        actions=trajectories.actions,
        rewards=trajectories.rewards,
        conditional_ratios=conditional,
        marginal_ratios=marginal,
    )


def _extended_score(
    proposal: FloatArray,
    *,
    expanded: bool,
    candidates: IntArray,
    action: int,
) -> FloatArray:
    if not expanded:
        score = -proposal.copy()
        score[action] += 1.0
        return score
    counts = np.bincount(candidates, minlength=proposal.size).astype(np.float64)
    return counts - candidates.size * proposal


def executed_marginal_score_table(controller: SearchMDPPolicy) -> FloatArray:
    """Return d log mu(a|s) / d raw-Actor logits for every state and action."""
    mdp = controller.mdp
    table = np.zeros(
        (
            mdp.horizon,
            mdp.state_count,
            mdp.action_count,
            mdp.action_count,
        ),
        dtype=np.float64,
    )
    for t in range(mdp.horizon):
        for state in range(mdp.state_count):
            state_policy = controller.state_policies[t][state]
            derivative = np.zeros((mdp.action_count, mdp.action_count))
            proposal = mdp.actor[t, state]
            gate = state_policy.expansion_probability
            action_eye = np.eye(mdp.action_count, dtype=np.float64)
            derivative += (
                (1.0 - gate)
                * proposal[:, None]
                * (action_eye - proposal[None, :])
            )
            if gate > 0.0:
                tuples = np.asarray(
                    list(
                        product(
                            range(mdp.action_count),
                            repeat=controller.candidate_count,
                        )
                    ),
                    dtype=np.int64,
                )
                candidate_probability = np.prod(proposal[tuples], axis=1)
                guided = softmax(
                    controller.beta * controller.critic_scores[t, state][tuples],
                    axis=-1,
                )
                selector = (
                    (1.0 - controller.epsilon) * guided
                    + controller.epsilon / controller.candidate_count
                )
                score = (
                    action_eye[tuples].sum(axis=1)
                    - controller.candidate_count * proposal[None, :]
                )
                for index in range(controller.candidate_count):
                    probability = (
                        gate * candidate_probability * selector[:, index]
                    )
                    np.add.at(
                        derivative,
                        tuples[:, index],
                        probability[:, None] * score,
                    )
            marginal = controller.executed_policy[t, state]
            table[t, state] = derivative / marginal[:, None]
    return table


def exact_e4_gradients(controller: SearchMDPPolicy) -> dict[str, FloatArray]:
    """Enumerate the expectations of all frozen E4 gradient estimators."""
    mdp = controller.mdp
    values, q_values = controller.values()
    advantages = q_values - values[:-1, :, None]
    occupancy = controller.occupancy()
    marginal_scores = executed_marginal_score_table(controller)
    output = {
        name: np.zeros_like(mdp.actor)
        for name in (
            "E4_executed_action_marginal",
            "E4_extended_candidate_set",
            "E4_selected_action_selector_proxy",
        )
    }
    for t in range(mdp.horizon):
        for state in range(mdp.state_count):
            proposal = mdp.actor[t, state]
            for outcome in controller.state_policies[t][state].outcomes():
                scalar = (
                    (mdp.gamma**t)
                    * occupancy[t, state]
                    * outcome.probability
                    * advantages[t, state, outcome.action]
                )
                output["E4_executed_action_marginal"][t, state] += (
                    scalar * marginal_scores[t, state, outcome.action]
                )
                output["E4_extended_candidate_set"][t, state] += scalar * (
                    _extended_score(
                        proposal,
                        expanded=outcome.expanded,
                        candidates=np.asarray(outcome.candidates, dtype=np.int64),
                        action=outcome.action,
                    )
                )
                selected_score = -proposal.copy()
                selected_score[outcome.action] += 1.0
                output["E4_selected_action_selector_proxy"][t, state] += (
                    scalar * selected_score
                )
    return output


def fast_exact_e4_gradients(
    controller: SearchMDPPolicy,
    *,
    marginal_scores: FloatArray | None = None,
) -> dict[str, FloatArray]:
    """Compute the same exact E4 expectations without enumerating outcomes.

    The extended-candidate result is the exact composite gradient by definition;
    the independent enumerated implementation above remains the regression oracle.
    """
    mdp = controller.mdp
    values, q_values = controller.values()
    advantages = q_values - values[:-1, :, None]
    occupancy = controller.occupancy()
    if marginal_scores is None:
        marginal_scores = executed_marginal_score_table(controller)
    discount = mdp.gamma ** np.arange(mdp.horizon)
    marginal = (
        discount[:, None, None]
        * occupancy[:, :, None]
        * np.einsum(
            "tsa,tsap->tsp",
            controller.executed_policy * advantages,
            marginal_scores,
        )
    )
    extended = composite_search_gradient(controller)
    proxy = actor_score_gradient(
        mdp,
        occupancy=occupancy,
        action_measure=controller.executed_policy,
        advantages=advantages,
    )
    return {
        "E4_executed_action_marginal": marginal,
        "E4_extended_candidate_set": extended,
        "E4_selected_action_selector_proxy": proxy,
    }


def estimate_e4_gradients(
    controller: SearchMDPPolicy,
    trajectories: SampledCompositeTrajectories,
    *,
    marginal_scores: FloatArray | None = None,
) -> tuple[dict[str, dict[str, object]], dict[str, FloatArray]]:
    """Estimate all E4 gradients from one common trajectory sample."""
    mdp = controller.mdp
    n = trajectories.sample_count
    values, q_values = controller.values()
    advantages = q_values - values[:-1, :, None]
    if marginal_scores is None:
        marginal_scores = executed_marginal_score_table(controller)
    names = (
        "E4_executed_action_marginal",
        "E4_extended_candidate_set",
        "E4_selected_action_selector_proxy",
    )
    sums = {name: np.zeros_like(mdp.actor) for name in names}
    second_norms = {name: 0.0 for name in names}

    action_eye = np.eye(mdp.action_count, dtype=np.float64)
    rows = np.arange(n)
    for t in range(mdp.horizon):
        state = trajectories.states[:, t]
        action = trajectories.actions[:, t]
        proposal = mdp.actor[t, state]
        scalar = (mdp.gamma**t) * advantages[t, state, action]
        selected_score = -proposal.copy()
        selected_score[rows, action] += 1.0
        extended_score = selected_score.copy()
        expanded_rows = np.flatnonzero(trajectories.expanded[:, t])
        if expanded_rows.size:
            candidate_set = trajectories.candidates[expanded_rows, t]
            counts = action_eye[candidate_set].sum(axis=1)
            extended_score[expanded_rows] = (
                counts
                - controller.candidate_count * proposal[expanded_rows]
            )
        score_by_name = {
            "E4_executed_action_marginal": marginal_scores[t, state, action],
            "E4_extended_candidate_set": extended_score,
            "E4_selected_action_selector_proxy": selected_score,
        }
        for name, score in score_by_name.items():
            contribution = scalar[:, None] * score
            np.add.at(sums[name][t], state, contribution)
            second_norms[name] += float(np.square(contribution).sum())

    target = composite_search_gradient(controller)
    truth_return = float(values[0, 0])
    discount = mdp.gamma ** np.arange(mdp.horizon)
    sampled_returns = np.sum(
        discount[None, :] * trajectories.rewards, axis=1
    )
    metrics: dict[str, dict[str, object]] = {}
    gradients: dict[str, FloatArray] = {}
    target_norm_squared = float(np.square(target).sum())
    for name in names:
        gradient = sums[name] / n
        gradients[name] = gradient
        diagnostics = gradient_diagnostics(gradient, target)
        variance_trace = max(
            0.0, second_norms[name] / n - float(np.square(gradient).sum())
        )
        bias_squared = float(np.square(gradient - target).sum())
        row: dict[str, object] = {
            **diagnostics,
            "gradient_bias_squared": bias_squared,
            "gradient_variance_trace": variance_trace,
            "normalized_final_weight_ess": 1.0,
            "maximum_final_weight": 1.0,
            "support_violations": 0,
            "final_weight_quantiles_0_50_90_95_99_100": [1.0] * 6,
            "return_mean": float(sampled_returns.mean()),
            "return_truth": truth_return,
            "return_bias": float(sampled_returns.mean() - truth_return),
            "return_variance": float(sampled_returns.var(ddof=1)),
        }
        for batch_size in BATCH_SIZES:
            mse = bias_squared + variance_trace / batch_size
            row[f"gradient_mse_batch_{batch_size}"] = mse
            row[f"gradient_relative_mse_batch_{batch_size}"] = (
                mse / target_norm_squared
            )
        row.update(post_update_metrics(controller, gradient))
        metrics[name] = row
    return metrics, gradients
