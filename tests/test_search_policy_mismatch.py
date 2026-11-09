from __future__ import annotations

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_policy_mismatch import (
    SearchMDPPolicy,
    SearchStatePolicy,
    candidate_pools_from_uniforms,
    central_finite_difference_gradient,
    composite_search_gradient,
    construction_diagnostics,
    estimator_expected_gradients,
    exact_values,
    mdp_with_actor_logits,
    raw_actor_gradient,
    gradient_diagnostics,
)


def state_policy(
    proposal=(0.15, 0.35, 0.50),
    scores=(-1.0, 0.5, 2.0),
    *,
    candidate_count=3,
    beta=1.7,
    epsilon=0.10,
    expansion=0.6,
):
    return SearchStatePolicy(
        proposal=np.asarray(proposal, dtype=np.float64),
        critic_scores=np.asarray(scores, dtype=np.float64),
        candidate_count=candidate_count,
        beta=beta,
        epsilon=epsilon,
        expansion_probability=expansion,
    )


def test_k_one_and_uniform_selector_recover_raw_actor() -> None:
    k_one = state_policy(candidate_count=1, beta=20.0, expansion=1.0)
    uniform = state_policy(candidate_count=4, beta=0.0, expansion=0.8)
    np.testing.assert_allclose(k_one.executed_marginal(), k_one.proposal, atol=1e-12)
    np.testing.assert_allclose(uniform.executed_marginal(), uniform.proposal, atol=1e-12)


def test_behavior_normalizes_and_has_full_support() -> None:
    policy = state_policy()
    outcomes = list(policy.outcomes())
    np.testing.assert_allclose(sum(row.probability for row in outcomes), 1.0, atol=1e-12)
    marginal = policy.executed_marginal()
    assert np.all(marginal > 0.0)
    np.testing.assert_allclose(marginal.sum(), 1.0, atol=1e-12)


def test_conditional_and_marginal_corrections_recover_raw_actor_target() -> None:
    policy = state_policy()
    test_function = np.asarray([-2.0, 0.25, 3.0])
    truth = float(np.dot(policy.proposal, test_function))
    conditional = 0.0
    marginal = 0.0
    for outcome in policy.outcomes():
        conditional += (
            outcome.probability
            * policy.conditional_propensity_weight(outcome)
            * test_function[outcome.action]
        )
        marginal += (
            outcome.probability
            * policy.raw_actor_marginal_weight(outcome)
            * test_function[outcome.action]
        )
    np.testing.assert_allclose(conditional, truth, atol=1e-12)
    np.testing.assert_allclose(marginal, truth, atol=1e-12)


def test_complete_extended_ratio_recovers_composite_target_but_selected_only_does_not() -> None:
    source = state_policy(expansion=0.45)
    target = state_policy(
        proposal=(0.40, 0.25, 0.35),
        scores=(0.7, -0.2, 1.1),
        expansion=0.75,
    )

    def outcome_value(outcome):
        candidate_sum = sum((index + 1) * action for index, action in enumerate(outcome.candidates))
        return 0.3 * candidate_sum + outcome.action + 0.2 * outcome.selected_index

    target_truth = sum(
        row.probability * outcome_value(row) for row in target.outcomes()
    )
    complete = 0.0
    selected_only = 0.0
    for outcome in source.outcomes():
        value = outcome_value(outcome)
        complete += (
            outcome.probability
            * source.extended_ratio_to(target, outcome)
            * value
        )
        selected_only += (
            outcome.probability
            * source.extended_ratio_to(
                target, outcome, include_all_candidates=False
            )
            * value
        )
    np.testing.assert_allclose(complete, target_truth, atol=1e-12)
    assert abs(selected_only - target_truth) > 1e-3


def test_executed_marginal_ratio_recovers_composite_action_expectation() -> None:
    source = state_policy(expansion=0.35)
    target = state_policy(
        proposal=(0.25, 0.60, 0.15),
        scores=(1.0, -0.5, 0.25),
        expansion=0.70,
    )
    action_value = np.asarray([1.5, -0.3, 2.2])
    truth = float(np.dot(target.executed_marginal(), action_value))
    recovered = sum(
        outcome.probability
        * source.executed_marginal_ratio_to(target, outcome)
        * action_value[outcome.action]
        for outcome in source.outcomes()
    )
    np.testing.assert_allclose(recovered, truth, atol=1e-12)


def test_candidate_pools_are_identical_across_estimators_and_selectors() -> None:
    proposal = np.asarray([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    uniforms = np.random.default_rng(91).random((2, 4))
    first = candidate_pools_from_uniforms(proposal, uniforms)
    second = candidate_pools_from_uniforms(proposal, uniforms.copy())
    np.testing.assert_array_equal(first, second)
    # Candidate generation sees neither Critic scores nor estimator labels.
    assert first.shape == (2, 4)


def _small_mdp(seed: int = 17) -> LayeredMDP:
    return LayeredMDP.random(
        seed=seed, horizon=3, state_count=3, action_count=3, gamma=0.93
    )


def test_exact_raw_actor_gradient_matches_central_finite_difference() -> None:
    mdp = _small_mdp()
    logits = np.log(mdp.actor)
    analytic = raw_actor_gradient(mdp)

    def objective(candidate_logits):
        candidate_mdp = mdp_with_actor_logits(mdp, candidate_logits)
        return float(exact_values(candidate_mdp, candidate_mdp.actor)[0][0, 0])

    finite = central_finite_difference_gradient(objective, logits)
    relative_error = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
    assert relative_error <= 1e-5


def test_exact_composite_gradient_includes_all_candidate_scores() -> None:
    mdp = _small_mdp(seed=23)
    _, critic = mdp.values()
    gate = np.full((mdp.horizon, mdp.state_count), 0.65)
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic,
        candidate_count=2,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=gate,
    )
    analytic = composite_search_gradient(controller)
    logits = np.log(mdp.actor)

    def objective(candidate_logits):
        candidate_mdp = mdp_with_actor_logits(mdp, candidate_logits)
        candidate_controller = SearchMDPPolicy(
            mdp=candidate_mdp,
            critic_scores=critic,
            candidate_count=2,
            beta=1.0,
            epsilon=0.10,
            expansion_probabilities=gate,
        )
        return float(candidate_controller.values()[0][0, 0])

    finite = central_finite_difference_gradient(objective, logits)
    relative_error = np.linalg.norm(analytic - finite) / np.linalg.norm(finite)
    assert relative_error <= 1e-5


def test_smoke_environment_suite_contains_preregistered_pathologies() -> None:
    for seed in (1000, 1001, 1002):
        diagnostics = construction_diagnostics(LayeredMDP.random(seed=seed))
        assert diagnostics["delayed_terminal_reward_dominates_per_layer"]
        assert diagnostics["misleading_local_reward_states"] > 0
        assert diagnostics["low_value_low_entropy_states"] > 0
        assert diagnostics["low_value_high_entropy_states"] > 0


def test_exact_cumulative_estimators_recover_raw_gradient_and_local_does_not() -> None:
    mdp = _small_mdp(seed=29)
    _, critic = mdp.values()
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic,
        candidate_count=2,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.full((mdp.horizon, mdp.state_count), 0.7),
    )
    gradients = estimator_expected_gradients(controller)
    target = gradients["raw_actor_target"]
    assert gradient_diagnostics(
        gradients["E1_trajectory"], target
    )["relative_bias"] < 1e-12
    assert gradient_diagnostics(
        gradients["E2_cumulative_marginal"], target
    )["relative_bias"] < 1e-12
    assert gradient_diagnostics(gradients["E1_local"], target)["relative_bias"] > 1e-4
