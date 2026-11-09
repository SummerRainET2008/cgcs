from __future__ import annotations

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_mismatch_estimators import (
    complete_partial_frontier,
    complete_partial_frontier_with_gradients,
    concatenate_trajectories,
    partial_weight_matrix,
    post_update_metrics,
    sample_search_trajectories,
    selected_raw_actor_estimators,
)
from acbpo.search_policy_mismatch import SearchMDPPolicy


def controller(seed: int = 37) -> SearchMDPPolicy:
    mdp = LayeredMDP.random(
        seed=seed, horizon=4, state_count=4, action_count=3, gamma=0.95
    )
    _, q_values = mdp.values()
    return SearchMDPPolicy(
        mdp=mdp,
        critic_scores=q_values,
        candidate_count=2,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.full((mdp.horizon, mdp.state_count), 0.6),
    )


def test_partial_weight_endpoints_and_constructions() -> None:
    ratios = np.asarray([[0.5, 2.0, 1.5], [1.2, 0.8, 0.4]])
    ones = np.ones_like(ratios)
    for method in ("local", "step_cumulative", "trajectory_blend"):
        np.testing.assert_allclose(
            partial_weight_matrix(ratios, method=method, strength=0.0), ones
        )
    exact = np.cumprod(ratios, axis=1)
    np.testing.assert_allclose(
        partial_weight_matrix(ratios, method="step_cumulative", strength=1.0),
        exact,
    )
    np.testing.assert_allclose(
        partial_weight_matrix(ratios, method="trajectory_blend", strength=1.0),
        exact,
    )
    np.testing.assert_allclose(
        partial_weight_matrix(ratios, method="local", strength=1.0), ratios
    )


def test_sampling_is_deterministic_and_estimator_independent() -> None:
    search = controller()
    first = sample_search_trajectories(search, samples=2000, seed=81)
    second = sample_search_trajectories(search, samples=2000, seed=81)
    for field in ("states", "actions", "rewards", "conditional_ratios", "marginal_ratios"):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))


def test_complete_frontier_has_both_ratio_bases_all_methods_and_lambdas() -> None:
    search = controller(seed=41)
    trajectories = sample_search_trajectories(search, samples=5000, seed=82)
    result = complete_partial_frontier(search, trajectories)
    assert len(result) == 2 * 3 * 5
    assert "conditional_local_lambda_0" in result
    assert "marginal_step_cumulative_lambda_1" in result
    for metrics in result.values():
        assert np.isfinite(metrics["gradient_relative_mse_batch_256"])
        assert 0.0 < metrics["normalized_final_weight_ess"] <= 1.0


def test_lambda_zero_is_identical_across_ratio_bases_and_methods() -> None:
    search = controller(seed=43)
    trajectories = sample_search_trajectories(search, samples=3000, seed=83)
    result = complete_partial_frontier(search, trajectories)
    baseline = result["conditional_local_lambda_0"]
    for ratio in ("conditional", "marginal"):
        for method in ("local", "step_cumulative", "trajectory_blend"):
            candidate = result[f"{ratio}_{method}_lambda_0"]
            for key in (
                "relative_bias",
                "gradient_variance_trace",
                "return_mean",
                "normalized_final_weight_ess",
            ):
                np.testing.assert_allclose(candidate[key], baseline[key])


def test_concatenation_and_kl_matched_update() -> None:
    search = controller(seed=47)
    left = sample_search_trajectories(search, samples=2000, seed=84)
    right = sample_search_trajectories(search, samples=2000, seed=85)
    joined = concatenate_trajectories([left, right])
    assert joined.sample_count == 4000
    metrics, gradients = complete_partial_frontier_with_gradients(search, joined)
    name = "marginal_step_cumulative_lambda_1"
    assert name in metrics and name in gradients
    update = post_update_metrics(search, gradients[name])
    np.testing.assert_allclose(update["post_update_actor_kl"], 0.01, atol=1e-10)
    assert np.isfinite(update["raw_actor_return_change"])
    assert np.isfinite(update["composite_search_return_change"])


def test_selected_cross_family_estimators_match_complete_frontier() -> None:
    search = controller(seed=49)
    trajectories = sample_search_trajectories(search, samples=4000, seed=86)
    selected, gradients = selected_raw_actor_estimators(search, trajectories)
    frontier = complete_partial_frontier(search, trajectories)
    mapping = {
        "E0_standard_selected_action": "conditional_local_lambda_0",
        "E1_local": "conditional_local_lambda_1",
        "E1_trajectory": "conditional_step_cumulative_lambda_1",
        "E2_local_marginal": "marginal_local_lambda_1",
        "E2_cumulative_marginal": "marginal_step_cumulative_lambda_1",
        "E3_step_cumulative_lambda_0.75": "marginal_step_cumulative_lambda_0.75",
        "E3_step_cumulative_lambda_1": "marginal_step_cumulative_lambda_1",
    }
    assert set(selected) == set(mapping)
    assert set(gradients) == set(mapping)
    for selected_name, frontier_name in mapping.items():
        for key in (
            "relative_bias",
            "gradient_variance_trace",
            "gradient_mse_batch_256",
            "normalized_final_weight_ess",
        ):
            np.testing.assert_allclose(
                selected[selected_name][key], frontier[frontier_name][key]
            )
        assert "post_update_actor_kl" in selected[selected_name]
