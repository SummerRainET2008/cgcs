from __future__ import annotations

import numpy as np

from acbpo.critic_quality import standardize_q
from acbpo.layered_mdp import LayeredMDP
from acbpo.search_composite_estimators import (
    as_raw_actor_trajectories,
    concatenate_composite_trajectories,
    estimate_e4_gradients,
    exact_e4_gradients,
    executed_marginal_score_table,
    fast_exact_e4_gradients,
    sample_composite_trajectories,
)
from acbpo.search_family_matrix import (
    build_adaptive_controller,
    build_best_of_n_controller,
)
from acbpo.search_policy_mismatch import composite_search_gradient


def make_controller(seed: int = 81):
    mdp = LayeredMDP.random(
        seed=seed, horizon=4, state_count=4, action_count=4, gamma=0.97
    )
    _, q_values = mdp.values()
    controller = build_adaptive_controller(
        mdp,
        standardize_q(q_values),
        candidate_count=4,
        budget=2.0,
        gate_name="entropy",
        selector_name="value",
    )
    return mdp, controller


def test_executed_marginal_score_has_zero_mean() -> None:
    _, controller = make_controller()
    scores = executed_marginal_score_table(controller)
    expected = np.sum(controller.executed_policy[..., None] * scores, axis=2)
    np.testing.assert_allclose(expected, 0.0, atol=1e-12, rtol=0.0)


def test_exact_marginal_and_extended_e4_recover_composite_gradient() -> None:
    _, controller = make_controller(seed=82)
    expected = composite_search_gradient(controller)
    gradients = exact_e4_gradients(controller)
    np.testing.assert_allclose(
        gradients["E4_executed_action_marginal"], expected, atol=1e-11, rtol=0.0
    )
    np.testing.assert_allclose(
        gradients["E4_extended_candidate_set"], expected, atol=1e-11, rtol=0.0
    )
    assert (
        np.linalg.norm(
            gradients["E4_selected_action_selector_proxy"] - expected
        )
        / np.linalg.norm(expected)
        > 1e-3
    )
    fast = fast_exact_e4_gradients(controller)
    for name in gradients:
        np.testing.assert_allclose(fast[name], gradients[name], atol=1e-11, rtol=0.0)


def test_composite_sampler_is_deterministic_and_logs_complete_candidates() -> None:
    _, controller = make_controller(seed=83)
    first = sample_composite_trajectories(controller, samples=128, seed=901)
    second = sample_composite_trajectories(controller, samples=128, seed=901)
    for field in (
        "states",
        "actions",
        "rewards",
        "expanded",
        "candidates",
        "selected_indices",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
    assert np.all(first.candidates[first.expanded] >= 0)
    selected = np.take_along_axis(
        first.candidates,
        first.selected_indices[..., None],
        axis=2,
    )[..., 0]
    np.testing.assert_array_equal(selected[first.expanded], first.actions[first.expanded])


def test_all_three_estimators_share_returns_but_not_gradient_scores() -> None:
    _, controller = make_controller(seed=84)
    trajectories = sample_composite_trajectories(
        controller, samples=2048, seed=902
    )
    metrics, gradients = estimate_e4_gradients(controller, trajectories)
    assert set(metrics) == {
        "E4_executed_action_marginal",
        "E4_extended_candidate_set",
        "E4_selected_action_selector_proxy",
    }
    assert len({row["return_mean"] for row in metrics.values()}) == 1
    assert all(row["normalized_final_weight_ess"] == 1.0 for row in metrics.values())
    assert not np.array_equal(
        gradients["E4_extended_candidate_set"],
        gradients["E4_selected_action_selector_proxy"],
    )


def test_one_extended_sample_exactly_reuses_e0_to_e3_trajectory_stream() -> None:
    _, controller = make_controller(seed=85)
    composite = sample_composite_trajectories(
        controller, samples=512, seed=903
    )
    projected = as_raw_actor_trajectories(controller, composite)
    for field in ("states", "actions", "rewards"):
        np.testing.assert_array_equal(
            getattr(projected, field), getattr(composite, field)
        )
    assert projected.conditional_ratios.shape == projected.states.shape
    assert projected.marginal_ratios.shape == projected.states.shape


def test_common_random_candidate_columns_are_nested_across_k() -> None:
    mdp, _ = make_controller(seed=87)
    _, q_values = mdp.values()
    scores = standardize_q(q_values)
    controller2 = build_best_of_n_controller(
        mdp, scores, candidate_count=2, budget=2.0
    )
    # K=4 always expansion is only a CRN diagnostic here, not a primary B<=2 row.
    controller4 = build_best_of_n_controller(
        mdp, scores, candidate_count=4, budget=4.0
    )
    sampled2 = sample_composite_trajectories(controller2, samples=128, seed=906)
    sampled4 = sample_composite_trajectories(controller4, samples=128, seed=906)
    np.testing.assert_array_equal(
        sampled2.candidates[:, 0], sampled4.candidates[:, 0, :2]
    )


def test_composite_trajectory_concatenation_preserves_every_logged_field() -> None:
    _, controller = make_controller(seed=86)
    left = sample_composite_trajectories(controller, samples=64, seed=904)
    right = sample_composite_trajectories(controller, samples=96, seed=905)
    joined = concatenate_composite_trajectories([left, right])
    assert joined.sample_count == 160
    for field in (
        "states",
        "actions",
        "rewards",
        "expanded",
        "candidates",
        "selected_indices",
    ):
        np.testing.assert_array_equal(
            getattr(joined, field)[:64], getattr(left, field)
        )
        np.testing.assert_array_equal(
            getattr(joined, field)[64:], getattr(right, field)
        )
