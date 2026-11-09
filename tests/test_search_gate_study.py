from __future__ import annotations

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_gate_study import (
    calibrate_gate,
    exact_expansion_gain,
    fit_ridge_gate,
    fixed_feature_sufficiency_groups,
    full_expansion_policy,
    gate_features,
    realized_expansion_rate,
    weighted_auc,
    weighted_oracle_labels,
)


def make_mdp(seed: int = 1011) -> LayeredMDP:
    return LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )


def test_gate_calibration_matches_exact_induced_occupancy() -> None:
    mdp = make_mdp()
    _, q_values = mdp.values()
    expanded = full_expansion_policy(
        mdp, q_values, candidate_count=4, beta=1.0
    )
    _, scores = gate_features(mdp)
    gate = calibrate_gate(
        mdp, expanded, scores["entropy"], target_expansion_rate=1.0 / 3.0
    )
    rate, _, _ = realized_expansion_rate(mdp, expanded, gate)
    np.testing.assert_allclose(rate, 1.0 / 3.0, atol=1e-10)


def test_uniform_selector_recovers_actor_under_every_gate() -> None:
    mdp = make_mdp(seed=1012)
    _, q_values = mdp.values()
    expanded = full_expansion_policy(
        mdp, q_values, candidate_count=4, beta=0.0
    )
    np.testing.assert_allclose(expanded, mdp.actor, atol=1e-12)


def test_oracle_labels_match_budget_and_perfect_score_has_unit_auc() -> None:
    mdp = make_mdp(seed=1013)
    _, q_values = mdp.values()
    expanded = full_expansion_policy(
        mdp, q_values, candidate_count=2, beta=1.0
    )
    gain = exact_expansion_gain(mdp, expanded, q_values)
    occupancy = np.zeros((mdp.horizon, mdp.state_count))
    occupancy[0, 0] = 1.0
    for t in range(mdp.horizon - 1):
        occupancy[t + 1] = np.einsum(
            "s,sa,san->n", occupancy[t], mdp.actor[t], mdp.transitions[t]
        )
    labels = weighted_oracle_labels(gain, occupancy, 0.5)
    np.testing.assert_allclose(np.sum(occupancy * labels) / mdp.horizon, 0.5)
    assert weighted_auc(gain, labels, occupancy) > 0.99


def test_ridge_gate_recovers_fixed_linear_signal() -> None:
    rng = np.random.default_rng(55)
    features = rng.normal(size=(500, 4))
    features[:, 0] = 1.0
    coefficient = np.asarray([0.2, -0.7, 1.1, 0.5])
    target = features @ coefficient
    fitted = fit_ridge_gate(features, target, penalty=1e-8)
    np.testing.assert_allclose(fitted, coefficient, rtol=1e-7, atol=1e-7)


def test_fixed_feature_groups_have_declared_shapes_and_finite_values() -> None:
    mdp = make_mdp(seed=1014)
    _, q_values = mdp.values()
    groups = fixed_feature_sufficiency_groups(mdp, q_values)
    assert set(groups) == {
        "value_entropy_degree_3",
        "actor_confidence",
        "critic_shape",
        "union_with_interactions",
    }
    assert groups["value_entropy_degree_3"].shape[-1] == 10
    assert groups["actor_confidence"].shape[-1] == 7
    assert groups["critic_shape"].shape[-1] == 7
    assert groups["union_with_interactions"].shape[-1] == 12
    assert all(np.all(np.isfinite(values)) for values in groups.values())
