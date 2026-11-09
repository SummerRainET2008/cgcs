from __future__ import annotations

import numpy as np

from acbpo.critic_quality import (
    build_critic_conditions,
    deterministic_seed,
    fit_tabular_ridge_q,
    lagged_q_at_kl,
)
from acbpo.layered_mdp import LayeredMDP


def make_mdp(seed: int = 61) -> LayeredMDP:
    return LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )


def test_lagged_snapshot_hits_frozen_kl_targets() -> None:
    mdp = make_mdp()
    for target in (0.01, 0.04):
        q_values, achieved = lagged_q_at_kl(mdp, target_kl=target)
        assert q_values.shape == mdp.rewards.shape
        np.testing.assert_allclose(achieved, target, atol=1e-10)


def test_learned_tabular_q_is_deterministic_and_finite() -> None:
    mdp = make_mdp(seed=62)
    seed = deterministic_seed(62, "learned", 256)
    first, first_meta = fit_tabular_ridge_q(mdp, trajectories=256, seed=seed)
    second, second_meta = fit_tabular_ridge_q(mdp, trajectories=256, seed=seed)
    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    assert np.all(np.isfinite(first))
    assert 0 < first_meta["visited_cells"] <= first_meta["total_cells"]


def test_frozen_critic_axis_has_all_nine_conditions() -> None:
    conditions = build_critic_conditions(make_mdp(seed=63), mdp_seed=63)
    assert [row.name for row in conditions] == [
        "exact",
        "noise_0.5",
        "noise_1",
        "noise_2",
        "inverted_diagnostic",
        "lagged_kl_0.01",
        "lagged_kl_0.04",
        "learned_256",
        "learned_1024",
    ]
    assert conditions[0].ranking_concordance == 1.0
    assert conditions[4].ranking_concordance == 0.0
    assert all(0.0 <= row.ranking_concordance <= 1.0 for row in conditions)
