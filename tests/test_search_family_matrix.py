from __future__ import annotations

import numpy as np
import pytest

from acbpo.critic_quality import standardize_q
from acbpo.layered_mdp import LayeredMDP
from acbpo.search_family_matrix import (
    build_adaptive_controller,
    build_best_of_n_controller,
    build_k1_controller,
    canonical_family_name,
    expected_candidate_budget,
)


def make_inputs(seed: int = 71):
    mdp = LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )
    _, q_values = mdp.values()
    return mdp, standardize_q(q_values)


def test_k1_is_exact_raw_actor_control() -> None:
    mdp, critic = make_inputs()
    controller = build_k1_controller(mdp, critic)
    np.testing.assert_allclose(controller.executed_policy, mdp.actor, atol=1e-12)
    assert expected_candidate_budget(controller) == 1.0


def test_best_of_n_means_always_expand_and_rejects_random_gate_budget() -> None:
    mdp, critic = make_inputs(seed=72)
    controller = build_best_of_n_controller(
        mdp, critic, candidate_count=2, budget=2.0
    )
    np.testing.assert_array_equal(
        controller.expansion_probabilities,
        np.ones((mdp.horizon, mdp.state_count)),
    )
    assert expected_candidate_budget(controller) == 2.0
    with pytest.raises(ValueError, match="always expands"):
        build_best_of_n_controller(
            mdp, critic, candidate_count=4, budget=2.0
        )


@pytest.mark.parametrize("candidate_count,budget", [(2, 1.5), (2, 2.0), (4, 1.5), (4, 2.0)])
@pytest.mark.parametrize("gate_name", ["random", "entropy", "value", "value_and_entropy"])
@pytest.mark.parametrize("selector_name", ["uniform", "value"])
def test_adaptive_families_hit_exact_occupancy_weighted_budget(
    candidate_count: int,
    budget: float,
    gate_name: str,
    selector_name: str,
) -> None:
    mdp, critic = make_inputs(seed=73)
    controller = build_adaptive_controller(
        mdp,
        critic,
        candidate_count=candidate_count,
        budget=budget,
        gate_name=gate_name,
        selector_name=selector_name,
    )
    assert abs(expected_candidate_budget(controller) - budget) < 1e-10


def test_uniform_selector_is_gate_only_isolation_control() -> None:
    mdp, critic = make_inputs(seed=74)
    controller = build_adaptive_controller(
        mdp,
        critic,
        candidate_count=4,
        budget=2.0,
        gate_name="entropy",
        selector_name="uniform",
    )
    np.testing.assert_allclose(controller.executed_policy, mdp.actor, atol=1e-12)


def test_canonical_names_reserve_best_of_n_for_always_expanded_factory() -> None:
    assert canonical_family_name("entropy", "value") == "entropy_value"
    assert canonical_family_name("learned_gate", "uniform") == "learned_gate_uniform"
    assert canonical_family_name("oracle_expansion_gain", "value") == "oracle_expansion_gain"
    with pytest.raises(ValueError):
        canonical_family_name("oracle_expansion_gain", "uniform")
