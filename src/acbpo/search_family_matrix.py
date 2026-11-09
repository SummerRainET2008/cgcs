"""Protocol-conformant construction of exact-truth search families.

The frozen protocol distinguishes an always-expanded best-of-N policy from a
compute-matched random expansion gate.  This module is the canonical factory for
the remaining estimator matrix and deliberately refuses to conflate the two.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .layered_mdp import LayeredMDP
from .search_gate_study import (
    calibrate_gate,
    exact_expansion_gain,
    full_expansion_policy,
    gate_features,
    standardize_per_layer,
)
from .search_policy_mismatch import SearchMDPPolicy


FloatArray = NDArray[np.float64]


def expected_candidate_budget(controller: SearchMDPPolicy) -> float:
    occupancy = controller.occupancy()
    expanded = float(
        np.sum(occupancy * controller.expansion_probabilities)
        / controller.mdp.horizon
    )
    return 1.0 + (controller.candidate_count - 1.0) * expanded


def build_k1_controller(
    mdp: LayeredMDP, critic_scores: FloatArray
) -> SearchMDPPolicy:
    return SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic_scores,
        candidate_count=1,
        beta=0.0,
        epsilon=0.10,
        expansion_probabilities=np.zeros(
            (mdp.horizon, mdp.state_count), dtype=np.float64
        ),
    )


def build_best_of_n_controller(
    mdp: LayeredMDP,
    critic_scores: FloatArray,
    *,
    candidate_count: int,
    budget: float,
) -> SearchMDPPolicy:
    """Construct true best-of-N: every visited state is expanded."""
    if candidate_count not in (2, 4):
        raise ValueError("candidate_count must be 2 or 4")
    if budget != float(candidate_count):
        raise ValueError(
            "best-of-N always expands, so its budget must equal candidate_count"
        )
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic_scores,
        candidate_count=candidate_count,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.ones(
            (mdp.horizon, mdp.state_count), dtype=np.float64
        ),
    )
    if abs(expected_candidate_budget(controller) - budget) > 1e-10:
        raise AssertionError("best-of-N budget mismatch")
    return controller


def build_adaptive_controller(
    mdp: LayeredMDP,
    critic_scores: FloatArray,
    *,
    candidate_count: int,
    budget: float,
    gate_name: str,
    selector_name: str,
    learned_coefficient: FloatArray | None = None,
) -> SearchMDPPolicy:
    """Construct an occupancy-calibrated adaptive search family."""
    if candidate_count not in (2, 4):
        raise ValueError("candidate_count must be 2 or 4")
    if budget not in (1.5, 2.0):
        raise ValueError("budget must be one of the two frozen primary budgets")
    if budget > candidate_count:
        raise ValueError("budget cannot exceed candidate_count")
    if selector_name not in ("uniform", "value"):
        raise ValueError("selector_name must be uniform or value")
    beta = 0.0 if selector_name == "uniform" else 1.0
    expanded_policy = full_expansion_policy(
        mdp,
        critic_scores,
        candidate_count=candidate_count,
        beta=beta,
        epsilon=0.10,
    )
    features, score_map = gate_features(mdp)
    if gate_name == "learned_gate":
        if learned_coefficient is None:
            raise ValueError("learned_gate requires its frozen coefficient")
        score = features @ np.asarray(learned_coefficient, dtype=np.float64)
    elif gate_name == "oracle_expansion_gain":
        _, true_q = mdp.values()
        score = standardize_per_layer(
            exact_expansion_gain(mdp, expanded_policy, true_q)
        )
    elif gate_name in ("random", "entropy", "value", "value_and_entropy"):
        score = score_map[gate_name]
    else:
        raise ValueError("unknown gate_name")

    target_rate = (budget - 1.0) / (candidate_count - 1.0)
    gate = calibrate_gate(
        mdp,
        expanded_policy,
        score,
        target_expansion_rate=target_rate,
        temperature=1.0,
    )
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=critic_scores,
        candidate_count=candidate_count,
        beta=beta,
        epsilon=0.10,
        expansion_probabilities=gate,
    )
    if abs(expected_candidate_budget(controller) - budget) > 1e-10:
        raise AssertionError("adaptive search budget mismatch")
    expected_policy = (
        (1.0 - gate[:, :, None]) * mdp.actor
        + gate[:, :, None] * expanded_policy
    )
    np.testing.assert_allclose(
        controller.executed_policy, expected_policy, atol=1e-12, rtol=0.0
    )
    return controller


def canonical_family_name(gate_name: str, selector_name: str) -> str:
    if gate_name == "learned_gate":
        return f"learned_gate_{selector_name}"
    if gate_name == "oracle_expansion_gain":
        if selector_name != "value":
            raise ValueError("the frozen oracle family uses the Value selector")
        return "oracle_expansion_gain"
    if gate_name in ("entropy", "value", "value_and_entropy"):
        return f"{gate_name}_{selector_name}"
    if gate_name == "random":
        return f"random_{selector_name}"
    raise ValueError("unknown gate_name")
