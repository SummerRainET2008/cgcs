from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

from acbpo.critic_quality import build_critic_conditions
from acbpo.layered_mdp import LayeredMDP
from acbpo.search_family_matrix import expected_candidate_budget


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_exact_truth_estimator_matrix",
    ROOT / "scripts/run_exact_truth_estimator_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


def make_inputs(seed: int = 91):
    mdp = LayeredMDP.random(
        seed=seed, horizon=4, state_count=4, action_count=4, gamma=0.97
    )
    critic = build_critic_conditions(mdp, mdp_seed=seed)[0]
    return mdp, critic


def test_primary_matrix_has_three_protocol_conformant_search_rows() -> None:
    mdp, critic = make_inputs()
    controllers = MATRIX.primary_controllers(mdp, critic.selector_scores)
    assert tuple(controllers) == MATRIX.PRIMARY_FAMILIES
    assert controllers["best_of_n"].candidate_count == 2
    assert np.all(controllers["best_of_n"].expansion_probabilities == 1.0)
    for name, controller in controllers.items():
        assert abs(expected_candidate_budget(controller) - 2.0) < 1e-10, name


def test_one_matrix_row_contains_named_targets_and_common_data_estimators() -> None:
    mdp, critic = make_inputs(seed=92)
    controller = MATRIX.primary_controllers(mdp, critic.selector_scores)[
        "entropy_value"
    ]
    row = MATRIX.evaluate_one(
        mdp,
        mdp_seed=92,
        critic=critic,
        family="entropy_value",
        controller=controller,
        stream_seeds=[501, 502],
        samples_per_stream=256,
    )
    assert row["total_trajectories"] == 512
    assert abs(row["expected_candidate_budget"] - 2.0) < 1e-10
    assert set(row["estimators"]) == {
        "E0_standard_selected_action",
        "E1_local",
        "E1_trajectory",
        "E2_local_marginal",
        "E2_cumulative_marginal",
        "E3_step_cumulative_lambda_0.75",
        "E3_step_cumulative_lambda_1",
        "E4_executed_action_marginal",
        "E4_extended_candidate_set",
        "E4_selected_action_selector_proxy",
    }
    exact_e4 = row["exact_composite_target_diagnostics"]
    assert exact_e4["E4_executed_action_marginal"]["relative_bias"] < 1e-10
    assert exact_e4["E4_extended_candidate_set"]["relative_bias"] == 0.0
    assert exact_e4["E4_selected_action_selector_proxy"]["relative_bias"] > 1e-3


def test_confirmation_cli_defaults_are_fail_closed(monkeypatch, tmp_path) -> None:
    output = tmp_path / "partial.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_exact_truth_estimator_matrix.py",
            "--split",
            "confirmation",
            "--mdp-limit",
            "1",
            "--output",
            str(output),
        ],
    )
    import pytest

    with pytest.raises(ValueError, match="partial confirmation"):
        MATRIX.main()
