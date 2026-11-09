from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_exact_truth_estimator_matrix",
    ROOT / "scripts/analyze_exact_truth_estimator_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_clustered_bootstrap_is_deterministic_and_counts_signs() -> None:
    differences = np.asarray([-2.0, -1.0, 0.0, 1.0])
    first = ANALYSIS.bootstrap_difference(differences, seed=17, resamples=1000)
    second = ANALYSIS.bootstrap_difference(differences, seed=17, resamples=1000)
    assert first == second
    assert first["cluster_count"] == 4
    assert (first["wins"], first["losses"], first["ties"]) == (1, 2, 1)


def test_contrast_names_keep_targets_and_metrics_explicit() -> None:
    assert ANALYSIS.CONTRAST_SPECS["E3_lambda075_vs_E0_mse256"] == (
        "E3_step_cumulative_lambda_0.75",
        "E0_standard_selected_action",
        "gradient_relative_mse_batch_256",
    )
    assert ANALYSIS.CONTRAST_SPECS["E4_proxy_vs_marginal_search_change"] == (
        "E4_selected_action_selector_proxy",
        "E4_executed_action_marginal",
        "composite_search_return_change",
    )
