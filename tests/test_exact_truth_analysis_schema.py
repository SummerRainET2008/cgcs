from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"
SCHEMA = ROOT / "results/search_policy_mismatch/ANALYSIS_SCHEMA.json"


def test_analysis_schema_is_bound_to_frozen_protocol_and_fail_closed() -> None:
    schema = json.loads(SCHEMA.read_text())
    digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert schema["protocol_sha256"] == digest
    assert schema["cluster_unit"] == "confirmation_mdp_seed"
    assert schema["repeated_measurement_unit"] == "monte_carlo_stream_seed"
    assert schema["inference"]["paired_analysis_required"] is True
    assert schema["inference"]["monte_carlo_streams_are_independent_clusters"] is False
    assert set(schema["fail_closed"].values()) == {"error"}


def test_analysis_schema_names_both_targets_and_all_estimator_families() -> None:
    schema = json.loads(SCHEMA.read_text())
    metrics = set(schema["required_per_condition_metrics"])
    assert {"raw_actor_return", "composite_search_return"} <= metrics
    estimators = schema["condition_axes"]["estimator"]
    for family in ("E0_", "E1_", "E2_", "E3_", "E4_"):
        assert any(name.startswith(family) for name in estimators)
    assert "E4_extended_candidate_set" in estimators
    assert schema["condition_axes"]["partial_ratio_basis"] == [
        "conditional_selector",
        "exact_executed_action_marginal",
    ]
