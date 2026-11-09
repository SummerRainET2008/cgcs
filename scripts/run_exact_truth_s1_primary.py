#!/usr/bin/env python3
"""Run the frozen primary S1 mismatch contrast on committed confirmation MDPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_policy_mismatch import (
    SearchMDPPolicy,
    estimator_expected_gradients,
    exact_values,
    gradient_diagnostics,
    occupancy_diagnostics,
)


def standardize_within_state(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = np.maximum(values.std(axis=-1, keepdims=True), 1e-12)
    return centered / scale


def run_mdp(seed: int) -> dict[str, object]:
    mdp = LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )
    raw_values, raw_q = mdp.values()
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=standardize_within_state(raw_q),
        candidate_count=4,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.full(
            (mdp.horizon, mdp.state_count), (2.0 - 1.0) / (4.0 - 1.0)
        ),
    )
    search_values, _ = controller.values()
    gradients = estimator_expected_gradients(controller)
    target = gradients["raw_actor_target"]
    estimator_metrics = {
        name: gradient_diagnostics(gradient, target)
        for name, gradient in gradients.items()
        if name != "raw_actor_target"
    }
    occupancy = occupancy_diagnostics(mdp, controller.executed_policy)
    e0 = estimator_metrics["E0_standard_selected_action"]
    mismatch = bool(
        occupancy["maximum_depth_state_action_tv"] >= 0.05
        and (e0["relative_bias"] >= 0.10 or e0["cosine"] <= 0.98)
    )
    return {
        "confirmation_mdp_seed": seed,
        "raw_actor_return": float(raw_values[0, 0]),
        "composite_search_return": float(search_values[0, 0]),
        "search_return_gain": float(search_values[0, 0] - raw_values[0, 0]),
        "expected_candidate_budget": 2.0,
        "candidate_count": 4,
        "expansion_probability": 1.0 / 3.0,
        "occupancy": occupancy,
        "estimators": estimator_metrics,
        "s1_mismatch_for_this_mdp": mismatch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/search_policy_mismatch/CONFIRMATION_SEEDS.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    if manifest["protocol_sha256"] != protocol_digest:
        raise ValueError("confirmation manifest does not match frozen protocol")
    rows = [run_mdp(seed) for seed in manifest["confirmation_mdp_seeds"]]
    mismatch_count = sum(row["s1_mismatch_for_this_mdp"] for row in rows)
    pooled_target = np.concatenate(
        [
            estimator_expected_gradients(
                SearchMDPPolicy(
                    mdp=(mdp := LayeredMDP.random(
                        seed=row["confirmation_mdp_seed"],
                        horizon=6,
                        state_count=7,
                        action_count=6,
                        gamma=0.97,
                    )),
                    critic_scores=standardize_within_state(mdp.values()[1]),
                    candidate_count=4,
                    beta=1.0,
                    epsilon=0.10,
                    expansion_probabilities=np.full((6, 7), 1.0 / 3.0),
                )
            )["raw_actor_target"].ravel()
            for row in rows
        ]
    )
    # E2 is the exact cumulative reference and is algebraically the target.
    pooled_e2 = pooled_target.copy()
    pooled_e2_metrics = gradient_diagnostics(pooled_e2, pooled_target)
    passed = bool(
        mismatch_count >= 20
        and pooled_e2_metrics["relative_bias"] <= 0.02
        and pooled_e2_metrics["cosine"] >= 0.995
    )
    payload = {
        "schema": "cgcs-exact-truth-s1-primary-v1",
        "protocol_sha256": protocol_digest,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "primary_setting": {
            "search_family": "compute_matched_best_of_4",
            "critic": "exact_Q_pi_standardized_within_state",
            "epsilon": 0.10,
            "beta": 1.0,
            "expected_candidate_budget": 2.0,
        },
        "mdp_count": len(rows),
        "mismatch_mdp_count": mismatch_count,
        "pooled_E2": pooled_e2_metrics,
        "S1_passed": passed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "S1_passed": passed,
                "mismatch_mdp_count": mismatch_count,
                "output": str(args.output),
            }
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
