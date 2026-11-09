#!/usr/bin/env python3
"""Run the frozen development-only gate feature-sufficiency diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_gate_study import (
    calibrate_gate,
    evaluate_gate,
    exact_expansion_gain,
    fit_ridge_gate,
    fixed_feature_sufficiency_groups,
    full_expansion_policy,
)


TRAIN_SEEDS = tuple(range(1000, 1020))
TEST_SEEDS = tuple(range(1020, 1030))


def standardize_q(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def one_seed(seed: int, candidate_count: int):
    mdp = LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )
    _, q_values = mdp.values()
    expanded = full_expansion_policy(
        mdp,
        standardize_q(q_values),
        candidate_count=candidate_count,
        beta=1.0,
        epsilon=0.10,
    )
    target = exact_expansion_gain(mdp, expanded, q_values)
    return mdp, q_values, expanded, target, fixed_feature_sufficiency_groups(mdp, q_values)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {}
    for candidate_count, target_rate in ((2, 0.5), (4, 1.0 / 3.0)):
        train = [one_seed(seed, candidate_count) for seed in TRAIN_SEEDS]
        test = [one_seed(seed, candidate_count) for seed in TEST_SEEDS]
        group_output = {}
        for group_name in train[0][4]:
            train_x = np.concatenate(
                [row[4][group_name].reshape(-1, row[4][group_name].shape[-1]) for row in train]
            )
            train_y = np.concatenate([row[3].ravel() for row in train])
            coefficient = fit_ridge_gate(train_x, train_y, penalty=1e-3)
            test_x = np.concatenate(
                [row[4][group_name].reshape(-1, row[4][group_name].shape[-1]) for row in test]
            )
            test_y = np.concatenate([row[3].ravel() for row in test])
            paired_return = []
            aucs = []
            for mdp, _, expanded, target, groups in test:
                prediction = groups[group_name] @ coefficient
                learned_gate = calibrate_gate(
                    mdp, expanded, prediction, target_expansion_rate=target_rate
                )
                random_gate = calibrate_gate(
                    mdp,
                    expanded,
                    np.zeros_like(target),
                    target_expansion_rate=target_rate,
                )
                learned = evaluate_gate(
                    mdp,
                    expanded,
                    learned_gate,
                    oracle_gain=target,
                    target_expansion_rate=target_rate,
                    candidate_count=candidate_count,
                ).metrics
                random = evaluate_gate(
                    mdp,
                    expanded,
                    random_gate,
                    oracle_gain=target,
                    target_expansion_rate=target_rate,
                    candidate_count=candidate_count,
                ).metrics
                paired_return.append(
                    learned["composite_search_return"]
                    - random["composite_search_return"]
                )
                aucs.append(learned["gate_auc_vs_oracle_allocation"])
            paired_return = np.asarray(paired_return)
            group_output[group_name] = {
                "feature_count": int(coefficient.size),
                "coefficient": coefficient.tolist(),
                "development_test_mse": float(
                    np.mean(np.square(test_x @ coefficient - test_y))
                ),
                "development_test_pearson": pearson(test_x @ coefficient, test_y),
                "mean_gate_auc": float(np.mean(aucs)),
                "paired_return_gain_vs_random_mean": float(paired_return.mean()),
                "paired_return_wins": int(np.sum(paired_return > 0.0)),
                "paired_return_losses": int(np.sum(paired_return < 0.0)),
            }
        output[str(candidate_count)] = group_output
    payload = {
        "schema": "cgcs-exact-truth-gate-feature-sufficiency-v1",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "confirmation_mdp_instantiated": False,
        "train_seeds": list(TRAIN_SEEDS),
        "development_test_seeds": list(TEST_SEEDS),
        "ridge_penalty": 1e-3,
        "conditions_by_candidate_count": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"conditions_by_candidate_count": output, "output": str(args.output)}))


if __name__ == "__main__":
    main()
