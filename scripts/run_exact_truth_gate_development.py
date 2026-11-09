#!/usr/bin/env python3
"""Fit the frozen linear Value--entropy gate on development MDPs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_gate_study import (
    exact_expansion_gain,
    fit_ridge_gate,
    full_expansion_policy,
    gate_features,
)


TRAIN_SEEDS = tuple(range(1000, 1020))
TEST_SEEDS = tuple(range(1020, 1030))


def standardize_q(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def rows_for_seed(seed: int, candidate_count: int):
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
    features, _ = gate_features(mdp)
    target = exact_expansion_gain(mdp, expanded, q_values)
    return features.reshape(-1, features.shape[-1]), target.ravel()


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    centered_prediction = prediction - prediction.mean()
    centered_target = target - target.mean()
    denominator = float(
        np.linalg.norm(centered_prediction) * np.linalg.norm(centered_target)
    )
    correlation = (
        float(np.dot(centered_prediction, centered_target) / denominator)
        if denominator > 0.0
        else 0.0
    )
    return {
        "mse": float(np.mean(np.square(prediction - target))),
        "pearson": correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {}
    for candidate_count in (2, 4):
        train = [rows_for_seed(seed, candidate_count) for seed in TRAIN_SEEDS]
        train_x = np.concatenate([row[0] for row in train], axis=0)
        train_y = np.concatenate([row[1] for row in train], axis=0)
        coefficient = fit_ridge_gate(train_x, train_y, penalty=1e-3)
        test = [rows_for_seed(seed, candidate_count) for seed in TEST_SEEDS]
        test_x = np.concatenate([row[0] for row in test], axis=0)
        test_y = np.concatenate([row[1] for row in test], axis=0)
        conditions[str(candidate_count)] = {
            "coefficient": coefficient.tolist(),
            "train": regression_metrics(train_x @ coefficient, train_y),
            "development_test": regression_metrics(test_x @ coefficient, test_y),
            "train_state_count": int(train_y.size),
            "development_test_state_count": int(test_y.size),
        }
    payload = {
        "schema": "cgcs-exact-truth-gate-development-v1",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "confirmation_mdp_instantiated": False,
        "ridge_penalty": 1e-3,
        "features": ["intercept", "low_value", "entropy", "interaction"],
        "train_seeds": list(TRAIN_SEEDS),
        "development_test_seeds": list(TEST_SEEDS),
        "conditions_by_candidate_count": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"conditions_by_candidate_count": conditions, "output": str(args.output)}))


if __name__ == "__main__":
    main()
