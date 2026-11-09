from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from acbpo.selector_utility_experiment import bootstrap_delta, evaluate, load_states


def evaluate_locked(
    data: Path,
    predictions: Path,
    *,
    method: str,
    beta: float,
    epsilon: float,
    uncertainty_weight: float = 0.0,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Evaluate a preregistered selector without searching on evaluation data."""
    states = load_states(data, predictions)
    config = {
        "method": method,
        "beta": beta,
        "epsilon": epsilon,
        "uncertainty_weight": uncertainty_weight,
    }
    metrics = evaluate(
        states,
        method,
        beta=beta,
        epsilon=epsilon,
        uncertainty_weight=uncertainty_weight,
    )
    metrics["second_moment_ratio_ci95"] = bootstrap_delta(
        states, config, seed=bootstrap_seed
    )
    return {
        "protocol": "locked_no_test_tuning",
        "config": {
            "data": str(data),
            "predictions": str(predictions),
            "states": len(states),
            "problems": len({state["problem_id"] for state in states}),
            "bootstrap_seed": bootstrap_seed,
            **config,
        },
        "metrics": metrics,
        "uniform_reference": evaluate(
            states, "uniform", beta=0.0, epsilon=0.0
        ),
    }


def evaluate_locked_ensemble(
    data: Path,
    prediction_paths: list[Path],
    *,
    method: str,
    betas: list[float],
    epsilons: list[float],
    labels: list[str],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Aggregate locked selectors over training seeds without test-time search."""
    count = len(prediction_paths)
    if not count or not (len(betas) == len(epsilons) == len(labels) == count):
        raise ValueError("prediction paths, betas, epsilons, and labels must align")

    state_sets = [load_states(data, path) for path in prediction_paths]
    reference = state_sets[0]
    for states in state_sets[1:]:
        if len(states) != len(reference):
            raise ValueError("state count mismatch across seeds")
        for expected, observed in zip(reference, states, strict=True):
            if expected["example_id"] != observed["example_id"]:
                raise ValueError("example order mismatch across seeds")
            if not np.array_equal(expected["ratings"], observed["ratings"]):
                raise ValueError("rating mismatch across seeds")

    per_seed = []
    for index, (label, states, beta, epsilon) in enumerate(
        zip(labels, state_sets, betas, epsilons, strict=True)
    ):
        config = {
            "method": method,
            "beta": beta,
            "epsilon": epsilon,
            "uncertainty_weight": 0.0,
        }
        metrics = evaluate(states, method, beta=beta, epsilon=epsilon)
        metrics["second_moment_ratio_ci95"] = bootstrap_delta(
            states, config, seed=bootstrap_seed + index + 1
        )
        per_seed.append(
            {
                "label": label,
                "predictions": str(prediction_paths[index]),
                "beta": beta,
                "epsilon": epsilon,
                "metrics": metrics,
            }
        )

    per_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for state_index, reference_state in enumerate(reference):
        ratings = reference_state["ratings"]
        baseline = float(np.mean(np.square(ratings - ratings.mean())))
        proposal_moments = []
        for states, beta, epsilon in zip(state_sets, betas, epsilons, strict=True):
            metrics = evaluate([states[state_index]], method, beta=beta, epsilon=epsilon)
            ratio = metrics["second_moment_ratio"]
            proposal_moments.append(0.0 if math.isnan(ratio) else ratio * baseline)
        cluster = str(reference_state.get("problem_id", state_index))
        per_cluster[cluster].append((baseline, float(np.mean(proposal_moments))))

    pairs = np.asarray(
        [np.asarray(values).sum(axis=0) for values in per_cluster.values()],
        dtype=np.float64,
    )
    denominator = float(pairs[:, 0].sum())
    numerator = float(pairs[:, 1].sum())
    ratio = numerator / denominator if denominator else math.nan
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_ratios = []
    for _ in range(20_000):
        sample = pairs[rng.integers(0, len(pairs), len(pairs))]
        sample_denominator = float(sample[:, 0].sum())
        bootstrap_ratios.append(
            float(sample[:, 1].sum()) / sample_denominator
            if sample_denominator
            else math.nan
        )

    return {
        "protocol": "locked_ensemble_no_test_tuning",
        "config": {
            "data": str(data),
            "method": method,
            "states": len(reference),
            "problems": len(per_cluster),
            "bootstrap_seed": bootstrap_seed,
            "labels": labels,
            "betas": betas,
            "epsilons": epsilons,
        },
        "per_seed": per_seed,
        "aggregate": {
            "second_moment_ratio": ratio,
            "second_moment_ratio_ci95": [
                float(np.nanquantile(bootstrap_ratios, 0.025)),
                float(np.nanquantile(bootstrap_ratios, 0.975)),
            ],
            "mean_ess_fraction": float(
                np.mean([row["metrics"]["mean_ess_fraction"] for row in per_seed])
            ),
            "mean_expected_rating_gain": float(
                np.mean([row["metrics"]["expected_rating_gain"] for row in per_seed])
            ),
        },
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a selector with preregistered, fixed hyperparameters"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", default="abs_advantage")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--uncertainty-weight", type=float, default=0.0)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    args = parser.parse_args()
    result = evaluate_locked(
        args.data,
        args.predictions,
        method=args.method,
        beta=args.beta,
        epsilon=args.epsilon,
        uncertainty_weight=args.uncertainty_weight,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
