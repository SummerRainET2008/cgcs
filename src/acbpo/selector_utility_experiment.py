from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def proposal_distribution(
    utility: Sequence[float], *, beta: float, epsilon: float, scale_floor: float
) -> np.ndarray:
    values = np.asarray(utility, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("utility must be a non-empty vector")
    if beta < 0.0 or scale_floor <= 0.0 or not 0.0 <= epsilon < 1.0:
        raise ValueError("invalid proposal hyperparameters")
    normalized = (values - values.mean()) / max(float(values.std()), scale_floor)
    logits = beta * normalized
    logits -= logits.max()
    shaped = np.exp(logits)
    shaped /= shaped.sum()
    return (1.0 - epsilon) * shaped + epsilon / len(values)


def entropy(probabilities: Sequence[float]) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-30)), axis=-1
    )


def load_states(data_path: Path, prediction_path: Path) -> list[dict[str, Any]]:
    data = {
        row["example_id"]: row
        for row in (json.loads(line) for line in data_path.open())
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for line in prediction_path.open():
        prediction = json.loads(line)
        example_id = prediction["example_id"]
        if example_id not in grouped:
            order.append(example_id)
        grouped[example_id].append(prediction)
    states = []
    for example_id in order:
        source = data[example_id]
        predictions = sorted(grouped[example_id], key=lambda row: row["candidate_index"])
        if len(predictions) != len(source["candidates"]):
            raise ValueError(f"candidate count mismatch for {example_id}")
        states.append(
            {
                "example_id": example_id,
                "problem_id": source["problem_id"],
                "ratings": np.asarray([row["rating"] for row in predictions], dtype=np.float64),
                "scores": np.asarray([row["score"] for row in predictions], dtype=np.float64),
                "probabilities": np.asarray(
                    [row["probabilities"] for row in predictions], dtype=np.float64
                ),
            }
        )
    return states


def proposal_for_state(
    state: dict[str, Any],
    method: str,
    *,
    beta: float,
    epsilon: float,
    uncertainty_weight: float,
    scale_floor: float,
) -> np.ndarray:
    scores = state["scores"]
    if method == "uniform":
        return np.full(len(scores), 1.0 / len(scores))
    if method == "high_q":
        utility = scores
    elif method == "abs_advantage":
        utility = np.abs(scores - scores.mean())
    elif method == "uncertainty":
        utility = entropy(state["probabilities"])
    elif method == "abs_advantage_uncertainty":
        absolute = np.abs(scores - scores.mean())
        uncertainty = entropy(state["probabilities"])
        absolute = (absolute - absolute.mean()) / max(float(absolute.std()), scale_floor)
        uncertainty = (uncertainty - uncertainty.mean()) / max(
            float(uncertainty.std()), scale_floor
        )
        utility = absolute + uncertainty_weight * uncertainty
    elif method == "oracle_abs_advantage":
        ratings = state["ratings"]
        utility = np.abs(ratings - ratings.mean())
    elif method == "oracle_abs_advantage_direct":
        ratings = state["ratings"]
        utility = np.abs(ratings - ratings.mean())
        if utility.sum() <= 0.0:
            return np.full(len(utility), 1.0 / len(utility))
        optimal = utility / utility.sum()
        return (1.0 - epsilon) * optimal + epsilon / len(utility)
    else:
        raise ValueError(f"unknown method {method}")
    return proposal_distribution(
        utility, beta=beta, epsilon=epsilon, scale_floor=scale_floor
    )


def evaluate(
    states: Sequence[dict[str, Any]],
    method: str,
    *,
    beta: float,
    epsilon: float,
    uncertainty_weight: float = 0.0,
    scale_floor: float = 0.05,
) -> dict[str, float]:
    target_second_moment = 0.0
    proposal_second_moment = 0.0
    expected_rating_gain = []
    ess_fractions = []
    informative_mass_gain = []
    informative_states = 0
    for state in states:
        ratings = state["ratings"]
        count = len(ratings)
        uniform = np.full(count, 1.0 / count)
        advantage = ratings - ratings.mean()
        proposal = proposal_for_state(
            state,
            method,
            beta=beta,
            epsilon=epsilon,
            uncertainty_weight=uncertainty_weight,
            scale_floor=scale_floor,
        )
        target_second_moment += float(np.sum(uniform * np.square(advantage)))
        proposal_second_moment += float(
            np.sum(np.square(uniform) * np.square(advantage) / proposal)
        )
        expected_rating_gain.append(float(proposal @ ratings - uniform @ ratings))
        ess_fractions.append(float(1.0 / np.sum(np.square(uniform) / proposal)))
        informative = np.abs(advantage) > 1e-12
        if informative.any():
            informative_states += 1
            informative_mass_gain.append(
                float(proposal[informative].sum() - uniform[informative].sum())
            )
    return {
        "states": float(len(states)),
        "informative_states": float(informative_states),
        "second_moment_ratio": (
            proposal_second_moment / target_second_moment
            if target_second_moment > 0.0
            else math.nan
        ),
        "expected_rating_gain": float(np.mean(expected_rating_gain)),
        "mean_ess_fraction": float(np.mean(ess_fractions)),
        "mean_informative_mass_gain": (
            float(np.mean(informative_mass_gain)) if informative_mass_gain else 0.0
        ),
    }


def tune(
    states: Sequence[dict[str, Any]], method: str, *, min_ess: float
) -> tuple[dict[str, float], list[dict[str, float]]]:
    grid = []
    uncertainty_weights = (
        (0.25, 0.5, 1.0, 2.0)
        if method == "abs_advantage_uncertainty"
        else (0.0,)
    )
    betas = (
        (0.0,)
        if method == "oracle_abs_advantage_direct"
        else (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    )
    epsilons = (
        (0.05, 0.1, 0.2, 0.3, 0.5)
        if method == "oracle_abs_advantage_direct"
        else (0.05, 0.1, 0.2)
    )
    for beta in betas:
        for epsilon in epsilons:
            for uncertainty_weight in uncertainty_weights:
                metrics = evaluate(
                    states,
                    method,
                    beta=beta,
                    epsilon=epsilon,
                    uncertainty_weight=uncertainty_weight,
                )
                grid.append(
                    {
                        "beta": beta,
                        "epsilon": epsilon,
                        "uncertainty_weight": uncertainty_weight,
                        **metrics,
                    }
                )
    eligible = [row for row in grid if row["mean_ess_fraction"] >= min_ess]
    if not eligible:
        raise RuntimeError(f"no {method} setting satisfies ESS constraint")
    return min(eligible, key=lambda row: row["second_moment_ratio"]), grid


def bootstrap_delta(
    states: Sequence[dict[str, Any]], config: dict[str, float], *, seed: int
) -> list[float]:
    per_cluster: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for state_index, state in enumerate(states):
        ratings = state["ratings"]
        baseline = float(np.mean(np.square(ratings - ratings.mean())))
        if baseline == 0.0:
            pair = (0.0, 0.0)
            cluster = str(state.get("problem_id", state_index))
            per_cluster[cluster].append(pair)
            continue
        selected = evaluate(
            [state],
            str(config["method"]),
            beta=float(config["beta"]),
            epsilon=float(config["epsilon"]),
            uncertainty_weight=float(config["uncertainty_weight"]),
        )
        # Aggregate second moments are additive. Store numerator-like values by
        # multiplying the state ratio by its uniform second moment.
        pair = (baseline, selected["second_moment_ratio"] * baseline)
        cluster = str(state.get("problem_id", state_index))
        per_cluster[cluster].append(pair)
    pairs = np.asarray(
        [np.asarray(values).sum(axis=0) for values in per_cluster.values()]
    )
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(20_000):
        sample = pairs[rng.integers(0, len(pairs), len(pairs))]
        denominator = sample[:, 0].sum()
        ratios.append(sample[:, 1].sum() / denominator if denominator else math.nan)
    return [float(np.nanquantile(ratios, 0.025)), float(np.nanquantile(ratios, 0.975))]


def run(args: argparse.Namespace) -> dict[str, Any]:
    states = load_states(args.data, args.predictions)
    if args.dev_problems is not None:
        problem_order = list(dict.fromkeys(state["problem_id"] for state in states))
        if len(problem_order) <= args.dev_problems:
            raise ValueError("not enough problems for dev/test split")
        dev_problem_ids = set(problem_order[: args.dev_problems])
        dev = [state for state in states if state["problem_id"] in dev_problem_ids]
        test = [state for state in states if state["problem_id"] not in dev_problem_ids]
    else:
        if len(states) <= args.dev_states:
            raise ValueError("not enough states for dev/test split")
        dev = states[: args.dev_states]
        test = states[args.dev_states :]
    methods = ("high_q", "abs_advantage", "uncertainty", "abs_advantage_uncertainty")
    selected = {}
    grids = {}
    test_metrics = {}
    for method in methods:
        best, grid = tune(dev, method, min_ess=args.min_ess)
        config = {
            "method": method,
            "beta": best["beta"],
            "epsilon": best["epsilon"],
            "uncertainty_weight": best["uncertainty_weight"],
        }
        selected[method] = {**config, "dev_metrics": best}
        grids[method] = grid
        test_metrics[method] = {
            **evaluate(
                test,
                method,
                beta=best["beta"],
                epsilon=best["epsilon"],
                uncertainty_weight=best["uncertainty_weight"],
            ),
            "second_moment_ratio_ci95": bootstrap_delta(
                test, config, seed=args.seed + len(test_metrics)
            ),
        }
    test_metrics["uniform"] = evaluate(test, "uniform", beta=0.0, epsilon=0.0)
    oracle_best, oracle_grid = tune(
        dev, "oracle_abs_advantage_direct", min_ess=args.min_ess
    )
    oracle_config = {
        "method": "oracle_abs_advantage_direct",
        "beta": oracle_best["beta"],
        "epsilon": oracle_best["epsilon"],
        "uncertainty_weight": 0.0,
    }
    oracle = {
        **evaluate(
            test,
            "oracle_abs_advantage_direct",
            beta=oracle_best["beta"],
            epsilon=oracle_best["epsilon"],
        ),
        "second_moment_ratio_ci95": bootstrap_delta(
            test, oracle_config, seed=args.seed + 100
        ),
    }
    return {
        "config": {
            "data": str(args.data),
            "predictions": str(args.predictions),
            "dev_states": len(dev),
            "test_states": len(test),
            "dev_problems": len({state["problem_id"] for state in dev}),
            "test_problems": len({state["problem_id"] for state in test}),
            "min_ess": args.min_ess,
            "seed": args.seed,
        },
        "selected_on_dev": selected,
        "test_metrics": test_metrics,
        "oracle_test_reference": oracle,
        "oracle_selected_on_dev": {**oracle_config, "dev_metrics": oracle_best},
        "oracle_dev_grid": oracle_grid,
        "dev_grids": grids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-states", type=int, default=100)
    parser.add_argument("--dev-problems", type=int)
    parser.add_argument("--min-ess", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=192001)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"config": result["config"], "selected_on_dev": result["selected_on_dev"], "test_metrics": result["test_metrics"], "oracle_test_reference": result["oracle_test_reference"]}, indent=2))


if __name__ == "__main__":
    main()
