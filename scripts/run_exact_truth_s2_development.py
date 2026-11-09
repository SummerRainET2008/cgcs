#!/usr/bin/env python3
"""Select frozen S2 partial-correction strengths on development MDPs only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_mismatch_estimators import (
    BATCH_SIZES,
    LAMBDAS,
    complete_partial_frontier,
    sample_search_trajectories,
)
from acbpo.search_policy_mismatch import SearchMDPPolicy


DEVELOPMENT_FIT_SEEDS = tuple(range(1000, 1020))
STREAM_DOMAIN = b"cgcs-exact-truth-s2-development-v1"


def stream_seed(mdp_seed: int) -> int:
    digest = hashlib.sha256(
        STREAM_DOMAIN + b":" + str(mdp_seed).encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 2_000_000_000


def standardize_within_state(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def run_mdp(seed: int, samples: int) -> dict[str, object]:
    mdp = LayeredMDP.random(
        seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
    )
    _, q_values = mdp.values()
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=standardize_within_state(q_values),
        candidate_count=4,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.full((6, 7), 1.0 / 3.0),
    )
    seed_for_stream = stream_seed(seed)
    trajectories = sample_search_trajectories(
        controller, samples=samples, seed=seed_for_stream
    )
    return {
        "development_mdp_seed": seed,
        "monte_carlo_stream_seed": seed_for_stream,
        "samples": samples,
        "frontier": complete_partial_frontier(controller, trajectories),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    names = rows[0]["frontier"].keys()
    metrics = rows[0]["frontier"][next(iter(names))].keys()
    output: dict[str, dict[str, float]] = {}
    for name in names:
        output[name] = {}
        for metric in metrics:
            value = rows[0]["frontier"][name][metric]
            if isinstance(value, (int, float)):
                values = np.asarray(
                    [row["frontier"][name][metric] for row in rows],
                    dtype=np.float64,
                )
                output[name][f"mean_{metric}"] = float(values.mean())
                output[name][f"std_across_mdp_{metric}"] = float(
                    values.std(ddof=1)
                )
    return output


def select_lambdas(
    summary: dict[str, dict[str, float]]
) -> dict[str, dict[str, object]]:
    selections = {}
    for batch_size in BATCH_SIZES:
        metric = f"mean_gradient_relative_mse_batch_{batch_size}"
        candidates = []
        for strength in LAMBDAS:
            name = f"marginal_step_cumulative_lambda_{strength:g}"
            candidates.append((summary[name][metric], strength, name))
        minimum = min(row[0] for row in candidates)
        tied = [row for row in candidates if np.isclose(row[0], minimum, atol=1e-12)]
        chosen = max(tied, key=lambda row: row[1])
        selections[str(batch_size)] = {
            "selection_metric": metric,
            "selected_lambda": chosen[1],
            "selected_estimator": chosen[2],
            "selected_mean_relative_gradient_mse": chosen[0],
            "all_lambdas": {
                str(strength): value for value, strength, _ in candidates
            },
            "tie_rule": "larger_lambda",
        }
    return selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-mdp", type=int, default=16_384)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_mdp < 1024:
        parser.error("--samples-per-mdp must be at least 1024")
    rows = [run_mdp(seed, args.samples_per_mdp) for seed in DEVELOPMENT_FIT_SEEDS]
    summary = aggregate(rows)
    selections = select_lambdas(summary)
    payload = {
        "schema": "cgcs-exact-truth-s2-development-v1",
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "confirmation_mdp_instantiated": False,
        "development_mdp_seeds": list(DEVELOPMENT_FIT_SEEDS),
        "stream_derivation_domain": STREAM_DOMAIN.decode("ascii"),
        "samples_per_mdp": args.samples_per_mdp,
        "primary_selection_frontier": "exact_marginal_step_cumulative",
        "selections_by_batch_size": selections,
        "aggregate": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"selections_by_batch_size": selections, "output": str(args.output)}))


if __name__ == "__main__":
    main()
