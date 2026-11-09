#!/usr/bin/env python3
"""Run the complete frozen S2 frontier on committed confirmation MDPs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_mismatch_estimators import (
    complete_partial_frontier_with_gradients,
    concatenate_trajectories,
    post_update_metrics,
    sample_search_trajectories,
)
from acbpo.search_policy_mismatch import SearchMDPPolicy


def standardize_within_state(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def run_mdp(
    mdp_seed: int, stream_seeds: list[int], samples_per_stream: int
) -> dict[str, object]:
    mdp = LayeredMDP.random(
        seed=mdp_seed, horizon=6, state_count=7, action_count=6, gamma=0.97
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
    trajectory_batches = [
        sample_search_trajectories(
            controller, samples=samples_per_stream, seed=stream_seed
        )
        for stream_seed in stream_seeds
    ]
    trajectories = concatenate_trajectories(trajectory_batches)
    metrics, gradients = complete_partial_frontier_with_gradients(
        controller, trajectories
    )
    update_cache: dict[bytes, dict[str, float]] = {}
    for name, gradient in gradients.items():
        key = np.round(gradient, decimals=15).tobytes()
        if key not in update_cache:
            update_cache[key] = post_update_metrics(controller, gradient)
        metrics[name].update(update_cache[key])
    return {
        "confirmation_mdp_seed": mdp_seed,
        "monte_carlo_stream_seeds": stream_seeds,
        "samples_per_stream": samples_per_stream,
        "total_trajectories": trajectories.sample_count,
        "frontier": metrics,
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    names = rows[0]["frontier"].keys()
    output: dict[str, dict[str, object]] = {}
    for name in names:
        output[name] = {}
        for metric, example in rows[0]["frontier"][name].items():
            if isinstance(example, (int, float)):
                values = np.asarray(
                    [row["frontier"][name][metric] for row in rows],
                    dtype=np.float64,
                )
                output[name][f"mean_{metric}"] = float(values.mean())
                output[name][f"std_across_mdp_{metric}"] = float(
                    values.std(ddof=1)
                )
    return output


def clustered_bootstrap_difference(
    rows: list[dict[str, object]],
    *,
    left: str,
    right: str,
    metric: str,
    seed: int,
    higher_is_better: bool = False,
    resamples: int = 10_000,
) -> dict[str, object]:
    differences = np.asarray(
        [
            row["frontier"][left][metric] - row["frontier"][right][metric]
            for row in rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    bootstrap = differences[indices].mean(axis=1)
    favorable = differences > 0.0 if higher_is_better else differences < 0.0
    unfavorable = differences < 0.0 if higher_is_better else differences > 0.0
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "mean_difference": float(differences.mean()),
        "clustered_bootstrap_95_ci": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "higher_is_better": higher_is_better,
        "wins": int(np.sum(favorable)),
        "losses": int(np.sum(unfavorable)),
        "ties": int(np.sum(differences == 0.0)),
        "cluster_count": int(differences.size),
        "resamples": resamples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-stream", type=int, default=4096)
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
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("results/search_policy_mismatch/S2_DEVELOPMENT.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples_per_stream < 1024:
        parser.error("--samples-per-stream must be at least 1024")
    protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    development = json.loads(args.development.read_text())
    if manifest["protocol_sha256"] != protocol_digest:
        raise ValueError("manifest/protocol digest mismatch")
    if development["protocol_sha256"] != protocol_digest:
        raise ValueError("development selection/protocol digest mismatch")

    rows = []
    for mdp_seed in manifest["confirmation_mdp_seeds"]:
        rows.append(
            run_mdp(
                mdp_seed,
                manifest["monte_carlo_streams_by_mdp"][str(mdp_seed)],
                args.samples_per_stream,
            )
        )
    summary = aggregate(rows)
    contrasts = {}
    for batch_size, selection in development["selections_by_batch_size"].items():
        selected = selection["selected_estimator"]
        metric = f"gradient_relative_mse_batch_{batch_size}"
        for endpoint in (
            "marginal_step_cumulative_lambda_0",
            "marginal_step_cumulative_lambda_1",
        ):
            name = f"batch_{batch_size}_selected_vs_{endpoint.rsplit('_', 1)[-1]}"
            contrasts[name] = clustered_bootstrap_difference(
                rows,
                left=selected,
                right=endpoint,
                metric=metric,
                seed=2026102800 + int(batch_size) + len(contrasts),
            )
            return_name = (
                f"batch_{batch_size}_selected_vs_"
                f"{endpoint.rsplit('_', 1)[-1]}_raw_return_change"
            )
            contrasts[return_name] = clustered_bootstrap_difference(
                rows,
                left=selected,
                right=endpoint,
                metric="raw_actor_return_change",
                seed=2026102900 + int(batch_size) + len(contrasts),
                higher_is_better=True,
            )

    contrasts["standard_E0_vs_exact_E2_raw_return_change"] = (
        clustered_bootstrap_difference(
            rows,
            left="marginal_step_cumulative_lambda_0",
            right="marginal_step_cumulative_lambda_1",
            metric="raw_actor_return_change",
            seed=2026103900,
            higher_is_better=True,
        )
    )

    expected_frontier_size = 2 * 3 * 5
    finite = all(
        len(row["frontier"]) == expected_frontier_size
        and all(
            np.isfinite(value)
            for metrics in row["frontier"].values()
            for value in metrics.values()
            if isinstance(value, (int, float))
        )
        for row in rows
    )
    s2_passed = bool(len(rows) == 30 and finite)
    payload = {
        "schema": "cgcs-exact-truth-s2-confirmation-v1",
        "protocol_sha256": protocol_digest,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "development_sha256": hashlib.sha256(args.development.read_bytes()).hexdigest(),
        "samples_per_stream": args.samples_per_stream,
        "streams_per_mdp": 10,
        "trajectories_per_mdp": args.samples_per_stream * 10,
        "development_selections_by_batch_size": development[
            "selections_by_batch_size"
        ],
        "complete_frontier_size": expected_frontier_size,
        "S2_completeness_passed": s2_passed,
        "contrasts": contrasts,
        "aggregate": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "S2_completeness_passed": s2_passed,
                "mdp_count": len(rows),
                "output": str(args.output),
            }
        )
    )
    if not s2_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
