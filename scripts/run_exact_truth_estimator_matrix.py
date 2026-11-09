#!/usr/bin/env python3
"""Run the protocol-conformant E0--E4 estimator/search-family matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.critic_quality import build_critic_conditions, deterministic_seed
from acbpo.layered_mdp import LayeredMDP
from acbpo.search_composite_estimators import (
    as_raw_actor_trajectories,
    concatenate_composite_trajectories,
    estimate_e4_gradients,
    executed_marginal_score_table,
    fast_exact_e4_gradients,
    sample_composite_trajectories,
)
from acbpo.search_family_matrix import (
    build_adaptive_controller,
    build_best_of_n_controller,
    expected_candidate_budget,
)
from acbpo.search_mismatch_estimators import selected_raw_actor_estimators
from acbpo.search_policy_mismatch import (
    estimator_expected_gradients,
    exact_values,
    gradient_diagnostics,
    occupancy_diagnostics,
    raw_actor_gradient,
)


PRIMARY_FAMILIES = ("best_of_n", "entropy_value", "value_value")


def primary_controllers(mdp: LayeredMDP, critic_scores: np.ndarray) -> dict:
    return {
        "best_of_n": build_best_of_n_controller(
            mdp, critic_scores, candidate_count=2, budget=2.0
        ),
        "entropy_value": build_adaptive_controller(
            mdp,
            critic_scores,
            candidate_count=4,
            budget=2.0,
            gate_name="entropy",
            selector_name="value",
        ),
        "value_value": build_adaptive_controller(
            mdp,
            critic_scores,
            candidate_count=4,
            budget=2.0,
            gate_name="value",
            selector_name="value",
        ),
    }


def exact_raw_diagnostics(controller) -> dict[str, dict[str, float]]:
    target = raw_actor_gradient(controller.mdp)
    gradients = estimator_expected_gradients(controller)
    gradients["E2_local_marginal"] = gradients["E1_local"]
    return {
        name: gradient_diagnostics(gradient, target)
        for name, gradient in gradients.items()
        if name != "raw_actor_target"
    }


def exact_composite_diagnostics(
    controller, *, marginal_scores
) -> dict[str, dict[str, float]]:
    gradients = fast_exact_e4_gradients(
        controller, marginal_scores=marginal_scores
    )
    target = gradients["E4_extended_candidate_set"]
    return {
        name: gradient_diagnostics(gradient, target)
        for name, gradient in gradients.items()
    }


def evaluate_one(
    mdp: LayeredMDP,
    *,
    mdp_seed: int,
    critic,
    family: str,
    controller,
    stream_seeds: list[int],
    samples_per_stream: int,
) -> dict[str, object]:
    sampled = [
        sample_composite_trajectories(
            controller, samples=samples_per_stream, seed=stream_seed
        )
        for stream_seed in stream_seeds
    ]
    composite = concatenate_composite_trajectories(sampled)
    raw_projection = as_raw_actor_trajectories(controller, composite)
    raw_metrics, _ = selected_raw_actor_estimators(controller, raw_projection)
    marginal_scores = executed_marginal_score_table(controller)
    composite_metrics, _ = estimate_e4_gradients(
        controller, composite, marginal_scores=marginal_scores
    )
    raw_return = float(exact_values(mdp, mdp.actor)[0][0, 0])
    search_return = float(controller.values()[0][0, 0])
    return {
        "confirmation_mdp_seed": mdp_seed,
        "critic_condition": critic.name,
        "critic_ranking_concordance": critic.ranking_concordance,
        "critic_metadata": critic.metadata,
        "search_family": family,
        "candidate_count": controller.candidate_count,
        "expected_candidate_budget": expected_candidate_budget(controller),
        "monte_carlo_stream_seeds": stream_seeds,
        "samples_per_stream": samples_per_stream,
        "total_trajectories": composite.sample_count,
        "raw_actor_return": raw_return,
        "composite_search_return": search_return,
        "search_gain_vs_raw": search_return - raw_return,
        "occupancy": occupancy_diagnostics(mdp, controller.executed_policy),
        "exact_raw_actor_target_diagnostics": exact_raw_diagnostics(controller),
        "exact_composite_target_diagnostics": exact_composite_diagnostics(
            controller, marginal_scores=marginal_scores
        ),
        "estimators": {**raw_metrics, **composite_metrics},
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
    parser.add_argument(
        "--split", choices=("development", "confirmation"), default="development"
    )
    parser.add_argument("--mdp-limit", type=int)
    parser.add_argument("--stream-limit", type=int)
    parser.add_argument("--samples-per-stream", type=int)
    parser.add_argument("--critic", action="append")
    parser.add_argument("--family", action="append", choices=PRIMARY_FAMILIES)
    parser.add_argument("--allow-partial-confirmation", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    if manifest["protocol_sha256"] != protocol_digest:
        raise ValueError("manifest/protocol digest mismatch")
    if args.stream_limit is None:
        args.stream_limit = 10 if args.split == "confirmation" else 2
    if args.samples_per_stream is None:
        args.samples_per_stream = 4096 if args.split == "confirmation" else 512
    if args.stream_limit <= 0 or args.samples_per_stream < 2:
        raise ValueError("stream_limit and samples_per_stream must be positive")
    partial_confirmation = args.split == "confirmation" and (
        args.mdp_limit is not None
        or args.stream_limit != 10
        or args.samples_per_stream != 4096
        or args.critic is not None
        or args.family is not None
    )
    if partial_confirmation and not args.allow_partial_confirmation:
        raise ValueError(
            "partial confirmation requires --allow-partial-confirmation; "
            "the default confirmation run is the complete frozen matrix"
        )

    if args.split == "confirmation":
        seeds = list(manifest["confirmation_mdp_seeds"])
    else:
        seeds = list(range(1000, 1020))
    if args.mdp_limit is not None:
        seeds = seeds[: args.mdp_limit]
    family_filter = tuple(args.family or PRIMARY_FAMILIES)
    rows = []
    for mdp_seed in seeds:
        mdp = LayeredMDP.random(
            seed=mdp_seed, horizon=6, state_count=7, action_count=6, gamma=0.97
        )
        if args.split == "confirmation":
            stream_seeds = list(
                manifest["monte_carlo_streams_by_mdp"][str(mdp_seed)]
            )[: args.stream_limit]
        else:
            stream_seeds = [
                deterministic_seed(mdp_seed, "estimator_matrix", index)
                for index in range(args.stream_limit)
            ]
        conditions = build_critic_conditions(mdp, mdp_seed=mdp_seed)
        if args.critic:
            requested = set(args.critic)
            conditions = [row for row in conditions if row.name in requested]
            missing = requested - {row.name for row in conditions}
            if missing:
                raise ValueError(f"unknown critic conditions: {sorted(missing)}")
        for critic in conditions:
            controllers = primary_controllers(mdp, critic.selector_scores)
            for family in family_filter:
                rows.append(
                    evaluate_one(
                        mdp,
                        mdp_seed=mdp_seed,
                        critic=critic,
                        family=family,
                        controller=controllers[family],
                        stream_seeds=stream_seeds,
                        samples_per_stream=args.samples_per_stream,
                    )
                )

    payload = {
        "schema": "cgcs-exact-truth-estimator-matrix-v1",
        "protocol_sha256": protocol_digest,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "split": args.split,
        "confirmation_mdp_instantiated": args.split == "confirmation",
        "complete_confirmation": args.split == "confirmation"
        and not partial_confirmation,
        "mdp_seeds": seeds,
        "critic_conditions": sorted({row["critic_condition"] for row in rows}),
        "search_families": list(family_filter),
        "stream_limit": args.stream_limit,
        "samples_per_stream": args.samples_per_stream,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"row_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
