#!/usr/bin/env python3
"""Aggregate and cluster-bootstrap the frozen estimator confirmation matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = (
    "relative_bias",
    "cosine",
    "component_sign_error",
    "gradient_variance_trace",
    "gradient_relative_mse_batch_256",
    "gradient_relative_mse_batch_1024",
    "normalized_final_weight_ess",
    "maximum_final_weight",
    "return_bias",
    "raw_actor_return_change",
    "composite_search_return_change",
    "post_update_actor_kl",
    "post_update_composite_kl",
)


CONTRAST_SPECS = {
    "E3_lambda075_vs_E0_mse256": (
        "E3_step_cumulative_lambda_0.75",
        "E0_standard_selected_action",
        "gradient_relative_mse_batch_256",
    ),
    "E3_lambda075_vs_E2_mse256": (
        "E3_step_cumulative_lambda_0.75",
        "E2_cumulative_marginal",
        "gradient_relative_mse_batch_256",
    ),
    "E3_lambda075_vs_E2_mse1024": (
        "E3_step_cumulative_lambda_0.75",
        "E2_cumulative_marginal",
        "gradient_relative_mse_batch_1024",
    ),
    "E3_lambda075_vs_E0_K1_change": (
        "E3_step_cumulative_lambda_0.75",
        "E0_standard_selected_action",
        "raw_actor_return_change",
    ),
    "E1_trajectory_vs_E2_mse256": (
        "E1_trajectory",
        "E2_cumulative_marginal",
        "gradient_relative_mse_batch_256",
    ),
    "E4_marginal_vs_extended_mse256": (
        "E4_executed_action_marginal",
        "E4_extended_candidate_set",
        "gradient_relative_mse_batch_256",
    ),
    "E4_proxy_vs_marginal_mse256": (
        "E4_selected_action_selector_proxy",
        "E4_executed_action_marginal",
        "gradient_relative_mse_batch_256",
    ),
    "E4_proxy_vs_marginal_search_change": (
        "E4_selected_action_selector_proxy",
        "E4_executed_action_marginal",
        "composite_search_return_change",
    ),
}


def bootstrap_difference(
    differences: np.ndarray, *, seed: int, resamples: int = 10_000
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "cluster_count": int(differences.size),
        "mean_difference": float(differences.mean()),
        "clustered_bootstrap_95_ci": np.quantile(
            means, [0.025, 0.975]
        ).tolist(),
        "wins": int(np.sum(differences > 0.0)),
        "losses": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
    }


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    conditions = sorted({row["critic_condition"] for row in rows})
    families = sorted({row["search_family"] for row in rows})
    estimators = list(rows[0]["estimators"])
    for condition in conditions:
        output[condition] = {}
        for family in families:
            selected = [
                row for row in rows
                if row["critic_condition"] == condition
                and row["search_family"] == family
            ]
            family_output: dict[str, object] = {
                "mdp_count": len(selected),
                "mean_critic_ranking_concordance": float(
                    np.mean([row["critic_ranking_concordance"] for row in selected])
                ),
                "mean_search_gain_vs_raw": float(
                    np.mean([row["search_gain_vs_raw"] for row in selected])
                ),
                "mean_maximum_depth_state_action_tv": float(
                    np.mean(
                        [
                            row["occupancy"]["maximum_depth_state_action_tv"]
                            for row in selected
                        ]
                    )
                ),
                "estimators": {},
            }
            for estimator in estimators:
                family_output["estimators"][estimator] = {
                    f"mean_{metric}": float(
                        np.mean(
                            [row["estimators"][estimator][metric] for row in selected]
                        )
                    )
                    for metric in METRICS
                }
            output[condition][family] = family_output
    return output


def contrasts(rows: list[dict[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    conditions = sorted({row["critic_condition"] for row in rows})
    families = sorted({row["search_family"] for row in rows})
    index = 0
    for condition in conditions:
        for family in families:
            selected = sorted(
                [
                    row for row in rows
                    if row["critic_condition"] == condition
                    and row["search_family"] == family
                ],
                key=lambda row: row["confirmation_mdp_seed"],
            )
            for name, (left, right, metric) in CONTRAST_SPECS.items():
                difference = np.asarray(
                    [
                        row["estimators"][left][metric]
                        - row["estimators"][right][metric]
                        for row in selected
                    ],
                    dtype=np.float64,
                )
                key = f"{condition}:{family}:{name}"
                output[key] = {
                    "critic_condition": condition,
                    "search_family": family,
                    "contrast": name,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    **bootstrap_difference(
                        difference, seed=2026110000 + index
                    ),
                }
                index += 1

    # Critic-robust result: average the nine paired differences within each MDP,
    # then bootstrap the 30 independent MDP clusters.
    for family in families:
        for name, (left, right, metric) in CONTRAST_SPECS.items():
            by_mdp = {}
            for row in rows:
                if row["search_family"] != family:
                    continue
                difference = (
                    row["estimators"][left][metric]
                    - row["estimators"][right][metric]
                )
                by_mdp.setdefault(row["confirmation_mdp_seed"], []).append(
                    difference
                )
            clustered = np.asarray(
                [np.mean(by_mdp[key]) for key in sorted(by_mdp)], dtype=np.float64
            )
            key = f"pooled_critics:{family}:{name}"
            output[key] = {
                "critic_condition": "pooled_critics_within_mdp_mean",
                "search_family": family,
                "contrast": name,
                "left": left,
                "right": right,
                "metric": metric,
                **bootstrap_difference(
                    clustered, seed=2026120000 + index
                ),
            }
            index += 1

    family_pairs = {
        "best_of_n_vs_entropy_value_search_gain": (
            "best_of_n",
            "entropy_value",
        ),
        "entropy_value_vs_value_value_search_gain": (
            "entropy_value",
            "value_value",
        ),
    }
    for condition in conditions:
        for name, (left_family, right_family) in family_pairs.items():
            by_key = {
                (row["confirmation_mdp_seed"], row["search_family"]): row
                for row in rows
                if row["critic_condition"] == condition
            }
            mdp_seeds = sorted(
                {key[0] for key in by_key if key[1] == left_family}
            )
            difference = np.asarray(
                [
                    by_key[(seed, left_family)]["search_gain_vs_raw"]
                    - by_key[(seed, right_family)]["search_gain_vs_raw"]
                    for seed in mdp_seeds
                ],
                dtype=np.float64,
            )
            key = f"{condition}:search_family:{name}"
            output[key] = {
                "critic_condition": condition,
                "contrast": name,
                "left": left_family,
                "right": right_family,
                "metric": "search_gain_vs_raw",
                **bootstrap_difference(
                    difference, seed=2026130000 + index
                ),
            }
            index += 1

    for name, (left_family, right_family) in family_pairs.items():
        by_mdp: dict[int, list[float]] = {}
        lookup = {
            (
                row["confirmation_mdp_seed"],
                row["critic_condition"],
                row["search_family"],
            ): row
            for row in rows
        }
        for mdp_seed in sorted({row["confirmation_mdp_seed"] for row in rows}):
            differences = []
            for condition in conditions:
                differences.append(
                    lookup[(mdp_seed, condition, left_family)][
                        "search_gain_vs_raw"
                    ]
                    - lookup[(mdp_seed, condition, right_family)][
                        "search_gain_vs_raw"
                    ]
                )
            by_mdp[mdp_seed] = differences
        clustered = np.asarray(
            [np.mean(by_mdp[key]) for key in sorted(by_mdp)], dtype=np.float64
        )
        key = f"pooled_critics:search_family:{name}"
        output[key] = {
            "critic_condition": "pooled_critics_within_mdp_mean",
            "contrast": name,
            "left": left_family,
            "right": right_family,
            "metric": "search_gain_vs_raw",
            **bootstrap_difference(clustered, seed=2026140000 + index),
        }
        index += 1
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if not payload.get("complete_confirmation") or len(payload["rows"]) != 810:
        raise ValueError("analysis requires the complete 810-row confirmation")
    output = {
        "schema": "cgcs-exact-truth-estimator-matrix-analysis-v1.1",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "cluster_unit": "confirmation_mdp_seed",
        "critic_conditions_are_repeated_within_mdp": True,
        "summary": aggregate(payload["rows"]),
        "paired_clustered_contrasts": contrasts(payload["rows"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"contrast_count": len(output["paired_clustered_contrasts"])}))


if __name__ == "__main__":
    main()
