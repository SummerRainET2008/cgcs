#!/usr/bin/env python3
"""Run the frozen Critic-quality axis across all exact search families."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.critic_quality import build_critic_conditions
from acbpo.layered_mdp import LayeredMDP
from acbpo.search_gate_study import (
    calibrate_gate,
    evaluate_gate,
    exact_expansion_gain,
    expected_selector_entropy_by_state,
    full_expansion_policy,
    gate_features,
    realized_expansion_rate,
    standardize_per_layer,
)


def evaluate_condition(
    mdp: LayeredMDP,
    *,
    critic,
    candidate_count: int,
    budget: float,
    learned_coefficient: np.ndarray,
) -> dict[str, dict[str, object]]:
    _, true_q = mdp.values()
    expanded = full_expansion_policy(
        mdp,
        critic.selector_scores,
        candidate_count=candidate_count,
        beta=1.0,
        epsilon=0.10,
    )
    oracle_gain = exact_expansion_gain(mdp, expanded, true_q)
    features, score_map = gate_features(mdp)
    score_map = {
        **score_map,
        "learned": features @ learned_coefficient,
        "oracle_gain": standardize_per_layer(oracle_gain),
    }
    selector_entropy = expected_selector_entropy_by_state(
        mdp.actor,
        critic.selector_scores,
        candidate_count=candidate_count,
        beta=1.0,
        epsilon=0.10,
    )
    target_rate = (budget - 1.0) / (candidate_count - 1.0)
    output = {}
    for gate_name, score in score_map.items():
        gate = calibrate_gate(
            mdp, expanded, score, target_expansion_rate=target_rate
        )
        evaluation = evaluate_gate(
            mdp,
            expanded,
            gate,
            oracle_gain=oracle_gain,
            target_expansion_rate=target_rate,
            candidate_count=candidate_count,
        )
        _, _, occupancy = realized_expansion_rate(mdp, expanded, gate)
        expansion_mass = float(np.sum(occupancy * gate))
        metrics = dict(evaluation.metrics)
        metrics.update(
            {
                "critic_ranking_concordance": critic.ranking_concordance,
                "mean_selector_entropy_when_expanded": float(
                    np.sum(occupancy * gate * selector_entropy)
                    / max(expansion_mass, 1e-300)
                ),
            }
        )
        output[gate_name] = metrics
    return output


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    output = {}
    condition_names = rows[0]["conditions"].keys()
    for condition in condition_names:
        output[condition] = {}
        setting_names = rows[0]["conditions"][condition]["settings"].keys()
        for setting in setting_names:
            output[condition][setting] = {}
            method_names = rows[0]["conditions"][condition]["settings"][setting].keys()
            for method in method_names:
                output[condition][setting][method] = {}
                metrics = rows[0]["conditions"][condition]["settings"][setting][method]
                for metric, example in metrics.items():
                    if isinstance(example, (int, float)):
                        values = np.asarray(
                            [
                                row["conditions"][condition]["settings"][setting][method][metric]
                                for row in rows
                            ]
                        )
                        output[condition][setting][method][f"mean_{metric}"] = float(
                            values.mean()
                        )
                        output[condition][setting][method][
                            f"std_across_mdp_{metric}"
                        ] = float(values.std(ddof=1))
    return output


def bootstrap_exact_difference(
    rows: list[dict[str, object]],
    *,
    condition: str,
    setting: str,
    method: str,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, object]:
    difference = np.asarray(
        [
            row["conditions"][condition]["settings"][setting][method][
                "composite_search_return"
            ]
            - row["conditions"]["exact"]["settings"][setting][method][
                "composite_search_return"
            ]
            for row in rows
        ]
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(resamples, len(rows)))
    means = difference[indices].mean(axis=1)
    return {
        "condition": condition,
        "reference": "exact",
        "setting": setting,
        "method": method,
        "mean_return_difference": float(difference.mean()),
        "clustered_bootstrap_95_ci": np.quantile(means, [0.025, 0.975]).tolist(),
        "wins": int(np.sum(difference > 0.0)),
        "losses": int(np.sum(difference < 0.0)),
        "ties": int(np.sum(difference == 0.0)),
    }


def pooled_critic_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    """Relate frozen Critic ranking quality to utility and shift diagnostics.

    The unit is one committed MDP x one preregistered Critic condition.  These
    correlations are descriptive diagnostics over the complete frozen axis;
    no condition is selected or discarded based on them.
    """
    output: dict[str, object] = {}
    for setting in ("K2_B1.5", "K4_B1.5", "K4_B2"):
        output[setting] = {}
        for method in ("random", "entropy"):
            observations = []
            for row in rows:
                for condition in row["conditions"].values():
                    metrics = condition["settings"][setting][method]
                    observations.append(
                        [
                            condition["ranking_concordance"],
                            metrics["return_gain_vs_raw"],
                            metrics["maximum_depth_state_action_tv"],
                            metrics["one_step_exact_marginal_ess_proxy"],
                            metrics["mean_selector_entropy_when_expanded"],
                        ]
                    )
            values = np.asarray(observations, dtype=np.float64)
            ranking = values[:, 0]
            output[setting][method] = {
                "observation_count": int(values.shape[0]),
                "pearson_ranking_vs_return_gain": float(
                    np.corrcoef(ranking, values[:, 1])[0, 1]
                ),
                "pearson_ranking_vs_maximum_depth_tv": float(
                    np.corrcoef(ranking, values[:, 2])[0, 1]
                ),
                "pearson_ranking_vs_ess_proxy": float(
                    np.corrcoef(ranking, values[:, 3])[0, 1]
                ),
                "pearson_ranking_vs_selector_entropy": float(
                    np.corrcoef(ranking, values[:, 4])[0, 1]
                ),
            }
    return output


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
        "--gate-development",
        type=Path,
        default=Path("results/search_policy_mismatch/GATE_DEVELOPMENT.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    gate_development = json.loads(args.gate_development.read_text())
    if manifest["protocol_sha256"] != protocol_digest:
        raise ValueError("manifest/protocol digest mismatch")
    if gate_development["protocol_sha256"] != protocol_digest:
        raise ValueError("gate development/protocol digest mismatch")
    settings = ((2, 1.5), (2, 2.0), (4, 1.5), (4, 2.0))
    rows = []
    for mdp_seed in manifest["confirmation_mdp_seeds"]:
        mdp = LayeredMDP.random(
            seed=mdp_seed, horizon=6, state_count=7, action_count=6, gamma=0.97
        )
        row_conditions = {}
        for critic in build_critic_conditions(mdp, mdp_seed=mdp_seed):
            condition_settings = {}
            for candidate_count, budget in settings:
                coefficient = np.asarray(
                    gate_development["conditions_by_candidate_count"][
                        str(candidate_count)
                    ]["coefficient"]
                )
                setting = f"K{candidate_count}_B{budget:g}"
                condition_settings[setting] = evaluate_condition(
                    mdp,
                    critic=critic,
                    candidate_count=candidate_count,
                    budget=budget,
                    learned_coefficient=coefficient,
                )
            row_conditions[critic.name] = {
                "metadata": critic.metadata,
                "ranking_concordance": critic.ranking_concordance,
                "settings": condition_settings,
            }
        rows.append(
            {"confirmation_mdp_seed": mdp_seed, "conditions": row_conditions}
        )
    summary = aggregate(rows)
    contrasts = {}
    index = 0
    for condition in summary:
        if condition == "exact":
            continue
        for setting in ("K2_B1.5", "K4_B1.5", "K4_B2"):
            for method in ("random", "entropy", "value", "value_and_entropy", "learned"):
                key = f"{condition}:{setting}:{method}_vs_exact"
                contrasts[key] = bootstrap_exact_difference(
                    rows,
                    condition=condition,
                    setting=setting,
                    method=method,
                    seed=2026107000 + index,
                )
                index += 1
    payload = {
        "schema": "cgcs-exact-truth-critic-confirmation-v1",
        "protocol_sha256": protocol_digest,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "gate_development_sha256": hashlib.sha256(
            args.gate_development.read_bytes()
        ).hexdigest(),
        "mdp_count": len(rows),
        "critic_conditions": list(rows[0]["conditions"].keys()),
        "settings": [f"K{k}_B{b:g}" for k, b in settings],
        "gate_families": list(
            next(iter(rows[0]["conditions"].values()))["settings"]["K2_B1.5"].keys()
        ),
        "summary": summary,
        "pooled_critic_diagnostics": pooled_critic_diagnostics(rows),
        "paired_clustered_contrasts": contrasts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"mdp_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
