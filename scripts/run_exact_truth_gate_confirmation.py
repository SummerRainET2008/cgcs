#!/usr/bin/env python3
"""Run compute-matched exact gate/search-family comparisons on confirmation MDPs."""
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
    full_expansion_policy,
    gate_features,
    standardize_per_layer,
)


def standardize_q(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / np.maximum(values.std(axis=-1, keepdims=True), 1e-12)


def run_condition(
    mdp: LayeredMDP,
    *,
    candidate_count: int,
    budget: float,
    learned_coefficient: np.ndarray,
) -> dict[str, dict[str, object]]:
    _, q_values = mdp.values()
    critic = standardize_q(q_values)
    value_expanded = full_expansion_policy(
        mdp,
        critic,
        candidate_count=candidate_count,
        beta=1.0,
        epsilon=0.10,
    )
    uniform_expanded = full_expansion_policy(
        mdp,
        critic,
        candidate_count=candidate_count,
        beta=0.0,
        epsilon=0.10,
    )
    oracle_gain = exact_expansion_gain(mdp, value_expanded, q_values)
    features, score_map = gate_features(mdp)
    score_map = {
        **score_map,
        "learned": features @ learned_coefficient,
        "oracle_gain": standardize_per_layer(oracle_gain),
    }
    target_rate = (budget - 1.0) / (candidate_count - 1.0)
    output = {}
    for selector_name, expanded_policy in (
        ("uniform", uniform_expanded),
        ("value", value_expanded),
    ):
        for gate_name, score in score_map.items():
            gate = calibrate_gate(
                mdp,
                expanded_policy,
                score,
                target_expansion_rate=target_rate,
                temperature=1.0,
            )
            evaluation = evaluate_gate(
                mdp,
                expanded_policy,
                gate,
                oracle_gain=oracle_gain,
                target_expansion_rate=target_rate,
                candidate_count=candidate_count,
            )
            output[f"{gate_name}_{selector_name}"] = evaluation.metrics
    # The compute-matched best-of-N baseline is random expansion plus Value selection.
    output["best_of_n"] = dict(output["random_value"])
    return output


def clustered_bootstrap(
    differences: np.ndarray, *, seed: int, resamples: int = 10_000
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    samples = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "clustered_bootstrap_95_ci": np.quantile(samples, [0.025, 0.975]).tolist(),
        "wins": int(np.sum(differences > 0.0)),
        "losses": int(np.sum(differences < 0.0)),
        "ties": int(np.sum(differences == 0.0)),
        "cluster_count": int(differences.size),
        "resamples": resamples,
    }


def aggregate(rows: list[dict[str, object]]) -> tuple[dict, dict]:
    settings = rows[0]["settings"].keys()
    summary = {}
    contrasts = {}
    contrast_index = 0
    for setting in settings:
        methods = rows[0]["settings"][setting].keys()
        summary[setting] = {}
        for method in methods:
            summary[setting][method] = {}
            for metric, example in rows[0]["settings"][setting][method].items():
                if isinstance(example, (int, float)):
                    values = np.asarray(
                        [row["settings"][setting][method][metric] for row in rows]
                    )
                    summary[setting][method][f"mean_{metric}"] = float(values.mean())
                    summary[setting][method][f"std_across_mdp_{metric}"] = float(
                        values.std(ddof=1)
                    )
        for method in methods:
            if method in {"random_uniform", "random_value", "best_of_n"}:
                continue
            selector = "uniform" if method.endswith("_uniform") else "value"
            baseline = f"random_{selector}"
            difference = np.asarray(
                [
                    row["settings"][setting][method]["composite_search_return"]
                    - row["settings"][setting][baseline]["composite_search_return"]
                    for row in rows
                ]
            )
            contrasts[f"{setting}:{method}_vs_{baseline}"] = clustered_bootstrap(
                difference, seed=2026105000 + contrast_index
            )
            contrast_index += 1
    return summary, contrasts


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
        "--development",
        type=Path,
        default=Path("results/search_policy_mismatch/GATE_DEVELOPMENT.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_digest = hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    manifest = json.loads(args.manifest.read_text())
    development = json.loads(args.development.read_text())
    if manifest["protocol_sha256"] != protocol_digest:
        raise ValueError("manifest/protocol digest mismatch")
    if development["protocol_sha256"] != protocol_digest:
        raise ValueError("development/protocol digest mismatch")

    settings = ((2, 1.5), (2, 2.0), (4, 1.5), (4, 2.0))
    rows = []
    for seed in manifest["confirmation_mdp_seeds"]:
        mdp = LayeredMDP.random(
            seed=seed, horizon=6, state_count=7, action_count=6, gamma=0.97
        )
        row_settings = {}
        for candidate_count, budget in settings:
            coefficient = np.asarray(
                development["conditions_by_candidate_count"][str(candidate_count)][
                    "coefficient"
                ]
            )
            key = f"K{candidate_count}_B{budget:g}"
            row_settings[key] = run_condition(
                mdp,
                candidate_count=candidate_count,
                budget=budget,
                learned_coefficient=coefficient,
            )
        rows.append({"confirmation_mdp_seed": seed, "settings": row_settings})
    summary, contrasts = aggregate(rows)
    payload = {
        "schema": "cgcs-exact-truth-gate-confirmation-v1",
        "protocol_sha256": protocol_digest,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "development_sha256": hashlib.sha256(args.development.read_bytes()).hexdigest(),
        "mdp_count": len(rows),
        "settings": [f"K{k}_B{b:g}" for k, b in settings],
        "methods": list(next(iter(rows[0]["settings"].values())).keys()),
        "summary": summary,
        "paired_clustered_contrasts": contrasts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"mdp_count": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
