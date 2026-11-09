#!/usr/bin/env python3
"""Render evidence-locked ICLR tables from completed CGCS artifacts.

This is an administrative renderer, not an experiment or a protocol.  It is
deliberately fail-closed: the S3 analysis and independent archive audit must
both be terminal and structurally complete before any table is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MECHANISM_ARMS = (
    "k1",
    "random_uniform",
    "entropy_uniform",
    "value_gate_uniform",
    "random_value",
    "value_guided",
)
MECHANISM_SEEDS = (99301, 99302, 99303)
S3_FAMILIES = ("best_of_n", "entropy_value", "value_value")
SEEDS = (99801, 99802, 99803)
S3_ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)
EVIDENCE_INPUTS = {
    "mechanism_analysis",
    "gate_confirmation",
    "critic_confirmation",
    "exact_confirmation",
    "exact_analysis",
    "s3_analysis",
    "s3_audit",
}
GATE_METHODS = (
    "random_uniform",
    "entropy_uniform",
    "value_uniform",
    "value_and_entropy_uniform",
    "learned_uniform",
    "oracle_gain_uniform",
)
EXACT_GATE_SETTINGS = ("K2_B1.5", "K2_B2", "K4_B1.5", "K4_B2")
EXACT_CRITIC_CONDITIONS = (
    "exact",
    "noise_0.5",
    "noise_1",
    "noise_2",
    "inverted_diagnostic",
    "lagged_kl_0.01",
    "lagged_kl_0.04",
    "learned_256",
    "learned_1024",
)
ESTIMATORS = (
    {
        "name": "E0",
        "implementation": "standard Actor ratio",
        "target": "search-distilled surrogate",
        "correction": "none for search allocation/retention",
        "main_finite_sample_issue": "selection and occupancy bias",
    },
    {
        "name": "E1",
        "implementation": "trajectory propensity",
        "target": "raw-Actor objective",
        "correction": "full candidate-selection trajectory",
        "main_finite_sample_issue": "product weights and low ESS",
    },
    {
        "name": "E2",
        "implementation": "executed-Action marginal",
        "target": "raw-Actor objective",
        "correction": "Rao--Blackwellized executed Action",
        "main_finite_sample_issue": "marginal probability computation",
    },
    {
        "name": "E3",
        "implementation": "partial cumulative correction",
        "target": "raw-Actor objective with controlled bias",
        "correction": "development-frozen lambda",
        "main_finite_sample_issue": "sample-size-dependent bias--variance tuning",
    },
    {
        "name": "E4",
        "implementation": "extended retained candidate set",
        "target": "retained composite-search objective",
        "correction": "scores the latent retained candidate set",
        "main_finite_sample_issue": "different estimand and greater compute",
    },
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"artifact is not a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    require(bool(rows), "cannot average an empty sequence")
    return sum(rows) / len(rows)


def valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_mechanism_analysis(payload: dict[str, Any]) -> None:
    require(
        payload.get("artifact_validation")
        == {
            "burnin_train_summaries": 3,
            "adaptive_train_summaries": 36,
            "evaluations": 18,
            "all_evaluations_exactly_200": True,
        },
        "mechanism replication artifact matrix is incomplete",
    )
    evaluations = payload.get("evaluation_by_seed")
    require(
        isinstance(evaluations, dict)
        and set(evaluations) == {str(seed) for seed in MECHANISM_SEEDS},
        "mechanism replication seed matrix drift",
    )
    for seed in MECHANISM_SEEDS:
        arms = evaluations[str(seed)]
        require(
            isinstance(arms, dict) and set(arms) == set(MECHANISM_ARMS),
            f"mechanism replication arm matrix drift for seed {seed}",
        )
        for arm, metrics in arms.items():
            require(
                isinstance(metrics, dict)
                and metrics.get("examples") == 200
                and all(
                    finite(metrics.get(key))
                    for key in ("accuracy", "answered_fraction", "mean_steps")
                )
                and 0.0 <= float(metrics["accuracy"])
                <= float(metrics["answered_fraction"])
                <= 1.0,
                f"invalid mechanism evaluation row {seed}/{arm}",
            )
    for key in ("adaptive_collection_by_arm", "adaptive_training_by_arm"):
        matrix = payload.get(key)
        require(
            isinstance(matrix, dict) and set(matrix) == set(MECHANISM_ARMS),
            f"mechanism replication {key} matrix drift",
        )
    gates = payload.get("mechanism_gates")
    gate_names = {
        "1_diversity",
        "2_value_guided_auc",
        "3_full_positive_vs_random_and_entropy",
        "4_factorized_contribution",
        "5_primary_clustered_ci_nonnegative",
    }
    require(
        isinstance(gates, dict)
        and set(gates) == gate_names | {"all_five"}
        and all(isinstance(gates[name], bool) for name in gate_names | {"all_five"})
        and gates["all_five"] == all(gates[name] for name in gate_names),
        "mechanism replication gate matrix is invalid",
    )
    manifest = payload.get("reproducibility_manifest")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    critical = {"protocol", "run_script", "implementation", "analysis"}
    require(
        isinstance(manifest, dict)
        and manifest.get("algorithm") == "sha256"
        and set(manifest.get("critical_source_labels", [])) == critical
        and isinstance(files, dict)
        and manifest.get("file_count") == len(files)
        and len(files) > 0
        and critical <= set(files)
        and all(valid_sha256(digest) for digest in files.values()),
        "mechanism replication reproducibility manifest is invalid",
    )


def mechanism_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    validate_mechanism_analysis(payload)
    value_metrics = payload.get("value_metrics")
    require(isinstance(value_metrics, dict), "mechanism Value metrics missing")
    require(set(value_metrics) == set(MECHANISM_ARMS), "mechanism arm matrix drift")
    rows: list[dict[str, Any]] = []
    for arm in MECHANISM_ARMS:
        for position in ("pre_action", "selected_post_action"):
            metrics = value_metrics[arm].get(position)
            require(isinstance(metrics, dict), f"missing {arm}/{position} metrics")
            for key in ("auc", "brier_clipped", "ece10_clipped", "observations"):
                require(finite(metrics.get(key)), f"non-finite {arm}/{position}/{key}")
            rows.append(
                {
                    "arm": arm,
                    "position": position,
                    "observations": int(metrics["observations"]),
                    "auc": float(metrics["auc"]),
                    "brier_clipped": float(metrics["brier_clipped"]),
                    "ece10_clipped": float(metrics["ece10_clipped"]),
                }
            )
    return rows


def gate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema") == "cgcs-exact-truth-gate-confirmation-v1",
        "exact gate schema mismatch",
    )
    require(
        payload.get("mdp_count") == 30
        and payload.get("settings") == list(EXACT_GATE_SETTINGS)
        and isinstance(payload.get("methods"), list)
        and set(GATE_METHODS) <= set(payload["methods"])
        and isinstance(payload.get("rows"), list)
        and len(payload["rows"]) == 30
        and isinstance(payload.get("paired_clustered_contrasts"), dict)
        and len(payload["paired_clustered_contrasts"]) == 40
        and all(
            valid_sha256(payload.get(key))
            for key in ("manifest_sha256", "protocol_sha256", "development_sha256")
        ),
        "exact gate confirmation is incomplete or unbound",
    )
    summary = payload.get("summary")
    require(
        isinstance(summary, dict) and set(summary) == set(EXACT_GATE_SETTINGS),
        "exact gate summary matrix drift",
    )
    rows: list[dict[str, Any]] = []
    for condition, methods in sorted(summary.items()):
        require(isinstance(methods, dict), f"invalid exact gate condition {condition}")
        for method in GATE_METHODS:
            metrics = methods.get(method)
            require(isinstance(metrics, dict), f"missing exact gate row {condition}/{method}")
            keys = (
                "mean_gate_auc_vs_oracle_allocation",
                "mean_gate_brier_vs_oracle_allocation",
                "mean_return_gain_vs_raw",
                "mean_mean_unique_candidates_when_expanded",
                "mean_all_candidates_identical_probability",
            )
            require(all(finite(metrics.get(key)) for key in keys), f"non-finite exact gate row {condition}/{method}")
            rows.append({"condition": condition, "method": method, **{key: float(metrics[key]) for key in keys}})
    return rows


def critic_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema") == "cgcs-exact-truth-critic-confirmation-v1",
        "exact Critic schema mismatch",
    )
    require(
        payload.get("mdp_count") == 30
        and isinstance(payload.get("critic_conditions"), list)
        and set(payload["critic_conditions"]) == set(EXACT_CRITIC_CONDITIONS)
        and payload.get("settings") == list(EXACT_GATE_SETTINGS)
        and isinstance(payload.get("gate_families"), list)
        and set(payload["gate_families"])
        == {"random", "entropy", "value", "value_and_entropy", "learned", "oracle_gain"}
        and isinstance(payload.get("rows"), list)
        and len(payload["rows"]) == 30
        and isinstance(payload.get("paired_clustered_contrasts"), dict)
        and len(payload["paired_clustered_contrasts"]) == 120
        and all(
            valid_sha256(payload.get(key))
            for key in ("manifest_sha256", "protocol_sha256", "gate_development_sha256")
        ),
        "exact Critic confirmation is incomplete or unbound",
    )
    pooled = payload.get("pooled_critic_diagnostics")
    require(isinstance(pooled, dict) and pooled, "pooled Critic diagnostics missing")
    rows: list[dict[str, Any]] = []
    for condition, gates in sorted(pooled.items()):
        require(isinstance(gates, dict) and gates, f"invalid Critic condition {condition}")
        for gate, metrics in sorted(gates.items()):
            keys = (
                "observation_count",
                "pearson_ranking_vs_return_gain",
                "pearson_ranking_vs_maximum_depth_tv",
                "pearson_ranking_vs_ess_proxy",
                "pearson_ranking_vs_selector_entropy",
            )
            require(all(finite(metrics.get(key)) for key in keys), f"non-finite Critic row {condition}/{gate}")
            rows.append({"condition": condition, "gate": gate, **{key: float(metrics[key]) for key in keys}})
    return rows


def validate_exact_confirmation(payload: dict[str, Any]) -> None:
    require(
        payload.get("schema") == "cgcs-exact-truth-estimator-matrix-v1",
        "exact estimator confirmation schema mismatch",
    )
    seeds = payload.get("mdp_seeds")
    rows = payload.get("rows")
    require(
        payload.get("complete_confirmation") is True
        and payload.get("confirmation_mdp_instantiated") is True
        and payload.get("split") == "confirmation"
        and isinstance(seeds, list)
        and len(seeds) == 30
        and len(set(seeds)) == 30
        and isinstance(payload.get("critic_conditions"), list)
        and set(payload["critic_conditions"]) == set(EXACT_CRITIC_CONDITIONS)
        and payload.get("search_families") == list(S3_FAMILIES)
        and payload.get("samples_per_stream") == 4096
        and payload.get("stream_limit") == 10
        and valid_sha256(payload.get("manifest_sha256"))
        and valid_sha256(payload.get("protocol_sha256"))
        and isinstance(rows, list)
        and len(rows) == 30 * len(EXACT_CRITIC_CONDITIONS) * len(S3_FAMILIES),
        "exact estimator confirmation matrix is incomplete or unbound",
    )
    identities = {
        (
            row.get("confirmation_mdp_seed"),
            row.get("critic_condition"),
            row.get("search_family"),
        )
        for row in rows
        if isinstance(row, dict)
    }
    require(
        identities
        == {
            (seed, condition, family)
            for seed in seeds
            for condition in EXACT_CRITIC_CONDITIONS
            for family in S3_FAMILIES
        },
        "exact estimator confirmation identity matrix drift",
    )


def exact_estimator_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema") == "cgcs-exact-truth-estimator-matrix-analysis-v1.1",
        "exact estimator-analysis schema mismatch",
    )
    exact = payload.get("summary", {}).get("exact")
    require(isinstance(exact, dict) and exact, "exact-Critic estimator summary missing")
    wanted = (
        "E0_standard_selected_action",
        "E1_trajectory",
        "E2_cumulative_marginal",
        "E3_step_cumulative_lambda_0.75",
        "E4_extended_candidate_set",
    )
    rows: list[dict[str, Any]] = []
    for family, family_payload in sorted(exact.items()):
        estimators = family_payload.get("estimators", {})
        for estimator in wanted:
            metrics = estimators.get(estimator)
            require(isinstance(metrics, dict), f"missing exact estimator {family}/{estimator}")
            keys = (
                "mean_relative_bias",
                "mean_gradient_relative_mse_batch_256",
                "mean_gradient_relative_mse_batch_1024",
                "mean_normalized_final_weight_ess",
                "mean_raw_actor_return_change",
                "mean_composite_search_return_change",
            )
            require(all(finite(metrics.get(key)) for key in keys), f"non-finite exact estimator {family}/{estimator}")
            rows.append({"family": family, "estimator": estimator, **{key: float(metrics[key]) for key in keys}})
    return rows


def exact_search_shift_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        payload.get("schema") == "cgcs-exact-truth-estimator-matrix-analysis-v1.1",
        "exact estimator-analysis schema mismatch",
    )
    summary = payload.get("summary")
    require(isinstance(summary, dict) and summary, "exact search-shift summary missing")
    exact = summary.get("exact")
    require(isinstance(exact, dict), "exact search-shift reference missing")
    rows: list[dict[str, Any]] = []
    for condition, families in sorted(summary.items()):
        require(isinstance(families, dict), f"invalid search-shift condition {condition}")
        require(
            set(families) == set(S3_FAMILIES),
            f"search-shift family matrix drift for {condition}",
        )
        for family in S3_FAMILIES:
            metrics = families[family]
            keys = (
                "mdp_count",
                "mean_critic_ranking_concordance",
                "mean_search_gain_vs_raw",
                "mean_maximum_depth_state_action_tv",
            )
            require(
                isinstance(metrics, dict)
                and all(finite(metrics.get(key)) for key in keys),
                f"non-finite search-shift row {condition}/{family}",
            )
            exact_gain = exact[family].get("mean_search_gain_vs_raw")
            require(finite(exact_gain), f"non-finite exact search gain for {family}")
            gain = float(metrics["mean_search_gain_vs_raw"])
            rows.append(
                {
                    "condition": condition,
                    "family": family,
                    "mdp_count": int(metrics["mdp_count"]),
                    "mean_critic_ranking_concordance": float(
                        metrics["mean_critic_ranking_concordance"]
                    ),
                    "mean_search_gain_vs_raw": gain,
                    "mean_maximum_depth_state_action_tv": float(
                        metrics["mean_maximum_depth_state_action_tv"]
                    ),
                    "oracle_search_gain_headroom": float(exact_gain) - gain,
                }
            )
    require(
        len(rows) == len(summary) * len(S3_FAMILIES),
        "exact search-shift row count mismatch",
    )
    return rows


def s3_value_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    require(
        audit.get("schema") == "cgcs-search-policy-mismatch-s3-archive-audit-v1",
        "S3 archive-audit schema mismatch",
    )
    require(audit.get("status") == "structurally_complete", "S3 archive is not terminal")
    rollouts = audit.get("training_rollout_audit")
    require(isinstance(rollouts, list) and len(rollouts) == 54, "S3 rollout-audit matrix must contain 54 rows")
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        for position, key in (
            ("pre_state", "pre_value_diagnostics"),
            ("candidate_zero_post_state", "candidate_zero_post_value_diagnostics"),
        ):
            metrics = rollout.get(key)
            require(isinstance(metrics, dict), f"S3 rollout missing {key}")
            required = ("states", "raw_brier_score", "clipped_ece10", "out_of_unit_fraction")
            require(all(finite(metrics.get(name)) for name in required), f"non-finite S3 Value diagnostics in {key}")
            auc = metrics.get("roc_auc")
            require(auc is None or finite(auc), f"non-finite S3 AUC in {key}")
            rows.append(
                {
                    "seed": int(rollout["seed"]),
                    "round": int(rollout["round"]),
                    "family": rollout["family"],
                    "estimator": rollout["estimator"],
                    "position": position,
                    "states": int(metrics["states"]),
                    "roc_auc": None if auc is None else float(auc),
                    "raw_brier_score": float(metrics["raw_brier_score"]),
                    "clipped_ece10": float(metrics["clipped_ece10"]),
                    "out_of_unit_fraction": float(metrics["out_of_unit_fraction"]),
                }
            )
    return rows


def aggregate_s3_value(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], row["estimator"], row["position"])].append(row)
    aggregated: list[dict[str, Any]] = []
    for (family, estimator, position), group in sorted(groups.items()):
        total = sum(row["states"] for row in group)
        require(total > 0, "empty S3 Value aggregation")
        def weighted(key: str) -> float | None:
            usable = [row for row in group if row[key] is not None]
            denominator = sum(row["states"] for row in usable)
            if denominator == 0:
                return None
            return sum(row["states"] * float(row[key]) for row in usable) / denominator
        aggregated.append(
            {
                "family": family,
                "estimator": estimator,
                "position": position,
                "rollouts": len(group),
                "states": total,
                "descriptive_state_weighted_auc": weighted("roc_auc"),
                "descriptive_state_weighted_raw_brier": weighted("raw_brier_score"),
                "descriptive_state_weighted_clipped_ece10": weighted("clipped_ece10"),
                "descriptive_state_weighted_out_of_unit_fraction": weighted("out_of_unit_fraction"),
            }
        )
    return aggregated


def s3_contrast_rows(contrasts: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        "raw_E3_minus_E0": {"best_of_n", "entropy_value", "value_value"},
        "raw_E1_minus_E0": {"best_of_n", "entropy_value", "value_value"},
        "composite_E4_minus_E0": {"best_of_n", "entropy_value", "value_value"},
        "raw_best_of_n_E3_minus_k1": {"best_of_n"},
    }
    rows: list[dict[str, Any]] = []
    for contrast, families in expected.items():
        contrast_rows = contrasts.get(contrast)
        require(isinstance(contrast_rows, list), f"missing S3 contrast {contrast}")
        require(
            {row.get("family") for row in contrast_rows} == families,
            f"S3 contrast-family matrix mismatch for {contrast}",
        )
        for row in contrast_rows:
            for key in ("estimate", "ci95_low", "ci95_high", "holm_adjusted_p"):
                require(finite(row.get(key)), f"non-finite S3 contrast {contrast}/{key}")
            wlt = row.get("pooled_wins_losses_ties")
            require(
                isinstance(wlt, dict)
                and all(isinstance(wlt.get(key), int) for key in ("wins", "losses", "ties"))
                and sum(wlt[key] for key in ("wins", "losses", "ties")) == 600,
                f"invalid pooled S3 W/L/T for {contrast}/{row.get('family')}",
            )
            per_seed = row.get("per_seed")
            require(
                isinstance(per_seed, list)
                and {seed_row.get("seed") for seed_row in per_seed}
                == {99801, 99802, 99803},
                f"invalid per-seed S3 contrast rows for {contrast}/{row.get('family')}",
            )
            rows.append(
                {
                    "contrast": contrast,
                    "family": row["family"],
                    "estimate": float(row["estimate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "holm_adjusted_p": float(row["holm_adjusted_p"]),
                    "pooled_wins_losses_ties": {
                        key: int(wlt[key]) for key in ("wins", "losses", "ties")
                    },
                    "per_seed": per_seed,
                }
            )
    require(len(rows) == 10, "S3 contrast matrix must contain 10 rows")
    return rows


def s3_compute_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    rows = audit.get("compute_by_family_estimator")
    require(isinstance(rows, list) and len(rows) == 12, "S3 compute matrix must contain 12 rows")
    require(
        {(row.get("family"), row.get("estimator")) for row in rows}
        == {(family, estimator) for family in S3_FAMILIES for estimator in S3_ESTIMATORS},
        "S3 compute family/estimator matrix mismatch",
    )
    keys = (
        "states",
        "optimizer_updates",
        "candidate_forwards",
        "candidate_tokens",
        "selected_tokens",
        "summed_invocation_elapsed_seconds",
        "candidate_token_ratio_to_same_family_E0",
        "candidate_forward_ratio_to_same_family_E0",
        "elapsed_ratio_to_same_family_E0",
        "minimum_ess",
        "maximum_external_weight",
        "maximum_clip_fraction",
        "maximum_kl_proxy",
    )
    for row in rows:
        require(all(finite(row.get(key)) for key in keys), "non-finite S3 compute row")
    return rows


def s3_evaluation_rows(audit: dict[str, Any]) -> list[dict[str, Any]]:
    completeness = audit.get("independent_artifact_completeness")
    require(isinstance(completeness, dict), "S3 independent completeness audit missing")
    rows = completeness.get("evaluation_metrics")
    require(isinstance(rows, list) and len(rows) == 75, "S3 evaluation matrix must contain 75 rows")
    require(sum(row.get("mode") == "raw_k1" for row in rows) == 39, "S3 raw-K1 evaluation count drift")
    require(sum(row.get("mode") == "composite" for row in rows) == 36, "S3 composite evaluation count drift")
    base_keys = ("seed", "accuracy", "answered_fraction", "elapsed_seconds")
    composite_keys = (
        "states",
        "registered_expansion_rate",
        "source_rollout_states",
        "source_rollout_expansion_rate",
        "expansion_rate",
        "expansion_rate_deviation",
        "evaluation_minus_source_rollout_expansion_rate",
        "generated_candidates",
        "registered_expected_candidates_per_state",
        "source_rollout_candidates_per_state",
        "realized_candidates_per_state",
        "evaluation_minus_source_rollout_candidates_per_state",
        "candidate_completion_tokens",
        "selected_completion_tokens",
        "candidate_completion_tokens_per_state",
        "selected_completion_tokens_per_state",
        "candidate_token_multiplier_vs_selected",
    )
    diversity_keys = (
        "mean_unique_candidates_when_expanded",
        "identical_expanded_fraction",
        "exact_token_pair_fraction",
        "mean_pairwise_prefix_divergence_fraction",
    )
    for row in rows:
        require(all(finite(row.get(key)) for key in base_keys), "non-finite S3 evaluation row")
        diagnostics = row.get("finalization_diagnostics")
        require(
            isinstance(diagnostics, dict),
            "S3 finalization diagnostics missing",
        )
        valid_finals = diagnostics.get("parseable_final_answers")
        unanswered = diagnostics.get("unanswered_examples")
        no_final_like = diagnostics.get("unanswered_no_final_like_marker")
        noncanonical = diagnostics.get(
            "unanswered_noncanonical_final_like_marker"
        )
        conditional = diagnostics.get("accuracy_given_parseable_final")
        mean_steps = diagnostics.get("mean_steps")
        mean_steps_answered = diagnostics.get("mean_steps_answered")
        mean_steps_unanswered = diagnostics.get("mean_steps_unanswered")
        require(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (
                    valid_finals,
                    unanswered,
                    no_final_like,
                    noncanonical,
                )
            ),
            "invalid S3 finalization counts",
        )
        require(
            valid_finals + unanswered == 200
            and no_final_like + noncanonical == unanswered,
            "S3 finalization counts do not close",
        )
        require(
            math.isclose(
                valid_finals / 200,
                float(row["answered_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            "S3 finalization counts disagree with answer rate",
        )
        require(
            diagnostics.get("all_unanswered_exhaust_max_steps") is True,
            "S3 unanswered examples did not all exhaust max_steps",
        )
        require(
            finite(mean_steps) and 1.0 <= float(mean_steps) <= 8.0,
            "invalid S3 mean step count",
        )
        require(
            (valid_finals == 0 and mean_steps_answered is None)
            or (
                valid_finals > 0
                and finite(mean_steps_answered)
                and 1.0 <= float(mean_steps_answered) <= 8.0
            ),
            "invalid S3 answered mean step count",
        )
        require(
            (unanswered == 0 and mean_steps_unanswered is None)
            or (
                unanswered > 0
                and finite(mean_steps_unanswered)
                and math.isclose(
                    float(mean_steps_unanswered),
                    8.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            "invalid S3 unanswered mean step count",
        )
        require(
            0.0 <= float(row["accuracy"]) <= float(row["answered_fraction"]),
            "S3 accuracy exceeds answer rate",
        )
        require(
            (
                valid_finals == 0
                and conditional is None
                and math.isclose(
                    float(row["accuracy"]), 0.0, rel_tol=0.0, abs_tol=1e-12
                )
            )
            or (
                valid_finals > 0
                and finite(conditional)
                and 0.0 <= float(conditional) <= 1.0
                and math.isclose(
                    float(conditional),
                    float(row["accuracy"]) / float(row["answered_fraction"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            "invalid S3 conditional accuracy",
        )
        if row.get("mode") == "composite":
            require(
                all(finite(row.get(key)) for key in composite_keys),
                "non-finite S3 composite compute row",
            )
            if math.isclose(
                float(row["expansion_rate"]), 0.0, rel_tol=0.0, abs_tol=1e-12
            ):
                require(
                    all(row.get(key) is None for key in diversity_keys),
                    "zero-expansion S3 composite must report diversity as unavailable",
                )
            else:
                require(
                    all(finite(row.get(key)) for key in diversity_keys),
                    "non-finite S3 composite diversity row",
                )
            family = row.get("family")
            expected_candidates = 2 if family == "best_of_n" else 4
            target_expansion = 1.0 if family == "best_of_n" else 1 / 3
            require(
                family in S3_FAMILIES
                and all(
                    isinstance(row[key], int)
                    and not isinstance(row[key], bool)
                    and row[key] > 0
                    for key in (
                        "candidate_completion_tokens",
                        "selected_completion_tokens",
                    )
                )
                and math.isclose(
                    float(row["registered_expansion_rate"]),
                    target_expansion,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["expansion_rate_deviation"]),
                    float(row["expansion_rate"]) - target_expansion,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and isinstance(row["source_rollout_states"], int)
                and not isinstance(row["source_rollout_states"], bool)
                and row["source_rollout_states"] > 0
                and 0.0 <= float(row["source_rollout_expansion_rate"]) <= 1.0
                and math.isclose(
                    float(row["evaluation_minus_source_rollout_expansion_rate"]),
                    float(row["expansion_rate"])
                    - float(row["source_rollout_expansion_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["registered_expected_candidates_per_state"]),
                    1.0 + (expected_candidates - 1) * target_expansion,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and 1.0
                <= float(row["source_rollout_candidates_per_state"])
                <= float(expected_candidates)
                and math.isclose(
                    float(row["source_rollout_candidates_per_state"]),
                    1.0
                    + (expected_candidates - 1)
                    * float(row["source_rollout_expansion_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["realized_candidates_per_state"]),
                    float(row["generated_candidates"]) / float(row["states"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["realized_candidates_per_state"]),
                    1.0
                    + (expected_candidates - 1)
                    * float(row["expansion_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(
                        row[
                            "evaluation_minus_source_rollout_candidates_per_state"
                        ]
                    ),
                    float(row["realized_candidates_per_state"])
                    - float(row["source_rollout_candidates_per_state"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["candidate_completion_tokens_per_state"]),
                    float(row["candidate_completion_tokens"]) / float(row["states"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(row["selected_completion_tokens_per_state"]),
                    float(row["selected_completion_tokens"]) / float(row["states"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and float(row["selected_completion_tokens"]) > 0.0
                and math.isclose(
                    float(row["candidate_token_multiplier_vs_selected"]),
                    float(row["candidate_completion_tokens"])
                    / float(row["selected_completion_tokens"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "invalid S3 composite target/realized compute decomposition",
            )
    return rows


def s3_contrast_decomposition_rows(
    audit: dict[str, Any], contrast_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = audit.get("contrast_metric_decomposition")
    require(
        isinstance(rows, list) and len(rows) == 30,
        "S3 contrast decomposition must contain 30 per-seed rows",
    )
    expected = {
        (contrast, family, seed)
        for contrast, families in (
            ("raw_E3_minus_E0", S3_FAMILIES),
            ("raw_E1_minus_E0", S3_FAMILIES),
            ("composite_E4_minus_E0", S3_FAMILIES),
            ("raw_best_of_n_E3_minus_k1", ("best_of_n",)),
        )
        for family in families
        for seed in SEEDS
    }
    identities = {
        (row.get("contrast"), row.get("family"), row.get("seed"))
        for row in rows
    }
    require(identities == expected, "S3 contrast decomposition identity drift")
    require(len(identities) == len(rows), "duplicate S3 contrast decomposition row")
    paired = {
        (row["contrast"], row["family"], seed_row["seed"]): seed_row["difference"]
        for row in contrast_rows
        for seed_row in row["per_seed"]
    }
    for row in rows:
        identity = (row["contrast"], row["family"], row["seed"])
        accuracy_difference = row.get("accuracy_difference")
        answer_rate_difference = row.get("answer_rate_difference")
        lhs_conditional = row.get("lhs_accuracy_given_parseable_final")
        rhs_conditional = row.get("rhs_accuracy_given_parseable_final")
        conditional_difference = row.get("conditional_accuracy_difference")
        require(
            finite(accuracy_difference)
            and finite(answer_rate_difference)
            and -1.0 <= float(accuracy_difference) <= 1.0
            and -1.0 <= float(answer_rate_difference) <= 1.0,
            f"{identity}: invalid S3 contrast decomposition",
        )
        require(
            math.isclose(
                float(accuracy_difference),
                float(paired[identity]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{identity}: decomposition disagrees with paired accuracy",
        )
        require(
            isinstance(row.get("lhs"), dict) and isinstance(row.get("rhs"), dict),
            f"{identity}: decomposition endpoints missing",
        )
        if lhs_conditional is None or rhs_conditional is None:
            require(
                conditional_difference is None,
                f"{identity}: undefined conditional accuracy has a difference",
            )
        else:
            require(
                finite(lhs_conditional)
                and finite(rhs_conditional)
                and finite(conditional_difference)
                and 0.0 <= float(lhs_conditional) <= 1.0
                and 0.0 <= float(rhs_conditional) <= 1.0
                and math.isclose(
                    float(conditional_difference),
                    float(lhs_conditional) - float(rhs_conditional),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{identity}: invalid conditional-accuracy decomposition",
            )
    return rows


def build_tables(
    mechanism: dict[str, Any],
    gate: dict[str, Any],
    critic: dict[str, Any],
    exact_confirmation: dict[str, Any],
    exact_analysis: dict[str, Any],
    s3_analysis: dict[str, Any],
    s3_audit: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    validate_exact_confirmation(exact_confirmation)
    require(
        set(exact_analysis.get("summary", {}))
        == set(exact_confirmation["critic_conditions"]),
        "exact estimator analysis condition matrix drift",
    )
    input_hashes = provenance.get("input_sha256")
    require(
        valid_sha256(provenance.get("renderer_sha256"))
        and isinstance(input_hashes, dict)
        and set(input_hashes) == EVIDENCE_INPUTS
        and all(valid_sha256(value) for value in input_hashes.values()),
        "evidence-table provenance matrix is incomplete or invalid",
    )
    require(
        exact_analysis.get("source_sha256")
        == input_hashes.get("exact_confirmation"),
        "exact estimator analysis is not bound to its confirmation artifact",
    )
    require(
        s3_analysis.get("schema") == "cgcs-search-policy-mismatch-s3-analysis-v1",
        "S3 analysis schema mismatch",
    )
    require(
        s3_analysis.get("artifact_counts")
        == {
            "burnin_training": 3,
            "k1_adaptive_training": 6,
            "search_adaptive_training": 72,
            "all_training": 81,
            "raw_k1_evaluations": 39,
            "composite_evaluations": 36,
        },
        "S3 terminal artifact counts mismatch",
    )
    expected_contrasts = {
        "raw_E3_minus_E0",
        "raw_E1_minus_E0",
        "composite_E4_minus_E0",
        "raw_best_of_n_E3_minus_k1",
    }
    contrasts = s3_analysis.get("contrasts")
    require(
        isinstance(contrasts, dict) and set(contrasts) == expected_contrasts,
        "S3 contrast matrix mismatch",
    )
    hashes = s3_audit.get("archive_sha256")
    require(isinstance(hashes, dict) and hashes, "S3 archive SHA-256 receipt missing")
    require(
        all(
            isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for digest in hashes.values()
        ),
        "invalid S3 archive SHA-256 receipt",
    )
    remote_tests = s3_audit.get("remote_postrun_tests")
    require(isinstance(remote_tests, dict), "S3 remote test receipt missing")
    require(
        int(remote_tests.get("tests", 0)) > 0
        and int(remote_tests.get("failures", -1)) == 0
        and int(remote_tests.get("errors", -1)) == 0
        and int(remote_tests.get("skipped", -1)) == 0,
        "S3 remote test receipt is not zero-failure and zero-skip",
    )
    for key in (
        "stability_gate_audit",
        "frozen_gate_logic_audit",
        "independent_run_session_audit",
        "manual_gate",
    ):
        require(isinstance(s3_audit.get(key), dict), f"S3 {key} missing")
    manual_gate = s3_audit["manual_gate"]
    require(
        manual_gate.get("verdict") in {"PASS", "FAIL"},
        "S3 manual compute gate must be explicitly reviewed",
    )
    require(
        isinstance(manual_gate.get("rationale"), str)
        and len(manual_gate["rationale"].strip()) >= 20,
        "S3 manual compute-review rationale missing",
    )
    for key in (
        "recorder_sha256",
        "preparer_sha256",
        "analysis_sha256",
        "report_sha256",
        "compute_evidence_sha256",
        "receipt_sha256",
    ):
        require(
            isinstance(manual_gate.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", manual_gate[key]),
            f"invalid S3 manual compute-review {key}",
        )
    raw_s3_value = s3_value_rows(s3_audit)
    flattened_contrasts = s3_contrast_rows(contrasts)
    compute_rows = s3_compute_rows(s3_audit)
    evaluation_rows = s3_evaluation_rows(s3_audit)
    decomposition_rows = s3_contrast_decomposition_rows(
        s3_audit, flattened_contrasts
    )
    return {
        "schema": "cgcs-search-policy-mismatch-iclr-tables-v1",
        "status": "evidence_complete",
        "provenance": provenance,
        "table_1_estimators": list(ESTIMATORS),
        "table_2_cross_stage": {
            "mechanism_replication_value": mechanism_rows(mechanism),
            "exact_gate_allocation": gate_rows(gate),
            "exact_critic_ranking_correlations": critic_rows(critic),
            "exact_estimator_bias_variance": exact_estimator_rows(exact_analysis),
            "exact_search_shift": exact_search_shift_rows(exact_analysis),
            "s3_value_diagnostics": raw_s3_value,
            "s3_value_descriptive_aggregation": aggregate_s3_value(raw_s3_value),
            "s3_contrasts": contrasts,
            "s3_contrast_rows": flattened_contrasts,
            "s3_contrast_metric_decomposition": decomposition_rows,
            "s3_compute_by_family_estimator": compute_rows,
            "s3_evaluation_metrics": evaluation_rows,
            "interpretation_boundary": (
                "S3 Value aggregates are state-count-weighted descriptive summaries "
                "of per-rollout metrics, not pooled calibration estimates or independent-sample inference."
            ),
        },
        "table_3_reproducibility": {
            "artifact_counts": s3_audit.get("artifact_counts"),
            "archive_sha256": s3_audit.get("archive_sha256"),
            "remote_postrun_tests": s3_audit.get("remote_postrun_tests"),
            "stability_gate_audit": s3_audit.get("stability_gate_audit"),
            "frozen_gate_logic_audit": s3_audit.get("frozen_gate_logic_audit"),
            "gpu_audit": s3_audit.get("independent_run_session_audit"),
            "manual_compute_gate": s3_audit.get("manual_gate"),
        },
    }


def fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# ICLR evidence tables",
        "",
        "Status: **evidence complete**. Generated from terminal validated artifacts; not a protocol amendment.",
        "",
        "## Table 1 — estimator definitions",
        "",
        "| Estimator | Implementation | Target | Correction | Main finite-sample issue |",
        "|---|---|---|---|---|",
    ]
    for row in payload["table_1_estimators"]:
        lines.append(
            f"| {row['name']} | {row['implementation']} | {row['target']} | "
            f"{row['correction']} | {row['main_finite_sample_issue']} |"
        )
    lines.extend(
        [
            "",
            "## Table 2 — cross-stage evidence",
            "",
            "### Joint Actor–Value mechanism replication",
            "",
            "| Arm | Position | n | AUC | Brier (clipped) | ECE-10 (clipped) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["mechanism_replication_value"]:
        lines.append(
            f"| {row['arm']} | {row['position']} | {row['observations']} | "
            f"{fmt(row['auc'])} | {fmt(row['brier_clipped'])} | {fmt(row['ece10_clipped'])} |"
        )
    lines.extend(
        [
            "",
            "### Exact-tree gate calibration and diversity",
            "",
            "| Condition | Gate | Allocation AUC | Allocation Brier | Return gain | Mean unique | Identical prob. |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["exact_gate_allocation"]:
        lines.append(
            f"| {row['condition']} | {row['method']} | "
            f"{fmt(row['mean_gate_auc_vs_oracle_allocation'])} | "
            f"{fmt(row['mean_gate_brier_vs_oracle_allocation'])} | "
            f"{fmt(row['mean_return_gain_vs_raw'])} | "
            f"{fmt(row['mean_mean_unique_candidates_when_expanded'])} | "
            f"{fmt(row['mean_all_candidates_identical_probability'])} |"
        )
    lines.extend(
        [
            "",
            "### Exact-tree Critic ranking diagnostics",
            "",
            "| Condition | Gate | n | Ranking vs gain | Ranking vs TV | Ranking vs ESS | Ranking vs selector entropy |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["exact_critic_ranking_correlations"]:
        lines.append(
            f"| {row['condition']} | {row['gate']} | {int(row['observation_count'])} | "
            f"{fmt(row['pearson_ranking_vs_return_gain'])} | "
            f"{fmt(row['pearson_ranking_vs_maximum_depth_tv'])} | "
            f"{fmt(row['pearson_ranking_vs_ess_proxy'])} | "
            f"{fmt(row['pearson_ranking_vs_selector_entropy'])} |"
        )
    lines.extend(
        [
            "",
            "### Exact-tree estimator bias--variance evidence",
            "",
            "| Family | Estimator | Relative bias | MSE@256 | MSE@1024 | ESS | Raw-Actor change | Composite change |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["exact_estimator_bias_variance"]:
        lines.append(
            f"| {row['family']} | {row['estimator']} | "
            f"{fmt(row['mean_relative_bias'])} | "
            f"{fmt(row['mean_gradient_relative_mse_batch_256'])} | "
            f"{fmt(row['mean_gradient_relative_mse_batch_1024'])} | "
            f"{fmt(row['mean_normalized_final_weight_ess'])} | "
            f"{fmt(row['mean_raw_actor_return_change'])} | "
            f"{fmt(row['mean_composite_search_return_change'])} |"
        )
    lines.extend(
        [
            "",
            "### Exact-tree search-shift conditions",
            "",
            "| Critic condition | Family | MDPs | Ranking concordance | Search gain | Occupancy TV | Headroom to exact |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["exact_search_shift"]:
        lines.append(
            f"| {row['condition']} | {row['family']} | {int(row['mdp_count'])} | "
            f"{fmt(row['mean_critic_ranking_concordance'])} | "
            f"{fmt(row['mean_search_gain_vs_raw'])} | "
            f"{fmt(row['mean_maximum_depth_state_action_tv'])} | "
            f"{fmt(row['oracle_search_gain_headroom'])} |"
        )
    lines.extend(
        [
            "",
            "### S3 Value diagnostics (descriptive aggregation)",
            "",
            "| Family | Estimator | Position | Rollouts | States | AUC | Raw Brier | Clipped ECE-10 | Out-of-unit |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["s3_value_descriptive_aggregation"]:
        lines.append(
            f"| {row['family']} | {row['estimator']} | {row['position']} | "
            f"{row['rollouts']} | {row['states']} | "
            f"{fmt(row['descriptive_state_weighted_auc'])} | "
            f"{fmt(row['descriptive_state_weighted_raw_brier'])} | "
            f"{fmt(row['descriptive_state_weighted_clipped_ece10'])} | "
            f"{fmt(row['descriptive_state_weighted_out_of_unit_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "### S3 paired confirmation contrasts",
            "",
            "| Contrast | Family | Difference | Clustered 95% CI | W/L/T | Holm p |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["s3_contrast_rows"]:
        wlt = row["pooled_wins_losses_ties"]
        lines.append(
            f"| {row['contrast']} | {row['family']} | {100 * row['estimate']:+.2f} pp | "
            f"[{100 * row['ci95_low']:+.2f}, {100 * row['ci95_high']:+.2f}] | "
            f"{wlt['wins']}/{wlt['losses']}/{wlt['ties']} | "
            f"{row['holm_adjusted_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "#### Per-seed accuracy/finalization decomposition",
            "",
            "The registered overall-accuracy difference remains primary. The "
            "answer-rate and conditional-accuracy differences diagnose whether "
            "the observed change is associated with completion/finalization, "
            "correctness among valid finals, or both.",
            "",
            "| Contrast | Family | Seed | Accuracy difference | Answer-rate "
            "difference | Conditional-accuracy difference |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"][
        "s3_contrast_metric_decomposition"
    ]:
        conditional = row["conditional_accuracy_difference"]
        lines.append(
            f"| {row['contrast']} | {row['family']} | {row['seed']} | "
            f"{100 * row['accuracy_difference']:+.2f} pp | "
            f"{100 * row['answer_rate_difference']:+.2f} pp | "
            f"{'NA' if conditional is None else f'{100 * conditional:+.2f} pp'} |"
        )
    lines.extend(
        [
            "",
            "### S3 matched-compute evidence",
            "",
            "| Family | Estimator | States | Updates | Candidate forwards | Candidate tokens | Selected tokens | Token ratio vs E0 | Time ratio vs E0 | Min ESS | Max clip | Max KL |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["s3_compute_by_family_estimator"]:
        lines.append(
            f"| {row['family']} | {row['estimator']} | {int(row['states'])} | "
            f"{int(row['optimizer_updates'])} | {int(row['candidate_forwards'])} | "
            f"{int(row['candidate_tokens'])} | {int(row['selected_tokens'])} | "
            f"{fmt(row['candidate_token_ratio_to_same_family_E0'])} | "
            f"{fmt(row['elapsed_ratio_to_same_family_E0'])} | "
            f"{fmt(row['minimum_ess'])} | {fmt(row['maximum_clip_fraction'])} | "
            f"{fmt(row['maximum_kl_proxy'])} |"
        )
    lines.extend(
        [
            "",
            "### S3 retained-composite diversity",
            "",
            "Answer rate denotes strict scalar `FINAL:` extraction rather than "
            "explicit abstention. Conditional accuracy is accuracy among those "
            "valid finals; unanswered failures are split by retained-text "
            "evidence.",
            "",
            "| Seed | Family | Estimator | Accuracy | Answer rate | Conditional accuracy | No final-like | Noncanonical final-like | Target expansion | Source expansion | Evaluation expansion | Eval-source expansion shift | Target candidates/state | Source candidates/state | Evaluation candidates/state | Eval-source candidate shift | Candidates | Candidate tokens | Selected tokens | Token multiplier | Mean unique | Identical | Exact-token pairs | Prefix divergence |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["table_2_cross_stage"]["s3_evaluation_metrics"]:
        if row["mode"] != "composite":
            continue
        finalization = row["finalization_diagnostics"]
        conditional = finalization["accuracy_given_parseable_final"]
        lines.append(
            f"| {int(row['seed'])} | {row['family']} | {row['estimator']} | "
            f"{fmt(row['accuracy'])} | {fmt(row['answered_fraction'])} | "
            f"{fmt(conditional) if conditional is not None else 'NA'} | "
            f"{int(finalization['unanswered_no_final_like_marker'])} | "
            f"{int(finalization['unanswered_noncanonical_final_like_marker'])} | "
            f"{fmt(row['registered_expansion_rate'])} | "
            f"{fmt(row['source_rollout_expansion_rate'])} | "
            f"{fmt(row['expansion_rate'])} | "
            f"{fmt(row['evaluation_minus_source_rollout_expansion_rate'])} | "
            f"{fmt(row['registered_expected_candidates_per_state'])} | "
            f"{fmt(row['source_rollout_candidates_per_state'])} | "
            f"{fmt(row['realized_candidates_per_state'])} | "
            f"{fmt(row['evaluation_minus_source_rollout_candidates_per_state'])} | "
            f"{int(row['generated_candidates'])} | "
            f"{int(row['candidate_completion_tokens'])} | "
            f"{int(row['selected_completion_tokens'])} | "
            f"{fmt(row['candidate_token_multiplier_vs_selected'])} | "
            f"{fmt(row['mean_unique_candidates_when_expanded'])} | "
            f"{fmt(row['identical_expanded_fraction'])} | "
            f"{fmt(row['exact_token_pair_fraction'])} | "
            f"{fmt(row['mean_pairwise_prefix_divergence_fraction'])} |"
        )
    lines.extend(
        [
            "",
            payload["table_2_cross_stage"]["interpretation_boundary"],
            "",
            "## Table 3 — frozen gates and reproducibility",
            "",
            "```json",
            json.dumps(payload["table_3_reproducibility"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    provenance = payload.get("provenance", {})
    if provenance:
        lines.extend(
            [
                "",
                "## Deterministic provenance",
                "",
                "| Input | SHA-256 |",
                "|---|---|",
                f"| renderer | `{provenance['renderer_sha256']}` |",
            ]
        )
        for label, digest in sorted(provenance["input_sha256"].items()):
            lines.append(f"| {label} | `{digest}` |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-analysis", type=Path, required=True)
    parser.add_argument("--gate-confirmation", type=Path, required=True)
    parser.add_argument("--critic-confirmation", type=Path, required=True)
    parser.add_argument("--exact-confirmation", type=Path, required=True)
    parser.add_argument("--exact-analysis", type=Path, required=True)
    parser.add_argument("--s3-analysis", type=Path, required=True)
    parser.add_argument("--s3-audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    inputs = {
        "mechanism_analysis": args.mechanism_analysis,
        "gate_confirmation": args.gate_confirmation,
        "critic_confirmation": args.critic_confirmation,
        "exact_confirmation": args.exact_confirmation,
        "exact_analysis": args.exact_analysis,
        "s3_analysis": args.s3_analysis,
        "s3_audit": args.s3_audit,
    }
    payload = build_tables(
        *(read_json(path) for path in inputs.values()),
        provenance={
            "renderer_sha256": sha256_file(Path(__file__).resolve()),
            "input_sha256": {
                label: sha256_file(path) for label, path in inputs.items()
            },
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_markdown(payload))
    print(json.dumps({"status": payload["status"], "schema": payload["schema"]}, indent=2))


if __name__ == "__main__":
    main()
