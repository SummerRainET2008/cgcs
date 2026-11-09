#!/usr/bin/env python3
"""Administrative post-run audit for the frozen S3 archive.

This script does not change the preregistered analyzer or its promotion rules.
It exposes the weight-distribution and compute-accounting evidence that the
frozen protocol requires an independent human audit to inspect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any


SEEDS = (99801, 99802, 99803)
FAMILIES = ("best_of_n", "entropy_value", "value_value")
ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)
EXPECTED_COUNTS = {
    "burnin_training": 3,
    "k1_adaptive_training": 6,
    "search_adaptive_training": 72,
    "all_training": 81,
    "raw_k1_evaluations": 39,
    "composite_evaluations": 36,
}
QUANTILE_KEY = "external_weight_quantiles_0_50_90_95_99_100"
PROTOCOL_SHA256 = "be485c2defcf04414b9ef925944951b1948c9c1f15bb088f70d37525085f3978"
DATA_SHA256 = "471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7"
MANIFEST_SHA256 = "ec96c0b63eb0dadb6132e12961c2efee85a11b9df1e92ca7a29a9933d560008d"
RUNNER_SHA256 = "466dd3608434f2244abd8861d55a554fee6d4c34aa2d17319423187acaf966a1"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
EXPECTED_GPU_UUID = "GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48"
PPO_CLIP = 0.20
EXPECTED_PARTITIONS = {
    "development": (0, 48),
    "confirmation": (48, 200),
    "reserve": (248, 120),
}
EXPECTED_CONTRAST_FAMILIES = {
    "raw_E3_minus_E0": FAMILIES,
    "raw_E1_minus_E0": FAMILIES,
    "composite_E4_minus_E0": FAMILIES,
    "raw_best_of_n_E3_minus_k1": ("best_of_n",),
}
FINAL_LIKE_PATTERN = re.compile(r"\bfinal\b", re.IGNORECASE)
STRICT_FINAL_MARKER_PATTERN = re.compile(r"\bfinal\s*:", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_finalization(
    examples: list[dict[str, Any]], *, max_steps: int
) -> dict[str, Any]:
    """Audit what the frozen `answered` field actually measures."""
    answered = 0
    correct = 0
    unanswered_no_final_like = 0
    unanswered_noncanonical_final_like = 0
    all_step_counts: list[int] = []
    answered_step_counts: list[int] = []
    unanswered_step_counts: list[int] = []
    for row in examples:
        result = row.get("result", {})
        is_answered = bool(result.get("answered"))
        is_correct = bool(result.get("correct"))
        require(
            not is_correct or is_answered,
            "correct evaluation example is not marked answered",
        )
        answered += int(is_answered)
        correct += int(is_correct)
        steps = row.get("steps")
        require(
            isinstance(steps, list) and 1 <= len(steps) <= max_steps,
            "evaluation example has invalid step count",
        )
        all_step_counts.append(len(steps))
        if is_answered:
            require(
                STRICT_FINAL_MARKER_PATTERN.search(str(steps[-1])) is not None,
                "answered evaluation example lacks a strict FINAL marker in its last step",
            )
            answered_step_counts.append(len(steps))
            continue
        unanswered_step_counts.append(len(steps))
        require(
            len(steps) == max_steps,
            "unanswered evaluation example did not exhaust max_steps",
        )
        transcript = "\n".join(str(step) for step in steps)
        require(
            STRICT_FINAL_MARKER_PATTERN.search(transcript) is None,
            "unanswered evaluation example contains a strict FINAL marker",
        )
        if FINAL_LIKE_PATTERN.search(transcript) is None:
            unanswered_no_final_like += 1
        else:
            unanswered_noncanonical_final_like += 1
    unanswered = len(examples) - answered
    require(
        unanswered_no_final_like + unanswered_noncanonical_final_like == unanswered,
        "unanswered finalization classification does not close",
    )
    return {
        "parseable_final_answers": answered,
        "unanswered_examples": unanswered,
        "all_unanswered_exhaust_max_steps": True,
        "unanswered_no_final_like_marker": unanswered_no_final_like,
        "unanswered_noncanonical_final_like_marker": (
            unanswered_noncanonical_final_like
        ),
        "accuracy_given_parseable_final": (
            correct / answered if answered else None
        ),
        "mean_steps": sum(all_step_counts) / len(all_step_counts),
        "mean_steps_answered": (
            sum(answered_step_counts) / len(answered_step_counts)
            if answered_step_counts
            else None
        ),
        "mean_steps_unanswered": (
            sum(unanswered_step_counts) / len(unanswered_step_counts)
            if unanswered_step_counts
            else None
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_review_evidence_sha256(
    compute_rows: list[dict[str, Any]],
    session_audit: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "compute_by_family_estimator": compute_rows,
            "independent_run_session_audit": session_audit,
        }
    )


def audit_manual_compute_review(
    path: Path,
    analysis_path: Path,
    report_path: Path,
    compute_rows: list[dict[str, Any]],
    session_audit: dict[str, Any],
    recorder_path: Path,
    preparer_path: Path,
) -> dict[str, Any]:
    require(path.is_file(), "manual compute-review receipt missing")
    payload = read_json(path)
    require(
        payload.get("schema")
        == "cgcs-search-policy-mismatch-s3-manual-compute-review-v1",
        "manual compute-review schema mismatch",
    )
    require(payload.get("status") == "reviewed", "manual compute review incomplete")
    require(
        payload.get("verdict") in {"PASS", "FAIL"},
        "manual compute-review verdict must be PASS or FAIL",
    )
    rationale = payload.get("rationale")
    require(
        isinstance(rationale, str) and len(rationale.strip()) >= 20,
        "manual compute-review rationale is missing or too short",
    )
    require(
        payload.get("recorder_sha256") == sha256_file(recorder_path),
        "manual compute-review recorder SHA-256 mismatch",
    )
    require(
        payload.get("preparer_sha256") == sha256_file(preparer_path),
        "manual compute-review preparer SHA-256 mismatch",
    )
    require(
        payload.get("analysis_sha256") == sha256_file(analysis_path),
        "manual compute-review analysis SHA-256 mismatch",
    )
    require(
        payload.get("report_sha256") == sha256_file(report_path),
        "manual compute-review report SHA-256 mismatch",
    )
    evidence_sha256 = compute_review_evidence_sha256(compute_rows, session_audit)
    require(
        payload.get("compute_evidence_sha256") == evidence_sha256,
        "manual compute-review evidence SHA-256 mismatch",
    )
    return {
        "required": True,
        "verdict": payload["verdict"],
        "rationale": rationale.strip(),
        "recorder_sha256": payload["recorder_sha256"],
        "preparer_sha256": payload["preparer_sha256"],
        "analysis_sha256": payload["analysis_sha256"],
        "report_sha256": payload["report_sha256"],
        "compute_evidence_sha256": evidence_sha256,
        "receipt_sha256": sha256_file(path),
    }


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit_remote_postrun_tests(root: Path) -> dict[str, Any]:
    marker = root / "REMOTE_POSTRUN_TESTS_COMPLETE"
    log = root / "REMOTE_POSTRUN_TESTS.log"
    junit = root / "REMOTE_POSTRUN_TESTS.xml"
    require(marker.is_file(), "remote post-run Torch completion marker missing")
    require(log.is_file(), "remote post-run Torch test log missing")
    require(junit.is_file(), "remote post-run Torch JUnit receipt missing")
    try:
        junit_root = ET.parse(junit).getroot()
    except ET.ParseError as error:
        raise ValueError("invalid remote post-run Torch JUnit receipt") from error
    suites = (
        [junit_root]
        if junit_root.tag.endswith("testsuite")
        else list(junit_root.findall(".//testsuite"))
    )
    require(bool(suites), "remote post-run Torch JUnit receipt has no test suites")
    totals = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    require(totals["tests"] > 0, "remote post-run Torch receipt has no tests")
    require(totals["failures"] == 0, "remote post-run Torch tests failed")
    require(totals["errors"] == 0, "remote post-run Torch tests errored")
    require(totals["skipped"] == 0, "remote post-run Torch tests were skipped")
    return {
        **totals,
        "log_sha256": sha256_file(log),
        "junit_sha256": sha256_file(junit),
    }


def ppo_clip_classification(ratio: float, actor_loss: float) -> dict[str, bool]:
    """Separate ratio support violations from PPO actor-branch saturation.

    The logged actor loss has the opposite sign of the normalized advantage
    because all frozen external weights are positive.  This is enough to tell
    whether the clipped surrogate, rather than the unclipped corrective branch,
    was active without reconstructing the unlogged old Value prediction.
    """
    require(finite_number(ratio) and ratio > 0, "invalid PPO sequence ratio")
    require(finite_number(actor_loss), "invalid PPO actor loss")
    lower = ratio < 1.0 - PPO_CLIP
    upper = ratio > 1.0 + PPO_CLIP
    zero_advantage = math.isclose(actor_loss, 0.0, abs_tol=1e-12, rel_tol=0.0)
    saturated = (upper and actor_loss < 0.0) or (lower and actor_loss > 0.0)
    return {
        "lower_outside": lower,
        "upper_outside": upper,
        "outside": lower or upper,
        "actor_surrogate_saturated": saturated,
        "outside_with_corrective_actor_gradient": (
            (lower or upper) and not saturated and not zero_advantage
        ),
        "zero_advantage": zero_advantage,
    }


def value_diagnostics(
    states: list[dict[str, Any]], value_key: str
) -> dict[str, Any]:
    """Compute descriptive calibration/ranking metrics without changing gates."""
    values = [float(state[value_key]) for state in states]
    labels = [float(state["reward"]) for state in states]
    require(
        all(finite_number(value) for value in values + labels),
        f"non-finite Value diagnostic for {value_key}",
    )
    require(
        all(label in (0.0, 1.0) for label in labels),
        f"non-binary reward in Value diagnostic for {value_key}",
    )
    clipped = [min(1.0, max(0.0, value)) for value in values]
    bin_counts = [0] * 10
    bin_predictions = [0.0] * 10
    bin_labels = [0.0] * 10
    for prediction, label in zip(clipped, labels, strict=True):
        index = min(9, int(prediction * 10))
        bin_counts[index] += 1
        bin_predictions[index] += prediction
        bin_labels[index] += label
    ece10 = 0.0
    for count, prediction_sum, label_sum in zip(
        bin_counts, bin_predictions, bin_labels, strict=True
    ):
        if count:
            ece10 += count / len(values) * abs(
                prediction_sum / count - label_sum / count
            )
    positives = [value for value, label in zip(values, labels, strict=True) if label == 1.0]
    negatives = [value for value, label in zip(values, labels, strict=True) if label == 0.0]
    auc = None
    if positives and negatives:
        favorable = sum(
            1.0 if positive > negative else 0.5 if positive == negative else 0.0
            for positive in positives
            for negative in negatives
        )
        auc = favorable / (len(positives) * len(negatives))
    return {
        "states": len(states),
        "mean_value": sum(values) / len(values),
        "mean_reward": sum(labels) / len(labels),
        "minimum_value": min(values),
        "maximum_value": max(values),
        "out_of_unit_fraction": sum(
            value < 0.0 or value > 1.0 for value in values
        )
        / len(values),
        "raw_brier_score": sum(
            (value - label) ** 2 for value, label in zip(values, labels, strict=True)
        )
        / len(values),
        "clipped_ece10": ece10,
        "roc_auc": auc,
        "positive_states": len(positives),
        "negative_states": len(negatives),
        "interpretation": (
            "Descriptive only: the scalar Value head is trained by MSE without a "
            "sigmoid constraint. Brier uses the raw prediction; ECE clips to [0,1]; "
            "AUC measures ranking. These are state-weighted metrics: states from "
            "one problem are correlated and longer trajectories contribute more, "
            "so they are not independent-sample inference or promotion gates."
        ),
    }


def token_prefix_diversity(states: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe sampled token-prefix divergence without claiming semantics."""
    pair_count = 0
    exact_pairs = 0
    common_prefix_tokens = 0
    normalized_common_prefix = 0.0
    for state in states:
        if not state.get("expanded"):
            continue
        candidates = state.get("candidates", [])
        if len(candidates) < 2:
            continue
        sequences: list[list[int]] = []
        for candidate in candidates:
            sequence = candidate.get("completion_ids")
            require(
                isinstance(sequence, list)
                and bool(sequence)
                and all(isinstance(token, int) for token in sequence),
                "expanded candidate completion_ids missing",
            )
            sequences.append(sequence)
        for left_index, left in enumerate(sequences):
            for right in sequences[left_index + 1 :]:
                limit = min(len(left), len(right))
                prefix = 0
                while prefix < limit and left[prefix] == right[prefix]:
                    prefix += 1
                pair_count += 1
                exact_pairs += int(left == right)
                common_prefix_tokens += prefix
                normalized_common_prefix += prefix / max(len(left), len(right))
    if pair_count == 0:
        return {
            "candidate_pairs": 0,
            "exact_token_pair_fraction": None,
            "mean_pairwise_common_prefix_tokens": None,
            "mean_pairwise_common_prefix_fraction": None,
            "mean_pairwise_prefix_divergence_fraction": None,
        }
    mean_normalized_prefix = normalized_common_prefix / pair_count
    return {
        "candidate_pairs": pair_count,
        "exact_token_pair_fraction": exact_pairs / pair_count,
        "mean_pairwise_common_prefix_tokens": common_prefix_tokens / pair_count,
        "mean_pairwise_common_prefix_fraction": mean_normalized_prefix,
        "mean_pairwise_prefix_divergence_fraction": 1.0 - mean_normalized_prefix,
    }


def validate_candidate_payloads(candidates: list[Any], context: str) -> None:
    """Fail closed on candidate records used by propensity and diversity audits."""
    require(bool(candidates), f"{context}: empty candidate set")
    for candidate in candidates:
        require(
            isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
            f"{context}: candidate text missing",
        )
        completion_ids = candidate.get("completion_ids")
        old_logprobs = candidate.get("old_token_logprobs")
        require(
            isinstance(completion_ids, list)
            and bool(completion_ids)
            and all(isinstance(token, int) for token in completion_ids),
            f"{context}: invalid candidate completion IDs",
        )
        require(
            isinstance(old_logprobs, list)
            and len(old_logprobs) == len(completion_ids)
            and all(finite_number(value) for value in old_logprobs),
            f"{context}: candidate token log-prob mismatch",
        )
        require(
            finite_number(candidate.get("mean_token_entropy"))
            and float(candidate["mean_token_entropy"]) >= 0.0,
            f"{context}: invalid candidate entropy",
        )
        require(
            finite_number(candidate.get("post_value")),
            f"{context}: invalid candidate post-Value",
        )


def audit_paired_contrasts(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate and expose the frozen analyzer's per-seed paired evidence."""
    contrasts = analysis.get("contrasts")
    require(isinstance(contrasts, dict), "analysis paired contrasts missing")
    require(
        set(contrasts) == set(EXPECTED_CONTRAST_FAMILIES),
        "analysis paired contrast set mismatch",
    )
    audited: list[dict[str, Any]] = []
    for name, expected_families in EXPECTED_CONTRAST_FAMILIES.items():
        rows = contrasts[name]
        require(isinstance(rows, list), f"{name}: contrast rows are not a list")
        require(
            [row.get("family") for row in rows] == list(expected_families),
            f"{name}: family order or membership mismatch",
        )
        order = sorted(
            range(len(rows)),
            key=lambda index: float(
                rows[index]["paired_cluster_randomization_p_two_sided"]
            ),
        )
        expected_holm: dict[int, float] = {}
        running = 0.0
        for rank, index in enumerate(order):
            raw_p = float(rows[index]["paired_cluster_randomization_p_two_sided"])
            require(0.0 <= raw_p <= 1.0, f"{name}: invalid paired p-value")
            running = max(running, min(1.0, (len(rows) - rank) * raw_p))
            expected_holm[index] = running
        for index, row in enumerate(rows):
            family = str(row["family"])
            require(int(row.get("draws", -1)) == 10_000, f"{name}/{family}: wrong draw count")
            estimate = float(row["estimate"])
            ci_low = float(row["ci95_low"])
            ci_high = float(row["ci95_high"])
            holm_p = float(row["holm_adjusted_p"])
            require(
                all(finite_number(value) for value in (estimate, ci_low, ci_high, holm_p)),
                f"{name}/{family}: non-finite paired statistic",
            )
            require(ci_low <= ci_high, f"{name}/{family}: invalid confidence interval")
            require(
                math.isclose(holm_p, expected_holm[index], abs_tol=1e-15, rel_tol=0.0),
                f"{name}/{family}: Holm adjustment mismatch",
            )
            per_seed = row.get("per_seed")
            require(isinstance(per_seed, list), f"{name}/{family}: per-seed rows missing")
            require(
                [int(seed_row.get("seed", -1)) for seed_row in per_seed] == list(SEEDS),
                f"{name}/{family}: per-seed identity/order mismatch",
            )
            pooled = {"wins": 0, "losses": 0, "ties": 0}
            audited_per_seed = []
            for seed_row in per_seed:
                wins = int(seed_row["wins"])
                losses = int(seed_row["losses"])
                ties = int(seed_row["ties"])
                difference = float(seed_row["difference"])
                require(
                    min(wins, losses, ties) >= 0 and wins + losses + ties == 200,
                    f"{name}/{family}: invalid per-seed wins/losses/ties",
                )
                require(
                    math.isclose(difference, (wins - losses) / 200, abs_tol=1e-15),
                    f"{name}/{family}: per-seed difference disagrees with W/L/T",
                )
                for key, value in (("wins", wins), ("losses", losses), ("ties", ties)):
                    pooled[key] += value
                audited_per_seed.append(
                    {
                        "seed": int(seed_row["seed"]),
                        "difference": difference,
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                    }
                )
            require(
                row.get("pooled_wins_losses_ties") == pooled,
                f"{name}/{family}: pooled W/L/T mismatch",
            )
            require(
                math.isclose(estimate, (pooled["wins"] - pooled["losses"]) / 600, abs_tol=1e-15),
                f"{name}/{family}: pooled estimate disagrees with W/L/T",
            )
            audited.append(
                {
                    "contrast": name,
                    "family": family,
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "holm_adjusted_p": holm_p,
                    "per_seed": audited_per_seed,
                    "pooled_wins_losses_ties": pooled,
                }
            )
    require(len(audited) == 10, f"expected 10 paired contrasts, found {len(audited)}")
    return audited


def audit_contrast_metric_decomposition(
    paired_contrasts: list[dict[str, Any]],
    evaluation_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decompose registered accuracy contrasts without changing their estimand."""
    keyed = {
        (row["mode"], int(row["seed"]), row["family"], row["estimator"]): row
        for row in evaluation_metrics
    }
    require(
        len(keyed) == len(evaluation_metrics),
        "duplicate per-seed evaluation metrics",
    )
    decomposed: list[dict[str, Any]] = []
    for contrast in paired_contrasts:
        name = str(contrast["contrast"])
        family = str(contrast["family"])
        if name == "raw_E3_minus_E0":
            mode = "raw_k1"
            lhs_family, lhs_estimator = family, ESTIMATORS[2]
            rhs_family, rhs_estimator = family, ESTIMATORS[0]
        elif name == "raw_E1_minus_E0":
            mode = "raw_k1"
            lhs_family, lhs_estimator = family, ESTIMATORS[1]
            rhs_family, rhs_estimator = family, ESTIMATORS[0]
        elif name == "composite_E4_minus_E0":
            mode = "composite"
            lhs_family, lhs_estimator = family, ESTIMATORS[3]
            rhs_family, rhs_estimator = family, ESTIMATORS[0]
        elif name == "raw_best_of_n_E3_minus_k1":
            mode = "raw_k1"
            lhs_family, lhs_estimator = "best_of_n", ESTIMATORS[2]
            rhs_family, rhs_estimator = "k1", ESTIMATORS[0]
        else:
            raise ValueError(f"unsupported registered contrast: {name}")
        for paired_seed in contrast["per_seed"]:
            seed = int(paired_seed["seed"])
            lhs_key = (mode, seed, lhs_family, lhs_estimator)
            rhs_key = (mode, seed, rhs_family, rhs_estimator)
            require(lhs_key in keyed, f"{name}/{family}/{seed}: missing lhs metric")
            require(rhs_key in keyed, f"{name}/{family}/{seed}: missing rhs metric")
            lhs = keyed[lhs_key]
            rhs = keyed[rhs_key]
            accuracy_difference = float(lhs["accuracy"]) - float(rhs["accuracy"])
            require(
                math.isclose(
                    accuracy_difference,
                    float(paired_seed["difference"]),
                    abs_tol=1e-15,
                    rel_tol=0.0,
                ),
                f"{name}/{family}/{seed}: evaluation and paired accuracy disagree",
            )
            lhs_conditional = lhs["finalization_diagnostics"][
                "accuracy_given_parseable_final"
            ]
            rhs_conditional = rhs["finalization_diagnostics"][
                "accuracy_given_parseable_final"
            ]
            conditional_difference = (
                None
                if lhs_conditional is None or rhs_conditional is None
                else float(lhs_conditional) - float(rhs_conditional)
            )
            decomposed.append(
                {
                    "contrast": name,
                    "family": family,
                    "seed": seed,
                    "lhs": {
                        "mode": mode,
                        "family": lhs_family,
                        "estimator": lhs_estimator,
                    },
                    "rhs": {
                        "mode": mode,
                        "family": rhs_family,
                        "estimator": rhs_estimator,
                    },
                    "accuracy_difference": accuracy_difference,
                    "answer_rate_difference": (
                        float(lhs["answered_fraction"])
                        - float(rhs["answered_fraction"])
                    ),
                    "lhs_accuracy_given_parseable_final": lhs_conditional,
                    "rhs_accuracy_given_parseable_final": rhs_conditional,
                    "conditional_accuracy_difference": conditional_difference,
                }
            )
    require(
        len(decomposed) == 30,
        f"expected 30 per-seed contrast decompositions, found {len(decomposed)}",
    )
    return decomposed


def audit_training_rollouts(root: Path) -> list[dict[str, Any]]:
    """Independently recompute every training-rollout mechanism summary."""
    expected: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = root / f"s{seed}"
        expected.append(
            {
                "path": seed_root / "burnin/rollout.json",
                "seed": seed,
                "round": 0,
                "family": "burnin",
                "estimator": "shared",
                "arm": "burnin",
                "offset": 6500,
                "prior": None,
                "candidates": 4,
                "target_expansion_rate": 0.5,
                "threshold_source": None,
            }
        )
        for round_id, offset in ((1, 6548), (2, 6596)):
            prior = (
                f"s{seed}/burnin/train"
                if round_id == 1
                else f"s{seed}/round1/k1/{ESTIMATORS[0]}/train"
            )
            expected.append(
                {
                    "path": seed_root / f"round{round_id}/k1/{ESTIMATORS[0]}/rollout.json",
                    "seed": seed,
                    "round": round_id,
                    "family": "k1",
                    "estimator": ESTIMATORS[0],
                    "arm": "k1",
                    "offset": offset,
                    "prior": prior,
                    "candidates": 4,
                    "target_expansion_rate": 0.5,
                    "threshold_source": None,
                }
            )
        for family in FAMILIES:
            candidates = 2 if family == "best_of_n" else 4
            target_expansion = 1.0 if family == "best_of_n" else 1 / 3
            expected.append(
                {
                    "path": seed_root / f"round1/{family}/shared_rollout.json",
                    "seed": seed,
                    "round": 1,
                    "family": family,
                    "estimator": "shared",
                    "arm": family,
                    "offset": 6548,
                    "prior": f"s{seed}/burnin/train",
                    "candidates": candidates,
                    "target_expansion_rate": target_expansion,
                    "threshold_source": (
                        None
                        if family == "best_of_n"
                        else f"s{seed}/burnin/rollout.json"
                    ),
                }
            )
            for estimator in ESTIMATORS:
                expected.append(
                    {
                        "path": seed_root / f"round2/{family}/{estimator}/rollout.json",
                        "seed": seed,
                        "round": 2,
                        "family": family,
                        "estimator": estimator,
                        "arm": family,
                        "offset": 6596,
                        "prior": f"s{seed}/round1/{family}/{estimator}/train",
                        "candidates": candidates,
                        "target_expansion_rate": target_expansion,
                        "threshold_source": (
                            None
                            if family == "best_of_n"
                            else f"s{seed}/round1/{family}/shared_rollout.json"
                        ),
                    }
                )
    require(len(expected) == 54, "internal expected-rollout matrix mismatch")
    actual = set(root.glob("s*/**/rollout.json")) | set(
        root.glob("s*/**/shared_rollout.json")
    )
    require(
        actual == {row["path"] for row in expected},
        "independent training-rollout path set mismatch",
    )

    audited: list[dict[str, Any]] = []
    for spec in expected:
        path = spec["path"]
        payload = read_json(path)
        config = payload.get("config", {})
        examples = payload.get("examples", [])
        summary = payload.get("summary", {})
        require(
            config.get("arm") == spec["arm"]
            and config.get("offset") == spec["offset"]
            and config.get("count") == 48
            and config.get("seed") == spec["seed"],
            f"{path}: training-rollout configuration drift",
        )
        require(
            config.get("model") == "Qwen/Qwen3-4B-Instruct-2507"
            and str(config.get("data", "")).replace("\\", "/").endswith(
                "data/gsm8k/train.jsonl"
            )
            and config.get("max_steps") == 7
            and config.get("step_tokens") == 80
            and config.get("max_prompt_tokens") == 384
            and config.get("head_tokens") == 128
            and config.get("sampling_temperature") == 1.3
            and config.get("beta") == 1.0
            and config.get("value_scale_floor") == 0.05,
            f"{path}: frozen model or generation configuration drift",
        )
        require(
            config.get("candidates") == spec["candidates"]
            and math.isclose(
                float(config.get("expansion_rate", float("nan"))),
                spec["target_expansion_rate"],
                abs_tol=1e-15,
                rel_tol=0.0,
            )
            and config.get("epsilon") == (1.0 if spec["family"] in {"burnin", "k1"} else 0.10),
            f"{path}: frozen search-budget configuration drift",
        )
        output_suffix = str(path.relative_to(root)).replace("\\", "/")
        require(
            str(config.get("output", "")).replace("\\", "/").endswith(output_suffix),
            f"{path}: output provenance mismatch",
        )
        prior = spec["prior"]
        if prior is None:
            require(
                config.get("actor_adapter") is None and config.get("value_head") is None,
                f"{path}: burn-in must start from the base model",
            )
        else:
            actor_suffix = f"{prior}/adapter"
            value_suffix = f"{prior}/value_head.pt"
            require(
                str(config.get("actor_adapter", "")).replace("\\", "/").endswith(actor_suffix)
                and str(config.get("value_head", "")).replace("\\", "/").endswith(value_suffix),
                f"{path}: Actor/Value checkpoint lineage mismatch",
            )
        threshold_source = spec["threshold_source"]
        if threshold_source is None:
            require(
                config.get("threshold_source") is None
                and config.get("entropy_threshold") is None
                and config.get("value_threshold") is None,
                f"{path}: unexpected gate threshold configuration",
            )
        else:
            require(
                str(config.get("threshold_source", "")).replace("\\", "/").endswith(
                    threshold_source
                ),
                f"{path}: threshold-source lineage mismatch",
            )
            if spec["family"] == "entropy_value":
                require(
                    finite_number(config.get("entropy_threshold"))
                    and config.get("value_threshold") is None,
                    f"{path}: entropy gate threshold mismatch",
                )
            else:
                require(
                    finite_number(config.get("value_threshold"))
                    and config.get("entropy_threshold") is None,
                    f"{path}: Value gate threshold mismatch",
                )
        require(
            len(examples) == 48 and summary.get("examples") == 48,
            f"{path}: incomplete training rollout",
        )
        require(
            [row.get("example_id") for row in examples]
            == list(range(spec["offset"], spec["offset"] + 48)),
            f"{path}: training-rollout ID order mismatch",
        )
        correctness: list[float] = []
        answered: list[float] = []
        states: list[dict[str, Any]] = []
        for example in examples:
            result = example.get("result", {})
            require(
                result.get("correct") in (True, False, 0, 1)
                and result.get("answered") in (True, False, 0, 1),
                f"{path}: invalid training-rollout outcome",
            )
            correctness.append(float(result["correct"]))
            answered.append(float(result["answered"]))
            states.extend(example.get("states", []))
        require(states, f"{path}: training rollout has no states")
        expanded_count = 0
        generated_candidates = 0
        unique_counts: list[int] = []
        for state in states:
            require(
                finite_number(state.get("pre_value"))
                and finite_number(state.get("candidate_zero_post_value"))
                and state.get("reward") in (0.0, 1.0, 0, 1),
                f"{path}: invalid Value/reward diagnostic fields",
            )
            candidates = state.get("candidates", [])
            probabilities = state.get("selector_probabilities", [])
            choice = state.get("choice")
            validate_candidate_payloads(candidates, str(path))
            require(
                candidates and len(candidates) == len(probabilities),
                f"{path}: candidate metadata mismatch",
            )
            require(
                isinstance(choice, int) and 0 <= choice < len(candidates),
                f"{path}: invalid candidate choice",
            )
            require(
                all(finite_number(value) and float(value) > 0 for value in probabilities)
                and math.isclose(sum(map(float, probabilities)), 1.0, abs_tol=1e-8),
                f"{path}: invalid selector support",
            )
            require(
                finite_number(state.get("selection_probability"))
                and math.isclose(
                    float(state["selection_probability"]),
                    float(probabilities[choice]),
                    abs_tol=1e-12,
                    rel_tol=0.0,
                ),
                f"{path}: selected propensity mismatch",
            )
            generated_candidates += len(candidates)
            if state.get("expanded"):
                expanded_count += 1
                require(
                    all(isinstance(candidate.get("text"), str) for candidate in candidates),
                    f"{path}: expanded candidate text missing",
                )
                unique_counts.append(len({candidate["text"] for candidate in candidates}))
        accuracy = sum(correctness) / len(correctness)
        answer_rate = sum(answered) / len(answered)
        expansion_rate = expanded_count / len(states)
        mean_unique = sum(unique_counts) / len(unique_counts) if unique_counts else None
        identical_fraction = (
            sum(count == 1 for count in unique_counts) / len(unique_counts)
            if unique_counts
            else None
        )
        prefix_diversity = token_prefix_diversity(states)
        expected_summary = {
            "accuracy": accuracy,
            "answered_fraction": answer_rate,
            "states": len(states),
            "expansion_rate": expansion_rate,
            "generated_candidates": generated_candidates,
            "mean_unique_candidates_when_expanded": mean_unique,
            "identical_expanded_fraction": identical_fraction,
        }
        for key, value in expected_summary.items():
            observed = summary.get(key)
            if value is None:
                require(observed is None, f"{path}: {key} must be null without expansion")
            elif isinstance(value, float):
                require(
                    finite_number(observed)
                    and math.isclose(float(observed), value, abs_tol=1e-12, rel_tol=0.0),
                    f"{path}: {key} summary mismatch",
                )
            else:
                require(observed == value, f"{path}: {key} summary mismatch")
        audited.append(
            {
                "seed": spec["seed"],
                "round": spec["round"],
                "family": spec["family"],
                "estimator": spec["estimator"],
                **expected_summary,
                **prefix_diversity,
                "pre_value_diagnostics": value_diagnostics(states, "pre_value"),
                "candidate_zero_post_value_diagnostics": value_diagnostics(
                    states, "candidate_zero_post_value"
                ),
            }
        )
    return audited


def audit_run_sessions(
    root: Path, frozen_gpu_audit: dict[str, Any]
) -> dict[str, Any]:
    session_path = root / "run_sessions.tsv"
    require(session_path.is_file(), "missing run session audit")
    active: dict[int, int] = {}
    sessions: list[dict[str, int]] = []
    for line_number, line in enumerate(session_path.read_text().splitlines(), 1):
        fields = line.split("\t")
        require(
            fields and fields[0] in {"START", "END"},
            f"invalid session row {line_number}",
        )
        if fields[0] == "START":
            require(len(fields) == 3, f"invalid session start row {line_number}")
            pid, start = int(fields[1]), int(fields[2])
            require(pid not in active, f"duplicate active session PID {pid}")
            active[pid] = start
            continue
        require(len(fields) == 4, f"invalid session end row {line_number}")
        pid, end, status = int(fields[1]), int(fields[2]), int(fields[3])
        require(pid in active, f"unpaired session end for PID {pid}")
        start = active.pop(pid)
        require(end >= start, f"negative session duration for PID {pid}")
        require(status == 0, f"failed run session for PID {pid}: status {status}")
        sessions.append(
            {"pid": pid, "start": start, "end": end, "status": status}
        )
    require(not active, "unfinished run session(s) in independent audit")
    require(sessions, "no completed run sessions")

    expected_logs = {
        f"gpu_usage_{session['start']}_{session['pid']}.csv" for session in sessions
    }
    actual_logs = {path.name for path in root.glob("gpu_usage_*.csv")}
    require(
        actual_logs == expected_logs,
        "GPU usage logs do not map one-to-one to completed run sessions",
    )
    sample_count = 0
    utilization: list[float] = []
    memory: list[float] = []
    for name in sorted(expected_logs):
        path = root / name
        rows = path.read_text().splitlines()
        require(rows, f"empty GPU usage log: {path}")
        for row_number, line in enumerate(rows, 1):
            fields = [field.strip() for field in line.split(",")]
            require(len(fields) >= 6, f"{path}:{row_number}: malformed GPU row")
            require(
                fields[1] == EXPECTED_GPU_UUID,
                f"{path}:{row_number}: unexpected GPU UUID {fields[1]}",
            )
            require(
                finite_number(fields[2]) and finite_number(fields[3]),
                f"{path}:{row_number}: non-finite utilization or memory",
            )
            utilization.append(float(fields[2]))
            memory.append(float(fields[3]))
            sample_count += 1

    reserved_seconds = sum(session["end"] - session["start"] for session in sessions)
    reserved_hours = reserved_seconds / 3600.0
    require(
        int(frozen_gpu_audit.get("successful_session_count", -1)) == len(sessions),
        "frozen analyzer and independent audit disagree on session count",
    )
    require(
        int(frozen_gpu_audit.get("usage_sample_count", -1)) == sample_count,
        "frozen analyzer and independent audit disagree on GPU sample count",
    )
    require(
        math.isclose(
            float(frozen_gpu_audit.get("gpu_reserved_hours", float("nan"))),
            reserved_hours,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "frozen analyzer and independent audit disagree on reserved GPU-hours",
    )
    return {
        "expected_gpu_uuid": EXPECTED_GPU_UUID,
        "sessions": sessions,
        "successful_session_count": len(sessions),
        "multiple_run_sessions": len(sessions) > 1,
        "all_status_zero": True,
        "gpu_usage_files": sorted(expected_logs),
        "usage_sample_count": sample_count,
        "total_reserved_seconds": reserved_seconds,
        "gpu_reserved_hours": reserved_hours,
        "peak_memory_mib": max(memory),
        "mean_sampled_utilization_percent": sum(utilization) / sample_count,
        "matches_frozen_analyzer": True,
    }


def load_search_summaries(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for round_id in (1, 2):
            for family in FAMILIES:
                for estimator in ESTIMATORS:
                    path = (
                        root
                        / f"s{seed}/round{round_id}/{family}/{estimator}/train/summary.json"
                    )
                    payload = read_json(path)
                    require(
                        payload.get("schema")
                        == "cgcs-search-policy-mismatch-llm-train-v1",
                        f"{path}: schema mismatch",
                    )
                    require(
                        payload.get("config", {}).get("estimator") == estimator,
                        f"{path}: estimator mismatch",
                    )
                    quantiles = payload.get(QUANTILE_KEY)
                    require(
                        isinstance(quantiles, list)
                        and len(quantiles) == 6
                        and all(finite_number(value) for value in quantiles),
                        f"{path}: invalid weight quantiles",
                    )
                    require(
                        all(
                            float(left) <= float(right)
                            for left, right in zip(quantiles, quantiles[1:])
                        ),
                        f"{path}: non-monotone weight quantiles",
                    )
                    row = {
                        "seed": seed,
                        "round": round_id,
                        "family": family,
                        "estimator": estimator,
                        "states": int(payload["states"]),
                        "optimizer_updates": int(payload["optimizer_updates"]),
                        "ess": float(payload["external_weight_ess"]),
                        "maximum_weight": float(payload["external_weight_maximum"]),
                        "weight_quantiles_0_50_90_95_99_100": [
                            float(value) for value in quantiles
                        ],
                        "clip_fraction": float(payload["sequence_ratio_clip_fraction"]),
                        "kl_proxy": float(payload["mean_selected_kl_proxy"]),
                        "candidate_forwards": int(payload["candidate_forwards"]),
                        "candidate_tokens": int(payload["candidate_tokens"]),
                        "selected_tokens": int(payload["selected_tokens"]),
                        "elapsed_seconds": float(payload["elapsed_seconds"]),
                    }
                    require(
                        row["candidate_forwards"] >= 0,
                        f"{path}: negative candidate forwards",
                    )
                    require(
                        all(
                            finite_number(row[key])
                            for key in (
                                "ess",
                                "maximum_weight",
                                "clip_fraction",
                                "kl_proxy",
                                "elapsed_seconds",
                            )
                        ),
                        f"{path}: non-finite diagnostic",
                    )
                    rows.append(row)
    require(len(rows) == 72, f"expected 72 search summaries, found {len(rows)}")
    return rows


def aggregate_compute(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["family"], row["estimator"])].append(row)
    output: list[dict[str, Any]] = []
    for family in FAMILIES:
        baseline = grouped[(family, ESTIMATORS[0])]
        baseline_candidate_forwards = sum(
            row["candidate_forwards"] for row in baseline
        )
        baseline_candidate_tokens = sum(row["candidate_tokens"] for row in baseline)
        baseline_elapsed = sum(row["elapsed_seconds"] for row in baseline)
        require(
            baseline_candidate_forwards > 0,
            f"{family}: zero E0 candidate forwards",
        )
        require(baseline_candidate_tokens > 0, f"{family}: zero E0 candidate tokens")
        require(baseline_elapsed > 0, f"{family}: zero E0 elapsed time")
        for estimator in ESTIMATORS:
            values = grouped[(family, estimator)]
            require(len(values) == 6, f"{family}/{estimator}: expected six summaries")
            candidate_forwards = sum(row["candidate_forwards"] for row in values)
            candidate_tokens = sum(row["candidate_tokens"] for row in values)
            elapsed = sum(row["elapsed_seconds"] for row in values)
            output.append(
                {
                    "family": family,
                    "estimator": estimator,
                    "states": sum(row["states"] for row in values),
                    "optimizer_updates": sum(
                        row["optimizer_updates"] for row in values
                    ),
                    "candidate_forwards": candidate_forwards,
                    "candidate_tokens": candidate_tokens,
                    "selected_tokens": sum(row["selected_tokens"] for row in values),
                    "summed_invocation_elapsed_seconds": elapsed,
                    "candidate_token_ratio_to_same_family_E0": (
                        candidate_tokens / baseline_candidate_tokens
                    ),
                    "candidate_forward_ratio_to_same_family_E0": (
                        candidate_forwards / baseline_candidate_forwards
                    ),
                    "elapsed_ratio_to_same_family_E0": elapsed / baseline_elapsed,
                    "minimum_ess": min(row["ess"] for row in values),
                    "maximum_external_weight": max(
                        row["maximum_weight"] for row in values
                    ),
                    "maximum_clip_fraction": max(
                        row["clip_fraction"] for row in values
                    ),
                    "maximum_kl_proxy": max(row["kl_proxy"] for row in values),
                }
            )
    return output


def audit_stability_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_violations = [row for row in rows if row["clip_fraction"] >= 0.20]
    kl_violations = [row for row in rows if row["kl_proxy"] >= 0.02]
    ess_comparisons = []
    keyed = {
        (row["seed"], row["round"], row["family"], row["estimator"]): row
        for row in rows
    }
    for seed in SEEDS:
        for round_id in (1, 2):
            for family in FAMILIES:
                e1 = keyed[
                    (seed, round_id, family, "E1_trajectory_propensity")
                ]["ess"]
                e3 = keyed[
                    (seed, round_id, family, "E3_step_cumulative_lambda_0.75")
                ]["ess"]
                ess_comparisons.append(
                    {
                        "seed": seed,
                        "round": round_id,
                        "family": family,
                        "e1_ess": e1,
                        "e3_ess": e3,
                        "e3_exceeds_e1": e3 > e1,
                    }
                )
    return {
        "clip_threshold_strictly_below": 0.20,
        "clip_violation_count": len(clip_violations),
        "all_clip_fractions_below_0_20": not clip_violations,
        "maximum_clip_fraction": max(row["clip_fraction"] for row in rows),
        "kl_threshold_strictly_below": 0.02,
        "kl_violation_count": len(kl_violations),
        "all_mean_kl_proxies_below_0_02": not kl_violations,
        "maximum_kl_proxy": max(row["kl_proxy"] for row in rows),
        "e3_vs_e1_ess_comparisons": ess_comparisons,
        "e3_ess_exceeds_e1_in_all_18_comparisons": all(
            row["e3_exceeds_e1"] for row in ess_comparisons
        ),
    }


def audit_frozen_gate_logic(
    analysis: dict[str, Any],
    stability: dict[str, Any],
    analyzer_source: Path,
) -> dict[str, Any]:
    safety = analysis.get("completeness_safety")
    gate = analysis.get("positive_transfer_gate")
    require(isinstance(safety, dict), "frozen analysis safety block missing")
    require(isinstance(gate, dict), "frozen analysis promotion gate missing")
    require(
        safety.get("support_violations") == 0,
        "frozen analysis reports selector support violations",
    )
    require(
        safety.get("all_clip_fractions_below_0_20")
        == stability["all_clip_fractions_below_0_20"],
        "frozen and independent clip gates disagree",
    )
    require(
        safety.get("all_mean_kl_proxies_below_0_02")
        == stability["all_mean_kl_proxies_below_0_02"],
        "frozen and independent KL gates disagree",
    )
    require(
        gate.get("E3_ESS_above_E1_every_seed_family_round")
        == stability["e3_ess_exceeds_e1_in_all_18_comparisons"],
        "frozen and independent ESS gates disagree",
    )
    require(
        isinstance(safety.get("no_raw_answer_rate_regression_over_2pp"), bool),
        "frozen answer-rate gate missing",
    )
    corrected_stability_gate = bool(safety.get("finite_and_complete")) and all(
        (
            safety["support_violations"] == 0,
            bool(safety["all_clip_fractions_below_0_20"]),
            bool(safety["all_mean_kl_proxies_below_0_02"]),
            bool(safety["no_raw_answer_rate_regression_over_2pp"]),
        )
    )
    corrected_positive_transfer = all(
        (
            bool(gate.get("raw_E3_family_gate")),
            bool(gate.get("E3_ESS_above_E1_every_seed_family_round")),
            bool(gate.get("composite_E4_family_gate")),
            corrected_stability_gate,
        )
    )
    source_text = analyzer_source.read_text()
    truthiness_expression = "all(completeness_safety.values())"
    require(
        truthiness_expression in source_text,
        "expected frozen analyzer stability expression not found",
    )
    frozen_stability_gate = bool(gate.get("stability_gate"))
    frozen_overall_gate = bool(gate.get("passed"))
    return {
        "support_violations": 0,
        "frozen_analyzer_expression": truthiness_expression,
        "zero_support_truthiness_defect_present": True,
        "frozen_analyzer_stability_gate": frozen_stability_gate,
        "independent_corrected_stability_gate": corrected_stability_gate,
        "frozen_analyzer_overall_gate": frozen_overall_gate,
        "independent_corrected_overall_gate": corrected_positive_transfer,
        "overall_verdict_invariant_to_defect": (
            frozen_overall_gate == corrected_positive_transfer
        ),
        "answer_rate_gate": safety["no_raw_answer_rate_regression_over_2pp"],
        "interpretation": (
            "The frozen analyzer treats integer zero support violations as false "
            "inside all(completeness_safety.values()). The independent audit "
            "recomputes the intended condition as support_violations == 0 without "
            "modifying the frozen analyzer."
        ),
    }
def audit_within_epoch_clip_dynamics(root: Path) -> dict[str, Any]:
    grouped: dict[
        str, dict[str, list[tuple[int, float, float, float, float]]]
    ] = {
        estimator: {"first_block": [], "later_blocks": []}
        for estimator in ESTIMATORS
    }
    stratified: dict[
        tuple[int, str, str, str],
        list[tuple[int, float, float, float, float]],
    ] = defaultdict(list)
    log_count = 0
    for seed in SEEDS:
        for round_id in (1, 2):
            for family in FAMILIES:
                for estimator in ESTIMATORS:
                    directory = (
                        root
                        / f"s{seed}/round{round_id}/{family}/{estimator}/train"
                    )
                    summary = read_json(directory / "summary.json")
                    log_path = directory / "train_log.jsonl"
                    require(log_path.is_file(), f"missing training log: {log_path}")
                    records = [
                        json.loads(line)
                        for line in log_path.read_text().splitlines()
                        if line
                    ]
                    require(
                        len(records) == int(summary["states"]),
                        f"{log_path}: state/log count mismatch",
                    )
                    require(
                        [int(row["step"]) for row in records]
                        == list(range(1, len(records) + 1)),
                        f"{log_path}: non-contiguous steps",
                    )
                    for row in records:
                        clipped = int(row["sequence_ratio_clipped"])
                        ratio = float(row["sequence_ratio"])
                        log_ratio = float(row["sequence_log_ratio"])
                        kl_proxy = float(row["selected_kl_proxy"])
                        actor_loss = float(row["actor_loss"])
                        outside_weight = float(row["outside_weight"])
                        require(clipped in (0, 1), f"{log_path}: invalid clip flag")
                        require(finite_number(ratio) and ratio > 0, f"{log_path}: invalid ratio")
                        require(finite_number(log_ratio), f"{log_path}: invalid log ratio")
                        require(
                            finite_number(actor_loss),
                            f"{log_path}: invalid actor loss",
                        )
                        require(
                            finite_number(outside_weight) and outside_weight > 0,
                            f"{log_path}: non-positive outside weight",
                        )
                        require(
                            finite_number(kl_proxy) and kl_proxy >= 0,
                            f"{log_path}: invalid KL proxy",
                        )
                        classification = ppo_clip_classification(ratio, actor_loss)
                        require(
                            clipped == int(classification["outside"]),
                            f"{log_path}: clip flag disagrees with sequence ratio",
                        )
                        phase = (
                            "first_block" if int(row["step"]) <= 32 else "later_blocks"
                        )
                        diagnostic = (
                            clipped,
                            ratio,
                            log_ratio,
                            kl_proxy,
                            actor_loss,
                        )
                        grouped[estimator][phase].append(diagnostic)
                        stratified[(round_id, family, estimator, phase)].append(
                            diagnostic
                        )
                    observed_clip_fraction = sum(
                        int(row["sequence_ratio_clipped"]) for row in records
                    ) / len(records)
                    require(
                        math.isclose(
                            observed_clip_fraction,
                            float(summary["sequence_ratio_clip_fraction"]),
                            abs_tol=1e-12,
                            rel_tol=0.0,
                        ),
                        f"{log_path}: log/summary clip fraction mismatch",
                    )
                    log_count += 1
    require(log_count == 72, f"expected 72 search training logs, found {log_count}")

    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def summarize(
        values: list[tuple[int, float, float, float, float]]
    ) -> dict[str, Any]:
        require(bool(values), "empty within-epoch diagnostic phase")
        log_ratios = [value[2] for value in values]
        absolute_log_ratios = [abs(value) for value in log_ratios]
        classifications = [
            ppo_clip_classification(value[1], value[4]) for value in values
        ]
        return {
            "states": len(values),
            "clip_fraction": sum(value[0] for value in values) / len(values),
            "lower_outside_fraction": sum(
                int(value["lower_outside"]) for value in classifications
            )
            / len(values),
            "upper_outside_fraction": sum(
                int(value["upper_outside"]) for value in classifications
            )
            / len(values),
            "actor_surrogate_saturation_fraction": sum(
                int(value["actor_surrogate_saturated"])
                for value in classifications
            )
            / len(values),
            "outside_with_corrective_actor_gradient_fraction": sum(
                int(value["outside_with_corrective_actor_gradient"])
                for value in classifications
            )
            / len(values),
            "zero_advantage_fraction": sum(
                int(value["zero_advantage"]) for value in classifications
            )
            / len(values),
            "mean_sequence_ratio": sum(value[1] for value in values) / len(values),
            "minimum_sequence_ratio": min(value[1] for value in values),
            "maximum_sequence_ratio": max(value[1] for value in values),
            "mean_selected_kl_proxy": sum(value[3] for value in values) / len(values),
            "log_ratio_quantiles_0_01_05_50_95_99_100": [
                quantile(log_ratios, probability)
                for probability in (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)
            ],
            "absolute_log_ratio_quantiles_50_90_95_99_100": [
                quantile(absolute_log_ratios, probability)
                for probability in (0.5, 0.9, 0.95, 0.99, 1.0)
            ],
        }

    per_estimator = {
        estimator: {
            phase: summarize(values)
            for phase, values in phases.items()
        }
        for estimator, phases in grouped.items()
    }
    per_round_family_estimator = []
    for round_id in (1, 2):
        for family in FAMILIES:
            for estimator in ESTIMATORS:
                first = summarize(
                    stratified[(round_id, family, estimator, "first_block")]
                )
                later = summarize(
                    stratified[(round_id, family, estimator, "later_blocks")]
                )
                per_round_family_estimator.append(
                    {
                        "round": round_id,
                        "family": family,
                        "estimator": estimator,
                        "first_block": first,
                        "later_blocks": later,
                    }
                )
    e4_later_drift_comparisons = []
    for round_id in (1, 2):
        for family in FAMILIES:
            cell = [
                row
                for row in per_round_family_estimator
                if row["round"] == round_id and row["family"] == family
            ]
            e4 = next(
                row
                for row in cell
                if row["estimator"] == "E4_extended_candidate_set"
            )["later_blocks"]
            others = [
                row["later_blocks"]
                for row in cell
                if row["estimator"] != "E4_extended_candidate_set"
            ]
            maximum_other_clip = max(row["clip_fraction"] for row in others)
            maximum_other_plateau = max(
                row["actor_surrogate_saturation_fraction"] for row in others
            )
            e4_later_drift_comparisons.append(
                {
                    "round": round_id,
                    "family": family,
                    "e4_clip_fraction": e4["clip_fraction"],
                    "maximum_other_clip_fraction": maximum_other_clip,
                    "e4_clip_margin_over_other_maximum": (
                        e4["clip_fraction"] - maximum_other_clip
                    ),
                    "e4_actor_plateau_fraction": e4[
                        "actor_surrogate_saturation_fraction"
                    ],
                    "maximum_other_actor_plateau_fraction": maximum_other_plateau,
                    "e4_actor_plateau_margin_over_other_maximum": (
                        e4["actor_surrogate_saturation_fraction"]
                        - maximum_other_plateau
                    ),
                }
            )
    all_first = [
        value
        for phases in grouped.values()
        for value in phases["first_block"]
    ]
    all_later = [
        value
        for phases in grouped.values()
        for value in phases["later_blocks"]
    ]
    return {
        "search_training_logs": log_count,
        "gradient_accumulation_block_size": 32,
        "first_block": summarize(all_first),
        "later_blocks": summarize(all_later),
        "first_block_all_ratios_one_within_1e_7": all(
            math.isclose(value[1], 1.0, abs_tol=1e-7, rel_tol=0.0)
            for value in all_first
        ),
        "per_estimator": per_estimator,
        "per_round_family_estimator": per_round_family_estimator,
        "e4_later_drift_comparisons": e4_later_drift_comparisons,
        "e4_highest_later_clip_cells": sum(
            row["e4_clip_margin_over_other_maximum"] > 0
            for row in e4_later_drift_comparisons
        ),
        "e4_highest_later_actor_plateau_cells": sum(
            row["e4_actor_plateau_margin_over_other_maximum"] > 0
            for row in e4_later_drift_comparisons
        ),
        "interpretation_boundary": (
            "This descriptive diagnostic distinguishes initial old-policy "
            "reconstruction from drift after within-epoch optimizer updates; "
            "it is not an additional preregistered promotion gate."
        ),
        "clip_fraction_semantics": (
            "sequence_ratio_clip_fraction is the fraction of state-action "
            "sequence ratios outside [0.8, 1.2]. It is a two-sided, "
            "sequence-level diagnostic, not a token-drop rate and not the "
            "fraction of examples with zero total gradient. PPO actor-branch "
            "saturation additionally depends on the advantage sign. The audit "
            "reports that plateau fraction separately, as well as ratios outside "
            "the band that retain a corrective Actor gradient; Value and KL "
            "terms can still contribute gradients in either case."
        ),
    }


def audit_data_seal(project_root: Path) -> dict[str, Any]:
    data_path = project_root / "data/math500_scalar.jsonl"
    manifest_path = project_root / "data/math500_scalar_manifest.json"
    runner_path = project_root / "scripts/run_search_policy_mismatch_s3_rsaturn.sh"
    require(sha256_file(data_path) == DATA_SHA256, "evaluation data SHA-256 mismatch")
    require(
        sha256_file(manifest_path) == MANIFEST_SHA256,
        "evaluation manifest SHA-256 mismatch",
    )
    manifest = read_json(manifest_path)
    require(manifest.get("revision") == MATH500_REVISION, "MATH-500 revision drift")
    require(manifest.get("output_sha256") == DATA_SHA256, "manifest data digest drift")
    require(manifest.get("examples") == 368, "prepared evaluation row count drift")
    observed_partitions = {
        name: (spec.get("offset"), spec.get("count"))
        for name, spec in manifest.get("partitions", {}).items()
    }
    require(
        observed_partitions == EXPECTED_PARTITIONS,
        "evaluation partition boundaries drift",
    )
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]
    require(len(rows) == 368, "prepared evaluation file must contain 368 rows")
    all_ids: list[str] = []
    for name, (offset, count) in EXPECTED_PARTITIONS.items():
        expected_ids = manifest["partitions"][name].get("unique_ids")
        require(
            isinstance(expected_ids, list) and len(expected_ids) == count,
            f"{name}: manifest unique-ID count drift",
        )
        selected = rows[offset : offset + count]
        observed_ids = [row.get("source", {}).get("unique_id") for row in selected]
        require(observed_ids == expected_ids, f"{name}: unique-ID order drift")
        require(
            all(row.get("source", {}).get("s3_partition") == name for row in selected),
            f"{name}: row partition labels drift",
        )
        all_ids.extend(observed_ids)
    require(len(set(all_ids)) == 368, "evaluation partitions contain duplicate IDs")

    runner_text = runner_path.read_text()
    for training_range in (
        "--arm burnin --offset 6500 --count 48",
        'offset=6548; else offset=6596',
    ):
        require(training_range in runner_text, "frozen GSM8K training range missing")
    require(
        "--offset 48 --count 200" in runner_text,
        "sealed confirmation range missing from runner",
    )
    require(
        "--offset 248" not in runner_text,
        "untouched reserve range appears in runner",
    )
    return {
        "dataset_revision": MATH500_REVISION,
        "prepared_rows": len(rows),
        "partitions": {
            name: {"offset": offset, "count": count}
            for name, (offset, count) in EXPECTED_PARTITIONS.items()
        },
        "all_unique_ids_distinct": True,
        "confirmation_range_present_in_runner": True,
        "reserve_range_absent_from_runner": True,
    }


def audit_artifact_completeness(root: Path, project_root: Path) -> dict[str, Any]:
    expected_training: dict[Path, str] = {}
    expected_raw: dict[Path, dict[str, Any]] = {}
    expected_composite: dict[Path, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_root = root / f"s{seed}"
        expected_training[seed_root / "burnin/train/summary.json"] = ESTIMATORS[0]
        for round_id in (1, 2):
            expected_training[
                seed_root
                / f"round{round_id}/k1/{ESTIMATORS[0]}/train/summary.json"
            ] = ESTIMATORS[0]
            for family in FAMILIES:
                for estimator in ESTIMATORS:
                    expected_training[
                        seed_root
                        / f"round{round_id}/{family}/{estimator}/train/summary.json"
                    ] = estimator
        k1_branch = (
            f"results/search_policy_mismatch/llm_s3/s{seed}/round2/"
            f"k1/{ESTIMATORS[0]}/train"
        )
        expected_raw[seed_root / f"eval/raw_k1/k1_{ESTIMATORS[0]}.json"] = {
            "composite": False,
            "family": None,
            "adapter_suffix": f"{k1_branch}/adapter",
            "value_head_suffix": None,
        }
        for family in FAMILIES:
            for estimator in ESTIMATORS:
                branch = (
                    f"results/search_policy_mismatch/llm_s3/s{seed}/round2/"
                    f"{family}/{estimator}/train"
                )
                expected_raw[
                    seed_root / f"eval/raw_k1/{family}_{estimator}.json"
                ] = {
                    "composite": False,
                    "family": None,
                    "adapter_suffix": f"{branch}/adapter",
                    "value_head_suffix": None,
                }
                expected_composite[
                    seed_root / f"eval/composite/{family}_{estimator}.json"
                ] = {
                    "composite": True,
                    "family": family,
                    "adapter_suffix": f"{branch}/adapter",
                    "value_head_suffix": f"{branch}/value_head.pt",
                    "threshold_rollout": (
                        seed_root / f"round2/{family}/{estimator}/rollout.json"
                    ),
                }

    observed_training = set(root.glob("s*/**/train/summary.json"))
    observed_raw = set(root.glob("s*/eval/raw_k1/*.json"))
    observed_composite = set(root.glob("s*/eval/composite/*.json"))
    require(
        observed_training == set(expected_training),
        "independent training artifact path set mismatch",
    )
    require(observed_raw == set(expected_raw), "independent raw-evaluation path set mismatch")
    require(
        observed_composite == set(expected_composite),
        "independent composite-evaluation path set mismatch",
    )

    for path, estimator in expected_training.items():
        payload = read_json(path)
        require(
            payload.get("schema") == "cgcs-search-policy-mismatch-llm-train-v1",
            f"{path}: training schema mismatch",
        )
        require(
            payload.get("config", {}).get("estimator") == estimator,
            f"{path}: training estimator mismatch",
        )
        require(
            int(payload.get("states", 0)) > 0
            and int(payload.get("optimizer_updates", 0)) > 0,
            f"{path}: empty training state or update count",
        )
        require(
            payload.get("controller_gradient_stopped") is True,
            f"{path}: controller gradient was not stopped",
        )
        require(
            finite_number(payload.get("mean_actor_loss"))
            and finite_number(payload.get("mean_value_loss")),
            f"{path}: non-finite training loss",
        )
        require(
            int(payload.get("candidate_forwards", 0)) >= int(payload["states"]),
            f"{path}: missing candidate forwards",
        )
        require(
            int(payload.get("candidate_tokens", 0))
            >= int(payload.get("selected_tokens", 0))
            > 0,
            f"{path}: invalid token accounting",
        )
        train_directory = path.parent
        require(
            (train_directory / "value_head.pt").is_file(),
            f"{train_directory}: missing Value head",
        )
        adapter = train_directory / "adapter"
        require(
            (adapter / "adapter_config.json").is_file(),
            f"{train_directory}: missing Actor adapter config",
        )
        require(
            bool(list(adapter.glob("adapter_model.*"))),
            f"{train_directory}: missing Actor adapter model",
        )
        train_log = train_directory / "train_log.jsonl"
        require(train_log.is_file(), f"{train_directory}: missing training log")
        log_rows = [line for line in train_log.read_text().splitlines() if line]
        require(
            len(log_rows) == int(payload["states"]),
            f"{train_log}: state/log count mismatch",
        )

    data_path = project_root / "data/math500_scalar.jsonl"
    all_rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]
    expected_rows = all_rows[48:248]
    expected_ids = list(range(48, 248))
    expected_questions = [row["question"] for row in expected_rows]
    expected_references = [
        row["answer"].split("####", 1)[-1].strip() for row in expected_rows
    ]

    evaluation_metrics: list[dict[str, Any]] = []
    for path, spec in {**expected_raw, **expected_composite}.items():
        payload = read_json(path)
        config = payload.get("config", {})
        examples = payload.get("examples", [])
        summary = payload.get("summary", {})
        require(
            config.get("offset") == 48
            and config.get("count") == 200
            and config.get("seed") == 100901,
            f"{path}: evaluation configuration drift",
        )
        require(
            config.get("model") == "Qwen/Qwen3-4B-Instruct-2507",
            f"{path}: evaluation model drift",
        )
        require(
            str(config.get("data", "")).replace("\\", "/").endswith(
                "data/math500_scalar.jsonl"
            ),
            f"{path}: evaluation data drift",
        )
        require(
            str(config.get("actor_adapter", "")).replace("\\", "/").endswith(
                spec["adapter_suffix"]
            ),
            f"{path}: Actor adapter provenance mismatch",
        )
        require(
            str(config.get("output", "")).replace("\\", "/").endswith(
                str(path.relative_to(root)).replace("\\", "/")
            ),
            f"{path}: evaluation output provenance mismatch",
        )
        require(
            config.get("max_steps") == 8
            and config.get("step_tokens") == 96
            and config.get("max_prompt_tokens") == 768
            and config.get("head_tokens") == 192,
            f"{path}: evaluation generation configuration drift",
        )
        require(
            len(examples) == 200 and summary.get("examples") == 200,
            f"{path}: incomplete evaluation",
        )
        require(
            [row.get("example_id") for row in examples] == expected_ids,
            f"{path}: evaluation ID order mismatch",
        )
        require(
            [row.get("question") for row in examples] == expected_questions,
            f"{path}: evaluation question mismatch",
        )
        require(
            [row.get("reference_answer") for row in examples]
            == expected_references,
            f"{path}: evaluation reference mismatch",
        )
        correctness: list[float] = []
        answered: list[float] = []
        for row in examples:
            result = row.get("result", {})
            require(
                result.get("correct") in (True, False, 0, 1),
                f"{path}: invalid correctness flag",
            )
            require(
                result.get("answered") in (True, False, 0, 1),
                f"{path}: invalid answered flag",
            )
            correctness.append(float(result["correct"]))
            answered.append(float(result["answered"]))
        require(
            math.isclose(
                float(summary.get("accuracy", float("nan"))),
                sum(correctness) / len(correctness),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{path}: accuracy summary mismatch",
        )
        require(
            math.isclose(
                float(summary.get("answered_fraction", float("nan"))),
                sum(answered) / len(answered),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"{path}: answer-rate summary mismatch",
        )
        finalization = audit_finalization(
            examples,
            max_steps=int(config["max_steps"]),
        )
        seed_part = next(part for part in path.parts if part.startswith("s99"))
        estimator = next(
            candidate
            for candidate in ESTIMATORS
            if path.stem == f"k1_{candidate}" or path.stem.endswith(f"_{candidate}")
        )
        family = (
            "k1"
            if path.stem == f"k1_{estimator}"
            else path.stem[: -(len(estimator) + 1)]
        )
        evaluation_metrics.append(
            {
                "mode": "composite" if spec["composite"] else "raw_k1",
                "seed": int(seed_part[1:]),
                "family": family,
                "estimator": estimator,
                "accuracy": sum(correctness) / len(correctness),
                "answered_fraction": sum(answered) / len(answered),
                "finalization_diagnostics": finalization,
                "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)),
            }
        )
        if spec["composite"]:
            require(
                config.get("arm") == spec["family"],
                f"{path}: composite arm mismatch",
            )
            require(
                config.get("sampling_temperature") == 1.3,
                f"{path}: composite temperature drift",
            )
            expected_candidates = 2 if spec["family"] == "best_of_n" else 4
            expected_expansion = 1.0 if spec["family"] == "best_of_n" else 1 / 3
            require(
                config.get("candidates") == expected_candidates
                and config.get("epsilon") == 0.10
                and config.get("beta") == 1.0
                and config.get("value_scale_floor") == 0.05
                and math.isclose(
                    float(config.get("expansion_rate", float("nan"))),
                    expected_expansion,
                    abs_tol=1e-15,
                    rel_tol=0.0,
                ),
                f"{path}: composite search-budget configuration drift",
            )
            require(
                str(config.get("value_head", "")).replace("\\", "/").endswith(
                    spec["value_head_suffix"]
                ),
                f"{path}: Value head provenance mismatch",
            )
            threshold_payload = read_json(spec["threshold_rollout"])
            threshold_config = threshold_payload["config"]
            require(
                config.get("threshold_source") is None,
                f"{path}: composite evaluation must use an explicit frozen threshold",
            )
            if spec["family"] == "best_of_n":
                require(
                    config.get("entropy_threshold") is None
                    and config.get("value_threshold") is None,
                    f"{path}: unexpected best-of-N threshold",
                )
            elif spec["family"] == "entropy_value":
                require(
                    finite_number(config.get("entropy_threshold"))
                    and math.isclose(
                        float(config["entropy_threshold"]),
                        float(threshold_config["entropy_threshold"]),
                        abs_tol=1e-15,
                        rel_tol=0.0,
                    )
                    and config.get("value_threshold") is None,
                    f"{path}: entropy threshold does not match round-two rollout",
                )
            else:
                require(
                    finite_number(config.get("value_threshold"))
                    and math.isclose(
                        float(config["value_threshold"]),
                        float(threshold_config["value_threshold"]),
                        abs_tol=1e-15,
                        rel_tol=0.0,
                    )
                    and config.get("entropy_threshold") is None,
                    f"{path}: Value threshold does not match round-two rollout",
                )
            states = [state for row in examples for state in row.get("states", [])]
            require(states, f"{path}: composite evaluation has no states")
            expanded = [state for state in states if state.get("expanded")]
            source_states = [
                state
                for row in threshold_payload.get("examples", [])
                for state in row.get("states", [])
            ]
            require(
                source_states,
                f"{spec['threshold_rollout']}: threshold-source rollout has no states",
            )
            source_expanded = [
                state for state in source_states if state.get("expanded")
            ]
            source_generated_candidates = sum(
                len(state.get("candidates", [])) for state in source_states
            )
            require(
                source_generated_candidates >= len(source_states),
                f"{spec['threshold_rollout']}: invalid source candidate count",
            )
            unique_counts: list[int] = []
            generated_candidates = 0
            candidate_completion_tokens = 0
            selected_completion_tokens = 0
            for state in states:
                candidates = state.get("candidates", [])
                probabilities = state.get("selector_probabilities", [])
                choice = state.get("choice")
                validate_candidate_payloads(candidates, str(path))
                require(
                    candidates and len(candidates) == len(probabilities),
                    f"{path}: candidate metadata mismatch",
                )
                require(
                    len(candidates)
                    == (expected_candidates if state.get("expanded") else 1),
                    f"{path}: candidate count disagrees with expansion decision",
                )
                require(
                    isinstance(choice, int) and 0 <= choice < len(candidates),
                    f"{path}: invalid candidate choice",
                )
                require(
                    all(finite_number(value) and float(value) > 0 for value in probabilities),
                    f"{path}: selector support violation",
                )
                require(
                    math.isclose(
                        sum(map(float, probabilities)),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    ),
                    f"{path}: selector probabilities do not sum to one",
                )
                generated_candidates += len(candidates)
                candidate_completion_tokens += sum(
                    len(candidate["completion_ids"]) for candidate in candidates
                )
                selected_completion_tokens += len(
                    candidates[choice]["completion_ids"]
                )
                if state.get("expanded"):
                    require(
                        all(
                            isinstance(candidate, dict)
                            and isinstance(candidate.get("text"), str)
                            for candidate in candidates
                        ),
                        f"{path}: candidate text missing",
                    )
                    unique_counts.append(
                        len({candidate["text"] for candidate in candidates})
                    )
            require(summary.get("states") == len(states), f"{path}: state count mismatch")
            require(
                summary.get("generated_candidates") == generated_candidates,
                f"{path}: generated-candidate summary mismatch",
            )
            require(
                math.isclose(
                    float(summary.get("expansion_rate", float("nan"))),
                    len(expanded) / len(states),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{path}: expansion-rate summary mismatch",
            )
            if unique_counts:
                require(
                    math.isclose(
                        float(
                            summary.get(
                                "mean_unique_candidates_when_expanded",
                                float("nan"),
                            )
                        ),
                        sum(unique_counts) / len(unique_counts),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    f"{path}: unique-candidate summary mismatch",
                )
                require(
                    math.isclose(
                        float(
                            summary.get(
                                "identical_expanded_fraction",
                                float("nan"),
                            )
                        ),
                        sum(count == 1 for count in unique_counts)
                        / len(unique_counts),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                    f"{path}: identical-candidate summary mismatch",
                )
            else:
                require(
                    summary.get("mean_unique_candidates_when_expanded") is None
                    and summary.get("identical_expanded_fraction") is None,
                    f"{path}: zero-expansion diversity summary mismatch",
                )
            prefix_diversity = token_prefix_diversity(states)
            realized_expansion = len(expanded) / len(states)
            source_rollout_expansion = len(source_expanded) / len(source_states)
            source_rollout_candidates_per_state = (
                source_generated_candidates / len(source_states)
            )
            evaluation_metrics[-1].update(
                {
                    "states": len(states),
                    "registered_expansion_rate": expected_expansion,
                    "source_rollout_states": len(source_states),
                    "source_rollout_expansion_rate": source_rollout_expansion,
                    "expansion_rate": realized_expansion,
                    "expansion_rate_deviation": (
                        realized_expansion - expected_expansion
                    ),
                    "evaluation_minus_source_rollout_expansion_rate": (
                        realized_expansion - source_rollout_expansion
                    ),
                    "generated_candidates": generated_candidates,
                    "registered_expected_candidates_per_state": (
                        1.0 + (expected_candidates - 1) * expected_expansion
                    ),
                    "source_rollout_candidates_per_state": (
                        source_rollout_candidates_per_state
                    ),
                    "realized_candidates_per_state": (
                        generated_candidates / len(states)
                    ),
                    "evaluation_minus_source_rollout_candidates_per_state": (
                        generated_candidates / len(states)
                        - source_rollout_candidates_per_state
                    ),
                    "candidate_completion_tokens": candidate_completion_tokens,
                    "selected_completion_tokens": selected_completion_tokens,
                    "candidate_completion_tokens_per_state": (
                        candidate_completion_tokens / len(states)
                    ),
                    "selected_completion_tokens_per_state": (
                        selected_completion_tokens / len(states)
                    ),
                    "candidate_token_multiplier_vs_selected": (
                        candidate_completion_tokens / selected_completion_tokens
                    ),
                    "mean_unique_candidates_when_expanded": (
                        sum(unique_counts) / len(unique_counts)
                        if unique_counts
                        else None
                    ),
                    "identical_expanded_fraction": (
                        sum(count == 1 for count in unique_counts)
                        / len(unique_counts)
                        if unique_counts
                        else None
                    ),
                    **prefix_diversity,
                }
            )
        else:
            require(
                config.get("arm") is None and config.get("value_head") is None,
                f"{path}: raw-K1 evaluation unexpectedly retained search state",
            )
            step_counts = [len(row.get("steps", [])) for row in examples]
            require(all(0 < count <= 8 for count in step_counts), f"{path}: invalid steps")
            require(
                math.isclose(
                    float(summary.get("mean_steps", float("nan"))),
                    sum(step_counts) / len(step_counts),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"{path}: mean-steps summary mismatch",
            )
        require(
            finite_number(payload.get("elapsed_seconds"))
            and float(payload["elapsed_seconds"]) >= 0,
            f"{path}: invalid elapsed time",
        )

    evaluation_metrics.sort(
        key=lambda row: (
            row["mode"], row["seed"], row["family"], row["estimator"]
        )
    )
    require(len(evaluation_metrics) == 75, "expected 75 independently audited evaluations")
    return {
        "training_summaries": len(expected_training),
        "burnin_training": len(SEEDS),
        "k1_adaptive_training": len(SEEDS) * 2,
        "search_adaptive_training": len(SEEDS) * 2 * len(FAMILIES) * len(ESTIMATORS),
        "raw_k1_evaluations": len(expected_raw),
        "composite_evaluations": len(expected_composite),
        "examples_per_evaluation": 200,
        "ordered_confirmation_example_ids": expected_ids,
        "all_training_branches_have_value_head_actor_adapter_and_log": True,
        "all_evaluations_match_frozen_questions_and_references": True,
        "all_evaluations_match_round2_branch_provenance": True,
        "all_evaluation_summaries_recompute_from_examples": True,
        "all_composite_selector_metadata_valid": True,
        "evaluation_metrics": evaluation_metrics,
    }


def build_archive_hashes(
    root: Path,
    project_root: Path,
    analysis_path: Path,
    report_path: Path,
    compute_review_path: Path | None = None,
) -> dict[str, str]:
    files = {
        "frozen_protocol": (
            project_root
            / "results/search_policy_mismatch/S3_LLM_TRANSFER_PROTOCOL.md"
        ),
        "evaluation_data": project_root / "data/math500_scalar.jsonl",
        "evaluation_manifest": project_root / "data/math500_scalar_manifest.json",
        "frozen_runner": (
            project_root / "scripts/run_search_policy_mismatch_s3_rsaturn.sh"
        ),
        "frozen_analysis_source": (
            project_root / "scripts/analyze_search_policy_mismatch_s3.py"
        ),
        "frozen_analysis_json": analysis_path,
        "frozen_report": report_path,
        "administrative_archive_audit_source": Path(__file__).resolve(),
        "manual_compute_review_recorder_source": (
            project_root
            / "scripts/record_search_policy_mismatch_s3_compute_review.py"
        ),
        "manual_compute_review_preparer_source": (
            project_root
            / "scripts/prepare_search_policy_mismatch_s3_compute_review.sh"
        ),
        "iclr_evidence_table_renderer_source": (
            project_root
            / "scripts/render_search_policy_mismatch_iclr_tables.py"
        ),
        "iclr_figure_renderer_source": (
            project_root
            / "scripts/render_search_policy_mismatch_iclr_figures.py"
        ),
        "iclr_terminal_results_renderer_source": (
            project_root
            / "scripts/render_search_policy_mismatch_iclr_results_tex.py"
        ),
        "iclr_manuscript_review_recorder_source": (
            project_root
            / "scripts/record_search_policy_mismatch_iclr_manuscript_review.py"
        ),
        "s3_final_archive_manifest_builder_source": (
            project_root
            / "scripts/build_search_policy_mismatch_s3_final_manifest.py"
        ),
        "remote_postrun_test_source": (
            project_root
            / "scripts/verify_search_policy_mismatch_s3_remote_postrun.sh"
        ),
        "remote_postrun_test_log": root / "REMOTE_POSTRUN_TESTS.log",
        "remote_postrun_test_junit": root / "REMOTE_POSTRUN_TESTS.xml",
    }
    if compute_review_path is not None:
        files["manual_compute_review_receipt"] = compute_review_path
    hashes = {label: sha256_file(path) for label, path in files.items()}
    require(
        hashes["frozen_protocol"] == PROTOCOL_SHA256,
        "frozen protocol SHA-256 mismatch",
    )
    require(
        hashes["evaluation_data"] == DATA_SHA256,
        "evaluation data SHA-256 mismatch",
    )
    require(
        hashes["evaluation_manifest"] == MANIFEST_SHA256,
        "evaluation manifest SHA-256 mismatch",
    )
    require(
        hashes["frozen_runner"] == RUNNER_SHA256,
        "frozen runner SHA-256 mismatch",
    )
    return hashes


def audit(
    root: Path,
    analysis_path: Path,
    report_path: Path,
    project_root: Path,
    compute_review_path: Path | None = None,
) -> dict[str, Any]:
    analysis = read_json(analysis_path)
    require(
        analysis.get("schema") == "cgcs-search-policy-mismatch-s3-analysis-v1",
        "analysis schema mismatch",
    )
    require(
        analysis.get("artifact_counts") == EXPECTED_COUNTS,
        "frozen artifact counts mismatch",
    )
    require(
        analysis.get("protocol_sha256") == PROTOCOL_SHA256,
        "analysis protocol SHA-256 mismatch",
    )
    require(
        analysis.get("data_sha256") == DATA_SHA256,
        "analysis data SHA-256 mismatch",
    )
    report_text = report_path.read_text()
    require(
        PROTOCOL_SHA256 in report_text and DATA_SHA256 in report_text,
        "report does not identify the frozen protocol and data",
    )
    require(
        isinstance(analysis.get("gpu_audit"), dict),
        "analysis GPU audit missing",
    )
    require((root / "COMPLETE").is_file(), "COMPLETE marker missing")
    require((root / "EVIDENCE_COMPLETE").is_file(), "EVIDENCE_COMPLETE marker missing")
    rows = load_search_summaries(root)
    stability = audit_stability_gates(rows)
    compute_rows = aggregate_compute(rows)
    session_audit = audit_run_sessions(root, analysis["gpu_audit"])
    completeness = audit_artifact_completeness(root, project_root)
    paired_contrasts = audit_paired_contrasts(analysis)
    contrast_decomposition = audit_contrast_metric_decomposition(
        paired_contrasts,
        completeness["evaluation_metrics"],
    )
    if compute_review_path is None:
        manual_gate = {
            "required": True,
            "reason": (
                "The frozen protocol intentionally defines compute competitiveness "
                "as an independently inspected condition, not a post-hoc numeric cutoff."
            ),
            "verdict": "UNSET",
        }
    else:
        manual_gate = audit_manual_compute_review(
            compute_review_path,
            analysis_path,
            report_path,
            compute_rows,
            session_audit,
            project_root
            / "scripts/record_search_policy_mismatch_s3_compute_review.py",
            project_root
            / "scripts/prepare_search_policy_mismatch_s3_compute_review.sh",
        )
    return {
        "schema": "cgcs-search-policy-mismatch-s3-archive-audit-v1",
        "status": "structurally_complete",
        "artifact_counts": EXPECTED_COUNTS,
        "archive_sha256": build_archive_hashes(
            root,
            project_root,
            analysis_path,
            report_path,
            compute_review_path,
        ),
        "remote_postrun_tests": audit_remote_postrun_tests(root),
        "data_seal_audit": audit_data_seal(project_root),
        "independent_artifact_completeness": completeness,
        "gpu_audit": analysis["gpu_audit"],
        "independent_run_session_audit": session_audit,
        "paired_contrast_audit": paired_contrasts,
        "contrast_metric_decomposition": contrast_decomposition,
        "training_rollout_audit": audit_training_rollouts(root),
        "per_summary_weight_and_stability": rows,
        "stability_gate_audit": stability,
        "frozen_gate_logic_audit": audit_frozen_gate_logic(
            analysis,
            stability,
            project_root / "scripts/analyze_search_policy_mismatch_s3.py",
        ),
        "within_epoch_clip_dynamics": audit_within_epoch_clip_dynamics(root),
        "compute_by_family_estimator": compute_rows,
        "manual_gate": manual_gate,
        "elapsed_time_caveat": (
            "Per-summary elapsed_seconds is invocation-local. The run-session/GPU "
            "audit is authoritative for total reserved GPU-hours. Multiple run sessions "
            "mean at least one runner-level resume occurred and artifact elapsed time must "
            "not be represented as exact end-to-end wall time."
        ),
    }


def render_audit_report(payload: dict[str, Any]) -> str:
    completeness = payload["independent_artifact_completeness"]
    stability = payload["stability_gate_audit"]
    sessions = payload["independent_run_session_audit"]
    remote_tests = payload["remote_postrun_tests"]
    dynamics = payload["within_epoch_clip_dynamics"]
    gate_logic = payload["frozen_gate_logic_audit"]
    lines = [
        "# S3 independent archive audit",
        "",
        "This administrative audit is independent of the frozen analyzer. "
        "Structural completeness does not imply that a promotion gate passed.",
        "",
        "## Artifact completeness",
        "",
        f"- Training summaries: {completeness['training_summaries']} "
        f"(burn-in {completeness['burnin_training']}, K1 "
        f"{completeness['k1_adaptive_training']}, search "
        f"{completeness['search_adaptive_training']}).",
        f"- Raw-K1 evaluations: {completeness['raw_k1_evaluations']}.",
        f"- Composite evaluations: {completeness['composite_evaluations']}.",
        f"- Examples per evaluation: {completeness['examples_per_evaluation']}.",
        "- Frozen question, reference, and order audit: PASS.",
        "- Actor adapter, Value head, and training-log audit: PASS.",
        "",
        "## Preregistered stability evidence",
        "",
        f"- Sequence-ratio clip violations (threshold < 0.20): "
        f"{stability['clip_violation_count']}/72; maximum "
        f"{stability['maximum_clip_fraction']:.6f}.",
        f"- Mean-KL violations (threshold < 0.02): "
        f"{stability['kl_violation_count']}/72; maximum "
        f"{stability['maximum_kl_proxy']:.6f}.",
        f"- E3 ESS exceeds E1 in all 18 comparisons: "
        f"{stability['e3_ess_exceeds_e1_in_all_18_comparisons']}.",
        f"- Raw answer-rate regression gate: {gate_logic['answer_rate_gate']}.",
        f"- Frozen analyzer stability gate: "
        f"{gate_logic['frozen_analyzer_stability_gate']}.",
        f"- Independently corrected stability gate: "
        f"{gate_logic['independent_corrected_stability_gate']}.",
        f"- Frozen analyzer overall gate: "
        f"{gate_logic['frozen_analyzer_overall_gate']}.",
        f"- Independently corrected overall gate: "
        f"{gate_logic['independent_corrected_overall_gate']}.",
        f"- Zero-support truthiness defect present in frozen analyzer: "
        f"{gate_logic['zero_support_truthiness_defect_present']}.",
        f"- Overall verdict invariant to that defect: "
        f"{gate_logic['overall_verdict_invariant_to_defect']}.",
        "",
        "## Within-epoch ratio diagnostic",
        "",
        f"- First accumulation block: {dynamics['first_block']['states']} states, "
        f"clip fraction {dynamics['first_block']['clip_fraction']:.6f}, Actor "
        f"surrogate saturation fraction "
        f"{dynamics['first_block']['actor_surrogate_saturation_fraction']:.6f}.",
        f"- Later blocks: {dynamics['later_blocks']['states']} states, clip "
        f"fraction {dynamics['later_blocks']['clip_fraction']:.6f}, Actor "
        f"surrogate saturation fraction "
        f"{dynamics['later_blocks']['actor_surrogate_saturation_fraction']:.6f}, "
        f"outside-band corrective-gradient fraction "
        f"{dynamics['later_blocks']['outside_with_corrective_actor_gradient_fraction']:.6f}.",
        f"- First-block ratios all reconstruct one within 1e-7: "
        f"{dynamics['first_block_all_ratios_one_within_1e_7']}.",
        f"- E4 has the largest later-block clip fraction in "
        f"{dynamics['e4_highest_later_clip_cells']}/6 round-by-family cells "
        f"and the largest Actor-platform fraction in "
        f"{dynamics['e4_highest_later_actor_plateau_cells']}/6 cells.",
        f"- Interpretation boundary: {dynamics['clip_fraction_semantics']}",
        "",
        "| Estimator | First clip | First plateau | Later clip | Later plateau | "
        "Later corrective outside | Later KL | Later abs-log-ratio p50 | p90 | "
        "p99 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for estimator in ESTIMATORS:
        first = dynamics["per_estimator"][estimator]["first_block"]
        later = dynamics["per_estimator"][estimator]["later_blocks"]
        absolute = later["absolute_log_ratio_quantiles_50_90_95_99_100"]
        lines.append(
            f"| {estimator} | {first['clip_fraction']:.6f} | "
            f"{first['actor_surrogate_saturation_fraction']:.6f} | "
            f"{later['clip_fraction']:.6f} | "
            f"{later['actor_surrogate_saturation_fraction']:.6f} | "
            f"{later['outside_with_corrective_actor_gradient_fraction']:.6f} | "
            f"{later['mean_selected_kl_proxy']:.6f} | {absolute[0]:.6f} | "
            f"{absolute[1]:.6f} | {absolute[3]:.6f} | {absolute[4]:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Later-block ratio diagnostics by round and search family",
            "",
            "| Round | Family | Estimator | States | Lower outside | Upper "
            "outside | All outside | Actor plateau | Corrective outside |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in dynamics["per_round_family_estimator"]:
        later = row["later_blocks"]
        lines.append(
            f"| {row['round']} | {row['family']} | {row['estimator']} | "
            f"{later['states']} | {later['lower_outside_fraction']:.6f} | "
            f"{later['upper_outside_fraction']:.6f} | "
            f"{later['clip_fraction']:.6f} | "
            f"{later['actor_surrogate_saturation_fraction']:.6f} | "
            f"{later['outside_with_corrective_actor_gradient_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed evaluation metrics",
            "",
            "| Mode | Seed | Family | Estimator | Accuracy | Answer rate |",
            "|---|---:|---|---|---:|---:|",
        ]
    )
    for row in completeness["evaluation_metrics"]:
        lines.append(
            f"| {row['mode']} | {row['seed']} | {row['family']} | "
            f"{row['estimator']} | {100 * row['accuracy']:.2f}% | "
            f"{100 * row['answered_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "### Valid-final and finalization diagnostics",
            "",
            "Answer rate is the fraction of examples with a scalar answer "
            "extracted from the frozen strict `FINAL:` protocol. It is not an "
            "explicit abstention probability. Every unanswered example must "
            "exhaust all frozen generation steps; the final two columns "
            "separate transcripts with no final-like wording from those with "
            "noncanonical final-like wording.",
            "",
            "| Mode | Seed | Family | Estimator | Valid finals | Unanswered | "
            "Accuracy given valid final | No final-like marker | Noncanonical "
            "final-like marker | Mean steps | Answered mean steps | Unanswered "
            "mean steps |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in completeness["evaluation_metrics"]:
        diagnostics = row["finalization_diagnostics"]
        conditional = diagnostics["accuracy_given_parseable_final"]
        conditional_text = (
            "n/a" if conditional is None else f"{100 * conditional:.2f}%"
        )
        answered_steps = diagnostics["mean_steps_answered"]
        unanswered_steps = diagnostics["mean_steps_unanswered"]
        lines.append(
            f"| {row['mode']} | {row['seed']} | {row['family']} | "
            f"{row['estimator']} | {diagnostics['parseable_final_answers']} | "
            f"{diagnostics['unanswered_examples']} | {conditional_text} | "
            f"{diagnostics['unanswered_no_final_like_marker']} | "
            f"{diagnostics['unanswered_noncanonical_final_like_marker']} | "
            f"{diagnostics['mean_steps']:.3f} | "
            f"{'n/a' if answered_steps is None else f'{answered_steps:.3f}'} | "
            f"{'n/a' if unanswered_steps is None else f'{unanswered_steps:.3f}'} |"
        )
    lines.extend(
        [
            "",
            "## Retained-composite expansion and diversity",
            "",
            "Exact-text uniqueness and pairwise token-prefix divergence measure "
            "surface-form branching only. They do not establish semantic or "
            "reasoning-path diversity because candidate-level parsed answers and "
            "counterfactual terminal rewards were not logged. Source rollout is "
            "the frozen round-two rollout that supplied the evaluation threshold; "
            "evaluation-minus-source shifts therefore isolate deployment-state "
            "shift from target-to-source threshold calibration error.",
            "",
            "| Seed | Family | Estimator | States | Target expansion | Source-rollout "
            "expansion | Evaluation expansion | Eval-source expansion shift | "
            "Target candidates/state | Source-rollout candidates/state | "
            "Evaluation candidates/state | Eval-source candidate shift | Candidates | Candidate tokens | Selected "
            "tokens | Candidate-token multiplier | Mean unique when expanded | "
            "Fully identical | Candidate pairs | Exact token pairs | Mean "
            "token-prefix divergence |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in completeness["evaluation_metrics"]:
        if row["mode"] != "composite":
            continue
        mean_unique = (
            "N/A"
            if row["mean_unique_candidates_when_expanded"] is None
            else f"{row['mean_unique_candidates_when_expanded']:.4f}"
        )
        identical_fraction = (
            "N/A"
            if row["identical_expanded_fraction"] is None
            else f"{100 * row['identical_expanded_fraction']:.2f}%"
        )
        exact_pair_fraction = (
            "N/A"
            if row["exact_token_pair_fraction"] is None
            else f"{100 * row['exact_token_pair_fraction']:.2f}%"
        )
        prefix_divergence = (
            "N/A"
            if row["mean_pairwise_prefix_divergence_fraction"] is None
            else f"{100 * row['mean_pairwise_prefix_divergence_fraction']:.2f}%"
        )
        lines.append(
            f"| {row['seed']} | {row['family']} | {row['estimator']} | "
            f"{row['states']} | {100 * row['registered_expansion_rate']:.2f}% | "
            f"{100 * row['source_rollout_expansion_rate']:.2f}% | "
            f"{100 * row['expansion_rate']:.2f}% | "
            f"{100 * row['evaluation_minus_source_rollout_expansion_rate']:+.2f} pp | "
            f"{row['registered_expected_candidates_per_state']:.4f} | "
            f"{row['source_rollout_candidates_per_state']:.4f} | "
            f"{row['realized_candidates_per_state']:.4f} | "
            f"{row['evaluation_minus_source_rollout_candidates_per_state']:+.4f} | "
            f"{row['generated_candidates']} | "
            f"{row['candidate_completion_tokens']} | "
            f"{row['selected_completion_tokens']} | "
            f"{row['candidate_token_multiplier_vs_selected']:.4f} | "
            f"{mean_unique} | "
            f"{identical_fraction} | "
            f"{row['candidate_pairs']} | "
            f"{exact_pair_fraction} | "
            f"{prefix_divergence} |"
        )
    lines.extend(
        [
            "",
            "## Training-rollout mechanism metrics",
            "",
            "| Seed | Round | Family | Estimator | Accuracy | Answer rate | "
            "States | Expansion | Candidates | Mean unique | Fully identical | "
            "Candidate pairs | Exact token pairs | Token-prefix divergence |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["training_rollout_audit"]:
        mean_unique = (
            "n/a"
            if row["mean_unique_candidates_when_expanded"] is None
            else f"{row['mean_unique_candidates_when_expanded']:.4f}"
        )
        identical = (
            "n/a"
            if row["identical_expanded_fraction"] is None
            else f"{100 * row['identical_expanded_fraction']:.2f}%"
        )
        exact_pairs = (
            "n/a"
            if row["exact_token_pair_fraction"] is None
            else f"{100 * row['exact_token_pair_fraction']:.2f}%"
        )
        prefix_divergence = (
            "n/a"
            if row["mean_pairwise_prefix_divergence_fraction"] is None
            else f"{100 * row['mean_pairwise_prefix_divergence_fraction']:.2f}%"
        )
        lines.append(
            f"| {row['seed']} | {row['round']} | {row['family']} | "
            f"{row['estimator']} | {100 * row['accuracy']:.2f}% | "
            f"{100 * row['answered_fraction']:.2f}% | {row['states']} | "
            f"{100 * row['expansion_rate']:.2f}% | "
            f"{row['generated_candidates']} | {mean_unique} | {identical} | "
            f"{row['candidate_pairs']} | {exact_pairs} | {prefix_divergence} |"
        )
    lines.extend(
        [
            "",
            "## Training-rollout Value calibration and ranking",
            "",
            "These are descriptive diagnostics, not additional promotion gates. "
            "Raw Brier uses the unconstrained scalar head; ECE10 clips predictions "
            "to `[0,1]`; AUC measures ranking. They are state-weighted, so correlated "
            "states and trajectory length preclude independent-sample inference.",
            "",
            "| Seed | Round | Family | Estimator | Mean reward | Pre mean | "
            "Pre Brier | Pre ECE10 | Pre AUC | Post Brier | Post ECE10 | "
            "Post AUC | Out-of-unit pre/post |",
            "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["training_rollout_audit"]:
        pre = row["pre_value_diagnostics"]
        post = row["candidate_zero_post_value_diagnostics"]
        pre_auc = "n/a" if pre["roc_auc"] is None else f"{pre['roc_auc']:.4f}"
        post_auc = "n/a" if post["roc_auc"] is None else f"{post['roc_auc']:.4f}"
        lines.append(
            f"| {row['seed']} | {row['round']} | {row['family']} | "
            f"{row['estimator']} | {pre['mean_reward']:.4f} | "
            f"{pre['mean_value']:.4f} | {pre['raw_brier_score']:.4f} | "
            f"{pre['clipped_ece10']:.4f} | {pre_auc} | "
            f"{post['raw_brier_score']:.4f} | {post['clipped_ece10']:.4f} | "
            f"{post_auc} | {pre['out_of_unit_fraction']:.4f}/"
            f"{post['out_of_unit_fraction']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Matched-compute evidence",
            "",
            "| Family | Estimator | States | Updates | Candidate forwards | "
            "Candidate tokens | Selected tokens | Forward ratio vs E0 | "
            "Token ratio vs E0 | Invocation-time ratio vs E0 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["compute_by_family_estimator"]:
        lines.append(
            f"| {row['family']} | {row['estimator']} | {row['states']} | "
            f"{row['optimizer_updates']} | {row['candidate_forwards']} | "
            f"{row['candidate_tokens']} | {row['selected_tokens']} | "
            f"{row['candidate_forward_ratio_to_same_family_E0']:.4f} | "
            f"{row['candidate_token_ratio_to_same_family_E0']:.4f} | "
            f"{row['elapsed_ratio_to_same_family_E0']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed paired evidence",
            "",
            "The frozen JSON stores the protocol-required per-seed paired "
            "differences even though its Markdown report shows pooled W/L/T only.",
            "The 10,000-resample clustered interval resamples the 200 problem "
            "columns while retaining all three seed observations for each problem. "
            "It therefore quantifies confirmation-problem uncertainty conditional "
            "on these three training seeds; it is not a precise estimate of "
            "training-seed superpopulation uncertainty.",
            "",
            "| Contrast | Family | Seed | Difference | W/L/T |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in payload["paired_contrast_audit"]:
        for seed_row in row["per_seed"]:
            lines.append(
                f"| {row['contrast']} | {row['family']} | {seed_row['seed']} | "
                f"{100 * seed_row['difference']:+.2f} pp | "
                f"{seed_row['wins']}/{seed_row['losses']}/{seed_row['ties']} |"
            )
    lines.extend(
        [
            "",
            "### Accuracy and finalization decomposition",
            "",
            "Each registered overall-accuracy contrast is decomposed into its "
            "answer-rate difference and descriptive correctness difference "
            "conditional on producing a parseable final answer. A difference "
            "driven by finalization remains an observed deployment-budget effect; "
            "it is not evidence of improved mathematical correctness conditional "
            "on answering. This diagnostic does not replace the registered overall-"
            "accuracy estimand.",
            "",
            "| Contrast | Family | Seed | Accuracy difference | Answer-rate "
            "difference | LHS conditional accuracy | RHS conditional accuracy | "
            "Conditional-accuracy difference |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["contrast_metric_decomposition"]:
        lhs_conditional = row["lhs_accuracy_given_parseable_final"]
        rhs_conditional = row["rhs_accuracy_given_parseable_final"]
        conditional_difference = row["conditional_accuracy_difference"]
        lines.append(
            f"| {row['contrast']} | {row['family']} | {row['seed']} | "
            f"{100 * row['accuracy_difference']:+.2f} pp | "
            f"{100 * row['answer_rate_difference']:+.2f} pp | "
            f"{'n/a' if lhs_conditional is None else f'{100 * lhs_conditional:.2f}%'} | "
            f"{'n/a' if rhs_conditional is None else f'{100 * rhs_conditional:.2f}%'} | "
            f"{'n/a' if conditional_difference is None else f'{100 * conditional_difference:+.2f} pp'} |"
        )
    lines.extend(
        [
            "",
            "## GPU and session audit",
            "",
            f"- Successful sessions: {sessions['successful_session_count']}.",
            f"- Multiple runner sessions: {sessions['multiple_run_sessions']}.",
            f"- Reserved GPU-hours: {sessions['gpu_reserved_hours']:.6f}.",
            f"- GPU UUID: `{sessions['expected_gpu_uuid']}`.",
            f"- GPU samples: {sessions['usage_sample_count']}.",
            "",
            "## Remote post-run Torch receipt",
            "",
            f"- Tests: {remote_tests['tests']}.",
            f"- Failures/errors/skips: {remote_tests['failures']}/"
            f"{remote_tests['errors']}/{remote_tests['skipped']}.",
            f"- Log SHA-256: `{remote_tests['log_sha256']}`.",
            f"- JUnit SHA-256: `{remote_tests['junit_sha256']}`.",
            "",
            "## Archive SHA-256 receipt",
            "",
            "This receipt is generated only after both terminal completion "
            "markers and the full artifact matrix pass the independent audit. "
            "The frozen analysis JSON contains the per-artifact reproducibility "
            "manifest; the table below binds its inputs, generator, outputs, "
            "and this administrative audit implementation.",
            "",
            "| Artifact role | SHA-256 |",
            "|---|---|",
        ]
    )
    for label, digest in sorted(payload["archive_sha256"].items()):
        lines.append(f"| `{label}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## Decision boundary",
            "",
            f"Manual compute-competitiveness review: "
            f"**{payload['manual_gate']['verdict']}**.",
            "A structurally complete archive with any failed frozen gate remains "
            "a mixed/negative S3 result and does not authorize a next-stage run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--compute-review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(
        args.root,
        args.analysis_json,
        args.report,
        args.project_root,
        args.compute_review,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(render_audit_report(payload))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "compute_rows": len(payload["compute_by_family_estimator"]),
                "manual_compute_gate": payload["manual_gate"]["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
