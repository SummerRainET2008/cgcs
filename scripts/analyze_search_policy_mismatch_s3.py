#!/usr/bin/env python3
"""Fail-closed validation and analysis for the frozen S3 LLM transfer.

The confirmation set is paired by problem and retained as a cluster across
training seeds.  This script will not emit a final report unless every planned
artifact is present and structurally valid.
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

import numpy as np


SEEDS = (99801, 99802, 99803)
FAMILIES = ("best_of_n", "entropy_value", "value_value")
ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)
DATA_SHA256 = "471d976a3bf547935eb67471ec5f754fb4bb2ffc1dd82e2ddc9711f68369aba7"
CONFIRMATION_OFFSET = 48
CONFIRMATION_COUNT = 200
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261028


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    require(bool(array.size), "cannot average an empty vector")
    return float(array.mean())


def _config_path_matches(value: Any, suffix: str) -> bool:
    return value is not None and str(value).replace("\\", "/").endswith(suffix)


def load_confirmation_rows(data_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    require(sha256_file(data_path) == DATA_SHA256, "MATH-500 scalar data digest mismatch")
    manifest = read_json(manifest_path)
    confirmation = manifest["partitions"]["confirmation"]
    require(confirmation["offset"] == CONFIRMATION_OFFSET, "confirmation offset drift")
    require(confirmation["count"] == CONFIRMATION_COUNT, "confirmation count drift")
    all_rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    rows = all_rows[CONFIRMATION_OFFSET : CONFIRMATION_OFFSET + CONFIRMATION_COUNT]
    observed_ids = [row["source"]["unique_id"] for row in rows]
    require(observed_ids == confirmation["unique_ids"], "confirmation unique-ID order drift")
    return rows


def validate_rollout(
    path: Path,
    *,
    arm: str,
    count: int,
    offset: int,
    seed: int,
    expected_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = read_json(path)
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    examples = payload.get("examples", [])
    require(config.get("arm") == arm, f"{path}: arm mismatch")
    require(config.get("count") == count and config.get("offset") == offset, f"{path}: range mismatch")
    require(config.get("seed") == seed, f"{path}: seed mismatch")
    require(len(examples) == count and summary.get("examples") == count, f"{path}: incomplete examples")
    require([row.get("example_id") for row in examples] == list(range(offset, offset + count)), f"{path}: ID order mismatch")
    if expected_rows is not None:
        require([row["question"] for row in examples] == [row["question"] for row in expected_rows], f"{path}: question mismatch")
        require([row["reference_answer"] for row in examples] == [row["answer"].split("####", 1)[-1].strip() for row in expected_rows], f"{path}: reference mismatch")
    flat = [state for row in examples for state in row.get("states", [])]
    require(flat, f"{path}: no states")
    for state in flat:
        candidates = state.get("candidates", [])
        probabilities = state.get("selector_probabilities", [])
        choice = state.get("choice")
        require(candidates and len(candidates) == len(probabilities), f"{path}: candidate metadata mismatch")
        require(isinstance(choice, int) and 0 <= choice < len(candidates), f"{path}: invalid choice")
        require(all(finite(x) and float(x) > 0 for x in probabilities), f"{path}: support violation")
        require(math.isclose(sum(map(float, probabilities)), 1.0, abs_tol=1e-8), f"{path}: selector probabilities do not sum to one")
    return payload


def validate_train(path: Path, estimator: str) -> dict[str, Any]:
    directory = path.parent
    payload = read_json(path)
    require(payload.get("schema") == "cgcs-search-policy-mismatch-llm-train-v1", f"{path}: schema mismatch")
    require(payload.get("controller_gradient_stopped") is True, f"{path}: controller gradient not stopped")
    require(payload.get("config", {}).get("estimator") == estimator, f"{path}: estimator mismatch")
    require(int(payload.get("states", 0)) > 0 and int(payload.get("optimizer_updates", 0)) > 0, f"{path}: empty training")
    for key in (
        "mean_actor_loss",
        "mean_value_loss",
        "mean_sequence_ratio",
        "sequence_ratio_clip_fraction",
        "mean_selected_kl_proxy",
        "external_weight_ess",
        "external_weight_maximum",
        "elapsed_seconds",
    ):
        require(finite(payload.get(key)), f"{path}: non-finite {key}")
    require(0 < float(payload["external_weight_ess"]) <= 1.0 + 1e-12, f"{path}: invalid ESS")
    require(float(payload["external_weight_maximum"]) > 0, f"{path}: invalid maximum weight")
    require(int(payload.get("candidate_forwards", 0)) >= int(payload["states"]), f"{path}: missing candidate forwards")
    require(int(payload.get("candidate_tokens", 0)) >= int(payload.get("selected_tokens", 0)) > 0, f"{path}: invalid token accounting")
    require((directory / "value_head.pt").is_file(), f"{directory}: missing Value head")
    require((directory / "adapter/adapter_config.json").is_file(), f"{directory}: missing adapter config")
    require(bool(list((directory / "adapter").glob("adapter_model.*"))), f"{directory}: missing adapter model")
    return payload


def validate_eval(path: Path, expected_rows: list[dict[str, Any]], *, composite: bool, arm: str | None = None) -> dict[str, Any]:
    payload = read_json(path)
    config = payload.get("config", {})
    examples = payload.get("examples", [])
    summary = payload.get("summary", {})
    require(config.get("offset") == CONFIRMATION_OFFSET and config.get("count") == CONFIRMATION_COUNT, f"{path}: evaluation range drift")
    require(config.get("seed") == 100901, f"{path}: evaluation seed drift")
    require(len(examples) == CONFIRMATION_COUNT and summary.get("examples") == CONFIRMATION_COUNT, f"{path}: incomplete evaluation")
    require([row.get("example_id") for row in examples] == list(range(CONFIRMATION_OFFSET, CONFIRMATION_OFFSET + CONFIRMATION_COUNT)), f"{path}: evaluation ID order mismatch")
    require([row["question"] for row in examples] == [row["question"] for row in expected_rows], f"{path}: evaluation question mismatch")
    expected_reference = [row["answer"].split("####", 1)[-1].strip() for row in expected_rows]
    require([row["reference_answer"] for row in examples] == expected_reference, f"{path}: evaluation reference mismatch")
    if composite:
        require(config.get("arm") == arm, f"{path}: composite arm mismatch")
        require(config.get("sampling_temperature") == 1.3, f"{path}: composite temperature drift")
    else:
        require(_config_path_matches(config.get("data"), "data/math500_scalar.jsonl"), f"{path}: raw evaluation data drift")
    for row in examples:
        result = row.get("result", {})
        require(result.get("correct") in (True, False, 0, 1), f"{path}: invalid correctness")
        require(result.get("answered") in (True, False, 0, 1), f"{path}: invalid answer flag")
    return payload


def outcome_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(row["result"][key]) for row in payload["examples"]], dtype=np.float64)


def clustered_bootstrap(matrix: np.ndarray, *, seed: int) -> dict[str, Any]:
    """Resample problem columns while retaining all seed observations."""
    require(matrix.shape == (len(SEEDS), CONFIRMATION_COUNT), "paired matrix shape mismatch")
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    block = 500
    for start in range(0, BOOTSTRAP_DRAWS, block):
        end = min(BOOTSTRAP_DRAWS, start + block)
        indices = rng.integers(0, CONFIRMATION_COUNT, size=(end - start, CONFIRMATION_COUNT))
        sampled = np.take(matrix, indices, axis=1)
        draws[start:end] = sampled.mean(axis=(0, 2))
    low, high = np.quantile(draws, [0.025, 0.975])
    # Problem-cluster sign flip gives a paired two-sided randomization p-value.
    cluster_means = matrix.mean(axis=0)
    permutations = rng.choice((-1.0, 1.0), size=(BOOTSTRAP_DRAWS, CONFIRMATION_COUNT))
    null = (permutations * cluster_means).mean(axis=1)
    observed = float(matrix.mean())
    p_value = float((1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (BOOTSTRAP_DRAWS + 1))
    return {
        "estimate": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_cluster_randomization_p_two_sided": p_value,
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["paired_cluster_randomization_p_two_sided"])
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * rows[index]["paired_cluster_randomization_p_two_sided"])
        running = max(running, adjusted)
        rows[index]["holm_adjusted_p"] = running


def paired_contrast(left: list[np.ndarray], right: list[np.ndarray], *, seed: int) -> dict[str, Any]:
    matrix = np.stack([a - b for a, b in zip(left, right, strict=True)])
    result = clustered_bootstrap(matrix, seed=seed)
    result["per_seed"] = []
    for training_seed, row in zip(SEEDS, matrix, strict=True):
        result["per_seed"].append(
            {
                "seed": training_seed,
                "difference": float(row.mean()),
                "wins": int(np.count_nonzero(row > 0)),
                "losses": int(np.count_nonzero(row < 0)),
                "ties": int(np.count_nonzero(row == 0)),
            }
        )
    result["pooled_wins_losses_ties"] = {
        "wins": int(np.count_nonzero(matrix > 0)),
        "losses": int(np.count_nonzero(matrix < 0)),
        "ties": int(np.count_nonzero(matrix == 0)),
    }
    return result


def parse_sessions(root: Path) -> dict[str, Any]:
    path = root / "run_sessions.tsv"
    require(path.is_file(), "missing run session audit")
    starts: dict[str, int] = {}
    completed: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        require(fields and fields[0] in {"START", "END"}, "invalid session audit row")
        if fields[0] == "START":
            require(len(fields) == 3 and fields[1] not in starts, "invalid or duplicate session start")
            starts[fields[1]] = int(fields[2])
        else:
            require(len(fields) == 4 and fields[1] in starts, "unpaired session end")
            start = starts.pop(fields[1])
            completed.append({"pid": int(fields[1]), "start": start, "end": int(fields[2]), "status": int(fields[3])})
    require(not starts, "unfinished audited run session")
    require(completed and all(row["status"] == 0 for row in completed), "one or more run sessions failed")
    seconds = sum(row["end"] - row["start"] for row in completed)
    usage_files = sorted(root.glob("gpu_usage_*.csv"))
    require(usage_files, "missing GPU usage audit")
    memory: list[float] = []
    utilization: list[float] = []
    uuids: set[str] = set()
    for usage in usage_files:
        for line in usage.read_text().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 5:
                uuids.add(fields[1])
                utilization.append(float(fields[2]))
                memory.append(float(fields[3]))
    require(uuids == {"GPU-a1d4c143-2850-4a5b-90d0-e8d8237fdc48"}, "GPU UUID audit mismatch")
    return {
        "sessions": completed,
        "successful_session_count": len(completed),
        "wall_hours": seconds / 3600.0,
        "gpu_reserved_hours": seconds / 3600.0,
        "peak_memory_mib": max(memory),
        "mean_sampled_utilization_percent": mean(utilization),
        "usage_sample_count": len(utilization),
    }


def build_reproducibility_manifest(root: Path, project_root: Path) -> dict[str, Any]:
    """Hash frozen sources and every evidence-bearing artifact."""
    files: dict[str, Path] = {
        "protocol": project_root / "results/search_policy_mismatch/S3_LLM_TRANSFER_PROTOCOL.md",
        "runner": project_root / "scripts/run_search_policy_mismatch_s3_rsaturn.sh",
        "analysis": Path(__file__).resolve(),
        "collector": project_root / "src/acbpo/math_joint_value_ppo.py",
        "estimators": project_root / "src/acbpo/math_search_policy_mismatch.py",
        "evaluation_data": project_root / "data/math500_scalar.jsonl",
        "evaluation_manifest": project_root / "data/math500_scalar_manifest.json",
        "session_audit": root / "run_sessions.tsv",
    }

    def add_train(label: str, directory: Path) -> None:
        files[f"{label}/summary.json"] = directory / "summary.json"
        files[f"{label}/train_log.jsonl"] = directory / "train_log.jsonl"
        files[f"{label}/value_head.pt"] = directory / "value_head.pt"
        files[f"{label}/adapter/adapter_config.json"] = directory / "adapter/adapter_config.json"
        models = sorted((directory / "adapter").glob("adapter_model.*"))
        require(len(models) == 1, f"{directory}: expected exactly one adapter model")
        files[f"{label}/adapter/{models[0].name}"] = models[0]

    for seed in SEEDS:
        seed_root = root / f"s{seed}"
        files[f"s{seed}/burnin/rollout.json"] = seed_root / "burnin/rollout.json"
        add_train(f"s{seed}/burnin/train", seed_root / "burnin/train")
        for round_id in (1, 2):
            k1 = seed_root / f"round{round_id}/k1/E0_standard_actor_ratio"
            files[f"s{seed}/round{round_id}/k1/rollout.json"] = k1 / "rollout.json"
            add_train(f"s{seed}/round{round_id}/k1/train", k1 / "train")
        for family in FAMILIES:
            files[f"s{seed}/round1/{family}/shared_rollout.json"] = seed_root / f"round1/{family}/shared_rollout.json"
            for estimator in ESTIMATORS:
                add_train(
                    f"s{seed}/round1/{family}/{estimator}/train",
                    seed_root / f"round1/{family}/{estimator}/train",
                )
                round2 = seed_root / f"round2/{family}/{estimator}"
                files[f"s{seed}/round2/{family}/{estimator}/rollout.json"] = round2 / "rollout.json"
                add_train(f"s{seed}/round2/{family}/{estimator}/train", round2 / "train")
                files[f"s{seed}/eval/raw_k1/{family}_{estimator}.json"] = seed_root / f"eval/raw_k1/{family}_{estimator}.json"
                files[f"s{seed}/eval/composite/{family}_{estimator}.json"] = seed_root / f"eval/composite/{family}_{estimator}.json"
        files[f"s{seed}/eval/raw_k1/k1_E0_standard_actor_ratio.json"] = seed_root / "eval/raw_k1/k1_E0_standard_actor_ratio.json"
    for usage in sorted(root.glob("gpu_usage_*.csv")):
        files[f"gpu_audit/{usage.name}"] = usage
    hashes = {label: sha256_file(path) for label, path in sorted(files.items())}
    return {
        "algorithm": "sha256",
        "file_count": len(hashes),
        "critical_source_labels": [
            "protocol",
            "runner",
            "analysis",
            "collector",
            "estimators",
            "evaluation_data",
            "evaluation_manifest",
        ],
        "files": hashes,
    }


def analyze(root: Path, project_root: Path) -> dict[str, Any]:
    protocol = project_root / "results/search_policy_mismatch/S3_LLM_TRANSFER_PROTOCOL.md"
    runner = project_root / "scripts/run_search_policy_mismatch_s3_rsaturn.sh"
    data = project_root / "data/math500_scalar.jsonl"
    manifest = project_root / "data/math500_scalar_manifest.json"
    protocol_text = protocol.read_text()
    require("Status: **frozen**" in protocol_text, "protocol is not frozen")
    protocol_sha = sha256_file(protocol)
    runner_text = runner.read_text()
    match = re.search(r"^expected_protocol_sha=([0-9a-f]{64})$", runner_text, re.MULTILINE)
    require(match is not None and match.group(1) == protocol_sha, "runner/protocol digest mismatch")
    confirmation_rows = load_confirmation_rows(data, manifest)

    trains: dict[tuple[int, str, str, int], dict[str, Any]] = {}
    raw: dict[tuple[int, str, str], dict[str, Any]] = {}
    composite: dict[tuple[int, str, str], dict[str, Any]] = {}
    rollout_metrics: list[dict[str, Any]] = []

    for seed in SEEDS:
        seed_root = root / f"s{seed}"
        require((seed_root / "COMPLETE").is_file(), f"seed {seed} is not complete")
        burnin = validate_rollout(seed_root / "burnin/rollout.json", arm="burnin", count=48, offset=6500, seed=seed)
        trains[(seed, "burnin", ESTIMATORS[0], 0)] = validate_train(seed_root / "burnin/train/summary.json", ESTIMATORS[0])
        rollout_metrics.append({"seed": seed, "round": 0, "family": "burnin", **burnin["summary"]})
        for round_id, offset in ((1, 6548), (2, 6596)):
            k1_path = seed_root / f"round{round_id}/k1/E0_standard_actor_ratio"
            k1_rollout = validate_rollout(k1_path / "rollout.json", arm="k1", count=48, offset=offset, seed=seed)
            trains[(seed, "k1", ESTIMATORS[0], round_id)] = validate_train(k1_path / "train/summary.json", ESTIMATORS[0])
            rollout_metrics.append({"seed": seed, "round": round_id, "family": "k1", **k1_rollout["summary"]})
        for family in FAMILIES:
            shared = validate_rollout(seed_root / f"round1/{family}/shared_rollout.json", arm=family, count=48, offset=6548, seed=seed)
            rollout_metrics.append({"seed": seed, "round": 1, "family": family, **shared["summary"]})
            for estimator in ESTIMATORS:
                trains[(seed, family, estimator, 1)] = validate_train(seed_root / f"round1/{family}/{estimator}/train/summary.json", estimator)
                round2_path = seed_root / f"round2/{family}/{estimator}"
                round2_rollout = validate_rollout(round2_path / "rollout.json", arm=family, count=48, offset=6596, seed=seed)
                rollout_metrics.append({"seed": seed, "round": 2, "family": family, "estimator": estimator, **round2_rollout["summary"]})
                trains[(seed, family, estimator, 2)] = validate_train(round2_path / "train/summary.json", estimator)
        raw[(seed, "k1", ESTIMATORS[0])] = validate_eval(seed_root / "eval/raw_k1/k1_E0_standard_actor_ratio.json", confirmation_rows, composite=False)
        for family in FAMILIES:
            for estimator in ESTIMATORS:
                raw[(seed, family, estimator)] = validate_eval(seed_root / f"eval/raw_k1/{family}_{estimator}.json", confirmation_rows, composite=False)
                composite[(seed, family, estimator)] = validate_eval(seed_root / f"eval/composite/{family}_{estimator}.json", confirmation_rows, composite=True, arm=family)

    require(len(trains) == 81, f"expected 81 total training summaries, found {len(trains)}")
    require(len(raw) == 39 and len(composite) == 36, "evaluation artifact count mismatch")
    require((root / "EVIDENCE_COMPLETE").is_file(), "evidence completion marker missing")

    metrics: list[dict[str, Any]] = []
    for (seed, family, estimator), payload in sorted(raw.items()):
        metrics.append({"mode": "raw_k1", "seed": seed, "family": family, "estimator": estimator, "accuracy": mean(outcome_vector(payload, "correct")), "answered_fraction": mean(outcome_vector(payload, "answered")), "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0))})
    for (seed, family, estimator), payload in sorted(composite.items()):
        metrics.append({"mode": "composite", "seed": seed, "family": family, "estimator": estimator, "accuracy": mean(outcome_vector(payload, "correct")), "answered_fraction": mean(outcome_vector(payload, "answered")), "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0)), **{key: payload["summary"].get(key) for key in ("states", "expansion_rate", "generated_candidates", "mean_unique_candidates_when_expanded", "identical_expanded_fraction")}})

    contrast_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    contrast_seed = BOOTSTRAP_SEED
    for family in FAMILIES:
        for name, mode, left_estimator, right_estimator in (
            ("raw_E3_minus_E0", "raw", ESTIMATORS[2], ESTIMATORS[0]),
            ("raw_E1_minus_E0", "raw", ESTIMATORS[1], ESTIMATORS[0]),
            ("composite_E4_minus_E0", "composite", ESTIMATORS[3], ESTIMATORS[0]),
        ):
            source = raw if mode == "raw" else composite
            left = [outcome_vector(source[(seed, family, left_estimator)], "correct") for seed in SEEDS]
            right = [outcome_vector(source[(seed, family, right_estimator)], "correct") for seed in SEEDS]
            row = {"family": family, **paired_contrast(left, right, seed=contrast_seed)}
            contrast_groups[name].append(row)
            contrast_seed += 1
    bestof_e3 = [outcome_vector(raw[(seed, "best_of_n", ESTIMATORS[2])], "correct") for seed in SEEDS]
    k1 = [outcome_vector(raw[(seed, "k1", ESTIMATORS[0])], "correct") for seed in SEEDS]
    contrast_groups["raw_best_of_n_E3_minus_k1"].append({"family": "best_of_n", **paired_contrast(bestof_e3, k1, seed=contrast_seed)})
    for rows in contrast_groups.values():
        holm_adjust(rows)

    stability = []
    for (seed, family, estimator, round_id), summary in sorted(trains.items()):
        stability.append({
            "seed": seed,
            "family": family,
            "estimator": estimator,
            "round": round_id,
            "states": int(summary["states"]),
            "optimizer_updates": int(summary["optimizer_updates"]),
            "ess": float(summary["external_weight_ess"]),
            "maximum_weight": float(summary["external_weight_maximum"]),
            "clip_fraction": float(summary["sequence_ratio_clip_fraction"]),
            "kl_proxy": float(summary["mean_selected_kl_proxy"]),
            "candidate_tokens": int(summary["candidate_tokens"]),
            "selected_tokens": int(summary["selected_tokens"]),
            "elapsed_seconds": float(summary["elapsed_seconds"]),
        })
    completeness_safety = {
        "finite_and_complete": True,
        "support_violations": 0,
        "all_clip_fractions_below_0_20": all(row["clip_fraction"] < 0.20 for row in stability),
        "all_mean_kl_proxies_below_0_02": all(row["kl_proxy"] < 0.02 for row in stability),
    }
    answer_regressions = []
    for seed in SEEDS:
        k1_answered = mean(outcome_vector(raw[(seed, "k1", ESTIMATORS[0])], "answered"))
        for family in FAMILIES:
            for estimator in ESTIMATORS:
                answered = mean(outcome_vector(raw[(seed, family, estimator)], "answered"))
                answer_regressions.append(answered - k1_answered)
    completeness_safety["no_raw_answer_rate_regression_over_2pp"] = min(answer_regressions) >= -0.02 - 1e-12

    e3_ess_dominates = all(
        trains[(seed, family, ESTIMATORS[2], round_id)]["external_weight_ess"]
        > trains[(seed, family, ESTIMATORS[1], round_id)]["external_weight_ess"]
        for seed in SEEDS for family in FAMILIES for round_id in (1, 2)
    )
    raw_winners = [row for row in contrast_groups["raw_E3_minus_E0"] if row["ci95_low"] >= 0 and all(seed_row["difference"] > 0 for seed_row in row["per_seed"])]
    composite_winners = [row for row in contrast_groups["composite_E4_minus_E0"] if row["ci95_low"] >= 0 and all(seed_row["difference"] > 0 for seed_row in row["per_seed"])]
    positive_transfer = {
        "raw_E3_family_gate": bool(raw_winners),
        "E3_ESS_above_E1_every_seed_family_round": e3_ess_dominates,
        "composite_E4_family_gate": bool(composite_winners),
        "stability_gate": all(completeness_safety.values()),
    }
    positive_transfer["passed"] = all(positive_transfer.values())

    return {
        "schema": "cgcs-search-policy-mismatch-s3-analysis-v1",
        "protocol_sha256": protocol_sha,
        "data_sha256": DATA_SHA256,
        "artifact_counts": {"burnin_training": 3, "k1_adaptive_training": 6, "search_adaptive_training": 72, "all_training": len(trains), "raw_k1_evaluations": len(raw), "composite_evaluations": len(composite)},
        "gpu_audit": parse_sessions(root),
        "evaluation_metrics": metrics,
        "rollout_metrics": rollout_metrics,
        "training_stability": stability,
        "contrasts": dict(contrast_groups),
        "completeness_safety": completeness_safety,
        "positive_transfer_gate": positive_transfer,
        "reproducibility_manifest": build_reproducibility_manifest(root, project_root),
    }


def render_report(analysis: dict[str, Any]) -> str:
    lines = [
        "# S3 LLM search-policy mismatch transfer report",
        "",
        f"Protocol SHA-256: `{analysis['protocol_sha256']}`.",
        f"Data SHA-256: `{analysis['data_sha256']}`.",
        "",
        "## Verdict",
        "",
        f"Positive-transfer gate: **{'PASS' if analysis['positive_transfer_gate']['passed'] else 'FAIL / MIXED'}**.",
        "",
        "## Paired confirmation contrasts",
        "",
        "| Contrast | Family | Difference | Clustered 95% CI | W/L/T | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, rows in analysis["contrasts"].items():
        for row in rows:
            wlt = row["pooled_wins_losses_ties"]
            lines.append(
                f"| {name} | {row['family']} | {100*row['estimate']:+.2f} pp | "
                f"[{100*row['ci95_low']:+.2f}, {100*row['ci95_high']:+.2f}] | "
                f"{wlt['wins']}/{wlt['losses']}/{wlt['ties']} | {row['holm_adjusted_p']:.4g} |"
            )
    lines.extend(["", "## Safety", ""])
    for key, value in analysis["completeness_safety"].items():
        lines.append(f"- {key}: **{value}**")
    lines.extend(["", "The raw K1 Actor and retained composite search policy are separate estimands. No search result is represented as an on-policy Actor result.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    analysis = analyze(args.root, args.project_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_report.write_text(render_report(analysis))
    print(json.dumps({"positive_transfer_gate": analysis["positive_transfer_gate"], "artifact_counts": analysis["artifact_counts"]}, indent=2))


if __name__ == "__main__":
    main()
