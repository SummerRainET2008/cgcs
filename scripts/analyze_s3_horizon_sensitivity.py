#!/usr/bin/env python3
"""Analyze the frozen S4 inference-horizon sensitivity matrix."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from acbpo.math_rollout_experiment import answers_match


SEEDS = (99801, 99802, 99803)
FAMILIES = ("best_of_n", "entropy_value", "value_value")
ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)
CONDITIONS = {
    "H8": (8, None),
    "H12": (12, None),
    "H16": (16, None),
    "C12": (12, 768),
    "C16": (16, 768),
}
CONTRASTS = tuple(name for name in CONDITIONS if name != "H8")
SECONDARY_FINAL = re.compile(
    r"(?i)^\s*(?:FINAL|FINAL\s+ANSWER|ANSWER)\s*[:=]\s*"
    r"([-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?)\s*[.。]?\s*$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def actor_names() -> tuple[str, ...]:
    return ("k1_E0_standard_actor_ratio",) + tuple(
        f"{family}_{estimator}"
        for family in FAMILIES
        for estimator in ESTIMATORS
    )


def actor_cell(actor: str) -> str:
    return "k1" if actor.startswith("k1_") else actor


def load_matrix(root: Path) -> dict[tuple[int, str, str], dict[str, Any]]:
    result: dict[tuple[int, str, str], dict[str, Any]] = {}
    expected_ids = list(range(248, 348))
    for seed in SEEDS:
        for condition, (max_steps, token_cap) in CONDITIONS.items():
            for actor in actor_names():
                path = root / f"s{seed}" / condition / f"{actor}.json"
                require(path.is_file(), f"missing evaluation: {path}")
                payload = json.loads(path.read_text())
                config = payload.get("config", {})
                examples = payload.get("examples", [])
                require(config.get("offset") == 248, f"{path}: offset drift")
                require(config.get("count") == 100, f"{path}: count drift")
                require(config.get("seed") == 101901, f"{path}: seed drift")
                require(config.get("max_steps") == max_steps, f"{path}: max_steps drift")
                require(config.get("step_tokens") == 96, f"{path}: step_tokens drift")
                require(
                    config.get("max_completion_tokens") == token_cap,
                    f"{path}: completion-token cap drift",
                )
                require(config.get("max_prompt_tokens") == 768, f"{path}: prompt drift")
                require(config.get("head_tokens") == 192, f"{path}: head drift")
                require(len(examples) == 100, f"{path}: incomplete examples")
                require(
                    [int(row["example_id"]) for row in examples] == expected_ids,
                    f"{path}: problem order drift",
                )
                for row in examples:
                    counts = [int(value) for value in row.get("step_token_counts", [])]
                    require(len(counts) == len(row.get("steps", [])), f"{path}: token lineage")
                    require(sum(counts) == int(row.get("completion_tokens", -1)), f"{path}: token sum")
                    require(0 < len(counts) <= max_steps, f"{path}: step count")
                    if token_cap is not None:
                        require(sum(counts) <= token_cap, f"{path}: token cap exceeded")
                    require(
                        row.get("termination_reason")
                        in {"valid_final", "max_steps", "max_completion_tokens"},
                        f"{path}: termination reason",
                    )
                result[(seed, condition, actor)] = payload
    require(len(result) == 39 * 5, "matrix cardinality drift")
    return result


def observation(row: dict[str, Any]) -> dict[str, float]:
    answered = float(bool(row["result"]["answered"]))
    correct = float(bool(row["result"]["correct"]))
    return {
        "accuracy": correct,
        "answer_rate": answered,
        "steps": float(len(row["steps"])),
        "completion_tokens": float(row["completion_tokens"]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    obs = [observation(row) for row in rows]
    answered = sum(item["answer_rate"] for item in obs)
    return {
        "examples": len(rows),
        "accuracy": float(np.mean([item["accuracy"] for item in obs])),
        "answer_rate": answered / len(rows),
        "conditional_accuracy": (
            sum(item["accuracy"] for item in obs) / answered if answered else math.nan
        ),
        "mean_steps": float(np.mean([item["steps"] for item in obs])),
        "mean_completion_tokens": float(
            np.mean([item["completion_tokens"] for item in obs])
        ),
        "total_completion_tokens": int(
            sum(item["completion_tokens"] for item in obs)
        ),
    }


def metric_value(rows: list[dict[str, Any]], metric: str) -> float:
    return aggregate(rows)[metric]


def clustered_contrast(
    left: dict[int, list[dict[str, Any]]],
    right: dict[int, list[dict[str, Any]]],
    *,
    metric: str,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, float]:
    ids = sorted(left)
    require(ids == sorted(right), "cluster IDs differ")
    observed = metric_value([row for key in ids for row in left[key]], metric) - metric_value(
        [row for key in ids for row in right[key]], metric
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        picked = rng.choice(ids, size=len(ids), replace=True)
        lrows = [row for key in picked for row in left[int(key)]]
        rrows = [row for key in picked for row in right[int(key)]]
        samples[index] = metric_value(lrows, metric) - metric_value(rrows, metric)
    pvalue = min(
        1.0,
        2.0
        * min(
            (float(np.count_nonzero(samples <= 0.0)) + 1.0) / (resamples + 1.0),
            (float(np.count_nonzero(samples >= 0.0)) + 1.0) / (resamples + 1.0),
        ),
    )
    return {
        "difference": float(observed),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "bootstrap_p": pvalue,
    }


def holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["bootstrap_p"])
    running = 0.0
    total = len(rows)
    for rank, (index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * float(row["bootstrap_p"]))
        running = max(running, adjusted)
        rows[index]["holm_p"] = running


def paired_counts(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, int]:
    require(len(left) == len(right), "paired rows differ")
    wins = losses = ties = 0
    for lrow, rrow in zip(left, right, strict=True):
        delta = int(bool(lrow["result"]["correct"])) - int(bool(rrow["result"]["correct"]))
        wins += delta > 0
        losses += delta < 0
        ties += delta == 0
    return {"wins": wins, "losses": losses, "ties": ties}


def secondary_answer(steps: list[str]) -> str | None:
    lines = [line.strip() for step in steps for line in str(step).splitlines() if line.strip()]
    if not lines:
        return None
    match = SECONDARY_FINAL.fullmatch(lines[-1])
    return match.group(1) if match else None


def parser_audit(s3_root: Path) -> dict[str, Any]:
    paths = sorted(s3_root.glob("s*/eval/raw_k1/*.json")) + sorted(
        s3_root.glob("s*/eval/composite/*.json")
    )
    require(len(paths) == 75, "terminal S3 evaluation count drift")
    table = defaultdict(int)
    recovered: list[dict[str, Any]] = []
    total = 0
    for path in paths:
        payload = json.loads(path.read_text())
        for row in payload["examples"]:
            total += 1
            strict = bool(row["result"]["answered"])
            prediction = secondary_answer(row["steps"])
            secondary = prediction is not None
            table[f"strict_{int(strict)}_secondary_{int(secondary)}"] += 1
            if secondary and not strict:
                recovered.append(
                    {
                        "path": str(path),
                        "example_id": int(row["example_id"]),
                        "prediction": prediction,
                        "correct": answers_match(prediction, row["reference_answer"]),
                        "last_step": row["steps"][-1] if row["steps"] else "",
                    }
                )
    recovered.sort(key=lambda row: (row["example_id"], row["path"]))
    return {
        "evaluations": len(paths),
        "transcripts": total,
        "table": dict(sorted(table.items())),
        "newly_recovered": len(recovered),
        "newly_recovered_correct": sum(int(row["correct"]) for row in recovered),
        "examples": recovered[:20],
    }


def analyze(root: Path, s3_root: Path) -> dict[str, Any]:
    matrix = load_matrix(root)
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    metrics = ("accuracy", "answer_rate", "conditional_accuracy", "mean_steps", "mean_completion_tokens")
    for actor in actor_names():
        for condition in CONDITIONS:
            rows = [
                row
                for seed in SEEDS
                for row in matrix[(seed, condition, actor)]["examples"]
            ]
            summaries.append({"cell": actor_cell(actor), "condition": condition, **aggregate(rows)})
        baseline = [
            row for seed in SEEDS for row in matrix[(seed, "H8", actor)]["examples"]
        ]
        for condition_index, condition in enumerate(CONTRASTS):
            candidate = [
                row for seed in SEEDS for row in matrix[(seed, condition, actor)]["examples"]
            ]
            left_clusters = {
                example_id: [
                    matrix[(seed, condition, actor)]["examples"][example_id - 248]
                    for seed in SEEDS
                ]
                for example_id in range(248, 348)
            }
            right_clusters = {
                example_id: [
                    matrix[(seed, "H8", actor)]["examples"][example_id - 248]
                    for seed in SEEDS
                ]
                for example_id in range(248, 348)
            }
            row: dict[str, Any] = {
                "cell": actor_cell(actor),
                "condition": condition,
                **paired_counts(candidate, baseline),
                "metrics": {},
            }
            for metric_index, metric in enumerate(metrics):
                row["metrics"][metric] = clustered_contrast(
                    left_clusters,
                    right_clusters,
                    metric=metric,
                    seed=440_000 + condition_index * 100 + metric_index,
                )
            contrasts.append(row)

    for condition in CONTRASTS:
        for metric in metrics:
            selected = [
                row["metrics"][metric]
                for row in contrasts
                if row["condition"] == condition and row["cell"] != "k1"
            ]
            holm_adjust(selected)

    pooled: dict[str, Any] = {}
    for condition_index, condition in enumerate(CONTRASTS):
        left = {
            example_id: [
                matrix[(seed, condition, actor)]["examples"][example_id - 248]
                for seed in SEEDS
                for actor in actor_names()
            ]
            for example_id in range(248, 348)
        }
        right = {
            example_id: [
                matrix[(seed, "H8", actor)]["examples"][example_id - 248]
                for seed in SEEDS
                for actor in actor_names()
            ]
            for example_id in range(248, 348)
        }
        pooled[condition] = {
            metric: clustered_contrast(
                left,
                right,
                metric=metric,
                seed=550_000 + condition_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    passing = []
    for condition in ("H12", "H16"):
        row = pooled[condition]
        passed = (
            row["answer_rate"]["difference"] >= 0.05
            and row["answer_rate"]["ci95_low"] > 0.0
            and row["accuracy"]["ci95_low"] >= 0.0
            and row["conditional_accuracy"]["ci95_low"] >= -0.02
        )
        row["horizon_gate_passed"] = passed
        if passed:
            passing.append(condition)
    selected_horizon = passing[0] if passing else "H8"
    return {
        "schema": "cgcs-s4-inference-horizon-sensitivity-v1",
        "matrix": {"actors": 39, "conditions": 5, "problems": 100, "evaluations": 19_500},
        "summaries": summaries,
        "contrasts": contrasts,
        "pooled_contrasts": pooled,
        "selected_horizon": selected_horizon,
        "parser_audit": parser_audit(s3_root),
    }


def percent(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{100 * value:.2f}%"


def report(payload: dict[str, Any]) -> str:
    lines = [
        "# S4 inference-horizon and answer-rate sensitivity report",
        "",
        f"Frozen horizon selection: **{payload['selected_horizon']}**.",
        "",
        "## Pooled contrasts versus H8",
        "",
        "| Condition | Accuracy | Answer rate | Conditional accuracy | Mean steps | Mean completion tokens | Horizon gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for condition in CONTRASTS:
        row = payload["pooled_contrasts"][condition]
        lines.append(
            f"| {condition} | {percent(row['accuracy']['difference'])} "
            f"[{percent(row['accuracy']['ci95_low'])}, {percent(row['accuracy']['ci95_high'])}] | "
            f"{percent(row['answer_rate']['difference'])} "
            f"[{percent(row['answer_rate']['ci95_low'])}, {percent(row['answer_rate']['ci95_high'])}] | "
            f"{percent(row['conditional_accuracy']['difference'])} "
            f"[{percent(row['conditional_accuracy']['ci95_low'])}, {percent(row['conditional_accuracy']['ci95_high'])}] | "
            f"{row['mean_steps']['difference']:+.3f} | "
            f"{row['mean_completion_tokens']['difference']:+.2f} | "
            f"{row.get('horizon_gate_passed', 'diagnostic')} |"
        )
    audit = payload["parser_audit"]
    lines.extend(
        [
            "",
            "## Offline extractor audit",
            "",
            f"Audited {audit['transcripts']} transcripts from {audit['evaluations']} terminal S3 evaluations. "
            f"The anchored secondary grammar newly recovered {audit['newly_recovered']} finals, "
            f"of which {audit['newly_recovered_correct']} were correct.",
            "",
            "This diagnostic does not alter S3 or replace strict S4 scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.root, args.s3_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(report(payload))


if __name__ == "__main__":
    main()
