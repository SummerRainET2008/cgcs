#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (99801, 99802, 99803)
HORIZONS = (8, 16)
ACTORS = (
    "k1_E0_standard_actor_ratio",
    "best_of_n_E0_standard_actor_ratio",
    "best_of_n_E3_step_cumulative_lambda_0.75",
    "best_of_n_E4_extended_candidate_set",
)
TARGET = "best_of_n_E4_extended_candidate_set"
BASELINES = (
    "best_of_n_E0_standard_actor_ratio",
    "best_of_n_E3_step_cumulative_lambda_0.75",
    "k1_E0_standard_actor_ratio",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def state(row: dict[str, Any]) -> str:
    if not row["result"]["answered"]:
        return "unanswered"
    return "correct" if row["result"]["correct"] else "incorrect"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = np.asarray([bool(row["result"]["correct"]) for row in rows])
    answered = np.asarray([bool(row["result"]["answered"]) for row in rows])
    return {
        "examples": len(rows),
        "accuracy": float(correct.mean()),
        "answer_rate": float(answered.mean()),
        "conditional_accuracy": float(correct.sum() / answered.sum()) if answered.any() else None,
        "mean_steps": float(np.mean([len(row["steps"]) for row in rows])),
        "mean_completion_tokens": float(np.mean([row["completion_tokens"] for row in rows])),
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
    }


def load_matrix(root: Path, expected_count: int) -> tuple[dict[tuple[int, int, str], list[dict[str, Any]]], list[str]]:
    matrix: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    ordered_ids: list[str] | None = None
    for seed in SEEDS:
        for horizon in HORIZONS:
            for actor in ACTORS:
                path = root / "eval" / f"s{seed}" / f"H{horizon}" / f"{actor}.json"
                payload = json.loads(path.read_text())
                rows = payload["examples"]
                if len(rows) != expected_count:
                    raise ValueError(f"count drift in {path}: {len(rows)}")
                if [int(row["example_id"]) for row in rows] != list(range(expected_count)):
                    raise ValueError(f"example order drift in {path}")
                source_ids = [str(row["source_id"]) for row in rows]
                if len(source_ids) != len(set(source_ids)):
                    raise ValueError(f"duplicate source IDs in {path}")
                if ordered_ids is None:
                    ordered_ids = source_ids
                elif source_ids != ordered_ids:
                    raise ValueError(f"source ID order drift in {path}")
                matrix[(seed, horizon, actor)] = rows
    if len(matrix) != 24 or ordered_ids is None:
        raise ValueError("S4-X matrix is incomplete")
    return matrix, ordered_ids


def _arrays(matrix, actor: str, horizons: tuple[int, ...], key: str) -> np.ndarray:
    values = []
    for seed in SEEDS:
        for horizon in horizons:
            rows = matrix[(seed, horizon, actor)]
            if key == "correct":
                values.append([float(row["result"]["correct"]) for row in rows])
            elif key == "answered":
                values.append([float(row["result"]["answered"]) for row in rows])
            elif key == "steps":
                values.append([float(len(row["steps"])) for row in rows])
            elif key == "tokens":
                values.append([float(row["completion_tokens"]) for row in rows])
            else:
                raise ValueError(key)
    return np.asarray(values, dtype=np.float64)


def _estimate(correct: np.ndarray, answered: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(correct.mean())
    if metric == "answer_rate":
        return float(answered.mean())
    if metric == "conditional_accuracy":
        return float(correct.sum() / answered.sum()) if answered.sum() else float("nan")
    raise ValueError(metric)


def bootstrap_contrast(
    matrix,
    target: str,
    baseline: str,
    horizons: tuple[int, ...],
    *,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    tc = _arrays(matrix, target, horizons, "correct")
    ta = _arrays(matrix, target, horizons, "answered")
    bc = _arrays(matrix, baseline, horizons, "correct")
    ba = _arrays(matrix, baseline, horizons, "answered")
    count = tc.shape[1]
    metrics = ("accuracy", "answer_rate", "conditional_accuracy")
    observed = {
        metric: _estimate(tc, ta, metric) - _estimate(bc, ba, metric)
        for metric in metrics
    }
    samples = {metric: np.empty(replicates, dtype=np.float64) for metric in metrics}
    for index in range(replicates):
        chosen = rng.integers(0, count, size=count)
        for metric in metrics:
            samples[metric][index] = _estimate(tc[:, chosen], ta[:, chosen], metric) - _estimate(
                bc[:, chosen], ba[:, chosen], metric
            )
    result: dict[str, Any] = {"target": target, "baseline": baseline, "horizons": list(horizons)}
    for metric in metrics:
        finite = samples[metric][np.isfinite(samples[metric])]
        if not len(finite):
            result[metric] = {"difference": observed[metric], "ci95": [None, None], "p_two_sided": None}
            continue
        p = min(1.0, 2.0 * min(float(np.mean(finite <= 0)), float(np.mean(finite >= 0))))
        result[metric] = {
            "difference": observed[metric],
            "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
            "p_two_sided": p,
        }
    return result


def analyze(root: Path, *, expected_count: int, replicates: int, bootstrap_seed: int) -> dict[str, Any]:
    matrix, ordered_ids = load_matrix(root, expected_count)
    cells = {
        f"s{seed}/H{horizon}/{actor}": _summary(matrix[(seed, horizon, actor)])
        for seed in SEEDS for horizon in HORIZONS for actor in ACTORS
    }
    rng = np.random.default_rng(bootstrap_seed)
    contrasts = []
    for horizons in ((8,), (16,), (8, 16)):
        for baseline in BASELINES:
            contrasts.append(
                bootstrap_contrast(
                    matrix, TARGET, baseline, horizons, replicates=replicates, rng=rng
                )
            )
    accuracy_ps = [float(row["accuracy"]["p_two_sided"]) for row in contrasts]
    for row, adjusted in zip(contrasts, holm_adjust(accuracy_ps), strict=True):
        row["accuracy"]["p_holm"] = adjusted

    transitions: dict[str, Any] = {}
    for horizon in HORIZONS:
        for baseline in BASELINES:
            counts: Counter[str] = Counter()
            for seed in SEEDS:
                before = matrix[(seed, horizon, baseline)]
                after = matrix[(seed, horizon, TARGET)]
                for left, right in zip(before, after, strict=True):
                    counts[f"{state(left)}->{state(right)}"] += 1
            transitions[f"H{horizon}:{baseline}->E4"] = dict(sorted(counts.items()))

    resource_effects = {}
    for horizon in HORIZONS:
        for baseline in BASELINES:
            resource_effects[f"H{horizon}:{baseline}->E4"] = {
                "mean_step_difference": float(
                    _arrays(matrix, TARGET, (horizon,), "steps").mean()
                    - _arrays(matrix, baseline, (horizon,), "steps").mean()
                ),
                "mean_token_difference": float(
                    _arrays(matrix, TARGET, (horizon,), "tokens").mean()
                    - _arrays(matrix, baseline, (horizon,), "tokens").mean()
                ),
            }

    per_seed = {}
    for seed in SEEDS:
        for horizon in HORIZONS:
            target_rows = matrix[(seed, horizon, TARGET)]
            for baseline in BASELINES:
                base_rows = matrix[(seed, horizon, baseline)]
                per_seed[f"s{seed}/H{horizon}/{baseline}->E4"] = {
                    "accuracy_difference": _summary(target_rows)["accuracy"] - _summary(base_rows)["accuracy"],
                    "answer_rate_difference": _summary(target_rows)["answer_rate"] - _summary(base_rows)["answer_rate"],
                }

    return {
        "schema": "cgcs-s4x-arc-challenge-analysis-v1",
        "expected_cells": 24,
        "observed_cells": len(matrix),
        "examples_per_cell": expected_count,
        "total_evaluations": 24 * expected_count,
        "ordered_source_ids": ordered_ids,
        "bootstrap": {"replicates": replicates, "seed": bootstrap_seed, "cluster": "source_id"},
        "cells": cells,
        "contrasts": contrasts,
        "per_seed_effects": per_seed,
        "transitions": transitions,
        "resource_effects": resource_effects,
    }


def report(payload: dict[str, Any]) -> str:
    lines = [
        "# S4-X ARC-Challenge Analysis",
        "",
        f"Validated {payload['observed_cells']}/24 cells and {payload['total_evaluations']:,} evaluations.",
        "",
        "## Registered accuracy contrasts",
        "",
        "| Horizon | Baseline | E4 difference | 95% CI | Holm p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["contrasts"]:
        metric = row["accuracy"]
        horizon = "+".join(f"H{x}" for x in row["horizons"])
        ci = metric["ci95"]
        lines.append(
            f"| {horizon} | {row['baseline']} | {metric['difference']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {metric['p_holm']:.4g} |"
        )
    lines.extend(["", "Answer-rate, conditional-accuracy, per-seed, transition, token, and termination audits are retained in the JSON artifact.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1172)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261106)
    args = parser.parse_args()
    payload = analyze(
        args.root,
        expected_count=args.expected_count,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_report.write_text(report(payload))


if __name__ == "__main__":
    main()
