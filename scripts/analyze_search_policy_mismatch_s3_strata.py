#!/usr/bin/env python3
"""Descriptive MATH-500 level/subject analysis for the terminal S3 archive.

This is an independent post-run diagnostic.  It deliberately leaves the
frozen S3 analyzer and every preregistered contrast unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_ANALYZER = SCRIPT_DIR / "analyze_search_policy_mismatch_s3.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_analyzer() -> Any:
    spec = importlib.util.spec_from_file_location("s3_frozen_analyzer", FROZEN_ANALYZER)
    require(spec is not None and spec.loader is not None, "cannot load frozen S3 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_strata(rows: list[dict[str, Any]], *, expected_count: int) -> dict[str, list[str]]:
    require(len(rows) == expected_count, "confirmation row count mismatch")
    result: dict[str, list[str]] = {"level": [], "subject": []}
    for index, row in enumerate(rows):
        source = row.get("source", {})
        require(isinstance(source, dict), f"confirmation row {index}: missing source metadata")
        for key in result:
            value = source.get(key)
            require(value is not None and str(value).strip(), f"confirmation row {index}: missing {key}")
            result[key].append(str(value))
    return result


def metrics_for_payload(
    payload: dict[str, Any],
    strata: dict[str, list[str]],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    examples = payload.get("examples", [])
    require(len(examples) == expected_count, "evaluation example count mismatch")
    rows: list[dict[str, Any]] = []
    for stratum_type, labels in sorted(strata.items()):
        require(len(labels) == expected_count, f"{stratum_type}: label count mismatch")
        for label in sorted(set(labels)):
            indices = [index for index, observed in enumerate(labels) if observed == label]
            correct = sum(bool(examples[index]["result"]["correct"]) for index in indices)
            answered = sum(bool(examples[index]["result"]["answered"]) for index in indices)
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum": label,
                    "examples": len(indices),
                    "correct": correct,
                    "answered": answered,
                    "accuracy": correct / len(indices),
                    "answered_fraction": answered / len(indices),
                }
            )
    return rows


def descriptive_stratified_contrast(
    paired_payloads: list[tuple[int, dict[str, Any], dict[str, Any]]],
    strata: dict[str, list[str]],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    """Return paired subgroup effects without CIs or hypothesis tests."""
    require(bool(paired_payloads), "paired contrast has no seeds")
    rows: list[dict[str, Any]] = []
    for stratum_type, labels in sorted(strata.items()):
        require(len(labels) == expected_count, f"{stratum_type}: label count mismatch")
        for label in sorted(set(labels)):
            indices = [index for index, observed in enumerate(labels) if observed == label]
            pooled: list[int] = []
            per_seed: list[dict[str, Any]] = []
            for seed, left, right in paired_payloads:
                left_examples = left.get("examples", [])
                right_examples = right.get("examples", [])
                require(
                    len(left_examples) == len(right_examples) == expected_count,
                    "paired evaluation example count mismatch",
                )
                differences = [
                    int(bool(left_examples[index]["result"]["correct"]))
                    - int(bool(right_examples[index]["result"]["correct"]))
                    for index in indices
                ]
                pooled.extend(differences)
                per_seed.append(
                    {
                        "seed": seed,
                        "difference": sum(differences) / len(differences),
                        "wins": sum(value > 0 for value in differences),
                        "losses": sum(value < 0 for value in differences),
                        "ties": sum(value == 0 for value in differences),
                    }
                )
            rows.append(
                {
                    "stratum_type": stratum_type,
                    "stratum": label,
                    "problem_seed_observations": len(pooled),
                    "difference": sum(pooled) / len(pooled),
                    "wins": sum(value > 0 for value in pooled),
                    "losses": sum(value < 0 for value in pooled),
                    "ties": sum(value == 0 for value in pooled),
                    "per_seed": per_seed,
                }
            )
    return rows


def build_analysis(root: Path, project_root: Path) -> dict[str, Any]:
    require((root / "EVIDENCE_COMPLETE").is_file(), "missing EVIDENCE_COMPLETE marker")
    require((root / "COMPLETE").is_file(), "missing COMPLETE marker")
    frozen = load_frozen_analyzer()
    data_path = project_root / "data/math500_scalar.jsonl"
    manifest_path = project_root / "data/math500_scalar_manifest.json"
    confirmation = frozen.load_confirmation_rows(data_path, manifest_path)
    strata = extract_strata(confirmation, expected_count=frozen.CONFIRMATION_COUNT)

    metrics: list[dict[str, Any]] = []
    payloads: dict[tuple[int, str, str, str], dict[str, Any]] = {}

    def add_payload(
        *, seed: int, mode: str, family: str, estimator: str, path: Path
    ) -> None:
        payload = frozen.validate_eval(
            path,
            confirmation,
            composite=mode == "composite",
            arm=family if mode == "composite" else None,
        )
        payloads[(seed, mode, family, estimator)] = payload
        for row in metrics_for_payload(
            payload, strata, expected_count=frozen.CONFIRMATION_COUNT
        ):
            metrics.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "family": family,
                    "estimator": estimator,
                    **row,
                }
            )

    for seed in frozen.SEEDS:
        seed_root = root / f"s{seed}/eval"
        add_payload(
            seed=seed,
            mode="raw_k1",
            family="k1",
            estimator=frozen.ESTIMATORS[0],
            path=seed_root / "raw_k1/k1_E0_standard_actor_ratio.json",
        )
        for family in frozen.FAMILIES:
            for estimator in frozen.ESTIMATORS:
                add_payload(
                    seed=seed,
                    mode="raw_k1",
                    family=family,
                    estimator=estimator,
                    path=seed_root / f"raw_k1/{family}_{estimator}.json",
                )
                add_payload(
                    seed=seed,
                    mode="composite",
                    family=family,
                    estimator=estimator,
                    path=seed_root / f"composite/{family}_{estimator}.json",
                )

    expected_evaluations = len(frozen.SEEDS) * (
        1 + 2 * len(frozen.FAMILIES) * len(frozen.ESTIMATORS)
    )
    strata_per_evaluation = sum(len(set(labels)) for labels in strata.values())
    require(
        len(metrics) == expected_evaluations * strata_per_evaluation,
        "stratified metric matrix is incomplete",
    )

    grouped: dict[tuple[str, str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"seed_count": 0, "examples": 0, "correct": 0, "answered": 0}
    )
    for row in metrics:
        key = (
            row["mode"],
            row["family"],
            row["estimator"],
            row["stratum_type"],
            row["stratum"],
        )
        aggregate = grouped[key]
        aggregate["seed_count"] += 1
        aggregate["examples"] += row["examples"]
        aggregate["correct"] += row["correct"]
        aggregate["answered"] += row["answered"]

    pooled = []
    for key, counts in sorted(grouped.items()):
        mode, family, estimator, stratum_type, stratum = key
        pooled.append(
            {
                "mode": mode,
                "family": family,
                "estimator": estimator,
                "stratum_type": stratum_type,
                "stratum": stratum,
                **counts,
                "accuracy": counts["correct"] / counts["examples"],
                "answered_fraction": counts["answered"] / counts["examples"],
            }
        )

    contrasts: list[dict[str, Any]] = []

    def add_contrast(
        name: str,
        left_mode: str,
        left_family: str,
        left_estimator: str,
        right_mode: str,
        right_family: str,
        right_estimator: str,
    ) -> None:
        paired = [
            (
                seed,
                payloads[(seed, left_mode, left_family, left_estimator)],
                payloads[(seed, right_mode, right_family, right_estimator)],
            )
            for seed in frozen.SEEDS
        ]
        for row in descriptive_stratified_contrast(
            paired, strata, expected_count=frozen.CONFIRMATION_COUNT
        ):
            contrasts.append(
                {
                    "contrast": name,
                    "left": {
                        "mode": left_mode,
                        "family": left_family,
                        "estimator": left_estimator,
                    },
                    "right": {
                        "mode": right_mode,
                        "family": right_family,
                        "estimator": right_estimator,
                    },
                    **row,
                }
            )

    for family in frozen.FAMILIES:
        add_contrast(
            "raw_E3_minus_E0",
            "raw_k1",
            family,
            frozen.ESTIMATORS[2],
            "raw_k1",
            family,
            frozen.ESTIMATORS[0],
        )
        add_contrast(
            "raw_E1_minus_E0",
            "raw_k1",
            family,
            frozen.ESTIMATORS[1],
            "raw_k1",
            family,
            frozen.ESTIMATORS[0],
        )
        add_contrast(
            "composite_E4_minus_E0",
            "composite",
            family,
            frozen.ESTIMATORS[3],
            "composite",
            family,
            frozen.ESTIMATORS[0],
        )
    add_contrast(
        "raw_best_of_n_E3_minus_k1",
        "raw_k1",
        "best_of_n",
        frozen.ESTIMATORS[2],
        "raw_k1",
        "k1",
        frozen.ESTIMATORS[0],
    )
    expected_contrast_groups = 3 * len(frozen.FAMILIES) + 1
    require(
        len(contrasts) == expected_contrast_groups * strata_per_evaluation,
        "stratified descriptive contrast matrix is incomplete",
    )

    return {
        "schema": "cgcs-search-policy-mismatch-s3-strata-v1",
        "scope": "descriptive post-run diagnostic; no additional confirmatory tests",
        "frozen_analyzer_sha256": sha256_file(FROZEN_ANALYZER),
        "data_sha256": sha256_file(data_path),
        "confirmation_composition": {
            kind: {label: labels.count(label) for label in sorted(set(labels))}
            for kind, labels in sorted(strata.items())
        },
        "evaluation_count": expected_evaluations,
        "per_seed_metrics": metrics,
        "pooled_descriptive_metrics": pooled,
        "stratified_descriptive_contrasts": contrasts,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# S3 MATH-500 level and subject diagnostics",
        "",
        "These are descriptive post-run diagnostics. They do not add hypotheses,",
        "change the frozen S3 verdict, or replace the preregistered paired analysis.",
        "",
        "## Sealed confirmation composition",
        "",
    ]
    for kind, counts in analysis["confirmation_composition"].items():
        lines.append(f"- {kind}: " + ", ".join(f"{key}: {value}" for key, value in counts.items()))
    lines.extend(
        [
            "",
            f"Validated evaluations: **{analysis['evaluation_count']}**.",
            "",
            "The machine-readable JSON retains every per-seed and pooled descriptive",
            "accuracy/answer-rate row for all levels and subjects. It also retains",
            "paired W/L/T subgroup differences for the preregistered contrast directions,",
            "without adding subgroup confidence intervals, p-values, or promotion gates.",
            "",
            "## Descriptive paired subgroup contrasts",
            "",
            "No subgroup confidence intervals or p-values are reported. Each observation",
            "is one training-seed/problem pair, and the table is exploratory only.",
            "",
            "| Contrast | Left | Right | Stratum | Observations | Difference | W/L/T |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in analysis["stratified_descriptive_contrasts"]:
        left = row["left"]
        right = row["right"]
        left_label = f"{left['mode']}:{left['family']}:{left['estimator']}"
        right_label = f"{right['mode']}:{right['family']}:{right['estimator']}"
        stratum = f"{row['stratum_type']}={row['stratum']}"
        lines.append(
            f"| {row['contrast']} | {left_label} | {right_label} | {stratum} | "
            f"{row['problem_seed_observations']} | {100 * row['difference']:+.2f} pp | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    analysis = build_analysis(args.root, args.project_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_markdown.write_text(render_markdown(analysis))


if __name__ == "__main__":
    main()
