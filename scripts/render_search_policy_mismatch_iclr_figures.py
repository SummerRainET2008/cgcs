#!/usr/bin/env python3
"""Render the five preregistered ICLR figures from terminal evidence tables.

The renderer is intentionally downstream of the fail-closed evidence-table
renderer.  It never reads partial experiment directories and refuses any
payload that is not marked evidence-complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cgcs-matplotlib-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


FAMILIES = ("best_of_n", "entropy_value", "value_value")
FAMILY_LABELS = {
    "best_of_n": "Best-of-N",
    "entropy_value": "Entropy branching",
    "value_value": "Value/PRM rollout",
}
FAMILY_COLORS = {
    "best_of_n": "#0072B2",
    "entropy_value": "#D55E00",
    "value_value": "#009E73",
}
ESTIMATORS = (
    "E0_standard_selected_action",
    "E1_trajectory",
    "E2_cumulative_marginal",
    "E3_step_cumulative_lambda_0.75",
)
ESTIMATOR_LABELS = {
    "E0_standard_selected_action": "E0",
    "E1_trajectory": "E1",
    "E2_cumulative_marginal": "E2",
    "E3_step_cumulative_lambda_0.75": "E3",
}
S3_ESTIMATORS = {
    "E0": "E0_standard_actor_ratio",
    "E1": "E1_trajectory_propensity",
    "E3": "E3_step_cumulative_lambda_0.75",
    "E4": "E4_extended_candidate_set",
}
CONDITION_ORDER = (
    "exact",
    "lagged_kl_0.01",
    "lagged_kl_0.04",
    "learned_1024",
    "learned_256",
    "noise_0.5",
    "noise_1",
    "noise_2",
    "inverted_diagnostic",
)
CONDITION_LABELS = {
    "exact": "Exact",
    "lagged_kl_0.01": "Lagged .01",
    "lagged_kl_0.04": "Lagged .04",
    "learned_1024": "Learned 1,024",
    "learned_256": "Learned 256",
    "noise_0.5": "Noise .5",
    "noise_1": "Noise 1",
    "noise_2": "Noise 2",
    "inverted_diagnostic": "Inverted",
}
FIGURE_NAMES = (
    "figure_1_estimands_and_data_generation",
    "figure_2_exact_bias_variance_frontier",
    "figure_3_when_search_shift_helps",
    "figure_4_no_search_llm_transfer",
    "figure_5_retained_search_objective_and_cost",
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    require(bool(rows), "cannot average an empty sequence")
    return sum(rows) / len(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def validate_tables(payload: dict[str, Any]) -> dict[str, Any]:
    require(
        payload.get("schema") == "cgcs-search-policy-mismatch-iclr-tables-v1",
        "ICLR evidence-table schema mismatch",
    )
    require(payload.get("status") == "evidence_complete", "evidence tables are not terminal")
    provenance = payload.get("provenance")
    require(isinstance(provenance, dict), "evidence-table provenance missing")
    require(
        re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("renderer_sha256", "")))
        is not None,
        "invalid evidence-table renderer hash",
    )
    input_hashes = provenance.get("input_sha256")
    require(
        isinstance(input_hashes, dict) and set(input_hashes) == EVIDENCE_INPUTS,
        "evidence-table input hash matrix drift",
    )
    require(
        all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in input_hashes.values()),
        "invalid evidence-table input hash",
    )
    table = payload.get("table_2_cross_stage")
    require(isinstance(table, dict), "cross-stage evidence missing")
    expected_counts = {
        "exact_estimator_bias_variance": 15,
        "exact_search_shift": 27,
        "s3_contrast_rows": 10,
        "s3_compute_by_family_estimator": 12,
        "s3_evaluation_metrics": 75,
    }
    for key, count in expected_counts.items():
        require(isinstance(table.get(key), list) and len(table[key]) == count, f"{key} count drift")

    exact_rows = table["exact_estimator_bias_variance"]
    require(
        {(row.get("family"), row.get("estimator")) for row in exact_rows}
        == {
            (family, estimator)
            for family in FAMILIES
            for estimator in (*ESTIMATORS, "E4_extended_candidate_set")
        },
        "exact estimator identity matrix drift",
    )
    exact_numeric = (
        "mean_relative_bias",
        "mean_gradient_relative_mse_batch_256",
        "mean_gradient_relative_mse_batch_1024",
        "mean_normalized_final_weight_ess",
        "mean_raw_actor_return_change",
        "mean_composite_search_return_change",
    )
    require(
        all(all(finite(row.get(key)) for key in exact_numeric) for row in exact_rows),
        "non-finite exact estimator evidence",
    )

    shift_rows = table["exact_search_shift"]
    require(
        {(row.get("condition"), row.get("family")) for row in shift_rows}
        == {(condition, family) for condition in CONDITION_ORDER for family in FAMILIES},
        "exact search-shift identity matrix drift",
    )
    shift_numeric = (
        "mdp_count",
        "mean_critic_ranking_concordance",
        "mean_search_gain_vs_raw",
        "mean_maximum_depth_state_action_tv",
        "oracle_search_gain_headroom",
    )
    require(
        all(all(finite(row.get(key)) for key in shift_numeric) for row in shift_rows),
        "non-finite exact search-shift evidence",
    )

    contrasts = table["s3_contrast_rows"]
    expected_contrasts = {
        (name, family)
        for name in ("raw_E3_minus_E0", "raw_E1_minus_E0", "composite_E4_minus_E0")
        for family in FAMILIES
    } | {("raw_best_of_n_E3_minus_k1", "best_of_n")}
    require(
        {(row.get("contrast"), row.get("family")) for row in contrasts}
        == expected_contrasts,
        "S3 contrast identity matrix drift",
    )
    for row in contrasts:
        require(
            all(finite(row.get(key)) for key in ("estimate", "ci95_low", "ci95_high", "holm_adjusted_p")),
            "non-finite S3 contrast evidence",
        )
        per_seed = row.get("per_seed")
        require(
            isinstance(per_seed, list)
            and {seed_row.get("seed") for seed_row in per_seed} == {99801, 99802, 99803}
            and all(finite(seed_row.get("difference")) for seed_row in per_seed),
            "S3 per-seed contrast matrix drift",
        )

    compute_rows = table["s3_compute_by_family_estimator"]
    require(
        {(row.get("family"), row.get("estimator")) for row in compute_rows}
        == {(family, estimator) for family in FAMILIES for estimator in S3_ESTIMATORS.values()},
        "S3 compute identity matrix drift",
    )
    compute_numeric = (
        "summed_invocation_elapsed_seconds",
        "candidate_token_ratio_to_same_family_E0",
        "elapsed_ratio_to_same_family_E0",
    )
    require(
        all(all(finite(row.get(key)) for key in compute_numeric) for row in compute_rows),
        "non-finite S3 compute evidence",
    )

    evaluations = table["s3_evaluation_metrics"]
    raw = [row for row in evaluations if row.get("mode") == "raw_k1"]
    composite = [row for row in evaluations if row.get("mode") == "composite"]
    expected_raw = {
        (seed, "k1", S3_ESTIMATORS["E0"])
        for seed in (99801, 99802, 99803)
    } | {
        (seed, family, estimator)
        for seed in (99801, 99802, 99803)
        for family in FAMILIES
        for estimator in S3_ESTIMATORS.values()
    }
    expected_composite = {
        (seed, family, estimator)
        for seed in (99801, 99802, 99803)
        for family in FAMILIES
        for estimator in S3_ESTIMATORS.values()
    }
    require(
        {(row.get("seed"), row.get("family"), row.get("estimator")) for row in raw}
        == expected_raw,
        "S3 raw-K1 evaluation identity matrix drift",
    )
    require(
        {(row.get("seed"), row.get("family"), row.get("estimator")) for row in composite}
        == expected_composite,
        "S3 composite evaluation identity matrix drift",
    )
    require(
        all(
            all(finite(row.get(key)) for key in ("accuracy", "answered_fraction", "elapsed_seconds"))
            for row in evaluations
        ),
        "non-finite S3 evaluation evidence",
    )
    require(
        all(
            0.0
            <= float(row["accuracy"])
            <= float(row["answered_fraction"])
            <= 1.0
            for row in evaluations
        ),
        "S3 evaluation accuracy/answer-rate bounds violated",
    )
    composite_compute_keys = (
        "registered_expansion_rate",
        "source_rollout_expansion_rate",
        "expansion_rate",
        "expansion_rate_deviation",
        "evaluation_minus_source_rollout_expansion_rate",
        "generated_candidates",
        "candidate_completion_tokens",
        "selected_completion_tokens",
        "candidate_token_multiplier_vs_selected",
    )
    diversity_keys = (
        "mean_unique_candidates_when_expanded",
        "identical_expanded_fraction",
        "exact_token_pair_fraction",
        "mean_pairwise_prefix_divergence_fraction",
    )
    require(
        all(
            all(finite(row.get(key)) for key in composite_compute_keys)
            for row in composite
        ),
        "non-finite S3 composite compute evidence",
    )
    for row in composite:
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
                "non-finite S3 composite diversity evidence",
            )
    return table


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "CGCS deterministic ICLR figure renderer"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "CGCS",
            "Subject": "Search-Induced Policy Mismatch",
            "Keywords": "CGCS, search, policy mismatch",
            "Creator": "CGCS deterministic ICLR figure renderer",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return [png, pdf]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=10)


def figure_1() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = (
        (0.2, 2.15, 1.4, 0.8, "Raw Actor\n$\\pi_{\\theta}$", "#E8F1FA"),
        (2.0, 2.15, 1.4, 0.8, "Allocation gate\n$g(s)$", "#FCE8D5"),
        (3.8, 2.15, 1.4, 0.8, "Candidate\ngenerator", "#E7F4EA"),
        (5.6, 2.15, 1.4, 0.8, "Retention /\nselection $q$", "#F4E8F7"),
        (7.4, 2.15, 1.4, 0.8, "Credit\nestimator", "#FFF4CC"),
        (9.15, 2.15, 0.65, 0.8, "Actor\nupdate", "#E8F1FA"),
    )
    for x, y, width, height, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.04", linewidth=1.2,
                edgecolor="#333333", facecolor=color,
            )
        )
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center")
    for left, right in zip(boxes[:-1], boxes[1:], strict=True):
        ax.add_patch(
            FancyArrowPatch(
                (left[0] + left[2], left[1] + left[3] / 2),
                (right[0], right[1] + right[3] / 2),
                arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color="#333333",
            )
        )
    ax.text(4.5, 3.35, "Search changes which states, actions, and prefixes enter training",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(1.0, 1.25, "Raw-Actor target", color="#0072B2", fontweight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((1.0, 1.45), (9.45, 1.45), arrowstyle="-|>",
                                 linewidth=1.5, color="#0072B2"))
    ax.text(6.3, 0.65, "Retained-composite target", color="#A23E9A", fontweight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((5.7, 0.9), (9.45, 0.9), arrowstyle="-|>",
                                 linewidth=1.5, color="#A23E9A"))
    ax.text(7.95, 1.72, "VinePPO / TRePO:\nprefix credit axis",
            ha="center", va="bottom", fontsize=7.7, color="#7A5C00")
    ax.text(5.55, 3.0, "propensity mismatch", fontsize=7.5, color="#7A2E76", ha="center")
    ax.set_title("Figure 1 — Estimands and search-generated training data", pad=8)
    return fig


def figure_2(rows: list[dict[str, Any]]) -> plt.Figure:
    lookup = {
        (row["family"], row["estimator"]): row
        for row in rows
        if row["estimator"] in ESTIMATORS
    }
    require(
        set(lookup) == {(family, estimator) for family in FAMILIES for estimator in ESTIMATORS},
        "exact estimator matrix drift",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    metrics = (
        ("mean_relative_bias", "Signed relative bias", "A"),
        ("mean_gradient_relative_mse_batch_256", "Relative gradient MSE (batch 256)", "B"),
        ("mean_gradient_relative_mse_batch_1024", "Relative gradient MSE (batch 1,024)", "C"),
        ("mean_normalized_final_weight_ess", "Normalized final-weight ESS", "D"),
    )
    x = list(range(len(ESTIMATORS)))
    for ax, (metric, title, label) in zip(axes.ravel(), metrics, strict=True):
        for family in FAMILIES:
            values = [float(lookup[(family, estimator)][metric]) for estimator in ESTIMATORS]
            ax.plot(x, values, marker="o", linewidth=1.6, color=FAMILY_COLORS[family],
                    label=FAMILY_LABELS[family])
        ax.axhline(0.0, color="#555555", linewidth=0.7)
        ax.set_xticks(x, [ESTIMATOR_LABELS[name] for name in ESTIMATORS])
        ax.set_title(title)
        panel_label(ax, label)
    axes[0, 1].legend(loc="upper right")
    fig.suptitle("Figure 2 — Exact bias–variance–ESS evidence across all search families",
                 fontsize=11, fontweight="bold")
    return fig


def figure_3(rows: list[dict[str, Any]]) -> plt.Figure:
    require(
        {(row["condition"], row["family"]) for row in rows}
        == {(condition, family) for condition in CONDITION_ORDER for family in FAMILIES},
        "search-shift condition matrix drift",
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    markers = {condition: marker for condition, marker in zip(
        CONDITION_ORDER, ("o", "s", "D", "^", "v", "P", "X", "*", "<"), strict=True
    )}
    for row in rows:
        color = FAMILY_COLORS[row["family"]]
        marker = markers[row["condition"]]
        axes[0, 0].scatter(row["mean_critic_ranking_concordance"], row["mean_search_gain_vs_raw"],
                           color=color, marker=marker, s=32)
        axes[0, 1].scatter(row["mean_maximum_depth_state_action_tv"], row["mean_search_gain_vs_raw"],
                           color=color, marker=marker, s=32)
        axes[1, 0].scatter(row["mean_critic_ranking_concordance"], row["oracle_search_gain_headroom"],
                           color=color, marker=marker, s=32)
    axes[0, 0].set(xlabel="Critic ranking concordance", ylabel="Search gain vs raw Actor",
                   title="Ranking quality predicts gain direction")
    axes[0, 1].set(xlabel="Maximum-depth occupancy TV", ylabel="Search gain vs raw Actor",
                   title="Shift magnitude alone is insufficient")
    axes[1, 0].set(
        xlabel="Critic ranking concordance",
        ylabel="Headroom to exact-Critic search gain",
        title="Oracle search-gain headroom",
    )
    for family in FAMILIES:
        ordered = [next(row for row in rows if row["condition"] == condition and row["family"] == family)
                   for condition in CONDITION_ORDER]
        axes[1, 1].plot(range(len(ordered)), [row["mean_search_gain_vs_raw"] for row in ordered],
                        marker="o", linewidth=1.4, color=FAMILY_COLORS[family],
                        label=FAMILY_LABELS[family])
    axes[1, 1].axhline(0.0, color="#555555", linewidth=0.7)
    axes[1, 1].set_xticks(range(len(CONDITION_ORDER)),
                          [CONDITION_LABELS[name] for name in CONDITION_ORDER], rotation=40, ha="right")
    axes[1, 1].set(ylabel="Search gain vs raw Actor", title="Exact, learned, noisy, and inverted Critics")
    axes[1, 1].legend(loc="best")
    for label, ax in zip(("A", "B", "C", "D"), axes.ravel(), strict=True):
        panel_label(ax, label)
    family_handles = [plt.Line2D([], [], color=FAMILY_COLORS[f], marker="o", linestyle="",
                                 label=FAMILY_LABELS[f]) for f in FAMILIES]
    axes[0, 0].legend(handles=family_handles, loc="best")
    fig.suptitle("Figure 3 — When search-induced shift helps rather than hurts",
                 fontsize=11, fontweight="bold")
    return fig


def _contrast_axis(ax: plt.Axes, rows: list[dict[str, Any]], title: str) -> None:
    y = list(range(len(rows)))
    for index, row in enumerate(rows):
        estimate = 100.0 * float(row["estimate"])
        low = 100.0 * float(row["ci95_low"])
        high = 100.0 * float(row["ci95_high"])
        ax.errorbar(estimate, index, xerr=[[estimate - low], [high - estimate]], fmt="o",
                    color=FAMILY_COLORS[row["family"]], capsize=3, linewidth=1.3)
        for seed_row in row["per_seed"]:
            ax.scatter(100.0 * float(seed_row["difference"]), index, s=17, alpha=0.55,
                       facecolors="none", edgecolors=FAMILY_COLORS[row["family"]])
    ax.axvline(0.0, color="#555555", linewidth=0.8)
    ax.set_yticks(
        y,
        [row.get("display_label", FAMILY_LABELS[row["family"]]) for row in rows],
    )
    ax.set_xlabel("Accuracy difference (percentage points)")
    ax.set_title(title)


def figure_4(contrasts: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.4, 4.1),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.1)},
    )
    e3 = sorted([row for row in contrasts if row["contrast"] == "raw_E3_minus_E0"],
                key=lambda row: FAMILIES.index(row["family"]))
    e3_vs_k1 = next(
        row for row in contrasts if row["contrast"] == "raw_best_of_n_E3_minus_k1"
    )
    e3_vs_k1 = {**e3_vs_k1, "display_label": "Best-of-N E3 − K1"}
    e3 = [*e3, e3_vs_k1]
    e1 = sorted([row for row in contrasts if row["contrast"] == "raw_E1_minus_E0"],
                key=lambda row: FAMILIES.index(row["family"]))
    _contrast_axis(axes[0], e3, "E3 partial correction comparisons")
    _contrast_axis(axes[1], e1, "E1 full propensity − E0")
    raw = [row for row in evaluations if row["mode"] == "raw_k1"]
    lookup = {(int(row["seed"]), row["family"], row["estimator"]): row for row in raw}
    short_family_labels = {
        "best_of_n": "Best-of-N",
        "entropy_value": "Entropy",
        "value_value": "Value/PRM",
    }
    comparisons = [
        (
            family,
            short,
            family,
            "E0",
            f"{short_family_labels[family]} {short}−E0",
            "o" if short == "E3" else "s",
        )
        for family in FAMILIES
        for short in ("E1", "E3")
    ]
    comparisons.append(
        (
            "best_of_n",
            "E3",
            "k1",
            "E0",
            "Best-of-N E3−K1",
            "^",
        )
    )
    labels = []
    for position, (
        left_family,
        left_short,
        right_family,
        right_short,
        label,
        marker,
    ) in enumerate(comparisons):
        differences = []
        for seed in (99801, 99802, 99803):
            left = lookup[
                (seed, left_family, S3_ESTIMATORS[left_short])
            ]["answered_fraction"]
            right = lookup[
                (seed, right_family, S3_ESTIMATORS[right_short])
            ]["answered_fraction"]
            differences.append(100.0 * (float(left) - float(right)))
        axes[2].scatter(
            differences,
            [position] * 3,
            color=FAMILY_COLORS[left_family],
            marker=marker,
            alpha=0.7,
        )
        axes[2].plot(
            [mean(differences)] * 2,
            [position - 0.18, position + 0.18],
            color="#111111",
            linewidth=1.5,
        )
        labels.append(label)
    axes[2].axvline(0.0, color="#555555", linewidth=0.8)
    axes[2].set_yticks(range(len(labels)), labels)
    axes[2].set_xlabel("Answer-rate difference (percentage points)")
    axes[2].set_title("Per-seed answer-rate transfer")
    for label, ax in zip(("A", "B", "C"), axes, strict=True):
        panel_label(ax, label)
    fig.suptitle("Figure 4 — No-search K1 Actor transfer on sealed MATH-500",
                 fontsize=11, fontweight="bold")
    return fig


def figure_5(
    contrasts: list[dict[str, Any]],
    compute: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2))
    composite = sorted(
        [row for row in contrasts if row["contrast"] == "composite_E4_minus_E0"],
        key=lambda row: FAMILIES.index(row["family"]),
    )
    _contrast_axis(axes[0, 0], composite, "Retained-composite accuracy: E4 − E0")
    compute_lookup = {(row["family"], row["estimator"]): row for row in compute}
    x = list(range(len(FAMILIES)))
    width = 0.34
    token_ratio = [float(compute_lookup[(family, S3_ESTIMATORS["E4"])]["candidate_token_ratio_to_same_family_E0"])
                   for family in FAMILIES]
    time_ratio = [float(compute_lookup[(family, S3_ESTIMATORS["E4"])]["elapsed_ratio_to_same_family_E0"])
                  for family in FAMILIES]
    axes[0, 1].bar([value - width / 2 for value in x], token_ratio, width, label="Candidate-token ratio")
    axes[0, 1].bar([value + width / 2 for value in x], time_ratio, width, label="GPU-time ratio")
    axes[0, 1].axhline(1.0, color="#555555", linewidth=0.8)
    axes[0, 1].set_xticks(x, [FAMILY_LABELS[f] for f in FAMILIES], rotation=18, ha="right")
    axes[0, 1].set_ylabel("E4 / E0 ratio")
    axes[0, 1].set_title("Matched-compute cost")
    axes[0, 1].legend()
    composite_evals = [row for row in evaluations if row["mode"] == "composite"]
    eval_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in composite_evals:
        eval_groups[(row["family"], row["estimator"])].append(row)
    diversity_metrics = (
        ("mean_unique_candidates_when_expanded", "Unique candidates"),
        ("identical_expanded_fraction", "Identical fraction"),
        ("exact_token_pair_fraction", "Exact-token pairs"),
        ("mean_pairwise_prefix_divergence_fraction", "Prefix divergence"),
    )
    for family in FAMILIES:
        differences = []
        for metric, _ in diversity_metrics:
            e4_by_seed = {
                row["seed"]: row[metric]
                for row in eval_groups[(family, S3_ESTIMATORS["E4"])]
            }
            e0_by_seed = {
                row["seed"]: row[metric]
                for row in eval_groups[(family, S3_ESTIMATORS["E0"])]
            }
            paired = [
                float(e4_by_seed[seed]) - float(e0_by_seed[seed])
                for seed in sorted(set(e4_by_seed) & set(e0_by_seed))
                if finite(e4_by_seed[seed]) and finite(e0_by_seed[seed])
            ]
            differences.append(mean(paired) if paired else float("nan"))
        axes[1, 0].plot(range(len(diversity_metrics)), differences, marker="o",
                        color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
    axes[1, 0].axhline(0.0, color="#555555", linewidth=0.8)
    axes[1, 0].set_xticks(range(len(diversity_metrics)), [label for _, label in diversity_metrics],
                          rotation=24, ha="right")
    axes[1, 0].set_ylabel("E4 − E0 (surface diagnostic)")
    axes[1, 0].set_title("Surface diversity (paired seeds with expansions)")
    axes[1, 0].legend()
    inference_cells = [
        (family, estimator_key)
        for family in FAMILIES
        for estimator_key in ("E0", "E4")
    ]
    inference_x = list(range(len(inference_cells)))
    target_expansion = []
    source_expansion = []
    evaluation_expansion = []
    token_multiplier = []
    for family, estimator_key in inference_cells:
        rows = eval_groups[(family, S3_ESTIMATORS[estimator_key])]
        target_expansion.append(
            mean(float(row["registered_expansion_rate"]) for row in rows)
        )
        source_expansion.append(
            mean(float(row["source_rollout_expansion_rate"]) for row in rows)
        )
        evaluation_expansion.append(
            mean(float(row["expansion_rate"]) for row in rows)
        )
        token_multiplier.append(
            mean(float(row["candidate_token_multiplier_vs_selected"]) for row in rows)
        )
    axes[1, 1].bar(
        [value - width / 2 for value in inference_x],
        source_expansion,
        width,
        label="Round-2 source",
    )
    axes[1, 1].bar(
        [value + width / 2 for value in inference_x],
        evaluation_expansion,
        width,
        label="Sealed evaluation",
    )
    axes[1, 1].plot(
        inference_x,
        target_expansion,
        marker="_",
        linestyle="none",
        markersize=12,
        color="#555555",
        label="Registered target",
    )
    secondary = axes[1, 1].twinx()
    secondary.plot(
        inference_x,
        token_multiplier,
        marker="o",
        color="#0072B2",
        label="Candidate/selected tokens",
    )
    axes[1, 1].set_xticks(
        inference_x,
        [f"{FAMILY_LABELS[family]}\n{key}" for family, key in inference_cells],
        rotation=18,
        ha="right",
    )
    axes[1, 1].set_ylabel("Expansion rate")
    secondary.set_ylabel("Candidate / selected completion-token ratio")
    axes[1, 1].set_title("Target, source, and evaluation compute")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[1, 1].legend(
        handles + handles2,
        labels + labels2,
        loc="upper right",
        fontsize=5.2,
        framealpha=0.9,
    )
    for label, ax in zip(("A", "B", "C", "D"), axes.ravel(), strict=True):
        panel_label(ax, label)
    fig.suptitle("Figure 5 — Retained-search objective, compute, and surface diversity",
                 fontsize=11, fontweight="bold")
    return fig


def render_all(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    configure_style()
    table = validate_tables(payload)
    expected_names = {
        f"{stem}.{extension}"
        for stem in FIGURE_NAMES
        for extension in ("png", "pdf")
    }
    if output_dir.is_dir():
        unexpected = {
            path.name
            for path in output_dir.iterdir()
            if path.is_file()
            and path.suffix in {".png", ".pdf"}
            and path.name not in expected_names
        }
        require(not unexpected, f"unexpected figure files in output directory: {sorted(unexpected)}")
    figures = (
        figure_1(),
        figure_2(table["exact_estimator_bias_variance"]),
        figure_3(table["exact_search_shift"]),
        figure_4(table["s3_contrast_rows"], table["s3_evaluation_metrics"]),
        figure_5(
            table["s3_contrast_rows"],
            table["s3_compute_by_family_estimator"],
            table["s3_evaluation_metrics"],
        ),
    )
    outputs: list[Path] = []
    for fig, name in zip(figures, FIGURE_NAMES, strict=True):
        outputs.extend(save_figure(fig, output_dir, name))
    require({path.name for path in outputs} == expected_names, "rendered figure set drift")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    require(args.evidence_tables.is_file(), f"missing evidence tables: {args.evidence_tables}")
    payload = json.loads(args.evidence_tables.read_text())
    require(isinstance(payload, dict), "evidence tables are not a JSON object")
    outputs = render_all(payload, args.output_dir)
    manifest = {
        "schema": "cgcs-search-policy-mismatch-iclr-figures-v1",
        "status": "evidence_complete",
        "source_evidence_tables_sha256": sha256_file(args.evidence_tables),
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
