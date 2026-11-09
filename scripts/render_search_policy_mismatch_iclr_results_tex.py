#!/usr/bin/env python3
"""Render the terminal S3 manuscript block from evidence-locked tables.

The renderer never reads an in-progress result tree.  Its only input is the
terminal ICLR evidence-table JSON emitted after the frozen analysis and the
independent archive audit both pass.  This prevents manual transcription of
interim accuracies into the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


FAMILIES = ("best_of_n", "entropy_value", "value_value")
FAMILY_LABELS = {
    "best_of_n": r"best-of-$N$",
    "entropy_value": r"entropy--Value",
    "value_value": r"Value/PRM",
}
CONTRASTS = (
    "raw_E3_minus_E0",
    "raw_E1_minus_E0",
    "raw_best_of_n_E3_minus_k1",
    "composite_E4_minus_E0",
)
CONTRAST_LABELS = {
    "raw_E3_minus_E0": r"raw \Ethree--\Ezero",
    "raw_E1_minus_E0": r"raw \Eone--\Ezero",
    "raw_best_of_n_E3_minus_k1": r"raw \Ethree--K1",
    "composite_E4_minus_E0": r"composite \Efour--\Ezero",
}
EXPECTED_IDENTITIES = {
    *((name, family) for name in CONTRASTS[:2] for family in FAMILIES),
    ("raw_best_of_n_E3_minus_k1", "best_of_n"),
    *(("composite_E4_minus_E0", family) for family in FAMILIES),
}
S3_ESTIMATORS = (
    "E0_standard_actor_ratio",
    "E1_trajectory_propensity",
    "E3_step_cumulative_lambda_0.75",
    "E4_extended_candidate_set",
)
ESTIMATOR_LABELS = {
    S3_ESTIMATORS[0]: r"\Ezero",
    S3_ESTIMATORS[1]: r"\Eone",
    S3_ESTIMATORS[2]: r"\Ethree",
    S3_ESTIMATORS[3]: r"\Efour",
}
SEEDS = (99801, 99802, 99803)
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


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file for SHA-256: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(
    payload: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    require(
        payload.get("schema") == "cgcs-search-policy-mismatch-iclr-tables-v1",
        "ICLR evidence-table schema mismatch",
    )
    require(payload.get("status") == "evidence_complete", "evidence tables are not terminal")
    provenance = payload.get("provenance")
    require(isinstance(provenance, dict), "evidence-table provenance missing")
    digests = [provenance.get("renderer_sha256")]
    input_hashes = provenance.get("input_sha256")
    require(
        isinstance(input_hashes, dict) and set(input_hashes) == EVIDENCE_INPUTS,
        "evidence-table input provenance matrix drift",
    )
    digests.extend(input_hashes.values())
    require(
        all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in digests),
        "invalid evidence-table SHA-256 provenance",
    )

    table_2 = payload.get("table_2_cross_stage")
    require(isinstance(table_2, dict), "cross-stage evidence table missing")
    rows = table_2.get("s3_contrast_rows")
    require(isinstance(rows, list) and len(rows) == 10, "S3 contrast matrix must contain ten rows")
    identities = {(row.get("contrast"), row.get("family")) for row in rows}
    require(identities == EXPECTED_IDENTITIES, "S3 contrast identity matrix drift")
    for row in rows:
        identity = f"{row.get('contrast')}/{row.get('family')}"
        require(
            all(finite(row.get(key)) for key in ("estimate", "ci95_low", "ci95_high", "holm_adjusted_p")),
            f"non-finite S3 contrast {identity}",
        )
        require(
            float(row["ci95_low"]) <= float(row["estimate"]) <= float(row["ci95_high"]),
            f"S3 contrast estimate outside interval for {identity}",
        )
        wlt = row.get("pooled_wins_losses_ties")
        require(
            isinstance(wlt, dict)
            and all(isinstance(wlt.get(key), int) for key in ("wins", "losses", "ties"))
            and sum(wlt[key] for key in ("wins", "losses", "ties")) == 600,
            f"invalid pooled wins/losses/ties for {identity}",
        )
        per_seed = row.get("per_seed")
        require(
            isinstance(per_seed, list)
            and {item.get("seed") for item in per_seed} == set(SEEDS),
            f"invalid per-seed rows for {identity}",
        )
        for item in per_seed:
            require(finite(item.get("difference")), f"non-finite per-seed difference for {identity}")
            require(
                all(isinstance(item.get(key), int) for key in ("wins", "losses", "ties"))
                and sum(item[key] for key in ("wins", "losses", "ties")) == 200,
                f"invalid per-seed wins/losses/ties for {identity}",
            )

    decomposition = table_2.get("s3_contrast_metric_decomposition")
    require(
        isinstance(decomposition, list) and len(decomposition) == 30,
        "S3 contrast decomposition must contain 30 rows",
    )
    expected_decomposition = {
        (contrast, family, seed)
        for contrast, family in EXPECTED_IDENTITIES
        for seed in SEEDS
    }
    decomposition_identities = {
        (row.get("contrast"), row.get("family"), row.get("seed"))
        for row in decomposition
    }
    require(
        decomposition_identities == expected_decomposition
        and len(decomposition_identities) == len(decomposition),
        "S3 contrast decomposition identity matrix drift",
    )
    paired_differences = {
        (row["contrast"], row["family"], item["seed"]): item["difference"]
        for row in rows
        for item in row["per_seed"]
    }
    for row in decomposition:
        identity = (row["contrast"], row["family"], row["seed"])
        require(
            finite(row.get("accuracy_difference"))
            and finite(row.get("answer_rate_difference"))
            and math.isclose(
                float(row["accuracy_difference"]),
                float(paired_differences[identity]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
            f"invalid S3 contrast decomposition for {identity}",
        )
        lhs_conditional = row.get("lhs_accuracy_given_parseable_final")
        rhs_conditional = row.get("rhs_accuracy_given_parseable_final")
        conditional_difference = row.get("conditional_accuracy_difference")
        require(
            (
                (lhs_conditional is None or rhs_conditional is None)
                and conditional_difference is None
            )
            or (
                finite(lhs_conditional)
                and finite(rhs_conditional)
                and finite(conditional_difference)
                and math.isclose(
                    float(conditional_difference),
                    float(lhs_conditional) - float(rhs_conditional),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            f"invalid conditional-accuracy decomposition for {identity}",
        )

    compute = table_2.get("s3_compute_by_family_estimator")
    require(isinstance(compute, list) and len(compute) == 12, "S3 compute matrix must contain twelve rows")
    require(
        {(row.get("family"), row.get("estimator")) for row in compute}
        == {(family, estimator) for family in FAMILIES for estimator in S3_ESTIMATORS},
        "S3 compute identity matrix drift",
    )
    require(
        all(finite(row.get("minimum_ess")) and finite(row.get("maximum_clip_fraction")) for row in compute),
        "non-finite S3 compute diagnostics",
    )

    evaluations = table_2.get("s3_evaluation_metrics")
    require(
        isinstance(evaluations, list) and len(evaluations) == 75,
        "S3 evaluation matrix must contain 75 rows",
    )
    expected_evaluations = {
        ("raw_k1", seed, "k1", S3_ESTIMATORS[0])
        for seed in SEEDS
    } | {
        (mode, seed, family, estimator)
        for mode in ("raw_k1", "composite")
        for seed in SEEDS
        for family in FAMILIES
        for estimator in S3_ESTIMATORS
    }
    require(
        {
            (
                row.get("mode"),
                row.get("seed"),
                row.get("family"),
                row.get("estimator"),
            )
            for row in evaluations
        }
        == expected_evaluations,
        "S3 evaluation identity matrix drift",
    )
    for row in evaluations:
        require(
            all(
                finite(row.get(key))
                for key in ("accuracy", "answered_fraction", "elapsed_seconds")
            ),
            "non-finite S3 evaluation metric",
        )
        diagnostics = row.get("finalization_diagnostics")
        require(isinstance(diagnostics, dict), "S3 finalization diagnostics missing")
        valid = diagnostics.get("parseable_final_answers")
        unanswered = diagnostics.get("unanswered_examples")
        no_final = diagnostics.get("unanswered_no_final_like_marker")
        noncanonical = diagnostics.get(
            "unanswered_noncanonical_final_like_marker"
        )
        conditional = diagnostics.get("accuracy_given_parseable_final")
        require(
            all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (valid, unanswered, no_final, noncanonical)
            )
            and valid + unanswered == 200
            and no_final + noncanonical == unanswered,
            "invalid S3 finalization counts",
        )
        require(
            math.isclose(
                valid / 200,
                float(row["answered_fraction"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and diagnostics.get("all_unanswered_exhaust_max_steps") is True,
            "S3 finalization diagnostics disagree with answer rate",
        )
        require(
            0.0 <= float(row["accuracy"]) <= float(row["answered_fraction"]),
            "S3 accuracy exceeds answer rate",
        )
        require(
            (
                valid == 0
                and conditional is None
                and math.isclose(
                    float(row["accuracy"]), 0.0, rel_tol=0.0, abs_tol=1e-12
                )
            )
            or (
                valid > 0
                and finite(conditional)
                and math.isclose(
                    float(conditional),
                    float(row["accuracy"]) / float(row["answered_fraction"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ),
            "S3 conditional accuracy mismatch",
        )

    table_3 = payload.get("table_3_reproducibility")
    require(isinstance(table_3, dict), "reproducibility table missing")
    stability = table_3.get("stability_gate_audit")
    require(isinstance(stability, dict), "stability audit missing")
    required_stability = (
        "clip_violation_count",
        "maximum_clip_fraction",
        "maximum_kl_proxy",
        "e3_ess_exceeds_e1_in_all_18_comparisons",
    )
    require(
        all(key in stability for key in required_stability)
        and isinstance(stability["clip_violation_count"], int)
        and 0 <= stability["clip_violation_count"] <= 72
        and finite(stability["maximum_clip_fraction"])
        and finite(stability["maximum_kl_proxy"])
        and isinstance(stability["e3_ess_exceeds_e1_in_all_18_comparisons"], bool),
        "invalid stability audit",
    )
    gate_logic = table_3.get("frozen_gate_logic_audit")
    require(isinstance(gate_logic, dict), "frozen gate-logic audit missing")
    require(
        isinstance(gate_logic.get("independent_corrected_overall_gate"), bool),
        "independent corrected overall gate missing",
    )
    manual = table_3.get("manual_compute_gate")
    require(
        isinstance(manual, dict) and manual.get("verdict") in {"PASS", "FAIL"},
        "manual compute gate must be explicitly reviewed",
    )
    require(
        isinstance(manual.get("rationale"), str)
        and len(manual["rationale"].strip()) >= 20,
        "manual compute-review rationale missing",
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
            isinstance(manual.get(key), str)
            and re.fullmatch(r"[0-9a-f]{64}", manual[key]),
            f"invalid manual compute-review {key}",
        )
    return rows, stability, evaluations, decomposition


def sort_key(row: dict[str, Any]) -> tuple[int, int]:
    return CONTRASTS.index(row["contrast"]), FAMILIES.index(row["family"])


def signed_pp(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def evaluation_configuration_label(family: str, estimator: str) -> str:
    if family == "k1":
        return "K1 control"
    return f"{FAMILY_LABELS[family]} {ESTIMATOR_LABELS[estimator]}"


def append_per_seed_evaluation_table(
    lines: list[str],
    evaluations: list[dict[str, Any]],
    *,
    mode: str,
) -> None:
    mode_rows = [row for row in evaluations if row["mode"] == mode]
    configurations = sorted(
        {(row["family"], row["estimator"]) for row in mode_rows},
        key=lambda item: (
            -1 if item[0] == "k1" else FAMILIES.index(item[0]),
            S3_ESTIMATORS.index(item[1]),
        ),
    )
    expected = 13 if mode == "raw_k1" else 12
    require(
        len(configurations) == expected,
        f"{mode} evaluation configuration count drift",
    )
    lookup = {
        (row["seed"], row["family"], row["estimator"]): row
        for row in mode_rows
    }
    title = "no-search raw-K1" if mode == "raw_k1" else "retained-composite"
    label = "raw" if mode == "raw_k1" else "composite"
    lines.extend(
        [
            "",
            r"\begin{table}[H]",
            (
                f"  \\caption{{Per-training-seed {title} accuracy (A), strict "
                r"final-answer extraction rate (R), and accuracy conditional "
                r"on a valid final (C), all in percent.}"
            ),
            f"  \\label{{tab:s3-{label}-per-seed-levels}}",
            r"  \centering",
            r"  \scriptsize",
            r"  \resizebox{\linewidth}{!}{%",
            r"  \begin{tabular}{lrrrrrrrrr}",
            r"    \toprule",
            r"    & \multicolumn{3}{c}{99801} & \multicolumn{3}{c}{99802} & \multicolumn{3}{c}{99803} \\",
            r"    Configuration & A & R & C & A & R & C & A & R & C \\",
            r"    \midrule",
        ]
    )
    for family, estimator in configurations:
        values: list[str] = []
        for seed in SEEDS:
            row = lookup[(seed, family, estimator)]
            conditional = row["finalization_diagnostics"][
                "accuracy_given_parseable_final"
            ]
            values.extend(
                [
                    percent(float(row["accuracy"])),
                    percent(float(row["answered_fraction"])),
                    "NA" if conditional is None else percent(float(conditional)),
                ]
            )
        lines.append(
            f"    {evaluation_configuration_label(family, estimator)} & "
            + " & ".join(values)
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
        ]
    )


def append_contrast_decomposition_table(
    lines: list[str], decomposition: list[dict[str, Any]]
) -> None:
    lookup = {
        (row["contrast"], row["family"], int(row["seed"])): row
        for row in decomposition
    }
    lines.extend(
        [
            "",
            r"\begin{table}[H]",
            r"  \caption{Per-training-seed decomposition of each registered overall-accuracy difference ($\Delta A$) into strict final-answer extraction-rate difference ($\Delta R$) and descriptive accuracy-given-answer difference ($\Delta C$), in percentage points. NA denotes an undefined conditional accuracy.}",
            r"  \label{tab:s3-terminal-finalization-decomposition}",
            r"  \centering",
            r"  \tiny",
            r"  \resizebox{\linewidth}{!}{%",
            r"  \begin{tabular}{llrrrrrrrrr}",
            r"    \toprule",
            r"    & & \multicolumn{3}{c}{99801} & \multicolumn{3}{c}{99802} & \multicolumn{3}{c}{99803} \\",
            r"    Contrast & Family & $\Delta A$ & $\Delta R$ & $\Delta C$ & $\Delta A$ & $\Delta R$ & $\Delta C$ & $\Delta A$ & $\Delta R$ & $\Delta C$ \\",
            r"    \midrule",
        ]
    )
    for contrast in CONTRASTS:
        families = (
            ("best_of_n",)
            if contrast == "raw_best_of_n_E3_minus_k1"
            else FAMILIES
        )
        for family in families:
            values: list[str] = []
            for seed in SEEDS:
                row = lookup[(contrast, family, seed)]
                conditional = row["conditional_accuracy_difference"]
                values.extend(
                    [
                        signed_pp(float(row["accuracy_difference"])),
                        signed_pp(float(row["answer_rate_difference"])),
                        "NA" if conditional is None else signed_pp(float(conditional)),
                    ]
                )
            lines.append(
                f"    {CONTRAST_LABELS[contrast]} & {FAMILY_LABELS[family]} & "
                + " & ".join(values)
                + r" \\"
            )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
        ]
    )


def render_tex(payload: dict[str, Any]) -> str:
    rows, stability, evaluations, decomposition = validate(payload)
    rows = sorted(rows, key=sort_key)
    table_3 = payload["table_3_reproducibility"]
    overall = table_3["frozen_gate_logic_audit"]["independent_corrected_overall_gate"]
    manual = table_3.get("manual_compute_gate", {}).get("verdict", "UNSET")
    e3_rows = [row for row in rows if row["contrast"] == "raw_E3_minus_E0"]
    e1_rows = [row for row in rows if row["contrast"] == "raw_E1_minus_E0"]
    composite_rows = [row for row in rows if row["contrast"] == "composite_E4_minus_E0"]
    bestof = next(row for row in rows if row["contrast"] == "raw_best_of_n_E3_minus_k1")

    lines = [
        "% Generated by render_search_policy_mismatch_iclr_results_tex.py.",
        "% Do not edit terminal numerical claims by hand.",
        f"% Manual compute-review receipt SHA-256: {table_3['manual_compute_gate']['receipt_sha256']}",
        r"\paragraph{Terminal promotion gate.}",
        (
            f"The independent audit finds {stability['clip_violation_count']}/72 "
            "search-training branches at or above the frozen 0.20 sequence-ratio "
            f"clip ceiling (maximum {float(stability['maximum_clip_fraction']):.4f}); "
            f"the maximum KL proxy is {float(stability['maximum_kl_proxy']):.4f}. "
            f"The corrected overall promotion gate is {'PASS' if overall else 'FAIL'} "
            f"and the independent compute verdict is {manual}. "
            "The review is hash-bound to its recorder and preliminary-audit "
            "preparer, the terminal analysis and report, the 12-row compute "
            "table, and GPU-session evidence. "
            "A failed gate does not authorize scale-up."
        ),
        "",
        r"\begin{table}[H]",
        r"  \caption{Terminal paired S3 contrasts on the sealed 200-problem MATH-500 scalar partition. Differences and clustered intervals are percentage points; W/L/T pools all three seeds.}",
        r"  \label{tab:s3-terminal-contrasts}",
        r"  \centering",
        r"  \small",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{llrrrr}",
        r"    \toprule",
        r"    Contrast & Family & Diff. & 95\% CI & W/L/T & Holm $p$ \\",
        r"    \midrule",
    ]
    for row in rows:
        wlt = row["pooled_wins_losses_ties"]
        lines.append(
            f"    {CONTRAST_LABELS[row['contrast']]} & {FAMILY_LABELS[row['family']]} & "
            f"{signed_pp(float(row['estimate']))} & "
            f"[{signed_pp(float(row['ci95_low']))}, {signed_pp(float(row['ci95_high']))}] & "
            f"{wlt['wins']}/{wlt['losses']}/{wlt['ties']} & "
            f"{float(row['holm_adjusted_p']):.4g} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}}",
            r"\end{table}",
            "",
            r"\begin{table}[H]",
            r"  \caption{Per-training-seed accuracy differences (percentage points) for every preregistered S3 contrast.}",
            r"  \label{tab:s3-terminal-per-seed}",
            r"  \centering",
            r"  \small",
            r"  \begin{tabular}{llrrr}",
            r"    \toprule",
            r"    Contrast & Family & 99801 & 99802 & 99803 \\",
            r"    \midrule",
        ]
    )
    for row in rows:
        per_seed = {int(item["seed"]): float(item["difference"]) for item in row["per_seed"]}
        lines.append(
            f"    {CONTRAST_LABELS[row['contrast']]} & {FAMILY_LABELS[row['family']]} & "
            + " & ".join(signed_pp(per_seed[seed]) for seed in SEEDS)
            + r" \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    append_per_seed_evaluation_table(lines, evaluations, mode="raw_k1")
    append_per_seed_evaluation_table(lines, evaluations, mode="composite")
    append_contrast_decomposition_table(lines, decomposition)
    lines.extend(
        [
            "",
            r"\paragraph{Precommitted interpretation.}",
            (
                f"Nonnegative clustered intervals occur for "
                f"{sum(float(row['ci95_low']) >= 0.0 for row in e3_rows)}/3 raw "
                f"\\Ethree--\\Ezero{{}} families, "
                f"{sum(float(row['ci95_low']) >= 0.0 for row in e1_rows)}/3 raw "
                f"\\Eone--\\Ezero{{}} families, and "
                f"{sum(float(row['ci95_low']) >= 0.0 for row in composite_rows)}/3 "
                f"composite \\Efour--\\Ezero{{}} families.  The preregistered best-of-$N$ "
                f"raw \\Ethree--K1 contrast is {signed_pp(float(bestof['estimate']))} pp "
                f"with clustered 95\\% CI "
                f"[{signed_pp(float(bestof['ci95_low']))}, {signed_pp(float(bestof['ci95_high']))}]. "
                f"E3 ESS exceeds E1 in all 18 matched comparisons: "
                f"{'yes' if stability['e3_ess_exceeds_e1_in_all_18_comparisons'] else 'no'}. "
                "These descriptive outcomes do not override the frozen promotion gate."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-tables", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.evidence_tables.read_text())
    require(isinstance(payload, dict), "evidence tables are not a JSON object")
    rendered = render_tex(payload)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(rendered)
    manifest = {
        "schema": "cgcs-search-policy-mismatch-iclr-results-tex-v1",
        "status": "evidence_complete",
        "renderer_sha256": sha256_file(Path(__file__).resolve()),
        "evidence_tables_sha256": sha256_file(args.evidence_tables),
        "terminal_tex_sha256": sha256_file(args.output_tex),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
