#!/usr/bin/env python3
"""Build the deterministic, post-review S3 evidence archive manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "cgcs-search-policy-mismatch-s3-final-archive-manifest-v1"
FIGURE_STEMS = (
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
MANUSCRIPT_REVIEW_CHECKS = {
    "compilation_clean",
    "tables_unclipped",
    "figures_legible",
    "labels_nonoverlapping",
    "content_within_text_area",
}


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing final archive artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON final archive artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def terminal_file_map(project_root: Path, root: Path) -> dict[str, Path]:
    project_files = {
        "project/readme": project_root / "README.md",
        "project/pyproject.toml": project_root / "pyproject.toml",
        "project/uv.lock": project_root / "uv.lock",
        "project/data/math500_scalar.jsonl": project_root / "data/math500_scalar.jsonl",
        "project/data/math500_scalar_manifest.json": (
            project_root / "data/math500_scalar_manifest.json"
        ),
        "project/mechanism/protocol": (
            project_root
            / "results/math_value_ppo/JOINT_VALUE_MECHANISM_REPLICATION_PROTOCOL.md"
        ),
        "project/mechanism/analysis": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/analysis.json"
        ),
        "project/mechanism/report": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/"
            "JOINT_VALUE_MECHANISM_REPLICATION_REPORT.md"
        ),
        "project/mechanism/archive_receipt": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/ARCHIVE_RECEIPT.md"
        ),
        "project/mechanism/source_snapshot": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/"
            "source_snapshot/math_joint_value_ppo.py"
        ),
        "project/mechanism/source_snapshot_recovery": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/"
            "SOURCE_SNAPSHOT_RECOVERY.md"
        ),
        "project/mechanism/manifest_verifier": (
            project_root
            / "scripts/verify_math_joint_value_replication_manifest.py"
        ),
        "project/mechanism/runner": (
            project_root
            / "scripts/run_math_joint_value_mechanism_replication_rsaturn.sh"
        ),
        "project/mechanism/analyzer": (
            project_root
            / "scripts/analyze_math_joint_value_mechanism_replication.py"
        ),
        "project/exact/protocol": (
            project_root / "results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"
        ),
        "project/exact/freeze_receipt": (
            project_root / "results/search_policy_mismatch/FREEZE_RECEIPT.md"
        ),
        "project/exact/confirmation_seeds": (
            project_root / "results/search_policy_mismatch/CONFIRMATION_SEEDS.json"
        ),
        "project/exact/analysis_schema": (
            project_root / "results/search_policy_mismatch/ANALYSIS_SCHEMA.json"
        ),
        "project/exact/prefreeze_checklist": (
            project_root / "results/search_policy_mismatch/PREFREEZE_CHECKLIST.md"
        ),
        "project/exact/search_family_conformance_audit": (
            project_root
            / "results/search_policy_mismatch/SEARCH_FAMILY_CONFORMANCE_AUDIT.md"
        ),
        "project/exact/s1_primary": (
            project_root / "results/search_policy_mismatch/S1_PRIMARY.json"
        ),
        "project/exact/s1_report": (
            project_root / "results/search_policy_mismatch/S1_PRIMARY_REPORT.md"
        ),
        "project/exact/s2_confirmation": (
            project_root / "results/search_policy_mismatch/S2_CONFIRMATION.json"
        ),
        "project/exact/s2_development": (
            project_root / "results/search_policy_mismatch/S2_DEVELOPMENT.json"
        ),
        "project/exact/s2_report": (
            project_root / "results/search_policy_mismatch/S2_BIAS_VARIANCE_REPORT.md"
        ),
        "project/exact/gate_confirmation": (
            project_root / "results/search_policy_mismatch/GATE_CONFIRMATION.json"
        ),
        "project/exact/gate_development": (
            project_root / "results/search_policy_mismatch/GATE_DEVELOPMENT.json"
        ),
        "project/exact/gate_feature_sufficiency": (
            project_root
            / "results/search_policy_mismatch/GATE_FEATURE_SUFFICIENCY.json"
        ),
        "project/exact/gate_report": (
            project_root / "results/search_policy_mismatch/GATE_SEARCH_FAMILY_REPORT.md"
        ),
        "project/exact/critic_confirmation": (
            project_root / "results/search_policy_mismatch/CRITIC_CONFIRMATION.json"
        ),
        "project/exact/critic_report": (
            project_root / "results/search_policy_mismatch/CRITIC_QUALITY_REPORT.md"
        ),
        "project/exact/estimator_confirmation": (
            project_root
            / "results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION.json"
        ),
        "project/exact/estimator_development": (
            project_root
            / "results/search_policy_mismatch/ESTIMATOR_MATRIX_DEVELOPMENT.json"
        ),
        "project/exact/estimator_development_report": (
            project_root
            / "results/search_policy_mismatch/ESTIMATOR_MATRIX_DEVELOPMENT_REPORT.md"
        ),
        "project/exact/estimator_analysis": (
            project_root
            / "results/search_policy_mismatch/"
            "ESTIMATOR_MATRIX_CONFIRMATION_ANALYSIS.json"
        ),
        "project/exact/estimator_report": (
            project_root
            / "results/search_policy_mismatch/"
            "ESTIMATOR_MATRIX_CONFIRMATION_REPORT.md"
        ),
        "project/exact/s1_runner": (
            project_root / "scripts/run_exact_truth_s1_primary.py"
        ),
        "project/exact/seed_generator": (
            project_root / "scripts/generate_exact_truth_confirmation_seeds.py"
        ),
        "project/exact/protocol_linter": (
            project_root / "scripts/lint_exact_truth_protocol.py"
        ),
        "project/exact/s2_development_runner": (
            project_root / "scripts/run_exact_truth_s2_development.py"
        ),
        "project/exact/s2_runner": (
            project_root / "scripts/run_exact_truth_s2_confirmation.py"
        ),
        "project/exact/gate_development_runner": (
            project_root / "scripts/run_exact_truth_gate_development.py"
        ),
        "project/exact/gate_feature_sufficiency_runner": (
            project_root / "scripts/run_exact_truth_gate_feature_sufficiency.py"
        ),
        "project/exact/gate_runner": (
            project_root / "scripts/run_exact_truth_gate_confirmation.py"
        ),
        "project/exact/critic_runner": (
            project_root / "scripts/run_exact_truth_critic_confirmation.py"
        ),
        "project/exact/estimator_runner": (
            project_root / "scripts/run_exact_truth_estimator_matrix.py"
        ),
        "project/exact/estimator_analyzer": (
            project_root / "scripts/analyze_exact_truth_estimator_matrix.py"
        ),
        "project/protocol": (
            project_root / "results/search_policy_mismatch/S3_LLM_TRANSFER_PROTOCOL.md"
        ),
        "project/postrun_checklist": (
            project_root
            / "results/search_policy_mismatch/S3_POSTRUN_ARCHIVE_CHECKLIST.md"
        ),
        "project/stage_gates": (
            project_root / "results/search_policy_mismatch/STAGE_GATES.md"
        ),
        "project/claim_evidence_matrix": (
            project_root
            / "results/search_policy_mismatch/ICLR_CLAIM_EVIDENCE_MATRIX.md"
        ),
        "project/data_exposure_ledger": (
            project_root / "results/search_policy_mismatch/DATA_EXPOSURE_LEDGER.md"
        ),
        "project/related_work_audit": (
            project_root / "results/search_policy_mismatch/RELATED_WORK_AUDIT.md"
        ),
        "project/manuscript_outline": (
            project_root / "results/search_policy_mismatch/ICLR_MANUSCRIPT_OUTLINE.md"
        ),
        "project/postrun_answer_rate_sensitivity_design": (
            project_root
            / "results/search_policy_mismatch/"
            "S3_POSTRUN_ANSWER_RATE_SENSITIVITY_DESIGN.md"
        ),
        "project/runner": (
            project_root / "scripts/run_search_policy_mismatch_s3_rsaturn.sh"
        ),
        "project/frozen_analyzer": (
            project_root / "scripts/analyze_search_policy_mismatch_s3.py"
        ),
        "project/stratified_diagnostic_analyzer": (
            project_root / "scripts/analyze_search_policy_mismatch_s3_strata.py"
        ),
        "project/archive_auditor": (
            project_root / "scripts/audit_search_policy_mismatch_s3_archive.py"
        ),
        "project/compute_review_recorder": (
            project_root / "scripts/record_search_policy_mismatch_s3_compute_review.py"
        ),
        "project/compute_review_preparer": (
            project_root / "scripts/prepare_search_policy_mismatch_s3_compute_review.sh"
        ),
        "project/evidence_table_renderer": (
            project_root / "scripts/render_search_policy_mismatch_iclr_tables.py"
        ),
        "project/figure_renderer": (
            project_root / "scripts/render_search_policy_mismatch_iclr_figures.py"
        ),
        "project/terminal_results_renderer": (
            project_root / "scripts/render_search_policy_mismatch_iclr_results_tex.py"
        ),
        "project/manuscript_review_recorder": (
            project_root
            / "scripts/record_search_policy_mismatch_iclr_manuscript_review.py"
        ),
        "project/final_manifest_builder": Path(__file__).resolve(),
        "project/local_archive_verifier": (
            project_root / "scripts/verify_search_policy_mismatch_s3_archive.sh"
        ),
        "project/remote_postrun_verifier": (
            project_root
            / "scripts/verify_search_policy_mismatch_s3_remote_postrun.sh"
        ),
        "project/runtime/acbpo/__init__.py": (
            project_root / "src/acbpo/__init__.py"
        ),
        "project/runtime/acbpo/candidate_policy.py": (
            project_root / "src/acbpo/candidate_policy.py"
        ),
        "project/runtime/acbpo/layered_mdp.py": (
            project_root / "src/acbpo/layered_mdp.py"
        ),
        "project/runtime/acbpo/search_policy_mismatch.py": (
            project_root / "src/acbpo/search_policy_mismatch.py"
        ),
        "project/runtime/acbpo/search_gate_study.py": (
            project_root / "src/acbpo/search_gate_study.py"
        ),
        "project/runtime/acbpo/critic_quality.py": (
            project_root / "src/acbpo/critic_quality.py"
        ),
        "project/runtime/acbpo/search_mismatch_estimators.py": (
            project_root / "src/acbpo/search_mismatch_estimators.py"
        ),
        "project/runtime/acbpo/search_family_matrix.py": (
            project_root / "src/acbpo/search_family_matrix.py"
        ),
        "project/runtime/acbpo/search_composite_estimators.py": (
            project_root / "src/acbpo/search_composite_estimators.py"
        ),
        "project/runtime/acbpo/math_joint_value_ppo.py": (
            project_root / "src/acbpo/math_joint_value_ppo.py"
        ),
        "project/runtime/acbpo/math_search_policy_mismatch.py": (
            project_root / "src/acbpo/math_search_policy_mismatch.py"
        ),
        "project/runtime/acbpo/actor_lora_experiment.py": (
            project_root / "src/acbpo/actor_lora_experiment.py"
        ),
        "project/runtime/acbpo/llama_cpp_policy.py": (
            project_root / "src/acbpo/llama_cpp_policy.py"
        ),
        "project/runtime/acbpo/math_rollout_experiment.py": (
            project_root / "src/acbpo/math_rollout_experiment.py"
        ),
        "project/runtime/acbpo/math_step_action_pilot.py": (
            project_root / "src/acbpo/math_step_action_pilot.py"
        ),
        "project/runtime/acbpo/step_critic_experiment.py": (
            project_root / "src/acbpo/step_critic_experiment.py"
        ),
        "project/runtime/acbpo/outcome_critic_experiment.py": (
            project_root / "src/acbpo/outcome_critic_experiment.py"
        ),
        "project/runtime/acbpo/qwen_tool_policy.py": (
            project_root / "src/acbpo/qwen_tool_policy.py"
        ),
        "project/runtime/acbpo/tau_retail.py": (
            project_root / "src/acbpo/tau_retail.py"
        ),
        "project/manuscript_main_tex": (
            project_root / "paper/search_policy_mismatch_iclr2027/main.tex"
        ),
        "project/manuscript_references": (
            project_root / "paper/search_policy_mismatch_iclr2027/references.bib"
        ),
        "project/manuscript_math_commands": (
            project_root / "paper/search_policy_mismatch_iclr2027/math_commands.tex"
        ),
        "project/manuscript_class_style": (
            project_root
            / "paper/search_policy_mismatch_iclr2027/iclr2027_conference.sty"
        ),
        "project/manuscript_bibliography_style": (
            project_root
            / "paper/search_policy_mismatch_iclr2027/iclr2027_conference.bst"
        ),
        "project/manuscript_fancyhdr_style": (
            project_root / "paper/search_policy_mismatch_iclr2027/fancyhdr.sty"
        ),
        "project/manuscript_natbib_style": (
            project_root / "paper/search_policy_mismatch_iclr2027/natbib.sty"
        ),
        "project/manuscript_pdf": (
            project_root / "paper/search_policy_mismatch_iclr2027/main.pdf"
        ),
    }
    archive_names = (
        "EVIDENCE_COMPLETE",
        "COMPLETE",
        "REMOTE_POSTRUN_TESTS_COMPLETE",
        "REMOTE_POSTRUN_TESTS.log",
        "REMOTE_POSTRUN_TESTS.xml",
        "S3_LLM_TRANSFER_ANALYSIS.json",
        "S3_LLM_TRANSFER_REPORT.md",
        "S3_MATH500_STRATA.json",
        "S3_MATH500_STRATA.md",
        "S3_MANUAL_COMPUTE_REVIEW.json",
        "S3_ARCHIVE_AUDIT.json",
        "S3_ARCHIVE_AUDIT.md",
        "ICLR_EVIDENCE_TABLES.json",
        "ICLR_EVIDENCE_TABLES.md",
        "ICLR_FIGURES_MANIFEST.json",
        "ICLR_S3_TERMINAL_RESULTS.tex",
        "ICLR_S3_TERMINAL_RESULTS_MANIFEST.json",
        "ICLR_MANUSCRIPT_REVIEW.json",
    )
    archive_files = {f"archive/{name}": root / name for name in archive_names}
    for stem in FIGURE_STEMS:
        for extension in ("png", "pdf"):
            relative = f"ICLR_FIGURES/{stem}.{extension}"
            archive_files[f"archive/{relative}"] = root / relative
    return {**project_files, **archive_files}


def validate_terminal_receipts(project_root: Path, root: Path) -> None:
    audit = read_json_object(root / "S3_ARCHIVE_AUDIT.json")
    if audit.get("status") != "structurally_complete":
        raise ValueError("S3 archive audit is not structurally complete")
    manual_gate = audit.get("manual_gate")
    compute_recorder = (
        project_root / "scripts/record_search_policy_mismatch_s3_compute_review.py"
    )
    compute_preparer = (
        project_root / "scripts/prepare_search_policy_mismatch_s3_compute_review.sh"
    )
    if (
        not isinstance(manual_gate, dict)
        or manual_gate.get("verdict") not in {"PASS", "FAIL"}
        or manual_gate.get("recorder_sha256") != sha256_file(compute_recorder)
        or manual_gate.get("preparer_sha256") != sha256_file(compute_preparer)
    ):
        raise ValueError("S3 manual compute gate was not explicitly reviewed")
    archive_hashes = audit.get("archive_sha256")
    archive_sources = {
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
        "frozen_analysis_json": root / "S3_LLM_TRANSFER_ANALYSIS.json",
        "frozen_report": root / "S3_LLM_TRANSFER_REPORT.md",
        "administrative_archive_audit_source": (
            project_root / "scripts/audit_search_policy_mismatch_s3_archive.py"
        ),
        "manual_compute_review_recorder_source": compute_recorder,
        "manual_compute_review_preparer_source": compute_preparer,
        "iclr_evidence_table_renderer_source": (
            project_root / "scripts/render_search_policy_mismatch_iclr_tables.py"
        ),
        "iclr_figure_renderer_source": (
            project_root / "scripts/render_search_policy_mismatch_iclr_figures.py"
        ),
        "iclr_terminal_results_renderer_source": (
            project_root / "scripts/render_search_policy_mismatch_iclr_results_tex.py"
        ),
        "iclr_manuscript_review_recorder_source": (
            project_root
            / "scripts/record_search_policy_mismatch_iclr_manuscript_review.py"
        ),
        "s3_final_archive_manifest_builder_source": Path(__file__).resolve(),
        "remote_postrun_test_source": (
            project_root
            / "scripts/verify_search_policy_mismatch_s3_remote_postrun.sh"
        ),
        "remote_postrun_test_log": root / "REMOTE_POSTRUN_TESTS.log",
        "remote_postrun_test_junit": root / "REMOTE_POSTRUN_TESTS.xml",
        "manual_compute_review_receipt": root / "S3_MANUAL_COMPUTE_REVIEW.json",
    }
    if not isinstance(archive_hashes, dict) or set(archive_hashes) != set(
        archive_sources
    ):
        raise ValueError("S3 archive SHA-256 matrix is incomplete")
    for label, path in archive_sources.items():
        if archive_hashes[label] != sha256_file(path):
            raise ValueError(f"S3 archive SHA-256 mismatch: {label}")
    strata = read_json_object(root / "S3_MATH500_STRATA.json")
    if strata.get("schema") != "cgcs-search-policy-mismatch-s3-strata-v1":
        raise ValueError("S3 MATH-500 strata schema mismatch")
    expected_lengths = {
        "per_seed_metrics": 900,
        "pooled_descriptive_metrics": 300,
        "stratified_descriptive_contrasts": 120,
    }
    if strata.get("evaluation_count") != 75 or any(
        len(strata.get(key, [])) != expected
        for key, expected in expected_lengths.items()
    ):
        raise ValueError("S3 MATH-500 strata matrix is incomplete")

    evidence_path = root / "ICLR_EVIDENCE_TABLES.json"
    evidence = read_json_object(evidence_path)
    provenance = evidence.get("provenance")
    input_hashes = provenance.get("input_sha256") if isinstance(provenance, dict) else None
    if (
        evidence.get("schema") != "cgcs-search-policy-mismatch-iclr-tables-v1"
        or evidence.get("status") != "evidence_complete"
        or not isinstance(provenance, dict)
        or not valid_sha256(provenance.get("renderer_sha256"))
        or not isinstance(input_hashes, dict)
        or set(input_hashes) != EVIDENCE_INPUTS
        or not all(valid_sha256(value) for value in input_hashes.values())
    ):
        raise ValueError("terminal ICLR evidence-table provenance is incomplete")
    evidence_renderer = (
        project_root / "scripts/render_search_policy_mismatch_iclr_tables.py"
    )
    if provenance["renderer_sha256"] != sha256_file(evidence_renderer):
        raise ValueError("terminal ICLR evidence-table renderer hash mismatch")
    evidence_inputs = {
        "mechanism_analysis": (
            project_root
            / "results/math_value_ppo/joint_value_mechanism_replication/analysis.json"
        ),
        "gate_confirmation": (
            project_root / "results/search_policy_mismatch/GATE_CONFIRMATION.json"
        ),
        "critic_confirmation": (
            project_root / "results/search_policy_mismatch/CRITIC_CONFIRMATION.json"
        ),
        "exact_confirmation": (
            project_root
            / "results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION.json"
        ),
        "exact_analysis": (
            project_root
            / "results/search_policy_mismatch/ESTIMATOR_MATRIX_CONFIRMATION_ANALYSIS.json"
        ),
        "s3_analysis": root / "S3_LLM_TRANSFER_ANALYSIS.json",
        "s3_audit": root / "S3_ARCHIVE_AUDIT.json",
    }
    for label, path in evidence_inputs.items():
        if input_hashes[label] != sha256_file(path):
            raise ValueError(f"terminal ICLR upstream evidence hash mismatch: {label}")
    evidence_sha = sha256_file(evidence_path)

    figure_manifest_path = root / "ICLR_FIGURES_MANIFEST.json"
    figure_manifest = read_json_object(figure_manifest_path)
    expected_figures = {
        f"{stem}.{extension}"
        for stem in FIGURE_STEMS
        for extension in ("png", "pdf")
    }
    figure_files = figure_manifest.get("files")
    if (
        figure_manifest.get("schema")
        != "cgcs-search-policy-mismatch-iclr-figures-v1"
        or figure_manifest.get("status") != "evidence_complete"
        or figure_manifest.get("source_evidence_tables_sha256") != evidence_sha
        or not valid_sha256(figure_manifest.get("renderer_sha256"))
        or not isinstance(figure_files, dict)
        or set(figure_files) != expected_figures
    ):
        raise ValueError("terminal ICLR figure manifest is incomplete or stale")
    figure_renderer = (
        project_root / "scripts/render_search_policy_mismatch_iclr_figures.py"
    )
    if figure_manifest["renderer_sha256"] != sha256_file(figure_renderer):
        raise ValueError("terminal ICLR figure renderer hash mismatch")
    for name, metadata in figure_files.items():
        path = root / "ICLR_FIGURES" / name
        if (
            not isinstance(metadata, dict)
            or not valid_sha256(metadata.get("sha256"))
            or not isinstance(metadata.get("bytes"), int)
            or metadata["bytes"] <= 0
            or metadata["sha256"] != sha256_file(path)
            or metadata["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"terminal ICLR figure receipt mismatch: {name}")

    results_manifest_path = root / "ICLR_S3_TERMINAL_RESULTS_MANIFEST.json"
    results_manifest = read_json_object(results_manifest_path)
    terminal_tex_path = root / "ICLR_S3_TERMINAL_RESULTS.tex"
    if (
        results_manifest.get("schema")
        != "cgcs-search-policy-mismatch-iclr-results-tex-v1"
        or results_manifest.get("status") != "evidence_complete"
        or not valid_sha256(results_manifest.get("renderer_sha256"))
        or results_manifest.get("evidence_tables_sha256") != evidence_sha
        or results_manifest.get("terminal_tex_sha256")
        != sha256_file(terminal_tex_path)
    ):
        raise ValueError("terminal ICLR results manifest is incomplete or stale")
    results_renderer = (
        project_root / "scripts/render_search_policy_mismatch_iclr_results_tex.py"
    )
    if results_manifest["renderer_sha256"] != sha256_file(results_renderer):
        raise ValueError("terminal ICLR results renderer hash mismatch")

    review = read_json_object(root / "ICLR_MANUSCRIPT_REVIEW.json")
    checks = review.get("checks")
    manuscript_main = project_root / "paper/search_policy_mismatch_iclr2027/main.tex"
    manuscript_pdf = project_root / "paper/search_policy_mismatch_iclr2027/main.pdf"
    manuscript_review_recorder = (
        project_root
        / "scripts/record_search_policy_mismatch_iclr_manuscript_review.py"
    )
    rationale = review.get("rationale")
    if (
        review.get("schema")
        != "cgcs-search-policy-mismatch-iclr-manuscript-review-v1"
        or review.get("status") != "reviewed"
        or review.get("verdict") != "PASS"
        or not isinstance(checks, dict)
        or set(checks) != MANUSCRIPT_REVIEW_CHECKS
        or not all(value is True for value in checks.values())
        or not isinstance(rationale, str)
        or len(rationale.strip()) < 40
        or review.get("recorder_sha256")
        != sha256_file(manuscript_review_recorder)
        or review.get("main_tex_sha256") != sha256_file(manuscript_main)
        or review.get("compiled_pdf_sha256") != sha256_file(manuscript_pdf)
        or review.get("terminal_results_manifest_sha256")
        != sha256_file(results_manifest_path)
        or review.get("figure_manifest_sha256")
        != sha256_file(figure_manifest_path)
    ):
        raise ValueError("ICLR manuscript review did not pass or is stale")
    if not (root / "EVIDENCE_COMPLETE").is_file() or not (root / "COMPLETE").is_file():
        raise ValueError("terminal S3 completion markers are missing")


def compact_text(path: Path) -> str:
    try:
        return " ".join(path.read_text().split())
    except OSError as error:
        raise ValueError(f"missing terminal project-state artifact: {path}") from error


def validate_project_terminal_state(project_root: Path) -> None:
    readme = compact_text(project_root / "README.md")
    if "S3 terminal status: `COMPLETE; STABILITY GATE FAILED`" not in readme:
        raise ValueError("S3 terminal README state is missing or inconsistent")
    if "S3 LLM transfer protocol is frozen and running" in readme:
        raise ValueError("S3 terminal README still claims that the run is active")

    stage_gates = compact_text(
        project_root / "results/search_policy_mismatch/STAGE_GATES.md"
    )
    required_gate_phrases = (
        "| S3 controlled LLM transfer | COMPLETE; STABILITY GATE FAILED |",
        "72/72 search branches exceed",
        "clip-fraction ceiling of 0.20",
        "No larger LLM experiment is authorized by S3",
    )
    if not all(phrase in stage_gates for phrase in required_gate_phrases):
        raise ValueError("S3 terminal stage-gate state is missing or inconsistent")
    if "| S3 controlled LLM transfer | PASS |" in stage_gates:
        raise ValueError("S3 terminal stage gate was incorrectly relabelled PASS")

    claims = compact_text(
        project_root
        / "results/search_policy_mismatch/ICLR_CLAIM_EVIDENCE_MATRIX.md"
    )
    if (
        "must be reported as a failed stage gate" not in claims
        or "cannot pass its full gate or authorize a larger" not in claims
    ):
        raise ValueError("S3 terminal claim boundary is missing or inconsistent")

    ledger = compact_text(
        project_root / "results/search_policy_mismatch/DATA_EXPOSURE_LEDGER.md"
    )
    if not all(
        phrase in ledger
        for phrase in (
            "0/48",
            "48/200",
            "248/120",
            "question/answer content and model outcomes remain unexposed",
        )
    ):
        raise ValueError("S3 terminal data-exposure ledger is incomplete")


def build_manifest(project_root: Path, root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    root = root.resolve()
    validate_terminal_receipts(project_root, root)
    validate_project_terminal_state(project_root)
    files = terminal_file_map(project_root, root)
    hashes = {label: sha256_file(path) for label, path in sorted(files.items())}
    return {
        "schema": SCHEMA,
        "status": "terminal_archive_reviewed",
        "artifact_count": len(hashes),
        "sha256": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(args.project_root, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "artifacts": payload["artifact_count"]}))


if __name__ == "__main__":
    main()
