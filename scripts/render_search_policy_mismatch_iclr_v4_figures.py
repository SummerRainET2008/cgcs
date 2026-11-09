#!/usr/bin/env python3
"""Render v4-only readability revisions of Figures 3 and 5.

The evidence and scientific content are unchanged; only crowded axis labels are
shortened. The v3 figures are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

BASE_PATH = Path(__file__).with_name("render_search_policy_mismatch_iclr_figures.py")
spec = importlib.util.spec_from_file_location("cgcs_iclr_figures_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-tables", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.evidence_tables.read_text())
    base.configure_style()
    table = base.validate_tables(payload)

    fig3 = base.figure_3(table["exact_search_shift"])
    axis3 = fig3.axes[3]
    labels3 = ("Exact", "Lag .01", "Lag .04", "Learn 1k", "Learn 256", "Noise .5", "Noise 1", "Noise 2", "Invert")
    axis3.set_xticks(range(len(labels3)), labels3, rotation=24, ha="right", fontsize=7.0)
    fig3.subplots_adjust(bottom=0.12, hspace=0.34, wspace=0.28)

    fig5 = base.figure_5(
        table["s3_contrast_rows"],
        table["s3_compute_by_family_estimator"],
        table["s3_evaluation_metrics"],
    )
    axis5c, axis5d = fig5.axes[2], fig5.axes[3]
    axis5c.set_xticks(range(4), ("Unique", "Identical", "Exact token", "Prefix div."), rotation=16, ha="right", fontsize=7.2)
    axis5d.set_xticks(
        range(6),
        ("BoN\nE0", "BoN\nE4", "Entropy\nE0", "Entropy\nE4", "Value\nE0", "Value\nE4"),
        rotation=0,
        ha="center",
        fontsize=7.0,
    )
    fig5.subplots_adjust(bottom=0.11, hspace=0.36, wspace=0.30)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs += base.save_figure(fig3, args.output_dir, "figure_3_when_search_shift_helps_v4")
    outputs += base.save_figure(fig5, args.output_dir, "figure_5_retained_search_objective_and_cost_v4")
    manifest = {
        "schema": "cgcs-search-policy-mismatch-iclr-v4-figures-v1",
        "status": "evidence_complete_layout_revision",
        "source_evidence_tables_sha256": sha256_file(args.evidence_tables),
        "renderer_sha256": sha256_file(Path(__file__)),
        "files": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(outputs)
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
