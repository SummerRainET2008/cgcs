from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_s4x_arc_challenge import ACTORS, HORIZONS, SEEDS, analyze, holm_adjust


def _write_matrix(root: Path, count: int = 4) -> None:
    for seed in SEEDS:
        for horizon in HORIZONS:
            for actor_index, actor in enumerate(ACTORS):
                rows = []
                for item in range(count):
                    correct = actor_index == 3 or item == 0
                    answered = not (actor_index == 0 and item == count - 1)
                    rows.append(
                        {
                            "example_id": item,
                            "source_id": f"id-{item}",
                            "steps": ["FINAL: A"],
                            "completion_tokens": 4 + actor_index,
                            "termination_reason": "valid_final" if answered else "max_steps",
                            "result": {"answered": answered, "correct": correct and answered},
                        }
                    )
                path = root / "eval" / f"s{seed}" / f"H{horizon}" / f"{actor}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"examples": rows}))


def test_holm_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_analysis_validates_matrix_and_finds_e4_gain(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    payload = analyze(tmp_path, expected_count=4, replicates=100, bootstrap_seed=7)
    assert payload["observed_cells"] == 24
    assert payload["total_evaluations"] == 96
    assert len(payload["contrasts"]) == 9
    assert all(row["accuracy"]["difference"] >= 0 for row in payload["contrasts"])
    assert len(payload["transitions"]) == 6
