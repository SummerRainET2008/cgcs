from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_s4y_drop import ACTORS, HORIZONS, SEEDS, analyze, holm_adjust


def _write_matrix(root: Path, count: int = 4) -> None:
    for seed in SEEDS:
        for horizon in HORIZONS:
            for actor_index, actor in enumerate(ACTORS):
                rows = []
                for item in range(count):
                    exact = actor_index == 3 or item == 0
                    answered = not (actor_index == 0 and item == count - 1)
                    f1 = 1.0 if exact and answered else (0.5 if answered else 0.0)
                    rows.append(
                        {
                            "example_id": item,
                            "source_id": f"id-{item}",
                            "steps": ["FINAL: answer"],
                            "completion_tokens": 4 + actor_index,
                            "termination_reason": (
                                "valid_final" if answered else "max_steps"
                            ),
                            "result": {
                                "answered": answered,
                                "correct": exact and answered,
                                "exact_match": float(exact and answered),
                                "f1": f1,
                            },
                        }
                    )
                path = root / "eval" / f"s{seed}" / f"H{horizon}" / f"{actor}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"config": {"task": "drop"}, "examples": rows})
                )


def test_holm_is_monotone_in_sorted_order() -> None:
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_analysis_validates_matrix_and_finds_e4_gain(tmp_path: Path) -> None:
    _write_matrix(tmp_path)
    payload = analyze(tmp_path, expected_count=4, replicates=100, bootstrap_seed=7)
    assert payload["schema"] == "cgcs-s4y-drop-h8-analysis-v1"
    assert payload["expected_cells"] == 12
    assert payload["observed_cells"] == 12
    assert payload["total_evaluations"] == 48
    assert payload["excluded_conditions"] == {
        "H16-W96": "design error; archived for provenance only"
    }
    assert len(payload["contrasts"]) == 3
    assert all(row["horizons"] == [8] for row in payload["contrasts"])
    assert all(row["f1"]["difference"] >= 0 for row in payload["contrasts"])
    assert all("p_holm" in row["f1"] for row in payload["contrasts"])
    assert all("p_holm" in row["exact_match"] for row in payload["contrasts"])
    assert len(payload["transitions"]) == 3
