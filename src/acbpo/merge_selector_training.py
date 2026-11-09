from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_training_splits(
    inputs: list[Path],
    output: Path,
    *,
    split_manifest: Path,
) -> dict[str, Any]:
    """Merge training decisions while protecting the registered dev/test split."""
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    problem_ids = manifest["problem_ids"]
    forbidden = set(map(str, problem_ids["dev"])) | set(map(str, problem_ids["test"]))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    input_counts: dict[str, int] = {}
    for path in inputs:
        current = load_jsonl(path)
        input_counts[str(path)] = len(current)
        for row in current:
            problem_id = str(row["problem_id"])
            if problem_id in forbidden:
                raise ValueError(f"training problem {problem_id} is in registered dev/test")
            if problem_id in seen:
                raise ValueError(f"duplicate training problem ID: {problem_id}")
            seen.add(problem_id)
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "inputs": input_counts,
        "output": str(output),
        "states": len(rows),
        "unique_problem_ids": len(seen),
        "registered_dev_states": len(problem_ids["dev"]),
        "registered_test_states": len(problem_ids["test"]),
        "train_dev_test_overlap": 0,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge selector training splits without opening sealed labels"
    )
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = merge_training_splits(
        args.train,
        args.output,
        split_manifest=args.split_manifest,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
