from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .math_prm import MathCandidate, MathDecision, summarize


def rating_from_return(value: float) -> int:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"return outside [0, 1]: {value}")
    if value == 0.0:
        return -1
    if value == 1.0:
        return 1
    return 0


def convert_payload(payload: dict[str, Any]) -> tuple[list[MathDecision], list[dict[str, Any]]]:
    decisions: list[MathDecision] = []
    targets: list[dict[str, Any]] = []
    for record_index, example in enumerate(payload["examples"]):
        problem = str(example["question"])
        problem_id = hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()[:16]
        for state in example["states"]:
            step_index = int(state["step_index"])
            example_id = f"step-q4:{example['example_id']}:{step_index}"
            candidates = tuple(
                MathCandidate(
                    text=str(candidate["step"]),
                    rating=rating_from_return(float(candidate["return_mean"])),
                    source_index=int(candidate["candidate"]),
                )
                for candidate in state["candidates"]
            )
            decisions.append(
                MathDecision(
                    example_id=example_id,
                    problem_id=problem_id,
                    problem=problem,
                    ground_truth_answer=str(example["reference_answer"]),
                    prefix=tuple(state["prefix"]),
                    candidates=candidates,
                    chosen_candidate=0,
                    record_index=record_index,
                    step_index=step_index,
                )
            )
            targets.append(
                {
                    "example_id": example_id,
                    "problem_id": problem_id,
                    "source_example_id": int(example["example_id"]),
                    "step_index": step_index,
                    "returns": [
                        float(candidate["return_mean"])
                        for candidate in state["candidates"]
                    ],
                }
            )
    return decisions, targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare step-action critic data")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    decisions, targets = convert_payload(payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")
    with (args.output_dir / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target) + "\n")
    manifest = {
        "source": str(args.input),
        "source_examples": len(payload["examples"]),
        "rating_mapping": {"0": -1, "(0,1)": 0, "1": 1},
        "dataset": summarize(decisions),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
