from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ARC_FINAL_PATTERN = re.compile(r"(?i)^FINAL\s*:\s*([A-E])\s*[.]?$")


@dataclass(frozen=True)
class ARCExample:
    example_id: int
    source_id: str
    question: str
    reference_answer: str


def load_arc_challenge(path: Path, *, offset: int, count: int) -> list[ARCExample]:
    rows: list[ARCExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            answer = str(item["reference_answer"]).strip().upper()
            labels = [str(choice["label"]).strip().upper() for choice in item["choices"]]
            if answer not in labels or len(labels) != len(set(labels)):
                raise ValueError(f"invalid ARC choices for {item.get('example_id')}")
            rows.append(
                ARCExample(
                    example_id=index,
                    source_id=str(item["example_id"]),
                    question=str(item["question"]).strip(),
                    reference_answer=answer,
                )
            )
    if offset < 0 or offset > len(rows):
        raise ValueError("offset is outside the dataset")
    return rows[offset : offset + count if count > 0 else None]


def arc_step_messages(question: str, steps: Sequence[str]) -> list[dict[str, str]]:
    prefix = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
    if not prefix:
        prefix = "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Answer the multiple-choice science question one concise reasoning step at "
                "a time. Return exactly one next step on one line. Do not repeat previous "
                "steps. When complete, return exactly FINAL: <option letter>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\nReasoning so far:\n{prefix}\n\nNext step:"
            ),
        },
    ]


def extract_arc_final_answer(text: str) -> str | None:
    match = ARC_FINAL_PATTERN.fullmatch(text.strip())
    return match.group(1).upper() if match else None


def arc_terminal_result(steps: Sequence[str], reference: str) -> dict[str, Any]:
    prediction = next(
        (
            answer
            for answer in (extract_arc_final_answer(step) for step in reversed(steps))
            if answer is not None
        ),
        None,
    )
    return {
        "prediction": prediction,
        "answered": prediction is not None,
        "correct": prediction == reference.strip().upper(),
    }
