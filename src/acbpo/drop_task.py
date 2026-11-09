from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np


DROP_FINAL_PATTERN = re.compile(r"(?i)^FINAL\s*:\s*(\S(?:.*\S)?)\s*$")
MULTISPAN_SEPARATOR = " || "
EXCLUDE = set(string.punctuation)


@dataclass(frozen=True)
class DROPExample:
    example_id: int
    source_id: str
    question: str
    reference_answer: str


def load_drop(path: Path, *, offset: int, count: int) -> list[DROPExample]:
    rows: list[DROPExample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            candidates = item["reference_answers"]
            # The official release contains nine training questions whose date
            # annotation normalizes to an empty string.  The official evaluator
            # retains them and awards zero unless another non-empty annotation
            # matches, so the loader must preserve (rather than reject) them.
            if not candidates or any(not isinstance(candidate, list) for candidate in candidates):
                raise ValueError(f"invalid DROP references for {item.get('source_id')}")
            rows.append(
                DROPExample(
                    example_id=index,
                    source_id=str(item["source_id"]),
                    question=str(item["question"]).strip(),
                    reference_answer=json.dumps(candidates, ensure_ascii=False),
                )
            )
    if offset < 0 or offset > len(rows):
        raise ValueError("offset is outside the dataset")
    return rows[offset : offset + count if count > 0 else None]


def drop_step_messages(question: str, steps: Sequence[str]) -> list[dict[str, str]]:
    prefix = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
    if not prefix:
        prefix = "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Answer the open-ended reading and numerical reasoning question one concise "
                "step at a time, using only the supplied passage. Return exactly one next "
                "step on one line and do not repeat previous steps. When complete, return "
                "exactly FINAL: <answer>. If the answer has multiple disjoint spans, join "
                f"them with {MULTISPAN_SEPARATOR.strip()}."
            ),
        },
        {
            "role": "user",
            "content": f"{question}\n\nReasoning so far:\n{prefix}\n\nNext step:",
        },
    ]


def extract_drop_final_answer(text: str) -> str | tuple[str, ...] | None:
    if "\n" in text or "\r" in text:
        return None
    match = DROP_FINAL_PATTERN.fullmatch(text.strip())
    if not match:
        return None
    spans = tuple(part.strip() for part in match.group(1).split(MULTISPAN_SEPARATOR))
    if any(not span for span in spans):
        return None
    return spans[0] if len(spans) == 1 else spans


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    return str(float(text)) if _is_number(text) else text


def _normalize_answer(text: str) -> str:
    parts = []
    for token in re.split(" |-", text.lower()):
        without_punctuation = token if _is_number(token) else "".join(
            character for character in token if character not in EXCLUDE
        )
        normalized = " ".join(
            _remove_articles(_normalize_number(without_punctuation)).split()
        )
        if normalized.strip():
            parts.append(normalized)
    return " ".join(parts).strip()


def _answer_to_bags(answer: str | Sequence[str]) -> tuple[list[str], list[set[str]]]:
    raw_spans = list(answer) if isinstance(answer, (list, tuple)) else [answer]
    normalized = [_normalize_answer(str(span)) for span in raw_spans]
    return normalized, [set(span.split()) for span in normalized]


def _compute_f1(predicted: set[str], gold: set[str]) -> float:
    intersection = len(gold.intersection(predicted))
    precision = intersection / len(predicted) if predicted else 1.0
    recall = intersection / len(gold) if gold else 1.0
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _numbers_match(gold: set[str], predicted: set[str]) -> bool:
    gold_numbers = {token for token in gold if _is_number(token)}
    predicted_numbers = {token for token in predicted if _is_number(token)}
    return not gold_numbers or bool(gold_numbers.intersection(predicted_numbers))


def _align_bags(predicted: list[set[str]], gold: list[set[str]]) -> list[float]:
    scores = np.zeros((len(gold), len(predicted)))
    for gold_index, gold_item in enumerate(gold):
        for predicted_index, predicted_item in enumerate(predicted):
            if _numbers_match(gold_item, predicted_item):
                scores[gold_index, predicted_index] = _compute_f1(
                    predicted_item, gold_item
                )
    # The official evaluator uses scipy's Hungarian assignment.  DROP answers
    # have very few spans, so an exact dependency-free dynamic program gives
    # the same maximum one-to-one score and keeps the frozen runtime minimal.
    if len(gold) <= len(predicted):
        matrix = scores
    else:
        matrix = scores.T

    @lru_cache(maxsize=None)
    def best(row: int, used_columns: int) -> tuple[float, tuple[int, ...]]:
        if row == matrix.shape[0]:
            return 0.0, ()
        candidates = []
        for column in range(matrix.shape[1]):
            if used_columns & (1 << column):
                continue
            tail_score, tail_columns = best(
                row + 1, used_columns | (1 << column)
            )
            candidates.append(
                (
                    float(matrix[row, column]) + tail_score,
                    (column, *tail_columns),
                )
            )
        return max(
            candidates,
            key=lambda candidate: candidate[0],
        )

    _, columns = best(0, 0)
    aligned = [0.0] * max(len(gold), len(predicted))
    for row, column in enumerate(columns):
        if len(gold) <= len(predicted):
            aligned[row] = float(matrix[row, column])
        else:
            aligned[column] = float(matrix[row, column])
    return aligned


def drop_metrics(
    predicted: str | Sequence[str], gold: str | Sequence[str]
) -> tuple[float, float]:
    """Official DROP exact-match/F1 normalization and span alignment."""
    predicted_normalized, predicted_bags = _answer_to_bags(predicted)
    gold_normalized, gold_bags = _answer_to_bags(gold)
    exact_match = float(
        set(predicted_normalized) == set(gold_normalized)
        and len(predicted_normalized) == len(gold_normalized)
    )
    f1 = float(round(np.mean(_align_bags(predicted_bags, gold_bags)), 2))
    return exact_match, f1


def drop_terminal_result(steps: Sequence[str], reference: str) -> dict[str, Any]:
    prediction = next(
        (
            answer
            for answer in (extract_drop_final_answer(step) for step in reversed(steps))
            if answer is not None
        ),
        None,
    )
    candidates = json.loads(reference)
    if prediction is None:
        exact_match = f1 = 0.0
    else:
        scores = [drop_metrics(prediction, candidate) for candidate in candidates]
        exact_match = max(score[0] for score in scores)
        f1 = max(score[1] for score in scores)
    return {
        "prediction": list(prediction) if isinstance(prediction, tuple) else prediction,
        "answered": prediction is not None,
        "correct": bool(exact_match),
        "exact_match": exact_match,
        "f1": f1,
    }
