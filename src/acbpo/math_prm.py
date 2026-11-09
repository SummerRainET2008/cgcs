from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


RATING_LABELS = {-1: "A", 0: "B", 1: "C"}


@dataclass(frozen=True)
class MathCandidate:
    text: str
    rating: int
    source_index: int
    multiplicity: int = 1


@dataclass(frozen=True)
class MathDecision:
    example_id: str
    problem_id: str
    problem: str
    ground_truth_answer: str
    prefix: tuple[str, ...]
    candidates: tuple[MathCandidate, ...]
    chosen_candidate: int | None
    record_index: int
    step_index: int


def _problem_id(problem: str) -> str:
    return hashlib.sha256(problem.strip().encode("utf-8")).hexdigest()[:16]


def _continuation(step: dict[str, Any]) -> str | None:
    chosen = step.get("chosen_completion")
    completions = step.get("completions") or []
    if isinstance(chosen, int) and 0 <= chosen < len(completions):
        text = completions[chosen].get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    human = step.get("human_completion")
    if isinstance(human, dict):
        human = human.get("text")
    if isinstance(human, str) and human.strip():
        return human.strip()
    return None


def _valid_candidates(step: dict[str, Any]) -> tuple[MathCandidate, ...]:
    """Return unflagged, labeled candidates; drop conflicting duplicate texts."""
    by_text: dict[str, list[tuple[int, int]]] = {}
    display_text: dict[str, str] = {}
    for index, completion in enumerate(step.get("completions") or []):
        if completion.get("flagged"):
            continue
        rating = completion.get("rating")
        text = completion.get("text")
        if rating not in RATING_LABELS or not isinstance(text, str) or not text.strip():
            continue
        clean = text.strip()
        key = " ".join(clean.split())
        display_text.setdefault(key, clean)
        by_text.setdefault(key, []).append((index, int(rating)))

    candidates: list[MathCandidate] = []
    for key, observations in by_text.items():
        ratings = {rating for _, rating in observations}
        if len(ratings) != 1:
            continue
        source_index, rating = observations[0]
        candidates.append(MathCandidate(display_text[key], rating, source_index))
    return tuple(candidates)


def iter_decisions(
    records: Iterable[dict[str, Any]], *, require_contrast: bool = True
) -> Iterable[MathDecision]:
    """Yield candidate-ranking decisions with a reconstructable trajectory prefix."""
    for record_index, record in enumerate(records):
        question = record.get("question") or {}
        problem = str(question.get("problem") or "").strip()
        if not problem:
            continue
        pid = _problem_id(problem)
        prefix: list[str] = []
        for step_index, step in enumerate((record.get("label") or {}).get("steps") or []):
            candidates = _valid_candidates(step)
            ratings = {candidate.rating for candidate in candidates}
            if len(candidates) >= 2 and (not require_contrast or len(ratings) >= 2):
                source_to_local = {
                    candidate.source_index: index
                    for index, candidate in enumerate(candidates)
                }
                chosen = step.get("chosen_completion")
                chosen_local = source_to_local.get(chosen) if isinstance(chosen, int) else None
                yield MathDecision(
                    example_id=f"{pid}:{record_index}:{step_index}",
                    problem_id=pid,
                    problem=problem,
                    ground_truth_answer=str(question.get("ground_truth_answer") or ""),
                    prefix=tuple(prefix),
                    candidates=candidates,
                    chosen_candidate=chosen_local,
                    record_index=record_index,
                    step_index=step_index,
                )
            continuation = _continuation(step)
            if continuation is None:
                break
            prefix.append(continuation)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def critic_messages(decision: MathDecision, candidate: MathCandidate) -> list[dict[str, str]]:
    prefix = "\n".join(
        f"Step {index}: {text}" for index, text in enumerate(decision.prefix, start=1)
    ) or "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Judge one proposed next step in a math solution. Output exactly one "
                "letter: A = incorrect, B = neutral or no progress, C = correct and useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{decision.problem}\n\nSolution so far:\n{prefix}\n\n"
                f"Proposed next step:\n{candidate.text}\n\nRating:"
            ),
        },
    ]


def terminal_return_critic_messages(
    decision: MathDecision, candidate: MathCandidate
) -> list[dict[str, str]]:
    """Prompt for predicting terminal success after one proposed math step."""
    prefix = "\n".join(
        f"Step {index}: {text}" for index, text in enumerate(decision.prefix, start=1)
    ) or "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Predict whether continuing a math solution after one proposed next "
                "step will reach the correct final numeric answer. Output exactly one "
                "letter: A = continuations fail, B = continuations have mixed success, "
                "C = continuations succeed."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{decision.problem}\n\nSolution so far:\n{prefix}\n\n"
                f"Proposed next step:\n{candidate.text}\n\nOutcome rating:"
            ),
        },
    ]


def summarize(decisions: list[MathDecision]) -> dict[str, Any]:
    rating_counts = Counter(
        candidate.rating for decision in decisions for candidate in decision.candidates
    )
    top_rating = [max(candidate.rating for candidate in item.candidates) for item in decisions]
    random_top1 = sum(
        sum(candidate.rating == best for candidate in item.candidates) / len(item.candidates)
        for item, best in zip(decisions, top_rating)
    ) / max(len(decisions), 1)
    chosen_items = [item for item in decisions if item.chosen_candidate is not None]
    chosen_top1 = sum(
        item.candidates[item.chosen_candidate].rating
        == max(candidate.rating for candidate in item.candidates)
        for item in chosen_items
    ) / max(len(chosen_items), 1)
    return {
        "decisions": len(decisions),
        "problems": len({item.problem_id for item in decisions}),
        "candidates": sum(len(item.candidates) for item in decisions),
        "candidate_count": dict(sorted(Counter(len(x.candidates) for x in decisions).items())),
        "rating_count": {str(key): value for key, value in sorted(rating_counts.items())},
        "mean_prefix_steps": (
            sum(len(item.prefix) for item in decisions) / max(len(decisions), 1)
        ),
        "uniform_random_top1": random_top1,
        "chosen_coverage": len(chosen_items) / max(len(decisions), 1),
        "chosen_top1": chosen_top1,
    }


def _write_jsonl(path: Path, decisions: list[MathDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(asdict(decision), ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PRM800K candidate-ranking data")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train = list(iter_decisions(load_jsonl(args.train)))
    test = list(iter_decisions(load_jsonl(args.test)))
    train_ids = {item.problem_id for item in train}
    test_ids = {item.problem_id for item in test}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise RuntimeError(f"train/test problem leakage: {len(overlap)} overlapping IDs")

    _write_jsonl(args.output_dir / "train.jsonl", train)
    _write_jsonl(args.output_dir / "test.jsonl", test)
    summary = {
        "train": summarize(train),
        "test": summarize(test),
        "problem_overlap": 0,
        "rating_labels": {str(key): value for key, value in RATING_LABELS.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
