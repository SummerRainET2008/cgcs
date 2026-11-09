from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def normalize_candidate_text(text: str) -> str:
    return " ".join(text.split())


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    problem_ids = [str(record["problem_id"]) for record in records]
    if len(problem_ids) != len(set(problem_ids)):
        duplicates = [key for key, count in Counter(problem_ids).items() if count > 1]
        raise ValueError(f"duplicate problem IDs: {duplicates[:5]}")
    return records


def split_problem_ids(
    records: Iterable[dict[str, Any]],
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    seed: int,
) -> dict[str, list[str]]:
    problem_ids = [str(record["problem_id"]) for record in records]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem IDs must be unique before splitting")
    requested = train_size + dev_size + test_size
    if requested != len(problem_ids):
        raise ValueError(f"split sizes total {requested}, expected {len(problem_ids)}")
    random.Random(seed).shuffle(problem_ids)
    train_end = train_size
    dev_end = train_end + dev_size
    return {
        "train": problem_ids[:train_end],
        "dev": problem_ids[train_end:dev_end],
        "test": problem_ids[dev_end:],
    }


def _converted_candidates(
    record: dict[str, Any], *, deduplicate: bool, drop_unanswered: bool
) -> tuple[list[dict[str, Any]], int, int, int]:
    converted: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    duplicate_count = 0
    conflicting_duplicates = 0
    unanswered_count = 0
    for source_index, candidate in enumerate(record["candidates"]):
        if drop_unanswered and candidate.get("prediction") is None:
            unanswered_count += 1
            continue
        text = str(candidate["text"]).strip()
        key = normalize_candidate_text(text)
        rating = 1 if int(candidate["reward"]) == 1 else -1
        if deduplicate and key in seen:
            duplicate_count += 1
            if converted[seen[key]]["rating"] != rating:
                conflicting_duplicates += 1
            else:
                converted[seen[key]]["multiplicity"] += 1
            continue
        seen.setdefault(key, len(converted))
        converted.append(
            {
                "text": text,
                "rating": rating,
                "source_index": source_index,
                "multiplicity": 1,
            }
        )
    return converted, duplicate_count, conflicting_duplicates, unanswered_count


def convert_record(
    record: dict[str, Any], *, deduplicate: bool, drop_unanswered: bool = False
) -> tuple[dict[str, Any], dict[str, int]]:
    candidates, duplicates, conflicts, unanswered = _converted_candidates(
        record, deduplicate=deduplicate, drop_unanswered=drop_unanswered
    )
    if not candidates:
        raise ValueError(f"state {record['problem_id']} has no candidates")
    decision = {
        "example_id": str(record["example_id"]),
        "problem_id": str(record["problem_id"]),
        "problem": str(record["question"]),
        "ground_truth_answer": str(record["reference_answer"]),
        "prefix": [],
        "candidates": candidates,
        "chosen_candidate": None,
        "record_index": -1,
        "step_index": 0,
    }
    return decision, {
        "duplicates_removed": duplicates,
        "conflicting_duplicates": conflicts,
        "unanswered_removed": unanswered,
    }


def summarize_raw_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    candidates = [candidate for row in rows for candidate in row["candidates"]]
    answered = [candidate for candidate in candidates if candidate.get("prediction") is not None]
    mixed = [
        row
        for row in rows
        if {int(candidate["reward"]) for candidate in row["candidates"]} == {0, 1}
    ]
    clean_mixed = [
        row
        for row in rows
        if any(int(candidate["reward"]) == 1 for candidate in row["candidates"])
        and any(
            int(candidate["reward"]) == 0 and candidate.get("prediction") is not None
            for candidate in row["candidates"]
        )
    ]
    unique_counts = [
        len({normalize_candidate_text(str(item["text"])) for item in row["candidates"]})
        for row in rows
    ]
    cap_hits = sum(
        int(candidate.get("token_count", -1)) >= int(row.get("max_new_tokens", 0))
        for row in rows
        for candidate in row["candidates"]
    )
    return {
        "states": len(rows),
        "candidates": len(candidates),
        "answer_rate": len(answered) / max(len(candidates), 1),
        "candidate_accuracy": sum(int(item["reward"]) for item in candidates)
        / max(len(candidates), 1),
        "pass_at_k": sum(
            any(int(item["reward"]) == 1 for item in row["candidates"]) for row in rows
        )
        / max(len(rows), 1),
        "mixed_states": len(mixed),
        "mixed_state_fraction": len(mixed) / max(len(rows), 1),
        "clean_mixed_states": len(clean_mixed),
        "clean_mixed_fraction": len(clean_mixed) / max(len(rows), 1),
        "all_correct_states": sum(
            all(int(item["reward"]) == 1 for item in row["candidates"]) for row in rows
        ),
        "all_wrong_states": sum(
            all(int(item["reward"]) == 0 for item in row["candidates"]) for row in rows
        ),
        "mean_unique_candidates": sum(unique_counts) / max(len(unique_counts), 1),
        "length_cap_hits": cap_hits,
    }


def prepare_dataset(
    records: list[dict[str, Any]],
    output_dir: Path,
    *,
    train_size: int,
    dev_size: int,
    test_size: int,
    seed: int,
) -> dict[str, Any]:
    splits = split_problem_ids(
        records,
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
        seed=seed,
    )
    by_id = {str(record["problem_id"]): record for record in records}
    output_dir.mkdir(parents=True, exist_ok=True)
    conversion: dict[str, Any] = {}
    split_sets = {name: set(ids) for name, ids in splits.items()}
    overlap = {
        "train_dev": len(split_sets["train"] & split_sets["dev"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "dev_test": len(split_sets["dev"] & split_sets["test"]),
    }
    for name, ids in splits.items():
        raw_rows = [by_id[problem_id] for problem_id in ids]
        # Exact-text duplicates are collapsed only in training so repeated samples
        # cannot overweight a target. Dev/test retain all K sampled positions and
        # therefore measure the original candidate distribution without rebalance.
        deduplicate = name == "train"
        decisions = []
        duplicate_count = 0
        conflict_count = 0
        unanswered_count = 0
        unusable_state_count = 0
        for row in raw_rows:
            if name == "train" and not any(
                candidate.get("prediction") is not None
                for candidate in row["candidates"]
            ):
                unanswered_count += len(row["candidates"])
                unusable_state_count += 1
                continue
            decision, stats = convert_record(
                row,
                deduplicate=deduplicate,
                drop_unanswered=name == "train",
            )
            decisions.append(decision)
            duplicate_count += stats["duplicates_removed"]
            conflict_count += stats["conflicting_duplicates"]
            unanswered_count += stats["unanswered_removed"]
        with (output_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for decision in decisions:
                handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
        conversion[name] = {
            "deduplicated_for_training": deduplicate,
            "states": len(decisions),
            "output_candidates": sum(len(row["candidates"]) for row in decisions),
            "effective_candidate_positions": sum(
                int(candidate.get("multiplicity", 1))
                for row in decisions
                for candidate in row["candidates"]
            ),
            "duplicates_removed": duplicate_count,
            "conflicting_duplicates": conflict_count,
            "unanswered_removed": unanswered_count,
            "unusable_states_removed": unusable_state_count,
            "raw": summarize_raw_records(raw_rows),
        }
    manifest = {
        "seed": seed,
        "problem_ids": splits,
        "overlap": overlap,
        "source": summarize_raw_records(records),
        "conversion": conversion,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split and convert on-policy candidate sets for selector training"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1600)
    parser.add_argument("--dev-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=75001)
    args = parser.parse_args()
    manifest = prepare_dataset(
        load_records(args.input),
        args.output_dir,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
