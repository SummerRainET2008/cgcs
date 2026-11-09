#!/usr/bin/env python3
"""Prepare a deterministic scalar-answer subset of MATH-500.

The subset is intentionally restricted to integers, decimals, and exact integer
fractions so the existing response-level verifier can score it without a model
grader or a permissive symbolic parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from fractions import Fraction
from pathlib import Path


DATASET_ID = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SPLIT_DOMAIN = "cgcs-search-policy-mismatch-s3-math500-v1"
DEVELOPMENT_COUNT = 48
CONFIRMATION_COUNT = 200
DECIMAL = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)")
FRACTION = re.compile(r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}")


def normalize_scalar_answer(answer: str) -> str | None:
    value = answer.strip().replace("$", "").replace(" ", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # A comma may denote a tuple/set as well as a thousands separator.  Reject
    # both rather than silently changing a structured answer into a scalar.
    if not value or "," in value:
        return None
    if DECIMAL.fullmatch(value):
        return value
    match = FRACTION.fullmatch(value)
    if match is None:
        return None
    sign, numerator, denominator = match.groups()
    fraction = Fraction(int(numerator), int(denominator))
    if sign == "-":
        fraction = -fraction
    if fraction.denominator == 1:
        return str(fraction.numerator)
    return f"{fraction.numerator}/{fraction.denominator}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_partition(unique_id: str) -> str:
    """Return only the hash key; partition counts are assigned after sorting."""
    return hashlib.sha256(f"{SPLIT_DOMAIN}:{unique_id}".encode()).hexdigest()


def prepare(output: Path, manifest: Path) -> dict:
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID, revision=DATASET_REVISION, split="test"
    )
    selected = []
    for source_index, row in enumerate(dataset):
        normalized = normalize_scalar_answer(str(row["answer"]))
        if normalized is None:
            continue
        selected.append(
            {
                "question": str(row["problem"]).strip(),
                "answer": f"#### {normalized}",
                "source": {
                    "dataset": DATASET_ID,
                    "revision": DATASET_REVISION,
                    "split": "test",
                    "source_index": source_index,
                    "unique_id": str(row["unique_id"]),
                    "subject": str(row["subject"]),
                    "level": int(row["level"]),
                    "original_answer": str(row["answer"]),
                },
            }
        )
    if len(selected) < DEVELOPMENT_COUNT + CONFIRMATION_COUNT:
        raise RuntimeError(f"unexpectedly small scalar subset: {len(selected)}")

    selected.sort(key=lambda row: deterministic_partition(row["source"]["unique_id"]))
    boundaries = {
        "development": (0, DEVELOPMENT_COUNT),
        "confirmation": (DEVELOPMENT_COUNT, DEVELOPMENT_COUNT + CONFIRMATION_COUNT),
        "reserve": (DEVELOPMENT_COUNT + CONFIRMATION_COUNT, len(selected)),
    }
    for partition, (start, stop) in boundaries.items():
        for row in selected[start:stop]:
            row["source"]["s3_partition"] = partition

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    subjects: dict[str, int] = {}
    levels: dict[str, int] = {}
    for row in selected:
        source = row["source"]
        subjects[source["subject"]] = subjects.get(source["subject"], 0) + 1
        key = str(source["level"])
        levels[key] = levels.get(key, 0) + 1
    payload = {
        "schema": "cgcs-math500-scalar-subset-v1",
        "dataset": DATASET_ID,
        "revision": DATASET_REVISION,
        "source_split": "test",
        "filter": "scalar integer, decimal, or integer fraction; commas rejected",
        "split_domain": SPLIT_DOMAIN,
        "partitions": {
            name: {
                "offset": start,
                "count": stop - start,
                "unique_ids": [
                    row["source"]["unique_id"] for row in selected[start:stop]
                ],
            }
            for name, (start, stop) in boundaries.items()
        },
        "examples": len(selected),
        "subjects": dict(sorted(subjects.items())),
        "levels": dict(sorted(levels.items())),
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.manifest), indent=2))


if __name__ == "__main__":
    main()
