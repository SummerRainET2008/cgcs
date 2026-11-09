#!/usr/bin/env python3
"""Prepare deterministic open-development and sealed DROP corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"
SOURCE_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
OFFICIAL_EVALUATOR_URL = (
    "https://github.com/allenai/allennlp-reading-comprehension/blob/master/"
    "allennlp_rc/eval/drop_eval.py"
)
OFFICIAL_EVALUATOR_SHA256 = (
    "589abfbb8a85760cdb3d30381068acb4545c82076311c2da5a112e283193e0a2"
)
ORDER_DOMAIN = "cgcs-s4y-drop-v1"
EXPECTED_COUNTS = {"train": 77409, "validation": 9536}
OPEN_DEV_COUNT = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(split: str, source_id: str) -> str:
    return hashlib.sha256(f"{ORDER_DOMAIN}\0{split}\0{source_id}".encode()).hexdigest()


def answer_to_strings(answer: dict[str, Any]) -> tuple[list[str], str]:
    if answer.get("number"):
        return [str(answer["number"])], "number"
    if answer.get("spans"):
        spans = [str(span) for span in answer["spans"]]
        return spans, "span" if len(spans) == 1 else "spans"
    if "date" in answer:
        date = answer["date"]
        return [f"{date['day']} {date['month']} {date['year']}"], "date"
    raise ValueError(f"invalid DROP answer: {answer}")


def normalize(
    *,
    passage_id: str,
    passage: str,
    qa_pair: dict[str, Any],
    qa_index: int,
    split: str,
) -> dict[str, Any]:
    query_id = str(qa_pair["query_id"])
    # The official validation release contains one repeated query_id.  Bind the
    # passage ID as well so every sealed unit has a stable, unique identity.
    source_id = f"{passage_id}:{query_id}:{qa_index}"
    raw_answers = [qa_pair["answer"], *qa_pair.get("validated_answers", [])]
    converted = [answer_to_strings(answer) for answer in raw_answers]
    reference_answers = [answer for answer, _ in converted]
    answer_types = [answer_type for _, answer_type in converted]
    prompt = (
        f"Passage:\n{str(passage).strip()}\n\n"
        f"Question:\n{str(qa_pair['question']).strip()}"
    )
    return {
        "source_id": source_id,
        "query_id": query_id,
        "passage_id": str(passage_id),
        "split": split,
        "question": prompt,
        "reference_answers": reference_answers,
        "answer_types": answer_types,
    }


def _flatten(payload: dict[str, Any], split: str) -> list[dict[str, Any]]:
    rows = []
    for passage_id, annotation in payload.items():
        for qa_index, qa_pair in enumerate(annotation["qa_pairs"]):
            rows.append(
                normalize(
                    passage_id=str(passage_id),
                    passage=str(annotation["passage"]),
                    qa_pair=qa_pair,
                    qa_index=qa_index,
                    split=split,
                )
            )
    expected = EXPECTED_COUNTS[split]
    if len(rows) != expected:
        raise ValueError(f"{split} count drift: {len(rows)} != {expected}")
    ids = [row["source_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate query_id in {split}")
    rows.sort(key=lambda row: stable_key(split, row["source_id"]))
    return rows


def _write(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "path": path.name,
        "count": len(rows),
        "sha256": sha256_file(path),
        "ordered_ids_sha256": hashlib.sha256(
            ("\n".join(row["source_id"] for row in rows) + "\n").encode()
        ).hexdigest(),
    }


def prepare(source_zip: Path, output_dir: Path) -> dict[str, Any]:
    source_digest = sha256_file(source_zip)
    if source_digest != SOURCE_SHA256:
        raise ValueError(f"source ZIP SHA-256 mismatch: {source_digest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_zip) as archive:
        if archive.testzip() is not None:
            raise ValueError("source ZIP integrity failure")
        train_payload = json.loads(
            archive.read("drop_dataset/drop_dataset_train.json")
        )
        validation_payload = json.loads(
            archive.read("drop_dataset/drop_dataset_dev.json")
        )
    train_all = _flatten(train_payload, "train")
    validation = _flatten(validation_payload, "validation")
    open_dev = train_all[:OPEN_DEV_COUNT]
    train = train_all[OPEN_DEV_COUNT:]
    manifest: dict[str, Any] = {
        "dataset": "DROP",
        "source_url": SOURCE_URL,
        "source_sha256": source_digest,
        "official_evaluator_url": OFFICIAL_EVALUATOR_URL,
        "official_evaluator_sha256": OFFICIAL_EVALUATOR_SHA256,
        "order_domain": ORDER_DOMAIN,
        "open_development_from_official_train": True,
        "official_validation_is_sealed": True,
        "splits": {},
    }
    manifest["splits"]["train"] = _write(output_dir / "drop_train.jsonl", train)
    manifest["splits"]["open_dev"] = _write(
        output_dir / "drop_open_dev.jsonl", open_dev
    )
    manifest["splits"]["sealed_validation"] = _write(
        output_dir / "drop_validation.jsonl", validation
    )
    manifest_path = output_dir / "DROP_PREPARATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_zip, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
