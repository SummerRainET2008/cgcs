#!/usr/bin/env python3
"""Prepare a sealed, deterministic ARC-Challenge corpus from the official ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


SOURCE_SHA256 = "6d2d5ab50b2ceec6ba5f79c921be77cf2de712ea25a2b3f4fff3acc101cecfa0"
HF_REVISION = "210d026faf9955653af8916fad021475a3f00453"
ORDER_DOMAIN = "cgcs-s4x-arc-challenge-v1"
SPLITS = {
    "train": ("Train", 1119),
    "validation": ("Dev", 299),
    "test": ("Test", 1172),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(split: str, source_id: str) -> str:
    return hashlib.sha256(f"{ORDER_DOMAIN}\0{split}\0{source_id}".encode()).hexdigest()


def normalize(row: dict[str, Any], split: str) -> dict[str, Any]:
    if set(row) != {"id", "question", "answerKey"}:
        raise ValueError(f"unexpected row keys in {split}: {sorted(row)}")
    source_id = str(row["id"])
    question = row["question"]
    if set(question) != {"stem", "choices"}:
        raise ValueError(f"unexpected question keys for {source_id}")
    choices = question["choices"]
    labels = [str(choice["label"]).strip().upper() for choice in choices]
    if not 3 <= len(labels) <= 5 or len(set(labels)) != len(labels) or any(
        not label for label in labels
    ):
        raise ValueError(f"invalid option labels for {source_id}: {labels}")
    answer_source = str(row["answerKey"]).strip().upper()
    if answer_source not in labels:
        raise ValueError(f"answer label missing from choices for {source_id}")
    canonical_labels = list("ABCDE"[: len(labels)])
    ordered = [
        {
            "label": canonical,
            "text": str(choice["text"]).strip(),
            "source_label": source,
        }
        for canonical, source, choice in zip(canonical_labels, labels, choices)
    ]
    answer = canonical_labels[labels.index(answer_source)]
    prompt = str(question["stem"]).strip() + "\n\nOptions:\n" + "\n".join(
        f"{choice['label']}. {choice['text']}" for choice in ordered
    )
    return {
        "example_id": source_id,
        "source_id": source_id,
        "split": split,
        "question": prompt,
        "reference_answer": answer,
        "choices": ordered,
    }


def prepare(source_zip: Path, output_dir: Path) -> dict[str, Any]:
    source_digest = sha256_file(source_zip)
    if source_digest != SOURCE_SHA256:
        raise ValueError(f"source ZIP SHA-256 mismatch: {source_digest}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "dataset": "ARC-Challenge",
        "source_url": "https://s3-us-west-2.amazonaws.com/ai2-website/data/ARC-V1-Feb2018.zip",
        "source_sha256": source_digest,
        "hf_repository": "allenai/ai2_arc",
        "hf_revision": HF_REVISION,
        "order_domain": ORDER_DOMAIN,
        "splits": {},
    }
    seen_ids: set[str] = set()
    with zipfile.ZipFile(source_zip) as archive:
        for split, (source_name, expected_count) in SPLITS.items():
            member = (
                "ARC-V1-Feb2018-2/ARC-Challenge/"
                f"ARC-Challenge-{source_name}.jsonl"
            )
            rows = [
                normalize(json.loads(line), split)
                for line in archive.read(member).splitlines()
                if line.strip()
            ]
            if len(rows) != expected_count:
                raise ValueError(f"{split} count drift: {len(rows)} != {expected_count}")
            ids = [row["example_id"] for row in rows]
            if len(ids) != len(set(ids)) or seen_ids.intersection(ids):
                raise ValueError(f"duplicate example ID detected in {split}")
            seen_ids.update(ids)
            rows.sort(key=lambda row: stable_key(split, row["example_id"]))
            output = output_dir / f"arc_challenge_{split}.jsonl"
            with output.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            manifest["splits"][split] = {
                "count": len(rows),
                "path": output.name,
                "sha256": sha256_file(output),
                "ordered_ids_sha256": hashlib.sha256(
                    ("\n".join(row["example_id"] for row in rows) + "\n").encode()
                ).hexdigest(),
            }
    manifest_path = output_dir / "ARC_CHALLENGE_PREPARATION_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.source_zip, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
