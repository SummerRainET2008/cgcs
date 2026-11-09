#!/usr/bin/env python3
"""Derive sealed exact-truth confirmation seeds from a frozen protocol hash.

The derivation contains no estimator or search-family label. Every method must
therefore reuse the same per-MDP Monte Carlo streams (common random numbers).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DOMAIN = b"cgcs-search-policy-mismatch-confirmation-v1"
DEVELOPMENT_SEEDS = frozenset(range(1000, 1030))
MDP_COUNT = 30
MC_STREAMS_PER_MDP = 10
MAX_SEED = 2_000_000_000


def protocol_digest(protocol_bytes: bytes) -> bytes:
    return hashlib.sha256(protocol_bytes).digest()


def draw_unique_seeds(
    digest: bytes,
    *,
    domain: bytes,
    count: int,
    excluded: frozenset[int] = frozenset(),
) -> list[int]:
    seeds: list[int] = []
    seen = set(excluded)
    counter = 0
    while len(seeds) < count:
        material = domain + b":" + digest + b":" + counter.to_bytes(8, "big")
        candidate = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % MAX_SEED
        counter += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        seeds.append(candidate)
    return seeds


def build_manifest(protocol_bytes: bytes) -> dict:
    digest = protocol_digest(protocol_bytes)
    mdp_seeds = draw_unique_seeds(
        digest,
        domain=DOMAIN + b":mdp",
        count=MDP_COUNT,
        excluded=DEVELOPMENT_SEEDS,
    )
    streams = {}
    for mdp_seed in mdp_seeds:
        stream_digest = hashlib.sha256(
            digest + b":" + str(mdp_seed).encode("ascii")
        ).digest()
        streams[str(mdp_seed)] = draw_unique_seeds(
            stream_digest,
            domain=DOMAIN + b":mc",
            count=MC_STREAMS_PER_MDP,
        )
    return {
        "schema": "cgcs-exact-truth-confirmation-seeds-v1",
        "protocol_sha256": digest.hex(),
        "derivation_domain": DOMAIN.decode("ascii"),
        "development_seeds_excluded": [min(DEVELOPMENT_SEEDS), max(DEVELOPMENT_SEEDS)],
        "confirmation_mdp_seeds": mdp_seeds,
        "monte_carlo_streams_by_mdp": streams,
        "common_random_numbers_across_methods": True,
        "estimator_or_search_label_used_in_derivation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol_text = protocol_bytes.decode("utf-8")
    if "Status: **frozen**" not in protocol_text:
        raise ValueError("refusing to derive confirmation seeds from a non-frozen protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite seed commitment: {args.output}")
    manifest = build_manifest(protocol_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
