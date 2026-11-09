#!/usr/bin/env python3
"""Fail-closed structural lint for the exact-truth protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_SNIPPETS = (
    "`H=6`, `S=7`, `A=6`",
    "`gamma=0.97`",
    "expected candidates `B in {1.5, 2.0}`",
    "`epsilon=0.10`",
    "`sigma in {0.5, 1.0, 2.0}`",
    "`(rho_bar,c_bar) in {(1,1),(2,1)}`",
    "`lambda in {0, 0.25, 0.5, 0.75, 1}`",
    "`{256, 1024}`",
    "mean state-weighted KL 0.01",
    "absolute tolerance `1e-10`",
    "relative L2 error at most `1e-5`",
    "Development MDP seeds 1000--1029 are permanently excluded",
    "common random numbers within an MDP",
    "### Gate S0",
    "### Gate S1",
    "### Gate S2",
    "### Gate S3",
)

BANNED_SNIPPETS = (
    "TODO",
    "TBD",
    "preregistered tolerance",
    "preregistered noise levels",
    "selected partial corrections",
    "appropriate E4",
    "initially `H=",
    "A smaller fixed smoke subset",
)


def lint(text: str, *, require_frozen: bool) -> dict:
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in text]
    banned = [snippet for snippet in BANNED_SNIPPETS if snippet in text]
    frozen = any(line.strip() == "Status: **frozen**" for line in text.splitlines())
    errors = []
    if missing:
        errors.append("missing required frozen choices")
    if banned:
        errors.append("contains unresolved language")
    if require_frozen and not frozen:
        errors.append("protocol is not marked frozen")
    return {
        "frozen": frozen,
        "require_frozen": require_frozen,
        "missing_required_snippets": missing,
        "banned_snippets_present": banned,
        "errors": errors,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    result = lint(args.protocol.read_text(), require_frozen=args.require_frozen)
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
