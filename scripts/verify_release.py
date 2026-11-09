#!/usr/bin/env python3
from hashlib import sha256
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = root / "MANIFEST.sha256"
if not manifest.exists():
    raise SystemExit("MANIFEST.sha256 is missing")

expected = {}
for line in manifest.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    expected[rel] = digest

paper_build_suffixes = {".aux", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".pdf"}
actual_files = {
    str(path.relative_to(root))
    for path in root.rglob("*")
    if path.is_file()
    and path != manifest
    and not (path.parent == root / "paper" and path.suffix in paper_build_suffixes)
    and ".git" not in path.parts
    and "reproduced" not in path.parts
    and ".pytest_cache" not in path.parts
    and ".venv" not in path.parts
    and not any(part.endswith(".egg-info") for part in path.parts)
    and "__pycache__" not in path.parts
}
if actual_files != set(expected):
    missing = sorted(set(expected) - actual_files)
    extra = sorted(actual_files - set(expected))
    raise SystemExit(f"manifest inventory mismatch; missing={missing}; extra={extra}")

for rel, want in expected.items():
    got = sha256((root / rel).read_bytes()).hexdigest()
    if got != want:
        raise SystemExit(f"hash mismatch: {rel}: {got} != {want}")

print(f"RELEASE_MANIFEST_PASS files={len(expected)}")
