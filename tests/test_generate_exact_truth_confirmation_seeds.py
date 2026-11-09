from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/generate_exact_truth_confirmation_seeds.py"
)
SPEC = importlib.util.spec_from_file_location("seed_commitment", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
seeds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seeds)


def test_manifest_is_deterministic_unique_and_excludes_development() -> None:
    protocol = b"Status: **frozen**\nfixed protocol\n"
    first = seeds.build_manifest(protocol)
    second = seeds.build_manifest(protocol)
    assert first == second
    mdp = first["confirmation_mdp_seeds"]
    assert len(mdp) == seeds.MDP_COUNT == len(set(mdp))
    assert set(mdp).isdisjoint(seeds.DEVELOPMENT_SEEDS)
    assert first["estimator_or_search_label_used_in_derivation"] is False
    for seed in mdp:
        streams = first["monte_carlo_streams_by_mdp"][str(seed)]
        assert len(streams) == seeds.MC_STREAMS_PER_MDP == len(set(streams))


def test_protocol_change_changes_commitment() -> None:
    left = seeds.build_manifest(b"Status: **frozen**\nprotocol A\n")
    right = seeds.build_manifest(b"Status: **frozen**\nprotocol B\n")
    assert left["protocol_sha256"] != right["protocol_sha256"]
    assert left["confirmation_mdp_seeds"] != right["confirmation_mdp_seeds"]


def test_cli_refuses_draft_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.md"
    output = tmp_path / "seeds.json"
    protocol.write_text("Status: **draft**\n")
    draft = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(protocol), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert draft.returncode != 0
    assert "non-frozen protocol" in draft.stderr
    protocol.write_text("Status: **frozen**\n")
    first = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(protocol), "--output", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(first.stdout)["schema"] == "cgcs-exact-truth-confirmation-seeds-v1"
    overwrite = subprocess.run(
        [sys.executable, str(MODULE_PATH), str(protocol), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert overwrite.returncode != 0
    assert "refusing to overwrite" in overwrite.stderr


def test_committed_manifest_matches_frozen_protocol_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = root / "results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"
    manifest = root / "results/search_policy_mismatch/CONFIRMATION_SEEDS.json"
    assert json.loads(manifest.read_text()) == seeds.build_manifest(protocol.read_bytes())
