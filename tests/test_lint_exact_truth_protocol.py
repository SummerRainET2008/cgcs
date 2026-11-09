from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "scripts/lint_exact_truth_protocol.py"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"
)
SPEC = importlib.util.spec_from_file_location("protocol_lint", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lint_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lint_module)


def test_current_protocol_has_explicit_choices_and_is_frozen() -> None:
    text = PROTOCOL_PATH.read_text()
    structural = lint_module.lint(text, require_frozen=False)
    assert structural["ok"], structural
    assert structural["frozen"] is True
    frozen = lint_module.lint(text, require_frozen=True)
    assert frozen["ok"] is True
    assert frozen["errors"] == []


def test_lint_catches_unresolved_and_missing_choices() -> None:
    result = lint_module.lint("Status: **frozen**\nTODO\n", require_frozen=True)
    assert result["ok"] is False
    assert "TODO" in result["banned_snippets_present"]
    assert result["missing_required_snippets"]
