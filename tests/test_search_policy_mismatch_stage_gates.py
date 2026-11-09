from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "results" / "search_policy_mismatch" / "STAGE_GATES.md"
PREFREEZE = (
    ROOT / "results" / "search_policy_mismatch" / "PREFREEZE_CHECKLIST.md"
)
CLAIMS = (
    ROOT / "results" / "search_policy_mismatch" / "ICLR_CLAIM_EVIDENCE_MATRIX.md"
)


def compact(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_completed_upstream_stages_remain_evidence_locked():
    gates = compact(GATES)
    assert "| S0 mathematical/software validity | PASS |" in gates
    assert "| S1 identifiable search-induced mismatch | PASS |" in gates
    assert "30/30 committed MDPs versus threshold 20/30" in gates
    assert "| S2 complete bias--variance finding | PASS |" in gates
    assert "2 ratio bases x 3 constructions x 5 lambdas" in gates

    prefreeze = compact(PREFREEZE)
    assert "Current three-seed mechanism replication complete | satisfied" in prefreeze
    assert "3/3 burn-ins, 36/36 adaptive summaries, and 18/18 evaluations" in prefreeze


def test_completed_s3_cannot_be_relabelled_as_pass_after_clip_gate_failure():
    gates = compact(GATES)
    assert "| S3 controlled LLM transfer | COMPLETE; STABILITY GATE FAILED |" in gates
    assert "72/72 search branches exceed" in gates
    assert "clip-fraction ceiling of 0.20" in gates
    assert "S3 cannot authorize expansion" in gates
    assert "39 raw-K1 evaluations" in gates
    assert "36 retained-composite evaluations" in gates
    assert "| S3 controlled LLM transfer | PASS |" not in gates

    claims = compact(CLAIMS)
    assert "must be reported as a failed stage gate" in claims
    assert "cannot pass its full gate or authorize a larger" in claims


def test_terminal_transition_is_intentionally_fail_closed():
    gates = compact(GATES)
    assert "This terminal row is immutable" in gates
    assert "negative/mixed final result remains in the archive" in gates
    assert "No larger LLM experiment is authorized by S3" in gates
