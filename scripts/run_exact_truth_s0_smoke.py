#!/usr/bin/env python3
"""Run fail-closed, development-only S0 smoke checks for the exact-truth study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from acbpo.layered_mdp import LayeredMDP
from acbpo.search_policy_mismatch import (
    SearchMDPPolicy,
    construction_diagnostics,
    monte_carlo_cumulative_recovery,
)


DEVELOPMENT_SMOKE_SEEDS = (1000, 1001, 1002)


def ranking_concordance(left: np.ndarray, right: np.ndarray) -> float:
    favorable = 0.0
    comparisons = 0
    for left_action in range(left.shape[-1]):
        for right_action in range(left_action + 1, left.shape[-1]):
            truth_delta = right[..., left_action] - right[..., right_action]
            signal_delta = left[..., left_action] - left[..., right_action]
            non_ties = truth_delta != 0.0
            favorable += float(
                np.sum((truth_delta * signal_delta > 0.0) & non_ties)
            )
            favorable += 0.5 * float(
                np.sum((signal_delta == 0.0) & non_ties)
            )
            comparisons += int(np.sum(non_ties))
    return favorable / comparisons


def standardize_within_state(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = np.maximum(values.std(axis=-1, keepdims=True), 1e-12)
    return centered / scale


def run_seed(seed: int, samples: int, stream_seed: int) -> dict[str, object]:
    mdp = LayeredMDP.random(
        seed=seed,
        horizon=6,
        state_count=7,
        action_count=6,
        gamma=0.97,
    )
    values, q_values = mdp.values()
    standardized_q = standardize_within_state(q_values)
    noise = np.random.default_rng(seed + 8_000_003).normal(size=q_values.shape)
    imperfect = standardize_within_state(standardized_q + 2.0 * noise)
    concordance = ranking_concordance(imperfect, q_values)
    controller = SearchMDPPolicy(
        mdp=mdp,
        critic_scores=standardized_q,
        candidate_count=2,
        beta=1.0,
        epsilon=0.10,
        expansion_probabilities=np.full(
            (mdp.horizon, mdp.state_count), 0.50, dtype=np.float64
        ),
    )
    recovery = monte_carlo_cumulative_recovery(
        controller,
        samples=samples,
        seed=stream_seed,
        initial_state=0,
    )
    truth = float(values[0, 0])
    recovery_checks = {}
    for name in ("e1_trajectory", "e2_cumulative"):
        error = abs(recovery[name]["mean"] - truth)
        three_se = 3.0 * recovery[name]["standard_error"]
        recovery_checks[name] = {
            "absolute_error": error,
            "three_standard_errors": three_se,
            "within_absolute_0.01": error <= 0.01,
            "truth_within_three_standard_errors": error <= three_se,
            "passed": error <= 0.01 and error <= three_se,
        }
    construction = construction_diagnostics(mdp)
    construction_passed = bool(
        construction["delayed_terminal_reward_dominates_per_layer"]
        and construction["misleading_local_reward_states"] > 0
        and construction["low_value_low_entropy_states"] > 0
        and construction["low_value_high_entropy_states"] > 0
        and concordance < 1.0
    )
    return {
        "development_mdp_seed": seed,
        "monte_carlo_stream_seed": stream_seed,
        "raw_actor_exact_return": truth,
        "construction": construction,
        "sigma_2_critic_ranking_concordance": concordance,
        "construction_passed": construction_passed,
        "estimators": recovery,
        "recovery_checks": recovery_checks,
        "passed": construction_passed
        and all(row["passed"] for row in recovery_checks.values()),
    }


def render_report(payload: dict[str, object]) -> str:
    lines = [
        "# Exact-truth S0 development smoke checks",
        "",
        "This is development-only construction and Monte Carlo evidence. It does not "
        "instantiate or inspect any confirmation environment.",
        "",
        f"- Samples per smoke MDP: {payload['samples_per_mdp']:,}",
        f"- Overall status: **{'PASS' if payload['passed'] else 'FAIL'}**",
        f"- Draft protocol SHA-256: `{payload['draft_protocol_sha256']}`",
        "",
        "| Dev seed | E1 trajectory error | E1 3SE | E2 cumulative error | E2 3SE | "
        "Noisy-Critic rank concordance | Status |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["seeds"]:
        e1 = row["recovery_checks"]["e1_trajectory"]
        e2 = row["recovery_checks"]["e2_cumulative"]
        lines.append(
            f"| {row['development_mdp_seed']} | {e1['absolute_error']:.6f} | "
            f"{e1['three_standard_errors']:.6f} | {e2['absolute_error']:.6f} | "
            f"{e2['three_standard_errors']:.6f} | "
            f"{row['sigma_2_critic_ranking_concordance']:.4f} | "
            f"{'PASS' if row['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "E1-local is retained in the JSON as a diagnostic only. It is not accepted "
            "as evidence that earlier search-induced state occupancy has been corrected.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "results/search_policy_mismatch/EXACT_TRUTH_PROTOCOL.md"
        ),
    )
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least two")
    protocol_bytes = args.protocol.read_bytes()
    rows = [
        run_seed(seed, args.samples, 2026102800 + offset)
        for offset, seed in enumerate(DEVELOPMENT_SMOKE_SEEDS)
    ]
    payload = {
        "kind": "development_only_exact_truth_s0_smoke",
        "confirmation_outcomes_seen": False,
        "draft_protocol": str(args.protocol),
        "draft_protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "samples_per_mdp": args.samples,
        "seeds": rows,
        "passed": all(row["passed"] for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(
        render_report(payload), encoding="utf-8"
    )
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
