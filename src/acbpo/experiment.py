from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .bandit import collect_gradient_samples, exact_policy_gradient, summarize
from .candidate_policy import CandidatePolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--phantom-sets", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    proposal = rng.dirichlet(np.linspace(0.8, 2.0, args.actions))
    rewards = rng.normal(size=args.actions)
    q_values = rewards + rng.normal(scale=0.35, size=args.actions)
    policy = CandidatePolicy(
        proposal=proposal,
        q_values=q_values,
        candidate_count=args.candidates,
        beta=args.beta,
        epsilon=args.epsilon,
    )
    truth = exact_policy_gradient(proposal, rewards)
    samples = collect_gradient_samples(
        policy,
        rewards,
        sample_count=args.samples,
        phantom_sets=args.phantom_sets,
        seed=args.seed + 1,
    )
    summaries = {
        name: asdict(summarize(values, truth)) for name, values in samples.items()
    }
    for summary in summaries.values():
        summary["mean"] = summary["mean"].tolist()
    result = {
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "proposal": proposal.tolist(),
        "rewards": rewards.tolist(),
        "q_values": q_values.tolist(),
        "behavior_marginal": policy.exact_marginal().tolist(),
        "true_gradient": truth.tolist(),
        "estimators": summaries,
    }
    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
