from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .layered_mdp import CandidateMDPBehavior, LayeredMDP
from .vtrace import discounted_return, per_decision_is_return, vtrace_values


def _summary(values: np.ndarray, truth: float) -> dict[str, float]:
    mean = float(values.mean())
    bias = mean - truth
    variance = float(values.var())
    return {
        "mean": mean,
        "truth": truth,
        "bias": bias,
        "absolute_bias": abs(bias),
        "variance": variance,
        "mse": variance + bias * bias,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--states", type=int, default=7)
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--phantom-sets", type=int, default=16)
    parser.add_argument("--critic-noise", type=float, default=0.35)
    parser.add_argument("--baseline-noise", type=float, default=0.30)
    parser.add_argument("--rho-bar", type=float, default=10.0)
    parser.add_argument("--c-bar", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path)
    return parser


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    phantom_rng = np.random.default_rng(args.seed + 1_000_003)
    mdp = LayeredMDP.random(
        seed=args.seed,
        horizon=args.horizon,
        state_count=args.states,
        action_count=args.actions,
    )
    true_values, true_q = mdp.values()
    critic_q = true_q + rng.normal(scale=args.critic_noise, size=true_q.shape)
    behavior = CandidateMDPBehavior(
        mdp=mdp,
        critic_q=critic_q,
        candidate_count=args.candidates,
        beta=args.beta,
        epsilon=args.epsilon,
    )
    baseline = true_values.copy()
    baseline[:-1] += rng.normal(
        scale=args.baseline_noise, size=baseline[:-1].shape
    )
    baseline[-1] = 0.0

    names = (
        "behavior_return",
        "joint_pdis",
        "marginal_pdis",
        "mc_marginal_pdis",
        "uncorrected_vtrace",
        "joint_vtrace",
        "marginal_vtrace",
        "mc_marginal_vtrace",
    )
    estimates = {
        name: np.empty(args.episodes, dtype=np.float64) for name in names
    }
    initial_state = 0
    for episode in range(args.episodes):
        trajectory = behavior.sample_trajectory(
            rng,
            initial_state=initial_state,
            phantom_sets=args.phantom_sets,
            phantom_rng=phantom_rng,
        )
        estimates["behavior_return"][episode] = discounted_return(
            trajectory.rewards, mdp.gamma
        )
        estimates["joint_pdis"][episode] = per_decision_is_return(
            trajectory.rewards, trajectory.joint_ratios, mdp.gamma
        )
        estimates["marginal_pdis"][episode] = per_decision_is_return(
            trajectory.rewards, trajectory.marginal_ratios, mdp.gamma
        )
        estimates["mc_marginal_pdis"][episode] = per_decision_is_return(
            trajectory.rewards, trajectory.mc_marginal_ratios, mdp.gamma
        )
        for name, ratios in (
            ("uncorrected_vtrace", np.ones(mdp.horizon)),
            ("joint_vtrace", trajectory.joint_ratios),
            ("marginal_vtrace", trajectory.marginal_ratios),
            ("mc_marginal_vtrace", trajectory.mc_marginal_ratios),
        ):
            estimates[name][episode] = vtrace_values(
                states=trajectory.states,
                rewards=trajectory.rewards,
                ratios=ratios,
                baseline_values=baseline,
                gamma=mdp.gamma,
                rho_bar=args.rho_bar,
                c_bar=args.c_bar,
            )[0]

    truth = float(true_values[0, initial_state])
    summaries = {name: _summary(values, truth) for name, values in estimates.items()}
    result = {
        "config": {key: value for key, value in vars(args).items() if key != "output"},
        "target_initial_value": truth,
        "behavior_initial_distribution": behavior.exact_marginals[0, initial_state].tolist(),
        "actor_initial_distribution": mdp.actor[0, initial_state].tolist(),
        "estimators": summaries,
    }
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(args)
    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
