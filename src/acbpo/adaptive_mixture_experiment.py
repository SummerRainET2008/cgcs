from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .layered_mdp import CandidateMDPBehavior, LayeredMDP, Trajectory
from .vtrace import batch_vtrace_initial


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TrajectoryArrays:
    states: IntArray
    rewards: FloatArray
    joint_ratios: FloatArray
    marginal_ratios: FloatArray
    mc_marginal_ratios: FloatArray


def _stack(trajectories: list[Trajectory]) -> TrajectoryArrays:
    return TrajectoryArrays(
        states=np.stack([trajectory.states for trajectory in trajectories]),
        rewards=np.stack([trajectory.rewards for trajectory in trajectories]),
        joint_ratios=np.stack(
            [trajectory.joint_ratios for trajectory in trajectories]
        ),
        marginal_ratios=np.stack(
            [trajectory.marginal_ratios for trajectory in trajectories]
        ),
        mc_marginal_ratios=np.stack(
            [trajectory.mc_marginal_ratios for trajectory in trajectories]
        ),
    )


def _collect(
    behavior: CandidateMDPBehavior,
    *,
    count: int,
    phantom_sets: int,
    rollout_seed: int,
    phantom_seed: int,
) -> TrajectoryArrays:
    rollout_rng = np.random.default_rng(rollout_seed)
    phantom_rng = np.random.default_rng(phantom_seed)
    return _stack(
        [
            behavior.sample_trajectory(
                rollout_rng,
                initial_state=0,
                phantom_sets=phantom_sets,
                phantom_rng=phantom_rng,
            )
            for _ in range(count)
        ]
    )


def _targets(
    batch: TrajectoryArrays,
    ratios: FloatArray,
    baseline: FloatArray,
    gamma: float,
    rho_bar: float,
    c_bar: float,
) -> FloatArray:
    return batch_vtrace_initial(
        states=batch.states,
        rewards=batch.rewards,
        ratios=ratios,
        baseline_values=baseline,
        gamma=gamma,
        rho_bar=rho_bar,
        c_bar=c_bar,
    )


def _statistics(values: FloatArray, truth: float, batch_size: int) -> dict[str, float]:
    mean = float(values.mean())
    bias = mean - truth
    variance = float(values.var())
    return {
        "mean": mean,
        "bias": bias,
        "absolute_bias": abs(bias),
        "variance": variance,
        "single_sample_mse": variance + bias * bias,
        "batch_mean_mse": variance / batch_size + bias * bias,
    }


def _configuration_grid() -> list[tuple[float, float, float]]:
    return [
        (float(mixture), rho_bar, c_bar)
        for mixture in np.linspace(0.0, 1.0, 11)
        for rho_bar in (2.0, 5.0, 10.0)
        for c_bar in (0.5, 1.0, 2.0)
        if c_bar <= rho_bar
    ]


def _safe_configuration_grid() -> list[tuple[float, float, float]]:
    """Return the low-trace-clipping region used for pilot selection."""
    return [
        (float(mixture), rho_bar, 0.5)
        for mixture in np.linspace(0.0, 1.0, 11)
        for rho_bar in (2.0, 5.0, 10.0)
    ]


def _median_of_means(values: FloatArray, groups: int = 20) -> float:
    """Estimate a location robustly when a few trajectories have huge weight."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        raise ValueError("median-of-means requires at least one value")
    group_count = min(groups, flat.size)
    group_means = [float(chunk.mean()) for chunk in np.array_split(flat, group_count)]
    return float(np.median(group_means))


def _mixed_ratios(
    joint: FloatArray, marginal: FloatArray, mixture: float
) -> FloatArray:
    return (1.0 - mixture) * joint + mixture * marginal


def _select_configuration(
    *,
    pilot: TrajectoryArrays,
    baseline: FloatArray,
    gamma: float,
    batch_size: int,
    mom_groups: int,
) -> tuple[float, float, float, dict[str, float]]:
    # A high-variance, weakly clipped MC-marginal target supplies a data-only
    # reference mean. Candidate configurations trade squared deviation from
    # this reference against their variance at the intended update batch size.
    reference = _targets(
        pilot,
        pilot.mc_marginal_ratios,
        baseline,
        gamma,
        rho_bar=100.0,
        c_bar=100.0,
    )
    reference_mean = _median_of_means(reference, groups=mom_groups)
    best: tuple[float, float, float] | None = None
    best_diagnostics: dict[str, float] | None = None
    default_score: float | None = None
    for mixture, rho_bar, c_bar in _safe_configuration_grid():
        ratios = _mixed_ratios(
            pilot.joint_ratios, pilot.mc_marginal_ratios, mixture
        )
        values = _targets(
            pilot, ratios, baseline, gamma, rho_bar=rho_bar, c_bar=c_bar
        )
        variance = float(values.var())
        candidate_mean = _median_of_means(values, groups=mom_groups)
        deviation = candidate_mean - reference_mean
        score = deviation * deviation + variance / batch_size
        if mixture == 0.0 and rho_bar == 2.0 and c_bar == 0.5:
            default_score = score
        if best_diagnostics is None or score < best_diagnostics["proxy_score"]:
            best = (mixture, rho_bar, c_bar)
            best_diagnostics = {
                "proxy_score": score,
                "reference_mean": reference_mean,
                "candidate_mean": candidate_mean,
                "candidate_variance": variance,
            }
    assert best is not None and best_diagnostics is not None
    assert default_score is not None
    best_diagnostics["default_proxy_score"] = default_score
    best_diagnostics["relative_proxy_improvement"] = (
        default_score - best_diagnostics["proxy_score"]
    ) / max(default_score, 1e-12)
    return (*best, best_diagnostics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(31, 41)))
    parser.add_argument("--pilot-episodes", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--phantom-sets", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mom-groups", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _run_seed(args: argparse.Namespace, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    mdp = LayeredMDP.random(seed=seed)
    true_values, true_q = mdp.values()
    critic_q = true_q + rng.normal(scale=0.35, size=true_q.shape)
    behavior = CandidateMDPBehavior(
        mdp=mdp,
        critic_q=critic_q,
        candidate_count=4,
        beta=2.0,
        epsilon=0.05,
    )
    baseline = true_values.copy()
    baseline[:-1] += rng.normal(scale=0.30, size=baseline[:-1].shape)
    baseline[-1] = 0.0
    pilot = _collect(
        behavior,
        count=args.pilot_episodes,
        phantom_sets=args.phantom_sets,
        rollout_seed=seed + 10_000,
        phantom_seed=seed + 20_000,
    )
    evaluation = _collect(
        behavior,
        count=args.eval_episodes,
        phantom_sets=args.phantom_sets,
        rollout_seed=seed + 30_000,
        phantom_seed=seed + 40_000,
    )
    mixture, rho_bar, c_bar, selection = _select_configuration(
        pilot=pilot,
        baseline=baseline,
        gamma=mdp.gamma,
        batch_size=args.batch_size,
        mom_groups=args.mom_groups,
    )
    truth = float(true_values[0, 0])

    fixed_configs = {
        "joint_clipped": (0.0, 10.0, 1.0),
        "mc_marginal_clipped": (1.0, 10.0, 1.0),
        "safe_static": (0.0, 2.0, 0.5),
        "adaptive_mixture": (mixture, rho_bar, c_bar),
    }
    metrics: dict[str, dict[str, float]] = {}
    for name, (mix, rho, c) in fixed_configs.items():
        ratios = _mixed_ratios(
            evaluation.joint_ratios, evaluation.mc_marginal_ratios, mix
        )
        values = _targets(evaluation, ratios, baseline, mdp.gamma, rho, c)
        metrics[name] = _statistics(values, truth, args.batch_size)

    oracle: dict[str, float] | None = None
    oracle_config: tuple[float, float, float] | None = None
    grid_metrics: dict[str, dict[str, float]] = {}
    for mix, rho, c in _configuration_grid():
        ratios = _mixed_ratios(
            evaluation.joint_ratios, evaluation.mc_marginal_ratios, mix
        )
        values = _targets(evaluation, ratios, baseline, mdp.gamma, rho, c)
        candidate = _statistics(values, truth, args.batch_size)
        config_key = f"lambda={mix:.1f},rho={rho:.1f},c={c:.1f}"
        grid_metrics[config_key] = candidate
        if oracle is None or candidate["batch_mean_mse"] < oracle["batch_mean_mse"]:
            oracle = candidate
            oracle_config = (mix, rho, c)
    assert oracle is not None and oracle_config is not None
    metrics["oracle_grid"] = oracle
    return {
        "seed": seed,
        "truth": truth,
        "selected": {
            "mixture": mixture,
            "rho_bar": rho_bar,
            "c_bar": c_bar,
            **selection,
        },
        "oracle": {
            "mixture": oracle_config[0],
            "rho_bar": oracle_config[1],
            "c_bar": oracle_config[2],
        },
        "metrics": metrics,
        "grid_metrics": grid_metrics,
    }


def _aggregate(per_seed: list[dict[str, object]]) -> dict[str, object]:
    names = per_seed[0]["metrics"].keys()
    metrics = {}
    for name in names:
        rows = [row["metrics"][name] for row in per_seed]
        metrics[name] = {
            key: float(np.mean([item[key] for item in rows]))
            for key in rows[0]
        }
    selected_mixtures = [row["selected"]["mixture"] for row in per_seed]
    grid_keys = per_seed[0]["grid_metrics"].keys()
    static_grid = {
        config_key: {
            metric: float(
                np.mean(
                    [row["grid_metrics"][config_key][metric] for row in per_seed]
                )
            )
            for metric in per_seed[0]["grid_metrics"][config_key]
        }
        for config_key in grid_keys
    }
    best_static_key = min(
        static_grid, key=lambda key: static_grid[key]["batch_mean_mse"]
    )
    return {
        "metrics": metrics,
        "best_static_config": best_static_key,
        "best_static_metrics": static_grid[best_static_key],
        "selected_mixture_mean": float(np.mean(selected_mixtures)),
        "selected_mixture_values": selected_mixtures,
        "selected_configs": [row["selected"] for row in per_seed],
        "oracle_configs": [row["oracle"] for row in per_seed],
    }


def main() -> None:
    args = build_parser().parse_args()
    per_seed = [_run_seed(args, seed) for seed in args.seeds]
    result = {
        "config": {
            "seeds": args.seeds,
            "pilot_episodes": args.pilot_episodes,
            "eval_episodes": args.eval_episodes,
            "phantom_sets": args.phantom_sets,
            "batch_size": args.batch_size,
            "mom_groups": args.mom_groups,
        },
        "aggregate": _aggregate(per_seed),
        "per_seed": per_seed,
    }
    serialized = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not args.quiet:
        print(serialized)


if __name__ == "__main__":
    main()
