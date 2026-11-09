from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .multistep_experiment import build_parser as build_single_parser
from .multistep_experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(31, 36)))
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--phantom-grid", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _single_args(seed: int, episodes: int, phantom_sets: int) -> argparse.Namespace:
    args = build_single_parser().parse_args([])
    args.seed = seed
    args.episodes = episodes
    args.phantom_sets = phantom_sets
    args.rho_bar = 10.0
    args.c_bar = 1.0
    args.output = None
    return args


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mean_absolute_bias": float(np.mean([row["absolute_bias"] for row in rows])),
        "mean_variance": float(np.mean([row["variance"] for row in rows])),
        "mean_mse": float(np.mean([row["mse"] for row in rows])),
    }


def main() -> None:
    args = build_parser().parse_args()
    estimator_names = (
        "joint_pdis",
        "marginal_pdis",
        "mc_marginal_pdis",
        "joint_vtrace",
        "marginal_vtrace",
        "mc_marginal_vtrace",
    )
    output: dict[str, object] = {
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "rho_bar": 10.0,
        "c_bar": 1.0,
        "phantom_grid": {},
    }
    for phantom_sets in args.phantom_grid:
        seed_results = [
            run_experiment(_single_args(seed, args.episodes, phantom_sets))
            for seed in args.seeds
        ]
        output["phantom_grid"][str(phantom_sets)] = {
            name: _aggregate(
                [result["estimators"][name] for result in seed_results]
            )
            for name in estimator_names
        }
    serialized = json.dumps(output, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not args.quiet:
        print(serialized)


if __name__ == "__main__":
    main()
