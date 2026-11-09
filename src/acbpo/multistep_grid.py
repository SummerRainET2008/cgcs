from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .multistep_experiment import build_parser as build_single_parser
from .multistep_experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(31, 41)))
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--phantom-sets", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _aggregate(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    estimator_names = results[0]["estimators"].keys()
    aggregate: dict[str, dict[str, float]] = {}
    for name in estimator_names:
        rows = [result["estimators"][name] for result in results]
        aggregate[name] = {
            "mean_absolute_bias": float(
                np.mean([row["absolute_bias"] for row in rows])
            ),
            "mean_variance": float(np.mean([row["variance"] for row in rows])),
            "mean_mse": float(np.mean([row["mse"] for row in rows])),
            "median_mse": float(np.median([row["mse"] for row in rows])),
        }
    return aggregate


def _single_args(
    *, seed: int, episodes: int, phantom_sets: int, rho_bar: float, c_bar: float
) -> argparse.Namespace:
    args = build_single_parser().parse_args([])
    args.seed = seed
    args.episodes = episodes
    args.phantom_sets = phantom_sets
    args.rho_bar = rho_bar
    args.c_bar = c_bar
    args.output = None
    return args


def main() -> None:
    args = build_parser().parse_args()
    settings = {
        "vtrace_clipped": (10.0, 1.0),
        "vtrace_moderate": (10.0, 2.0),
        "vtrace_unclipped": (100.0, 100.0),
    }
    output: dict[str, object] = {
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "phantom_sets": args.phantom_sets,
        "settings": {},
    }
    for setting_name, (rho_bar, c_bar) in settings.items():
        seed_results = [
            run_experiment(
                _single_args(
                    seed=seed,
                    episodes=args.episodes,
                    phantom_sets=args.phantom_sets,
                    rho_bar=rho_bar,
                    c_bar=c_bar,
                )
            )
            for seed in args.seeds
        ]
        output["settings"][setting_name] = {
            "rho_bar": rho_bar,
            "c_bar": c_bar,
            "aggregate": _aggregate(seed_results),
            "per_seed": seed_results,
        }
    serialized = json.dumps(output, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if not args.quiet:
        print(serialized)


if __name__ == "__main__":
    main()
