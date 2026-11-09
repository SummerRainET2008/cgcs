from __future__ import annotations

import argparse
import json
from pathlib import Path

from acbpo.locked_selector_evaluation import evaluate_locked_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate preregistered selector evaluations over training seeds"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--beta", type=float, action="append", required=True)
    parser.add_argument("--epsilon", type=float, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--method", default="abs_advantage")
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_locked_ensemble(
        args.data,
        args.predictions,
        method=args.method,
        betas=args.beta,
        epsilons=args.epsilon,
        labels=args.label,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
