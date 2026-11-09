from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .llama_cpp_policy import LlamaCppEndpoints, LlamaCppToolPolicy
from .llama_critic_experiment import _parallel_map, _run_episode, _summarize


ARGUMENT_PROMPT_VERSION = "schema-and-exact-identifiers-v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--actor-urls", nargs="+", required=True)
    parser.add_argument("--eval-split", choices=["train", "dev"], default="train")
    parser.add_argument("--eval-start", type=int, default=400)
    parser.add_argument("--eval-count", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[17001])
    parser.add_argument("--selector", choices=["actor", "mask", "logprob"], default="mask")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--progress-scale", type=float, default=2.0)
    parser.add_argument("--error-penalty", type=float, default=0.5)
    parser.add_argument("--repeat-penalty", type=float, default=0.2)
    parser.add_argument("--failed-respond-penalty", type=float, default=0.5)
    parser.add_argument("--failed-timeout-penalty", type=float, default=0.25)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".run/actor-cache"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policies = [
        LlamaCppToolPolicy(LlamaCppEndpoints(actor=url))
        for url in args.actor_urls
    ]

    def execute(policy: LlamaCppToolPolicy, job: dict[str, Any]):
        return _run_episode(policy=policy, args=args, **job)

    fingerprint_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"output", "cache_dir", "actor_urls"}
    }
    fingerprint_config["argument_prompt_version"] = ARGUMENT_PROMPT_VERSION
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    jobs = [
        {
            "critic": None,
            "selector_name": args.selector,
            "tau_root": args.tau_root,
            "split": args.eval_split,
            "task_id": task_id,
            "rollout": seed_index,
            "seed": eval_seed,
        }
        for seed_index, eval_seed in enumerate(args.eval_seeds)
        for task_id in range(args.eval_start, args.eval_start + args.eval_count)
    ]
    episodes = _parallel_map(
        policies,
        jobs,
        execute,
        cache_dir=args.cache_dir,
        fingerprint=fingerprint,
    )
    result = {
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "argument_prompt_version": ARGUMENT_PROMPT_VERSION,
        },
        "summaries": {args.selector: _summarize(episodes)},
        "evaluation_tasks": {
            args.selector: [episode.result for episode in episodes]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
