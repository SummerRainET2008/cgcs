from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llama_critic_experiment import _run_episode, _summarize, paired_bootstrap
from .peft_actor_policy import PeftActorOnlyPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--eval-split", choices=["train", "dev"], default="train")
    parser.add_argument("--eval-start", type=int, default=420)
    parser.add_argument("--eval-count", type=int, default=4)
    parser.add_argument("--eval-seed", type=int, default=78001)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--choice-temperature", type=float, default=4.0)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--argument-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--strict-intent-mask", action="store_true")
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument(
        "--arms", nargs="+", choices=["base", "lora"], default=["base", "lora"]
    )
    parser.add_argument("--progress-scale", type=float, default=2.0)
    parser.add_argument("--error-penalty", type=float, default=0.5)
    parser.add_argument("--repeat-penalty", type=float, default=0.2)
    parser.add_argument("--failed-respond-penalty", type=float, default=0.5)
    parser.add_argument("--failed-timeout-penalty", type=float, default=0.25)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    policy = PeftActorOnlyPolicy(
        args.model,
        args.adapter,
        use_adapter_for_choices=False,
        max_context_tokens=args.max_context_tokens,
        head_tokens=args.head_tokens,
    )
    arms: dict[str, list] = {arm: [] for arm in args.arms}
    for arm, use_adapter in (
        (arm, arm == "lora") for arm in args.arms
    ):
        policy.use_adapter_for_choices = use_adapter
        for task_id in range(args.eval_start, args.eval_start + args.eval_count):
            episode = _run_episode(
                policy=policy,
                critic=None,
                selector_name="mask",
                tau_root=args.tau_root,
                split=args.eval_split,
                task_id=task_id,
                rollout=0,
                seed=args.eval_seed,
                args=args,
            )
            arms[arm].append(episode)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "task_id": task_id,
                        "success": episode.result["reward"],
                        "steps": episode.result["steps"],
                        "progress": episode.result["database_progress"],
                    }
                ),
                flush=True,
            )
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "invariants": {
            "critic_used": False,
            "action_mask_shared": True,
            "strict_intent_mask": args.strict_intent_mask,
            "argument_generator": "base_nf4_adapter_disabled",
            "lora_scope": "action_name_choice_only",
        },
        "summaries": {arm: _summarize(episodes) for arm, episodes in arms.items()},
        "evaluation_tasks": {
            arm: [episode.result for episode in episodes]
            for arm, episodes in arms.items()
        },
    }
    if set(arms) == {"base", "lora"}:
        result["comparison"] = paired_bootstrap(
            arms["base"], arms["lora"], seed=args.seed, samples=20_000
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "summaries": result["summaries"],
                "comparison": result.get("comparison"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
