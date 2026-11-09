from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Sequence
from urllib.request import Request, urlopen

from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import extract_final_answer, load_gsm8k
from .math_step_action_pilot import EndpointPool, step_messages, terminal_result


ARMS = (
    "actor_zero",
    "random",
    "critic_95001",
    "critic_95002",
    "critic_95003",
    "ensemble",
    "gated_ensemble",
)


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def select_actions(
    critic_scores: Sequence[Sequence[float]],
    *,
    threshold: float,
    random_index: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    tops = [argmax(scores) for scores in critic_scores]
    ensemble = [mean(scores[index] for scores in critic_scores) for index in range(4)]
    ensemble_top = argmax(ensemble)
    agreement = len(set(tops)) == 1
    advantage = ensemble[ensemble_top] - ensemble[0]
    gated = ensemble_top if agreement and advantage >= threshold else 0
    choices = {
        "actor_zero": 0,
        "random": random_index,
        "critic_95001": tops[0],
        "critic_95002": tops[1],
        "critic_95003": tops[2],
        "ensemble": ensemble_top,
        "gated_ensemble": gated,
    }
    return choices, {
        "critic_tops": tops,
        "ensemble_scores": ensemble,
        "agreement": agreement,
        "ensemble_advantage_over_zero": advantage,
        "gate_eligible": agreement and advantage >= threshold,
    }


def post_json(url: str, payload: dict[str, Any], timeout: float = 240.0) -> dict[str, Any]:
    request = Request(
        url.rstrip("/") + "/score",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        runs = [example["arms"][arm] for example in examples]
        result[arm] = {
            "examples": len(runs),
            "accuracy": mean(float(run["result"]["correct"]) for run in runs),
            "answered_fraction": mean(float(run["result"]["answered"]) for run in runs),
            "mean_steps": mean(len(run["steps"]) for run in runs),
            "intervention_fraction": (
                mean(
                    float(state["choice"] != 0)
                    for run in runs
                    for state in run["trace"]
                )
                if any(run["trace"] for run in runs)
                else 0.0
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gated step-critic closed loop")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--actor-url", required=True)
    parser.add_argument("--critic-urls", nargs=3, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--step-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=97001)
    args = parser.parse_args()
    if args.candidates != 4:
        raise ValueError("the frozen protocol requires exactly four candidates")
    gate = json.loads(args.gate.read_text())
    threshold = float(gate["selected"]["threshold"])
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)
    actor = EndpointPool([args.actor_url])

    def generate(question: str, prefix: Sequence[str], seed: int) -> str:
        prompt = qwen_chat_prompt(step_messages(question, prefix))
        return actor.complete(prompt, seed=seed, args=args)

    existing: list[dict[str, Any]] = []
    if args.output.exists():
        existing = json.loads(args.output.read_text()).get("examples", [])
    completed = {row["example_id"] for row in existing}
    started = time.time()
    for example in examples:
        if example.example_id in completed:
            continue
        arms = {arm: {"steps": [], "trace": [], "done": False} for arm in ARMS}
        seed_base = args.seed + example.example_id * 100_003
        for step_index in range(args.max_steps):
            active = [arm for arm in ARMS if not arms[arm]["done"]]
            if not active:
                break
            groups: dict[tuple[str, ...], list[str]] = {}
            for arm in active:
                groups.setdefault(tuple(arms[arm]["steps"]), []).append(arm)
            for prefix_tuple, group_arms in groups.items():
                prefix = list(prefix_tuple)
                seeds = [seed_base + step_index * 10_007 + i for i in range(4)]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    candidates = list(
                        executor.map(
                            lambda seed: generate(example.question, prefix, seed), seeds
                        )
                    )
                payload = {
                    "problem": example.question,
                    "prefix": prefix,
                    "candidates": candidates,
                }
                with ThreadPoolExecutor(max_workers=3) as executor:
                    critic_rows = list(
                        executor.map(lambda url: post_json(url, payload), args.critic_urls)
                    )
                critic_scores = [row["scores"] for row in critic_rows]
                random_index = random.Random(
                    seed_base + 9_000_019 + step_index * 1_009
                ).randrange(4)
                choices, diagnostics = select_actions(
                    critic_scores, threshold=threshold, random_index=random_index
                )
                for arm in group_arms:
                    choice = choices[arm]
                    step = candidates[choice]
                    arms[arm]["steps"].append(step)
                    arms[arm]["trace"].append(
                        {
                            "step_index": step_index,
                            "prefix": prefix,
                            "candidates": candidates,
                            "critic_scores": critic_scores,
                            "choice": choice,
                            **diagnostics,
                        }
                    )
                    arms[arm]["done"] = extract_final_answer(step) is not None
        for arm in ARMS:
            arms[arm]["result"] = terminal_result(
                arms[arm]["steps"], example.reference_answer
            )
            del arms[arm]["done"]
        existing.append(
            {
                "example_id": example.example_id,
                "question": example.question,
                "reference_answer": example.reference_answer,
                "arms": arms,
            }
        )
        payload = {
            "exploratory": True,
            "config": vars(args) | {"data": str(args.data), "output": str(args.output), "gate": str(args.gate)},
            "gate": gate,
            "elapsed_seconds": time.time() - started,
            "summary": summarize(existing),
            "examples": existing,
        }
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"example_id": example.example_id, "summary": payload["summary"]}), flush=True)


if __name__ == "__main__":
    main()
