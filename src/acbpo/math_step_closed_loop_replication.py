from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Sequence
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import extract_final_answer, load_gsm8k
from .math_step_action_pilot import step_messages, terminal_result


ARMS = (
    "actor_zero",
    "random",
    "gated_random",
    "ensemble",
    "gated_ensemble",
)


def argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=values.__getitem__)


def select_replication_actions(
    critic_scores: Sequence[Sequence[float]],
    *,
    threshold: float,
    random_index: int,
    gated_random_index: int,
) -> tuple[dict[str, int], dict[str, Any]]:
    if len(critic_scores) != 3 or any(len(row) != 4 for row in critic_scores):
        raise ValueError("the frozen replication requires three critics and K=4")
    tops = [argmax(scores) for scores in critic_scores]
    ensemble = [mean(scores[index] for scores in critic_scores) for index in range(4)]
    ensemble_top = argmax(ensemble)
    agreement = len(set(tops)) == 1
    advantage = ensemble[ensemble_top] - ensemble[0]
    gate_eligible = agreement and advantage >= threshold
    gated_choice = ensemble_top if gate_eligible else 0
    intervene = gated_choice != 0
    choices = {
        "actor_zero": 0,
        "random": random_index,
        "gated_random": gated_random_index if intervene else 0,
        "ensemble": ensemble_top,
        "gated_ensemble": gated_choice,
    }
    return choices, {
        "critic_tops": tops,
        "ensemble_scores": ensemble,
        "agreement": agreement,
        "ensemble_advantage_over_zero": advantage,
        "gate_eligible": gate_eligible,
        "gate_intervenes": intervene,
    }


def post_json(url: str, path: str, payload: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    request = Request(
        url.rstrip("/") + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"service {path} returned HTTP {exc.code}: {body[:1000]}") from exc
    if "error" in result:
        raise RuntimeError(str(result["error"]))
    return result


def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        runs = [example["arms"][arm] for example in examples]
        states = [state for run in runs for state in run["trace"]]
        result[arm] = {
            "examples": len(runs),
            "accuracy": mean(float(run["result"]["correct"]) for run in runs),
            "answered_fraction": mean(float(run["result"]["answered"]) for run in runs),
            "mean_steps": mean(len(run["steps"]) for run in runs),
            "intervention_fraction": (
                mean(float(state["choice"] != 0) for state in states) if states else 0.0
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BF16 step-action replication")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--step-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.candidates != 4:
        raise ValueError("the frozen replication requires exactly four candidates")
    gate = json.loads(args.gate.read_text())
    threshold = float(gate["selected"]["threshold"])
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)

    def generate(question: str, prefix: Sequence[str], seed: int) -> str:
        prompt = qwen_chat_prompt(step_messages(question, prefix))
        row = post_json(
            args.server_url,
            "/completion",
            {
                "prompt": prompt,
                "seed": seed,
                "n_predict": args.step_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": 40,
                "repeat_penalty": 1.05,
            },
        )
        text = str(row.get("content", "")).replace("<|im_end|>", "").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0] if lines else "(empty step)"

    existing: list[dict[str, Any]] = []
    if args.output.exists():
        prior = json.loads(args.output.read_text())
        expected = {"offset": args.offset, "count": args.count, "seed": args.seed}
        observed = {key: prior["config"][key] for key in expected}
        if observed != expected:
            raise ValueError(f"resume config mismatch: {observed} != {expected}")
        existing = prior.get("examples", [])
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
                seeds = [seed_base + step_index * 10_007 + index for index in range(4)]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    candidates = list(executor.map(lambda seed: generate(example.question, prefix, seed), seeds))
                scored = post_json(
                    args.server_url,
                    "/score_all",
                    {"problem": example.question, "prefix": prefix, "candidates": candidates},
                )
                random_index = random.Random(seed_base + 9_000_019 + step_index * 1_009).randrange(4)
                gated_random_index = random.Random(
                    seed_base + 19_000_033 + step_index * 2_003
                ).choice((1, 2, 3))
                choices, diagnostics = select_replication_actions(
                    scored["scores"],
                    threshold=threshold,
                    random_index=random_index,
                    gated_random_index=gated_random_index,
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
                            "critic_scores": scored["scores"],
                            "choice": choice,
                            **diagnostics,
                        }
                    )
                    arms[arm]["done"] = extract_final_answer(step) is not None
        for arm in ARMS:
            arms[arm]["result"] = terminal_result(arms[arm]["steps"], example.reference_answer)
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
            "formal_replication": True,
            "precision": "bfloat16",
            "config": vars(args)
            | {"data": str(args.data), "output": str(args.output), "gate": str(args.gate)},
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
