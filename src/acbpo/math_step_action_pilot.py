from __future__ import annotations

import argparse
import json
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .llama_cpp_policy import qwen_chat_prompt
from .math_rollout_experiment import (
    GSM8KExample,
    answers_match,
    extract_final_answer,
    load_gsm8k,
)


def step_messages(question: str, steps: Sequence[str]) -> list[dict[str, str]]:
    prefix = "\n".join(f"Step {index}: {step}" for index, step in enumerate(steps, 1))
    if not prefix:
        prefix = "(no previous steps)"
    return [
        {
            "role": "system",
            "content": (
                "Solve the math problem one concise reasoning step at a time. "
                "Return exactly one next step on one line. Do not repeat previous "
                "steps. If the solution is complete, return FINAL: <numeric answer>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\nSolution so far:\n{prefix}\n\nNext step:"
            ),
        },
    ]


def clean_step(text: str) -> str:
    text = text.replace("<|im_end|>", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "(empty step)"
    first = lines[0]
    if first.lower().startswith("step ") and ":" in first:
        first = first.split(":", 1)[1].strip()
    return first or "(empty step)"


class EndpointPool:
    def __init__(self, urls: Sequence[str], *, timeout: float = 240.0):
        if not urls:
            raise ValueError("at least one llama.cpp URL is required")
        self._urls: queue.Queue[str] = queue.Queue()
        for url in urls:
            self._urls.put(url.rstrip("/"))
        self.timeout = timeout

    def complete(self, prompt: str, *, seed: int, args: argparse.Namespace) -> str:
        url = self._urls.get()
        try:
            payload = {
                "prompt": prompt,
                "n_predict": args.step_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": 40,
                "min_p": 0.0,
                "repeat_penalty": 1.05,
                "seed": seed,
                "stop": ["\n", "<|im_end|>"],
                "cache_prompt": True,
            }
            request = Request(
                url + "/completion",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = json.load(response)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"llama.cpp returned HTTP {exc.code}: {body[:500]}"
                ) from exc
            except (URLError, TimeoutError) as exc:
                raise RuntimeError(f"llama.cpp request failed: {exc}") from exc
            return clean_step(str(result.get("content", "")))
        finally:
            self._urls.put(url)


def terminal_result(steps: Sequence[str], reference: str) -> dict[str, Any]:
    prediction = next(
        (
            answer
            for answer in (extract_final_answer(step) for step in reversed(steps))
            if answer is not None
        ),
        None,
    )
    return {
        "prediction": prediction,
        "answered": prediction is not None,
        "correct": answers_match(prediction, reference),
    }


def summarize_examples(examples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    states = [state for example in examples for state in example["states"]]
    candidates = [candidate for state in states for candidate in state["candidates"]]
    continuations = [row for candidate in candidates for row in candidate["continuations"]]
    mixed = [
        state
        for state in states
        if len({candidate["return_mean"] for candidate in state["candidates"]}) > 1
    ]
    uniform = [
        mean(candidate["return_mean"] for candidate in state["candidates"])
        for state in states
    ]
    oracle = [
        max(candidate["return_mean"] for candidate in state["candidates"])
        for state in states
    ]
    return {
        "examples": len(examples),
        "states": len(states),
        "candidate_actions": len(candidates),
        "continuations": len(continuations),
        "answered_fraction": (
            mean(float(row["answered"]) for row in continuations)
            if continuations
            else math.nan
        ),
        "correct_fraction": (
            mean(float(row["correct"]) for row in continuations)
            if continuations
            else math.nan
        ),
        "mixed_state_fraction": len(mixed) / len(states) if states else math.nan,
        "uniform_expected_return": mean(uniform) if uniform else math.nan,
        "oracle_expected_return": mean(oracle) if oracle else math.nan,
        "oracle_headroom": (
            mean(o - u for o, u in zip(oracle, uniform)) if uniform else math.nan
        ),
        "empty_step_fraction": (
            mean(float(candidate["step"] == "(empty step)") for candidate in candidates)
            if candidates
            else math.nan
        ),
    }


def collect_example(
    example: GSM8KExample,
    *,
    generate: Callable[[str, Sequence[str], int], str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    prefix: list[str] = []
    states: list[dict[str, Any]] = []
    seed_base = args.seed + example.example_id * 100_003

    for step_index in range(args.max_steps):
        candidate_seeds = [seed_base + step_index * 10_007 + i for i in range(args.candidates)]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            candidate_steps = list(
                executor.map(
                    lambda seed: generate(example.question, prefix, seed), candidate_seeds
                )
            )

        candidate_rows: list[dict[str, Any]] = []
        jobs: list[tuple[int, int]] = [
            (candidate_index, continuation_index)
            for candidate_index in range(args.candidates)
            for continuation_index in range(args.continuations)
        ]

        def run_continuation(job: tuple[int, int]) -> tuple[int, dict[str, Any]]:
            candidate_index, continuation_index = job
            steps = [*prefix, candidate_steps[candidate_index]]
            continuation_seed = (
                seed_base
                + 1_000_003
                + step_index * 100_003
                + candidate_index * 1_009
                + continuation_index
            )
            while len(steps) < args.max_steps and extract_final_answer(steps[-1]) is None:
                steps.append(generate(example.question, steps, continuation_seed + len(steps)))
            result = terminal_result(steps, example.reference_answer)
            return candidate_index, {
                "continuation": continuation_index,
                "steps": steps[len(prefix) + 1 :],
                **result,
            }

        grouped: list[list[dict[str, Any]]] = [[] for _ in candidate_steps]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for candidate_index, result in executor.map(run_continuation, jobs):
                grouped[candidate_index].append(result)
        for candidate_index, step in enumerate(candidate_steps):
            rows = sorted(grouped[candidate_index], key=lambda row: row["continuation"])
            candidate_rows.append(
                {
                    "candidate": candidate_index,
                    "step": step,
                    "terminal": extract_final_answer(step) is not None,
                    "return_mean": mean(float(row["correct"]) for row in rows),
                    "answered_mean": mean(float(row["answered"]) for row in rows),
                    "continuations": rows,
                }
            )
        states.append(
            {
                "step_index": step_index,
                "prefix": list(prefix),
                "candidates": candidate_rows,
            }
        )

        # Candidate zero is an ordinary actor sample and defines the on-policy
        # main trajectory used to harvest the next state.  Continuation labels
        # never influence this choice.
        prefix.append(candidate_steps[0])
        if extract_final_answer(candidate_steps[0]) is not None:
            break

    return {
        "example_id": example.example_id,
        "question": example.question,
        "reference_answer": example.reference_answer,
        "main_steps": prefix,
        "main_result": terminal_result(prefix, example.reference_answer),
        "states": states,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory step-action branch pilot")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-urls", nargs="+", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--step-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--workers", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("count", "candidates", "continuations", "max_steps", "step_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    args.workers = args.workers or len(args.actor_urls)
    if args.workers < 1 or args.workers > len(args.actor_urls):
        raise ValueError("workers must be between 1 and the number of actor URLs")
    random.seed(args.seed)
    examples = load_gsm8k(args.data, offset=args.offset, count=args.count)
    pool = EndpointPool(args.actor_urls)

    def generate(question: str, steps: Sequence[str], seed: int) -> str:
        return pool.complete(
            qwen_chat_prompt(step_messages(question, steps)), seed=seed, args=args
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    for example in examples:
        completed.append(collect_example(example, generate=generate, args=args))
        payload = {
            "exploratory": True,
            "config": {
                **vars(args),
                "data": str(args.data),
                "output": str(args.output),
            },
            "elapsed_seconds": time.monotonic() - started,
            "summary": summarize_examples(completed),
            "examples": completed,
        }
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "example_id": example.example_id,
                    "examples": len(completed),
                    "summary": payload["summary"],
                }
            ),
            flush=True,
        )

    print(json.dumps(summarize_examples(completed), indent=2), flush=True)


if __name__ == "__main__":
    main()
