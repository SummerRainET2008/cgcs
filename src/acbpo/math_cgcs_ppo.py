from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np

from .actor_lora_experiment import middle_truncate
from .cgcs_ppo import (
    effective_sample_size,
    uniform_selection_weight,
    weighted_ppo_loss,
)
from .math_rollout_experiment import answers_match, extract_final_answer
from .selector_utility_experiment import proposal_distribution


@dataclass(frozen=True)
class MathProblem:
    problem_id: str
    question: str
    answer: str


NUMERIC_ANSWER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d+)?")


def load_unique_prm_problems(path: Path) -> list[MathProblem]:
    problems: dict[str, MathProblem] = {}
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            if "problem" in item and "ground_truth_answer" in item:
                problem_id = str(item["problem_id"])
                question = str(item["problem"]).strip()
                answer = str(item["ground_truth_answer"]).strip()
            elif "question" in item and "answer" in item:
                problem_id = f"gsm8k-test-{row_index}"
                question = str(item["question"]).strip()
                raw_answer = str(item["answer"])
                if "####" not in raw_answer:
                    raise ValueError("GSM8K answer is missing the #### delimiter")
                answer = raw_answer.rsplit("####", 1)[1].strip()
            else:
                raise ValueError("unrecognized math problem schema")
            problem = MathProblem(
                problem_id=problem_id,
                question=question,
                answer=answer,
            )
            previous = problems.setdefault(problem_id, problem)
            if previous != problem:
                raise ValueError(f"inconsistent duplicate problem {problem_id}")
    return list(problems.values())


def solution_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return exactly two lines and fewer than 80 tokens total. Do not restate "
                "the problem. Line 1 must be a compact calculation. Line 2 must be "
                "FINAL: <numeric answer>."
            ),
        },
        {"role": "user", "content": question},
    ]


def solution_critic_messages(question: str, solution: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge one proposed next step in a math solution. Output exactly one "
                "letter: A = incorrect, B = neutral or no progress, C = correct and useful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{question}\n\nSolution so far:\n(no previous steps)\n\n"
                f"Proposed next step:\n{solution}\n\nRating:"
            ),
        },
    ]


def _prompt_ids(tokenizer, messages: list[dict[str, str]], max_length: int, head: int):
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return middle_truncate(
        tokenizer(prompt, add_special_tokens=False).input_ids, max_length, head
    )


def advantage_selector_distribution(
    critic_scores: Sequence[float], *, beta: float, epsilon: float, scale_floor: float
) -> np.ndarray:
    """Use the exact absolute-advantage proposal validated by the sealed gate."""
    scores = np.asarray(critic_scores, dtype=np.float64)
    utility = np.abs(scores - scores.mean())
    return proposal_distribution(
        utility, beta=beta, epsilon=epsilon, scale_floor=scale_floor
    )


def signed_selector_distribution(
    critic_scores: Sequence[float], *, beta: float, epsilon: float, scale_floor: float
) -> np.ndarray:
    """Prefer candidates the critic predicts are correct and useful.

    Unlike :func:`advantage_selector_distribution`, this proposal preserves the
    sign of the critic score.  It is therefore a return-oriented selector rather
    than a variance-oriented absolute-advantage proposal.
    """
    return proposal_distribution(
        critic_scores, beta=beta, epsilon=epsilon, scale_floor=scale_floor
    )


def interpolated_selection_weight(
    exact_weight: float, correction_strength: float
) -> float:
    """Interpolate between biased selector sampling and exact correction.

    ``correction_strength=0`` keeps the critic-selected sample uncorrected,
    while ``correction_strength=1`` recovers the exact conditional importance
    weight.  Unlike exponentiating the weight, this linear path preserves unit
    expected weight for every strength when the exact weight is valid:

        E_q[(1-lambda) + lambda * p/q] = 1.
    """
    if not 0.0 <= correction_strength <= 1.0:
        raise ValueError("selection correction strength must be in [0, 1]")
    return 1.0 + correction_strength * (exact_weight - 1.0)


def training_items(
    rows: Sequence[dict[str, Any]], arm: str
) -> list[tuple[dict[str, Any], int, float]]:
    """Return ``(row, candidate_index, objective_weight)`` training items.

    The selected-candidate arms retain their historical behavior.  The all-K
    arms expand every prompt into all of its candidates.  Because the expanded
    training loop averages over candidates, ``K*q_i`` implements
    ``sum_i q_i loss_i`` while a unit weight implements the uniform
    ``(1/K) sum_i loss_i`` objective.
    """
    items: list[tuple[dict[str, Any], int, float]] = []
    if arm in {"all_k_uniform", "all_k_selector"}:
        for row in rows:
            candidates = row.get("candidates", [])
            probabilities = row.get("selector_probabilities", [])
            if not candidates:
                raise ValueError("all-K training requires non-empty candidates")
            if len(probabilities) != len(candidates):
                raise ValueError("selector probabilities must match all-K candidates")
            if not math.isclose(
                sum(float(x) for x in probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6
            ):
                raise ValueError("selector probabilities must sum to one")
            k = len(candidates)
            for index in range(k):
                weight = (
                    1.0
                    if arm == "all_k_uniform"
                    else k * float(probabilities[index])
                )
                items.append((row, index, weight))
        return items

    for row in rows:
        selected = int(row[f"{arm}_selected"])
        exact_weight = float(row["cgcs_selection_weight"]) if arm == "cgcs" else 1.0
        items.append((row, selected, exact_weight))
    return items


def collect(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if not math.isclose(args.actor_temperature, 1.0) or not math.isclose(args.top_p, 1.0):
        raise ValueError(
            "exact PPO correction requires proposal sampling from the raw actor "
            "distribution (actor-temperature=1 and top-p=1)"
        )
    problems = load_unique_prm_problems(args.data)
    if args.numeric_only:
        problems = [x for x in problems if NUMERIC_ANSWER.fullmatch(x.answer)]
    problems = problems[args.offset : args.offset + args.count]
    if not problems:
        raise RuntimeError("the requested problem slice is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    labels = [tokenizer(x, add_special_tokens=False).input_ids for x in ("A", "B", "C")]
    if any(len(ids) != 1 for ids in labels):
        raise RuntimeError(f"critic labels are not single tokens: {labels}")
    label_ids = torch.tensor([ids[0] for ids in labels], device="cuda")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    actor_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    if args.actor_adapter is not None:
        actor_model = PeftModel.from_pretrained(actor_model, args.actor_adapter)
    actor_model.eval()
    critic_base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    critic_model = PeftModel.from_pretrained(critic_base, args.critic_adapter)
    critic_model.eval()
    rows: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for problem_start in range(0, len(problems), args.problem_batch_size):
        problem_batch = problems[problem_start : problem_start + args.problem_batch_size]
        prompts = [
            _prompt_ids(
                tokenizer,
                solution_messages(problem.question),
                args.max_prompt_tokens,
                args.head_tokens,
            )
            for problem in problem_batch
        ]
        padded = tokenizer.pad(
            [{"input_ids": prompt} for prompt in prompts],
            padding=True,
            return_tensors="pt",
        )
        inputs = padded["input_ids"].to("cuda")
        attention_mask = padded["attention_mask"].to("cuda")
        with torch.no_grad():
            output = actor_model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=args.actor_temperature,
                top_p=args.top_p,
                num_return_sequences=args.candidates,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            generated = output.sequences[:, inputs.shape[1] :]
            transition = actor_model.compute_transition_scores(
                output.sequences, output.scores, normalize_logits=True
            )

        candidate_groups: list[list[dict[str, Any]]] = []
        critic_inputs = []
        for problem_index, (problem, prompt) in enumerate(zip(problem_batch, prompts)):
            candidates = []
            start = problem_index * args.candidates
            end = start + args.candidates
            for token_row, score_row in zip(generated[start:end], transition[start:end]):
                mask = token_row != tokenizer.pad_token_id
                if not bool(mask.any()):
                    mask = torch.ones_like(token_row, dtype=torch.bool)
                completion = token_row[mask].tolist()
                token_logprobs = score_row[mask].float().cpu().tolist()
                text = tokenizer.decode(completion, skip_special_tokens=True).strip()
                prediction = extract_final_answer(text)
                candidates.append(
                    {
                        "text": text,
                        "completion_ids": completion,
                        "old_token_logprobs": token_logprobs,
                        "actor_logprob": float(np.mean(token_logprobs)),
                        "prediction": prediction,
                        "reward": float(answers_match(prediction, problem.answer)),
                    }
                )
                critic_inputs.append(
                    {
                        "input_ids": _prompt_ids(
                            tokenizer,
                            solution_critic_messages(problem.question, text),
                            args.max_critic_tokens,
                            args.head_tokens,
                        )
                    }
                )
            candidate_groups.append(candidates)

        critic_batch = tokenizer.pad(critic_inputs, padding=True, return_tensors="pt")
        critic_batch = {key: value.to("cuda") for key, value in critic_batch.items()}
        with torch.no_grad():
            logits = critic_model(**critic_batch, logits_to_keep=1).logits[
                :, -1, label_ids
            ].float()
            critic_probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        probability_index = 0
        for local_index, (problem, prompt, candidates) in enumerate(
            zip(problem_batch, prompts, candidate_groups)
        ):
            for candidate in candidates:
                probabilities = critic_probabilities[probability_index]
                probability_index += 1
                candidate["critic_probabilities"] = [float(x) for x in probabilities]
                candidate["critic_score"] = -float(probabilities[0]) + float(probabilities[2])
            q = advantage_selector_distribution(
                [x["critic_score"] for x in candidates],
                beta=args.beta,
                epsilon=args.selector_epsilon,
                scale_floor=args.critic_scale_floor,
            )
            if args.selector_method == "signed_score":
                q = signed_selector_distribution(
                    [x["critic_score"] for x in candidates],
                    beta=args.beta,
                    epsilon=args.selector_epsilon,
                    scale_floor=args.critic_scale_floor,
                )
            elif args.selector_method == "probability_c":
                q = proposal_distribution(
                    [x["critic_probabilities"][2] for x in candidates],
                    beta=args.beta,
                    epsilon=args.selector_epsilon,
                    scale_floor=args.critic_scale_floor,
                )
            baseline_index = int(rng.integers(args.candidates))
            cgcs_index = int(rng.choice(args.candidates, p=q))
            cap = args.selection_weight_cap if args.selection_weight_cap > 0 else None
            row = {
                **asdict(problem),
                "prompt_ids": prompt,
                "candidates": candidates,
                "k1_selected": 0,
                "baseline_selected": baseline_index,
                "cgcs_selected": cgcs_index,
                "selector_probabilities": q.tolist(),
                "cgcs_selection_weight": uniform_selection_weight(
                    float(q[cgcs_index]), args.candidates, cap=cap
                ),
            }
            rows.append(row)
            problem_index = problem_start + local_index
            print(
                json.dumps(
                    {
                        "problem": args.offset + problem_index,
                        "candidate_accuracy": mean(x["reward"] for x in candidates),
                        "baseline_reward": candidates[baseline_index]["reward"],
                        "cgcs_reward": candidates[cgcs_index]["reward"],
                        "q_max": float(q.max()),
                    }
                ),
                flush=True,
            )
        args.output.write_text(
            json.dumps(
                {"config": _json_config(args), "examples": rows},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    weights = [float(x["cgcs_selection_weight"]) for x in rows]
    print(
        json.dumps(
            {
                "examples": len(rows),
                "candidate_accuracy": mean(
                    c["reward"] for row in rows for c in row["candidates"]
                ),
                "baseline_accuracy": mean(
                    row["candidates"][row["baseline_selected"]]["reward"] for row in rows
                ),
                "cgcs_accuracy": mean(
                    row["candidates"][row["cgcs_selected"]]["reward"] for row in rows
                ),
                "selection_weight_mean": mean(weights),
                "selection_weight_ess": effective_sample_size(weights),
            },
            indent=2,
        )
    )


def _json_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "func"
    }


def merge(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in args.inputs:
        payload = json.loads(path.read_text())
        for row in payload["examples"]:
            if row["problem_id"] in seen:
                raise ValueError(f"duplicate problem {row['problem_id']}")
            seen.add(row["problem_id"])
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"inputs": [str(x) for x in args.inputs], "examples": rows}) + "\n"
    )
    print(json.dumps({"examples": len(rows), "output": str(args.output)}, indent=2))


def train(args: argparse.Namespace) -> None:
    import torch
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = json.loads(args.rollouts.read_text())
    rows = list(payload["examples"])
    if args.rollout_offset < 0:
        raise ValueError("rollout offset must be non-negative")
    rows = rows[args.rollout_offset :]
    if args.max_examples:
        rows = rows[: args.max_examples]
    if not rows:
        raise RuntimeError("no rollout examples")

    prompt_examples = len(rows)
    items = training_items(rows, args.arm)
    random.Random(args.seed).shuffle(items)

    rewards = []
    exact_weights = []
    weights = []
    for row, selected, exact_weight in items:
        rewards.append(float(row["candidates"][selected]["reward"]))
        exact_weights.append(exact_weight)
        weights.append(
            (
                interpolated_selection_weight(
                    exact_weight, args.selection_correction_strength
                )
                if args.arm == "cgcs"
                else exact_weight
            )
        )
    baseline = float(np.mean(rewards))
    reward_scale = max(float(np.std(rewards)), 0.25)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ),
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if args.initial_adapter is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model, args.initial_adapter, is_trainable=True
        )
    model.config.use_cache = False
    model.train()
    hidden_size = int(model.config.hidden_size)
    value_head = torch.nn.Linear(hidden_size, 1, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        value_head.weight.zero_()
        value_head.bias.fill_(baseline)
    optimizer = torch.optim.AdamW(
        [x for x in model.parameters() if x.requires_grad] + list(value_head.parameters()),
        lr=args.learning_rate,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    records = []
    steps = 0
    updates = 0

    for epoch in range(args.epochs):
        if epoch:
            random.Random(args.seed + epoch).shuffle(items)
        for row, selected, exact_selection in items:
            candidate = row["candidates"][selected]
            prompt = list(row["prompt_ids"])
            completion = list(candidate["completion_ids"])
            old = list(candidate["old_token_logprobs"])
            if not completion or len(completion) != len(old):
                continue
            full = torch.tensor([prompt + completion], device="cuda", dtype=torch.long)
            output = model(
                input_ids=full,
                attention_mask=torch.ones_like(full),
                output_hidden_states=True,
            )
            logits = output.logits[0].float()
            positions = torch.arange(
                len(prompt) - 1, len(prompt) + len(completion) - 1, device="cuda"
            )
            tokens = torch.tensor(completion, device="cuda", dtype=torch.long)
            new_logprob = torch.log_softmax(logits[positions], dim=-1)[
                torch.arange(len(tokens), device="cuda"), tokens
            ]
            old_logprob = torch.tensor(old, device="cuda", dtype=torch.float32)
            advantage_value = (float(candidate["reward"]) - baseline) / reward_scale
            advantage = torch.full_like(new_logprob, advantage_value)
            selection = (
                interpolated_selection_weight(
                    exact_selection, args.selection_correction_strength
                )
                if args.arm == "cgcs"
                else exact_selection
            )
            selection_tensor = torch.full_like(new_logprob, selection)
            actor_loss, ratio = weighted_ppo_loss(
                new_logprob,
                old_logprob,
                advantage,
                selection_tensor,
                clip_epsilon=args.ppo_clip,
            )
            value_prediction = value_head(
                output.hidden_states[-1][0, len(prompt) - 1].float()
            ).squeeze()
            value_target = torch.tensor(
                float(candidate["reward"]), device="cuda", dtype=torch.float32
            )
            value_loss = selection * torch.square(value_prediction - value_target)
            # A small sampled KL proxy prevents a single tiny pilot from moving
            # the adapter too far from the frozen base policy.
            kl_proxy = torch.square(new_logprob - old_logprob).mean()
            total = (
                actor_loss
                + args.value_coefficient * value_loss
                + args.kl_coefficient * kl_proxy
            ) / args.gradient_accumulation
            total.backward()
            steps += 1
            if steps % args.gradient_accumulation == 0 or steps == len(items) * args.epochs:
                torch.nn.utils.clip_grad_norm_(
                    [x for x in model.parameters() if x.requires_grad]
                    + list(value_head.parameters()),
                    args.max_grad_norm,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
            record = {
                "step": steps,
                "epoch": epoch,
                "reward": float(candidate["reward"]),
                "advantage": advantage_value,
                "selection_weight": selection,
                "actor_loss": float(actor_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "value_prediction": float(value_prediction.detach().cpu()),
                "kl_proxy": float(kl_proxy.detach().cpu()),
                "ratio_mean": float(ratio.detach().mean().cpu()),
                "ratio_clip_fraction": float(
                    ((ratio < 1 - args.ppo_clip) | (ratio > 1 + args.ppo_clip))
                    .float()
                    .mean()
                    .cpu()
                ),
            }
            records.append(record)
            if steps % args.log_every == 0:
                print(json.dumps(record), flush=True)

    model.save_pretrained(args.output_dir / "adapter")
    tokenizer.save_pretrained(args.output_dir / "adapter")
    torch.save(value_head.state_dict(), args.output_dir / "value_head.pt")
    summary = {
        "config": _json_config(args),
        "examples": prompt_examples,
        "candidate_examples": len(items),
        "updates": updates,
        "reward_mean": mean(rewards),
        "weighted_reward_mean": float(np.mean(np.asarray(weights) * np.asarray(rewards))),
        "advantage_baseline": baseline,
        "selection_weight_mean": mean(weights),
        "selection_weight_ess": effective_sample_size(weights),
        "selection_weight_ess_fraction": effective_sample_size(weights) / len(weights),
        "exact_selection_weight_mean": mean(exact_weights),
        "exact_selection_weight_ess_fraction": effective_sample_size(exact_weights)
        / len(exact_weights),
        "weighted_advantage_second_moment": float(
            np.mean(
                np.square(
                    np.asarray(weights)
                    * ((np.asarray(rewards) - baseline) / reward_scale)
                )
            )
        ),
        "final_ratio_clip_fraction": mean(x["ratio_clip_fraction"] for x in records),
        "final_kl_proxy": mean(x["kl_proxy"] for x in records[-len(rows) :]),
        "final_value_loss": mean(x["value_loss"] for x in records[-len(rows) :]),
    }
    (args.output_dir / "train_log.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in records)
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    problems = load_unique_prm_problems(args.data)
    if args.numeric_only:
        problems = [x for x in problems if NUMERIC_ANSWER.fullmatch(x.answer)]
    problems = problems[args.offset : args.offset + args.count]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    examples = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, problem in enumerate(problems):
        prompt = _prompt_ids(
            tokenizer, solution_messages(problem.question), args.max_prompt_tokens, args.head_tokens
        )
        inputs = torch.tensor([prompt], device="cuda", dtype=torch.long)
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )[0, len(prompt) :]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        prediction = extract_final_answer(text)
        item = {
            **asdict(problem),
            "prediction": prediction,
            "correct": answers_match(prediction, problem.answer),
            "text": text,
        }
        examples.append(item)
        args.output.write_text(
            json.dumps(
                {
                    "config": _json_config(args),
                    "accuracy": mean(float(x["correct"]) for x in examples),
                    "examples": examples,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        print(json.dumps({"index": args.offset + index, "correct": item["correct"]}), flush=True)
    print(json.dumps({"examples": len(examples), "accuracy": mean(float(x["correct"]) for x in examples)}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CGCS-PPO frozen-critic math pilot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    common.add_argument("--data", type=Path, required=True)
    common.add_argument("--offset", type=int, default=0)
    common.add_argument("--count", type=int, default=32)
    common.add_argument("--max-prompt-tokens", type=int, default=768)
    common.add_argument("--head-tokens", type=int, default=256)
    common.add_argument("--max-new-tokens", type=int, default=96)
    common.add_argument("--seed", type=int, default=181001)
    common.add_argument("--numeric-only", action="store_true")

    collect_parser = subparsers.add_parser("collect", parents=[common])
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--actor-adapter", type=Path)
    collect_parser.add_argument("--critic-adapter", type=Path, required=True)
    collect_parser.add_argument("--candidates", type=int, default=4)
    collect_parser.add_argument("--problem-batch-size", type=int, default=1)
    collect_parser.add_argument("--actor-temperature", type=float, default=1.0)
    collect_parser.add_argument("--top-p", type=float, default=1.0)
    collect_parser.add_argument("--max-critic-tokens", type=int, default=1024)
    collect_parser.add_argument("--beta", type=float, default=0.5)
    collect_parser.add_argument(
        "--selector-method",
        choices=("absolute_deviation", "signed_score", "probability_c"),
        default="absolute_deviation",
    )
    collect_parser.add_argument("--selector-epsilon", type=float, default=0.1)
    collect_parser.add_argument("--critic-scale-floor", type=float, default=0.05)
    collect_parser.add_argument(
        "--selection-weight-cap",
        type=float,
        default=0.0,
        help="zero keeps the exact uncapped correction",
    )
    collect_parser.set_defaults(func=collect)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.set_defaults(func=merge)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    train_parser.add_argument("--rollouts", type=Path, required=True)
    train_parser.add_argument("--initial-adapter", type=Path)
    train_parser.add_argument(
        "--arm",
        choices=("k1", "baseline", "cgcs", "all_k_uniform", "all_k_selector"),
        required=True,
    )
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--rollout-offset", type=int, default=0)
    train_parser.add_argument("--max-examples", type=int, default=0)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=2e-6)
    train_parser.add_argument("--gradient-accumulation", type=int, default=8)
    train_parser.add_argument("--lora-rank", type=int, default=8)
    train_parser.add_argument("--lora-alpha", type=int, default=16)
    train_parser.add_argument("--ppo-clip", type=float, default=0.2)
    train_parser.add_argument("--kl-coefficient", type=float, default=0.02)
    train_parser.add_argument("--value-coefficient", type=float, default=0.5)
    train_parser.add_argument(
        "--selection-correction-strength",
        type=float,
        default=1.0,
        help=(
            "lambda in w_lambda=(1-lambda)+lambda*((1/K)/q); "
            "0 disables selector correction and 1 is exact"
        ),
    )
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--log-every", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=181001)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("evaluate", parents=[common])
    eval_parser.add_argument("--adapter", type=Path)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
