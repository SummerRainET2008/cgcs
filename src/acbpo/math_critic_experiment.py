from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .actor_lora_experiment import middle_truncate
from .math_prm import (
    MathCandidate,
    MathDecision,
    RATING_LABELS,
    critic_messages,
    terminal_return_critic_messages,
)


@dataclass(frozen=True)
class ScoredCandidate:
    example_id: str
    candidate_index: int
    rating: int
    probabilities: tuple[float, float, float]

    @property
    def score(self) -> float:
        return sum(rating * probability for rating, probability in zip((-1, 0, 1), self.probabilities))


def multiplicity_aware_listwise_loss(
    scores,
    target_utility,
    multiplicities,
    *,
    selection_temperature: float,
    target_temperature: float,
):
    """Listwise cross entropy over sampled positions, encoding duplicates once.

    Each unique candidate has one score but represents ``multiplicity`` sampled
    positions. The returned selection log probabilities are per-position; the
    target masses are aggregated over identical positions.
    """
    import torch

    log_multiplicity = torch.log(multiplicities)
    selection_logits = scores / selection_temperature
    selection_log_norm = torch.logsumexp(
        selection_logits + log_multiplicity, dim=0
    )
    selection_log_probs = selection_logits - selection_log_norm
    target_logits = target_utility / target_temperature
    target_log_norm = torch.logsumexp(target_logits + log_multiplicity, dim=0)
    target_mass = torch.exp(target_logits + log_multiplicity - target_log_norm)
    loss = -(target_mass * selection_log_probs).sum()
    return loss, selection_log_probs, target_mass


def load_decisions(path: Path) -> list[MathDecision]:
    decisions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["prefix"] = tuple(item["prefix"])
            item["candidates"] = tuple(MathCandidate(**x) for x in item["candidates"])
            decisions.append(MathDecision(**item))
    return decisions


def is_mixed_decision(decision: MathDecision) -> bool:
    """Return whether a state contains more than one observed outcome."""
    return len({candidate.rating for candidate in decision.candidates}) > 1


def select_balanced_training_items(
    decisions: list[MathDecision], *, per_rating: int, seed: int
) -> list[tuple[MathDecision, int]]:
    by_rating: dict[int, list[tuple[MathDecision, int]]] = {-1: [], 0: [], 1: []}
    for decision in decisions:
        for index, candidate in enumerate(decision.candidates):
            by_rating[candidate.rating].append((decision, index))
    rng = random.Random(seed)
    selected = []
    for rating in (-1, 0, 1):
        rng.shuffle(by_rating[rating])
        selected.extend(by_rating[rating][:per_rating] if per_rating > 0 else by_rating[rating])
    rng.shuffle(selected)
    return selected


def select_pairwise_training_items(
    decisions: list[MathDecision], *, max_pairs: int, seed: int
) -> list[tuple[MathDecision, int, int]]:
    """Sample higher/lower labeled candidates, with at most one pair per state."""
    rng = random.Random(seed)
    shuffled = list(decisions)
    rng.shuffle(shuffled)
    pairs = []
    for decision in shuffled:
        valid = [
            (high, low)
            for high in range(len(decision.candidates))
            for low in range(len(decision.candidates))
            if decision.candidates[high].rating > decision.candidates[low].rating
        ]
        if valid:
            high, low = rng.choice(valid)
            pairs.append((decision, high, low))
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    return pairs


def ranking_metrics(
    decisions: list[MathDecision], scored: list[ScoredCandidate]
) -> dict[str, Any]:
    lookup = {(x.example_id, x.candidate_index): x for x in scored}
    confusion = np.zeros((3, 3), dtype=np.int64)
    top1 = 0
    selected_rating = 0.0
    pair_correct = 0.0
    pair_total = 0
    covered = 0
    for decision in decisions:
        group = [lookup.get((decision.example_id, index)) for index in range(len(decision.candidates))]
        if any(item is None for item in group):
            continue
        typed_group = [item for item in group if item is not None]
        covered += 1
        scores = [item.score for item in typed_group]
        selected = max(range(len(scores)), key=scores.__getitem__)
        best_rating = max(candidate.rating for candidate in decision.candidates)
        selected_rating += decision.candidates[selected].rating
        top1 += int(decision.candidates[selected].rating == best_rating)
        for item in typed_group:
            pred = int(np.argmax(item.probabilities))
            confusion[item.rating + 1, pred] += 1
        for left in range(len(typed_group)):
            for right in range(left + 1, len(typed_group)):
                left_rating = typed_group[left].rating
                right_rating = typed_group[right].rating
                if left_rating == right_rating:
                    continue
                pair_total += 1
                delta = scores[left] - scores[right]
                truth = left_rating - right_rating
                pair_correct += 1.0 if delta * truth > 0 else 0.5 if delta == 0 else 0.0

    total = int(confusion.sum())
    accuracy = float(np.trace(confusion) / max(total, 1))
    f1s = []
    for label in range(3):
        tp = confusion[label, label]
        fp = confusion[:, label].sum() - tp
        fn = confusion[label, :].sum() - tp
        f1s.append(float(2 * tp / max(2 * tp + fp + fn, 1)))
    return {
        "decisions": covered,
        "candidates": total,
        "classification_accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "pairwise_accuracy": float(pair_correct / max(pair_total, 1)),
        "pairwise_comparisons": pair_total,
        "top1_accuracy": float(top1 / max(covered, 1)),
        "mean_selected_rating": float(selected_rating / max(covered, 1)),
        "confusion_rows_true_cols_pred": confusion.tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a PRM math step critic")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--objective",
        choices=(
            "pointwise",
            "pairwise",
            "listwise",
            "advantage",
            "calibrated_advantage",
        ),
        default="pointwise",
    )
    parser.add_argument("--train-per-rating", type=int, default=512)
    parser.add_argument("--train-pairs", type=int, default=512)
    parser.add_argument("--train-decisions", type=int, default=512)
    parser.add_argument("--test-offset", type=int, default=0)
    parser.add_argument("--pairwise-ce-weight", type=float, default=0.1)
    parser.add_argument("--listwise-ce-weight", type=float, default=0.05)
    parser.add_argument("--mixed-listwise-weight", type=float, default=1.0)
    parser.add_argument("--base-kl-weight", type=float, default=0.2)
    parser.add_argument("--target-temperature", type=float, default=0.25)
    parser.add_argument("--selection-temperature", type=float, default=0.25)
    parser.add_argument("--max-test-decisions", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--head-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--prompt-style",
        choices=("prm_rating", "terminal_return"),
        default="prm_rating",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.max_length <= args.head_tokens:
        raise ValueError("max-length must exceed head-tokens")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_decisions = load_decisions(args.train)
    test_decisions = load_decisions(args.test)
    if args.test_offset < 0 or args.test_offset > len(test_decisions):
        raise ValueError("test-offset is outside the test decision range")
    test_decisions = test_decisions[args.test_offset :]
    if args.max_test_decisions > 0:
        test_decisions = test_decisions[: args.max_test_decisions]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    label_tokens = [RATING_LABELS[rating] for rating in (-1, 0, 1)]
    label_ids = [tokenizer(x, add_special_tokens=False).input_ids for x in label_tokens]
    if any(len(ids) != 1 for ids in label_ids):
        raise RuntimeError(f"rating labels are not single tokens: {label_ids}")
    label_ids_tensor = torch.tensor([ids[0] for ids in label_ids], device="cuda")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=not args.eval_only)
    elif not args.eval_only:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            ),
        )
    model.config.use_cache = False

    def token_ids(decision: MathDecision, candidate_index: int) -> list[int]:
        message_builder = (
            terminal_return_critic_messages
            if args.prompt_style == "terminal_return"
            else critic_messages
        )
        prompt = tokenizer.apply_chat_template(
            message_builder(decision, decision.candidates[candidate_index]),
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        return middle_truncate(ids, args.max_length, args.head_tokens)

    def batch_tensors(items: list[tuple[MathDecision, int]]):
        encoded = [{"input_ids": token_ids(decision, index)} for decision, index in items]
        padded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
        return {key: value.to("cuda") for key, value in padded.items()}

    def evaluate() -> tuple[dict[str, Any], list[ScoredCandidate]]:
        items = [
            (decision, index)
            for decision in test_decisions
            for index in range(len(decision.candidates))
        ]
        scored: list[ScoredCandidate] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(items), args.eval_batch_size):
                batch = items[start : start + args.eval_batch_size]
                tensors = batch_tensors(batch)
                logits = model(**tensors, logits_to_keep=1).logits[
                    :, -1, label_ids_tensor
                ].float()
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                for (decision, index), probs in zip(batch, probabilities):
                    scored.append(
                        ScoredCandidate(
                            decision.example_id,
                            index,
                            decision.candidates[index].rating,
                            tuple(float(x) for x in probs),
                        )
                    )
        return ranking_metrics(test_decisions, scored), scored

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        if args.objective == "pointwise":
            training = select_balanced_training_items(
                train_decisions, per_rating=args.train_per_rating, seed=args.seed
            )
        elif args.objective == "pairwise":
            training = select_pairwise_training_items(
                train_decisions, max_pairs=args.train_pairs, seed=args.seed
            )
        else:
            training = list(train_decisions)
            random.Random(args.seed).shuffle(training)
            if args.train_decisions > 0:
                training = training[: args.train_decisions]
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
        )
        optimizer.zero_grad(set_to_none=True)
        updates = 0
        losses = []
        model.train()
        with (args.output_dir / "train_log.jsonl").open("w") as log_handle:
            for epoch in range(args.epochs):
                for start in range(0, len(training), args.batch_size):
                    batch = training[start : start + args.batch_size]
                    if args.objective == "pointwise":
                        candidate_batch = batch
                    elif args.objective == "pairwise":
                        candidate_batch = [
                            item
                            for decision, high, low in batch
                            for item in ((decision, high), (decision, low))
                        ]
                    else:
                        candidate_batch = []
                        listwise_spans = []
                        for decision in batch:
                            span_start = len(candidate_batch)
                            candidate_batch.extend(
                                (decision, index)
                                for index in range(len(decision.candidates))
                            )
                            listwise_spans.append(
                                (decision, span_start, len(candidate_batch))
                            )
                    tensors = batch_tensors(candidate_batch)
                    targets = torch.tensor(
                        [decision.candidates[index].rating + 1 for decision, index in candidate_batch],
                        device="cuda",
                    )
                    if args.objective in {
                        "listwise",
                        "advantage",
                        "calibrated_advantage",
                    } and args.base_kl_weight > 0:
                        with model.disable_adapter(), torch.no_grad():
                            base_logits = model(**tensors, logits_to_keep=1).logits[
                                :, -1, label_ids_tensor
                            ].float()
                    logits = model(**tensors, logits_to_keep=1).logits[
                        :, -1, label_ids_tensor
                    ].float()
                    if args.objective == "pointwise":
                        loss = F.cross_entropy(logits, targets)
                    elif args.objective == "pairwise":
                        ordinal = logits[:, 2] - logits[:, 0]
                        high_scores = ordinal[0::2]
                        low_scores = ordinal[1::2]
                        rank_loss = F.softplus(low_scores - high_scores).mean()
                        calibration_loss = F.cross_entropy(logits, targets)
                        loss = rank_loss + args.pairwise_ce_weight * calibration_loss
                    else:
                        label_values = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
                        state_losses = []
                        calibration_losses = []
                        mixed_guidance_losses = []
                        for decision, span_start, span_end in listwise_spans:
                            state_logits = logits[span_start:span_end]
                            scores = (
                                torch.softmax(state_logits, dim=-1) * label_values
                            ).sum(dim=-1)
                            ratings = torch.tensor(
                                [candidate.rating for candidate in decision.candidates],
                                device="cuda",
                                dtype=torch.float32,
                            )
                            multiplicities = torch.tensor(
                                [
                                    candidate.multiplicity
                                    for candidate in decision.candidates
                                ],
                                device="cuda",
                                dtype=torch.float32,
                            )
                            if args.objective in {"advantage", "calibrated_advantage"}:
                                target_mean = (
                                    ratings * multiplicities
                                ).sum() / multiplicities.sum()
                                score_mean = (
                                    scores * multiplicities
                                ).sum() / multiplicities.sum()
                                target_utility = torch.abs(ratings - target_mean)
                                selection_scores = torch.abs(scores - score_mean)
                            else:
                                target_utility = ratings
                                selection_scores = scores
                            listwise_loss, selection_log_probs, _ = (
                                multiplicity_aware_listwise_loss(
                                    selection_scores,
                                    target_utility,
                                    multiplicities,
                                    selection_temperature=args.selection_temperature,
                                    target_temperature=args.target_temperature,
                                )
                            )
                            calibration_items = F.cross_entropy(
                                state_logits,
                                targets[span_start:span_end],
                                reduction="none",
                            )
                            calibration_loss = (
                                calibration_items * multiplicities
                            ).sum() / multiplicities.sum()

                            if args.objective == "calibrated_advantage":
                                calibration_losses.append(calibration_loss)
                                if not is_mixed_decision(decision):
                                    continue
                                guidance_loss = listwise_loss
                                if args.base_kl_weight > 0:
                                    state_base_logits = base_logits[span_start:span_end]
                                    base_scores = (
                                        torch.softmax(state_base_logits, dim=-1)
                                        * label_values
                                    ).sum(dim=-1)
                                    base_mean = (
                                        base_scores * multiplicities
                                    ).sum() / multiplicities.sum()
                                    base_scores = torch.abs(base_scores - base_mean)
                                    base_logits_for_selection = (
                                        base_scores / args.selection_temperature
                                    )
                                    log_multiplicity = torch.log(multiplicities)
                                    base_log_probs = (
                                        base_logits_for_selection
                                        - torch.logsumexp(
                                            base_logits_for_selection + log_multiplicity,
                                            dim=0,
                                        )
                                    )
                                    base_mass = torch.exp(
                                        base_log_probs + log_multiplicity
                                    )
                                    anchor_kl = (
                                        base_mass
                                        * (base_log_probs - selection_log_probs)
                                    ).sum()
                                    guidance_loss = (
                                        guidance_loss + args.base_kl_weight * anchor_kl
                                    )
                                mixed_guidance_losses.append(guidance_loss)
                                continue

                            state_loss = (
                                listwise_loss
                                + args.listwise_ce_weight * calibration_loss
                            )
                            if args.base_kl_weight > 0:
                                state_base_logits = base_logits[span_start:span_end]
                                base_scores = (
                                    torch.softmax(state_base_logits, dim=-1)
                                    * label_values
                                ).sum(dim=-1)
                                if args.objective == "advantage":
                                    base_mean = (
                                        base_scores * multiplicities
                                    ).sum() / multiplicities.sum()
                                    base_scores = torch.abs(base_scores - base_mean)
                                base_logits_for_selection = (
                                    base_scores / args.selection_temperature
                                )
                                log_multiplicity = torch.log(multiplicities)
                                base_log_probs = (
                                    base_logits_for_selection
                                    - torch.logsumexp(
                                        base_logits_for_selection + log_multiplicity,
                                        dim=0,
                                    )
                                )
                                base_mass = torch.exp(
                                    base_log_probs + log_multiplicity
                                )
                                anchor_kl = (
                                    base_mass
                                    * (base_log_probs - selection_log_probs)
                                ).sum()
                                state_loss = (
                                    state_loss + args.base_kl_weight * anchor_kl
                                )
                            state_losses.append(state_loss)
                        if args.objective == "calibrated_advantage":
                            loss = torch.stack(calibration_losses).mean()
                            if mixed_guidance_losses:
                                loss = loss + args.mixed_listwise_weight * torch.stack(
                                    mixed_guidance_losses
                                ).mean()
                        else:
                            loss = torch.stack(state_losses).mean()
                    (loss / args.gradient_accumulation).backward()
                    losses.append(float(loss.detach().cpu()))
                    batch_number = start // args.batch_size + 1
                    final_batch = start + args.batch_size >= len(training)
                    if batch_number % args.gradient_accumulation == 0 or final_batch:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        updates += 1
                        if updates % args.log_every == 0 or final_batch:
                            row = {
                                "epoch": epoch,
                                "update": updates,
                                "mean_recent_loss": float(np.mean(losses[-args.log_every * args.gradient_accumulation :])),
                            }
                            print(json.dumps(row), flush=True)
                            log_handle.write(json.dumps(row) + "\n")
                            log_handle.flush()
        model.save_pretrained(args.output_dir / "adapter")

    metrics, scored = evaluate()
    metrics["model"] = args.model
    metrics["adapter"] = str(args.adapter) if args.adapter else ("trained" if not args.eval_only else None)
    metrics["seed"] = args.seed
    metrics["objective"] = None if args.eval_only else args.objective
    metrics["prompt_style"] = args.prompt_style
    metrics["train_examples"] = 0 if args.eval_only else len(training)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps({**item.__dict__, "score": item.score}) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
