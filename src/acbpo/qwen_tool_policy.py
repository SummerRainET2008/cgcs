from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from model text."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"no JSON object found in: {text[:200]!r}")


def selector_probabilities(
    candidates: np.ndarray,
    q_values: np.ndarray,
    *,
    beta: float,
    epsilon: float,
) -> np.ndarray:
    del candidates  # Candidate identity matters only when marginalizing below.
    shifted = beta * (q_values - np.max(q_values))
    softmax = np.exp(shifted)
    softmax /= softmax.sum()
    return (1.0 - epsilon) * softmax + epsilon / len(q_values)


def mc_selected_action_probability(
    action: int,
    actor_probabilities: np.ndarray,
    action_scores: np.ndarray,
    *,
    candidate_count: int,
    phantom_sets: int,
    beta: float,
    epsilon: float,
    rng: np.random.Generator,
) -> float:
    estimates = np.empty(phantom_sets, dtype=np.float64)
    action_ids = np.arange(actor_probabilities.size)
    for phantom in range(phantom_sets):
        candidates = rng.choice(
            action_ids, size=candidate_count, p=actor_probabilities
        )
        selector = selector_probabilities(
            candidates,
            action_scores[candidates],
            beta=beta,
            epsilon=epsilon,
        )
        estimates[phantom] = selector[candidates == action].sum()
    return float(estimates.mean())


@lru_cache(maxsize=8)
def _candidate_tuples(action_count: int, candidate_count: int) -> np.ndarray:
    return np.indices(
        (action_count,) * candidate_count, dtype=np.int16
    ).reshape(candidate_count, -1).T


def exact_selected_action_probability(
    action: int,
    actor_probabilities: np.ndarray,
    action_scores: np.ndarray,
    *,
    candidate_count: int,
    beta: float,
    epsilon: float,
) -> float:
    """Exactly marginalize the selector over every finite candidate tuple."""
    tuples = _candidate_tuples(actor_probabilities.size, candidate_count)
    tuple_probability = actor_probabilities[tuples].prod(axis=1)
    logits = beta * action_scores[tuples]
    logits -= logits.max(axis=1, keepdims=True)
    selector = np.exp(logits)
    selector /= selector.sum(axis=1, keepdims=True)
    selector = (1.0 - epsilon) * selector + epsilon / candidate_count
    selected_probability = (selector * (tuples == action)).sum(axis=1)
    return float(np.dot(tuple_probability, selected_probability))


@dataclass(frozen=True)
class GeneratedArguments:
    arguments: dict[str, Any]
    raw_text: str
    valid_json: bool


@dataclass(frozen=True)
class ChoiceDistribution:
    probabilities: np.ndarray
    state_features: np.ndarray


class QwenToolPolicy:
    def __init__(self, model_name: str, device: str = "cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map=device,
        )
        self.model.eval()

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def choice_probabilities(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        temperature: float = 1.0,
    ) -> np.ndarray:
        return self.choice_distribution(
            messages, choices, temperature=temperature
        ).probabilities

    def choice_distribution(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        temperature: float = 1.0,
    ) -> ChoiceDistribution:
        torch = self.torch
        if temperature <= 0.0:
            raise ValueError("choice temperature must be positive")
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        choice_lines = "\n".join(
            f"{label}: {choice}" for label, choice in zip(labels, choices)
        )
        scoring_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Choose the single best next action name from this list. "
                    "Return only its letter.\n" + choice_lines + "\nAnswer:"
                ),
            }
        ]
        prompt = self._prompt(scoring_messages)
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        label_ids = [
            self.tokenizer(label, add_special_tokens=False).input_ids for label in labels
        ]
        if any(len(ids) != 1 for ids in label_ids):
            raise RuntimeError("Qwen action labels are not single tokens")
        with torch.inference_mode():
            state_features = self.model.model(**encoded).last_hidden_state[0, -1]
            next_token_logits = self.model.lm_head(state_features).float()
        choice_logits = next_token_logits[
            torch.asarray([ids[0] for ids in label_ids], device=self.model.device)
        ]
        probabilities = torch.softmax(choice_logits / temperature, dim=0)
        result = probabilities.detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(result)) or result.sum() <= 0.0:
            raise RuntimeError("non-finite action probabilities from Qwen")
        result /= result.sum()
        return ChoiceDistribution(
            probabilities=result,
            state_features=state_features.detach().float().cpu().numpy(),
        )

    def generate_arguments(
        self,
        messages: list[dict[str, str]],
        action_name: str,
        schema: dict[str, Any],
        *,
        max_new_tokens: int = 256,
    ) -> GeneratedArguments:
        request = (
            f"The next action is {action_name}. Return only one JSON object "
            "containing its arguments. Do not include Markdown.\n"
            f"Argument schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        prompt = self._prompt(messages + [{"role": "user", "content": request}])
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        raw = self.tokenizer.decode(
            output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        try:
            arguments = extract_json_object(raw)
            return GeneratedArguments(arguments, raw, True)
        except ValueError:
            fallback = {"content": raw} if action_name == "respond" else {}
            return GeneratedArguments(fallback, raw, False)

    def action_values(
        self,
        messages: list[dict[str, str]],
        action_names: list[str],
    ) -> dict[str, float]:
        """Use Qwen as a zero-shot critic over a small candidate action set."""
        torch = self.torch
        label_ids = {
            label: self.tokenizer(label, add_special_tokens=False).input_ids
            for label in ("A", "B")
        }
        if any(len(ids) != 1 for ids in label_ids.values()):
            raise RuntimeError("Qwen critic labels are not single tokens")
        values: dict[str, float] = {}
        for action_name in dict.fromkeys(action_names):
            request = (
                "Judge the proposed next action in the conversation. It is good only if "
                "it advances the customer's task now, follows policy, and is not an "
                "unnecessary repeated call.\n"
                f"Proposed action: {action_name}\n"
                "A: bad next action\nB: good next action\nAnswer:"
            )
            prompt = self._prompt(messages + [{"role": "user", "content": request}])
            encoded = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False
            ).to(self.model.device)
            with torch.inference_mode():
                logits = self.model(**encoded).logits[0, -1].float()
            pair = torch.stack(
                [logits[label_ids["A"][0]], logits[label_ids["B"][0]]]
            )
            values[action_name] = float(torch.softmax(pair, dim=0)[1].item())
        return values
