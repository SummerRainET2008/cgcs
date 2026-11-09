from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .qwen_tool_policy import ChoiceDistribution, GeneratedArguments, extract_json_object


def qwen_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Render the Qwen3 chat template used by the GGUF model."""
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported chat role: {role}")
        parts.append(
            f"<|im_start|>{role}\n{message['content']}<|im_end|>\n"
        )
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _choice_grammar(labels: list[str]) -> str:
    alternatives = " | ".join(json.dumps(label) for label in labels)
    return f"root ::= {alternatives}"


def probabilities_from_completion(
    response: dict[str, Any], labels: list[str], *, allow_missing: bool = False
) -> np.ndarray:
    """Extract the grammar-conditioned next-token distribution."""
    try:
        completion = response["completion_probabilities"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("llama.cpp response omitted token probabilities") from exc
    top = completion.get("top_probs", completion.get("top_logprobs"))
    if top is None:
        raise RuntimeError("llama.cpp response omitted ranked token probabilities")
    probability_by_label: dict[str, float] = {}
    for item in top:
        token = str(item.get("token", "")).strip()
        if token in labels:
            if "prob" in item:
                probability_by_label[token] = probability_by_label.get(
                    token, 0.0
                ) + float(item["prob"])
            else:
                probability_by_label[token] = probability_by_label.get(
                    token, 0.0
                ) + math.exp(float(item["logprob"]))
    missing = [label for label in labels if label not in probability_by_label]
    if missing and not allow_missing:
        raise RuntimeError(f"llama.cpp omitted constrained labels: {missing}")
    probabilities = np.asarray(
        [probability_by_label.get(label, 0.0) for label in labels], dtype=np.float64
    )
    total = probabilities.sum()
    if not math.isfinite(float(total)) or total <= 0.0:
        raise RuntimeError("non-finite action probabilities from llama.cpp")
    return probabilities / total


@dataclass(frozen=True)
class LlamaCppEndpoints:
    actor: str
    embedding: str | None = None


class LlamaCppToolPolicy:
    """Qwen tool policy served by llama.cpp HTTP endpoints."""

    def __init__(
        self,
        endpoints: LlamaCppEndpoints,
        *,
        timeout: float = 180.0,
        model_name: str = "Qwen3-4B-Instruct-2507-Q4_K_M",
    ):
        self.endpoints = endpoints
        self.timeout = timeout
        self.model_name = model_name

    def _post(self, base_url: str, path: str, payload: dict[str, Any]) -> Any:
        request = Request(
            base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"llama.cpp {path} returned HTTP {exc.code}: {body[:500]}"
                )
                if exc.code < 500:
                    raise last_error from exc
            except (URLError, TimeoutError) as exc:
                last_error = RuntimeError(f"llama.cpp {path} request failed: {exc}")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise last_error

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return qwen_chat_prompt(messages)

    def choice_probabilities(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        temperature: float = 1.0,
    ) -> np.ndarray:
        return self.choice_distribution(
            messages, choices, temperature=temperature, with_features=False
        ).probabilities

    def choice_distribution(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        temperature: float = 1.0,
        with_features: bool = True,
    ) -> ChoiceDistribution:
        if temperature <= 0.0:
            raise ValueError("choice temperature must be positive")
        if len(choices) > 26:
            raise ValueError("letter-scored policy supports at most 26 choices")
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        choice_lines = "\n".join(
            f"{label}: {choice}" for label, choice in zip(labels, choices)
        )
        scoring_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Choose the single best next action name from this list. "
                    "Return only its letter.\n"
                    + choice_lines
                    + "\nAnswer:"
                ),
            }
        ]
        prompt = self._prompt(scoring_messages)
        payload = {
            "prompt": prompt,
            "n_predict": 1,
            "temperature": temperature,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "typical_p": 1.0,
            "repeat_penalty": 1.0,
            "grammar": _choice_grammar(labels),
            # A grammar may admit multiple tokenizations of the same single
            # character. Ask for enough ranked entries to recover all labels.
            "n_probs": max(128, len(labels)),
            "post_sampling_probs": True,
            "cache_prompt": True,
        }
        response = self._post(self.endpoints.actor, "/completion", payload)
        try:
            probabilities = probabilities_from_completion(response, labels)
        except RuntimeError as exc:
            if "omitted constrained labels" not in str(exc):
                raise
            # Rarely llama.cpp omits a legal grammar token from its ranked
            # list even with n_probs=128. Retry with a wider list, then treat
            # any still-unreported label as negligible instead of aborting a
            # resumable multi-hour experiment.
            payload["n_probs"] = max(2048, len(labels))
            response = self._post(self.endpoints.actor, "/completion", payload)
            probabilities = probabilities_from_completion(
                response, labels, allow_missing=True
            )
        if with_features:
            if self.endpoints.embedding is None:
                raise RuntimeError("an embedding endpoint is required for critic features")
            embedding = self._post(
                self.endpoints.embedding,
                "/v1/embeddings",
                {"input": prompt, "model": self.model_name},
            )
            try:
                state_features = np.asarray(
                    embedding["data"][0]["embedding"], dtype=np.float32
                )
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("llama.cpp returned an invalid embedding") from exc
        else:
            state_features = np.empty(0, dtype=np.float32)
        return ChoiceDistribution(
            probabilities=probabilities,
            state_features=state_features,
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
            "containing its arguments. Do not include Markdown. "
            "Copy identifiers exactly from the customer request or tool results; "
            "never remove prefixes such as # and never invent an identifier.\n"
            f"Argument schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        prompt = self._prompt(messages + [{"role": "user", "content": request}])
        response = self._post(
            self.endpoints.actor,
            "/completion",
            {
                "prompt": prompt,
                "n_predict": max_new_tokens,
                "temperature": 0.0,
                "json_schema": schema,
                "cache_prompt": True,
            },
        )
        raw = str(response.get("content", "")).strip()
        try:
            arguments = extract_json_object(raw)
            return GeneratedArguments(arguments, raw, True)
        except ValueError:
            fallback = {"content": raw} if action_name == "respond" else {}
            return GeneratedArguments(fallback, raw, False)
