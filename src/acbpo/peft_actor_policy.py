from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from .actor_lora_experiment import middle_truncate
from .qwen_tool_policy import ChoiceDistribution, GeneratedArguments, extract_json_object


def normalize_generated_arguments(
    value: dict[str, Any], action_name: str
) -> dict[str, Any]:
    """Accept Qwen's occasional tool-call envelope around the requested arguments."""
    nested = value.get("arguments")
    if (
        isinstance(nested, dict)
        and set(value).issubset({"name", "arguments"})
        and value.get("name", action_name) == action_name
    ):
        return nested
    return value


class PeftActorOnlyPolicy:
    """NF4 Qwen policy with LoRA enabled only for action-name scoring."""

    def __init__(
        self,
        model_name: str,
        adapter: Path,
        *,
        use_adapter_for_choices: bool,
        max_context_tokens: int = 2048,
        head_tokens: int = 384,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.use_adapter_for_choices = use_adapter_for_choices
        self.max_context_tokens = max_context_tokens
        self.head_tokens = head_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()

    def _encode_prompt(self, prompt: str):
        token_ids = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids
        token_ids = middle_truncate(
            token_ids, self.max_context_tokens, self.head_tokens
        )
        return self.torch.tensor(
            [token_ids], device=self.model.device, dtype=self.torch.long
        )

    def _prompt(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def choice_distribution(
        self,
        messages: list[dict[str, str]],
        choices: list[str],
        *,
        temperature: float = 1.0,
        with_features: bool = False,
    ) -> ChoiceDistribution:
        if with_features:
            raise RuntimeError("actor-only PEFT policy does not expose critic features")
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        choice_lines = "\n".join(
            f"{label}: {choice}" for label, choice in zip(labels, choices)
        )
        prompt = self._prompt(
            messages
            + [
                {
                    "role": "user",
                    "content": (
                        "Choose the single best next action name from this list. "
                        "Return only its letter.\n" + choice_lines + "\nAnswer:"
                    ),
                }
            ]
        )
        input_ids = self._encode_prompt(prompt)
        label_ids = [
            self.tokenizer(label, add_special_tokens=False).input_ids
            for label in labels
        ]
        if any(len(ids) != 1 for ids in label_ids):
            raise RuntimeError("action labels are not single tokens")
        context = (
            nullcontext()
            if self.use_adapter_for_choices
            else self.model.disable_adapter()
        )
        with context, self.torch.inference_mode():
            logits = self.model(
                input_ids=input_ids, attention_mask=self.torch.ones_like(input_ids)
            ).logits[0, -1].float()
        indices = self.torch.tensor(
            [ids[0] for ids in label_ids], device=self.model.device
        )
        probabilities = self.torch.softmax(logits[indices] / temperature, dim=0)
        result = probabilities.cpu().numpy().astype(np.float64)
        result /= result.sum()
        return ChoiceDistribution(result, np.empty(0, dtype=np.float32))

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
        input_ids = self._encode_prompt(prompt)
        with self.model.disable_adapter(), self.torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        raw = self.tokenizer.decode(
            output[0, input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S)
        try:
            parsed = extract_json_object(raw)
            return GeneratedArguments(
                normalize_generated_arguments(parsed, action_name), raw, True
            )
        except ValueError:
            fallback = {"content": raw} if action_name == "respond" else {}
            return GeneratedArguments(fallback, raw, False)
