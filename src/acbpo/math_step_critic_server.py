from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .math_prm import MathCandidate, MathDecision, terminal_return_critic_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one frozen math step critic")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--head-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def middle_truncate(ids: list[int], max_length: int, head_tokens: int) -> list[int]:
    if len(ids) <= max_length:
        return ids
    return ids[:head_tokens] + ids[-(max_length - head_tokens) :]


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    label_ids = [
        tokenizer(label, add_special_tokens=False).input_ids[0]
        for label in ("A", "B", "C")
    ]
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
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.config.use_cache = False
    model.eval()
    label_tensor = torch.tensor(label_ids, device="cuda")
    lock = threading.Lock()

    def score(payload: dict[str, Any]) -> dict[str, Any]:
        candidates = tuple(
            MathCandidate(str(text), 0, index)
            for index, text in enumerate(payload["candidates"])
        )
        decision = MathDecision(
            example_id="online",
            problem_id="online",
            problem=str(payload["problem"]),
            ground_truth_answer="",
            prefix=tuple(str(x) for x in payload.get("prefix", [])),
            candidates=candidates,
            chosen_candidate=None,
            record_index=0,
            step_index=len(payload.get("prefix", [])),
        )
        prompts = [
            tokenizer.apply_chat_template(
                terminal_return_critic_messages(decision, candidate),
                tokenize=False,
                add_generation_prompt=True,
            )
            for candidate in candidates
        ]
        encoded = [
            {
                "input_ids": middle_truncate(
                    tokenizer(prompt, add_special_tokens=False).input_ids,
                    args.max_length,
                    args.head_tokens,
                )
            }
            for prompt in prompts
        ]
        probabilities: list[list[float]] = []
        with lock, torch.no_grad():
            for start in range(0, len(encoded), args.batch_size):
                batch = tokenizer.pad(
                    encoded[start : start + args.batch_size],
                    padding=True,
                    return_tensors="pt",
                )
                batch = {key: value.to("cuda") for key, value in batch.items()}
                logits = model(**batch, logits_to_keep=1).logits[:, -1, label_tensor].float()
                probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return {
            "probabilities": probabilities,
            "scores": [row[2] - row[0] for row in probabilities],
        }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            return

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            self.send_json(200 if self.path == "/health" else 404, {"status": "ok"})

        def do_POST(self) -> None:
            if self.path != "/score":
                self.send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self.send_json(200, score(payload))
            except Exception as exc:  # pragma: no cover - surfaced to runner
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
