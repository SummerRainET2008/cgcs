from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .math_prm import MathCandidate, MathDecision, terminal_return_critic_messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve one full-precision actor and three frozen step critics"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--actor-adapter")
    parser.add_argument("--adapters", nargs=3, required=True)
    parser.add_argument("--adapter-names", nargs=3, default=("95001", "95002", "95003"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--head-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser


def middle_truncate(ids: list[int], max_length: int, head_tokens: int) -> list[int]:
    if len(ids) <= max_length:
        return ids
    return ids[:head_tokens] + ids[-(max_length - head_tokens) :]


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    label_ids = [
        tokenizer(label, add_special_tokens=False).input_ids[0]
        for label in ("A", "B", "C")
    ]
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": 0},
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    first_path = args.actor_adapter or args.adapters[0]
    first_name = "actor" if args.actor_adapter else args.adapter_names[0]
    model = PeftModel.from_pretrained(
        base, first_path, adapter_name=first_name, is_trainable=False
    )
    critic_pairs = zip(args.adapter_names, args.adapters) if args.actor_adapter else zip(
        args.adapter_names[1:], args.adapters[1:]
    )
    for name, path in critic_pairs:
        model.load_adapter(path, adapter_name=name, is_trainable=False)
    model.eval()
    label_tensor = torch.tensor(label_ids, device="cuda")
    lock = threading.Lock()

    def complete(payload: dict[str, Any]) -> dict[str, Any]:
        prompt = str(payload["prompt"])
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with lock, torch.inference_mode():
            # Generation is serialized by the server lock, so setting the
            # process RNG here gives deterministic per-request sampling.
            torch.manual_seed(int(payload["seed"]))
            if args.actor_adapter is None:
                with model.disable_adapter():
                    output = model.generate(
                        **encoded,
                        max_new_tokens=int(payload.get("n_predict", 80)),
                        do_sample=True,
                        temperature=float(payload.get("temperature", 0.8)),
                        top_p=float(payload.get("top_p", 0.95)),
                        top_k=int(payload.get("top_k", 40)),
                        repetition_penalty=float(payload.get("repeat_penalty", 1.05)),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                model.set_adapter("actor")
                output = model.generate(
                    **encoded,
                    max_new_tokens=int(payload.get("n_predict", 80)),
                    do_sample=True,
                    temperature=float(payload.get("temperature", 0.8)),
                    top_p=float(payload.get("top_p", 0.95)),
                    top_k=int(payload.get("top_k", 40)),
                    repetition_penalty=float(payload.get("repeat_penalty", 1.05)),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        new_tokens = output[0, encoded["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"content": text.splitlines()[0].strip() if text.strip() else ""}

    def score_all(payload: dict[str, Any]) -> dict[str, Any]:
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
        all_scores: list[list[float]] = []
        all_probabilities: list[list[list[float]]] = []
        with lock, torch.inference_mode():
            for adapter_name in args.adapter_names:
                model.set_adapter(adapter_name)
                probabilities: list[list[float]] = []
                for start in range(0, len(encoded), args.batch_size):
                    batch = tokenizer.pad(
                        encoded[start : start + args.batch_size],
                        padding=True,
                        return_tensors="pt",
                    )
                    batch = {key: value.to("cuda") for key, value in batch.items()}
                    logits = model(**batch, logits_to_keep=1).logits[:, -1, label_tensor].float()
                    probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
                all_probabilities.append(probabilities)
                all_scores.append([row[2] - row[0] for row in probabilities])
        return {
            "adapter_names": list(args.adapter_names),
            "probabilities": all_probabilities,
            "scores": all_scores,
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
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(
                200,
                {
                    "status": "ok",
                    "model": args.model,
                    "dtype": args.dtype,
                    "actor_adapter": args.actor_adapter or "base",
                    "adapter_names": list(args.adapter_names),
                },
            )

        def do_POST(self) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                if self.path == "/completion":
                    result = complete(payload)
                elif self.path == "/score_all":
                    result = score_all(payload)
                else:
                    self.send_json(404, {"error": "not found"})
                    return
                self.send_json(200, result)
            except Exception as exc:  # pragma: no cover - surfaced to runner
                print(f"{self.path}: {type(exc).__name__}: {exc}", flush=True)
                self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
