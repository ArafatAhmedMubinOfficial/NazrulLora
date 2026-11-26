#!/usr/bin/env python3
"""Interactive and single-prompt inference CLI for Nazrul-1B."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import peft.import_utils as _peft_iu

    _orig_is_torchao = _peft_iu.is_torchao_available

    def _safe_is_torchao_available():
        try:
            return _orig_is_torchao()
        except ImportError:
            return False

    _peft_iu.is_torchao_available = _safe_is_torchao_available
except Exception:
    pass

from peft import PeftModel

BASE_MODEL_ID = "hishab/titulm-llama-3.2-1b-v2.0"

SYSTEM_PROMPT = (
    "আপনি জাতীয় কবি কাজী নজরুল ইসলামের সাহিত্যিক ভাবধারা, ছন্দ, ওজস্বী ভাষা ও সৃজনশীল শৈলীতে "
    "দক্ষ একজন বাংলা এআই লেখক। ব্যবহারকারীর প্রম্পট অনুযায়ী কাজী নজরুলের ভঙ্গিতে কবিতা, "
    "গান (নজরুলগীতি), গল্প, প্রবন্ধ, নাটক বা গদ্য রচনা করুন এবং সাধারণ বাংলা প্রশ্নেরও নির্ভুল উত্তর দিন।"
)

NAZRUL_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
    "{{ '<|begin_of_text|>' }}"
    "{% if system_message is defined %}{{ '<|start_header_id|>system<|end_header_id|>\n\n' + system_message + '<|eot_id|>' }}{% endif %}"
    "{% for message in loop_messages %}"
    "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def load_nazrul_model(model_path: Path, base_model_id: str = BASE_MODEL_ID):
    """Load either a merged HuggingFace model directory or base model + LoRA adapter."""
    tok = AutoTokenizer.from_pretrained(model_path)
    if not tok.chat_template or "<|start_header_id|>" not in tok.chat_template:
        tok.chat_template = NAZRUL_CHAT_TEMPLATE
    if "<|finetune_right_pad_id|>" in tok.get_vocab():
        tok.pad_token = "<|finetune_right_pad_id|>"
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    if (model_path / "adapter_config.json").exists():
        base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=dtype, device_map=device_map)
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 240,
    temperature: float = 0.75,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt").to(model.device)
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tok.eos_token_id] if tok.eos_token_id == eot_id else [tok.eos_token_id, eot_id]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(1e-4, temperature),
        top_p=top_p,
        top_k=50,
        repetition_penalty=repetition_penalty,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chat with Nazrul-1B")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/nazrul-lora"))
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run and exit")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.75)
    args = parser.parse_args()

    model, tok = load_nazrul_model(args.model_path)
    if args.prompt:
        print(generate(model, tok, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature))
        return

    print("Nazrul-1B interactive mode (type 'exit' to quit):")
    while True:
        try:
            q = input("\nআপনি> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        reply = generate(model, tok, q, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        print(f"\nনজরুল-১বি>\n{reply}")


if __name__ == "__main__":
    main()
