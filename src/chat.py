#!/usr/bin/env python3
"""Interactive and single-prompt inference CLI for Nazrul-4B (Gemma-3 QLoRA)."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from src.peft_compat import patch_peft_torchao
except ImportError:
    from peft_compat import patch_peft_torchao

patch_peft_torchao()
from peft import PeftModel

BASE_MODEL_ID = "unsloth/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "আপনি জাতীয় কবি কাজী নজরুল ইসলামের সাহিত্যিক ভাবধারা, ছন্দ, অন্ত্যমিল, ওজস্বী ভাষা ও সৃজনশীল "
    "শৈলীতে দক্ষ একজন বাংলা এআই লেখক। ব্যবহারকারীর যেকোনো প্রম্পট (আধুনিক বা ধ্রুপদী) বুঝে কাজী নজরুলের "
    "ভঙ্গিতে ছন্দোবদ্ধ ও অন্ত্যমিলযুক্ত কবিতা, নজরুলগীতি, গল্প বা প্রবন্ধ রচনা করুন এবং সাধারণ বাংলা প্রশ্নেরও নির্ভুল উত্তর দিন।"
)


def score_bangla_rhyme(text: str) -> float:
    """Measure fraction of verse lines sharing end-rhyme syllables."""
    lines: List[str] = [ln.strip() for ln in text.splitlines() if len(re.sub(r"[^ঀ-৿]", "", ln)) >= 4]
    if len(lines) < 4:
        return 0.0
    keys = [re.sub(r"[^ঀ-৿]", "", ln)[-2:] for ln in lines]
    matched = sum(
        1
        for i, k in enumerate(keys)
        if k and k in [keys[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(keys) and j != i]
    )
    return matched / len(keys)


def load_nazrul_model(model_path: Path, base_model_id: str = BASE_MODEL_ID, load_in_4bit: bool = True):
    """Load base Gemma-3-4B (4-bit on GPU, fp32 on CPU) with the Nazrul LoRA adapter."""
    tok = AutoTokenizer.from_pretrained(model_path if model_path.exists() else base_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available() and load_in_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb, device_map="auto")
    elif torch.cuda.is_available():
        base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float16, device_map="auto")
    else:
        base = AutoModelForCausalLM.from_pretrained(base_model_id, dtype=torch.float32)

    if (model_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        model = base
    model.eval()
    return model, tok


@torch.no_grad()
def generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 260,
    temperature: float = 0.72,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    rhyme_candidates: int = 1,
) -> str:
    msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{prompt}"}]
    formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(formatted, return_tensors="pt").to(model.device)
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [i for i in {tok.eos_token_id, eot_id} if isinstance(i, int) and i >= 0]

    best_text = ""
    best_score = -1.0
    for _ in range(max(1, rhyme_candidates)):
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
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        if rhyme_candidates <= 1:
            return text
        sc = score_bangla_rhyme(text)
        if sc > best_score:
            best_score = sc
            best_text = text
    return best_text


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Chat with Nazrul-4B")
    parser.add_argument("--model-path", type=Path, default=Path("outputs/nazrul-lora"))
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to run and exit")
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--temperature", type=float, default=0.72)
    parser.add_argument("--rhyme-candidates", type=int, default=1, help="Sample N outputs and pick highest Bengali rhyme")
    args = parser.parse_args()

    model, tok = load_nazrul_model(args.model_path)
    if args.prompt:
        print(
            generate(
                model,
                tok,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                rhyme_candidates=args.rhyme_candidates,
            )
        )
        return

    print("Nazrul-4B interactive mode (type 'exit' to quit):")
    while True:
        try:
            q = input("\nআপনি> ").strip()
        except EOFError:
            break
        if not q or q.lower() in {"exit", "quit"}:
            break
        reply = generate(
            model,
            tok,
            q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            rhyme_candidates=args.rhyme_candidates,
        )
        print(f"\nনজরুল-৪বি>\n{reply}")


if __name__ == "__main__":
    main()
