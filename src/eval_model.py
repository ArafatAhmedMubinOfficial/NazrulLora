#!/usr/bin/env python3
"""Evaluate Nazrul-1B against the base model on held-out Nazrul perplexity and diverse prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

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

TEST_PROMPTS = [
    ("revolutionary_poem", "কাজী নজরুল ইসলামের ভঙ্গিতে অন্যায় ও পরাধীনতার বিরুদ্ধে জাগরণ ও বিদ্রোহের উপর একটি কবিতা লেখো।"),
    ("nazrul_geeti", "কাজী নজরুলের ভঙ্গিতে বর্ষা, মেঘ ও বিরহ নিয়ে একটি নজরুলগীতি রচনা করো।"),
    ("prose_essay", "কাজী নজরুল ইসলামের ওজস্বী গদ্যশৈলীতে তরুণ সমাজ ও তারুণ্যের শক্তি নিয়ে একটি অনুচ্ছেদ লেখো।"),
    ("short_story", "নজরুলের গল্প বলার ভঙ্গিতে গ্রামের সন্ধ্যা ও এক বাউণ্ডুলে তরুণের গল্প শুরু করো।"),
    ("verse_continuation", "কাজী নজরুলের শৈলী অনুসরণ করে নিচের লেখাটি এগিয়ে নিয়ে যাও:\n\nগাহি সাম্যের গান—\nযেখানে আসিয়া এক হয়ে গেছে সব বাধা-ব্যবধান,"),
    ("english_prompt", "Write a passionate Bangla poem in the style of Kazi Nazrul Islam about freedom and youth."),
    ("general_bangla", "বাংলাদেশের জাতীয় কবি কে এবং তাঁর সাহিত্যকর্মের মূল সুর কী? সংক্ষেপে বলো।"),
]


@torch.no_grad()
def compute_val_ppl(model: torch.nn.Module, tok: AutoTokenizer, val_jsonl: Path, seq_len: int = 512, max_blocks: int = 60) -> float:
    sep_id = tok.convert_tokens_to_ids("<|end_of_text|>")
    stream: List[int] = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            ids = tok.encode(json.loads(line)["text"], add_special_tokens=False)
            stream.extend(ids + [sep_id])
    n_blocks = min(max_blocks, len(stream) // seq_len)
    total_loss = 0.0
    for i in range(n_blocks):
        x = torch.tensor(stream[i * seq_len : (i + 1) * seq_len], dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.float16) if model.device.type == "cuda" else torch.no_grad():
            loss = model(input_ids=x, labels=x).loss
        total_loss += float(loss.item())
    return math.exp(min(20.0, total_loss / max(1, n_blocks)))


@torch.no_grad()
def generate_reply(model: torch.nn.Module, tok: AutoTokenizer, prompt: str, use_chat_template: bool = True, max_new_tokens: int = 180) -> str:
    if use_chat_template:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt").to(model.device)
    eot_id = tok.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tok.eos_token_id] if tok.eos_token_id == eot_id else [tok.eos_token_id, eot_id]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.75,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.15,
        eos_token_id=eos_ids,
        pad_token_id=tok.pad_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1] :]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Nazrul-1B model")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/nazrul-lora"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data/processed/cpt_val.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("outputs/eval_report.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter_dir)
    if "<|finetune_right_pad_id|>" in tok.get_vocab():
        tok.pad_token = "<|finetune_right_pad_id|>"
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16, device_map="auto")
    base_model.eval()
    base_ppl = compute_val_ppl(base_model, tok, args.val_jsonl)
    print(f"[eval] Base TituLM-1B held-out Nazrul PPL: {base_ppl:.2f}", flush=True)

    ft_model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    ft_model.eval()
    ft_ppl = compute_val_ppl(ft_model, tok, args.val_jsonl)
    print(f"[eval] Fine-tuned Nazrul-1B held-out Nazrul PPL: {ft_ppl:.2f}", flush=True)

    samples = []
    for name, prompt in TEST_PROMPTS:
        reply = generate_reply(ft_model, tok, prompt, use_chat_template=True, max_new_tokens=200)
        print(f"\n--- [{name}] ---\nPROMPT: {prompt}\nNAZRUL-1B:\n{reply}", flush=True)
        samples.append({"name": name, "prompt": prompt, "response": reply})

    report = {
        "base_model": BASE_MODEL_ID,
        "adapter_dir": str(args.adapter_dir),
        "heldout_nazrul_ppl_base": round(base_ppl, 2),
        "heldout_nazrul_ppl_finetuned": round(ft_ppl, 2),
        "samples": samples,
    }
    args.out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval] Saved report to {args.out_file}", flush=True)


if __name__ == "__main__":
    main()
