#!/usr/bin/env python3
"""Evaluate Nazrul-4B against base Gemma-3-4B on held-out Nazrul perplexity, rhyme rate, and custom prompts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from src.peft_compat import patch_peft_torchao
    from src.chat import generate, score_bangla_rhyme
except ImportError:
    from peft_compat import patch_peft_torchao
    from chat import generate, score_bangla_rhyme

patch_peft_torchao()
from peft import PeftModel

BASE_MODEL_ID = "coder3101/gemma-4-E4B-it-heretic"

TEST_PROMPTS = [
    ("revolutionary_poem", "কাজী নজরুল ইসলামের ভঙ্গিতে ছন্দ ও অন্ত্যমিল বজায় রেখে অন্যায় ও পরাধীনতার বিরুদ্ধে জাগরণের উপর ৮-১০ লাইনের একটি কবিতা লেখো।"),
    ("custom_social_satire", "কাজী নজরুল ইসলামের ভঙ্গিতে ছন্দ ও অন্ত্যমিল বজায় রেখে আধুনিক সমাজে নারীর সম্মানের আদর্শ কীভাবে ভোগবাদ, নগ্নতা ও অশ্লীলতার নামে বিকৃত হয়ে গেছে—তা নিয়ে ৮-১০ লাইনের একটি প্রতিবাদী ও আক্ষেপভরা কবিতা লেখো।"),
    ("nazrul_geeti", "কাজী নজরুলের ভঙ্গিতে বর্ষা, মেঘ ও বিরহ নিয়ে একটি ছন্দোবদ্ধ নজরুলগীতি রচনা করো।"),
    ("craft_and_poem", "কাজী নজরুল ইসলামের লেখার কৌশল, ছন্দ ও শব্দচয়নের বৈশিষ্ট্য সংক্ষেপে ব্যাখ্যা করে তাঁর ভঙ্গিতে তারুণ্যের শক্তি নিয়ে একটি ছন্দোবদ্ধ কবিতা লেখো।"),
    ("prose_essay", "কাজী নজরুল ইসলামের ওজস্বী গদ্যশৈলীতে তরুণ সমাজ ও তারুণ্যের দায়িত্ব নিয়ে একটি অনুচ্ছেদ লেখো।"),
    ("verse_continuation", "কাজী নজরুলের ছন্দ ও অন্ত্যমিল অনুসরণ করে নিচের লেখাটি এগিয়ে নিয়ে যাও:\n\nগাহি সাম্যের গান—\nযেখানে আসিয়া এক হয়ে গেছে সব বাধা-ব্যবধান,"),
    ("english_prompt", "Write a rhymed Bangla poem in the style of Kazi Nazrul Islam about freedom, youth, and standing against hypocrisy."),
    ("general_bangla", "বাংলাদেশের জাতীয় কবি কে এবং তাঁর সাহিত্যকর্মের মূল সুর ও ছন্দ-বৈশিষ্ট্য কী? সংক্ষেপে বলো।"),
]


@torch.no_grad()
def compute_val_ppl(model: torch.nn.Module, tok: AutoTokenizer, val_jsonl: Path, seq_len: int = 512, max_blocks: int = 35) -> float:
    eos_cand = tok.convert_tokens_to_ids("<eos>")
    sep_id = eos_cand if isinstance(eos_cand, int) and eos_cand >= 0 else tok.eos_token_id
    stream: List[int] = []
    with val_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            ids = tok.encode(json.loads(line)["text"], add_special_tokens=False)
            stream.extend(ids + [sep_id])
    n_blocks = min(max_blocks, len(stream) // seq_len)
    total_loss = 0.0
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for i in range(n_blocks):
        x = torch.tensor(stream[i * seq_len : (i + 1) * seq_len], dtype=torch.long, device=model.device).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            loss = model(input_ids=x, labels=x).loss
        total_loss += float(loss.item())
    return math.exp(min(20.0, total_loss / max(1, n_blocks)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Nazrul-4B model")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/nazrul-lora"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data/processed/cpt_val.jsonl"))
    parser.add_argument("--out-file", type=Path, default=Path("outputs/eval_report.json"))
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.adapter_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, dtype=compute_dtype, device_map="auto")
    base_model.eval()
    base_ppl = compute_val_ppl(base_model, tok, args.val_jsonl)
    print(f"[eval] Base Gemma-3-4B held-out Nazrul PPL: {base_ppl:.2f}", flush=True)

    ft_model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    ft_model.eval()
    ft_ppl = compute_val_ppl(ft_model, tok, args.val_jsonl)
    print(f"[eval] Fine-tuned Nazrul-4B held-out Nazrul PPL: {ft_ppl:.2f}", flush=True)

    samples = []
    for name, prompt in TEST_PROMPTS:
        cands = 3 if any(k in name for k in ("poem", "geeti", "satire", "english")) else 1
        reply = generate(ft_model, tok, prompt, max_new_tokens=240, temperature=0.72, rhyme_candidates=cands)
        rhyme_sc = round(score_bangla_rhyme(reply), 2)
        print(f"\n--- [{name}] (rhyme_rate={rhyme_sc}) ---\nPROMPT: {prompt}\nNAZRUL-4B:\n{reply}", flush=True)
        samples.append({"name": name, "prompt": prompt, "rhyme_rate": rhyme_sc, "response": reply})

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
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
