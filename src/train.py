#!/usr/bin/env python3
"""Two-stage Colab T4 training for Nazrul-1B:
Stage 1: Domain Continual Pretraining (Basic Bangla + Kazi Nazrul Islam writings)
Stage 2: Instruction & Style Fine-Tuning (Assistant-only loss masking)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.peft_compat import patch_peft_torchao
except ImportError:
    from peft_compat import patch_peft_torchao

patch_peft_torchao()
from peft import LoraConfig, PeftModel, get_peft_model

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


class PackedCPTDataset(Dataset):
    """Packs tokenized documents into fixed seq_len blocks with <|end_of_text|> boundaries."""

    def __init__(self, jsonl_path: Path, tok: AutoTokenizer, seq_len: int = 512, max_blocks: int | None = None):
        sep_id = tok.convert_tokens_to_ids("<|end_of_text|>")
        stream: List[int] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                txt = json.loads(line)["text"]
                ids = tok.encode(txt, add_special_tokens=False)
                if ids:
                    stream.extend(ids)
                    stream.append(sep_id)
        n_blocks = len(stream) // seq_len
        if max_blocks is not None:
            n_blocks = min(n_blocks, max_blocks)
        self.blocks = [
            torch.tensor(stream[i * seq_len : (i + 1) * seq_len], dtype=torch.long)
            for i in range(n_blocks)
        ]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.blocks[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }


class SFTDataset(Dataset):
    """Formats prompt/response pairs into Llama-3 chat format with assistant-only loss masking."""

    def __init__(self, jsonl_path: Path, tok: AutoTokenizer, max_len: int = 576):
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.pad_id = tok.pad_token_id
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt_msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ]
                prompt_text = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
                prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
                resp_ids = tok.encode(row["response"] + "<|eot_id|>", add_special_tokens=False)

                full_ids = prompt_ids + resp_ids
                labels = [-100] * len(prompt_ids) + resp_ids
                if len(full_ids) > max_len:
                    if len(prompt_ids) >= max_len - 48:
                        continue
                    full_ids = full_ids[:max_len]
                    labels = labels[:max_len]

                self.samples.append((full_ids, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


def make_sft_collate(pad_id: int):
    """Pad each SFT mini-batch dynamically to its longest sequence (multiple of 8)."""
    def collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        max_l = max(len(x[0]) for x in batch)
        max_l = ((max_l + 7) // 8) * 8
        ids_list, mask_list, lbl_list = [], [], []
        for ids, lbl in batch:
            pad_len = max_l - len(ids)
            ids_list.append(ids + [pad_id] * pad_len)
            mask_list.append([1] * len(ids) + [0] * pad_len)
            lbl_list.append(lbl + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long),
            "labels": torch.tensor(lbl_list, dtype=torch.long),
        }
    return collate_fn


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, max_batches: int = 40) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(**batch)
        if not torch.isnan(out.loss):
            total_loss += float(out.loss.item())
            count += 1
    model.train()
    return total_loss / max(1, count)


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def run_stage(
    name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    grad_accum: int,
    max_epochs: float,
    max_steps: int,
    max_minutes: float,
    log_every: int = 5,
) -> dict:
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    epoch_steps = max(1, int(steps_per_epoch * max_epochs))
    total_opt_steps = min(epoch_steps, max_steps) if max_steps > 0 else epoch_steps
    warmup_steps = max(10, int(0.06 * total_opt_steps))

    val_before = evaluate_loss(model, val_loader)
    print(f"\n=== [{name}] start | micro_batches={len(train_loader)} | target_opt_steps={total_opt_steps} | max_min={max_minutes:.1f} | val_loss_before={val_before:.4f} (ppl={math.exp(min(20, val_before)):.2f}) ===", flush=True)

    t0 = time.time()
    opt_step = 0
    step_loss = 0.0
    log_loss_sum = 0.0
    log_steps = 0
    tokens_seen = 0
    epoch = 0
    stop_early = False

    opt.zero_grad()
    while opt_step < total_opt_steps and not stop_early:
        epoch += 1
        for micro_idx, batch in enumerate(train_loader):
            batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
            tokens_seen += int(batch["attention_mask"].sum().item())
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / grad_accum
            scaler.scale(loss).backward()
            step_loss += float(loss.item())

            if (micro_idx + 1) % grad_accum == 0 or (micro_idx + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                cur_lr = cosine_lr(opt_step, total_opt_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                opt_step += 1
                log_loss_sum += step_loss
                log_steps += 1
                step_loss = 0.0

                if opt_step % log_every == 0 or opt_step == total_opt_steps:
                    elapsed = time.time() - t0
                    tok_s = tokens_seen / max(0.1, elapsed)
                    avg_loss = log_loss_sum / max(1, log_steps)
                    print(
                        f"[{name}] step {opt_step}/{total_opt_steps} (ep {epoch}) | loss={avg_loss:.4f} | lr={cur_lr:.2e} | {tok_s:.0f} tok/s | {elapsed/60:.1f}m",
                        flush=True,
                    )
                    log_loss_sum = 0.0
                    log_steps = 0

                if (time.time() - t0) / 60.0 >= max_minutes:
                    print(f"[{name}] Reached time budget ({max_minutes:.1f}m), finishing stage cleanly.", flush=True)
                    stop_early = True
                    break
                if opt_step >= total_opt_steps:
                    break

    val_after = evaluate_loss(model, val_loader)
    elapsed_min = (time.time() - t0) / 60.0
    print(
        f"=== [{name}] done in {elapsed_min:.1f}m | opt_steps={opt_step} | tokens={tokens_seen:,} | val_loss_after={val_after:.4f} (ppl={math.exp(min(20, val_after)):.2f}) ===",
        flush=True,
    )
    return {
        "stage": name,
        "elapsed_min": round(elapsed_min, 2),
        "opt_steps": opt_step,
        "tokens_seen": tokens_seen,
        "val_loss_before": round(val_before, 4),
        "val_loss_after": round(val_after, 4),
        "val_ppl_before": round(math.exp(min(20, val_before)), 2),
        "val_ppl_after": round(math.exp(min(20, val_after)), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Nazrul-1B on Colab T4 in under 1 hour")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--stage1-minutes", type=float, default=27.0)
    parser.add_argument("--stage2-minutes", type=float, default=17.0)
    parser.add_argument("--stage1-max-steps", type=int, default=320)
    parser.add_argument("--stage2-max-steps", type=int, default=360)
    parser.add_argument("--stage1-epochs", type=float, default=1.0)
    parser.add_argument("--stage2-epochs", type=float, default=2.0)
    parser.add_argument("--cpt-seq-len", type=int, default=512)
    parser.add_argument("--sft-max-len", type=int, default=576)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--resume-lora", type=Path, default=None)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--sft-lr", type=float, default=8e-5)
    parser.add_argument("--save-merged", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] Loading tokenizer & base model: {BASE_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.resume_lora if args.resume_lora else BASE_MODEL_ID)
    if "<|finetune_right_pad_id|>" in tok.get_vocab():
        tok.pad_token = "<|finetune_right_pad_id|>"
    elif tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.chat_template = NAZRUL_CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=torch.float16, device_map="auto")
    if args.resume_lora:
        print(f"[train] Resuming trainable LoRA adapter from {args.resume_lora}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.resume_lora), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    s1_metrics = None
    if not args.skip_stage1:
        print("[train] Loading Stage 1 CPT datasets...", flush=True)
        cpt_train_ds = PackedCPTDataset(args.data_dir / "cpt_train.jsonl", tok, seq_len=args.cpt_seq_len)
        cpt_val_ds = PackedCPTDataset(args.data_dir / "cpt_val.jsonl", tok, seq_len=args.cpt_seq_len, max_blocks=80)
        cpt_train_loader = DataLoader(cpt_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        cpt_val_loader = DataLoader(cpt_val_ds, batch_size=args.batch_size, shuffle=False)
        print(f"[train] CPT train blocks={len(cpt_train_ds)} ({len(cpt_train_ds)*args.cpt_seq_len:,} tokens), val blocks={len(cpt_val_ds)}", flush=True)

        s1_metrics = run_stage(
            name="Stage1_CPT",
            model=model,
            train_loader=cpt_train_loader,
            val_loader=cpt_val_loader,
            lr=1.5e-4,
            grad_accum=args.grad_accum,
            max_epochs=args.stage1_epochs,
            max_steps=args.stage1_max_steps,
            max_minutes=args.stage1_minutes,
            log_every=5,
        )

        s1_dir = args.out_dir / "nazrul-lora-stage1"
        model.save_pretrained(s1_dir)
        tok.save_pretrained(s1_dir)

    print("[train] Loading Stage 2 SFT datasets...", flush=True)
    sft_train_ds = SFTDataset(args.data_dir / "sft_train.jsonl", tok, max_len=args.sft_max_len)
    sft_val_ds = SFTDataset(args.data_dir / "sft_val.jsonl", tok, max_len=args.sft_max_len)
    collate_fn = make_sft_collate(tok.pad_token_id)
    sft_train_loader = DataLoader(sft_train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    sft_val_loader = DataLoader(sft_val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[train] SFT train examples={len(sft_train_ds)}, val examples={len(sft_val_ds)}", flush=True)

    s2_metrics = run_stage(
        name="Stage2_SFT",
        model=model,
        train_loader=sft_train_loader,
        val_loader=sft_val_loader,
        lr=args.sft_lr,
        grad_accum=args.grad_accum,
        max_epochs=args.stage2_epochs,
        max_steps=args.stage2_max_steps,
        max_minutes=args.stage2_minutes,
        log_every=5,
    )

    final_lora_dir = args.out_dir / "nazrul-lora"
    # Cast adapter weights to fp16 before saving so adapter_model.safetensors (~86 MB) stays under GitHub's 100 MB limit
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.half()
    model.save_pretrained(final_lora_dir)
    tok.save_pretrained(final_lora_dir)
    print(f"[train] Saved final fp16 LoRA adapter & tokenizer to {final_lora_dir}", flush=True)

    if args.save_merged:
        merged_dir = args.out_dir / "nazrul-1b-merged"
        print(f"[train] Merging LoRA weights and saving to {merged_dir}...", flush=True)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)

    summary = {
        "base_model": BASE_MODEL_ID,
        "total_training_min": round((s1_metrics["elapsed_min"] if s1_metrics else 0.0) + s2_metrics["elapsed_min"], 2),
        "stage1": s1_metrics,
        "stage2": s2_metrics,
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[train] Complete summary:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
