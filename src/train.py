#!/usr/bin/env python3
"""Two-stage Colab T4 QLoRA training for Nazrul-4B (coder3101/gemma-4-E4B-it-heretic):
Stage 1: Domain Continual Pretraining (Nazrul corpus + Bengali poetics + modern Bangla)
Stage 2: Craft, Rhyme-Blueprint & Custom-Prompt SFT (assistant-only loss masking)
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from src.peft_compat import patch_peft_torchao
except ImportError:
    from peft_compat import patch_peft_torchao

patch_peft_torchao()
from peft import LoraConfig, PeftModel, get_peft_model

BASE_MODEL_ID = "coder3101/gemma-4-E4B-it-heretic"
LANGUAGE_MODEL_LORA_REGEX = (
    r"(?:model\.language_model|model)\.layers\.\d+\.(?:self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)

SYSTEM_PROMPT = (
    "আপনি জাতীয় কবি কাজী নজরুল ইসলামের সাহিত্যিক ভাবধারা, ছন্দ, অন্ত্যমিল, ওজস্বী ভাষা ও সৃজনশীল "
    "শৈলীতে দক্ষ একজন বাংলা এআই লেখক। ব্যবহারকারীর যেকোনো প্রম্পট (আধুনিক বা ধ্রুপদী) বুঝে কাজী নজরুলের "
    "ভঙ্গিতে ছন্দোবদ্ধ ও অন্ত্যমিলযুক্ত কবিতা, নজরুলগীতি, গল্প বা প্রবন্ধ রচনা করুন এবং সাধারণ বাংলা প্রশ্নেরও নির্ভুল উত্তর দিন।"
)


def format_gemma_prompt(tok: AutoTokenizer, user_prompt: str) -> str:
    """Format user turn with system guidance using Gemma-3's native turn template."""
    msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class PackedCPTDataset(Dataset):
    """Packs tokenized documents into fixed seq_len blocks with document EOS boundaries."""

    def __init__(self, jsonl_path: Path, tok: AutoTokenizer, seq_len: int = 512, max_blocks: int | None = None):
        eos_cand = tok.convert_tokens_to_ids("<eos>")
        sep_id = eos_cand if isinstance(eos_cand, int) and eos_cand >= 0 else tok.eos_token_id
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
    """Formats prompt/response pairs into Gemma-3 turn format with assistant-only loss masking."""

    def __init__(self, jsonl_path: Path, tok: AutoTokenizer, max_len: int = 576):
        self.samples: List[Tuple[List[int], List[int]]] = []
        self.pad_id = tok.pad_token_id
        eot_tok = "<end_of_turn>" if "<end_of_turn>" in tok.get_vocab() else (tok.eos_token or "")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                prompt_text = format_gemma_prompt(tok, row["prompt"])
                prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
                resp_ids = tok.encode(row["response"] + eot_tok + "\n", add_special_tokens=False)

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


def _compute_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def evaluate_loss(model: torch.nn.Module, loader: DataLoader, compute_dtype: torch.dtype, max_batches: int = 30) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            out = model(**batch)
        if not torch.isnan(out.loss):
            total_loss += float(out.loss.item())
            count += 1
    model.train()
    torch.cuda.empty_cache()
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
    tok: AutoTokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    grad_accum: int,
    max_epochs: float,
    max_steps: int,
    max_minutes: float,
    compute_dtype: torch.dtype,
    ckpt_dir: Path | None = None,
    save_every: int = 50,
    log_every: int = 2,
) -> dict:
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train_loader) // grad_accum)
    epoch_steps = max(1, int(steps_per_epoch * max_epochs))
    total_opt_steps = min(epoch_steps, max_steps) if max_steps > 0 else epoch_steps
    warmup_steps = max(4, int(0.08 * total_opt_steps))

    val_before = evaluate_loss(model, val_loader, compute_dtype)
    print(
        f"\n=== [{name}] start | micro_batches={len(train_loader)} | target_opt_steps={total_opt_steps} | max_min={max_minutes:.1f} | lr={lr:.2e} | val_loss_before={val_before:.4f} (ppl={math.exp(min(20, val_before)):.2f}) ===",
        flush=True,
    )

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
            with torch.amp.autocast("cuda", dtype=compute_dtype):
                loss = model(**batch).loss / grad_accum
            loss.backward()
            step_loss += float(loss.item())

            if (micro_idx + 1) % grad_accum == 0 or (micro_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                cur_lr = cosine_lr(opt_step, total_opt_steps, warmup_steps, lr)
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
                opt.step()
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

                if ckpt_dir is not None and save_every > 0 and opt_step % save_every == 0:
                    latest = ckpt_dir / "latest"
                    model.save_pretrained(latest)
                    tok.save_pretrained(latest)

                if (time.time() - t0) / 60.0 >= max_minutes:
                    print(f"[{name}] Reached time budget ({max_minutes:.1f}m), finishing stage cleanly.", flush=True)
                    stop_early = True
                    break
                if opt_step >= total_opt_steps:
                    break

    val_after = evaluate_loss(model, val_loader, compute_dtype)
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
    parser = argparse.ArgumentParser(description="Multi-round QLoRA training for Nazrul-4B (Gemma-4-E4B on L4/T4)")
    parser.add_argument("--preset", choices=["l4", "t4"], default="l4", help="l4=2h V1->V4 loop (24GB bf16); t4=<1h (15GB fp16)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--rounds", "--versions", dest="rounds", type=int, default=None, help="Number of iterative training versions (V1..VN)")
    parser.add_argument("--start-version", type=int, default=1, help="Starting version number (set 2 when resuming from V1 to save V2, etc.)")
    parser.add_argument("--stage1-minutes", type=float, default=None)
    parser.add_argument("--stage2-minutes", type=float, default=None)
    parser.add_argument("--stage1-max-steps", type=int, default=None)
    parser.add_argument("--stage2-max-steps", type=int, default=None)
    parser.add_argument("--stage1-epochs", type=float, default=1.0)
    parser.add_argument("--stage2-epochs", type=float, default=3.0)
    parser.add_argument("--cpt-seq-len", type=int, default=None)
    parser.add_argument("--sft-max-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-preview", action="store_true", help="Skip short sample poem preview after each version")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--resume-lora", type=Path, default=None)
    parser.add_argument("--skip-stage1", action="store_true")
    parser.add_argument("--cpt-lr", type=float, default=1e-4)
    parser.add_argument("--sft-lr", type=float, default=6e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    presets = {
        # L4 2-hour 4-version loop (V1 ~36m, V2 ~28m, V3 ~26m, V4 ~24m => ~114m total)
        "l4": dict(rounds=4, s1_min=22.0, s2_min=14.0, s1_steps=180, s2_steps=160, cpt_len=768, sft_len=768, bs=2, accum=8),
        "t4": dict(rounds=1, s1_min=18.0, s2_min=27.0, s1_steps=45, s2_steps=115, cpt_len=512, sft_len=576, bs=1, accum=16),
    }
    p = presets[args.preset]
    rounds = args.rounds if args.rounds is not None else p["rounds"]
    s1_min = args.stage1_minutes if args.stage1_minutes is not None else p["s1_min"]
    s2_min = args.stage2_minutes if args.stage2_minutes is not None else p["s2_min"]
    s1_steps = args.stage1_max_steps if args.stage1_max_steps is not None else p["s1_steps"]
    s2_steps = args.stage2_max_steps if args.stage2_max_steps is not None else p["s2_steps"]
    cpt_seq_len = args.cpt_seq_len if args.cpt_seq_len is not None else p["cpt_len"]
    sft_max_len = args.sft_max_len if args.sft_max_len is not None else p["sft_len"]
    batch_size = args.batch_size if args.batch_size is not None else p["bs"]
    grad_accum = args.grad_accum if args.grad_accum is not None else p["accum"]

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = _compute_dtype()
    print(f"[train] Preset={args.preset} | rounds={rounds} | compute_dtype={compute_dtype} | base={BASE_MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.resume_lora if args.resume_lora else BASE_MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb, dtype=compute_dtype, device_map="auto")
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad_(False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.resume_lora:
        print(f"[train] Resuming trainable QLoRA adapter from {args.resume_lora}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.resume_lora), is_trainable=True)
        for param in model.parameters():
            if param.requires_grad:
                param.data = param.data.to(compute_dtype)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LANGUAGE_MODEL_LORA_REGEX,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    cpt_train_loader = cpt_val_loader = None
    if not args.skip_stage1:
        print("[train] Loading Stage 1 CPT datasets...", flush=True)
        cpt_train_ds = PackedCPTDataset(args.data_dir / "cpt_train.jsonl", tok, seq_len=cpt_seq_len)
        cpt_val_ds = PackedCPTDataset(args.data_dir / "cpt_val.jsonl", tok, seq_len=cpt_seq_len, max_blocks=40)
        cpt_train_loader = DataLoader(cpt_train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        cpt_val_loader = DataLoader(cpt_val_ds, batch_size=1, shuffle=False)
        print(f"[train] CPT train blocks={len(cpt_train_ds)} ({len(cpt_train_ds)*cpt_seq_len:,} tokens), val blocks={len(cpt_val_ds)}", flush=True)

    print("[train] Loading Stage 2 SFT datasets...", flush=True)
    sft_train_ds = SFTDataset(args.data_dir / "sft_train.jsonl", tok, max_len=sft_max_len)
    sft_val_ds = SFTDataset(args.data_dir / "sft_val.jsonl", tok, max_len=sft_max_len)
    collate_fn = make_sft_collate(tok.pad_token_id)
    sft_train_loader = DataLoader(sft_train_ds, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn)
    sft_val_loader = DataLoader(sft_val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"[train] SFT train examples={len(sft_train_ds)}, val examples={len(sft_val_ds)}", flush=True)

    versions_dir = args.out_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    round_logs = []
    total_min = 0.0
    # L4 4-version progression: V1 (broad CPT+SFT) -> V2 -> V3 -> V4 (SFT-heavy polish with decayed LR)
    l4_cpt_scales = [1.0, 0.45, 0.27, 0.18]
    l4_sft_scales = [1.0, 1.29, 1.43, 1.43]
    l4_lr_scales = [1.0, 0.72, 0.50, 0.35]

    for step_idx in range(rounds):
        ver_num = args.start_version + step_idx
        ver_tag = f"v{ver_num}"
        scale_i = min(ver_num - 1, len(l4_lr_scales) - 1)
        decay = l4_lr_scales[scale_i] if ver_num <= 4 else 0.35 * (0.7 ** (ver_num - 4))
        cpt_scale = l4_cpt_scales[scale_i] if args.preset == "l4" else (1.0 if step_idx == 0 else 0.35)
        sft_scale = l4_sft_scales[scale_i] if args.preset == "l4" else 1.0

        r_s1_min = round(s1_min * cpt_scale, 2)
        r_s2_min = round(s2_min * sft_scale, 2)
        r_s1_steps = max(20, int(s1_steps * cpt_scale))
        r_s2_steps = max(40, int(s2_steps * sft_scale))

        print(f"\n==================== Training Version {ver_tag.upper()} ====================", flush=True)
        s1_metrics = None
        if not args.skip_stage1 and cpt_train_loader is not None and r_s1_min > 0:
            s1_metrics = run_stage(
                name=f"{ver_tag.upper()}_CPT",
                model=model,
                tok=tok,
                train_loader=cpt_train_loader,
                val_loader=cpt_val_loader,
                lr=args.cpt_lr * decay,
                grad_accum=grad_accum,
                max_epochs=args.stage1_epochs,
                max_steps=r_s1_steps,
                max_minutes=r_s1_min,
                compute_dtype=compute_dtype,
                ckpt_dir=ckpt_dir,
                save_every=args.save_every,
                log_every=5,
            )
            total_min += s1_metrics["elapsed_min"]

        s2_metrics = run_stage(
            name=f"{ver_tag.upper()}_SFT",
            model=model,
            tok=tok,
            train_loader=sft_train_loader,
            val_loader=sft_val_loader,
            lr=args.sft_lr * decay,
            grad_accum=grad_accum,
            max_epochs=args.stage2_epochs,
            max_steps=r_s2_steps,
            max_minutes=r_s2_min,
            compute_dtype=compute_dtype,
            ckpt_dir=ckpt_dir,
            save_every=args.save_every,
            log_every=5,
        )
        total_min += s2_metrics["elapsed_min"]

        v_dir = versions_dir / ver_tag
        model.save_pretrained(v_dir)
        tok.save_pretrained(v_dir)
        print(f"[train] Saved checkpoint version {ver_tag.upper()} -> {v_dir}", flush=True)

        preview_text = None
        if not args.no_preview:
            model.eval()
            model.config.use_cache = True
            with torch.no_grad():
                p_txt = format_gemma_prompt(
                    tok,
                    "কাজী নজরুল ইসলামের ভঙ্গিতে কঠোর ছন্দ ও অন্ত্যমিল বজায় রেখে অন্যায় ও ভণ্ডামির বিরুদ্ধে ৬-৮ লাইনের একটি কবিতা লেখো।",
                )
                p_in = tok(p_txt, return_tensors="pt").to(model.device)
                eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
                eos_ids = [i for i in {tok.eos_token_id, eot_id} if isinstance(i, int) and i >= 0]
                with torch.amp.autocast("cuda", dtype=compute_dtype):
                    p_out = model.generate(
                        **p_in,
                        max_new_tokens=160,
                        do_sample=True,
                        temperature=0.72,
                        top_p=0.9,
                        repetition_penalty=1.15,
                        eos_token_id=eos_ids,
                        pad_token_id=tok.pad_token_id,
                    )
                preview_text = tok.decode(p_out[0][p_in["input_ids"].shape[1] :], skip_special_tokens=True).strip()
            model.config.use_cache = False
            model.train()
            print(f"[{ver_tag.upper()} preview]\n{preview_text}\n", flush=True)
            (versions_dir / f"{ver_tag}_preview.txt").write_text(preview_text, encoding="utf-8")

        round_logs.append(
            {"version": ver_tag, "adapter_dir": str(v_dir), "stage1": s1_metrics, "stage2": s2_metrics, "preview": preview_text}
        )
        (versions_dir / "version_history.json").write_text(
            json.dumps(round_logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    final_lora_dir = args.out_dir / "nazrul-lora"
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.half()
    model.save_pretrained(final_lora_dir)
    tok.save_pretrained(final_lora_dir)
    print(f"[train] Saved final fp16 LoRA adapter & tokenizer to {final_lora_dir}", flush=True)

    summary = {
        "base_model": BASE_MODEL_ID,
        "preset": args.preset,
        "rounds_completed": rounds,
        "total_training_min": round(total_min, 2),
        "rounds": round_logs,
    }
    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[train] Complete summary:", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
