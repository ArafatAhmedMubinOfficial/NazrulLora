#!/usr/bin/env bash
# End-to-end 2-hour L4 multi-round dataset build, training, and evaluation launcher for Camber / cloud Linux GPUs.
set -euo pipefail

echo "=== [1/4] Installing dependencies ==="
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r requirements.txt

echo "=== [2/4] Building big Nazrul + Poetics + Modern Bangla dataset ==="
mkdir -p data/cache data/processed outputs
python3 src/prepare_data.py \
  --cache-dir data/cache \
  --out-dir data/processed

echo "=== [3/4] Running 2-hour multi-round L4 QLoRA training (bf16, seq=768, 2 rounds) ==="
python3 src/train.py \
  --preset l4 \
  --data-dir data/processed \
  --out-dir outputs \
  --rounds "${ROUNDS:-2}" \
  --save-every 50

echo "=== [4/4] Evaluating trained Gemma-4-E4B Nazrul LoRA ==="
python3 src/eval_model.py \
  --adapter-dir outputs/nazrul-lora \
  --val-jsonl data/processed/cpt_val.jsonl \
  --out-file outputs/eval_report.json

echo "=== Done! Saved adapter in outputs/nazrul-lora and report in outputs/eval_report.json ==="
