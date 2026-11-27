#!/usr/bin/env bash
# End-to-end 2-hour L4 iterative V1 -> V2 -> V3 -> V4 -> Final training launcher for Camber.
set -euo pipefail

echo "=== [1/4] Installing dependencies ==="
python3 -m pip install -q --upgrade pip "brotli>=1.2.0" "brotlicffi>=1.2.0" "urllib3>=2.2.0"
python3 -m pip install -q -r requirements.txt

echo "=== [2/4] Building big Nazrul + Poetics + Modern Bangla dataset ==="
mkdir -p data/cache data/processed outputs
python3 src/prepare_data.py \
  --cache-dir data/cache \
  --out-dir data/processed

echo "=== [3/4] Running 2-hour L4 iterative training (V1 -> V2 -> V3 -> V4 -> Final) ==="
python3 src/train.py \
  --preset l4 \
  --data-dir data/processed \
  --out-dir outputs \
  --versions "${VERSIONS:-4}" \
  --save-every 50

echo "=== [4/4] Evaluating final Nazrul Gemma-4-E4B LoRA ==="
python3 src/eval_model.py \
  --adapter-dir outputs/nazrul-lora \
  --val-jsonl data/processed/cpt_val.jsonl \
  --out-file outputs/eval_report.json

echo "=== Done! Saved V1..V4 in outputs/versions/ and final adapter in outputs/nazrul-lora ==="
