# Nazrul-4B (`NazrulLora`) — Kazi Nazrul Islam Bangla AI

A compact, fluent **4.3B Bangla generative AI** (`unsloth/gemma-3-4b-it` + 4-bit QLoRA `r=16`, `~57 MB` fp16 adapter stored directly in Git without LFS) trained on Google Colab T4 via **Colab CLI** in under 1 hour.

## Why Gemma-3-4B (`unsloth/gemma-3-4b-it`)?
After benchmarking five 2025–2026 `2B–4B` models head-to-head on Colab T4 (`hishab/titulm-llama-3.2-3b-v2.0-Instruct-v1.0`, `BanglaLLM/Bangla-s1k-qwen-2.5-3B-Instruct`, `Qwen/Qwen3-4B-Instruct-2507`, `Qwen/Qwen3.5-4B`, and `unsloth/gemma-3-4b-it`), `unsloth/gemma-3-4b-it` led in Bengali tokenizer efficiency (`3.74` chars/token), modern custom-prompt adherence, and zero-shot Bengali end-rhyme (`অন্ত্যমিল`), while fitting in `~3.25 GB` VRAM in 4-bit.

## Two-Stage Training Pipeline (< 1 Hour on Colab T4)
1. **Stage 1 — Continual Pretraining (Nazrul Corpus + Bengali Poetics + Modern Context)**:
   - **Kazi Nazrul Islam Works (`2,534+` unique clean writings):** Poems, *Nazrul Geeti* songs, novels (*Kuhelika*, *Bandhanhara*, *Mrityukshudha*), short stories, essays, plays, and letters.
   - **Bengali Poetics & Modern Bangla Grounding:** Bangla Wikipedia articles on Bengali meter (*ছন্দ*), poetic ornaments (*অলংকার*), *Bidrohi*, *Nazrul Geeti*, and modern encyclopedic Bangla (`sayurio/bangla-wikipedia`) + classic literature (`barunsaha/bangla_sahitya`).
2. **Stage 2 — Craft, Rhyme-Blueprint & Custom-Prompt SFT**:
   - Assistant-only loss masking over poetics craft Q&A, rhyme-planned Nazrul stanzas (`[ছন্দ ও মিল-পরিকল্পনা]` + rhymed verse), Nazrul Geeti, verse continuations, and modern Bangla instructions (`BanglaLLM/bangla-alpaca-orca`).

---

## Run Anywhere (Colab, Kaggle, Lightning AI, Cimber, Modal)

### 1. Google Colab / Kaggle / Lightning AI / Cimber
Open [`notebooks/Nazrul_1B_Playground.ipynb`](notebooks/Nazrul_1B_Playground.ipynb) or run in any notebook / cloud terminal:

```bash
git clone https://github.com/ArafatAhmedMubinOfficial/NazrulLora.git && cd NazrulLora
pip install -q -r requirements.txt
python src/chat.py --rhyme-candidates 2 --prompt "কাজী নজরুল ইসলামের ভঙ্গিতে ছন্দ ও অন্ত্যমিল বজায় রেখে জাগরণ নিয়ে একটি কবিতা লেখো।"
```

Interactive chat mode:
```bash
python src/chat.py
```

### 2. Modal (Serverless T4 GPU)
```bash
pip install modal && modal setup
modal run scripts/modal_app.py --prompt "কাজী নজরুলের ভঙ্গিতে বর্ষা ও মেঘ নিয়ে একটি নজরুলগীতি লেখো।"
```

### 3. Retrain on Google Colab via Colab CLI (< 1 Hour)
```powershell
.\scripts\run_colab_train.ps1
```
