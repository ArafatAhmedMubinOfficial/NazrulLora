# Nazrul-4B (`NazrulLora`) — Kazi Nazrul Islam Bangla AI

A compact, fluent **Gemma 4 E4B uncensored Bangla generative AI** (`coder3101/gemma-4-E4B-it-heretic` + 4-bit QLoRA `r=16`, fp16 adapter stored directly in Git without LFS) trained on Google Colab T4 via **Colab CLI** in under 1 hour.

## Why Heretic Gemma-4-E4B (`coder3101/gemma-4-E4B-it-heretic`)?
After benchmarking standard and abliterated 3B–4B candidates on Colab T4 (`unsloth/gemma-3-4b-it`, `mlabonne/gemma-3-4b-it-abliterated-v2`, `saidutta69/gemma-3-4b-it-heretic`, and `coder3101/gemma-4-E4B-it-heretic`), `coder3101/gemma-4-E4B-it-heretic` (decensored via Heretic v1.2.0 ARA) avoided both the repetition collapse of standard abliteration and English-word mixing, writing fluent rhymed Bengali couplets out of the box while fitting in 4-bit on T4 (`~8.68 GB`).

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

### 3. Train 2-Hour Multi-Round L4 Run on Camber (or any L4 / A100 cloud GPU)
The `unsensored` branch packs a large multi-round training dataset (`2,534` unique Nazrul works upsampled `3x`, `18` Bengali poetics/meter/Nazrul craft articles, `4,500` Bangla Wikipedia docs, `1,800` classic Bangla literature docs, `3,500` non-Nazrul Bengali poems, and `5,000` Bangla Alpaca-Orca instructions):

```bash
git clone -b unsensored https://github.com/ArafatAhmedMubinOfficial/NazrulLora.git && cd NazrulLora
bash scripts/run_camber_l4.sh
```
Or customize rounds/budgets directly:
```bash
python3 src/prepare_data.py --cache-dir data/cache --out-dir data/processed
python3 src/train.py --preset l4 --rounds 2 --data-dir data/processed --out-dir outputs
python3 src/eval_model.py --adapter-dir outputs/nazrul-lora
```

### 4. Retrain on Google Colab T4 via Colab CLI (< 1 Hour)
```powershell
.\scripts\run_colab_train.ps1
```
