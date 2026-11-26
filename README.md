# Nazrul-1B — Kazi Nazrul Islam Bangla AI

A compact, fluent **1.32B Bangla generative model** (`hishab/titulm-llama-3.2-1b-v2.0` + high-rank LoRA `r=64`, saved in fp16 at ~86 MB so the full trained adapter fits directly in Git without LFS) trained on Google Colab T4 via **Colab CLI** in under 1 hour.

## Training Architecture

1. **Base Backbone (`hishab/titulm-llama-3.2-1b-v2.0`)**:
   - 1.32B parameters with 42k extended Bangla BPE tokens (~2.32 Bangla chars/token), pretrained on 37B Bangla tokens so general Bangla grammar and reasoning stay coherent ("small, not dumb").
2. **Stage 1 — Continual Pretraining (Basic Bangla + Clean Kazi Nazrul Corpus)**:
   - **Main Corpus (2,584 unique clean Nazrul works, ~6.25M chars)**: Poems (*Agnibina*, *Samyabadi*, *Sarbahara*, *Dolonchapa*, etc.), *Nazrul Geeti* songs, novels (*Kuhelika*, *Bandhanhara*, *Mrityukshudha*), short stories, essays, plays, and letters from `ekpatagolpo`, `bangla-kobita`, `Bengali-Poem-Dataset`, `rishiraj/bengali-poems`, `BanglaMusicStylo`, `Shakil2448868`, and `NazrulRachanabali`.
   - **Basic Bangla Grounding (2,650 docs)**: Clean Bangla Wikipedia (`sayurio/bangla-wikipedia`) and classic Bengali literature (`barunsaha/bangla_sahitya` & classic poems).
3. **Stage 2 — Instruction & Nazrul Style Fine-Tuning (SFT)**:
   - Assistant-only masked instruction tuning over 6,635 Nazrul creative writing prompt pairs (Bangla & English prompts for poems, Nazrul Geeti, essays, stories, continuations) + 2,200 general Bangla instruction pairs (`BanglaLLM/bangla-alpaca-orca`).

---

## Run Anywhere (Colab, Kaggle, Lightning AI, Cimber, Modal)

### 1. Google Colab / Kaggle / Lightning AI / Cimber (Notebook or Terminal)
Open [`notebooks/Nazrul_1B_Playground.ipynb`](notebooks/Nazrul_1B_Playground.ipynb) directly, or run in any notebook cell / cloud terminal:

```bash
git clone https://github.com/ArafatAhmedMubinOfficial/NazrulLora.git && cd NazrulLora
pip install -q -r requirements.txt
python src/chat.py --prompt "কাজী নজরুল ইসলামের ভঙ্গিতে জাগরণ ও তারুণ্য নিয়ে একটি কবিতা লেখো।"
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
