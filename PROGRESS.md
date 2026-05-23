# KMVision-1 Progress

Last updated: 2026-05-24

## Project goal

Train a vision-language model (SigLIP2 + projector + Qwen2.5-Coder-1.5B with LoRA) to extract structured JSON from clinical chart images — Kaplan-Meier curves, forest plots, waterfall plots, and anchor charts.

---

## Data layout

Original synthetic data lived under `C:\sem4\KMVision-1 Data\dataset\images` and `labels\`. It was split into:

| Folder | Purpose | Count |
|--------|---------|-------|
| `train_1/images` + `train_1/labels` | Phase B/C training pool (100k, balanced) | 100,000 pairs |
| `testing/images` + `testing/labels` | Holdout eval (never used in training) | ~399,972 pairs |
| `split_manifest.json` | Reproducible train/test path lists (seed=42) | at dataset root |

**Phase B weights and Phase B evals** are archived separately (not moved with data):

- `archives/phase_b_baseline/checkpoints/phase_b_final/` — copy of `checkpoints/phase_b/final`
- `archives/phase_b_baseline/evaluation_results/` — all pre-Phase-C eval JSONL/summaries
- `archives/phase_b_baseline/EVAL_REPORT.md` — baseline metrics write-up

---

## What has been done so far

### Phase A — Projector warm-up
- Trained MLP projector only (vision + LLM frozen).
- Output: `checkpoints/checkpoints_projector/projector_weights.pth`

### Phase B — QLoRA fine-tuning (complete)
- **Architecture:** 5-crop spatial pooling (global 384² + 4 quadrants from 768²) → 3,645 visual tokens + text.
- **Trainable:** LoRA (r=64) + projector; vision frozen; LLM 4-bit.
- **Data:** Up to 100k samples, balanced across 8 chart categories.
- **Constraint:** `max_length=768` with verbose JSON labels → long KM JSON was **truncated during training** (the “768-token guillotine”).
- **Output:** `checkpoints/phase_b/final/` (+ step checkpoints every 250 optimizer steps under `checkpoints/phase_b/step_*`).

### Evaluation pipeline (built after Phase B)
- `evaluation/metrics.py` — structure-aware scoring (text, arms, curves, censoring).
- `eval_inference.py` — run checkpoint on `testing/` holdout.
- **Phase B baseline (12 KM charts):** ~0.54 overall after JSON repair; chart-type and schema triggering strong; structure/censoring weak due to truncation.

### Phase C preparation (done, training not started by you yet)
- `evaluation/schema_compact.py` — minified JSON keys, step-aware KM coordinate subsampling, caps, token-budget filter.
- `scripts/audit_label_token_lengths.py` — verify labels fit 768 tokens.
- `train_phase_c.py` — continues from Phase B `final`, compact labels, same prompt (no ChatML yet).
- Corrupt images: **logged and deleted** (`corrupted_images.log`); sample removed from in-memory list so training continues.
- Checkpoints: every **250** global steps → `checkpoints/phase_c/step_N/` with `adapter_model.safetensors`, `projector_weights.pth`, `checkpoint_meta.json`.

A short Phase C smoke run (~18 global steps) confirmed stability (~12.4 GB VRAM, loss ~0.7–1.0, no OOM). That run was **stopped** and **partial progress deleted** so you can start clean.

---

## Phase C — what it is and how it proceeds

**Problem Phase C fixes:** Verbose JSON labels exceeded the 768-token training window. The model never saw full multi-arm KM JSON, censoring ticks, or closing braces.

**Approach (single variable first):**
1. **Minified JSON** — short keys (`ct`, `ax`, `a`, `p`, `c`, etc.).
2. **Step-aware subsampling** — keep only KM step corners (drops + endpoints), max ~10 points/arm.
3. **Token budget filter** — only train pairs where compact label + prompt ≤ 768 tokens (~25,749 of 30k sampled).
4. **Same prompt as Phase B** — no ChatML yet (planned for a later pass after loss stabilizes).
5. **Init from** `checkpoints/phase_b/final` — vision + schema understanding preserved.

**Training hyperparameters:**
- 30k target subset → ~25.7k after budget filter
- Up to **2,000** optimizer steps (global step = every 16 micro-batches)
- LR `5e-5`, batch 1, grad accum 16
- Checkpoints every 250 steps

**Expected runtime:** ~1 min/optimizer step → ~33 hours for full 2,000 steps on RTX 4080 class GPU.

---

## Commands

### Start Phase C training (from repo root)

```powershell
cd c:\sem4\KMVision-1
python train_phase_c.py 2>&1 | Tee-Object -FilePath phase_c_training.log
```

### Monitor

```powershell
Get-Content c:\sem4\KMVision-1\phase_c_training.log -Tail 20 -Wait
```

### Manual checkpoint (while training)

Create an empty file `save_now.txt` in the repo root; training saves `checkpoints/phase_c/manual_step_{N}/` and deletes the trigger file.

### Audit label token lengths

```powershell
python scripts/audit_label_token_lengths.py --sample 300
```

### Eval (after Phase C checkpoint exists)

```powershell
python eval_inference.py --checkpoint checkpoints/phase_c/step_250 --category km --max-samples 12
```

(`eval_inference.py` decompresses compact model output before scoring.)

---

## Future plans

1. **Finish Phase C** (~2,000 steps) — target loss stabilizing near Phase B levels (~0.9 or lower) on compact JSON.
2. **Re-eval 12–30 fixed KM IDs** from `split_manifest.json` — compare structure/censoring vs Phase B baseline.
3. **ChatML prompt wrapper** — second Phase C pass *after* minified format is stable (one variable at a time).
4. **Investigate 0.001 time bug** — after full JSON fits, check if axis scale reading improves.
5. **Optional:** regenerate synthetic KM labels with subsampling at source (`generate_km.py`) for cleaner long-term data.
6. **Real PMC charts** — still unlabeled; manual annotation or weak supervision later.

---

## Key file map

| File | Role |
|------|------|
| `model.py` | SigLIP2 + projector + Qwen LLM |
| `train_phase_b.py` | Phase B training (reference) |
| `train_phase_c.py` | Phase C training |
| `evaluation/schema_compact.py` | Minify / decompress labels |
| `evaluation/metrics.py` | Eval scoring |
| `eval_inference.py` | Inference + eval |
| `scripts/organize_train_test.py` | Build train_1 / testing split |
| `scripts/archive_phase_b_baseline.py` | Archive baseline artifacts |
