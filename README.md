# KMVision-1

Vision-language model for extracting structured JSON from clinical trial charts (Kaplan-Meier, forest plots, waterfall plots, and general anchor charts).

## Architecture

`ClinicalMicroVLM` (`model.py`) combines three components:

| Component | Model | Role |
|-----------|-------|------|
| Vision encoder | `google/siglip2-so400m-patch14-384` | Frozen. Patch embeddings at 1152-dim, 729 patches per 384x384 crop. |
| Projector | 2-layer MLP (1152 -> 1536 -> 1536, GELU) | Maps vision tokens into the LLM embedding space. Trained in both phases. |
| LLM decoder | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | Phase A: frozen. Phase B: 4-bit quantized with QLoRA adapters (rank 64, alpha 128). |

**Inference flow:** image patches are projected and prepended as a visual prefix to the text prompt embeddings. The LLM generates JSON autoregressively from the concatenated sequence.

**Phase B image input (5-crop pooling):** each chart is split into one global 384x384 resize plus four 384x384 quadrants from a 768x768 resize. All five crops pass through SigLIP, producing 3645 image tokens (5 x 729) prepended to the text.

Base models are downloaded from Hugging Face on first run.

## Chart types and labels

Labels are JSON files paired with PNG images, organized by category subdirectory under `images/` and `labels/`.

| Category | Schema | Source |
|----------|--------|--------|
| `km` | `KMChartSchema`: axes, treatment arms, step-function coordinates, censoring ticks | Synthetic (50% of generation) + PMC real-world collection |
| `forest` | `ForestChartSchema`: studies with HR and CI, overall effect | Synthetic (10%) + PMC |
| `waterfall` | `WaterfallChartSchema`: per-subject bar values | Synthetic (10%) + PMC |
| `anchor` | `AnchorChartSchema`: bar, line, scatter, stacked bar, dual-axis combo | Synthetic (30%) |

Schema definitions: `synth_dataset/schemas.py`.

## Data pipeline

### Synthetic data

`synth_dataset/main.py` generates charts with matplotlib/lifelines into an external dataset root (default: `C:\sem4\KMVision-1 Data\dataset`). Distribution: 50% KM, 10% forest, 10% waterfall, 30% anchor variants. After generation, `augment.py` applies adversarial augmentations to 20% of images.

Lexical labels use word lists fetched by `synth_dataset/setup_data.py` into `C:\sem4\KMVision-1 Data\config`.

### Real-world data

`real_dataset/` collects charts from PubMed Central:

1. `scraper.py` searches PMC for article IDs by chart type.
2. `extracter.py` downloads figure images from those articles.
3. `labeler.py` provides manual labeling UI.
4. `reindexer.py` renames and organizes accepted images.

Targets: 250 KM, 125 forest, 125 waterfall (`real_dataset/config.py`).

### Train/test split

`scripts/organize_train_test.py` moves the Phase B training subset (100k balanced samples) into `train_1/` and the remainder into `testing/`. Selection logic mirrors `train_phase_b.py` with a fixed seed (42) for reproducibility.

## Training

### Phase A: projector warm-up (`train_phase_a.py`)

- Trains only the projector; vision encoder and LLM frozen.
- Single global crop per image (384x384).
- 25,000 samples, 1 epoch.
- Batch size 1, gradient accumulation 8, LR 1e-3.
- Max text sequence length 1536.
- Output: `checkpoints/phase_a_projector/projector_weights.pth`

### Phase B: QLoRA fine-tuning (`train_phase_b.py`)

- Loads Phase A projector weights from `checkpoints/checkpoints_projector/projector_weights.pth`.
- LLM loaded in 4-bit (bitsandbytes NF4), LoRA on attention and MLP projections, gradient checkpointing enabled.
- Projector remains trainable alongside LoRA.
- 100,000 samples with equal per-category balancing.
- Batch size 1, gradient accumulation 16, LR 5e-5, PagedAdamW8bit optimizer.
- Max text sequence length 768.
- 5-crop spatial pooling (3645 image tokens).
- 5% of samples use a chart-type classification prompt instead of full extraction.
- Checkpoints saved every 250 optimizer steps to `checkpoints/phase_b/step_N/`.
- Final weights: `checkpoints/phase_b/final/` (LoRA adapter + projector).

**Hardware:** developed on a single CUDA GPU (Windows). Phase B targets ~8 GB VRAM through 4-bit quantization, LoRA, and sequence length limits.

### Current training status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase A | Complete | Projector weights saved |
| Phase B | Complete | Trained through step 6250; final checkpoint at `checkpoints/phase_b/final/` |

Intermediate Phase B checkpoints exist at steps 250 through 6250 (every 250 steps).

## Evaluation

`eval_inference.py` runs inference on the `testing/` holdout and scores output with `evaluation/metrics.py`.

```bash
python eval_inference.py
python eval_inference.py --max-samples 50 --category km
python eval_inference.py --checkpoint checkpoints/phase_b/final
```

**Metrics (`evaluation/metrics.py`):**

- Kaplan-Meier: weighted composite of JSON validity, chart type match, label text similarity, arm structure, numeric curve accuracy (RMSE/MAE on step functions), and censoring tick overlap (F1).
- Other chart types: recursive field comparison with float tolerance.

Results are written to `evaluation/results/` as JSONL per-sample records and a summary JSON.

Unit tests: `python -m pytest evaluation/test_metrics.py` (or `python evaluation/test_metrics.py`).

## Project layout

```
model.py                  # ClinicalMicroVLM architecture
train_phase_a.py          # Phase A projector training
train_phase_b.py          # Phase B QLoRA training
eval_inference.py         # Holdout evaluation
evaluation/               # Metrics, parsing, data indexing, preprocessing
synth_dataset/            # Synthetic chart generation
real_dataset/             # PMC scraping, extraction, labeling
scripts/                  # Dataset organization utilities
checkpoints/              # Saved weights (not in git)
```

Dataset files live outside the repo at `C:\sem4\KMVision-1 Data\dataset` by default. Training scripts hardcode this path; update `IMAGE_DIR`, `LABEL_DIR`, and `DEFAULT_DATASET_ROOT` if your layout differs.

## Dependencies

Core ML stack (install separately; not pinned in `requirements.txt`):

```
torch
transformers
peft
accelerate
bitsandbytes
```

Data generation and collection (`requirements.txt`):

```
matplotlib seaborn lifelines numpy pandas pydantic albumentations tqdm
requests beautifulsoup4 selenium Pillow
```

## Usage summary

```bash
# 1. Generate synthetic data
python synth_dataset/setup_data.py
python synth_dataset/main.py --num_samples 500000

# 2. (Optional) Collect real-world charts
python real_dataset/run_collection.py

# 3. Train
python train_phase_a.py
python train_phase_b.py

# 4. Split dataset and evaluate
python scripts/organize_train_test.py
python eval_inference.py --checkpoint checkpoints/phase_b/final
```

Manual checkpoint trigger during Phase B training: create `save_now.txt` in the repo root.
