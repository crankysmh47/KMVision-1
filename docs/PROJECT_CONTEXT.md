# KMVision-1: Complete Project Context

This document contains every piece of context needed to work on this project without reading the codebase. It covers the goal, architecture, data pipeline, training history, evaluation metrics, current results, known issues, file layout, and remaining work.

Last updated: 2026-06-06.

---

## 1. Project Goal

Extract structured JSON data from clinical trial chart images — primarily Kaplan-Meier (KM) survival curves, but also forest plots, waterfall plots, and general anchor charts (bar, line, scatter, stacked bar, dual-axis combo).

Given a chart image (PNG), the model outputs a JSON object containing:
- Chart type
- Axis labels and max values
- Per-arm treatment labels, dense coordinate arrays, and censoring tick locations

**Target benchmark:** 95% overall score on a weighted composite metric (see Section 7). Macro model: 0.73 on 12-chart holdout. Stage 2 v2.2: 77.3% strict JSON (parser-rescored), 53% point match. **First E2E score: 0.59 overall** on 12 charts (tile stitch → clinical → `evaluation/metrics.py`).

---

## 2. Architecture

### ClinicalMicroVLM (`model.py`, 108 lines)

Three components wired together:

```
Image (384x384 crops) --> SigLIP2 Vision Encoder --> MLP Projector --> Qwen 2.5 Coder 1.5B LLM
                          (frozen, 1152-dim)        (1152->1536)       (4-bit QLoRA)
```

| Component | HuggingFace ID | Hidden Dim | Parameters | Notes |
|-----------|----------------|------------|------------|-------|
| Vision Encoder | `google/siglip2-so400m-patch14-384` | 1152 | ~400M | Always frozen. Outputs 729 patch tokens per 384x384 crop. Uses `.vision_model` only (text tower discarded). |
| Projector | Custom 2-layer MLP | 1152 -> 1536 | ~3.5M | `Linear(1152,1536) -> GELU -> Linear(1536,1536)`. bfloat16. Trained in ALL phases. |
| LLM Decoder | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | 1536 | ~1.5B (4-bit: ~750M effective) | Phase A: frozen. Phase B+: 4-bit NF4 quantized, QLoRA adapters. SDPA attention. |

### Forward Pass

1. `pixel_values` shape `(B, num_crops, 3, 384, 384)` -- flatten to `(B*crops, 3, 384, 384)`
2. Vision encoder (under `torch.no_grad()`): `(B*crops, 729, 1152)`
3. Projector: `(B*crops, 729, 1536)` -- reshape to `(B, crops*729, 1536)`
4. Text embeddings from LLM's embedding layer (under `torch.no_grad()`): `(B, seq_len, 1536)`
5. Concatenate: `[image_embeds | text_embeds]` along sequence dimension
6. Extend attention mask (all-ones for image tokens) and labels (-100 for image tokens)
7. Feed `inputs_embeds` + mask + labels into LLM for autoregressive generation

**Macro model (Phase B/C):** 5 crops (global + 4 quadrants) = 3645 image tokens  
**Stage 2 (tiles):** 1 crop = 729 image tokens

### LoRA Configuration

- Rank: 64, Alpha: 128
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Task type: `CAUSAL_LM`
- Gradient checkpointing enabled

### Quantization

BitsAndBytes 4-bit: NF4 type, double quantization, bfloat16 compute dtype.

---

## 3. Data Pipeline

### 3.1 Synthetic Data Generation

**Generator:** `synth_dataset/main.py` with multiprocessing.

| Chart Type | Fraction | Generator | Schema |
|------------|----------|-----------|--------|
| Kaplan-Meier | 50% | `generate_km.py` (lifelines KaplanMeierFitter, Weibull survival) | `KMChartSchema` |
| Forest Plot | 10% | `generate_clinical.py` (errorbar studies + diamond overall) | `ForestChartSchema` |
| Waterfall Plot | 10% | `generate_clinical.py` (per-subject bars) | `WaterfallChartSchema` |
| Anchor Charts | 30% | `generate_anchor.py` (bar/line/scatter/stacked/dual-axis) | `AnchorChartSchema` |

**Volume:** ~500k total images generated. Schemas defined in `synth_dataset/schemas.py`.

**Augmentation:** `synth_dataset/augment.py` applies adversarial transforms (JPEG compression, Gaussian noise/blur, coarse dropout) to 20% of images via Albumentations.

**Lexical engine:** `synth_dataset/lexical_engine.py` generates axis labels and treatment names. 80% clinical combinatorics (medical terms), 20% dictionary words. 30% OCR-style typo noise. Word lists fetched by `synth_dataset/setup_data.py`.

### 3.2 Train/Test Split

`scripts/organize_train_test.py` creates:
- `train_1/`: 100k balanced samples (25k per category) -- used for Phase B/C training
- `testing/`: remaining ~238k samples -- used for macro evaluation and Stage 2 tile generation

Selection uses seed 42 for reproducibility. Manifest: `split_manifest.json`.

### 3.3 Label Formats

**Verbose (on-disk, all phases):**
```json
{
  "chart_type": "kaplan_meier",
  "axes": {
    "x": {"label": "Time (Months)", "max_value": 50.0},
    "y": {"label": "Survival Probability", "max_value": 1.0}
  },
  "arms": [{
    "treatment_label": "Drug A",
    "coordinates": [[0, 1.0], [5, 0.9], [10, 0.7], ...],
    "censoring_ticks": [8.0, 15.0]
  }]
}
```

**Compact / minified (Phase C training targets):**
```json
{"ct":"km","ax":{"x":{"l":"Time (Months)","m":50.0},"y":{"l":"Survival Probability","m":1.0}},"a":[{"id":"Drug A","p":[[0,1.0],[5,0.9]],"c":[8.0]}]}
```
Key map: `ct`=chart_type, `ax`=axes, `l`=label, `m`=max_value, `a`=arms, `id`=treatment_label, `p`=coordinates, `c`=censoring_ticks. Step-aware subsampling caps to 10 coordinates per arm, 6 censors per arm. Code: `evaluation/schema_compact.py`.

**Stage 2 tile labels (on-disk, nested):**
```json
{
  "arm_id": "Drug A",
  "points": [[0.145, 0.112], [0.300, 0.250]],
  "censors": [[0.500, 0.300]],
  "_meta": {
    "coordinate_space": "normalized_local",
    "coord_decimals": 3,
    "source_chart": "chart_abc123_km",
    "tile_origin": [108, 92],
    "plot_bbox": [108, 92, 737, 599],
    "axis_max": {"x": 50.0, "y": 1.0},
    "points_before_cap": 87,
    "censors_before_cap": 12,
    "max_points_per_tile": 40,
    "max_censors_per_tile": 10
  }
}
```

**Stage 2 training target (flat, generated at load time by `stage2_common.stage2_target_json()`):**
```json
{"arm_id":"Drug A","points":[0.145,0.112,0.300,0.250],"censors":[0.500,0.300]}
```
The `_meta` block is excluded from training. Points/censors are flattened from `[[x,y],...]` to `[x,y,x,y,...]` at load time -- no tile regeneration needed.

### 3.4 Stage 2 Tile Generation

**Script:** `scripts/generate_stage2_tiles.py`

**Pipeline:**
1. Load 768x768 chart image + verbose JSON label
2. Estimate plot bounding box from raster (non-white pixel mask with margin shrink; fallback to fixed insets)
3. Slide 384x384 windows horizontally with 50px overlap across the plot area (vertically centered)
4. For each window, for each arm: filter points/censors that fall inside the tile (pixel-in-tile test)
5. Skip tiles with < 2 points for that arm
6. Step-aware subsample in clinical space, cap to 40 points / 10 censors
7. Convert coordinates: clinical -> global pixel -> local tile pixel -> normalized [0,1]

**Coordinate transform (clinical -> normalized local):**
```
px = plot.x0 + (t / x_max) * plot.width
py = plot.y1 - (s / y_max) * plot.height    # y-axis inverted (image y down)
x_local = px - tile.x0
y_local = py - tile.y0
x_norm = round(x_local / 384, 3)            # clamped to [0, 1]
y_norm = round(y_local / 384, 3)
```

**Inverse transform:** `scripts/stage2_coordinate_transform.py` reverses using `_meta` fields.

**Output:** `stage2_v2_1/images/km/*.png` + `stage2_v2_1/labels/km/*.json`  
**Holdout:** 5% of source charts -> `stage2_v2_1_holdout/`  
**Volume:** 71,451 train tiles, 3,778 holdout tiles from 11,400 source charts (from `testing/`)

### 3.5 Real-World Data

**Status:** ~100 KM images collected from PMC, unlabeled. Target: 250 KM, 125 forest, 125 waterfall.

**Pipeline:** `real_dataset/scraper.py` (PMC search) -> `real_dataset/extracter.py` (download) -> `real_dataset/labeler.py` (Tkinter manual annotation) -> `real_dataset/reindexer.py` (organize).

---

## 4. Training History

### Phase A: Projector Warm-Up (`train_phase_a.py`)

| Setting | Value |
|---------|-------|
| Trainable | Projector only (~3.5M params) |
| Frozen | Vision encoder + LLM |
| Input | Single 384x384 crop -> 729 image tokens |
| Data | 25k samples, 1 epoch |
| LR | 1e-3, AdamW |
| Grad accum | 8 |
| max_length | 1536 |
| Output | `checkpoints/phase_a_projector/projector_weights.pth` |

### Phase B: QLoRA Fine-Tuning (`train_phase_b.py`)

| Setting | Value |
|---------|-------|
| Init | Phase A projector weights |
| Trainable | LoRA adapters + projector |
| Frozen | Vision encoder; LLM 4-bit quantized |
| Input | 5-crop pooling -> 3645 image tokens |
| Data | 100k balanced (25k per category), 1 epoch |
| LR | 5e-5, PagedAdamW8bit |
| Grad accum | 16 |
| max_length | 768 (**caused truncation bug -- see Section 6**) |
| Classify router | 5% of samples use chart-type classification prompt |
| Checkpoint | Every 250 steps -> `checkpoints/phase_b/step_N/` |
| Final | `checkpoints/phase_b/final/` (step 6250) |

### Phase C: Compact JSON Fine-Tuning (`train_phase_c.py`)

Three runs via automated week queue (`scripts/week_queue.py`):

| Run | Config | Init | Steps | Overall | Numeric | Censoring | Gate |
|-----|--------|------|-------|---------|---------|-----------|------|
| Run 1 | Minified JSON keys | Phase B final | ~1609 | **0.710** | 0.560 | 0.156 | PASS |
| **Run 2** | + ChatML wrapping | Run 1 final | 2000 | **0.730** | 0.614 | 0.152 | **PASS** |
| Run 3 | + Low LR (1e-5) | Run 2 final | ~1588 | 0.717 | 0.587 | 0.138 | FAIL |

**Production macro model:** `checkpoints/phase_c_run2_chatml/final/`

Run 3 failed the censoring gate (ratio 0.909, needed >= 0.95). Gate config: `config/eval_gates.json`.

### Stage 2: Tile Micro-Extractor (`train_stage2.py`)

Four format iterations:

| Version | Coords | Format | Steps | Strict JSON | Point Match | Key Issue |
|---------|--------|--------|-------|-------------|-------------|-----------|
| v1 | Clinical `(t, s)` | Nested `[[t,s],...]` | 3000 | 0% | 2.1% | `max_length=512` truncated 800+ token labels |
| v2 | Clinical, capped 40pts/10cens | Nested | 3000 | 0% | 3.1% | Model can't output clinical coords from axisless crops |
| v2.1 | Normalized [0,1] | Nested `[[x,y],...]` | 500 | 20% | n/a | Bracket chaos from nested array depth |
| **v2.2** | Normalized [0,1] | **Flat `[x,y,x,y,...]`** | 3000 | **77%** (rescored) | **53.2%** | Loss not converged (0.4875); 10k train in progress |

**v2.2 details:**
- Init: Fresh LoRA + Phase A projector only (no Phase B/C LoRA)
- Prefix-masked loss: mask through `{"arm_id": "...", "points": [`; model only learns coordinate stream
- Inference: `--force-json-prefix` pre-fills assistant through `"points": [`
- `max_length=1024`, grad_accum=8, LR 5e-5 (same as Phase B)
- Checkpoint: `checkpoints/stage2_v2_1/final`

---

## 5. File Layout

### Core Model + Training

| File | Purpose |
|------|---------|
| `model.py` | `ClinicalMicroVLM` class definition (108 lines) |
| `train_phase_a.py` | Phase A projector warm-up |
| `train_phase_b.py` | Phase B QLoRA fine-tuning |
| `train_phase_c.py` | Phase C compact JSON training |
| `train_stage2.py` | Stage 2 tile micro-extractor training |
| `stage2_common.py` | Shared Stage 2 helpers: prompt template, JSON prefix forcing, flat_xy conversions, mask length computation |

### Evaluation

| File | Purpose |
|------|---------|
| `eval_inference.py` | Macro model eval on `testing/` holdout |
| `eval_stage2.py` | Stage 2 tile eval (RMSE + censoring F1) |
| `eval_e2e.py` | End-to-end eval: tile JSONL → stitch → score vs chart GT |
| `eval_realworld.py` | Macro eval on labeled real-world KM charts |
| `train_realworld.py` | Fine-tune Phase C on real-world labeled data |
| `scripts/realworld_status.py` | Real-world collection/labeling progress report |
| `evaluation/metrics.py` | 690-line weighted composite scorer (see Section 7) |
| `evaluation/parse_output.py` | Robust JSON extraction from model text (fences, truncation repair, `json_repair` fallback) |
| `evaluation/schema_compact.py` | Minify/decompress JSON labels, `build_training_text()`, step-aware subsampling |
| `evaluation/image_preprocess.py` | 5-crop builder for macro model |
| `evaluation/data_index.py` | Dataset indexing and balanced sampling |
| `evaluation/rescore_jsonl.py` | Re-score saved eval JSONL after parser improvements |

### Scripts

| File | Purpose |
|------|---------|
| `scripts/generate_stage2_tiles.py` | Tile dataset generator (clinical -> normalized coords, sliding windows) |
| `scripts/stitch_tiles.py` | Stitch tile preds → clinical KM JSON (overlap dedupe) |
| `scripts/stage2_coordinate_transform.py` | Inverse transform: normalized tile -> clinical using `_meta` |
| `scripts/stage2_sanity_check.py` | Quick JSON validity check on holdout tiles |
| `scripts/training_checkpoint.py` | Checkpoint infra: ordered dirs, `latest.json`, resume with RNG state |
| `scripts/training_lock.py` | Windows-safe GPU mutex |
| `scripts/compress_labels.py` | Build `labels_compressed/` from verbose labels |
| `scripts/organize_train_test.py` | Create `train_1/` and `testing/` splits |
| `scripts/check_eval_gate.py` | Automated pass/fail gate for queue runs |
| `scripts/check_compress_gate.py` | Verify labels fit 768-token budget |
| `scripts/week_queue.py` | Phase C 3-run chained pipeline |

### Synthetic Data Generation

| File | Purpose |
|------|---------|
| `synth_dataset/main.py` | Orchestrator (multiprocessing, category distribution, augment pass) |
| `synth_dataset/schemas.py` | Pydantic schemas for all chart types |
| `synth_dataset/generate_km.py` | KM curves via lifelines (Weibull survival, random arms/colors/styles) |
| `synth_dataset/generate_clinical.py` | Forest and waterfall plot generation |
| `synth_dataset/generate_anchor.py` | Bar, line, scatter, stacked bar, dual-axis combo |
| `synth_dataset/augment.py` | Albumentations adversarial augmentations |
| `synth_dataset/lexical_engine.py` | Treatment/axis label text generator with OCR noise |

### Real-World Data

| File | Purpose |
|------|---------|
| `real_dataset/scraper.py` | PMC article ID search via NCBI E-utilities |
| `real_dataset/extracter.py` | Download figure images (Selenium + requests) |
| `real_dataset/labeler.py` | Tkinter manual labeling UI |
| `real_dataset/reindexer.py` | Rename/organize accepted images |
| `real_dataset/config.py` | Shared paths and collection targets |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Project overview, architecture, usage |
| `Update.md` | Full chronological changelog (Phase B through Stage 2 v2.2) |
| `docs/STAGE2_DECISIONS.md` | Design decision log for Stage 2 (12 sections) |
| `docs/RETROSPECTIVE.md` | Project retrospective and roadmap |
| `docs/PROJECT_CONTEXT.md` | This file |
| `docs/V2_ARCHITECTURE.md` | V2 single-stage architecture plan |

### Batch Scripts (roadmap)

| File | Purpose |
|------|---------|
| `run_stage2_v2_2_train10k.bat` | Resume Stage 2 3000→10000 + checkpoint evals + E2E |
| `run_stage2_regen_train1.bat` | Regenerate tiles from `train_1/` → `stage2_train1/` |
| `run_stage2_train1_10k.bat` | Train 10k on train_1 tiles + eval + E2E |
| `run_realworld_pipeline.bat` | PMC scrape → label → real-world fine-tune → eval |

### Batch Scripts

| File | Purpose |
|------|---------|
| `run_week_queue.bat` | Phase C 3-run pipeline |
| `run_stage2_v2_1_sanity.bat` | Tile regen + 500-step train + sanity |
| `run_stage2_v2_2_overnight.bat` | 500 sanity -> gate -> 3000 train -> eval |
| `run_stage2_v2_sanity.bat` / `full.bat` / `eval_only.bat` | Legacy v2 scripts |

### Config

| File | Purpose |
|------|---------|
| `config/eval_gates.json` | Automated gate thresholds for queue runs |
| `config/phase_b_baseline_summary.json` | Phase B baseline numbers for comparison |

---

## 6. Key Decisions and Their Rationale

### Decision 1: SigLIP2 + MLP + Qwen Coder (architecture choice)

- **SigLIP2** chosen for strong patch-level features at 384x384 without requiring a text tower
- **MLP projector** (simplest possible) chosen for fast iteration; modern VLMs use cross-attention projectors
- **Qwen 2.5 Coder 1.5B** chosen because: (a) Coder variant has JSON-native training, (b) 1.5B fits in 8GB VRAM with 4-bit quantization, (c) the `Instruct` variant supports ChatML

### Decision 2: 5-crop spatial pooling (macro model)

Charts resized to 768x768, then 5 crops of 384x384: 1 global resize + 4 non-overlapping quadrants. Gives the LLM 3645 image tokens. Rationale: captures both overall layout and fine-grained tick marks.

### Decision 3: Phase C compact labels (truncation fix)

Phase B's `max_length=768` truncated multi-arm KM JSON (often 1500+ tokens). Phase C minified keys (`chart_type` -> `ct`, `coordinates` -> `p`, etc.) and capped to 10 step-corners per arm, 6 censors per arm. This fit labels within 768 tokens. The eval decompresses back to verbose schema for scoring.

### Decision 4: Two-stage pipeline (macro + tile)

Macro model outputs chart-level JSON (axes, arms, ~10 sparse coordinates per arm). Stage 2 "sniper" crops 384x384 tiles from the plot area and extracts dense per-arm coordinates. Rationale: macro resolution insufficient for precise coordinate reading.

**Retrospective flaw:** The numeric weakness (0.56) was from truncation, not resolution. The macro model with accurate 10-point step corners could potentially score 0.85+ on numeric. Stage 2 introduces arm disambiguation problems (no legend in tiles).

### Decision 5: Normalized local coordinates (Stage 2 v2.1)

Clinical coordinates (e.g., time=60.3, survival=0.42) are ill-posed for axisless tile crops. Switched to normalized `[0,1]` tile-local coordinates. `[0,0]` = top-left of crop, `[1,1]` = bottom-right. 3 decimal places. Middleware inverts using stored `_meta`.

### Decision 6: Flat interleaved arrays (Stage 2 v2.2)

Nested `[[x,y],[x,y],...]` caused bracket chaos -- the model lost nesting depth after ~40 pairs. Switched to flat `[x,y,x,y,...]`. The prefix `{"arm_id": "...", "points": [` means the next tokens are just `num, num, num, ...` with no inner brackets.

### Decision 7: Prefix-masked loss + forced prefix

Training includes the full JSON prefix `{"arm_id": "...", "points": [` in the sequence but masks its loss (labels = -100). The model only learns the coordinate stream onward. At inference, the same prefix is pre-filled (`--force-json-prefix`). This ensures JSON routing without wasting model capacity on learning to emit the boilerplate.

### Decision 8: Stage 2 uses Phase A projector only

Stage 2 initializes from the Phase A projector (vision-to-LLM mapping) but does NOT load Phase B/C LoRA weights. Rationale: macro LoRA encodes global layout priors (5-crop, full chart) that interfere with local tile reading (single crop, no context).

---

## 7. Evaluation Metrics (Complete Reference)

### 7.1 Macro Model: Weighted Composite (`evaluation/metrics.py`)

**KM charts -- component weights (sum to 1.0):**

| Component | Weight | Computation |
|-----------|--------|-------------|
| JSON valid | 0.10 | Binary: model output parses as JSON |
| Chart type | 0.05 | Normalized string match of `chart_type` field |
| Text | 0.20 | `SequenceMatcher` similarity on axis labels + arm treatment labels (threshold 0.85) |
| Structure | 0.15 | 50% arm count accuracy + 50% greedy one-to-one arm matching by label similarity |
| Numeric | 0.40 | Axis max relative error + per-arm step-curve RMSE (survival at GT event times). RMSE mapped to [0,1]: `max(0, 1 - rmse/0.25)`. Match tolerance: time +/-0.5, survival +/-0.05. |
| Censoring | 0.10 | Set overlap F1 on censoring tick times with +/-1.0 tolerance (greedy matching) |

**Other chart types:** 0.1 * JSON_valid + 0.1 * type_match + 0.8 * recursive_field_ratio (float tolerance: 5% relative, 0.001 absolute; text similarity >= 0.85).

### 7.2 Stage 2: Tile-Level Metrics (`eval_stage2.py`)

| Metric | Description |
|--------|-------------|
| `json_valid_strict_rate` | Full `{"arm_id","points","censors"}` parsed |
| `json_valid_rate` | Any parseable JSON (relaxed regex recovery) |
| `point_match_rate` | `matched_points / gt_points`. Match: nearest x within tolerance (0.05 normalized, 0.5 clinical) |
| `pooled_coordinate_rmse` | 2D RMSE across all matched point pairs (in tile-normalized space) |
| `mean_coordinate_rmse` | Average per-tile RMSE |
| `micro_censoring_f1` | Pooled TP/FP/FN across all tiles, F1 on censor x-times |
| `mean_censoring_f1` | Average per-tile F1 |

### 7.3 Gate System (`config/eval_gates.json`)

Automated pass/fail for Phase C queue runs:
- Absolute minimums on JSON valid rate, chart type accuracy, overall score, structure, numeric
- Relative-to-previous run minimums (e.g., >= 0.95 ratio on censoring)
- Baseline comparison (Phase B numbers in `config/phase_b_baseline_summary.json`)

### 7.4 End-to-End Pipeline (`eval_e2e.py`)

Stitches Stage 2 tile predictions (from eval JSONL or live inference) back to per-chart clinical KM JSON via `scripts/stitch_tiles.py`, then scores against verbose chart GT using `evaluation/metrics.py` (same weighted composite as macro eval).

| Metric | Description |
|--------|-------------|
| `mean_overall_score` | Full pipeline score on stitched chart JSON |
| Components | Same as §7.1 (JSON, type, text, structure, numeric, censoring) |

First measurement (2026-06-06, 12 charts, oracle tile boundaries from holdout labels):

| Metric | Score |
|--------|-------|
| Overall | **0.590** |
| Numeric | 0.653 |
| Structure | 0.382 |
| Censoring | 0.000 |

Results: `evaluation/results/e2e/latest_summary.json`

### 7.5 Stage 2 JSON Repair (`evaluation/parse_output.py`)

Post-inference cleanup before strict parse:

- Leading/trailing commas in flat arrays
- Orphan integer at array start (truncation artifact)
- Default missing `"censors": []`
- Close truncated brackets/braces

Rescore without GPU: `python eval_stage2.py --rescore-only <jsonl>`

---

## 8. Current Results

### Macro Model (Phase C Run 2)

Evaluated on 12 KM charts from `testing/` holdout:

| Metric | Score |
|--------|-------|
| JSON valid | 100% |
| Chart type | 100% |
| Text | ~0.63 |
| Structure | 1.000 |
| Numeric | 0.614 |
| Censoring | 0.152 |
| RMSE | 0.143 |
| **Overall** | **0.730** |

Checkpoint: `checkpoints/phase_c_run2_chatml/final`

### Stage 2 v2.2 (3000 steps)

Evaluated on 150 holdout tiles with `--force-json-prefix`:

| Metric | Raw | Parser-rescored |
|--------|-----|-----------------|
| JSON valid (relaxed) | 99.3% | 100% |
| JSON valid (strict) | 70.0% | **77.3%** |
| Point match rate | 53.2% | 54.1% |
| Pooled RMSE | 0.272 | 0.261 |
| Micro censoring F1 | 27.1% | 29.3% |
| Train loss @ 3000 | 0.4875 | — |

Checkpoint: `checkpoints/stage2_v2_1/final`  
Raw eval: `evaluation/results/stage2_v2_1_holdout/eval_20260606T044115Z_summary.json`  
Rescored: `evaluation/results/stage2_v2_1_holdout_rescored/latest_summary.json`

### End-to-End (first measurement, 2026-06-06)

12 charts via `eval_e2e.py` (stitched tile preds → clinical → macro metric):

| Metric | Score |
|--------|-------|
| **Overall** | **0.590** |
| JSON valid | 100% |
| Numeric | 0.653 |
| Structure | 0.382 |
| Censoring | 0.000 |

Results: `evaluation/results/e2e/latest_summary.json`

### In-Progress GPU Work (2026-06-06)

| Job | Status | Output |
|-----|--------|--------|
| Tile regen from `train_1/` (12.5k charts) | Running | `stage2_train1/`, `stage2_train1_holdout/` |
| Stage 2 train 3000→10000 steps | Running | `checkpoints/stage2_v2_1/` |

### Historical Progression

| Version | Strict JSON | Point Match | Key Change |
|---------|-------------|-------------|------------|
| Stage 2 v1 | 0% | 2.1% | max_length=512 truncation |
| Stage 2 v2 | 0% | 3.1% | Capped 40pts/10cens, max_length=1024 |
| v2 + prefix | 47% (15 tiles) | ~0.7% | Force JSON prefix, no retraining |
| Stage 2 v2.1 | 20% (500 steps) | n/a | Normalized coords, nested arrays |
| **Stage 2 v2.2** | **70%** | **53.2%** | Flat arrays, 3000 steps |

---

## 9. Checkpoint Map

All checkpoints are gitignored; stored on local disk.

| Role | Path | Status |
|------|------|--------|
| Phase A projector | `checkpoints/checkpoints_projector/projector_weights.pth` | Complete |
| Phase B final | `checkpoints/phase_b/final/` | Complete (step 6250) |
| Phase C Run 1 | `checkpoints/phase_c_run1_minified/final/` | Complete |
| **Phase C Run 2 (production)** | `checkpoints/phase_c_run2_chatml/final/` | **Active macro model** |
| Phase C Run 3 | `checkpoints/phase_c_run3_low_lr/final/` | Gate failed |
| Stage 2 v1 | `checkpoints/stage2/final/` | Obsolete |
| Stage 2 v2 | `checkpoints/stage2_v2/final/` | Obsolete |
| **Stage 2 v2.2** | `checkpoints/stage2_v2_1/final/` | **Active tile model** |

**External dataset root:** `C:\sem4\KMVision-1 Data\dataset\`

Contains: `train_1/`, `testing/`, `labels_compressed/`, `stage2/`, `stage2_v2/`, `stage2_v2_1/`, `stage2_v2_1_holdout/`, `stage2_train1/`, `stage2_train1_holdout/`, `split_manifest.json`

### Real-World Data (`real_dataset/`)

| Path | Content |
|------|---------|
| `images_km/` | 128 curated KM images (accepted) |
| `inbox/km/` | ~1001 raw scraped images awaiting review |
| `labels/km/` | Manual labels (0 as of 2026-06-06) |
| `status_report.json` | Generated by `scripts/realworld_status.py` |

Targets: 250 KM, 125 forest, 125 waterfall. Label with `python real_dataset/labeler.py`.

---

## 10. Known Issues and Bugs

1. **768-Token Guillotine (Phase B):** Training with `max_length=768` truncated multi-arm KM JSON, causing 0% censoring and low structure scores. Fixed in Phase C with minified keys but the 10-coordinate cap remains.

2. **Stage 2 v1 Truncation:** `max_length=512` while tile labels were 800+ tokens. Model learned headless fragments.

3. **Bracket Chaos:** Nested `[[x,y],...]` overwhelms autoregressive nesting depth for 40+ points. Fixed by flat `[x,y,x,y,...]` in v2.2.

4. **Tokenizer Entropy:** Clinical floats like `60.3207` split into many subword tokens. Fixed by 3-decimal normalized coords.

5. **Parser gaps (partially fixed):** Leading commas, missing censors, orphan integers — handled by `repair_stage2_json()` (strict JSON 70%→77%). Remaining ~23% need model/training fixes.

6. **E2E censoring is zero:** Stitched pipeline scores 0.0 on censoring component despite 27% tile-level censor F1 — stitch/eval mismatch or sparse censor recovery at chart level.

7. **Tile Arm Disambiguation:** Tiles lack legend context — model guesses which overlapping curve to trace. Hard ceiling on accuracy.

8. **Loss Not Converged:** 0.4875 at 3000 steps (~3.4 epochs). 10k-step training in progress.

9. **GPU Driver Recovery (Windows):** `Ctrl+Shift+Win+B` interrupted unattended training. Required separate eval-only reruns.

10. **Real-world gap:** 0 labeled charts; domain shift unmeasured until labeling + `train_realworld.py`.

---

## 11. Remaining Work

### Completed (D1, 2026-06-06)

1. Parser cleanup + rescore — `evaluation/parse_output.py`
2. `scripts/stitch_tiles.py` — inverse transform + dedupe
3. `eval_e2e.py` — first E2E benchmark (0.59 on 12 charts)
4. Docs: `docs/RETROSPECTIVE.md`, `docs/V2_ARCHITECTURE.md`

### In Progress (D2)

1. **Regenerate tiles from `train_1/`** — `run_stage2_regen_train1.bat` → `stage2_train1/`
2. **Train Stage 2 to 10,000 steps** — `run_stage2_v2_2_train10k.bat` (resume from 3000)
3. **Re-run E2E** after best checkpoint selected

### Next (D3)

1. **Label real-world KM** — `python real_dataset/labeler.py` (1129 in queue)
2. **Fine-tune macro** — `train_realworld.py` → `checkpoints/realworld_macro_km/`
3. **Real-world eval** — `eval_realworld.py`

### V2 Architecture (future)

See `docs/V2_ARCHITECTURE.md`: single-stage 7B, Perceiver projector, GRPO, curriculum. Targets: 0.85 E2E synthetic, 0.70 real-world.

---

## 12. Conventions and Gotchas

### Code Conventions

- All training scripts accept `--dataset_root` (default: `C:\sem4\KMVision-1 Data\dataset`)
- Checkpoint infra: `scripts/training_checkpoint.py` manages `step_NNNNNN/` dirs + `latest.json`
- GPU mutex: `scripts/training_lock.py` prevents concurrent training
- Windows paths with spaces must be quoted in batch scripts

### Important Constants

| Constant | Location | Value | Meaning |
|----------|----------|-------|---------|
| `MAX_COORDS_PER_ARM` | `evaluation/schema_compact.py` | 10 | Macro model coordinate cap |
| `MAX_CENSORS_PER_ARM` | `evaluation/schema_compact.py` | 6 | Macro model censor cap |
| `MAX_POINTS_PER_TILE` | `scripts/generate_stage2_tiles.py` | 40 | Stage 2 point cap |
| `MAX_CENSORS_PER_TILE` | `scripts/generate_stage2_tiles.py` | 10 | Stage 2 censor cap |
| `TILE_SIZE` | `scripts/generate_stage2_tiles.py` | 384 | Tile crop dimensions |
| `TILE_OVERLAP` | `scripts/generate_stage2_tiles.py` | 50 | Horizontal overlap between tiles |
| `COORD_DECIMALS` | `stage2_common.py` | 3 | Normalized coord precision |
| `NUM_IMAGE_TOKENS` | `train_stage2.py` | 729 | 27x27 patches from SigLIP |
| `POINT_X_MATCH_TOL_NORMALIZED` | `eval_stage2.py` | 0.05 | Match tolerance in [0,1] space |
| `CENSOR_X_TOL_NORMALIZED` | `eval_stage2.py` | 0.05 | Censor match tolerance |

### Dataset Layout on Disk

```
C:\sem4\KMVision-1 Data\dataset\
  train_1/
    images/km/    (100k training images)
    labels/km/    (verbose JSON labels)
  testing/
    images/km/    (238k holdout images)
    labels/km/
  labels_compressed/km/    (minified Phase C labels)
  stage2_v2_1/
    images/km/    (71k tile crops)
    labels/km/    (normalized local JSON labels)
  stage2_v2_1_holdout/
    images/km/    (3.8k holdout tiles)
    labels/km/
  split_manifest.json
```

### Git Rules

- **Never commit:** checkpoints, `*.pt`, `*.safetensors`, dataset dirs, `evaluation/results/`, `logs/`
- **Always commit:** Python code, docs, batch scripts, config JSON
- `.gitignore` is comprehensive -- see root `.gitignore`

### How to Resume Work

1. Read `Update.md` for chronological history (§10 = latest roadmap work)
2. Read `docs/STAGE2_DECISIONS.md` for design rationale
3. Read `docs/RETROSPECTIVE.md` for identified flaws and roadmap
4. Read `docs/V2_ARCHITECTURE.md` for V2 plan
5. Check `evaluation/results/stage2_v2_1_holdout_rescored/latest_summary.json` for tile metrics
6. Check `evaluation/results/e2e/latest_summary.json` for end-to-end score
7. Check `checkpoints/stage2_v2_1/latest.json` for training state
8. Check `scripts/realworld_status.py` for labeling progress
