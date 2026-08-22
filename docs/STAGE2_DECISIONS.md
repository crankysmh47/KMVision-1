# Stage 2 MVP — Design Decision Log

This document records the reasoning behind the Stage 2 "Sniper" infrastructure.
Implementation files: `scripts/generate_stage2_tiles.py`, `train_stage2.py`, `eval_stage2.py`.

---

## 1. Plot bounding box (no pixel metadata in synthetic JSON)

**Observation:** `synth_dataset/generate_km.py` saves only clinical schema (`axes`, `arms`); it does **not** persist matplotlib axes pixel bounds.

**Decision:** Estimate `plot_bbox` from the raster image after normalizing to **768×768**, using a non-background mask (luminance below a white threshold), then shrink the mask by a small margin to avoid frame/legend bleed.

**Fallback:** If the mask is empty or too small, use fixed fractional insets tuned for `bbox_inches='tight'` KM figures (`left=0.14`, `top=0.12`, `right=0.96`, `bottom=0.22`).

**Why not re-render:** Offline tile generation must run on existing PNGs without calling matplotlib again.

---

## 2. Clinical ↔ pixel mapping

**Decision:** Affine map from axis limits in the dense JSON:

- `x_px = x0 + (t / x_max) * (x1 - x0)`
- `y_px = y1 - (s / y_max) * (y1 - y0)` (image y grows downward; survival 1.0 at top)

**Rationale:** Matches matplotlib survival plots (origin bottom-left, y upward in data space).

**Filtering rule:** A point `(t, s)` is kept for a tile if its projected pixel lies inside the tile rectangle `[tx0, tx0+384) × [ty0, ty0+384)`.

---

## 3. Horizontal sliding only

**User spec:** Slide 384×384 windows left→right with **50 px overlap** across the plot bbox.

**Decision:** Vertically center a 384 px band inside the plot bbox (`ty0 = y0 + max(0, (h-384)//2)`). Do not vertical-slide in MVP (reduces combinatorial explosion; KM curves are wide).

**Skip:** Charts whose plot bbox is narrower than 128 px or shorter than 128 px after normalization.

---

## 4. Censoring labels as `[t, s]` pairs

**User spec:** `"censors": [[t, s], ...]` in clinical coordinates.

**Decision:** For each censoring tick time `t_c`, set `s = survival_at_time(coordinates, t_c)` using a right-continuous step function (standard KM convention). Filter by the same pixel-in-tile test as curve points.

**Why:** Vertical tick marks sit on the curve; Stage 2 learns localized (time, survival) pairs, not isolated times.

---

## 5. One tile file per (window × arm)

**Decision:** If an arm has **≥ 2** points inside the tile, emit one PNG + one JSON. Same spatial window can yield multiple training pairs (different arms).

**Naming:** `{source_stem}_x{tx0}_arm{idx}_{slug}.png` / `.json`

**Metadata in label (non-training):** `_meta` records `source_chart`, `tile_origin`, `time_window`, `plot_bbox` for debugging and future reassembly — excluded from loss target string.

---

## 6. Input data paths

**Decision:** Default `--source testing` → `testing/images/km` + `testing/labels/km`. This holdout was **never** used in Phase B/C (`train_1/` is the 100k training pool).

**Exclusion:** Also load stems from `train_1/labels/km` and `split_manifest.json` `train` entries; drop any chart whose stem appears in the Phase B/C set (belt-and-suspenders if paths are ever rescanned).

**Do not use:** `train_1/` for Stage 2 tile generation (same charts as Phase C `labels_compressed` source). Prior pilot tiles from `train_1` should be regenerated with `--clear_output`.

**Volume:** Default `--max_charts 12000` from ~238k unused KM testing charts.

**Holdout:** `--holdout_fraction` (default 0.05) writes to `dataset/stage2_holdout/` for `eval_stage2.py`; remainder to `dataset/stage2/`. (v2.1 uses `stage2_v2_1_holdout/`.)

---

## 7. `train_stage2.py` vision pipeline

**Decision:** Keep `ClinicalMicroVLM` unchanged; feed **one** 384×384 crop as `pixel_values` shape `(B, 1, C, H, W)` → **729** image tokens (not 3645).

**Why not edit `model.py`:** `num_crops=1` is already valid in the forward pass; avoids duplicating the architecture module.

**Init:** Fresh **LoRA** on Qwen + trainable projector; load **Phase A projector weights** only (`checkpoints/checkpoints_projector/projector_weights.pth`). **Do not** load Phase B/C LoRA (global layout prior hurts local tick reading).

**Prompt:** ChatML via `build_training_text` with dynamic arm string (same semantics as user template).

**No 5% classify router:** Stage 2 is extraction-only.

---

## 8. `eval_stage2.py` metrics

**Decision:** Only two MVP metrics (no Phase B tier system):

1. **Coordinate RMSE** — match GT/pred points by nearest time (≤ 0.5 month tolerance); RMSE on survival `s` over matched pairs; aggregate macro-mean across tiles.
2. **Censoring F1** — treat censor locations as sets of `t` (from `[t,s]` pairs); match with **1.0** clinical time tolerance (per user); report precision, recall, F1.

**Inference:** Single-crop preprocessor (384 tile as-is, no resize). ChatML prompt mirrors training.

---

## 9. Stage 2 v2 — label caps (truncation fix)

**Problem:** v1 trained with `max_length=512` while tile labels were often 800+ tokens. The model learned headless coordinate fragments, not full JSON.

**Decision:**

- **Cap per tile:** max **40** points, **10** censors (after pixel filtering).
- **Point subsample:** `subsample_km_coordinates` (step-aware corners), then `_cap_evenly` if still > 40.
- **Censor subsample:** sort by time, `_cap_evenly` to 10.
- **Training:** `max_length=1024`; fresh LoRA + Phase A projector only — **do not** resume from `checkpoints/stage2/`.
- **Paths:** `dataset/stage2_v2/`, `checkpoints/stage2_v2/`.
- **Sanity:** 500 steps, then `scripts/stage2_sanity_check.py` (strict JSON with `arm_id`, `points`, `censors`).

---

## 10. Stage 2 v2.1 — normalized local coordinate space

**Problem (v2 clinical labels):** A 384×384 crop has no visible axes. Asking the model to emit global `(time, survival)` from pixels alone is ill-posed. The ~3% point match on v2 was largely curve memorization, not measurement.

**Problem (bracket chaos / tokenizer entropy):** High-entropy floats like `60.3207` split into many subword tokens (e.g. Qwen: `[60]`, `[.]`, `[32]`, `[07]`). With ~40 coordinate pairs, the autoregressive stream is dominated by arbitrary digit subwords; the model loses track of JSON syntax (`[`, `,`, `]`) and emits malformed pairs like `[11, 0.422979)`. Prefix forcing (§8 in `Update.md`) raised strict JSON from 0% → 47% without retraining — proving routing failure, not vision — but bracket chaos persists on the remaining failures until coordinates are simplified.

**Decision:** Train and eval in **normalized local tile space**:

| Corner | Normalized `[x, y]` |
|--------|---------------------|
| Top-left of tile | `[0.000, 0.000]` |
| Bottom-right of tile | `[1.000, 1.000]` |

All coordinates rounded to **3 decimal places** (`0.000`–`1.000`).

**Label pipeline (`generate_stage2_tiles.py`):**

1. **Clinical → global pixel** on 768×768 canvas:  
   `px = plot.x0 + (t / x_max) * plot.width`  
   `py = plot.y1 - (s / y_max) * plot.height`
2. **Global pixel → local tile pixel:**  
   `x_local = px - tile.x0`, `y_local = py - tile.y0`
3. **Local → normalized:**  
   `x_norm = round(x_local / 384, 3)`, same for y (image y down)

Step-aware KM subsampling runs in **clinical space before** conversion (preserves step corners).

**Middleware (production handoff):** `scripts/stage2_coordinate_transform.py` inverts normalized → clinical using `_meta.tile_origin`, `_meta.plot_bbox`, `_meta.axis_max`.

**Training (`train_stage2.py` v2.1):**

- Data: `dataset/stage2_v2_1/`, checkpoint: `checkpoints/stage2_v2_1/`
- Prompt mentions normalized `[x,y]` in `[0,1]` (`stage2_common.py`)
- **Prefix-masked loss:** mask user ChatML + forced prefix `{"arm_id": "<id>", "points": [`; loss only on coordinate stream + closing JSON
- Fresh LoRA + Phase A projector; `max_length=1024`

**Inference (`eval_stage2.py`):**

- Use `--force-json-prefix` (pre-fill through `"points": [`)
- Compare predictions to normalized labels directly (RMSE in tile space)

**Paths:**

| Artifact | Location |
|----------|----------|
| Train tiles | `{dataset}/stage2_v2_1/` |
| Holdout | `{dataset}/stage2_v2_1_holdout/` |
| Weights | `checkpoints/stage2_v2_1/final` |

**Orchestration:** `run_stage2_v2_1_sanity.bat` (regen → 500-step train → prefix sanity check)

**Benefits (three-way fix):**

1. **Vision accuracy** — SigLIP sees a tick at tile center and predicts `0.500`; no global clinical guess from an axisless crop.
2. **Syntax stability** — `0.500` is a short, predictable token pattern; bracket/comma rhythm stays intact.
3. **Token budget** — 3-decimal normalized coords use fewer subwords; 40 points + 10 censors fit in `max_length=1024`.

---

## 11. Stage 2 v2.2 — flat interleaved coordinates (syntax fix)

**Problem (v2.1 nested JSON):** Even with normalized 3-decimal values, training target used nested `"points": [[x,y], [x,y], ...]`. The model opened `"points": [` then emitted sibling arrays — bracket chaos like `[.10], [0.145, 0.112]` (20% strict JSON after 500 steps).

**Root cause:** Each point requires an inner `[` `]` pair inside the outer array. The autoregressive model loses nesting depth after ~40 pairs.

**Decision:** Flat interleaved lists — no nested coordinate arrays:

```json
{"arm_id": "Drug X", "points": [0.145, 0.112, 0.146, 0.124], "censors": [0.01, 0.652]}
```

| Aspect | Detail |
|--------|--------|
| Training target | `stage2_common.stage2_target_json()` flattens disk labels at load time |
| Disk labels | Unchanged nested `[[x,y],...]` in `stage2_v2_1/` (no regen) |
| Prefix | Still `{"arm_id": "<id>", "points": [` — next tokens are plain numbers |
| Eval | `_as_points()` / `coords_to_pairs()` accept nested or flat |
| Checkpoint | Same path `checkpoints/stage2_v2_1/` — **fresh train** after format change |

**Orchestration:** `run_stage2_v2_2_overnight.bat` (500 sanity → gate → 3000 train → 150-tile eval)

---

## 12. Explicit non-goals (MVP)

- No Stage 1 inference in the generator (uses dense GT for bbox calibration quality on synthetic data).
- No path-guided windows from Run 2 predictions yet.
- No multi-GPU / queue integration.
- End-to-end 2-pass eval (Run 2 macro → crop → Stage 2) not built yet.
