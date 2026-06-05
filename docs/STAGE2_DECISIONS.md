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

**Holdout:** `--holdout_fraction` (default 0.05) writes to `dataset/stage2_holdout/` for `eval_stage2.py`; remainder to `dataset/stage2/`.

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

## 9. Explicit non-goals (MVP)

- No Stage 1 inference in the generator (uses dense GT for bbox calibration quality on synthetic data).
- No path-guided windows from Run 2 predictions yet.
- No multi-GPU / queue integration.
- Scripts are **write-only** in this PR — user requested no execution.
