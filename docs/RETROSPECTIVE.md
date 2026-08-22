# KMVision-1: Full Project Retrospective and Roadmap

Generated 2026-06-06 from a comprehensive review of all code, metrics, and decisions.

## Current State at a Glance

**Macro model (Phase C Run 2):** 0.73 overall on 12-chart synthetic holdout  
**Stage 2 v2.2 (3000 steps):** 70% strict JSON, 53.2% point match, 0.272 RMSE on 150 holdout tiles  
**End-to-end pipeline:** Not built  
**Real-world data:** ~5-10% collected, unlabeled

---

## Part 1: Flaws in Judgment (Honest Audit)

### Flaw 1 (Critical): The two-stage pipeline may not have been necessary

The single biggest architectural bet was splitting extraction into macro + tile. The reasoning was:

- Macro numeric score was 0.56 (the main drag on 0.73 overall)
- Diagnosis: "the model needs higher resolution to read precise coordinates"
- Solution: crop 384x384 tiles and run a second model on each

**What was missed:** The macro model's numeric weakness was primarily caused by the **768-token guillotine** truncating labels, not by vision resolution. Phase C fixed the truncation with minified keys, but **kept the 10-point-per-arm cap** (`evaluation/schema_compact.py`, line 25: `MAX_COORDS_PER_ARM = 10`). The macro model was never given a fair chance to output dense coordinates with an adequate token budget.

A KM step function with 10 accurately-placed corners can score very high on numeric evaluation (the metric interpolates survival at GT event times). The bottleneck was coordinate **accuracy**, not coordinate **density** — and that's a training maturity issue, not a resolution issue.

**Impact:** Months of work on Stage 2 (v1 -> v2 -> v2.1 -> v2.2), each iteration discovering a new formatting issue, while the macro model sits at 0.73 with a plausible path to 0.85+ through more training data alone.

### Flaw 2 (Significant): Stage 2 tiles lack critical context

Each 384x384 tile crop has:
- No legend (which curve is which color?)
- No axis labels or tick marks (usually outside the plot area)
- Multiple overlapping curves with no way to distinguish them except color

The model is asked to trace "Drug X" through a tile where it sees 3-4 colored lines but has no legend context telling it which color maps to which arm. The 53% point match likely hits a hard ceiling here — the model is essentially guessing which curve to follow based on the arm name string matching training priors.

### Flaw 3 (Significant): No end-to-end metric exists

The 95% benchmark target has **never been measured**. The macro model reports 0.73 on its own metric. Stage 2 reports 53% point match on its own metric. These don't compose. Without `stitch_tiles.py` and an end-to-end eval, we're optimizing proxy metrics that may not correlate with the actual goal.

### Flaw 4 (Moderate): Training loss hasn't converged

Stage 2 v2.2 finished at loss **0.4875** after 3000 steps. That's not converged — a well-fitted model should approach 0.1-0.2. With 71k tiles and grad_accum=8, 3000 global steps only covers ~3.4 passes through the data. The model hasn't seen enough examples.

### Flaw 5 (Moderate): Generous match tolerances inflate numbers

Stage 2 uses `POINT_X_MATCH_TOL_NORMALIZED = 0.05` — a point matches if x is within 5% of tile width (~19 pixels). The 53% point match at this tolerance is actually underwhelming. At tighter tolerance (0.02), it would likely drop to ~30%.

### Flaw 6 (Minor): Stage 2 trains on `testing/` charts

Tiles are generated from `testing/` (the macro model's holdout), not from `train_1/` (the training set). This isn't wrong per se (Stage 2 is a separate model), but it means 71k tiles come from only 11.4k of the 238k available charts. Using `train_1/` (100k charts) would give ~6x more training data.

### Flaw 7 (Minor): Iteration pattern — format changes without enough training

Each Stage 2 variant (v1, v2, v2.1, v2.2) changed the label format and ran 500-3000 steps, then diagnosed the next format issue. But format issues (JSON structure) and vision issues (coordinate accuracy) are separate — we kept conflating them. The flat format was the right call, but it should have been the first format, not the fourth.

---

## Part 2: What We Can Improve Now (No Architecture Changes)

### Improvement A: Free JSON uplift via parser cleanup

The 70% strict JSON rate is artificially low. Looking at sanity failures, common patterns are:
- Leading comma: `"points": [, 0.299, ...]` — trivially fixable with regex
- Truncated output missing `"censors"` key — recoverable by defaulting
- Stray integers at array start: `"points": [9, 0.144, ...]` — first orphan value is a truncation artifact

This is a zero-GPU fix in `evaluation/parse_output.py`. Expected uplift: **70% to 85-92% strict JSON**.

### Improvement B: More training steps (low risk, high expected value)

Loss at 3000 steps: 0.4875. The model has only seen ~3.4 epochs. Running to **6000-10000 steps** (7-12 epochs) should push loss toward 0.2-0.3 and improve point match from 53% to potentially 65-75%.

### Improvement C: Train on `train_1/` charts (6x more data)

Currently using 11.4k charts from `testing/`. Regenerating tiles from `train_1/` (100k charts) would give ~400-600k tiles. Even capped at 50k, the diversity improvement matters.

### Improvement D: Build the end-to-end eval

Without this, we cannot measure progress toward 95%. This is the **most important missing piece** — not another training run.

---

## Part 3: Recommended Next Steps (Priority Order)

### Phase D1: Measure what matters (1-2 days, no GPU)

1. **Parser cleanup** in `evaluation/parse_output.py` — fix leading/trailing commas, missing "censors" default, orphan first values
2. **Rescore existing JSONL** — measure true strict JSON rate without re-inference
3. **Draft `stitch_tiles.py`** — normalized flat arrays + `_meta` inverse transform + overlap deduplication
4. **End-to-end eval script** — Run 2 macro output + synthetic GT tile boundaries + Stage 2 inference + stitch + score vs original GT using `evaluation/metrics.py`
5. Get the **first real end-to-end number**

### Phase D2: Push training ceiling (2-3 days, GPU)

1. **Regenerate tiles from `train_1/`** — 100k source charts, `--source train_1`
2. **Train Stage 2 to 10,000 steps** — let loss converge
3. **Eval at checkpoints** (2500, 5000, 7500, 10000) — find the plateau
4. **Re-run end-to-end eval** with best checkpoint

### Phase D3: Real-world data (1-2 weeks, depends on labeling)

1. Complete PMC image collection (currently ~100 images, ~5-10%)
2. Label images using the existing Tkinter tool (`real_dataset/labeler.py`)
3. Fine-tune the macro model (Phase C checkpoint) on real-world data
4. Fine-tune Stage 2 on real-world tile crops
5. Evaluate on held-out real-world test set

---

## Part 4: Honest Assessment — Can v1 Hit 95%?

**On synthetic data:** Plausibly 80-85% end-to-end with the improvements above. The macro model at 0.73 plus better Stage 2 plus stitching could get there. But 95% is very hard because:
- Censoring is fundamentally difficult (small tick marks, low visual salience)
- Tiles without legend context have a hard ceiling for arm disambiguation
- The LLM at 1.5B is capacity-limited for spatial reasoning

**On real-world data:** Unlikely to hit 95% without real-world fine-tuning. Domain gap (synthetic matplotlib -> real publication figures) is significant.

**Recommendation:** Set the v1 target at **85% on synthetic, 70% on real-world** after fine-tuning. Reserve 95% for v2.

---

## Part 5: V2 Architecture Improvements

### V2.1: Single-stage dense extraction (biggest bang for buck)

Skip tiling entirely. Instead:
- Use a **dynamic resolution** vision encoder (SigLIP with variable patch count, or InternViT)
- Input the **full chart at native resolution** (e.g., 1024x1024 -> ~5000 vision tokens)
- Output dense coordinates directly with `max_length=2048`
- The model sees the full chart with legend, axes, and all arms in one pass
- Eliminates: tile generation, stitching, arm disambiguation without legend, coordinate space transforms

### V2.2: Larger LLM backbone

Qwen 2.5 Coder **7B** (or 14B with 4-bit) gives 4-9x more capacity for spatial reasoning and longer coordinate sequences. The memory cost is manageable with QLoRA.

### V2.3: Better projector

Replace the 2-layer MLP with a **Perceiver resampler** or **cross-attention** projector. This compresses vision tokens more intelligently and gives the LLM a cleaner signal. Modern VLMs (LLaVA-1.6, Qwen-VL) all use more sophisticated projectors.

### V2.4: Reinforcement learning on coordinate quality

After SFT, run **DPO or GRPO** using coordinate RMSE as the reward signal. The model learns to self-correct coordinate drift rather than just imitating training labels. This is how frontier models push from 80% to 95%.

### V2.5: Curriculum learning

Train on easy charts (2 arms, clear separation) first, then gradually introduce hard charts (4+ arms, overlapping curves, dense censoring). The current training is random — curriculum would help the model build spatial reasoning incrementally.

---

## Summary Decision Matrix

| Action | Effort | Expected Impact | Do Now? |
|--------|--------|-----------------|---------|
| Parser regex cleanup | 1 hour | 70% -> 85-92% strict JSON | Yes |
| End-to-end eval script | 1 day | Unlocks real benchmark | Yes |
| `stitch_tiles.py` | 0.5 day | Required for E2E eval | Yes |
| More training (10k steps) | 8-12 hr GPU | 53% -> 65-75% point match | Yes |
| Train on `train_1/` tiles | 2 hr regen + GPU | Better diversity | Yes |
| Macro model with 20-30 pts/arm | 1 day + GPU | Alternative to tiling | Consider |
| Real-world data pipeline | 1-2 weeks | Domain gap closure | Next sprint |
| V2 architecture | Weeks | 95% target | Future |
