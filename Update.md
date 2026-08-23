### Current Updates

The model has completed training as of last week, but I couldnt find the time to test its performance on any data, synthetic or real. Real world data collection is about 5-10% complete only, with almost 100 Km curve unlabelled images
being the all of it. The testing phase is owing its delays mostly due to the lack of a good evaluation metric for the json data for now. I plan to search for a good approximation online, and if I cant find anything, then
Ill have to come up with something myself. If the model succeeds with synthetic data, and struggles with the real world, we will begin fine tuning on the real world dataset, after we complete it of course. If the results 
dont hold up, even on the synthetic data, we have two options, continue further training on the remaining images that include almost 80% of the 500k or so we initially geenrated. It might turn out to be a long grind.

The second option is that we come up with another strategy for how our loss is calculated during backpropagation, so we might give up on the current trajectory completely and pivot to Reinforcement learning, or we might have to
come up with an entirely new loss function. I know all of these options sound crazy, the latest one perhaps most of all, but nevertheless they are still options fro consideration given how the model might not be generalizing well on the current ADAM based optimizer.

I hope we clear at least the synthetic testing with grace so we have some ground to stand up for our next steps.

---

## Updates since the note above (May–June 2026)

This section logs everything done after the original paragraph: evaluation infrastructure, Phase B diagnosis, Phase C three-run pipeline, production checkpoint selection, and Stage 2 (Sniper tile) work. Dataset and checkpoint files live outside git (see `.gitignore`); paths below are on disk.

### 1. Evaluation system (Phase B holdout)

We built a full eval stack because there was no scoring code in the repo.

**New / updated code**

| File | Role |
|------|------|
| `evaluation/metrics.py` | Multi-tier KM scoring: JSON validity, chart type, text, structure, numeric RMSE, censoring F1 |
| `evaluation/parse_output.py` | Robust JSON extraction from model text (fences, prose) |
| `evaluation/data_index.py` | Index image/label pairs under `testing/` |
| `evaluation/image_preprocess.py` | 5-crop pixel tensor builder for macro charts |
| `eval_inference.py` | Run checkpoint on holdout → JSONL + summary |
| `evaluation/rescore_jsonl.py` | Re-score saved JSONL after parser fixes |

**Phase B baseline (12 KM charts, `testing/` holdout)**

Checkpoint: `checkpoints/phase_b/final` (archived copy in `archives/phase_b_baseline/`).

| Metric | Score |
|--------|-------|
| JSON valid | 100% |
| Chart type | 100% |
| **Overall** | **0.538** |
| Structure | 0.410 |
| Numeric | 0.570 |
| Censoring | **0.000** |
| Coordinate RMSE | 0.314 |

Canonical numbers: `config/phase_b_baseline_summary.json`.

**Key diagnosis — the 768-token guillotine**

Phase B trained with `max_length=768` and `truncation=True`. Multi-arm KM JSON often exceeds 1,500 tokens. Everything after the cut (extra arms, `censoring_ticks`, closing braces) never appeared in the loss. That explains structure ~0.41, censoring 0.0, and truncated inference outputs — not a vision failure (JSON valid and chart type were already 100%).

---

### 2. Phase C: compact labels + three-run week queue

**Label compression**

| File | Role |
|------|------|
| `evaluation/schema_compact.py` | Minified keys (`ct`, `a`, `p`, `c`), KM step-aware subsampling, token budget helpers |
| `scripts/compress_labels.py` | Build `labels_compressed/` from `train_1/labels/` |
| `scripts/check_compress_gate.py` | Reject run if >5% of sample labels exceed 768 tokens |
| `train_phase_c.py` | Phase C training (compact JSON targets, ChatML optional) |
| `scripts/week_queue.py` | Chained compress → train → eval → gate |
| `scripts/check_eval_gate.py` | Automated pass/fail vs baseline / previous run |
| `config/eval_gates.json` | Gate thresholds |

**Infrastructure fixes during queue runs**

- `scripts/training_lock.py` — Windows-safe GPU mutex (`logs/week_queue/training.lock`)
- `scripts/training_checkpoint.py` — Ordered `step_NNNNNN/`, `latest.json`, resume with CPU-safe RNG reload
- `evaluation/parse_output.py`, `evaluation/rescore_jsonl.py` — Relaxed pair parsing / rescoring for truncated outputs

**Three-run pipeline** (`run_week_queue.bat`, logs in `logs/week_queue/`)

| Run | Config | Steps | Overall | Structure | Numeric | Censoring | RMSE | Gate |
|-----|--------|-------|---------|-----------|---------|-----------|------|------|
| **Run 1** minified JSON | `phase_c_run1_minified` | ~1609* | **0.710** | 1.0 | 0.560 | 0.156 | 0.174 | **PASS** |
| **Run 2** + ChatML | `phase_c_run2_chatml` | 2000 | **0.730** | 1.0 | 0.614 | 0.152 | 0.143 | **PASS** |
| **Run 3** + low LR polish (1e-5) | `phase_c_run3_low_lr` | ~1588* | 0.717 | 1.0 | 0.587 | 0.138 | 0.152 | **FAIL** |

\*Effective steps capped by compact dataset size (~25.7k samples ÷ grad_accum 16 ≈ 1600).

**Run 3 gate failure:** `mean_censoring_score` ratio vs Run 2 was **0.909** (need ≥ 0.95). Censoring dropped from 0.152 → 0.138. Queue stopped after Run 3 eval; no promotion to production.

**Production macro model (Stage 1 skeleton):** `checkpoints/phase_c_run2_chatml/final` — best overall, all gates passed (+36% vs Phase B overall).

Eval summaries: `evaluation/results/run1_minified/`, `run2_chatml/`, `run3_low_lr/` (`latest_summary.json`, `gate_runN.json`).

---

### 3. Stage 2 “Sniper” — tile micro-extractor

Design log: `docs/STAGE2_DECISIONS.md`. Code: `scripts/generate_stage2_tiles.py`, `train_stage2.py`, `eval_stage2.py`, `scripts/stage2_sanity_check.py`.

**Architecture**

- Single **384×384** tile → **729** image tokens (not 5-crop / 3645)
- Fresh LoRA + **Phase A projector only** (no Phase B/C LoRA)
- ChatML prompt per arm: extract `{"arm_id","points","censors"}` for that tile
- Data from `testing/` KM charts **disjoint from Phase B/C `train_1`** (12k charts → ~71k train tiles, ~3.8k holdout)

**Stage 2 v1** (`dataset/stage2/`, `checkpoints/stage2/final`)

- Training: 3000 steps, completed 2026-06-04
- Bug: trained with `max_length=512` while labels were often **800+ tokens** → supervision was mid-`points` array; model learned headless coordinate fragments

**Holdout eval v1** (`evaluation/results/stage2_holdout/`, 150 tiles, `checkpoints/stage2/final`)

| Metric | Value |
|--------|-------|
| Strict full JSON | **0%** |
| Relaxed coordinate parse | 100% |
| Point time-match rate | **2.1%** |
| Pooled RMSE (matched) | 0.170 |
| Censoring F1 | **0.0** |

---

### 4. Stage 2 v2 — truncation fix (label caps)

**Planned fix (user spec)**

1. Cap tiles: **40 points**, **10 censors** (step-aware subsample + even cap)
2. `max_length=1024` in training
3. Fresh init — do **not** resume from `checkpoints/stage2/final`
4. **500-step sanity** first; only run 3000 if strict JSON passes

**What was implemented**

- Regenerated tiles → `dataset/stage2_v2/`, holdout `stage2_v2_holdout/` (same 71,451 / 3,778 counts; all labels ≤ 919 tokens)
- `checkpoints/stage2_v2/`: 500-step sanity (0/5 strict JSON) then **full 500→3000 resume** (deviation from gate — JSON had not passed at 500)
- Training **completed** 3000 steps 2026-06-05 (`checkpoints/stage2_v2/final`)
- GPU driver recovery (Ctrl+Shift+Win+B) interrupted the unattended pipeline **during post-train eval**; holdout eval was **not** finished in that run

**500-step sanity result (before full resume):** 0/5 tiles emitted strict `{"arm_id","points","censors"}`; previews were still coordinate fragments (one tile showed closing `]}}`).

**Eval status (v2):** Completed via `run_stage2_v2_eval_only.bat` (2026-06-05). Results: `evaluation/results/stage2_v2_holdout/latest_summary.json`.

| Metric | v2 holdout (150 tiles) |
|--------|------------------------|
| Strict full JSON | **0%** |
| Relaxed parse | 100% |
| Point time-match rate | **3.1%** (92 / 3013) |
| Pooled RMSE (matched) | 0.183 |
| Censoring F1 | **0.0** |
| Sanity (10 tiles) | **0/10** strict JSON |

Slight improvement vs v1 (~2.1% point match) but still no valid JSON envelope or censors. Original unattended pipeline (`run_stage2_v2_full.bat`) finished **training** at step 3000; post-train eval was interrupted by GPU/display reset and completed in a separate eval-only run.

**Orchestration scripts**

| Script | Purpose |
|--------|---------|
| `run_stage2_v2_sanity.bat` | Regen tiles + 500-step train + sanity |
| `run_stage2_v2_full.bat` | Resume 500→3000 + sanity + 150-tile eval |
| `run_stage2_v2_eval_only.bat` | Eval only (after training complete) |

---

### 5. Checkpoint map (on disk, gitignored)

| Role | Path |
|------|------|
| Phase A projector | `checkpoints/checkpoints_projector/projector_weights.pth` |
| Phase B final | `checkpoints/phase_b/final/` |
| **Phase C production (macro)** | `checkpoints/phase_c_run2_chatml/final/` |
| Phase C Run 1 / 3 | `phase_c_run1_minified/final/`, `phase_c_run3_low_lr/final/` |
| Stage 2 v1 | `checkpoints/stage2/final/` |
| Stage 2 v2 | `checkpoints/stage2_v2/final/` |

**External dataset root:** `C:\sem4\KMVision-1 Data\dataset\` (`train_1`, `testing`, `labels_compressed`, `stage2`, `stage2_v2`, `split_manifest.json`).

---

### 6. Current model status (summary)

| Layer | Status | Notes |
|-------|--------|-------|
| Macro KM extraction | **Production-ready for synthetic** | Run 2 @ 0.730 overall; use for Stage 1 bbox / chart JSON |
| Censoring (macro) | Weak but non-zero | ~0.15 F1 component; Run 3 polish regressed |
| Tile micro-extractor v1 | **Not usable** | Truncation-trained; ~2% point match, 0% censors |
| Tile micro-extractor v2 | **Eval complete — still weak** | Capped labels; 3.1% point match, 0% strict JSON/censors |

**Open decisions for next session**

1. v2 holdout eval is in `evaluation/results/stage2_v2_holdout/latest_summary.json` — strict JSON still 0%
2. If strict JSON still 0%: consider longer sanity-only run, true base init (no Phase A projector), or label format simplification
3. Do **not** resume from v2 weights for a third attempt if JSON structure is still broken — restart fresh per original v2 spec
4. Real-world PMC collection still ~5–10% (unchanged from top of this file)

---

### 7. Git / commit readiness

Repo code and docs are ready to commit; large artifacts are gitignored (`.gitignore` updated for checkpoints, `*.pt`, tile dataset dir names, `evaluation/results/`, `logs/`, HF cache). Run `git status` and commit when ready — no auto-commit from agent runs.

---

### 8. Prefix forcing experiment + Stage 2 v2.1 (normalized local space)

**Prefix forcing (`--force-json-prefix` in `eval_stage2.py`, `scripts/stage2_sanity_check.py`)**

Inference pre-fills the assistant through `{"arm_id": "<id>", "points": [` before generation. No weight changes.

| Condition | Strict JSON (15 tiles) |
|-----------|------------------------|
| No prefix | 0% |
| With prefix | **47%** (7/15) |

This is a major diagnostic win: **0% → 47% without changing a single weight** proves the model *can* speak JSON when anchored, and that the text-routing engine was failing to initialize from a blank assistant — not a vision failure.

Remaining failures show **bracket chaos**: malformed pairs like `[11, 0.422979)` mixing `[` and `(`, or losing comma/bracket rhythm mid-stream.

**Root cause — tokenizer entropy on clinical floats**

Language models struggle with arbitrary floating-point numbers. A value like `60.3207` is not one token; the Qwen tokenizer may split it into `[60]`, `[.]`, `[32]`, `[07]`. When the model must emit ~40 coordinate pairs of these high-entropy, multi-token floats, the autoregressive engine loses syntax context — it forgets whether it just opened `[` or needs `,` because attention is dominated by random digit subwords.

**Shared code:** `stage2_common.py` — prompt, `force_json_assistant_prefix()`, `mask_len_through_json_prefix()`.

**Training alignment (v2.1):** `train_stage2.py` keeps the full JSON (including `{"arm_id": "...", "points": [`) in the training sequence but **masks loss** on user ChatML + that forced prefix; the model only learns the coordinate stream + closing `], "censors": [...]}`. Eval always uses `--force-json-prefix` to match.

**Stage 2 v2.1 — normalized local labels**

| Change | Detail |
|--------|--------|
| Coordinate system | `[x_norm, y_norm]` in `[0.000, 1.000]` per 384×384 tile (3 decimals), not clinical time/survival |
| Generator | `scripts/generate_stage2_tiles.py` — 3-step transform below; step-aware cap in **clinical space first** |
| Middleware | `scripts/stage2_coordinate_transform.py` — inverse normalized → clinical using `_meta` |
| Data paths | `stage2_v2_1/`, `stage2_v2_1_holdout/` |
| Checkpoints | `checkpoints/stage2_v2_1/` (fresh LoRA; **do not** resume `stage2_v2/`) |
| Eval | `--force-json-prefix`; x-tolerance 0.05 + 2D RMSE in tile space for normalized labels |
| Orchestration | `run_stage2_v2_1_sanity.bat` |

**Label transform pipeline (`generate_stage2_tiles.py`)**

1. **Clinical → global pixel** on 768×768 canvas (matplotlib axes metadata):  
   `px = plot.x0 + (t / x_max) * plot.width`, `py = plot.y1 - (s / y_max) * plot.height`
2. **Global pixel → local tile pixel:**  
   `x_local = px - tile.x0`, `y_local = py - tile.y0`
3. **Local tile pixel → normalized:**  
   `x_norm = round(x_local / 384, 3)` (clamped to `[0, 1]`); same for y. Image y increases downward.

Corner semantics: `[0.000, 0.000]` = top-left of crop; `[1.000, 1.000]` = bottom-right.

Each label stores `_meta.coordinate_space = "normalized_local"`, `tile_origin`, `plot_bbox`, `axis_max` for middleware inverse.

**Why v2.1 fixes both accuracy and formatting**

| Problem | Fix |
|---------|-----|
| Vision (clinical coords on axisless crop) | Model predicts position *within the tile* — tick at horizontal center → `0.500` |
| Bracket chaos (high-entropy floats) | Short 3-decimal tokens like `0.500` are low-entropy and rhythmic for the tokenizer |
| Token budget (`max_length=1024`) | Fewer subwords per coordinate → full 40-point + 10-censor JSON fits comfortably |

Full design: `docs/STAGE2_DECISIONS.md` §10.

**Status (2026-06-05 run):**

| Step | Result |
|------|--------|
| Tile regen | **Done** — 71,451 train tiles → `stage2_v2_1/` |
| Bugfix | `mask_len_through_json_prefix` — Coder tokenizer returns flat `input_ids`; fixed via `_encode_token_len()` |
| 500-step train | **Done** — loss 0.4827 @ step 500 → `checkpoints/stage2_v2_1/final` (~76 min) |
| Prefix sanity (10 tiles) | **FAILED** — 2/10 strict JSON (20%); normalized coords visible but flat `[x,y], [x,y]` bracket chaos persists |

Log: `logs/stage2_v2_1_train_sanity_20260605_233242.log`

---

### 9. Stage 2 v2.2 — flat interleaved coordinates + overnight run

**Diagnosis:** v2.1 nested `[[x,y],...]` still caused bracket chaos (`[.10], [0.145, 0.112]`) — the model cannot maintain inner/outer array depth for 40 points.

**Fix (`stage2_common.py` v2.2):**

| Before | After |
|--------|-------|
| `"points": [[0.145, 0.112], [0.146, 0.124]]` | `"points": [0.145, 0.112, 0.146, 0.124]` |
| Nested `[` per point | Single rhythm: `num, num, num, num, ...` |

- Training flattens at load time (`stage2_target_json`); **no tile regen**
- Prompt updated to forbid nested pairs
- Eval/middleware accept nested (GT) or flat (predictions)

**Overnight pipeline:** `run_stage2_v2_2_overnight.bat`

1. 500-step sanity (fresh LoRA, flat targets)
2. Sanity gate (≥80% strict JSON on 10 tiles)
3. If pass → 3000-step full train
4. Holdout eval (150 tiles, `--force-json-prefix`)

**95% benchmark — honest morning expectations**

| Metric | v2.1 @ 500 sanity | Expected @ 3000 (v2.2) | Path to 95% |
|--------|-------------------|------------------------|-------------|
| Strict JSON | 20% | **70–95%** (flat removes nesting failure mode) | Prefix forcing + flat format |
| Point match (150 holdout) | ~0% usable | **25–55%** (vision task now fair + parseable) | More steps, path-guided crops from Run 2 |
| Normalized RMSE (matched pts) | n/a | **0.05–0.12** tile space | Fine-tune tolerance; more train data |
| Censor F1 | 0% | **10–30%** | Separate censor head or post-process |

95% overall on tile extraction is **not realistic in one night** — that target applies to the full macro pipeline. This run should deliver **parseable JSON at scale** and **first meaningful point-match rates** (10–50× above v2’s 3.1%), which is the prerequisite for iterating toward 95%.

**Updated status table**

| Layer | Status |
|-------|--------|
| Tile v2 (clinical, capped) | Eval done — 3.1% match, 0% strict JSON without prefix |
| Tile v2 + prefix forcing | 47% strict JSON on 15-tile spot check — routing diagnosed |
| Tile v2.1 (normalized nested) | 500-step sanity FAILED — 2/10 strict JSON |
| Tile v2.2 (normalized flat_xy) | **3000-step eval done** — 70% strict JSON, **53.2% point match**, RMSE 0.27 |

**v2.2 full run results (150 holdout tiles, 2026-06-06):**

| Metric | v2 clinical | v2.2 @ 3000 |
|--------|-------------|-------------|
| Strict JSON | 0% | **70%** |
| Point match | 3.1% | **53.2%** |
| Pooled RMSE (tile space) | — | **0.272** |
| Censor F1 (micro) | 0% | **27.1%** |
| Train loss @ 3000 | — | 0.4875 |

Log: `logs/stage2_v2_2_full_20260606_020401.log`  
Eval: `evaluation/results/stage2_v2_1_holdout/eval_20260606T044115Z_summary.json`

---

### 10. Roadmap implementation (2026-06-06) — D1 complete, D2/D3 in flight

Implemented the retrospective roadmap (`docs/RETROSPECTIVE.md`). Planning docs: `docs/PROJECT_CONTEXT.md`, `docs/V2_ARCHITECTURE.md`.

#### Phase D1 — Measure what matters (complete, no GPU)

**Parser cleanup (`evaluation/parse_output.py`)**

- `repair_stage2_json()` / `extract_stage2_json()` — fixes leading/trailing commas, orphan integers at array start, missing `"censors"`, truncated bracket closure
- Wired into `eval_stage2.py` strict parse path
- Tests: `evaluation/test_parse_output.py` (6 passing)

**Rescore (same 150-tile JSONL, no re-inference)**

| Metric | Raw @ 3000 | After parser rescored |
|--------|------------|------------------------|
| Strict JSON | 70.0% | **77.3%** |
| Point match | 53.2% | 54.1% |
| Pooled RMSE | 0.272 | 0.261 |

Rescore: `python eval_stage2.py --rescore-only evaluation/results/stage2_v2_1_holdout/eval_20260606T044115Z.jsonl`  
Summary: `evaluation/results/stage2_v2_1_holdout_rescored/latest_summary.json`

**Tile stitching (`scripts/stitch_tiles.py`)**

- Normalized flat_xy → clinical via `_meta` + `stage2_coordinate_transform.py`
- Overlap dedupe by clinical time (0.25 month tolerance)
- Groups Stage 2 eval JSONL by source chart; emits verbose KM JSON per chart
- Tests: `evaluation/test_stitch_tiles.py` (4 passing)

**End-to-end eval (`eval_e2e.py`) — first benchmark number**

Pipeline: holdout tile predictions → stitch per chart → score vs verbose GT (`evaluation/metrics.py`).

| Metric | Score (12 charts) |
|--------|-------------------|
| **Overall** | **0.590** |
| JSON valid | 100% |
| Structure | 0.382 |
| Numeric | 0.653 |
| Censoring | **0.000** |
| RMSE | 0.391 |

Results: `evaluation/results/e2e/latest_summary.json`  
Usage: `python eval_e2e.py --stage2-jsonl evaluation/results/stage2_v2_1_holdout/eval_20260606T044115Z.jsonl --max-charts 12`

#### Phase D2 — Push training ceiling (in progress)

**Bugfix:** `generate_stage2_tiles.py` — `--source train_1` no longer excludes all train stems (was filtering entire Phase B/C pool).

**Tile regen from `train_1/`** — started 2026-06-06

- Source: 12,500 KM charts from `train_1/images/km`
- Output: `stage2_train1/`, `stage2_train1_holdout/` on external dataset root
- Script: `run_stage2_regen_train1.bat`

**Stage 2 continue 3000 → 10,000 steps** — started 2026-06-06

- Resumes from `checkpoints/stage2_v2_1/final` (`--auto_resume`)
- Checkpoint evals at 2500/5000/7500/10000
- Script: `run_stage2_v2_2_train10k.bat`
- Post-train on `train_1` tiles: `run_stage2_train1_10k.bat`

#### Phase D3 — Real-world pipeline (infra complete; labeling pending)

| File | Role |
|------|------|
| `train_realworld.py` | Fine-tune Phase C macro on labeled `real_dataset/` KM charts |
| `eval_realworld.py` | Score macro checkpoint on real-world labeled holdout |
| `run_realworld_pipeline.bat` | Scrape → label → fine-tune → eval |
| `scripts/realworld_status.py` | Collection/labeling progress report |

Status (2026-06-06): 128 curated KM images, 1129 unlabeled in queue, **0 labeled**. Targets: 250 KM / 125 forest / 125 wf.

#### V2 architecture plan

`docs/V2_ARCHITECTURE.md` — single-stage 7B, Perceiver projector, GRPO on coordinate RMSE, curriculum; target 0.85 E2E synthetic / 0.70 real-world.

#### Updated status table

| Layer | Status |
|-------|--------|
| Macro (Phase C Run 2) | 0.730 overall on 12-chart holdout |
| Stage 2 v2.2 @ 3000 | 77.3% strict JSON (rescored), 53% point match |
| **E2E (stitch)** | **0.590 overall** on 12 charts (first measured) |
| train_1 tile regen | In progress → `stage2_train1/` |
| Stage 2 @ 10k steps | In progress (resume from step 3000) |
| Real-world | Infra ready; needs manual labeling |

---

### 11. Phase 0 — Stitcher repair: the first honest E2E numbers (2026-08-22)

**Root cause found and fixed.** The 0.590 E2E number measured **no model
output at all**. In `stitch_arm_tiles`, `prediction_raw` was parsed only
inside the `if not meta:` branch — but `group_eval_jsonl_by_chart`
pre-attaches `_meta` to every eval record, so the branch never ran and
every arm was stitched with EMPTY coordinates. There was no macro fallback;
predictions were silently *discarded*. Verified directly: feeding two
different prediction streams produced byte-identical stitched output.
That explains both the frozen 0.590 and the 0.000 censoring component.
**The 0.590 figure is retired and must not be used as a baseline.**

**Fixes (all under strict mode — failures are loud, never silent):**

| File | Change |
|------|--------|
| `scripts/stitch_tiles.py` | Rewritten: unconditional prediction parsing (`prediction`/`parsed` dict or `prediction_raw` via repair parser); `StitchError` raised in strict mode on missing/unparseable/out-of-bounds predictions; provenance per tile (`prediction_source`, `tile_id`, counts) + chart-level `_meta.stitch` (version `stitch_v2_provenance_strict`); bounds assertions in normalized and clinical space; censors travel through the same inverse transform as points in their own list |
| `eval_e2e.py` | Catches `StitchError` per chart → recorded as `stitch_failed` (never silently dropped); summary reports `scored_charts` / `error_charts`; exit code 2 if any chart failed to stitch |
| `evaluation/parse_output.py` | New repairs, all verified by tests: bare decimals `.623`→`0.623`; leading-zero integers `01`→`1`; stray blank string `]," "censors":`→`],"censors":`; restart pathology (concatenated second `{"arm_id":…}` object truncated to the first); LIFO bracket repair `_close_open_containers` replacing the count-based closer that appended `]` after a trailing `}` |
| `evaluation/test_stitch_tiles.py`, `evaluation/test_parse_output.py` | 25 tests total (was 10): regression tests for the silent-discard bug (different predictions must change output), strict-raise behavior, provenance, censor inverse transform, and every new parser repair |

**Parser effect on existing eval JSONLs (no re-inference):**

| JSONL | Unparseable before | After |
|-------|-------------------:|------:|
| Stage 2 3k (`eval_20260606T044115Z`) | 34 / 150 | **0 / 150** |
| Stage 2 10k (`eval_20260607T012712Z`) | 15 / 150 | **0 / 150** |

**First honest E2E numbers** (12 charts, same oracle tile boundaries;
results in `evaluation/results/e2e_repaired_3k/` and `e2e_repaired_10k/`):

| Metric | Broken 0.590 (retired) | 3k tiles | 10k tiles |
|--------|------------------------|---------:|----------:|
| Overall | 0.590 | **0.6127** | **0.6121** |
| Numeric | 0.653 | 0.703 | 0.701 |
| Structure | 0.382 | 0.382 | 0.382 |
| Censoring | 0.000 | 0.026 | 0.028 |
| RMSE | 0.391 | 0.270 | 0.278 |
| Text | 0.608 | 0.608 | 0.608 |
| Charts scored | 12/12* | 12/12 | 12/12 |

\* the "broken" run scored 12/12 only because it scored the GT skeleton.

**Interpretation (honest):**

1. Predictions now genuinely contribute: numeric 0.653 → ~0.70, RMSE
   0.391 → ~0.27, censoring 0.000 → 0.027 (nonzero at last).
2. **3k ≈ 10k at the E2E level** (0.6127 vs 0.6121). The 10k tile-level
   gains (strict JSON 77%→91%, censor F1 0.29→0.34) do **not** translate
   to end-to-end improvement on these 12 charts. Structure (0.38) and
   text (0.61) are inherited from the GT skeleton / arm naming and
   dominate the residual gap.
3. 12 charts remain statistically insufficient — this number has error
   bars wider than the 3k-vs-10k difference. The 500+ chart validation
   set (plan Phase 0 / §3.5) is the next prerequisite before any
   Stage-2-vs-macro decision.
4. Censoring is still ~0.03 end-to-end despite 0.34 tile-level F1 —
   the remaining censor loss happens at stitch time (tile-level censors
   are spatially sparse per tile and dedupe/aggregation discards most);
   needs its own investigation, separate from the stitcher bug.

---

### 12. Phase 0 validation benchmark: the frozen 500-chart split (2026-08-23/24)

**Split** (`scripts/make_validation_split.py`, seed 42): 500 KM charts sampled
from `testing/`, disjoint from every prior holdout (incl. stage2 holdouts).
3,122 oracle-boundary tiles generated for them → `stage2_validation/`.
Frozen: no training, tuning, or architecture selection on this split.
Runner: `scripts/run_validation_benchmark.py` (arms: `macro`, `e2e_oracle`,
`macro_baseline`; per-chart JSONL + mean/median/stdev/P10/P90 summaries).

**Hardware incident (new, blocking long runs).** The RTX 5060 Ti
(Blackwell SM 12.0, driver 591.86, torch 2.6.0+cu124, bitsandbytes 0.45.1)
develops **cumulative in-process CUDA corruption**: after ~16–24 generations
one process starts raising `AcceleratorError` (misaligned address / invalid
argument) and *every* later generation in that process fails; a fresh process
runs the exact same charts perfectly. Not model or code — deterministic per
process age. Retroactively explains the 2026-06-06 Ctrl+Shift+Win+B recovery.
Workaround for eval: batch-restart loop (`--stop-after N` per process,
incremental partial JSONL, resume skips scored charts). This is an eval-only
mitigation — **the driver/torch/bnb stack must be fixed before Phase 1
training** (checkpoint-corruption risk).

**Macro arm (Phase C Run 2) — COMPLETE: 500/500 charts clean, 0 errors.**

| Stat | overall | numeric | censoring | coord RMSE |
|------|--------:|--------:|----------:|-----------:|
| mean | **0.7050** | 0.5892 | 0.1378 | 0.1569 |
| median | 0.7238 | 0.6013 | 0.1382 | 0.1568 |
| stdev | 0.1272 | 0.1547 | 0.0370 | 0.0428 |
| P10 / P90 | 0.6514 / 0.7828 | 0.4317 / 0.7525 | 0.1048 / 0.1767 | 0.1017 / 0.2090 |

95% CI on mean overall ≈ [0.694, 0.716]. Strict JSON valid: 97.4%.
This supersedes all previous macro numbers (0.730 was n=12).

**Stage-2 E2E-oracle arm — INCOMPLETE: 14/500 clean so far.**

- Runner loads `checkpoints/stage2_v2_1/final` (the v2.2 @10k lineage — same
  family as the §11 honest numbers), tiles from `stage2_validation/`.
- Same GPU degradation hits this arm; additionally the first batch loop ran
  with `--no-resume` plus a completion counter that counted duplicate rows,
  burning passes on charts 1–20. Loop killed and relaunched fixed
  (resume on, unique-count gate, AcceleratorError → exit 3 → restart).
- Preliminary paired result on the 14 charts scored by both arms:
  e2e 0.7518 vs macro 0.7199 (macro wins 3/14). **This is not a reversal of
  §11**: n=14 manifest-order survivors, paired-diff SE ≈ 0.04, and it
  contradicts the 12-chart result more than it confirms anything. It does
  mean the "loses by ~0.12 decisively" framing needs the full-500 error bars
  before being treated as final.

**Standing interpretation (honest).** The §11 evidence stands: both tile
checkpoints lost to macro by ~0.12 on their own 12-chart oracle comparison,
3k ≈ 10k, and tile-level gains did not translate E2E. Engineering-wise the
single-stage v2 branch subsumes this pipeline regardless of whether v2.2 ties
macro here (segmentation cost + ~6 generations/chart vs 1). But per plan Rule 1
the deprecation record will cite the full-500 paired numbers once accumulated;
they are running now (`logs/val500_e2e_batch4.log`).

**Gate 0 checklist status (plan §3):** stitcher strict ✓ · censor inverse
transform ✓ · loud assertions ✓ · provenance ✓ · 3k/10k re-evaluated ✓ ·
validation ≥500 ✓ (this split) · **frozen synthetic test set ✗ (still owed —
val500 is the development validation)** · real dev/frozen split exists but
unlabeled ⚠ · per-chart metrics ✓ · no silent failures ✓.
