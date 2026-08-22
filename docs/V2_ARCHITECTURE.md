# KMVision V2 Architecture Plan

Last updated: 2026-06-06. Companion to [RETROSPECTIVE.md](RETROSPECTIVE.md) Part 5.

## Goal

Achieve **95% end-to-end extraction accuracy** on synthetic KM charts and **80%+** on real-world publication figures without a fragile two-stage tile pipeline.

## V2 Design Principles

1. **Single-pass extraction** — model sees full chart (legend, axes, all arms) in one forward pass.
2. **Token budget headroom** — `max_length=2048`, no 10-point caps; step-aware subsampling only at eval if needed.
3. **Stronger spatial backbone** — larger LLM + better vision-language connector.
4. **Quality beyond imitation** — RL/DPO on coordinate RMSE after SFT.

---

## V2.1 Single-Stage Dense Extraction (Priority 1)

### Current pain (v1)

- Macro model: 0.73 overall, 10 coords/arm cap, 768-token history
- Stage 2 tiles: no legend, arm disambiguation ceiling, stitching complexity
- E2E (12 charts): **0.59 overall** with oracle tile boundaries

### V2 approach

```
Full chart (1024×1024 or native res)
    → Dynamic vision encoder (variable patch count)
    → Perceiver / cross-attention projector (compress to ~512–1024 tokens)
    → Qwen 7B (4-bit QLoRA)
    → Dense verbose or compact JSON (2048 tokens)
```

### Implementation sketch

| Component | v1 | v2 |
|-----------|----|----|
| Input | 768×768, 5×384 crops | 1024×1024 native or multi-scale |
| Vision tokens | 3645 (macro) / 729 (tile) | ~2000–5000 (configurable) |
| Projector | 2-layer MLP | Perceiver resampler (16–64 latents) or Q-Former |
| LLM | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-7B-Instruct |
| Output | Compact JSON, 10 pts/arm | Compact JSON, 30–50 pts/arm |
| Pipeline stages | Macro + tile + stitch | **One model** |

### Files to add (greenfield branch)

- `model_v2.py` — `ClinicalVLMv2` with pluggable projector
- `projectors/perceiver.py` — resampler module
- `train_v2_sft.py` — single-stage SFT on full charts
- Retire: `generate_stage2_tiles.py`, `stitch_tiles.py`, `eval_e2e.py` tile path

### Migration path

1. Train v2 SFT on existing `train_1/` 100k charts (same labels, no tile split).
2. Compare macro-only v1 (0.73) vs v2 single-pass on same 12-chart holdout.
3. If v2 ≥ 0.80 E2E, deprecate Stage 2.

---

## V2.2 Larger LLM Backbone (Priority 2)

| Model | VRAM (4-bit + LoRA r=64) | Expected gain |
|-------|--------------------------|---------------|
| Qwen2.5-Coder-1.5B | ~8 GB | Baseline |
| Qwen2.5-Coder-7B | ~14–16 GB | Better JSON rhythm, longer coord streams |
| Qwen2.5-Coder-14B | ~24+ GB | Diminishing returns unless multi-GPU |

**Decision:** Target **7B** on single 16 GB GPU; 14B only if cloud training (HF Jobs).

LoRA config unchanged (r=64, α=128). Increase `max_length` to 2048.

---

## V2.3 Better Projector (Priority 2)

### Problem with MLP projector

Maps every vision patch 1:1 to LLM space. With 3000+ patches, the LLM attention is diluted.

### Options (pick one for v2 MVP)

1. **Perceiver resampler** (LLaVA-NeXT style): 729→64 learned latents, cross-attn from latents to patches.
2. **Q-Former** (BLIP-2 style): BERT-like queries attend to frozen vision features.
3. **2D pooling + MLP**: Cheap baseline — pool 27×27 patches to 7×7 before MLP.

**Recommendation:** Perceiver with **32 latents** — 10× token reduction, proven in VLMs.

```python
# Sketch
class PerceiverProjector(nn.Module):
    def __init__(self, vision_dim=1152, llm_dim=1536, num_latents=32, num_layers=2):
        self.latents = nn.Parameter(torch.randn(num_latents, llm_dim))
        self.cross_attn_layers = nn.ModuleList([...])
    def forward(self, vision_tokens):  # (B, 729, 1152)
        return cross_attn(self.latents, vision_tokens)  # (B, 32, 1536)
```

Train projector warm-up (Phase A equivalent) for 1 epoch before LLM LoRA.

---

## V2.4 Reinforcement Learning on Coordinate Quality (Priority 3)

After SFT converges (~0.85 E2E):

1. **Reward:** `-coordinate_rmse` from `evaluation/metrics.py` (differentiable proxy: survival match at GT times).
2. **Method:** GRPO or DPO with chosen/rejected pairs from same chart (perturbed coordinates).
3. **Data:** 1k hard charts (4+ arms, dense censoring, overlapping curves).

Expected lift: 0.85 → 0.92 on synthetic.

---

## V2.5 Curriculum Learning (Priority 4)

| Stage | Charts | Criteria |
|-------|--------|----------|
| 1 | 2-arm, wide separation | Learn basic axis mapping |
| 2 | 3-arm, moderate overlap | Arm disambiguation |
| 3 | 4+ arm, dense censoring | Full difficulty |

Implement via `scripts/curriculum_sampler.py` filtering `train_1/` by metadata (arm count, censor density).

---

## Hardware & Timeline Estimate

| Milestone | GPU time | Deliverable |
|-----------|----------|-------------|
| v2 projector warm-up | 4 hr | `checkpoints/v2_phase_a/` |
| v2 SFT 7B, 50k steps | 24–48 hr | `checkpoints/v2_sft/final` |
| v2 eval vs v1 E2E | 1 hr | `evaluation/results/v2_e2e/` |
| v2 GRPO polish | 12 hr | `checkpoints/v2_rl/final` |

**Total:** ~1–2 weeks wall clock with one 16 GB GPU.

---

## Success Criteria

| Metric | v1 (current) | v2 target |
|--------|--------------|------------|
| E2E overall (12-chart holdout) | 0.59 | **≥ 0.85** |
| Macro numeric | 0.56 | **≥ 0.80** |
| Censoring F1 component | 0.16 | **≥ 0.50** |
| Real-world (after RW fine-tune) | unmeasured | **≥ 0.70** |

---

## Non-Goals for V2 MVP

- Multi-chart batch inference
- Forest/waterfall (KM only first)
- Real-time deployment / ONNX export
- Automatic PMC labeling (keep manual labeler)

---

## Open Questions

1. **SigLIP2 vs InternViT** for dynamic resolution — SigLIP2 is proven in v1; InternViT may scale better to 1024+.
2. **Compact vs verbose JSON at train time** — keep compact keys for token budget even at 2048?
3. **Freeze vision entirely?** — v1 froze SigLIP; v2 may benefit from last-layer unfreeze with small LR.

See [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for v1 baseline and [RETROSPECTIVE.md](RETROSPECTIVE.md) for rationale to pivot.
