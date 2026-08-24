"""Reproduce the cumulative GPU corruption using the REAL Stage-2 pipeline.

Iterates real tiles from stage2_validation through eval_stage2's actual
loaders and generation path until failure or --tiles exhausted.

  --quant nf4   : exact production path (BitsAndBytesConfig NF4 double-quant)
  --quant none  : identical model/graph but LLM in bf16 (no bitsandbytes)

Exit codes: 0 = survived all tiles, 2 = CUDA/generation failure at tile N.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_ROOT = Path(r"C:\sem4\KMVision-1 Data\dataset")
CKPT = "checkpoints/stage2_v2_1/final"


def load_model_quant(device):
    from eval_stage2 import load_model

    return load_model(CKPT, device)


def load_model_no_quant(device, attn_impl="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM
    from model import ClinicalMicroVLM
    from peft import PeftModel

    if attn_impl == "sdpa":
        model = ClinicalMicroVLM(bnb_config=None)
    else:
        model = ClinicalMicroVLM.__new__(ClinicalMicroVLM)
        import torch.nn as nn
        from transformers import AutoModel

        nn.Module.__init__(model)
        print("Loading Vision Encoder (SigLIP 2)...", flush=True)
        model.vision_encoder = AutoModel.from_pretrained(
            "google/siglip2-so400m-patch14-384",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).vision_model
        print("Loading LLM Decoder (Qwen 2.5 Coder 1.5B)...", flush=True)
        model.llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        print("Initializing Projector...", flush=True)
        model.projector = nn.Sequential(
            nn.Linear(1152, 1536, dtype=torch.bfloat16),
            nn.GELU(),
            nn.Linear(1536, 1536, dtype=torch.bfloat16),
        )
    model.vision_encoder.requires_grad_(False)
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)
    model.projector.load_state_dict(
        torch.load(Path(CKPT) / "projector_weights.pth", map_location=device)
    )
    model.llm = model.llm.to(device)
    model.llm = PeftModel.from_pretrained(model.llm, CKPT)
    model.llm.config.use_cache = True
    model.eval()
    return model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quant", choices=["nf4", "none"], required=True)
    p.add_argument("--tiles", type=int, default=30)
    p.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    args = p.parse_args()

    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from eval_stage2 import generate_tile_json

    img_dir = DATASET_ROOT / "stage2_validation" / "images" / "km"
    lbl_dir = DATASET_ROOT / "stage2_validation" / "labels" / "km"
    tile_pairs = sorted(lbl_dir.glob("*.json"))[: args.tiles]

    device = torch.device("cuda:0")
    print(f"quant={args.quant} tiles={len(tile_pairs)} torch={torch.__version__}",
          flush=True)

    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    if args.quant == "nf4":
        model = load_model_quant(device)
    else:
        model = load_model_no_quant(device, attn_impl=args.attn)
    print(f"model loaded in {time.time() - t0:.1f}s", flush=True)

    for i, lf in enumerate(tile_pairs):
        meta = json.loads(lf.read_text(encoding="utf-8"))
        arm_id = meta.get("arm_id", "unknown")
        img = img_dir / f"{lf.stem}.png"
        if not img.is_file():
            continue
        t1 = time.time()
        try:
            out = generate_tile_json(model, processor, tok, img, arm_id,
                                     device, force_json_prefix=True)
            n_chars = len(out)
        except Exception as exc:
            print(f"[{args.quant}] FAILED at tile {i} "
                  f"({time.time() - t1:.1f}s into gen): {type(exc).__name__}: {exc}",
                  flush=True)
            traceback.print_exc()
            return 2
        torch.cuda.empty_cache()
        print(f"[{args.quant}] tile {i} ok ({time.time() - t1:.2f}s, {n_chars} chars)",
              flush=True)
    print(f"[{args.quant}] SURVIVED all {len(tile_pairs)} tiles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
