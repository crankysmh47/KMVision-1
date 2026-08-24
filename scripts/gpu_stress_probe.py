"""Isolate the cumulative CUDA corruption on the RTX 5060 Ti.

Three modes, each in its OWN process (corruption is per-process):

  raw : pure torch CUDA matmul/alloc loop, no model, no bnb
  f16 : Qwen2.5-Coder-1.5B-Instruct loaded in bf16, generate() loop
  nf4 : same model loaded 4-bit NF4 double-quant via bitsandbytes

Exit codes: 0 = survived all iterations, 2 = failed at iteration N.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback

PROMPT_SEEDS = [
    "Extract the Kaplan-Meier curve coordinates as JSON:",
    "List the survival probabilities at months",
    "The treatment arm shows a hazard ratio of",
    "Censoring events occur at time points",
    "At risk table values for cohort",
]


def run_raw(iters: int, device) -> int:
    import torch

    sizes = [512, 729, 1024, 1536, 2048]
    state = {s: torch.randn(8, s, s, device=device) for s in sizes}
    for i in range(iters):
        t0 = time.time()
        s = sizes[i % len(sizes)]
        x = state[s] @ state[s].transpose(-1, -2)
        y = torch.softmax(x[:, :, : s // 2], dim=-1)
        _ = float(y.sum())
        del x, y
        torch.cuda.empty_cache()
        print(f"[raw] iter {i} ok ({time.time() - t0:.2f}s)", flush=True)
    return 0


def _load_llm(nf4: bool):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    if nf4:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
    model.to("cuda")
    model.eval()
    return model, tok


def run_llm(iters: int, nf4: bool) -> int:
    import torch

    model, tok = _load_llm(nf4)
    rng = random.Random(7)
    for i in range(iters):
        t0 = time.time()
        seed_txt = PROMPT_SEEDS[i % len(PROMPT_SEEDS)]
        n_fake = rng.randint(3, 12)
        fake_pts = ", ".join(f"{rng.random():.3f}" for _ in range(2 * n_fake))
        prompt = f"{seed_txt} [{fake_pts}]. Continue with the next points."
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            _ = tok.decode(out[0], skip_special_tokens=True)
        except Exception as exc:
            print(f"[{'nf4' if nf4 else 'f16'}] FAILED at gen {i}: "
                  f"{type(exc).__name__}", flush=True)
            traceback.print_exc()
            return 2
        torch.cuda.empty_cache()
        print(f"[{'nf4' if nf4 else 'f16'}] gen {i} ok ({time.time() - t0:.2f}s)",
              flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["raw", "f16", "nf4"], required=True)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args()

    import torch

    device = torch.device("cuda:0")
    print(f"mode={args.mode} iters={args.iters} torch={torch.__version__} "
          f"device={torch.cuda.get_device_name(0)}", flush=True)
    rc = run_raw(args.iters, device) if args.mode == "raw" \
        else run_llm(args.iters, args.mode == "nf4")
    print(f"[{args.mode}] finished rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
