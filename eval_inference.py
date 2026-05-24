"""
Run checkpoint inference on the testing/ holdout and score with evaluation.metrics.

Usage:
  python eval_inference.py
  python eval_inference.py --max-samples 50 --category km
  python eval_inference.py --checkpoint checkpoints/phase_b/final
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

from evaluation.data_index import collect_all_valid_samples
from evaluation.image_preprocess import pixel_values_from_path
from evaluation.metrics import aggregate_scores, score_extraction
from evaluation.parse_output import extract_json_from_text
from evaluation.schema_compact import decompress_chart
from model import ClinicalMicroVLM

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
DEFAULT_CHECKPOINT = "checkpoints/phase_b/final"
EXTRACTION_PROMPT = "\nExtract the underlying data from this clinical chart in strict JSON format.\n"
NUM_IMAGE_TOKENS = 3645


def decompress_json(minified: dict) -> dict:
    """
    Expand Phase C minified keys back to verbose schema for metric calculators.

    Handles KM t/s/c arrays (time_points, survival_probabilities, censoring_ticks)
    and legacy compact keys via evaluation.schema_compact.decompress_chart.
    """
    if not isinstance(minified, dict):
        raise TypeError("decompress_json expects a dict")

    if minified.get("chart_type") in (None, "") and "ct" in minified:
        expanded = decompress_chart(minified)
    elif "chart_type" in minified:
        return minified
    else:
        expanded = decompress_chart(minified)

    # Normalize verbose KM arms if model used t/s/c at top level (non-compact ct format).
    if expanded.get("chart_type") == "kaplan_meier":
        arms = []
        for arm in expanded.get("arms", []):
            if "coordinates" not in arm and "t" in arm and "s" in arm:
                arm = {
                    "treatment_label": arm.get("treatment_label", arm.get("id", "")),
                    "coordinates": [
                        [float(t), float(s)] for t, s in zip(arm["t"], arm["s"])
                    ],
                    "censoring_ticks": arm.get("censoring_ticks", arm.get("c", [])),
                }
            arms.append(arm)
        expanded["arms"] = arms
    return expanded


def resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return path
    alt = os.path.join("checkpoints/phase_b", path)
    if os.path.isdir(alt):
        return alt
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_model(checkpoint_dir: str, device: torch.device) -> ClinicalMicroVLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = ClinicalMicroVLM(bnb_config=bnb_config)
    model.vision_encoder.requires_grad_(False)
    model.vision_encoder = model.vision_encoder.to(device)
    model.projector = model.projector.to(device)

    projector_path = os.path.join(checkpoint_dir, "projector_weights.pth")
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Missing projector weights: {projector_path}")
    model.projector.load_state_dict(torch.load(projector_path, map_location=device))

    model.llm = PeftModel.from_pretrained(model.llm, checkpoint_dir)
    model.llm.config.use_cache = True
    model.eval()
    return model


@torch.inference_mode()
def build_inputs_embeds(
    model: ClinicalMicroVLM,
    pixel_values: torch.Tensor,
    input_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (inputs_embeds, attention_mask) for the visual prefix + prompt."""
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    input_ids = input_ids.to(device)

    b_val, num_crops, _, _, _ = pixel_values.shape
    flat = pixel_values.view(b_val * num_crops, *pixel_values.shape[2:])
    with torch.no_grad():
        vision_out = model.vision_encoder(pixel_values=flat)
        image_embeds = vision_out.last_hidden_state
    projected = model.projector(image_embeds)
    _, num_patches, embed_dim = projected.shape
    projected = projected.view(b_val, num_crops * num_patches, embed_dim)

    text_embeds = model.llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)

    image_mask = torch.ones((b_val, projected.shape[1]), dtype=torch.long, device=device)
    text_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([image_mask, text_mask], dim=1)
    return inputs_embeds, attention_mask


@torch.inference_mode()
def generate_extraction(
    model: ClinicalMicroVLM,
    processor,
    tokenizer,
    image_path: str,
    device: torch.device,
    *,
    max_new_tokens: int = 768,
) -> str:
    pixel_values = pixel_values_from_path(image_path, processor)
    encoded = tokenizer(EXTRACTION_PROMPT, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded.input_ids

    inputs_embeds, attention_mask = build_inputs_embeds(model, pixel_values, input_ids, device)

    output_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # With inputs_embeds, generate() returns only *new* token ids — do not strip prompt_len.
    if input_ids is not None and output_ids.shape[1] > input_ids.shape[1] + 8:
        # Full sequence returned (prompt + completion): keep completion only.
        new_ids = output_ids[:, input_ids.shape[1] :]
    else:
        new_ids = output_ids

    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()


def collect_test_samples(
    testing_root: str,
    *,
    category: Optional[str],
    max_samples: int,
    seed: int,
) -> List[Tuple[str, str]]:
    img_root = os.path.join(testing_root, "images")
    lbl_root = os.path.join(testing_root, "labels")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(
            f"Testing folder not found: {img_root}. Run scripts/organize_train_test.py first."
        )

    by_cat = collect_all_valid_samples(img_root, lbl_root)
    if category:
        if category not in by_cat:
            raise ValueError(f"Category '{category}' not in testing set. Available: {list(by_cat)}")
        pool = by_cat[category]
    else:
        pool = [p for samples in by_cat.values() for p in samples]

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VLM checkpoint on testing holdout.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--category", default=None, help="Limit to one category (e.g. km).")
    parser.add_argument("--max-samples", type=int, default=40, help="Max test images to evaluate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="KM JSON is long; training used 768 total seq len so outputs may still truncate.",
    )
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    testing_root = os.path.join(args.dataset_root, "testing")
    checkpoint_dir = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Checkpoint: {checkpoint_dir}")

    samples = collect_test_samples(
        testing_root,
        category=args.category,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    if not samples:
        print("No test samples found.")
        sys.exit(1)
    print(f"Evaluating {len(samples)} samples from {testing_root}")

    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(checkpoint_dir, device)

    os.makedirs(args.output_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jsonl_path = os.path.join(args.output_dir, f"eval_{stamp}.jsonl")
    summary_path = os.path.join(args.output_dir, f"eval_{stamp}_summary.json")

    scores = []
    records = []

    for img_path, label_path in tqdm(samples, desc="inference"):
        with open(label_path, encoding="utf-8", errors="replace") as f:
            ground_truth = json.load(f)

        try:
            prediction_raw = generate_extraction(
                model,
                processor,
                tokenizer,
                img_path,
                device,
                max_new_tokens=args.max_new_tokens,
            )
            err = None
        except Exception as e:
            prediction_raw = ""
            err = str(e)

        try:
            parsed, _ = extract_json_from_text(prediction_raw)
            if parsed is None:
                raise ValueError("could not parse prediction")
            expanded = decompress_json(parsed)
            score = score_extraction(ground_truth, expanded)
        except (ValueError, TypeError, KeyError):
            score = score_extraction(ground_truth, prediction_raw)
        scores.append(score)

        records.append(
            {
                "image": img_path,
                "label": label_path,
                "category": os.path.basename(os.path.dirname(label_path)),
                "inference_error": err,
                "prediction_raw": prediction_raw[:4000],
                "score": score.to_dict(),
            }
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = aggregate_scores(scores)
    summary["checkpoint"] = checkpoint_dir
    summary["eval_samples"] = len(samples)
    summary["category_filter"] = args.category
    summary["timestamp_utc"] = stamp

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    latest_path = os.path.join(args.output_dir, "latest_summary.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Evaluation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nPer-sample results: {jsonl_path}")
    print(f"Summary JSON:       {summary_path}")


if __name__ == "__main__":
    main()
