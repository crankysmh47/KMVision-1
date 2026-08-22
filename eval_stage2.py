"""
Evaluate Stage 2 tile micro-extractor (coordinate RMSE + censoring F1).

Mirrors eval_inference.py output layout: JSONL per run, timestamped summary,
and latest_summary.json in the results folder.

Usage:
  python eval_stage2.py --checkpoint checkpoints/stage2/final
  python eval_stage2.py --max-samples 200 --seed 0
  python eval_stage2.py --max-samples 0   # all holdout tiles (slow)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.parse_output import extract_json_from_text, extract_stage2_json
from model import ClinicalMicroVLM
from stage2_common import (
    as_point_tuples,
    chat_prefix_for_user,
    force_json_assistant_prefix,
    stage2_user_prompt,
)
from train_stage2 import NUM_IMAGE_TOKENS

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
DEFAULT_CHECKPOINT = "checkpoints/stage2_v2_1/final"
DEFAULT_OUTPUT_DIR = "evaluation/results/stage2_v2_1_holdout"
POINT_TIME_MATCH_TOL = 0.5
POINT_X_MATCH_TOL_NORMALIZED = 0.05
CENSOR_TIME_TOL = 1.0
CENSOR_X_TOL_NORMALIZED = 0.05


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        return path
    alt = os.path.join("checkpoints/stage2_v2_1", path)
    if os.path.isdir(alt):
        return alt
    alt = os.path.join("checkpoints/stage2_v2", path)
    if os.path.isdir(alt):
        return alt
    alt = os.path.join("checkpoints/stage2", path)
    if os.path.isdir(alt):
        return alt
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def load_tile_pairs(image_dir: Path, label_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for label_path in sorted(label_dir.glob("*.json")):
        stem = label_path.stem
        for ext in (".png", ".jpg"):
            img = image_dir / f"{stem}{ext}"
            if img.is_file():
                pairs.append((img, label_path))
                break
    return pairs


def sample_pairs(
    pairs: List[Tuple[Path, Path]],
    *,
    max_samples: int,
    seed: int,
) -> List[Tuple[Path, Path]]:
    if max_samples <= 0 or max_samples >= len(pairs):
        return pairs
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    return [pairs[i] for i in idx[:max_samples]]


_PAIR_RE = re.compile(
    r"\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?",
)


def parse_stage2_output(text: str, *, arm_id: str = "") -> Optional[dict]:
    """Strict JSON parse (expects full {"arm_id","points","censors"} object)."""
    parsed, _ = extract_stage2_json(text)
    if isinstance(parsed, dict) and ("points" in parsed or "censors" in parsed):
        return parsed
    return None


def parse_stage2_output_relaxed(text: str, *, arm_id: str = "") -> Optional[dict]:
    """
    Recover coordinates when the model emits a truncated points stream
    (common when train_stage2 used max_length=512 but labels are ~800+ tokens).
    """
    strict = parse_stage2_output(text)
    if strict is not None:
        return strict

    if not text or not _PAIR_RE.search(text):
        return None

    pairs = [
        [float(a), float(b)]
        for a, b in _PAIR_RE.findall(text)
    ]
    if not pairs:
        return None

    # Heuristic: censors often share survival with nearby step — keep all pairs as points;
    # eval censor F1 uses time-only matching on censors list (empty unless JSON had "censors").
    return {"arm_id": arm_id, "points": pairs, "censors": []}


def _as_points(raw: Any) -> List[Tuple[float, float]]:
    """Nested [[x,y],...] or flat [x,y,x,y,...] from model or label."""
    if isinstance(raw, list) and raw and isinstance(raw[0], dict):
        out: List[Tuple[float, float]] = []
        for item in raw:
            if isinstance(item, dict) and "t" in item and "s" in item:
                out.append((float(item["t"]), float(item["s"])))
        return out
    return as_point_tuples(raw)


def _censor_times(raw: Any) -> List[float]:
    return sorted(t for t, _ in _as_points(raw))


def _label_coordinate_space(label_obj: dict) -> str:
    return str(label_obj.get("_meta", {}).get("coordinate_space", "clinical"))


def match_points(
    gt: Sequence[Tuple[float, float]],
    pred: Sequence[Tuple[float, float]],
    *,
    time_tol: float = POINT_TIME_MATCH_TOL,
    normalized_space: bool = False,
) -> Tuple[List[float], int, int]:
    if not gt:
        return [], 0, 0
    pred_remaining = list(pred)
    sq_errors: List[float] = []
    matched = 0

    for gt_a, gt_b in gt:
        best_j = -1
        best_da = time_tol + 1.0
        for j, (p_a, p_b) in enumerate(pred_remaining):
            da = abs(p_a - gt_a)
            if da <= time_tol and da < best_da:
                best_da = da
                best_j = j
        if best_j >= 0:
            p_a, p_b = pred_remaining.pop(best_j)
            if normalized_space:
                sq_errors.append((p_a - gt_a) ** 2 + (p_b - gt_b) ** 2)
            else:
                sq_errors.append((p_b - gt_b) ** 2)
            matched += 1

    return sq_errors, matched, len(gt)


def coordinate_rmse(
    gt: Sequence[Tuple[float, float]],
    pred: Sequence[Tuple[float, float]],
    *,
    time_tol: float = POINT_TIME_MATCH_TOL,
    normalized_space: bool = False,
) -> Tuple[Optional[float], int, int]:
    sq, matched, n_gt = match_points(
        gt, pred, time_tol=time_tol, normalized_space=normalized_space
    )
    if matched == 0:
        return None, 0, n_gt
    return math.sqrt(sum(sq) / len(sq)), matched, n_gt


def set_prf1(
    gt_times: Sequence[float],
    pred_times: Sequence[float],
    *,
    tol: float = CENSOR_TIME_TOL,
) -> Tuple[float, float, float, int, int, int]:
    if not gt_times and not pred_times:
        return 1.0, 1.0, 1.0, 0, 0, 0
    if not gt_times:
        return 0.0, 1.0, 0.0, 0, len(pred_times), 0
    if not pred_times:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_times)

    pred_used = [False] * len(pred_times)
    tp = 0
    for g in gt_times:
        for j, p in enumerate(pred_times):
            if pred_used[j]:
                continue
            if abs(g - p) <= tol:
                tp += 1
                pred_used[j] = True
                break

    fp = sum(1 for u in pred_used if not u)
    fn = len(gt_times) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


@torch.inference_mode()
def build_inputs_embeds(model, pixel_values, input_ids, device):
    if pixel_values.dim() == 4:
        pixel_values = pixel_values.unsqueeze(0)
    pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    input_ids = input_ids.to(device)

    b_val, num_crops, _, _, _ = pixel_values.shape
    flat = pixel_values.view(b_val * num_crops, *pixel_values.shape[2:])
    vision_out = model.vision_encoder(pixel_values=flat)
    projected = model.projector(vision_out.last_hidden_state)
    _, num_patches, embed_dim = projected.shape
    projected = projected.view(b_val, num_crops * num_patches, embed_dim)

    text_embeds = model.llm.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    image_mask = torch.ones((b_val, projected.shape[1]), dtype=torch.long, device=device)
    text_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([image_mask, text_mask], dim=1)
    return inputs_embeds, attention_mask


@torch.inference_mode()
def generate_tile_json(
    model,
    processor,
    tokenizer,
    image_path: Path,
    arm_id: str,
    device: torch.device,
    *,
    max_new_tokens: int = 384,
    force_json_prefix: bool = False,
) -> str:
    image = Image.open(image_path).convert("RGB")
    if image.size != (384, 384):
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
    pixel_values = processor(images=[image], return_tensors="pt").pixel_values

    user_prompt = stage2_user_prompt(arm_id)
    chat_prefix = chat_prefix_for_user(user_prompt, tokenizer)
    assistant_forced = force_json_assistant_prefix(arm_id) if force_json_prefix else ""
    full_prefix = chat_prefix + assistant_forced

    encoded = tokenizer(full_prefix, return_tensors="pt", add_special_tokens=True)
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

    if input_ids is not None and output_ids.shape[1] > input_ids.shape[1] + 8:
        new_ids = output_ids[:, input_ids.shape[1] :]
    else:
        new_ids = output_ids

    continuation = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
    if force_json_prefix:
        return assistant_forced + continuation
    return continuation


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
    model.projector.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "projector_weights.pth"), map_location=device)
    )
    model.llm = PeftModel.from_pretrained(model.llm, checkpoint_dir)
    model.llm.config.use_cache = True
    model.eval()
    return model


def aggregate_stage2_metrics(records: List[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"count": 0}

    json_valid = sum(1 for r in records if r.get("json_valid"))
    json_valid_strict = sum(1 for r in records if r.get("json_valid_strict"))
    rmse_vals = [r["coordinate_rmse"] for r in records if r.get("coordinate_rmse") is not None]
    precisions = [r.get("censoring_precision", 0.0) for r in records]
    recalls = [r.get("censoring_recall", 0.0) for r in records]
    f1s = [r.get("censoring_f1", 0.0) for r in records]

    total_matched = sum(r.get("points_matched", 0) for r in records)
    total_gt_points = sum(r.get("n_gt_points", 0) for r in records)
    pooled_sq = sum(r.get("points_sq_error_sum", 0.0) for r in records)
    pooled_rmse = math.sqrt(pooled_sq / total_matched) if total_matched else None

    total_tp = sum(r.get("censor_tp", 0) for r in records)
    total_fp = sum(r.get("censor_fp", 0) for r in records)
    total_fn = sum(r.get("censor_fn", 0) for r in records)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    return {
        "count": n,
        "json_valid_rate": json_valid / n,
        "json_valid_strict_rate": json_valid_strict / n,
        "mean_coordinate_rmse": sum(rmse_vals) / len(rmse_vals) if rmse_vals else None,
        "pooled_coordinate_rmse": pooled_rmse,
        "point_match_rate": total_matched / total_gt_points if total_gt_points else 0.0,
        "points_matched": total_matched,
        "points_ground_truth": total_gt_points,
        "mean_censoring_precision": sum(precisions) / n,
        "mean_censoring_recall": sum(recalls) / n,
        "mean_censoring_f1": sum(f1s) / n,
        "micro_censoring_precision": micro_p,
        "micro_censoring_recall": micro_r,
        "micro_censoring_f1": micro_f1,
        "censor_tp": total_tp,
        "censor_fp": total_fp,
        "censor_fn": total_fn,
        "point_time_match_tol": POINT_TIME_MATCH_TOL,
        "censor_time_tol": CENSOR_TIME_TOL,
        "num_image_tokens": NUM_IMAGE_TOKENS,
    }


def run_evaluation(
    model,
    processor,
    tokenizer,
    pairs: List[Tuple[Path, Path]],
    device: torch.device,
    *,
    max_new_tokens: int,
    force_json_prefix: bool = False,
) -> List[dict]:
    records: List[dict] = []

    for img_path, label_path in tqdm(pairs, desc="eval_stage2"):
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)
        arm_id = gt.get("arm_id", "unknown")
        normalized_space = _label_coordinate_space(gt) == "normalized_local"
        match_tol = POINT_X_MATCH_TOL_NORMALIZED if normalized_space else POINT_TIME_MATCH_TOL
        censor_tol = CENSOR_X_TOL_NORMALIZED if normalized_space else CENSOR_TIME_TOL
        gt_points = _as_points(gt.get("points", []))
        gt_censors = _censor_times(gt.get("censors", []))

        rec: dict = {
            "image": str(img_path),
            "label": str(label_path),
            "arm_id": arm_id,
            "coordinate_space": _label_coordinate_space(gt),
            "n_gt_points": len(gt_points),
            "n_gt_censors": len(gt_censors),
            "force_json_prefix": force_json_prefix,
        }

        try:
            raw = generate_tile_json(
                model, processor, tokenizer, img_path, arm_id, device,
                max_new_tokens=max_new_tokens,
                force_json_prefix=force_json_prefix,
            )
            rec["inference_error"] = None
        except Exception as exc:
            raw = ""
            rec["inference_error"] = str(exc)

        rec["prediction_raw"] = raw[:4000]
        pred_strict = parse_stage2_output(raw)
        pred = parse_stage2_output_relaxed(raw, arm_id=arm_id)

        if pred is None:
            rec.update(
                {
                    "json_valid": False,
                    "json_valid_strict": False,
                    "parse_mode": "failed",
                    "censoring_precision": 0.0,
                    "censoring_recall": 0.0,
                    "censoring_f1": 0.0,
                }
            )
            records.append(rec)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        rec["json_valid"] = True
        rec["json_valid_strict"] = pred_strict is not None
        rec["parse_mode"] = "strict_json" if pred_strict is not None else "relaxed_pairs"
        pred_points = _as_points(pred.get("points", []))
        pred_censors = _censor_times(pred.get("censors", []))
        rec["n_pred_points"] = len(pred_points)
        rec["n_pred_censors"] = len(pred_censors)

        rmse, matched, n_gt = coordinate_rmse(
            gt_points, pred_points, time_tol=match_tol, normalized_space=normalized_space
        )
        sq_sum = 0.0
        if matched:
            sq, _, _ = match_points(
                gt_points,
                pred_points,
                time_tol=match_tol,
                normalized_space=normalized_space,
            )
            sq_sum = sum(sq)
        rec["coordinate_rmse"] = round(rmse, 6) if rmse is not None else None
        rec["points_matched"] = matched
        rec["points_sq_error_sum"] = sq_sum

        p, r, f1, tp, fp, fn = set_prf1(gt_censors, pred_censors, tol=censor_tol)
        rec.update(
            {
                "censoring_precision": round(p, 4),
                "censoring_recall": round(r, 4),
                "censoring_f1": round(f1, 4),
                "censor_tp": tp,
                "censor_fp": fp,
                "censor_fn": fn,
            }
        )
        records.append(rec)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return records


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 tile evaluator (RMSE + censoring F1).")
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--holdout-dir", type=str, default=None)
    p.add_argument("--image-dir", type=str, default=None)
    p.add_argument("--label-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Holdout tiles to score (0 = all). Default 150 like a Phase C spot-check.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Generation cap (tile labels are often 800+ tokens).",
    )
    p.add_argument(
        "--force-json-prefix",
        action="store_true",
        help='Pre-fill assistant with {"arm_id": "<id>", "points": [ before generation.',
    )
    p.add_argument(
        "--rescore-only",
        type=str,
        default=None,
        metavar="JSONL",
        help="Re-score an existing eval JSONL without re-running inference.",
    )
    return p.parse_args()


def rescore_jsonl(jsonl_path: Path, *, checkpoint: str) -> dict:
    """Re-aggregate metrics from a saved eval JSONL using current parsers."""
    records: List[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        raw = rec.get("prediction_raw", "")
        arm_id = rec.get("arm_id", "")
        with open(rec["label"], encoding="utf-8") as f:
            gt = json.load(f)
        gt_points = _as_points(gt.get("points", []))
        gt_censors = _censor_times(gt.get("censors", []))
        normalized_space = _label_coordinate_space(gt) == "normalized_local"
        match_tol = POINT_X_MATCH_TOL_NORMALIZED if normalized_space else POINT_TIME_MATCH_TOL
        censor_tol = CENSOR_X_TOL_NORMALIZED if normalized_space else CENSOR_TIME_TOL
        pred_strict = parse_stage2_output(raw)
        pred = parse_stage2_output_relaxed(raw, arm_id=arm_id)
        if pred is None:
            rec.update(
                json_valid=False,
                json_valid_strict=False,
                parse_mode="failed",
                censoring_precision=0.0,
                censoring_recall=0.0,
                censoring_f1=0.0,
            )
            records.append(rec)
            continue
        pred_points = _as_points(pred.get("points", []))
        pred_censors = _censor_times(pred.get("censors", []))
        rmse, matched, _ = coordinate_rmse(
            gt_points, pred_points, time_tol=match_tol, normalized_space=normalized_space
        )
        sq_sum = 0.0
        if matched:
            sq, _, _ = match_points(
                gt_points,
                pred_points,
                time_tol=match_tol,
                normalized_space=normalized_space,
            )
            sq_sum = sum(sq)
        p, r, f1, tp, fp, fn = set_prf1(gt_censors, pred_censors, tol=censor_tol)
        rec.update(
            json_valid=True,
            json_valid_strict=pred_strict is not None,
            parse_mode="strict_json" if pred_strict is not None else "relaxed_pairs",
            n_pred_points=len(pred_points),
            n_pred_censors=len(pred_censors),
            coordinate_rmse=round(rmse, 6) if rmse is not None else None,
            points_matched=matched,
            points_sq_error_sum=sq_sum,
            censoring_precision=round(p, 4),
            censoring_recall=round(r, 4),
            censoring_f1=round(f1, 4),
            censor_tp=tp,
            censor_fp=fp,
            censor_fn=fn,
        )
        records.append(rec)

    summary = aggregate_stage2_metrics(records)
    summary["checkpoint"] = checkpoint.replace("\\", "/")
    summary["eval_samples"] = len(records)
    summary["rescored_from"] = str(jsonl_path)
    return summary


def main() -> int:
    args = parse_args()

    if args.rescore_only:
        jsonl_path = Path(args.rescore_only)
        if not jsonl_path.is_file():
            print(f"JSONL not found: {jsonl_path}")
            return 1
        checkpoint_dir = resolve_checkpoint(args.checkpoint)
        summary = rescore_jsonl(jsonl_path, checkpoint=checkpoint_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        latest_path = out_dir / "latest_summary.json"
        with open(latest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("\n=== Stage 2 rescore summary ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"\nWrote {latest_path}")
        return 0

    root = Path(args.dataset_root)
    holdout = Path(args.holdout_dir or root / "stage2_v2_1_holdout")
    image_dir = Path(args.image_dir or holdout / "images" / "km")
    label_dir = Path(args.label_dir or holdout / "labels" / "km")

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"Missing holdout tiles: {image_dir} or {label_dir}")
        return 1

    all_pairs = load_tile_pairs(image_dir, label_dir)
    pairs = sample_pairs(all_pairs, max_samples=args.max_samples, seed=args.seed)
    if not pairs:
        print("No tile pairs found.")
        return 1

    checkpoint_dir = resolve_checkpoint(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Checkpoint: {checkpoint_dir}")
    print(f"Evaluating {len(pairs)} / {len(all_pairs)} holdout tiles from {holdout}")
    if args.force_json_prefix:
        print("  force_json_prefix: ON (assistant pre-filled through points array open)")

    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(checkpoint_dir, device)
    records = run_evaluation(
        model,
        processor,
        tokenizer,
        pairs,
        device,
        max_new_tokens=args.max_new_tokens,
        force_json_prefix=args.force_json_prefix,
    )

    stamp = _utc_stamp()
    summary = aggregate_stage2_metrics(records)
    summary["checkpoint"] = checkpoint_dir.replace("\\", "/")
    summary["holdout_dir"] = str(holdout)
    summary["eval_samples"] = len(pairs)
    summary["holdout_pool_size"] = len(all_pairs)
    summary["seed"] = args.seed
    summary["force_json_prefix"] = args.force_json_prefix
    summary["timestamp_utc"] = stamp

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"eval_{stamp}.jsonl"
    summary_path = out_dir / f"eval_{stamp}_summary.json"
    metrics_path = out_dir / "stage2_metrics.json"
    latest_path = out_dir / "latest_summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage 2 evaluation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nPer-tile results: {jsonl_path}")
    print(f"Summary JSON:     {summary_path}")
    print(f"Metrics copy:     {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
