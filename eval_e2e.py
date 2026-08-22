"""
End-to-end evaluation: macro chart -> tile crops -> Stage 2 -> stitch -> score.

Modes:
  1. Full GPU pipeline (macro + stage2 inference on holdout tiles)
  2. Fast rescoring from saved Stage 2 JSONL (--stage2-jsonl) — no GPU

Usage:
  python eval_e2e.py --max-charts 12 --stage2-jsonl evaluation/results/stage2_v2_1_holdout/eval_20260606T044115Z.jsonl
  python eval_e2e.py --max-charts 5 --run-inference
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import aggregate_scores, score_extraction
from evaluation.parse_output import extract_json_from_text
from eval_inference import decompress_json, generate_extraction, load_model as load_macro_model
from eval_stage2 import (
    generate_tile_json,
    load_model as load_stage2_model,
    load_tile_pairs,
    parse_stage2_output_relaxed,
    sample_pairs,
)
from scripts.stitch_tiles import StitchError, group_eval_jsonl_by_chart, stitch_chart_from_tiles
from transformers import AutoProcessor, AutoTokenizer

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
DEFAULT_MACRO_CKPT = "checkpoints/phase_c_run2_chatml/final"
DEFAULT_STAGE2_CKPT = "checkpoints/stage2_v2_1/final"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_chart_gt(dataset_root: Path, chart_stem: str) -> Optional[dict]:
    for sub in ("testing", "train_1"):
        path = dataset_root / sub / "labels" / "km" / f"{chart_stem}.json"
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def charts_from_tile_jsonl(jsonl_path: Path) -> Dict[str, List[dict]]:
    return group_eval_jsonl_by_chart(jsonl_path)


def run_stage2_on_chart_tiles(
    chart_stem: str,
    tile_records: List[dict],
    model,
    processor,
    tokenizer,
    device,
    *,
    force_json_prefix: bool = True,
) -> List[dict]:
    """Run Stage 2 inference on tiles belonging to one chart (if not already in JSONL)."""
    out: List[dict] = []
    for rec in tile_records:
        if rec.get("prediction_raw"):
            out.append(rec)
            continue
        img_path = Path(rec.get("image", ""))
        label_path = Path(rec.get("label", ""))
        if not img_path.is_file() or not label_path.is_file():
            continue
        with open(label_path, encoding="utf-8") as f:
            label_obj = json.load(f)
        arm_id = label_obj.get("arm_id", "unknown")
        raw = generate_tile_json(
            model,
            processor,
            tokenizer,
            img_path,
            arm_id,
            device,
            force_json_prefix=force_json_prefix,
        )
        out.append({**rec, "prediction_raw": raw, "arm_id": arm_id})
    return out


def score_stitched_chart(chart_stem: str, tile_records: List[dict], dataset_root: Path) -> dict:
    gt = load_chart_gt(dataset_root, chart_stem)
    if gt is None:
        return {"chart": chart_stem, "error": "missing_gt"}

    try:
        stitched = stitch_chart_from_tiles(tile_records, chart_gt=gt)
    except StitchError as exc:
        # Loud, explicit failure: never silently score a chart with dropped predictions.
        return {"chart": chart_stem, "error": "stitch_failed", "detail": str(exc)}
    score = score_extraction(gt, stitched)
    return {
        "chart": chart_stem,
        "label": str(dataset_root / "testing" / "labels" / "km" / f"{chart_stem}.json"),
        "n_tiles": len(tile_records),
        "stitched_arms": len(stitched.get("arms", [])),
        "score": score.to_dict(),
        "stitched": stitched,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end KM extraction eval (macro + tiles + stitch).")
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--macro-checkpoint", default=DEFAULT_MACRO_CKPT)
    p.add_argument("--stage2-checkpoint", default=DEFAULT_STAGE2_CKPT)
    p.add_argument(
        "--stage2-jsonl",
        default=None,
        help="Reuse Stage 2 tile predictions (fast, no tile GPU inference).",
    )
    p.add_argument("--holdout-dir", default=None, help="Tile holdout root (default: stage2_v2_1_holdout).")
    p.add_argument("--max-charts", type=int, default=12, help="Max source charts to score.")
    p.add_argument("--run-inference", action="store_true", help="Run Stage 2 GPU inference on missing preds.")
    p.add_argument("--macro-eval", action="store_true", help="Also score macro-only baseline per chart.")
    p.add_argument("--output-dir", default="evaluation/results/e2e")
    p.add_argument("--force-json-prefix", action="store_true", default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    holdout = Path(args.holdout_dir or dataset_root / "stage2_v2_1_holdout")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage2_jsonl:
        chart_groups = charts_from_tile_jsonl(Path(args.stage2_jsonl))
    else:
        image_dir = holdout / "images" / "km"
        label_dir = holdout / "labels" / "km"
        pairs = load_tile_pairs(image_dir, label_dir)
        chart_groups_map: DefaultDict[str, List[dict]] = defaultdict(list)
        for img, lbl in pairs:
            with open(lbl, encoding="utf-8") as f:
                meta = json.load(f).get("_meta", {})
            stem = meta.get("source_chart", lbl.stem)
            chart_groups_map[stem].append({"image": str(img), "label": str(lbl)})
        chart_groups = dict(chart_groups_map)

    chart_stems = sorted(chart_groups.keys())[: args.max_charts]
    if not chart_stems:
        print("No charts to evaluate.")
        return 1

    device = __import__("torch").device("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    stage2_model = None
    processor = None
    tokenizer = None

    if args.run_inference and not args.stage2_jsonl:
        processor = AutoProcessor.from_pretrained(
            "google/siglip2-so400m-patch14-384", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        stage2_model = load_stage2_model(args.stage2_checkpoint, device)

    macro_model = None
    if args.macro_eval:
        processor = processor or AutoProcessor.from_pretrained(
            "google/siglip2-so400m-patch14-384", trust_remote_code=True
        )
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        macro_model = load_macro_model(args.macro_checkpoint, device)

    e2e_records: List[dict] = []
    macro_records: List[dict] = []

    for chart_stem in chart_stems:
        tiles = chart_groups[chart_stem]
        if stage2_model is not None:
            tiles = run_stage2_on_chart_tiles(
                chart_stem,
                tiles,
                stage2_model,
                processor,
                tokenizer,
                device,
                force_json_prefix=args.force_json_prefix,
            )

        rec = score_stitched_chart(chart_stem, tiles, dataset_root)
        e2e_records.append(rec)

        if macro_model is not None:
            gt = load_chart_gt(dataset_root, chart_stem)
            if gt:
                img_candidates = list((dataset_root / "testing" / "images" / "km").glob(f"{chart_stem}.*"))
                if img_candidates:
                    raw = generate_extraction(
                        macro_model, processor, tokenizer, str(img_candidates[0]), device
                    )
                    parsed, _ = extract_json_from_text(raw)
                    expanded = decompress_json(parsed) if parsed else raw
                    macro_score = score_extraction(gt, expanded)
                    macro_records.append(
                        {
                            "chart": chart_stem,
                            "score": macro_score.to_dict(),
                            "prediction_raw": raw[:2000],
                        }
                    )

    from evaluation.metrics import ChartScore

    scores = [ChartScore(**r["score"]) for r in e2e_records if "score" in r and "error" not in r]
    error_records = [r for r in e2e_records if "error" in r]
    summary = aggregate_scores(scores) if scores else {"count": 0}
    summary["eval_charts"] = len(e2e_records)
    summary["scored_charts"] = len(scores)
    summary["error_charts"] = len(error_records)
    summary["timestamp_utc"] = _utc_stamp()
    summary["stage2_jsonl"] = args.stage2_jsonl
    summary["macro_checkpoint"] = args.macro_checkpoint
    summary["stage2_checkpoint"] = args.stage2_checkpoint

    if macro_records:
        macro_scores = [ChartScore(**r["score"]) for r in macro_records]
        macro_summary = aggregate_scores(macro_scores)
        summary["macro_only"] = macro_summary

    stamp = _utc_stamp()
    jsonl_path = out_dir / f"e2e_{stamp}.jsonl"
    summary_path = out_dir / f"e2e_{stamp}_summary.json"

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in e2e_records:
            slim = {k: v for k, v in rec.items() if k != "stitched"}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "latest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== End-to-end evaluation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}: {json.dumps(v, indent=2)[:200]}...")
        else:
            print(f"  {k}: {v}")
    print(f"\nPer-chart results: {jsonl_path}")
    print(f"Summary JSON:      {summary_path}")

    if error_records:
        print(f"\nE2E FAILURE: {len(error_records)} chart(s) failed to stitch and were NOT scored.")
        for r in error_records:
            print(f"  - {r['chart']}: {r.get('error')} :: {str(r.get('detail', ''))[:200]}")
        print("Mean scores above exclude these charts. This run is incomplete.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
