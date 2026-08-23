"""
Definitive macro vs Stage-2-E2E benchmark on the frozen 500-chart validation set.

Three arms, one shared chart list (validation_manifest.json, seed=42):

  1. MACRO      : Phase C Run 2 checkpoint, direct full-chart inference.
  2. E2E-ORACLE : Stage 2 tile predictions on oracle tile boundaries
                  (tiles from the holdout pool of the SAME 500 charts,
                  stitched with the repaired strict stitcher).
                  Upper bound for the two-stage pipeline: real deployments
                  would need a segmentation model to find tiles; this arm
                  does not pay that cost.

Outputs per arm:
  evaluation/results/val500/<arm>/eval_<stamp>.jsonl   (per-chart records)
  evaluation/results/val500/<arm>/latest_summary.json  (mean + percentiles)

Usage:
  venv/Scripts/python.exe scripts/run_validation_benchmark.py --arm macro
  venv/Scripts/python.exe scripts/run_validation_benchmark.py --arm e2e_oracle
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_DATASET_ROOT = r"C:\sem4\KMVision-1 Data\dataset"
MACRO_CKPT = "checkpoints/phase_c_run2_chatml/final"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_manifest(dataset_root: Path) -> list[str]:
    man = dataset_root / "validation_manifest.json"
    if not man.is_file():
        raise FileNotFoundError(
            f"{man} missing - run scripts/make_validation_split.py first"
        )
    data = json.loads(man.read_text(encoding="utf-8"))
    ids = data["categories"]["km"]
    if len(ids) < 500:
        print(f"WARNING: manifest has only {len(ids)} charts")
    return ids


def summarize(scores: list[dict]) -> dict:
    def pct(vals: list[float], p: float) -> float:
        s = sorted(vals)
        idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
        return s[idx]

    out: dict = {}
    keys = sorted({k for s in scores for k in s})
    for k in keys:
        vals = [float(s[k]) for s in scores if k in s and isinstance(s[k], (int, float))]
        if not vals:
            continue
        out[f"mean_{k}"] = statistics.fmean(vals)
        out[f"median_{k}"] = statistics.median(vals)
        out[f"stdev_{k}"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[f"p10_{k}"] = pct(vals, 10)
        out[f"p90_{k}"] = pct(vals, 90)
    return out


def run_macro(chart_ids: list[str], root: Path, device, *, stop_after: int = 0,
              resume: bool = True) -> tuple[list[dict], list]:
    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from eval_inference import decompress_json, generate_extraction, load_model as load_macro
    from evaluation.metrics import ChartScore, score_extraction
    from evaluation.parse_output import extract_json_from_text

    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_macro(MACRO_CKPT, device)

    # Resume support: skip charts already scored in prior partial runs.
    done_ids: dict[str, dict] = {}
    if resume:
        for prev in sorted(Path("evaluation/results/val500/macro").glob("partial_*.jsonl")):
            for line in prev.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "score" in rec and "error" not in rec:
                    done_ids[rec["chart"]] = rec["score"]

    empty_score = None

    scores: list = []
    records: list[dict] = []
    partial_path = Path("evaluation/results/val500/macro/partial_current.jsonl")
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    with open(partial_path, "a", encoding="utf-8") as partial_f:
        fresh = 0
        for i, cid in enumerate(chart_ids):
            if cid in done_ids:
                sc = dict(done_ids[cid])
                sc["chart"] = cid
                scores.append(sc)
                records.append({"chart": cid, "score": sc, "resumed": True})
                continue
            if stop_after and fresh >= stop_after:
                print(f"[macro] stop-after limit reached ({fresh} fresh charts this process)", flush=True)
                break
            fresh += 1
            gt_path = root / "testing" / "labels" / "km" / f"{cid}.json"
            img_candidates = list((root / "testing" / "images" / "km").glob(f"{cid}.*"))
            if not gt_path.is_file() or not img_candidates:
                records.append({"chart": cid, "error": "missing_gt_or_image"})
                continue
            gt = json.loads(gt_path.read_text(encoding="utf-8", errors="replace"))
            raw = ""
            expanded = None
            err = None
            score = None
            try:
                raw = generate_extraction(model, processor, tokenizer, str(img_candidates[0]), device)
                parsed, _ = extract_json_from_text(raw)
                expanded = decompress_json(parsed) if parsed else None
                score = score_extraction(gt, expanded).to_dict() if expanded else \
                    score_extraction(gt, "").to_dict()
            except torch.AcceleratorError as exc:
                # GPU-level failure: record and keep going; model may still work.
                err = f"gpu:{type(exc).__name__}:{exc}"[:300]
                if empty_score is None:
                    empty_score = score_extraction(gt, "").to_dict()
                score = dict(empty_score)
            except Exception as exc:  # noqa: BLE001 - record and continue
                err = f"inference:{type(exc).__name__}:{exc}"[:300]
                if empty_score is None:
                    empty_score = score_extraction(gt, "").to_dict()
                score = dict(empty_score)
            score["chart"] = cid
            scores.append(score)
            rec_out = {"chart": cid, "score": score, "prediction_raw": raw[:4000]}
            if err:
                rec_out["error"] = err
            records.append(rec_out)
            partial_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            partial_f.flush()
            if (i + 1) % 25 == 0:
                print(f"[macro] {i+1}/{len(chart_ids)}", flush=True)
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass  # CUDA hiccup on cache clear must not kill the run
    return records, scores


def run_e2e_oracle(chart_ids: list[str], root: Path, device,
                   stop_after: int = 0, resume: bool = True) -> tuple[list[dict], list]:
    """Stage-2 tile predictions stitched to charts, scored against chart GT."""
    from collections import defaultdict

    from eval_e2e import load_chart_gt
    from eval_stage2 import (
        generate_tile_json,
        load_model as load_stage2,
    )
    from evaluation.metrics import ChartScore, score_extraction
    from scripts.stitch_tiles import stitch_chart_from_tiles
    from transformers import AutoProcessor, AutoTokenizer

    holdout = root / "stage2_validation"
    img_dir, lbl_dir = holdout / "images" / "km", holdout / "labels" / "km"

    # Collect ALL holdout tiles belonging to the 500 validation charts.
    by_chart: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    found_charts: set[str] = set()
    scanned = 0
    for lf in lbl_dir.glob("*.json"):
        scanned += 1
        meta = json.loads(lf.read_text(encoding="utf-8")).get("_meta", {})
        sc = meta.get("source_chart")
        if sc in set(chart_ids):
            stem = lf.stem
            img = img_dir / f"{stem}.png"
            if img.is_file():
                by_chart[sc].append((img, lf))
                found_charts.add(sc)
    print(f"Scanned {scanned} holdout tiles; matched {len(found_charts)}/500 "
          f"validation charts with tiles.")

    # Device passed from main; use it.
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_stage2("checkpoints/stage2_v2_1/final", device)

    import torch

    # Resume support: skip charts already scored in prior partial runs.
    done_scores: dict[str, dict] = {}
    if resume:
        import glob
        for prev in sorted(Path("evaluation/results/val500/e2e_oracle").glob("partial_*.jsonl")):
            for line in open(prev, encoding="utf-8"):
                r = json.loads(line)
                if "score" in r and "error" not in r:
                    done_scores[r["chart"]] = r["score"]

    scores: list = []
    records: list[dict] = []
    stitch_errors = 0
    fresh = 0
    partial_dir = Path("evaluation/results/val500/e2e_oracle")
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial_path = partial_dir / "partial_current.jsonl"
    with open(partial_path, "a", encoding="utf-8") as partial_f:
        for i, cid in enumerate(chart_ids):
            if cid in done_scores:
                sc = dict(done_scores[cid])
                sc["chart"] = cid
                scores.append(sc)
                records.append({"chart": cid, "n_tiles": 0, "score": sc, "resumed": True})
                continue

            tiles = by_chart.get(cid, [])
            gt = load_chart_gt(root, cid)
            if gt is None:
                records.append({"chart": cid, "error": "missing_gt"})
                continue
            if not tiles:
                score = score_extraction(gt, {"chart_type": "kaplan_meier",
                                              "axes": {}, "arms": []}).to_dict()
                score["chart"] = cid
                score["no_tiles"] = True
                scores.append(score)
                records.append({"chart": cid, "error": "no_tiles_in_holdout_pool", "score": score})
                continue
            tile_records = []
            stitched = None
            error_msg = None
            try:
                for img, lf in tiles:
                    lbl = json.loads(lf.read_text(encoding="utf-8", errors="replace"))
                    arm_id = lbl.get("arm_id", "unknown")
                    raw = generate_tile_json(model, processor, tokenizer, str(img), arm_id,
                                             device, force_json_prefix=True)
                    tile_records.append({"label": str(lf), "prediction_raw": raw,
                                         "_meta": lbl.get("_meta", {}), "arm_id": arm_id})
                stitched = stitch_chart_from_tiles(tile_records, chart_gt=gt, strict=True)
                score = score_extraction(gt, stitched).to_dict()
            except torch.AcceleratorError as exc:
                # Cumulative in-process GPU corruption: every later generation in
                # this process would fail too. Abort cleanly (exit 3) so the outer
                # restart loop resumes from the partial file with a fresh process.
                print(f"[e2e_oracle] GPU state degraded at chart {cid}: {exc}", flush=True)
                raise SystemExit(3)
            except Exception as exc:  # noqa: BLE001
                stitch_errors += 1
                error_msg = f"stitch/inference:{type(exc).__name__}:{exc}"[:300]
                score = score_extraction(gt, {"chart_type": "kaplan_meier",
                                              "axes": {}, "arms": []}).to_dict()
            score["chart"] = cid
            score["n_tiles"] = len(tiles)
            scores.append(score)
            if error_msg:
                records.append({"chart": cid, "error": error_msg,
                                "n_tiles": len(tiles), "score": score})
            else:
                records.append({"chart": cid, "n_tiles": len(tiles), "score": score,
                                "stitched_arms": [a.get("treatment_label") for a in stitched.get("arms", [])]})
            if (i + 1) % 25 == 0:
                print(f"[e2e_oracle] {i+1}/{len(chart_ids)} (stitch errors so far: {stitch_errors})", flush=True)
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:  # CUDA hiccup on cache clear must not kill the run
                    pass

            partial_f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
            partial_f.flush()
            fresh += 1
            if stop_after and fresh >= stop_after:
                break

    return records, scores


def main() -> int:
    parser = argparse.ArgumentParser(description="500-chart validation benchmark.")
    parser.add_argument("--arm", choices=["macro", "e2e_oracle", "macro_baseline"], required=True)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", default="evaluation/results/val500")
    parser.add_argument(
        "--stop-after", type=int, default=0,
        help="Macro only: max fresh charts this process should run, then exit cleanly "
             "(0 = no limit). Used to work around cumulative in-process GPU degradation: "
             "run repeatedly with e.g. --stop-after 100 until the partial file covers all charts.",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore prior partial results and start from scratch.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    chart_ids = load_manifest(root)

    out_dir = Path(args.output_dir) / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    device = __import__("torch").device("cuda:0" if __import__("torch").cuda.is_available() else "cpu")
    print(f"Arm: {args.arm} | charts: {len(chart_ids)} | device: {device}")

    if args.arm == "macro":
        records, scores = run_macro(chart_ids, root, device,
                                    stop_after=args.stop_after,
                                    resume=not args.no_resume)
    elif args.arm == "macro_baseline":
        # Same as macro arm: Phase C Run 2 checkpoint, direct full-chart inference.
        # Run separately for explicit baseline comparison.
        records, scores = run_macro(chart_ids, root, device,
                                    stop_after=args.stop_after,
                                    resume=not args.no_resume)
    else:
        records, scores = run_e2e_oracle(chart_ids, root, device,
                                         stop_after=args.stop_after,
                                         resume=not args.no_resume)

    stamp = utc()
    summary = {
        "arm": args.arm,
        "n_charts": len(chart_ids),
        "scored": len(scores),
        "errors": len([r for r in records if "error" in r]),
        "manifest_seed": 42,
        "timestamp_utc": stamp,
        **summarize(scores),
    }
    with open(out_dir / f"eval_{stamp}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / f"summary_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "latest_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Validation benchmark summary ===")
    for k in sorted(summary):
        v = summary[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
