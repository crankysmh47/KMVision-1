"""Evaluate a V2-SFT checkpoint on the validation_v2 partition.

Legacy metrics via evaluation.metrics.score_extraction (curve/censor/etc.)
plus per-field accuracy for the six new schema fields:
  time_unit (exact), title (exact + soft), hazard_ratio / ci bounds
  (relative tolerance), p_value (tolerance ladder), at_risk_table
  (timepoint recall + cell accuracy).

Usage:
  venv/Scripts/python.exe scripts/eval_v2_fields.py --ckpt checkpoints/v2_sft_run1/final
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_ROOT = Path(r"C:\sem4\KMVision-1 Data\dataset")
MANIFEST = DATASET_ROOT / "validation_v2_manifest.json"
OUT_DIR = ROOT / "evaluation" / "results" / "valv2"


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def rel_err(pred: float, gt: float) -> float:
    if gt == 0:
        return 0.0 if pred == 0 else float("inf")
    return abs(pred - gt) / abs(gt)


def field_metrics(gt: dict, pred: dict | None) -> dict:
    out: dict = {}
    if not isinstance(pred, dict):
        out.update({f"f_{k}": 0.0 for k in
                    ("unit_match", "title_exact", "title_soft", "hr_ok",
                     "lo_ok", "hi_ok", "pv_log10_ok")})
        out["f_rt_timepoint_recall"] = 0.0
        out["f_rt_cell_acc"] = 0.0
        return out

    out["f_unit_match"] = 1.0 if pred.get("time_unit") == gt.get("time_unit") else 0.0

    gt_t, pr_t = norm_text(gt.get("title", "")), norm_text(pred.get("title", ""))
    out["f_title_exact"] = 1.0 if gt_t and gt_t == pr_t else 0.0
    g_ws, p_ws = set(gt_t.split()), set(pr_t.split())
    out["f_title_soft"] = (2 * len(g_ws & p_ws) / (len(g_ws) + len(p_ws))) if (g_ws or p_ws) else 0.0

    try:
        out["f_hr_ok"] = 1.0 if rel_err(float(pred["hazard_ratio"]),
                                        float(gt["hazard_ratio"])) <= 0.10 else 0.0
    except (KeyError, TypeError, ValueError):
        out["f_hr_ok"] = 0.0
    for key, name in (("ci_lower", "lo_ok"), ("ci_upper", "hi_ok")):
        try:
            out[f"f_{name}"] = 1.0 if rel_err(float(pred[key]), float(gt[key])) <= 0.10 else 0.0
        except (KeyError, TypeError, ValueError):
            out[f"f_{name}"] = 0.0

    try:
        gp, pp = float(gt["p_value"]), float(pred["p_value"])
        if gp <= 0 and pp <= 0:
            out["f_pv_log10_ok"] = 1.0
        elif gp <= 0 or pp <= 0:
            out["f_pv_log10_ok"] = 0.0
        else:
            out["f_pv_log10_ok"] = 1.0 if abs(__import__("math").log10(pp) -
                                              __import__("math").log10(gp)) <= 1.0 else 0.0
    except (KeyError, TypeError, ValueError):
        out["f_pv_log10_ok"] = 0.0

    gt_rt = gt.get("at_risk_table") or []
    if not gt_rt:
        out["f_rt_timepoint_recall"] = -1.0  # N/A: not rendered on chart
        out["f_rt_cell_acc"] = -1.0
        return out
    pr_rt = {(round(float(r.get("timepoint", -1)), 2)): (r.get("counts") or {})
             for r in (pred.get("at_risk_table") or [])
             if isinstance(r, dict)}
    matched_cells = total_cells = matched_tp = 0
    for row in gt_rt:
        tp = round(float(row.get("timepoint", 0)), 2)
        gt_counts = row.get("counts") or {}
        tol = max(0.05 * tp, 0.5)
        near = next((k for k in pr_rt if abs(k - tp) <= tol), None)
        if near is not None:
            matched_tp += 1
            pr_counts = pr_rt[near]
            for arm, gv in gt_counts.items():
                total_cells += 1
                pv = pr_counts.get(arm)
                if pv is not None and int(pv) == int(gv):
                    matched_cells += 1
    n_gt_tp = len(gt_rt)
    out["f_rt_timepoint_recall"] = matched_tp / n_gt_tp if n_gt_tp else -1.0
    out["f_rt_cell_acc"] = matched_cells / total_cells if total_cells else -1.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v2_sft_run1/final")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoTokenizer

    from eval_inference import decompress_json, generate_extraction, load_model as load_macro
    from evaluation.metrics import score_extraction
    from evaluation.parse_output import extract_json_from_text

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = json.loads(MANIFEST.read_text(encoding="utf-8"))["categories"]["km"]
    if args.limit:
        ids = ids[: args.limit]

    done: dict[str, dict] = {}
    partial = OUT_DIR / "partial_current.jsonl"
    if partial.is_file():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r:
                done[r["chart"]] = r

    device = torch.device("cuda:0")
    processor = AutoProcessor.from_pretrained(
        "google/siglip2-so400m-patch14-384", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_macro(args.ckpt, device)

    img_dir = DATASET_ROOT / "train_v2" / "images" / "km"
    lbl_dir = DATASET_ROOT / "train_v2" / "labels" / "km"

    with open(partial, "a", encoding="utf-8") as pf:
        for i, cid in enumerate(ids):
            if cid in done:
                continue
            t0 = time.time()
            gt_path = lbl_dir / f"{cid}.json"
            img_path = img_dir / f"{cid}.png"
            if not gt_path.is_file() or not img_path.is_file():
                rec = {"chart": cid, "error": "missing_gt_or_image"}
                pf.write(json.dumps(rec) + "\n"); pf.flush()
                continue
            gt = json.loads(gt_path.read_text(encoding="utf-8", errors="replace"))
            rec_extra: dict = {}
            try:
                raw = generate_extraction(model, processor, tok, str(img_path),
                                          device, max_new_tokens=2816)
                parsed, _ = extract_json_from_text(raw)
                pred = decompress_json(parsed) if parsed else None
                legacy = score_extraction(gt, pred).to_dict() if pred \
                    else score_extraction(gt, "").to_dict()
                rec_extra = field_metrics(gt, pred)
            except Exception as exc:
                legacy = score_extraction(gt, "").to_dict()
                rec_extra = {"error": f"{type(exc).__name__}: {exc}"[:300],
                             **field_metrics(gt, None)}
            rec = {"chart": cid, "score": legacy,
                   "prediction": pred if isinstance(pred, dict) else None,
                   **rec_extra}
            pf.write(json.dumps(rec, ensure_ascii=False) + "\n"); pf.flush()
            print(f"[{i+1}/{len(ids)}] {cid} overall={legacy.get('overall_score'):.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)

    records = []
    for line in partial.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if "error" not in r or "score" in r:
                records.append(r)

    def mean(key: str) -> tuple[float, int]:
        vals = [r[key] for r in records if key in r and isinstance(r[key], (int, float))
                and r[key] >= 0]
        return (statistics.fmean(vals), len(vals)) if vals else (float("nan"), 0)

    summary: dict = {"arm": "v2_sft", "ckpt": args.ckpt, "n": len(records),
                     "timestamp_utc": utc()}
    for k in sorted({k for r in records for k in r if k.startswith(("f_",))}):
        m, n = mean(k)
        summary[k] = {"mean": round(m, 4), "n": n}
    overall = [r["score"]["overall_score"] for r in records if "score" in r]
    numeric = [r["score"].get("numeric_score", 0) for r in records if "score" in r]
    rmse = [r["score"].get("coordinate_rmse", 0) for r in records if "score" in r]
    censor = [r["score"].get("censoring_score", 0) for r in records if "score" in r]
    for name, arr in (("overall", overall), ("numeric", numeric),
                      ("rmse", rmse), ("censoring", censor)):
        if arr:
            summary[name] = {"mean": round(statistics.fmean(arr), 4),
                             "median": round(statistics.median(arr), 4)}

    stamp = utc()
    (OUT_DIR / f"summary_{stamp}.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "latest_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / f"eval_{stamp}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    print("\n=== validation_v2 summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
