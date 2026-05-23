"""Re-score a saved eval JSONL (e.g. after parse/repair fixes)."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import aggregate_scores, score_extraction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    scores = []
    out_path = args.output or args.jsonl_path.replace(".jsonl", "_rescored_summary.json")

    for line in open(args.jsonl_path, encoding="utf-8"):
        rec = json.loads(line)
        with open(rec["label"], encoding="utf-8") as f:
            gt = json.load(f)
        sc = score_extraction(gt, rec["prediction_raw"])
        rec["score"] = sc.to_dict()
        scores.append(sc)

    summary = aggregate_scores(scores)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
