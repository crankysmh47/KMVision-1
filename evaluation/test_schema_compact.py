"""Round-trip tests for compact chart schema."""

import json

from evaluation.metrics import score_extraction
from evaluation.schema_compact import compact_json_string, decompress_chart, minify_chart


def test_km_roundtrip():
    gt = {
        "chart_type": "kaplan_meier",
        "axes": {
            "x": {"label": "Time", "max_value": 50.0},
            "y": {"label": "Survival", "max_value": 1.0},
        },
        "arms": [
            {
                "treatment_label": "A",
                "coordinates": [[0, 1.0], [5, 1.0], [5, 0.8], [10, 0.8], [10, 0.5]],
                "censoring_ticks": [5.0, 10.0],
            }
        ],
    }
    compact = minify_chart(gt)
    assert len(compact["a"][0]["p"]) <= 4
    expanded = decompress_chart(compact)
    s1 = score_extraction(gt, gt)
    s2 = score_extraction(gt, expanded)
    assert s2.overall_score > 0.95


if __name__ == "__main__":
    test_km_roundtrip()
    print("schema_compact round-trip OK")
