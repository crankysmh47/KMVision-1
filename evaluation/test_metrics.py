"""Smoke tests for chart extraction metrics (run: python -m evaluation.test_metrics)."""

import json

from evaluation.metrics import aggregate_scores, score_extraction


SAMPLE_GT = {
    "chart_type": "kaplan_meier",
    "axes": {
        "x": {"label": "Time (Months)", "max_value": 50.0},
        "y": {"label": "Survival Probability", "max_value": 1.0},
    },
    "arms": [
        {
            "treatment_label": "Treatment A",
            "coordinates": [[0.0, 1.0], [10.0, 0.8], [20.0, 0.5]],
            "censoring_ticks": [10.0, 20.0],
        },
        {
            "treatment_label": "Treatment B",
            "coordinates": [[0.0, 1.0], [10.0, 0.9], [20.0, 0.7]],
            "censoring_ticks": [15.0],
        },
    ],
}


def test_perfect_match():
    pred = json.dumps(SAMPLE_GT)
    score = score_extraction(SAMPLE_GT, pred)
    assert score.json_valid
    assert score.overall_score > 0.9
    assert score.fields_correct == score.fields_total


def test_parse_markdown_fence():
    pred = "```json\n" + json.dumps(SAMPLE_GT) + "\n```"
    score = score_extraction(SAMPLE_GT, pred)
    assert score.json_valid
    assert score.overall_score > 0.9


def test_partial_numeric_drift():
    pred = json.loads(json.dumps(SAMPLE_GT))
    pred["arms"][0]["coordinates"][1][1] = 0.75  # survival off by 0.05
    score = score_extraction(SAMPLE_GT, pred)
    assert score.json_valid
    assert 0.5 < score.overall_score < 1.0
    assert score.coordinate_rmse is not None


def test_wrong_arm_count():
    pred = json.loads(json.dumps(SAMPLE_GT))
    pred["arms"] = pred["arms"][:1]
    score = score_extraction(SAMPLE_GT, pred)
    assert score.arms_matched == 1
    assert score.overall_score < 0.95


def test_invalid_json():
    score = score_extraction(SAMPLE_GT, "not json at all")
    assert not score.json_valid
    assert score.overall_score == 0.0


def test_aggregate():
    scores = [
        score_extraction(SAMPLE_GT, json.dumps(SAMPLE_GT)),
        score_extraction(SAMPLE_GT, "garbage"),
    ]
    summary = aggregate_scores(scores)
    assert summary["count"] == 2
    assert summary["json_valid_rate"] == 0.5


if __name__ == "__main__":
    test_perfect_match()
    test_parse_markdown_fence()
    test_partial_numeric_drift()
    test_wrong_arm_count()
    test_invalid_json()
    test_aggregate()
    print("All evaluation metric smoke tests passed.")
