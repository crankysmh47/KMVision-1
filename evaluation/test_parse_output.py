"""Unit tests for Stage 2 JSON repair helpers."""

import unittest

from evaluation.parse_output import (
    extract_json_from_text,
    extract_stage2_json,
    repair_stage2_json,
)


class TestStage2JsonRepair(unittest.TestCase):
    def test_leading_comma_in_points(self):
        raw = '{"arm_id": "Drug A", "points": [, 0.299, 0.412, 0.350, 0.380], "censors": []}'
        obj, err = extract_stage2_json(raw)
        self.assertIsNone(err)
        self.assertEqual(obj["points"][:4], [0.299, 0.412, 0.350, 0.380])

    def test_trailing_comma(self):
        raw = '{"arm_id": "X", "points": [0.1, 0.2,], "censors": [0.5, 0.3,]}'
        obj, err = extract_stage2_json(raw)
        self.assertIsNone(err)
        self.assertEqual(len(obj["points"]), 2)

    def test_missing_censors_defaults_empty(self):
        raw = '{"arm_id": "X", "points": [0.1, 0.2, 0.3, 0.4]}'
        obj, err = extract_stage2_json(raw)
        self.assertIsNone(err)
        self.assertEqual(obj["censors"], [])

    def test_orphan_integer_dropped(self):
        raw = '{"arm_id": "X", "points": [9, 0.144, 0.112, 0.200, 0.180], "censors": []}'
        obj, err = extract_stage2_json(raw)
        self.assertIsNone(err)
        self.assertEqual(obj["points"][0], 0.144)

    def test_truncated_closes_brackets(self):
        raw = '{"arm_id": "X", "points": [0.1, 0.2, 0.3, 0.4'
        repaired = repair_stage2_json(raw)
        obj, err = extract_json_from_text(repaired)
        self.assertIsNone(err)
        self.assertIn("points", obj)

    def test_valid_unchanged(self):
        raw = '{"arm_id":"Drug A","points":[0.145,0.112,0.300,0.250],"censors":[0.500,0.300]}'
        obj, err = extract_stage2_json(raw)
        self.assertIsNone(err)
        self.assertEqual(obj["points"], [0.145, 0.112, 0.300, 0.250])


if __name__ == "__main__":
    unittest.main()
