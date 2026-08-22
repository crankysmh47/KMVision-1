"""Unit tests for tile stitching middleware."""

import unittest

from scripts.stitch_tiles import (
    _dedupe_censors_by_time,
    _dedupe_points_by_time,
    stitch_arm_tiles,
    tile_pred_to_clinical,
)


class TestStitchTiles(unittest.TestCase):
    def test_dedupe_points_by_time(self):
        pts = [[0.0, 1.0], [0.1, 0.9], [0.2, 0.8], [0.21, 0.79]]
        out = _dedupe_points_by_time(pts, tol=0.05)
        self.assertEqual(len(out), 3)

    def test_dedupe_censors(self):
        cens = [[1.0, 0.5], [1.1, 0.48], [5.0, 0.3]]
        out = _dedupe_censors_by_time(cens, tol=0.15)
        self.assertEqual(len(out), 2)

    def test_tile_pred_to_clinical(self):
        meta = {
            "coordinate_space": "normalized_local",
            "tile_origin": [100, 80],
            "plot_bbox": [100, 80, 700, 600],
            "axis_max": {"x": 50.0, "y": 1.0},
        }
        pred = {"points": [0.0, 0.0, 1.0, 1.0], "censors": []}
        pts, cens = tile_pred_to_clinical(pred, meta)
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0][0], 0.0, places=1)
        self.assertAlmostEqual(pts[1][0], 32.0, places=0)
        self.assertGreater(pts[0][1], pts[1][1])  # survival decreases left-to-right in time

    def test_stitch_arm_tiles_merges_overlap(self):
        meta = {
            "coordinate_space": "normalized_local",
            "tile_origin": [100, 80],
            "plot_bbox": [100, 80, 700, 600],
            "axis_max": {"x": 50.0, "y": 1.0},
        }
        rec_a = {
            "prediction": {"points": [0.0, 1.0, 0.5, 0.5], "censors": []},
            "meta": meta,
        }
        rec_b = {
            "prediction": {"points": [0.5, 0.5, 1.0, 0.0], "censors": []},
            "meta": meta,
        }
        pts, cens = stitch_arm_tiles([rec_a, rec_b])
        self.assertGreaterEqual(len(pts), 2)


if __name__ == "__main__":
    unittest.main()
