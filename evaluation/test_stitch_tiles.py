"""Unit tests for tile stitching middleware."""

import json
import unittest

from scripts.stitch_tiles import (
    STITCH_VERSION,
    StitchError,
    _dedupe_censors_by_time,
    _dedupe_points_by_time,
    parse_tile_prediction,
    stitch_arm_tiles,
    stitch_chart_from_tiles,
    tile_pred_to_clinical,
)

META = {
    "coordinate_space": "normalized_local",
    "tile_origin": [100, 80],
    "plot_bbox": [100, 80, 700, 600],
    "axis_max": {"x": 50.0, "y": 1.0},
}


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
        pred = {"points": [0.0, 0.0, 1.0, 1.0], "censors": []}
        pts, cens = tile_pred_to_clinical(pred, META, tile_id="t1")
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0][0], 0.0, places=1)
        self.assertAlmostEqual(pts[1][0], 32.0, places=0)
        self.assertGreater(pts[0][1], pts[1][1])  # survival decreases left-to-right in time

    def test_tile_pred_out_of_bounds_raises(self):
        pred = {"points": [1.5, 0.2], "censors": []}
        with self.assertRaises(StitchError):
            tile_pred_to_clinical(pred, META, tile_id="t-bad")

    def test_stitch_arm_tiles_merges_overlap(self):
        rec_a = {
            "prediction": {"points": [0.0, 1.0, 0.5, 0.5], "censors": []},
            "meta": META,
        }
        rec_b = {
            "prediction": {"points": [0.5, 0.5, 1.0, 0.0], "censors": []},
            "meta": META,
        }
        pts, cens, prov = stitch_arm_tiles([rec_a, rec_b])
        self.assertGreaterEqual(len(pts), 2)
        self.assertEqual(len(prov), 2)
        self.assertEqual(prov[0]["prediction_source"], "stage2_tile")


class TestPredictionRawFlow(unittest.TestCase):
    """Regression: records with _meta pre-attached must still have their
    prediction_raw parsed and stitched (the 2026-08 silent-discard bug)."""

    RAW = '{"arm_id": "Drug X", "points": [0.10, 0.20, 0.30, 0.40], "censors": [0.55, 0.60]}'

    def _rec(self, raw: str = RAW):
        return {"prediction_raw": raw, "_meta": dict(META), "arm_id": "Drug X"}

    def test_parse_tile_prediction_from_raw(self):
        pred, source = parse_tile_prediction(self._rec())
        self.assertIsNotNone(pred)
        self.assertEqual(source, "stage2_tile_raw")
        self.assertEqual(len(pred["points"]), 4)

    def test_raw_predictions_flow_through_stitch(self):
        recs = [self._rec(), self._rec()]
        out = stitch_chart_from_tiles(recs, strict=True)
        arm = out["arms"][0]
        self.assertGreater(len(arm["coordinates"]), 0, "stitched coordinates must not be empty")
        self.assertGreater(len(arm["censoring_ticks"]), 0, "stitched censors must not be empty")

    def test_predictions_actually_change_output(self):
        recs_a = [self._rec('{"arm_id": "Drug X", "points": [0.10, 0.20], "censors": []}')]
        recs_b = [self._rec('{"arm_id": "Drug X", "points": [0.90, 0.95], "censors": []}')]
        out_a = stitch_chart_from_tiles(recs_a, strict=True)
        out_b = stitch_chart_from_tiles(recs_b, strict=True)
        self.assertNotEqual(
            out_a["arms"][0]["coordinates"],
            out_b["arms"][0]["coordinates"],
            "different predictions must produce different stitched output",
        )

    def test_strict_raises_on_unparseable_prediction(self):
        recs = [self._rec("not json at all {{{")]
        with self.assertRaises(StitchError):
            stitch_chart_from_tiles(recs, strict=True)

    def test_strict_raises_on_missing_prediction(self):
        recs = [{"prediction_raw": "", "_meta": dict(META), "arm_id": "Drug X"}]
        with self.assertRaises(StitchError):
            stitch_chart_from_tiles(recs, strict=True)

    def test_lenient_skips_unparseable_and_records_it(self):
        recs = [self._rec("not json at all {{{"), self._rec()]
        out = stitch_chart_from_tiles(recs, strict=False)
        stitch_meta = out["_meta"]["stitch"]
        self.assertEqual(stitch_meta["skipped_tiles"], 1)
        self.assertFalse(stitch_meta["strict"])
        self.assertGreater(len(out["arms"][0]["coordinates"]), 0)

    def test_provenance_recorded(self):
        out = stitch_chart_from_tiles([self._rec()], strict=True)
        stitch_meta = out["_meta"]["stitch"]
        self.assertEqual(stitch_meta["version"], STITCH_VERSION)
        self.assertTrue(stitch_meta["strict"])
        rows = stitch_meta["arms"][0]["tiles"]
        self.assertEqual(rows[0]["prediction_source"], "stage2_tile_raw")
        self.assertEqual(rows[0]["points_in"], 2)
        self.assertEqual(rows[0]["censors_in"], 1)  # RAW fixture: 4 flat coords -> 2 pts, 2 floats -> 1 censor

    def test_censors_survive_inverse_transform(self):
        # Censor at tile x=0.5: px = 100 + 192 = 292; t = (292-100)/600 * 50 = 16.0
        recs = [self._rec('{"arm_id": "Drug X", "points": [0.1, 0.2], "censors": [0.50, 0.30]}')]
        out = stitch_chart_from_tiles(recs, strict=True)
        ticks = out["arms"][0]["censoring_ticks"]
        self.assertEqual(len(ticks), 1)
        self.assertAlmostEqual(ticks[0], 16.0, places=1)


class TestGroupJsonl(unittest.TestCase):
    def test_group_eval_jsonl_roundtrip(self, tmp_dir=None):
        import tempfile
        from pathlib import Path
        from scripts.stitch_tiles import group_eval_jsonl_by_chart

        with tempfile.TemporaryDirectory() as td:
            label_path = Path(td) / "tile_a.json"
            label_path.write_text(json.dumps({
                "arm_id": "Drug X", "points": [[0.1, 0.2]], "censors": [],
                "_meta": {**META, "source_chart": "chart_test_km"},
            }), encoding="utf-8")
            jsonl_path = Path(td) / "eval.jsonl"
            jsonl_path.write_text(json.dumps({
                "label": str(label_path),
                "prediction_raw": TestPredictionRawFlow.RAW,
            }) + "\n", encoding="utf-8")
            groups = group_eval_jsonl_by_chart(jsonl_path)
            self.assertIn("chart_test_km", groups)
            rec = groups["chart_test_km"][0]
            self.assertEqual(rec["arm_id"], "Drug X")
            self.assertIn("_meta", rec)


if __name__ == "__main__":
    unittest.main()
