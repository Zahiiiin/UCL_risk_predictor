"""
Basic checks for the UCL Risk Predictor.
Run from the project folder with:   python -m unittest -v
"""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app as A  # noqa: E402

M1 = {"model": "1", "type": "PWR", "country": "USA", "net_capacity_mw": 1100, "age_years": 30,
      "construction_years": 6, "nsss_group": "Western_PWR", "model_family": "Other"}
M2 = dict(M1, model="2", ucl_p1=1, ucl_p2=0.8)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.client = A.app.test_client()

    def post(self, payload):
        return self.client.post("/api/predict", json=payload)

    def test_page_and_health(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        page.close()
        self.assertEqual(self.client.get("/healthz").get_json()["status"], "ok")

    def test_options(self):
        o = self.client.get("/api/options").get_json()
        self.assertEqual(len(o["countries"]), 31)
        self.assertEqual({t["value"] for t in o["types"]}, {"PWR", "BWR", "PHWR", "GCR", "LWGR", "FBR"})
        self.assertEqual(o["models"]["model1"]["n_reactors"], 387)
        self.assertEqual(o["models"]["model2"]["n_reactors"], 324)

    def test_saved_test_metrics(self):
        with open(os.path.join(A.ART, "meta.json")) as fh:
            meta = json.load(fh)
        m1, m2 = meta["models"]["model1"], meta["models"]["model2"]
        # cross-validated values; the benchmark (research/expected_results/benchmark) gives
        # LR 66.0 for Model 1 and RF 71.0 for Model 2
        self.assertAlmostEqual(m1["test"]["accuracy"], 66.0, delta=1.0)
        self.assertAlmostEqual(m2["test"]["accuracy"], 71.0, delta=1.0)
        self.assertAlmostEqual(m1["test"]["baselines"]["country_majority"], 69.0, delta=0.5)
        self.assertAlmostEqual(m2["test"]["baselines"]["country_majority"], 69.3, delta=0.5)
        for m in (m1, m2):
            cm = m["test"]["confusion_matrix"]
            self.assertEqual([sum(r) for r in cm], [m["class_counts"][c] for c in ("Low", "Medium", "High")])
            self.assertEqual(m["test"]["n_test"], m["n_reactors"])

    def test_predictions_are_valid(self):
        for payload, n_drivers in ((M1, 6), (M2, 7)):
            r = self.post(payload)
            self.assertEqual(r.status_code, 200)
            d = r.get_json()
            self.assertIn(d["predicted"], ["Low", "Medium", "High"])
            self.assertAlmostEqual(sum(d["probabilities"].values()), 1.0, places=2)
            self.assertEqual(len(d["drivers"]), n_drivers)
            self.assertEqual(d["predicted"], max(d["probabilities"], key=d["probabilities"].get))

    def test_history_changes_the_answer(self):
        good = self.post(dict(M2, ucl_p1=0.5, ucl_p2=0.4)).get_json()
        poor = self.post(dict(M2, ucl_p1=15, ucl_p2=18)).get_json()
        self.assertGreater(poor["probabilities"]["High"], good["probabilities"]["High"])

    def test_bad_input_is_rejected(self):
        self.assertEqual(self.post(dict(M1, net_capacity_mw=5000)).status_code, 400)
        self.assertEqual(self.post(dict(M1, type="XYZ")).status_code, 400)
        self.assertEqual(self.post(dict(M1, age_years="old")).status_code, 400)
        self.assertEqual(self.post(dict(M1, model="3")).status_code, 400)
        self.assertEqual(self.post(dict(M2, ucl_p1=None)).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", data="not json").status_code, 400)

    def test_soft_warnings(self):
        d = self.post(dict(M1, country="NORWAY", net_capacity_mw=1900)).get_json()
        fields = {w["field"] for w in d["warnings"]}
        self.assertEqual(fields, {"country", "net_capacity_mw"})
        d = self.post(dict(M2, model_family="EPR")).get_json()
        self.assertIn("model_family", {w["field"] for w in d["warnings"]})


if __name__ == "__main__":
    unittest.main()
