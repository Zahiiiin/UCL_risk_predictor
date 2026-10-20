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
        self.assertEqual(o["models"]["model1"]["n_reactors"], 407)
        self.assertEqual(o["models"]["model2"]["n_reactors"], 368)

    def test_saved_test_metrics(self):
        with open(os.path.join(A.ART, "meta.json")) as fh:
            meta = json.load(fh)
        self.assertAlmostEqual(meta["models"]["model1"]["test"]["accuracy"], 75.61, places=1)
        self.assertAlmostEqual(meta["models"]["model2"]["test"]["accuracy"], 87.84, places=1)
        # F1 and AUC as reported in the paper (needs the pinned scikit-learn 1.8.0)
        m1 = meta["models"]["model1"]["test"]
        self.assertAlmostEqual(m1["f1_weighted"], 74.9, delta=0.1)
        self.assertAlmostEqual(m1["auc_ovr_weighted"], 86.5, delta=0.1)

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
