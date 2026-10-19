"""
Writes tools/reference_cases.json: random reactor inputs together with the
answers the Python server gives for them. check_local_engine.mjs then asks the
in-browser engine the same questions and compares.

Run from the project folder:   python tools/make_reference.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app as A  # noqa: E402


def main(n_per_model=300, seed=2024):
    rng = random.Random(seed)
    client = A.app.test_client()
    opt = client.get("/api/options").get_json()
    countries = [c["value"] for c in opt["countries"]] + ["OTHER", "OTHER", "NOWHERE"]
    types = [t["value"] for t in opt["types"]]
    groups = [g["value"] for g in opt["nsss_groups"]]
    fams = [f["value"] for f in opt["families"]]

    cases = []
    for model in (1, 2):
        for i in range(n_per_model):
            wild = i % 15 == 0  # now and then, deliberately invalid or extreme values
            lo_c, hi_c = (10, 3000) if wild else (50, 2500)
            payload = {
                "model": model,
                "type": rng.choice(types),
                "country": rng.choice(countries),
                "nsss_group": rng.choice(groups),
                "model_family": rng.choice(fams),
                "net_capacity_mw": round(rng.uniform(lo_c, hi_c), rng.choice([0, 1, 2])),
                "age_years": round(rng.uniform(0, 85 if wild else 80), rng.choice([0, 0, 1])),
                "construction_years": round(rng.uniform(1, 60), rng.choice([0, 1])),
            }
            if model == 2:
                payload["ucf_p1"] = round(rng.uniform(0, 100), rng.choice([0, 1, 2]))
                payload["ucf_p2"] = round(rng.uniform(0, 100), rng.choice([0, 1, 2]))
            try:
                out = {"ok": True, "json": A.run_prediction(payload)}
            except A.InputError as exc:
                out = {"ok": False, "json": {"error": str(exc)}}
            cases.append({"payload": payload, "expected": out})

    # a few hand-made edge cases
    edge = [
        {"model": 1, "type": "PWR", "country": "USA", "nsss_group": "Western_PWR", "model_family": "Other",
         "net_capacity_mw": "950", "age_years": "38", "construction_years": "6"},
        {"model": 2, "type": "PWR", "country": "FRANCE", "nsss_group": "French_PWR", "model_family": "Other",
         "net_capacity_mw": 1300, "age_years": 5, "construction_years": 6, "ucf_p1": 100, "ucf_p2": 0},
        {"model": 2, "type": "PWR", "country": "", "nsss_group": "French_PWR", "model_family": "Other",
         "net_capacity_mw": 1300, "age_years": 0, "construction_years": 6, "ucf_p1": "", "ucf_p2": 50},
        {"model": 3, "type": "PWR"},
    ]
    for payload in edge:
        try:
            out = {"ok": True, "json": A.run_prediction(payload)}
        except A.InputError as exc:
            out = {"ok": False, "json": {"error": str(exc)}}
        cases.append({"payload": payload, "expected": out})

    path = ROOT / "tools" / "reference_cases.json"
    path.write_text(json.dumps(cases))
    n_err = sum(1 for c in cases if not c["expected"]["ok"])
    print(f"Wrote {len(cases)} cases ({n_err} of them invalid inputs) to {path.name}")


if __name__ == "__main__":
    main()
