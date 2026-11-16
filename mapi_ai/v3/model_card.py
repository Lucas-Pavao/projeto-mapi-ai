"""
Cartão do modelo v3 para a tela Modelo do front (v2 e v3 lado a lado). Só lê o que o treino e a avaliação gravaram:
models/v3/metadata.json e reports/v3/*.json. Nenhum número é recalculado aqui.

Saída: reports/v3/model_card.json (copiado para projeto-mapi-front/src/features/model/data/v3_model_card.json).
"""
import csv
import hashlib
import json

from mapi_ai.v3 import config as C

FILES = ["models/v3/metadata.json", "reports/v3/metrics.csv", "reports/v3/evaluation.json",
         "reports/v3/explanation_eval.json", "reports/v3/physical_comparison.json", "reports/v3/dataset_summary.json"]


def _metric(rows, h, model, part, inputs="com lacunas", prob="bruta"):
    for r in rows:
        if (int(r["horizon_h"]) == h and r["model"] == model and r["part"] == part and r["inputs"] == inputs
                and r["probability"].startswith(prob)):
            return {k: float(r[k]) for k in ("pr_auc", "roc_auc", "brier", "bss", "ece")}
    return None


def build() -> dict:
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    ev = json.loads((C.REPORTS_DIR / "evaluation.json").read_text())
    xai = json.loads((C.REPORTS_DIR / "explanation_eval.json").read_text())["v3"]
    phys = json.loads((C.REPORTS_DIR / "physical_comparison.json").read_text())["variants"]
    ds = json.loads((C.REPORTS_DIR / "dataset_summary.json").read_text())
    rows = list(csv.DictReader(open(C.REPORTS_DIR / "metrics.csv")))
    hp = C.PRIMARY_HORIZON
    chosen = meta["production"]["model"]
    horizons = []
    for h in C.HORIZONS:
        e = ev["horizons"][f"h{h}"]
        t = e["alerts_and_events"]["test"]
        horizons.append({
            "hours": h, "model": e["model"],
            "validation": {r["model"]: r["val_pr_auc"] for r in meta["horizons"][f"h{h}"]["selection"]["table"]},
            "test": _metric(rows, h, e["model"], "test"),
            "testComplete": _metric(rows, h, e["model"], "test", inputs="sem lacunas"),
            "testCalibrated": _metric(rows, h, e["model"], "test", prob="calibrada"),
            "events": {lvl: {"alertDaysPerPointYear": t[lvl]["alert_days_per_point_year"], "eventRecall": t[lvl]["event_recall"],
                             "rainEventRecall": t[lvl]["rain_event_recall"], "leadMedianH": t[lvl]["lead_median_h"],
                             "leadP10H": t[lvl]["lead_p10_h"], "falseAlertsPerEvent": t[lvl]["false_alerts_per_event"]}
                       for lvl in ("medium", "high")},
        })
    sha = {f: hashlib.sha256((C.ROOT / f).read_bytes()).hexdigest() for f in FILES if (C.ROOT / f).exists()}
    return {
        "modelVersion": meta["model_version"], "model": chosen, "horizonHours": hp, "tag": C.SYNTHETIC_TAG,
        "trainingData": meta["training_data"], "rainSource": meta["rain_source"],
        "features": meta["features"], "removedFromV2": meta["removed_from_v2"],
        "selectionCriterion": meta["selection_criterion"],
        "thresholds": {"medium": meta["thresholds"]["medium"], "high": meta["thresholds"]["high"],
                       "rule": meta["thresholds"]["rule"], "probability": meta["thresholds"]["probability"]},
        "calibration": {"adopted": meta["calibration"]["adopted"], "check": meta["calibration"]["check_second_half"]},
        "prior": {"piReal": meta["prior_correction"]["pi_real"], "piSint": meta["prior_correction"]["pi_sint"],
                  "source": meta["prior_correction"]["pi_real_source"]},
        "prevalenceContinuous": ds["continuous"]["prevalence_continuous"],
        "horizons": horizons,
        "explanations": {k: {"jaccard": v["jaccard_top5_shap_lime"], "spearman": v["spearman_shap_lime"],
                             "limeStability": v["lime_stability_10_seeds"], "deletionShap": v["deletion_k1"]["SHAP"],
                             "deletionRandom": v["deletion_k1"]["aleatório"],
                             "formulaRecovery": (v.get("formula_recovery") or {}).get("spearman_shap_vs_truth")}
                         for k, v in xai.items()},
        "physical": {k: {"testPrAuc": v["test_pr_auc_chosen"], "soilAbsentPrAuc": v.get("test_pr_auc_soil_absent_as_in_production")}
                     for k, v in phys.items()},
        "sources": sha, "attribution": C.ATTRIBUTION,
    }


if __name__ == "__main__":
    card = build()
    (C.REPORTS_DIR / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print("reports/v3/model_card.json")
