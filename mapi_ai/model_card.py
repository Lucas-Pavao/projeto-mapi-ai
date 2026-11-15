"""
Cartão do modelo em produção (xgb_v2) para a tela "Modelo" do front.

Lê só arquivos já gerados pelo pipeline (nenhum número é digitado aqui) e grava reports/model_card.json, com o
SHA-256 de cada fonte. O front copia o arquivo para src/features/model/data/xgb_v2_model_card.json.
Uso: python -m mapi_ai.model_card
"""
import hashlib
import json

import pandas as pd

from mapi_ai import in1165_config as C

NAMES = {
    "nunca_alaga": "Nunca alaga", "logistica": "Regressão logística", "arvore_rasa": "Árvore rasa",
    "xgb_v2": "xgb_v2 (em produção)", "xgb_v2_calibrado": "xgb_v2 calibrado", "xgb_v2_sem_local": "xgb_v2 sem lat/lon",
    "xgb_v2_reanalise": "xgb_v2 com reanálise",
}
ABLATIONS = {
    ("xgb_v2", "xgb_v2_reanalise"): "Chuva medida (Cemaden) × reanálise",
    ("xgb_v2", "xgb_v2_sem_local"): "Com × sem latitude/longitude",
    ("xgb_v2", "logistica"): "xgb_v2 × regressão logística",
}
LABELS = {
    "rainfall_3h_accumulated": "Chuva acumulada em 3 horas", "rainfall_24h_accumulated": "Chuva acumulada em 24 horas",
    "current_rainfall": "Chuva atual", "tide_level": "Nível da maré", "rainfall_12h_accumulated": "Chuva acumulada em 12 horas",
    "rainfall_6h_accumulated": "Chuva acumulada em 6 horas",
}


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    files = {
        "metrics": C.REPORTS_DIR / "metrics.csv", "significance": C.REPORTS_DIR / "significance.csv",
        "explanations": C.REPORTS_DIR / "explanation_eval.json", "xai": C.REPORTS_DIR / "xai_cases.json",
        "metadata": C.MODELS_DIR / "metadata.json", "dataset": C.REPORTS_DIR / "dataset_summary.json",
    }
    met = pd.read_csv(files["metrics"])
    sig = pd.read_csv(files["significance"]).set_index(["a", "b"])
    ev = json.loads(files["explanations"].read_text())
    xai = json.loads(files["xai"].read_text())
    meta = json.loads(files["metadata"].read_text())
    ds = json.loads(files["dataset"].read_text())["dataset"]
    t = {r["metrica"]: r["media"] for r in ev["table"]}
    rows = []
    for _, r in met.iterrows():
        if r["modelo"].startswith("regra_chuva3h>"):
            name = "Regra: chuva 3 h > " + r["modelo"].split(">")[1].replace("mm", " mm")
            rid = "regra"
        else:
            name, rid = NAMES.get(r["modelo"], r["modelo"]), r["modelo"]
        rows.append({"id": rid, "name": name, "prAuc": round(float(r["pr_auc"]), 4),
                     "rocAuc": round(float(r["roc_auc"]), 4), "brier": round(float(r["brier"]), 4)})
    abl = [{"label": lbl, "delta": round(float(sig.loc[k, "delta_pr_auc"]), 4), "lo": round(float(sig.loc[k, "ic95_lo"]), 4),
            "hi": round(float(sig.loc[k, "ic95_hi"]), 4)} for k, lbl in ABLATIONS.items()]
    return {
        "modelVersion": meta["model_version"], "trainingData": meta["training_data"], "rainSource": meta["rain_source"],
        "tag": C.SYNTHETIC_TAG,
        "synthetic": {"n": ds["rows"], "positiveRate": ds["positive_rate"], "seed": C.SEED},
        "thresholds": {"medium": meta["thresholds"]["medium"], "high": meta["thresholds"]["high"],
                       "criteria": meta["thresholds"]["criteria"]},
        "calibration": {"used": meta["calibration"]["used"], "valEceRaw": round(meta["calibration"]["val_ece_raw"], 4),
                        "criterion": "aplicar se ECE de validação > 0,05"},
        "split": meta["split"], "rows": rows, "ablations": abl,
        "xai": {"top5": [LABELS.get(f, f) for f in list(xai["global_shap_mean_abs"])[:5]],
                "jaccard": t["concordancia_jaccard_top5_shap_lime"], "spearman": t["concordancia_spearman_shap_lime"],
                "lime10": t["estabilidade_lime_jaccard_top5_10_sementes"], "delShap": t["delecao_queda_prob_SHAP_k1"],
                "delLime": t["delecao_queda_prob_LIME_k1"], "delRandom": t["delecao_queda_prob_aleatório_k1"],
                "planBSpearman": round(ev["plan_b"]["spearman_shap_vs_truth"], 4)},
        "sources": {p.name: _sha(p) for p in files.values()},
        "attribution": C.ATTRIBUTION,
    }


if __name__ == "__main__":
    card = build()
    out = C.REPORTS_DIR / "model_card.json"
    out.write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"{out} ({len(card['rows'])} modelos)")
