"""Grava models/v4/rain/metadata.json a partir da seleção e da avaliação. Uso: python -m mapi_ai.rain.metadata"""
import hashlib
import json
import subprocess

import pandas as pd
import sklearn
import xgboost

from mapi_ai.rain import config as C
from mapi_ai.rain import features as F


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    ev = json.loads((C.REPORTS_DIR / "evaluation.json").read_text())
    info = json.loads((C.MODELS_DIR / "train_info.json").read_text())
    hz = {}
    for h in C.HORIZONS:
        H = ev["horizons"][f"{h}h"]
        m = H["escolhido"]
        t = H["teste"][m]
        hz[f"{h}h"] = {"model": m, "artifact": f"models/v4/rain/{m}.joblib",
                       "features": F.ALL_FEATURES if m == "gb_rede_om" else F.BASE_FEATURES if m == "gb_rede" else "chave discreta (models.py)",
                       "validation": H["validacao"], "alert_threshold_FORTE": H["limiar_aviso_FORTE"][m]["limiar"],
                       "test": {"pinball": t["todas"]["pinball"], "coverage": {q: t["todas"][f"cobertura_{q}"] for q in ("q50", "q80", "q95")},
                                "FORTE": t["FORTE"]["todas"], "lead": t["FORTE"]["eventos"]}}
    meta = {"model_version": C.MODEL_VERSION, "target": "MEASURED", "training_data": "MEASURED",
            "target_note": "chuva medida do ponto (maior acumulado entre as estações válidas a até 5 km), em (t, t+H]",
            "attribution": C.ATTRIBUTION, "horizons_h": list(C.HORIZONS), "quantiles": list(C.QUANTILES),
            "thresholds_mm_h": C.THRESHOLDS, "selection_criterion": C.SELECTION, "alert_rule": C.ALERT_RULE,
            "split": {"train_end": str(C.TRAIN_END), "valid_end": str(C.VALID_END), "guard_h": 6, "data_start": str(C.DATA_START), "data_end": str(C.DATA_END)},
            "train": info, "seed": C.SEED, "xgb_params": {k: v for k, v in C.XGB.items()},
            "data_sha256": {p.name: sha(p) for p in (C.READINGS, C.STATIONS, C.OM_PREV, C.OM_HIST)},
            "open_meteo": "rodada da véspera (previous-runs-api, precipitation_previous_day1): emitida pelo menos 24 h antes da hora prevista",
            "versions": {"xgboost": xgboost.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__},
            "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.ROOT).stdout.strip(),
            "horizons": hz}
    (C.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    print(json.dumps({h: (v["model"], v["alert_threshold_FORTE"]) for h, v in run()["horizons"].items()}))
