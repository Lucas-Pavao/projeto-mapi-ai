"""
Treina os cinco modelos no treino e grava as previsões de validação e teste. Uso: python -m mapi_ai.rain.train
O open_meteo (e a entrada om_* do gb_rede_om) usa a rodada da VÉSPERA: emitida antes de t com certeza.
"""
import json

import joblib
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import models as M

PRED = C.DATA_DIR / "predictions.pkl.gz"


def run():
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = D.load()
    train = df[df.part == "train"].iloc[::C.TRAIN_STEP]
    held = df[df.part.isin(["valid", "test"])]
    preds = {}
    for name in C.MODELS:
        m = M.make(name).fit(train)
        joblib.dump(m, C.MODELS_DIR / f"{name}.joblib", compress=3)
        preds[name] = m.predict(held)
        print(name, "ok", flush=True)
    keep = ["slug", "t", "part", "raining", "p_10", "p_60"] + [c for c in held if c[:2] in ("y_", "ev") or c[:5] == "peak_"]
    pd.to_pickle({"rows": held[keep], "pred": preds}, PRED)
    info = {"train_rows": len(train), "train_from": str(train.t.min()), "train_to": str(train.t.max()),
            "best_iteration": {n: {f"{k}_{h}h": int(m.best_iteration) for (k, h), m in joblib.load(C.MODELS_DIR / f"{n}.joblib").models.items()}
                               for n in ("gb_rede", "gb_rede_om")}}
    (C.MODELS_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
