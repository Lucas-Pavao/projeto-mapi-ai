"""
Previsão de chuva FORA DA AMOSTRA para os 5 anos, para treinar o modelo de alagamento (Fase 4) com a previsão que
teria sido emitida, nunca com a chuva futura verdadeira.
- validação e teste: o modelo final (treinado só no treino) já é fora da amostra;
- treino: 4 blocos contíguos no tempo; cada bloco é previsto por um modelo ajustado nos outros três, com folga de 6 h.
  O modelo de um bloco pode ter visto anos POSTERIORES ao bloco, mas nunca o próprio bloco (out-of-fold em blocos).
Saída horária (instantes de hora cheia): data/v4/rain_forecast_oof.csv.gz. Uso: python -m mapi_ai.rain.oof
"""
import json

import joblib
import numpy as np
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import models as M

OUT = C.ROOT / "data" / "v4" / "rain_forecast_oof.csv.gz"
K = 4


def chosen():
    sel = json.loads((C.REPORTS_DIR / "selection.json").read_text())["horizons"]
    return {h: sel[f"{h}h"]["escolhido"] for h in C.HORIZONS}


def run():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ch = chosen()
    df = D.load()
    hourly = df[df.t.dt.minute == 0]
    tr = df[df.part == "train"]
    edges = pd.date_range(tr.t.min(), tr.t.max(), periods=K + 1)
    parts = []
    held = hourly[hourly.part != "train"]
    final = {n: joblib.load(C.MODELS_DIR / f"{n}.joblib") for n in set(ch.values())}
    p = pd.concat([final[ch[h]].predict(held)[M.out_columns(h)] for h in C.HORIZONS], axis=1)
    parts.append(pd.concat([held[["slug", "t", "part"]], p], axis=1).assign(fold="final"))
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        inside = (tr.t >= lo) & (tr.t <= hi) if k == K - 1 else (tr.t >= lo) & (tr.t < hi)
        fit = tr[(tr.t < lo - C.GUARD) | (tr.t > hi + C.GUARD)].iloc[::C.TRAIN_STEP]
        blk = tr[inside & (tr.t.dt.minute == 0)]
        ms = {n: M.make(n).fit(fit) for n in set(ch.values())}
        p = pd.concat([ms[ch[h]].predict(blk)[M.out_columns(h)] for h in C.HORIZONS], axis=1)
        parts.append(pd.concat([blk[["slug", "t", "part"]], p], axis=1).assign(fold=f"bloco{k + 1}"))
        print("bloco", k + 1, lo, hi, len(fit), len(blk), flush=True)
    out = pd.concat(parts).sort_values(["slug", "t"])
    out["t"] = out.t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    num = out.select_dtypes("number").columns
    out[num] = out[num].astype(float).round(4)
    out.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
    return out


if __name__ == "__main__":
    o = run()
    print(len(o), o.fold.value_counts().to_dict())
