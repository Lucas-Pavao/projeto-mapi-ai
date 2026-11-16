"""
SHAP global (contribuições exatas de árvore do XGBoost, pred_contribs) dos modelos escolhidos, na VALIDAÇÃO.
Pergunta: a previsão vem das estações vizinhas ou da tendência no próprio ponto?
Uso: python -m mapi_ai.rain.explain
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import features as F

GROUP_LABEL = {"ponto": "chuva no próprio ponto (até 5 km)", "vizinhas": "estações vizinhas (5 a 40 km)",
               "rede": "estado da rede", "relogio": "hora e mês", "open_meteo": "Open-Meteo (rodada da véspera)"}


def group(f):
    if f in F.POINT_FEATURES:
        return "ponto"
    if f in F.SECTOR_FEATURES or f in F.RING_FEATURES:
        return "vizinhas"
    if f in F.NET_FEATURES:
        return "rede"
    if f in F.OM_FEATURES:
        return "open_meteo"
    return "relogio"


def contribs(model, X):
    c = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return np.abs(c[..., :-1])                       # tira o termo de base; (n, [saídas,] entradas)


def run(n=20000):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sel = json.loads((C.REPORTS_DIR / "selection.json").read_text())["horizons"]
    df = D.load()
    v = df[df.part == "valid"].sample(n, random_state=C.SEED)
    out = {"amostra": "validação, %d linhas, semente %d" % (n, C.SEED), "metodo": "contribuições de árvore do XGBoost (pred_contribs), média do módulo", "horizons": {}}
    fig, axes = plt.subplots(1, len(C.HORIZONS), figsize=(16, 6))
    for ax, h in zip(axes, C.HORIZONS):
        name = sel[f"{h}h"]["escolhido"]
        m = joblib.load(C.MODELS_DIR / f"{name}.joblib")
        if not hasattr(m, "models"):
            out["horizons"][f"{h}h"] = {"modelo": name, "nota": "modelo sem árvore: não há SHAP"}
            continue
        res = {"modelo": name}
        for regime, mask in (("todas", np.ones(len(v), bool)), ("ja_chove", v.raining.to_numpy()), ("ainda_nao_chove", ~v.raining.to_numpy())):
            X = v.loc[mask, m.features]
            cq = contribs(m.models[("q", h)], X)
            q80 = cq[:, 1, :].mean(axis=0) if cq.ndim == 3 else cq.mean(axis=0)
            cf = contribs(m.models[("FORTE", h)], X).mean(axis=0)
            r = {}
            for label, vals in (("quantil_80_mm", q80), ("prob_FORTE_logit", cf)):
                s = pd.Series(vals, index=m.features)
                g = s.groupby([group(f) for f in m.features]).sum()
                r[label] = {"por_grupo_pct": (100 * g / g.sum()).round(1).to_dict(),
                            "top10": s.sort_values(ascending=False).head(10).round(4).to_dict()}
            res[regime] = r
        out["horizons"][f"{h}h"] = res
        s = pd.Series(res["todas"]["quantil_80_mm"]["top10"]).iloc[::-1]
        colors = {"ponto": "#1f6feb", "vizinhas": "#d97706", "rede": "#6b7280", "relogio": "#9ca3af", "open_meteo": "#059669"}
        ax.barh(s.index, s.values, color=[colors[group(f)] for f in s.index])
        ax.set_title(f"{h} h — {name}"); ax.set_xlabel("média de |SHAP| no quantil de 80% (mm)")
    fig.suptitle("Previsão de chuva: de onde vem a previsão (azul = próprio ponto, laranja = vizinhas, verde = Open-Meteo)\n" + C.ATTRIBUTION, fontsize=10)
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "shap_global.png", dpi=120); plt.close(fig)
    (C.REPORTS_DIR / "shap_global.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run()
    for h, r in o["horizons"].items():
        for reg in ("todas", "ja_chove", "ainda_nao_chove"):
            print(h, r["modelo"], reg, r[reg]["quantil_80_mm"]["por_grupo_pct"], "| FORTE", r[reg]["prob_FORTE_logit"]["por_grupo_pct"])
        print("  top", list(r["todas"]["quantil_80_mm"]["top10"].items())[:6])
