"""
Referência da conferência semanal do rain_v1 (onda 16, Fase 1.1): a tabela da persistência (faixa da chuva da última hora no
ponto -> q50, q80, q95 por prazo, ajustada no treino) e o ganho do modelo escolhido sobre a persistência no TESTE, com IC
95% por bootstrap de dia (o mesmo bootstrap do relatório do rain_v1). Nada é reajustado: lê models/v4/rain/persistencia.joblib
e data/rain/predictions.pkl.gz (previsões fora da amostra) e grava reports/v4/rain/referencia_conferencia_semanal.json.

Ganho (skill) = 1 - pinball(modelo) / pinball(persistência), pinball = média dos três quantis (como no relatório).
Uso: python -m scripts.exportar_referencia_conferencia_rain
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import evaluate as E

ROOT = Path(__file__).resolve().parent.parent
OUT = C.REPORTS_DIR / "referencia_conferencia_semanal.json"
N_BOOT = 2000
SEED = 20261122


def skill_ci(loss_m: np.ndarray, loss_p: np.ndarray, blocks: np.ndarray) -> tuple[float, float, float]:
    ok = ~np.isnan(loss_m) & ~np.isnan(loss_p)
    d = pd.DataFrame({"m": loss_m[ok], "p": loss_p[ok], "b": blocks[ok]}).groupby("b").agg(m=("m", "sum"), p=("p", "sum"))
    m, p = d.m.to_numpy(), d.p.to_numpy()
    idx = np.random.default_rng(SEED).integers(0, len(m), size=(N_BOOT, len(m)))
    s = 1 - m[idx].sum(axis=1) / p[idx].sum(axis=1)
    return float(1 - m.sum() / p.sum()), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))


def main() -> dict:
    pers = joblib.load(C.MODELS_DIR / "persistencia.joblib")
    ev = json.loads((C.REPORTS_DIR / "evaluation.json").read_text())
    pk = pd.read_pickle(C.DATA_DIR / "predictions.pkl.gz")
    rows, pred = pk["rows"], pk["pred"]
    test = (rows.part == "test").to_numpy()
    out = {
        "gerado_por": "scripts/exportar_referencia_conferencia_rain.py",
        "modelo": C.MODEL_VERSION,
        "faixas_p60_mm": {"seco": "p_60 <= 0", "0-1": "0 < p_60 <= 1", "1-5": "1 < p_60 <= 5", ">=5": "p_60 > 5",
                          "sem_dado": "p_60 ausente (usa a distribuição global do treino)"},
        "persistencia": {}, "teste": {},
        "sha256_persistencia_joblib": hashlib.sha256((C.MODELS_DIR / "persistencia.joblib").read_bytes()).hexdigest(),
    }
    for h in C.HORIZONS:
        t = pers.table[h]
        out["persistencia"][f"{h}h"] = {
            **{b: {q: round(float(t.loc[b, f"{q}_{h}h"]), 4) for q in ("q50", "q80", "q95")} for b in t.index},
            "sem_dado": {q: round(float(pers.glob[h][f"{q}_{h}h"]), 4) for q in ("q50", "q80", "q95")},
        }
        chosen = ev["horizons"][f"{h}h"]["escolhido"]
        r = rows[test]
        lm = E.row_loss(r, pred[chosen][test], h)
        lp = E.row_loss(r, pred["persistencia"][test], h)
        s, lo, hi = skill_ci(lm, lp, E.day_blocks(r))
        out["teste"][f"{h}h"] = {"modelo_escolhido": chosen, "pinball_modelo": round(float(np.nanmean(lm)), 4),
                                 "pinball_persistencia": round(float(np.nanmean(lp)), 4),
                                 "ganho": round(s, 4), "ganho_ic95": [round(lo, 4), round(hi, 4)]}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print(json.dumps(main()["teste"], indent=1, ensure_ascii=False))
