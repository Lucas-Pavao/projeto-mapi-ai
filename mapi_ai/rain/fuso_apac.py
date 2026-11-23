"""
Teste de fuso da APAC meteorologia (onda 16, Fase 7.1): correlação cruzada da chuva de 10 min de uma estação que aparece na
APAC (horário sem fuso declarado) e na PED do Cemaden (UTC), com defasagens de -4 h a +4 h, em 3 a 5 eventos de chuva.

Regra escrita antes de rodar: evento = hora com >= 5 mm na PED (horas a menos de 3 h uma da outra são um evento); a defasagem escolhida é o pico da correlação somada nos
eventos; confirma o fuso X se o pico cair em -X h (± 10 min) em pelo menos 3 eventos e a correlação no pico for >= 0,8.
Sem 3 eventos com chuva nas duas fontes: inconclusivo (a fonte continua em quarentena).
Uso: python -m mapi_ai.rain.fuso_apac <apac.csv: ts_texto,mm> <ped.csv: ts_utc,mm>
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

LAGS = range(-24, 25)            # passos de 10 min: -4 h a +4 h
EVENTO_MM_H = 5.0
MIN_EVENTOS = 3
MIN_R = 0.8


def serie(df: pd.DataFrame, col_ts: str, tz: str | None) -> pd.Series:
    t = pd.to_datetime(df[col_ts])
    t = t.dt.tz_localize(tz) if tz else t.dt.tz_localize("UTC")
    return pd.Series(df.mm.to_numpy(dtype=float), index=t.dt.tz_convert("UTC")).sort_index()


def corr_lag(a: pd.Series, b: pd.Series, lag: int) -> float:
    """Correlação entre a (APAC lida como UTC) deslocada de lag passos e b (PED, UTC)."""
    x = a.copy()
    x.index = x.index + pd.Timedelta(minutes=10 * lag)
    j = pd.concat([x, b], axis=1, join="inner").dropna()
    if len(j) < 6 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return float("nan")
    return float(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])


def testa(apac: pd.Series, ped: pd.Series) -> dict:
    hora = ped.resample("1h").sum(min_count=1)
    chuvosas = hora[hora >= EVENTO_MM_H]
    # horas com chuva a menos de 3 h uma da outra são o mesmo evento; o evento fica na hora de maior chuva
    grupo = (chuvosas.index.to_series().diff() > pd.Timedelta(hours=3)).cumsum()
    eventos = [g.idxmax() for _, g in chuvosas.groupby(grupo.to_numpy())]
    out = {"eventos": [], "regra": __doc__.strip()}
    for e in eventos:
        a = apac[(apac.index > e - pd.Timedelta(hours=8)) & (apac.index <= e + pd.Timedelta(hours=8))]
        b = ped[(ped.index > e - pd.Timedelta(hours=4)) & (ped.index <= e + pd.Timedelta(hours=4))]
        rs = {lag: corr_lag(a, b, lag) for lag in LAGS}
        rs = {k: v for k, v in rs.items() if np.isfinite(v)}
        if not rs:
            continue
        best = max(rs, key=rs.get)
        out["eventos"].append({"hora": str(e), "defasagem_min": best * 10, "r": round(rs[best], 3)})
    ok = [e for e in out["eventos"] if e["r"] >= MIN_R]
    out["n_eventos"] = len(out["eventos"])
    if len(out["eventos"]) < MIN_EVENTOS:
        out["conclusao"] = "INCONCLUSIVO: menos de 3 eventos com chuva nas duas fontes"
        return out
    lags = pd.Series([e["defasagem_min"] for e in ok])
    moda = int(lags.mode().iloc[0]) if len(lags) else None
    concordam = int((lags.sub(moda).abs() <= 10).sum()) if moda is not None else 0
    out["defasagem_moda_min"] = moda
    if moda is not None and concordam >= MIN_EVENTOS:
        out["conclusao"] = (f"FUSO CONFIRMADO: a APAC lida como UTC precisa andar {moda} min para casar com a PED "
                            f"({concordam} de {len(out['eventos'])} eventos, r >= {MIN_R}); fuso = UTC{-moda / 60:+.0f} h")
    else:
        out["conclusao"] = "NÃO CONFIRMADO: os eventos não concordam numa defasagem; descartar a fonte com esta evidência"
    return out


if __name__ == "__main__":
    a = serie(pd.read_csv(sys.argv[1]), "ts", None)
    b = serie(pd.read_csv(sys.argv[2]), "ts", None)
    print(json.dumps(testa(a, b), ensure_ascii=False, indent=1))
