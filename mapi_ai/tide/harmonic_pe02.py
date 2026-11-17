"""
Maré prevista do Porto do Recife (pe02) por análise harmônica, para 2027 (operação O3 do prompt do alerta de 20 min).
Uso: python -m mapi_ai.tide.harmonic_pe02

A ficha F-41 (estação 30645) não traz constantes harmônicas: só a descrição da estação, o Z0 = 128,2 cm e as RRNN
(projeto-mapi-api/docs/extratos/f41-porto-recife.md). Por isso as constantes são estimadas com o UTide (Codiga, 2011)
sobre os extremos oficiais da DHN de 2021-10 a 2024-12 (CSVs extraídos dos PDFs, projeto-mapi-api/src/main/resources/tide),
e a previsão é validada contra as preamares e baixa-mares de 2025 e 2026, que NÃO entram no ajuste.
Fonte nova TIDE_HARMONIC_F41: gravada, NÃO ligada. A troca de fonte da maré é decisão do Lucas.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import utide

ROOT = Path(__file__).resolve().parents[2]
API_TIDE = ROOT.parent / "projeto-mapi-api" / "src" / "main" / "resources" / "tide"
OUT_DATA = ROOT / "data" / "tide"
OUT_REPORT = ROOT / "reports" / "tide"
LAT = -(8 + 3.4 / 60)          # F-41: 08º 03'.4 S
FIT_FROM, FIT_TO = "2021-10-01", "2025-01-01"
# Lista explícita: com amostragem irregular (só extremos) o critério automático do UTide não escolhe nenhuma.
# 3,25 anos separam todas pelo critério de Rayleigh (SA e SSA incluídas: o ajuste cobre mais de 1 ano).
CONSTIT = ["SA", "SSA", "MM", "MSF", "MF", "2Q1", "Q1", "O1", "NO1", "P1", "K1", "J1", "OO1", "2N2", "MU2", "N2", "NU2",
           "M2", "L2", "T2", "S2", "K2", "MO3", "M3", "MK3", "SK3", "MN4", "M4", "MS4", "S4", "2MK5", "2MN6", "M6", "2MS6"]


def load_year(year: int) -> pd.DataFrame:
    p = API_TIDE / f"dhn-pe02-{year}.csv"
    d = pd.read_csv(p, comment="#", header=None, names=["data", "hora", "h"])
    t = pd.to_datetime(d.data + " " + d.hora).dt.tz_localize("Etc/GMT+3").dt.tz_convert("UTC")
    return pd.DataFrame({"t": t, "h": d.h.astype(float), "fonte": p.name,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})


def cosine_series(ext: pd.DataFrame, step="10min") -> pd.DataFrame:
    """Série regular entre extremos consecutivos por meia onda de cosseno (método do CosineTideInterpolator da API)."""
    t = pd.date_range(ext.t.iloc[0].ceil(step), ext.t.iloc[-1].floor(step), freq=step)
    te = ext.t.astype("int64").to_numpy()
    he = ext.h.to_numpy()
    tv = t.astype("int64").to_numpy()
    k = np.clip(np.searchsorted(te, tv, side="right") - 1, 0, len(te) - 2)
    frac = (tv - te[k]) / (te[k + 1] - te[k])
    h = he[k] + (he[k + 1] - he[k]) * (1 - np.cos(np.pi * frac)) / 2
    return pd.DataFrame({"t": t, "h": h})


def classify(df: pd.DataFrame) -> pd.Series:
    """Preamar (PM) ou baixa-mar (BM): maior ou menor que os dois vizinhos."""
    h = df.h.to_numpy()
    prev = np.r_[np.nan, h[:-1]]
    nxt = np.r_[h[1:], np.nan]
    return pd.Series(np.where((h > np.nan_to_num(prev, nan=-9)) & (h > np.nan_to_num(nxt, nan=-9)), "PM", "BM"), index=df.index)


def extremes(t: pd.DatetimeIndex, h: np.ndarray) -> pd.DataFrame:
    d = np.diff(h)
    i = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    j = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    return pd.concat([pd.DataFrame({"t": t[i], "h": h[i], "tipo": "PM"}),
                      pd.DataFrame({"t": t[j], "h": h[j], "tipo": "BM"})]).sort_values("t").reset_index(drop=True)


def predict(coef, start, end, step="1min"):
    t = pd.date_range(start, end, freq=step, tz="UTC")
    tt = (t.tz_convert(None) - pd.Timestamp("1970-01-01")).total_seconds().to_numpy() / 86400.0
    return t, utide.reconstruct(tt, coef, verbose=False, epoch="1970-01-01").h


def match(obs: pd.DataFrame, pred: pd.DataFrame, tol_h=3.0):
    rows = []
    for tipo in ("PM", "BM"):
        o, p = obs[obs.tipo == tipo], pred[pred.tipo == tipo]
        if p.empty:
            continue
        for r in o.itertuples():
            k = int(np.argmin(np.abs((p.t - r.t).dt.total_seconds().to_numpy())))
            dt = (p.t.iloc[k] - r.t).total_seconds() / 60
            if abs(dt) <= tol_h * 60:
                rows.append({"t": r.t, "tipo": tipo, "dh_cm": 100 * (p.h.iloc[k] - r.h), "dt_min": dt})
    return pd.DataFrame(rows)


def stats(m: pd.DataFrame):
    out = {}
    for k, g in [("todos", m), ("PM", m[m.tipo == "PM"]), ("BM", m[m.tipo == "BM"])]:
        out[k] = {"n": int(len(g)),
                  "altura_cm": {"media": float(g.dh_cm.mean()), "media_abs": float(g.dh_cm.abs().mean()),
                                "p95_abs": float(g.dh_cm.abs().quantile(.95)), "max_abs": float(g.dh_cm.abs().max())},
                  "horario_min": {"media": float(g.dt_min.mean()), "media_abs": float(g.dt_min.abs().mean()),
                                  "p95_abs": float(g.dt_min.abs().quantile(.95)), "max_abs": float(g.dt_min.abs().max())}}
    return out


def run():
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    allx = pd.concat([load_year(y) for y in range(2021, 2027)], ignore_index=True).sort_values("t").reset_index(drop=True)
    allx["tipo"] = classify(allx)
    fit_ext = allx[(allx.t >= FIT_FROM) & (allx.t < FIT_TO)].reset_index(drop=True)
    fit = cosine_series(fit_ext)
    tt = (fit.t.dt.tz_convert(None) - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() / 86400.0
    coef = utide.solve(tt, fit.h.to_numpy(), lat=LAT, method="ols", conf_int="linear", constit="auto", verbose=False, epoch="1970-01-01")
    out = {"metodo": "UTide solve (OLS, constituintes 'auto' pelo critério de Rayleigh) sobre série de 10 min interpolada por cosseno entre os extremos oficiais da DHN",
           "ajuste": {"de": FIT_FROM, "ate": FIT_TO, "extremos": int(len(fit_ext)), "passos_10min": int(len(fit))}, "utide": utide.__version__,
           "constituintes": int(len(coef.name)), "nivel_medio_m": float(coef.mean),
           "principais": {n: {"amp_m": float(a), "fase_graus": float(g)} for n, a, g in
                          sorted(zip(coef.name, coef.A, coef.g), key=lambda x: -x[1])[:8]},
           "fontes": sorted({(r.fonte, r.sha256) for r in allx.itertuples()}), "validacao": {}}
    for year in (2025, 2026):
        obs = allx[(allx.t >= f"{year}-01-01") & (allx.t < f"{year + 1}-01-01")]
        t, h = predict(coef, f"{year - 1}-12-31", f"{year + 1}-01-02")
        m = match(obs, extremes(t, h))
        out["validacao"][str(year)] = {"extremos_oficiais": int(len(obs)), "pareados": int(len(m)), **stats(m)}
    t, h = predict(coef, "2027-01-01 03:00", "2028-01-01 03:00")
    ex = extremes(t, h)
    ex["data_local"] = ex.t.dt.tz_convert("Etc/GMT+3").dt.strftime("%Y-%m-%d")
    ex["hora_local"] = ex.t.dt.tz_convert("Etc/GMT+3").dt.strftime("%H:%M")
    path = OUT_DATA / "harmonic-pe02-2027.csv"
    with open(path, "w") as f:
        f.write("# fonte: TIDE_HARMONIC_F41 (NAO LIGADA); UTide sobre extremos DHN 2021-10..2024-12; mapi_ai/tide/harmonic_pe02.py\n")
        f.write("# data_local,hora_local,altura_m (UTC-3 como na tabua; altura acima do NR da DHN)\n")
        ex[["data_local", "hora_local", "h"]].round({"h": 2}).to_csv(f, index=False, header=False)
    out["previsao_2027"] = {"arquivo": str(path.relative_to(ROOT)), "extremos": int(len(ex)),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT_REPORT / "harmonica_2027.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: v for k, v in o.items() if k != "fontes"}, indent=1, ensure_ascii=False))
