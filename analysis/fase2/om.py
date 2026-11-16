"""Fase 2.3: acerto da previsão ARQUIVADA do Open-Meteo contra a chuva medida no ponto (regra chuva-ponto)."""
import sys, json
import numpy as np, pandas as pd
F2, AI = sys.argv[1], sys.argv[2]
slugs = ["AV_RECIFE_IBURA", "CIN_UFPE", "AGAMENON_DERBY", "JABOATAO_CENTRO", "MASCARENHAS_IMBIRIBEIRA"]
def load(fn, cols):
    out = []
    for slug, d in zip(slugs, json.load(open(f"{F2}/om/{fn}"))):
        h = pd.DataFrame(d["hourly"]); h["slug"] = slug; h["t"] = pd.to_datetime(h.time, utc=True); out.append(h[["slug", "t"] + cols])
    return pd.concat(out)
hist = load("hist.json", ["precipitation"]).rename(columns={"precipitation": "om_curto"})
prev = load("prev.json", ["precipitation_previous_day1", "precipitation_previous_day2"]).rename(
    columns={"precipitation_previous_day1": "om_24h", "precipitation_previous_day2": "om_48h"})
m = pd.read_csv(f"{AI}/data/in1165/cemaden_point_hourly.csv.gz", parse_dates=["hour_end_utc"]).rename(columns={"hour_end_utc": "t"})
df = m[["slug", "t", "current_rainfall"]].merge(hist, on=["slug", "t"]).merge(prev, on=["slug", "t"], how="left")
res = []
for agg in (1, 3, 6):
    parts = []
    for slug, g in df.sort_values("t").groupby("slug"):
        g = g.set_index("t").asfreq("h")
        r = g[["current_rainfall", "om_curto", "om_24h", "om_48h"]].rolling(agg, min_periods=agg).sum()
        r["persist"] = g.current_rainfall.rolling(agg, min_periods=agg).sum().shift(agg)   # o mesmo acumulado, janela anterior
        r["clim"] = np.nan; parts.append(r.dropna(subset=["current_rainfall"]))
    R = pd.concat(parts)
    thr = {1: 5.03, 3: 10.0, 6: 15.0}[agg]
    for f in ("om_curto", "om_24h", "om_48h", "persist"):
        x = R.dropna(subset=[f]); y, p = x.current_rainfall, x[f]
        hit = ((y >= thr) & (p >= thr)).sum(); miss = ((y >= thr) & (p < thr)).sum(); fa = ((y < thr) & (p >= thr)).sum()
        # melhor limiar da previsão para o mesmo evento (o modelo pode subestimar sistematicamente)
        best = max(((((y >= thr) & (p >= q)).sum() / max(1, ((y >= thr) | (p >= q)).sum()), q) for q in np.quantile(p[p > 0], np.linspace(.5, .995, 40))), default=(np.nan, np.nan))
        res.append(dict(janela_h=agg, previsao=f, horas=len(x), corr=round(float(np.corrcoef(y, p)[0, 1]), 3), mae_mm=round(float((y - p).abs().mean()), 3),
                        media_medida=round(float(y.mean()), 3), media_prevista=round(float(p.mean()), 3), limiar_mm=thr,
                        eventos=int((y >= thr).sum()), POD=round(hit / max(1, hit + miss), 3), FAR=round(fa / max(1, hit + fa), 3),
                        CSI=round(hit / max(1, hit + miss + fa), 3), CSI_melhor_limiar=round(float(best[0]), 3), limiar_prev_melhor=round(float(best[1]), 2)))
T = pd.DataFrame(res); print(T.to_string(index=False)); T.to_csv(f"{F2}/om_skill.csv", index=False)
print("periodo", df.t.min(), df.t.max(), "grade", [ (d["latitude"], d["longitude"]) for d in json.load(open(f"{F2}/om/hist.json"))])
