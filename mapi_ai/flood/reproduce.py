"""
Reprodução do alerta por regressão (Etapa 4.1). A partir só dos arquivos de data/ e dos artefatos de models/v4/rain:

1. reconstrói o alerta de um dia de TESTE (CIN_UFPE, 2026-05-01, o dia de teste com mais chuva medida no ponto: 219,7 mm)
   a cada 10 min, com o mesmo código da validação (validate.reconstruct), e confere o hash SHA-256 do resultado
   canônico com o gravado em reports/regressao/reproducao.json;
2. confere que o caminho do SERVIÇO (flood.forecast.forecast_point, com as leituras brutas da rede e a série do ponto)
   dá a mesma chuva prevista e a mesma lâmina que a reconstrução no instante calculável (sem lacuna na janela de 24 h) de maior chuva do dia.
Uso: python -m mapi_ai.flood.reproduce [--gravar]
"""
from __future__ import annotations

import hashlib
import json
import sys

import joblib
import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B10
from mapi_ai.flood import validate as V
from mapi_ai.flood.forecast import forecast_point
from mapi_ai.rain import config as RC
from mapi_ai.rain import dataset as RD
from mapi_ai.rain import features as F

SLUG, DAY = "CIN_UFPE", "2026-05-01"
EXPECTED = V.OUT / "reproducao.json"


def canonical(day: pd.DataFrame) -> str:
    cols = ["t", "level"] + [f"{k}_{h}h" for h in (1, 3, 6) for k in ("level", "sheet_q50", "sheet_q95")]
    d = day[cols].copy()
    d["t"] = d.t.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    for c in cols:
        if c.startswith("sheet"):
            d[c] = d[c].round(2)
    return json.dumps(d.to_dict("records"), separators=(",", ":"), ensure_ascii=False)


def run(gravar: bool = False) -> dict:
    terr = {t["slug"]: t for t in json.loads((V.SERIES / "terreno.json").read_text())}[SLUG]
    ds = RD.load()
    ds = ds[(ds.slug == SLUG) & (ds.part == "test")].reset_index(drop=True)
    keep = ["slug", "t", "part", "raining", "p_10", "p_60"]
    pred = {n: joblib.load(RC.MODELS_DIR / f"{n}.joblib").predict(ds) for n in sorted(set(V.CHOSEN.values()))}
    rec = V.reconstruct(ds[keep], pred, {SLUG: terr})
    local = rec.t.dt.tz_convert("America/Recife").dt.strftime("%Y-%m-%d")
    day = rec[local == DAY].reset_index(drop=True)
    blob = canonical(day)
    digest = hashlib.sha256(blob.encode()).hexdigest()

    # caminho do serviço no instante de maior chuva medida do dia
    i = int(day.mm.where(day.level != "SEM_DADO").fillna(-1).idxmax())  # instante calculável de maior chuva
    t = day.t.iloc[i]
    stations = pd.read_csv(RC.STATIONS)
    geo = F.station_geometry(stations, -8.05531, -34.95116)
    r = RD.load_readings()
    r = r[r.station_id.isin(geo.station_id) & (r.ts > t - F.LOOKBACK) & (r.ts <= t)].merge(
        stations[["station_id", "lat", "lon"]], on="station_id")
    readings = [{"station_id": a, "latitude": b, "longitude": c, "ts": d, "mm": e}
                for a, b, c, d, e in zip(r.station_id, r.lat, r.lon, r.ts, r.mm)]
    om = RD.load_om(RC.OM_PREV).get(SLUG)
    om_list = [{"hour_end_utc": k, "mm": float(v)} for k, v in om[(om.index > t - pd.Timedelta("2h")) & (om.index <= t + pd.Timedelta("7h"))].items()]
    serie = pd.read_csv(V.SERIES / f"serie10min_{SLUG}.csv.gz", parse_dates=["slot_end_utc"]).set_index("slot_end_utc").mm
    s144 = serie[(serie.index > t - pd.Timedelta(minutes=10 * V.WINDOW)) & (serie.index <= t)]
    ter = B10.Terreno(float(terr["area_m2"]), np.asarray(terr["z"], float), np.asarray(terr["v"], float), float(terr["rua_rel_m"]))
    from mapi_ai.rain.serve import RainForecaster
    svc = forecast_point(latitude=-8.05531, longitude=-34.95116, readings=readings,
                         rain_10min_mm=[None if pd.isna(v) else float(v) for v in s144], terrain=ter,
                         open_meteo=om_list, issued_at=t.isoformat(), rain_model=RainForecaster(), slug=SLUG)
    row = day.iloc[i]
    comp = {"instante": str(t), "nivel_servico": svc["level"], "nivel_reconstrucao": row.level}
    if svc["level"] != "SEM_DADO":
        k = ds.index[ds.t == t][0]
        comp["chuva_q50_mm"] = {h["hours"]: [h["rain_q50_mm"], round(float(pred[V.CHOSEN[h["hours"]]].loc[k, f"q50_{h['hours']}h"]), 2)]
                                for h in svc["horizons"]}
        comp["lamina_q95_cm"] = {h["hours"]: [h["sheet_q95_cm"], round(float(row[f"sheet_q95_{h['hours']}h"]), 1)]
                                 for h in svc["horizons"]}
        comp["max_diferenca_chuva_mm"] = max(abs(a - b) for a, b in comp["chuva_q50_mm"].values())
    out = {"ponto": SLUG, "dia_local": DAY, "passos": int(len(day)), "sha256": digest,
           "niveis_no_dia": day.level.value_counts().to_dict(), "lamina_max_q95_cm": float(np.nanmax(day[[f"sheet_q95_{h}h" for h in (1, 3, 6)]].to_numpy())),
           "chuva_medida_no_dia_mm": round(float(day.mm.sum()), 1), "servico_x_reconstrucao": comp}
    if gravar:
        EXPECTED.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
        out["conferencia"] = "gravado"
    else:
        exp = json.loads(EXPECTED.read_text())
        out["conferencia"] = "OK" if exp["sha256"] == digest else f"DIVERGE (esperado {exp['sha256']})"
    return out


if __name__ == "__main__":
    o = run("--gravar" in sys.argv)
    print(json.dumps(o, indent=1, ensure_ascii=False, default=str))
    sys.exit(0 if o["conferencia"] in ("OK", "gravado") else 1)
