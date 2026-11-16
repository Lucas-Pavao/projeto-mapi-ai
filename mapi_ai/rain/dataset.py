"""
Monta o dataset da previsão de chuva: grade de 10 min x 5 pontos, entradas (features.py) e alvo medido.
Uso: python -m mapi_ai.rain.dataset
"""
import glob
import json

import numpy as np
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import features as F


def load_readings() -> pd.DataFrame:
    r = pd.read_csv(C.READINGS, header=None, names=["station_id", "epoch", "mm"])
    r["ts"] = pd.to_datetime(r.epoch, unit="s", utc=True)
    return r[["station_id", "ts", "mm"]]


def build_om_prev():
    """Junta os JSON da previous-runs-api (data/rain/raw) num CSV: slug, hour_end_utc, om_mm (rodada da véspera)."""
    rows = []
    for fn in sorted(glob.glob(str(C.DATA_DIR / "raw" / "prev_*.json"))):
        slug = fn.split("prev_")[1].rsplit("_", 1)[0]
        d = json.load(open(fn))
        h = pd.DataFrame(d["hourly"])
        rows.append(pd.DataFrame({"slug": slug, "hour_end_utc": pd.to_datetime(h.time, utc=True),
                                  "om_mm": h.precipitation_previous_day1, "grid_lat": d["latitude"], "grid_lon": d["longitude"]}))
    out = pd.concat(rows).dropna(subset=["om_mm"]).drop_duplicates(["slug", "hour_end_utc"])
    out.to_csv(C.OM_PREV, index=False, compression={"method": "gzip", "mtime": 0})
    return out


def load_om(path) -> dict:
    o = pd.read_csv(path)
    o["hour_end_utc"] = pd.to_datetime(o.hour_end_utc, utc=True)
    return {s: g.set_index("hour_end_utc").om_mm for s, g in o.groupby("slug")}


def build() -> pd.DataFrame:
    readings, stations, points = load_readings(), pd.read_csv(C.STATIONS), pd.read_csv(C.POINTS)
    if not C.OM_PREV.exists():
        build_om_prev()
    om_prev, om_hist = load_om(C.OM_PREV), load_om(C.OM_HIST)
    times = pd.date_range(C.DATA_START, C.DATA_END, freq=F.SLOT)
    parts = []
    for p in points.itertuples():
        x = F.features_at(readings, stations, p.latitude, p.longitude, times, om_prev.get(p.slug))
        for h in C.HORIZONS:                       # variante otimista: rodadas emendadas (emissão não separada)
            x[f"omh_{h}h"] = F.om_sum(om_hist[p.slug], times, h) if p.slug in om_hist else np.nan
        y = F.point_targets(readings, stations, p.latitude, p.longitude, times, C.HORIZONS, C.THRESHOLDS)
        d = pd.concat([x, y], axis=1)
        d.insert(0, "slug", p.slug)
        d.insert(1, "t", times)
        parts.append(d.reset_index(drop=True))
    df = pd.concat(parts, ignore_index=True)
    df["part"] = C.part(df.t)
    df["raining"] = df.p_60 >= C.RAINING_MM
    df.to_pickle(C.DATASET)
    return df


def load() -> pd.DataFrame:
    return pd.read_pickle(C.DATASET)


if __name__ == "__main__":
    d = build()
    print(len(d), d.part.value_counts().to_dict())
    print(d[[f"y_{h}h" for h in C.HORIZONS]].describe().T)
    print(d.groupby("part")[[f"ev_FORTE_{h}h" for h in C.HORIZONS]].mean())
    print(d[F.ALL_FEATURES].isna().mean().round(3).to_dict())
