"""
Previsão de chuva no serviço. As entradas saem do MESMO código do treino (features.features_at), a partir das leituras
de 10 min das últimas 6 h que a API manda. Ausente nunca vira zero: estação sem leitura em 70 min fica fora, e sem
nenhuma estação válida a até 5 km do ponto a previsão é recusada (MissingFeaturesError -> HTTP 422).
"""
import json

import joblib
import numpy as np
import pandas as pd

from mapi_ai.feature_engineering import MissingFeaturesError
from mapi_ai.rain import config as C
from mapi_ai.rain import features as F
from mapi_ai.rain import models as M


class RainForecaster:
    def __init__(self):
        self.meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
        self.chosen = {int(h[:-1]): v["model"] for h, v in self.meta["horizons"].items()}
        self.alert = {int(h[:-1]): v["alert_threshold_FORTE"] for h, v in self.meta["horizons"].items()}
        self.models = {n: joblib.load(C.MODELS_DIR / f"{n}.joblib") for n in set(self.chosen.values())}

    def forecast(self, latitude: float, longitude: float, readings: list, open_meteo: list | None = None,
                 issued_at: str | None = None, now: str | None = None) -> dict:
        return self.forecast_with_inputs(latitude, longitude, readings, open_meteo, issued_at, now)[0]

    def forecast_with_inputs(self, latitude: float, longitude: float, readings: list, open_meteo: list | None = None,
                             issued_at: str | None = None, now: str | None = None) -> tuple[dict, pd.DataFrame]:
        """A mesma previsão de `forecast` e a linha de entradas usada (para o SHAP local do alerta por regressão)."""
        r = pd.DataFrame(readings, columns=["station_id", "latitude", "longitude", "ts", "mm"])
        r = r.dropna(subset=["mm"])                                   # leitura sem valor não é zero: sai
        if r.empty:
            raise MissingFeaturesError(F.POINT_FEATURES)
        r["ts"] = pd.to_datetime(r.ts, utc=True)
        stations = r.groupby("station_id")[["latitude", "longitude"]].first().reset_index().rename(
            columns={"latitude": "lat", "longitude": "lon"})
        latest = r.ts.max()
        t = pd.Timestamp(issued_at).tz_convert("UTC").floor(F.SLOT) if issued_at else latest.ceil(F.SLOT)
        r = r[r.ts <= t]                                              # nada posterior ao instante da previsão
        om = None
        if open_meteo:
            o = pd.DataFrame(open_meteo).dropna(subset=["mm"])
            if len(o):
                om = pd.Series(o.mm.to_numpy(dtype=float), index=pd.to_datetime(o.hour_end_utc, utc=True))
        x = F.features_at(r[["station_id", "ts", "mm"]], stations, latitude, longitude, [t], om)
        row = x.iloc[0]
        if row[F.POINT_FEATURES].isna().all():
            raise MissingFeaturesError(F.POINT_FEATURES)
        x.insert(0, "t", [t])
        horizons, used = [], set()
        for h in C.HORIZONS:
            name = self.chosen[h]
            p = self.models[name].predict(x).iloc[0]
            feats = getattr(self.models[name], "features", F.POINT_FEATURES)
            used.update(feats)
            pf = float(p[f"pFORTE_{h}h"])
            horizons.append({"horizon_hours": h, "valid_from": _iso(t), "valid_until": _iso(t + pd.Timedelta(hours=h)),
                             "quantiles_mm": {n: round(float(p[f"{n}_{h}h"]), 2) for n in M.QN},
                             "prob_media": round(float(p[f"pMEDIA_{h}h"]), 4), "prob_forte": round(pf, 4),
                             "forte_alert": bool(pf >= self.alert[h]), "forte_alert_threshold": self.alert[h], "model": name})
        missing = sorted(f for f in used if pd.isna(row.get(f)))
        sources = ["CEMADEN_PED"] + (["OPEN_METEO_FORECAST"] if om is not None and any(f in used for f in F.OM_FEATURES)
                                     and not row[F.OM_FEATURES].isna().all() else [])
        ref = pd.Timestamp(now).tz_convert("UTC") if now else pd.Timestamp.now(tz="UTC")
        return {"model_version": self.meta["model_version"], "issued_at": _iso(t), "horizons": horizons,
                "sources": sources, "missing_inputs": missing,
                "latest_reading_utc": _iso(latest), "latest_reading_age_minutes": round((ref - latest).total_seconds() / 60, 1),
                "stations_valid_near": int(row["n_near_valid"]), "network_valid_fraction": round(float(row["net_valid_frac"]), 3),
                "rain_last_60min_mm": None if pd.isna(row["p_60"]) else round(float(row["p_60"]), 2),
                "thresholds_mm_h": C.THRESHOLDS, "target": "MEASURED", "attribution": C.ATTRIBUTION}, x


def _iso(t: pd.Timestamp) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")
