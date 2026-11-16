"""
Entradas da previsão de chuva (Fase 3 do plano de 16/11). O MESMO código serve o treino e o serviço:
o treino passa a série de 5 anos, o serviço passa as leituras das últimas horas, e as duas chamam `features_at`.

Grade de 10 min. Leitura com instante `ts` (UTC) conta para o intervalo que TERMINA em `ts`.
Janela "últimos N min" no instante t = leituras com ts em (t - N, t].

Estação válida em t: tem ao menos uma leitura em (t - 70 min, t] (a PED manda a hora seca como um bloco; sem
leitura nenhuma em 70 min, a estação está fora). Dentro de uma estação válida, intervalo sem leitura = 0 mm.
Estação inválida = ausente (NaN), nunca zero.

Chuva do ponto: o maior acumulado entre as estações válidas a até 5 km (regra chuva-ponto, ADR 0022, sem a
checagem mensal nem a retirada de pico isolado, que precisam do mês inteiro).
"""
import math

import numpy as np
import pandas as pd

SLOT = pd.Timedelta("10min")
NEAR_KM = 5.0
MAX_KM = 40.0
WET_MM = 0.2          # estação "com chuva" numa janela
ONSET_MM = 0.4        # início: intervalo de 10 min com 0,4 mm ou mais depois de 2 h secas
SECTORS = ("N", "S", "L", "O")
LOOKBACK = pd.Timedelta("6h")   # quanto de leitura o serviço precisa receber

POINT_FEATURES = ["p_10", "p_30", "p_60", "p_180", "p_trend"]
SECTOR_FEATURES = [f"{s}_{k}" for s in SECTORS for k in ("max_30", "max_60", "frac_wet_60", "min_since_onset")]
RINGS = {"r5_15": (5.0, 15.0), "r15_40": (15.0, 40.0)}     # anéis de distância ao ponto, km
RING_FEATURES = [f"{r}_{k}" for r in RINGS for k in ("max_60", "mean_60", "frac_wet_60")]
NET_FEATURES = ["net_valid_frac", "n_near_valid"]
TIME_FEATURES = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
BASE_FEATURES = POINT_FEATURES + SECTOR_FEATURES + RING_FEATURES + NET_FEATURES + TIME_FEATURES
OM_FEATURES = ["om_1h", "om_3h", "om_6h"]
ALL_FEATURES = BASE_FEATURES + OM_FEATURES


def km(lat1, lon1, lat2, lon2) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def sector(lat0, lon0, lat, lon) -> str:
    ang = math.degrees(math.atan2(lat - lat0, (lon - lon0) * math.cos(math.radians(lat0))))
    return "L" if -45 <= ang < 45 else "N" if 45 <= ang < 135 else "S" if -135 <= ang < -45 else "O"


def station_geometry(stations: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Distância e setor de cada estação em relação ao ponto; só as de até 40 km. `stations`: station_id, lat, lon."""
    g = stations.dropna(subset=["lat", "lon"]).copy()
    g["km"] = [km(lat, lon, a, b) for a, b in zip(g.lat, g.lon)]
    g["sector"] = [sector(lat, lon, a, b) for a, b in zip(g.lat, g.lon)]
    g["near"] = g.km <= NEAR_KM
    return g[g.km <= MAX_KM].sort_values("km").reset_index(drop=True)


def grid(readings: pd.DataFrame, station_ids: list, start: pd.Timestamp, end: pd.Timestamp):
    """(valor, válida) em grade de 10 min: matrizes tempo x estação. readings: station_id, ts (UTC), mm."""
    idx = pd.date_range(start.floor(SLOT), end.ceil(SLOT), freq=SLOT)
    r = readings[readings.station_id.isin(station_ids)].copy()
    r["slot"] = pd.to_datetime(r.ts, utc=True).dt.ceil(SLOT)
    m = r.groupby(["slot", "station_id"]).mm.sum().unstack().reindex(index=idx, columns=station_ids)
    seen = m.notna().astype(float)
    valid = seen.rolling(7, min_periods=1).max().astype(bool)          # alguma leitura em (t-70 min, t]
    return m, valid


def _window(m: pd.DataFrame, valid: pd.DataFrame, n: int) -> pd.DataFrame:
    s = m.fillna(0.0).rolling(n, min_periods=1).sum()
    return s.where(valid)


def _since_onset(m: pd.DataFrame) -> pd.DataFrame:
    """Minutos desde o início da chuva em cada estação, se ele foi nas últimas 3 h; senão NaN."""
    wet = (m.fillna(0.0) >= ONSET_MM)
    prev_dry = ~wet.shift(1, fill_value=False).rolling(12, min_periods=1).max().astype(bool)
    onset = wet & prev_dry
    t = pd.Series(np.arange(len(m), dtype=float), index=m.index)
    last = pd.DataFrame(np.where(onset, t.to_numpy()[:, None], np.nan), index=m.index, columns=m.columns).ffill()
    mins = (t.to_numpy()[:, None] - last) * 10.0
    return mins.where(mins <= 180)


def features_at(readings: pd.DataFrame, stations: pd.DataFrame, lat: float, lon: float, times,
                om_hourly: pd.Series | None = None) -> pd.DataFrame:
    """
    Entradas nos instantes `times` (UTC, múltiplos de 10 min) para o ponto (lat, lon).
    om_hourly: previsão do Open-Meteo para a célula do ponto, mm por hora, índice = fim da hora (UTC); opcional.
    """
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    geo = station_geometry(stations, lat, lon)
    ids = list(geo.station_id)
    m, valid = grid(readings, ids, times.min() - LOOKBACK, times.max())
    w = {n: _window(m, valid, n // 10) for n in (10, 30, 60, 180)}
    near = list(geo.station_id[geo.near])
    out = pd.DataFrame(index=times)
    for n in (10, 30, 60, 180):
        out[f"p_{n}"] = w[n][near].max(axis=1, skipna=True).reindex(times) if near else np.nan
    prev30 = (w[60] - w[30])[near].max(axis=1).reindex(times) if near else np.nan
    out["p_trend"] = out["p_30"] - prev30
    since = _since_onset(m)
    for s in SECTORS:
        cols = list(geo.station_id[(geo.sector == s) & ~geo.near])
        if not cols:
            for k in ("max_30", "max_60", "frac_wet_60", "min_since_onset"):
                out[f"{s}_{k}"] = np.nan
            continue
        out[f"{s}_max_30"] = w[30][cols].max(axis=1).reindex(times)
        out[f"{s}_max_60"] = w[60][cols].max(axis=1).reindex(times)
        nv = valid[cols].sum(axis=1)
        out[f"{s}_frac_wet_60"] = ((w[60][cols] >= WET_MM).sum(axis=1) / nv.where(nv > 0)).reindex(times)
        # a estação mais próxima do ponto (colunas já em ordem de distância) que começou a chover nas últimas 3 h
        sv = since[cols].where(valid[cols])
        out[f"{s}_min_since_onset"] = sv.bfill(axis=1).iloc[:, 0].reindex(times)
    for r, (lo, hi) in RINGS.items():
        cols = list(geo.station_id[(geo.km > lo) & (geo.km <= hi)])
        if not cols:
            for k in ("max_60", "mean_60", "frac_wet_60"):
                out[f"{r}_{k}"] = np.nan
            continue
        nv = valid[cols].sum(axis=1)
        out[f"{r}_max_60"] = w[60][cols].max(axis=1).reindex(times)
        out[f"{r}_mean_60"] = w[60][cols].mean(axis=1).reindex(times)
        out[f"{r}_frac_wet_60"] = ((w[60][cols] >= WET_MM).sum(axis=1) / nv.where(nv > 0)).reindex(times)
    out["net_valid_frac"] = valid.mean(axis=1).reindex(times)
    out["n_near_valid"] = valid[near].sum(axis=1).reindex(times).astype(float) if near else 0.0
    local = times.tz_convert("America/Recife")
    out["hour_sin"] = np.sin(2 * np.pi * (local.hour + local.minute / 60) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (local.hour + local.minute / 60) / 24)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * local.month / 12), np.cos(2 * np.pi * local.month / 12)
    for h in (1, 3, 6):
        out[f"om_{h}h"] = om_sum(om_hourly, times, h) if om_hourly is not None else np.nan
    return out


def om_sum(om_hourly: pd.Series, times: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Soma da previsão horária do Open-Meteo nas horas que terminam em (t, t+h]. Hora faltando = NaN."""
    s = om_hourly.sort_index()
    s = s[~s.index.duplicated()]
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    fwd = full.rolling(h, min_periods=h).sum().shift(-(h - 1))       # no índice e: soma de e .. e+h-1
    first = times.floor("h") + pd.Timedelta("1h")
    return fwd.reindex(first).to_numpy()


def point_targets(readings: pd.DataFrame, stations: pd.DataFrame, lat: float, lon: float, times,
                  horizons=(1, 3, 6), thresholds=None) -> pd.DataFrame:
    """
    Alvo medido: chuva do ponto em (t, t+H] = maior acumulado entre as estações a até 5 km válidas no fim da janela;
    NaN se nenhuma estação próxima esteve válida. Evento: alguma hora cheia dentro de (t, t+H] com chuva do ponto
    acima do limiar (pico horário, como no ADR 0017).
    """
    thresholds = thresholds or {}
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    geo = station_geometry(stations, lat, lon)
    near = list(geo.station_id[geo.near])
    hmax = max(horizons)
    m, valid = grid(readings, near, times.min() - LOOKBACK, times.max() + pd.Timedelta(hours=hmax))
    vals = m.fillna(0.0)
    out = pd.DataFrame(index=times)
    cum = vals.cumsum()
    for h in horizons:
        n = h * 6
        fut = (cum.shift(-n) - cum)                                   # soma em (t, t+h]
        ok = valid.shift(-n, fill_value=False) & valid                # válida no início e no fim
        out[f"y_{h}h"] = fut.where(ok).max(axis=1).reindex(times)
        r60 = vals.rolling(6, min_periods=6).sum().where(valid)       # hora cheia terminando em cada slot
        peak = r60.max(axis=1)
        # maior hora cheia dentro de (t, t+h]: janelas que terminam em (t+60 min, t+h]
        fmax = peak[::-1].rolling(max(1, n - 5), min_periods=1).max()[::-1].shift(-6)
        out[f"peak_{h}h"] = fmax.reindex(times)
        for name, thr in thresholds.items():
            out[f"ev_{name}_{h}h"] = (out[f"peak_{h}h"] >= thr).where(out[f"peak_{h}h"].notna())
    return out
