"""
Simulação do dado como chega (prompt do alerta de 20 min, item 2.3). Em cada instante t o modelo só vê as leituras
que a coleta ao vivo teria entregue até t: leitura com fim em ts e atraso d é visível em t se ts + d <= t.

Formato do bloco: a coleta ao vivo entrega as MESMAS leituras da PED (2.142 blocos iguais e 0 diferentes na
sobreposição de 2026-11-08 a 11-17, consulta em reports/rain_v1b/SIMULACAO.md). Por isso a única transformação é o
atraso; o tamanho do bloco não é re-sorteado.

Atraso: sorteado por leitura, com semente fixa, da distribuição empírica da estação em regime estável (coleta que veio
até 6 min depois da anterior). Estação com menos de MIN_SAMPLES amostras usa a distribuição de todas.

As entradas são as mesmas de features.features_at, calculadas com a máscara de visibilidade: em t, o passo de idade a
(t - a) só conta se o atraso da leitura em passos for <= a. Sem leitura visível em 70 min, a estação fica ausente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mapi_ai.rain import features as F

MIN_SAMPLES = 20
SLOT_MIN = 10


def sample_delays(readings: pd.DataFrame, delays: pd.DataFrame, seed: int = 42, scenario: str = "estavel") -> np.ndarray:
    """
    Atraso (min) por leitura. scenario:
      - "lab": zero (o laboratório);
      - "estavel": distribuição da estação em regime estável (todas as leituras, secas e com chuva);
      - "chuva": leitura com chuva (mm > 0) sorteia das leituras ao vivo COM chuva (todas as estações juntas, poucas
        amostras: cenário pessimista declarado); leitura seca, da distribuição da estação.
    """
    if scenario == "lab":
        return np.zeros(len(readings))
    rng = np.random.default_rng(seed)
    pooled = delays.delay_min.to_numpy(dtype=float)
    wet_pool = delays[delays.wet == 1].delay_min.to_numpy(dtype=float)
    by_st = {s: g.delay_min.to_numpy(dtype=float) for s, g in delays.groupby("canonical_id")}
    out = np.empty(len(readings))
    st = readings.station_id.to_numpy()
    mm = readings.mm.to_numpy(dtype=float)
    for s in np.unique(st):
        idx = np.flatnonzero(st == s)
        pool = by_st.get(s)
        if pool is None or len(pool) < MIN_SAMPLES:
            pool = pooled
        out[idx] = rng.choice(pool, size=len(idx))
    if scenario == "chuva" and len(wet_pool):
        w = np.flatnonzero(mm > 0)
        out[w] = rng.choice(wet_pool, size=len(w))
    return out


def _grid_lag(readings: pd.DataFrame, ids: list, start: pd.Timestamp, end: pd.Timestamp, offset_min: float = 0.0):
    """
    Valor por passo (soma) e atraso em passos (máximo das leituras do passo), tempo x estação. As entradas ficam
    ancoradas no passo t, como no serviço (serve.py ancora na leitura mais nova), e são calculadas no ciclo que roda
    offset_min depois de t: a leitura do passo de idade a é visível se ts + d <= t + offset_min (+ 10 a).
    """
    idx = pd.date_range(start.floor(F.SLOT), end.ceil(F.SLOT), freq=F.SLOT)
    r = readings[readings.station_id.isin(ids)].copy()
    ts = pd.to_datetime(r.ts, utc=True)
    r["slot"] = ts.dt.ceil(F.SLOT)
    avail = ts + pd.to_timedelta(r.delay_min.to_numpy(), unit="min")
    # passos depois do passo da leitura até ela ficar visível: t >= slot + k*10 min  <=>  t >= ts + d
    r["k"] = np.ceil(np.maximum(0.0, (avail - r.slot).dt.total_seconds().to_numpy() / 60.0 - offset_min) / SLOT_MIN - 1e-9).astype(int)
    m = r.groupby(["slot", "station_id"]).mm.sum().unstack().reindex(index=idx, columns=ids)
    k = r.groupby(["slot", "station_id"]).k.max().unstack().reindex(index=idx, columns=ids)
    return m, k


def _visible_sum(m: pd.DataFrame, k: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Soma em (t - n passos, t] só do que já tinha chegado em t: a soma do laboratório (mesmo rolling de features.py, para
    que atraso zero reproduza o laboratório bit a bit) menos as leituras ainda não chegadas (idade a < atraso k).
    """
    full = m.fillna(0.0).rolling(n, min_periods=1).sum()
    mv = np.nan_to_num(m.to_numpy(dtype=float), nan=0.0)
    kv = np.nan_to_num(k.to_numpy(dtype=float), nan=0.0)
    hidden = np.zeros_like(mv)
    kmax = int(kv.max()) if kv.size else 0
    for a in range(min(n, kmax)):
        h = np.where(kv > a, mv, 0.0)
        hidden[a:] += h[:len(h) - a] if a else h
    if not hidden.any():
        return full
    return full - pd.DataFrame(hidden, index=m.index, columns=m.columns)


def _visible_valid(m: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """Estação válida em t: alguma leitura visível em (t - 70 min, t]."""
    seen, kv = m.notna().to_numpy(), k.to_numpy(dtype=float)
    out = np.zeros(seen.shape, bool)
    for a in range(7):
        vis = seen & (kv <= a)
        out[a:] |= vis[:len(vis) - a] if a else vis
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def _since_onset_visible(m: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
    """Minutos desde o início da chuva mais recente JÁ VISÍVEL em t, se nas últimas 3 h; senão NaN."""
    wet = (m.fillna(0.0) >= F.ONSET_MM)
    prev_dry = ~wet.shift(1, fill_value=False).rolling(12, min_periods=1).max().astype(bool)
    onset = (wet & prev_dry).to_numpy()
    kv = np.nan_to_num(k.to_numpy(dtype=float), nan=0.0).astype(int)
    T, S = onset.shape
    out = np.full((T, S), np.nan)
    for j in range(S):
        place = np.full(T, -1)
        for s in np.flatnonzero(onset[:, j]):
            vt = s + kv[s, j]
            if vt < T:
                place[vt] = max(place[vt], s)
        last = np.maximum.accumulate(place)
        mins = (np.arange(T) - last) * 10.0
        mins[last < 0] = np.nan
        out[:, j] = np.where(mins <= 180, mins, np.nan)
    return pd.DataFrame(out, index=m.index, columns=m.columns)


def features_at_visible(readings: pd.DataFrame, stations: pd.DataFrame, lat: float, lon: float, times,
                        om_hourly: pd.Series | None = None, offset_min: float = 0.0) -> pd.DataFrame:
    """Mesmas colunas de features.features_at; readings precisa de delay_min. offset_min: ciclo depois da âncora."""
    times = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    geo = F.station_geometry(stations, lat, lon)
    ids = list(geo.station_id)
    m, k = _grid_lag(readings, ids, times.min() - F.LOOKBACK, times.max(), offset_min)
    valid = _visible_valid(m, k)
    w = {n: _visible_sum(m, k, n // 10).where(valid) for n in (10, 30, 60, 180)}
    near = list(geo.station_id[geo.near])
    out = pd.DataFrame(index=times)
    for n in (10, 30, 60, 180):
        out[f"p_{n}"] = w[n][near].max(axis=1, skipna=True).reindex(times) if near else np.nan
    prev30 = (w[60] - w[30])[near].max(axis=1).reindex(times) if near else np.nan
    out["p_trend"] = out["p_30"] - prev30
    since = _since_onset_visible(m, k)
    for s in F.SECTORS:
        cols = list(geo.station_id[(geo.sector == s) & ~geo.near])
        if not cols:
            for kk in ("max_30", "max_60", "frac_wet_60", "min_since_onset"):
                out[f"{s}_{kk}"] = np.nan
            continue
        out[f"{s}_max_30"] = w[30][cols].max(axis=1).reindex(times)
        out[f"{s}_max_60"] = w[60][cols].max(axis=1).reindex(times)
        nv = valid[cols].sum(axis=1)
        out[f"{s}_frac_wet_60"] = ((w[60][cols] >= F.WET_MM).sum(axis=1) / nv.where(nv > 0)).reindex(times)
        sv = since[cols].where(valid[cols])
        out[f"{s}_min_since_onset"] = sv.bfill(axis=1).iloc[:, 0].reindex(times)
    for r, (lo, hi) in F.RINGS.items():
        cols = list(geo.station_id[(geo.km > lo) & (geo.km <= hi)])
        if not cols:
            for kk in ("max_60", "mean_60", "frac_wet_60"):
                out[f"{r}_{kk}"] = np.nan
            continue
        nv = valid[cols].sum(axis=1)
        out[f"{r}_max_60"] = w[60][cols].max(axis=1).reindex(times)
        out[f"{r}_mean_60"] = w[60][cols].mean(axis=1).reindex(times)
        out[f"{r}_frac_wet_60"] = ((w[60][cols] >= F.WET_MM).sum(axis=1) / nv.where(nv > 0)).reindex(times)
    out["net_valid_frac"] = valid.mean(axis=1).reindex(times)
    out["n_near_valid"] = valid[near].sum(axis=1).reindex(times).astype(float) if near else 0.0
    local = times.tz_convert("America/Recife")
    out["hour_sin"] = np.sin(2 * np.pi * (local.hour + local.minute / 60) / 24)
    out["hour_cos"] = np.cos(2 * np.pi * (local.hour + local.minute / 60) / 24)
    out["month_sin"], out["month_cos"] = np.sin(2 * np.pi * local.month / 12), np.cos(2 * np.pi * local.month / 12)
    for h in (1, 3, 6):
        out[f"om_{h}h"] = F.om_sum(om_hourly, times, h) if om_hourly is not None else np.nan
    return out
