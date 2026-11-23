"""
Paridade treino × serviço do rain_v1 (onda 16, Fase 7.2). Para instantes passados, as entradas do serviço (6 h de leitura, a
lista de dicionários que a API manda) e as do treino (features_at sobre semanas de histórico) têm de bater.
Tolerância escrita antes: somas e máximos em mm, |diferença| <= 1e-6; frações, contagens e minutos, iguais; NaN dos dois lados.
"""
import numpy as np
import pandas as pd
import pytest

from mapi_ai.rain import config as C
from mapi_ai.rain import features as F
from mapi_ai.rain.serve import RainForecaster

TOL_MM = 1e-6
PONTOS = pd.read_csv(C.POINTS)
INSTANTES = ["2025-03-18T21:10:00Z", "2025-05-02T09:40:00Z", "2025-06-14T15:00:00Z"]
needs_data = pytest.mark.skipif(not C.READINGS.exists() if hasattr(C, "READINGS") else False, reason="sem leituras da PED")


def _readings():
    from mapi_ai.rain import dataset as D
    return D.load_readings()


@pytest.fixture(scope="module")
def dados():
    try:
        r = _readings()
    except FileNotFoundError:
        pytest.skip("sem leituras da PED no ambiente")
    return r, pd.read_csv(C.STATIONS), RainForecaster()


@pytest.mark.parametrize("t", INSTANTES)
def test_entradas_do_servico_batem_com_as_do_treino(dados, t):
    readings, stations, rf = dados
    t = pd.Timestamp(t)
    hist = readings[(pd.to_datetime(readings.ts, utc=True) > t - pd.Timedelta(days=21)) & (pd.to_datetime(readings.ts, utc=True) <= t)]
    for p in PONTOS.itertuples():
        treino = F.features_at(hist, stations, p.latitude, p.longitude, [t]).iloc[0]
        win = hist[pd.to_datetime(hist.ts, utc=True) > t - F.LOOKBACK]
        st = stations.set_index("station_id")
        lista = [{"station_id": s, "latitude": st.loc[s, "lat"], "longitude": st.loc[s, "lon"], "ts": ts, "mm": mm}
                 for s, ts, mm in zip(win.station_id, win.ts, win.mm) if s in st.index]
        if not lista:
            continue
        try:
            _, x = rf.forecast_with_inputs(p.latitude, p.longitude, lista, None, t.isoformat())
        except Exception:
            assert treino[F.POINT_FEATURES].isna().all()
            continue
        serv = x.iloc[0]
        for f in F.BASE_FEATURES:
            a, b = treino[f], serv[f]
            if pd.isna(a) or pd.isna(b):
                assert pd.isna(a) and pd.isna(b), (p.slug, str(t), f, a, b)
            elif f.endswith("frac_wet_60") or f in ("net_valid_frac", "n_near_valid") or f.endswith("min_since_onset"):
                assert a == pytest.approx(b, abs=1e-12), (p.slug, str(t), f, a, b)
            else:
                assert abs(a - b) <= TOL_MM, (p.slug, str(t), f, a, b)


def test_estacao_so_ao_vivo_fica_de_fora_e_e_listada(dados):
    _, stations, rf = dados
    p = PONTOS.iloc[0]
    s0 = stations.iloc[int(((stations.lat - p.latitude) ** 2 + (stations.lon - p.longitude) ** 2).idxmin())]
    t = pd.Timestamp("2025-05-02T09:40:00Z")
    ts = pd.date_range(t - pd.Timedelta("50min"), t, freq="10min")
    lista = [{"station_id": s0.station_id, "latitude": s0.lat, "longitude": s0.lon, "ts": x.isoformat(), "mm": 0.0} for x in ts]
    lista += [{"station_id": "SO_AO_VIVO", "latitude": p.latitude, "longitude": p.longitude, "ts": x.isoformat(), "mm": 50.0}
              for x in ts]
    out, x = rf.forecast_with_inputs(p.latitude, p.longitude, lista, None, t.isoformat())
    assert out["stations_ignored_live_only"] == ["SO_AO_VIVO"]
    assert out["network_valid_fraction"] < 0.1     # as outras 45 estações da rede contam no denominador
    assert (x.iloc[0]["p_60"] != 300.0)
