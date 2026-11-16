"""Previsão de chuva: nenhuma entrada usa dado posterior a t; ausente não vira zero; contrato da rota."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai.rain import config as C
from mapi_ai.rain import features as F

LAT, LON = -8.05531, -34.95116
ST = pd.DataFrame({"station_id": ["perto", "norte", "oeste"], "lat": [LAT + 0.01, LAT + 0.1, LAT], "lon": [LON, LON, LON - 0.1]})
T = pd.Timestamp("2025-05-10T12:00:00Z")


def readings(until, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(T - pd.Timedelta("8h"), until, freq="10min")
    return pd.concat([pd.DataFrame({"station_id": s, "ts": ts, "mm": rng.gamma(0.3, 2.0, len(ts)).round(2)}) for s in ST.station_id])


def test_entradas_nao_usam_nada_depois_de_t():
    future = readings(T + pd.Timedelta("6h"))
    past = future[future.ts <= T]                                      # as mesmas leituras, sem o futuro
    future.loc[future.ts > T, "mm"] = 99.0
    a = F.features_at(past, ST, LAT, LON, [T])
    b = F.features_at(future, ST, LAT, LON, [T])
    pd.testing.assert_frame_equal(a, b)


def test_alvo_usa_so_o_futuro():
    r = readings(T + pd.Timedelta("6h"))
    y1 = F.point_targets(r, ST, LAT, LON, [T], C.HORIZONS, C.THRESHOLDS)
    r2 = r.copy()
    r2.loc[r2.ts <= T, "mm"] = 0.0                                     # mexer no passado não muda o alvo
    y2 = F.point_targets(r2, ST, LAT, LON, [T], C.HORIZONS, C.THRESHOLDS)
    assert np.allclose(y1[[f"y_{h}h" for h in C.HORIZONS]], y2[[f"y_{h}h" for h in C.HORIZONS]])
    near = r[(r.station_id == "perto") & (r.ts > T) & (r.ts <= T + pd.Timedelta("1h"))].mm.sum()
    assert y1["y_1h"].iloc[0] == pytest.approx(near)


def test_estacao_sem_leitura_fica_ausente_e_nao_zero():
    r = readings(T)
    r = r[~((r.station_id == "perto") & (r.ts > T - pd.Timedelta("3h")))]   # a estação do ponto parou há 3 h
    x = F.features_at(r, ST, LAT, LON, [T])
    assert np.isnan(x["p_60"].iloc[0]) and x["n_near_valid"].iloc[0] == 0


def test_split_temporal_com_folga():
    t = pd.Series(pd.to_datetime(["2024-07-03T07:00Z", "2024-07-03T10:00Z", "2024-07-03T15:00Z", "2025-07-30T20:00Z", "2025-08-01T00:00Z"]))
    assert list(C.part(t)) == ["train", "guard", "valid", "guard", "test"]


needs_model = pytest.mark.skipif(not (C.MODELS_DIR / "metadata.json").exists(), reason="treine a previsão de chuva antes")


@pytest.fixture
def client(monkeypatch):
    from mapi_ai.rain.serve import RainForecaster
    monkeypatch.setattr(app_module, "rain_model", RainForecaster())
    return TestClient(app_module.app)


def payload(r):
    geo = ST.set_index("station_id")
    return {"latitude": LAT, "longitude": LON,
            "readings": [{"station_id": x.station_id, "latitude": geo.lat[x.station_id], "longitude": geo.lon[x.station_id],
                          "ts": x.ts.isoformat(), "mm": x.mm} for x in r.itertuples()]}


@needs_model
def test_rota_devolve_quantis_e_probabilidades_por_horizonte(client):
    j = client.post("/v1/forecast/rain", json=payload(readings(T))).json()
    assert j["model_version"] == C.MODEL_VERSION and j["target"] == "MEASURED"
    assert [h["horizon_hours"] for h in j["horizons"]] == [1, 3, 6]
    for h in j["horizons"]:
        q = h["quantiles_mm"]
        assert 0 <= q["q50"] <= q["q80"] <= q["q95"]
        assert 0 <= h["prob_forte"] <= 1 and 0 <= h["prob_media"] <= 1
    assert j["issued_at"] == "2025-05-10T12:00:00Z" and j["sources"] == ["CEMADEN_PED"]
    assert "om_6h" in j["missing_inputs"]                            # sem Open-Meteo: declarado, não zerado
    assert "CEMADEN/MCTIC" in j["attribution"]


@needs_model
def test_sem_estacao_valida_perto_do_ponto_e_422(client):
    r = readings(T)
    r = r[r.station_id != "perto"]
    resp = client.post("/v1/forecast/rain", json=payload(r))
    assert resp.status_code == 422 and "p_60" in resp.json()["missing_features"]


@needs_model
def test_chuva_forte_agora_sobe_a_previsao(client):
    dry = readings(T); dry["mm"] = 0.0
    wet = dry.copy(); wet.loc[wet.ts > T - pd.Timedelta("1h"), "mm"] = 6.0
    a = client.post("/v1/forecast/rain", json=payload(dry)).json()["horizons"][0]
    b = client.post("/v1/forecast/rain", json=payload(wet)).json()["horizons"][0]
    assert b["quantiles_mm"]["q80"] > a["quantiles_mm"]["q80"] and b["prob_forte"] > a["prob_forte"]
