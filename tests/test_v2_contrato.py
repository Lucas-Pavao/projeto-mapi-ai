"""Etapa 2 do IN1165: o app carrega o artefato v2 (sem scaler, com metadata) sem quebrar o contrato."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai import in1165_config as C

PAYLOAD = {
    "station_id": "p1", "lat": -8.05, "lon": -34.9,
    "current_rainfall": 30.0, "rainfall_3h_accumulated": 60.0, "rainfall_6h_accumulated": 70.0,
    "rainfall_12h_accumulated": 80.0, "rainfall_24h_accumulated": 90.0,
    "tide_level": 2.2, "timestamp": "2026-05-10T15:00:00Z",
}

pytestmark = pytest.mark.skipif(not (C.MODELS_DIR / "metadata.json").exists(), reason="treine antes (Etapa 2)")


@pytest.fixture
def client():
    clf, cal, meta = app_module.load_v2()
    app_module.v2_model, app_module.v2_calibrator, app_module.metadata = clf, cal, meta
    return TestClient(app_module.app)


def test_v2_responde_com_versao_fonte_e_aviso(client):
    r = client.post("/v1/predict/flood", json=PAYLOAD)
    assert r.status_code == 200, r.text
    j = r.json()
    assert 0.0 <= j["flood_probability"] <= 1.0
    assert j["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert j["model_version"] == "xgb_v2"
    assert j["rain_source"] == "MEASURED"
    assert j["training_data"] == "SYNTHETIC" and "SINTÉTICO" in j["synthetic_warning"]
    assert set(j["thresholds"]) == {"medium", "high", "source"}


def test_v2_chuva_forte_com_mare_alta_tem_mais_risco_que_seco(client):
    seco = PAYLOAD | {k: 0.0 for k in C.RAIN_FEATURES} | {"tide_level": 1.0}
    p_seco = client.post("/v1/predict/flood", json=seco).json()["flood_probability"]
    p_chuva = client.post("/v1/predict/flood", json=PAYLOAD).json()["flood_probability"]
    assert p_chuva > p_seco


def test_v2_sem_coordenada_422(client):
    body = {k: v for k, v in PAYLOAD.items() if k not in ("lat", "lon")}
    r = client.post("/v1/predict/flood", json=body)
    assert r.status_code == 422
    assert set(r.json()["missing_features"]) == {"latitude", "longitude"}


def test_v2_sem_scaler_e_features_na_ordem_do_metadata():
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    assert meta["features"] == C.FEATURES
    clf = app_module.load_v2()[0]
    assert list(clf.get_booster().feature_names) == C.FEATURES


def test_split_temporal_sem_sobreposicao():
    import pandas as pd
    from mapi_ai.trainer import temporal_split
    df = pd.read_csv(C.DATASET_CSV, parse_dates=["timestamp"])
    tr, va, te = temporal_split(df)
    assert tr.timestamp.max() <= va.timestamp.min() <= va.timestamp.max() <= te.timestamp.min()
    assert len(tr) == 3000 and len(va) == 1000 and len(te) == 1000


def test_nan_continua_nan_no_treino():
    import pandas as pd
    from mapi_ai.trainer import feature_matrix
    df = pd.read_csv(C.DATASET_CSV, parse_dates=["timestamp"]).head(3).copy()
    df.loc[0, "rainfall_6h_accumulated"] = np.nan
    X, _ = feature_matrix(df)
    assert np.isnan(X.iloc[0]["rainfall_6h_accumulated"])
