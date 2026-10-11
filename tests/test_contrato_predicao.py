"""Contrato de predição (ADR 0007 do backend, §3 e Emenda 1): sem imputação, 422 e validation status."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai.feature_engineering import MissingFeaturesError, prepare_scenario_features

PAYLOAD = {
    "station_id": "p1", "lat": -8.05, "lon": -34.9,
    "current_rainfall": 1.0, "rainfall_3h_accumulated": 2.0, "rainfall_6h_accumulated": 3.0,
    "rainfall_12h_accumulated": 4.0, "rainfall_24h_accumulated": 5.0,
    "tide_level": 1.2, "river_level": 0.8, "timestamp": "2026-10-10T12:00:00",
    "wind_speed": 3.0, "temperature": 28.0, "apparent_temperature": 30.0, "humidity": 80.0,
    "pressure": 1010.0, "wave_height": 0.5, "wave_period": 6.0, "wave_direction": 90.0,
    "solar_radiation": 400.0,
}


class FakeModel:
    def predict_proba(self, X):
        return np.array([[0.7, 0.3]])


class FakeScaler:
    def transform(self, X):
        return X


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module, "model", FakeModel())
    monkeypatch.setattr(app_module, "scaler", FakeScaler())
    monkeypatch.setattr(app_module, "lstm_model", None)
    return TestClient(app_module.app)


def test_features_completas_nao_alteram_valores():
    df = pd.DataFrame([PAYLOAD | {"latitude": -8.05, "longitude": -34.9}])
    X = prepare_scenario_features(df, is_training=False)
    assert X.iloc[0]["river_level"] == 0.8
    assert X.iloc[0]["tide_level"] == 1.2


def test_inferencia_nao_imputa_coluna_ausente():
    df = pd.DataFrame([{"latitude": -8.0, "longitude": -34.9, "timestamp": "2026-10-10T12:00:00"}])
    with pytest.raises(MissingFeaturesError) as e:
        prepare_scenario_features(df, is_training=False)
    assert "river_level" in e.value.missing_features
    assert "tide_level" in e.value.missing_features


def test_inferencia_nao_imputa_valor_nulo():
    row = PAYLOAD | {"latitude": -8.05, "longitude": -34.9, "river_level": np.nan}
    with pytest.raises(MissingFeaturesError) as e:
        prepare_scenario_features(pd.DataFrame([row]), is_training=False)
    assert e.value.missing_features == ["river_level"]


def test_treino_mantem_preenchimento_legado():
    df = pd.DataFrame([{"latitude": -8.0, "longitude": -34.9, "timestamp": "2026-10-10T12:00:00"}])
    X = prepare_scenario_features(df, is_training=True)
    assert X.iloc[0]["river_level"] == 0.0


def test_endpoint_422_com_missing_features(client):
    body = {k: v for k, v in PAYLOAD.items() if k != "river_level"}
    r = client.post("/v1/predict/flood", json=body)
    assert r.status_code == 422
    assert r.json()["missing_features"] == ["river_level"]
    assert "detail" in r.json()


def test_endpoint_422_sem_meteorologicos(client):
    chaves = ("station_id", "lat", "lon", "current_rainfall", "rainfall_3h_accumulated",
              "rainfall_6h_accumulated", "rainfall_12h_accumulated", "rainfall_24h_accumulated",
              "tide_level", "river_level", "timestamp")
    r = client.post("/v1/predict/flood", json={k: PAYLOAD[k] for k in chaves})
    assert r.status_code == 422
    assert set(r.json()["missing_features"]) == {
        "wind_speed", "temperature", "apparent_temperature", "humidity", "pressure",
        "wave_height", "wave_period", "wave_direction", "solar_radiation"}


def test_endpoint_sem_coordenadas_422(client):
    body = {k: v for k, v in PAYLOAD.items() if k not in ("lat", "lon")}
    r = client.post("/v1/predict/flood", json=body)
    assert r.status_code == 422
    assert r.json()["missing_features"] == ["latitude", "longitude"]


def test_endpoint_sucesso_envia_versao_e_status(client):
    r = client.post("/v1/predict/flood", json=PAYLOAD)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["model_version"] == "unversioned"
    assert j["model_validation_status"] == "NOT_VALIDATED"
    assert j["risk_level"] == "LOW"
    assert "imputed_features" not in j
