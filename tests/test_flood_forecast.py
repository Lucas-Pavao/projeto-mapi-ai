"""Alerta de alagamento por regressão (mapi_ai/flood, POST /v1/flood/forecast). Testes obrigatórios das Etapas 1 e 3."""
import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mapi_ai import app as app_module
from mapi_ai import bucket10 as B
from mapi_ai.flood import config as FC
from mapi_ai.flood import explain as FX
from mapi_ai.flood.forecast import forecast_point

client = TestClient(app_module.app)
LAT, LON = -8.05531, -34.95116
T = pd.Timestamp("2025-05-10T12:00:00Z")
STATIONS = [("perto", LAT + 0.01, LON), ("norte", LAT + 0.1, LON), ("oeste", LAT, LON - 0.1)]


def readings(mm_per_10min: float, hours=6, missing=()):
    ts = pd.date_range(T - pd.Timedelta(hours=hours) + pd.Timedelta("10min"), T, freq="10min")
    return [{"station_id": s, "latitude": la, "longitude": lo, "ts": t.isoformat(),
             "mm": None if s in missing else mm_per_10min} for s, la, lo in STATIONS for t in ts]


def point(area=20000.0, rua_rel=0.10):
    z = np.linspace(0, 2.0, 41)
    return {"slug": "TESTE", "area_m2": area, "z_curva_m": z.tolist(), "v_curva_m3": (100.0 * z).tolist(), "rua_rel_m": rua_rel}


def body(mm=0.0, series=None, **kw):
    b = {"latitude": LAT, "longitude": LON, "readings": readings(mm), "issued_at": T.isoformat(),
         "rain_10min_mm": series if series is not None else [mm] * 36, "point": point()}
    b.update(kw)
    return b


class FakeRain:
    """rain_v1 de mentira: quantis fixos, para testar a física e a regra sem depender do modelo treinado."""
    meta = {"model_version": "rain_v1", "training_data": "MEASURED", "target": "MEASURED"}

    def __init__(self, q50, q95):
        self.q50, self.q95 = q50, q95
        real = app_module.rain_model
        self.models, self.chosen = real.models, real.chosen
        self._real = real

    def forecast_with_inputs(self, *a, **k):
        rain, x = self._real.forecast_with_inputs(*a, **k)
        for h in rain["horizons"]:
            h["quantiles_mm"] = {"q50": self.q50 * h["horizon_hours"], "q80": self.q95 * h["horizon_hours"],
                                 "q95": self.q95 * h["horizon_hours"]}
        return rain, x


needs_rain = pytest.mark.skipif(app_module.rain_model is None, reason="rain_v1 não carregado")


@needs_rain
def test_chuva_zero_da_lamina_zero_e_sem_alerta():
    r = client.post("/v1/flood/forecast", json=body(0.0))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["level"] == "SEM_ALERTA" and out["worst_horizon_hours"] == 6
    assert out["h_now_cm"] == 0.0
    assert all(h["sheet_q50_cm"] == 0.0 for h in out["horizons"]) or all(h["level"] == "SEM_ALERTA" for h in out["horizons"])
    assert out["experimental"] is True and out["attribution"] == "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC"
    assert out["code_version"] and out["issued_at"] == "2025-05-10T12:00:00Z"
    assert [h["valid_until"] for h in out["horizons"]] == ["2025-05-10T13:00:00Z", "2025-05-10T15:00:00Z", "2025-05-10T18:00:00Z"]


@needs_rain
def test_chuva_forte_sustentada_da_alerta():
    # 40 mm a cada 10 min por 6 h (240 mm/h) num terreno raso: a lâmina passa de 15 cm já agora
    out = client.post("/v1/flood/forecast", json=body(40.0)).json()
    assert out["level"] == "ALERTA", out
    assert out["worst_horizon_hours"] == 1
    assert out["h_now_cm"] >= FC.HCRIT_CM


@needs_rain
def test_regra_atencao_quando_so_o_q95_chega(monkeypatch):
    t = B.Terreno(20000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.10)
    out = forecast_point(latitude=LAT, longitude=LON, readings=readings(0.0), rain_10min_mm=[0.0] * 36, terrain=t,
                         issued_at=T.isoformat(), rain_model=FakeRain(q50=1.0, q95=200.0))
    assert out["level"] == "ATENCAO"
    h1 = out["horizons"][0]
    assert h1["sheet_q50_cm"] < FC.HCRIT_CM <= h1["sheet_q95_cm"]
    assert out["explanation"]["served_scenario"] == "q95"


@needs_rain
def test_chuva_ausente_em_estacao_vira_sem_dado_nunca_zero():
    b = body(5.0)
    b["readings"] = readings(5.0, missing=("perto", "norte", "oeste"))
    out = client.post("/v1/flood/forecast", json=b).json()
    assert out["level"] == "SEM_DADO" and out["reason"] == "CHUVA_AUSENTE"
    assert out["reason_text"]


@needs_rain
def test_lacuna_na_serie_do_ponto_vira_sem_dado():
    s = [1.0] * 36
    s[30] = None
    out = client.post("/v1/flood/forecast", json=body(1.0, series=s)).json()
    assert out["level"] == "SEM_DADO" and out["reason"] == "SERIE_INCOMPLETA"


def test_rain_v1_fora_vira_sem_dado(monkeypatch):
    monkeypatch.setattr(app_module, "rain_model", None)
    out = client.post("/v1/flood/forecast", json=body(10.0)).json()
    assert out["level"] == "SEM_DADO" and out["reason"] == "PREVISAO_INDISPONIVEL"
    assert out["level"] != "SEM_ALERTA"


def test_rain_v1_quebrado_vira_sem_dado():
    class Quebrado:
        meta = FakeRain.meta

        def forecast_with_inputs(self, *a, **k):
            raise RuntimeError("boom")
    t = B.Terreno(1000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.10)
    out = forecast_point(latitude=LAT, longitude=LON, readings=[], rain_10min_mm=[0.0] * 6, terrain=t, rain_model=Quebrado())
    assert out["level"] == "SEM_DADO" and out["reason"] == "PREVISAO_INDISPONIVEL"


@pytest.mark.parametrize("status,reason", [("SEM_TERRENO", "SEM_TERRENO"), ("EM_REVISAO", "TERRENO_EM_REVISAO")])
def test_terreno(status, reason):
    out = client.post("/v1/flood/forecast", json=body(1.0, terrain_status=status)).json()
    assert out["level"] == "SEM_DADO" and out["reason"] == reason


def test_sem_ponto_e_sem_terreno():
    out = client.post("/v1/flood/forecast", json=body(1.0, point=None)).json()
    assert out["reason"] == "SEM_TERRENO"


@needs_rain
def test_nenhum_modelo_usado_e_treinado_com_rotulo_sintetico():
    out = client.post("/v1/flood/forecast", json=body(2.0)).json()
    mv = out["model_versions"]
    for k in ("rain", "bucket"):
        assert mv[k]["training_data"] not in ("SYNTHETIC", "SYNTHETIC_LABEL"), mv
    assert app_module.rain_model.meta["training_data"] == "MEASURED"
    # o pacote não importa os classificadores de rótulo (v2, v3, sintético)
    proibidos = ("mapi_ai.v3", "mapi_ai.synthetic", "mapi_ai.trainer", "mapi_ai.build_labels", "xgb_v2")
    for f in pathlib.Path(FC.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] + ([node.module] if getattr(node, "module", None) else [])
                assert not any(m and m.startswith(p) for m in mods for p in proibidos), (f, mods)


@needs_rain
def test_shap_local_soma_igual_a_previsao():
    out = client.post("/v1/flood/forecast", json=body(3.0)).json()
    s = out["explanation"]["shap"]
    assert s["available"] and len(s["top"]) == 3
    assert s["sum_check_mm"] == pytest.approx(s["raw_prediction_mm"], abs=1e-3)
    assert all("mm" in i["text"] for i in s["top"])


@needs_rain
def test_contrafactual_reproduz_o_limiar():
    t = B.Terreno(20000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.10)
    p = B.Params10()
    cf = FX.counterfactual_mm(0.0, 0.0, 2.0, t, p)
    assert cf["reached"] and cf["extra_mm"] > 0
    assert FX._scenario_sheet_cm(0.0, 0.0, 2.0 + cf["extra_mm"], 1, t, p) >= FC.HCRIT_CM
    assert FX._scenario_sheet_cm(0.0, 0.0, 2.0 + cf["extra_mm"] - 5 * FC.CF_TOL_MM, 1, t, p) < FC.HCRIT_CM


@needs_rain
def test_decomposicao_e_sensibilidade():
    out = client.post("/v1/flood/forecast", json=body(3.0)).json()
    e = out["explanation"]
    d = e["decomposition"]
    assert [x["kind"] for x in d].count("medido") == FC.DECOMP_HISTORY_STEPS
    assert [x["kind"] for x in d].count("previsto") == out["worst_horizon_hours"] * 6
    assert all(x["outflow_m3_h"] <= x["capacity_m3_h"] + 1e-9 for x in d)
    s = e["sensitivity"]
    assert s["min_cm"] <= s["max_cm"] and s["n_combinations"] == 6 * 3 * 3 * 3
    assert set(s["one_at_a_time"]) == {"f", "C", "tr_anos", "te_min"}


def test_simular_passos_igual_a_simular():
    t = B.Terreno(5000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.05)
    P = np.r_[np.zeros(5), np.full(6, 25.0), np.zeros(10)]
    h, V, fora = B.simular(P, t, B.Params10())
    d = B.simular_passos(P, t, B.Params10())
    assert np.allclose(h, d["h_mm"]) and np.allclose(V, d["V_m3"]) and fora == d["n_fora_da_curva"]


def test_health_traz_commit_real(monkeypatch):
    from mapi_ai import version
    version.code_version.cache_clear()
    monkeypatch.delenv("MAPI_CODE_VERSION", raising=False)
    h = client.get("/health").json()
    assert h["code_version"] != "unknown" and len(h["code_version"]) == 40, h
    assert h["code_version_source"] == "git"
    version.code_version.cache_clear()
    monkeypatch.setenv("MAPI_CODE_VERSION", "abc1234")
    assert client.get("/health").json()["code_version"] == "abc1234"
    version.code_version.cache_clear()


def test_balde_vetorizado_da_validacao_igual_ao_simular():
    from mapi_ai.flood.validate import VBucket, WINDOW, state, project_max_cm
    t = B.Terreno(3000.0, np.linspace(0, 2, 41), 100.0 * np.linspace(0, 2, 41), 0.05)
    rng = np.random.default_rng(3)
    mm = rng.gamma(0.5, 12.0, 400)
    mm[250] = np.nan
    ok, V, h = state(mm, VBucket(t, B.Params10()))
    assert not ok[250] and not ok[250 + WINDOW - 1] and ok[250 + WINDOW]
    for i in (WINDOW - 1, 249, 399):
        hh, VV, _ = B.simular(mm[i - WINDOW + 1:i + 1], t, B.Params10())
        assert hh[-1] == pytest.approx(h[i], abs=1e-9) and VV[-1] == pytest.approx(V[i], abs=1e-9)
    # projeção vetorizada = a do serviço
    cm = project_max_cm(np.array([V[399]]), np.array([h[399]]), np.array([40.0]), 1, VBucket(t, B.Params10()))[0]
    assert cm == pytest.approx(FX._scenario_sheet_cm(V[399], h[399] / 10, 40.0, 1, t, B.Params10()), abs=1e-9)
