"""Fatores por ponto do alerta único (ADR 0030): refluxo de maré a cada passo do horizonte e limpeza no /v1/flood/forecast."""
import numpy as np
import pandas as pd
import pytest

from mapi_ai import app as app_module
from mapi_ai import bucket10 as B
from mapi_ai.flood import config as FC
from mapi_ai.flood import fatores as FT
from mapi_ai.flood.forecast import forecast_point
from tests.test_flood_forecast import LAT, LON, T, FakeRain, client, point, readings

needs_rain = pytest.mark.skipif(app_module.rain_model is None, reason="rain_v1 não carregado")

N_OBS = 36
SILL_MDT, D, DELTA = 0.5, 0.6, 0.821          # soleira no NR = 1,321 m; topo do tubo = 1,921 m
SILL_NR = SILL_MDT + DELTA


def terr():
    pt = point()
    return B.Terreno(pt["area_m2"], np.asarray(pt["z_curva_m"]), np.asarray(pt["v_curva_m3"]), pt["rua_rel_m"])


def tide(obs: float, fut: float | list):
    futs = fut if isinstance(fut, list) else [fut] * FC.HORIZON_STEPS
    return [obs] * N_OBS + futs


def run(q50=2.0, q95=4.0, series=None, refluxo=None, limpeza=None):
    return forecast_point(latitude=LAT, longitude=LON, readings=readings(0.0), rain_10min_mm=series or [0.0] * N_OBS,
                          terrain=terr(), issued_at=T.isoformat(), rain_model=FakeRain(q50, q95), slug="TESTE",
                          refluxo=refluxo, limpeza=limpeza)


def refl(mode="NIVEL", obs=0.0, fut=0.0, **kw):
    return FT.Refluxo(mode=mode, tide_nr_m=tide(obs, fut), sill_mdt_m=SILL_MDT, diameter_m=D, delta_m=DELTA, **kw)


def test_simular_passos_com_fator_por_passo_igual_ao_simular():
    rng = np.random.default_rng(3)
    P = rng.gamma(0.5, 6.0, 80)
    f = rng.uniform(0, 1, 80)
    h, V, _ = B.simular(P, terr(), B.Params10(), f_passo=f)
    out = B.simular_passos(P, terr(), B.Params10(), f_passo=f)
    assert np.allclose(out["h_mm"], h) and np.allclose(out["V_m3"], V)
    assert np.allclose(out["outflow_factor"], f)


@needs_rain
def test_mare_abaixo_da_soleira_nao_muda_nada():
    base = run(q50=2.0)
    out = run(q50=2.0, refluxo=refl(obs=SILL_NR - 0.3, fut=SILL_NR - 0.1))
    assert out["level"] == base["level"] == "SEM_ALERTA"
    assert [h["sheet_q50_cm"] for h in out["horizons"]] == [h["sheet_q50_cm"] for h in base["horizons"]]
    r = out["explanation"]["factors"]["refluxo"]
    assert r["status"] == "OK" and r["applied_to_level"] and r["periods"] == [] and r["reduction_now_pct"] == 0.0
    assert "abaixo da saída" in r["text"]


@needs_rain
def test_mare_entre_soleira_e_topo_reduz_a_saida_pela_metade():
    out = run(q50=2.0, refluxo=refl(obs=SILL_NR + D / 2, fut=SILL_NR + D / 2))
    r = out["explanation"]["factors"]["refluxo"]
    assert r["reduction_now_pct"] == 50.0
    assert r["periods"][0]["mean_reduction_pct"] == 50.0 and r["periods"][0]["from"] == "2025-05-10T12:00:00Z"
    assert r["periods"][0]["to"] == "2025-05-10T18:00:00Z" and "até" in r["text"]
    prev = [s for s in out["explanation"]["decomposition"] if s["kind"] == "previsto"]
    assert all(s["outflow_factor"] == 0.5 and s["tide_nr_m"] == round(SILL_NR + D / 2, 3) for s in prev)
    assert all(s["capacity_step_m3_h"] == pytest.approx(s["capacity_m3_h"] / 2, abs=1e-3) for s in prev)


@needs_rain
def test_mare_acima_do_topo_fecha_a_saida_e_chuva_fraca_alaga():
    # 2 mm em 1 h não alaga com a galeria livre (precisa de > 20,9 mm em 10 min); com a saída fechada, alaga
    assert run(q50=2.0)["level"] == "SEM_ALERTA"
    out = run(q50=2.0, refluxo=refl(obs=0.0, fut=SILL_NR + D + 0.2))
    assert out["level"] == "ALERTA"
    r = out["explanation"]["factors"]["refluxo"]
    assert r["periods"][0]["mean_reduction_pct"] == 100.0
    assert r["served_delta_cm"] > FC.HCRIT_CM and "menor" in r["text"]


@needs_rain
def test_modo_info_devolve_a_contribuicao_sem_mudar_o_nivel():
    out = run(q50=2.0, refluxo=refl(mode="INFO", obs=0.0, fut=SILL_NR + D + 0.2))
    r = out["explanation"]["factors"]["refluxo"]
    assert out["level"] == "SEM_ALERTA" and not r["applied_to_level"]
    assert r["delta_sheet_cm"]["q50"]["1"] > FC.HCRIT_CM


@needs_rain
def test_previsao_usa_a_mare_do_horizonte_nao_a_de_agora():
    # maré baixa agora e na 1ª hora, saída fechada da 2ª à 6ª hora: 1 h sem efeito, 3 h e 6 h com efeito
    fut = [0.0] * 6 + [SILL_NR + D + 0.2] * 30
    out = run(q50=2.0, refluxo=refl(obs=0.0, fut=fut))
    r = out["explanation"]["factors"]["refluxo"]
    assert r["reduction_now_pct"] == 0.0
    assert r["periods"][0]["from"] == "2025-05-10T13:00:00Z" and "entre" in r["text"]
    assert r["delta_sheet_cm"]["q50"]["1"] == 0.0 and r["delta_sheet_cm"]["q50"]["3"] > 0
    lv = {h["hours"]: h["level"] for h in out["horizons"]}
    assert lv[1] == "SEM_ALERTA" and lv[3] == "ALERTA"


@needs_rain
def test_ponto_sem_soleira_nao_se_aplica():
    sem = FT.Refluxo(mode="NIVEL", tide_nr_m=tide(0.0, 3.0), sill_mdt_m=None, diameter_m=None, delta_m=DELTA)
    out = run(q50=2.0, refluxo=sem)
    r = out["explanation"]["factors"]["refluxo"]
    assert r["status"] == "NAO_SE_APLICA" and out["level"] == "SEM_ALERTA" and "não tem saída" in r["text"]


@needs_rain
def test_mare_ausente_com_refluxo_no_nivel_vira_sem_dado():
    t = tide(0.0, 0.0)
    t[N_OBS + 10] = None
    out = run(refluxo=FT.Refluxo(mode="NIVEL", tide_nr_m=t, sill_mdt_m=SILL_MDT, diameter_m=D, delta_m=DELTA))
    assert out["level"] == "SEM_DADO" and out["reason"] == "MARE_AUSENTE" and "1 de 72" in out["reason_detail"]
    info = run(refluxo=FT.Refluxo(mode="INFO", tide_nr_m=t, sill_mdt_m=SILL_MDT, diameter_m=D, delta_m=DELTA))
    assert info["level"] == "SEM_ALERTA"
    assert info["explanation"]["factors"]["refluxo"]["status"] == "SEM_DADO"


@needs_rain
def test_cenario_de_vies_sobe_a_soleira_e_so_informa():
    # maré 0,2 m acima do topo no cenário central; com a soleira 0,369 m mais alta, sobra saída (fator > 0)
    out = run(q50=2.0, refluxo=refl(obs=0.0, fut=SILL_NR + D + 0.2))
    b = out["explanation"]["factors"]["refluxo"]["bias_scenario"]
    assert b["sill_nr_m"] == pytest.approx(SILL_NR + 0.369, abs=1e-3)
    assert b["sheet_cm"]["q50"]["1"] < out["horizons"][0]["sheet_q50_cm"]
    assert out["level"] == "ALERTA"   # o nível é o do cenário central


def lz(mode="NIVEL", dias=None, regime=None, lam=0.0333, de=44.2, f_e=0.0):
    return FT.Limpeza(mode=mode, days_since_cleaning=dias, lambda_re_per_day=lam, regime=regime, mean_clogged_days=de,
                      f_clogged=f_e)


@needs_rain
def test_limpeza_recente_nao_muda_o_nivel():
    out = run(q50=2.0, limpeza=lz(dias=1))
    l_ = out["explanation"]["factors"]["limpeza"]
    assert l_["p_clogged"] == pytest.approx(1 - np.exp(-0.0333), abs=1e-4)
    assert out["level"] == "SEM_ALERTA" and l_["mixture_level"] == "SEM_ALERTA"
    assert l_["sheet_clogged_cm"]["q50"]["6"] > FC.HCRIT_CM     # entupida alagaria, mas a chance é de 3%


@needs_rain
def test_limpeza_antiga_no_nivel_da_alerta_e_em_info_so_informa():
    nivel = run(q50=2.0, limpeza=lz(dias=1000))
    assert nivel["level"] == "ALERTA" and nivel["explanation"]["factors"]["limpeza"]["applied_to_level"]
    info = run(q50=2.0, limpeza=lz(mode="INFO", dias=1000))
    l_ = info["explanation"]["factors"]["limpeza"]
    assert info["level"] == "SEM_ALERTA" and l_["mixture_level"] == "ALERTA" and "não muda o nível" in l_["text"]


@needs_rain
def test_limpeza_regime_permanente_e_sem_taxa():
    perm = run(q50=2.0, limpeza=lz(mode="INFO", regime="PERMANENTE"))
    l_ = perm["explanation"]["factors"]["limpeza"]
    assert l_["regime"] == "PERMANENTE" and l_["days_since_cleaning"] is None
    assert l_["p_clogged"] == pytest.approx(44.2 / (1 / 0.0333 + 44.2), abs=1e-4)
    sem = run(q50=2.0, limpeza=lz(lam=None, dias=10))
    assert sem["explanation"]["factors"]["limpeza"]["status"] == "SEM_DADO" and sem["level"] == "SEM_ALERTA"


@needs_rain
def test_rota_aceita_os_fatores_e_sem_eles_responde_como_antes():
    b = {"latitude": LAT, "longitude": LON, "readings": readings(0.0), "issued_at": T.isoformat(),
         "rain_10min_mm": [0.0] * N_OBS, "point": point()}
    antes = client.post("/v1/flood/forecast", json=b).json()
    assert antes["explanation"]["factors"]["refluxo"]["status"] == "DESLIGADO"
    b2 = dict(b, refluxo_mode="NIVEL", tide_nr_m=tide(0.0, 0.0), outlet_sill_mdt_m=SILL_MDT, outlet_diameter_m=D,
              delta_m=DELTA, limpeza_mode="INFO", lambda_re_per_day=0.0333, clog_regime="PERMANENTE",
              mean_clogged_days=44.2)
    r = client.post("/v1/flood/forecast", json=b2)
    assert r.status_code == 200, r.text
    f = r.json()["explanation"]["factors"]
    assert f["refluxo"]["status"] == "OK" and f["limpeza"]["status"] == "OK"
