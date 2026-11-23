"""Capacidade efetiva em faixa (Fase 4.1) e pico de 10 min previsto (Fase 4.2) no /v1/flood/forecast."""
import json

import numpy as np
import pytest

from mapi_ai import app as app_module
from mapi_ai import bucket10 as B
from mapi_ai.flood import capacidade as CAP
from mapi_ai.flood import config as FC
from mapi_ai.flood.forecast import forecast_point
from mapi_ai.rain import perfil as PF
from tests.test_flood_forecast import LAT, LON, T, FakeRain, point, readings

needs_rain = pytest.mark.skipif(app_module.rain_model is None, reason="rain_v1 não carregado")


def terr():
    pt = point()
    return B.Terreno(pt["area_m2"], np.asarray(pt["z_curva_m"]), np.asarray(pt["v_curva_m3"]), pt["rua_rel_m"])


def test_capacidade_em_m3s_no_balde():
    p = B.Params10(q_cap_m3s=0.1)
    assert p.q_passo_m3(20000.0) == pytest.approx(60.0)
    P = np.full(6, 2.0)   # 2 mm por passo em 20.000 m2 com C 0,86 = 34,4 m3 por passo
    out = B.simular_passos(P, terr(), p)
    assert out["outflow_m3_h"][0] == pytest.approx(34.4 * 6)
    # com a capacidade de projeto (IDF), o mesmo valor de antes
    assert B.Params10().q_passo_m3(20000.0) == pytest.approx(0.86 * B.Params10().i_cap_mm_h * 20 * 10 / 60)


def test_perfil_pico_no_meio_limitado_ao_acumulado():
    perf = CAP.Perfil({"q50": {1: 4.0, 3: 50.0, 6: 1.0}, "q95": {1: 8.0, 3: 9.0, 6: 2.0}})
    s = perf.serie(10.0, 1, "q50")
    assert len(s) == 6 and s[2] == 4.0 and s.sum() == pytest.approx(10.0) and np.allclose(np.delete(s, 2), 1.2)
    s3 = perf.serie(12.0, 3, "q50")          # pico previsto maior que o acumulado: o pico vira o acumulado
    assert s3[8] == 12.0 and s3.sum() == pytest.approx(12.0)
    assert np.allclose(CAP.Perfil(None).serie(6.0, 1), 1.0)
    w = PF.alt_block_profile(1)
    assert w.sum() == pytest.approx(1.0) and int(np.argmax(w)) == 2 and w.max() == pytest.approx(PF.alt_block_fraction(1))


def faixa_json(tmp_path, qmin, qmax, kappa=None):
    f = tmp_path / "cap.json"
    f.write_text(json.dumps({"pontos": {"TESTE": {"q_min_m3s": qmin, "q_max_m3s": qmax, "base": "teste"}}}))
    c = tmp_path / "cal.json"
    c.write_text(json.dumps({"pontos": {} if kappa is None else {"TESTE": {"kappa": kappa}}}))
    return f, c


def run(q50, q95):
    return forecast_point(latitude=LAT, longitude=LON, readings=readings(0.0), rain_10min_mm=[0.0] * 36, terrain=terr(),
                          issued_at=T.isoformat(), rain_model=FakeRain(q50, q95), slug="TESTE")


@needs_rain
def test_regra_da_faixa_alerta_com_qmax_e_atencao_com_qmin(tmp_path, monkeypatch):
    f, c = faixa_json(tmp_path, 0.005, 0.5)
    monkeypatch.setattr(CAP, "ARQUIVO", f)
    monkeypatch.setattr(CAP, "CALIBRACAO", c)
    # 6 mm/h por igual: 5,7 m3 por passo de 10 min no terreno de teste (20.000 m2); Q_min tira 3 m3, Q_max tira 300 m3
    out = run(6.0, 6.0)
    h1 = out["horizons"][0]
    assert h1["sheet_q50_cm"] == 0.0 and h1["sheet_q95_cm"] >= FC.HCRIT_CM and h1["level"] == "ATENCAO"
    assert h1["sheet_q50_cm_qmin"] >= FC.HCRIT_CM and h1["sheet_q95_cm_qmax"] == 0.0
    cap = out["explanation"]["capacity"]
    assert cap["q_min_m3s"] == 0.005 and cap["q_max_m3s"] == 0.5 and cap["base"] == "teste"
    assert out["explanation"]["served_capacity"] == "q_min" and out["explanation"]["sensitivity"]["ranges"]["q_m3s"][0] == 0.005
    # com Q_max pequeno também, ALERTA
    f2, c2 = faixa_json(tmp_path, 0.005, 0.006)
    monkeypatch.setattr(CAP, "ARQUIVO", f2)
    assert run(6.0, 6.0)["level"] == "ALERTA"


@needs_rain
def test_kappa_calibrado_vale_nas_duas_pontas(tmp_path, monkeypatch):
    f, c = faixa_json(tmp_path, 0.005, 0.5, kappa=1.0)
    monkeypatch.setattr(CAP, "ARQUIVO", f)
    monkeypatch.setattr(CAP, "CALIBRACAO", c)
    out = run(6.0, 6.0)
    assert out["explanation"]["capacity"]["q_min_m3s"] == out["explanation"]["capacity"]["q_max_m3s"] == 0.5
    assert out["level"] == "SEM_ALERTA"


@needs_rain
def test_modelo_real_traz_o_pico_e_o_perfil():
    rm = app_module.rain_model
    assert rm.peak is not None
    out = forecast_point(latitude=LAT, longitude=LON, readings=readings(3.0), rain_10min_mm=[3.0] * 36, terrain=terr(),
                         issued_at=T.isoformat(), rain_model=rm, slug="TESTE")
    assert out["explanation"]["profile"]["model"] == "rain_v1p"
    for h in out["horizons"]:
        assert 0 <= h["peak10_q50_mm"] <= h["rain_q50_mm"] + 1e-9
    prev = [s for s in out["explanation"]["decomposition"] if s["kind"] == "previsto"]
    n = len(prev)
    assert prev[(n - 1) // 2]["rain_mm"] >= max(s["rain_mm"] for s in prev) - 1e-9
