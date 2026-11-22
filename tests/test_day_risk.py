"""Maquinaria do "dia de risco" (bloco C) em dados sintéticos: vazamento, rótulo com lacuna, janelas, bootstrap."""
import numpy as np
import pandas as pd
import pytest

from mapi_ai.dayrisk import config as C
from mapi_ai.dayrisk import evaluate as E


def _p60(values, start="2025-03-01T00:00:00Z"):
    idx = pd.date_range(start, periods=len(values), freq="10min", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_antecedencia_cobre_a_janela_com_atraso_de_publicacao():
    for h in C.HORIZONS:
        n = C.lead_days(h)
        assert h + C.PUBLICATION_DELAY_H <= 24 * n
    assert [C.lead_days(h) for h in (6, 12, 24)] == [1, 1, 2]


def test_rotulo_um_quando_hora_cheia_passa_do_limiar_dentro_da_janela():
    v = np.zeros(200)
    v[6 + 30] = 30.0                                  # hora cheia que termina 6 h depois de t
    s = _p60(v)
    t = pd.DatetimeIndex([s.index[6]])
    assert E.label(s, t, 6)[0] == 1.0
    assert E.label(s, t, 3)[0] == 0.0                 # fora da janela de 3 h


def test_hora_cheia_que_comeca_antes_de_t_nao_conta():
    v = np.zeros(200)
    v[6 + 5] = 30.0                                   # termina 50 min depois de t: começou antes de t
    s = _p60(v)
    assert E.label(s, pd.DatetimeIndex([s.index[6]]), 6)[0] == 0.0


def test_lacuna_nunca_vira_sem_evento():
    v = np.zeros(200)
    v[10:30] = np.nan                                 # 20 de 31 passos da janela de 6 h sem dado
    s = _p60(v)
    assert np.isnan(E.label(s, pd.DatetimeIndex([s.index[6]]), 6)[0])
    v[12] = 40.0                                      # mas um evento válido basta para o 1
    s = _p60(v)
    assert E.label(s, pd.DatetimeIndex([s.index[6]]), 6)[0] == 1.0


def test_obs24_ausente_com_qualquer_passo_faltando():
    v = np.ones(300) * 0.1
    s = _p60(v)
    t = pd.DatetimeIndex([s.index[200]])
    assert E.obs24(s, t)[0] == pytest.approx(14.4)
    v[150] = np.nan
    assert np.isnan(E.obs24(_p60(v), t)[0])


def _om(hours, lead, values, slug="X", model="m"):
    t = pd.date_range("2025-03-01T01:00:00Z", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"slug": slug, "model": model, "hour_end_utc": t, "lead_days": lead, "precip_mm": values,
                         "cape": values})


def test_janela_om_usa_so_horas_que_terminam_em_t_mais_1_ate_t_mais_h():
    vals = np.zeros(72)
    vals[9] = 7.0                                     # hora que termina em t0 + 10 h
    om = _om(72, 1, vals)
    t = pd.DatetimeIndex([pd.Timestamp("2025-03-01T00:00:00Z")])
    assert E.om_window(om, "X", "m", t, 12)[0] == 7.0
    assert E.om_window(om, "X", "m", t, 6)[0] == 0.0
    vals[3] = np.nan
    om = _om(72, 1, vals)
    assert np.isnan(E.om_window(om, "X", "m", t, 6)[0])   # sem soma parcial


def test_janela_om_recusa_antecedencia_curta(monkeypatch):
    monkeypatch.setattr(C, "lead_days", lambda h: 1)
    om = _om(72, 1, np.zeros(72))
    with pytest.raises(AssertionError, match="vazamento"):
        E.om_window(om, "X", "m", pd.DatetimeIndex([pd.Timestamp("2025-03-01T00:00:00Z")]), 24)


def test_bss_da_climatologia_contra_ela_mesma_e_zero_e_bootstrap_e_deterministico():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(float)
    pc = np.full(500, 0.1)
    days = np.repeat(np.arange(100), 5).astype(str)
    a = E.bss_ci(y, pc, pc, days)
    assert a["bss"] == 0.0 and a["ic95"] == [0.0, 0.0]
    p = np.clip(y * 0.6 + 0.05, 0, 1)
    assert E.bss_ci(y, p, pc, days) == E.bss_ci(y, p, pc, days)
    assert E.bss_ci(y, p, pc, days)["bss"] > 0


def test_emissao_das_06h_local_e_09_utc():
    t = E.emissions(pd.Timestamp("2025-03-01T00:00:00Z"), pd.Timestamp("2025-03-03T00:00:00Z"), (6,))
    assert list(t.hour) == [9, 9] and str(t.tz) == "UTC"


def test_selecao_cai_na_climatologia_quando_o_ic_do_melhor_inclui_zero():
    ms = {m: {"bss": -0.1, "ic95": [-0.3, 0.1]} for m in C.MODELS}
    ms["climatologia"] = {"bss": 0.0, "ic95": [0.0, 0.0]}
    ms["om_media"] = {"bss": 0.05, "ic95": [-0.02, 0.12]}
    assert E.select({"modelos": ms})["escolhido"] == "climatologia"
    ms["om_media"] = {"bss": 0.08, "ic95": [0.03, 0.12]}
    ms["om_icon_global"] = {"bss": 0.04, "ic95": [0.0, 0.08]}
    assert E.select({"modelos": ms})["escolhido"] == "om_icon_global"
