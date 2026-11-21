"""Entupimento e limpeza recente (mapi_ai/entupimento.py)."""
import math

import numpy as np
import pytest

from mapi_ai import bucket10 as B
from mapi_ai import entupimento as E


def terreno(area=1000.0, rua_rel=0.10):
    z = np.linspace(0, 2.0, 41)
    return B.Terreno(area, z, 100.0 * z, rua_rel)


def test_p_entupida_cresce_com_o_tempo_desde_a_limpeza():
    lam = 0.02349  # Ibura, 150 m (data/entupimento/parametros_156.json)
    assert E.p_entupida(0, lam) == 0.0
    assert round(E.p_entupida(30, lam), 3) == round(1 - math.exp(-0.7047), 3)
    assert E.p_entupida(7, lam) < E.p_entupida(30, lam) < E.p_entupida(90, lam) < 1.0


def test_sem_dado_nunca_vira_numero():
    assert E.p_entupida(None, 0.02) is None
    assert E.p_entupida(10, None) is None
    with pytest.raises(ValueError):
        E.p_entupida(-1, 0.02)


def test_parametros_do_156():
    assert E.lambda_do_ponto("AV_RECIFE_IBURA") == pytest.approx(0.02349)
    assert E.lambda_do_ponto("CIN_UFPE") is None                    # nenhuma limpeza casada
    assert E.lambda_do_ponto("MASCARENHAS_IMBIRIBEIRA") is None     # 4 limpezas < MIN_LIMPEZAS
    assert E.lambda_do_ponto("AGAMENON_DERBY") is not None          # 24 limpezas


def test_limpeza_recente_da_menos_chance_de_alagar():
    # chuva sintética que alaga só com a galeria entupida: 15 mm em 10 min < 20,9 mm (capacidade TR 5)
    P = np.array([15.0] * 6)
    t, p = terreno(), B.Params10()
    r1 = E.prob_alagar(P, t, p, t_dias_desde_limpeza=3, lambda_re_por_dia=0.02349)
    r2 = E.prob_alagar(P, t, p, t_dias_desde_limpeza=90, lambda_re_por_dia=0.02349)
    assert r1["h_livre_mm"] < p.hcrit_mm <= r1["h_entupida_mm"]
    assert r1["prob_lamina_ge_hcrit"] == pytest.approx(r1["p_entupida"])
    assert r1["prob_lamina_ge_hcrit"] < 0.1 < 0.8 < r2["prob_lamina_ge_hcrit"]


def test_sem_limpeza_conhecida_mantem_o_comportamento_de_hoje():
    P = np.array([15.0] * 6)
    r = E.prob_alagar(P, terreno(), B.Params10(), None, 0.02)
    assert r["prob_lamina_ge_hcrit"] is None
    h_hoje = B.simular(P, terreno(), B.Params10())[0][-1]
    assert r["h_livre_mm"] == pytest.approx(h_hoje)
