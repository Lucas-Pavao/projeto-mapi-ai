import numpy as np

from avaliacao.balde import Params, Ponto, simular_ponto
from avaliacao.entradas import dia_local


def ponto_sintetico(area=1000.0, rua_rel=0.10):
    # curva linear: 1 m3 por cm de coluna -> V = 100 * z (z em m)
    z = np.linspace(0, 2.0, 41)
    return Ponto("SINT", {5: area, 10: area, 20: area}, z, 100.0 * z, rua_rel)


def test_sem_chuva_nao_acumula():
    h, fora = simular_ponto(np.zeros(48), ponto_sintetico(), Params())
    assert h.max() == 0 and fora == 0


def test_chuva_abaixo_da_capacidade_escoa_tudo():
    # P = 0,9 * f * iT: Q_in < q_p, V sempre 0
    P = np.full(24, 0.9 * 60.1)
    h, _ = simular_ponto(P, ponto_sintetico(), Params())
    assert h.max() == 0


def test_excesso_acumula_volume_exato():
    # C=1, A=1000, iT=60: q_p = 60 m3/h; P = 100 -> Q_in = 100; sobra 40 m3/h.
    # curva: V = 100 z  -> z = 0,4 m apos 1 h; rua_rel = 0,1 -> h = 300 mm
    p = Params(C=1.0, iT_mm_h=60.0)
    h, fora = simular_ponto(np.array([100.0]), ponto_sintetico(), p)
    assert abs(h[0] - 300.0) < 1e-9 and fora == 0


def test_memoria_entre_horas_e_esvaziamento():
    p = Params(C=1.0, iT_mm_h=60.0)  # q_p = 60 m3/h
    P = np.array([100.0, 0.0, 0.0])
    h, _ = simular_ponto(P, ponto_sintetico(), p)
    # V1 = 40 (z 0,4 -> h 300); h2: V = 40-40 -> saida limitada pelo que ha: 0 apos hora 2
    assert abs(h[0] - 300.0) < 1e-9
    assert h[1] == 0 and h[2] == 0


def test_balanco_de_massa_fecha():
    # soma(Q_in) - soma(Q_out) = V_final; verificamos via curva linear: V = 100 z
    p = Params(C=1.0, iT_mm_h=60.0)
    P = np.array([100.0, 100.0, 100.0])  # V = 40, 80, 120 (sem escoar alem de 60/h)
    pt = ponto_sintetico(rua_rel=0.0)
    h, _ = simular_ponto(P, pt, p)
    assert np.allclose(h, [400.0, 800.0, 1200.0])


def test_fora_da_curva_satura_e_conta():
    p = Params(C=1.0, iT_mm_h=60.0)
    pt = ponto_sintetico(rua_rel=0.1)  # vmax = 200 m3
    h, fora = simular_ponto(np.array([400.0]), pt, p)  # V = 340 > 200
    assert fora == 1 and abs(h[0] - 1900.0) < 1e-9


def test_f_zero_so_armazena():
    p = Params(C=1.0, iT_mm_h=60.0, f=0.0)
    h, _ = simular_ponto(np.array([30.0, 0.0]), ponto_sintetico(rua_rel=0.0), p)
    assert abs(h[0] - 300.0) < 1e-9 and abs(h[1] - 300.0) < 1e-9  # nada escoa


def test_escala_da_curva():
    p = Params(C=1.0, iT_mm_h=60.0, curve_scale=0.5)  # V = 50 z
    h, _ = simular_ponto(np.array([100.0]), ponto_sintetico(rua_rel=0.0), p)
    assert abs(h[0] - 800.0) < 1e-9  # V = 40 -> z = 0,8


def test_chuva_nan_rejeitada():
    try:
        simular_ponto(np.array([np.nan]), ponto_sintetico(), Params())
        assert False
    except ValueError:
        pass


def test_dia_local_fim_do_intervalo():
    import pandas as pd
    t = pd.to_datetime(["2021-04-01 00:00", "2021-04-01 03:00", "2021-04-01 04:00", "2021-04-02 03:00",
                        "2021-04-02 04:00"])
    d = dia_local(t)
    # hora que termina 00:00Z cobre 20h-21h local de 03-31; 04:00Z cobre 00h-01h local de 04-01
    assert [str(x) for x in d.date] == ["2021-03-31", "2021-03-31", "2021-04-01", "2021-04-01", "2021-04-02"]
