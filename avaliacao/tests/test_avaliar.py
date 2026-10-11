import math
import numpy as np
from avaliacao.avaliar_h1_h2 import cp, matriz_w, mcnemar_exato, episodios
from datetime import date


def test_auc_empates_valem_meio():
    sp = np.array([[1.0, 0.0], [2.0, 0.0]])
    sn = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert matriz_w(sp, sn).mean() == (0.5 + 1 + 1 + 1) / 4


def test_lexicografico_desempata_por_R():
    sp = np.array([[0.0, 2.0]])
    sn = np.array([[0.0, 1.0], [0.0, 3.0]])
    assert matriz_w(sp, sn).mean() == 0.5


def test_constante_da_auc_meio():
    assert matriz_w(np.zeros((3, 2)), np.zeros((5, 2))).mean() == 0.5


def test_clopper_pearson_apendice_a():
    lo, hi = cp(7, 14)
    assert abs(lo - 0.230) < 1e-3 and abs(hi - 0.770) < 1e-3
    lo, hi = cp(0, 689)
    assert lo == 0 and abs(hi - 0.005) < 1e-3
    lo, hi = cp(14, 14)
    assert abs(lo - 0.768) < 1e-3 and hi == 1.0


def test_mcnemar():
    assert mcnemar_exato(0, 0) == 1.0
    assert abs(mcnemar_exato(6, 0) - 0.03125) < 1e-9


def test_episodios():
    ds = [date(2023, 6, 27), date(2023, 6, 30), date(2022, 7, 8), date(2022, 7, 12)]
    assert len(set(episodios(ds))) == 3
