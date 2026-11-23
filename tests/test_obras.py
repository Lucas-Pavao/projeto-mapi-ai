"""Obras por ponto (onda 16, Fase 3.4): a validação nunca mistura antes e depois."""
import pandas as pd
import pytest

from mapi_ai import obras as O

OBRAS = {"MASCARENHAS_IMBIRIBEIRA": [{"nome": "reservatório", "inicio_operacao": "2027-03-01"}]}


def df():
    t = pd.to_datetime(["2027-02-28T23:00:00Z", "2027-03-01T02:59:00Z", "2027-03-01T03:00:00Z", "2027-03-05T12:00:00Z"])
    return pd.DataFrame({"slug": ["MASCARENHAS_IMBIRIBEIRA"] * 4 + ["CIN_UFPE"], "t": list(t) + [t[0]]})


def test_arquivo_do_repositorio_sem_data_tudo_antes():
    assert O.periodo("MASCARENHAS_IMBIRIBEIRA", pd.Timestamp("2030-01-01T00:00:00Z")) == "ANTES"
    assert O.periodo("CIN_UFPE", pd.Timestamp("2030-01-01T00:00:00Z")) == "SEM_OBRA"


def test_dia_local_decide_e_mistura_levanta_erro():
    per = O.periodo("MASCARENHAS_IMBIRIBEIRA", df().t[:4], OBRAS)
    assert list(per) == ["ANTES", "ANTES", "DEPOIS", "DEPOIS"]      # 01/03 02h59 UTC ainda é 28/02 no Recife
    with pytest.raises(O.ObraMisturada):
        O.garante_sem_mistura(df(), obras=OBRAS)
    partes = O.separa(df(), obras=OBRAS)
    assert set(partes) == {("MASCARENHAS_IMBIRIBEIRA", "ANTES"), ("MASCARENHAS_IMBIRIBEIRA", "DEPOIS"), ("CIN_UFPE", "SEM_OBRA")}
    assert len(partes[("MASCARENHAS_IMBIRIBEIRA", "DEPOIS")]) == 2


def test_sem_data_nao_levanta():
    out = O.garante_sem_mistura(df())
    assert set(out.obra_periodo) == {"ANTES", "SEM_OBRA"}
