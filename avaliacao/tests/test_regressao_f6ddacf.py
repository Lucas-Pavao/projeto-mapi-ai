"""Teste de REGRESSÃO do código de avaliação F1 contra a saída congelada do commit f6ddacf (F5-ACAD-02).

O QUE ESTE TESTE É: protege contra MUDANÇA DE CÓDIGO. Para 8 casos de dias (PRONTIDAO-ACADEMICA §2), roda
`gerar_escores` sobre as entradas congeladas (SHA-256 conferido) e exige que cada escore de cada modelo seja igual ao
valor congelado em `regressao_f6ddacf.json` (recorte de `escores_cegos.csv`, SHA-256 4615d416…).

O QUE ESTE TESTE NÃO É: não mede acerto. Ele NUNCA compara com rótulo (não lê rótulos nem ocorrências), não calcula AUC,
recall nem contagem de alertas por positivo, e não existe `assert alerta(dia) == 1` na versão central. Os 18 dias do
pré-registro já foram gastos; qualquer avaliação nova sobre eles é exploratória (INTERPRETACAO §6.2).
Os casos foram escolhidos por quem já conhecia os resultados (RESULTADOS §6 e §8): são canários de mecanismo, não
evidência.

Se o modelo mudar DE PROPÓSITO (M1b, pluviômetro), regenere o JSON com `congelar_regressao.py`, registre no changelog e
suba a `modelVersion`. Nunca ajuste parâmetro para o teste passar.

Entradas congeladas: variável AVALIACAO_F1_DIR (pasta com entradas/ e escores_cegos.csv; padrão ../avaliacao-f1 ao lado
do repositório). Sem elas, os testes que dependem delas são pulados (e dizem por quê); o de episódios roda sempre.
"""
import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avaliacao import gerar_escores
from avaliacao.avaliar_h1_h2 import episodios

AQUI = Path(__file__).resolve().parent
AVAL = AQUI.parent
CONGELADO = json.loads((AQUI / "regressao_f6ddacf.json").read_text(encoding="utf-8"))
BASE = Path(os.environ.get("AVALIACAO_F1_DIR", AVAL.parents[1] / "avaliacao-f1"))
NORMAL = AVAL / "dados" / "inmet_normal_82900_prec.json"
COLS = CONGELADO["origem"]["colunas"]
HASH_CSV = "4615d416e788729a7e5d5b34b1e4df8abff84cfc7e84a79860f14ed2e826b1a5"


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _hashes_esperados():
    out = {}
    for linha in (AVAL / "HASHES.sha256").read_text().splitlines():
        h, nome = linha.split(None, 1)
        out[nome.strip()] = h
    return out


@pytest.fixture(scope="module")
def escores(tmp_path_factory):
    if not (BASE / "entradas" / "chuva_open_meteo_archive.csv").exists():
        pytest.skip(f"entradas congeladas ausentes em {BASE} (defina AVALIACAO_F1_DIR)")
    for nome, h in _hashes_esperados().items():
        if nome.startswith("entradas/"):
            assert _sha(BASE / nome) == h, f"entrada {nome} diverge do hash registrado em HASHES.sha256"
    saida = tmp_path_factory.mktemp("escores")
    gerar_escores.main(["--entradas", str(BASE / "entradas"), "--normal", str(NORMAL), "--saida", str(saida)])
    return saida / "escores_cegos.csv"


def test_determinismo_csv_inteiro(escores):
    """Mesmas entradas e mesmo código => CSV byte a byte igual ao congelado (4615d416…)."""
    assert _sha(escores) == HASH_CSV


def _valor_igual(a, b):
    if b is None:
        return a is None or (isinstance(a, float) and math.isnan(a))
    if isinstance(b, (int, float)):
        return a is not None and not (isinstance(a, float) and math.isnan(a)) and math.isclose(
            float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


@pytest.mark.parametrize("caso,dias", sorted(CONGELADO["casos"].items()))
def test_escores_iguais_ao_congelado(escores, caso, dias):
    """R1–R8: cada linha (modelo, variante, ponto, dia) igual ao valor congelado. Compara com o código, não com rótulo."""
    atual = pd.read_csv(escores)
    atual["alerta"] = atual["alerta"].astype(float)
    chave = ["modelo", "variante", "ponto", "data"]
    idx = atual.set_index(chave)
    n = 0
    for linha in CONGELADO["linhas"]:
        reg = dict(zip(COLS, linha))
        if reg["data"] not in dias:
            continue
        n += 1
        row = idx.loc[tuple(reg[c] for c in chave)]
        for c in ("escore", "desempate", "alerta", "horas_no_dia"):
            v = row[c]
            v = None if (isinstance(v, float) and math.isnan(v)) else v
            assert _valor_igual(v, reg[c]), f"{caso} {reg['data']} {reg['modelo']}/{reg['variante']}/{reg['ponto']} {c}: {v} != {reg[c]}"
        assert row["periodo"] == reg["periodo"]
    assert n >= 30 * len(dias), "poucas linhas congeladas para o caso"


def test_R6_episodios_do_codigo_igual_ao_congelado():
    """R6 (junção em episódio, gap <= 3 dias): a função `episodios` devolve 13 episódios em P_principal e 17 nos 18 dias.
    As datas vêm de resultado_etapa2.json congelado (copiadas no JSON), sem consultar rótulos."""
    ep = CONGELADO["episodios"]
    parse = lambda L: [date.fromisoformat(x) for x in L]
    p, extra = parse(ep["positivos_principal"]), parse(ep["positivos_extra"])
    assert int(episodios(p).max() + 1) == ep["esperado_principal"] == 13
    assert int(episodios(p + extra).max() + 1) == ep["esperado_18"] == 17
    ids = episodios([date(2023, 6, 27), date(2023, 6, 30)])  # 3 dias de intervalo => mesmo episódio
    assert ids[0] == ids[1]


def test_R8_invariante_fisica_dia_sem_chuva(escores):
    """R8: 2021-04-02 é o primeiro dia (período de aquecimento, sem rótulo) com 0 mm nas 24 h dos 4 pontos. O balde está
    vazio: nas variantes com f > 0, S = 0, R = 0 e nenhum alerta; B3 e B0 não alertam. A variante f = 0 só armazena."""
    dia = CONGELADO["casos"]["R8"][0]
    chuva = pd.read_csv(BASE / "entradas" / "chuva_open_meteo_archive.csv", parse_dates=["ts_utc_fim_intervalo"])
    local = (chuva.ts_utc_fim_intervalo - pd.Timedelta(hours=4)).dt.normalize()
    mx = chuva.groupby(local).precipitation_mm.max()
    assert mx[mx == 0].index[0] == pd.Timestamp(dia), "regra de escolha do dia R8 mudou"
    e = pd.read_csv(escores)
    e = e[e.data == dia]
    m1 = e[e.modelo.isin(["M1a", "M1-sens"]) & (e.variante != "f0.0")]
    assert len(m1) > 20
    assert (m1.escore == 0).all() and (m1.desempate == 0).all() and (m1.alerta == 0).all()
    assert (e[e.modelo.isin(["B3", "B0"])].alerta == 0).all()
    f0 = e[e.variante == "f0.0"]
    assert (f0.desempate == 0).all()  # R = 0 também com f = 0; S > 0 é só água armazenada (nada escoa)
