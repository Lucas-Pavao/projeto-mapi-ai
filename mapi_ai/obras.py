"""
Obras que mudam a física de um ponto (onda 16, Fase 3.4; data/obras.json, a mesma lista de ObrasProperties da API).
Nenhuma validação mistura antes e depois: `garante_sem_mistura` levanta erro se um ponto tiver linhas dos dois períodos, e
`separa` devolve um pedaço por (ponto, período). Data de início vazia = a obra ainda não opera: tudo é ANTES.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ARQUIVO = Path(__file__).resolve().parents[1] / "data" / "obras.json"
TZ = "America/Recife"


class ObraMisturada(ValueError):
    """Um conjunto de validação junta instantes de antes e de depois da operação de uma obra no mesmo ponto."""


def carrega(caminho: Path = ARQUIVO) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))["pontos"]


def periodo(slug: str, t, obras: dict | None = None) -> pd.Series | str:
    """ANTES, DEPOIS ou SEM_OBRA. `t` pode ser um instante ou uma série de instantes (UTC); vale o dia em America/Recife."""
    obras = carrega() if obras is None else obras
    lista = obras.get(slug) or []
    serie = isinstance(t, (pd.Series, pd.DatetimeIndex, list))
    ts = pd.to_datetime(pd.Series(t) if serie else pd.Series([t]), utc=True)
    if not lista:
        out = pd.Series("SEM_OBRA", index=ts.index)
    else:
        dia = ts.dt.tz_convert(TZ).dt.date
        inicios = [pd.Timestamp(o["inicio_operacao"]).date() for o in lista if o.get("inicio_operacao")]
        out = pd.Series("ANTES", index=ts.index)
        for d0 in inicios:
            out[dia >= d0] = "DEPOIS"
    return out if serie else out.iloc[0]


def garante_sem_mistura(df: pd.DataFrame, col_slug: str = "slug", col_t: str = "t", obras: dict | None = None) -> pd.DataFrame:
    """Acrescenta `obra_periodo` e levanta ObraMisturada se algum ponto tiver ANTES e DEPOIS juntos."""
    out = df.copy()
    out["obra_periodo"] = "SEM_OBRA"
    for slug, g in out.groupby(col_slug):
        per = periodo(slug, g[col_t].reset_index(drop=True), obras)
        out.loc[g.index, "obra_periodo"] = per.to_numpy()
        if {"ANTES", "DEPOIS"} <= set(per):
            raise ObraMisturada(f"{slug}: o conjunto junta antes e depois da obra; separe com obras.separa()")
    return out


def separa(df: pd.DataFrame, col_slug: str = "slug", col_t: str = "t", obras: dict | None = None) -> dict:
    """{(ponto, período): pedaço}. Cada pedaço é validado sozinho."""
    out = df.copy()
    out["obra_periodo"] = "SEM_OBRA"
    for slug, g in out.groupby(col_slug):
        out.loc[g.index, "obra_periodo"] = periodo(slug, g[col_t].reset_index(drop=True), obras).to_numpy()
    return {k: g for k, g in out.groupby([col_slug, "obra_periodo"])}
