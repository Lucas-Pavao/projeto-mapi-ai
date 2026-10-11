"""Carga das entradas da etapa 1 (CEGA). Só chuva, terreno e normal climatológica.

Garantia de cegueira, no código: nenhum arquivo carregado aqui pode conter coluna de rótulo. `_recusa_rotulos`
lê SÓ o cabeçalho de cada CSV e aborta se achar qualquer nome proibido, ANTES de ler os dados em memória.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .balde import Ponto

PONTOS = ["AV_RECIFE_IBURA", "CIN_UFPE", "AGAMENON_DERBY", "MASCARENHAS_IMBIRIBEIRA"]
PROIBIDAS = ("label", "isflooded", "severity", "occurrence", "flood_occurrence", "ckan")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _recusa_rotulos(path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        cab = next(csv.reader(f))
    ruins = [c for c in cab if any(p in c.lower() for p in PROIBIDAS)]
    if ruins:
        raise RuntimeError(f"{path}: colunas de rótulo proibidas na etapa cega: {ruins}")
    return cab


def dia_local(ts_fim_utc) -> pd.DatetimeIndex:
    """Dia local (America/Recife, UTC-3) da hora que TERMINA em t (UTC): data de t - 4 h (MODELAGEM §2.4)."""
    return pd.DatetimeIndex(pd.to_datetime(ts_fim_utc)) - pd.Timedelta(hours=4)


def carregar_chuva(path) -> dict[str, pd.Series]:
    _recusa_rotulos(path)
    df = pd.read_csv(path, parse_dates=["ts_utc_fim_intervalo"])
    assert set(df.columns) == {"slug", "ts_utc_fim_intervalo", "precipitation_mm"}
    out = {}
    for slug in PONTOS:
        s = df[df.slug == slug].set_index("ts_utc_fim_intervalo")["precipitation_mm"].sort_index()
        assert s.index.is_unique and s.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all(), slug
        out[slug] = s
    idx0 = out[PONTOS[0]].index
    assert all(s.index.equals(idx0) for s in out.values())
    return out


def carregar_terreno(path_pt, path_curva) -> dict[str, Ponto]:
    _recusa_rotulos(path_pt)
    _recusa_rotulos(path_curva)
    pt = pd.read_csv(path_pt).set_index("slug")
    cv = pd.read_csv(path_curva)
    out = {}
    for slug in PONTOS:
        r = pt.loc[slug]
        c = cv[cv.slug == slug].sort_values("altura_acima_fundo_m")
        z = c.altura_acima_fundo_m.to_numpy(float)
        v = c.volume_acumulado_m3.to_numpy(float)
        assert z[0] == 0 and np.all(np.diff(z) > 0) and np.all(np.diff(v) >= 0), slug
        out[slug] = Ponto(
            slug=slug,
            area_m2={5: float(r.area_contribuinte_m2_snap5m), 10: float(r.area_contribuinte_m2_snap10m),
                     20: float(r.area_contribuinte_m2_snap20m)},
            z_curva_m=z, v_curva_m3=v,
            rua_rel_m=float(r.cota_rua_m_raio3m - r.fundo_bacia_cota_m),  # só diferença de cotas (ADR 0009)
        )
    return out


def carregar_normal_inmet(path) -> dict[int, float]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(k): float(v) for k, v in d["mm_por_mes"].items()}
