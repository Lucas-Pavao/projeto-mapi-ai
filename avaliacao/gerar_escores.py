"""Etapa 1 cega (F4-IA-13): escores por dia, sem rótulos. Para aqui; não junta nada.

Uso: python -m avaliacao.gerar_escores --entradas DIR --normal JSON --saida DIR
Saída: escores_cegos.csv (formato longo) e metadata.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .balde import Params, simular_ponto
from .entradas import (PONTOS, carregar_chuva, carregar_normal_inmet, carregar_terreno, dia_local, sha256)

SEED = 20261011
INICIO = pd.Timestamp("2021-04-01")
FIM = pd.Timestamp("2024-12-31")
AQUECIMENTO_ATE = pd.Timestamp("2021-04-30")      # 30 dias de aquecimento: 04-01 a 04-30
AVAL_INI = pd.Timestamp("2023-01-01")             # período de avaliação (N e P_principal): 2023-2024

# Faixas congeladas (PRE-REGISTRO §4.2 e §6.3)
CENTRAL = Params()
UM_POR_VEZ = {
    "C0.70": CENTRAL.com(C=0.70), "C0.95": CENTRAL.com(C=0.95),
    "iT50.3": CENTRAL.com(iT_mm_h=50.3), "iT68.7": CENTRAL.com(iT_mm_h=68.7),
    "hcrit10cm": CENTRAL.com(hcrit_mm=100.0), "hcrit30cm": CENTRAL.com(hcrit_mm=300.0),
    "f0.0": CENTRAL.com(f=0.0), "f0.2": CENTRAL.com(f=0.2), "f0.4": CENTRAL.com(f=0.4),
    "f0.6": CENTRAL.com(f=0.6), "f0.8": CENTRAL.com(f=0.8),
    "curva-30%": CENTRAL.com(curve_scale=0.7), "curva+30%": CENTRAL.com(curve_scale=1.3),
    "snap5m": CENTRAL.com(snap_m=5), "snap20m": CENTRAL.com(snap_m=20),
}
CANTOS = {}
for f in (0.2, 1.0):
    for cs in (0.7, 1.3):
        for sn in (5, 20):
            CANTOS[f"canto_f{f}_curva{int(round((cs - 1) * 100)):+d}%_snap{sn}m"] = CENTRAL.com(
                f=f, curve_scale=cs, snap_m=sn)
VARIANTES_M1 = {"central": CENTRAL, **UM_POR_VEZ, **CANTOS}

B3_LIMIARES = {"B3": 50.3, "B3-47.4": 47.4, "B3-55": 55.0, "B3-50": 25.2, "B3-70": 35.2}  # mm em 60 min


def periodo(d: pd.Timestamp) -> str:
    if d <= AQUECIMENTO_ATE:
        return "aquecimento"
    return "avaliacao" if d >= AVAL_INI else "fora_avaliacao"


def agrega_dia(serie_h: pd.Series, dias: pd.DatetimeIndex, how: str) -> pd.Series:
    g = serie_h.groupby(dias.normalize())
    return g.max() if how == "max" else g.sum()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entradas", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--extra-meta", default=None, help="JSON com proveniência extra (backfill, dump)")
    a = ap.parse_args(argv)
    np.random.seed(SEED)
    ent, saida = Path(a.entradas), Path(a.saida)
    f_chuva, f_pt, f_cv = ent / "chuva_open_meteo_archive.csv", ent / "ponto_terreno.csv", ent / "curva_cota_volume.csv"

    chuva = carregar_chuva(f_chuva)
    terreno = carregar_terreno(f_pt, f_cv)
    normal = carregar_normal_inmet(a.normal)
    idx = chuva[PONTOS[0]].index
    dias_h = dia_local(idx)
    dias = pd.date_range(INICIO, FIM, freq="D")

    # completude: >= 23 de 24 horas por dia local (MODELAGEM §2.4)
    n_h = pd.Series(1, index=dias_h.normalize()).groupby(level=0).sum().reindex(dias).fillna(0)
    completo = n_h >= 23
    registros = []

    def add(modelo, variante, ponto, serie_dia, desemp=None, alerta=None):
        for d in dias:
            registros.append((modelo, variante, ponto, d.date().isoformat(),
                              float(serie_dia[d]) if completo[d] else np.nan,
                              float(desemp[d]) if (desemp is not None and completo[d]) else np.nan,
                              (int(alerta[d]) if (alerta is not None and completo[d]) else ""),
                              periodo(d), int(n_h[d])))

    # ---- M1 (balde) -----------------------------------------------------------------------------------
    fora_curva = {}
    for nome, prm in VARIANTES_M1.items():
        S_pts, R_pts = {}, {}
        for slug in PONTOS:
            P = chuva[slug].to_numpy()
            h, fora = simular_ponto(P, terreno[slug], prm)
            fora_curva[f"{nome}|{slug}"] = int(fora)
            S_pts[slug] = agrega_dia(pd.Series(h, index=idx), dias_h, "max").reindex(dias)
            # R = max P / (f * iT); com f = 0 a razão é indefinida: usa f = 1 (declarado no relatório)
            den = (prm.f if prm.f > 0 else 1.0) * prm.iT_mm_h
            R_pts[slug] = agrega_dia(chuva[slug] / den, dias_h, "max").reindex(dias)
        S_c = pd.concat(S_pts, axis=1).max(axis=1)          # máximo sobre os 4 pontos (MODELAGEM §3)
        R_c = pd.concat(R_pts, axis=1).max(axis=1)
        A_c = (S_c >= prm.hcrit_mm).astype(int)             # OU lógico dos alertas == alerta do máximo
        modelo = "M1a" if nome == "central" else "M1-sens"
        add(modelo, nome, "CIDADE", S_c, R_c, A_c)
        if nome == "central":
            for slug in PONTOS:
                add(modelo, nome, slug, S_pts[slug], R_pts[slug], (S_pts[slug] >= prm.hcrit_mm).astype(int))

    # ---- B3 / B2 / B0 ------------------------------------------------------------------------------------
    pmax = pd.concat({s: agrega_dia(chuva[s], dias_h, "max").reindex(dias) for s in PONTOS}, axis=1).max(axis=1)
    for nome, lim in B3_LIMIARES.items():
        add("B3", f"{nome}", "CIDADE", pmax, None, (pmax >= lim).astype(int))
    # B3-24h: máximo, nas horas do dia local, da soma móvel das 24 h anteriores, depois máximo nos 4 pontos
    roll = {s: agrega_dia(chuva[s].rolling(24, min_periods=24).sum(), dias_h, "max").reindex(dias) for s in PONTOS}
    r24 = pd.concat(roll, axis=1).max(axis=1)
    add("B3", "B3-24h", "CIDADE", r24, None, (r24 >= 30.0).astype(int))
    b2 = pd.Series([normal[d.month] for d in dias], index=dias)
    add("B2", "normal_mensal_INMET_82900", "CIDADE", b2)
    add("B0", "nunca_alaga", "CIDADE", pd.Series(0.0, index=dias), None, pd.Series(0, index=dias))
    # B1 (taxa de positivos do teste) NÃO é calculado aqui: depende de rótulos; constante, não afeta AUC.

    df = pd.DataFrame(registros, columns=["modelo", "variante", "ponto", "data", "escore",
                                          "desempate", "alerta", "periodo", "horas_no_dia"])
    saida.mkdir(parents=True, exist_ok=True)
    f_csv = saida / "escores_cegos.csv"
    df.to_csv(f_csv, index=False, float_format="%.10g")

    repo = Path(__file__).resolve().parents[1]
    git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
    extra = json.loads(Path(a.extra_meta).read_text()) if a.extra_meta else {}
    meta = {
        "etapa": "1 (cega) - F4-IA-13",
        "declaracao": "Nenhum rótulo de alagamento foi lido, consultado ou juntado. Entradas: chuva Open-Meteo archive, "
                      "terreno e normal INMET.",
        "csv_escores_sha256": sha256(f_csv),
        "linhas_csv": int(len(df)),
        "entradas_sha256": {p.name: sha256(p) for p in (f_chuva, f_pt, f_cv)}
                           | {"inmet_normal_82900_prec.json": sha256(a.normal)},
        "seed": SEED,
        "codigo": {"commit": git("rev-parse", "HEAD"), "arvore_suja": bool(git("status", "--porcelain"))},
        "versoes": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "plataforma": platform.platform()},
        "periodo_simulado": [str(INICIO.date()), str(FIM.date())],
        "aquecimento_ate": str(AQUECIMENTO_ATE.date()),
        "periodo_avaliacao": [str(AVAL_INI.date()), str(FIM.date())],
        "dias_incompletos": [str(d.date()) for d in dias if not completo[d]],
        "parametros_central": CENTRAL.__dict__,
        "variantes_M1": {k: v.__dict__ for k, v in VARIANTES_M1.items()},
        "limiares_B3_mm_60min": B3_LIMIARES, "B3-24h_mm": 30.0,
        "pontos": PONTOS,
        "horas_fora_da_curva": fora_curva,
        "proveniencia": extra,
    }
    (saida / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(meta["csv_escores_sha256"], len(df))


if __name__ == "__main__":
    sys.exit(main())
