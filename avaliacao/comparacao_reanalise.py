"""PRE-REGISTRO §4.3: distribuição da chuva horária Open-Meteo (reanálise) × pluviômetro Cemaden (fuso verificado),
SEM nenhuma informação de alagamento. Só chuva.

Uso: python -m avaliacao.comparacao_reanalise --entradas DIR --saida arquivo.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .entradas import PONTOS, _recusa_rotulos, carregar_chuva, sha256

QS = [0.5, 0.9, 0.99, 0.999]
LIMS = [25.2, 35.2, 50.3, 60.1]


def resumo(x: pd.Series) -> dict:
    return {"n_horas": int(x.size), "max_mm_h": float(x.max()),
            **{f"q{q}": float(x.quantile(q)) for q in QS},
            **{f"n_horas_ge_{l}": int((x >= l).sum()) for l in LIMS},
            "horas_chuva_gt0": int((x > 0).sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--entradas", required=True)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args(argv)
    ent = Path(a.entradas)
    om = carregar_chuva(ent / "chuva_open_meteo_archive.csv")
    f_cem = ent / "chuva_cemaden_recife.csv"
    _recusa_rotulos(f_cem)
    cem = pd.read_csv(f_cem, parse_dates=["ts_utc"])
    # Cemaden: leituras EVENT (mm acumulados no intervalo da leitura); soma por hora, rotulada pelo FIM da hora (ADR 0006)
    cem["fim_hora"] = cem.ts_utc.dt.ceil("h")
    cem_h = cem.groupby(["station_id", "fim_hora"]).precipitation_mm.sum()

    out = {"open_meteo_2021-04_a_2024-12": {s: resumo(om[s]) for s in PONTOS}}
    om_all = pd.concat(om.values())
    out["open_meteo_4_pontos_juntos"] = resumo(om_all)
    om_min, om_max = om[PONTOS[0]].index.min(), om[PONTOS[0]].index.max()
    comum = cem_h[(cem_h.index.get_level_values(1) >= om_min) & (cem_h.index.get_level_values(1) <= om_max)]
    out["cobertura"] = {
        "open_meteo_janela": [str(om_min), str(om_max)],
        "cemaden_janela_no_banco": [str(cem.ts_utc.min()), str(cem.ts_utc.max())],
        "cemaden_estacoes_recife": int(cem.station_id.nunique()),
        "cemaden_leituras": int(len(cem)),
        "horas_comuns_cemaden_x_open_meteo": int(len(comum)),
        "criterio_minimo_do_protocolo": ">= 1 ano de horas comuns (MODELAGEM §2.3)",
        "comparacao_possivel": bool(len(comum) >= 8760),
    }
    out["cemaden_descritivo_sem_comparacao"] = resumo(cem_h) if len(cem_h) else {}
    out["entradas_sha256"] = {p.name: sha256(p) for p in (ent / "chuva_open_meteo_archive.csv", f_cem)}
    Path(a.saida).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["cobertura"], indent=1))


if __name__ == "__main__":
    main()
