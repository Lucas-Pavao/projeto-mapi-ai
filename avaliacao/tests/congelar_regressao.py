"""Gera regressao_f6ddacf.json a partir da saída JÁ CONGELADA (escores_cegos.csv, SHA-256 4615d416…, commit f6ddacf).

Uso: python avaliacao/tests/congelar_regressao.py ESCORES_CEGOS.csv RESULTADO_ETAPA2.json
Não calcula nada novo: só recorta linhas do arquivo congelado. Os valores NÃO vêm de rótulos, e sim das saídas dos modelos.
Rodar de novo só faz sentido se o modelo mudar de propósito (nova modelVersion + linha no changelog).
Os dias de R6 vêm de resultado_etapa2.json (positivos_principal/positivos_extra), só para testar a função `episodios`.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

CASOS = {
    "R1": ["2024-06-15"], "R2": ["2023-02-06"], "R3": ["2024-01-15"], "R4": ["2024-02-16"],
    "R5": ["2022-07-12"], "R6": ["2023-06-27", "2023-06-30"], "R7": ["2023-02-22"],
    "R8": ["2021-04-02"],
}


def main(escores, resultado):
    h = hashlib.sha256(Path(escores).read_bytes()).hexdigest()
    assert h.startswith("4615d416"), h
    e = pd.read_csv(escores)
    datas = sorted({d for v in CASOS.values() for d in v})
    linhas = []
    for d in datas:
        for r in e[e.data == d].itertuples(index=False):
            f = lambda x: None if pd.isna(x) else x
            linhas.append([r.modelo, r.variante, r.ponto, r.data, f(r.escore), f(r.desempate), f(r.alerta),
                           r.periodo, int(r.horas_no_dia)])
    res = json.loads(Path(resultado).read_text())
    out = {
        "origem": {"codigo": "f6ddacf97b43943126679c5e1289ce4b8bde60b8", "escores_cegos_sha256": h,
                   "colunas": ["modelo", "variante", "ponto", "data", "escore", "desempate", "alerta", "periodo",
                               "horas_no_dia"]},
        "casos": CASOS,
        "linhas": linhas,
        "episodios": {"positivos_principal": res["positivos_principal"], "positivos_extra": res["positivos_extra"],
                      "esperado_principal": res["episodios_principal"], "esperado_18": res["episodios_18"]},
    }
    linhas_txt = ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in out.pop("linhas"))
    cab = json.dumps(out, indent=1, ensure_ascii=False)
    texto = cab[:-2] + ',\n "linhas": [\n' + linhas_txt + "\n ]\n}\n"
    Path(__file__).with_name("regressao_f6ddacf.json").write_text(texto)


if __name__ == "__main__":
    main(*sys.argv[1:3])
