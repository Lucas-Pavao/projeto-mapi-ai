"""
Cartão do alerta por regressão para a tela Modelo do front (aba "Alagamento por regressão"). Lê só relatórios gerados:
reports/regressao/validacao.json, reports/v4/rain/evaluation.json e reports/bucket10/balde10_5anos.json.
Uso: python -I scripts/cartao_regressao.py <arquivo de saída .json>
"""
import json
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "reports"


def br(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def pct(x):
    return f"{100 * x:.0f}%"


def main(out: Path):
    v = json.loads((R / "regressao" / "validacao.json").read_text())
    ev = json.loads((R / "v4" / "rain" / "evaluation.json").read_text())
    b = json.loads((R / "bucket10" / "balde10_5anos.json").read_text())
    ext = v["2.3_e_2.4_teste_externo"]
    rule = lambda k: ext["regras"][k]
    coh = v["2.2_coerencia_fisica"]["lamina_medida_chega_a_15cm"]
    rain = []
    for h in ("1h", "3h", "6h"):
        H = ev["horizons"][h]
        e = H["teste"][H["escolhido"]]["FORTE"]["eventos"]
        rain.append({"horizonHours": int(h[:-1]), "events": e["eventos"], "detected": e["detectados"],
                     "leadMedianMin": e["antecedencia_mediana_min"], "falsePerEvent": round(e["falsos_por_evento"], 1),
                     "warnedBeforeRainPct": round(e["avisados_antes_do_inicio_da_chuva_pct"], 1)})
    bucket = []
    for slug, p in b["pontos"].items():
        if isinstance(p["balde"], str):
            bucket.append({"slug": slug, "noTerrain": True})
            continue
        bucket.append({"slug": slug, "noTerrain": False, **{f"tr{t[2:]}": p["balde"][t]["eventos_ge_15cm"] for t in ("TR2", "TR5", "TR10")},
                       "dates": [d[:10] for t in ("TR2", "TR5", "TR10") for d in p["balde"][t]["datas"]]})
    cov = v["2.1_cobertura_quantis_teste"]["valores"]
    card = {
        "version": v["versao"], "rule": v["regra"],
        "pipeline": [
            {"step": "Chuva medida", "kind": "dado", "detail": "rede Cemaden/PED, 10 min"},
            {"step": "rain_v1", "kind": "validado", "detail": "quantis q50 e q95 em 1, 3 e 6 h"},
            {"step": "Balde a 10 min", "kind": "físico, não calibrado", "detail": "lâmina no terreno do ponto"},
            {"step": "Regra fixa", "kind": "fixada antes do teste", "detail": "15 cm: q50 = Alerta, só q95 = Atenção"},
        ],
        "validation": [
            {"id": "2.1", "item": "Cobertura dos quantis da chuva (teste)", "status": "medido",
             "result": " · ".join(f"{h} h: q50 {pct(cov[h]['q50'])}, q95 {pct(cov[h]['q95'])}" for h in ("1", "3", "6")), "ci": None},
            {"id": "2.2", "item": "Balde com chuva medida chega a 15 cm nos eventos FORTE", "status": "medido",
             "result": f"{coh['acertos']} de {coh['n']} eventos", "ci": coh["ic95_exato_por_dia"]},
            {"id": "2.3", "item": "Dias da Defesa Civil (2 km) com alerta", "status": "medido · uma vez",
             "result": f"{rule('alerta_regressao')['sensibilidade']['acertos']} de {rule('alerta_regressao')['sensibilidade']['n']} ponto-dias ({rule('alerta_regressao')['sensibilidade']['dias']} dias); {br(rule('alerta_regressao')['falsos_alertas_por_ponto_mes'], 2)} falsos por ponto-mês",
             "ci": rule("alerta_regressao")["sensibilidade"]["ic95_exato_por_dia"]},
            {"id": "2.4", "item": "Baseline: chuva medida 6 h ≥ 25 mm", "status": "medido",
             "result": f"{rule('baseline_chuva6h_ge_25mm')['sensibilidade']['acertos']} de {rule('baseline_chuva6h_ge_25mm')['sensibilidade']['n']}; {br(rule('baseline_chuva6h_ge_25mm')['falsos_alertas_por_ponto_mes'], 2)} falsos por ponto-mês",
             "ci": rule("baseline_chuva6h_ge_25mm")["sensibilidade"]["ic95_exato_por_dia"]},
        ],
        "smallSampleNote": f"Com {ext['regras']['alerta_regressao']['sensibilidade']['dias']} dias da Defesa Civil, o intervalo é largo: 0 de 12 dias é compatível com até {pct(rule('alerta_regressao')['sensibilidade']['ic95_exato_por_dia'][1])} de sensibilidade.",
        "finding": v["diagnostico"]["leitura"],
        "neverFired": all(x["ALERTA"] == 0 and x["ATENCAO"] == 0 for x in v["frequencia_no_teste_horas_por_ano"].values()),
        "rainWarning": rain,
        "bucket5y": bucket,
        "sources": ["reports/RELATORIO-REGRESSAO.md", "reports/regressao/validacao.json", "reports/v4/rain/evaluation.json",
                    "reports/bucket10/balde10_5anos.json"],
        "attribution": v["attribution"],
    }
    out.write_text(json.dumps(card, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(card, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
