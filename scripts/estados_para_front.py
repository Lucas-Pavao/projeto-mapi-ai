"""
Converte as respostas da IA dos 10 estados (scripts/gerar_estados_alagamento.py) para o contrato de
GET /api/alagamento/pontos (camelCase), com as mesmas regras de FloodForecastMapper do projeto-mapi-api.
Saída: mocks msw do front. Uso: python -I scripts/estados_para_front.py <pasta com as respostas da IA> <pasta de saída>
"""
import json
import pathlib
import sys

REASONS = {"CHUVA_AUSENTE": "sem leitura de chuva válida perto do ponto",
           "PREVISAO_INDISPONIVEL": "previsão de chuva (rain_v1) fora do ar", "SEM_TERRENO": "sem terreno do ponto",
           "TERRENO_EM_REVISAO": "terreno do ponto em revisão", "SERIE_INCOMPLETA": "série de chuva de 10 min com lacuna"}
RULE = "ALERTA se a lâmina provável (q50) chega a 15 cm; ATENÇÃO se só a pessimista (q95) chega. Experimental."
# chuva medida da última hora que a API acrescenta (a IA não devolve): a mesma das séries do gerador
RAIN_HOUR = {"alerta_agora": 180.0, "alerta_1h": 45.0, "atencao_3h": 45.0, "chuva_forte_sem_encher": 45.0,
             "atencao_6h": 45.0, "sem_dado_previsao_fora": 45.0}


def explanation(e):
    if not e:
        return None
    return {"servedScenario": e["served_scenario"], "servedHorizonHours": e["served_horizon_hours"],
            "shapTop3": [{"text": s["text"], "contributionMm": s["contribution_mm"]} for s in e["shap"]["top"]],
            "counterfactualMm": e["counterfactual"].get("extra_mm"), "counterfactualText": e["counterfactual"]["text"],
            "sensitivity": {"minCm": e["sensitivity"]["min_cm"], "maxCm": e["sensitivity"]["max_cm"],
                            "levelChanges": e["sensitivity"]["level_changes"],
                            "heaviestParameter": e["sensitivity"]["heaviest_parameter"], "text": e["sensitivity"]["text"]},
            "decomposition": [{"t": d["t"], "minutesFromNow": d["minutes_from_now"], "kind": d["kind"], "rainMm": d["rain_mm"],
                               "inflowM3H": d["inflow_m3_h"], "outflowM3H": d["outflow_m3_h"], "volumeM3": d["volume_m3"],
                               "sheetCm": d["sheet_cm"]} for d in e["decomposition"]],
            "capacityM3H": e["capacity_m3_h"]}


def main(src: pathlib.Path, dst: pathlib.Path):
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.json")):
        a, name = json.loads(f.read_text()), f.stem
        sd = a["level"] == "SEM_DADO"
        absent = a.get("reason") == "CHUVA_AUSENTE"
        mv = a.get("model_versions", {})
        dto = {"floodPointId": 5, "slug": "MASCARENHAS_IMBIRIBEIRA", "name": "Av. Mascarenhas de Morais", "level": a["level"],
               "worstHorizonHours": None if sd else a["worst_horizon_hours"], "reason": a["reason"] if sd else None,
               "reasonText": REASONS[a["reason"]] if sd else None, "hNowCm": a.get("h_now_cm"),
               "rainLastHourMm": None if absent else RAIN_HOUR.get(name, 0.0),
               "horizons": [{"hours": h["hours"], "level": h["level"], "rainQ50Mm": h["rain_q50_mm"], "rainQ95Mm": h["rain_q95_mm"],
                             "sheetQ50Cm": h["sheet_q50_cm"], "sheetQ95Cm": h["sheet_q95_cm"], "validUntil": h["valid_until"]}
                            for h in a.get("horizons", [])],
               "explanation": explanation(a.get("explanation")),
               "rainStation": None if absent else "261160609A", "rainStationName": None if absent else "Imbiribeira",
               "rainStationDistanceKm": None if absent else 1.1,
               "latestReadingAgeMinutes": a.get("latest_reading_age_minutes"),
               "modelVersions": {} if not mv else {"flood": mv["flood"], "rain": mv["rain"]["version"], "bucket": mv["bucket"]["version"]},
               "sources": a.get("sources", []), "missingInputs": a.get("missing_inputs", []),
               "codeVersion": None if sd else a.get("code_version"), "issuedAt": a.get("issued_at"),
               "evaluatedAt": "2026-11-20T17:00:00Z", "experimental": True, "shadow": True, "rule": RULE,
               "attribution": "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC"}
        (dst / f"{name}.json").write_text(json.dumps(dto, ensure_ascii=False, indent=1) + "\n")
        print(name, dto["level"], dto["worstHorizonHours"], dto["reason"], dto["hNowCm"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
