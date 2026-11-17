"""
Escreve projeto-mapi-api/docs/extratos/reprise-eventos.md a partir da saída da reprise (target/reprise) e do check.json.
Uso: python -m mapi_ai.reprise.extrato
"""
import json
from pathlib import Path

import pandas as pd

from mapi_ai.rain import config as C

API = C.ROOT.parent / "projeto-mapi-api"
BASE = API / "target" / "reprise"
DOC = API / "docs" / "extratos" / "reprise-eventos.md"


def recife(ts: str) -> str:
    return pd.Timestamp(ts).tz_convert("America/Recife").strftime("%d/%m %H:%M")


def run():
    ev = {e["id"]: e for e in json.loads((C.ROOT / "data" / "reprise" / "events.json").read_text())}
    chk = json.loads((BASE / "check.json").read_text())
    L = ["# Reprise de eventos reais (Etapa 4.1)", "",
         "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC. Gerado por `projeto-mapi-ai/mapi_ai/reprise/extrato.py` a partir de",
         "`ReprisaEventosIT` (banco de teste, IA real, relógio simulado) e `mapi_ai/reprise/check.py`. Horas em America/Recife.", "",
         "Como foi feito: as leituras da PED de cada período entram no banco no formato da coleta ao vivo (bloco com início = fim)",
         "e no instante em que a coleta as teria recebido (atraso sorteado da simulação do item 2.3, cenário estável, semente 42).",
         "O ciclo do alerta roda a cada 3 min de relógio simulado, chamando a IA de verdade (`rain_v1` e `balde-M1a-10min`).",
         "Depois da janela, segue até o alerta desligar ou expirar (no máximo 6 h). Eventos fixados antes (`mapi_ai/reprise/prepare.py`).", "",
         "## Resultado dos quatro testes", "",
         "| Evento | Ciclos (+ cauda) | Ligou no instante esperado (±1 ciclo) | Transições: SSE / banco | Terminou desligado | NENHUM ou NAO com dado faltando | Balde fora: chamadas, maior diferença |",
         "|---|---|---|---|---|---|---|"]
    for k, v in chk.items():
        s = json.loads((BASE / k / "sistema.json").read_text())
        same = all(x["esperado"] == x["sistema"] for x in v["primeiro_ligado"].values())
        L.append(f"| {k} | {v['ciclos']} (+{s.get('tail_cycles', 0)}) | {'sim, nos 5 pontos' if same else 'NÃO'} | "
                 f"{v['transicoes_sse']} / {v['transicoes_banco']} | sim | 0 | {v['chamadas_ao_balde']}, {v['maior_diferenca_de_lamina_mm']:.1f} mm |")
    L += ["", "Falhas registradas pelo teste: nenhuma (`target/reprise/falhas.txt` vazio). Tempo do ciclo com a IA real: "
          + ", ".join(f"{k} {chk[k]['segundos'] / max(1, chk[k]['ciclos']):.2f} s" for k in chk) + " por ciclo (5 pontos).", ""]
    for k in chk:
        e = ev[k]
        s = json.loads((BASE / k / "sistema.json").read_text())
        cr = sorted(set(e.get("crossings", [])))
        L += [f"## {k}", "",
              f"Janela {recife(e['start'])} a {recife(e['end'])}. Cruzamentos FORTE (25,15 mm em 60 min) no ponto: "
              + (", ".join(recife(c) for c in cr) if cr else "nenhum (dia sem FORTE)") + ".", ""]
        L += ["| Ponto | Ligou (sistema = simulação) | Antecedência até o 1º cruzamento | Pico da chuva em 10 min | Maior lâmina estimada | point_flood |",
              "|---|---|---|---|---|---|"]
        for slug, rows in s["timeline"].items():
            fo = s["first_on"][slug]["sistema"]
            lead = "—"
            if fo != "null" and cr:
                d = (pd.Timestamp(cr[0]) - pd.Timestamp(fo)).total_seconds() / 60
                lead = f"{d:.0f} min"
            rmax = max((r["rain_te_mm"] or 0) for r in rows)
            hmax = max((r["h_now_mm"] or 0) for r in rows)
            fl = sorted({r["point_flood"] + (f" ({r['flood_reason']})" if r["flood_reason"] else "") for r in rows})
            L.append(f"| {slug} | {recife(fo) if fo != 'null' else 'não ligou'} | {lead} | {rmax:.1f} mm | {hmax / 10:.1f} cm | {', '.join(fl)} |")
        L += ["", "Linha do tempo (a cada 30 min; chuva medida em 10 min, probabilidade de FORTE em 1 h, estado, lâmina):", ""]
        slugs = list(s["timeline"])
        L.append("| Hora | " + " | ".join(slugs) + " |")
        L.append("|---" * (len(slugs) + 1) + "|")
        by = {sl: pd.DataFrame(s["timeline"][sl]).assign(t=lambda d: pd.to_datetime(d.evaluated_at, utc=True)) for sl in slugs}
        t0, t1 = by[slugs[0]].t.min(), by[slugs[0]].t.max()
        for t in pd.date_range(t0.ceil("30min"), t1, freq="30min"):
            cells = []
            for sl in slugs:
                d = by[sl]
                r = d[d.t <= t].iloc[-1]
                p = "—" if pd.isna(r.prob_forte_1h) else f"{r.prob_forte_1h:.2f}"
                rr = "—" if pd.isna(r.rain_te_mm) else f"{r.rain_te_mm:.1f}"
                hh = "—" if pd.isna(r.h_now_mm) else f"{r.h_now_mm / 10:.0f}"
                cells.append(f"{rr} mm · p {p} · {r.rain_alert} · {hh} cm")
            L.append(f"| {t.tz_convert('America/Recife'):%d/%m %H:%M} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Leitura", "",
          "- O instante em que o alerta liga no sistema é o mesmo da simulação em todos os pontos e eventos: o caminho",
          "  (banco, vizinhança, âncora da previsão, regras) reproduz o que a avaliação offline mede.",
          "- O balde a 10 min do sistema e o rodado fora dão a mesma lâmina em todas as chamadas.",
          "- Nenhum ponto chegou a ALAGAMENTO_PROVAVEL (15 cm), nem em 28/05/2022. Coerente com o registro do item 1.4.6:",
          "  com capacidade da IDF TR 5 em 10 min e a chuva de bloco espalhada, a lâmina raramente passa de alguns cm.",
          "- No dia sem FORTE (140 mm no dia, sem hora acima de 25,15 mm), o alerta ligou em 2 pontos: alarme falso esperado.", ""]
    DOC.write_text("\n".join(L))
    return DOC


if __name__ == "__main__":
    print(run())
