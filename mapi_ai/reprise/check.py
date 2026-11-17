"""
Confere a reprise (Etapa 4.1, item 4.1.4, último critério): o point_flood do sistema bate com o balde a 10 min rodado
FORA do sistema sobre a mesma série (mesma lâmina, mesmo instante de PROVAVEL). Lê a saída do ReprisaEventosIT
(projeto-mapi-api/target/reprise/<evento>/) e escreve check.json e a linha do tempo de cada evento.
Uso: python -m mapi_ai.reprise.check [diretório da saída]
"""
import json
import sys
from pathlib import Path

import numpy as np

from mapi_ai import bucket10 as B
from mapi_ai.rain import config as C

TOL_MM = 1e-6


def run(base: Path):
    out = {}
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        calls = json.loads((d / "balde_chamadas.json").read_text())
        sysj = json.loads((d / "sistema.json").read_text())
        worst, n, prov_sys, prov_off = 0.0, 0, {}, {}
        for c in calls:
            t = c["terrain"]
            terr = B.Terreno(t["areaM2"], np.asarray(t["zCurveM"], float), np.asarray(t["vCurveM3"], float), t["streetRelM"])
            o = B.avaliar(np.asarray(c["rain"], float), terr, B.Params10(tr_anos=c["tr"], te_min=c["te"]), c["q50"], c["q80"])
            r = c["result"]
            worst = max(worst, abs(o["h_now_mm"] - r["hNowMm"]))
            if o["projection"]["q50"] and r["q50"]:
                worst = max(worst, float(np.max(np.abs(np.asarray(o["projection"]["q50"]["h_mm"]) - np.asarray(r["q50"]["hMm"])))))
            n += 1
            prov = o["h_now_mm"] >= 150 or (o["projection"]["q50"] and o["projection"]["q50"]["minutes_to_hcrit"] is not None
                                             and o["projection"]["q50"]["minutes_to_hcrit"] <= 60)
            if prov and c["slug"] not in prov_off:
                prov_off[c["slug"]] = c["at"]
        for slug, rows in sysj["timeline"].items():
            first = next((r["evaluated_at"] for r in rows if r["point_flood"] == "PROVAVEL"), None)
            if first:
                prov_sys[slug] = first
        same_prov = {s: (prov_sys.get(s), prov_off.get(s)) for s in set(prov_sys) | set(prov_off)}
        ok = worst <= TOL_MM and all(_same(a, b) for a, b in same_prov.values())
        out[d.name] = {"chamadas_ao_balde": n, "maior_diferenca_de_lamina_mm": worst, "provavel_sistema_x_fora": same_prov,
                       "bate": bool(ok), "primeiro_ligado": sysj["first_on"], "transicoes_sse": sysj["sse_transitions"],
                       "transicoes_banco": sysj["db_transitions"], "ciclos": sysj["cycles"], "segundos": sysj["seconds"]}
    (base / "check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    import pandas as pd
    return abs((pd.Timestamp(a.replace(" ", "T")) - pd.Timestamp(b)).total_seconds()) < 1


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else C.ROOT.parent / "projeto-mapi-api" / "target" / "reprise"
    print(json.dumps(run(base), indent=1, ensure_ascii=False))
