"""
Item 1.4.6 do prompt do alerta de 20 min: o balde M1a a 10 min sobre os 5 anos, por ponto e TR. REGISTRO, não
calibração: nenhum parâmetro é mexido para acertar frequência.
Entradas: data/bucket10/serie10min_<ponto>.csv.gz (exportada por GET /api/alerta/serie-10min.csv, mesma regra do alerta)
e data/bucket10/terreno.json (point_terrain, terrain_run 2).
Série com lacuna: o balde roda por trecho contínuo, começando vazio depois de cada lacuna (ausente nunca vira zero).
Evento = episódio com lâmina >= 15 cm, separado do seguinte por pelo menos 1 passo abaixo de 15 cm.
Uso: python -m mapi_ai.bucket10_5anos
"""
import gzip
import json

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B
from mapi_ai.rain import config as C

DIR = C.ROOT / "data" / "bucket10"
OUT = C.ROOT / "reports" / "bucket10"
TRS = (2.0, 5.0, 10.0)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    terr = {t["slug"]: t for t in json.loads((DIR / "terreno.json").read_text())}
    out = {"regra": __doc__.strip(), "pontos": {}}
    for slug in ["AV_RECIFE_IBURA", "CIN_UFPE", "AGAMENON_DERBY", "MASCARENHAS_IMBIRIBEIRA", "JABOATAO_CENTRO"]:
        s = pd.read_csv(DIR / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"])
        t = terr.get(slug)
        info = {"passos": int(len(s)), "passos_sem_estacao_pct": round(100 * s.mm.isna().mean(), 2),
                "chuva_total_mm": round(float(s.mm.sum()), 1),
                "maior_chuva_10min_mm": round(float(s.mm.max()), 1),
                "passos_com_bloco_longo_e_chuva_pct": round(100 * float((s.long_block.astype(bool) & (s.mm > 0)).sum() / max(1, (s.mm > 0).sum())), 1)}
        if t is None or t["status"] != "OK" or not t["z"]:
            info["balde"] = "sem terreno (SEM_DADO no alerta)"
            out["pontos"][slug] = info
            continue
        ter = B.Terreno(float(t["area_m2"]), np.asarray(t["z"], float), np.asarray(t["v"], float), float(t["rua_rel_m"]))
        years = (s.slot_end_utc.max() - s.slot_end_utc.min()).total_seconds() / (365.25 * 86400)
        info["anos"] = round(years, 2)
        info["terreno"] = {"area_m2": t["area_m2"], "rua_rel_m": round(t["rua_rel_m"], 3), "volume_topo_m3": t["v"][-1]}
        info["balde"] = {}
        gap = s.mm.isna().to_numpy()
        seg_id = np.cumsum(gap)
        for tr in TRS:
            p = B.Params10(tr_anos=tr)
            h = np.zeros(len(s))
            for sid in np.unique(seg_id[~gap]):
                idx = np.flatnonzero((seg_id == sid) & ~gap)
                h[idx] = B.simular(s.mm.to_numpy()[idx], ter, p)[0]
            over = h >= p.hcrit_mm
            starts = np.flatnonzero(over & ~np.r_[False, over[:-1]])
            info["balde"][f"TR{int(tr)}"] = {"capacidade_mm_h": round(p.i_cap_mm_h, 1), "eventos_ge_15cm": int(len(starts)),
                                            "eventos_por_ano": round(len(starts) / years, 2), "lamina_max_cm": round(float(h.max()) / 10, 1),
                                            "passos_com_lamina_gt_0": int((h > 0).sum()),
                                            "datas": [str(s.slot_end_utc.iloc[i]) for i in starts[:20]]}
        out["pontos"][slug] = info
        print(slug, {k: (v["eventos_ge_15cm"], v["lamina_max_cm"]) for k, v in info["balde"].items()}, flush=True)
    out["referencia_cidade"] = ("3,6 dias por ano com alagamento registrado na cidade inteira (CKAN, Defesa Civil). A comparação "
                                "mistura unidades: cidade inteira contra um ponto.")
    (OUT / "balde10_5anos.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "balde"} | ({"balde": {t: {a: b for a, b in x.items() if a != "datas"} for t, x in v["balde"].items()}} if isinstance(v.get("balde"), dict) else {"balde": v.get("balde")})
                      for k, v in o["pontos"].items()}, indent=1, ensure_ascii=False))
