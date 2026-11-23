"""
Onda 16, Fase 2.4: com a chuva MEDIDA de 5 anos, quantas vezes o refluxo de maré e a limpeza mudam a lâmina do balde a
10 min em CIn e Mascarenhas, e em quanto. Sem dia de alagamento: nada aqui é comparado com ocorrência, nada é ajustado.

Entradas (as mesmas do registro de 5 anos, mapi_ai/bucket10_5anos.py):
- data/bucket10/serie10min_<ponto>.csv.gz (GET /api/alerta/serie-10min.csv, regra do alerta) e data/bucket10/terreno.json;
- maré: extremos da tábua DHN do Porto do Recife 2021-2026 (projeto-mapi-api/src/main/resources/tide/dhn-pe02-AAAA.csv,
  hora UTC-3, altura acima do NR), interpolados por meia onda de cosseno a cada 10 min (o método do TideInterpolation da
  API). O valor de cada passo é o do fim do passo.
- refluxo (ADR 0030): soleira e diâmetro de AlertaProperties.Refluxo; Δ = 0,821 m; cenário central (viés 0) e de viés
  (MDT - RN = -0,369 m).
- limpeza: regime permanente da emenda 1 da 156 (o que o ciclo usa até a carga completa da 156): π_E = D_E/(1/λ + D_E),
  com λ e D_E de LimpezaProperties; galeria entupida com f_E = 0 (padrão, [sem fonte]) e f_E = 0,5 (ponta da faixa).

Série com lacuna: o balde roda por trecho contínuo, vazio depois de cada lacuna (ausente nunca vira zero).
Evento = episódio com lâmina >= 15 cm, separado do seguinte por pelo menos 1 passo abaixo de 15 cm.
Uso: python -m mapi_ai.fatores_5anos
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B
from mapi_ai import entupimento as ENT
from mapi_ai.rain import config as RC

DIR = RC.ROOT / "data" / "bucket10"
TIDE_DIR = RC.ROOT.parent / "projeto-mapi-api" / "src" / "main" / "resources" / "tide"
OUT = RC.ROOT / "reports" / "fatores"
HCRIT = 150.0
DELTA = 0.821
BIAS = -0.369
# AlertaProperties.Refluxo (soleira no MDT, diâmetro) e LimpezaProperties (λ, limpezas, D_E), emenda 1 da 156
SAIDA = {"MASCARENHAS_IMBIRIBEIRA": (0.638, 0.6)}
LIMPEZA = {"MASCARENHAS_IMBIRIBEIRA": (0.03331, 130, 44.2), "CIN_UFPE": (None, 0, 185.5)}
PONTOS = ("CIN_UFPE", "MASCARENHAS_IMBIRIBEIRA")


def tide_10min(index: pd.DatetimeIndex) -> tuple[np.ndarray, list[dict]]:
    frames, files = [], []
    for y in range(2021, 2027):
        p = TIDE_DIR / f"dhn-pe02-{y}.csv"
        d = pd.read_csv(p, comment="#", header=None, names=["data", "hora", "h"])
        d["t"] = pd.to_datetime(d.data + " " + d.hora).dt.tz_localize("Etc/GMT+3").dt.tz_convert("UTC")
        frames.append(d[["t", "h"]])
        files.append({"arquivo": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    ext = pd.concat(frames).sort_values("t").drop_duplicates("t")
    te = ext.t.astype("int64").to_numpy()
    he = ext.h.to_numpy(dtype=float)
    tv = index.astype("int64").to_numpy()
    k = np.searchsorted(te, tv, side="right") - 1
    ok = (k >= 0) & (k < len(te) - 1)
    k = np.clip(k, 0, len(te) - 2)
    gap = (te[k + 1] - te[k]) / 3.6e12
    frac = (tv - te[k]) / (te[k + 1] - te[k])
    h = he[k] + (he[k + 1] - he[k]) * (1 - np.cos(np.pi * frac)) / 2
    h[~ok | (gap > 12)] = np.nan
    return h, files


def run_segments(P: np.ndarray, gap: np.ndarray, ter: B.Terreno, p: B.Params10, f_passo=None, f: float = 1.0) -> np.ndarray:
    h = np.zeros(len(P))
    seg = np.cumsum(gap)
    for sid in np.unique(seg[~gap]):
        idx = np.flatnonzero((seg == sid) & ~gap)
        fp = None if f_passo is None else f * f_passo[idx]
        if fp is None and f != 1.0:
            fp = np.full(len(idx), f)
        h[idx] = B.simular(P[idx], ter, p, f_passo=fp)[0]
    return h


def events(h: np.ndarray, ts: pd.Series) -> dict:
    over = h >= HCRIT
    starts = np.flatnonzero(over & ~np.r_[False, over[:-1]])
    return {"eventos_ge_15cm": int(len(starts)), "lamina_max_cm": round(float(h.max()) / 10, 1),
            "passos_com_lamina_gt_0": int((h > 0).sum()), "datas": [str(ts.iloc[i]) for i in starts[:15]]}


def diff(a: np.ndarray, b: np.ndarray, years: float) -> dict:
    d = (a - b) / 10.0
    ch = np.abs(d) > 0.1
    starts = np.flatnonzero(ch & ~np.r_[False, ch[:-1]])
    out = {"passos_com_mudanca_gt_1mm": int(ch.sum()), "episodios_com_mudanca": int(len(starts)),
           "episodios_por_ano": round(len(starts) / years, 2)}
    if ch.any():
        out.update({"mudanca_cm_mediana": round(float(np.median(d[ch])), 2), "mudanca_cm_p95": round(float(np.quantile(d[ch], 0.95)), 2),
                    "mudanca_cm_max": round(float(d[ch].max()), 2)})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    terr = {t["slug"]: t for t in json.loads((DIR / "terreno.json").read_text())}
    p = B.Params10()
    res = {"regra": __doc__.strip(), "parametros": {"delta_m": DELTA, "vies_cenario_m": BIAS, "hcrit_cm": HCRIT / 10,
                                                   "tr_anos": p.tr_anos, "te_min": p.te_min, "C": p.C}, "pontos": {}}
    for slug in PONTOS:
        s = pd.read_csv(DIR / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"])
        t = terr[slug]
        ter = B.Terreno(float(t["area_m2"]), np.asarray(t["z"], float), np.asarray(t["v"], float), float(t["rua_rel_m"]))
        P = s.mm.fillna(0.0).to_numpy()
        gap = s.mm.isna().to_numpy()
        years = (s.slot_end_utc.max() - s.slot_end_utc.min()).total_seconds() / (365.25 * 86400)
        base = run_segments(P, gap, ter, p)
        info = {"anos": round(years, 2), "passos": int(len(s)), "passos_sem_estacao_pct": round(100 * gap.mean(), 2),
                "terreno": {"area_m2": t["area_m2"], "rua_rel_m": round(t["rua_rel_m"], 3)}, "sem_fatores": events(base, s.slot_end_utc)}
        # refluxo
        if slug in SAIDA:
            sill_mdt, D = SAIDA[slug]
            mare, files = tide_10min(pd.DatetimeIndex(s.slot_end_utc))
            res["mare_fonte"] = files
            sem_mare = np.isnan(mare)
            gap_r = gap | sem_mare
            ref = {}
            for nome, bias in (("central", 0.0), ("vies_-0,369", BIAS)):
                sill = sill_mdt - bias + DELTA
                f = np.ones(len(P))
                f[~sem_mare] = B.fator_refluxo(mare[~sem_mare], sill, D)
                h = run_segments(P, gap_r, ter, p, f_passo=f)
                b2 = run_segments(P, gap_r, ter, p)
                wet = (P > 0) & ~gap_r
                ref[nome] = {"soleira_nr_m": round(sill, 3), "diametro_m": D,
                             "tempo_com_saida_reduzida_pct": round(100 * float((f[~gap_r] < 1).mean()), 1),
                             "tempo_com_saida_fechada_pct": round(100 * float((f[~gap_r] == 0).mean()), 1),
                             "passos_com_chuva_e_saida_reduzida_pct": round(100 * float((f[wet] < 1).mean()), 1),
                             "com_refluxo": events(h, s.slot_end_utc), "mudanca_vs_sem_refluxo": diff(h, b2, years)}
            info["refluxo"] = {"status": "OK", "passos_sem_mare": int(sem_mare.sum()), "cenarios": ref}
        else:
            info["refluxo"] = {"status": "NAO_SE_APLICA", "motivo": "ponto sem saída de galeria cadastrada no ESIG"}
        # limpeza
        lam, n_limp, de = LIMPEZA[slug]
        pe = ENT.pi_entupida_permanente(lam, de) if lam is not None and n_limp >= ENT.MIN_LIMPEZAS else None
        if pe is None:
            info["limpeza"] = {"status": "SEM_DADO", "motivo": f"{n_limp} limpezas casadas a 150 m (mínimo {ENT.MIN_LIMPEZAS}): sem taxa"}
        else:
            lim = {"regime": "PERMANENTE", "lambda_re_por_dia": lam, "dias_entupida_media": de, "p_entupida": round(pe, 4)}
            for fe in (0.0, 0.5):
                h = run_segments(P, gap, ter, p, f=fe) if fe > 0 else run_segments(P, gap, ter, p, f_passo=np.zeros(len(P)))
                ev = events(h, s.slot_end_utc)
                both = (h >= HCRIT) & ~(base >= HCRIT)
                lim[f"f_E_{str(fe).replace('.', ',')}"] = {
                    "galeria_entupida": ev, "mudanca_vs_livre": diff(h, base, years),
                    "passos_em_que_a_mistura_chega_a_0,5": int(((1 - pe) * (base >= HCRIT) + pe * (h >= HCRIT) >= 0.5).sum()),
                    "passos_so_entupida_alaga": int(both.sum())}
            info["limpeza"] = {"status": "OK", **lim}
        res["pontos"][slug] = info
        print(slug, json.dumps({k: v for k, v in info.items() if k in ("sem_fatores",)}, ensure_ascii=False)[:300], flush=True)
    (OUT / "fatores_5anos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


if __name__ == "__main__":
    main()
