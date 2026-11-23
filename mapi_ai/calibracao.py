"""
Calibração inversa da capacidade (onda 16, Fase 4.3; projeto-mapi-api/docs/PRE-REGISTRO-CALIBRACAO-INVERSA.md, escrito antes).

Um parâmetro por ponto: kappa em [0, 1], Q(kappa) = Q_min^(1-kappa) * Q_max^kappa (faixa de data/capacidade/capacidade_efetiva.json).
Positivos: os 18 dias da Defesa Civil (ponto-dia local, registro a até 2 km). Negativos de controle por regra: dias de 01/09/2021 a
25/06/2024 com 15 a 40 mm medidos no ponto, dia completo, sem registro a 2 km. Balde sobre a chuva MEDIDA, refluxo no modo do
ponto, limpeza fora. Objetivo: acurácia balanceada; IC por bootstrap de dias estratificado. Uso: python -m mapi_ai.calibracao
"""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B
from mapi_ai import fatores_5anos as F5
from mapi_ai.flood import capacidade as CAP
from mapi_ai.rain import config as RC

OUT = RC.ROOT / "reports" / "calibracao"
OCC = RC.ROOT / "data" / "in1165" / "flood_occurrences.csv.gz"
PONTOS_CSV = RC.ROOT.parent / "projeto-mapi-api" / "tools" / "terreno" / "pontos_criticos.csv"
TERRENO = RC.ROOT / "data" / "bucket10" / "terreno.json"
RAIO_KM = 2.0
INICIO, FIM = pd.Timestamp("2021-09-01").date(), pd.Timestamp("2024-06-25").date()
NEG_MM = (15.0, 40.0)
KAPPAS = np.round(np.arange(0, 1.0001, 0.05), 2)
N_BOOT = 1000
HCRIT = 150.0
REFLUXO_NIVEL = {"MASCARENHAS_IMBIRIBEIRA": (0.638, 0.6)}   # Ibura em INFO (ADR 0030 emenda 1): fora da conta
ADOTA = {"min_positivos": 3, "max_largura_ic": 0.5, "min_ganho_vs_meio": 0.05}


def km(a, b, c, d):
    p = math.pi / 180
    x = (d - b) * p * math.cos((a + c) / 2 * p)
    return 6371.0 * math.hypot(x, (c - a) * p)


def bal_acc(pred: np.ndarray, obs: np.ndarray) -> float:
    pos, neg = obs, ~obs
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    return 0.5 * (pred[pos].mean() + (~pred[neg]).mean())


def melhor(ba: np.ndarray) -> int:
    """Índice do maior valor; empate: o kappa mais perto de 0,5."""
    m = np.nanmax(ba)
    cand = np.flatnonzero(np.isclose(ba, m))
    return int(cand[np.argmin(np.abs(KAPPAS[cand] - 0.5))])


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RC.SEED)
    occ = pd.read_csv(OCC)
    occ["day"] = pd.to_datetime(occ.occurred_from, utc=True).dt.tz_convert("America/Recife").dt.date
    pts = {r["slug"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(open(PONTOS_CSV, encoding="utf-8"))}
    terr = {t["slug"]: t for t in json.loads(TERRENO.read_text()) if t["status"] == "OK"}
    res = {"pre_registro": "projeto-mapi-api/docs/PRE-REGISTRO-CALIBRACAO-INVERSA.md", "kappas": KAPPAS.tolist(),
           "regra_adocao": ADOTA, "pontos": {}}
    adotados = {}
    for slug in ("AV_RECIFE_IBURA", "CIN_UFPE", "MASCARENHAS_IMBIRIBEIRA", "JABOATAO_CENTRO"):
        fx = CAP.faixa(slug)
        if slug not in terr or fx is None:
            res["pontos"][slug] = {"status": "fora (sem terreno ou sem faixa)"}
            continue
        lat, lon = pts[slug]
        pos_days = {d for d, a, b in zip(occ.day, occ.geocoded_lat, occ.geocoded_lon) if km(lat, lon, a, b) <= RAIO_KM}
        s = pd.read_csv(RC.ROOT / "data" / "bucket10" / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"])
        s = s[(s.slot_end_utc > pd.Timestamp(INICIO, tz="America/Recife").tz_convert("UTC"))
              & (s.slot_end_utc <= (pd.Timestamp(FIM, tz="America/Recife") + pd.Timedelta(days=1)).tz_convert("UTC"))]
        s = s.reset_index(drop=True)
        local_day = (s.slot_end_utc - pd.Timedelta(minutes=10)).dt.tz_convert("America/Recife").dt.date
        P = s.mm.fillna(0.0).to_numpy()
        gap = s.mm.isna().to_numpy()
        daily = pd.DataFrame({"day": local_day, "mm": s.mm, "gap": gap}).groupby("day").agg(mm=("mm", "sum"), gap=("gap", "any"),
                                                                                             n=("gap", "size"))
        daily = daily[(~daily.gap) & (daily.n == 144)]
        obs_pos = daily.index.isin(list(pos_days))
        neg = (~obs_pos) & daily.mm.between(*NEG_MM)
        keep = obs_pos | neg
        days = daily.index[keep]
        obs = obs_pos[keep]
        f_passo = None
        if slug in REFLUXO_NIVEL:
            sill, D = REFLUXO_NIVEL[slug]
            mare, _ = F5.tide_10min(pd.DatetimeIndex(s.slot_end_utc))
            gap = gap | np.isnan(mare)
            f_passo = np.ones(len(P))
            f_passo[~np.isnan(mare)] = B.fator_refluxo(mare[~np.isnan(mare)], sill + F5.DELTA, D)
        ter = B.Terreno(float(terr[slug]["area_m2"]), np.asarray(terr[slug]["z"], float), np.asarray(terr[slug]["v"], float),
                        float(terr[slug]["rua_rel_m"]))
        day_idx = pd.Index(local_day)
        pred = np.zeros((len(days), len(KAPPAS)), dtype=bool)
        hmax_day = {}
        for j, k in enumerate(KAPPAS):
            p = B.Params10(q_cap_m3s=fx.q(float(k)))
            h = F5.run_segments(P, gap, ter, p, f_passo=f_passo)
            mx = pd.Series(h).groupby(day_idx).max()
            pred[:, j] = mx.reindex(days).fillna(0.0).to_numpy() >= HCRIT
            hmax_day[float(k)] = (mx.reindex(days[obs]).fillna(0.0) / 10.0).round(1).tolist()
        ba = np.array([bal_acc(pred[:, j], obs) for j in range(len(KAPPAS))])
        info = {"faixa_m3s": [fx.q_min_m3s, fx.q_max_m3s], "base": fx.base, "positivos": int(obs.sum()),
                "negativos_controle": int((~obs).sum()), "dias_positivos": [str(d) for d in days[obs]],
                "acuracia_balanceada": {f"{k:.2f}": None if np.isnan(v) else round(float(v), 3) for k, v in zip(KAPPAS, ba)},
                "sensibilidade_e_especificidade": {f"{k:.2f}": [round(float(pred[obs, j].mean()), 3) if obs.any() else None,
                                                                round(float((~pred[~obs, j]).mean()), 3) if (~obs).any() else None]
                                                   for j, k in enumerate(KAPPAS) if k in (0.0, 0.5, 1.0)},
                "lamina_max_cm_nos_dias_positivos": {k: v for k, v in hmax_day.items() if k in (0.0, 0.5, 1.0)}}
        if np.all(np.isnan(ba)):
            info["status"] = "sem positivo ou sem negativo: não calibrável"
            res["pontos"][slug] = info
            continue
        jhat = melhor(ba)
        boots = []
        ip, ineg = np.flatnonzero(obs), np.flatnonzero(~obs)
        for _ in range(N_BOOT):
            sel = np.r_[rng.choice(ip, len(ip)), rng.choice(ineg, len(ineg))]
            b = np.array([bal_acc(pred[sel, j], obs[sel]) for j in range(len(KAPPAS))])
            boots.append(KAPPAS[melhor(b)])
        ic = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
        ganho = float(ba[jhat] - ba[list(KAPPAS).index(0.5)])
        crit = {"positivos_ge_3": bool(obs.sum() >= ADOTA["min_positivos"]),
                "largura_ic_le_0,5": bool(ic[1] - ic[0] <= ADOTA["max_largura_ic"]),
                "ganho_vs_kappa_0,5_ge_0,05": bool(ganho >= ADOTA["min_ganho_vs_meio"])}
        info.update({"kappa_hat": float(KAPPAS[jhat]), "ic95_kappa": ic, "acuracia_em_kappa_hat": round(float(ba[jhat]), 3),
                     "ganho_vs_kappa_0,5": round(ganho, 3), "criterios": crit, "adotado": all(crit.values()),
                     "q_calibrado_m3s": round(fx.q(float(KAPPAS[jhat])), 4)})
        if info["adotado"]:
            adotados[slug] = {"kappa": float(KAPPAS[jhat]), "ic95": ic, "q_m3s": info["q_calibrado_m3s"]}
        res["pontos"][slug] = info
        print(slug, {k: info[k] for k in ("positivos", "negativos_controle", "kappa_hat", "ic95_kappa", "acuracia_em_kappa_hat",
                                           "ganho_vs_kappa_0,5", "adotado")}, flush=True)
    (OUT / "calibracao.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (RC.ROOT / "data" / "capacidade" / "calibracao_adotada.json").write_text(json.dumps(
        {"nota": "kappa adotado pela regra do pré-registro da calibração inversa; ponto fora daqui usa a regra da faixa",
         "pontos": adotados}, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    run()
