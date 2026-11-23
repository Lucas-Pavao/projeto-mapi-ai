"""
Onda 16, Fase 4.4: o alerta único reconstruído nos 5 anos com a capacidade efetiva em faixa (Fase 4.1), o pico de 10 min
previsto (Fase 4.2) e o refluxo no modo de cada ponto (ADR 0030). Relata SÓ a frequência de ALERTA e ATENÇÃO por ponto-mês
(quantos alertas o sistema daria). Nada aqui é comparado com dia de alagamento (isso é a Fase 9).

Mesma máquina de mapi_ai/flood/validate.py (balde vetorizado; estado = 24 h medidas antes de t, começando vazio; janela com
lacuna = SEM_DADO), com: saída por passo = Q·Δt (Q_min e Q_max), fator de refluxo por passo (maré da tábua DHN, cosseno a cada
10 min) e chuva prevista com o pico no passo do meio. Previsões do rain_v1/rain_v1p sobre o período inteiro (no treino,
dentro da amostra: declarado). Uso: python -m mapi_ai.flood.reconstrucao
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B10
from mapi_ai import fatores_5anos as F5
from mapi_ai.flood import capacidade as CAP
from mapi_ai.flood import config as FC
from mapi_ai.rain import config as RC
from mapi_ai.rain import dataset as RD
from mapi_ai.rain import perfil as PF

OUT = RC.ROOT / "reports" / "reconstrucao"
SERIES = RC.ROOT / "data" / "bucket10"
CHOSEN = {1: "gb_rede", 3: "gb_rede_om", 6: "gb_rede_om"}
WINDOW = 144
REFLUXO = {"MASCARENHAS_IMBIRIBEIRA": (0.638, 0.6)}      # modo NIVEL; Ibura em INFO (fora da conta)
VARIANTES = {
    "antes_onda16": dict(cap="idf", perfil=False, refluxo=False),
    "so_capacidade": dict(cap="faixa", perfil=False, refluxo=False),
    "capacidade_e_perfil": dict(cap="faixa", perfil=True, refluxo=False),
    "alerta_unico_v2": dict(cap="faixa", perfil=True, refluxo=True),
}


class VB:
    def __init__(self, ter: B10.Terreno, q_step: float, C: float = 0.86):
        self.t, self.q, self.C = ter, q_step, C

    def step(self, V, P, f):
        v = V + self.C * P * self.t.area_m2 / 1000.0
        return np.maximum(0.0, v - np.minimum(self.q * f, v))

    def sheet_mm(self, V):
        z = np.interp(V, self.t.v_curva_m3, self.t.z_curva_m)
        return np.where(V > 0, 1000.0 * np.maximum(0.0, z - self.t.rua_rel_m), 0.0)


def state(mm, f, vb):
    gaps = pd.Series(np.isnan(mm).astype(int)).rolling(WINDOW, min_periods=WINDOW).sum().to_numpy()
    ok = gaps == 0
    idx = np.flatnonzero(ok)
    V = np.zeros(len(idx))
    for k in range(WINDOW):
        j = idx - WINDOW + 1 + k
        V = vb.step(V, mm[j], f[j])
    Vt = np.full(len(mm), np.nan)
    Vt[idx] = V
    return ok, Vt


def project_max_cm(V0, h0_mm, acc, peak, h, f_full, pos, vb):
    """Lâmina máxima no horizonte; chuva com o pico (ou None = por igual) no passo do meio; f do passo pos+k+1."""
    n = h * 6
    acc = np.maximum(acc, 0.0)
    if peak is None:
        pk, rest, mid = acc / n, acc / n, -1
    else:
        pk = np.minimum(np.maximum(peak, 0.0), acc)
        rest, mid = (acc - pk) / (n - 1), (n - 1) // 2
    V, hmax = V0.copy(), h0_mm.copy()
    L = len(f_full)
    for k in range(n):
        P = pk if k == mid else rest
        fk = f_full[np.minimum(pos + k + 1, L - 1)]
        V = vb.step(V, P, fk)
        hmax = np.maximum(hmax, vb.sheet_mm(V))
    return hmax / 10.0


def episodes(flag: np.ndarray, t: pd.Series) -> int:
    """Episódios: trechos contínuos (passos de 10 min consecutivos) com o nível ligado."""
    on = np.flatnonzero(flag)
    if len(on) == 0:
        return 0
    tt = t.to_numpy()[on]
    return int(1 + np.sum(np.diff(tt) > np.timedelta64(10, "m")))


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = RD.load()
    ds = ds[ds.part != "guard"].reset_index(drop=True)
    terr = {t["slug"]: t for t in json.loads((SERIES / "terreno.json").read_text()) if t["status"] == "OK" and t["z"]}
    ds = ds[ds.slug.isin(terr)].reset_index(drop=True)
    pred = {n: joblib.load(RC.MODELS_DIR / f"{n}.joblib").predict(ds) for n in sorted(set(CHOSEN.values()))}
    pk = PF.predict_peak(joblib.load(PF.MODEL_PATH)["models"], ds)
    res = {"metodo": __doc__.strip(), "regra": FC.RULE_TEXT, "variantes": VARIANTES, "pontos": {}}
    for slug, t in terr.items():
        g = ds.index[ds.slug == slug]
        rows = ds.loc[g, ["t", "part"]].reset_index(drop=True)
        s = pd.read_csv(SERIES / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"]).set_index("slot_end_utc")
        mm_full = s.mm.to_numpy(dtype=float)
        pos = s.index.get_indexer(rows.t)
        okpos = pos >= 0
        ter = B10.Terreno(float(t["area_m2"]), np.asarray(t["z"], float), np.asarray(t["v"], float), float(t["rua_rel_m"]))
        fx = CAP.faixa(slug)
        f_ref = np.ones(len(mm_full))
        if slug in REFLUXO:
            mare, _ = F5.tide_10min(pd.DatetimeIndex(s.index))
            sill, D = REFLUXO[slug]
            bad = np.isnan(mare)
            f_ref[~bad] = B10.fator_refluxo(mare[~bad], sill + F5.DELTA, D)
            mm_full = np.where(bad, np.nan, mm_full)
        meses = (rows.t.max() - rows.t.min()).total_seconds() / (30.44 * 86400)
        out = {"meses": round(meses, 1), "faixa_m3s": None if fx is None else [fx.q_min_m3s, fx.q_max_m3s],
               "base": None if fx is None else fx.base, "variantes": {}}
        for nome, v in VARIANTES.items():
            p0 = B10.Params10()
            if v["cap"] == "idf" or fx is None:
                q_lo = q_hi = p0.q_passo_m3(ter.area_m2)
            else:
                q_lo, q_hi = fx.q_min_m3s * 600.0, fx.q_max_m3s * 600.0
            f_use = f_ref if v["refluxo"] else np.ones(len(mm_full))
            lv = {}
            for tag, q in (("lo", q_lo), ("hi", q_hi)):
                vb = VB(ter, q)
                ok_f, V_f = state(mm_full, f_use, vb)
                ok = np.zeros(len(rows), bool)
                ok[okpos] = ok_f[pos[okpos]]
                V0 = np.zeros(len(rows))
                V0[okpos] = np.nan_to_num(V_f[pos[okpos]])
                h0 = vb.sheet_mm(V0)
                for h in FC.HORIZONS:
                    pr = pred[CHOSEN[h]].loc[g].reset_index(drop=True)
                    sc = "q50" if tag == "hi" else "q95"
                    acc = pr[f"{sc}_{h}h"].to_numpy()
                    peak = pk.loc[g, f"pk_{sc}_{h}h"].to_numpy() if v["perfil"] else None
                    cm = np.full(len(rows), np.nan)
                    i = np.flatnonzero(ok & ~np.isnan(acc))
                    cm[i] = project_max_cm(V0[i], h0[i], acc[i], None if peak is None else peak[i], h, f_use, pos[i], vb)
                    lv[(tag, h)] = cm
            okrow = ~np.isnan(lv[("hi", 1)])
            alerta = np.zeros(len(rows), bool)
            atencao = np.zeros(len(rows), bool)
            for h in FC.HORIZONS:
                alerta |= np.nan_to_num(lv[("hi", h)]) >= FC.HCRIT_CM
                atencao |= np.nan_to_num(lv[("lo", h)]) >= FC.HCRIT_CM
            atencao &= ~alerta
            ligado = alerta | atencao
            out["variantes"][nome] = {
                "passos_calculados": int(okrow.sum()), "passos_sem_dado": int((~okrow).sum()),
                "horas_em_ALERTA": round(float(alerta.sum()) / 6, 1), "horas_em_ATENCAO": round(float(atencao.sum()) / 6, 1),
                "episodios_ALERTA": episodes(alerta, rows.t), "episodios_ALERTA_ou_ATENCAO": episodes(ligado, rows.t),
                "episodios_por_ponto_mes": {"ALERTA": round(episodes(alerta, rows.t) / meses, 2),
                                            "ALERTA_ou_ATENCAO": round(episodes(ligado, rows.t) / meses, 2)},
                "fracao_do_tempo_ligado_pct": round(100 * float(ligado[okrow].mean()), 2) if okrow.any() else None}
            print(slug, nome, out["variantes"][nome]["episodios_por_ponto_mes"], out["variantes"][nome]["fracao_do_tempo_ligado_pct"],
                  flush=True)
        res["pontos"][slug] = out
    res["referencia"] = ("linha de base do prompt: 'chuva medida em 6 h >= 25 mm' dava 3,77 falsos por ponto-mês na validação de "
                         "20/11 (reports/regressao/validacao.json). Aqui não há comparação com alagamento.")
    (OUT / "reconstrucao.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    run()
