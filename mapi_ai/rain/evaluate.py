"""
Avalia os cinco modelos. Escolha na validação (critério em config.SELECTION); o teste é lido uma vez, no fim.
Uso: python -m mapi_ai.rain.evaluate
"""
import json

import numpy as np
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import models as M
from mapi_ai.rain.train import PRED

RNG = np.random.default_rng(C.SEED)
SLOT_MIN = 10


def pinball_rows(y, q, a):
    d = y - q
    return np.maximum(a * d, (a - 1) * d)


def row_loss(rows, pred, h):
    """Pinball médio dos três quantis, por linha (NaN onde o alvo falta)."""
    y = rows[f"y_{h}h"].to_numpy()
    return np.mean([pinball_rows(y, pred[f"{n}_{h}h"].to_numpy(), a) for n, a in zip(M.QN, C.QUANTILES)], axis=0)


def day_blocks(rows):
    return rows.t.dt.floor("D").astype("int64").to_numpy()


def boot_diff(loss_a, loss_b, blocks, n=C.N_BOOT):
    """IC 95% da diferença de médias (a - b), bootstrap pareado por bloco de dia (os 5 pontos juntos)."""
    ok = ~np.isnan(loss_a) & ~np.isnan(loss_b)
    d = pd.DataFrame({"d": (loss_a - loss_b)[ok], "b": blocks[ok]}).groupby("b").d.agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = RNG.integers(0, len(s), size=(n, len(s)))
    v = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return float(s.sum() / c.sum()), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def metrics(rows, pred, h, mask=None):
    m = np.ones(len(rows), bool) if mask is None else mask
    y = rows[f"y_{h}h"].to_numpy()
    ok = m & ~np.isnan(y)
    out = {"n": int(ok.sum()), "pinball": float(np.nanmean(row_loss(rows, pred, h)[ok]))}
    for n, a in zip(M.QN, C.QUANTILES):
        q = pred[f"{n}_{h}h"].to_numpy()
        out[f"pinball_{n}"] = float(pinball_rows(y[ok], q[ok], a).mean())
        out[f"cobertura_{n}"] = float((y[ok] <= q[ok]).mean())
    for k in C.THRESHOLDS:
        e = rows[f"ev_{k}_{h}h"].to_numpy(dtype=float)
        p = pred[f"p{k}_{h}h"].to_numpy()
        okk = m & ~np.isnan(e)
        out[f"brier_{k}"] = float(np.mean((p[okk] - e[okk]) ** 2))
        out[f"prevalencia_{k}"] = float(e[okk].mean())
    return out


def best_csi_threshold(e, p):
    ok = ~np.isnan(e)
    e, p = e[ok].astype(bool), p[ok]
    best = (0.0, 1.0)
    for thr in np.unique(np.quantile(p[p > 0], np.linspace(0.5, 0.9999, 400))) if (p > 0).any() else []:
        a = p >= thr
        hit, miss, fa = (a & e).sum(), (~a & e).sum(), (a & ~e).sum()
        csi = hit / max(1, hit + miss + fa)
        if csi > best[0]:
            best = (float(csi), float(thr))
    return best


def prf(e, p, thr):
    ok = ~np.isnan(e)
    e, a = e[ok].astype(bool), p[ok] >= thr
    hit, miss, fa = int((a & e).sum()), int((~a & e).sum()), int((a & ~e).sum())
    return {"precisao": hit / max(1, hit + fa), "recall": hit / max(1, hit + miss), "csi": hit / max(1, hit + miss + fa),
            "acertos": hit, "perdidos": miss, "falsos": fa}


def events(rows, thr_mm):
    """
    Eventos FORTE por ponto. Cruzamento c: primeiro instante em que a chuva do ponto nos últimos 60 min chega ao limiar,
    sem outro cruzamento nas 6 h anteriores. Início o: começo da chuva (10 min com 0,4 mm ou mais depois de 2 h sem isso)
    mais próximo antes de c.
    """
    out = []
    for slug, g in rows.groupby("slug"):
        g = g.sort_values("t").reset_index(drop=True)
        hot = (g.p_60 >= thr_mm).to_numpy()
        wet = (g.p_10.fillna(0) >= 0.4)
        onset = (wet & ~(wet.shift(1, fill_value=False).rolling(12, min_periods=1).max().astype(bool))).to_numpy()
        last = -10**9
        for i in np.flatnonzero(hot):
            if i - last >= 36:
                on = np.flatnonzero(onset[max(0, i - 144):i + 1])
                out.append({"slug": slug, "i": int(i), "c": g.t[i], "o": g.t[max(0, i - 144) + on[-1]] if len(on) else pd.NaT})
            last = i
    return pd.DataFrame(out)


def lead_stats(rows, pred, h, thr_p, ev):
    """Para cada evento: primeiro aviso em [c - h, c). Antecedência até o cruzamento e até o início da chuva (min)."""
    n = h * 6
    res, false_ep, alert_slots = [], 0, 0
    for slug, g in rows.assign(a=(pred[f"pFORTE_{h}h"] >= thr_p).to_numpy()).groupby("slug"):
        g = g.sort_values("t").reset_index(drop=True)
        a = g.a.to_numpy()
        alert_slots += int(a.sum())
        cs = ev[ev.slug == slug]
        cross = np.zeros(len(g), bool)
        cross[cs.i.to_numpy()] = True
        for e in cs.itertuples():
            w = np.flatnonzero(a[max(0, e.i - n):e.i])
            if len(w) == 0:
                res.append({"detectado": False, "lead_cruz": 0.0, "lead_inicio": np.nan, "seco": False})
                continue
            first = max(0, e.i - n) + w[0]
            t1 = g.t[first]
            res.append({"detectado": True, "lead_cruz": (e.i - first) * SLOT_MIN,
                        "lead_inicio": (e.o - t1).total_seconds() / 60 if pd.notna(e.o) else np.nan,
                        "seco": bool(g.p_60[first] < C.RAINING_MM) if pd.notna(g.p_60[first]) else False})
        # episódios de aviso (avisos a menos de 1 h um do outro) sem cruzamento até h depois do fim
        idx = np.flatnonzero(a)
        if len(idx):
            starts = idx[np.r_[True, np.diff(idx) > 6]]
            ends = idx[np.r_[np.diff(idx) > 6, True]]
            for s, e2 in zip(starts, ends):
                if not cross[s:min(len(g), e2 + n + 1)].any():
                    false_ep += 1
    r = pd.DataFrame(res)
    if r.empty:
        return {"eventos": 0}
    years = (rows.t.max() - rows.t.min()).total_seconds() / (365.25 * 86400)
    med = lambda v: float(np.median(v)) if len(v) else float("nan")
    lc = r.lead_cruz.to_numpy()
    bs = np.array([np.median(RNG.choice(lc, len(lc))) for _ in range(C.N_BOOT)])       # bootstrap por evento
    det = r[r.detectado]
    return {"eventos": int(len(r)), "detectados": int(r.detectado.sum()), "detectados_pct": float(100 * r.detectado.mean()),
            "antecedencia_mediana_min": med(lc), "antecedencia_ic95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "antecedencia_p10_min": float(np.quantile(lc, 0.10)), "antecedencia_p90_min": float(np.quantile(lc, 0.90)),
            "antecedencia_mediana_dos_detectados_min": med(det.lead_cruz.to_numpy()),
            "avisados_ainda_sem_chuva_pct": float(100 * r.seco.mean()),
            "antecedencia_ate_inicio_da_chuva_mediana_min": med(det.lead_inicio.dropna().to_numpy()),
            "avisados_antes_do_inicio_da_chuva_pct": float(100 * (r.lead_inicio > 0).mean()),
            "episodios_falsos": int(false_ep), "falsos_por_evento": float(false_ep / max(1, len(r))),
            "horas_de_aviso_por_ponto_ano": float(alert_slots / 6 / rows.slug.nunique() / years)}


def run():
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    blob = pd.read_pickle(PRED)
    rows, pred = blob["rows"].reset_index(drop=True), {k: v.reset_index(drop=True) for k, v in blob["pred"].items()}
    V, T = (rows.part == "valid").to_numpy(), (rows.part == "test").to_numpy()
    out = {"attribution": C.ATTRIBUTION, "selection_criterion": C.SELECTION, "alert_rule": C.ALERT_RULE,
           "thresholds_mm_h": C.THRESHOLDS, "quantiles": C.QUANTILES, "horizons": {}}
    # ---------- validação: escolha
    rv, blocks = rows[V], day_blocks(rows[V])
    for h in C.HORIZONS:
        loss = {m: row_loss(rv, pred[m][V], h) for m in C.MODELS}
        mean = {m: float(np.nanmean(v)) for m, v in loss.items()}
        best = min(mean, key=mean.get)
        table, chosen = [], None
        for m in C.MODELS:
            d, lo, hi = (0.0, 0.0, 0.0) if m == best else boot_diff(loss[m], loss[best], blocks)
            tied = lo <= 0.0
            if tied and chosen is None:
                chosen = m
            table.append({"modelo": m, "pinball_validacao": mean[m], "diff_para_o_melhor": d, "ic95": [lo, hi], "empata_com_o_melhor": bool(tied)})
        thr = {m: dict(zip(("csi_validacao", "limiar"), best_csi_threshold(rv[f"ev_FORTE_{h}h"].to_numpy(dtype=float), pred[m][V][f"pFORTE_{h}h"].to_numpy())))
               for m in C.MODELS}
        out["horizons"][f"{h}h"] = {"validacao": table, "melhor": best, "escolhido": chosen, "limiar_aviso_FORTE": thr}
    (C.REPORTS_DIR / "selection.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # ---------- teste: uma vez
    rt, tblocks = rows[T], day_blocks(rows[T])
    ev_t = events(rt, C.THRESHOLDS["FORTE"])
    out["eventos_FORTE_no_teste"] = int(len(ev_t))
    rain_t, dry_t = rt.raining.to_numpy(), ~rt.raining.to_numpy()
    flat = []
    for h in C.HORIZONS:
        H = out["horizons"][f"{h}h"]
        chosen = H["escolhido"]
        loss = {m: row_loss(rt, pred[m][T], h) for m in C.MODELS}
        H["teste"] = {}
        for m in C.MODELS:
            pm = pred[m][T]
            r = {"todas": metrics(rt, pm, h), "ja_chove": metrics(rt, pm, h, rain_t), "ainda_nao_chove": metrics(rt, pm, h, dry_t)}
            d, lo, hi = (0.0, 0.0, 0.0) if m == chosen else boot_diff(loss[m], loss[chosen], tblocks)
            r["diff_para_o_escolhido"] = {"diff": d, "ic95": [lo, hi], "significativa": bool(lo > 0 or hi < 0)}
            thr = H["limiar_aviso_FORTE"][m]["limiar"]
            e = rt[f"ev_FORTE_{h}h"].to_numpy(dtype=float)
            p = pm[f"pFORTE_{h}h"].to_numpy()
            r["FORTE"] = {"limiar": thr, "todas": prf(e, p, thr), "ja_chove": prf(e[rain_t], p[rain_t], thr),
                          "ainda_nao_chove": prf(e[dry_t], p[dry_t], thr), "eventos": lead_stats(rt, pm, h, thr, ev_t)}
            H["teste"][m] = r
            flat.append({"horizonte_h": h, "modelo": m, "escolhido": m == chosen, **{f"{k}": v for k, v in r["todas"].items()},
                         "pinball_ja_chove": r["ja_chove"]["pinball"], "pinball_ainda_nao_chove": r["ainda_nao_chove"]["pinball"],
                         "diff_escolhido": d, "ic_lo": lo, "ic_hi": hi, **{f"FORTE_{k}": v for k, v in r["FORTE"]["todas"].items()},
                         **{f"ev_{k}": v for k, v in r["FORTE"]["eventos"].items() if not isinstance(v, list)}})
    (C.REPORTS_DIR / "evaluation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pd.DataFrame(flat).to_csv(C.REPORTS_DIR / "metrics.csv", index=False)
    return out


if __name__ == "__main__":
    o = run()
    t = pd.read_csv(C.REPORTS_DIR / "metrics.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    for h in C.HORIZONS:
        H = o["horizons"][f"{h}h"]
        print(h, "melhor", H["melhor"], "escolhido", H["escolhido"], [(r["modelo"], round(r["pinball_validacao"], 4), [round(x, 4) for x in r["ic95"]]) for r in H["validacao"]])
    print(t[["horizonte_h", "modelo", "pinball", "pinball_ja_chove", "pinball_ainda_nao_chove", "cobertura_q50", "cobertura_q80", "cobertura_q95", "ic_lo", "ic_hi"]].round(4).to_string())
    print(t[["horizonte_h", "modelo", "FORTE_precisao", "FORTE_recall", "ev_eventos", "ev_detectados_pct", "ev_antecedencia_mediana_min", "ev_antecedencia_p10_min", "ev_antecedencia_mediana_dos_detectados_min", "ev_avisados_ainda_sem_chuva_pct", "ev_avisados_antes_do_inicio_da_chuva_pct", "ev_falsos_por_evento", "ev_horas_de_aviso_por_ponto_ano"]].round(3).to_string())
