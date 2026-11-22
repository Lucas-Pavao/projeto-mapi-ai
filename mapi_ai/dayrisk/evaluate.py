"""
Avaliação do "dia de risco" (bloco C), na ordem do pré-registro (mapi_ai/dayrisk/config.py, commit 3d9e5af).

Uso: python -m mapi_ai.dayrisk.evaluate   (lê data/rain/dataset.pkl.gz e data/day_risk/om_previous_runs.csv.gz;
grava reports/day_risk/evaluation.json)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from mapi_ai.dayrisk import config as C
from mapi_ai.rain import config as RC
from mapi_ai.rain import evaluate as RE

STEP = pd.Timedelta(minutes=10)


# ---------------------------------------------------------------------------------------------------------------- dados
def point_series(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Por ponto: p_10 e p_60 numa grade regular de 10 min (lacuna = NaN)."""
    out = {}
    for slug, g in rows.groupby("slug"):
        g = g.set_index("t").sort_index()[["p_10", "p_60"]]
        full = pd.date_range(g.index.min(), g.index.max(), freq=STEP, tz="UTC")
        out[slug] = g.reindex(full)
    return out


def emissions(start: pd.Timestamp, end: pd.Timestamp, local_hours: tuple[int, ...]) -> pd.DatetimeIndex:
    """Instantes de emissão (UTC) nas horas locais dadas, em [start, end)."""
    days = pd.date_range(start.tz_convert(C.TZ).normalize() - pd.Timedelta(days=1), end.tz_convert(C.TZ).normalize(),
                         freq="D")
    ts = [d + pd.Timedelta(hours=h) for d in days for h in local_hours]
    idx = pd.DatetimeIndex(ts).tz_convert("UTC")
    return idx[(idx >= start) & (idx < end)]


def _window_flag(p60: pd.Series, t: pd.DatetimeIndex, first: pd.Timedelta, last: pd.Timedelta) -> np.ndarray:
    """1 se alguma hora cheia válida com fim em [t+first, t+last] passa do limiar; 0 se >= 90% válidas e nenhuma
    passa; NaN senão."""
    n = int((last - first) / STEP) + 1
    v = p60.to_numpy()
    idx = p60.index
    pos = idx.get_indexer(t + first)
    out = np.full(len(t), np.nan)
    for k, p in enumerate(pos):
        if p < 0 or p + n > len(v):
            continue
        w = v[p:p + n]
        ok = ~np.isnan(w)
        if (w[ok] >= C.FORTE_MM).any():
            out[k] = 1.0
        elif ok.mean() >= C.LABEL_MIN_VALID:
            out[k] = 0.0
    return out


def label(p60: pd.Series, t: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Evento FORTE em (t, t+H]: horas cheias que terminam em [t+60 min, t+H]."""
    return _window_flag(p60, t, pd.Timedelta(minutes=60), pd.Timedelta(hours=h))


def persist(p60: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    """Choveu forte em (t-24 h, t]: horas cheias que terminam em [t-23 h, t]."""
    return _window_flag(p60, t, pd.Timedelta(hours=-23), pd.Timedelta(0))


def obs24(p10: pd.Series, t: pd.DatetimeIndex) -> np.ndarray:
    """Chuva medida em (t-24 h, t]: soma dos 144 passos; ausente se faltar qualquer passo."""
    s = p10.rolling(144, min_periods=144).sum()
    return s.reindex(t).to_numpy()


def om_window(om: pd.DataFrame, slug: str, model: str, t: pd.DatetimeIndex, h: int, col: str = "precip_mm") -> np.ndarray:
    """Maior valor previsto nas horas que terminam em (t, t+H], antecedência N(H). Ausente se faltar qualquer hora.
    Trava de vazamento: a hora h só vale se h - 24N + atraso <= t."""
    n = C.lead_days(h)
    assert h <= 24 * n - C.PUBLICATION_DELAY_H, "vazamento: a rodada desta antecedência não estava publicada em t"
    s = om[(om.slug == slug) & (om.model == model) & (om.lead_days == n)].set_index("hour_end_utc")[col].sort_index()
    s = s[~s.index.duplicated()]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h", tz="UTC")
    s = s.reindex(full)
    # janela das H horas que terminam em t+1h..t+H: rolling sobre o fim t+H
    m = s.rolling(h, min_periods=h).max()
    return m.reindex(t + pd.Timedelta(hours=h)).to_numpy()


def build(rows: pd.DataFrame, om: pd.DataFrame, h: int, local_hours: tuple[int, ...]) -> pd.DataFrame:
    series = point_series(rows)
    first = RC.DATA_START
    out = []
    for slug, g in series.items():
        t = emissions(first, C.END - pd.Timedelta(hours=h) + pd.Timedelta(seconds=1), local_hours)
        df = pd.DataFrame({"slug": slug, "t": t})
        df["y"] = label(g.p_60, t, h)
        df["persist"] = persist(g.p_60, t)
        df["obs24"] = obs24(g.p_10, t)
        for m in C.OM_MODELS:
            df[f"om_{m}"] = om_window(om, slug, m, t, h)
        df["cape"] = om_window(om, slug, C.OM_CAPE_MODEL, t, h, col="cape")
        out.append(df)
    d = pd.concat(out, ignore_index=True)
    loc = d.t.dt.tz_convert(C.TZ)
    d["month"] = loc.dt.month
    d["lhour"] = loc.dt.hour
    d["day"] = loc.dt.date.astype(str)
    end = d.t + pd.Timedelta(hours=h)
    d["part"] = None
    d.loc[(d.t >= C.START) & (end <= C.TRAIN_END), "part"] = "train"
    d.loc[(d.t >= C.TRAIN_END) & (end <= C.VALID_END), "part"] = "valid"
    d.loc[(d.t >= C.VALID_END) & (end <= C.END), "part"] = "test"
    oms = [f"om_{m}" for m in C.OM_MODELS]
    d["om_media"] = d[oms].mean(axis=1, skipna=False)
    d["om_maximo"] = d[oms].max(axis=1, skipna=False)
    d["complete"] = d[["y", "persist", "obs24", "cape", *oms]].notna().all(axis=1)
    return d


# ------------------------------------------------------------------------------------------------------------- modelos
def climatology(d: pd.DataFrame, part_start: pd.Timestamp, h: int, target: pd.DataFrame, by_hour: bool) -> np.ndarray:
    fit = d[(d.t + pd.Timedelta(hours=h) <= part_start) & d.y.notna()]
    pbar = fit.y.mean()
    keys = ["month", "lhour"] if by_hour else ["month"]
    g = fit.groupby(keys).y.agg(["sum", "count"])
    rate = (g["sum"] + C.CLIMATOLOGY_PRIOR_N * pbar) / (g["count"] + C.CLIMATOLOGY_PRIOR_N)
    idx = pd.MultiIndex.from_frame(target[keys]) if by_hour else target["month"]
    return rate.reindex(idx).fillna(pbar).to_numpy()


def persistence(fit: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    pbar = fit.y.mean()
    g = fit.groupby("persist").y.agg(["sum", "count"])
    rate = (g["sum"] + C.PERSIST_PRIOR_N * pbar) / (g["count"] + C.PERSIST_PRIOR_N)
    return target.persist.map(rate).fillna(pbar).to_numpy()


def _feats(df: pd.DataFrame, name: str) -> np.ndarray:
    if name.startswith("om_") and name != "om_media" and name != "om_maximo":
        return np.log1p(df[[name]].to_numpy())
    if name in ("om_media", "om_maximo"):
        return np.log1p(df[[name]].to_numpy())
    if name == "logistica_completa":
        return np.column_stack([np.log1p(df.om_media), df.cape / 1000.0, np.log1p(df.obs24), df.persist])
    raise KeyError(name)


def logistic(fit: pd.DataFrame, target: pd.DataFrame, name: str) -> np.ndarray:
    sc = StandardScaler().fit(_feats(fit, name))
    lr = LogisticRegression(C=C.LOGIT_C, max_iter=1000).fit(sc.transform(_feats(fit, name)), fit.y.astype(int))
    return lr.predict_proba(sc.transform(_feats(target, name)))[:, 1]


def predict_all(d: pd.DataFrame, part: str, h: int, by_hour: bool) -> pd.DataFrame:
    target = d[(d.part == part) & d.complete].copy()
    fit_parts = ["train"] if part == "valid" else ["train", "valid"]
    fit = d[d.part.isin(fit_parts) & d.complete]
    start = C.TRAIN_END if part == "valid" else C.VALID_END
    target["climatologia"] = climatology(d, start, h, target, by_hour)
    target["persistencia"] = persistence(fit, target)
    for m in C.MODELS[2:]:
        target[m] = logistic(fit, target, m.replace("om_", "om_") if m != "logistica_completa" else m)
    return target


# ------------------------------------------------------------------------------------------------------------ métricas
def _boot_counts(days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u, inv = np.unique(days, return_inverse=True)
    rng = np.random.default_rng(C.SEED)
    w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=C.N_BOOT)
    return w, inv


def bss_ci(y: np.ndarray, p: np.ndarray, pc: np.ndarray, days: np.ndarray) -> dict:
    w, inv = _boot_counts(days)
    sm = np.bincount(inv, (p - y) ** 2)
    sc = np.bincount(inv, (pc - y) ** 2)
    bs = 1 - (w @ sm) / np.maximum(w @ sc, 1e-12)
    point = 1 - sm.sum() / sc.sum()
    return {"bss": round(float(point), 4), "ic95": [round(float(np.quantile(bs, 0.025)), 4),
                                                     round(float(np.quantile(bs, 0.975)), 4)]}


def ece(y: np.ndarray, p: np.ndarray) -> dict:
    q = pd.qcut(pd.Series(p).rank(method="first"), C.RELIABILITY_BINS, labels=False)
    curve = []
    e = 0.0
    for b in range(C.RELIABILITY_BINS):
        m = (q == b).to_numpy()
        if not m.any():
            continue
        mp, my = float(p[m].mean()), float(y[m].mean())
        curve.append({"p_media": round(mp, 4), "freq_obs": round(my, 4), "n": int(m.sum())})
        e += m.mean() * abs(mp - my)
    return {"ece": round(e, 4), "curva": curve}


def best_csi(y: np.ndarray, p: np.ndarray) -> float:
    best, thr = -1.0, 1.0
    for c in np.unique(np.round(p, 4)):
        a = p >= c
        hit, miss, fa = (a & (y == 1)).sum(), (~a & (y == 1)).sum(), (a & (y == 0)).sum()
        csi = hit / max(1, hit + miss + fa)
        if csi > best:
            best, thr = csi, float(c)
    return thr


def warn_counts(t: pd.DataFrame, y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    a = p >= thr
    hit, miss, fa = int((a & (y == 1)).sum()), int((~a & (y == 1)).sum()), int((a & (y == 0)).sum())
    months = t.groupby("slug").t.agg(lambda s: (s.max() - s.min()).days / 30.44).sum()
    return {"limiar": round(thr, 4), "eventos": hit + miss, "avisados": hit, "fracao_avisada": round(hit / max(1, hit + miss), 3),
            "falsos": fa, "ponto_meses": round(float(months), 1), "falsos_por_ponto_mes": round(fa / max(months, 1e-9), 3)}


def score(t: pd.DataFrame, thresholds: dict | None) -> dict:
    y = t.y.to_numpy()
    pc = t.climatologia.to_numpy()
    days = t.day.to_numpy()
    out = {"n": int(len(t)), "positivos": int(y.sum()), "dias": int(t.day.nunique()),
           "dias_com_evento": int(t[t.y == 1].day.nunique()), "modelos": {}}
    for m in C.MODELS:
        p = t[m].to_numpy()
        r = {"brier": round(float(np.mean((p - y) ** 2)), 5), **bss_ci(y, p, pc, days),
             "pr_auc": round(float(average_precision_score(y, p)), 4) if y.sum() else None, **ece(y, p)}
        if thresholds is not None:
            r["aviso"] = warn_counts(t, y, p, thresholds[m])
        out["modelos"][m] = r
    return out


def select(val: dict) -> dict:
    ms = val["modelos"]
    best = max((m for m in C.MODELS if m != "climatologia"), key=lambda m: ms[m]["bss"])
    lo = ms[best]["ic95"][0]
    chosen = next(m for m in C.MODELS if ms[m]["bss"] >= lo)
    return {"melhor_validacao": best, "bss_melhor": ms[best]["bss"], "ic95_melhor": ms[best]["ic95"],
            "escolhido": chosen}


def run() -> dict:
    rows = pd.read_pickle(RC.DATASET)[["slug", "t", "p_10", "p_60"]]
    om = pd.read_csv(C.OM_TABLE, parse_dates=["hour_end_utc"])
    out = {"versao": C.VERSION, "pre_registro": "mapi_ai/dayrisk/config.py @ 3d9e5af", "regra_adocao": C.ADOPTION_RULE,
           "atribuicao": [C.ATTRIBUTION, C.OM_ATTRIBUTION], "esquemas": {}}
    schemes = {"06h": (C.EMISSION_MORNING_LOCAL_HOUR,), "a_cada_3h": C.EMISSION_EVERY_3H_LOCAL}
    for sname, hours in schemes.items():
        out["esquemas"][sname] = {}
        for h in C.HORIZONS:
            d = build(rows, om, h, hours)
            excl = {p: {"instantes": int((d.part == p).sum()), "sem_rotulo": int(((d.part == p) & d.y.isna()).sum()),
                        "sem_entrada": int(((d.part == p) & d.y.notna() & ~d.complete).sum())} for p in ("train", "valid", "test")}
            by_hour = sname != "06h"
            v = predict_all(d, "valid", h, by_hour)
            thr = {m: best_csi(v.y.to_numpy(), v[m].to_numpy()) for m in C.MODELS}
            val = score(v, thr)
            te = predict_all(d, "test", h, by_hour)
            tst = score(te, thr)
            out["esquemas"][sname][f"{h}h"] = {"lead_days": C.lead_days(h), "contagem": excl, "validacao": val,
                                                "teste": tst}
            print(sname, h, "valid BSS", {m: val["modelos"][m]["bss"] for m in C.MODELS}, flush=True)
    sel = select(out["esquemas"][C.SELECTION_EMISSION][f"{C.SELECTION_HORIZON}h"]["validacao"])
    test_sel = out["esquemas"][C.SELECTION_EMISSION][f"{C.SELECTION_HORIZON}h"]["teste"]["modelos"][sel["escolhido"]]
    sel["teste_bss"] = test_sel["bss"]
    sel["teste_ic95"] = test_sel["ic95"]
    sel["adotado"] = bool(sel["escolhido"] != "climatologia" and test_sel["ic95"][0] > 0)
    out["decisao"] = sel
    # eventos FORTE no teste (definição do rain_v1: cruzamento sem outro nas 6 h anteriores)
    test_rows = rows[(rows.t >= C.VALID_END) & (rows.t < C.END)]
    ev = RE.events(test_rows, C.FORTE_MM)
    out["eventos_FORTE_teste"] = {"eventos": int(len(ev)), "dias": int(ev.c.dt.tz_convert(C.TZ).dt.date.nunique()) if len(ev) else 0,
                                  "por_ponto": ev.groupby("slug").size().to_dict() if len(ev) else {},
                                  "abaixo_do_minimo": bool(len(ev) < C.MIN_TEST_EVENTS)}
    out["inmet_avisos_sobreposicao"] = ("nenhuma: os avisos do INMET são arquivados desde 2026-11-16 e a série medida "
                                        "termina em 2026-10-31")
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "evaluation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return out


if __name__ == "__main__":
    r = run()
    json.dump(r["decisao"], sys.stdout, indent=1, ensure_ascii=False)
    print()
    print(r["eventos_FORTE_teste"])
