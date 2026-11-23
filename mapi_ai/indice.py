"""
Índice de risco caixa branca (onda 16, Fase 5; ADR 0031 e projeto-mapi-api/docs/PRE-REGISTRO-INDICE-CAIXA-BRANCA.md,
escritos antes). Só comparação: nada daqui vai para a tela.

Unidade ponto-dia local, 01/09/2021 a 25/06/2024, dia com série de 10 min completa. Positivo = registro da Defesa Civil a até
2 km. Entradas: chuva_6h_mm, max_30min_mm, mare_nr_m. Modelos: regra 6 h >= 25 mm; logística L2 C = 1; EBM com 1 interação.
Validação cruzada agrupada por dia (5 dobras). Uso: python -m mapi_ai.indice
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mapi_ai import calibracao as CB
from mapi_ai import fatores_5anos as F5
from mapi_ai.rain import config as RC

OUT = RC.ROOT / "reports" / "indice"
FEATS = ["chuva_6h_mm", "max_30min_mm", "mare_nr_m"]
SLUGS = ("AV_RECIFE_IBURA", "CIN_UFPE", "AGAMENON_DERBY", "JABOATAO_CENTRO", "MASCARENHAS_IMBIRIBEIRA")
N_BOOT = 1000
REGRA_MM = 25.0


def dataset() -> pd.DataFrame:
    occ = pd.read_csv(CB.OCC)
    occ["day"] = pd.to_datetime(occ.occurred_from, utc=True).dt.tz_convert("America/Recife").dt.date
    pts = {r["slug"]: (float(r["latitude"]), float(r["longitude"])) for r in csv.DictReader(open(CB.PONTOS_CSV, encoding="utf-8"))}
    rows = []
    for slug in SLUGS:
        lat, lon = pts[slug]
        pos = {d for d, a, b in zip(occ.day, occ.geocoded_lat, occ.geocoded_lon) if CB.km(lat, lon, a, b) <= CB.RAIO_KM}
        s = pd.read_csv(RC.ROOT / "data" / "bucket10" / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"])
        ini = pd.Timestamp(CB.INICIO, tz="America/Recife").tz_convert("UTC")
        fim = (pd.Timestamp(CB.FIM, tz="America/Recife") + pd.Timedelta(days=1)).tz_convert("UTC")
        s = s[(s.slot_end_utc > ini - pd.Timedelta(hours=6)) & (s.slot_end_utc <= fim)].reset_index(drop=True)
        mm = s.mm
        r6 = mm.rolling(36, min_periods=36).sum()
        r30 = mm.rolling(3, min_periods=3).sum()
        mare, _ = F5.tide_10min(pd.DatetimeIndex(s.slot_end_utc))
        day = (s.slot_end_utc - pd.Timedelta(minutes=10)).dt.tz_convert("America/Recife").dt.date
        df = pd.DataFrame({"day": day, "mm": mm, "r6": r6, "r30": r30, "mare": mare})
        df = df[(df.day >= CB.INICIO) & (df.day <= CB.FIM)]
        for d, g in df.groupby("day"):
            if len(g) != 144 or g.mm.isna().any() or g.r6.isna().any():
                continue
            i = g.r6.idxmax()
            rows.append({"slug": slug, "day": d, "chuva_6h_mm": float(g.r6.max()), "max_30min_mm": float(g.r30.max()),
                         "mare_nr_m": float(g.mare.loc[i]), "y": int(d in pos)})
    return pd.DataFrame(rows)


def ebm():
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(interactions=1, outer_bags=8, random_state=RC.SEED)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))


def oof(model_fn, X, y, groups) -> np.ndarray:
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = model_fn().fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    return p


def metricas(p, y, days, rng, p_brier=None) -> dict:
    """p: escore para AUC e PR-AUC; p_brier: probabilidade para o Brier (a regra usa 0/1)."""
    pb = p if p_brier is None else p_brier
    out = {"auc": round(float(roc_auc_score(y, p)), 3), "pr_auc": round(float(average_precision_score(y, p)), 3),
           "brier": round(float(brier_score_loss(y, pb)), 4)}
    ud = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in ud}
    boots = {"auc": [], "pr_auc": []}
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
        if y[sel].min() == y[sel].max():
            continue
        boots["auc"].append(roc_auc_score(y[sel], p[sel]))
        boots["pr_auc"].append(average_precision_score(y[sel], p[sel]))
    for k, v in boots.items():
        out[f"{k}_ic95"] = [round(float(np.quantile(v, 0.025)), 3), round(float(np.quantile(v, 0.975)), 3)]
    return out


def operacao(score, y, n_marcados, meses) -> dict:
    order = np.argsort(-score, kind="stable")
    flag = np.zeros(len(y), bool)
    flag[order[:n_marcados]] = True
    tp = int((flag & (y == 1)).sum())
    return {"marcados": int(n_marcados), "pod": round(tp / max(1, int(y.sum())), 3), "acertos": tp,
            "falsos_por_ponto_mes": round(float((flag & (y == 0)).sum()) / meses, 3)}


def confiabilidade(p, y, n=10) -> list:
    bins = np.linspace(0, 1, n + 1)
    k = np.clip(np.digitize(p, bins) - 1, 0, n - 1)
    return [{"faixa": [round(bins[i], 2), round(bins[i + 1], 2)], "n": int((k == i).sum()),
             "prevista": None if (k == i).sum() == 0 else round(float(p[k == i].mean()), 3),
             "observada": None if (k == i).sum() == 0 else round(float(y[k == i].mean()), 3)} for i in range(n)]


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RC.SEED)
    d = dataset()
    X, y, groups = d[FEATS], d.y.to_numpy(), d.day.astype(str).to_numpy()
    meses = d.groupby("slug").day.nunique().sum() / 30.44
    regra = (d.chuva_6h_mm >= REGRA_MM).to_numpy()
    n_regra = int(regra.sum())
    p_log = oof(logit, X, y, groups)
    p_ebm = oof(ebm, X, y, groups)
    res = {"pre_registro": "projeto-mapi-api/docs/PRE-REGISTRO-INDICE-CAIXA-BRANCA.md", "adr": "0031",
           "training_data": "REAL_LABEL_DEFESA_CIVIL", "on_screen": False, "orientador": "pendente",
           "dados": {"ponto_dias": int(len(d)), "positivos": int(y.sum()), "dias": int(d.day.nunique()),
                     "positivos_por_ponto": d.groupby("slug").y.sum().astype(int).to_dict(), "ponto_meses": round(float(meses), 1)},
           "fora_da_dobra": {}}
    tp = int((regra & (y == 1)).sum())
    res["fora_da_dobra"]["regra_6h_ge_25mm"] = {
        **metricas(d.chuva_6h_mm.to_numpy(), y, groups, rng, p_brier=regra.astype(float)),
        "operacao": {"marcados": n_regra, "pod": round(tp / max(1, int(y.sum())), 3), "acertos": tp,
                     "falsos_por_ponto_mes": round(float((regra & (y == 0)).sum()) / meses, 3)}}
    for nome, p in (("logistica", p_log), ("ebm", p_ebm)):
        res["fora_da_dobra"][nome] = {**metricas(p, y, groups, rng), "operacao": operacao(p, y, n_regra, meses),
                                      "confiabilidade": confiabilidade(p, y)}
    # interpretação: ajuste no conjunto todo
    lg = logit().fit(X, y)
    sc, lr = lg.named_steps["standardscaler"], lg.named_steps["logisticregression"]
    beta = lr.coef_[0] / sc.scale_                    # por unidade física
    passo = {"chuva_6h_mm": 10.0, "max_30min_mm": 10.0, "mare_nr_m": 0.5}
    boots = []
    ud = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ud}
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[g] for g in rng.choice(ud, len(ud))])
        if y[sel].min() == y[sel].max():
            continue
        m = logit().fit(X.iloc[sel], y[sel])
        boots.append(m.named_steps["logisticregression"].coef_[0] / m.named_steps["standardscaler"].scale_)
    boots = np.array(boots)
    res["logistica_razao_de_chances"] = {
        f: {"por": passo[f], "razao": round(float(np.exp(beta[i] * passo[f])), 3),
            "ic95": [round(float(np.exp(np.quantile(boots[:, i] * passo[f], 0.025))), 3),
                     round(float(np.exp(np.quantile(boots[:, i] * passo[f], 0.975))), 3)],
            "texto": f"cada {passo[f]:g} {'mm' if 'mm' in f else 'm'} a mais de {f} multiplica a chance por "
                     f"{np.exp(beta[i] * passo[f]):.2f}".replace(".", ",")}
        for i, f in enumerate(FEATS)}
    e = ebm().fit(X, y)
    g = e.explain_global()
    curvas = {}
    for i, nome in enumerate(g.data()["names"]):
        di = g.data(i)
        sc = di.get("scores")
        um = sc is not None and np.ndim(sc) == 1
        curvas[nome] = {"importancia": round(float(g.data()["scores"][i]), 4),
                        "x": [round(float(v), 3) for v in di.get("names")] if um else None,
                        "score": [round(float(v), 4) for v in sc] if um else None}
    res["ebm_curvas"] = curvas
    _figuras(curvas, res)
    (OUT / "indice.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    return res


def _figuras(curvas, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, f in zip(axs, FEATS):
        c = curvas.get(f)
        if c and c["x"] and c["score"]:
            xs, ys = c["x"], c["score"]
            n = min(len(xs) - 1, len(ys))
            ax.step(xs[:n], ys[:n], where="post")
        ax.set_title(f)
        ax.set_ylabel("contribuição (logit)")
    fig.tight_layout()
    fig.savefig(OUT / "curvas_ebm.png", dpi=110)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for nome in ("logistica", "ebm"):
        cf = [b for b in res["fora_da_dobra"][nome]["confiabilidade"] if b["n"] > 0]
        ax.plot([b["prevista"] for b in cf], [b["observada"] for b in cf], marker="o", label=nome)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("probabilidade prevista (fora da dobra)")
    ax.set_ylabel("fração observada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "confiabilidade.png", dpi=110)


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k not in ("ebm_curvas",)}, indent=1, ensure_ascii=False, default=str)[:5000])
