"""
Contrafactuais do verdadeiro positivo (Etapa 5): quanto MENOS chuva levaria o risco a LOW?

Leitura correta: chuva e maré não são controláveis. Um contrafactual aqui é um LIMIAR ("com X mm a menos em 3 h, o
risco seria LOW"), não uma ação recomendada.

Dois métodos:
1. DiCE (método "random"), variando só as 5 features de chuva, cada uma entre 0 e o valor observado (só menos chuva);
   local, hora, mês e maré fixos. A DiCE sorteia cada acumulado de forma independente e pode gerar combinações
   impossíveis (chuva de 1 h maior que a de 3 h). Por isso pedimos muitos candidatos e ficamos com os 3 fisicamente
   coerentes (1 h ≤ 3 h ≤ 6 h ≤ 12 h ≤ 24 h) que mais se aproximam do caso.
2. Escala: toda a chuva multiplicada pelo mesmo fator k (mantém a coerência por construção); busca-se o maior k que
   leva o risco a LOW. É o limiar mais fácil de comunicar.

"Risco LOW" = probabilidade abaixo do limiar MEDIUM do metadata (escolhido na validação, Etapa 4).
"""
import dice_ml
import numpy as np
import pandas as pd

from mapi_ai import in1165_config as C
from mapi_ai.xai.common import load, pick_cases, proba


def coherent(row) -> bool:
    v = [row[f] for f in C.RAIN_FEATURES]
    return all(np.isfinite(v)) and all(a <= b + 1e-9 for a, b in zip(v, v[1:]))


def scale_threshold(model, x: pd.Series, low_thr: float, outcome: str = "o risco seria LOW") -> dict:
    """Maior fator k ∈ [0, 1] tal que p(k · chuva) < low_thr (bisseção; p cresce com a chuva neste modelo)."""
    def p_at(k):
        z = x.copy()
        for f in C.RAIN_FEATURES:
            z[f] = x[f] * k
        return float(proba(model, z.to_frame().T)[0])
    if p_at(0.0) >= low_thr:
        return {"reachable": False, "p_without_rain": p_at(0.0)}
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_at(mid) < low_thr else (lo, mid)
    k = lo
    return {"reachable": True, "k": round(k, 4), "p_at_k": p_at(k),
            "rain_3h_mm_observed": float(x["rainfall_3h_accumulated"]),
            "rain_3h_mm_low": float(x["rainfall_3h_accumulated"] * k),
            "mm_less_in_3h": float(x["rainfall_3h_accumulated"] * (1 - k)),
            "statement": (f"Com {x['rainfall_3h_accumulated'] * (1 - k):.1f} mm a menos em 3 h (de "
                          f"{x['rainfall_3h_accumulated']:.1f} para {x['rainfall_3h_accumulated'] * k:.1f} mm, e a chuva "
                          f"das outras janelas reduzida na mesma proporção), {outcome}.")}


def repair(cf: pd.DataFrame) -> pd.DataFrame:
    """
    Torna o candidato coerente: cada janela longa passa a ser o máximo entre ela e as mais curtas
    (24 h >= 12 h >= 6 h >= 3 h >= 1 h). Como o caso observado é coerente e cada janela do candidato está abaixo da
    observada, o reparo nunca passa do valor observado: continua sendo "menos chuva".
    """
    cf = cf.copy()
    for a, b in zip(C.RAIN_FEATURES, C.RAIN_FEATURES[1:]):
        cf[b] = np.maximum(cf[b], cf[a])
    return cf


def dice_counterfactuals(model, X_train, y_train, x: pd.Series, low_thr: float, n_candidates=300) -> list:
    train = X_train.fillna(X_train.median()).copy()
    train[C.LABEL] = y_train
    data = dice_ml.Data(dataframe=train, continuous_features=C.FEATURES, outcome_name=C.LABEL)
    m = dice_ml.Model(model=model, backend="sklearn")
    exp = dice_ml.Dice(data, m, method="random")
    q = x.to_frame().T.astype(float)
    ranges = {f: [0.0, float(x[f])] for f in C.RAIN_FEATURES}
    # desired_class=0 com stopping_threshold: P(classe 0) >= 1 − low_thr, isto é, P(alaga) < limiar MEDIUM.
    res = exp.generate_counterfactuals(q, total_CFs=n_candidates, desired_class=0, features_to_vary=C.RAIN_FEATURES,
                                       permitted_range=ranges, stopping_threshold=1 - low_thr, random_seed=C.SEED)
    cfs = res.cf_examples_list[0].final_cfs_df
    if cfs is None or len(cfs) == 0:
        return []
    raw = cfs[C.FEATURES].astype(float)
    n_raw_below = int((proba(model, raw) < low_thr).sum())
    n_raw_coherent = int(raw.apply(coherent, axis=1).sum())
    cfs = repair(raw)
    cfs["p"] = proba(model, cfs[C.FEATURES])
    ok = cfs[(cfs["p"] < low_thr) & cfs.apply(coherent, axis=1)].drop_duplicates(subset=C.RAIN_FEATURES).copy()
    # Mais próximos do caso: menor redução total de chuva (soma em mm, só das janelas de chuva).
    ok["mm_reduced"] = sum((x[f] - ok[f]) for f in C.RAIN_FEATURES)
    ok = ok.sort_values("mm_reduced").head(3)
    return [{"features": {f: round(float(r[f]), 3) for f in C.RAIN_FEATURES}, "probability": round(float(r["p"]), 5),
             "mm_less_in_3h": round(float(x["rainfall_3h_accumulated"] - r["rainfall_3h_accumulated"]), 2),
             "statement": f"Com {x['rainfall_3h_accumulated'] - r['rainfall_3h_accumulated']:.1f} mm a menos em 3 h "
                          f"(e as demais janelas como na linha), o risco seria LOW."}
            for _, r in ok.iterrows()] + [{"note": f"{len(raw)} candidatos da DiCE: {n_raw_below} abaixo do limiar e "
                                                 f"{n_raw_coherent} coerentes como saíram; depois do reparo, {len(ok)} "
                                                 f"coerentes e abaixo do limiar (até 3 usados)"}]


def run(ctx=None) -> dict:
    ctx = ctx or load()
    model, X_te, y_te, meta = ctx["model"], ctx["X_test"], ctx["y_test"], ctx["meta"]
    cases = ctx.get("cases") or pick_cases(model, X_te, y_te, meta["thresholds"]["high"])
    i = cases["TP"]
    x = X_te.iloc[i]
    low = meta["thresholds"]["medium"]
    return {"case": "TP", "index_in_test": i, "probability": float(proba(model, x.to_frame().T)[0]),
            "low_threshold": low, "dice": dice_counterfactuals(model, ctx["X_train"], ctx["y_train"], x, low),
            "scale": scale_threshold(model, x, low),
            # Mesmo cálculo para sair de HIGH (p abaixo do limiar HIGH): "o risco cairia para MEDIUM ou LOW".
            "scale_below_high": scale_threshold(model, x, meta["thresholds"]["high"], "o risco sairia de HIGH (MEDIUM)"),
            "reading": "limiar, não ação: chuva e maré não são controláveis"}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, ensure_ascii=False, default=float))
