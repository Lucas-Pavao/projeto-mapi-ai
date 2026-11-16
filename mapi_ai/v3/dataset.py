"""
Dataset do modelo v3 (etapas B1, B2 e B3).

O que muda em relação ao v2 (mapi_ai/synthetic.py), e por quê:

1. SÉRIE CONTÍNUA. O rótulo é gerado para TODAS as horas de cada ponto, não só para a amostra. Disso saem o alvo com
   antecedência (B1), a avaliação hora a hora e por evento (B7) e os alertas por ano (B6).
2. ALVO COM ANTECEDÊNCIA (B1). O v2 era nowcast: o rótulo da hora t dependia da chuva até t ("está alagando"). Aqui as
   entradas são as da hora t e o rótulo é "alaga em algum momento de t a t+H", com H = 0 (o nowcast, só para comparar),
   3 e 6 h. Nenhuma entrada usa chuva depois de t (tests/test_v3_dataset.py testa isso).
3. LACUNAS (B2). O dataset do v2 tinha 0% de ausentes; a série real tem lacunas. As lacunas são aplicadas DEPOIS de
   gerar o rótulo: o rótulo vem da chuva verdadeira e o modelo vê a chuva com buracos, como em produção. O padrão é o
   medido na série real (projeto-mapi-api, etapa A2): mesma fração por ponto, mesma distribuição de duração, parte
   simultânea entre os pontos, e independente da chuva (a medida não achou dependência).
4. JANELAS DISJUNTAS (B3). 0-1 h, 1-3 h, 3-6 h, 6-12 h, 12-24 h no lugar dos acumulados aninhados. O contrato não muda:
   as janelas saem dos acumulados por subtração (windows_from_accumulated), e a ausência se propaga.

A fórmula geradora, as escalas e o intercepto são os do v2 (reports/synthetic_ground_truth.json), sem alteração, para
que o H = 0 seja comparável. Os sorteios são outros (a série inteira é sorteada, não a amostra).
"""
import hashlib
import json

import numpy as np
import pandas as pd

from mapi_ai import in1165_config as C2
from mapi_ai.sources import load_hourly
from mapi_ai.synthetic import _event_hours, linear_score
from mapi_ai.v3 import config as C

V2_TRUTH = C2.REPORTS_DIR / "synthetic_ground_truth.json"


def windows_from_accumulated(df: pd.DataFrame) -> pd.DataFrame:
    """
    Janelas disjuntas a partir dos acumulados do contrato (/v1/predict/flood). Usada no treino E no serviço.

    Ausência se propaga: uma janela depende de dois acumulados e fica ausente se qualquer um deles falta. Diferença
    negativa por arredondamento vira 0.
    """
    cur, a3, a6, a12, a24 = (df[c].astype(float) for c in C.ACCUMULATED)
    out = pd.DataFrame({"rain_0_1h": cur, "rain_1_3h": a3 - cur, "rain_3_6h": a6 - a3, "rain_6_12h": a12 - a6,
                        "rain_12_24h": a24 - a12}, index=df.index)
    return out.clip(lower=0)   # clip mantém NaN


def accumulate(hourly: pd.Series) -> pd.DataFrame:
    """Acumulados de 3/6/12/24 h de UMA série horária contínua. Janela com qualquer hora ausente fica ausente."""
    out = {"current_rainfall": hourly}
    for n in (3, 6, 12, 24):
        out[f"rainfall_{n}h_accumulated"] = hourly.rolling(n, min_periods=n).sum()
    return pd.DataFrame(out)


def horizon_label(y0: pd.Series, h: int) -> pd.Series:
    """
    1 se alaga em alguma hora de t a t+h (inclusive), dentro da série de UM ponto. Indefinido (NaN) se alguma hora da
    janela não tem rótulo (fim da série, ou hora sem chuva verdadeira).
    """
    if h == 0:
        return y0.astype(float)
    rev = y0[::-1].rolling(h + 1, min_periods=h + 1).max()
    return rev[::-1]


# --------------------------------------------------------------------------------------------------- lacunas (B2)

def _durations(rng, cls: str, n: int) -> np.ndarray:
    if cls == "1h":
        return np.ones(n, dtype=int)
    lo, hi = (2, 6) if cls == "2_6h" else (7, 24)
    return rng.integers(lo, hi + 1, n)


def gap_mask(index_by_slug: dict, pattern: dict, rng) -> tuple[dict, dict]:
    """
    Máscara de lacunas por ponto (True = hora sem valor), com o padrão medido na série real.

    - Só lacuna curta (até 24 h). A longa (mês inteiro sem estação válida) vira UNKNOWN em produção, não previsão com
      entrada ausente; por isso não é simulada.
    - Número de lacunas por classe de duração = o medido por ponto, na proporção do tamanho da série.
    - Parte das lacunas é simultânea nos pontos (queda da rede): na série real, 93 das 472 horas distintas com lacuna
      curta atingem 4 ou 5 pontos. Essas são sorteadas uma vez e aplicadas a todos.
    - Início uniforme no tempo: a medida não achou dependência da chuva (taxa de 0,35% a 0,41% em todas as faixas).
    """
    slugs = list(index_by_slug)
    n_hours = {s: len(index_by_slug[s]) for s in slugs}
    mask = {s: np.zeros(n_hours[s], dtype=bool) for s in slugs}
    sim = pattern["simultaneidade_horas_lacuna_curta"]
    shared_hours_target = sim["4_pontos"] + sim["5_pontos"]
    ref_hours = next(iter(pattern["por_ponto"].values()))["horas"]
    n0 = n_hours[slugs[0]]

    # lacunas da rede: mesmas horas em todos os pontos (as séries têm a mesma grade)
    shared = np.zeros(n0, dtype=bool)
    target = shared_hours_target * n0 / ref_hours
    while shared.sum() < target:
        cls = rng.choice(["1h", "2_6h", "7_24h"], p=[0.5, 0.35, 0.15])
        d = int(_durations(rng, cls, 1)[0])
        start = int(rng.integers(0, n0 - d))
        shared[start:start + d] = True
    for s in slugs:
        mask[s] |= shared[:n_hours[s]]

    # lacunas do ponto: completa até o total de horas em lacuna curta medido para o ponto
    for s in slugs:
        info = pattern["por_ponto"][s]
        scale = n_hours[s] / info["horas"]
        goal = info["curtas_h"] * scale
        runs = {k: info["runs"][k] for k in ("1h", "2_6h", "7_24h")}
        probs = np.array([runs["1h"], runs["2_6h"], runs["7_24h"]], dtype=float)
        probs /= probs.sum()
        guard = 0
        while mask[s].sum() < goal and guard < 10000:
            guard += 1
            cls = rng.choice(["1h", "2_6h", "7_24h"], p=probs)
            d = int(_durations(rng, cls, 1)[0])
            start = int(rng.integers(0, n_hours[s] - d))
            mask[s][start:start + d] = True
    achieved = {s: {"alvo_fracao_curta": round(pattern["por_ponto"][s]["curtas_h"] / pattern["por_ponto"][s]["horas"], 5),
                    "obtida": round(float(mask[s].mean()), 5), "horas": int(mask[s].sum())} for s in slugs}
    return mask, achieved


# ------------------------------------------------------------------------------------------------- série contínua

def build_continuous(seed: int = C.SEED) -> tuple[pd.DataFrame, dict]:
    """Uma linha por ponto e hora: chuva verdadeira, rótulos por horizonte, chuva observada (com lacunas) e janelas."""
    rng = np.random.default_rng(seed)
    truth_v2 = json.loads(V2_TRUTH.read_text())
    anchors, b0 = truth_v2["anchors"], truth_v2["b0"]
    pattern = json.loads(C.GAPS_JSON.read_text())

    df = load_hourly("cemaden").sort_values(["slug", "hour_end_utc"]).reset_index(drop=True)
    events = pd.read_csv(C2.RAIN_EVENTS)
    events = events[events.source == "CEMADEN_PED"]

    # --- rótulo: fórmula do v2 sobre a chuva VERDADEIRA; indefinido onde a fórmula não tem as entradas
    defined = df[["current_rainfall", "rainfall_3h_accumulated", "rainfall_24h_accumulated", "tide_level"]].notna().all(axis=1)
    p = np.full(len(df), np.nan)
    p[defined.to_numpy()] = 1 / (1 + np.exp(-(b0 + linear_score(df[defined], anchors))))
    df["p_true"] = p
    draw = rng.random(len(df))
    df["flood_h0"] = np.where(defined, (draw < p).astype(float), np.nan)
    for h in C.HORIZONS:
        df[C.label(h)] = df.groupby("slug")["flood_h0"].transform(lambda s, h=h: horizon_label(s, h))

    # --- estrato de cada hora (tabela de eventos de 5 anos), como no v2
    eh = _event_hours(events)
    order = {"FORTE": 0, "MEDIA": 1, "FRACA": 2}
    eh = eh.assign(o=eh.stratum.map(order)).sort_values("o").drop_duplicates(["slug", "hour_end_utc"]).drop(columns="o")
    eh["hour_end_utc"] = pd.to_datetime(eh["hour_end_utc"], utc=True)
    df = df.merge(eh, on=["slug", "hour_end_utc"], how="left")
    df["stratum"] = df["stratum"].fillna("SECO")

    # --- lacunas: aplicadas DEPOIS do rótulo, só na chuva que o modelo vê
    index_by_slug = {s: g.index.to_numpy() for s, g in df.groupby("slug", sort=False)}
    mask, achieved = gap_mask(index_by_slug, pattern, rng)
    df["gap_injected"] = False
    for s, idx in index_by_slug.items():
        df.loc[idx, "gap_injected"] = mask[s]
    true_acc = df[C.ACCUMULATED].copy()
    obs_hourly = df["current_rainfall"].where(~df["gap_injected"])
    obs = pd.concat([accumulate(obs_hourly.loc[idx]) for idx in index_by_slug.values()]).loc[df.index]
    for c in C.ACCUMULATED:
        df[c + "_true"] = true_acc[c]
        df[c] = obs[c]

    # --- janelas disjuntas: com lacunas (o que o modelo vê) e sem lacunas (para separar o efeito das lacunas)
    df[C.WINDOWS] = windows_from_accumulated(df)
    full = windows_from_accumulated(df[[c + "_true" for c in C.ACCUMULATED]].rename(columns=lambda c: c[:-5]))
    for w in C.WINDOWS:
        df[w + "_full"] = full[w]

    local = df["hour_end_utc"].dt.tz_convert("America/Recife")
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * local.dt.hour / 24), np.cos(2 * np.pi * local.dt.hour / 24)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * local.dt.month / 12), np.cos(2 * np.pi * local.dt.month / 12)
    df = df.rename(columns={"hour_end_utc": "timestamp"})
    df["part"] = part_of(df["timestamp"])

    labeled = df[[C.label(h) for h in C.HORIZONS]].notna().all(axis=1) & df["tide_level"].notna()
    df["labeled"] = labeled
    info = {
        "formula": truth_v2["formula"], "b0": b0, "anchors": anchors, "coefficients": truth_v2["coefficients"],
        "formula_source": "reports/synthetic_ground_truth.json (v2, sem alteração)",
        "rows": int(len(df)), "labeled_rows": int(labeled.sum()),
        "prevalence_continuous": {f"h{h}": float(df.loc[labeled, C.label(h)].mean()) for h in C.HORIZONS},
        "positives_in_dry_hours_h0": float((df.loc[labeled & (df.stratum == "SECO"), "flood_h0"].sum())
                                           / max(df.loc[labeled, "flood_h0"].sum(), 1)),
        "gaps": {"pattern_source": "docs/handoff/v3-lacunas.json", "applied_after_label": True, "by_point": achieved,
                 "missing_rate_features": {w: float(df.loc[labeled, w].isna().mean()) for w in C.WINDOWS}},
        "seed": seed,
    }
    return df, info


def part_of(ts: pd.Series) -> pd.Series:
    """train / validation / test pelos cortes do v2, com folga de GUARD_H horas antes de cada corte (purga)."""
    guard = pd.Timedelta(hours=C.GUARD_H)
    part = pd.Series("test", index=ts.index)
    part[ts <= C.VALID_END] = "validation"
    part[ts <= C.TRAIN_END] = "train"
    part[(ts > C.TRAIN_END - guard) & (ts <= C.TRAIN_END)] = "guard"
    part[(ts > C.VALID_END - guard) & (ts <= C.VALID_END)] = "guard"
    return part


def sample(cont: pd.DataFrame, seed: int = C.SEED) -> tuple[pd.DataFrame, dict]:
    """Amostra por estrato (mesmos tamanhos do v2) entre as horas com rótulo definido e fora da folga do split."""
    rng = np.random.default_rng(seed + 1)
    pool_all = cont[cont.labeled & (cont.part != "guard")]
    parts, weights = [], {}
    for stratum, n in C.STRATA.items():
        pool = pool_all[pool_all.stratum == stratum]
        k = min(n, len(pool))
        parts.append(pool.sample(n=k, random_state=int(rng.integers(1 << 31))))
        weights[stratum] = {"pool_hours": int(len(pool)), "sampled": int(k), "inclusion_prob": k / max(len(pool), 1)}
    out = pd.concat(parts).sort_values(["timestamp", "slug"]).reset_index(drop=True)
    info = {"strata": weights, "rows": int(len(out)),
            "split": {name: {"rows": int((out.part == name).sum()),
                             **{f"positives_h{h}": int(out.loc[out.part == name, C.label(h)].sum()) for h in C.HORIZONS},
                             "from": str(out.loc[out.part == name, "timestamp"].min()),
                             "to": str(out.loc[out.part == name, "timestamp"].max())}
                      for name in ("train", "validation", "test")},
            "missing_rate": {w: float(out[w].isna().mean()) for w in C.WINDOWS}}
    return out, info


KEEP = (["timestamp", "slug", "part", "stratum", "event_id", "p_true", "gap_injected", "labeled"]
        + [C.label(h) for h in C.HORIZONS] + C.ACCUMULATED + C.WINDOWS + [w + "_full" for w in C.WINDOWS]
        + ["tide_level"] + C.TIME)


def write() -> dict:
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cont, info = build_continuous()
    smp, sinfo = sample(cont)
    cont[KEEP].to_csv(C.CONTINUOUS, index=False, compression=C.GZIP)
    smp[KEEP].to_csv(C.DATASET, index=False, compression=C.GZIP)
    summary = {"tag": C.SYNTHETIC_TAG, "continuous": info, "sample": sinfo,
               "sha256": {"continuous": hashlib.sha256(C.CONTINUOUS.read_bytes()).hexdigest(),
                          "dataset": hashlib.sha256(C.DATASET.read_bytes()).hexdigest()},
               "horizons": list(C.HORIZONS), "primary_horizon": C.PRIMARY_HORIZON, "attribution": C.ATTRIBUTION}
    (C.REPORTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def load(which: str = "dataset") -> pd.DataFrame:
    df = pd.read_csv(C.DATASET if which == "dataset" else C.CONTINUOUS, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


if __name__ == "__main__":
    s = write()
    print(json.dumps({"prevalence": s["continuous"]["prevalence_continuous"], "split": s["sample"]["split"],
                      "gaps": s["continuous"]["gaps"]["by_point"], "missing": s["sample"]["missing_rate"],
                      "dry_share_h0": s["continuous"]["positives_in_dry_hours_h0"]}, indent=1, ensure_ascii=False))
