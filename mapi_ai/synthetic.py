"""
Plano B do IN1165: rótulo SINTÉTICO, em experimento controlado.

Por que existe: o banco tem 0 ocorrências de alagamento a 500 m dos pontos e nenhuma com hora (Etapa 1,
mapi_ai/build_labels.py). Um classificador supervisionado não se sustenta com isso. Em vez de inventar números, o
experimento usa:

1. FEATURES REAIS. Cada linha é uma hora real de um ponto real: a chuva MEDIDA pelo Cemaden (série do ponto), a
   chuva da REANÁLISE na mesma hora e a maré prevista. Nenhuma feature é sorteada.
2. AMOSTRAGEM CALIBRADA PELOS EVENTOS REAIS. As horas são sorteadas por estrato da tabela de eventos de 5 anos
   (FORTE, MEDIA, FRACA e horas fora de evento). Intensidade, duração e frequência vêm, portanto, da distribuição
   real; o peso amostral de cada estrato fica gravado (a amostra tem mais chuva forte que o clima).
3. RÓTULO POR FÓRMULA CONHECIDA. is_flooded ~ Bernoulli(sigmoide(z)), com z escrito em reports/synthetic_ground_truth.md.
   Os limiares de z são quantis das distribuições reais. Como a fórmula é conhecida, dá para testar se SHAP e LIME
   recuperam as features certas (Etapa 6) — esse é o ponto forte do Plano B.

Limite que precisa estar no relatório: o rótulo é função da chuva MEDIDA. A ablação Cemaden × reanálise mede quanta
informação sobre a chuva medida no local a reanálise preserva; por construção, favorece o Cemaden.
"""
import hashlib
import json

import numpy as np
import pandas as pd

from mapi_ai import in1165_config as C
from mapi_ai.sources import load_hourly

N_ROWS = 5000
# Quantas horas sortear de cada estrato. FORTE é raro (132 eventos em 5 anos): entram todas as horas disponíveis até
# o teto. O resto completa com MEDIA, FRACA e seco, para o modelo ver também o caso comum (não chove).
STRATA = {"FORTE": 900, "MEDIA": 1300, "FRACA": 1300, "SECO": 1500}
# Depois do fim da chuva a rua ainda pode estar alagada: as 3 h seguintes ao evento entram no estrato do evento.
AFTER_EVENT_H = 3
# Taxa de positivos alvo na amostra; b0 é resolvido para atingi-la (não é escolhido à mão).
TARGET_POSITIVE_RATE = 0.08

# Coeficientes escolhidos ANTES de treinar qualquer modelo, olhando só a taxa de positivos por estrato: com
# (1,5; 2,5; 0,8; 3,0) a hora seca tinha 4,3% de positivos (rótulo quase ruído); com (6; 10; 2; 6) fica 0,5% no seco e
# ~28% nas horas de evento FORTE. A comparação está no PROGRESSO.md.
COEF = {"current_rainfall": 6.0, "rainfall_3h_accumulated": 10.0, "rainfall_24h_accumulated": 2.0, "tide_x_rain": 6.0}
# Ordem pela ATRIBUIÇÃO medida no teste (|termo − média| na escala do logit, xai/evaluate_explanations.py), não pelos
# coeficientes: a chuva de 24 h tem coeficiente menor que a de 1 h, mas varia muito mais e pesa mais na prática.
TRUE_FEATURES = ["rainfall_3h_accumulated", "rainfall_24h_accumulated", "current_rainfall", "tide_level"]


def anchors(events: pd.DataFrame, cem: pd.DataFrame, tide: pd.Series) -> dict:
    """Escalas da fórmula, todas medidas nos dados reais (origem ao lado de cada uma)."""
    return {
        "R1_mm": float(events.loc[events.type == "FORTE", "maxHourlyMmH"].median()),   # pico horário típico de evento FORTE
        "R3_mm": float(cem["rainfall_3h_accumulated"].quantile(0.999)),               # 3 h: quantil 99,9% da série do ponto
        "R24_mm": float(cem["rainfall_24h_accumulated"].quantile(0.99)),              # 24 h: quantil 99%
        "T_hi_m": float(tide.quantile(0.90)),                                         # maré alta: quantil 90%
        "T_span_m": float(tide.quantile(0.99) - tide.quantile(0.90)),                 # escala da maré acima do alto
        "wet3_mm": float(cem["rainfall_3h_accumulated"].quantile(0.95)),              # "está chovendo" para a interação
    }


def linear_score(df: pd.DataFrame, a: dict) -> np.ndarray:
    """Parte de z sem o intercepto. Unidades físicas na entrada; cada termo é adimensional."""
    tide_ex = np.clip(df["tide_level"] - a["T_hi_m"], 0, None) / a["T_span_m"]
    wet = (df["rainfall_3h_accumulated"] > a["wet3_mm"]).astype(float)
    return (COEF["current_rainfall"] * df["current_rainfall"] / a["R1_mm"]
            + COEF["rainfall_3h_accumulated"] * df["rainfall_3h_accumulated"] / a["R3_mm"]
            + COEF["rainfall_24h_accumulated"] * df["rainfall_24h_accumulated"] / a["R24_mm"]
            + COEF["tide_x_rain"] * tide_ex * wet).to_numpy()


def solve_intercept(s: np.ndarray, target: float) -> float:
    """b0 tal que a média de sigmoide(b0 + s) = target (bisseção; a média é monótona em b0)."""
    lo, hi = -30.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(mid + s)))).mean() > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _event_hours(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for e in events.itertuples():
        start = pd.Timestamp(e.startUtc).ceil("h")
        end = pd.Timestamp(e.endUtc) + pd.Timedelta(hours=AFTER_EVENT_H)
        for h in pd.date_range(start, end, freq="1h"):
            rows.append((e.floodPointSlug, h, e.type, e.eventId))
    return pd.DataFrame(rows, columns=["slug", "hour_end_utc", "stratum", "event_id"])


def build(seed: int = C.SEED) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    cem = load_hourly("cemaden")
    rea = load_hourly("reanalysis")[["slug", "hour_end_utc"] + C.RAIN_FEATURES]
    events = pd.read_csv(C.RAIN_EVENTS)
    events = events[events.source == "CEMADEN_PED"]

    # Junta medida e reanálise na mesma hora e ponto. Sufixo _rea = reanálise.
    df = cem.merge(rea, on=["slug", "hour_end_utc"], how="left", suffixes=("", "_rea"))
    # Só horas em que a fórmula está definida: chuva medida de 1/3/24 h e maré presentes.
    df = df.dropna(subset=["current_rainfall", "rainfall_3h_accumulated", "rainfall_24h_accumulated", "tide_level"])

    eh = _event_hours(events)
    # Hora que cai em mais de um evento (eventos colados): fica o estrato mais forte.
    order = {"FORTE": 0, "MEDIA": 1, "FRACA": 2}
    eh = eh.assign(o=eh.stratum.map(order)).sort_values("o").drop_duplicates(["slug", "hour_end_utc"]).drop(columns="o")
    df = df.merge(eh, on=["slug", "hour_end_utc"], how="left")
    df["stratum"] = df["stratum"].fillna("SECO")

    parts, weights = [], {}
    for stratum, n in STRATA.items():
        pool = df[df.stratum == stratum]
        k = min(n, len(pool))
        take = pool.sample(n=k, random_state=int(rng.integers(1 << 31)))
        weights[stratum] = {"pool_hours": int(len(pool)), "sampled": int(k), "inclusion_prob": k / max(len(pool), 1)}
        parts.append(take)
    sample = pd.concat(parts)
    # Completa até N_ROWS com horas secas, se algum estrato teve menos horas que o pedido.
    short = N_ROWS - len(sample)
    if short > 0:
        extra = df[(df.stratum == "SECO") & ~df.index.isin(sample.index)].sample(n=short, random_state=seed)
        sample = pd.concat([sample, extra])
        weights["SECO"]["sampled"] += short
    sample = sample.sort_values("hour_end_utc").reset_index(drop=True)

    tide_all = pd.read_csv(C.TIDE_HOURLY)["tide_level"]
    a = anchors(events, cem, tide_all)
    s = linear_score(sample, a)
    b0 = solve_intercept(s, TARGET_POSITIVE_RATE)
    p = 1 / (1 + np.exp(-(b0 + s)))
    sample["p_true"] = p
    sample[C.LABEL] = (rng.random(len(sample)) < p).astype(int)

    # Features de tempo em hora LOCAL (o ciclo do dia em Recife, não em UTC).
    local = sample["hour_end_utc"].dt.tz_convert("America/Recife")
    sample["hour_sin"] = np.sin(2 * np.pi * local.dt.hour / 24)
    sample["hour_cos"] = np.cos(2 * np.pi * local.dt.hour / 24)
    sample["month_sin"] = np.sin(2 * np.pi * local.dt.month / 12)
    sample["month_cos"] = np.cos(2 * np.pi * local.dt.month / 12)
    sample = sample.rename(columns={"hour_end_utc": "timestamp"})

    k = COEF
    truth = {"formula": (f"is_flooded ~ Bernoulli(sigmoid(b0 + {k['current_rainfall']}*P1h/R1 + "
                         f"{k['rainfall_3h_accumulated']}*P3h/R3 + {k['rainfall_24h_accumulated']}*P24h/R24 + "
                         f"{k['tide_x_rain']}*max(0, mare - T_hi)/T_span * 1[P3h > wet3]))"),
             "b0": b0, "coefficients": COEF, "anchors": a, "target_positive_rate": TARGET_POSITIVE_RATE,
             "true_features_ranked": TRUE_FEATURES, "strata": weights, "seed": seed, "n_rows": int(len(sample)),
             "positive_rate": float(sample[C.LABEL].mean())}
    return sample, truth


def write(sample: pd.DataFrame, truth: dict) -> None:
    keep = (["timestamp", "slug", "stratum", "event_id", "station_id", "station_distance_m", "p_true", C.LABEL]
            + C.FEATURES + [f + "_rea" for f in C.RAIN_FEATURES])
    sample[keep].to_csv(C.DATASET_CSV, index=False, compression=C.GZIP)
    truth["dataset_sha256"] = hashlib.sha256(C.DATASET_CSV.read_bytes()).hexdigest()
    C.REPORTS_DIR.mkdir(exist_ok=True)
    (C.REPORTS_DIR / "synthetic_ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False))
    a = truth["anchors"]
    md = f"""# Fórmula geradora do rótulo sintético

**{C.SYNTHETIC_TAG}.** Nenhum número desta pasta é desempenho em alagamento real.

Por que sintético: 0 ocorrências de alagamento a 500 m dos pontos e nenhuma com hora no banco (`mapi_ai/build_labels.py`,
`reports/dataset_summary.json`). O piso para um classificador supervisionado era {C.MIN_REAL_POSITIVES} positivos.

## Fórmula

```
{truth['formula']}
```

P1h, P3h e P24h são a chuva **medida** (Cemaden, série do ponto) em mm; maré em m sobre o NR da DHN.

| Escala | Valor | Origem (dado real) |
|---|---|---|
| R1 | {a['R1_mm']:.2f} mm | mediana do pico horário dos eventos FORTE (tabela de eventos de 5 anos) |
| R3 | {a['R3_mm']:.2f} mm | quantil 99,9% da chuva de 3 h da série dos pontos |
| R24 | {a['R24_mm']:.2f} mm | quantil 99% da chuva de 24 h |
| T_hi | {a['T_hi_m']:.3f} m | quantil 90% da maré horária |
| T_span | {a['T_span_m']:.3f} m | quantil 99% − quantil 90% da maré |
| wet3 | {a['wet3_mm']:.2f} mm | quantil 95% da chuva de 3 h ("está chovendo" para a interação maré × chuva) |
| b0 | {truth['b0']:.3f} | resolvido por bisseção para {TARGET_POSITIVE_RATE:.0%} de positivos esperados na amostra |

Coeficientes (escolhidos antes de treinar, `[sem fonte]` como valor físico): {COEF}. Com (1,5; 2,5; 0,8; 3,0) a hora seca tinha 4,3% de positivos e o rótulo virava ruído. Ordem de importância verdadeira,
medida como atribuição de cada termo nos dados do teste (|termo − média|, escala do logit;
`mapi_ai/xai/evaluate_explanations.py`): **chuva de 3 h > chuva de 24 h > chuva de 1 h > maré (só com chuva)**. A ordem dos
coeficientes é outra (3 h > 1 h > maré > 24 h): a chuva de 24 h tem coeficiente menor, mas varia muito mais entre as horas. Não entram na fórmula: chuva de 6 h e
12 h (correlacionadas com 3 h e 24 h), hora, mês, latitude e longitude.

## Amostra

{truth['n_rows']} horas reais, {truth['positive_rate']:.1%} positivas, semente {truth['seed']}. Estratos (horas do evento + 3 h depois):

| Estrato | Horas disponíveis | Sorteadas | Probabilidade de inclusão |
|---|---|---|---|
""" + "\n".join(f"| {k} | {v['pool_hours']} | {v['sampled']} | {v['inclusion_prob']:.4f} |" for k, v in truth["strata"].items()) + f"""

A amostra super-representa chuva forte em relação ao clima; métricas valem para esta amostra, não para "uma hora qualquer".

## Limites

- O rótulo é função da chuva **medida**. A ablação Cemaden × reanálise mede quanta informação sobre a chuva medida no
  local a reanálise preserva; por construção favorece o Cemaden e não prova que o Cemaden prevê alagamento real melhor.
- Ruído do rótulo = sorteio de Bernoulli; não há erro de notificação simulado.
- SHA-256 do dataset: `{truth['dataset_sha256']}`.

{C.ATTRIBUTION}
"""
    (C.REPORTS_DIR / "synthetic_ground_truth.md").write_text(md)


if __name__ == "__main__":
    s, t = build()
    write(s, t)
    print(f"{len(s)} linhas, {s[C.LABEL].mean():.3%} positivas -> {C.DATASET_CSV}")
