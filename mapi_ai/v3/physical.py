"""
Variante do v3 com entradas físicas (etapa B5).

O GERADOR MUDA: o rótulo desta variante vem de uma fórmula em que o pico de 10 min, a tendência da maré e a umidade do
solo têm papel físico declarado. Por isso há duas variantes sobre o MESMO rótulo novo, e é entre elas que se compara:

- fisico_base: as 10 entradas do v3 (sem as físicas);
- fisico:      as 10 + rain_max_10min_mm, tide_trend_m_h e soil_moisture.

Comparar "fisico" com o v2 ou com o v3 principal como se fosse ganho de modelo é errado: o rótulo é outro.

Fórmula (logit), sobre a do v2 (z_v2, sem alteração), com o intercepto resolvido para manter a prevalência de H = 0 da
série contínua igual à do v3 principal:

    z = b0' + (z_v2 - b0)
        + K_PEAK  * I10 / CAP10                                   pico de 10 min contra a capacidade da galeria
        + K_TREND * max(0, dM/dt) / TREND_REF * mare_alta * chovendo   maré enchendo segura mais que vazando
        + K_SOIL  * (solo - SOIL_MED) / SOIL_SPAN * P3h / R3      solo úmido aumenta o escoamento da mesma chuva

O nível do rio NÃO entra: a régua do Duas Unas só tem leituras desde 2026-11-09 (sem histórico).

Ausência: o modelo vê rain_max_10min_mm como a série real o tem (falta em 40% a 80% das horas com chuva; a ausência
depende da chuva) e ausente também nas lacunas injetadas. Onde o pico verdadeiro não é conhecido, o RÓTULO usa o piso
P1h/6 (chuva uniforme na hora). A umidade do solo existe no histórico, mas chega com ~2 semanas de atraso: ao vivo
vem ausente, e por isso a variante também é avaliada com o solo ausente no teste.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from mapi_ai import in1165_config as C2
from mapi_ai.synthetic import solve_intercept
from mapi_ai.v3 import config as C
from mapi_ai.v3 import dataset as D
from mapi_ai.v3 import evaluate as E
from mapi_ai.v3 import train as T
from mapi_ai.v3.sources import load_physical

# Capacidade de projeto da galeria: 60,1 mm/h (projeto-mapi-api/docs/EVENTOS-DE-CHUVA.md) = 10,02 mm em 10 min.
CAP10_MM = 60.1 / 6
# Coeficientes escolhidos ANTES de treinar, [sem fonte] como valor físico: o pico na capacidade vale 4 no logit (dois
# terços do termo de 1 h do v2 no seu ponto de escala); maré enchendo no quantil 90% soma metade do termo de maré alta;
# solo no quantil 90% com chuva de 3 h na escala R3 soma 3.
K_PEAK, K_TREND, K_SOIL = 4.0, 3.0, 3.0
HORIZONS = (C.PRIMARY_HORIZON,)


def build() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(C.SEED + 7)
    truth_v2 = json.loads((C2.REPORTS_DIR / "synthetic_ground_truth.json").read_text())
    a, b0 = truth_v2["anchors"], truth_v2["b0"]
    cont = D.load("continuous")
    phys = load_physical().rename(columns={"hour_end_utc": "timestamp"})
    cont = cont.merge(phys, on=["slug", "timestamp"], how="left")

    p1 = cont["rain_0_1h_full"]
    p3 = p1 + cont["rain_1_3h_full"]
    tide_ex = np.clip(cont["tide_level"] - a["T_hi_m"], 0, None) / a["T_span_m"]
    wet = (p3 > a["wet3_mm"]).astype(float)
    i10_true = cont["rain_max_10min_mm"].fillna(p1 / 6)            # piso onde o pico não é conhecido
    anchors = {"CAP10_mm": CAP10_MM, "TREND_REF_m_h": float(cont["tide_trend_m_h"].quantile(0.90)),
               "SOIL_MED": float(cont["soil_moisture"].median()),
               "SOIL_SPAN": float(cont["soil_moisture"].quantile(0.90) - cont["soil_moisture"].median())}
    extra = (K_PEAK * i10_true / CAP10_MM
             + K_TREND * np.clip(cont["tide_trend_m_h"], 0, None) / anchors["TREND_REF_m_h"] * tide_ex * wet
             + K_SOIL * (cont["soil_moisture"] - anchors["SOIL_MED"]) / anchors["SOIL_SPAN"] * p3 / a["R3_mm"])
    z_wo = E.logit(cont["p_true"]) - b0 + extra
    defined = z_wo.notna()
    target = float(cont.loc[cont.labeled, "flood_h0"].mean())       # mesma prevalência contínua do v3 principal
    b0_new = solve_intercept(z_wo[defined & cont.labeled].to_numpy(), target)
    p = 1 / (1 + np.exp(-(b0_new + z_wo)))
    cont["p_true"] = p
    cont["flood_h0"] = np.where(defined, (rng.random(len(cont)) < p).astype(float), np.nan)
    for h in C.HORIZONS:
        cont[C.label(h)] = cont.groupby("slug")["flood_h0"].transform(lambda s, h=h: D.horizon_label(s, h))
    cont["labeled"] = cont[[C.label(h) for h in C.HORIZONS]].notna().all(axis=1) & cont["tide_level"].notna()
    # o que o modelo vê: pico ausente onde a série real não o tem e nas lacunas injetadas; maré e solo como estão
    cont.loc[cont["gap_injected"], "rain_max_10min_mm"] = np.nan
    smp, sinfo = D.sample(cont, seed=C.SEED + 7)
    lab = cont[cont.labeled]
    info = {"formula": "z = b0' + (z_v2 - b0) + K_PEAK*I10/CAP10 + K_TREND*max(0,dM/dt)/TREND_REF*mare_alta*chovendo "
                       "+ K_SOIL*(solo - SOIL_MED)/SOIL_SPAN*P3h/R3",
            "coefficients": {"K_PEAK": K_PEAK, "K_TREND": K_TREND, "K_SOIL": K_SOIL}, "anchors": anchors, "b0": b0_new,
            "prevalence_continuous_h0": float(lab["flood_h0"].mean()),
            "peak_unknown_label_uses": "P1h/6 (chuva uniforme na hora)",
            "missing_rate_sample": {f: float(smp[f].isna().mean()) for f in C.PHYSICAL},
            "missing_rate_peak_by_rain": {
                "seco": float(lab.loc[lab["rain_0_1h_full"] == 0, "rain_max_10min_mm"].isna().mean()),
                "0_a_5mm": float(lab.loc[(lab["rain_0_1h_full"] > 0) & (lab["rain_0_1h_full"] < 5), "rain_max_10min_mm"].isna().mean()),
                "5mm_ou_mais": float(lab.loc[lab["rain_0_1h_full"] >= 5, "rain_max_10min_mm"].isna().mean())},
            "river_level_m": "fora: régua do Duas Unas só desde 2026-11-09", "sample": sinfo}
    return cont, smp, info


def run() -> dict:
    cont, smp, info = build()
    out = {"tag": C.SYNTHETIC_TAG, "generator": info, "variants": {}}
    te = smp[smp.part == "test"]
    y_te = te[C.label(C.PRIMARY_HORIZON)].astype(int).to_numpy()
    for tag, feats in (("fisico_base", C.FEATURES), ("fisico", C.FEATURES_PHYSICAL),
                       ("fisico_sem_solo", C.FEATURES + ["rain_max_10min_mm", "tide_trend_m_h"])):
        meta = T.run(features=feats, tag=tag, dataset=smp, horizons=HORIZONS,
                     extra_meta={"generator": "reports/v3/physical_comparison.json (fórmula com entradas físicas)"})
        ev = E.run(tag=tag, features=feats, cont=cont, smp=smp, horizons=HORIZONS)
        hk = f"h{C.PRIMARY_HORIZON}"
        sel = meta["horizons"][hk]["selection"]
        import joblib
        m = joblib.load(C.ROOT / meta["production"]["artifact"])
        res = {"features": feats, "chosen": sel["chosen"], "validation_pr_auc": {r["model"]: r["val_pr_auc"] for r in sel["table"]},
               "test_pr_auc_chosen": float(average_precision_score(y_te, m.predict_proba(T.X_of(te, feats))[:, 1])),
               "alerts_and_events_test": ev["horizons"][hk]["alerts_and_events"]["test"],
               "calibration_adopted": ev["horizons"][hk]["calibration"]["adopted"]}
        if "soil_moisture" in feats:
            X = T.X_of(te, feats)
            X["soil_moisture"] = np.nan                             # como em produção: o ERA5-Land ainda não chegou
            res["test_pr_auc_soil_absent_as_in_production"] = float(average_precision_score(y_te, m.predict_proba(X)[:, 1]))
        out["variants"][tag] = res
        print(tag, res["chosen"], {k: round(v, 3) for k, v in res["validation_pr_auc"].items()}, "teste", round(res["test_pr_auc_chosen"], 3),
              res.get("test_pr_auc_soil_absent_as_in_production"))
    (C.REPORTS_DIR / "physical_comparison.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    from mapi_ai.v3 import physical as _p
    _p.run()
