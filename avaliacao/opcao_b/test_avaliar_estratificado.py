"""DADOS SINTÉTICOS — teste de unidade da avaliação estratificada (Opção B)."""
import numpy as np
import pandas as pd

from avaliacao.opcao_b.avaliar_estratificado import evaluate_stratified

SYNTHETIC_LABEL = "DADOS SINTÉTICOS — teste de unidade"


def _synthetic_frame(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 60
    ts = pd.date_range("2026-05-01T00:00:00Z", periods=n, freq="1h")
    rain = rng.uniform(0.0, 40.0, size=n)
    y = (rain > 25.0).astype(int)
    tiers = ["HIGH" if i % 3 != 1 else "MEDIUM" for i in range(n)]
    tiers[-2] = "LOW"
    tiers[-1] = "DISCARD"
    weights = [1.0 if t == "HIGH" else (0.7 if t == "MEDIUM" else (0.4 if t == "LOW" else 0.0)) for t in tiers]
    rain_tiers = ["MEASURED_PRIMARY"] * n
    rain_tiers[5] = "REANALYSIS_FALLBACK"
    rain_tiers[-5] = "REANALYSIS_FALLBACK"
    return pd.DataFrame({
        "datasetLabel": SYNTHETIC_LABEL,
        "floodPointSlug": ["PA"] * n,
        "timestampUtc": ts.astype(str),
        "label_h6": y,
        "tier": tiers,
        "sample_weight": weights,
        "temporal_match": ["EVENT_WINDOW" if v == 1 else "CLEAN_NEGATIVE" for v in y],
        "rain_tier": rain_tiers,
        "rain_confidence": [0.6 if rt == "REANALYSIS_FALLBACK" else 1.0 for rt in rain_tiers],
        "weatherPrecipitation": rain,
        "accumulated3h": rain * 1.5,
        "accumulated6h": rain * 2.0,
        "accumulated12h": rain * 2.2,
        "accumulated24h": rain * 2.5,
        "tideHeight": rng.uniform(0.3, 2.2, size=n),
    })


def test_estratos_vazios_reporta_mensagem_explicita_sem_quebrar():
    df = _synthetic_frame()
    df["label_h6"] = 0  # nenhum positivo em nenhum estrato
    df["temporal_match"] = "TEMPORAL_DAY_ONLY"
    res = evaluate_stratified(df)
    assert res["status"] == "nao_executavel"
    assert "0 positivos no estrato" in res["main_table_high"]["status"]
    assert "não executável" in res["main_table_high"]["status"]
    assert "0 positivos no estrato" in res["bronze_exploratory"]["status"]
    assert res["temporal_day_only_rows"] == len(df)


def test_split_temporal_separa_high_bronze_e_reanalise():
    df = _synthetic_frame()
    res = evaluate_stratified(df)
    assert res["status"] == "ok"
    assert res["reanalysis_report"]["rows"] == 2
    # Tabela principal só entra HIGH medido no período de teste
    main = res["main_table_high"]
    assert main["status"] == "ok"
    assert main["positives"] >= 1
    assert 0.0 <= main["pr_auc"] <= 1.0
    assert 0.0 <= main["brier"] <= 1.0
    # Sensibilidade inclui MEDIUM além de HIGH
    assert res["sensitivity_high_medium"]["rows"] >= main["rows"]
