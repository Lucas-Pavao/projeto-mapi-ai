"""
Configuração única do projeto final IN1165 (Advanced ML & Explainable AI, CIn/UFPE).

Tudo que define o experimento fica aqui, para que o notebook, os scripts e os testes usem os mesmos números:
semente, caminhos, ordem das features e a regra do split temporal.
"""
from pathlib import Path

# Semente única. Toda aleatoriedade (amostragem, rótulo sintético, modelos, LIME, DiCE) parte dela,
# para que "Run All" reproduza os mesmos números.
SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "in1165"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

# Arquivos extraídos do banco do projeto-mapi-api (handoff etapa0.json). Ficam versionados para que o notebook
# rode sem o banco. O SHA-256 de cada um vai para reports/dataset_summary.json.
CEMADEN_HOURLY = DATA_DIR / "cemaden_point_hourly.csv.gz"     # point_rain_accum_v, chuva-ponto-v4 (medida)
REANALYSIS_HOURLY = DATA_DIR / "reanalysis_point_hourly.csv.gz"  # observation_hourly_v, OPEN_METEO_ARCHIVE
TIDE_HOURLY = DATA_DIR / "tide_hourly.csv.gz"                 # observation_hourly_v, TIDE_PE02_HOURLY (m, NR DHN)
OCCURRENCES = DATA_DIR / "flood_occurrences.csv.gz"           # flood_occurrence (CKAN Recife)
POINTS = DATA_DIR / "flood_points.csv"
RAIN_EVENTS = DATA_DIR / "eventos_CEMADEN_PED.csv.gz"         # tabela de eventos de 5 anos (ADR 0017/0022)

DATASET_CSV = DATA_DIR / "dataset.csv.gz"   # dataset final (real ou sintético) que o treino lê

# Ordem EXATA das features que o modelo espera. Os nomes são os do contrato /v1/predict/flood
# (PredictionInput), para que a inferência não precise de tradução.
RAIN_FEATURES = [
    "current_rainfall",          # mm na última hora
    "rainfall_3h_accumulated",   # mm nas últimas 3 h
    "rainfall_6h_accumulated",
    "rainfall_12h_accumulated",
    "rainfall_24h_accumulated",
]
FEATURES = RAIN_FEATURES + [
    "tide_level",                # m, NR da DHN, maré horária prevista do Porto do Recife
    "hour_sin", "hour_cos",      # hora local do dia, codificada em círculo (23 h fica perto de 0 h)
    "month_sin", "month_cos",    # mês, idem (sazonalidade)
    "latitude", "longitude",     # graus; o ponto crítico
]
UNITS = {
    "current_rainfall": "mm", "rainfall_3h_accumulated": "mm", "rainfall_6h_accumulated": "mm",
    "rainfall_12h_accumulated": "mm", "rainfall_24h_accumulated": "mm", "tide_level": "m (NR DHN)",
    "hour_sin": "adimensional", "hour_cos": "adimensional", "month_sin": "adimensional", "month_cos": "adimensional",
    "latitude": "graus", "longitude": "graus",
}
LABEL = "is_flooded"

# Split temporal em três partes, sem embaralhar: o passado treina, o futuro valida e testa.
SPLIT = (0.60, 0.20, 0.20)

# Piso de positivos reais para um classificador supervisionado (prompt IN1165, Etapa 1).
MIN_REAL_POSITIVES = 30

SYNTHETIC_TAG = "DADOS SINTÉTICOS — experimento controlado"
ATTRIBUTION = "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC"
