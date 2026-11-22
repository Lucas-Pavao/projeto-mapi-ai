"""
"Dia de risco" (prompt dos horizontes, bloco C): PRÉ-REGISTRO. Tudo neste arquivo foi escrito e commitado ANTES de
baixar o histórico de rodadas e de rodar qualquer avaliação (22/11/2026). Nada aqui é mudado depois de olhar o teste.

Pergunta: a previsão numérica do Open-Meteo (ECMWF, ICON e o modelo usado hoje), vista de manhã, diz se haverá evento
FORTE no ponto nas próximas H horas melhor do que a climatologia?

Não usa rótulo de alagamento. O alvo é chuva MEDIDA (rede Cemaden/PED). O rain_v1 não é tocado.
"""
from __future__ import annotations

import math

import pandas as pd

from mapi_ai.rain import config as RC

ROOT = RC.ROOT
DATA_DIR = ROOT / "data" / "day_risk"
RAW_DIR = DATA_DIR / "raw"
OM_TABLE = DATA_DIR / "om_previous_runs.csv.gz"     # derivado dos brutos (python -m mapi_ai.dayrisk.fetch --parse)
REPORTS_DIR = ROOT / "reports" / "day_risk"
VERSION = "day_risk_v0"                              # v0 = avaliação; só vira serviço se passar no critério
ATTRIBUTION = "DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC"
OM_ATTRIBUTION = "Open-Meteo (ECMWF/DWD/NOAA), CC BY 4.0"

# ---- Alvo (C1) ----------------------------------------------------------------------------------------------------
# Evento FORTE (ADR 0017): alguma hora cheia dentro de (t, t+H] com chuva do ponto >= 25,15 mm (pico horário), a mesma
# definição do rain_v1 (features.targets): janelas de 60 min que terminam em [t+60 min, t+H]. Chuva do ponto = p_60 de
# data/rain/dataset.pkl.gz (maior valor entre as estações válidas a até 5 km, ADR 0022).
FORTE_MM = RC.THRESHOLDS["FORTE"]
HORIZONS = (6, 12, 24)
# Rótulo 1 se alguma hora cheia válida da janela passa do limiar; 0 só se >= 90% das horas cheias da janela são
# válidas e nenhuma passa; senão ausente (fora da avaliação, contado). Lacuna nunca vira "sem evento".
LABEL_MIN_VALID = 0.90

# ---- Instante de emissão (C1) -------------------------------------------------------------------------------------
TZ = "America/Recife"
EMISSION_MORNING_LOCAL_HOUR = 6                      # o aviso de manhã (06h local = 09 UTC)
EMISSION_EVERY_3H_LOCAL = tuple(range(0, 24, 3))     # teste adicional

# ---- Período e cortes ---------------------------------------------------------------------------------------------
# Início pelo histórico de rodadas (o ECMWF 0,25° começa em 2024-02-03 no previous-runs-api; o _previous_day2 em
# 2024-02-04). Fim pelo fim da série medida do ponto (rain.config.DATA_END). Uma janela (t, t+H] nunca cruza um corte.
START = pd.Timestamp("2024-02-06T00:00:00Z")
TRAIN_END = pd.Timestamp("2025-02-01T03:00:00Z")     # 2025-02-01 00h local
VALID_END = pd.Timestamp("2025-11-01T03:00:00Z")     # 2025-11-01 00h local
END = RC.DATA_END                                    # 2026-10-31 00:00 UTC
# treino: [START, TRAIN_END) · validação: [TRAIN_END, VALID_END) · teste: [VALID_END, END), cada um com uma estação
# chuvosa (mar-ago) inteira: 2024, 2025 e 2026.

# ---- Fontes de previsão (C2) --------------------------------------------------------------------------------------
# previous-runs-api do Open-Meteo: <var>_previous_dayN = valor previsto N x 24 h antes da hora válida (documentação,
# lida em 22/11/2026). Precipitação horária = soma da hora que termina no carimbo.
OM_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
OM_MODELS = ("ecmwf_ifs025", "icon_global", "best_match")   # ECMWF IFS 0,25° (o de 9 km não tem histórico), ICON, atual
OM_CAPE_MODEL = "best_match"                                 # único dos três com cape_previous_dayN no histórico
OM_START_DATE = "2024-01-15"
OM_END_DATE = "2026-11-01"
OM_PAUSE_S = 2.0                                     # uma requisição por vez, pausa entre elas
# 850 hPa: o previous-runs-api recusa relative_humidity_850hPa_previous_day1 (conferido em 22/11). Fica de fora.

# Trava de vazamento: a rodada que gera <var>_previous_dayN para a hora h foi iniciada até h - 24N e publicada até
# ~7 h depois (maior atraso visto em meta.json em 22/11: ECMWF 0,25°, 7h05). Margem de 8 h. Só vale a hora h se
# h - 24N + 8 h <= t (emissão). N(H) é o menor N que cobre a janela inteira.
PUBLICATION_DELAY_H = 8


def lead_days(h: int) -> int:
    """Menor N com 24 N >= H + atraso de publicação: H=6 -> 1, H=12 -> 1, H=24 -> 2."""
    return math.ceil((h + PUBLICATION_DELAY_H) / 24)


# ---- Entradas (C2) ------------------------------------------------------------------------------------------------
# om_<modelo>: maior chuva horária prevista entre as horas que terminam em (t, t+H] (lead N(H)); ausente se faltar
# qualquer hora da janela (sem soma parcial). cape: maior CAPE previsto na janela (best_match). obs24: chuva medida do
# ponto em (t-24 h, t], soma dos 144 passos de 10 min, ausente se faltar qualquer passo. persist: alguma hora cheia
# medida em (t-24 h, t] >= 25,15 mm (mesma regra de ausência do rótulo).
# Conjunto avaliado por H: instantes com rótulo e todas as entradas presentes. Os mesmos instantes para todos os
# modelos. Exclusões contadas no relatório.

# ---- Modelos, do mais simples ao mais complexo (C2) ---------------------------------------------------------------
MODELS = (
    "climatologia",          # 1. taxa por mês (e hora de emissão, nas emissões a cada 3 h)
    "persistencia",          # 2. P(evento | choveu forte nas últimas 24 h), tabela 2x1
    "om_ecmwf_ifs025",       # 3. cada modelo sozinho: logística em log1p(máx previsto)
    "om_icon_global",
    "om_best_match",
    "om_media",              # 4. combinação: logística em log1p(média dos máximos) e em log1p(máximo dos máximos)
    "om_maximo",
    "logistica_completa",    # 5. log1p(média), cape/1000, log1p(obs24), persist
)
CLIMATOLOGY_PRIOR_N = 30     # suavização da taxa mensal rumo à taxa geral (pseudo-instantes)
PERSIST_PRIOR_N = 30
LOGIT_C = 1.0                # sklearn LogisticRegression(C=1.0), entradas padronizadas no ajuste
# Climatologia: ajustada em toda a série medida ANTES do início da parte avaliada (desde 2021-09; só chuva medida).
# Demais modelos: coeficientes no treino para avaliar a validação; reajustados em treino+validação (mesma
# especificação) para avaliar o teste.

# ---- Critério de adoção (C2), escrito antes de rodar ---------------------------------------------------------------
SELECTION_HORIZON = 12
SELECTION_EMISSION = "06h"
ADOPTION_RULE = (
    "Em H = 12 h, emissão das 06h: o melhor modelo é o de maior Brier skill score (BSS) contra a climatologia na "
    "VALIDAÇÃO. Fica o mais simples (ordem de MODELS) cujo BSS de validação está dentro do IC 95% do melhor. O 'dia de "
    "risco' só vai para a tela se esse modelo, no TESTE, tiver BSS contra a climatologia com IC 95% sem o zero "
    "(limite inferior > 0). Se não passar, o resultado é esse: o horizonte do dia fica só com o INMET.")
N_BOOT = 2000
SEED = 20261122
BOOT_UNIT = "dia local (America/Recife): todos os pontos e emissões do dia entram juntos"
# Limiar de "dia de risco" para as contagens de aviso: o de maior CSI na validação, por H e modelo.
RELIABILITY_BINS = 10        # curva de confiabilidade e ECE: 10 faixas de mesma contagem
MIN_TEST_EVENTS = 15         # abaixo disso o relatório diz, em destaque, que os intervalos são largos
