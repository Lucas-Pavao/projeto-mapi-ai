# Previsão de chuva (rain_v1) — relatório da Fase 3

DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC.

O alvo é **chuva medida**: maior acumulado entre as estações válidas a até 5 km do ponto, em `(t, t+1h]`, `(t, t+3h]` e `(t, t+6h]`. Não há rótulo sintético neste relatório.

Regenera com `python -m mapi_ai.rain.dataset`, `train`, `evaluate`, `explain`, `metadata`, `oof` (cerca de 4 min no total). Semente 42. Números: `reports/v4/rain/evaluation.json`, `metrics.csv`, `selection.json`, `shap_global.json`.

## Resposta em uma linha

**A rede avisa um evento FORTE 20 minutos antes, em mediana (IC 95%: 10 a 40 min; modelo de 1 h, 20 de 28 eventos do teste), e sempre com a chuva já caindo no ponto: nenhum dos avisos de 1 h saiu antes de começar a chover.**

Com o modelo de 3 h a mediana é de 40 min (IC 95%: 20 a 140), com 1 aviso em 28 dado ainda sem chuva.

## Dados e split

| Item | Valor | Fonte |
|---|---|---|
| Leituras de 10 min | 1.485.181, de 46 estações da PED na RMR, 2021-09 a 2026-11 | `data/rain/ped_10min.csv.gz` |
| Grade | 10 min × 5 pontos = 1.357.745 linhas | `python -m mapi_ai.rain.dataset` |
| Treino | até 2024-07-03 (745.985 linhas; 1 a cada 30 min no ajuste: 248.662) | `models/v4/rain/train_info.json` |
| Validação | até 2025-07-31 (282.360) | idem |
| Teste | até 2026-10-31 (329.040), lido uma vez | idem |
| Folga entre as partes | 6 h (o maior horizonte) | `mapi_ai/rain/config.py` |
| Eventos FORTE no teste | 28 cruzamentos de 25,15 mm/h, em **15 dias distintos** | `evaluate.events` |

O anel de 15 a 40 km fora da RMR não foi carregado: a carga pela PED exige token ADMIN. O modelo usa as 46 estações já carregadas (17 delas a mais de 15 km de algum ponto).

Entradas (30 na rede, 3 do Open-Meteo), todas com informação até `t` (teste `test_entradas_nao_usam_nada_depois_de_t`): chuva do ponto em 10, 30, 60 e 180 min e a tendência; por setor (N, S, L, O), maior valor em 30 e 60 min, fração de estações com chuva e minutos desde o início da chuva; por anel (5–15 e 15–40 km), maior valor, média e fração com chuva; fração de estações válidas na rede; hora e mês. Estação sem leitura em 70 min fica ausente, nunca zero.

Open-Meteo: **rodada da véspera** (`previous-runs-api`, `precipitation_previous_day1`), emitida pelo menos 24 h antes da hora prevista. Só existe de 2024-01 em diante, então o treino tem 6 meses dela. A previsão emendada (`historical-forecast-api`) não foi usada como entrada porque não separa a emissão.

Período sem arquivo (conferido em 2026-11-22, onda 16): antes de 2024-01-19 a `previous-runs-api` não tem a rodada, e `om_1h`, `om_3h` e `om_6h` ficam ausentes (NaN, nunca zero). Isso vale para **84% das linhas de treino** (só 16% têm a entrada); validação e teste têm 100%. O XGBoost aprende um ramo próprio para "ausente", o mesmo que usa no serviço quando o Open-Meteo falta. Em 2026-11-30 o `best_match` muda por dentro (ICON na grade nativa): quebra de série registrada na api (V43) e conferida toda semana (`rain_forecast_check`, V44; referência em `referencia_conferencia_semanal.json`).

## Os cinco modelos

Critério fixado antes do teste: por horizonte, fica o mais simples cuja diferença de erro de quantil para o melhor, na validação, tem IC 95% contendo zero (bootstrap pareado por bloco de dia, 1.000 reamostras).

Erro de quantil (pinball médio de 50%, 80% e 95%, em mm; menor é melhor). Validação entre parênteses.

| Modelo | 1 h | 3 h | 6 h |
|---|---|---|---|
| 1. Climatologia (mês × hora) | 0,226 (0,247) | 0,552 (0,605) | 0,973 (1,051) |
| 2. Persistência (faixa da chuva da última hora) | 0,172 (0,184) | 0,447 (0,485) | 0,841 (0,898) |
| 3. Open-Meteo sozinho | 0,228 (0,239) | 0,566 (0,573) | 0,987 (0,978) |
| 4. Gradient boosting com a rede | **0,144 (0,152)** | 0,385 (0,414) | 0,743 (0,790) |
| 5. O 4 mais o Open-Meteo | 0,146 (0,153) | **0,381 (0,403)** | **0,725 (0,755)** |
| Escolhido | 4 (empata com o 5; o 4 é mais simples) | 5 | 5 |

No teste, a diferença de cada modelo para o escolhido é significativa em todos os casos (IC 95% sem o zero). Em 1 h o Open-Meteo piora um pouco (0,146 contra 0,144); em 3 h e 6 h melhora (1% e 2%).

- **A rede vence a persistência** em 16% (1 h), 15% (3 h) e 14% (6 h).
- **O Open-Meteo sozinho não vence a climatologia** em 1 h nem em 6 h no teste (resultado negativo). Só ajuda somado à rede.

## Por regime: já chove contra ainda não chove

"Já chove" = 0,2 mm ou mais no ponto nos últimos 60 min (18,9% das linhas do teste).

| Horizonte (modelo) | Pinball, já chove | Pinball, ainda não chove | Recall FORTE, já chove | Recall FORTE, ainda não chove |
|---|---|---|---|---|
| 1 h (4) | 0,480 | 0,065 | 32% (de 159 linhas) | **0%** (de 4) |
| 3 h (5) | 1,106 | 0,212 | 44% (de 461) | **4%** (de 68) |
| 6 h (5) | 1,899 | 0,452 | 46% (de 840) | **5%** (de 193) |

**O modelo prevê a continuação e a intensificação de uma chuva que já começou. O início ele não prevê** (resultado negativo, e o esperado pela Fase 2: a chuva vem do mar, onde não há pluviômetro).

## Cobertura dos quantis (teste)

| Horizonte | 50% | 80% | 95% | 95%, só quando já chove |
|---|---|---|---|---|
| 1 h | 83,6% | 89,7% | **96,4%** | 95,3% |
| 3 h | 75,1% | 88,1% | **97,1%** | 96,2% |
| 6 h | 68,5% | 88,6% | **97,0%** | 96,0% |

O quantil de 95% contém a medida em 96% a 97% das vezes. Os de 50% e 80% cobrem mais que o nominal porque a maior parte das horas tem chuva zero e previsão zero. Quando já chove: 50% cobre 53% a 57%; 80% cobre 80% a 84%.

## Evento FORTE: precisão, recall e antecedência real (teste)

Aviso = probabilidade de FORTE acima do limiar de maior CSI na validação. Evento = primeiro instante em que a chuva do ponto em 60 min chega a 25,15 mm. Antecedência = minutos entre o primeiro aviso (dentro do horizonte) e o cruzamento; evento sem aviso conta 0. IC por bootstrap de evento.

| Horizonte (modelo) | Precisão | Recall | Eventos avisados | Antecedência mediana (IC 95%) | p10 | Avisos ainda sem chuva | Falsos por evento | Horas de aviso por ponto por ano |
|---|---|---|---|---|---|---|---|---|
| 1 h (4) | 15% | 31% | 20 de 28 | **20 min (10–40)** | 0 | 0 de 28 | 3,0 | 9 |
| 3 h (5) | 20% | 39% | 21 de 28 | **40 min (20–140)** | 0 | 1 de 28 | 4,2 | 28 |
| 6 h (5) | 15% | 38% | 24 de 28 | 195 min (40–310) | 0 | 2 de 28 | 11,4 | 68 |
| Persistência, 1 h | 2% | 80% | 28 de 28 | 55 min (40–60) | 30 | 0 de 28 | 22,8 | 148 |

- A persistência "avisa" todos com 55 min porque dispara sempre que choveu 5 mm na última hora: 23 falsos por evento. Não é comparável sem olhar essa coluna.
- O aviso de 6 h tem 11 falsos por evento. É atenção, não alerta.
- **p10 = 0 em todos:** em pelo menos 1 de cada 10 eventos o aviso não sai antes do cruzamento.
- Os 28 eventos caem em 15 dias: os pontos compartilham a mesma tempestade. O IC por evento é otimista.

## SHAP global: de onde vem a previsão

Média de |SHAP| no quantil de 80%, validação, 20.000 linhas (`figures/shap_global.png`).

| Horizonte | Próprio ponto (até 5 km) | Estações vizinhas (5–40 km) | Open-Meteo | Hora e mês | Estado da rede |
|---|---|---|---|---|---|
| 1 h | **50,5%** | 42,7% | — | 5,4% | 1,4% |
| 3 h | 23,5% | **51,3%** | 8,9% | 13,2% | 3,1% |
| 6 h | 16,2% | **42,8%** | 13,6% | 22,2% | 5,2% |

Em 1 h a previsão vem metade do próprio ponto (chuva dos últimos 180 e 10 min). De 3 h em diante, a entrada que mais pesa é a **fração de estações com chuva no anel de 5 a 15 km**: o modelo mede o tamanho da chuva, não a direção de onde ela vem. Em 6 h, hora e mês pesam 22%: boa parte é climatologia.

## Limitações

- Atraso do dado: a PED chega em carga horária e a coleta ao vivo em blocos de 10 min. A antecedência acima conta a partir da leitura; em operação desconta-se a idade da leitura, que a rota devolve (`latest_reading_age_minutes`). Não foi medida em dia de chuva `[sem fonte]`.
- Na produção a checagem mensal de estação e a retirada de pico isolado (ADR 0022) não se aplicam à previsão: precisam do mês inteiro.
- 6 meses de Open-Meteo no treino.
- 28 eventos FORTE em 15 dias no teste.

## Previsão fora da amostra para a Fase 4

`data/v4/rain_forecast_oof.csv.gz`: 226.295 linhas horárias. Validação e teste vêm do modelo final; o treino, de 4 blocos no tempo, cada um previsto por um modelo ajustado nos outros três com folga de 6 h.
