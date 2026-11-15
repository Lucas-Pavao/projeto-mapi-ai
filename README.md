# MAPI AI - Inteligência Artificial e Predição 🧠🌊

O **MAPI AI** é o componente de inteligência do ecossistema MAPI, responsável pela análise avançada de séries temporais e predição de riscos de inundação urbana em tempo real.

## 🌐 Ecossistema MAPI

Este projeto atua como um **microserviço especializado de inferência síncrona**:

```text
  [ Agências: APAC, ANA, Cemaden, TabuaMare, Open-Meteo, CKAN Recife ]
        │   (coleta por jobs agendados na própria API)
        ▼
  [  MAPI API  ] (Java 21 / Spring Boot / TimescaleDB) 🌊🚀
        │ ▲
        │ │ (Dados em Tempo Real via HTTP POST / Resposta com Probabilidade e Risco)
        ▼ │ <-- (Este Serviço interage aqui de forma síncrona)
  [  MAPI AI  ] (Python / FastAPI / XGBoost & LSTM) 🧠
        │
        │ (Consumo da REST API e Exibição Geoespacial)
        ▼
  [ MAPI Front ] (React 19 / MapLibre GL) 💻✨
```

### Interdependência de Fluxo:

1. **Fase de Treinamento (Batch):** O MAPI AI conecta-se ao banco de dados do MAPI API para ler dados históricos consolidados.
2. **Fase de Inferência (Tempo Real):** O MAPI API envia um HTTP POST para o MAPI AI com o estado atual dos sensores. O serviço calcula a predição e retorna o nível de risco imediatamente.

### Componentes do Ecossistema

> O antigo MAPI Edge (Python/MQTT, repositório projeto-mapi-sensores) foi descontinuado; a coleta agora é feita pela MAPI API.

Para o funcionamento completo, o MAPI é dividido em três componentes principais:
- **[MAPI AI](https://github.com/Lucas-Pavao/projeto-mapi-ai) (Este Repositório):** Microserviço de inferência para predição de riscos de inundação.
- **[MAPI API](https://github.com/Lucas-Pavao/projeto-mapi-api) (Backend):** Orquestrador central: coleta agendada das agências (APAC, ANA, Cemaden, TabuaMare, Open-Meteo, CKAN), persistência temporal em TimescaleDB e REST.
- **[MAPI Front](https://github.com/Lucas-Pavao/projeto-mapi-front) (Dashboard):** Interface geoespacial para monitoramento e alertas em tempo real.

## 🛠️ Tecnologias Escolhidas

| Categoria | Tecnologia | Justificativa Técnica |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.10+ | Padrão para ciência de dados e frameworks estáveis de ML. |
| **Framework API** | FastAPI | Alta performance assíncrona e tipagem com Pydantic. |
| **Machine Learning** | XGBoost, TensorFlow (LSTM), Scikit-learn | Gradient Boosting para dados tabulares rápidos e LSTM para memória temporal. |
| **Manipulação de Dados** | Pandas, NumPy, Geopandas | Tratamento de DataFrames e cálculos de proximidade geométrica. |
| **Containerização** | Docker | Isolamento completo das dependências nativas C++ de ML. |

## 📂 Estrutura de Pastas

```text
.
├── Dockerfile                  # Definição do container de runtime
├── main.py                     # Entrypoint (CLI para treino e serviço)
├── mapi_ai/                    # Código-fonte do módulo
│   ├── app.py                  # Definição dos endpoints FastAPI
│   ├── config.py               # Configurações e variáveis de ambiente
│   ├── data_engineering.py     # Pipeline de ETL e extração do banco
│   ├── feature_engineering.py  # Transformações e criação de features
│   ├── models.py               # Arquiteturas dos modelos (LSTM/XGB)
│   └── trainer.py              # Orquestrador do treinamento
├── models/                     # Artefatos binários dos modelos (.joblib, .h5)
└── requirements.txt            # Dependências Python
```

## 🔄 Comunicação entre Sistemas e Fluxo de Dados

- **Treinamento:** Executa o ETL em `data_engineering.py`, puxando as séries históricas do PostgreSQL da API Central (TimescaleDB).
- **Inferência:** A API faz uma requisição HTTP POST para `/v1/predict/flood`. O serviço processa o payload e devolve um JSON padronizado com o nível de risco (`LOW`, `MEDIUM`, `HIGH`).

## 🚀 Como Rodar a Aplicação

### 1. Configuração do Ambiente
Crie um arquivo `.env` baseado nas variáveis lidas em `mapi_ai/config.py`:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mapi_db
DB_USER=postgres
DB_PASS=postgres
```

### 2. Instalação Local
```bash
# Criar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# Instalar dependências
pip install -r requirements.txt
```

### 3. Execução
O projeto possui dois modos de operação via `main.py`:

**Modo de Treinamento:**
```bash
python main.py --mode train
```

**Modo de Serviço (API):**
```bash
python main.py --mode serve
```

### 4. Docker Compose (Ecossistema Completo)
Para rodar todo o ecossistema MAPI (API, AI e Front) de forma orquestrada, utilize o Docker Compose centralizado no repositório da API. Este método garante que todas as dependências de rede e banco de dados sejam configuradas automaticamente.

- **Guia de Orquestração:** [MAPI API - Docker Compose Context](https://github.com/Lucas-Pavao/projeto-mapi-api)

## 🔌 Endpoints Principais

- `GET /health`: Diagnóstico de saúde do modelo e status de carregamento.
- `POST /v1/predict/flood`: Envio dos payloads de sensores e retorno dos riscos calculados.
- `POST /v1/simulate/bucket`: simulação ilustrativa (não é previsão) do balde M1a, que chama `avaliacao/balde.py` sem cópia da fórmula (ADR 0014 do backend). Sem maré, sem escrita em disco/banco.
  - Entrada: `P_mm_h` (1 a 72 valores horários, finitos, >= 0); `point` (`slug`, `area_m2` no `snap_m` padrão, `z_curva_m` estritamente crescente, `v_curva_m3` não decrescente e do mesmo tamanho, `rua_rel_m`); `params` opcional só com `f` em {0, 0.2, 0.4, 0.6, 0.8, 1.0} e `curve_scale` em [0.7, 1.3]. `C`, `iT_mm_h`, `hcrit_mm` e `snap_m` são fixos em `Params`; enviar qualquer um deles ou valor fora da faixa devolve 422.
  - Saída: `h_mm[]`, `n_fora_da_curva`, `hmax_mm`, `t_cross_h` (índice da primeira hora com `h >= hcrit_mm`, ou `null`), `params_usados`, `model = "M1a"`, `illustrative = true`.

**Exemplo de Payload de Entrada:**
```json
{
  "station_id": "ANA-12345",
  "lat": -8.05,
  "lon": -34.90,
  "current_rainfall": 12.5,
  "rainfall_3h_accumulated": 45.0,
  "rainfall_6h_accumulated": 60.0,
  "rainfall_12h_accumulated": 80.0,
  "rainfall_24h_accumulated": 100.0,
  "tide_level": 2.1,
  "timestamp": "2026-06-05T14:30:00Z"
}
```

## 📄 Licença

Este projeto está sob a licença **MIT**.


## Projeto final IN1165 (Advanced ML & Explainable AI, CIn/UFPE)

> **DADOS SINTÉTICOS — experimento controlado.** As features são horas reais; o rótulo é gerado por fórmula conhecida
> (`reports/synthetic_ground_truth.md`), porque não há registro real de alagamento suficiente. DADOS DA REDE
> OBSERVACIONAL DO CEMADEN/MCTIC.

### Reproduzir

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/in1165_projeto_final.ipynb   # ou "Run All" no Jupyter
```

Passo a passo, sem notebook:

```bash
python -m mapi_ai.build_labels          # rótulo real: 0 positivos a 500 m -> Plano B
python -m mapi_ai.synthetic             # dataset sintético (5 000 horas reais)
python -m mapi_ai.dataset_report        # reports/dataset_summary.json
python -m mapi_ai.trainer --synthetic   # XGBoost v2 e ablações -> models/
python -m mapi_ai.baselines             # logística e árvore rasa
python -m mapi_ai.evaluate              # métricas, calibração, limiares -> models/metadata.json
python -m mapi_ai.xai.run_all           # SHAP, LIME, contrafactuais
python -m mapi_ai.xai.evaluate_explanations
pytest
```

Os dados vêm de `data/in1165/`, extraídos do banco do projeto-mapi-api (`python -m mapi_ai.sources`, só para regenerar).
Semente 42 em tudo; o gzip é gravado com `mtime=0`, então o SHA-256 do dataset é estável.

### Estrutura de `reports/`

| Arquivo | Conteúdo |
|---|---|
| `dataset_summary.json` | rótulo real, decisão do Plano B, dataset, split, SHA-256 das fontes |
| `synthetic_ground_truth.md` / `.json` | fórmula geradora e de onde vem cada escala |
| `metrics.csv`, `significance.csv` | modelos no teste; IC 95% por bootstrap pareado |
| `baseline_tree.txt`, `baselines.json` | regras da árvore e coeficientes da logística |
| `xai_cases.json` | SHAP e LIME dos casos TP/FN/FP; contrafactuais |
| `explanation_eval.csv` / `.json` | concordância, estabilidade, deleção, comparação com a fórmula |
| `figures/` | confusão, PR, ROC, calibração, SHAP, LIME, deleção, árvore |

### Resultados (teste, sintético)

| Modelo | PR-AUC | ROC-AUC | Recall (HIGH) | Precisão (HIGH) | Brier |
|---|---|---|---|---|---|
| nunca_alaga | 0.063 | 0.5 | 0.0 | 0.0 | 0.059 |
| logistica | 0.8 | 0.928 | 0.794 | 0.505 | 0.049 |
| arvore_rasa | 0.67 | 0.925 | 0.825 | 0.374 | 0.057 |
| xgb_v2 | 0.764 | 0.923 | 0.778 | 0.426 | 0.037 |
| xgb_v2_calibrado | 0.764 | 0.923 | 0.778 | 0.426 | 0.026 |
| xgb_v2_sem_local | 0.792 | 0.934 | 0.778 | 0.48 | 0.049 |
| xgb_v2_reanalise | 0.318 | 0.797 | 0.714 | 0.139 | 0.15 |

Limiares (validação): MEDIUM = 0.0159, HIGH = 0.2441.
SHAP recupera as 4 features da fórmula no top-4 e na ordem verdadeira; LIME, 3 de 4.

### Limitações conhecidas

- **Rótulo:** 0 ocorrências a 500 m e nenhuma com hora no CKAN; tudo é sintético. O "recall" de qualquer rótulo real
  futuro será recall sobre o que foi notificado.
- **Fórmula logística:** favorece a regressão logística, que vence o XGBoost aqui.
- **Fonte da chuva:** o rótulo vem da chuva medida (Cemaden), então a ablação contra a reanálise favorece o Cemaden por
  construção.
- **Correlação entre os acumulados de chuva (1 a 24 h):** SHAP intervencional e LIME podem dividir ou trocar o crédito
  entre eles.
- **Calibração:** o critério pré-fixado (ECE de validação > 0,05) não aplicou o Platt; no teste o calibrado teria
  Brier menor.
