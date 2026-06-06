# MAPI AI - Inteligência Artificial e Predição 🧠🌊

O **MAPI AI** é o componente de inteligência do ecossistema MAPI, responsável pela análise avançada de séries temporais e predição de riscos de inundação urbana em tempo real.

## 🌐 Ecossistema MAPI

Este projeto atua como um **microserviço especializado de inferência síncrona**:

```text
  [ MAPI Edge ] (Python / MQTT) 📡
        │   (Pulsações Telemétricas e Inteligência de Borda)
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

## 🔌 Endpoints Principais

- `GET /health`: Diagnóstico de saúde do modelo e status de carregamento.
- `POST /v1/predict/flood`: Envio dos payloads de sensores e retorno dos riscos calculados.

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
