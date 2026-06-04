# Projeto MAPI - Inteligência Artificial (AI) 🧠🌊

O **MAPI AI** é o componente de inteligência do ecossistema MAPI, responsável pela análise de dados e predição de riscos de inundação em tempo real. Este repositório contém o pipeline de Machine Learning, desde a engenharia de dados até a disponibilização de modelos via API de alta performance.

## 📖 Descrição do Projeto

O objetivo principal deste projeto é transformar dados brutos de sensores de nível de rio, pluviômetros (chuva) e tábuas de maré em insights acionáveis. Através de modelos preditivos, o sistema estima a probabilidade de transbordamento e inundações urbanas, permitindo que autoridades e cidadãos recebam alertas antecipados.

## 🛠️ Tecnologias Escolhidas

- **Linguagem:** [Python 3.10+](https://www.python.org/)
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) para inferência de baixa latência.
- **Machine Learning:** 
  - **XGBoost:** Para classificação robusta de risco.
  - **TensorFlow (LSTM):** Para análise de séries temporais e tendências.
  - **Scikit-learn:** Para pré-processamento e métricas de avaliação.
- **Manipulação de Dados:** Pandas, NumPy e Geopandas para análise espacial.
- **Banco de Dados:** PostgreSQL com SQLAlchemy para persistência e ETL.
- **Servidor:** Uvicorn.
- **Containerização:** Docker.

## 📂 Estrutura de Pastas

```text
projeto-mapi-ai/
├── mapi_ai/                # Core da aplicação
│   ├── app.py              # API FastAPI para inferência (Servidor de Predição)
│   ├── trainer.py          # Script de treinamento dos modelos
│   ├── data_engineering.py # ETL e conexão com banco de dados
│   ├── feature_engineering.py # Criação de variáveis preditivas
│   ├── models.py           # Definição das classes/arquiteturas de ML
│   └── config.py           # Variáveis de ambiente e hiperparâmetros
├── models/                 # Modelos serializados (.joblib, .h5)
├── main.py                 # CLI para gerenciar treino e execução
├── requirements.txt        # Dependências Python
├── Dockerfile              # Configuração de containerização
└── README.md               # Documentação principal
```

## 🔄 Comunicação entre Sistemas

O ecossistema MAPI é composto por dois serviços principais que trabalham de forma síncrona:

1.  **MAPI API (Main Backend):** Gerencia usuários, dispositivos, notificações e armazena os dados históricos no PostgreSQL.
2.  **MAPI AI (Este serviço):** Atua como um microserviço especializado em processamento numérico e predição.

### Fluxo de Dados:
- **Treinamento:** O `MAPI AI` conecta-se diretamente ao banco de dados PostgreSQL mantido pelo `MAPI API` para ler dados históricos e realizar o treinamento do modelo.
- **Inferência (Tempo Real):** Quando o `MAPI API` recebe novos dados de sensores, ele faz uma requisição **HTTP POST** para o endpoint `/v1/predict/flood` deste serviço, enviando o estado atual dos sensores.
- **Resposta:** Este serviço processa os dados, aplica o modelo de ML e retorna um JSON contendo a probabilidade (0 a 1) e o nível de risco (LOW, MEDIUM, HIGH).

## 🚀 Como Rodar a Aplicação

### 1. Instalação Local (Desenvolvimento)

**Pré-requisitos:** Python 3.10+, PostgreSQL.

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/Lucas-Pavao/projeto-mapi-ai.git
    cd projeto-mapi-ai
    ```

2.  **Ambiente Virtual:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate
    ```

3.  **Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuração:**
    Crie um arquivo `.env` na raiz:
    ```env
    DATABASE_URL=postgresql://user:pass@localhost:5432/mapi_db
    MODEL_PATH=models/flood_model.joblib
    ```

5.  **Execução:**
    ```bash
    # Para rodar a API de predição
    python main.py --mode serve

    # Para treinar o modelo com dados do banco
    python main.py --mode train
    ```

### 2. Rodando via Docker (Produção)

Se preferir rodar em containers:

1.  **Build da imagem:**
    ```bash
    docker build -t mapi-ai .
    ```

2.  **Execução do container:**
    ```bash
    docker run -p 8000:8000 --env-file .env mapi-ai
    ```

## 🔌 Endpoints Principais

- `GET /health`: Verifica a saúde do serviço e se o modelo está carregado.
- `POST /v1/predict/flood`: Recebe dados de sensores e retorna a predição de risco.

---
Desenvolvido por **Lucas Pavão** como parte do ecossistema MAPI.
