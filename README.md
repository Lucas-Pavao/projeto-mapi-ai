# MAPI AI - Sistema de Predição de Inundações

O **MAPI AI** é uma solução avançada de inteligência artificial voltada para a predição de riscos de inundação em tempo real. O sistema integra dados meteorológicos, níveis de maré e sensores fluviais para fornecer alertas precoces e análises de probabilidade de alagamento, auxiliando na gestão de riscos e resposta a desastres.

## 🚀 Tecnologias Escolhidas

O projeto utiliza uma stack moderna voltada para ciência de dados e performance:

-   **Linguagem:** Python 3.10+
-   **Manipulação de Dados:** Pandas, NumPy, Geopandas
-   **Machine Learning:** XGBoost (Classificação de Risco), Scikit-learn, TensorFlow (Modelagem de Séries Temporais)
-   **API Framework:** FastAPI (Inferência de alta performance)
-   **Servidor Web:** Uvicorn
-   **Banco de Dados:** PostgreSQL com SQLAlchemy
-   **Gerenciamento de Modelos:** Joblib
-   **Geolocalização:** Shapely

## 📁 Estrutura do Projeto

A arquitetura do projeto é modular, facilitando a manutenção e escalabilidade:

```text
projeto-mapi-ai/
├── mapi_ai/                # Core da aplicação
│   ├── app.py              # API FastAPI para inferência em tempo real
│   ├── trainer.py          # Pipeline automatizado de treinamento
│   ├── data_engineering.py # ETL e carregamento de dados do banco
│   ├── feature_engineering.py # Criação de features cíclicas e temporais
│   ├── models.py           # Definição das arquiteturas de modelos (XGBoost/LSTM)
│   └── config.py           # Configurações globais e variáveis de ambiente
├── models/                 # Armazenamento de modelos treinados (.joblib, .h5)
├── main.py                 # Ponto de entrada principal do sistema
├── requirements.txt        # Dependências do projeto
└── .env                    # Configurações de ambiente (Banco de dados, etc.)
```

## 🛠️ Como Rodar a Aplicação

### 1. Preparação do Ambiente
Certifique-se de ter o Python instalado. Recomendamos o uso de um ambiente virtual:

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar no Linux/macOS
source .venv/bin/activate

# Ativar no Windows
.venv\Scripts\activate
```

### 2. Instalação de Dependências
Instale todas as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

### 3. Configuração
Crie um arquivo `.env` na raiz do projeto com as credenciais do seu banco de dados PostgreSQL:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mapi_db
DB_USER=seu_usuario
DB_PASS=sua_senha
```

### 4. Treinamento do Modelo
Antes de rodar a API, é necessário treinar o modelo com os dados históricos:

```bash
python main.py --mode train
```

### 5. Execução da API de Inferência
Para iniciar o servidor e disponibilizar os endpoints de predição:

```bash
python main.py --mode serve
```
A API estará disponível em `http://localhost:8000`. Você pode acessar a documentação interativa (Swagger UI) em `http://localhost:8000/docs`.

---
Desenvolvido como parte do ecossistema MAPI.
