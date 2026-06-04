# Projeto MAPI - Inteligência Artificial (AI) 🧠🌊

O **MAPI AI** é o componente de inteligência do ecossistema MAPI, responsável pela predição de riscos de inundação em tempo real. Utilizando modelos de Machine Learning avançados, o sistema analisa dados históricos e em tempo real de sensores, marés e meteorologia para fornecer alertas precoces e análises de probabilidade.

## 🛠️ Tecnologias Escolhidas

- **Linguagem:** Python 3.10+
- **Machine Learning:** XGBoost (Classificação), TensorFlow (LSTM para séries temporais), Scikit-learn
- **API Framework:** FastAPI (Inferência de alta performance)
- **Manipulação de Dados:** Pandas, NumPy, Geopandas
- **Banco de Dados:** PostgreSQL com SQLAlchemy
- **Gerenciamento de Modelos:** Joblib

## ✨ Funcionalidades / Features

- 🧠 **Predição em Tempo Real:** Endpoints de inferência de baixa latência para risco de inundação.
- 🔄 **Pipeline de Treinamento:** Sistema automatizado para retreinamento de modelos com novos dados históricos.
- 📈 **Feature Engineering:** Criação de variáveis temporais, cíclicas e geográficas para aumentar a precisão.
- 🗺️ **Análise Espacial:** Integração com Geopandas para processamento de dados georreferenciados.

## 📂 Estrutura de Pastas

```text
projeto-mapi-ai/
├── mapi_ai/                # Core da aplicação
│   ├── app.py              # API FastAPI para inferência
│   ├── trainer.py          # Pipeline de treinamento automatizado
│   ├── data_engineering.py # ETL e carregamento de dados
│   ├── feature_engineering.py # Transformação e criação de features
│   ├── models.py           # Definição das arquiteturas de modelos
│   └── config.py           # Configurações e variáveis de ambiente
├── models/                 # Modelos treinados (.joblib, .h5)
├── main.py                 # Ponto de entrada (CLI para treino ou serviço)
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação principal
```

## 📋 Pré-requisitos

- Python 3.10 ou superior.
- Banco de Dados PostgreSQL configurado.
- Pip instalado.

## 🚀 Como instalar e rodar

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/Lucas-Pavao/projeto-mapi-ai.git
   cd projeto-mapi-ai
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure as variáveis de ambiente:**
   Crie um arquivo `.env` com as credenciais do banco:
   ```env
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=mapi_db
   DB_USER=seu_usuario
   DB_PASS=sua_senha
   ```

5. **Treine o modelo (opcional se já houver um):**
   ```bash
   python main.py --mode train
   ```

6. **Inicie a API de Inferência:**
   ```bash
   python main.py --mode serve
   ```

## 🤝 Como contribuir

1. Faça um **Fork** do projeto.
2. Crie uma **Branch** para sua modificação (`git checkout -b feature/melhoria-modelo`).
3. Faça o **Commit** de suas alterações (`git commit -m 'Improve: ajuste hiperparâmetros'`).
4. Faça o **Push** para a sua Branch (`git push origin feature/melhoria-modelo`).
5. Abra um **Pull Request**.

## 📄 Licença

Este projeto está sob a licença **MIT**.
