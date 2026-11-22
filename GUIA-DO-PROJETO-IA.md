# Guia do MAPI AI — explicado do zero

> Documento de estudo. Explica o projeto `projeto-mapi-ai` para quem **nunca estudou IA**.
> Tudo aqui foi tirado do código deste repositório (`mapi_ai/`, `avaliacao/`) e dos documentos de
> modelagem do repositório da API (`projeto-mapi-api/docs/`). Quando algo é opinião ou ponto de atenção, está marcado.

> **O que o serviço serve de fato (conferido no código em 21/11/2026).**
> - `POST /v1/predict/flood` foi **removida em 21/11** (commit `864ba40`, decisão de 20/11). Ela servia a **`logreg_v3`** (regressão logística, horizonte de 6 h), treinada com **rótulo sintético** (`NOT_VALIDATED`); o `xgb_v2` só respondia se a v3 não carregasse, e a **LSTM nunca era alcançada**. Os artefatos e relatórios ficam no git e em `reports/`.
> - `POST /v1/forecast/rain` serve o **`rain_v1`**: previsão da chuva **medida** (quantis em 1, 3 e 6 h), validada em `reports/v4/rain/RELATORIO.md`.
> - `POST /v1/flood/forecast` (novo, `mapi_ai/flood/`): **alerta de alagamento por regressão + física**. Chuva prevista pelo `rain_v1` → balde M1a a 10 min → `ALERTA` se a lâmina provável (q50) chega a 15 cm, `ATENCAO` se só a pessimista (q95), `SEM_ALERTA` ou `SEM_DADO` com motivo. Nenhum modelo treinado com rótulo de alagamento. **Experimental**: na validação (Etapa 2) o alerta nunca disparou em 5 anos; ver seção 4.3 e `reports/RELATORIO-REGRESSAO.md`.
> - `POST /v1/bucket/point10` e `POST /v1/simulate/bucket`: balde M1a (físico, sem treino).
> - `GET /health`: `rain_model_version`, `bucket_version`, `flood_version` e `code_version` (o commit: variável do build ou, rodando local, `git rev-parse HEAD`).
>
> As seções abaixo que descrevem XGBoost e LSTM como "modelo do serviço" contam a história do projeto e ficam para a dissertação.

---

## 0. Respostas rápidas (o que você perguntou)

| Pergunta | Resposta curta |
|---|---|
| **Qual dataset é usado?** | São **dois mundos**: (1) o serviço em produção treina com a tabela `flood_scenario_labels` do banco PostgreSQL/TimescaleDB da MAPI API (ou um CSV, ou, se nada existir, **dados sintéticos inventados** pelo próprio código); (2) a avaliação científica (pasta `avaliacao/`) usa **chuva horária do Open-Meteo (reanálise)**, **terreno LiDAR do PE3D**, a **normal climatológica do INMET** e os **registros de alagamento da Defesa Civil do Recife (CKAN)** — só **18 dias** com alagamento entre 2021 e 2024. |
| **Qual modelo "tradicional" é usado? KNN? Árvore?** | **Nenhum KNN.** O modelo tradicional **servido hoje** é a **regressão logística v3** (rótulo sintético; sai do serviço por decisão de 20/11). Antes dela, era o **XGBoost**, que é um **conjunto de árvores de decisão** (técnica chamada *gradient boosting*). Além dele há uma **rede neural LSTM** (prevê nível do rio) e, na avaliação científica, um **modelo físico chamado "balde"** (não é aprendizado de máquina, é física/hidrologia). |
| **Por que esse e não outro?** | XGBoost é o padrão de mercado para dados em tabela (linhas × colunas), lida bem com valores faltando e com relações não lineares (ex.: "chuva forte **e** maré alta"). Mas o projeto descobriu que **há poucos exemplos de alagamento (18 dias)** para treinar qualquer modelo de ML com confiança, por isso a avaliação oficial passou a usar o modelo físico. Detalhes na seção 5. |
| **O modelo atual está validado?** | **Não.** A própria API responde `"model_validation_status": "NOT_VALIDATED"` (`mapi_ai/app.py:211`). |

---

## 1. IA do zero: os conceitos que você precisa

Pense em IA (mais precisamente **aprendizado de máquina**, *machine learning*, ML) como **ensinar por exemplos** em vez
de escrever regras à mão.

### 1.1 A analogia da receita

- **Programação tradicional:** você escreve a regra. "Se chover mais de 50 mm em uma hora, avise alagamento."
- **Aprendizado de máquina:** você mostra milhares de exemplos do passado ("neste dia choveu X, a maré era Y, e
  **alagou**"; "neste outro choveu Z e **não alagou**") e o algoritmo **descobre sozinho** a regra.

### 1.2 Vocabulário básico

| Termo | O que é | Exemplo no MAPI |
|---|---|---|
| **Dataset (conjunto de dados)** | A tabela de exemplos do passado | Tabela `flood_scenario_labels` |
| **Amostra / linha** | Um exemplo | "Ponto CIN_UFPE, 14h de 06/02/2023" |
| **Feature (variável de entrada)** | Uma coluna que o modelo usa para decidir | chuva acumulada 6h, nível da maré, altitude |
| **Rótulo / target (alvo)** | A resposta certa daquele exemplo | `is_flooded` = alagou (1) ou não (0) |
| **Treino** | Fase em que o modelo "estuda" os exemplos | `python main.py --mode train` |
| **Inferência / predição** | Usar o modelo treinado em um caso novo | `POST /v1/predict/flood` |
| **Modelo** | O "cérebro" resultante do treino, salvo em arquivo | `models/flood_classifier.joblib` |
| **Classificação** | Prever uma **categoria** (sim/não, baixo/médio/alto) | Vai alagar? |
| **Regressão** | Prever um **número** | Qual será o nível do rio em metros? |
| **Treino × teste** | Separa-se parte dos dados para **provar** que o modelo funciona em dados que ele nunca viu | 80% treino, 20% teste (`trainer.py:774`) |
| **Overfitting (decoreba)** | O modelo decora os exemplos em vez de aprender a regra; vai bem no treino e mal no mundo real | Grande risco com poucos exemplos |
| **Baseline** | Um "modelo burro" de referência. Se a IA não vence o baseline, ela não serve | "Nunca alaga", "alaga se chover > 50 mm/h" |
| **AUC** | Nota de 0,5 (chute) a 1,0 (perfeito) que mede se o modelo dá nota mais alta aos dias que alagaram do que aos que não alagaram | 0,891 na avaliação F1 |
| **Intervalo de confiança (IC)** | Faixa de incerteza de um número. Com poucos dados, a faixa é larga | AUC entre 0,75 e 0,97 |

### 1.3 Por que "poucos exemplos" é o grande problema

Um modelo de ML é como um aluno: aprende vendo exemplos. Se você mostrar só 18 provas resolvidas, ele pode decorar
as 18 e não aprender a matéria. E, pior, você não tem provas sobrando para **testar** se ele aprendeu. Esse é o
problema central do MAPI (seção 6).

---

## 2. O que é o MAPI e onde este repositório entra

O MAPI é um sistema de **monitoramento e previsão de alagamentos urbanos no Recife**. São três peças:

```text
 Agências públicas (APAC, ANA, Cemaden, Tábua de Maré, Open-Meteo, CKAN Recife)
        │  coleta automática
        ▼
 MAPI API  (Java/Spring Boot + banco TimescaleDB)   ← guarda tudo
        │  manda o "estado atual" via HTTP POST
        ▼
 MAPI AI   (Python/FastAPI)  ← ESTE REPOSITÓRIO: calcula a probabilidade de alagar
        │  devolve probabilidade + risco (LOW / MEDIUM / HIGH)
        ▼
 MAPI Front (React + mapa)   ← mostra no mapa para o usuário
```

O MAPI AI faz **duas coisas** em momentos diferentes:

1. **Treino (de vez em quando, "em lote"):** lê o histórico do banco e gera os arquivos de modelo em `models/`.
2. **Serviço (o tempo todo):** fica no ar como uma API; a MAPI API pergunta "com esta chuva e esta maré, vai alagar?"
   e ele responde na hora.

E existe uma terceira parte, separada: a **avaliação científica** (`avaliacao/`), que testa se a abordagem realmente
funciona com dados reais. Ela é a parte "da dissertação".

---

## 3. Os datasets (dados) em detalhe

### 3.1 Dataset do serviço (treino do XGBoost e da LSTM)

**De onde vem** (`mapi_ai/trainer.py:731` — função `load_dataset`), nesta ordem de tentativa:

1. Um **CSV** passado na linha de comando (`--csv arquivo.csv`);
2. Senão, a tabela **`flood_scenario_labels`** do PostgreSQL da MAPI API;
3. Senão (banco vazio ou fora do ar), **200 linhas sintéticas** geradas por `generate_dummy_scenario_data`
   (`trainer.py:676`).

> ⚠️ **Atenção ao item 3:** os dados sintéticos são **inventados** com números aleatórios e uma regra fixa
> ("alaga se `chuva×0,15 + chuva6h×0,05 + 0,3 se maré>2m + 0,4 se rio>3m` passar de 0,6"). Um modelo treinado nisso
> só aprende essa regra inventada. Serve para testar se o código roda, **não** para prever alagamento de verdade.
> Não dá para saber, só pelo arquivo, com qual dos três os modelos em `models/` foram gerados.

**Colunas (features) que o modelo recebe** — são 28, definidas em `feature_engineering.py:304`:

| Grupo | Features | De onde vem |
|---|---|---|
| Localização | `latitude`, `longitude` | Payload |
| Chuva | `current_rainfall`, `rainfall_3h/6h/12h/24h_accumulated` | Payload (sensores) |
| Água | `tide_level` (maré), `river_level` (rio), `is_high_tide` (maré > 2 m → 1) | Payload; `is_high_tide` é calculada |
| Terreno | `altitude`, `distance_to_channel` (distância ao canal) | Tabela `flood_points` (ponto de alagamento mais próximo) |
| Proximidade | `distance_to_flood_point`, `distance_to_rain_sensor`, `distance_to_river_sensor` | Calculadas com a fórmula de Haversine (distância na esfera da Terra) |
| Clima | `wind_speed`, `temperature`, `apparent_temperature`, `humidity`, `pressure`, `solar_radiation` | Payload (Open-Meteo) |
| Mar | `wave_height`, `wave_period`, `wave_direction` | Payload (Open-Meteo Marine) |
| Tempo | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` | Calculadas do `timestamp` |

**Rótulos (respostas certas):**
- XGBoost aprende a coluna `is_flooded` (alagou ou não).
- LSTM aprende a coluna `river_level` (nível do rio).

> 💡 **Por que seno e cosseno da hora?** Se você passar a hora como número (0 a 23), o modelo acha que 23h e 0h estão
> "longe" (23 de distância), quando na verdade são vizinhas. Transformar em um ponto num círculo (seno/cosseno) resolve.
> O mesmo vale para o mês (dezembro e janeiro são vizinhos).

### 3.2 Datasets da avaliação científica (`avaliacao/`)

Aqui os dados são reais e cada arquivo tem um hash (impressão digital SHA-256) para garantir que ninguém o alterou.

| Dado | Fonte | Período / tamanho | Para que serve |
|---|---|---|---|
| **Chuva horária** | Open-Meteo *archive* (reanálise: estimativa de chuva feita por modelo meteorológico, não um pluviômetro) | 2021-04-01 a 2024-12-31, 1 valor por hora, em 4 pontos | Entrada do modelo |
| **Terreno** | PE3D (LiDAR do governo de PE): área que drena para o ponto, curva "altura × volume" da bacia, cota da rua | 4 pontos | Entrada do modelo físico |
| **Normal climatológica** | INMET, estação 82900 Recife (Curado), média 1991–2020 (`dados/inmet_normal_82900_prec.json`) | 12 valores (um por mês) | Baseline B2 ("meses chuvosos alagam mais") |
| **Rótulos** | Ocorrências de alagamento da Defesa Civil do Recife (portal CKAN), tabela `flood_occurrence` | **18 dias** com registro (`dados/rotulos_apendice_b.csv`): 14 em 2023–2024 e 4 antes | **Só** para dar a nota no final |

Os 4 pontos estudados: `AV_RECIFE_IBURA`, `CIN_UFPE`, `AGAMENON_DERBY`, `MASCARENHAS_IMBIRIBEIRA`
(Jaboatão ficou de fora: outro município e sem terreno do PE3D).

> 💡 **"Etapa cega":** o código que calcula as notas dos modelos (`gerar_escores.py`) é **proibido** de ler qualquer
> rótulo. A função `_recusa_rotulos` (`entradas.py`) aborta se encontrar uma coluna com nome tipo `isflooded` ou
> `occurrence`. Só depois, numa segunda etapa (`avaliar_h1_h2.py`), as notas são comparadas com os 18 dias. Isso evita
> "colar na prova".

---

## 4. Os modelos usados

### 4.1 XGBoost — o classificador principal do serviço

- **Arquivo:** `mapi_ai/models.py:626` · **Salvo em:** `models/flood_classifier.joblib`
- **Tarefa:** classificação → "vai alagar?" (devolve probabilidade de 0 a 1)
- **Configuração:** 100 árvores, profundidade máxima 6, taxa de aprendizado 0,1.

**Como funciona, sem matemática:**
1. Uma **árvore de decisão** é um fluxograma de perguntas:
   "Chuva 6h > 40 mm?" → sim → "Maré > 2 m?" → sim → "alto risco".
2. Uma árvore sozinha erra bastante. O **XGBoost** cria **100 árvores em sequência**: a 2ª tenta corrigir os erros da
   1ª, a 3ª corrige o que sobrou, e assim por diante (*boosting*). A resposta final é a soma de todas.
3. É como pedir opinião a 100 especialistas em que cada um se concentra nos casos que os anteriores erraram.

**Como vira risco** (`app.py:199`):
- probabilidade > 0,8 → `HIGH`
- probabilidade > 0,5 → `MEDIUM`
- senão → `LOW`

### 4.2 LSTM — a rede neural para o nível do rio

- **Arquivo:** `mapi_ai/models.py:638` · **Salvo em:** `models/water_level_lstm.h5`
- **Tarefa:** regressão → "qual o nível do rio?" (número em metros)
- **Estrutura:** 2 camadas LSTM (64 e 32 neurônios) + camadas densas, olhando uma **janela de 12 passos** de tempo.

**Como funciona, sem matemática:** uma rede neural "comum" olha uma foto. Uma **LSTM** (*Long Short-Term Memory*,
memória de curto e longo prazo) olha um **filme**: ela lê as últimas 12 leituras em ordem e mantém uma "memória" do
que viu. Faz sentido para rio, porque o nível de agora depende do que choveu nas horas anteriores.

É **opcional**: se o arquivo não existir ou der erro, a API responde sem o campo `predicted_river_level`.

### 4.3 StandardScaler — não é modelo, é "padronizador"

- **Salvo em:** `models/scaler.joblib`
- Coloca todas as colunas na mesma escala (média 0, desvio 1). Sem isso, pressão (≈1010) "gritaria" mais alto que
  chuva (≈5) só por ter números maiores. Tecnicamente árvores não precisam disso, mas a LSTM precisa.

### 4.4 Modelo físico "balde" (M1) — o modelo da avaliação científica

- **Arquivo:** `avaliacao/balde.py`
- **Não é aprendizado de máquina.** É uma conta de física (balanço de massa), com parâmetros tirados de livros e
  normas, **sem olhar os dias de alagamento**.

**A analogia:** cada ponto da cidade é um **balde** com um **ralo** no fundo.
- A chuva enche o balde (quanto mais área drena para ali, mais água entra).
- O ralo (a galeria de drenagem) esvazia numa velocidade máxima fixa (calculada pela chuva de projeto de 60,1 mm/h).
- Se entra mais água do que o ralo dá conta, o nível sobe. Usando o terreno (curva altura × volume), calcula-se a
  altura da água. Se passar da cota da rua, há **lâmina d'água na rua**.
- Alerta se a lâmina passar de **15 cm**.

### 4.5 Baselines (os "modelos burros" de comparação)

| Nome | Regra | Para que |
|---|---|---|
| **B0** | Nunca alaga | Piso mínimo |
| **B2** | Nota = chuva média histórica do mês (INMET) | "Mês chuvoso alaga mais" |
| **B3** | Alerta se a maior chuva horária do dia ≥ 50,3 mm/h (curva IDF de Recife) | Regra de chuva simples |

Se o balde não ganhar do B2 e do B3, ele não acrescenta nada.

---

### 4.3 Alerta de alagamento por regressão (novo, 21/11/2026)

**Por que os rótulos saíram do treino.** O classificador servido até 21/11 (`logreg_v3`) aprendia a coluna
`is_flooded` do dataset IN1165. Essa coluna é **sintética**: os autores sortearam alagamento com uma fórmula escrita à mão
sobre a chuva medida (`reports/synthetic_ground_truth.md`). AUC 0,92 media quanto o modelo recuperava essa fórmula, não
alagamento. O único dado real (Defesa Civil) tem 18 dias, sem hora, e nenhum registro a menos de 500 m dos pontos.
Com isso não se treina nada. A decisão de 20/11: prever o que tem dado medido (a chuva) e passar pela física (o balde).

**Como funciona (`mapi_ai/flood/`, `POST /v1/flood/forecast`):**

1. o `rain_v1` prevê quanto vai chover em 1, 3 e 6 h, em dois cenários: provável (q50) e pessimista (q95);
2. o balde M1a a 10 min parte da água das últimas 24 h medidas e recebe a chuva prevista, espalhada por igual no prazo;
3. regra fixa: **ALERTA** se a lâmina provável chega a 15 cm; **ATENÇÃO** se só a pessimista chega; **SEM ALERTA**
   caso contrário; **SEM DADO** com motivo quando falta entrada (nunca vira "sem alerta").

**Explicação de cada alerta:** o que mais pesou na chuva prevista (SHAP local, em mm), quantos mm a mais levariam a 15 cm
(contrafactual calculado no balde), quanto a lâmina muda com os parâmetros sem fonte (sensibilidade) e a conta do balde
a cada 10 min (entrada, saída, volume).

**O que a validação mostrou** (`reports/RELATORIO-REGRESSAO.md`): em 5 anos o alerta **nunca disparou**. A chuva prevista,
espalhada, chega no máximo a 60 mm/h, e a galeria do modelo escoa 125 mm/h (chuva de projeto de 10 min). O balde não
enche. A chuva medida das últimas 6 h ≥ 25 mm pegou os 12 dias da Defesa Civil perto dos pontos, com 3,8 falsos por
ponto-mês. Por isso o alerta é **experimental**: falta uma capacidade de galeria com fonte e uma forma de distribuir a
chuva prevista no tempo, decididas sem olhar os 18 dias, e uma validação nova.

### 4.4 "Dia de risco" com a previsão numérica (avaliado em 22/11/2026, não adotado)

Pergunta: o ECMWF, o ICON ou o modelo atual do Open-Meteo, vistos às 06h, dizem se haverá evento FORTE no ponto nas
próximas 6, 12 ou 24 h melhor que a climatologia? O critério foi escrito e commitado antes de baixar os dados
(`mapi_ai/dayrisk/config.py`). O alvo é a chuva medida, sem rótulo de alagamento. Há trava de vazamento: só vale a
rodada publicada até a emissão (`h − 24N + 8 h ≤ t`).

Resultado: **não passou.** Em H = 12 h, o melhor na validação (ICON, BSS 0,071) teve IC 95% de −0,081 a 0,260. Pela
regra, fica a climatologia. O teste tinha 28 eventos FORTE em 14 dias, então os intervalos são largos. O horizonte do
dia fica só com o INMET na tela. Números: `reports/day_risk/RELATORIO.md`. Reproduzir:
`python -m mapi_ai.dayrisk.fetch --parse && python -m mapi_ai.dayrisk.evaluate`. Não há serviço
`/v1/forecast/day-risk`, porque ele só existiria se o critério passasse.

## 5. Comparação de algoritmos: por que usar, quando usar, por que (não) estamos usando

Contexto do MAPI que pesa na escolha: **dados em tabela**, **muitas colunas (28)**, **relações que se combinam**
(chuva + maré), **valores faltando às vezes**, **pouquíssimos exemplos positivos** e **resposta rápida na API**.

| Algoritmo | Como pensa (analogia) | Bom quando… | Ruim quando… | No MAPI |
|---|---|---|---|---|
| **KNN** (K vizinhos mais próximos) | "Me diga com quem andas…": acha os K dias do passado mais parecidos com hoje e vota | Poucas colunas, dados bem escalados, quer algo simples de explicar ("parecido com 06/02/2023") | Muitas colunas (a noção de "parecido" se perde), colunas irrelevantes atrapalham, precisa guardar todo o histórico e comparar a cada previsão (lento), sensível a escala, **com 18 positivos os vizinhos quase sempre serão "não alagou"** | ❌ **Não usado.** Funcionaria mal com 28 features e classes tão desbalanceadas. Poderia servir como baseline didático |
| **Árvore de decisão** (uma só) | Um fluxograma de perguntas sim/não | Precisa **explicar** a regra para um gestor; dados pequenos; quer ver a lógica | Instável (muda muito com poucos dados novos), decora fácil, precisão menor | ❌ **Não usado sozinho**, mas é a **peça básica do XGBoost**. Seria uma boa escolha se o objetivo fosse explicabilidade |
| **Random Forest** (floresta aleatória) | Muitas árvores independentes, cada uma vendo um pedaço dos dados, e votam | Bom resultado "de primeira", poucos ajustes, robusto a ruído | Um pouco menos preciso que boosting em geral; modelo pesado | ❌ Não usado. Alternativa honesta ao XGBoost, mais difícil de errar no ajuste. Faria sentido testar como comparação |
| **XGBoost** (gradient boosting) | Árvores em sequência, cada uma corrige a anterior | **Dados em tabela** (é o campeão nesse tipo), relações não lineares, lida nativamente com valor faltando, rápido para prever | Muitos parâmetros para ajustar; com poucos exemplos decora fácil; menos explicável | ✅ **Usado** no serviço (`flood_classifier.joblib`). Escolha correta **se** houvesse muitos exemplos rotulados |
| **Regressão logística** | Soma ponderada das colunas → probabilidade ("cada mm de chuva soma tanto") | Poucos dados, quer coeficientes interpretáveis, baseline sólido | Não capta "chuva **e** maré" sem você criar a combinação à mão | ❌ Não usado. Com 18 positivos, seria **mais apropriada** que XGBoost como modelo de ML (menos parâmetros = menos decoreba) |
| **SVM** | Traça a "melhor fronteira" separando os dois grupos | Dados médios, fronteira complexa | Probabilidades ruins sem calibração, lento com muitos dados, difícil explicar | ❌ Não usado. Sem vantagem clara aqui |
| **Naive Bayes** | Probabilidades por coluna, supondo que são independentes | Texto, classificação rápida | Chuva 3h, 6h, 12h e 24h são **fortemente dependentes** → quebra a suposição | ❌ Não usado. Inadequado para estas features |
| **LSTM** (rede neural recorrente) | Lê uma sequência no tempo com memória | Séries temporais longas, **muitos dados** | Precisa de muitos dados, demora para treinar, difícil explicar | ✅ **Usado** para nível do rio. Questionável com pouco histórico |
| **Modelo físico (balde)** | Equações de água entrando e saindo | **Poucos ou nenhum rótulo**; quer entender o mecanismo; parâmetros vêm de normas | Depende de bons parâmetros (galerias, terreno); simplifica a realidade | ✅ **Usado na avaliação científica** — justamente porque não precisa de rótulos para ser construído |

### 5.1 Regra prática: qual usar em que momento

```text
Quantos exemplos ROTULADOS (alagou/não alagou) eu tenho?

  Quase nenhum (dezenas)  → Modelo físico + baselines simples. Rótulos só para TESTAR.   ← MAPI hoje
  Poucos (centenas)       → Regressão logística ou árvore rasa; validação cuidadosa.
  Muitos (milhares+)      → XGBoost / Random Forest para tabela; LSTM se a ordem no tempo importar.

Preciso explicar a decisão para alguém (Defesa Civil)?
  Sim → árvore de decisão, regressão logística, modelo físico.
  Não → XGBoost, redes neurais.

Os dados são "parecidos com quem"? (poucas colunas, bem escaladas, e posso guardar tudo)
  Sim → KNN pode ser uma referência simples.
```

### 5.2 Onde o ML ainda entra no plano

Pelos documentos de modelagem (`projeto-mapi-api/docs/MODELAGEM-IA.md` §0), o ML deixou de ser "o modelo que decide se
alaga" e passou a ser planejado para:
- **Emulador:** uma IA treinada para imitar rapidamente um modelo físico mais pesado (rede de Petri) — aí os "rótulos"
  são gerados por simulação e existem aos milhares. Algoritmos citados: gradient boosting e MLP. **Trabalho futuro**,
  fora da avaliação atual.
- **Correção da chuva de entrada** (ajustar a reanálise para parecer com pluviômetro).

---

## 6. Ponta a ponta: o caminho dos dados

### 6.1 Fluxo de treino (`python main.py --mode train`)

```text
main.py
  └─ trainer.train_pipeline()
       1. load_dataset()          → CSV, ou banco (flood_scenario_labels), ou dados sintéticos
       2. prepare_scenario_features(is_training=True)
              - renomeia colunas (lat→latitude…)
              - preenche vazios com 0.0   (só no treino)
              - is_high_tide = maré > 2 m
              - seno/cosseno de hora e mês
              - busca no banco os pontos de alagamento e sensores e calcula distâncias/altitude
              - devolve as 28 colunas na ordem fixa
       3. StandardScaler           → padroniza as colunas
       4. divide 80% treino / 20% teste (em ordem de tempo)
       5a. XGBoost.fit(...)        → imprime relatório (precisão, recall) → salva .joblib
       5b. LSTM: monta janelas de 12 passos → treina 15 épocas → imprime MSE/MAE → salva .h5
```

### 6.2 Fluxo de previsão do artefato legado (`python main.py --mode serve`, depois `POST /v1/predict/flood`)

> Histórico: este é o caminho do artefato legado, que só roda sem a v3 e sem o v2. Hoje a rota serve a `logreg_v3` e não chega na LSTM.

```text
1. Ao subir, a API carrega os modelos e o cache de pontos/sensores do banco.
2. A MAPI API manda um JSON: estação, lat/lon, chuvas, maré, rio, clima, ondas, sensores próximos.
3. prepare_features_for_inference()
      - se faltar QUALQUER campo obrigatório → responde 422 com a lista "missing_features"
        (decisão ADR 0007: não inventar zeros na previsão)
      - senão monta as mesmas 28 colunas do treino
4. scaler.transform → XGBoost.predict_proba → probabilidade de alagar
5. Converte em LOW / MEDIUM / HIGH
6. (Opcional) LSTM: busca as últimas 11 linhas históricas perto daquela coordenada,
   junta com a atual (12 passos) e prevê o nível do rio
7. Devolve:
   {
     "flood_probability": 0.73,
     "risk_level": "MEDIUM",
     "model_version": "unversioned",
     "model_validation_status": "NOT_VALIDATED",
     "station_id": "...", "timestamp": "...",
     "predicted_river_level": 2.4      ← só se a LSTM rodou
   }
```

`GET /health` diz se XGBoost e LSTM estão carregados.

### 6.3 Fluxo da avaliação científica (`avaliacao/reproduzir.sh congelado`)

```text
Etapa 1 — CEGA (gerar_escores.py)
  entradas: chuva Open-Meteo + terreno + normal INMET   (proibido ler rótulos)
  para cada dia de 2021-04 a 2024-12:
     - roda o balde (M1) em 4 pontos, com a versão central e ~25 variações de parâmetros
     - calcula baselines B0, B2, B3
  saída: escores_cegos.csv

Etapa 2 — JUNTA COM A VERDADE (avaliar_h1_h2.py)
  lê os 18 dias com alagamento
  período avaliado: 2023-2024 → 14 dias positivos × 689 dias negativos
  calcula AUC, intervalo de confiança (bootstrap 10 000 vezes, semente fixa 20261011)
  testa:
     H1: o balde é melhor que a climatologia (B2)?
     H2: o balde é melhor que a regra de chuva (B3)?
  saída: resultado_etapa2.json
```

Tudo com semente fixa e hashes conferidos: qualquer pessoa reroda e obtém **exatamente** o mesmo número.

---

## 7. O que a avaliação descobriu (resultados honestos)

Fonte: `../avaliacao-f1/resultado_etapa2.json` e `projeto-mapi-api/docs/INTERPRETACAO-F1.md`.

| Modelo | AUC | IC 95% |
|---|---|---|
| B2 (climatologia do mês) | 0,777 | 0,641 – 0,890 |
| B3 (regra de chuva) | 0,891 | 0,747 – 0,970 |
| M1 (balde, central) | 0,891 | 0,747 – 0,970 |

O que isso quer dizer, em português:

1. **O balde nunca transbordou.** Em todos os 14 dias que alagaram, a lâmina simulada foi **zero**, e o alerta disparou
   em 0 dos 689 dias normais. O modelo físico, na prática, não "funcionou" — ele não chegou a ser exercitado.
2. O AUC de 0,891 do balde veio do **critério de desempate** (maior chuva horária do dia ÷ constante), que ordena os
   dias exatamente como a regra de chuva B3. Por isso os números são **idênticos**.
3. **Por que o balde não encheu?** A chuva de reanálise do Open-Meteo **suaviza os picos**: o máximo foi 16,7 mm/h,
   enquanto o ralo do balde aguenta 60,1 mm/h. Chuva real de pluviômetro em temporais passa disso; a reanálise não.
4. **H1 = inconclusiva** (o balde parece melhor que a climatologia, +0,113, mas o intervalo vai de −0,010 a +0,252,
   ou seja, inclui "nenhuma diferença"). **H2 não foi testada** (regra de parada do protocolo).
5. **Conclusão:** com estes dados, não dá para afirmar que algum modelo tem habilidade além de "chuva forte → alaga".
   Um teste com poder precisa de **chuva de pluviômetro**, **mais episódios de alagamento** e **dados de drenagem**.

> Isso **não é fracasso do projeto**: é um resultado negativo bem documentado e reproduzível, que mostra exatamente
> o que falta. Em pesquisa, isso tem valor.

---

## 8. Pontos de atenção no código (observações minhas, para você conhecer)

Não são correções feitas, apenas coisas que vale entender se alguém perguntar:

1. **Dados sintéticos como plano B** (`trainer.py:754`): se o banco falhar, o treino segue em silêncio com dados
   inventados. Um modelo assim treinado não deveria ir para produção.
2. **Treino preenche vazios com 0, previsão recusa vazios** (`feature_engineering.py:563`): coerente com a ADR 0007 do
   lado da previsão, mas o treino ainda "inventa" zeros (comentário no código reconhece isso).
3. **O scaler é ajustado em todos os dados antes de separar treino/teste** (`trainer.py:770`): o teste "vaza" um pouco
   de informação para o treino. Efeito pequeno, mas o correto é ajustar só no treino.
4. **Classes desbalanceadas:** com quase nenhum dia de alagamento, o XGBoost pode aprender a dizer "não alaga" sempre e
   ainda acertar 98%. Por isso a acurácia engana e o projeto usa AUC.
5. **LSTM com histórico curto** (`app.py:159`): se faltar histórico, a sequência é completada **repetindo a leitura
   atual**, o que dá ao modelo um "filme" artificial.
6. **SQL montado com f-string** (`app.py:150`): os valores são números do payload validados pelo Pydantic, então o risco
   é baixo, mas o padrão recomendado é consulta parametrizada.

---

## 9. Glossário rápido

| Termo | Significado |
|---|---|
| **Reanálise** | Estimativa do tempo no passado feita por um modelo meteorológico que combina observações. Cobre tudo, mas suaviza extremos |
| **Pluviômetro** | Aparelho que mede a chuva num ponto. Mais preciso localmente |
| **LiDAR / MDT** | Laser aéreo que mede o relevo; gera o Modelo Digital de Terreno |
| **Curva IDF** | Intensidade–Duração–Frequência: diz qual chuva é "rara" numa cidade (ex.: 60,1 mm em 1h a cada 5 anos) |
| **Método racional** | Fórmula clássica de drenagem: vazão = coeficiente × chuva × área |
| **CKAN** | Portal de dados abertos da Prefeitura do Recife, de onde vêm os registros de alagamento |
| **Pré-registro** | Escrever **antes** como vai avaliar e o que conta como sucesso, para não ajustar a análise depois de ver o resultado |
| **Bootstrap** | Reamostrar os dados milhares de vezes para estimar a incerteza |
| **Hindcast** | "Previsão do passado": rodar o modelo com dados já observados para ver se ele teria acertado |
| **ADR** | *Architecture Decision Record*, documento que registra uma decisão técnica e o porquê |
| **joblib / .h5** | Formatos de arquivo onde os modelos treinados ficam salvos |
| **FastAPI** | Biblioteca Python que transforma funções em endpoints HTTP |

---

## 10. Mapa de arquivos para estudar, em ordem

1. `README.md` — visão geral
2. `main.py` — os dois modos (train / serve)
3. `mapi_ai/feature_engineering.py` — como os dados viram as 28 colunas
4. `mapi_ai/models.py` — definição do XGBoost e da LSTM
5. `mapi_ai/trainer.py` — o treino
6. `mapi_ai/app.py` — a API de previsão
7. `avaliacao/balde.py` — o modelo físico (curto, ~70 linhas)
8. `avaliacao/gerar_escores.py` e `avaliar_h1_h2.py` — a avaliação científica
9. `projeto-mapi-api/docs/MEMO-ESCASSEZ-DE-ROTULO.md`, `MODELAGEM-IA.md`, `INTERPRETACAO-F1.md` — o raciocínio por trás
