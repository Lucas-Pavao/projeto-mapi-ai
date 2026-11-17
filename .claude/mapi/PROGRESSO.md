# Progresso — projeto-mapi-ai (IN1165)

Estado geral da revisão: `projeto-mapi-api/.claude/mapi/PROGRESSO.md`. Aqui, só as etapas do projeto final IN1165.
Contrato de entrada: `projeto-mapi-api/docs/handoff/etapa0.json`.

| Etapa | Status | Commit | Notas |
|---|---|---|---|
| 1 Dataset | concluida | (este) | flood_scenario_labels 0 linhas; CKAN 225 ocorrências / 18 dias / 0 com hora; 0 pontos-dia a 500 m (1 km: 6; 2 km: 45) → Plano B. Sintético: 5 000 horas reais (estratos da tabela de 4 408 eventos Cemaden), 7,7% positivos, fórmula em `reports/synthetic_ground_truth.md`. `load_dataset()` exige `--synthetic`. Nada gravado no banco. |
| 2 Treino | concluida | (este) | split temporal 60/20/20 (treino até 2024-07-03, validação até 2025-07-31, teste até 2026-10-07); XGBoost sem scaler, NaN nativo, `scale_pos_weight` = 11,0 (só treino); grade de 12 combinações escolhida por PR-AUC na validação; ablações sem lat/lon e com reanálise; `models/xgb_v2*.joblib` + `metadata.json`; app carrega v2 com fallback para o legado; pytest 45 passam |
| 3 Baselines | concluida | (este) | logística em Pipeline (imputer→scaler→LR balanced, ajuste só no treino) e árvore profundidade 3 (`reports/baseline_tree.txt` e `figures/baseline_tree.png`): a árvore usa só chuva de 3 h e 24 h |
| 4 Avaliação | concluida | (este) | teste (1 000 linhas, 63 positivas): PR-AUC logística 0,800 > XGB 0,764 (Δ −0,035, IC95 [−0,070; −0,006], significativo) — caixa-preta NÃO se justifica neste experimento; Cemaden 0,764 × reanálise 0,318 (Δ +0,446 [0,336; 0,548]); sem lat/lon 0,792 (local é ruído). Calibração Platt na validação: critério pré-fixado ECE_val > 0,05 não disparou (0,047), servido o bruto; no teste o calibrado teria Brier 0,026 × 0,037. Limiares (validação): HIGH 0,2441 (recall 0,83, precisão 0,46), MEDIUM 0,0159 (recall 0,96, precisão 0,13) |
| 5 XAI | concluida | (este) | SHAP probabilidade/interventional, fundo 200 (masker explícito: o padrão reduzia a 100), aditividade com assert; top-5 global: chuva 3 h, 24 h, 1 h, maré, 12 h; casos TP p=0,999, FN p=0,239, FP p=0,975 (FP puxado pela chuva de 24 h = 169 mm). LIME nos mesmos casos. DiCE: 133 candidatos, 18 coerentes como saíram; com reparo de coerência, 3 válidos; todos exigem ~74 mm a menos em 3 h, porque LOW = p < 0,016. Escala: sair de HIGH = 64,6 mm a menos em 3 h. app.py devolve top_contributors (SHAP, unidade física) |
| 6 Avaliação das explicações | concluida | (este) | 50 linhas (25 com chuva 3 h > 1 mm): SHAP×LIME Jaccard top-5 0,53 ± 0,18, Spearman 0,53 ± 0,24; estabilidade LIME (10 sementes) 0,61 ± 0,10; deleção (prob. da classe prevista) k=1: SHAP 0,113, LIME 0,104, aleatório 0,016; k=5: 0,179 / 0,151 / 0,088. Plano B: SHAP recupera as 4 features da fórmula no top-4, na ordem verdadeira (3 h > 24 h > 1 h > maré), Spearman com a verdade 0,84; LIME 3 de 4 (troca maré por 12 h), Spearman 0,80. Corrigidos antes de aceitar: deleção com sinal da classe prevista, Spearman com empates, ordem verdadeira no .md (era a dos coeficientes), gzip com mtime=0 (hash mudava a cada geração) |
| 7 Notebook e docs | concluida (verificação em ambiente limpo: ver linha seguinte) | (este) | `notebooks/in1165_projeto_final.ipynb` (10 seções, markdown em português) executado por nbconvert em 89 s, regenerando as 20 figuras; requirements travados (shap, lime, dice-ml, nbconvert); README "Projeto final IN1165"; `docs/handoff/etapas1a7.json`; Dockerfile 3.10 → 3.12 (pandas 3 exige ≥ 3.11) |

## Decisões
| Data | Decisão | Por quê |
|---|---|---|
| 2026-11-15 | Plano B | 0 positivos reais a 500 m, nenhum com hora (piso 30) |
| 2026-11-15 | Features e chuva reais; só o rótulo é sintético | calibração pelas distribuições reais sem inventar números |
| 2026-11-15 | Coeficientes do gerador (6; 10; 2; 6) em vez de (1,5; 2,5; 0,8; 3,0) | com os primeiros a hora seca tinha 4,3% de positivos (rótulo ≈ ruído); escolhido antes de treinar, só pela taxa por estrato: seco 0,5%, FORTE 28% |
| 2026-11-15 | Dados extraídos versionados em `data/in1165/` com SHA-256 | o notebook precisa rodar em ambiente limpo, sem o banco |
| 2026-11-15 | `import psycopg2` antes do TensorFlow no app.py | TF carregado antes faz a 1ª conexão do psycopg2-binary abortar o processo (reproduzido); o mesmo afetaria o `lifespan` em produção |
| 2026-11-15 | Features do v2 com os nomes do contrato /v1/predict/flood | inferência sem tradução; contrato só ganha campos por adição |
| 2026-11-15 | Usar calibrador só se ECE de validação > 0,05 (fixado antes do teste) | não disparou (0,047); o teste mostrou ganho do calibrado, mas trocar depois de ver o teste seria vazamento — fica como próximo passo |
| 2026-11-15 | HIGH = recall ≥ 0,80; MEDIUM = recall ≥ 0,95, ambos na validação | perder alagamento custa mais que alarme falso; MEDIUM é atenção |
| 2026-11-15 | Contrafactual da DiCE com reparo de coerência (janela longa = máx. das curtas) | a DiCE sorteia janelas independentes e gerava chuva de 1 h > 3 h; o reparo nunca passa do observado |
| 2026-11-15 | Deleção mede a prob. da CLASSE PREVISTA | com p(alaga) em todas as linhas, as horas secas (onde remover chuva aumenta p) diluíam a curva |
| 2026-11-15 | Atribuição verdadeira = |termo − média| do logit no teste | é o SHAP exato dos termos lineares da fórmula; o termo maré×chuva vai para a maré |

## Plano de 16/11 (estado geral em `projeto-mapi-api/.claude/mapi/PROGRESSO.md`)

| Fase | Status | Commit | Notas |
|---|---|---|---|
| 1.1 v3: notebook, relatório, handoff | concluida | c40a089 | `reports/v3/RELATORIO.md` com os dois resultados negativos; `docs/handoff/v3-ai.json` |
| 2 Medidas da descoberta | concluida | 4ff6ca1 | `analysis/fase2/` |
| 3 Previsão de chuva `rain_v1` | concluida | 0aba78a | `reports/v4/rain/RELATORIO.md`; rede em 1 h, rede + Open-Meteo da véspera em 3 h e 6 h; antecedência FORTE 20 min (IC 10–40) |
| 4 Rótulo do balde e v4 | parcial | 0636f2b | balde central: 0 evento em 5 anos; v4 não treinável; `--sensibilidade` à espera do ok do Lucas |
| ALERTA-E1-IA | 10 | IA | Opus 5.5 | Balde M1a a 10 min (`mapi_ai/bucket10.py`, `balde-M1a-10min`), `POST /v1/bucket/point10`, versão do código no `/health` | concluido (sem push/PR) | `alerta-20min` | 2026-11-17 · `docs/handoff/alerta-etapa1.json`; capacidade pela IDF em Te = 10 min, TR 5: 125,4 mm/h; chuva constante de 1 h dá a mesma lâmina do balde horário; pytest 84 passed |
| ALERTA-E2-IA | 10 | IA | Opus 5.5 | Simulação do dado como chega (âncora e ciclos de 3 min), rain_v1 reavaliado sem retreino, rain_v1b treinado na simulação | concluido; PARE da 2.3 | `alerta-20min` | 2026-11-17 · `reports/rain_v1b/SIMULACAO.md`; 1 h: laboratório 20 min (10–40), estável 16 (3–33), chuva 0 (0–8); fica o rain_v1 em 1 h; condição de parada cumprida no cenário chuva (11 amostras de atraso com chuva) |
| ALERTA-O3-IA | 10 | IA | Opus 5.5 | Maré de 2027: previsão harmônica (UTide) como TIDE_HARMONIC_F41, não ligada | concluido; PARE do O3 | `alerta-20min` | 2026-11-17 · F-41 sem constantes: ajuste na tábua DHN 2021-10..2024-12; 2025: altura p95 1,8 cm, horário p95 8 min; 2026: 2,2 cm e 8 min; tábua DHN 2027 não encontrada (TabuaMare ainda em 2026, site da Marinha atrás de Cloudflare) |
| ALERTA-E3-IA | 10 | IA | Opus 5.5 | Regras de liga e desliga escolhidas na validação da simulação (regra escrita antes) | concluido | `alerta-20min` | 2026-11-17 · `reports/rain_v1b/regras_alerta.json`; L_desliga = 0,3 x L_liga (0,0316), duração mínima 0; teste: trocas por evento 9,64 → 7,86, falsos por evento 2,61 antes do agrupamento e 2,25 depois, antecedência 20 min (5–40) igual |
| ALERTA-E2.4-IA | 10 | IA | Opus 5.5 | Efeito do controle de qualidade ao vivo nos 5 anos (simulação estável, rain_v1 de 1 h) | concluido | `alerta-20min` | 2026-11-17 · `reports/rain_v1b/controle_qualidade_ao_vivo.json`; filtro tira 489 leituras por pico isolado (151 horas) e 154.690 por estação suspeita (7.422 de 87.538 estação-dias); validação: falsos 82 → 66, eventos perdidos 0, antecedência 26 → 23 min; teste: falsos 87 → 73, 1 evento perdido, antecedência 16 → 6 min (0–16); decisão: o filtro vale só para a série medida do alerta (ALERTA e balde), não para as entradas do rain_v1 |
| ALERTA-E1.4.6-IA | 10 | IA | Opus 5.5 | Balde M1a a 10 min nos 5 anos, por ponto e TR (registro) | concluido | `alerta-20min` | 2026-11-17 · `reports/bucket10/balde10_5anos.json`; série exportada pela API (mesma regra do alerta); TR 5: 0 evento ≥ 15 cm em todos os pontos com terreno; TR 2: 1 evento no CIN (32,5 cm); maior chuva de 10 min no ponto 17,6 a 22,1 mm (capacidade TR 5 em 10 min: 20,9); 64% a 69% dos passos com chuva vêm de bloco longo espalhado |
| ALERTA-E4.1-IA | 10 | IA | Opus 5.5 | Reprise: preparo (eventos fixados, atrasos, esperado ciclo a ciclo), conferência do balde fora do sistema e extrato | concluido | `alerta-20min` | 2026-11-17 · 5 eventos; ligou no mesmo instante da simulação nos 13 pares ponto-evento; 2.237 chamadas ao balde com diferença 0,0 mm; nenhum ALAGAMENTO_PROVAVEL |
