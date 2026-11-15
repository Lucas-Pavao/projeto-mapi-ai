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
