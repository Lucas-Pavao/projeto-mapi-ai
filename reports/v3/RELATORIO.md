# Modelo v3 — relatório

**DADOS SINTÉTICOS — experimento controlado.** O rótulo vem da fórmula do v2 (`reports/synthetic_ground_truth.md`), sem alteração. Nenhum número deste relatório é desempenho em alagamento real.

DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC.

Regenera com `notebooks/v3_modelo.ipynb` (ou `python -m mapi_ai.v3.dataset`, `train`, `evaluate`, `physical`, `explain`, `model_card`). Semente 42. O v2 (`reports/`, `models/`) não é tocado.

## Os cinco resultados

1. **O modelo não antecipa o alagamento.** No teste, na série contínua, com 50 dias de alerta por ponto por ano, o v3 de 6 h detecta 33,2% dos eventos (51,2% dos que acontecem em chuva) com **antecedência mediana de 0 h** e p10 de 0 h. Os modelos de 0 e 3 h dão o mesmo: 0 h. Com chuva e maré só da hora `t`, o alerta sai quando o evento começa.
2. **O mundo sintético tem eventos demais.** A fórmula do v2, aplicada à série contínua de 5 anos, deixa 3,3% das horas alagadas (H = 0); são cerca de 107 eventos por ponto por ano, contra uns 4 declarados para a realidade. Com orçamento de 50 dias de alerta por ano, a maioria dos eventos não tem como ser avisada.
3. **Janelas disjuntas pioraram as explicações (resultado negativo).** A correlação média entre as entradas de chuva caiu de 0,65 para 0,25, mas a concordância SHAP × LIME caiu (Jaccard do top-5: **0,77 → 0,62** em H = 0; 0,84 → 0,70 em H = 6) e a estabilidade do LIME também (0,80 → 0,75; 0,89 → 0,87). A recuperação da fórmula pelo SHAP caiu de 0,87 para 0,53 (Spearman, H = 0). Só a fidelidade por deleção melhorou um pouco (H = 6: 0,015 → 0,028, contra 0,009 do acaso). A hipótese do pedido não se confirmou.
4. **Entradas físicas não ajudaram (resultado negativo).** Com um gerador em que pico de 10 min, tendência da maré e umidade do solo têm papel físico, a PR-AUC de teste ficou em **0,642 com elas contra 0,647 sem**. O pico de 10 min falta em 77% das horas com até 5 mm e em 52% das horas com 5 mm ou mais (blocos longos do Cemaden); a umidade do solo chega com ~2 semanas de atraso (sem ela, como em produção: 0,645); o rio ficou de fora por não ter histórico.
5. **A regra contínua de 3 h é um baseline forte.** "Risco = chuva de 3 h" faz 0,585 de PR-AUC em H = 6 (teste), contra 0,650 da logística. No v2 a regra entrava como saída 0/1 e marcava 0,460, injustamente baixo.

## Escolha do modelo (B4)

Critério gravado no metadata antes de treinar: maior PR-AUC na validação, no horizonte principal (6 h); empate dentro do IC 95% do bootstrap pareado fica com o mais simples.

| Horizonte | Regra 3 h | Logística | Árvore rasa | XGBoost | Escolhido |
|---|---|---|---|---|---|
| 0 h | 0,725 | **0,790** | 0,626 | 0,771 (empate: IC −0,010 a 0,053) | Logística |
| 3 h | 0,654 | **0,729** | 0,550 | 0,720 (empate: IC −0,009 a 0,028) | Logística |
| 6 h | 0,601 | **0,672** | 0,541 | 0,665 (empate: IC −0,007 a 0,020) | Logística |

PR-AUC de validação, amostra estratificada. A logística foi a melhor e a mais simples entre as empatadas nos três horizontes.

## Teste (olhado uma vez)

| Horizonte | PR-AUC com lacunas | PR-AUC sem lacunas | ROC-AUC | Brier bruto → calibrado | ECE bruto → calibrado |
|---|---|---|---|---|---|
| 0 h | 0,677 | 0,727 | 0,873 | 0,047 → 0,031 | 0,089 → 0,015 |
| 3 h | 0,633 | 0,657 | 0,808 | 0,107 → 0,080 | 0,155 → 0,038 |
| 6 h | 0,651 | 0,657 | 0,816 | 0,131 → 0,110 | 0,145 → 0,022 |

v2, para comparação (amostra e sorteios do rótulo diferentes): XGBoost 0,764; logística do v2 0,800.

As lacunas custam pouco em 6 h (0,651 contra 0,657) e mais em 0 h (0,677 contra 0,727): sem a chuva da hora, o nowcast perde o que mais pesa.

## Calibração, prevalência e limiares (B6)

- **Calibração**, regra gravada antes: Platt ajustado na primeira metade (no tempo) da validação, adotado se ECE e Brier melhoram na segunda. Em 6 h: ECE 0,215 → 0,047, Brier 0,159 → 0,111. Adotado nos três horizontes.
- **Prevalência**: `logit(p') = logit(p) + logit(π_real) − logit(π_sint)`. π_sint = 16,8% (metade da validação que ajusta o calibrador); π_real = 0,35% para 6 h, derivada de 0,1% das horas e 18 dias de alagamento em 5 anos (CKAN). A duração do alagamento por dia não é conhecida `[sem fonte]`. A probabilidade servida é a corrigida.
- **Limiares**, sobre a probabilidade servida: o menor limiar com no máximo 50 (moderado) e 10 (alto) **dias de alerta por ponto por ano**, na série contínua da validação. Em 6 h: moderado 4,9%, alto 92,8%. A contagem é em dias, não em episódios: a contagem de episódios não é monótona no limiar (limiar baixo funde episódios vizinhos e a contagem cai), e um limiar quase zero passaria no orçamento com um único alerta ligado o ano inteiro.
- A curva "dias de alerta por ano × eventos detectados" está em `reports/v3/alert_curve_h{0,3,6}.csv` e `figures/alert_curve_h*.png`, para escolher os números finais.

## Eventos no teste, série contínua, 6 h (B7)

| Nível | Dias de alerta por ponto por ano | Eventos detectados | Em chuva | Antecedência mediana | Falsos por evento |
|---|---|---|---|---|---|
| Moderado | 49,9 | 33,2% | 51,2% | 0 h | 0,10 |
| Alto | 8,9 | 8,2% | 12,7% | 0 h | 0,00 |

PR-AUC por hora na série contínua (teste): 0,440, com prevalência de 10,8%; na amostra estratificada, 0,651. A amostra super-representa chuva forte.

## Lacunas (B2)

Padrão medido na série real (`projeto-mapi-api/docs/extratos/lacunas-chuva-ponto.md`): lacuna curta independente da chuva, fração por ponto de 0,18% a 0,78% das horas, parte simultânea entre os pontos. Aplicado depois de gerar o rótulo. Lacuna longa (mês inteiro sem estação válida) não é simulada: em produção vira `UNKNOWN`.

Regra de degradação da API (sem a chuva da hora e sem a de 3 h → `UNKNOWN`): nas 99 horas da validação sem a chuva da hora, o modelo de 6 h ainda ordena bem (PR-AUC 0,70 contra 0,55 nas mesmas horas com a chuva completa; só 13 positivos, número frágil). O de 0 h não (0,016; 2 positivos). A regra fica como está até haver mais horas para medir.

## Limitações

- Rótulo sintético: o modelo reaprende a fórmula; SHAP recuperar as entradas certas é quase garantido por construção.
- Antecedência de 0 h: precisa de previsão de chuva como entrada.
- π_real depende de uma duração de alagamento por dia sem fonte.
- A variante física mudou o gerador; não se compara com o v2 como ganho de modelo.
