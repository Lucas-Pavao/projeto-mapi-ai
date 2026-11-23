# Calibração inversa da capacidade (onda 16, Fase 4.3)

Pré-registro: `projeto-mapi-api/docs/PRE-REGISTRO-CALIBRACAO-INVERSA.md` (escrito antes). Rodado por `python -m mapi_ai.calibracao`
em 2026-11-23 (`reports/calibracao/calibracao.json`). Aval do orientador: **pendente** (o prompt pede; o Lucas deu o aval geral).

## Resultado

| Ponto | Faixa (m³/s) | Positivos / negativos de controle | κ̂ | IC95 de κ | Acurácia balanceada em κ̂ (em κ = 0,5) | Adotado |
|---|---|---|---|---|---|---|
| Ibura | 0,084 – 0,335 | 11 / 167 | — | — | 0,50 em todos os κ | não |
| CIn | 2,07 – 2,82 (IDF, sem cadastro) | 6 / 184 | — | — | 0,50 em todos os κ | não |
| Mascarenhas | 0,138 – 0,265 | 10 / 140 | 0,85 | 0,30 – 0,95 | 0,650 (0,625) | não |
| Jaboatão | 0,078 – 0,106 (IDF) | 0 / 172 | — | — | sem positivo | não calibrável |

Critérios de adoção (todos precisam passar): pelo menos 3 positivos; IC95 de κ com largura ≤ 0,5; ganho ≥ 0,05 sobre κ = 0,5.
Mascarenhas tem positivos, mas o IC tem largura 0,65 e o ganho é 0,025. **Nenhum κ foi adotado**: todos os pontos ficam com a
regra da faixa (ALERTA com Q_max, ATENÇÃO com Q_min). `data/capacidade/calibracao_adotada.json` está vazio.

## Leitura

- Ibura e CIn: em nenhum dos 178 e 190 dias o balde chega a 15 cm, em nenhum κ da faixa. A faixa com fonte deixa a galeria
  grande demais para a área desses pontos (Ibura: 306 a 1.222 mm/h equivalentes em 1.146 m²; CIn: a capacidade de projeto),
  e a calibração não tem o que discriminar. Para esses pontos o balde, com o que temos de cadastro, não dispara.
- Mascarenhas: com Q_min a sensibilidade é 0,80 e a especificidade 0,40 (dias de 15 a 40 mm sem registro também passam de 15 cm);
  com Q_max, 0,60 e 0,64. A troca entre as duas pontas existe, mas 10 positivos não seguram um κ.
- Lembrete do pré-registro: dia sem registro da Defesa Civil não prova rua seca. A especificidade aqui é contra "sem registro".
