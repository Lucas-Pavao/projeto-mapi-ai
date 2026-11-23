# Maré e limpeza no balde: efeito em 5 anos de chuva medida (onda 16, Fase 2.4)

Gerado por `python -m mapi_ai.fatores_5anos` em 2026-11-22 (resultado completo em `reports/fatores/fatores_5anos.json`).
Pergunta: com a chuva **medida** de 5 anos, quantas vezes o refluxo de maré e a limpeza mudam a lâmina do balde a 10 min
em CIn e Mascarenhas, e em quanto? **Nenhum dia de alagamento foi usado**: nada aqui é comparado com ocorrência e nenhum
parâmetro foi ajustado. As datas no JSON são saídas do balde, não acertos.

## Entradas

| Entrada | Valor | Fonte |
|---|---|---|
| Chuva | série de 10 min de 01/09/2021 a 01/11/2026 (5,17 anos), regra do alerta | `data/bucket10/serie10min_<ponto>.csv.gz` (exportada por `GET /api/alerta/serie-10min.csv`) |
| Terreno | CIn: 82.378 m², rua 0,109 m acima do fundo; Mascarenhas: 713 m², 0,126 m | `data/bucket10/terreno.json` |
| Maré | extremos da tábua DHN do Porto do Recife 2021–2026, cosseno a cada 10 min | `projeto-mapi-api/src/main/resources/tide/dhn-pe02-AAAA.csv` (SHA-256 no JSON) |
| Soleira de Mascarenhas | 0,638 m no MDT (CG 88084) + Δ 0,821 m = 1,459 m NR; D = 0,6 m | `AlertaProperties.Refluxo`; ADR 0009 emendas 1 e 3 |
| Cenário de viés | MDT − RN = −0,369 m: soleira 1,828 m NR | `projeto-mapi-api/docs/extratos/referencia-vertical.md:101` |
| Limpeza de Mascarenhas | regime permanente, λ = 0,03331/dia, D_E = 44,2 dias, π_E = 0,596 | `LimpezaProperties` (emenda 1 da 156) |
| Galeria entupida | f_E = 0 (padrão, sem fonte) e f_E = 0,5 (ponta da faixa) | `docs/PARAMETROS-ENTUPIMENTO.md` |

## Resultado

| Ponto | Sem fatores | Maré, cenário central | Maré, cenário de viés | Limpeza |
|---|---|---|---|---|
| CIn | 0 evento ≥ 15 cm; lâmina máx. 0,0 cm | não se aplica (sem saída cadastrada no ESIG) | não se aplica | sem dado (0 limpezas casadas a 150 m; mínimo 10) |
| Mascarenhas | 0 evento; lâmina máx. 0,0 cm | 113 episódios com a lâmina mudando (21,9 por ano); mudança mediana +2,7 cm, p95 +8,2 cm, máx. +10,6 cm; 0 evento ≥ 15 cm | 12 episódios (2,3 por ano); máx. +6,4 cm; 0 evento | f_E = 0: 41 eventos ≥ 15 cm e lâmina > 0 em 86% dos passos (a galeria fechada nunca esvazia); f_E = 0,5: 8 episódios, máx. +2,3 cm, 0 evento |

Maré em Mascarenhas, cenário central: a saída fica reduzida em 42,3% do tempo e fechada em 10,7% (cenário de viés: 21,5% e
1,0%). Dos passos com chuva, 42,2% caem com a saída reduzida.

## O que isso quer dizer

1. **A maré muda a lâmina com frequência, mas pouco.** Em Mascarenhas, o refluxo põe água na rua em cerca de 22 episódios
   por ano, com 2,7 cm de mediana. Sozinho, nunca leva a lâmina a 15 cm nos 5 anos. A capacidade da galeria (Fase 4)
   continua sendo o que decide se o alerta pode disparar.
2. **O viés do MDT pesa.** Com a soleira 0,369 m mais alta, os episódios caem de 113 para 12. Por isso o cenário de viés vai
   em todo ciclo, como sensibilidade (ADR 0030).
3. **A galeria totalmente fechada não é um estado físico útil.** Com f_E = 0 e sem infiltração nem evaporação no balde, a
   água de qualquer chuva fica para sempre: lâmina acima de zero em 86% dos passos de 5 anos. Isso confirma a decisão do
   ADR 0030 de deixar a limpeza em `INFO` até a Fase 4 dar uma faixa de capacidade com fonte. Com f_E = 0,5, o efeito
   some (máx. 2,3 cm).
4. **CIn não tem nenhum dos dois fatores**: falta saída cadastrada e faltam limpezas casadas. O alerta de CIn continua
   sendo só chuva e terreno, e a tela diz isso ("não se aplica", "sem dado").

## Limitações

- Balde sobre a chuva medida (o "agora" de cada passo), não sobre a previsão: mede o efeito dos fatores na física, não o
  efeito no nível previsto.
- A maré é a da tábua (previsão), não o nível observado no porto.
- A soleira vem do cadastro ESIG (um ponto de galeria a até 150 m); o tubo real de saída para o canal não foi conferido em
  campo.
