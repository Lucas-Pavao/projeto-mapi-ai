# Alerta único reconstruído em 5 anos: quantos alertas o sistema daria (onda 16, Fase 4.4)

Rodado por `python -m mapi_ai.flood.reconstrucao` em 2026-11-23 (`reports/reconstrucao/reconstrucao.json`). Período: 01/09/2021 a
31/10/2026 (61,9 meses por ponto). **Só frequência**: nenhum dia de alagamento foi olhado; a comparação é da Fase 9.
Previsões do rain_v1 e do rain_v1p sobre o período inteiro (no treino, dentro da amostra). Regra fixada no pré-registro da
capacidade: ALERTA se a lâmina q50 chega a 15 cm com Q_max; ATENÇÃO se a q95 chega com Q_min. Nenhum κ calibrado foi adotado.

| Ponto | Antes da onda 16 (IDF, por igual) | Só a capacidade | Capacidade + pico previsto | Alerta único v2 (+ maré no nível) |
|---|---|---|---|---|
| Ibura (maré só informa) | 0 | 0 | 0 | **0** |
| CIn | 0 | 0,02 | 0,13 | **0,13** (só ATENÇÃO; 8 h em 5 anos) |
| Jaboatão | 0 | 0 | 0 | **0** |
| Mascarenhas | 0 | ALERTA 1,81 · A+A 8,6 | ALERTA 1,81 · A+A 47,7 | **ALERTA 9,75 · ALERTA ou ATENÇÃO 60,1** (ligado 27% do tempo) |

Episódios por ponto-mês (A+A = ALERTA ou ATENÇÃO). Episódio = passos de 10 min seguidos com o nível ligado.

## Leitura

1. **O balde agora pode encher.** Antes desta onda, 0 alerta em todos os pontos e em todas as variantes (ajuste 2 do prompt).
2. **Em Mascarenhas ele enche demais.** A capacidade do cano (0,14 a 0,26 m³/s, sem boca de lobo cadastrada no alcance) fica
   entre 14 e 28 mm/h equivalentes numa bacia de 40.159 m², e o ponto está numa depressão rasa (15 cm de lâmina na rua com a
   água a 0,38 m do fundo). Com a maré no nível, a ATENÇÃO (q95 com Q_min, as duas pontas pessimistas juntas) liga em quase
   toda preamar com alguma chuva prevista: 60 episódios por mês, 27% do tempo. Isso não serve como alerta para gente.
3. **Nos outros pontos ele quase não dispara.** Ibura e CIn têm galeria grande para a área com o cadastro que existe;
   Jaboatão usa a capacidade de projeto.
4. Para comparar: a linha de base "chuva medida em 6 h ≥ 25 mm" dava 3,77 falsos por ponto-mês na validação de 20/11.

## O que fica decidido e o que não

- O alerta no ar (modo sombra) passa a usar esta regra. Em Mascarenhas ele vai ficar em ATENÇÃO com frequência: está
  registrado e não muda nada para o público (sombra ligada).
- **Não mudo a regra da ATENÇÃO agora**: ela foi fixada antes de rodar, e mudar depois de ver a frequência seria escolher o
  limiar pelo resultado. A proposta para a Fase 9 (pré-registro da validação de 2027) é decidir, antes da quadra chuvosa,
  se a ATENÇÃO usa só uma das pontas pessimistas (q95 com Q_max ou q50 com Q_min).
- O que mais pesa em Mascarenhas, em ordem: a soleira da saída (cenário de viés do MDT divide os eventos por 11, Fase 3), a
  capacidade sem boca de lobo cadastrada (pedido 06 à EMLURB) e o q95 do pico (cobertura de 71% a 89%, abaixo do nominal).
