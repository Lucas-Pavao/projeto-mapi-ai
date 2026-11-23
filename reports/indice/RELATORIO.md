# Índice caixa branca com rótulo real (onda 16, Fase 5; ADR 0031)

Pré-registro: `projeto-mapi-api/docs/PRE-REGISTRO-INDICE-CAIXA-BRANCA.md` (escrito antes). Rodado por `python -m mapi_ai.indice`
em 2026-11-23 (`reports/indice/indice.json`, `curvas_ebm.png`, `confiabilidade.png`). **Só comparação: nada vai para a tela.**
Aval do orientador: pendente.

## Dados

4.858 ponto-dias calculáveis (5 pontos, 01/09/2021 a 25/06/2024, série de 10 min completa no dia), 1.015 dias, 159,6
ponto-meses. **37 positivos** (registro da Defesa Civil a até 2 km): Derby 10, Ibura 11, CIn 6, Mascarenhas 10, Jaboatão 0.
O prompt fala em 41 ponto-dias: 4 caem em dias sem a série completa ou antes de 01/09/2021 e ficam fora pela regra escrita.

## Resultado fora da dobra (5 dobras agrupadas por dia)

| Modelo | AUC [IC95] | PR-AUC [IC95] | Brier | No mesmo número de ponto-dias marcados que a regra (787): POD · falsos por ponto-mês |
|---|---|---|---|---|
| Regra "chuva medida em 6 h ≥ 25 mm" | 0,968 [0,949; 0,984] | 0,157 [0,062; 0,335] | 0,154 (0/1) | **1,000** (37/37) · 4,70 |
| Logística L2, 3 entradas | 0,964 [0,937; 0,985] | 0,155 [0,059; 0,323] | 0,063 | 0,946 (35/37) · 4,71 |
| EBM, 3 entradas + 1 interação | 0,917 [0,842; 0,968] | 0,096 [0,030; 0,253] | 0,008 | 0,838 (31/37) · 4,74 |

## O que a logística diz (ajuste no conjunto todo; IC por bootstrap de dia)

- **Cada 10 mm a mais de chuva medida em 6 h multiplica a chance de registro por 2,75** (IC95 2,12 a 4,92).
- Cada 10 mm a mais no máximo de 30 min: × 0,81 (IC95 0,08 a 2,04): sem efeito separado da chuva de 6 h.
- Cada 0,5 m a mais de maré na hora da chuva: × 2,04 (IC95 0,91 a 3,86): sinal de efeito, intervalo inclui 1.

EBM: a chuva de 6 h pesa mais (importância 1,14), depois a maré (0,73), o máximo de 30 min (0,55) e a interação chuva × pico
(0,55). As curvas estão em `curvas_ebm.png`.

## Leitura

1. **Nenhum dos dois índices bate a regra simples de 25 mm em 6 h** neste rótulo. A logística empata em AUC e perde 2 dos 37
   positivos no mesmo número de marcações; o EBM perde 6.
2. A regra e os índices marcam muito: 787 ponto-dias para 37 registros (4,7 falsos por ponto-mês). Com 18 dias da Defesa Civil
   em 3 anos, boa parte desses "falsos" pode ser alagamento sem registro: a métrica de falso alarme aqui é contra "sem registro".
3. Calibração: a logística (pesos balanceados, como fixado) superestima muito a chance (na faixa de 0,9 a 1,0 a fração
   observada é 0,13); o EBM é mais calibrado no baixo, mas superestima entre 0,1 e 0,4 (`confiabilidade.png`). Nenhum dos dois
   serve como probabilidade para a tela.
4. A maré aparece com sinal positivo nos dois modelos, coerente com o refluxo, mas sem precisão para afirmar.

## Limitações

Os 18 dias já tinham sido olhados (regra 1): isso é ajuste e diagnóstico dentro da amostra de calibração, não validação. A
validação dos índices, do balde e da regra é a de 2027 (Fase 9), com os rótulos lacrados.
