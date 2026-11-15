# Fórmula geradora do rótulo sintético

**DADOS SINTÉTICOS — experimento controlado.** Nenhum número desta pasta é desempenho em alagamento real.

Por que sintético: 0 ocorrências de alagamento a 500 m dos pontos e nenhuma com hora no banco (`mapi_ai/build_labels.py`,
`reports/dataset_summary.json`). O piso para um classificador supervisionado era 30 positivos.

## Fórmula

```
is_flooded ~ Bernoulli(sigmoid(b0 + 6.0*P1h/R1 + 10.0*P3h/R3 + 2.0*P24h/R24 + 6.0*max(0, mare - T_hi)/T_span * 1[P3h > wet3]))
```

P1h, P3h e P24h são a chuva **medida** (Cemaden, série do ponto) em mm; maré em m sobre o NR da DHN.

| Escala | Valor | Origem (dado real) |
|---|---|---|
| R1 | 31.83 mm | mediana do pico horário dos eventos FORTE (tabela de eventos de 5 anos) |
| R3 | 54.68 mm | quantil 99,9% da chuva de 3 h da série dos pontos |
| R24 | 104.69 mm | quantil 99% da chuva de 24 h |
| T_hi | 2.077 m | quantil 90% da maré horária |
| T_span | 0.339 m | quantil 99% − quantil 90% da maré |
| wet3 | 7.40 mm | quantil 95% da chuva de 3 h ("está chovendo" para a interação maré × chuva) |
| b0 | -5.056 | resolvido por bisseção para 8% de positivos esperados na amostra |

Coeficientes (escolhidos antes de treinar, `[sem fonte]` como valor físico): {'current_rainfall': 6.0, 'rainfall_3h_accumulated': 10.0, 'rainfall_24h_accumulated': 2.0, 'tide_x_rain': 6.0}. Com (1,5; 2,5; 0,8; 3,0) a hora seca tinha 4,3% de positivos e o rótulo virava ruído. Ordem de importância verdadeira
(pela fórmula): **chuva de 3 h > chuva de 1 h > maré (só com chuva) > chuva de 24 h**. Não entram na fórmula: chuva de 6 h e
12 h (correlacionadas com 3 h e 24 h), hora, mês, latitude e longitude.

## Amostra

5000 horas reais, 7.7% positivas, semente 42. Estratos (horas do evento + 3 h depois):

| Estrato | Horas disponíveis | Sorteadas | Probabilidade de inclusão |
|---|---|---|---|
| FORTE | 6052 | 900 | 0.1487 |
| MEDIA | 39493 | 1300 | 0.0329 |
| FRACA | 39116 | 1300 | 0.0332 |
| SECO | 127550 | 1500 | 0.0118 |

A amostra super-representa chuva forte em relação ao clima; métricas valem para esta amostra, não para "uma hora qualquer".

## Limites

- O rótulo é função da chuva **medida**. A ablação Cemaden × reanálise mede quanta informação sobre a chuva medida no
  local a reanálise preserva; por construção favorece o Cemaden e não prova que o Cemaden prevê alagamento real melhor.
- Ruído do rótulo = sorteio de Bernoulli; não há erro de notificação simulado.
- SHA-256 do dataset: `5c20d270dbe06cd67079bd9e8e924bb902b83ab10448ab576c5a362428d73430`.

DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC
