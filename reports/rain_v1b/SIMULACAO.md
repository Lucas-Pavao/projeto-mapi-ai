# Simulação do dado como chega e rain_v1b (Etapa 2.3 do prompt do alerta de 20 min)

DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC. Data: 2026-11-17. Semente 42.
Código: `mapi_ai/rain/availability.py`, `mapi_ai/rain/simulate_v1b.py`; testes `tests/test_rain_availability.py`.
Números: `evaluation_estavel.json`, `evaluation_chuva.json`, `evaluation_estavel_relogio_ingenuo.json`.

## 1. O que a coleta ao vivo entrega (item 2.3.1)

**Formato do bloco: igual ao da PED.** Na sobreposição de 2026-11-08 a 11-17, por estação canônica e instante:
2.142 blocos iguais (mesmo fim e mesmo valor), 0 com valor diferente. Dos 2.456 blocos que só a PED tem, 2.432 caem em
horas sem nenhuma coleta ao vivo (paradas da API, inclusive 223 blocos com chuva entre 10 e 13/11); os outros 24 são
secos. A coleta ao vivo não agrega nem muda o bloco: ela só chega depois, ou não chega quando o coletor está parado.
(fonte: consulta em `observation_current` × `station_alias`, `mapi_db`, 2026-11-17)

**Atraso de chegada** = `raw_observation.collected_at − ts`, só em regime estável (coleta que veio até 6 min depois da
anterior; a reentrega depois de uma parada não é atraso da fonte). Arquivo: `data/rain_v1b/live_delays.csv`
(2026-11-09 a 11-17, 1.498 leituras, 45 estações, 33 com 20 ou mais amostras).

| Leituras | n | Mediana | p90 | p99 | Máximo |
|---|---|---|---|---|---|
| Secas | 1.487 | 3,6 min | 7,7 min | 9,7 min | 12,8 min |
| Com chuva | 11 (7 estações) | 8,7 min | 18,0 min | 248,9 min | 274,5 min |

Com chuva há 11 amostras: não há distribuição de atraso em dia de chuva que mereça o nome. Por isso dois cenários.

## 2. A simulação (item 2.3.2)

- Atraso sorteado por leitura, com semente 42.
  - **estavel**: da distribuição da própria estação em regime estável (todas as amostras; estação com menos de 20 usa
    todas as estações juntas).
  - **chuva** (pessimista, declarado): leitura com chuva sorteia das 11 leituras ao vivo com chuva (todas as estações
    juntas); leitura seca, como no estável.
- **Âncora e ciclo, como no serviço.** O `serve.py` ancora as entradas na leitura mais nova que chegou (não no relógio),
  e o ciclo do alerta recalcula a cada 3 min. Na simulação, as entradas do passo `t` são calculadas nos ciclos que
  rodam 1, 4 e 7 min depois de `t`, cada um vendo só as leituras com `ts + atraso ≤ momento do ciclo`. O aviso conta a
  partir do primeiro ciclo que disparou: antecedência = cruzamento − (t + minutos do ciclo).
- A verdade (evento FORTE, alvo) é sempre a série completa da PED, a mesma do relatório da Fase 3 (28 eventos no teste,
  em 15 dias).
- Com atraso zero, as entradas simuladas são idênticas às do laboratório, bit a bit
  (`test_atraso_zero_reproduz_o_laboratorio`).

**Achado sobre o cálculo no relógio.** Uma primeira versão calculou as entradas no instante do relógio (múltiplo de
10 min), sem âncora. Nesse instante o bloco que termina em `t` ainda não chegou, então `p_10` sai zero, e o modelo de
1 h não avisou nenhum dos 28 eventos (antecedência 0 min; `evaluation_estavel_relogio_ingenuo.json`). O serviço não
funciona assim, mas o resultado mostra que **a âncora na leitura mais nova é o que sustenta a antecedência**: se ela
deixar de existir (por exemplo, passando `issued_at` = agora), a antecedência some.

## 3. Resultado (itens 2.3.3 a 2.3.5): teste, 28 eventos FORTE

Antecedência mediana em min, IC 95% por bootstrap de evento (1.000 reamostras). Evento sem aviso conta 0. Limiar do
`rain_v1`: o da validação do laboratório (1 h: 0,1055). Limiar do `rain_v1b`: maior CSI na validação da simulação.

### Modelo de 1 h (o que dispara ATENCAO)

| | Avisados | Antecedência mediana (IC 95%) | Falsos por evento | Recall FORTE (já chove / ainda não) |
|---|---|---|---|---|
| Laboratório, rain_v1 | 20/28 | **20** (10–40) | 3,04 | 32% / 0% |
| Simulação estável, rain_v1 | 19/28 | **16** (3–33) | 3,11 | 31% / 0% |
| Simulação estável, rain_v1b | 20/28 | 16 (3–36) | 3,79 | 35% / 0% |
| Simulação chuva, rain_v1 | 11/28 | **0** (0–8) | 1,46 | 20% / 0% |
| Simulação chuva, rain_v1b | 16/28 | 13 (0–23) | 3,21 | 32% / 0% |

- Simulação estável − laboratório (rain_v1): −4 min (IC −20 a −4).
- Critério (fixado antes): fica o `rain_v1b` se a antecedência dele for maior que a do `rain_v1` com IC sem o zero.
  Estável: diferença 0 (IC −10 a +16) → **fica o rain_v1**. Chuva: +13 (IC 0 a +19,5) → o IC toca o zero → **fica o rain_v1**.

### Modelos de 3 h e 6 h

| Horizonte | Laboratório | Estável v1 / v1b | Chuva v1 / v1b | Fica (estável / chuva) |
|---|---|---|---|---|
| 3 h | 40 (20–145) | 36 (16–140) / 36 (13–143) | 1,5 (0–131) / 34,5 (3–160) | rain_v1 / **rain_v1b** (+33, IC 1,5 a 133) |
| 6 h | 195 (40–300) | 190 (36–293) / 253 (31–316) | 171 (28–303) / 208 (40–309) | rain_v1 / rain_v1 |

## 4. Leitura

- Com o atraso medido em hora seca (3,6 min de mediana), o aviso de 1 h perde 4 min: **16 min em vez de 20**.
- Com o atraso das poucas leituras com chuva (8,7 min de mediana e uma cauda de horas), o aviso de 1 h do `rain_v1`
  **some** (0 min, IC 0 a 8). O `rain_v1b`, treinado com o atraso, recupera parte (13 min), sem passar no critério.
- O cenário que decide é o da chuva, e ele tem 11 amostras. Esta é a condição de parada da Etapa 2.3 ("antecedência
  mediana na simulação de 5 min ou menos"): **cumprida no cenário chuva, não cumprida no estável.**
- O modelo continua sem prever o início da chuva (recall de 0% quando ainda não chove nos dois cenários).

## 5. Limitações

- 11 leituras com chuva para o atraso em dia de chuva; o p99 vem de uma leitura de 249 min.
- Atraso sorteado independente por leitura: na prática, a fonte atrasa em bloco (todas as estações juntas).
- Ciclos de 3 min com fase fixa de 1 min; tempo de processamento (2 s por ciclo, medido) ignorado.
- Parada da coleta ao vivo não entra: vira SEM_DADO no alerta, não aviso atrasado.
