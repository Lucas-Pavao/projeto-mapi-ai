# Modelo v4 — relatório da Fase 4

**RÓTULO DO MODELO FÍSICO (balde M1a), não alagamento observado.** DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC.

Regenera com `python -m mapi_ai.v4.labels` e `python -m mapi_ai.v4.experiment`. Semente 42. Números: `bucket_label_frequency.json` e `experiment_p1.json`. A previsão de chuva usada é a do `rain_v1`, fora da amostra (`reports/v4/rain/RELATORIO.md`).

## Resposta em uma linha

**Não há número: com os parâmetros de fonte, o balde não produz nenhum alagamento em 5 anos, então não há evento para avisar nem antecedência para medir.** O experimento dos três braços está pronto e não foi rodado em outra capacidade de galeria, porque isso seria escolher parâmetro pela frequência.

## 4.1 Frequência de eventos do balde, por ponto

Balde M1a (`avaliacao/balde.py`, ADR 0014) sobre a chuva horária medida do ponto, 2021-09 a 2026-10 (5,0 anos por ponto), terreno do PE3D (`avaliacao-f1/entradas/`, congelado). Positivo: lâmina de 15 cm ou mais. Evento: horas positivas a menos de 6 h uma da outra. Jaboatão fora (sem terreno).

| Ponto | Central: f = 1,0 (60,1 mm/h) | f = 0,8 | f = 0,6 | f = 0,4 | f = 0,2 |
|---|---|---|---|---|---|
| Av. Recife (Ibura) | **0** | 0 | 0 | 2 (0,4 por ano) | 15 (3,1 por ano) |
| CIn - UFPE | **0** | 0 | 7 (1,4) | 31 (6,2) | 138 (27,4) |
| Agamenon (Derby) | **0** | 0 | 0 | 0 | 0 |
| Mascarenhas (Imbiribeira) | **0** | 1 (0,2) | 2 (0,4) | 18 (3,7) | 68 (13,9) |

Fonte: `python -m mapi_ai.v4.labels`. `f` é a fração da capacidade de projeto (TR 5, 60 min). A faixa de 0 a 60 mm/h em passos de 20% está em `PARAMETROS-MODELO-FISICO.md` como sensibilidade, porque a capacidade real da galeria é `[sem fonte]`.

- **Central: 0 evento em 5 anos nos 4 pontos**, contra "cerca de 4 por ano" (18 dias com alagamento registrado em 5 anos no CKAN = 3,6 por ano, na cidade inteira e sem hora). A diferença fica registrada; **nenhum parâmetro foi ajustado**.
- Por quê: o balde só enche quando a chuva de uma hora passa da capacidade. A maior chuva horária medida nos pontos foi de 48,7 a 61,3 mm; a capacidade é 60,1 mm/h. O passo de 1 h esconde o pico de 10 min (até 26,9 mm em 10 min, ou 161 mm/h).
- O Derby não alaga em nenhum cenário: a área de contribuição do ponto no terreno é de 62 m².
- O M1a não tem maré (o M1b não roda, ADR 0009). No rótulo, a maré não tem papel.
- Hora sem chuva medida não vira hora seca: ela e as 6 h seguintes ficam sem rótulo (0,2% a 4,2% das horas, conforme o ponto).

O gerador sintético do v3 (107 eventos por ponto por ano, sem atraso físico) não foi recalibrado: o plano o troca pelo balde, e o balde central dá zero. **Os dois extremos falham em sentidos opostos: 107 contra 0, para um valor declarado de 4.**

## 4.2 O experimento que responde P1

Pronto em `mapi_ai/v4/experiment.py`, com tudo o mais igual nos três braços:

| Braço | Entradas |
|---|---|
| v4 sem previsão de chuva | as do v3 (janelas de chuva, maré, hora e mês) mais o terreno do ponto (cota da rua sobre o fundo, profundidade da depressão, área de contribuição) |
| v4 com previsão de chuva | as anteriores mais quantis de 50%, 80% e 95% e probabilidades de MÉDIA e FORTE em 1, 3 e 6 h, **fora da amostra** |
| v4 com a chuva futura verdadeira (teto) | as do primeiro mais a chuva medida de `(t, t+1h]`, `(t, t+3h]`, `(t, t+6h]` e a maior hora |

Critérios iguais aos do v3: família escolhida na validação, Platt se melhora na validação, limiares por orçamento de dias de alerta na série contínua, teste lido uma vez. Antecedência com IC por bootstrap de evento.

**Resultado no cenário central: não treinável** (0 evento em treino, validação e teste; `experiment_p1.json`).

### Faixa pré-registrada de capacidade (`--sensibilidade`, aprovada pelo Lucas em 2026-11-16)

> **CONDICIONAL a um parâmetro sem fonte** (capacidade da galeria `f`, `PARAMETROS-MODELO-FISICO.md`). RÓTULO DO MODELO FÍSICO (balde M1a), não alagamento observado. Nenhum `f` foi escolhido para acertar a frequência: a faixa inteira é reportada.

Eventos do balde (4 pontos, treino / validação / teste; mínimo para treinar: 20 / 5 / 5):

| f | capacidade (mm/h) | eventos | por ponto por ano (aprox.) | treinável |
|---|---|---|---|---|
| 1,0 (central) | 60,1 | 0 / 0 / 0 | 0 | não |
| 0,8 | 48,1 | 1 / 0 / 0 | 0,05 | não |
| 0,6 | 36,1 | 7 / 2 / 0 | 0,45 | não |
| 0,4 | 24,0 | 41 / 5 / 5 | 2,5 | sim (no limite) |
| 0,2 | 12,0 | 162 / 29 / 30 | 11 | sim |

A frequência declarada (~3,6 por ano, cidade inteira) cai entre f = 0,4 e f = 0,6. Isso é registro, não calibração.

Três braços no teste, família `xgboost` nos dois cenários (escolhida na validação, IC da diferença acima de 0). Orçamento **alto** (~9 a 11 dias de alerta por ponto por ano):

| f | Braço | PR-AUC teste | detectados | antecedência mediana (IC95) | p10 | ≥ 1 h | falsos por evento |
|---|---|---|---|---|---|---|---|
| 0,4 | sem previsão | 0,198 | 4 / 5 | 0 min (0–360) | 0 | 40% | 7,0 |
| 0,4 | com previsão | 0,394 | 4 / 5 | 180 min (0–360) | 0 | 60% | 7,6 |
| 0,4 | chuva futura verdadeira | 0,944 | 5 / 5 | 360 min (360–360) | 360 | 100% | 6,8 |
| 0,2 | sem previsão | 0,467 | 28 / 30 | 0 min (0–0) | 0 | 7% | 0,33 |
| 0,2 | com previsão | 0,506 | 27 / 30 | **0 min (0–0)** | 0 | 7% | 0,33 |
| 0,2 | chuva futura verdadeira | 0,996 | 28 / 30 | 360 min (300–360) | 60 | 93% | 0,00 |

Orçamento **médio** (~41 a 48 dias de alerta por ponto por ano), f = 0,2: sem previsão 180 min (0–270), p10 0; **com previsão 240 min (90–300)**, p10 54; teto 360 min (360–360). Em f = 0,4 os três braços dão mediana de 360 min, com IC de 0 a 360 (5 eventos) e ~35 falsos por evento.

Leitura:

- **O teto é 360 min por construção**: o horizonte é de 6 h, e o aviso só conta a partir de `início − 6 h`.
- **A previsão de chuva sobe a PR-AUC** (0,198 → 0,394 em f = 0,4; 0,467 → 0,506 em f = 0,2), **mas no orçamento alto não muda a antecedência** no único cenário com eventos suficientes (f = 0,2: 0 min nos dois braços).
- No orçamento médio ganha cerca de 1 h de mediana (180 → 240 min), pagando 45 a 48 dias de alerta por ponto por ano. Isso é ~12 vezes a frequência declarada de eventos.
- Em f = 0,4 o teste tem 5 eventos: os IC vão de 0 a 360 min e não sustentam nenhuma conclusão.
- **A diferença entre "com previsão" e "teto" é quase toda a antecedência possível**: a previsão `rain_v1` entrega pouco do que uma previsão perfeita daria, coerente com a Fase 3 (o modelo de chuva não prevê o início).

## Limitações

- O terreno identifica o ponto: com 4 pontos, as 3 entradas de terreno são 4 valores distintos.
- Com passo horário, o rótulo responde na mesma hora da chuva: o atraso físico só aparece em chuva de várias horas.
