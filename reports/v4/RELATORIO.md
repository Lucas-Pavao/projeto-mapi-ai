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

### Decisão que depende de você

Para o experimento ter evento, é preciso um destes caminhos:

1. Rodar a faixa pré-registrada de capacidade (`--sensibilidade`). Cenários com evento em todas as partes: provavelmente f = 0,4 e f = 0,2. A resposta sai **condicional** a um parâmetro sem fonte.
2. Rodar o balde em passo de 10 min com a chuva de 10 min. É mudar o modelo físico (a capacidade vem de uma IDF de 60 min): pede emenda ao ADR 0014.
3. Esperar o cadastro de microdrenagem (pedido por LAI à EMLURB, já proposto em `PARAMETROS-MODELO-FISICO.md`).

Recomendo a 1, com a resposta dada como faixa e a etiqueta de condicional.

## Limitações

- O terreno identifica o ponto: com 4 pontos, as 3 entradas de terreno são 4 valores distintos.
- Com passo horário, o rótulo responde na mesma hora da chuva: o atraso físico só aparece em chuva de várias horas.
