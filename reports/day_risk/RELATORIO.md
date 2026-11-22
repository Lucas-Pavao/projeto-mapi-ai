# "Dia de risco": a previsão numérica vence a climatologia? (bloco C do prompt dos horizontes)

Data: 2026-11-22 · Pré-registro: `mapi_ai/dayrisk/config.py` (commit `3d9e5af`, antes de baixar dados e rodar) ·
Números: `reports/day_risk/evaluation.json` · Reprodução: `python -m mapi_ai.dayrisk.fetch --parse && python -m mapi_ai.dayrisk.evaluate`

## Resposta curta

**Não passou no critério.** O "dia de risco" do MAPi **não vai para a tela**. O horizonte do dia fica só com o INMET
(avisos e previsão por município).

Pela regra escrita antes, em H = 12 h e emissão às 06h, o melhor modelo na validação foi o ICON sozinho (BSS 0,071).
O IC 95% dele foi de −0,081 a 0,260 e contém o zero. Por isso a regra escolhe a climatologia. Com a climatologia
escolhida, não há o que adotar. O teste foi olhado uma vez e não muda a decisão.

## O que foi medido

- **Alvo:** evento FORTE no ponto em (t, t+H], H = 6, 12 e 24 h. Evento FORTE = alguma hora cheia com chuva do ponto
  ≥ 25,15 mm (ADR 0017), medida pela rede Cemaden/PED (ADR 0022). Mesma definição de evento do rain_v1.
- **Emissão:** às 06h local, mais um teste com emissões a cada 3 h.
- **Cortes:** treino de 06/02/2024 a 31/01/2025, validação de 01/02 a 31/10/2025, teste de 01/11/2025 a 31/10/2026.
  Cada corte tem uma estação chuvosa inteira.
- **Previsão numérica:** histórico de rodadas do Open-Meteo (`previous-runs-api`) para o ECMWF IFS 0,25°, o ICON
  Global e o `best_match` (o modelo que o rain_v1 usa hoje). Brutos com SHA-256 em `data/day_risk/raw/manifest.json`.
- **Sem vazamento:** só vale a hora h se `h − 24N + 8 h ≤ t`. N é a antecedência da rodada: 1 dia para H = 6 e 12 h, 2
  dias para H = 24 h. As 8 h são o maior atraso de publicação visto (7h05), com folga. O código tem um assert para isso,
  e há teste.
- **Modelos, do mais simples ao mais complexo:** climatologia (mês, e hora nas emissões a cada 3 h), persistência,
  cada modelo numérico sozinho, a média e o máximo dos três, e uma logística com a média, o CAPE, a chuva medida em
  24 h e a persistência.

## Teste (olhado uma vez), emissão às 06h

BSS = Brier skill score contra a climatologia. Entre colchetes, o IC 95% por bootstrap de dia local.

| Modelo | H = 6 h | H = 12 h | H = 24 h |
|---|---|---|---|
| persistência | −0,007 [−0,018; 0,102] | 0,001 [−0,022; 0,119] | 0,005 [−0,040; 0,101] |
| ECMWF IFS 0,25° | 0,008 [−0,001; 0,092] | −0,019 [−0,303; 0,068] | 0,077 [−0,037; 0,184] |
| ICON Global | 0,112 [−0,011; 0,193] | 0,109 [−0,002; 0,195] | −0,006 [−0,258; 0,127] |
| best_match (o atual) | −1,419 [−13,46; 0,165] | −2,136 [−11,95; −0,234] | −0,781 [−2,017; −0,341] |
| média dos três | 0,093 [−1,289; 0,329] | −0,099 [−2,369; 0,380] | −0,169 [−0,658; 0,066] |
| máximo dos três | 0,104 [−0,204; 0,177] | −0,061 [−1,778; 0,304] | −0,353 [−1,022; −0,074] |
| logística completa | −0,004 [−0,017; 0,132] | 0,008 [−0,017; 0,154] | 0,004 [−0,049; 0,127] |

Positivos no teste: 9 (H = 6 h), 10 (H = 12 h) e 28 (H = 24 h) ponto-instantes, em 5, 6 e 14 dias.
PR-AUC, curva de confiabilidade, ECE e o aviso com o limiar da validação estão no `evaluation.json`.

## Quantos eventos havia, e o que isso quer dizer

- No teste há **28 eventos FORTE em 14 dias**, somando os 5 pontos (mínimo pré-registrado: 15). Na emissão das 06h com
  H = 12 h, só 10 ponto-instantes são positivos, e caem em 6 dias.
- Com tão poucos dias de evento, **os intervalos são largos**: quase todos cruzam o zero. Isso pesa tanto quanto
  qualquer número positivo da tabela. Um modelo que acerta um dia a mais ou a menos muda o BSS em décimos.

## O que fica registrado, sem mudar a decisão

- **O ICON chegou perto no teste.** Em H = 12 h às 06h, BSS 0,109, com IC de −0,002 a 0,195. Mas o ICON não foi o
  escolhido pela regra (o IC dele na validação continha o zero), e escolher agora, depois de ver o teste, seria olhar o
  teste duas vezes. Além disso, o Open-Meteo marca o **ICON Global como descontinuado em 30/11/2026**. Não daria para
  servir o modelo avaliado.
- **O `best_match`, a previsão que o rain_v1 usa hoje, fica muito pior que a climatologia** quando é usado sozinho para
  o dia (BSS −2,1 em H = 12 h). A logística ajustada até out/2025 ficou confiante demais no teste. Isso não afeta o
  rain_v1: lá o Open-Meteo é uma entrada entre várias, e o relatório da Fase 3 já mostrava ganho de só 1% a 2%.
- Nas emissões a cada 3 h com H = 6 h, a logística completa teve IC sem o zero (0,023 [0,007; 0,057]). Não é o
  horizonte nem a emissão do critério, e o ganho é pequeno. Fica como pista para a próxima rodada, não como resultado.

## Comparação com os avisos do INMET

Não há período em comum. Os avisos são arquivados desde 16/11/2026 (ADR 0024), e a série medida termina em 31/10/2026.
A comparação fica para quando houver uma estação chuvosa arquivada.

## O que não foi feito

- 850 hPa: o `previous-runs-api` não serve umidade nem vento em 850 hPa nas rodadas anteriores (`relative_humidity_850hPa_previous_day1`
  é recusado). O CAPE vem do `best_match`, o único dos três modelos com CAPE no histórico.
- O ECMWF IFS de 9 km (`ecmwf_ifs`) não tem histórico de rodadas. A avaliação usa o de 0,25°. A coleta ao vivo da API
  grava os dois.
- CPTEC WRF (C3): não foi aprovado na lista do prompt. Não entrou.
- C5 (serviço `POST /v1/forecast/day-risk`): não existe, porque o critério não passou.

`DADOS DA REDE OBSERVACIONAL DO CEMADEN/MCTIC` · Previsão: Open-Meteo (ECMWF/DWD/NOAA), CC BY 4.0.
