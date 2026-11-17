# Avaliação Estratificada — Opção B (ADR 0025, proposta)

Módulo separado dos arquivos congelados da avaliação F1 (`avaliacao/avaliar_h1_h2.py`, `avaliacao/reproduzir.sh`, `avaliacao/HASHES.sha256`).

## Regras implementadas

- **Treino ponderado:** `tier IN ('HIGH', 'MEDIUM')` com `sample_weight`, restrito à chuva medida (`rain_tier IN ('MEASURED_PRIMARY', 'MEASURED_EXTENDED')`).
- **Tabela principal:** exclusivamente `tier == 'HIGH'` e chuva medida (zero contaminação por reanálise).
- **Bronze (`LOW`):** relatório exploratório separado (fora de treino e teste principal).
- **Reanálise (`REANALYSIS_FALLBACK`):** relatório separado; nunca entra na métrica estrita.
- **Estratos vazios:** quando um estrato ou partição tem zero positivos (estado atual em `2026-11-16`, pois todas as 225 ocorrências do banco são `DAY` e estão a ≥ 619 m dos pontos), o script registra explicitamente `"0 positivos no estrato <X> — avaliação não executável"` sem falhar nem inventar rótulos.

## Uso

```bash
python -m avaliacao.opcao_b.avaliar_estratificado dataset_estratificado.csv saida.json
```
