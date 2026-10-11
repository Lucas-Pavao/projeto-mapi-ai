# avaliacao/ — avaliação F1 do MAPi (pré-registro)

Código que gera os escores cegos (etapa 1) e avalia H1 e H2 (etapa 2) do `docs/PRE-REGISTRO-AVALIACAO.md` (repo da API).
Esta pasta **reproduz** o que já foi feito. Ela não roda análise nova sobre rótulos e não altera o pré-registro.

## Reproduzir (um comando por modo)

```bash
avaliacao/reproduzir.sh congelado   # offline, ~1 min
avaliacao/reproduzir.sh fontes      # a partir do banco local
```

Pré-requisitos: Python 3.12 (o `metadata.json` registra 3.12.3), `requirements.txt` com versões exatas (numpy 1.26.4,
pandas 2.2.3, psycopg2-binary 2.9.10, pytest 8.3.4; o script cria `.venv-repro/` ou usa `REPRO_VENV`). Semente fixa
`20261011` no código (`gerar_escores.SEED`, `avaliar_h1_h2.SEMENTE`). As entradas congeladas ficam **fora do git**, em
`AVALIACAO_F1_DIR` (padrão: `../avaliacao-f1`, com `entradas/`, `escores_cegos.csv`, `resultado_etapa2.json`).

| Modo | Cadeia | Falha se |
|---|---|---|
| `congelado` | entradas com hash → `gerar_escores` → `avaliar_h1_h2` → `pytest` | qualquer hash abaixo divergir |
| `fontes` | banco → `extrair_entradas.sh` → escores → resultado | sai com código 3 e descreve a divergência (as fontes mudam) |

### Hashes esperados (`HASHES.sha256`, `dados/HASHES.sha256`)

| Artefato | SHA-256 (início) |
|---|---|
| `entradas/chuva_open_meteo_archive.csv` | `ef6772b7…` |
| `entradas/ponto_terreno.csv` / `curva_cota_volume.csv` | `114ddd32…` / `91c6945e…` |
| `dados/inmet_normal_82900_prec.json` | `cef4a97f…` |
| `escores_cegos.csv` | `4615d416…` |
| `dados/rotulos_apendice_b.csv` | `3753ada8…` |
| `resultado_etapa2.json` | `dcf020cb…` |
| Dump pós-carga `MAPI/snapshots/tide_db-2026-10-10-pos-carga.dump` | `24aa31ce…` |

### O que o modo `fontes` depende de fora do git

1. **Banco `tide_db`** com a chuva `OPEN_METEO_ARCHIVE`. Restaurar o dump pós-carga (SHA-256 `24aa31ce…`): banco vazio com
   TimescaleDB 2.30.2, `timescaledb_pre_restore()`, `pg_restore --no-owner`, `timescaledb_post_restore()`
   (`docs/snapshots/2026-10-10-troca-de-imagem.md`). Compose sempre com `-p projeto-mapi-api`.
2. **O backfill 2021-02→09 não está nesse dump.** Foi feito depois, em dois runs (`902c2b09…`, 2021-03-31→07-01; `8cba199e…`,
   2021-07-01→10-01). Para refazer, pelo endpoint de runs (`docs/INGESTAO.md`), um de cada vez, esperando `status` sair de
   `RUNNING`: `POST /api/admin/ingestion/runs` com `{"source":"OPEN_METEO_ARCHIVE","mode":"REFETCH","from":...,"to":...}`.
   Sem ele, `reproduzir.sh fontes` avisa que faltam horas e o hash `ef6772b7…` diverge.
3. **Reanálise revisável:** o refetch do Open-Meteo pode devolver valores diferentes dos de 2026-10-11. Diferença aí é
   achado (RC-17), não defeito do script. A âncora é o modo `congelado`.
4. **Terreno e xlsx do INMET** não são refeitos aqui: o terreno vem de `tools/terreno/run.sh` (repo da API, exige os zips do
   PE3D, ainda sem termo de uso); a normal do INMET está em `dados/` (xlsx de origem SHA-256 `903ba7d9…`).

Verificado em 2026-10-11 no banco local que já tem o backfill: extração idêntica a `ef6772b7…` e rótulos idênticos a
`3753ada8…`. **Não** foi testado a partir de um banco restaurado do dump.

## Arquivos

| Arquivo | Papel |
|---|---|
| `reproduzir.sh` | Roteiro dos dois modos, com conferência de hash a cada etapa |
| `extrair_entradas.sh`, `dados/extrair_entradas.sql` | Chuva horária de entrada a partir do banco (só `observation` e `flood_points`) |
| `dados/rotulos_apendice_b.sql` / `.csv` | Consulta do Apêndice B e sua saída congelada (usada só na etapa 2) |
| `balde.py`, `entradas.py`, `gerar_escores.py` | Etapa 1 cega: modelo M1, baselines e escores (recusa colunas de rótulo) |
| `avaliar_h1_h2.py`, `comparacao_reanalise.py` | Etapa 2 e comparação reanálise × pluviômetro |
| `tests/` | `test_balde.py`, `test_avaliar.py` e a regressão abaixo |

## Teste de regressão (`tests/test_regressao_f6ddacf.py`)

Oito casos de dias (R1–R8 da `PRONTIDAO-ACADEMICA.md` §2). Para cada um, os escores de cada modelo e variante têm de ser
iguais ao recorte congelado da saída do commit `f6ddacf` (`tests/regressao_f6ddacf.json`, de `escores_cegos.csv`).
**Protege contra mudança de código; não mede acerto.** Nunca compara com rótulo, e os dias foram escolhidos sabendo-se
os resultados. R8 foi adaptado: o escore só cobre até 2024-12-31, então vale o primeiro dia sem chuva do período de
aquecimento (2021-04-02), regra que só olha a chuva. Se o modelo mudar de propósito, regenere o JSON com
`tests/congelar_regressao.py`, registre no changelog e suba a `modelVersion`.
