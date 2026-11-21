#!/usr/bin/env bash
# Reproduz o alerta de alagamento por regressão (Etapa 4.1) só com os arquivos de data/ e models/v4/rain.
# 1. regenera data/rain/dataset.pkl.gz a partir de data/rain/ped_10min.csv.gz (versionado);
# 2. refaz a validação (reports/regressao/validacao.json);
# 3. reconstrói o alerta de CIN_UFPE em 2026-05-01 e confere o SHA-256 com reports/regressao/reproducao.json,
#    e compara o caminho do serviço com a reconstrução num instante do dia.
# Uso: scripts/reproduzir_regressao.sh   (sai com erro se o hash divergir)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
"$PY" -c "from mapi_ai.rain import dataset as D; D.build()"
"$PY" -m mapi_ai.flood.validate > /dev/null
"$PY" -m mapi_ai.flood.reproduce
sha256sum reports/regressao/validacao.json
