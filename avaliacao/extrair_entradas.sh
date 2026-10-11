#!/usr/bin/env bash
# Extrai do banco local as entradas que dependem do banco: a chuva horária e (opcional) a contagem diária de rótulos.
# Uso: avaliacao/extrair_entradas.sh SAIDA_CHUVA.csv [SAIDA_ROTULOS.csv]
# Variável PSQL: comando que abre o psql no banco tide_db. Padrão: o contêiner do compose do MAPI.
#   PSQL="psql -h 127.0.0.1 -p 5433 -U mapi_user -d tide_db"   (senha em PGPASSWORD)
# A consulta de rótulos é a do Apêndice B do pré-registro, só para REPRODUZIR o arquivo já congelado.
set -euo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSQL="${PSQL:-docker exec -i mapi-db psql -U mapi_user -d tide_db}"
[ $# -ge 1 ] || { echo "uso: $0 SAIDA_CHUVA.csv [SAIDA_ROTULOS.csv]" >&2; exit 2; }
$PSQL -q -v ON_ERROR_STOP=1 < "$AQUI/dados/extrair_entradas.sql" > "$1"
echo "chuva: $1 ($(sha256sum "$1" | cut -c1-16)…, $(wc -l < "$1") linhas)"
if [ $# -ge 2 ]; then
  $PSQL -At -F, -q -v ON_ERROR_STOP=1 < "$AQUI/dados/rotulos_apendice_b.sql" > "$2"
  echo "rotulos: $2 ($(sha256sum "$2" | cut -c1-16)…)"
fi
