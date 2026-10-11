#!/usr/bin/env bash
# Reproduz a cadeia da avaliação F1 (escores -> resultado) com um comando por modo. F5-ACAD-02.
#
#   avaliacao/reproduzir.sh congelado   offline, ~1 min. Parte dos arquivos de entrada com hash registrado.
#   avaliacao/reproduzir.sh fontes      parte do BANCO local (extração -> escores -> resultado). Ver o aviso abaixo.
#
# Variáveis: AVALIACAO_F1_DIR  pasta com entradas/ e as saídas congeladas (padrão: ../avaliacao-f1 ao lado do repo)
#            SAIDA             onde escrever (padrão: avaliacao/saida-repro/<modo>)
#            PSQL              comando do psql, só no modo fontes (padrão: docker exec -i mapi-db psql -U mapi_user -d tide_db)
# Semente: 20261011, fixa no código (gerar_escores.SEED e avaliar_h1_h2.SEMENTE); PYTHONHASHSEED=0 aqui.
# Não roda análise nova: só refaz o que já foi feito e CONFERE os hashes. Divergência => falha (código de saída != 0).
set -euo pipefail
export PYTHONHASHSEED=0 LC_ALL=C
AVAL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$AVAL")"
MODO="${1:-}"
BASE="${AVALIACAO_F1_DIR:-$REPO/../avaliacao-f1}"
SAIDA="${SAIDA:-$AVAL/saida-repro/$MODO}"
VENV="${REPRO_VENV:-$REPO/.venv-repro}"

H_ESCORES=4615d416e788729a7e5d5b34b1e4df8abff84cfc7e84a79860f14ed2e826b1a5
H_RESULTADO=dcf020cb57a3d1cce5e4a7c0efc36db0fc47d4841759dad5fb58be3e6c9fac14
H_CHUVA=ef6772b73e34b6d73b167af5140005c0f7a5242e488e7ce3c3e023f5cf032e76
H_ROTULOS=3753ada8218f77503f403f216d1d73038c63bcab1400379a993e6890872b25d3
H_DUMP=24aa31ced046b85fe3fddda1fd9fe2d86102635f39bf1412a79f489db0e7f2d0
DUMP="snapshots/tide_db-2026-10-10-pos-carga.dump"

passo() { printf '\n== %s\n' "$*"; }
falha() { printf 'FALHA: %s\n' "$*" >&2; exit 1; }
confere() {  # confere ARQUIVO HASH_ESPERADO ROTULO
  local h; h="$(sha256sum "$1" | cut -d' ' -f1)"
  [ "$h" = "$2" ] || falha "$3: SHA-256 $h != esperado $2 ($1)"
  echo "ok  $3  ${h:0:8}…"
}

[ "$MODO" = congelado ] || [ "$MODO" = fontes ] || { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2; }
mkdir -p "$SAIDA"

passo "0. Ambiente (versões travadas)"
PY="$(command -v python3.12 || command -v python3 || true)"
[ -n "$PY" ] || falha "python3.12 não encontrado"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,12) else 1)' || falha "precisa de Python 3.12.x (metadata.json registra 3.12.3); achei $("$PY" -V)"
if [ ! -x "$VENV/bin/python" ]; then
  "$PY" -m venv "$VENV" || falha "não consegui criar o venv (Debian/Ubuntu: apt install python3.12-venv) ou aponte REPRO_VENV para um venv existente"
fi
# confere as versões travadas pelo próprio Python (não depende do pip); instala só se faltar algo e houver pip
confere_pacotes() {
  "$VENV/bin/python" - "$AVAL/requirements.txt" "$SAIDA/pacotes-instalados.txt" <<'PYEOF'
import sys, importlib.metadata as m
pins = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
inst = sorted(f"{d.metadata['Name']}=={d.version}" for d in m.distributions())
open(sys.argv[2], "w").write("\n".join(inst) + "\n")
low = {x.lower().replace("_", "-") for x in inst}
falta = [p for p in pins if p.lower().replace("_", "-") not in low]
if falta:
    print("faltam/divergem:", ", ".join(falta)); sys.exit(1)
PYEOF
}
if ! confere_pacotes >/dev/null 2>&1; then
  "$VENV/bin/python" -m pip --version >/dev/null 2>&1 || falha "pacotes fora de requirements.txt e o venv não tem pip"
  "$VENV/bin/python" -m pip install -q --disable-pip-version-check -r "$AVAL/requirements.txt"
  confere_pacotes || falha "versão fora do requirements.txt (veja $SAIDA/pacotes-instalados.txt)"
fi
echo "ok  $("$VENV/bin/python" -V), pacotes conforme requirements.txt (lista completa em $SAIDA/pacotes-instalados.txt)"

passo "1. Hashes registrados"
( cd "$AVAL/dados" && sha256sum -c --quiet HASHES.sha256 ) || falha "avaliacao/dados diverge de dados/HASHES.sha256"
echo "ok  normal INMET (cef4a97f…), rótulos do git (3753ada8…) e SQL"

if [ "$MODO" = congelado ]; then
  [ -d "$BASE/entradas" ] || falha "entradas congeladas não encontradas em $BASE (defina AVALIACAO_F1_DIR)"
  ENT="$BASE/entradas"
  ( cd "$BASE" && sha256sum -c --quiet "$AVAL/HASHES.sha256" ) || falha "entradas/saídas congeladas divergem de HASHES.sha256"
  echo "ok  chuva ef6772b7…, terreno 114ddd32…, curva 91c6945e…, escores 4615d416…, resultado dcf020cb…"
  ROT="$AVAL/dados/rotulos_apendice_b.csv"
else
  ENT="$SAIDA/entradas"; mkdir -p "$ENT"
  cat <<AVISO

  O QUE O MODO fontes DEPENDE DE FORA DO GIT:
   - O banco local tide_db com a série OPEN_METEO_ARCHIVE de 2021-04-01 a 2025-01-01. Restaure o dump pós-carga:
       MAPI/$DUMP  (SHA-256 ${H_DUMP:0:8}…, ${H_DUMP})
       (banco vazio, TimescaleDB 2.30.2; timescaledb_pre_restore(); pg_restore --no-owner; timescaledb_post_restore();
        ver docs/snapshots/2026-10-10-troca-de-imagem.md no repo da API)
   - O backfill de 2021-02 a 2021-09 NÃO está nesse dump. Sem ele a chuva de 2021-04 a 2021-09 falta e o hash de
     entrada diverge. Refaça pelo endpoint de runs (docs/INGESTAO.md), um run por vez, esperando o fim de cada um:
       POST /api/admin/ingestion/runs {"source":"OPEN_METEO_ARCHIVE","mode":"REFETCH","from":"2021-03-31T00:00:00Z","to":"2021-07-01T00:00:00Z"}
       POST /api/admin/ingestion/runs {"source":"OPEN_METEO_ARCHIVE","mode":"REFETCH","from":"2021-07-01T00:00:00Z","to":"2021-10-01T00:00:00Z"}
     (runs originais: 902c2b09-b239-4418-8b24-c93030089306 e 8cba199e-814b-4e03-8a60-e96dc50e794f, kind BACKFILL)
   - O Open-Meteo archive é reanálise e pode ser revisada: o refetch pode NÃO dar os mesmos valores. Se der diferença,
     é achado, não erro do script; o modo congelado continua sendo a âncora.
   - O terreno (ponto_terreno.csv, curva) e o xlsx do INMET NÃO são refeitos aqui: vêm de $BASE/entradas (terreno exige
     os zips do PE3D, ainda sem termo; ver tools/terreno/README.md no repo da API) e do JSON versionado.

AVISO
  passo "2. Extração do banco"
  PSQL="${PSQL:-docker exec -i mapi-db psql -U mapi_user -d tide_db}"
  export PSQL
  $PSQL -At -c "select 1" >/dev/null || falha "sem acesso ao banco (suba o postgres do compose com -p projeto-mapi-api)"
  FALTAM="$($PSQL -At -c "select count(*) from (select station_id, count(*) n from observation where source='OPEN_METEO_ARCHIVE' and variable='precipitation' and status='ACTIVE' and ts_utc between '2021-04-01 00:00+00' and '2025-01-01 03:00+00' group by 1) t where n < 32908")"
  [ "$FALTAM" = 0 ] || echo "AVISO: $FALTAM estação(ões) com menos de 32 908 horas no intervalo: backfill 2021-02→09 ausente?"
  "$AVAL/extrair_entradas.sh" "$ENT/chuva_open_meteo_archive.csv" "$SAIDA/rotulos_apendice_b.csv"
  cp "$BASE/entradas/ponto_terreno.csv" "$BASE/entradas/curva_cota_volume.csv" "$ENT/"
  DIVERGE=0
  if ! cmp -s "$ENT/chuva_open_meteo_archive.csv" "$BASE/entradas/chuva_open_meteo_archive.csv"; then
    DIVERGE=1
    echo "DIVERGE: chuva extraída != chuva congelada (ef6772b7…). Linhas diferentes:"
    diff <(cat "$BASE/entradas/chuva_open_meteo_archive.csv") "$ENT/chuva_open_meteo_archive.csv" | grep -c '^>' || true
  else echo "ok  chuva extraída idêntica a ef6772b7…"; fi
  cmp -s "$SAIDA/rotulos_apendice_b.csv" "$AVAL/dados/rotulos_apendice_b.csv" \
    && echo "ok  rótulos extraídos idênticos a 3753ada8…" || { DIVERGE=1; echo "DIVERGE: rótulos extraídos != 3753ada8…"; }
  ROT="$SAIDA/rotulos_apendice_b.csv"
fi

passo "3. Escores (gerar_escores, semente 20261011)"
cd "$REPO"
"$VENV/bin/python" -m avaliacao.gerar_escores --entradas "$ENT" --normal "$AVAL/dados/inmet_normal_82900_prec.json" --saida "$SAIDA"
if [ "$MODO" = congelado ]; then confere "$SAIDA/escores_cegos.csv" "$H_ESCORES" "escores_cegos.csv"
else
  [ "$(sha256sum "$SAIDA/escores_cegos.csv" | cut -d' ' -f1)" = "$H_ESCORES" ] && echo "ok  escores_cegos.csv 4615d416…" \
    || { DIVERGE=1; echo "DIVERGE: escores_cegos.csv != 4615d416… (esperado se a chuva divergiu)"; }
fi

passo "4. Resultado (avaliar_h1_h2, semente 20261011, 10 000 réplicas)"
"$VENV/bin/python" -m avaliacao.avaliar_h1_h2 "$SAIDA/escores_cegos.csv" "$ROT" "$SAIDA/resultado_etapa2.json"
if [ "$MODO" = congelado ]; then
  confere "$SAIDA/resultado_etapa2.json" "$H_RESULTADO" "resultado_etapa2.json"
  cmp -s "$SAIDA/resultado_etapa2.json" "$BASE/resultado_etapa2.json" && echo "ok  byte a byte igual ao arquivo congelado"
  passo "5. Teste de regressão (código contra a saída congelada f6ddacf; não mede acerto)"
  AVALIACAO_F1_DIR="$BASE" "$VENV/bin/python" -m pytest avaliacao -q
  printf '\nCONGELADO: tudo reproduzido (4615d416…, dcf020cb…). Saídas em %s\n' "$SAIDA"
else
  [ "$(sha256sum "$SAIDA/resultado_etapa2.json" | cut -d' ' -f1)" = "$H_RESULTADO" ] && echo "ok  resultado_etapa2.json dcf020cb…" \
    || { DIVERGE=1; echo "DIVERGE: resultado_etapa2.json != dcf020cb…"; }
  if [ "$DIVERGE" = 0 ]; then printf '\nFONTES: cadeia banco -> resultado reproduziu os hashes. Saídas em %s\n' "$SAIDA"
  else printf '\nFONTES: houve divergência (acima). Isso descreve a mudança das fontes; não invalida o modo congelado. Saídas em %s\n' "$SAIDA"; exit 3; fi
fi
