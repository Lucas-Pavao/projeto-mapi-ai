-- Extração da chuva horária de entrada (F4-IA-13) a partir do banco local. Reproduz byte a byte
-- avaliacao-f1/entradas/chuva_open_meteo_archive.csv (SHA-256 ef6772b73e34b6d73b167af5140005c0f7a5242e488e7ce3c3e023f5cf032e76).
--
-- Entrada: SOMENTE a tabela observation (fonte OPEN_METEO_ARCHIVE, variável precipitation) e flood_points
-- (slug, latitude, longitude, para nomear os pontos). Nenhuma tabela de rótulo é lida.
-- O station_id do archive é "lat,lon" arredondado a 4 casas (ex.: -8.0526,-34.8944 = AGAMENON_DERBY).
-- Intervalo: 2021-04-01 00:00Z a 2025-01-01 03:00Z, ambos inclusive; ts_utc é o FIM do intervalo horário (ADR 0001).
-- Só 4 pontos (JABOATAO_CENTRO fica de fora, como no pré-registro).
-- Saída: CSV com cabeçalho, linhas ordenadas por slug e hora, sem fuso na hora (UTC), LF.
-- Uso (feito por avaliacao/extrair_entradas.sh):  psql ... -f extrair_entradas.sql > chuva_open_meteo_archive.csv
SET TIME ZONE 'UTC';
COPY (
  SELECT f.slug,
         (o.ts_utc AT TIME ZONE 'UTC') AS ts_utc_fim_intervalo,
         o.value AS precipitation_mm
  FROM observation o
  JOIN flood_points f
    ON o.station_id = round(f.latitude::numeric, 4)::text || ',' || round(f.longitude::numeric, 4)::text
  WHERE o.source = 'OPEN_METEO_ARCHIVE'
    AND o.variable = 'precipitation'
    AND o.status = 'ACTIVE'
    AND f.slug IN ('AV_RECIFE_IBURA', 'CIN_UFPE', 'AGAMENON_DERBY', 'MASCARENHAS_IMBIRIBEIRA')
    AND o.ts_utc >= TIMESTAMPTZ '2021-04-01 00:00:00+00'
    AND o.ts_utc <= TIMESTAMPTZ '2025-01-01 03:00:00+00'
  ORDER BY f.slug, o.ts_utc
) TO STDOUT WITH (FORMAT csv, HEADER true);
