WITH canon AS (
  SELECT o.* FROM flood_occurrence o
  WHERE NOT EXISTS (SELECT 1 FROM flood_occurrence d
                    WHERE d.ckan_record_id = o.ckan_record_id
                      AND d.occurred_from = o.occurred_from AND d.id < o.id))
SELECT (occurred_from AT TIME ZONE 'America/Recife')::date AS dia, count(*) AS ocorrencias
FROM canon
WHERE (occurred_from AT TIME ZONE 'America/Recife')::date BETWEEN DATE '2021-01-01' AND DATE '2024-12-31'
GROUP BY 1 ORDER BY 1;
