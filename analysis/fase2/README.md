# Fase 2 — medidas da descoberta da previsão de chuva

Relatório: `projeto-mapi-api/docs/extratos/fase2-previsao-de-chuva-descoberta.md`.

| Script | Mede | Saída |
|---|---|---|
| `lag2.py` | defasagem do início da chuva por setor, com controle deslocado 3 dias | `lag2.json` |
| `lag.py` | persistência no ponto (e a primeira versão da defasagem, sem controle) | `lag_persist.json` |
| `om.py` | previsão arquivada do Open-Meteo contra a chuva do ponto | `om_skill.csv` |

Entradas: `ped10.csv.gz` (export de `observation_current`, fonte CEMADEN_PED: `station_id, epoch, mm`), `ped_st.csv`
(tabela `station` da PED), `om/hist.json` e `om/prev.json` (Open-Meteo, 5 pontos), e `data/in1165/`.
Uso: `python lag2.py <pasta com as entradas> <raiz do projeto-mapi-ai>`.
