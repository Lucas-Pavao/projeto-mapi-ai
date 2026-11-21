# Linha de base do alerta por regressão (Etapa 0)

Data: 2026-11-21 · Base: `projeto-mapi-ai` @ `c686d13` (main), antes de qualquer mudança.

## Suíte de testes

Comando: `.venv/bin/python -m pytest tests avaliacao/tests -q -p no:cacheprovider`

| Momento | Resultado |
|---|---|
| Antes das mudanças (`c686d13`) | **114 passed, 0 skipped**, 6 avisos (depreciação do `httpx` no `TestClient` e `.dict()` do Pydantic) |
| Relatório de 20/11 | 96 passed, 18 skipped |

Os 18 testes pulados no relatório de 20/11 dependem dos dados locais (`data/rain/`, `data/in1165/`). Nesta máquina os dados existem, então os 18 rodam e passam. Não é mudança de código.

A suíte passou antes de qualquer mudança: não houve motivo para o PARE da Etapa 0.

## O que o serviço servia em `c686d13`

- `POST /v1/predict/flood` → `logreg_v3` (`MAPI_MODEL_GENERATION=v3`, padrão). Rótulo sintético, `NOT_VALIDATED`. O `xgb_v2` só responde se a v3 falhar. A LSTM não é alcançada.
- `POST /v1/forecast/rain` → `rain_v1` (alvo medido).
- `GET /health` → `code_version: "unknown"` quando o serviço roda sem a variável `MAPI_CODE_VERSION` (uvicorn direto, ou `docker compose build` sem `scripts/implantar.sh`).

## Correção do `code_version` (Etapa 0.4)

O build já recebia o commit (`Dockerfile`: `ARG GIT_COMMIT` → `MAPI_CODE_VERSION`; `projeto-mapi-api/scripts/implantar.sh` exporta `AI_GIT_COMMIT`). O `"unknown"` aparecia quando o serviço rodava fora desse caminho. Agora `mapi_ai/version.py` segue esta ordem:

1. a variável do build;
2. sem ela (ou com o valor-sentinela `unknown...`), `git rev-parse HEAD` do repositório;
3. só sem nenhum dos dois, `unknown-build-without-git`, explícito.

O `/health` diz de onde veio o valor (`code_version_source`: `env`, `git` ou `none`). Teste: `tests/test_flood_forecast.py::test_health_traz_commit_real`.

## Depois da Etapa 1

`130 passed` (os 114 de antes e 16 novos em `tests/test_flood_forecast.py`).
