"""
Versão do código servido (Etapa 0.4 do alerta por regressão). Ordem:
1. MAPI_CODE_VERSION do build (Dockerfile: ARG GIT_COMMIT; scripts/implantar.sh do projeto-mapi-api exporta o commit);
2. sem a variável (ou com o valor-sentinela "unknown..." de um build sem git), o commit do repositório local
   (`git rev-parse HEAD`), para o serviço rodado direto com uvicorn;
3. sem nenhum dos dois, "unknown-build-without-git", explícito.
"""
from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNKNOWN = "unknown-build-without-git"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def code_version() -> dict:
    """{"commit": str, "dirty": "true"|"false"|"unknown", "source": "env"|"git"|"none"}."""
    env = os.getenv("MAPI_CODE_VERSION", "").strip()
    if env and not env.startswith("unknown"):
        return {"commit": env, "dirty": os.getenv("MAPI_CODE_DIRTY", "unknown"), "source": "env"}
    commit = _git("rev-parse", "HEAD")
    if commit:
        status = _git("status", "--porcelain", "--untracked-files=no")
        return {"commit": commit, "dirty": "unknown" if status is None else str(bool(status)).lower(), "source": "git"}
    return {"commit": UNKNOWN, "dirty": "unknown", "source": "none"}
