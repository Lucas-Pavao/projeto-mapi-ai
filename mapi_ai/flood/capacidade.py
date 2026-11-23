"""
Capacidade efetiva por ponto (onda 16, Fase 4.1; projeto-mapi-api/docs/PRE-REGISTRO-CAPACIDADE-EFETIVA.md) e perfil da chuva
prevista (Fase 4.2; docs/PRE-REGISTRO-PERFIL-CHUVA.md).

Regra do nível fixada no pré-registro: ALERTA se a lâmina q50 chega a 15 cm com Q_max; ATENÇÃO se a q95 chega com Q_min.
Ponto sem faixa (fora de data/capacidade/capacidade_efetiva.json): capacidade de projeto da IDF (TR do pedido) nas duas pontas.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mapi_ai import bucket10 as B10

ARQUIVO = Path(__file__).resolve().parents[2] / "data" / "capacidade" / "capacidade_efetiva.json"
# calibração inversa adotada (Fase 4.3; docs/PRE-REGISTRO-CALIBRACAO-INVERSA.md): {slug: kappa}; vazio = regra da faixa
CALIBRACAO = Path(__file__).resolve().parents[2] / "data" / "capacidade" / "calibracao_adotada.json"


@dataclass(frozen=True)
class Faixa:
    q_min_m3s: float
    q_max_m3s: float
    base: str
    area_m2: float | None = None
    kappa: float | None = None

    def q(self, kappa: float) -> float:
        return float(self.q_min_m3s ** (1 - kappa) * self.q_max_m3s ** kappa)


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def faixa(slug: str | None, path: Path | None = None, cal_path: Path | None = None) -> Faixa | None:
    if not slug:
        return None
    p = _load(path or ARQUIVO).get("pontos", {}).get(slug)
    if not p:
        return None
    k = _load(cal_path or CALIBRACAO).get("pontos", {}).get(slug, {}).get("kappa")
    return Faixa(float(p["q_min_m3s"]), float(p["q_max_m3s"]), p["base"], p.get("area_m2"), k)


def params(f: Faixa | None, tr_anos: float, te_min: float) -> tuple[B10.Params10, B10.Params10]:
    """(Params com Q_min, Params com Q_max). Calibrado: Q(kappa) nas duas pontas. Sem faixa: IDF nas duas pontas."""
    base = B10.Params10(tr_anos=tr_anos, te_min=te_min)
    if f is None:
        return base, base
    if f.kappa is not None:
        q = f.q(f.kappa)
        return base.com(q_cap_m3s=q), base.com(q_cap_m3s=q)
    return base.com(q_cap_m3s=f.q_min_m3s), base.com(q_cap_m3s=f.q_max_m3s)


class Perfil:
    """Série de chuva prevista no horizonte. Com o pico previsto (rain_v1p): pico no passo do meio, limitado ao acumulado, e o
    resto por igual; sem ele, o acumulado por igual (como antes)."""

    def __init__(self, peaks: dict | None):
        self.peaks = peaks

    @property
    def metodo(self) -> str:
        return ("pico de 10 min previsto (rain_v1p) no passo do meio; o resto do acumulado por igual" if self.peaks
                else "acumulado espalhado por igual nos passos de 10 min")

    def serie(self, acc_mm: float, h: int, s: str = "q50") -> np.ndarray:
        n = int(h * 6)
        acc = max(float(acc_mm), 0.0)
        if not self.peaks:
            return np.full(n, acc / n)
        pk = min(max(float(self.peaks[s][h]), 0.0), acc)
        rest = (acc - pk) / (n - 1) if n > 1 else 0.0
        out = np.full(n, rest)
        out[(n - 1) // 2] = pk
        return out
