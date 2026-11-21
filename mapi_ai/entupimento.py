"""
Entupimento da galeria e limpeza recente (prompt 3; api docs/MODELO-DISPONIBILIDADE-GALERIA.md, seção 8, e
docs/PRE-REGISTRO-ENTUPIMENTO-156.md).

Ideia (aprovada pelo Lucas em 2026-11-21): se a galeria foi limpa há pouco, a chance de estar entupida é menor, e a chance
de alagar também. Dois estados:
    livre    -> capacidade cheia (f = 1)
    entupida -> capacidade f_E (padrão 0; faixa 0 a 0,5, [sem fonte])
P(entupida | t dias desde a última limpeza) = 1 - exp(-lambda_re * t), com lambda_re = taxa de reentupimento medida na
Central 156 do Recife: tempo entre uma limpeza ATENDIDA e o próximo chamado de limpeza a até 150 m do ponto
(data/entupimento/parametros_156.json). Como o chamado é entupimento PERCEBIDO, lambda_re é limite inferior.
P(lâmina >= h_crit) = (1 - p_E) * 1[h_livre >= h_crit] + p_E * 1[h_entupida >= h_crit]   (mistura, não média de capacidade)

Nenhum valor aqui foi ajustado para acertar alagamento. Este módulo não está ligado ao alerta ao vivo: para isso a API
precisa saber a data da última limpeza de cada ponto (Dados Vivos da 156), o que é um passo separado.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from mapi_ai import bucket10 as B

PARAMETROS = Path(__file__).resolve().parents[1] / "data" / "entupimento" / "parametros_156.json"
MIN_LIMPEZAS = 10      # abaixo disso a taxa do ponto não é estimada (Imbiribeira tem 4): sem dado, nunca um valor inventado
F_ENTUPIDA = 0.0


def p_entupida(t_dias: float | None, lambda_re_por_dia: float | None) -> float | None:
    """Probabilidade de a galeria estar entupida t dias depois da última limpeza. None se faltar dado."""
    if t_dias is None or lambda_re_por_dia is None:
        return None
    if t_dias < 0:
        raise ValueError("t_dias negativo: a limpeza não pode ser depois de agora")
    return 1.0 - math.exp(-lambda_re_por_dia * t_dias)


def lambda_do_ponto(slug: str, caminho: Path = PARAMETROS) -> float | None:
    """Taxa de reentupimento do ponto (150 m), ou None se houver menos de MIN_LIMPEZAS limpezas casadas."""
    p = json.loads(caminho.read_text(encoding="utf-8"))["pontos"].get(slug)
    if not p or not p.get("lambda_re_por_dia_ponto") or int(p.get("limpezas_ponto") or 0) < MIN_LIMPEZAS:
        return None
    return float(p["lambda_re_por_dia_ponto"])


def prob_alagar(P_mm_passo: np.ndarray, t: B.Terreno, p: B.Params10, t_dias_desde_limpeza: float | None,
                lambda_re_por_dia: float | None, f_entupida: float = F_ENTUPIDA) -> dict:
    """
    Roda o balde nos dois estados e mistura pela probabilidade de entupimento.
    Sem dado de limpeza (t ou lambda None): devolve prob = None e só a lâmina com galeria livre (o comportamento de hoje).
    """
    livre = B.simular_passos(P_mm_passo, t, p, f=1.0)
    ent = B.simular_passos(P_mm_passo, t, p, f=f_entupida)
    h_l, h_e = float(livre["h_mm"][-1]), float(ent["h_mm"][-1])
    pe = p_entupida(t_dias_desde_limpeza, lambda_re_por_dia)
    out = {"h_livre_mm": h_l, "h_entupida_mm": h_e, "p_entupida": pe, "f_entupida": f_entupida}
    if pe is None:
        out["prob_lamina_ge_hcrit"] = None
        return out
    out["prob_lamina_ge_hcrit"] = (1 - pe) * (h_l >= p.hcrit_mm) + pe * (h_e >= p.hcrit_mm)
    return out
