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


REGIMES = ("TEMPO_DESDE_LIMPEZA", "PERMANENTE")


def pi_entupida_permanente(lambda_re_por_dia: float | None, dias_entupida_media: float | None) -> float | None:
    """
    Regime permanente (emenda 1 do pré-registro da 156, E4): renovação alternada, livre por 1/lambda_re em média e
    entupida por D_E em média (do chamado à limpeza). pi_E = D_E / (1/lambda_re + D_E). None se faltar dado.
    """
    if lambda_re_por_dia is None or dias_entupida_media is None or lambda_re_por_dia <= 0 or dias_entupida_media < 0:
        return None
    return dias_entupida_media / (1.0 / lambda_re_por_dia + dias_entupida_media)


def p_entupida_regime(regime: str | None, t_dias: float | None, lambda_re_por_dia: float | None,
                      dias_entupida_media: float | None) -> float | None:
    """TEMPO_DESDE_LIMPEZA (padrão, o de antes) usa t; PERMANENTE usa pi_E. Sem dado, None (nunca um número inventado)."""
    if regime == "PERMANENTE":
        return pi_entupida_permanente(lambda_re_por_dia, dias_entupida_media)
    return p_entupida(t_dias, lambda_re_por_dia)


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


def _alaga(av: dict, nome: str | None, hcrit: float) -> bool:
    """Lâmina >= h_crit agora (nome None) ou em algum passo da projeção `nome` (q50/q80)."""
    if nome is None:
        return av["h_now_mm"] >= hcrit
    pr = av["projection"].get(nome)
    return av["h_now_mm"] >= hcrit or (pr is not None and pr["minutes_to_hcrit"] is not None)


def avaliar_com_limpeza(P_medido: np.ndarray, t: B.Terreno, p: B.Params10, q50: float | None, q80: float | None,
                        dias_desde_limpeza: float | None, lambda_re_por_dia: float | None,
                        f_entupida: float = F_ENTUPIDA, f_medido: np.ndarray | None = None,
                        f_projecao: np.ndarray | None = None, regime: str | None = None,
                        dias_entupida_media: float | None = None) -> dict | None:
    """
    Bloco "limpeza" da resposta do /v1/bucket/point10. None se faltar dado (comportamento de antes).
    prob_* = (1 - p_E) * 1[galeria livre alaga] + p_E * 1[galeria entupida alaga], agora e até 60 min (q50, q80).
    regime PERMANENTE (limpeza desconhecida, emenda 1 E3/E4): p_E = pi_E, sem dias desde a limpeza.
    """
    pe = p_entupida_regime(regime, dias_desde_limpeza, lambda_re_por_dia, dias_entupida_media)
    if pe is None:
        return None
    livre = B.avaliar(P_medido, t, p, q50, q80, f_medido=f_medido, f_projecao=f_projecao)
    ent = B.avaliar(P_medido, t, p, q50, q80, i_cap_mm_h=f_entupida * p.i_cap_mm_h, f_medido=f_medido, f_projecao=f_projecao)
    out = {"regime": "PERMANENTE" if regime == "PERMANENTE" else "TEMPO_DESDE_LIMPEZA",
           "dias_desde_limpeza": None if regime == "PERMANENTE" else float(dias_desde_limpeza),
           "dias_entupida_media": dias_entupida_media, "lambda_re_por_dia": float(lambda_re_por_dia),
           "p_entupida": pe, "f_entupida": f_entupida, "h_now_entupida_mm": ent["h_now_mm"],
           "projection_entupida": ent["projection"]}
    for chave, nome in (("prob_hcrit_agora", None), ("prob_hcrit_60min_q50", "q50"), ("prob_hcrit_60min_q80", "q80")):
        if nome is not None and livre["projection"].get(nome) is None:
            out[chave] = None
            continue
        out[chave] = (1 - pe) * _alaga(livre, nome, p.hcrit_mm) + pe * _alaga(ent, nome, p.hcrit_mm)
    return out

