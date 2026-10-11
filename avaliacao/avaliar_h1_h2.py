"""Etapa 2 do pre-registro (F4-IA-15): juncao dos escores cegos com os rotulos e avaliacao de H1 e H2.

Uso: python -m avaliacao.avaliar_h1_h2 ESCORES.csv ROTULOS.csv SAIDA.json [--sintetico SEMENTE]
ROTULOS.csv: linhas "AAAA-MM-DD,n_ocorrencias" (saida da consulta do Apendice B).
Com --sintetico, ignora ROTULOS e sorteia 14 positivos em 2023-2024 (teste do script, sem rotulos reais).
Semente de toda reamostragem: 20261011; 10 000 replicas; IC percentil de 95% (pre-registro, secao 6).
Escore de M1: ordenacao lexicografica (escore=S, depois desempate=R); demais modelos: so o escore.
"""
import json
import math
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEMENTE = 20261011
N_REP = 10_000
INI, FIM = date(2023, 1, 1), date(2024, 12, 31)
SEM_FONTE = ("f", "curva", "snap")  # parametros sem fonte (pre-registro 7)


def episodios(dias, gap=3):
    """Dias a <= gap dias formam um episodio (encadeado). Devolve vetor de ids."""
    ordem = np.argsort([d.toordinal() for d in dias])
    ids = np.zeros(len(dias), int)
    atual = 0
    for k, i in enumerate(ordem):
        if k and (dias[i] - dias[ordem[k - 1]]).days > gap:
            atual += 1
        ids[i] = atual
    return ids


def cp(k, n, alfa=0.05):
    """Clopper-Pearson exato por bissecao sobre a binomial."""
    def cdf(x, p):  # P(X <= x)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x + 1))
    def sf(x, p):   # P(X >= x)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(x, n + 1))
    lo = 0.0
    if k > 0:
        a, b = 0.0, 1.0
        for _ in range(100):
            m = (a + b) / 2
            if sf(k, m) < alfa / 2:
                a = m
            else:
                b = m
        lo = (a + b) / 2
    hi = 1.0
    if k < n:
        a, b = 0.0, 1.0
        for _ in range(100):
            m = (a + b) / 2
            if cdf(k, m) > alfa / 2:
                a = m
            else:
                b = m
        hi = (a + b) / 2
    return lo, hi


def mcnemar_exato(b, c):
    m = b + c
    if m == 0:
        return 1.0
    mn = min(b, c)
    p = 2 * sum(math.comb(m, i) for i in range(mn + 1)) / 2**m
    return min(1.0, p)


def matriz_w(sp, sn):
    """W[i,j] = 1 se sp_i > sn_j, 0,5 se empate, 0 caso contrario; sp, sn sao (n,2) lexicograficos."""
    gt = (sp[:, None, 0] > sn[None, :, 0]) | ((sp[:, None, 0] == sn[None, :, 0]) & (sp[:, None, 1] > sn[None, :, 1]))
    eq = (sp[:, None, 0] == sn[None, :, 0]) & (sp[:, None, 1] == sn[None, :, 1])
    return gt.astype(float) + 0.5 * eq


def pesos(rng, grupos_p, grupos_n):
    """Pesos de bootstrap: positivos por episodio, negativos por bloco (semana ISO)."""
    def amostra(grupos):
        g = np.unique(grupos)
        sorteio = rng.integers(0, len(g), size=(N_REP, len(g)))
        cont = np.zeros((N_REP, len(g)))
        for j in range(len(g)):
            cont[:, j] = (sorteio == j).sum(1)
        idx = np.searchsorted(g, grupos)
        return cont[:, idx]
    return amostra(grupos_p), amostra(grupos_n)


def auc_rep(W, wp, wn):
    num = ((wp @ W) * wn).sum(1)
    return num / (wp.sum(1) * wn.sum(1))


def carregar(escores_csv):
    d = pd.read_csv(escores_csv)
    d["data"] = pd.to_datetime(d["data"]).dt.date
    return d[d.ponto == "CIDADE"].copy()


def serie(d, modelo, variante):
    g = d[(d.modelo == modelo) & (d.variante == variante)].set_index("data").sort_index()
    return g


def vetor(g, dias, lexico):
    e = g.loc[dias, "escore"].to_numpy(float)
    r = g.loc[dias, "desempate"].to_numpy(float) if lexico else np.zeros(len(dias))
    return np.column_stack([e, np.nan_to_num(r)])


def rotulos(rotulos_csv):
    pos = {}
    for ln in open(rotulos_csv):
        ln = ln.strip()
        if ln:
            a, b = ln.split(",")
            pos[date.fromisoformat(a)] = int(b)
    return pos


def conjuntos(pos_dias, d):
    aval = d[d.periodo == "avaliacao"]
    completos = set(aval.loc[aval.horas_no_dia >= 23, "data"])
    p_princ = sorted(x for x in pos_dias if INI <= x <= FIM)
    p_extra = sorted(x for x in pos_dias if x < INI)
    excl = set()
    for x in p_princ:
        excl |= {x, x - timedelta(1), x + timedelta(1)}
    todos = [INI + timedelta(i) for i in range((FIM - INI).days + 1)]
    n_bruto = [x for x in todos if x not in excl]
    n_dias = [x for x in n_bruto if x in completos]
    return p_princ, p_extra, n_bruto, n_dias, sorted(set(todos) - completos)


def categoria(lo, hi):
    return "positivo" if lo > 0 else ("negativo" if hi < 0 else "inconclusivo")


def main(escores_csv, rotulos_csv, saida, sintetico=None):
    d = carregar(escores_csv)
    if sintetico is not None:
        r = np.random.default_rng(int(sintetico))
        todos = [INI + timedelta(i) for i in range((FIM - INI).days + 1)]
        pos = {todos[i]: 1 for i in r.choice(len(todos), 14, replace=False)}
    else:
        pos = rotulos(rotulos_csv)
    p_princ, p_extra, n_bruto, n_dias, incompletos = conjuntos(list(pos), d)
    ep_p = episodios(p_princ)
    p18 = p_extra + p_princ
    ep_18 = episodios(p18)
    sem_n = np.array([f"{x.isocalendar()[0]}-{x.isocalendar()[1]:02d}" for x in n_dias])
    rng = np.random.default_rng(SEMENTE)
    wp, wn = pesos(rng, ep_p, sem_n)
    rng18 = np.random.default_rng(SEMENTE)
    wp18, wn18 = pesos(rng18, ep_18, sem_n)

    def sc(modelo, variante):
        g = serie(d, modelo, variante)
        lex = modelo.startswith("M1")
        return g, lex

    def auc(modelo, variante, P=p_princ, wP=wp, wN=wn):
        g, lex = sc(modelo, variante)
        W = matriz_w(vetor(g, P, lex), vetor(g, n_dias, lex))
        pt = W.mean()
        reps = auc_rep(W, wP, wN)
        return pt, reps, W

    def ic(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    out = {"n_pos_principal": len(p_princ), "n_pos_extra": len(p_extra), "n_neg_bruto": len(n_bruto),
           "n_neg": len(n_dias), "dias_incompletos_avaliacao": [str(x) for x in incompletos],
           "episodios_principal": int(ep_p.max() + 1), "episodios_18": int(ep_18.max() + 1),
           "positivos_principal": [str(x) for x in p_princ], "positivos_extra": [str(x) for x in p_extra]}

    # --- H1 e H2 (principais) e todas as variantes de M1
    a_b2, r_b2, _ = auc("B2", "normal_mensal_INMET_82900")
    a_b3, r_b3, _ = auc("B3", "B3")
    var = ["central"] + []
    m1 = {"central": ("M1a", "central")}
    for v in sorted(d[d.modelo == "M1-sens"].variante.unique()):
        m1[v] = ("M1-sens", v)
    res = {}
    for nome, (m, v) in m1.items():
        a, r, _ = auc(m, v)
        h1 = r - r_b2
        h2 = r - r_b3
        res[nome] = {"auc": a, "auc_ic": ic(r),
                     "d_h1": a - a_b2, "d_h1_ic": ic(h1), "cat_h1": categoria(*ic(h1)),
                     "d_h2": a - a_b3, "d_h2_ic": ic(h2), "cat_h2": categoria(*ic(h2))}
    out["auc_B2"] = [a_b2, ic(r_b2)]
    out["auc_B3"] = [a_b3, ic(r_b3)]
    out["m1"] = res

    # --- classificacao pela regra do protocolo
    def sem_fonte(nome):
        return nome.startswith(("f0", "curva", "snap")) or nome.startswith("canto_")

    def classifica(h):
        cat = res["central"][f"cat_{h}"]
        if cat != "positivo":
            return cat, []
        mudam = [n for n in res if sem_fonte(n) and res[n][f"cat_{h}"] != "positivo"]
        return ("positivo robusto" if not mudam else "positivo condicional (= inconclusivo)"), mudam
    out["class_h1"] = classifica("h1")
    out["class_h2"] = classifica("h2")  # so valido se H1 positivo (regra de parada); marcado no relatorio

    # --- degeneracao (7)
    g, _ = sc("M1a", "central")
    ga = g.loc[n_dias, "alerta"]
    out["degenerado"] = {"alerta_M1a_em_N": int(ga.sum()), "S_zero_em_todos_positivos": bool((g.loc[p_princ, "escore"] == 0).all()),
                         "S_max_N": float(g.loc[n_dias, "escore"].max())}

    # --- secundarias
    sec = {}
    for (m, v), _g in d.groupby(["modelo", "variante"]):
        if m == "B0":
            continue
        a, r, _ = auc(m, v)
        a18, r18, _ = auc(m, v, p18, wp18, wn18)
        sec[f"{m}/{v}"] = {"auc": a, "ic": ic(r), "exclui_05": bool(ic(r)[0] > 0.5 or ic(r)[1] < 0.5),
                           "auc18": a18, "ic18": ic(r18)}
    out["auc_todos"] = sec
    # binarias
    bins = {}
    for m, v in [("M1a", "central"), ("B3", "B3"), ("B3", "B3-24h"), ("B3", "B3-47.4"), ("B3", "B3-55"),
                 ("B3", "B3-50"), ("B3", "B3-70"), ("B2", "normal_mensal_INMET_82900")]:
        g = serie(d, m, v)
        k14 = int(g.loc[p_princ, "alerta"].sum())
        k18 = int(g.loc[p18, "alerta"].sum())
        kn = int(g.loc[n_dias, "alerta"].sum())
        k4 = int(g.loc[p_extra, "alerta"].sum())
        bins[f"{m}/{v}"] = {"k14": k14, "ic14": cp(k14, len(p_princ)), "k18": k18, "ic18": cp(k18, len(p18)),
                            "k4": k4, "kn": kn, "icn": cp(kn, len(n_dias))}
    out["binarias"] = bins
    a1 = serie(d, "M1a", "central")["alerta"]
    a3 = serie(d, "B3", "B3")["alerta"]
    b = int(((a1.loc[p_princ] == 1) & (a3.loc[p_princ] == 0)).sum())
    c = int(((a1.loc[p_princ] == 0) & (a3.loc[p_princ] == 1)).sum())
    bn = int(((a1.loc[n_dias] == 1) & (a3.loc[n_dias] == 0)).sum())
    cn = int(((a1.loc[n_dias] == 0) & (a3.loc[n_dias] == 1)).sum())
    out["mcnemar"] = {"pos": [b, c, mcnemar_exato(b, c)], "neg": [bn, cn, mcnemar_exato(bn, cn)]}
    # P_extra: posto percentil
    extra = {}
    for m, v in [("M1a", "central"), ("B3", "B3"), ("B2", "normal_mensal_INMET_82900")]:
        g, lex = sc(m, v)
        sn = vetor(g, n_dias, lex)
        row = {}
        for x in p_extra:
            W = matriz_w(vetor(g, [x], lex), sn)
            row[str(x)] = float(W.mean())
        extra[f"{m}/{v}"] = row
    out["p_extra_posto"] = extra
    json.dump(out, open(saida, "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    sint = a[a.index("--sintetico") + 1] if "--sintetico" in a else None
    main(a[0], a[1], a[2], sint)
