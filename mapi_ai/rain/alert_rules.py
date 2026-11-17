"""
Etapa 3.1 do prompt do alerta de 20 min: regras de liga e desliga do rain_alert, escolhidas na VALIDAÇÃO da simulação
do dado como chega (cenário estável, rain_v1 de 1 h, ciclos 1, 4 e 7 min depois da âncora). O teste só é lido no fim.
Uso: python -m mapi_ai.rain.alert_rules

Máquina de estados igual à da API (RainAlertRules.java), em passos de 10 min:
- renova se p >= L_liga ou a chuva medida em 10 min passa da galeria (20,9 mm, IDF TR 5, Te 10 min);
- ligado e L_desliga <= p < L_liga: continua, sem renovar (histerese);
- desliga: CANCELA na hora se a chuva dos últimos 30 min < 0,2 mm; senão DESLIGA, segurado pela duração mínima;
- EXPIRA sem renovação por 60 min;
- agrupamento: episódios separados por menos de 60 min seguidos de chuva < 0,2 mm (em 10 min) são o mesmo evento.

REGRA DE ESCOLHA (escrita antes de rodar):
1. L_desliga em {1,0; 0,8; 0,6; 0,5; 0,4; 0,3} x L_liga. Fica o que dá MENOS trocas de estado por evento FORTE na
   validação, desde que não perca evento: avisados >= os do sem histerese e antecedência mediana >= a do sem histerese.
   Empate: o maior L_desliga (mais perto de não ter histerese).
2. Com esse L_desliga, duração mínima em {0, 10, 20, 30} min: fica a menor cujo passo seguinte reduz menos de
   0,1 troca por evento (cotovelo), sem perder evento pelo mesmo critério.
3. Expiração 60 min, cancelamento (0,2 mm em 30 min) e agrupamento (60 min) vêm do prompt, não são escolhidos.
"""
import json

import numpy as np
import pandas as pd

from mapi_ai.rain import config as C
from mapi_ai.rain import evaluate as E
from mapi_ai.rain import simulate_v1b as S

CAP_10MIN_MM = 20.9
DRY_MM = 0.2
EXPIRY_STEPS = 6
GROUP_GAP_STEPS = 6
OFF_FACTORS = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
MIN_DUR = (0, 10, 20, 30)


def machine(p, p10, p30, on_thr, off_thr, min_dur_steps):
    """Estado ligado (bool) por passo e número de trocas, para um ponto (arrays em ordem de tempo)."""
    n = len(p)
    on = np.zeros(n, bool)
    state, since, last_renew, changes = False, 0, -10**9, 0
    for i in range(n):
        alerta = p10[i] >= CAP_10MIN_MM if not np.isnan(p10[i]) else False
        renew = alerta or p[i] >= on_thr
        dry = (not np.isnan(p30[i])) and p30[i] < DRY_MM
        new = state
        if renew:
            new = True
            last_renew = i
        elif state and p[i] >= off_thr:
            new = True
        elif state:
            new = False if dry or (i - since) >= min_dur_steps else True
        if new and not renew and i - last_renew >= EXPIRY_STEPS:
            new = False
        if new != state:
            changes += 1
            since = i
        state = new
        on[i] = state
    return on, changes


def evaluate_rules(rows, p_any, p10, p30, ev, on_thr, off_thr, min_dur_steps, h=1):
    n_h = h * 6
    lead, changes, false_raw, false_grp, episodes, groups = [], 0, 0, 0, 0, 0
    for slug, idx in rows.groupby("slug").indices.items():
        idx = np.sort(idx)
        on, ch = machine(p_any[idx], p10[idx], p30[idx], on_thr, off_thr, min_dur_steps)
        changes += ch
        cs = ev[ev.slug == slug].i.to_numpy()
        cross = np.zeros(len(idx), bool)
        cross[cs] = True
        for c in cs:
            w = np.flatnonzero(on[max(0, c - n_h):c])
            lead.append(0.0 if len(w) == 0 else float((c - (max(0, c - n_h) + w[0])) * 10))
        st = np.flatnonzero(on & ~np.r_[False, on[:-1]])
        en = np.flatnonzero(on & ~np.r_[on[1:], False])
        episodes += len(st)
        wet = ~(np.nan_to_num(p10[idx], nan=0.0) < DRY_MM)
        grp = []
        for s, e in zip(st, en):
            if grp:
                gs, ge = grp[-1]
                gap = wet[ge + 1:s]
                dry_run = 0
                longest = 0
                for x in gap:
                    dry_run = 0 if x else dry_run + 1
                    longest = max(longest, dry_run)
                if longest < GROUP_GAP_STEPS:
                    grp[-1] = (gs, e)
                    continue
            grp.append((s, e))
            if not cross[s:min(len(idx), e + n_h + 1)].any():
                false_raw += 1
        for s, e in grp:
            if not cross[s:min(len(idx), e + n_h + 1)].any():
                false_grp += 1
        groups += len(grp)
    lead = np.array(lead)
    ne = max(1, len(lead))
    rng = np.random.default_rng(C.SEED)
    bs = np.median(lead[rng.integers(0, len(lead), size=(C.N_BOOT, len(lead)))], axis=1) if len(lead) else np.array([0.0])
    return {"eventos": int(len(lead)), "avisados": int((lead > 0).sum()), "antecedencia_mediana_min": float(np.median(lead)) if len(lead) else 0.0,
            "antecedencia_ic95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
            "trocas_por_evento": changes / ne, "episodios": int(episodes), "eventos_agrupados": int(groups),
            "falsos_por_evento_antes_do_agrupamento": false_raw / ne, "falsos_por_evento_depois": false_grp / ne}


def run():
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    on_thr = meta["horizons"]["1h"]["alert_threshold_FORTE"]
    import joblib
    model = joblib.load(C.MODELS_DIR / "gb_rede.joblib")
    sims = {o: S.simulated_dataset("estavel", o) for o in S.OFFSETS}
    out = {"regra": __doc__.split("REGRA DE ESCOLHA (escrita antes de rodar):")[1].strip(), "L_liga": on_thr, "partes": {}}
    data = {}
    for part in ("valid", "test"):
        held = {o: d[d.part == part].reset_index(drop=True) for o, d in sims.items()}
        rows = S.truth_rows(held[S.TRAIN_OFFSET])
        p_any = np.max(np.vstack([model.predict(held[o])["pFORTE_1h"].to_numpy() for o in S.OFFSETS]), axis=0)
        last = held[max(S.OFFSETS)]
        data[part] = (rows, p_any, last.p_10.to_numpy(), last.p_30.to_numpy(), E.events(rows, C.THRESHOLDS["FORTE"]))
    # sem histerese (o que a API fazia até aqui) e "sem regra nenhuma" (aviso cru por passo, como no relatório da Fase 3)
    def ev(part, f, md):
        rows, p, p10, p30, evs = data[part]
        return evaluate_rules(rows, p, p10, p30, evs, on_thr, on_thr * f, md // 10)
    base_v = ev("valid", 1.0, 0)
    grid = {f: ev("valid", f, 0) for f in OFF_FACTORS}
    ok = {f: r for f, r in grid.items() if r["avisados"] >= base_v["avisados"] and r["antecedencia_mediana_min"] >= base_v["antecedencia_mediana_min"]}
    f_best = min(ok, key=lambda f: (round(ok[f]["trocas_por_evento"], 6), -f))
    dur = {md: ev("valid", f_best, md) for md in MIN_DUR}
    md_best = MIN_DUR[-1]
    for a, b in zip(MIN_DUR, MIN_DUR[1:]):
        lost = dur[a]["avisados"] < base_v["avisados"] or dur[a]["antecedencia_mediana_min"] < base_v["antecedencia_mediana_min"]
        if not lost and dur[a]["trocas_por_evento"] - dur[b]["trocas_por_evento"] < 0.1:
            md_best = a
            break
    out["validacao"] = {"sem_histerese": base_v, "grade_L_desliga": {str(f): r for f, r in grid.items()},
                        "grade_duracao_minima": {str(m): r for m, r in dur.items()}}
    out["escolha"] = {"L_desliga_fator": f_best, "L_desliga": on_thr * f_best, "duracao_minima_min": md_best,
                      "expiracao_min": 60, "cancelamento": "chuva < 0,2 mm em 30 min e p < L_desliga", "agrupamento_min": 60}
    out["teste"] = {"sem_histerese": ev("test", 1.0, 0), "escolhido": ev("test", f_best, md_best)}
    path = C.ROOT / "reports" / "rain_v1b" / "regras_alerta.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: v for k, v in o.items() if k != "regra"}, indent=1, ensure_ascii=False))
