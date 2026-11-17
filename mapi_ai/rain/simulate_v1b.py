"""
Etapa 2.3 do prompt do alerta de 20 min: o rain_v1 com o dado como chega, e o rain_v1b treinado nessa simulação.
Uso: python -m mapi_ai.rain.simulate_v1b [--cenario estavel|chuva]

1. Entradas simuladas nos 5 anos (availability.features_at_visible), com o atraso sorteado por estação (semente 42).
2. rain_v1 SEM retreinar sobre elas: recall de FORTE, antecedência (IC 95% por bootstrap de evento), falsos por evento,
   nos regimes já chove / ainda não chove. Limiar: o do rain_v1 (escolhido na validação do laboratório).
3. rain_v1b: mesmas entradas e método (gb_rede em 1 h; gb_rede_om em 3 h e 6 h, como o rain_v1), treinado na simulação;
   limiar de maior CSI na VALIDAÇÃO da simulação.
4. Critério fixado antes: fica o rain_v1b se a antecedência mediana dele no teste da simulação for maior que a do
   rain_v1 com IC 95% da diferença (bootstrap pareado por evento) sem o zero; senão fica o rain_v1.
A verdade (evento, alvo) é sempre a série completa da PED. Nada do rain_v1, v3 ou v4 é sobrescrito.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from mapi_ai.rain import availability as A
from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import evaluate as E
from mapi_ai.rain import features as F
from mapi_ai.rain import models as M

OUT_MODELS = C.ROOT / "models" / "rain_v1b"
OUT_REPORT = C.ROOT / "reports" / "rain_v1b"
OUT_DATA = C.ROOT / "data" / "rain_v1b"
DELAYS = OUT_DATA / "live_delays.csv"
CHOSEN = {1: "gb_rede", 3: "gb_rede_om", 6: "gb_rede_om"}
# Ciclos do alerta depois da âncora (passo de 10 min): a cada 3 min, fase de 1 min. O serviço ancora as entradas na
# leitura mais nova (serve.py) e recalcula a cada ciclo com o que já chegou; o aviso conta a partir do ciclo que disparou.
OFFSETS = (1, 4, 7)
TRAIN_OFFSET = 4


def simulated_dataset(scenario: str, offset: int) -> pd.DataFrame:
    path = OUT_DATA / f"dataset_sim_{scenario}_o{offset}.pkl.gz"
    if path.exists():
        return pd.read_pickle(path)
    lab = D.load()
    readings, stations, points = D.load_readings(), pd.read_csv(C.STATIONS), pd.read_csv(C.POINTS)
    delays = pd.read_csv(DELAYS)
    readings["delay_min"] = A.sample_delays(readings, delays, seed=C.SEED, scenario=scenario)
    om_prev = D.load_om(C.OM_PREV)
    times = pd.date_range(C.DATA_START, C.DATA_END, freq=F.SLOT)
    parts = []
    truth_cols = ["slug", "t", "part", "raining", "p_10", "p_60"] + [c for c in lab if c[:2] in ("y_", "ev") or c[:5] == "peak_"]
    for p in points.itertuples():
        x = A.features_at_visible(readings, stations, p.latitude, p.longitude, times, om_prev.get(p.slug),
                                  offset_min=offset).reset_index(drop=True)
        tr = lab[lab.slug == p.slug][truth_cols].reset_index(drop=True)
        assert (tr.t.to_numpy() == times.to_numpy()).all()
        # entradas simuladas; a verdade (p_10, p_60 dos eventos, alvos) fica com o sufixo _true
        d = pd.concat([tr.rename(columns={"p_10": "p_10_true", "p_60": "p_60_true"}), x], axis=1)
        parts.append(d)
        print("simulado", p.slug, "ciclo +", offset, "min", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_pickle(path)
    return df


def truth_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Linhas para evaluate.events/lead_stats: p_10 e p_60 da série COMPLETA (definição do evento)."""
    r = df[["slug", "t", "part", "raining", "p_10_true", "p_60_true"] + [c for c in df if c[:2] in ("y_", "ev") or c[:5] == "peak_"]]
    return r.rename(columns={"p_10_true": "p_10", "p_60_true": "p_60"})


def boot_lead_diff(a: np.ndarray, b: np.ndarray, n=C.N_BOOT, seed=C.SEED):
    """IC 95% da diferença de medianas de antecedência (a - b), pareado por evento."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    v = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.median(a) - np.median(b)), float(np.quantile(v, .025)), float(np.quantile(v, .975))


def lead_vector(rows, pred, h, thr, ev):
    """Antecedência por evento (0 se não avisado), na ordem de ev (para o pareamento)."""
    n = h * 6
    out = []
    a_all = (pred[f"pFORTE_{h}h"].to_numpy() >= thr)
    for e in ev.itertuples():
        g_idx = np.flatnonzero((rows.slug == e.slug).to_numpy())
        a = a_all[g_idx]
        w = np.flatnonzero(a[max(0, e.i - n):e.i])
        out.append(0.0 if len(w) == 0 else float((e.i - (max(0, e.i - n) + w[0])) * 10))
    return np.array(out)


def lead_vector_cycles(rows, preds: dict, h, thr, ev):
    """Antecedência (min) por evento contada do primeiro ciclo que disparou: t do passo + offset do ciclo."""
    n = h * 6
    fire = np.full(len(rows), np.inf)
    for o in sorted(preds):
        hit = (preds[o][f"pFORTE_{h}h"].to_numpy() >= thr) & np.isinf(fire)
        fire[hit] = o
    out = []
    for e in ev.itertuples():
        g_idx = np.flatnonzero((rows.slug == e.slug).to_numpy())
        f = fire[g_idx]
        lo = max(0, e.i - n)
        best = None
        for j in range(lo, e.i):
            if np.isfinite(f[j]):
                best = (e.i - j) * 10 - f[j]
                break
        out.append(0.0 if best is None else float(max(0.0, best)))
    return np.array(out)


def summary(lead: np.ndarray, seed=C.SEED):
    rng = np.random.default_rng(seed)
    bs = np.median(lead[rng.integers(0, len(lead), size=(C.N_BOOT, len(lead)))], axis=1)
    return {"eventos": int(len(lead)), "avisados": int((lead > 0).sum()), "mediana_min": float(np.median(lead)),
            "ic95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "p10_min": float(np.quantile(lead, .1)),
            "p90_min": float(np.quantile(lead, .9))}


def block(rows, pred, h, thr, ev):
    rain, dry = rows.raining.to_numpy(), ~rows.raining.to_numpy()
    e = rows[f"ev_FORTE_{h}h"].to_numpy(dtype=float)
    p = pred[f"pFORTE_{h}h"].to_numpy()
    return {"limiar": thr, "todas": E.prf(e, p, thr), "ja_chove": E.prf(e[rain], p[rain], thr),
            "ainda_nao_chove": E.prf(e[dry], p[dry], thr), "eventos": E.lead_stats(rows, pred, h, thr, ev)}


def run(scenario: str):
    t0 = time.time()
    OUT_REPORT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    sims = {o: simulated_dataset(scenario, o) for o in OFFSETS}
    helds = {o: d[d.part.isin(["valid", "test"])].reset_index(drop=True) for o, d in sims.items()}
    rows = truth_rows(helds[TRAIN_OFFSET])
    V, T = (rows.part == "valid").to_numpy(), (rows.part == "test").to_numpy()
    rt = rows[T].reset_index(drop=True)
    ev_t = E.events(rt, C.THRESHOLDS["FORTE"])
    blob = pd.read_pickle(C.DATA_DIR / "predictions.pkl.gz")
    lab_rows = blob["rows"].reset_index(drop=True)
    labT = (lab_rows.part == "test").to_numpy()
    lab_rt = lab_rows[labT].reset_index(drop=True)
    ev_lab = E.events(lab_rt, C.THRESHOLDS["FORTE"])
    assert len(ev_lab) == len(ev_t), "a verdade da simulação precisa dar os mesmos eventos do laboratório"
    v1 = {n: joblib.load(C.MODELS_DIR / f"{n}.joblib") for n in set(CHOSEN.values())}
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    train = sims[TRAIN_OFFSET][sims[TRAIN_OFFSET].part == "train"].iloc[::C.TRAIN_STEP]
    v1b = {}
    for n in sorted(set(CHOSEN.values())):
        path = OUT_MODELS / f"{n}_{scenario}.joblib"
        if path.exists():
            v1b[n] = joblib.load(path)
        else:
            v1b[n] = M.make(n).fit(train)
            joblib.dump(v1b[n], path, compress=3)
        print("rain_v1b", n, "ok", flush=True)
    out = {"cenario": scenario, "ciclos_min_apos_ancora": list(OFFSETS), "treino_v1b_ciclo": TRAIN_OFFSET,
           "attribution": C.ATTRIBUTION, "eventos_FORTE_no_teste": int(len(ev_t)),
           "eventos_em_dias_distintos": int(ev_t.c.dt.floor("D").nunique()), "horizons": {}}
    for h in C.HORIZONS:
        name = CHOSEN[h]
        thr_v1 = meta["horizons"][f"{h}h"]["alert_threshold_FORTE"]
        p1 = {o: v1[name].predict(helds[o]) for o in OFFSETS}
        p1b = {o: v1b[name].predict(helds[o]) for o in OFFSETS}
        csi_v, thr_v1b = E.best_csi_threshold(rows[V][f"ev_FORTE_{h}h"].to_numpy(dtype=float),
                                              p1b[TRAIN_OFFSET][V][f"pFORTE_{h}h"].to_numpy())
        any1 = pd.concat([p1[o] for o in OFFSETS]).groupby(level=0).max()
        any1b = pd.concat([p1b[o] for o in OFFSETS]).groupby(level=0).max()
        T1 = {o: p1[o][T].reset_index(drop=True) for o in OFFSETS}
        T1b = {o: p1b[o][T].reset_index(drop=True) for o in OFFSETS}
        lab_pred = blob["pred"][name].reset_index(drop=True)[labT].reset_index(drop=True)
        l_lab = lead_vector(lab_rt, lab_pred, h, thr_v1, ev_lab)
        l_v1 = lead_vector_cycles(rt, T1, h, thr_v1, ev_t)
        l_v1b = lead_vector_cycles(rt, T1b, h, thr_v1b, ev_t)
        H = {"modelo": name,
             "laboratorio_rain_v1": {**block(lab_rt, lab_pred, h, thr_v1, ev_lab), "antecedencia": summary(l_lab)},
             "simulacao_rain_v1": {**block(rt, any1[T].reset_index(drop=True), h, thr_v1, ev_t), "antecedencia": summary(l_v1)},
             "simulacao_rain_v1b": {**block(rt, any1b[T].reset_index(drop=True), h, thr_v1b, ev_t), "antecedencia": summary(l_v1b)},
             "limiar_rain_v1b_validacao": {"limiar": thr_v1b, "csi_validacao": csi_v}}
        H["dif_v1b_menos_v1_sim"] = dict(zip(("diff_mediana_min", "ic95_lo", "ic95_hi"), boot_lead_diff(l_v1b, l_v1)))
        H["dif_sim_menos_lab_v1"] = dict(zip(("diff_mediana_min", "ic95_lo", "ic95_hi"), boot_lead_diff(l_v1, l_lab)))
        H["fica"] = "rain_v1b" if H["dif_v1b_menos_v1_sim"]["ic95_lo"] > 0 else "rain_v1"
        H["antecedencia_por_evento"] = {"lab": l_lab.tolist(), "sim_v1": l_v1.tolist(), "sim_v1b": l_v1b.tolist()}
        out["horizons"][f"{h}h"] = H
        print(h, "h:", "lab", np.median(l_lab), "sim v1", np.median(l_v1), "sim v1b", np.median(l_v1b), "->", H["fica"], flush=True)
    out["segundos"] = round(time.time() - t0, 1)
    (OUT_REPORT / f"evaluation_{scenario}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cenario", default="estavel", choices=["estavel", "chuva"])
    run(ap.parse_args().cenario)
