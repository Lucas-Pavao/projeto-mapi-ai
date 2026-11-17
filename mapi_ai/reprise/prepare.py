"""
Reprise de eventos reais (Etapa 4.1 do prompt do alerta de 20 min): escolhe os eventos, sorteia o atraso de chegada
(simulação do item 2.3, cenário estável, semente 42) e calcula, ciclo a ciclo (3 min), o que a simulação espera:
probabilidade de FORTE em 1 h do rain_v1 (o MESMO RainForecaster do serviço, com as leituras visíveis no ciclo) e o
estado do rain_alert pela mesma máquina da API. A API reproduz o período num banco de teste
(ReprisaEventosIT) e o check.py compara.
Uso: python -m mapi_ai.reprise.prepare

Eventos fixados ANTES de rodar a reprise:
1. a chuva de 28/05/2022 (o dia inteiro, UTC);
2-4. os 3 maiores eventos FORTE do teste do rain_v1 (2025-07-31 a 2026-10-31), pelo maior acumulado em 60 min do ponto;
5. o dia de chuva do teste com o maior acumulado diário num ponto sem nenhum cruzamento FORTE em nenhum ponto.
Janela de cada evento: de 3 h antes do primeiro cruzamento até 3 h depois do último (máx. 12 h); no dia sem FORTE,
6 h centradas na hora de maior chuva. Antes da janela, 30 h de leituras já chegadas (histórico do balde e da previsão).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mapi_ai.rain import availability as A
from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import evaluate as E
from mapi_ai.rain import features as F
from mapi_ai.rain.serve import RainForecaster

OUT = C.ROOT / "data" / "reprise"
CYCLE = pd.Timedelta("3min")
HISTORY = pd.Timedelta("30h")
TAIL = pd.Timedelta("6h")       # depois da janela, a reprise segue recebendo leituras até o alerta desligar ou expirar
L_OFF = 0.0316          # L_desliga escolhido na validação (reports/rain_v1b/regras_alerta.json)
CAP_10 = 20.9           # galeria em 10 min (IDF TR 5, Te 10 min)
FRESH_MIN = 15


def choose_events(lab: pd.DataFrame) -> list[dict]:
    test = lab[lab.part == "test"].reset_index(drop=True)
    ev = E.events(test, C.THRESHOLDS["FORTE"])
    ev["peak"] = [test[(test.slug == r.slug) & (test.t >= r.c) & (test.t <= r.c + pd.Timedelta("1h"))].p_60.max() for r in ev.itertuples()]
    out = []
    day = pd.Timestamp("2022-05-28", tz="UTC")
    d22 = lab[(lab.t >= day) & (lab.t < day + pd.Timedelta("1D"))]
    ev22 = E.events(d22.reset_index(drop=True), C.THRESHOLDS["FORTE"])
    out.append(window("2022-05-28", ev22.c.tolist(), d22))
    for k, r in enumerate(ev.sort_values("peak", ascending=False).drop_duplicates(subset=["c"]).head(3).itertuples()):
        same = ev[(ev.c - r.c).abs() <= pd.Timedelta("6h")].c.tolist()
        out.append(window(f"forte{k + 1}-{r.c:%Y%m%d}", same, test))
    test["day"] = test.t.dt.floor("D")
    crossed = set(ev.c.dt.floor("D"))
    daily = test.groupby(["day", "slug"]).p_10.sum().reset_index()
    daily = daily[~daily.day.isin(crossed)].sort_values("p_10", ascending=False)
    best = daily.iloc[0]
    g = test[(test.day == best.day) & (test.slug == best.slug)]
    t_peak = g.loc[g.p_60.idxmax(), "t"]
    out.append({"id": f"moderada-{best.day:%Y%m%d}", "start": str(t_peak - pd.Timedelta("3h")), "end": str(t_peak + pd.Timedelta("3h")),
                "crossings": [], "max_daily_mm": float(best.p_10)})
    return out


def window(eid, crossings, rows):
    c = sorted(pd.to_datetime(crossings, utc=True))
    s, e = c[0] - pd.Timedelta("3h"), c[-1] + pd.Timedelta("3h")
    if e - s > pd.Timedelta("12h"):
        e = s + pd.Timedelta("12h")
    return {"id": eid, "start": str(s), "end": str(e), "crossings": [str(x) for x in c]}


def machine_step(st, now, p, measured, prob_on):
    """Uma iteração da máquina da API (RainAlertRules), só o rain_alert. st: dict de estado."""
    age, r10, r30 = measured
    alerta = age is not None and age <= FRESH_MIN and r10 is not None and r10 >= CAP_10
    if alerta:
        cand, renew = "ALERTA", True
    elif age is None or age > FRESH_MIN or r10 is None:
        cand, renew = "SEM_DADO", False
    elif p is None:
        cand, renew = "SEM_DADO", False
    elif p >= prob_on:
        cand, renew = "ATENCAO", True
    elif st["state"] in ("ATENCAO", "ALERTA") and p >= L_OFF:
        cand, renew = "ATENCAO", False
    else:
        cand, renew = "NENHUM", False
    if renew:
        st["renew"] = now
    if cand in ("ATENCAO", "ALERTA") and not renew and st["renew"] is not None and (now - st["renew"]) >= pd.Timedelta("60min"):
        cand = "NENHUM"
    st["state"] = cand
    return cand


def measured(vis: pd.DataFrame, near: list, now):
    v = vis[vis.station_id.isin(near)]
    if v.empty:
        return None, None, None
    newest = v.ts.max()
    age = (now - newest).total_seconds() / 60
    if age > 70:
        return None, None, None
    slot = newest.ceil("10min")
    def win(n):
        lo = slot - pd.Timedelta(minutes=10 * n)
        per = []
        for s, g in v.groupby("station_id"):
            g = g.sort_values("ts")
            prev = g.ts.shift(1)
            ln = ((g.ts - prev).dt.total_seconds() / 60).clip(upper=60).fillna(10)
            tot = 0.0
            for ts, mm, L in zip(g.ts, g.mm, ln):
                a, b = ts - pd.Timedelta(minutes=L), ts
                ov = (min(b, slot) - max(a, lo)).total_seconds() / 60
                if ov > 0 and L > 0:
                    tot += mm * ov / L
            per.append(tot)
        return max(per) if per else None
    return age, win(1), win(3)


def run(only_readings: bool = False):
    OUT.mkdir(parents=True, exist_ok=True)
    lab = D.load()
    readings, stations, points = D.load_readings(), pd.read_csv(C.STATIONS), pd.read_csv(C.POINTS)
    readings["delay_min"] = A.sample_delays(readings, pd.read_csv(C.ROOT / "data" / "rain_v1b" / "live_delays.csv"),
                                            seed=C.SEED, scenario="estavel")
    rf = RainForecaster()
    prob_on = rf.alert[1]
    events = json.loads((OUT / "events.json").read_text()) if only_readings else choose_events(lab)
    st_pos = stations.set_index("station_id")
    for ev in events:
        s, e = pd.Timestamp(ev["start"]), pd.Timestamp(ev["end"])
        win = readings[(readings.ts > s - HISTORY) & (readings.ts <= e + TAIL)].copy()
        d = OUT / ev["id"]
        d.mkdir(parents=True, exist_ok=True)
        w = win.assign(lat=win.station_id.map(st_pos.lat), lon=win.station_id.map(st_pos.lon))
        w[["station_id", "ts", "mm", "delay_min"]].to_csv(d / "readings.csv", index=False)
        if only_readings:
            continue
        w = w[w.ts <= e]
        cycles = pd.date_range(s.ceil("3min"), e, freq=CYCLE)
        exp = {"event": ev, "prob_on": prob_on, "L_off": L_OFF, "cycles": [str(c) for c in cycles], "points": {}}
        avail = w.ts + pd.to_timedelta(w.delay_min, unit="min")
        for p in points.itertuples():
            geo = F.station_geometry(stations, p.latitude, p.longitude)
            ids, near = set(geo.station_id), list(geo.station_id[geo.near])
            st = {"state": "NENHUM", "renew": None}
            probs, states = [], []
            for c in cycles:
                vis = w[(avail <= c) & (w.ts > c - pd.Timedelta("390min")) & w.station_id.isin(ids)]
                try:
                    out = rf.forecast(p.latitude, p.longitude, [
                        {"station_id": r.station_id, "latitude": r.lat, "longitude": r.lon, "ts": r.ts, "mm": r.mm}
                        for r in vis.itertuples()], None, None, str(c))
                    pr = next(h for h in out["horizons"] if h["horizon_hours"] == 1)["prob_forte"]
                except Exception:
                    pr = None
                probs.append(pr)
                states.append(machine_step(st, c, pr, measured(vis, near, c), prob_on))
            first_on = next((cycles[i] for i, x in enumerate(states) if x in ("ATENCAO", "ALERTA")), None)
            exp["points"][p.slug] = {"prob": probs, "state": states, "first_on": None if first_on is None else str(first_on)}
            print(ev["id"], p.slug, "primeiro ligado:", first_on, flush=True)
        (d / "expected.json").write_text(json.dumps(exp, indent=1, ensure_ascii=False, default=str))
    (OUT / "events.json").write_text(json.dumps(events, indent=2, ensure_ascii=False))
    return events


if __name__ == "__main__":
    import sys
    print(json.dumps(run("--so-leituras" in sys.argv), indent=1, ensure_ascii=False))
