"""
Item 2.4.3 do prompt do alerta de 20 min: efeito do controle de qualidade ao vivo sobre os 5 anos, na simulação do
dado como chega (cenário estável). Mesmas regras da API (emenda ao ADR 0022):
- pico isolado: hora de uma estação com 10 mm ou mais, com pelo menos 2 vizinhas a até 6 km informando a mesma hora,
  todas abaixo de 1 mm (RainSpatialCheck) -> as leituras dessa hora saem;
- estação suspeita: critérios da checagem mensal (StationMonthChecker: cobertura parcial < 80%, travada em zero,
  subcaptação < 0,5, excesso > 2, com pelo menos 2 vizinhas a até 10 km com 20 mm ou mais nas horas em comum) numa janela
  móvel de 30 dias terminando no início de cada dia -> a estação sai naquele dia.
Leitura retirada some (a estação fica ausente se não sobrar leitura em 70 min), nunca vira zero.
Uso: python -m mapi_ai.rain.live_qc
"""
import json

import joblib
import numpy as np
import pandas as pd

from mapi_ai.rain import availability as A
from mapi_ai.rain import config as C
from mapi_ai.rain import dataset as D
from mapi_ai.rain import evaluate as E
from mapi_ai.rain import features as F
from mapi_ai.rain import simulate_v1b as S

PEAK_MM, PEAK_KM, DRY_NEIGHBOR_MM, MIN_NEIGHBORS = 10.0, 6.0, 1.0, 2
CHECK_KM, MIN_COV, MIN_NEIGHBOR_MM, STUCK_MM, MIN_RATIO, MAX_RATIO = 10.0, 0.80, 20.0, 1.0, 0.5, 2.0
WINDOW_H = 720


def hourly(readings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chuva por hora (soma dos blocos com fim na hora) e minutos cobertos, hora x estação (NaN = sem leitura)."""
    r = readings.sort_values(["station_id", "ts"]).copy()
    r["len"] = r.groupby("station_id").ts.diff().dt.total_seconds().div(60).clip(upper=60).fillna(10)
    r["hour"] = r.ts.dt.ceil("h")
    mm = r.groupby(["hour", "station_id"]).mm.sum().unstack()
    cov = r.groupby(["hour", "station_id"]).len.sum().unstack().clip(upper=60)
    idx = pd.date_range(mm.index.min(), mm.index.max(), freq="h")
    return mm.reindex(idx), cov.reindex(idx)


def dist_km(stations: pd.DataFrame) -> pd.DataFrame:
    s = stations.set_index("station_id")
    ids = list(s.index)
    d = np.array([[F.km(s.lat[a], s.lon[a], s.lat[b], s.lon[b]) for b in ids] for a in ids])
    return pd.DataFrame(d, index=ids, columns=ids)


def peak_mask(mm: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    """True onde a hora da estação é pico isolado."""
    out = pd.DataFrame(False, index=mm.index, columns=mm.columns)
    for s in mm.columns:
        nb = [o for o in mm.columns if o != s and o in dist.index and s in dist.index and dist.loc[s, o] <= PEAK_KM]
        if not nb:
            continue
        v = mm[s]
        n = mm[nb]
        n_rep = n.notna().sum(axis=1)
        wet_nb = (n >= DRY_NEIGHBOR_MM).any(axis=1)
        out[s] = (v >= PEAK_MM) & (n_rep >= MIN_NEIGHBORS) & ~wet_nb
    return out


def station_fail(mm: pd.DataFrame, cov: pd.DataFrame, dist: pd.DataFrame) -> pd.DataFrame:
    """FAIL por estação e dia (janela de 30 dias terminando no início do dia)."""
    days = pd.date_range(mm.index.min().ceil("D"), mm.index.max().floor("D"), freq="D")
    roll = lambda x: x.rolling(WINDOW_H, min_periods=1).sum()
    has = mm.notna()
    covered = roll(cov.where(has).fillna(0))
    nval = roll(has.astype(float))
    out = pd.DataFrame(False, index=days, columns=mm.columns)
    for s in mm.columns:
        part = (covered[s] / (nval[s] * 60).where(nval[s] > 0)) < MIN_COV
        ratios, sums_s = [], []
        for o in mm.columns:
            if o == s or s not in dist.index or o not in dist.index or dist.loc[s, o] > CHECK_KM:
                continue
            common = has[s] & has[o]
            ss = roll(mm[s].where(common).fillna(0))
            nn = roll(mm[o].where(common).fillna(0))
            ok = nn >= MIN_NEIGHBOR_MM
            ratios.append((ss / nn).where(ok))
            sums_s.append(ss.where(ok))
        fail = part.fillna(False)
        if ratios:
            R = pd.concat(ratios, axis=1)
            Ss = pd.concat(sums_s, axis=1)
            nok = R.notna().sum(axis=1)
            med = R.median(axis=1)
            stuck = Ss.median(axis=1) < STUCK_MM
            checked = nok >= MIN_NEIGHBORS
            fail = fail | (checked & (stuck | (med < MIN_RATIO) | (med > MAX_RATIO)))
        out[s] = fail.reindex(days - pd.Timedelta("1h"), method="ffill").fillna(False).to_numpy()
    return out


def filtered(readings, stations):
    mm, cov = hourly(readings)
    dist = dist_km(stations)
    pk = peak_mask(mm, dist)
    fl = station_fail(mm, cov, dist)
    r = readings.copy()
    hour = r.ts.dt.ceil("h")
    day = r.ts.dt.floor("D")
    drop_pk = pk.stack().rename("pk").reset_index().rename(columns={"level_0": "hour", "level_1": "station_id"})
    drop_pk = drop_pk[drop_pk.pk]
    drop_fl = fl.stack().rename("fl").reset_index().rename(columns={"level_0": "day", "level_1": "station_id"})
    drop_fl = drop_fl[drop_fl.fl]
    r["hour"], r["day"] = hour, day
    r = r.merge(drop_pk, on=["hour", "station_id"], how="left").merge(drop_fl, on=["day", "station_id"], how="left")
    stats = {"leituras": int(len(r)), "retiradas_pico_isolado": int(r.pk.fillna(False).sum()),
             "retiradas_estacao_suspeita": int(r.fl.fillna(False).sum()),
             "estacao_dias_reprovados": int(fl.to_numpy().sum()), "estacao_dias": int(fl.size),
             "horas_pico_isolado": int(pk.to_numpy().sum())}
    keep = ~(r.pk.fillna(False) | r.fl.fillna(False))
    return r[keep][readings.columns], stats


def run():
    readings, stations, points = D.load_readings(), pd.read_csv(C.STATIONS), pd.read_csv(C.POINTS)
    readings["delay_min"] = A.sample_delays(readings, pd.read_csv(S.DELAYS), seed=C.SEED, scenario="estavel")
    rq, stats = filtered(readings, stations)
    print(stats, flush=True)
    om_prev = D.load_om(C.OM_PREV)
    times = pd.date_range(C.DATA_START, C.DATA_END, freq=F.SLOT)
    model = joblib.load(C.MODELS_DIR / "gb_rede.joblib")
    meta = json.loads((C.MODELS_DIR / "metadata.json").read_text())
    thr = meta["horizons"]["1h"]["alert_threshold_FORTE"]
    out = {"regras": __doc__.strip(), "filtro": stats, "partes": {}}
    base = {o: S.simulated_dataset("estavel", o) for o in S.OFFSETS}
    filt = {o: pd.concat([A.features_at_visible(rq, stations, p.latitude, p.longitude, times, om_prev.get(p.slug), offset_min=o)
                          .reset_index(drop=True) for p in points.itertuples()], ignore_index=True) for o in S.OFFSETS}
    for part in ("valid", "test"):
        res = {}
        for name, rd in (("sem_filtro", None), ("com_filtro", rq)):
            preds = {}
            for o in S.OFFSETS:
                b = base[o]
                held = b[b.part == part].reset_index(drop=True)
                if rd is None:
                    x = held
                else:
                    x = filt[o][(b.part == part).to_numpy()].reset_index(drop=True)
                preds[o] = model.predict(x)
            rows = S.truth_rows(base[S.TRAIN_OFFSET][base[S.TRAIN_OFFSET].part == part].reset_index(drop=True))
            ev = E.events(rows, C.THRESHOLDS["FORTE"])
            p_any = pd.concat(list(preds.values())).groupby(level=0).max()
            ls = E.lead_stats(rows, p_any, 1, thr, ev)
            lead = S.lead_vector_cycles(rows, preds, 1, thr, ev)
            res[name] = {"eventos": int(len(ev)), "avisados": int((lead > 0).sum()), "antecedencia": S.summary(lead),
                         "episodios_falsos": ls["episodios_falsos"], "falsos_por_evento": ls["falsos_por_evento"]}
            print(part, name, res[name]["avisados"], res[name]["episodios_falsos"], flush=True)
        res["alarmes_falsos_evitados"] = res["sem_filtro"]["episodios_falsos"] - res["com_filtro"]["episodios_falsos"]
        res["eventos_que_deixariam_de_avisar"] = max(0, res["sem_filtro"]["avisados"] - res["com_filtro"]["avisados"])
        out["partes"][part] = res
    (C.ROOT / "reports" / "rain_v1b" / "controle_qualidade_ao_vivo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: v for k, v in o.items() if k != "regras"}, indent=1, ensure_ascii=False, default=float))
