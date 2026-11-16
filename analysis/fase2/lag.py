"""Fase 2.2: defasagem do início da chuva por setor (estações PED da RMR) e persistência no ponto."""
import sys, math, json
import numpy as np, pandas as pd
F2, AI = sys.argv[1], sys.argv[2]
obs = pd.read_csv(f"{F2}/ped10.csv.gz", names=["st", "ts", "v"])
st = pd.read_csv(f"{F2}/ped_st.csv").set_index("station_id")
st = st[~st.index.str.endswith("G")]                       # geotécnicas: sem pluviômetro confiável
obs = obs[obs.st.isin(st.index)]
ev = pd.read_csv(f"{AI}/data/in1165/eventos_CEMADEN_PED.csv.gz")
pts = pd.read_csv(f"{AI}/data/in1165/flood_points.csv").set_index("slug")
ev = ev[ev.type.isin(["MEDIA", "FORTE"])]
WET = 0.4   # mm em 10 min (2 basculadas): início da chuva

def km(a, b, c, d):
    la1, lo1, la2, lo2 = map(math.radians, (a, b, c, d))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
def sector(lat0, lon0, lat, lon):
    ang = math.degrees(math.atan2(lat - lat0, (lon - lon0) * math.cos(math.radians(lat0))))
    return "L" if -45 <= ang < 45 else "N" if 45 <= ang < 135 else "S" if -135 <= ang < -45 else "O"

wet = obs[obs.v >= WET].sort_values("ts")
by_st = {s: g.ts.to_numpy() for s, g in wet.groupby("st")}
rows = []
# um evento por início (os 5 pontos repetem o mesmo temporal): dedup pelo início e pela estação do ponto
ev = ev.drop_duplicates(["startUtc", "stationId"])
for _, e in ev.iterrows():
    p = pts.loc[e.floodPointSlug]
    t0 = pd.Timestamp(e.startUtc).timestamp()
    own = by_st.get(e.stationId)
    if own is None: continue
    w = own[(own >= t0 - 3600) & (own <= t0 + 6 * 3600)]
    if len(w) == 0: continue
    onset = w[0]                                             # início da chuva na estação do ponto
    for s, info in st.iterrows():
        if s == e.stationId or pd.isna(info.lat): continue
        a = by_st.get(s)
        if a is None: continue
        x = a[(a >= onset - 4 * 3600) & (a <= onset + 4 * 3600)]
        d = km(p.latitude, p.longitude, info.lat, info.lon)
        rows.append(dict(event=e.eventId, type=e.type, point=e.floodPointSlug, station=s, km=d,
                         ring="0-5" if d < 5 else "5-15" if d < 15 else "15-40",
                         sector=sector(p.latitude, p.longitude, info.lat, info.lon),
                         lead_min=(onset - x[0]) / 60 if len(x) else np.nan))
L = pd.DataFrame(rows)
n_ev = L.event.nunique()
first = L.dropna().groupby("event").lead_min.max()          # a estação que começou primeiro, por evento
out = {"eventos_MEDIA_FORTE": int(n_ev), "limiar_inicio_mm_10min": WET,
       "por_setor": L.groupby("sector").lead_min.describe(percentiles=[.1, .5, .9]).round(1).reset_index().to_dict("records"),
       "por_anel": L.groupby("ring").lead_min.describe(percentiles=[.1, .5, .9]).round(1).reset_index().to_dict("records"),
       "estacoes_por_setor_e_anel": L.drop_duplicates(["point", "station"]).groupby(["sector", "ring"]).size().unstack(fill_value=0).to_dict(),
       "melhor_estacao_por_evento_min": first.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_dict(),
       "fracao_eventos_com_alguma_estacao_30min_antes": float((first >= 30).mean()),
       "fracao_eventos_com_alguma_estacao_60min_antes": float((first >= 60).mean())}
for t in ("MEDIA", "FORTE"):
    out[f"melhor_estacao_{t}"] = L[L.type == t].dropna().groupby("event").lead_min.max().describe(percentiles=[.5, .9]).round(1).to_dict()

# persistência: série horária do ponto (regra chuva-ponto), P(chuva nas próximas 1, 2, 3 h | chuva agora)
h = pd.read_csv(f"{AI}/data/in1165/cemaden_point_hourly.csv.gz", parse_dates=["hour_end_utc"])
pers = []
for slug, g in h.groupby("slug"):
    s = g.set_index("hour_end_utc").current_rainfall.asfreq("h")
    for k in (1, 2, 3):
        fut = sum(s.shift(-i) for i in range(1, k + 1))
        pers.append(pd.DataFrame({"now": s, "fut": fut, "k": k}).dropna())
P = pd.concat(pers)
def cls(v): return pd.cut(v, [-0.01, 0.0, 1, 5.03, 1e9], labels=["seco", "0-1 mm", "1-5 mm", ">=5 mm (MEDIA)"])
P["agora"] = cls(P.now)
tab = P.groupby(["k", "agora"], observed=True).fut.describe(percentiles=[.5, .8, .95]).round(2)
tab["P(fut>=1mm)"] = P.groupby(["k", "agora"], observed=True).fut.apply(lambda x: (x >= 1).mean()).round(3)
tab["P(fut>=5mm)"] = P.groupby(["k", "agora"], observed=True).fut.apply(lambda x: (x >= 5.03).mean()).round(3)
out["persistencia"] = tab.reset_index().astype({"agora": str}).to_dict("records")
json.dump(out, open(f"{F2}/lag_persist.json", "w"), indent=1, ensure_ascii=False, default=float)
L.to_csv(f"{F2}/lag_rows.csv.gz", index=False)
print(json.dumps({k: v for k, v in out.items() if k != "persistencia"}, indent=1, ensure_ascii=False, default=float))
print(tab.to_string())
