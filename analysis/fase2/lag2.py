"""Defasagem com controle: início = primeira leitura >= 0,4 mm/10 min depois de 2 h secas, na estação do ponto e em cada
vizinha; só conta a vizinha se o início dela cair em +-2 h. Controle: o mesmo cálculo com o início do ponto deslocado 3 dias
(chuva que não é a mesma), para saber quanto da "antecedência" sai por acaso."""
import sys, math, json
import numpy as np, pandas as pd
F2, AI = sys.argv[1], sys.argv[2]
obs = pd.read_csv(f"{F2}/ped10.csv.gz", names=["st", "ts", "v"])
st = pd.read_csv(f"{F2}/ped_st.csv").set_index("station_id"); st = st[~st.index.str.endswith("G")]
obs = obs[obs.st.isin(st.index)]
ev = pd.read_csv(f"{AI}/data/in1165/eventos_CEMADEN_PED.csv.gz"); ev = ev[ev.type.isin(["MEDIA", "FORTE"])]
pts = pd.read_csv(f"{AI}/data/in1165/flood_points.csv").set_index("slug")
ev = ev.drop_duplicates(["startUtc", "stationId"])
def km(a, b, c, d):
    la1, lo1, la2, lo2 = map(math.radians, (a, b, c, d))
    return 6371 * 2 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
def sector(lat0, lon0, lat, lon):
    ang = math.degrees(math.atan2(lat - lat0, (lon - lon0) * math.cos(math.radians(lat0))))
    return "L" if -45 <= ang < 45 else "N" if 45 <= ang < 135 else "S" if -135 <= ang < -45 else "O"
onsets = {}
for s, g in obs[obs.v >= 0.4].groupby("st"):
    t = np.sort(g.ts.to_numpy()); gap = np.diff(t, prepend=-10**12)
    onsets[s] = t[gap >= 2 * 3600]                       # início depois de 2 h sem chuva relevante
def leads(shift):
    rows = []
    for _, e in ev.iterrows():
        own = onsets.get(e.stationId)
        if own is None: continue
        t0 = pd.Timestamp(e.startUtc).timestamp()
        w = own[(own >= t0 - 3600) & (own <= t0 + 6 * 3600)]
        if not len(w): continue
        o = w[0] + shift; p = pts.loc[e.floodPointSlug]
        for s, info in st.iterrows():
            if s == e.stationId or s not in onsets: continue
            a = onsets[s]; x = a[(a >= o - 7200) & (a <= o + 7200)]
            d = km(p.latitude, p.longitude, info.lat, info.lon)
            rows.append(dict(event=e.eventId, type=e.type, km=d, ring="0-5" if d < 5 else "5-15" if d < 15 else "15-40",
                             sector=sector(p.latitude, p.longitude, info.lat, info.lon),
                             lead=(o - x[0]) / 60 if len(x) else np.nan))
    return pd.DataFrame(rows)
R, C = leads(0), leads(3 * 86400)
def summ(L):
    best = L.dropna().groupby("event").lead.max().reindex(L.event.unique())
    sec = L.groupby("sector").agg(pares=("lead", "size"), com_inicio_em_2h=("lead", lambda x: x.notna().mean()),
                                  mediana_min=("lead", "median"), p_antes_30min=("lead", lambda x: (x >= 30).mean()))
    return dict(eventos=int(L.event.nunique()), por_setor=sec.round(3).reset_index().to_dict("records"),
                por_anel=L.groupby("ring").lead.median().to_dict(),
                melhor_mediana=float(best.median()), melhor_p10=float(best.quantile(.1)),
                frac_alguma_30min=float((best >= 30).mean()), frac_alguma_60min=float((best >= 60).mean()),
                FORTE=dict(n=int(L[L.type == "FORTE"].event.nunique()),
                           melhor_mediana=float(L[L.type == "FORTE"].dropna().groupby("event").lead.max().median())))
out = {"real": summ(R), "controle_deslocado_3_dias": summ(C)}
json.dump(out, open(f"{F2}/lag2.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
