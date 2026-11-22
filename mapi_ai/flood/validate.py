"""
Validação do alerta por regressão (Etapa 2), SEM treinar com rótulo. Tudo aqui foi escrito antes de rodar:

Reconstrução histórica do alerta, por ponto com terreno e por instante t (grade de 10 min):
- chuva prevista: as previsões fora da amostra do rain_v1 (data/rain/predictions.pkl.gz, modelo escolhido por horizonte);
- estado inicial: balde sobre as últimas 24 h de chuva MEDIDA do ponto (data/bucket10/serie10min_<ponto>, a mesma série
  do alerta), começando vazio, como o serviço recebe; lacuna em qualquer passo da janela = SEM_DADO (nunca zero);
- projeção, lâmina máxima e regra: idênticas a mapi_ai/flood/forecast.py (config.RULE), vetorizadas sobre os instantes.

2.1 Calibração da chuva: só reporta a cobertura dos quantis do rain_v1 no teste (reports/v4/rain/evaluation.json).
2.2 Coerência física: para cada evento FORTE do TESTE do rain_v1 (rain.evaluate.events), a lâmina do balde com a chuva
    MEDIDA chega a 15 cm entre 3 h antes e 6 h depois do cruzamento? Taxa com IC 95% por bootstrap de dia.
2.3 Teste externo (Defesa Civil, uma vez, critério aprovado pelo Lucas em 21/11): unidade = ponto-dia local
    (America/Recife). Dia observado positivo: algum registro de data/in1165/flood_occurrences.csv.gz a até 2 km do
    ponto naquele dia. Dia previsto positivo: ALERTA ou ATENCAO em algum instante e horizonte do dia. Ponto-dia sem
    nenhum instante calculável fica fora (contado). Sensibilidade, falsos alertas por ponto-mês, IC 95% por bootstrap
    de dia (todos os pontos do dia juntos).
2.4 Baselines sem chuva prevista, no mesmo critério do 2.3: "nunca alerta" e "alerta se a chuva medida das últimas
    6 h >= 25 mm".
Uso: python -m mapi_ai.flood.validate
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from mapi_ai import bucket10 as B10
from mapi_ai.flood import config as FC
from mapi_ai.rain import config as RC
from mapi_ai.rain import evaluate as RE

OUT = RC.ROOT / "reports" / "regressao"
SERIES = RC.ROOT / "data" / "bucket10"
OCC = RC.ROOT / "data" / "in1165" / "flood_occurrences.csv.gz"
POINTS = RC.ROOT / "data" / "in1165" / "flood_points.csv"
WINDOW = 144                 # 24 h de estado inicial
RADIUS_KM = 2.0
BASE_6H_MM = 25.0
EVENT_WINDOW = (-18, 36)     # passos de 10 min em torno do cruzamento FORTE: 3 h antes, 6 h depois
N_BOOT = 1000
CHOSEN = {1: "gb_rede", 3: "gb_rede_om", 6: "gb_rede_om"}


def km(a, b, c, d):
    la1, lo1, la2, lo2 = map(math.radians, (a, b, c, d))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(h))


class VBucket:
    """O balde M1a a 10 min (bucket10.simular) vetorizado sobre muitos instantes ao mesmo tempo."""

    def __init__(self, ter: B10.Terreno, p: B10.Params10):
        self.t, self.p = ter, p
        self.cap = p.C * p.i_cap_mm_h * ter.area_m2 / 1000.0 * p.dt_min / 60.0
        self.vc = ter.v_curva_m3 * p.curve_scale

    def step(self, V, P):
        v = V + self.p.C * P * self.t.area_m2 / 1000.0
        return np.maximum(0.0, v - np.minimum(self.cap, v))

    def sheet_mm(self, V):
        z = np.interp(V, self.vc, self.t.z_curva_m)
        return np.where(V > 0, 1000.0 * np.maximum(0.0, z - self.t.rua_rel_m), 0.0)


def state(mm: np.ndarray, vb: VBucket):
    """(ok, V_t, h_t_mm) por índice: balde nas últimas WINDOW leituras, começando vazio; ok = janela sem lacuna."""
    gap = np.isnan(mm).astype(int)
    gaps = pd.Series(gap).rolling(WINDOW, min_periods=WINDOW).sum().to_numpy()
    ok = gaps == 0
    idx = np.flatnonzero(ok)
    V = np.zeros(len(idx))
    for k in range(WINDOW):
        V = vb.step(V, mm[idx - WINDOW + 1 + k])
    Vt, ht = np.full(len(mm), np.nan), np.full(len(mm), np.nan)
    Vt[idx], ht[idx] = V, vb.sheet_mm(V)
    return ok, Vt, ht


def project_max_cm(V0, h0_mm, rain_mm, horizon, vb: VBucket):
    n = int(horizon * 60 / FC.DT_MIN)
    P = rain_mm / n
    V, hmax = V0.copy(), h0_mm.copy()
    for _ in range(n):
        V = vb.step(V, P)
        hmax = np.maximum(hmax, vb.sheet_mm(V))
    return hmax / 10.0


def reconstruct(rows: pd.DataFrame, pred: dict, terr: dict) -> pd.DataFrame:
    """Alerta histórico por ponto e instante: nível por horizonte e nível do instante."""
    out = []
    for slug, t in terr.items():
        g = rows.index[rows.slug == slug]
        r = rows.loc[g, ["slug", "t", "part"]].copy()
        s = pd.read_csv(SERIES / f"serie10min_{slug}.csv.gz", parse_dates=["slot_end_utc"])
        mm = s.set_index("slot_end_utc").mm.reindex(r.t).to_numpy(dtype=float)
        ter = B10.Terreno(float(t["area_m2"]), np.asarray(t["z"], float), np.asarray(t["v"], float), float(t["rua_rel_m"]))
        vb = VBucket(ter, B10.Params10())
        # estado: calculado na série completa (com lacunas) e lido nos instantes das linhas
        full = s.set_index("slot_end_utc").mm
        ok_f, V_f, h_f = state(full.to_numpy(dtype=float), vb)
        st = pd.DataFrame({"ok": ok_f, "V": V_f, "h": h_f}, index=full.index).reindex(r.t)
        r["ok_state"] = st.ok.fillna(False).to_numpy().astype(bool)
        r["V"], r["h_now_mm"] = st.V.to_numpy(), st.h.to_numpy()
        r["mm"] = mm
        r["rain6h_mm"] = full.rolling(36, min_periods=36).sum().reindex(r.t).to_numpy()
        levels = []
        for h in FC.HORIZONS:
            p = pred[CHOSEN[h]].loc[g]
            q50, q95 = p[f"q50_{h}h"].to_numpy(), p[f"q95_{h}h"].to_numpy()
            okh = r.ok_state.to_numpy() & ~np.isnan(q50) & ~np.isnan(q95)
            c50, c95 = np.full(len(r), np.nan), np.full(len(r), np.nan)
            i = np.flatnonzero(okh)
            V0, h0 = r.V.to_numpy()[i], r.h_now_mm.to_numpy()[i]
            c50[i] = project_max_cm(V0, h0, q50[i], h, vb)
            c95[i] = project_max_cm(V0, h0, q95[i], h, vb)
            r[f"sheet_q50_{h}h"], r[f"sheet_q95_{h}h"] = c50, c95
            lvl = np.where(~okh, "SEM_DADO", np.where(c50 >= FC.HCRIT_CM, "ALERTA",
                                                      np.where(c95 >= FC.HCRIT_CM, "ATENCAO", "SEM_ALERTA")))
            r[f"level_{h}h"] = lvl
            levels.append(lvl)
        L = np.stack(levels)
        r["level"] = np.where((L == "ALERTA").any(0), "ALERTA", np.where((L == "ATENCAO").any(0), "ATENCAO",
                              np.where((L == "SEM_DADO").all(0), "SEM_DADO", "SEM_ALERTA")))
        out.append(r)
        print(slug, r.level.value_counts().to_dict(), flush=True)
    return pd.concat(out, ignore_index=True)


def boot_rate(hit: np.ndarray, day: np.ndarray, rng, n=N_BOOT):
    """Proporção com IC 95% por bootstrap de dia (reamostra dias inteiros)."""
    if len(hit) == 0:
        return {"n": 0, "taxa": None, "ic95": [None, None]}
    d = pd.DataFrame({"h": hit.astype(float), "d": day}).groupby("d").h.agg(["sum", "count"])
    s, c = d["sum"].to_numpy(), d["count"].to_numpy()
    idx = rng.integers(0, len(s), size=(n, len(s)))
    v = s[idx].sum(1) / c[idx].sum(1)
    # Com 0 (ou todos) acertos o bootstrap degenera em [0, 0]: o IC exato de Clopper-Pearson com o DIA como unidade
    # (dias com algum acerto / dias) dá a incerteza honesta para poucos dias.
    from scipy.stats import beta
    k, nd = int((s > 0).sum()), int(len(s))
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, nd - k + 1))
    hi = 1.0 if k == nd else float(beta.ppf(0.975, k + 1, nd - k))
    return {"n": int(c.sum()), "acertos": int(s.sum()), "dias": nd, "taxa": round(float(s.sum() / c.sum()), 4),
            "ic95": [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)],
            "ic95_exato_por_dia": [round(lo, 4), round(hi, 4)], "dias_com_acerto": k}


def daily(rec: pd.DataFrame, flag: np.ndarray) -> pd.DataFrame:
    """Ponto-dia local: previsto positivo se algum instante calculável do dia tem flag; calculável se algum instante tem."""
    d = rec.assign(day=rec.t.dt.tz_convert("America/Recife").dt.date, flag=flag, calc=rec.level != "SEM_DADO")
    return d.groupby(["slug", "day"]).agg(flag=("flag", "any"), calc=("calc", "any")).reset_index()


def external(rec: pd.DataFrame, occ: pd.DataFrame, pts: pd.DataFrame, rng) -> dict:
    occ = occ.copy()
    occ["day"] = pd.to_datetime(occ.occurred_from, utc=True).dt.tz_convert("America/Recife").dt.date
    pos = set()
    dist_min = {}
    for _, p in pts.iterrows():
        dists = [km(p.latitude, p.longitude, a, b) for a, b in zip(occ.geocoded_lat, occ.geocoded_lon)]
        dist_min[p.slug] = round(min(dists), 2)
        for dday, dk in zip(occ.day, dists):
            if dk <= RADIUS_KM:
                pos.add((p.slug, dday))
    rules = {
        "alerta_regressao": rec.level.isin(["ALERTA", "ATENCAO"]).to_numpy(),
        "so_ALERTA_q50": (rec.level == "ALERTA").to_numpy(),
        "baseline_nunca": np.zeros(len(rec), bool),
        "baseline_chuva6h_ge_25mm": (rec.rain6h_mm >= BASE_6H_MM).fillna(False).to_numpy(),
    }
    base = daily(rec, rules["alerta_regressao"])
    base["obs"] = [(s, d) in pos for s, d in zip(base.slug, base.day)]
    calc = base.calc.to_numpy()
    res = {"dias_registro_cidade": int(occ.day.nunique()), "registros": int(len(occ)),
           "ponto_dias_positivos_2km_total": int(len(pos)),
           "ponto_dias_positivos_no_periodo_com_serie": int(base.obs.sum()),
           "ponto_dias_positivos_calculaveis": int((base.obs & calc).sum()),
           "ponto_dias_sem_calculo": int((~calc).sum()),
           "distancia_minima_km_por_ponto": dist_min, "regras": {}}
    months = (base[calc].groupby("slug").day.nunique() / 30.44).sum()
    for name, f in rules.items():
        d = daily(rec, f).merge(base[["slug", "day", "obs"]], on=["slug", "day"])
        d = d[d.calc]
        P = d[d.obs]
        N = d[~d.obs]
        sens = boot_rate(P.flag.to_numpy(), P.day.astype(str).to_numpy(), rng)
        # falsos por ponto-mês, IC por bootstrap de dia
        nd = N.groupby("day").flag.agg(["sum", "count"])
        s, c = nd["sum"].to_numpy(), nd["count"].to_numpy()
        idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
        fr = s[idx].sum(1) / c[idx].sum(1) * 30.44
        res["regras"][name] = {
            "sensibilidade": sens,
            "falsos_alertas_por_ponto_mes": round(float(s.sum() / c.sum() * 30.44), 3),
            "falsos_ic95": [round(float(np.quantile(fr, 0.025)), 3), round(float(np.quantile(fr, 0.975)), 3)],
            "ponto_dias_com_alerta": int(d.flag.sum()), "ponto_meses": round(float(months), 1),
            "positivos_por_ponto": P.groupby("slug").flag.agg(["sum", "count"]).astype(int).to_dict("index")}
    res["dias_positivos_detalhe"] = [
        {"ponto": s, "dia": str(dd), "calculavel": bool(c), "alerta_regressao": bool(f)}
        for s, dd, c, f in base[base.obs][["slug", "day", "calc", "flag"]].itertuples(index=False)]
    return res


def coherence(rec: pd.DataFrame, rows_test: pd.DataFrame, rng) -> dict:
    ev = RE.events(rows_test, RC.THRESHOLDS["FORTE"])
    hit, hit_alert, days, skipped = [], [], [], 0
    for e in ev.itertuples():
        g = rec[rec.slug == e.slug].reset_index(drop=True)
        if g.empty:
            skipped += 1      # ponto sem terreno
            continue
        i = int(g.t.searchsorted(pd.Timestamp(e.c).tz_convert("UTC")))
        w = g.iloc[max(0, i + EVENT_WINDOW[0]): i + EVENT_WINDOW[1] + 1]
        if w.h_now_mm.isna().all():
            skipped += 1
            continue
        hit.append(bool(np.nanmax(w.h_now_mm) >= FC.HCRIT_CM * 10))
        pre = g.iloc[max(0, i - 36): i + 7]
        hit_alert.append(bool(pre.level.isin(["ALERTA", "ATENCAO"]).any()))
        days.append(str(e.c.tz_convert("America/Recife").date()))
    return {"eventos_FORTE_teste": int(len(ev)), "eventos_sem_terreno_ou_sem_serie": skipped,
            "janela": "3 h antes a 6 h depois do cruzamento (chuva medida)",
            "lamina_medida_chega_a_15cm": boot_rate(np.array(hit), np.array(days), rng),
            "alerta_regressao_de_6h_antes_a_1h_depois": boot_rate(np.array(hit_alert), np.array(days), rng)}


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RC.SEED)
    # Previsões do rain_v1 (artefatos servidos, sem retreino) no período INTEIRO: predictions.pkl.gz só tem validação e
    # teste (2024-07 em diante), e os dias da Defesa Civil são de 2021-05 a 2024-06. No treino a previsão é dentro da
    # amostra do modelo de chuva (declarado em "aviso_em_amostra").
    import joblib
    from mapi_ai.rain import dataset as RD
    ds = RD.load().reset_index(drop=True)
    ds = ds[ds.part != "guard"].reset_index(drop=True)
    keep = ["slug", "t", "part", "raining", "p_10", "p_60"] + [c for c in ds if c[:2] in ("y_", "ev") or c[:5] == "peak_"]
    rows = ds[keep]
    pred = {n: joblib.load(RC.MODELS_DIR / f"{n}.joblib").predict(ds) for n in sorted(set(CHOSEN.values()))}
    terr = {t["slug"]: t for t in json.loads((SERIES / "terreno.json").read_text()) if t["status"] == "OK" and t["z"]}
    rec = reconstruct(rows, pred, terr)
    test = rec.part == "test"
    ev = json.loads((RC.REPORTS_DIR / "evaluation.json").read_text())
    cal = {h: {k: round(ev["horizons"][f"{h}h"]["teste"][CHOSEN[h]]["todas"][f"cobertura_{k}"], 4) for k in ("q50", "q80", "q95")}
           for h in FC.HORIZONS}
    years = {s: (g.t.max() - g.t.min()).total_seconds() / (365.25 * 86400) for s, g in rec[test].groupby("slug")}
    freq = {s: {lvl: round(float((g.level == lvl).sum()) / 6 / years[s], 1) for lvl in ("ALERTA", "ATENCAO", "SEM_DADO")}
            for s, g in rec[test].groupby("slug")}
    out = {
        "versao": FC.VERSION, "regra": FC.RULE_TEXT, "metodo": __doc__.strip(),
        "pontos_com_terreno": sorted(terr), "pontos_sem_terreno": ["JABOATAO_CENTRO"],
        "2.1_cobertura_quantis_teste": {"valores": cal, "nominal": {"q50": 0.5, "q80": 0.8, "q95": 0.95},
                                        "fonte": "reports/v4/rain/evaluation.json (modelo escolhido por horizonte)"},
        "2.2_coerencia_fisica": coherence(rec, rows[rows.part == "test"], rng),
        "2.3_e_2.4_teste_externo": external(rec, pd.read_csv(OCC), pd.read_csv(POINTS), rng),
        "frequencia_no_teste_horas_por_ano": freq,
        "diagnostico": {
            "capacidade_galeria_mm_h": round(B10.Params10().i_cap_mm_h, 1),
            "capacidade_por_passo_10min_mm": round(B10.Params10().i_cap_mm_h / 6, 2),
            "maior_chuva_prevista_mm": {f"{h}h": {q: round(float(np.nanmax(pred[CHOSEN[h]][f"{q}_{h}h"])), 1)
                                                  for q in ("q50", "q95")} for h in FC.HORIZONS},
            "maior_intensidade_prevista_espalhada_mm_h": {f"{h}h": round(float(np.nanmax(pred[CHOSEN[h]][f"q95_{h}h"])) / h, 1)
                                                         for h in FC.HORIZONS},
            "maior_chuva_10min_medida_mm": {s: round(float(g.mm.max()), 2) for s, g in rec.groupby("slug")},
            "maior_lamina_cm": {s: {"agora_medida": round(float(np.nanmax(g.h_now_mm)) / 10, 2),
                                    **{f"q95_{h}h": round(float(np.nanmax(g[f"sheet_q95_{h}h"])), 2) for h in FC.HORIZONS}}
                                for s, g in rec.groupby("slug")},
            "leitura": ("a chuva prevista, espalhada no horizonte, nunca passa da capacidade da galeria (IDF TR 5 em Te = 10 "
                        "min), e a chuva medida de 10 min fica abaixo da capacidade por passo: o balde não acumula e a regra "
                        "não tem como disparar com os parâmetros atuais. Nenhum parâmetro foi mudado depois deste resultado."),
        },
        "periodo": {p: [str(rec[rec.part == p].t.min()), str(rec[rec.part == p].t.max())] for p in ("train", "valid", "test")},
        "aviso_em_amostra": ("os dias da Defesa Civil (2021–2024) caem no período de TREINO e validação do rain_v1: a chuva "
                             "prevista nesses dias é dentro da amostra do modelo de chuva (otimista). Nenhum modelo viu os "
                             "rótulos de alagamento."),
        "attribution": RC.ATTRIBUTION,
    }
    (OUT / "validacao.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    rec.to_pickle(OUT / "alerta_historico.pkl.gz")
    return out


if __name__ == "__main__":
    o = run()
    print(json.dumps({k: v for k, v in o.items() if k not in ("metodo",)}, indent=1, ensure_ascii=False, default=str)[:6000])
