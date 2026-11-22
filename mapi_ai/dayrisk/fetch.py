"""
Histórico de rodadas do Open-Meteo (previous-runs-api) para o "dia de risco". Bruto antes de extrair: cada resposta é
gravada inteira (gzip) em data/day_risk/raw/ com o SHA-256 do corpo no manifesto, antes de ser lida.

Uso:
  python -m mapi_ai.dayrisk.fetch           # baixa o que falta (uma requisição por vez, pausa entre elas; retomável)
  python -m mapi_ai.dayrisk.fetch --parse   # só remonta data/day_risk/om_previous_runs.csv.gz a partir dos brutos
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from mapi_ai.dayrisk import config as C
from mapi_ai.rain import config as RC

USER_AGENT = "mapi-ai/day-risk (pesquisa academica; uso nao comercial)"
LEADS = (1, 2, 3)
MANIFEST = C.RAW_DIR / "manifest.json"


def _vars(model: str) -> list[str]:
    v = [f"precipitation_previous_day{n}" for n in LEADS]
    if model == C.OM_CAPE_MODEL:
        v += [f"cape_previous_day{n}" for n in LEADS]
    return v


def _url(lat: float, lon: float, model: str) -> str:
    q = {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "models": model, "hourly": ",".join(_vars(model)),
         "start_date": C.OM_START_DATE, "end_date": C.OM_END_DATE, "timezone": "UTC"}
    return C.OM_URL + "?" + urllib.parse.urlencode(q)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def fetch() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    pts = pd.read_csv(RC.POINTS)
    man = _manifest()
    for _, p in pts.iterrows():
        for model in C.OM_MODELS:
            name = f"prev_{p.slug}_{model}.json.gz"
            if name in man and (C.RAW_DIR / name).exists():
                continue
            url = _url(p.latitude, p.longitude, model)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            (C.RAW_DIR / name).write_bytes(gzip.compress(body, mtime=0))
            man[name] = {"sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "url": url,
                         "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "slug": p.slug, "model": model}
            MANIFEST.write_text(json.dumps(man, indent=1, ensure_ascii=False))
            print(f"{name}: {len(body)} bytes", flush=True)
            time.sleep(C.OM_PAUSE_S)


def parse() -> pd.DataFrame:
    """Tabela longa: slug, model, hour_end_utc, lead_days, precip_mm, cape. Confere o SHA-256 de cada bruto."""
    man = _manifest()
    rows = []
    for name, m in sorted(man.items()):
        body = gzip.decompress((C.RAW_DIR / name).read_bytes())
        if hashlib.sha256(body).hexdigest() != m["sha256"]:
            raise RuntimeError(f"SHA-256 diverge do manifesto: {name}")
        d = json.loads(body)
        if d.get("utc_offset_seconds", -1) != 0:
            raise RuntimeError(f"{name}: resposta fora de UTC")
        h = d["hourly"]
        t = pd.to_datetime(h["time"], utc=True)
        for n in LEADS:
            df = pd.DataFrame({"slug": m["slug"], "model": m["model"], "hour_end_utc": t, "lead_days": n,
                               "precip_mm": h.get(f"precipitation_previous_day{n}"),
                               "cape": h.get(f"cape_previous_day{n}", [None] * len(t)),
                               "grid_lat": d.get("latitude"), "grid_lon": d.get("longitude")})
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="coerce")
    out["cape"] = pd.to_numeric(out["cape"], errors="coerce")
    out.to_csv(C.OM_TABLE, index=False, compression="gzip")
    return out


if __name__ == "__main__":
    if "--parse" not in sys.argv:
        fetch()
    df = parse()
    print(df.groupby(["model", "lead_days"]).precip_mm.agg(["count", "mean"]).round(3).to_string())
