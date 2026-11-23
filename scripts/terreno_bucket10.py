"""
Gera data/bucket10/terreno.json a partir da saída do terreno (projeto-mapi-api/tools/terreno/run.sh atributos):
ponto_terreno.csv e curva_cota_volume.csv. Regras de status da onda 16, Fase 3 (projeto-mapi-api/docs/ONDA16-FASE3-PONTOS.md),
escritas antes de gerar:
  R3: OK só se há depressão no MDT a até 5 m do ponto (fill - MDT > 2 cm); senão EM_REVISAO, com o motivo.
  Trava (a) (ADR 0027, emenda A) só nos pontos com exutório pelo snap de 10 m: A(20 m)/A(5 m) > 10 -> EM_REVISAO.
  R3b (conferência de cálculo, escrita depois de ver o Derby): a curva cota-volume vai do fundo a fundo + 2 m; se a rua
  + 15 cm fica acima do topo da curva, o balde não consegue calcular lâmina na rua (satura abaixo dela) -> EM_REVISAO.
Uso: python -I scripts/terreno_bucket10.py <pasta do terreno> [saída]
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

DD_MIN = 0.02
TRAVA = 10.0


def main(src: Path, out: Path):
    pts = list(csv.DictReader(open(src / "ponto_terreno.csv", encoding="utf-8")))
    curvas = {}
    for r in csv.DictReader(open(src / "curva_cota_volume.csv", encoding="utf-8")):
        curvas.setdefault(r["slug"], []).append((float(r["altura_acima_fundo_m"]), float(r["volume_acumulado_m3"])))
    shas = {f: hashlib.sha256((src / f).read_bytes()).hexdigest() for f in ("ponto_terreno.csv", "curva_cota_volume.csv")}
    res = []
    for p in pts:
        slug = p["slug"]
        c = sorted(curvas.get(slug, []))
        rua_rel = float(p["cota_rua_m_raio3m"]) - float(p["fundo_bacia_cota_m"])
        dd5 = float(p["profundidade_depressao_max_raio5m"])
        regra = p.get("regra_exutorio") or "snap 10 m"
        a5, a20 = float(p["area_contribuinte_m2_snap5m"]), float(p["area_contribuinte_m2_snap20m"])
        razao = a20 / a5 if a5 else None
        motivos = []
        if dd5 <= DD_MIN:
            motivos.append(f"sem depressão no MDT a até 5 m do ponto (fill - MDT máx. {dd5:.3f} m <= {DD_MIN} m): o balde de "
                           "acúmulo não descreve este lugar (regra R3, onda 16)")
        zmax = max((z for z, _ in c), default=0.0)
        if rua_rel + 0.15 > zmax:
            motivos.append(f"a curva cota-volume (até {zmax:.2f} m acima do fundo) não chega à rua + 15 cm ({rua_rel:.2f} + 0,15 m): "
                           "o fundo da bacia está bem abaixo da rua (regra R3b, onda 16)")
        if regra.startswith("snap") and razao is not None and razao > TRAVA:
            motivos.append(f"trava da área: A(20 m)/A(5 m) = {razao:.1f} > {TRAVA:.0f}")
        item = {"slug": slug, "status": "EM_REVISAO" if motivos else "OK", "area_m2": int(p["area_contribuinte_m2"]),
                "rua_rel_m": rua_rel, "run_id": None, "regra_exutorio": regra,
                "profundidade_depressao_max_raio5m": dd5, "z": [z for z, _ in c], "v": [v for _, v in c],
                "fonte": {"pasta": src.name, **shas}}
        if motivos:
            item["motivo"] = "; ".join(motivos)
        res.append(item)
        print(slug, item["status"], item["area_m2"], round(rua_rel, 3), regra, dd5, item.get("motivo", ""))
    out.write_text(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "data" / "bucket10" / "terreno.json")
