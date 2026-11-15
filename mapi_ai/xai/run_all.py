"""Roda SHAP, LIME e contrafactuais nos mesmos casos e grava reports/xai_cases.json."""
import json

from mapi_ai import in1165_config as C
from mapi_ai.xai import counterfactual, lime_explain, shap_explain
from mapi_ai.xai.common import load


def run() -> dict:
    ctx = load()
    s = shap_explain.run(ctx)
    lm = lime_explain.run(ctx)
    cf = counterfactual.run(ctx)
    for name, rec in s["cases"].items():
        rec["lime"] = lm.get(name)
    out = {"tag": C.SYNTHETIC_TAG, "model_version": ctx["meta"]["model_version"], "shap_base_value": s["base_value"],
           "threshold_high": s["threshold_high"], "global_shap_mean_abs": s["global_mean_abs"],
           "additivity_checked": s["additivity_checked"], "cases": s["cases"], "counterfactual_TP": cf,
           "attribution": C.ATTRIBUTION}
    (C.REPORTS_DIR / "xai_cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    return out


if __name__ == "__main__":
    r = run()
    print("SHAP global (top 5):", list(r["global_shap_mean_abs"].items())[:5])
    for k, v in r["cases"].items():
        print(k, round(v["probability"], 3), "real", v["y_true"], "| SHAP top3:", list(v["shap"])[:3],
              "| LIME top3:", [x["feature"] for x in v["lime"]["rules"][:3]])
    print("Contrafactual (escala, até LOW):", r["counterfactual_TP"]["scale"].get("statement"))
    print("Contrafactual (escala, sair de HIGH):", r["counterfactual_TP"]["scale_below_high"].get("statement"))
    for c in r["counterfactual_TP"]["dice"]:
        print("DiCE:", c)
