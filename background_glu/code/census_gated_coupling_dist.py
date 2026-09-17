# -*- coding: utf-8 -*-
"""Within-gate-branch coupling DISTRIBUTION (Olli, 2026-09-12, for background_glu).

The gate branch carries the b_L coupling, but is the coupling uniform across gate
units within a layer? Per gate row w_n, the coupling is c_n = w_n . b_hat_L; the
per-row cosine cos_n = c_n / ||w_n||. This quantifies the distribution across
units, per layer, for the gate branch (and the value branch for contrast):

  cos_p10/50/90     signed cosine quantiles (sign = inhibitory if <0)
  abscos_p50/90     magnitude quantiles
  frac_uncoupled    fraction with |cos_n| < 2*floor (floor = sqrt(2/(pi d)),
                    the random-direction |cos| baseline) -- "effectively uncoupled"
  frac_neg          fraction with c_n < 0 (rests-closed sign)
  corr_absc_norm    Spearman(|c_n|, ||w_n||) -- is coupling norm-scaled?
  gini_absc         Gini of |c_n| across units -- is coupling concentrated in few?

Qwen2.5-3B / 1.5B (strong), TinyLlama-1.1B (moderate). CPU/float32, weights + b_L.
Usage: .venv/Scripts/python.exe superposition/code/census_gated_coupling_dist.py
"""
import os
import sys
import json
import time
import math

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import torch

DEV = "cpu"
DTYPE = torch.float32
MAXLEN, SKIP = 160, 16
N_CAL = 16
MODELS = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-coupling-dist.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def gini(x):
    x = np.sort(np.abs(x))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def branch_stats(W, bL, bLn, floor):
    W = W.float()
    rown = W.norm(dim=1).clamp(min=1e-9)
    c = (W @ bL).numpy()                       # signed coupling (unnormalised)
    cosn = ((W @ bLn) / rown).numpy()          # signed per-row cosine
    absc = np.abs(cosn)
    return {"cos_p10": round(float(np.percentile(cosn, 10)), 4),
            "cos_p50": round(float(np.percentile(cosn, 50)), 4),
            "cos_p90": round(float(np.percentile(cosn, 90)), 4),
            "abscos_p50": round(float(np.percentile(absc, 50)), 4),
            "abscos_p90": round(float(np.percentile(absc, 90)), 4),
            "frac_uncoupled_2xfloor": round(float(np.mean(absc < 2 * floor)), 3),
            "frac_neg": round(float(np.mean(c < 0)), 3),
            "corr_absc_norm": round(spearman(np.abs(c), rown.numpy()), 3),
            "gini_absc": round(gini(np.abs(c)), 3),
            "floor": round(floor, 4)}


def analyze(name, docs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    d = m.config.hidden_size
    NL = len(Lz)
    floor = math.sqrt(2.0 / (math.pi * d))
    layers = sorted(set(int(round(f * NL)) for f in (0.25, 0.5, 0.75)))

    cap = {}
    hooks = [Lz[L].mlp.gate_proj.register_forward_hook(
        (lambda L: (lambda mod, inp, out: cap.__setitem__(
            L, inp[0][0].detach())))(L)) for L in layers]
    s1 = {L: torch.zeros(d, dtype=torch.float64) for L in layers}
    n = 0
    for text in docs[:N_CAL]:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            s1[L] += cap[L][SKIP:].double().sum(0)
        n += 1
    for h in hooks:
        h.remove()

    out = {"n_layers": NL, "d": d, "d_ff": m.config.intermediate_size,
           "floor": round(floor, 4), "layers": layers, "per_layer": {}}
    for L in layers:
        bL = (s1[L] / max(n, 1)).float()
        bLn = bL / bL.norm().clamp(min=1e-9)
        Wg = Lz[L].mlp.gate_proj.weight.detach()
        Wv = Lz[L].mlp.up_proj.weight.detach()
        out["per_layer"][f"L{L}"] = {
            "gate": branch_stats(Wg, bL, bLn, floor),
            "value": branch_stats(Wv, bL, bLn, floor)}
    del m
    return out


def main():
    t0 = time.time()
    docs = json.load(open(PILE, encoding="utf-8"))
    res = {}
    print(f"{'model':>26} {'L':>4} {'br':>5} {'cos p10/50/90':>20} "
          f"{'|cos|p50':>9} {'unc%':>6} {'neg%':>6} {'r(|c|,‖w‖)':>11} {'gini':>6}",
          flush=True)
    for name in MODELS:
        try:
            r = analyze(name, docs)
        except Exception as ex:
            print(f"  {name}: SKIP ({type(ex).__name__}: {str(ex)[:60]})", flush=True)
            continue
        res[name] = r
        for L in r["layers"]:
            for br in ("gate", "value"):
                e = r["per_layer"][f"L{L}"][br]
                print(f"{name[-24:]:>26} {L:>4} {br:>5} "
                      f"{e['cos_p10']:+.3f}/{e['cos_p50']:+.3f}/{e['cos_p90']:+.3f}  "
                      f"{e['abscos_p50']:>9.3f} "
                      f"{100*e['frac_uncoupled_2xfloor']:>5.1f} "
                      f"{100*e['frac_neg']:>5.1f} {e['corr_absc_norm']:>11.3f} "
                      f"{e['gini_absc']:>6.3f}", flush=True)
        json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: gate frac_neg high + |cos|p50 >> floor + gini moderate => a "
          "graded,\nmostly-inhibitory coupling spread across units (not uniform, "
          "not one-neuron).\nvalue frac_uncoupled high + |cos|p50 ~ floor => value "
          "branch is uncoupled.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
