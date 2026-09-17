# -*- coding: utf-8 -*-
"""Behavioural check of the switch<->multiplier labels (Olli, 2026-09-13).

The paper labels gate units by GEOMETRY (coupling to b_L). This tests whether the
BEHAVIOUR matches: does a unit's actual output SiLU(g)*v track the bilinear
product g*v (the multiplier reading), and does that fidelity fall off with
coupling (the switch reading)?

Per gate unit, on real tokens:
  g = w_g.x_ln, v = w_v.x_ln, out = SiLU(g)*v (actual output coeff), prod = g*v
  R2_bilinear = corr(out, prod)^2       (1 => output IS the product; multiplier)
  a = cov(out,prod)/var(prod)           (best-fit slope; SiLU'(0)=0.5 predicts ~0.5)
  |cos| = |cos(w_g, b_hat_L)|           (the geometric coordinate)
Prediction if geometry==behaviour: uncoupled (knee) units R2->1, a->~0.5;
coupled (deep switch) units R2 low (saturation decorrelates out from the product).
Report R2 by group + Spearman(R2, |cos|) (predict negative).

Qwen2.5-3B / 1.5B, GPU fp16 (cached). Usage:
  ./.venv/Scripts/python.exe superposition/code/census_gate_bilinear_fidelity.py
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
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEV == "cuda" else torch.float32
MODELS = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B"]
MAXLEN, SKIP = 256, 16
N_CAL = 24
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gate-bilinear-fidelity.json")


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def analyze(name, docs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    NL, d = len(Lz), m.config.hidden_size
    dff = m.config.intermediate_size
    floor = math.sqrt(2.0 / (math.pi * d))
    layers = sorted(set(int(round(f * NL)) for f in (0.25, 0.5, 0.75)))

    cap = {}
    hooks = [Lz[L].mlp.gate_proj.register_forward_hook(
        (lambda L: (lambda mod, inp, out: cap.__setitem__(
            L, inp[0][0].detach())))(L)) for L in layers]
    # streaming sums per unit (float64 on device)
    acc = {L: {k: torch.zeros(dff, dtype=torch.float64, device=DEV)
               for k in ("so", "so2", "sp", "sp2", "sop")} for L in layers}
    n = {L: 0 for L in layers}
    sx = {L: torch.zeros(d, dtype=torch.float64, device=DEV) for L in layers}

    for k, text in enumerate(docs):
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            x = cap[L][SKIP:].float()                          # [P, d]
            Wg = Lz[L].mlp.gate_proj.weight.float()
            Wv = Lz[L].mlp.up_proj.weight.float()
            g = x @ Wg.T                                        # [P, dff]
            v = x @ Wv.T
            out = F.silu(g) * v
            prod = g * v
            a = acc[L]
            a["so"] += out.sum(0).double(); a["so2"] += (out * out).sum(0).double()
            a["sp"] += prod.sum(0).double(); a["sp2"] += (prod * prod).sum(0).double()
            a["sop"] += (out * prod).sum(0).double()
            sx[L] += x.sum(0).double()
            n[L] += x.shape[0]
        if (k + 1) % 8 == 0:
            print(f"    [{name[-12:]}] doc {k+1}/{len(docs)}", flush=True)
    for h in hooks:
        h.remove()

    res = {"n_layers": NL, "d": d, "floor": round(floor, 4), "layers": layers,
           "per_layer": {}}
    for L in layers:
        N = n[L]; a = acc[L]
        mo, mp = a["so"] / N, a["sp"] / N
        vo = (a["so2"] / N - mo * mo).clamp(min=1e-12)
        vp = (a["sp2"] / N - mp * mp).clamp(min=1e-12)
        cov = a["sop"] / N - mo * mp
        r2 = (cov * cov / (vo * vp)).cpu().numpy()             # corr^2
        slope = (cov / vp).cpu().numpy()
        bL = (sx[L] / N).float()
        bn = bL / bL.norm().clamp(min=1e-9)
        Wg = Lz[L].mlp.gate_proj.weight.float()
        cosv = ((Wg @ bn) / Wg.norm(dim=1).clamp(min=1e-9)).abs().cpu().numpy()
        fin = np.isfinite(r2) & np.isfinite(slope)
        r2, slope, cosv = r2[fin], slope[fin], cosv[fin]
        unc = cosv < 2 * floor
        cpl = ~unc
        e = {"r2_uncoupled_med": round(float(np.median(r2[unc])), 3) if unc.any() else None,
             "r2_coupled_med": round(float(np.median(r2[cpl])), 3) if cpl.any() else None,
             "slope_uncoupled_med": round(float(np.median(slope[unc])), 3) if unc.any() else None,
             "slope_coupled_med": round(float(np.median(slope[cpl])), 3) if cpl.any() else None,
             "spearman_r2_vs_coupling": round(spearman(r2, cosv), 3),
             "frac_uncoupled": round(float(unc.mean()), 3),
             "r2_all_med": round(float(np.median(r2)), 3)}
        res["per_layer"][f"L{L}"] = e
    del m
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    t0 = time.time()
    docs = json.load(open(PILE, encoding="utf-8"))[:N_CAL]   # <-- slice; N_CAL calibration docs
    res = {}
    print(f"{'model':>18} {'L':>4} {'R2 unc':>7} {'R2 cpl':>7} {'a unc':>6} "
          f"{'a cpl':>6} {'spear(R2,cos)':>14} {'unc%':>6}", flush=True)
    for name in MODELS:
        try:
            r = analyze(name, docs)
        except Exception as ex:
            print(f"  {name}: SKIP ({type(ex).__name__}: {str(ex)[:60]})", flush=True)
            continue
        res[name] = r
        for L in r["layers"]:
            e = r["per_layer"][f"L{L}"]
            print(f"{name[-18:]:>18} {L:>4} {str(e['r2_uncoupled_med']):>7} "
                  f"{str(e['r2_coupled_med']):>7} {str(e['slope_uncoupled_med']):>6} "
                  f"{str(e['slope_coupled_med']):>6} "
                  f"{e['spearman_r2_vs_coupling']:>14.3f} "
                  f"{100*e['frac_uncoupled']:>5.1f}", flush=True)
        json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: geometry==behaviour if uncoupled (knee) units have R2->1 and "
          "slope~0.5\n(their output IS the bilinear product), coupled (switch) "
          "units have low R2, and\nSpearman(R2,|cos|) is negative (more coupled => "
          "less bilinear).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
