# -*- coding: utf-8 -*-
"""CONTROL A for background_glu (2026-09-10): is the gate-branch b_L localization
carried by the DISTRIBUTED reference, or merely by the handful of massive
(high-variance) coordinates of x_ln?

In Qwen2.5-3B, b_L places a large share of its norm on ~8 coordinates (the
project's own cross-model sweep found 64-99% on 8 coords in Qwen). A referee can
then object that the gate simply reads those few massive coordinates, not a
distributed reference direction. This control decomposes, per layer,
    b_L = b_massive (top-K variance coords) + b_rest (the complement),
renormalises each, and recomputes the gate/value alignment ratio on EACH. If the
ratio (gate rows align with b_L >> value rows) survives on b_rest, the
localization is to the reference direction, not to the massive coordinates.

Also reports the fraction of ||b_L||^2 on the top-K coords (the 64-99% check) and
whether the gate still "rests closed" (SiLU(W_g . b_rest) < 0) on b_rest.

CPU, float32. Qwen2.5-3B (SwiGLU/RMSNorm), weights + b_L only (static).
Usage: .venv/Scripts/python.exe superposition/code/census_gated_massive_control.py
"""
import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cpu"
DTYPE = torch.float32
MODEL = "Qwen/Qwen2.5-3B"
MAXLEN, SKIP = 192, 16
N_CAL = 24
LAYERS = [9, 18, 27]
KS = [8, 32, 64]                 # massive-coord counts to split out
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-massive-control.json")


def ratio_on(Wg, Wv, bvec):
    """median |cos(w_g, bhat)| / median |cos(w_v, bhat)| and the two medians."""
    bn = bvec / bvec.norm().clamp(min=1e-9)
    cg = ((Wg @ bn) / Wg.norm(dim=1).clamp(min=1e-9)).abs().median()
    cv = ((Wv @ bn) / Wv.norm(dim=1).clamp(min=1e-9)).abs().median()
    return float(cg), float(cv), float(cg / cv.clamp(min=1e-6))


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    d = m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))[:N_CAL]
    print(f"  {MODEL} on {DEV}: {len(Lz)} layers, d={d}", flush=True)

    cap = {}
    hooks = [Lz[L].mlp.gate_proj.register_forward_hook(
        (lambda L: (lambda mod, inp, out: cap.__setitem__(
            L, inp[0][0].detach())))(L)) for L in LAYERS]

    s1 = {L: torch.zeros(d, dtype=torch.float64) for L in LAYERS}
    s2 = {L: torch.zeros(d, dtype=torch.float64) for L in LAYERS}
    n = 0
    for k, text in enumerate(docs):
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in LAYERS:
            x = cap[L][SKIP:].double()
            s1[L] += x.sum(0)
            s2[L] += (x * x).sum(0)
        n += len(ids) - SKIP
        if (k + 1) % 6 == 0:
            print(f"    cal {k+1}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()

    res = {}
    print(f"\n  b_L from {n} positions. split ratio = gate/value |cos| to b_L\n")
    hdr = (f"  {'L':>3} {'variant':>12} {'|cos g|':>9} {'|cos v|':>9} "
           f"{'ratio':>7} {'restSiLU':>9} {'norm-frac':>9}")
    print(hdr)
    for L in LAYERS:
        bL = (s1[L] / n).float()
        var = (s2[L] / n - (s1[L] / n) ** 2).float()
        order = torch.argsort(var, descending=True)
        Wg = Lz[L].mlp.gate_proj.weight.detach().float()
        Wv = Lz[L].mlp.up_proj.weight.detach().float()
        entry = {}
        # full reference
        cg, cv, r = ratio_on(Wg, Wv, bL)
        rest_full = float(F.silu(Wg @ bL).median())
        entry["full"] = {"cos_g": cg, "cos_v": cv, "ratio": r,
                         "rest_silu": rest_full, "norm_frac": 1.0}
        print(f"  {L:>3} {'full':>12} {cg:>9.4f} {cv:>9.4f} {r:>7.1f} "
              f"{rest_full:>9.4f} {1.0:>9.4f}")
        for K in KS:
            top = order[:K]
            nf = float((bL[top] ** 2).sum() / (bL ** 2).sum().clamp(min=1e-12))
            b_rest = bL.clone(); b_rest[top] = 0.0
            b_mass = torch.zeros_like(bL); b_mass[top] = bL[top]
            cgr, cvr, rr = ratio_on(Wg, Wv, b_rest)
            cgm, cvm, rm = ratio_on(Wg, Wv, b_mass)
            rest_r = float(F.silu(Wg @ b_rest).median())
            entry[f"rest_top{K}"] = {"cos_g": cgr, "cos_v": cvr, "ratio": rr,
                                     "rest_silu": rest_r, "norm_frac_removed": nf}
            entry[f"massive_top{K}"] = {"cos_g": cgm, "cos_v": cvm, "ratio": rm,
                                        "norm_frac_kept": nf}
            print(f"  {L:>3} {('rest-top'+str(K)):>12} {cgr:>9.4f} {cvr:>9.4f} "
                  f"{rr:>7.1f} {rest_r:>9.4f} {nf:>9.4f}  (removed)")
            print(f"  {L:>3} {('mass-top'+str(K)):>12} {cgm:>9.4f} {cvm:>9.4f} "
                  f"{rm:>7.1f} {'-':>9} {nf:>9.4f}  (kept)")
        res[f"L{L}"] = entry

    json.dump({"model": MODEL, "device": DEV, "layers": LAYERS, "n_cal": n,
               "ks": KS, "results": res}, open(OUT, "w", encoding="utf-8"),
              indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: if rest-topK ratio stays >> 1 while the norm-fraction removed "
          "is large,\nthe gate localization is to the DISTRIBUTED reference, not "
          "the massive coords.\nIf the ratio collapses on b_rest and lives only "
          "in massive-topK, the gate just\nreads the massive dimensions.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
