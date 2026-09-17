# -*- coding: utf-8 -*-
"""Does the b_L reference coupling localize to the GATE branch across SwiGLU
models, or is it Qwen2.5-3B-specific? (Olli, 2026-09-09; cross-model replication
of census_gated_gate_structure.py Leg B.)

Per model, per mid-stack layer (depth ~0.25/0.5/0.75), weights + b_L only:
  cos(w_g,w_v)        self-gating check (random baseline ~ 0)
  |cos(w_g,bL)| vs |cos(w_v,bL)|   THE headline: does the coupling sit in the gate
  gate/value hivar-energy          does the gate over-read the amplitude rail
  resting SiLU(w_g.bL)             does the gate rest closed
The claim replicates if, in every SwiGLU model, gate rows align with b_L markedly
more than value rows (ratio >> 1) and the gate rests closed.

All Llama-style access (mlp.gate_proj/up_proj, post_attention_layernorm). Local.
Usage: .venv/Scripts/python.exe superposition/code/census_gated_branch_xmodel.py
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = [
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "HuggingFaceTB/SmolLM2-1.7B",
    "meta-llama/Llama-3.2-1B",          # gated repo; skipped cleanly if unavailable
]
MAXLEN, SKIP = 256, 16
N_CAL = 16
HIGHVAR_K = 64
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-branch-xmodel.json")


def analyze(name, docs, tok_cache):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    d = m.config.hidden_size
    NL = len(Lz)
    if not hasattr(Lz[0].mlp, "gate_proj"):
        print(f"  {name}: not a GLU MLP, skipping")
        del m
        return None
    layers = [int(round(f * NL)) for f in (0.25, 0.5, 0.75)]

    cap = {}
    hooks = [Lz[L].mlp.gate_proj.register_forward_hook(
        (lambda L: (lambda mod, inp, out: cap.__setitem__(
            L, inp[0][0].detach())))(L)) for L in layers]
    s1 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in layers}
    s2 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in layers}
    n = 0
    for text in docs[:N_CAL]:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            x = cap[L][SKIP:].double()
            s1[L] += x.sum(0); s2[L] += (x * x).sum(0)
        n += len(ids) - SKIP
    for h in hooks:
        h.remove()

    out = {"n_layers": NL, "d": d, "d_ff": m.config.intermediate_size,
           "layers": layers, "per_layer": {}}
    for L in layers:
        bL = (s1[L] / n).float()
        bLn = bL / bL.norm().clamp(min=1e-9)
        var = (s2[L] / n - (s1[L] / n) ** 2).float()
        hv = torch.topk(var, HIGHVAR_K).indices
        mask = torch.zeros(d, dtype=torch.bool, device=DEV); mask[hv] = True
        Wg = Lz[L].mlp.gate_proj.weight.detach().float()
        Wv = Lz[L].mlp.up_proj.weight.detach().float()
        cgv = F.cosine_similarity(Wg, Wv, dim=1)
        cgb = ((Wg @ bLn) / Wg.norm(dim=1).clamp(min=1e-9)).abs()
        cvb = ((Wv @ bLn) / Wv.norm(dim=1).clamp(min=1e-9)).abs()
        g_hi = ((Wg[:, mask] ** 2).sum(1) / (Wg ** 2).sum(1).clamp(min=1e-9))
        v_hi = ((Wv[:, mask] ** 2).sum(1) / (Wv ** 2).sum(1).clamp(min=1e-9))
        rest = Wg @ bL
        e = {"cos_wg_wv_med": float(cgv.median()),
             "abscos_wg_bL_med": float(cgb.median()),
             "abscos_wv_bL_med": float(cvb.median()),
             "gate_over_val_bL": float(cgb.median() / cvb.median().clamp(min=1e-6)),
             "gate_hivar_egy": float(g_hi.median()),
             "val_hivar_egy": float(v_hi.median()),
             "hivar_iso": HIGHVAR_K / d,
             "rest_open_silu_med": float(F.silu(rest).median()),
             "frac_gate_open": float((rest > 0).float().mean())}
        out["per_layer"][f"L{L}"] = e
    del m
    torch.cuda.empty_cache() if DEV == "cuda" else None
    return out


def main():
    t0 = time.time()
    docs = json.load(open(PILE, encoding="utf-8"))
    res = {}
    print(f"{'model':>34} {'L':>4} {'cos(wg,wv)':>10} {'|cos wg,bL|':>11} "
          f"{'|cos wv,bL|':>11} {'g/v bL':>7} {'g hi':>7} {'v hi':>7} "
          f"{'restSiLU':>9}")
    for name in MODELS:
        try:
            r = analyze(name, docs, {})
        except Exception as ex:
            print(f"  {name}: FAILED ({type(ex).__name__}: {str(ex)[:60]})")
            continue
        if r is None:
            continue
        res[name] = r
        for L in r["layers"]:
            e = r["per_layer"][f"L{L}"]
            print(f"{name[-32:]:>34} {L:>4} {e['cos_wg_wv_med']:>10.3f} "
                  f"{e['abscos_wg_bL_med']:>11.3f} {e['abscos_wv_bL_med']:>11.3f} "
                  f"{e['gate_over_val_bL']:>7.1f} {e['gate_hivar_egy']:>7.3f} "
                  f"{e['val_hivar_egy']:>7.3f} {e['rest_open_silu_med']:>9.3f}",
                  flush=True)
        # incremental save so a later download/timeout keeps earlier results
        json.dump({"models": list(res.keys()), "highvar_k": HIGHVAR_K,
                   "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)

    json.dump({"models": list(res.keys()), "highvar_k": HIGHVAR_K,
               "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: g/v bL >> 1 in every model => the reference coupling localizes")
    print("to the gate branch (not Qwen-specific). restSiLU < 0 => gate rests closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
