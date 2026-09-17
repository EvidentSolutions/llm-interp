# -*- coding: utf-8 -*-
"""MoE pilot (Olli, 2026-09-13, for background_glu discussion): does the
switch/reference reading extend INSIDE experts, with an expert-conditional
reference?

Each expert in an MoE is a SwiGLU MLP. Our dense finding says the gate couples to
the carried reference = mean post-LN MLP input. In an MoE, an expert only sees the
tokens ROUTED to it, so its reference should be the mean over its OWN routed
tokens, b_L^(e), not the global mean. This pilot measures, per expert e, at a few
mid-stack layers of OLMoE-1B-7B:
  gate/value ratio      median |cos(w_g,b_e)| / |cos(w_v,b_e)|  (coupling in gate?)
  rest SiLU(w_g.b_e)    does the expert gate rest closed on its OWN routed ref?
  cos(b_e, b_global)    routed reference vs the global mean  (<1 => conditional)
  cos(b_e, b_e') mean   cross-expert reference divergence     (<1 => expert-specific)
  cos(b_e, r_e)         routed reference vs the expert's ROUTER direction
                        (the bridge to Li et al. 2606.00761)
  own vs global gate    median |cos(w_g,b_e)| vs |cos(w_g,b_global)|
                        (is the gate more aligned to its own routed ref?)

CPU, bf16. Weights + routed means only (no per-token storage). Router top-k is
recomputed from the captured block input and the router weight.
Usage: .venv/Scripts/python.exe superposition/code/census_moe_expert_reference.py
"""
import os
import sys
import json
import time
import itertools

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ["HF_HUB_OFFLINE"] = "0"        # allow the one-time download
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cpu"
DTYPE = torch.bfloat16
MODEL = "allenai/OLMoE-1B-7B-0924"
MAXLEN, SKIP = 256, 16
N_CAL = 24
MIN_TOK = 50                              # min routed tokens to keep an expert
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-moe-expert-reference.json")


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    NL = len(Lz)
    d = m.config.hidden_size
    ne = m.config.num_experts
    topk = m.config.num_experts_per_tok
    layers = sorted(set(int(round(f * NL)) for f in (0.25, 0.5, 0.75)))
    print(f"  {MODEL}: {NL} layers, d={d}, {ne} experts, top-{topk} "
          f"({time.time()-t0:.0f}s to load)", flush=True)

    cap = {}
    hooks = [Lz[L].mlp.register_forward_pre_hook(
        (lambda L: (lambda mod, args: cap.__setitem__(L, args[0].detach())))(L))
        for L in layers]

    docs = json.load(open(PILE, encoding="utf-8"))[:N_CAL]
    se = {L: torch.zeros(ne, d, dtype=torch.float64) for L in layers}   # per-expert sum
    ce = {L: torch.zeros(ne, dtype=torch.float64) for L in layers}      # per-expert count
    sg = {L: torch.zeros(d, dtype=torch.float64) for L in layers}       # global sum
    cg = {L: 0 for L in layers}
    for k, text in enumerate(docs):
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            x = cap[L][0][SKIP:].float()                     # [P, d]
            rw = Lz[L].mlp.gate.weight.detach().float()      # [ne, d]
            logits = x @ rw.T                                # [P, ne]
            top = logits.topk(topk, dim=-1).indices          # [P, topk]
            oneh = torch.zeros(x.shape[0], ne)
            oneh.scatter_(1, top, 1.0)                       # [P, ne] routed mask
            se[L] += (oneh.T.double() @ x.double())          # [ne, d]
            ce[L] += oneh.sum(0).double()
            sg[L] += x.sum(0).double(); cg[L] += x.shape[0]
        if (k + 1) % 6 == 0:
            print(f"    doc {k+1}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()

    res = {}
    print(f"\n  {'L':>3} {'#exp':>5} {'g/v ratio':>10} {'restSiLU':>9} "
          f"{'cos(be,bg)':>11} {'cos(be,be)':>11} {'cos(be,re)':>11} "
          f"{'own/glob gate':>13}", flush=True)
    for L in layers:
        b_g = (sg[L] / max(cg[L], 1)).float()
        bgn = b_g / b_g.norm().clamp(min=1e-9)
        rw = Lz[L].mlp.gate.weight.detach().float()          # router dirs [ne,d]
        keep = [e for e in range(ne) if ce[L][e] >= MIN_TOK]
        gv, rest, cbg, cbr, own, glob = [], [], [], [], [], []
        bns = []
        for e in keep:
            b_e = (se[L][e] / ce[L][e]).float()
            bn = b_e / b_e.norm().clamp(min=1e-9)
            bns.append(bn)
            Wg = Lz[L].mlp.experts[e].gate_proj.weight.detach().float()
            Wv = Lz[L].mlp.experts[e].up_proj.weight.detach().float()
            cg_ = ((Wg @ bn) / Wg.norm(dim=1).clamp(min=1e-9)).abs()
            cv_ = ((Wv @ bn) / Wv.norm(dim=1).clamp(min=1e-9)).abs()
            gv.append(float(cg_.median() / cv_.median().clamp(min=1e-6)))
            rest.append(float(F.silu(Wg @ b_e).median()))
            cbg.append(float(F.cosine_similarity(bn, bgn, dim=0)))
            rn = rw[e] / rw[e].norm().clamp(min=1e-9)
            cbr.append(float(F.cosine_similarity(bn, rn, dim=0)))
            own.append(float(cg_.median()))
            glob.append(float(((Wg @ bgn) / Wg.norm(dim=1).clamp(min=1e-9)).abs().median()))
        # cross-expert reference divergence (sample 300 pairs)
        pairs = list(itertools.combinations(range(len(bns)), 2))
        if len(pairs) > 300:
            idx = np.random.default_rng(0).choice(len(pairs), 300, replace=False)
            pairs = [pairs[i] for i in idx]
        cbe = [float(F.cosine_similarity(bns[i], bns[j], dim=0)) for i, j in pairs]
        e = {"n_experts_kept": len(keep),
             "gate_over_val_ratio_med": round(med(gv), 2),
             "rest_silu_med": round(med(rest), 4),
             "cos_be_bglobal_med": round(med(cbg), 3),
             "cos_be_be_med": round(med(cbe), 3),
             "cos_be_router_med": round(med(cbr), 3),
             "gate_own_med": round(med(own), 4),
             "gate_global_med": round(med(glob), 4)}
        res[f"L{L}"] = e
        print(f"  {L:>3} {len(keep):>5} {e['gate_over_val_ratio_med']:>10.2f} "
              f"{e['rest_silu_med']:>9.4f} {e['cos_be_bglobal_med']:>11.3f} "
              f"{e['cos_be_be_med']:>11.3f} {e['cos_be_router_med']:>11.3f} "
              f"{e['gate_own_med']:.4f}/{e['gate_global_med']:.4f}", flush=True)
    json.dump({"model": MODEL, "num_experts": ne, "top_k": topk,
               "layers": layers, "results": res},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: g/v ratio >1 => coupling in the gate inside experts (dense "
          "finding holds).\nrestSiLU<0 => expert gate rests closed on its OWN "
          "routed reference. cos(be,bg)<1 and\ncos(be,be)<1 => the reference is "
          "expert-conditional. gate_own > gate_global => the\ngate is tuned to its "
          "routed mean, not the global one. cos(be,re) = router-alignment.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
