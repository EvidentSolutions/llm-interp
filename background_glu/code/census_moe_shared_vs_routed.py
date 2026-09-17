# -*- coding: utf-8 -*-
"""MoE shared-vs-routed contrast (Olli, 2026-09-13, background_glu follow-up).

Qwen1.5-MoE-A2.7B has 60 ROUTED experts (top-4) plus a SHARED expert applied to
every token. Prediction from the switch/reference reading: the shared expert sees
the full distribution, so its reference is the GLOBAL mean and it should read like
a dense MLP (gate couples to b_global, rests closed); routed experts see only
their routed subpopulation, so they couple to their OWN routed reference b_e,
tilted off the global mean. This is the clean contrast OLMoE (no shared expert)
could not provide.

Per mid-stack layer:
  SHARED expert vs b_global : gate/value ratio, rest SiLU(w_g.b_global)
  ROUTED experts vs b_e     : gate/value ratio (med), rest SiLU (med),
                              cos(b_e,b_global), cross-expert cos(b_e,b_e'),
                              cos(b_e,router_e)
  coupling to GLOBAL        : shared gate |cos(w_g,b_global)| vs routed (med)
                              -- is the shared expert more aligned to global?

CPU, bf16. Weights + routed means only. Usage:
  .venv/Scripts/python.exe superposition/code/census_moe_shared_vs_routed.py
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
os.environ["HF_HUB_OFFLINE"] = "0"
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cpu"
DTYPE = torch.bfloat16
MODEL = "Qwen/Qwen1.5-MoE-A2.7B"
MAXLEN, SKIP = 256, 16
N_CAL = 24
MIN_TOK = 50
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-moe-shared-vs-routed.json")


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def ratio_rest(mlpmod, bvec, bn):
    Wg = mlpmod.gate_proj.weight.detach().float()
    Wv = mlpmod.up_proj.weight.detach().float()
    cg = ((Wg @ bn) / Wg.norm(dim=1).clamp(min=1e-9)).abs()
    cv = ((Wv @ bn) / Wv.norm(dim=1).clamp(min=1e-9)).abs()
    return (float(cg.median() / cv.median().clamp(min=1e-6)),
            float(F.silu(Wg @ bvec).median()), float(cg.median()))


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
    has_shared = hasattr(Lz[0].mlp, "shared_expert")
    print(f"  {MODEL}: {NL} layers, d={d}, {ne} routed experts, top-{topk}, "
          f"shared_expert={has_shared} ({time.time()-t0:.0f}s)", flush=True)

    cap = {}
    hooks = [Lz[L].mlp.register_forward_pre_hook(
        (lambda L: (lambda mod, args: cap.__setitem__(L, args[0].detach())))(L))
        for L in layers]

    docs = json.load(open(PILE, encoding="utf-8"))[:N_CAL]
    se = {L: torch.zeros(ne, d, dtype=torch.float64) for L in layers}
    ce = {L: torch.zeros(ne, dtype=torch.float64) for L in layers}
    sg = {L: torch.zeros(d, dtype=torch.float64) for L in layers}
    cg = {L: 0 for L in layers}
    for k, text in enumerate(docs):
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            x = cap[L][0][SKIP:].float()
            rw = Lz[L].mlp.gate.weight.detach().float()
            top = (x @ rw.T).topk(topk, dim=-1).indices
            oneh = torch.zeros(x.shape[0], ne)
            oneh.scatter_(1, top, 1.0)
            se[L] += (oneh.T.double() @ x.double())
            ce[L] += oneh.sum(0).double()
            sg[L] += x.sum(0).double(); cg[L] += x.shape[0]
        if (k + 1) % 6 == 0:
            print(f"    doc {k+1}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
    for h in hooks:
        h.remove()

    res = {}
    print(f"\n  {'L':>3} || {'SHARED g/v':>10} {'sh rest':>8} {'sh|cos|g':>8} "
          f"|| {'rtd g/v':>8} {'rtd rest':>8} {'rtd|cos|g':>9} "
          f"{'cos(be,bg)':>10} {'cos(be,be)':>10} {'cos(be,re)':>10}", flush=True)
    for L in layers:
        b_g = (sg[L] / max(cg[L], 1)).float()
        bgn = b_g / b_g.norm().clamp(min=1e-9)
        rw = Lz[L].mlp.gate.weight.detach().float()
        # SHARED expert reads the full distribution => reference is b_global
        sh = Lz[L].mlp.shared_expert
        sh_ratio, sh_rest, sh_cosg = ratio_rest(sh, b_g, bgn)
        # ROUTED experts vs their own routed reference
        keep = [e for e in range(ne) if ce[L][e] >= MIN_TOK]
        gv, rest, cbg, cbr, glob = [], [], [], [], []
        bns = []
        for e in keep:
            b_e = (se[L][e] / ce[L][e]).float()
            bn = b_e / b_e.norm().clamp(min=1e-9)
            bns.append(bn)
            r, rs, _ = ratio_rest(Lz[L].mlp.experts[e], b_e, bn)
            gv.append(r); rest.append(rs)
            cbg.append(float(F.cosine_similarity(bn, bgn, dim=0)))
            rn = rw[e] / rw[e].norm().clamp(min=1e-9)
            cbr.append(float(F.cosine_similarity(bn, rn, dim=0)))
            Wg = Lz[L].mlp.experts[e].gate_proj.weight.detach().float()
            glob.append(float(((Wg @ bgn) / Wg.norm(dim=1).clamp(min=1e-9)).abs().median()))
        pairs = list(itertools.combinations(range(len(bns)), 2))
        if len(pairs) > 300:
            idx = np.random.default_rng(0).choice(len(pairs), 300, replace=False)
            pairs = [pairs[i] for i in idx]
        cbe = [float(F.cosine_similarity(bns[i], bns[j], dim=0)) for i, j in pairs]
        e = {"n_routed_kept": len(keep),
             "shared_gate_over_val": round(sh_ratio, 2),
             "shared_rest_silu": round(sh_rest, 4),
             "shared_gate_cos_global": round(sh_cosg, 4),
             "routed_gate_over_val_med": round(med(gv), 2),
             "routed_rest_silu_med": round(med(rest), 4),
             "routed_gate_cos_global_med": round(med(glob), 4),
             "cos_be_bglobal_med": round(med(cbg), 3),
             "cos_be_be_med": round(med(cbe), 3),
             "cos_be_router_med": round(med(cbr), 3)}
        res[f"L{L}"] = e
        print(f"  {L:>3} || {e['shared_gate_over_val']:>10.2f} "
              f"{e['shared_rest_silu']:>8.4f} {e['shared_gate_cos_global']:>8.4f} "
              f"|| {e['routed_gate_over_val_med']:>8.2f} "
              f"{e['routed_rest_silu_med']:>8.4f} "
              f"{e['routed_gate_cos_global_med']:>9.4f} "
              f"{e['cos_be_bglobal_med']:>10.3f} {e['cos_be_be_med']:>10.3f} "
              f"{e['cos_be_router_med']:>10.3f}", flush=True)
    json.dump({"model": MODEL, "num_experts": ne, "top_k": topk,
               "layers": layers, "results": res},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: prediction = SHARED expert reads like a dense MLP against "
          "b_global (rest<0,\nratio>1, high |cos|g); ROUTED experts couple to "
          "their own b_e (cos(be,bg)<1,\ncross-expert cos<1). Compare shared vs "
          "routed |cos|_global to see if the shared\nexpert is the one tuned to "
          "the global reference.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
