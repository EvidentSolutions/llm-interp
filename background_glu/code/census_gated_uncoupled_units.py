# -*- coding: utf-8 -*-
"""What are the UNCOUPLED gate units, and what (if anything) do they couple to?
(Olli, 2026-09-13, for background_glu.)

census_gated_coupling_dist found a growing lower tail of gate units effectively
uncoupled from b_L (|cos(w_g, b_hat_L)| < 2*floor), rising from ~4% (shallow) to
~40% (deep). In a bias-free RMSNorm SwiGLU a gate's only fixed operating point is
w_g . b_L, so an uncoupled unit either (a) rests near the SiLU knee and is
input-driven (a CONTENT gate, no maintained bias), or (b) couples to a SECOND
carried direction. This test decides which, per layer, splitting gate units into
UNCOUPLED (|cos|<2*floor) vs COUPLED:

  rest_med / silu_rest   median w_g.b_L and SiLU(w_g.b_L)  (near 0 => rests at knee)
  duty_med               median P(pre-activation > 0) over positions (~0.5 => open)
  constancy_med          median |mean_z|/std_z per unit (>>1 => constant/off-by-default;
                         ~0 => input-driven content gate)   [z = gate pre-activation]
  cos_gv_med             median cos(w_g, w_v)  (value-likeness of the gate row)
  svd top1/top3 energy of the uncoupled rows + |cos(top right-singular, b_hat_L)|
                         (a shared secondary axis => a second reference)
  abscos bins            is "uncoupled" a discrete cluster or a smooth tail?

Qwen2.5-3B / 1.5B, TinyLlama. CPU/float32; one eval pass for duty/variance.
Usage: .venv/Scripts/python.exe superposition/code/census_gated_uncoupled_units.py
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
DTYPE = torch.float32
MAXLEN, SKIP = 160, 16
N_CAL = 16
MODELS = ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-1.5B",
          "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-gated-uncoupled-units.json")


def med(x):
    return float(np.median(x)) if len(x) else float("nan")


def svd_axis(W, bLn):
    """top-1 / top-3 singular energy share of rows W, and |cos(V[0], b_hat_L)|."""
    if W.shape[0] < 3:
        return None
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    s2 = (S * S)
    tot = float(s2.sum().clamp(min=1e-12))
    return {"top1_energy": round(float(s2[0] / tot), 3),
            "top3_energy": round(float(s2[:3].sum() / tot), 3),
            "cos_top_bL": round(float((Vh[0] @ bLn).abs()), 3)}


def analyze(name, docs):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=DTYPE, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    d = m.config.hidden_size
    NL = len(Lz)
    dff = m.config.intermediate_size
    floor = math.sqrt(2.0 / (math.pi * d))
    layers = sorted(set(int(round(f * NL)) for f in (0.25, 0.5, 0.75)))

    cap = {}
    hooks = []
    for L in layers:
        hooks.append(Lz[L].mlp.gate_proj.register_forward_hook(
            (lambda L: (lambda mod, inp, out: cap.__setitem__(
                L, (inp[0][0].detach(), out[0].detach()))))(L)))

    s1x = {L: torch.zeros(d, dtype=torch.float64, device=DEV) for L in layers}
    sz = {L: torch.zeros(dff, dtype=torch.float64, device=DEV) for L in layers}
    sz2 = {L: torch.zeros(dff, dtype=torch.float64, device=DEV) for L in layers}
    pos = {L: torch.zeros(dff, dtype=torch.float64, device=DEV) for L in layers}
    npos = 0
    for text in docs[:N_CAL]:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))
        for L in layers:
            x, z = cap[L]
            x = x[SKIP:].double(); z = z[SKIP:].double()
            s1x[L] += x.sum(0)
            sz[L] += z.sum(0); sz2[L] += (z * z).sum(0)
            pos[L] += (z > 0).sum(0)
        npos += (len(ids) - SKIP)
    for h in hooks:
        h.remove()

    out = {"n_layers": NL, "d": d, "d_ff": dff, "floor": round(floor, 4),
           "n_pos": npos, "layers": layers, "per_layer": {}}
    for L in layers:
        bL = (s1x[L] / npos).float().cpu()
        bLn = bL / bL.norm().clamp(min=1e-9)
        Wg = Lz[L].mlp.gate_proj.weight.detach().float().cpu()
        Wv = Lz[L].mlp.up_proj.weight.detach().float().cpu()
        rown = Wg.norm(dim=1).clamp(min=1e-9)
        cosg = ((Wg @ bLn) / rown)
        abscos = cosg.abs().numpy()
        unc = abscos < 2 * floor
        cpl = ~unc
        rest = (sz[L] / npos).float().cpu().numpy()          # mean pre-activation = w_g.b_L
        stdz = (sz2[L] / npos - (sz[L] / npos) ** 2).clamp(min=0).sqrt().float().cpu().numpy()
        duty = (pos[L] / npos).float().cpu().numpy()
        constancy = np.abs(rest) / (stdz + 1e-9)
        cosgv = F.cosine_similarity(Wg, Wv, dim=1).numpy()

        def grp(mask):
            return {"n": int(mask.sum()),
                    "frac": round(float(mask.mean()), 3),
                    "rest_med": round(med(rest[mask]), 4),
                    "silu_rest_med": round(float(F.silu(torch.tensor(
                        med(rest[mask]))).item()), 4),
                    "duty_med": round(med(duty[mask]), 3),
                    "constancy_med": round(med(constancy[mask]), 3),
                    "cos_gv_med": round(med(cosgv[mask]), 4),
                    "svd": svd_axis(Wg[torch.tensor(mask)], bLn)}
        b = [float((abscos < floor).mean()),
             float(((abscos >= floor) & (abscos < 2 * floor)).mean()),
             float(((abscos >= 2 * floor) & (abscos < 4 * floor)).mean()),
             float((abscos >= 4 * floor).mean())]
        out["per_layer"][f"L{L}"] = {
            "uncoupled": grp(unc), "coupled": grp(cpl),
            "abscos_bins_<f_<2f_<4f_ge4f": [round(x, 3) for x in b]}
    del m
    return out


def main():
    t0 = time.time()
    docs = json.load(open(PILE, encoding="utf-8"))
    res = {}
    for name in MODELS:
        try:
            r = analyze(name, docs)
        except Exception as ex:
            print(f"  {name}: SKIP ({type(ex).__name__}: {str(ex)[:70]})", flush=True)
            continue
        res[name] = r
        print(f"\n=== {name} (floor {r['floor']}) ===", flush=True)
        for L in r["layers"]:
            e = r["per_layer"][f"L{L}"]
            u, c = e["uncoupled"], e["coupled"]
            print(f"  L{L}: uncoupled {u['frac']*100:4.1f}%  "
                  f"[rest {u['rest_med']:+.3f} duty {u['duty_med']:.2f} "
                  f"const {u['constancy_med']:.2f} cos_gv {u['cos_gv_med']:+.3f} "
                  f"svd1 {u['svd']['top1_energy'] if u['svd'] else float('nan')} "
                  f"cosVbL {u['svd']['cos_top_bL'] if u['svd'] else float('nan')}]",
                  flush=True)
            print(f"       coupled   {c['frac']*100:4.1f}%  "
                  f"[rest {c['rest_med']:+.3f} duty {c['duty_med']:.2f} "
                  f"const {c['constancy_med']:.2f} cos_gv {c['cos_gv_med']:+.3f} "
                  f"svd1 {c['svd']['top1_energy'] if c['svd'] else float('nan')} "
                  f"cosVbL {c['svd']['cos_top_bL'] if c['svd'] else float('nan')}]",
                  flush=True)
            print(f"       abscos bins [<f <2f <4f >=4f]: "
                  f"{e['abscos_bins_<f_<2f_<4f_ge4f']}", flush=True)
        json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    json.dump({"results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: uncoupled with duty~0.5 + constancy~0 + cos_gv up + svd1 low "
          "=> input-driven\ncontent gates (no maintained bias). uncoupled with a "
          "high svd1 + low cosVbL =>\nthey share a SECOND carried direction "
          "(a second reference).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
