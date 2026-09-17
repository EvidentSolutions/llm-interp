"""Cross-arm operating-point spectrum (Olli, 2026-09-13, for background_glu).

Thesis to ground: a GLU gate spans a SWITCH<->MULTIPLIER continuum, its position
set by coupling to the carried reference b_L, and this is why one unit type covers
both roles. bilinear can only be a multiplier (no off-state); a single LU unit can
only be a switch (no key/value decoupling). Show it on the bit-identical-init
ladder at step 150000.

Per arm, for the "gate/switch-axis" read branch (gelu dense_h_to_4h, swiglu
w_gate, bilinear w_a, relu2 w_in), pooled over 3 mid-stack layers, per unit:
  resting = w . b_L                       operating point (mean pre-activation)
  duty    = P(pre-activation > 0)          off-by-default if low, knee if ~0.5
  |cos|   = |cos(w, b_hat_L)|              coupling to the reference
Report: median resting, frac off-by-default (duty<0.15), frac at-knee
(0.35<duty<0.65), |cos| median and p10/p90 spread, and (gated arms) cos(gate,val)
= decoupling. SwiGLU should show BOTH a closed-switch population and a knee
population (wide spread); bilinear only the knee (frac_off ~0); gelu/relu2 mostly
closed switches but with no second branch to decouple.

Usage: ./.venv/Scripts/python.exe minout/code/census_ladder_gate_spectrum.py
"""
import os
import sys
import json

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_probe import load_snapshot, eval_batch, empty_cache            # noqa: E402

STEP = os.environ.get("STEP", "150000")
LDIR = "minout/data/ladder"
OUT = "minout/data/ladder-gate-spectrum.json"
SKIP = 8
# gate/switch-axis branch, value branch (None if single matrix), down attr
BRANCH = {"gelu": ("dense_h_to_4h", None), "swiglu": ("w_gate", "w_up"),
          "bilinear": ("w_a", "w_b"), "relu2": ("w_in", None)}


@torch.no_grad()
def measure(model, arm, ids, layers):
    Lz = model.gpt_neox.layers
    a1, a2 = BRANCH[arm]
    cap = {}
    hks = [Lz[L].mlp.register_forward_pre_hook(
        (lambda L: (lambda mod, args: cap.__setitem__(L, args[0].detach())))(L))
        for L in layers]
    # b_L accumulation
    s1 = {L: None for L in layers}
    cnt = {L: 0 for L in layers}
    xs = {L: [] for L in layers}
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
        for L in layers:
            x = cap[L][0][SKIP:].float()
            s1[L] = x.sum(0) if s1[L] is None else s1[L] + x.sum(0)
            cnt[L] += x.shape[0]
            xs[L].append(x)
    for h in hks:
        h.remove()

    rest_all, duty_all, cos_all = [], [], []
    cos_gv_all = []
    for L in layers:
        bL = s1[L] / cnt[L]
        bLn = bL / bL.norm().clamp_min(1e-9)
        W1 = getattr(Lz[L].mlp, a1).weight.detach().float()      # [units, d]
        X = torch.cat(xs[L], 0)                                  # [P, d]
        Z = X @ W1.T                                             # [P, units]
        rest = (W1 @ bL)                                         # [units] = w.b_L
        duty = (Z > 0).float().mean(0)                           # [units]
        cos1 = ((W1 @ bLn) / W1.norm(dim=1).clamp_min(1e-9)).abs()
        rest_all.append(rest.cpu().numpy())
        duty_all.append(duty.cpu().numpy())
        cos_all.append(cos1.cpu().numpy())
        if a2 is not None:
            W2 = getattr(Lz[L].mlp, a2).weight.detach().float()
            cos_gv = torch.nn.functional.cosine_similarity(W1, W2, dim=1)
            cos_gv_all.append(cos_gv.cpu().numpy())
    rest = np.concatenate(rest_all)
    duty = np.concatenate(duty_all)
    cos = np.concatenate(cos_all)
    return {
        "rest_med": round(float(np.median(rest)), 4),
        "rest_p10": round(float(np.percentile(rest, 10)), 4),
        "frac_off_by_default": round(float(np.mean(duty < 0.15)), 3),
        "frac_at_knee": round(float(np.mean((duty > 0.35) & (duty < 0.65))), 3),
        "duty_med": round(float(np.median(duty)), 3),
        "abscos_med": round(float(np.median(cos)), 4),
        "abscos_p10": round(float(np.percentile(cos, 10)), 4),
        "abscos_p90": round(float(np.percentile(cos, 90)), 4),
        "cos_gate_val_med": (round(float(np.median(np.concatenate(cos_gv_all))), 4)
                             if cos_gv_all else None),
        "decoupled_branches": a2 is not None,
        "n_units_pooled": int(len(rest))}


def main():
    ids = eval_batch()
    res = {}
    print(f"{'arm':>10} {'rest_med':>9} {'rest_p10':>9} {'off%':>6} {'knee%':>6} "
          f"{'|cos|med':>9} {'|cos|p10-p90':>14} {'cos(g,v)':>9} {'decouple':>9}")
    for arm in ("gelu", "swiglu", "bilinear", "relu2"):
        p = f"{LDIR}/{arm}_step{STEP}.pt"
        if not os.path.exists(p):
            print(f"  {arm}: missing {p}")
            continue
        model, a, st = load_snapshot(p)
        NL = len(model.gpt_neox.layers)
        layers = sorted(set(int(round(f * NL)) for f in (0.33, 0.5, 0.67)))
        e = measure(model, arm, ids, layers)
        res[arm] = {"step": st, "layers": layers, **e}
        print(f"{arm:>10} {e['rest_med']:>9.3f} {e['rest_p10']:>9.3f} "
              f"{100*e['frac_off_by_default']:>5.1f} {100*e['frac_at_knee']:>5.1f} "
              f"{e['abscos_med']:>9.4f} "
              f"{e['abscos_p10']:.3f}-{e['abscos_p90']:.3f}   "
              f"{str(e['cos_gate_val_med']):>9} "
              f"{str(e['decoupled_branches']):>9}")
        del model
        empty_cache()
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}")
    print("\nREAD: swiglu = BOTH off-by-default switches (off%>0) AND knee "
          "multipliers (knee%>0),\nwide |cos| spread + decoupled branches. "
          "bilinear = knee only (off%~0), decoupled but\nno switch. gelu/relu2 = "
          "switches (off%>0) but NOT decoupled (single matrix). => only\nswiglu "
          "spans the switch<->multiplier continuum with independent key/value.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
