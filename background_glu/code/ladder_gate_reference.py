"""THE CONNECTING EXPERIMENT: does adding a gate relocate b_L BECAUSE the b_L
reference coupling moves into the gate branch? (Olli, 2026-09-09.)

glu_paper's one alive result: from bit-identical init, swiglu-vs-bilinear b_L
cosine falls to ~0.118 (adding a SiLU relocates b_L) while gelu-vs-relu2 holds at
~0.926 (swapping the elementwise fn does not). The 2026-09-09 Qwen finding: in a
SwiGLU MLP the b_L coupling concentrates in the GATE branch (gate rows align with
b_L 4-33x the value; gate rests closed). This script unifies them ON THE LADDER:

  RELOCATION   cos(b_L_swiglu, b_L_bilinear) vs cos(b_L_gelu, b_L_relu2), per layer.
  MECHANISM    swiglu: |cos(w_gate,b_L)| / |cos(w_up,b_L)| (>> 1 => coupling in the
               gate) + gate resting SiLU(w_gate.b_L) (< 0 => rests closed).
               bilinear: |cos(w_a,b_L)| / |cos(w_b,b_L)| -- the BUILT-IN NULL, the
               two branches are structurally symmetric (bit-identical init to
               swiglu's gate/up), so the ratio must be ~1 and there is no
               resting-closed threshold (no SiLU). If swiglu's ratio >> 1 while
               bilinear's ~1, the SiLU is what puts the coupling in the gate, and
               that is why swiglu's b_L relocates.
  non-gated    gelu/relu2: within-matrix coupling band |c_n|/||w_n|| + resting
               w_in.b_L -- the paper's single-matrix decomposition, insensitive to
               the elementwise fn (hence b_L stays put).

Usage: ./.venv/Scripts/python.exe minout/code/ladder_gate_reference.py
       STEPS=0,12000,150000
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_probe import load_snapshot, eval_batch, empty_cache   # noqa: E402

STEPS = os.environ.get("STEPS", "0,12000,150000").split(",")
OUT = os.environ.get("OUT", "minout/data/ladder-gate-reference.json")
SKIP = 8
ARMS = ("swiglu", "bilinear", "doublegate", "gelu", "relu2")
# arm -> (read1 attr, elementwise fn on read1, read2 attr or None)
BRANCH = {
    "swiglu": ("w_gate", F.silu, "w_up"),
    "bilinear": ("w_a", None, "w_b"),
    "doublegate": ("w_a", F.silu, "w_b"),   # both branches SiLU; ratio = g1/g2
    "gelu": ("dense_h_to_4h", None, None),
    "relu2": ("w_in", None, None),
}


@torch.no_grad()
def bL_per_layer(model, ids):
    """Mean MLP input (post-attention-LN output) per layer over tokens."""
    layers = model.gpt_neox.layers
    acc = [None] * len(layers)
    cnt = [0] * len(layers)

    def cap(L):
        def h(mod, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])[SKIP:]
            acc[L] = x.sum(0) if acc[L] is None else acc[L] + x.sum(0)
            cnt[L] += x.shape[0]
        return h
    hks = [lay.mlp.register_forward_pre_hook(cap(L))
           for L, lay in enumerate(layers)]
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
    for h in hks:
        h.remove()
    return [acc[L] / cnt[L] for L in range(len(layers))]


def arm_stats(model, arm, bL):
    a1, fn, a2 = BRANCH[arm]
    layers = model.gpt_neox.layers
    out = {"gate_over_val_bL": [], "abscos_read1_bL": [], "abscos_read2_bL": [],
           "rest_med": [], "band_med": []}
    for L, lay in enumerate(layers):
        b = bL[L]
        bn = b / b.norm().clamp_min(1e-9)
        W1 = getattr(lay.mlp, a1).weight.detach().float()
        c1 = ((W1 @ bn) / W1.norm(dim=1).clamp_min(1e-9)).abs()
        rest1 = W1 @ b
        out["abscos_read1_bL"].append(float(c1.median()))
        # resting: SiLU(gate.b_L) for swiglu, else the raw resting w.b_L median
        r = fn(rest1) if fn is not None else rest1
        out["rest_med"].append(float(r.median()))
        # within-matrix coupling band |c_n|/||w_n|| (the paper's decomposition)
        out["band_med"].append(float((rest1.abs() / W1.norm(dim=1).clamp_min(1e-9)
                                      / b.norm().clamp_min(1e-9)).median()))
        if a2 is not None:
            W2 = getattr(lay.mlp, a2).weight.detach().float()
            c2 = ((W2 @ bn) / W2.norm(dim=1).clamp_min(1e-9)).abs()
            out["abscos_read2_bL"].append(float(c2.median()))
            out["gate_over_val_bL"].append(
                float(c1.median() / c2.median().clamp_min(1e-6)))
    return out


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    ids = eval_batch()
    for step in STEPS:
        bLs = {}
        stats = {}
        for arm in ARMS:
            p = "minout/data/ladder/{}_step{}.pt".format(arm, step)
            if not os.path.exists(p):
                continue
            model, a, st = load_snapshot(p)
            bL = bL_per_layer(model, ids)
            bLs[arm] = [v.cpu() for v in bL]
            stats[arm] = arm_stats(model, arm, bL)
            del model
            empty_cache()
        # cross-arm b_L relocation cosines, per layer -> median
        def xcos(x, y):
            if x not in bLs or y not in bLs:
                return None
            cs = [float(F.cosine_similarity(bLs[x][L], bLs[y][L], dim=0))
                  for L in range(len(bLs[x]))]
            return round(float(np.median(cs)), 4)
        entry = {"step": step,
                 "cos_bL_swiglu_bilinear": xcos("swiglu", "bilinear"),
                 "cos_bL_gelu_relu2": xcos("gelu", "relu2"),
                 "cos_bL_doublegate_swiglu": xcos("doublegate", "swiglu"),
                 "cos_bL_doublegate_bilinear": xcos("doublegate", "bilinear"),
                 "arms": {a: {k: [round(x, 5) for x in v] for k, v in s.items()}
                          for a, s in stats.items()}}
        res[str(step)] = entry
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
        print(f"\n=== step {step} ===")
        print(f"  RELOCATION: cos(b_L swiglu,bilinear) = "
              f"{entry['cos_bL_swiglu_bilinear']}  | "
              f"cos(b_L gelu,relu2) = {entry['cos_bL_gelu_relu2']}")
        for a in ARMS:
            if a not in stats:
                continue
            s = stats[a]
            gv = (f"gate/val b_L {np.median(s['gate_over_val_bL']):.1f}"
                  if s["gate_over_val_bL"] else "single-matrix")
            print(f"  {a:>9}: {gv:>22}  rest(med) "
                  f"{np.median(s['rest_med']):+.3f}  "
                  f"|cos(read1,b_L)| {np.median(s['abscos_read1_bL']):.3f}")
    print("\nREAD: swiglu gate/val >> 1 + rest<0 while bilinear ~1 + no closed "
          "rest\n=> the SiLU puts the coupling in the gate, relocating b_L "
          "(low swiglu-bilinear\ncosine) while gelu-relu2 (coupling in the single "
          "matrix) stays aligned.")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
