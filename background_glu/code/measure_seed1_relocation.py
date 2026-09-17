"""Seed replicate (INIT_SEED=1) of the b_L relocation, measured at step 12000.

Loads the seed-1 arm checkpoints (minout/data/glu_ckpt_seed1/{arm}.pt), computes
b_L = mean post-LN MLP input per layer on the same fixed eval batch as the
original run, and reports the cross-arm cosines. Comparison target (seed 0, step
12000, from ladder-gate-reference.json): swiglu-bilinear 0.224, gelu-relu2 0.892.
Dissociation replicates if swiglu-bilinear << gelu-relu2 again.
"""
import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoConfig, AutoModelForCausalLM          # noqa: E402
from ffn_variants import apply_ffn_variant                          # noqa: E402
from ladder_probe import eval_batch, ARMS                           # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKDIR = os.path.join(os.path.dirname(__file__), "..", "data", "glu_ckpt_seed1")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "seed1-relocation.json")
SKIP = 8
ARMSET = ["gelu", "swiglu", "bilinear", "relu2"]


def build_load(name):
    cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m-deduped")
    cfg.torch_dtype = torch.float32
    m = AutoModelForCausalLM.from_config(cfg).to(DEV, dtype=torch.float32)
    apply_ffn_variant(m, ARMS[name])
    st = torch.load(os.path.join(CKDIR, name + ".pt"),
                    map_location=DEV, weights_only=False)
    m.load_state_dict({k: v.float() for k, v in st["model"].items()})
    m.to(DEV).eval()
    return m, st["step"]


@torch.no_grad()
def bL_per_layer(m, ids):
    layers = m.gpt_neox.layers
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
        m(ids[i:i + 2])
    for h in hks:
        h.remove()
    return [acc[L] / cnt[L] for L in range(len(layers))]


def main():
    ids = eval_batch().to(DEV)
    bLs, steps = {}, {}
    for name in ARMSET:
        m, step = build_load(name)
        bLs[name] = [v.cpu() for v in bL_per_layer(m, ids)]
        steps[name] = step
        del m
        if DEV == "cuda":
            torch.cuda.empty_cache()

    def xcos(a, b):
        cs = [float(F.cosine_similarity(bLs[a][L], bLs[b][L], dim=0))
              for L in range(len(bLs[a]))]
        return float(np.median(cs)), cs

    sb_med, sb = xcos("swiglu", "bilinear")
    gr_med, gr = xcos("gelu", "relu2")
    print("=== seed 1, step 12000 (median over layers) ===")
    print(f"  swiglu-bilinear (add a gate)     cos(b_L) = {sb_med:.4f}   "
          f"[seed0: 0.224]")
    print(f"  gelu-relu2      (swap fn)         cos(b_L) = {gr_med:.4f}   "
          f"[seed0: 0.892]")
    print(f"  steps: {steps}")
    verdict = ("REPLICATES: dissociation present (swiglu-bilinear << gelu-relu2)"
               if sb_med < 0.6 * gr_med else "does NOT clearly replicate")
    print(f"  {verdict}")
    json.dump({"seed": 1, "step": 12000, "steps": steps,
               "cos_bL_swiglu_bilinear": sb_med, "cos_bL_gelu_relu2": gr_med,
               "seed0_swiglu_bilinear": 0.224, "seed0_gelu_relu2": 0.892,
               "per_layer_swiglu_bilinear": [round(v, 4) for v in sb],
               "per_layer_gelu_relu2": [round(v, 4) for v in gr]},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
