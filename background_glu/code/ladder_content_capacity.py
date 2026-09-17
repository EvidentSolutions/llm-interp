"""ANGLE 1: does gating buy content capacity by moving the shared b_L coupling
OUT of the content read matrix? (Olli, 2026-09-09.)

Background paper (GELU): every read row spends a slice of its norm on the shared
b_L coupling (Phi-2: 7-12%). Because it is the SAME direction b_hat_L for all rows,
that coupling is a RANK-1 shared overhead sitting inside the one read matrix -- it
carries no content, yet occupies read-matrix energy and (if strong enough) a
principal component. 2026-09-09 finding: SwiGLU moves that overhead into the GATE
branch, leaving the VALUE (up) branch free of it.

So the content-capacity question: in the content read matrix W, how much energy /
how many principal components does the shared b_hat_L direction occupy, gelu (one
entangled matrix) vs swiglu-up (freed) vs swiglu-gate (where the coupling went)?

Per arm, per layer, on the relevant read matrix W [d_ff, d]:
  tax_med         median row |w.b_hat| / ||w||   (the per-row coupling fraction)
  shared_over_iso ||W b_hat||^2 / ||W||_F^2 * d   (read energy on b_hat vs the 1/d
                  isotropic baseline; >> 1 = a real shared overhead, ~1 = free)
  bL_pc_index     which principal component of W is most aligned with b_hat (1 =
                  top PC), its |cos|, and its singular-energy share
  eff_rank/d      participation ratio of singular values, W and W with b_hat
                  projected out (does removing the shared axis free a dimension)

Content read matrix per arm: gelu dense_h_to_4h, relu2 w_in, swiglu w_up (+ w_gate
as the coupling branch), bilinear w_a (+ w_b, symmetric null).
Usage: ./.venv/Scripts/python.exe minout/code/ladder_content_capacity.py
       STEPS=150000
"""
import os
import sys
import json

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_probe import load_snapshot, eval_batch, empty_cache        # noqa: E402
from ladder_gate_reference import bL_per_layer                          # noqa: E402

STEPS = os.environ.get("STEPS", "150000").split(",")
OUT = os.environ.get("OUT", "minout/data/ladder-content-capacity.json")
# arm -> {label: read-weight attr}. gelu/relu2 = one entangled matrix; the gated
# arms expose the content branch AND the coupling branch separately.
READS = {
    "gelu": {"read": "dense_h_to_4h"},
    "relu2": {"read": "w_in"},
    "swiglu": {"value": "w_up", "gate": "w_gate"},
    "bilinear": {"a": "w_a", "b": "w_b"},
    "doublegate": {"g1": "w_a", "g2": "w_b"},
}


def eff_rank(W):
    s = torch.linalg.svdvals(W.float())
    s2 = (s * s)
    return float((s.sum() ** 2) / s2.sum().clamp_min(1e-12)), s


def mat_stats(W, bhat, d):
    W = W.float()
    row_n = W.norm(dim=1).clamp_min(1e-9)
    c = W @ bhat                                   # [d_ff]
    tax = (c.abs() / row_n).median()
    shared = (c @ c) / (W * W).sum().clamp_min(1e-12)
    er, s = eff_rank(W)
    # which right-singular direction aligns with b_hat
    _, _, Vh = torch.linalg.svd(W, full_matrices=False)
    al = (Vh @ bhat).abs()                         # [min(d_ff,d)]
    i_star = int(al.argmax())
    s2 = (s * s)
    er_perp, _ = eff_rank(W - torch.outer(c, bhat))
    return {"tax_med": round(float(tax), 4),
            "shared_over_iso": round(float(shared * d), 2),
            "bL_pc_index": i_star + 1,
            "bL_pc_cos": round(float(al[i_star]), 3),
            "bL_pc_energy": round(float(s2[i_star] / s2.sum()), 4),
            "eff_rank_over_d": round(er / d, 4),
            "eff_rank_perp_over_d": round(er_perp / d, 4)}


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    ids = eval_batch()
    for step in STEPS:
        res.setdefault(step, {})
        for arm, reads in READS.items():
            p = "minout/data/ladder/{}_step{}.pt".format(arm, step)
            if not os.path.exists(p):
                continue
            model, a, st = load_snapshot(p)
            d = model.config.hidden_size
            bL = bL_per_layer(model, ids)
            layers = model.gpt_neox.layers
            out = {label: {k: [] for k in
                           ("tax_med", "shared_over_iso", "bL_pc_index",
                            "bL_pc_cos", "bL_pc_energy", "eff_rank_over_d",
                            "eff_rank_perp_over_d")}
                   for label in reads}
            for L, lay in enumerate(layers):
                bhat = (bL[L] / bL[L].norm().clamp_min(1e-9))
                for label, attr in reads.items():
                    W = getattr(lay.mlp, attr).weight.detach()
                    st_ = mat_stats(W, bhat, d)
                    for k, v in st_.items():
                        out[label][k].append(v)
            res[step][arm] = {label: {k: (round(float(np.median(v)), 4))
                                      for k, v in s.items()}
                              for label, s in out.items()}
            del model
            empty_cache()

        print(f"\n=== step {step} (medians over layers) ===")
        print(f"  {'arm/branch':>16} {'tax':>6} {'shared/iso':>11} "
              f"{'b_L PC#':>8} {'PC cos':>7} {'PC egy':>7} {'effrank/d':>10} "
              f"{'perp/d':>7}")
        for arm in READS:
            if arm not in res[step]:
                continue
            for label, s in res[step][arm].items():
                tag = f"{arm}:{label}"
                print(f"  {tag:>16} {s['tax_med']:>6.3f} "
                      f"{s['shared_over_iso']:>11.1f} {int(s['bL_pc_index']):>8d} "
                      f"{s['bL_pc_cos']:>7.3f} {s['bL_pc_energy']:>7.4f} "
                      f"{s['eff_rank_over_d']:>10.4f} "
                      f"{s['eff_rank_perp_over_d']:>7.4f}")
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nREAD: gelu = one matrix with the shared b_L overhead inside it "
          "(shared/iso >> 1,\nb_L a top PC). swiglu:value should be FREE "
          "(shared/iso ~1, b_L NOT a top PC),\nswiglu:gate carries it. bilinear "
          "a/b ~symmetric. The freed content dimension is\nthe capacity gating "
          "buys.")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
