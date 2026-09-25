"""b_L, STRUCTURE, AND NOISE: how the nonlinearity partitions the residual stream's energy.

Second instrument of the 2026-08-26 pivot. Decomposes the residual entering each layer into three
parts whose shares sum to 1 by construction (this one IS a partition, unlike the variance ratios in
`massive_partition.py` -- see PLAN.md 2026-08-26c D2):

    x  =  mu           the DC / reference direction, this project's b_L
       +  P_k(x - mu)  the STRUCTURED remainder: top-k principal directions of the centred residual
       +  rest         NOISE: everything below the k-th principal direction

    dc_share        ||mu||^2 / E||x||^2
    struct_share    (1 - dc_share) * (variance in top-k PCs / total centred variance)
    noise_share     (1 - dc_share) * (1 - that fraction)

WHY b_L. The background paper's central object: the mean residual direction that sets a uniform
resting operating point for every gate. SETTLED.md has it at 80-87% of the norm, and 15o finds the
resting pre-activation is 93.6-97.2% w.b_L against 0.0-0.2% learned bias -- re-measured by the
concurrent session here at 80-93% vs 0.8-3.1%. What has never been asked is whether the CHOICE OF
NONLINEARITY changes b_L, which is exactly what four matched arms can answer.

[!!] THE CROSS-ARM CONTROL IS FREE AND UNUSUAL. All four arms share INIT_SEED for the embedding, so
the residual BASIS is identical across arms -- `run_glu_matched.py` seeds before `from_config`, and
only the MLP differs. So cos(mu_armA(L), mu_armB(L)) is meaningful, which it would not be for two
independently initialised models. High cross-arm cosine => b_L is set by data and embedding; low =>
by the nonlinearity. Oskin 2608.10251 Sec 8.2 found two seeds of the SAME architecture hold their
concept frames ~90 degrees apart (89.7 vs 91.4 random), so a shared basis is the only way this
question is askable at all.

[!!] PARTICIPATION RATIO IS ON THE CENTRED COVARIANCE. Uncentred PR collapses toward 1 because the
DC term dominates -- this project measures ~1.3 when it forgets. Centring is what makes PR report
the workspace rather than b_L.

[!!] Per-layer profiles only. Never a single mean.

Usage:
  ./.venv/Scripts/python.exe minout/code/bl_and_noise.py
  STEPS=...  KPC=10
"""
import os
import sys
import json
import itertools

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder_probe import load_snapshot, eval_batch, ARMS, empty_cache   # noqa: E402

STEPS = os.environ.get("STEPS", "0,3000,12000,32000,75000,150000").split(",")
KPC = int(os.environ.get("KPC", "10"))
OUT = os.environ.get("OUT", "minout/data/bl-noise.json")


@torch.no_grad()
def profile(model, ids):
    layers = model.gpt_neox.layers
    H = {}

    def cap(L):
        def fn(mod, inp):
            H.setdefault(L, []).append(inp[0].detach().float().reshape(-1, inp[0].shape[-1]))
        return fn

    hks = [lay.register_forward_pre_hook(cap(L)) for L, lay in enumerate(layers)]
    for i in range(0, ids.shape[0], 2):
        model(ids[i:i + 2])
    for h in hks:
        h.remove()

    out = {k: [] for k in ("dc_share", "struct_share", "noise_share", "pr_centered",
                           "bl_norm", "bl_cos_next")}
    mus = []
    for L in range(len(layers)):
        x = torch.cat(H[L]).double()
        mu = x.mean(0)
        tot = float(x.pow(2).sum(1).mean())
        dc = float(mu.pow(2).sum()) / max(tot, 1e-12)
        c = x - mu
        C = (c.T @ c) / c.shape[0]
        ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
        tr, tr2 = float(ev.sum()), float((ev ** 2).sum())
        frac_k = float(ev[:KPC].sum()) / max(tr, 1e-12)
        out["dc_share"].append(round(dc, 5))
        out["struct_share"].append(round((1 - dc) * frac_k, 5))
        out["noise_share"].append(round((1 - dc) * (1 - frac_k), 5))
        out["pr_centered"].append(round(tr * tr / max(tr2, 1e-12), 2))
        out["bl_norm"].append(round(float(mu.norm()), 3))
        mus.append((mu / mu.norm().clamp_min(1e-12)).cpu().numpy())
    for L in range(len(layers) - 1):
        out["bl_cos_next"].append(round(float(np.dot(mus[L], mus[L + 1])), 4))
    out["bl_cos_next"].append(None)
    return out, mus


def main():
    res = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    ids = eval_batch()
    for step in STEPS:
        mus_by_arm = {}
        for arm in ARMS:
            p = "minout/data/ladder/{}_step{}.pt".format(arm, step)
            if not os.path.exists(p):
                continue
            if step in res.get(arm, {}):
                continue
            model, a, st = load_snapshot(p)
            pr, mus = profile(model, ids)
            mus_by_arm[arm] = mus
            res.setdefault(arm, {})[str(st)] = pr
            print("[{:>8s}] step {:>6s}  b_L share {:5.1%}  struct {:5.1%}  noise {:5.1%}  "
                  "PR(centred) {:6.1f}  b_L cos(L,L+1) {:+.3f}".format(
                      arm, step, float(np.mean(pr["dc_share"])),
                      float(np.mean(pr["struct_share"])), float(np.mean(pr["noise_share"])),
                      float(np.mean(pr["pr_centered"])),
                      float(np.mean([v for v in pr["bl_cos_next"] if v is not None]))), flush=True)
            del model
            empty_cache()
        if len(mus_by_arm) > 1:
            xa = res.setdefault("_cross_arm", {}).setdefault(str(step), {})
            for a, b in itertools.combinations(sorted(mus_by_arm), 2):
                cs = [float(np.dot(mus_by_arm[a][L], mus_by_arm[b][L]))
                      for L in range(len(mus_by_arm[a]))]
                xa["{}-{}".format(a, b)] = [round(v, 4) for v in cs]
            print("   cross-arm cos(b_L): " + "  ".join(
                "{} {:+.3f}".format(k, float(np.mean(v))) for k, v in xa.items()), flush=True)
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
