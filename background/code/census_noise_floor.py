"""What is the residual stream's NOISE FLOOR, in the same currency as our energy
readings? (Olli, 2026-08-13: "how much noise does a transformer tolerate before
performance degrades? add a uniform noise source to each layer just before the
layer norm ... to give our energy/variance readings a reference point.")

Injection site is the layer input, i.e. exactly the paper's alpha-scaling site
(`register_forward_pre_hook` on `model.layers[L]`), so results are directly
comparable to the b_L removal numbers. Positions < SKIP are left untouched,
following the paper's convention.

PARAMETERISATION. Noise is drawn per-coordinate uniform and rescaled so that
||n_t|| = eps * ||x_t|| at every position. Because n is ~orthogonal to x in
d=2560, the injected ENERGY SHARE is eps^2 / (1 + eps^2):

    eps    0.01   0.05   0.10   0.20   0.33   0.50   0.70   1.00
    share  0.01%  0.25%  1.0%   3.8%   9.8%   20%    33%    50%

**eps = 0.33 injects the same energy share that b_L occupies (10-12%,
`census_reference_energy_budget.py`).** That is the reference point: if isotropic
noise at b_L's energy is harmless while removing b_L is not, b_L's 10% is
structured rather than generic.

ARMS
  iso_all     uniform noise at EVERY layer (the question as asked; damage
              compounds down the stack)
  iso_one     uniform noise at ONE layer (isolates depth sensitivity, and is the
              apples-to-apples comparison with single-site b_L removal)
  dim_all     noise matched to each layer's PER-DIMENSION std instead of
              isotropic. Same energy, shaped like the data. Phi-2's residual is
              wildly anisotropic (massive dims), so this asks whether the
              tolerance is about energy or about WHERE the energy goes.
  bL_one      remove the projection on b_hat_L at one layer -- the paper's
              alpha=0 manipulation, for reference on the same metrics.
  rand1_one   remove the projection on ONE random direction at one layer
              (norm-matched control, as in the paper).

METRICS, over positions >= SKIP: KL(clean || perturbed), delta NLL on the true
next token, top-1 agreement. Baselines are recomputed per document.

Usage: .venv/Scripts/python.exe superposition/code/census_noise_floor.py
       NDOCS=n to change the document count
"""
import sys
import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
N_DOCS = int(os.environ.get("NDOCS", "24"))
MAXLEN, SKIP = 256, 16
EPS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.33, 0.50, 0.70, 1.00]
ONE_LAYERS = [6, 10, 14, 22]
REF_EPS = 0.33                      # the b_L-energy-matched point

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-noise-floor.json")


def share(eps):
    return eps * eps / (1.0 + eps * eps)


def perturb_hook(mode, eps, gen, dstd=None, bhat=None):
    """Return a forward_pre_hook that edits hidden_states at positions >= SKIP."""
    def f(mod, args, kwargs):
        h = (args[0] if args else kwargs["hidden_states"])
        x = h[0]
        seg = x[SKIP:]
        if seg.shape[0] == 0:
            return None
        nrm = seg.norm(dim=-1, keepdim=True)
        if mode == "iso":
            n = torch.rand(seg.shape, device=seg.device, dtype=torch.float32,
                           generator=gen) * 2.0 - 1.0
        elif mode == "dim":
            n = (torch.rand(seg.shape, device=seg.device, dtype=torch.float32,
                            generator=gen) * 2.0 - 1.0) * dstd
        elif mode in ("proj", "rand1"):
            u = bhat
            coef = (seg.float() @ u).unsqueeze(-1)
            new = seg.float() - coef * u
            x2 = x.clone()
            x2[SKIP:] = new.to(x.dtype)
            h2 = h.clone()
            h2[0] = x2
            return ((h2,) + args[1:], kwargs) if args else (args, {**kwargs, "hidden_states": h2})
        n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9) * nrm.float() * eps
        x2 = x.clone()
        x2[SKIP:] = (seg.float() + n).to(x.dtype)
        h2 = h.clone()
        h2[0] = x2
        return ((h2,) + args[1:], kwargs) if args else (args, {**kwargs, "hidden_states": h2})
    return f


@torch.no_grad()
def run_cfg(m, ids, layers, mode, eps, gen, dstd, bhats, clean):
    hs = []
    for L in layers:
        hs.append(m.model.layers[L].register_forward_pre_hook(
            perturb_hook(mode, eps, gen,
                         dstd=None if dstd is None else dstd[L],
                         bhat=None if bhats is None else bhats[L]),
            with_kwargs=True))
    lg = m(input_ids=ids).logits[0].float()
    for h in hs:
        h.remove()
    lp_c, lp_n = clean["lp"], F.log_softmax(lg, -1)
    sl = slice(SKIP, lg.shape[0] - 1)
    kl = float((lp_c[sl].exp() * (lp_c[sl] - lp_n[sl])).sum(-1).mean())
    tgt = clean["tgt"]
    nll = float(-lp_n[sl].gather(1, tgt[:, None]).squeeze(1).mean())
    top1 = float((lp_n[sl].argmax(-1) == lp_c[sl].argmax(-1)).float().mean())
    return kl, nll - clean["nll"], top1


@torch.no_grad()
def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    docs = json.load(open(DOCS, encoding="utf-8"))[:N_DOCS]
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    d, n_layers = m.config.hidden_size, len(m.model.layers)
    ALL = list(range(n_layers))

    ids_list = []
    for t in docs:
        ii = tok(t, truncation=True, max_length=MAXLEN,
                 return_tensors="pt")["input_ids"].to(DEV)
        if ii.shape[1] >= SKIP + 48:
            ids_list.append(ii)
    print(f"{len(ids_list)} docs, d={d}, layers={n_layers}")

    # ---- pass 0: per-layer per-dim std and mean direction (for dim/proj arms)
    print("pass 0: per-dim std + b_L ...", flush=True)
    s1 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in ALL}
    s2 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in ALL}
    cnt = 0
    for ids in ids_list:
        cap, hs = {}, []
        for L in ALL:
            def mk(L=L):
                def f(mod, args, kwargs):
                    hh = args[0] if args else kwargs["hidden_states"]
                    cap[L] = hh[0].detach().double()
                return f
            hs.append(m.model.layers[L].register_forward_pre_hook(
                mk(), with_kwargs=True))
        m(input_ids=ids)
        for h in hs:
            h.remove()
        for L in ALL:
            x = cap[L][SKIP:]
            s1[L] += x.sum(0)
            s2[L] += (x * x).sum(0)
        cnt += cap[0].shape[0] - SKIP
        del cap
    dstd, bhats = {}, {}
    for L in ALL:
        mu = s1[L] / cnt
        var = (s2[L] / cnt - mu * mu).clamp(min=0)
        sd = var.sqrt().float()
        dstd[L] = (sd / sd.norm().clamp(min=1e-9) * np.sqrt(d)).to(torch.float32)
        bhats[L] = (mu / mu.norm().clamp(min=1e-12)).float()

    gen = torch.Generator(device=DEV)
    rows = []

    def sweep(name, layers, mode, eps_list, bh=None):
        for eps in eps_list:
            gen.manual_seed(SEED)
            kls, dnl, t1s = [], [], []
            for ids in ids_list:
                lg = m(input_ids=ids).logits[0].float()
                lp = F.log_softmax(lg, -1)
                sl = slice(SKIP, lg.shape[0] - 1)
                tgt = ids[0][SKIP + 1:]
                clean = {"lp": lp, "tgt": tgt,
                         "nll": float(-lp[sl].gather(1, tgt[:, None]).squeeze(1).mean())}
                k, dn, t1 = run_cfg(m, ids, layers, mode, eps, gen, dstd,
                                    bh if bh is not None else bhats, clean)
                kls.append(k)
                dnl.append(dn)
                t1s.append(t1)
            r = {"arm": name, "eps": eps, "energy_share": round(share(eps), 5),
                 "n_layers": len(layers),
                 "KL": round(float(np.median(kls)), 4),
                 "dNLL": round(float(np.median(dnl)), 4),
                 "top1_agree": round(float(np.median(t1s)), 4)}
            rows.append(r)
            print(f"  {name:>12} eps={eps:<5} energy={share(eps)*100:>5.2f}%  "
                  f"KL={r['KL']:>8.4f}  dNLL={r['dNLL']:>+8.4f}  "
                  f"top1={r['top1_agree']:.4f}", flush=True)

    print("\n--- iso, ALL layers ---")
    sweep("iso_all", ALL, "iso", EPS)
    print("\n--- iso, ONE layer (L10) ---")
    sweep("iso_L10", [10], "iso", EPS)
    print("\n--- dim-matched (data-shaped), ALL layers ---")
    sweep("dim_all", ALL, "dim", [0.05, 0.10, 0.20, 0.33, 0.50, 1.00])
    print("\n--- iso, ONE layer, at the b_L-matched energy, by depth ---")
    for L in ONE_LAYERS:
        sweep(f"iso_L{L}", [L], "iso", [REF_EPS])
    print("\n--- REFERENCE: remove b_L / a random direction, one layer ---")
    for L in ONE_LAYERS:
        sweep(f"bL_L{L}", [L], "proj", [0.0])
        rd = {L: F.normalize(torch.randn(d, generator=torch.Generator(
            device=DEV).manual_seed(1234 + L), device=DEV), dim=0)}
        sweep(f"rand1_L{L}", [L], "proj", [0.0], bh=rd)

    json.dump({"config": {"n_docs": len(ids_list), "skip": SKIP, "eps": EPS,
                          "seed": SEED, "d": d}, "rows": rows},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")
    del m
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
