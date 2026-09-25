"""Did the noise actually average to zero? (Olli, 2026-08-13.)

13g/13h drew symmetric noise, so E[n] = 0 by construction -- but nothing ENFORCED
it on the finite sample. With T ~ 200 positions per document, the empirical mean
of the injected noise has norm ~ ||n|| / sqrt(T) ~= 7% of ||n||.

That matters most for the arm carrying 13h's headline. `along_bL` puts +/- eps*||x||
along b_hat_L with a random sign per position, so a residual mean of 7% of
eps*||x|| is a systematic shift of the operating point of about
0.07 * 0.33 / 0.326 ~= 7% -- i.e. a small alpha-scale, which is precisely the
background paper's intervention. Part of the reported 20-46x could therefore be
a systematic alpha-shift rather than the per-position jitter it was billed as.

This script separates them by ENFORCING a zero empirical mean:

  along_bL / along_rand          random +/- sign  (13h, unbalanced)
  along_bL_bal / along_rand_bal  EXACTLY balanced signs (equal counts of +/-,
                                 randomly permuted) -> empirical mean exactly 0
  iso / iso_bal                  isotropic, and isotropic with the per-position
                                 column mean subtracted and a single global
                                 rescale (keeps the mean exactly 0 while
                                 matching total energy)

It also REPORTS the achieved residual mean, ||mean(n)|| / ||b_L||, per arm, so
the size of the confound is on the record rather than assumed.

BRANCHES (pre-registered):
  If along_bL_bal ~= along_bL, the 13h effect is genuine per-position jitter and
    the conclusion stands unchanged.
  If along_bL_bal << along_bL, a material part of 13h's 20-46x was a residual
    alpha-shift and 13h must be corrected.

Usage: .venv/Scripts/python.exe superposition/code/census_noise_zero_mean.py
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
N_DOCS = int(os.environ.get("NDOCS", "23"))
MAXLEN, SKIP = 256, 16
ARMS = ["iso", "iso_bal", "along_rand", "along_rand_bal",
        "along_bL", "along_bL_bal"]

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
DOCS = os.path.join(DATA, "census-docs-phi-2.json")
OUT = os.path.join(DATA, "census-noise-zero-mean.json")

RESID = {}      # arm -> list of ||mean(n)|| / ||b_L||


def balanced_signs(T, dev, gen):
    k = T // 2
    s = torch.cat([torch.ones(k, device=dev),
                   -torch.ones(T - k, device=dev)])
    return s[torch.randperm(T, device=dev, generator=gen)].unsqueeze(-1)


def mk_hook(arm, eps, gen, bh, rh, bnorm, rec):
    def f(mod, args, kwargs):
        h = (args[0] if args else kwargs["hidden_states"])
        x = h[0]
        seg = x[SKIP:].float()
        T = seg.shape[0]
        if T == 0:
            return None
        tgt = seg.norm(dim=-1, keepdim=True) * eps
        if arm.startswith("along"):
            u = bh if "bL" in arm else rh
            sgn = (balanced_signs(T, seg.device, gen) if arm.endswith("_bal")
                   else torch.randint(0, 2, (T, 1), device=seg.device,
                                      generator=gen).float() * 2.0 - 1.0)
            n = sgn * tgt * u
        else:
            n = torch.rand(seg.shape, device=seg.device, dtype=torch.float32,
                           generator=gen) * 2.0 - 1.0
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-9) * tgt
            if arm.endswith("_bal"):
                n = n - n.mean(0, keepdim=True)
                cur = n.norm(dim=-1).mean().clamp(min=1e-9)
                n = n * (tgt.mean() / cur)
        rec.append(float(n.mean(0).norm()) / max(bnorm, 1e-9))
        x2 = x.clone()
        x2[SKIP:] = (seg + n).to(x.dtype)
        h2 = h.clone()
        h2[0] = x2
        return ((h2,) + args[1:], kwargs) if args else (args, {**kwargs, "hidden_states": h2})
    return f


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
    print(f"{len(ids_list)} docs, d={d}, median T="
          f"{int(np.median([i.shape[1]-SKIP for i in ids_list]))}")

    s1 = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in ALL}
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
            s1[L] += cap[L][SKIP:].sum(0)
        cnt += cap[0].shape[0] - SKIP
        del cap
    mu = {L: (s1[L] / cnt) for L in ALL}
    bnorm = {L: float(mu[L].norm()) for L in ALL}
    bh = {L: (mu[L] / mu[L].norm()).float() for L in ALL}
    g0 = torch.Generator(device=DEV).manual_seed(999)
    rh = {L: F.normalize(torch.randn(d, device=DEV, generator=g0), dim=0)
          for L in ALL}

    gen = torch.Generator(device=DEV)
    rows = []

    def run(tag, layers, arm, eps):
        gen.manual_seed(SEED)
        kls, dn, t1, rec = [], [], [], []
        for ids in ids_list:
            lg = m(input_ids=ids).logits[0].float()
            lp = F.log_softmax(lg, -1)
            sl = slice(SKIP, lg.shape[0] - 1)
            tg = ids[0][SKIP + 1:]
            base = float(-lp[sl].gather(1, tg[:, None]).squeeze(1).mean())
            hs = [m.model.layers[L].register_forward_pre_hook(
                mk_hook(arm, eps, gen, bh[L], rh[L], bnorm[L], rec),
                with_kwargs=True) for L in layers]
            lg2 = m(input_ids=ids).logits[0].float()
            for h in hs:
                h.remove()
            lp2 = F.log_softmax(lg2, -1)
            kls.append(float((lp[sl].exp() * (lp[sl] - lp2[sl])).sum(-1).mean()))
            dn.append(float(-lp2[sl].gather(1, tg[:, None]).squeeze(1).mean()) - base)
            t1.append(float((lp2[sl].argmax(-1) == lp[sl].argmax(-1)).float().mean()))
        r = {"site": tag, "arm": arm, "eps": eps,
             "KL": round(float(np.median(kls)), 4),
             "dNLL": round(float(np.median(dn)), 4),
             "top1": round(float(np.median(t1)), 4),
             "resid_mean_over_bL": round(float(np.median(rec)), 5)}
        rows.append(r)
        return r

    for tag, layers, eps_list in (("L10", [10], [0.33, 0.50]),
                                  ("ALL", ALL, [0.10, 0.20])):
        for eps in eps_list:
            print(f"\n--- {tag}, eps={eps} ---")
            print(f"   {'arm':>15} {'KL':>10} {'dNLL':>9} {'top1':>7} "
                  f"{'||mean n||/||b_L||':>19}")
            for arm in ARMS:
                r = run(tag, layers, arm, eps)
                print(f"   {arm:>15} {r['KL']:>10.4f} {r['dNLL']:>+9.4f} "
                      f"{r['top1']:>7.4f} {r['resid_mean_over_bL']:>19.5f}",
                      flush=True)

    json.dump({"config": {"n_docs": len(ids_list), "arms": ARMS, "seed": SEED,
                          "d": d, "skip": SKIP}, "rows": rows},
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
