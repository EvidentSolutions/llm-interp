# -*- coding: utf-8 -*-
"""WHERE does the variance of the reference component come from, and what
removes it? (Olli, 2026-08-19; the measurement 15bm left open.)

THE QUESTION. Write the post-LN gate input as x_ln(p) = b_L + delta(p). Then
    <x_ln(p), b_hat_L> = ||b_L|| + <delta(p), b_hat_L>
and since b_L is DEFINED as the corpus mean, <delta, b_hat_L> has mean zero BY
CONSTRUCTION. Nothing forces its VARIANCE to be small -- yet the observed spread
is 0.326 +- 0.044. So: what keeps it small, per position?

Three mechanisms 15bm could not separate:
  (i)  each write is individually ~orthogonal to b_hat_L;
  (ii) writes have b_hat_L components that CANCEL across sources (negative
       covariance -- this is also the last surviving rescue for the conservation
       hypothesis, in its cancellation-aware signed form, since 15bm used squared
       norms which cannot see cancellation);
  (iii) writes have large b_hat_L components and LN's per-position denominator
       scales them away.

THE DECOMPOSITION IS AN IDENTITY, NOT AN APPROXIMATION. LayerNorm is
    x_ln = gamma * (h - mu(p)*1) / sigma(p) + beta
with mu, sigma the per-position COORDINATE mean and std. The map h -> h - mu*1 is
linear and 1/sigma(p) is a scalar, so with u = gamma * b_hat_L,

    <x_ln(p), b_hat_L> = (1/sigma(p)) * SUM_s [ <s(p), u> - mean_coord(s(p))*sum(u) ]
                         + <beta, b_hat_L>

over sources s = {embedding output h_0, attn_k, mlp_k for k < L}, exactly. The
script checks that identity numerically before reading anything off it.

  LEG A  SOURCE ATTRIBUTION. Per source: mean and sd of its contribution. Which
         sources carry the variance of the reference component?
  LEG B  CANCELLATION. R = Var(sum)/sum(Var) on the SIGNED contributions, plus
         the mean pairwise correlation. R < 1 with negative covariance = sources
         cancel (mechanism ii, and the conservation rescue fires). R ~ 1 =
         independent. R > 1 = they reinforce.
  LEG C  WHAT LN'S DENOMINATOR IS WORTH. Recompute every contribution with
         sigma(p) replaced by its corpus mean -- i.e. LN centres and scales by a
         CONSTANT instead of per-position. The ratio of Var(sum) between the two
         is exactly what the per-position gain control buys (mechanism iii).
  LEG D  THE CHAIN, in comparable units (coefficient of variation):
         <h, m_hat> raw  ->  <h/||h||, m_hat>  ->  <x_ln, b_hat_L>/||x_ln||.
         Isolates "divide by the norm" from "full LN".

b_L, m_L and mean sigma are estimated on documents DISJOINT from the evaluation
set. Positions >= SKIP only, since 15bm showed the first ~16 positions are a
settling transient.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_variance_sources.py
"""
import os
import sys
import json
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
MAXLEN, SKIP = 256, 16
N_BL, N_EVAL = 20, 30
LAYERS = [8, 16, 24]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-reference-variance-sources.json")


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz = m.model.layers
    NL, d = len(Lz), m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    bl_docs, ev_docs = docs[:N_BL], docs[N_BL:N_BL + N_EVAL]

    cap = {}
    hooks = []
    for L in range(NL):
        hooks.append(Lz[L].register_forward_pre_hook(
            (lambda L: (lambda mod_, a, kw: cap.__setitem__(
                f"h{L}", (a[0] if a else kw["hidden_states"])[0].detach())))(L),
            with_kwargs=True))
        hooks.append(Lz[L].self_attn.register_forward_hook(
            (lambda L: (lambda mod_, i, o: cap.__setitem__(
                f"a{L}", (o[0] if isinstance(o, tuple) else o)[0].detach())))(L)))
        hooks.append(Lz[L].mlp.register_forward_hook(
            (lambda L: (lambda mod_, i, o: cap.__setitem__(
                f"m{L}", (o[0] if isinstance(o, tuple) else o)[0].detach())))(L)))

    def fwd(text):
        ids = tok(text, truncation=True, max_length=MAXLEN,
                  return_tensors="pt")["input_ids"].to(DEV)
        if ids.shape[1] < SKIP + 8:
            return None
        with torch.no_grad():
            m(input_ids=ids)
        return ids.shape[1]

    # ---- references on the DISJOINT set ------------------------------------
    accM = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    accB = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    accS = {L: 0.0 for L in LAYERS}
    n = 0
    for text in bl_docs:
        T = fwd(text)
        if T is None:
            continue
        for L in LAYERS:
            h = cap[f"h{L}"][SKIP:].float()
            accM[L] += h.double().sum(0)
            with torch.no_grad():
                x = Lz[L].input_layernorm(cap[f"h{L}"])[SKIP:].float()
            accB[L] += x.double().sum(0)
            accS[L] += float(h.var(-1, unbiased=False).add(
                Lz[L].input_layernorm.eps).sqrt().sum())
        n += T - SKIP
    mhat, bhat, bnorm, sig_mean = {}, {}, {}, {}
    for L in LAYERS:
        mv = (accM[L] / n).float(); mhat[L] = mv / mv.norm()
        bv = (accB[L] / n).float(); bnorm[L] = float(bv.norm())
        bhat[L] = bv / bv.norm()
        sig_mean[L] = accS[L] / n
    print(f"  references from {n} positions of {len(bl_docs)} docs")
    for L in LAYERS:
        print(f"    L{L}: ||b_L|| {bnorm[L]:.3f}, mean sigma {sig_mean[L]:.3f}, "
              f"cos(m_hat, b_hat) {float(mhat[L] @ bhat[L]):+.4f}")

    # ---- evaluation pass ---------------------------------------------------
    res = {}
    for L in LAYERS:
        ln = Lz[L].input_layernorm
        gam, bet, eps = (ln.weight.detach().float(), ln.bias.detach().float(),
                         ln.eps)
        u = gam * bhat[L]                       # effective reference
        su = float(u.sum())
        beta_term = float(bet @ bhat[L])
        names = ["embed"] + [f"a{k}" for k in range(L)] + \
                [f"m{k}" for k in range(L)]
        C, Cfix, tot, chain = [], [], [], {"raw": [], "unitnorm": [], "ln": []}
        gate = []
        for text in ev_docs:
            T = fwd(text)
            if T is None:
                continue
            h = cap[f"h{L}"].float()[SKIP:]
            srcs = [cap["h0"].float()[SKIP:]] + \
                   [cap[f"a{k}"].float()[SKIP:] for k in range(L)] + \
                   [cap[f"m{k}"].float()[SKIP:] for k in range(L)]
            sigma = h.var(-1, unbiased=False).add(eps).sqrt()      # (P,)
            # exact per-source contribution to <x_ln, b_hat>
            raw = torch.stack([(s @ u) - s.mean(-1) * su for s in srcs], 1)
            C.append((raw / sigma[:, None]).cpu().numpy())
            Cfix.append((raw / sig_mean[L]).cpu().numpy())
            with torch.no_grad():
                x = ln(cap[f"h{L}"]).float()[SKIP:]
            tot.append((x @ bhat[L]).cpu().numpy())
            gate.append(float(((raw / sigma[:, None]).sum(1) + beta_term
                               - (x @ bhat[L])).abs().max()
                              / (x @ bhat[L]).abs().mean()))
            chain["raw"].extend((h @ mhat[L]).cpu().numpy().tolist())
            chain["unitnorm"].extend(
                ((h / h.norm(dim=-1, keepdim=True)) @ mhat[L])
                .cpu().numpy().tolist())
            chain["ln"].extend(
                ((x / x.norm(dim=-1, keepdim=True)) @ bhat[L])
                .cpu().numpy().tolist())
        C = np.concatenate(C, 0); Cfix = np.concatenate(Cfix, 0)
        tot = np.concatenate(tot, 0)
        g = float(np.max(gate))
        print(f"\n=== L{L} ===  {C.shape[0]} positions, {C.shape[1]} sources")
        print(f"  IDENTITY GATE (sum of contributions + beta vs actual): "
              f"max rel err {g:.2e}  {'PASS' if g < 5e-2 else '**FAIL**'}")

        def stats(X):
            s = X.sum(1)
            vs, sv = float(s.var()), float(X.var(0).sum())
            R = vs / sv if sv > 1e-30 else float("nan")
            Cm = np.corrcoef(X, rowvar=False)
            iu = np.triu_indices(X.shape[1], 1)
            return dict(var_sum=vs, sum_var=sv, R=R,
                        rho_bar=float(np.nanmean(Cm[iu])),
                        cov_frac=(vs - sv) / vs if abs(vs) > 1e-30 else float("nan"))

        A, B = stats(C), stats(Cfix)
        print(f"  LEG B  signed contributions: Var(sum) {A['var_sum']:.4f}, "
              f"sum(Var) {A['sum_var']:.4f}, R {A['R']:.3f}, "
              f"rho_bar {A['rho_bar']:+.4f}")
        print(f"         -> covariance carries {100*A['cov_frac']:+.1f}% of "
              f"Var(sum)  "
              f"({'CANCELLATION' if A['R'] < 0.8 else 'independent' if A['R'] < 1.25 else 'REINFORCEMENT'})")
        print(f"  LEG C  with sigma FIXED at its corpus mean: Var(sum) "
              f"{B['var_sum']:.4f} -> per-position sigma gives {A['var_sum']:.4f}"
              f"  = LN's gain control removes "
              f"{100*(1-A['var_sum']/B['var_sum']):.1f}%")
        print(f"  LEG A  top sources by sd of contribution:")
        sd = C.std(0); mu = C.mean(0)
        order = np.argsort(-sd)[:8]
        for i in order:
            print(f"         {names[i]:>7}  mean {mu[i]:+8.3f}  sd {sd[i]:7.3f}")
        print(f"         (total reference component: mean {tot.mean():+.3f}, "
              f"sd {tot.std():.3f}; ||b_L|| {bnorm[L]:.3f})")
        cvs = {k: float(np.std(v) / abs(np.mean(v)))
               for k, v in chain.items()}
        print(f"  LEG D  CV along the chain: raw <h,m> {cvs['raw']:.4f} -> "
              f"unit-norm {cvs['unitnorm']:.4f} -> full LN {cvs['ln']:.4f}")
        res[f"L{L}"] = dict(
            n_pos=int(C.shape[0]), n_src=int(C.shape[1]), identity_gate=g,
            signed=dict(R=round(A["R"], 4), rho_bar=round(A["rho_bar"], 4),
                        var_sum=round(A["var_sum"], 4),
                        sum_var=round(A["sum_var"], 4),
                        cov_frac=round(A["cov_frac"], 4)),
            sigma_fixed=dict(R=round(B["R"], 4), var_sum=round(B["var_sum"], 4)),
            ln_gain_removes=round(1 - A["var_sum"] / B["var_sum"], 4),
            total=dict(mean=round(float(tot.mean()), 4),
                       sd=round(float(tot.std()), 4), bnorm=round(bnorm[L], 4)),
            chain_cv={k: round(v, 4) for k, v in cvs.items()},
            top_sources=[{"src": names[int(i)], "mean": round(float(mu[i]), 4),
                          "sd": round(float(sd[i]), 4)} for i in order])

    for h in hooks:
        h.remove()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL, "skip": SKIP, "layers": LAYERS,
                   "n_bl_docs": N_BL, "n_eval_docs": N_EVAL,
                   "results": res}, f, indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: R < 1 with negative rho_bar = sources CANCEL (mechanism ii,")
    print("and the signed conservation rescue fires). R ~ 1 = independent.")
    print("A large 'LN gain control removes' = mechanism iii, the denominator.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
