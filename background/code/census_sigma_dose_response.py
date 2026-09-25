# -*- coding: utf-8 -*-
"""Measure sigma directly, and is its inflation a smooth function of input
degeneracy? (Olli, 2026-08-19; the quantity 15bp derived rather than measured.)

15bp established BY ARITHMETIC that LN's denominator must inflate ~17.5x under
repeated-token input: the module-bias contribution to the reference is a CONSTANT
numerator over sigma(p), and it collapsed +1.745 -> +0.100. Nothing else in that
expression can move. But sigma itself was never measured, and the duty measure
refused to give a dose-response (mild corruptions sat slightly BELOW prose duty
while total degeneracy exploded), so whether the failure is smooth or threshold-like
is open.

THE DOSE. Two knobs, both interpolating to natural text, so "degeneracy" is a
number rather than a category:
  PERIOD k -- take the document's first k tokens and TILE them to full length.
    k=1 is the 15bo/15bp `repeat` condition; k=256 IS the prose document. Local
    statistics stay real text at every k; only the global variety changes.
  TYPES V -- sample i.i.d. from the document's V most frequent token types
    (renormalised), so the frequency profile stays Zipf-ish while variety is set
    directly. V=1 is degenerate, large V approaches a bag-of-words draw.

MEASURED per condition per layer:
  sigma        the actual LN denominator, mean over positions, plus E[1/sigma]
               (the quantity the mechanism predicts scales the reference)
  sigma_nomass the same computed over the NON-massive coordinates only, to ask
               whether sigma inflation is a massive-dimension phenomenon (prime
               suspect: they carry 52.4% of ||b_L||^2 and co-activate at +0.354)
  total        the reference component <x_ln, b_hat_prose>
  bias_pred    C_total * E[1/sigma], where C_total is computed FROM WEIGHTS ALONE
               (the constant numerator of the module-bias contributions). Its
               agreement with the measured bias part is an identity check, and its
               size against `total` is the mechanism.
  duty, entropy, n_types

THE QUANTITATIVE TEST. If the mechanism is right, the reference component across
doses should be predicted by 1/sigma. Reported as the R^2 of total ~ a + b/sigma
across the dose ladder. A high R^2 means the reference is being divided out, not
rotated away.

Usage: .venv/Scripts/python.exe superposition/code/census_sigma_dose_response.py
"""
import os
import sys
import json
import time
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
MAXLEN, SKIP = 256, 16
N_BL, N_EVAL = 20, 16
LAYERS = [2, 8, 16, 24, 31]
N_MASSIVE = 12
PERIODS = [1, 2, 4, 8, 16, 32, 64, 128]
TYPES = [1, 2, 4, 16, 64, 256]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-sigma-dose-response.json")
RNG = np.random.default_rng(20260819)


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz, NL, d = m.model.layers, len(m.model.layers), m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    bl_docs, ev_docs = docs[:N_BL], docs[N_BL:N_BL + N_EVAL]

    cap, hooks = {}, []
    for L in LAYERS:
        hooks.append(Lz[L].register_forward_pre_hook(
            (lambda L: (lambda mod_, a, kw: cap.__setitem__(
                f"h{L}", (a[0] if a else kw["hidden_states"])[0].detach())))(L),
            with_kwargs=True))
        hooks.append(Lz[L].mlp.fc1.register_forward_hook(
            (lambda L: (lambda mod_, i, o: cap.__setitem__(
                f"z{L}", o[0].detach())))(L)))

    def run(ids):
        with torch.no_grad():
            return m(input_ids=torch.tensor([ids], device=DEV)).logits[0].float()

    # ---- b_L^prose, massive dims, and the bias constant --------------------
    acc = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    absm = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    n = 0
    for text in bl_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run(ids)
        for L in LAYERS:
            with torch.no_grad():
                acc[L] += Lz[L].input_layernorm(
                    cap[f"h{L}"])[SKIP:].double().sum(0)
            absm[L] += cap[f"h{L}"][SKIP:].float().abs().double().sum(0)
        n += len(ids) - SKIP
    bhat, massive, Cbias, keep = {}, {}, {}, {}
    abias = {k: Lz[k].self_attn.dense.bias.detach().float() for k in range(NL)}
    mbias = {k: Lz[k].mlp.fc2.bias.detach().float() for k in range(NL)}
    for L in LAYERS:
        v = (acc[L] / n).float()
        bhat[L] = v / v.norm()
        massive[L] = torch.topk((absm[L] / n).float(), N_MASSIVE).indices
        kp = torch.ones(d, dtype=torch.bool, device=DEV)
        kp[massive[L]] = False
        keep[L] = kp
        ln = Lz[L].input_layernorm
        u = ln.weight.detach().float() * bhat[L]
        su = float(u.sum())
        C = 0.0
        for k in range(L):
            for b in (abias[k], mbias[k]):
                C += float(b @ u) - float(b.mean()) * su
        Cbias[L] = C
    print(f"  b_L^prose from {n} positions")
    print("  bias-constant C (from weights alone): " +
          ", ".join(f"L{L} {Cbias[L]:+.3f}" for L in LAYERS))

    # ---- the dose ladder ---------------------------------------------------
    def variants(text):
        base = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(base) < 200:
            return None
        T = len(base)
        out = {}
        for k in PERIODS:
            blk = base[:k]
            out[f"period{k}"] = [blk[i % k] for i in range(T)]
        out["prose"] = base
        cn = Counter(base)
        types = [t for t, _ in cn.most_common()]
        for V in TYPES:
            sub = types[:V]
            w = np.array([cn[t] for t in sub], dtype=float)
            w /= w.sum()
            out[f"types{V}"] = [int(sub[i]) for i in
                                RNG.choice(len(sub), size=T, p=w)]
        return out

    CONDS = [f"period{k}" for k in PERIODS] + ["prose"] + \
            [f"types{V}" for V in TYPES]
    agg = {c: {L: {"sig": [], "siginv": [], "signm": [], "tot": [],
                   "cos": [], "duty": [], "hn": []} for L in LAYERS}
           for c in CONDS}
    extra = {c: {"ent": [], "ntypes": []} for c in CONDS}
    for text in ev_docs:
        V = variants(text)
        if V is None:
            continue
        for c in CONDS:
            lg = run(V[c])
            lp = torch.log_softmax(lg[SKIP:], -1)
            extra[c]["ent"].append(float((-(lp.exp() * lp).sum(-1)).mean()))
            extra[c]["ntypes"].append(len(set(V[c])))
            for L in LAYERS:
                h = cap[f"h{L}"].float()[SKIP:]
                ln = Lz[L].input_layernorm
                sg = h.var(-1, unbiased=False).add(ln.eps).sqrt()
                sgn = h[:, keep[L]].var(-1, unbiased=False).add(ln.eps).sqrt()
                with torch.no_grad():
                    x = ln(cap[f"h{L}"]).float()[SKIP:]
                a = agg[c][L]
                a["sig"].append(float(sg.mean()))
                a["siginv"].append(float((1.0 / sg).mean()))
                a["signm"].append(float(sgn.mean()))
                a["hn"].append(float(h.norm(dim=-1).mean()))
                a["tot"].append(float((x @ bhat[L]).mean()))
                a["cos"].append(float(
                    ((x / x.norm(dim=-1, keepdim=True)) @ bhat[L]).mean()))
                z = cap[f"z{L}"][SKIP:].float()
                a["duty"].append(float((z > 0).float().mean()))

    def M(c, L, k):
        return float(np.mean(agg[c][L][k]))

    print("\n=== PERIOD DOSE (k=1 is fully degenerate; prose is the document) ===")
    res = {}
    for L in LAYERS:
        print(f"\n--- L{L} ---")
        print(f"  {'cond':>9} {'ntypes':>7} {'sigma':>8} {'sig/prose':>10} "
              f"{'E[1/sig]':>9} {'sig_nomass':>11} {'||h||':>8} "
              f"{'reference':>10} {'cos':>7} {'duty':>7} {'entropy':>8}")
        sp = M("prose", L, "sig")
        res[f"L{L}"] = {}
        for c in [f"period{k}" for k in PERIODS] + ["prose"] + \
                 [f"types{V}" for V in TYPES]:
            e = dict(ntypes=round(float(np.mean(extra[c]["ntypes"])), 1),
                     sigma=round(M(c, L, "sig"), 4),
                     sigma_over_prose=round(M(c, L, "sig") / sp, 3),
                     e_inv_sigma=round(M(c, L, "siginv"), 4),
                     sigma_nomass=round(M(c, L, "signm"), 4),
                     h_norm=round(M(c, L, "hn"), 2),
                     reference=round(M(c, L, "tot"), 4),
                     cos=round(M(c, L, "cos"), 4),
                     duty=round(M(c, L, "duty"), 4),
                     entropy=round(float(np.mean(extra[c]["ent"])), 3))
            res[f"L{L}"][c] = e
            print(f"  {c:>9} {e['ntypes']:>7.1f} {e['sigma']:>8.3f} "
                  f"{e['sigma_over_prose']:>10.2f} {e['e_inv_sigma']:>9.4f} "
                  f"{e['sigma_nomass']:>11.3f} {e['h_norm']:>8.1f} "
                  f"{e['reference']:>10.3f} {e['cos']:>7.4f} "
                  f"{e['duty']:>7.4f} {e['entropy']:>8.3f}")

        # the quantitative test: is the reference predicted by 1/sigma?
        ks = [f"period{k}" for k in PERIODS] + ["prose"]
        x = np.array([M(c, L, "siginv") for c in ks])
        y = np.array([M(c, L, "tot") for c in ks])
        A = np.vstack([np.ones_like(x), x]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - float(np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        # and the pure mechanism prediction, with NO fitting
        mech = Cbias[L] * x
        print(f"  reference ~ a + b*E[1/sigma] over the period ladder: "
              f"R^2 = {r2:.4f}  (a {coef[0]:+.3f}, b {coef[1]:+.3f})")
        print(f"  UNFITTED bias-mechanism prediction C*E[1/sigma]: "
              f"{mech[0]:+.3f} (k=1) -> {mech[-1]:+.3f} (prose), "
              f"C = {Cbias[L]:+.3f}")
        print(f"  sigma ratio repeat/prose = "
              f"{M('period1', L, 'sig')/sp:.2f}x, "
              f"E[1/sigma] ratio prose/repeat = "
              f"{M('prose', L, 'siginv')/M('period1', L, 'siginv'):.2f}x")
        res[f"L{L}"]["_fit"] = dict(r2=round(r2, 4), a=round(float(coef[0]), 4),
                                    b=round(float(coef[1]), 4),
                                    C_bias=round(Cbias[L], 4),
                                    sigma_ratio_deg_over_prose=round(
                                        M("period1", L, "sig") / sp, 3),
                                    einv_ratio_prose_over_deg=round(
                                        M("prose", L, "siginv")
                                        / M("period1", L, "siginv"), 3))

    for h_ in hooks:
        h_.remove()
    json.dump({"model": MODEL, "layers": LAYERS, "periods": PERIODS,
               "types": TYPES, "skip": SKIP, "n_massive": N_MASSIVE,
               "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD: a high R^2 of reference ~ 1/sigma across the ladder means the")
    print("reference is DIVIDED OUT rather than rotated away. Compare sigma to")
    print("sigma_nomass to see whether the inflation is the massive dims.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
