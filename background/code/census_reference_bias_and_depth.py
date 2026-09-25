# -*- coding: utf-8 -*-
"""What generates the INPUT-INDEPENDENT bulk of the reference, and why is the
calibration most fragile at L24? (Olli, 2026-08-19; the two openings 15bo left.)

15bo found that per-position cos(x_ln, b_hat_prose) survives at 82% of its prose
value under RANDOM TOKEN IDS, so most of the reference is generated almost
regardless of input. It also found the calibration error is worst at L24
(RMS/sd 0.316 prose -> 3.034 repeat) and mildest at L8 (0.189 -> 2.025).

TEST 3 -- WHERE DOES THE INPUT-INDEPENDENT BULK COME FROM? Phi-2 has BIASES, and
they are added regardless of input. Exact split of the reference component:

    <x_ln(p), b_hat> = <beta, b_hat>                                  [LN bias]
                     + (1/sigma(p)) * SUM_s [ <s(p) - b_s, u> - ... ] [computed]
                     + (1/sigma(p)) * SUM_s [ <b_s, u> - ... ]        [module bias]

with u = gamma*b_hat, b_s the module bias (mlp.fc2.bias, self_attn.dense.bias;
the embedding has none, so h0 is fully input-dependent). The LN beta term carries
NO 1/sigma factor, so it is the only fully constant part. The module-bias terms
are CONSTANT VECTORS scaled by a per-position scalar. If those two groups are what
survives random ids, the "input-independent bulk" is literally the accumulated
bias vectors, and the reference is in part a learned constant the stack adds to
itself.

TEST 2 -- WHY L24? Two candidate explanations, separated here:
  (a) DENOMINATOR ARTIFACT. 15bo normalised the calibration error by the resting
      spread, and that spread SHRINKS with depth (sd 0.631 / 0.522 / 0.430 at
      L8/L16/L24), so RMS/sd is inflated at L24 by construction. Fixed by
      reporting the RAW RMS error alongside.
  (b) COMPOSITION. 15bn found the reference at L24 is dominated by the LATE MLPs
      (m23, m22, m21, m20), which are the least constant sources by mean/sd
      (1.30-1.47) whereas L8's reference is dominated by a0/m0 at 3.06-4.03. If
      so, the depth gradient is about WHO contributes, and the per-source shift
      under corruption should localise it.

Conditions: prose / randid / repeat, with weights and b_L^prose FIXED (estimated
on 20 disjoint prose docs), so every change is the input's doing.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_bias_and_depth.py
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
N_BL, N_EVAL = 20, 24
LAYERS = [8, 16, 24]
CONDS = ["prose", "randid", "repeat"]
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
OUT = os.path.join(DATA, "census-reference-bias-and-depth.json")
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
    for L in LAYERS:
        hooks.append(Lz[L].mlp.fc1.register_forward_hook(
            (lambda L: (lambda mod_, i, o: cap.__setitem__(
                f"z{L}", o[0].detach())))(L)))

    def run(ids):
        with torch.no_grad():
            m(input_ids=torch.tensor([ids], device=DEV))

    # module biases
    abias = {k: Lz[k].self_attn.dense.bias.detach().float() for k in range(NL)}
    mbias = {k: Lz[k].mlp.fc2.bias.detach().float() for k in range(NL)}
    print("  bias norms: attn.dense " +
          f"{np.mean([float(abias[k].norm()) for k in range(NL)]):.3f} mean, "
          f"mlp.fc2 {np.mean([float(mbias[k].norm()) for k in range(NL)]):.3f} mean")

    # ---- b_L^prose ---------------------------------------------------------
    acc = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
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
        n += len(ids) - SKIP
    bhat, resting = {}, {}
    for L in LAYERS:
        v = (acc[L] / n).float()
        bhat[L] = v / v.norm()
        W1 = Lz[L].mlp.fc1.weight.detach().float()
        b1 = Lz[L].mlp.fc1.bias.detach().float()
        resting[L] = (W1 @ v + b1).cpu().numpy()
    print(f"  b_L^prose from {n} positions; resting sd " +
          ", ".join(f"L{L} {resting[L].std():.4f}" for L in LAYERS))

    def variants(text):
        base = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(base) < SKIP + 16:
            return None
        common = Counter(base).most_common(1)[0][0]
        return {"prose": base,
                "randid": [int(x) for x in RNG.integers(0, 50000, len(base))],
                "repeat": [int(common)] * len(base)}

    res = {}
    for L in LAYERS:
        ln = Lz[L].input_layernorm
        gam, bet, eps = (ln.weight.detach().float(),
                         ln.bias.detach().float(), ln.eps)
        u = gam * bhat[L]
        su = float(u.sum())
        beta_term = float(bet @ bhat[L])
        names = ["embed"] + [f"a{k}" for k in range(L)] + \
                [f"m{k}" for k in range(L)]
        bvecs = [None] + [abias[k] for k in range(L)] + [mbias[k] for k in range(L)]

        def contrib(vecs, sigma):
            """exact per-source contribution rows -> (P, nsrc)"""
            return torch.stack([(s @ u) - s.mean(-1) * su for s in vecs], 1) \
                / sigma[:, None]

        store = {c: {"comp": [], "bias": [], "tot": [], "z": [], "duty": []}
                 for c in CONDS}
        nseen = {c: 0 for c in CONDS}
        zs = {c: np.zeros(Lz[L].mlp.fc1.out_features) for c in CONDS}
        ds = {c: np.zeros(Lz[L].mlp.fc1.out_features) for c in CONDS}
        gate = []
        for text in ev_docs:
            V = variants(text)
            if V is None:
                continue
            for c in CONDS:
                run(V[c])
                h = cap[f"h{L}"].float()
                srcs = [cap["h0"].float()] + \
                       [cap[f"a{k}"].float() for k in range(L)] + \
                       [cap[f"m{k}"].float() for k in range(L)]
                sigma = h.var(-1, unbiased=False).add(eps).sqrt()
                # computed part = output minus its own bias
                comp_src = [srcs[0]] + [srcs[i] - bvecs[i][None, :]
                                        for i in range(1, len(srcs))]
                bias_src = [torch.zeros_like(srcs[0])] + \
                           [bvecs[i][None, :].expand_as(srcs[i])
                            for i in range(1, len(srcs))]
                Ccomp = contrib(comp_src, sigma)[SKIP:]
                Cbias = contrib(bias_src, sigma)[SKIP:]
                with torch.no_grad():
                    x = ln(cap[f"h{L}"]).float()[SKIP:]
                tot = x @ bhat[L]
                gate.append(float(
                    (Ccomp.sum(1) + Cbias.sum(1) + beta_term - tot).abs().max()
                    / tot.abs().mean()))
                store[c]["comp"].append(Ccomp.cpu().numpy())
                store[c]["bias"].append(Cbias.cpu().numpy())
                store[c]["tot"].append(tot.cpu().numpy())
                z = cap[f"z{L}"][SKIP:].float()
                zs[c] += z.sum(0).cpu().numpy()
                ds[c] += (z > 0).float().sum(0).cpu().numpy()
                nseen[c] += z.shape[0]
        g = float(np.max(gate))
        print(f"\n=== L{L} === identity gate {g:.2e} "
              f"{'PASS' if g < 5e-2 else '**FAIL**'}   "
              f"beta term {beta_term:+.4f}")

        r = resting[L]
        entry = {"identity_gate": g, "beta_term": round(beta_term, 4),
                 "resting_sd": round(float(r.std()), 4), "conds": {}}
        print(f"  {'cond':>8} {'total':>8} {'beta':>7} {'bias-sum':>9} "
              f"{'computed':>9} {'RAW rms':>8} {'rms/sd':>7} {'duty':>7}")
        for c in CONDS:
            Cc = np.concatenate(store[c]["comp"], 0)
            Cb = np.concatenate(store[c]["bias"], 0)
            tt = np.concatenate(store[c]["tot"], 0)
            mz = zs[c] / max(nseen[c], 1)
            du = ds[c] / max(nseen[c], 1)
            raw = float(np.sqrt(np.mean((r - mz) ** 2)))
            e = dict(total=round(float(tt.mean()), 4),
                     beta=round(beta_term, 4),
                     bias_sum=round(float(Cb.sum(1).mean()), 4),
                     bias_sd=round(float(Cb.sum(1).std()), 4),
                     computed=round(float(Cc.sum(1).mean()), 4),
                     computed_sd=round(float(Cc.sum(1).std()), 4),
                     rms_raw=round(raw, 4),
                     rms_over_sd=round(raw / float(r.std()), 4),
                     duty=round(float(du.mean()), 4))
            entry["conds"][c] = e
            print(f"  {c:>8} {e['total']:>8.3f} {beta_term:>7.3f} "
                  f"{e['bias_sum']:>9.3f} {e['computed']:>9.3f} "
                  f"{e['rms_raw']:>8.4f} {e['rms_over_sd']:>7.3f} "
                  f"{e['duty']:>7.4f}")
        # how much of the reference is input-independent, and what survives
        pr, rd = entry["conds"]["prose"], entry["conds"]["randid"]
        const_share = (beta_term + pr["bias_sum"]) / pr["total"]
        print(f"  -> input-INDEPENDENT share (beta + module biases) of the "
              f"prose reference: {100*const_share:.1f}%")
        print(f"  -> under randid: beta+bias {beta_term + rd['bias_sum']:+.3f} "
              f"(kept {100*(beta_term+rd['bias_sum'])/(beta_term+pr['bias_sum']):.1f}%)"
              f", computed {rd['computed']:+.3f} "
              f"(kept {100*rd['computed']/pr['computed']:.1f}%)")
        entry["const_share_prose"] = round(float(const_share), 4)

        # TEST 2: which sources shift most under corruption?
        Cp = np.concatenate(store["prose"]["comp"], 0).mean(0)
        shifts = {}
        for c in ["randid", "repeat"]:
            Cx = np.concatenate(store[c]["comp"], 0).mean(0)
            dv = Cx - Cp
            order = np.argsort(-np.abs(dv))[:6]
            shifts[c] = [{"src": names[int(i)], "prose": round(float(Cp[i]), 4),
                          "shift": round(float(dv[i]), 4)} for i in order]
            print(f"  biggest computed-part shifts under {c}: " + ", ".join(
                f"{names[int(i)]} {dv[i]:+.3f}" for i in order))
        entry["shifts"] = shifts
        res[f"L{L}"] = entry

    for h_ in hooks:
        h_.remove()
    json.dump({"model": MODEL, "conds": CONDS, "layers": LAYERS, "skip": SKIP,
               "results": res}, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")
    print("\nREAD TEST 3: a large beta+bias share that SURVIVES randid means the")
    print("input-independent bulk is the accumulated learned bias vectors.")
    print("READ TEST 2: compare RAW rms across layers (no denominator artifact)")
    print("and see whether L24's fragility localises to the late MLPs.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
