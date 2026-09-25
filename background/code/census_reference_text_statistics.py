# -*- coding: utf-8 -*-
"""WHICH properties of natural text supply the gates' operating point, and are
the individual contributions interpretable? (Olli, 2026-08-19.)

BACKGROUND. 15bn showed the reference component <x_ln, b_hat_L> is stable because
(a) LN's per-position denominator removes 66-85% of its variance and (b) what is
left is a sum of 17-49 small steady contributions. It also showed the sum is far
steadier than its parts: at L8 the biggest contributor a0 has CV 0.327 and m7 has
CV 6.65, while the TOTAL has CV 0.118. So stability is partly aggregation over
sources that each react to something different -- Olli's composition reading.

And b_L is input-conditioned: prose-vs-prose cos 0.97-0.98, code 0.76-0.84,
random ids 0.48-0.52, repeated token 0.02-0.05. So the gates lean on a property
of natural text. WHICH property?

LEG 1 -- THE STATISTIC-ABLATION LADDER. Each condition destroys ONE property of
natural text while preserving others. The decisive cell is SHUFFLE, which
preserves the unigram multiset EXACTLY and destroys all order:

    prose      untouched
    shuffle    token order permuted within the document
               -> unigram distribution IDENTICAL, syntax and bigrams destroyed
    freqmatch  each token replaced by a random token from its own frequency
               decile -> frequency PROFILE preserved, identity destroyed
    randid     uniform random token ids -> frequency distribution destroyed too
    repeat     the document's commonest token repeated (the degenerate floor)
    lower      lowercased -> case statistics destroyed
    nopunct    punctuation stripped -> delimiter statistics destroyed

  PREDICTION if the operative statistic is the token FREQUENCY DISTRIBUTION:
  shuffle and freqmatch keep the reference (cos_prose high), randid degrades,
  repeat collapses. If instead SYNTAX is required, shuffle should already break.

  Measured per condition, per layer:
    cos_prose   cos(x_ln, b_hat_L estimated on held-out PROSE) -- is the
                reference the gates were tuned against still present?
    resting_err THE NON-TAUTOLOGICAL CALIBRATION MEASURE. resting_n =
                w_n . b_L^prose + b_n is a WEIGHT-SPACE constant: the operating
                point the gate carries. Compare it against the MEASURED mean
                pre-activation under this condition -- corr and RMS error scaled
                by the resting spread. Under prose these agree by construction;
                the mismatch elsewhere is exactly "the calibration no longer
                holds". (Deliberately not rho(resting, duty) alone, which is
                near-tautological -- see the 2026-08-13 retraction -- though it
                is reported for continuity since the weights are FIXED across
                conditions, so its CHANGE is informative.)
    duty        mean firing fraction and its spread
    entropy     output entropy / top-1 probability, as coarse health

LEG 2 -- ARE THE INDIVIDUAL MECHANISMS INTERPRETABLE? For each source s, its
exact contribution c_s(p) to the reference component (the 15bn decomposition) is
regressed on TEXT-ONLY interpretable features, held out, against a
label-permutation null:
    log unigram frequency of the current and previous token, log position,
    is_punct, has_leading_space, is_capitalised, is_digit, is_subword,
    token length, in_quote (running quote parity), paren depth,
    is_sentence_start.
Reports per-source held-out R^2 and its largest standardised coefficient, so
"each write reacts to a different feature" becomes a measurement rather than a
story.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_text_statistics.py
"""
import os
import re
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
LEG2_LAYER = 16
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
PILE = os.path.join(DATA, "pile-big-80000.json")
FREQ = os.path.join(DATA, "pile-token-freq-phi2.json")
OUT = os.path.join(DATA, "census-reference-text-statistics.json")
CONDS = ["prose", "shuffle", "freqmatch", "randid", "repeat", "lower", "nopunct"]
RNG = np.random.default_rng(20260819)


def ridge_r2(X, y, folds=4, alphas=(1e-2, 1e-1, 1.0, 10.0, 100.0)):
    n = len(y)
    idx = RNG.permutation(n)
    X, y = X[idx], y[idx]
    fold = np.array_split(np.arange(n), folds)
    pred = np.zeros(n)
    W = None
    for f in range(folds):
        te = fold[f]
        tr = np.concatenate([fold[g] for g in range(folds) if g != f])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Z, ym = (X[tr] - mu) / sd, y[tr].mean()
        best, bw, ba = None, None, alphas[0]
        cut = int(0.8 * len(tr))
        for a in alphas:
            w = np.linalg.solve(Z[:cut].T @ Z[:cut] + a * np.eye(X.shape[1]),
                                Z[:cut].T @ (y[tr][:cut] - ym))
            e = float(np.mean((Z[cut:] @ w - (y[tr][cut:] - ym)) ** 2))
            if best is None or e < best:
                best, ba = e, a
        w = np.linalg.solve(Z.T @ Z + ba * np.eye(X.shape[1]), Z.T @ (y[tr] - ym))
        pred[te] = ((X[te] - mu) / sd) @ w + ym
        W = w
    ss = float(np.sum((y - pred) ** 2)) / max(float(np.sum((y - y.mean()) ** 2)), 1e-30)
    return 1.0 - ss, W


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()
    Lz, NL, d = m.model.layers, len(m.model.layers), m.config.hidden_size
    docs = json.load(open(PILE, encoding="utf-8"))
    bl_docs, ev_docs = docs[:N_BL], docs[N_BL:N_BL + N_EVAL]

    raw = json.load(open(FREQ, encoding="utf-8"))
    cnt = {int(k): v for k, v in raw["counts"].items()}
    print(f"  freq table {raw['total']:,} tokens, {len(cnt):,} types")
    ids_sorted = sorted(cnt, key=lambda t: -cnt[t])
    decile = {}
    for i, t in enumerate(ids_sorted):
        decile[t] = min(9, int(10 * i / len(ids_sorted)))
    by_dec = {q: [] for q in range(10)}
    for t, q in decile.items():
        by_dec[q].append(t)
    logf = {t: np.log10(c + 1.0) for t, c in cnt.items()}
    DEFLF = float(np.median(list(logf.values())))

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

    def run_ids(idlist):
        t = torch.tensor([idlist], device=DEV)
        with torch.no_grad():
            out = m(input_ids=t)
        return out.logits[0].float()

    # ---- b_L^prose on the disjoint set -------------------------------------
    acc = {L: torch.zeros(d, device=DEV, dtype=torch.float64) for L in LAYERS}
    n = 0
    for text in bl_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 8:
            continue
        run_ids(ids)
        for L in LAYERS:
            with torch.no_grad():
                acc[L] += Lz[L].input_layernorm(
                    cap[f"h{L}"])[SKIP:].double().sum(0)
        n += len(ids) - SKIP
    bL, bhat, resting = {}, {}, {}
    for L in LAYERS:
        v = (acc[L] / n).float()
        bL[L] = v
        bhat[L] = v / v.norm()
        W1 = Lz[L].mlp.fc1.weight.detach().float()
        b1 = Lz[L].mlp.fc1.bias.detach().float()
        resting[L] = (W1 @ v + b1).cpu().numpy()      # weight-space constant
    print(f"  b_L^prose from {n} positions; resting sd per layer: " +
          ", ".join(f"L{L} {resting[L].std():.3f}" for L in LAYERS))

    PUNCT = re.compile(r"^[\s]*[^\w\s]+$")

    def variants(text):
        base = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(base) < SKIP + 16:
            return None
        out = {"prose": base}
        sh = list(base)
        RNG.shuffle(sh)
        out["shuffle"] = sh
        out["freqmatch"] = [
            int(by_dec[decile.get(t, 9)][RNG.integers(
                0, len(by_dec[decile.get(t, 9)]))]) for t in base]
        out["randid"] = [int(x) for x in RNG.integers(0, 50000, len(base))]
        common = Counter(base).most_common(1)[0][0]
        out["repeat"] = [int(common)] * len(base)
        for name, fn in (("lower", str.lower),
                         ("nopunct", lambda s: re.sub(r"[^\w\s]", "", s))):
            v = tok(fn(text), truncation=True, max_length=MAXLEN)["input_ids"]
            out[name] = v if len(v) >= SKIP + 16 else base
        return out

    # ---- LEG 1 --------------------------------------------------------------
    agg = {c: {L: {"cos": [], "z": [], "duty": []} for L in LAYERS}
           for c in CONDS}
    health = {c: {"ent": [], "top1": []} for c in CONDS}
    zsum = {c: {L: np.zeros(Lz[L].mlp.fc1.out_features) for L in LAYERS}
            for c in CONDS}
    dsum = {c: {L: np.zeros(Lz[L].mlp.fc1.out_features) for L in LAYERS}
            for c in CONDS}
    npos = {c: 0 for c in CONDS}
    for text in ev_docs:
        V = variants(text)
        if V is None:
            continue
        for c in CONDS:
            lg = run_ids(V[c])
            lp = torch.log_softmax(lg[SKIP:], -1)
            p = lp.exp()
            health[c]["ent"].extend((-(p * lp).sum(-1)).cpu().numpy().tolist())
            health[c]["top1"].extend(p.max(-1).values.cpu().numpy().tolist())
            for L in LAYERS:
                with torch.no_grad():
                    x = Lz[L].input_layernorm(cap[f"h{L}"])[SKIP:].float()
                xn = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                agg[c][L]["cos"].extend((xn @ bhat[L]).cpu().numpy().tolist())
                z = cap[f"z{L}"][SKIP:].float()
                zsum[c][L] += z.sum(0).cpu().numpy()
                dsum[c][L] += (z > 0).float().sum(0).cpu().numpy()
            npos[c] += V[c].__len__() - SKIP

    print("\n=== LEG 1: which text statistic supplies the operating point? ===")
    leg1 = {}
    for L in LAYERS:
        print(f"\n--- L{L} (resting is FIXED weight-space; b_L from prose) ---")
        print(f"  {'condition':>10} {'cos_prose':>10} {'cos sd':>7} "
              f"{'corr(rest,meanz)':>17} {'RMSerr/sd':>10} "
              f"{'duty':>6} {'rho(rest,duty)':>15} {'entropy':>8}")
        leg1[f"L{L}"] = {}
        for c in CONDS:
            cs = np.array(agg[c][L]["cos"])
            mz = zsum[c][L] / max(npos[c], 1)
            du = dsum[c][L] / max(npos[c], 1)
            r = resting[L]
            corr = float(np.corrcoef(r, mz)[0, 1])
            rmse = float(np.sqrt(np.mean((r - mz) ** 2)) / r.std())
            from scipy.stats import spearmanr
            rho = float(spearmanr(r, du).statistic)
            e = dict(cos_prose=round(float(cs.mean()), 4),
                     cos_sd=round(float(cs.std()), 4),
                     corr_rest_meanz=round(corr, 4),
                     rms_err_over_sd=round(rmse, 4),
                     duty=round(float(du.mean()), 4),
                     rho_rest_duty=round(rho, 4),
                     entropy=round(float(np.mean(health[c]["ent"])), 3),
                     top1=round(float(np.mean(health[c]["top1"])), 4))
            leg1[f"L{L}"][c] = e
            print(f"  {c:>10} {e['cos_prose']:>10.4f} {e['cos_sd']:>7.4f} "
                  f"{e['corr_rest_meanz']:>17.4f} {e['rms_err_over_sd']:>10.3f} "
                  f"{e['duty']:>6.4f} {e['rho_rest_duty']:>15.4f} "
                  f"{e['entropy']:>8.3f}")

    # ---- LEG 2 --------------------------------------------------------------
    print(f"\n=== LEG 2: are the individual contributions interpretable? "
          f"(L{LEG2_LAYER}) ===")
    L = LEG2_LAYER
    ln = Lz[L].input_layernorm
    gam, eps = ln.weight.detach().float(), ln.eps
    u = gam * bhat[L]
    su = float(u.sum())
    names = ["embed"] + [f"a{k}" for k in range(L)] + [f"m{k}" for k in range(L)]
    FEAT = ["logf_cur", "logf_prev", "log_pos", "is_punct", "lead_space",
            "is_cap", "is_digit", "is_subword", "tok_len", "in_quote",
            "paren_depth", "sent_start"]
    C, F = [], []
    for text in ev_docs:
        ids = tok(text, truncation=True, max_length=MAXLEN)["input_ids"]
        if len(ids) < SKIP + 16:
            continue
        run_ids(ids)
        h = cap[f"h{L}"].float()
        srcs = [cap["h0"].float()] + [cap[f"a{k}"].float() for k in range(L)] \
            + [cap[f"m{k}"].float() for k in range(L)]
        sigma = h.var(-1, unbiased=False).add(eps).sqrt()
        con = torch.stack([(s @ u) - s.mean(-1) * su for s in srcs], 1) \
            / sigma[:, None]
        C.append(con[SKIP:].cpu().numpy())
        strs = [tok.decode([t]) for t in ids]
        q, pd = 0, 0
        rows = []
        for i, s in enumerate(strs):
            q ^= s.count('"') & 1
            pd += s.count("(") - s.count(")")
            prev = strs[i - 1] if i else ""
            body = s.strip()
            rows.append([
                logf.get(ids[i], DEFLF), logf.get(ids[i - 1], DEFLF) if i else DEFLF,
                np.log10(i + 1.0),
                1.0 if PUNCT.match(s) else 0.0,
                1.0 if s[:1] == " " else 0.0,
                1.0 if body[:1].isupper() else 0.0,
                1.0 if body[:1].isdigit() else 0.0,
                1.0 if (s[:1] != " " and body[:1].isalpha()) else 0.0,
                float(len(s)), float(q), float(max(pd, 0)),
                1.0 if prev.strip()[-1:] in ".!?" else 0.0])
        F.append(np.array(rows)[SKIP:])
    C = np.concatenate(C, 0); F = np.concatenate(F, 0)
    print(f"  {C.shape[0]} positions, {len(FEAT)} features, {C.shape[1]} sources")
    r2n, _ = ridge_r2(F, RNG.permutation(C[:, 0].copy()))
    print(f"  permutation null R^2 = {r2n:+.4f}")
    rows = []
    for i in range(C.shape[1]):
        r2, w = ridge_r2(F, C[:, i].copy())
        top = FEAT[int(np.argmax(np.abs(w)))]
        rows.append((names[i], r2, top, float(C[:, i].mean()),
                     float(C[:, i].std())))
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    print(f"  {'source':>7} {'R^2':>8} {'top feature':>13} {'mean':>8} {'sd':>7}")
    for nm, r2, top, mu, sd in rows_sorted[:12]:
        print(f"  {nm:>7} {r2:>8.4f} {top:>13} {mu:>+8.3f} {sd:>7.3f}")
    print("  ... worst three:")
    for nm, r2, top, mu, sd in rows_sorted[-3:]:
        print(f"  {nm:>7} {r2:>8.4f} {top:>13} {mu:>+8.3f} {sd:>7.3f}")
    tot = C.sum(1)
    r2t, wt = ridge_r2(F, tot.copy())
    print(f"  TOTAL reference component: R^2 {r2t:.4f}, "
          f"top feature {FEAT[int(np.argmax(np.abs(wt)))]}, "
          f"mean {tot.mean():+.3f}, sd {tot.std():.3f}")

    for h_ in hooks:
        h_.remove()
    json.dump({"model": MODEL, "conds": CONDS, "layers": LAYERS, "skip": SKIP,
               "leg1": leg1,
               "leg2": {"layer": L, "features": FEAT, "n_pos": int(C.shape[0]),
                        "perm_null_r2": round(r2n, 4),
                        "total": {"r2": round(r2t, 4),
                                  "top": FEAT[int(np.argmax(np.abs(wt)))]},
                        "sources": [{"src": nm, "r2": round(r2, 4), "top": top,
                                     "mean": round(mu, 4), "sd": round(sd, 4)}
                                    for nm, r2, top, mu, sd in rows]}},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSaved {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
