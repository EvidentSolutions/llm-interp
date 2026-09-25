"""How can b_L be non-atomic AND stable? (Olli, 2026-08-13.)

The apparent tension: 13d found register-conditioned lines whose deviations run
0.19-1.45x ||b_L||, yet b_L's split-half cosine is 0.95-0.99 and it reproduces
at 0.98 on a disjoint Pile sample.

There is no tension, if b_L is a MIXTURE MEAN. Writing the population as classes
c with proportions pi_c and class means mu_c,

    b_L = sum_c pi_c mu_c        and        sum_c pi_c (mu_c - b_L) = 0

exactly, by construction. Large deviations cancel *at the true weights*. So
b_L's stability is the stability of pi, not the homogeneity of the population,
and it should move exactly when the register mix moves.

That is a prediction, not a restatement, and this script tests it three ways:

  1 LINES vs WEIGHTS. cos(mu_c^A, mu_c^B) per class across corpora. If the lines
    are shared and only pi differs, these are high while cos(b_A, b_B) is low.
  2 REWEIGHTING (the decisive test). Build b~ = sum_c pi_c^B mu_c^A -- corpus
    A's lines under corpus B's register mix -- and ask how much of the
    cos(b_A, b_B) gap it closes. If the mixture account is right, most of it.
  3 MIXTURE STABILITY. split-half stability of pi within a corpus, and total
    variation distance between corpora's pi, against the observed cos(b_A, b_B).
    The account predicts these track.

Also reported: the between-class / within-class variance split (how much of the
spread around b_L the register partition actually captures), and the
class-size-vs-deviation relation (a mixture is anchored by its largest class,
which is mechanically its most typical one).

Usage: .venv/Scripts/python.exe superposition/code/census_reference_mixture.py
"""
import sys
import os
import gc
import json
import time
import string
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SEED = 0
LAYERS = [6, 10, 14, 22]
N_DOCS = int(os.environ.get("NDOCS", "40"))
MAXLEN, SKIP = 256, 16
MIN_CLASS = 30

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "census-reference-mixture.json")
PUNCT = set(string.punctuation)


def tok_class(s):
    t = s.strip()
    if "\n" in s:
        return "newline"
    if t == "":
        return "space"
    if any(ch.isdigit() for ch in t):
        return "digit"
    if all(ch in PUNCT for ch in t):
        return "punct"
    if s[:1] == " ":
        return "word_init"
    return "subword"


CLASSES = ["word_init", "subword", "punct", "newline", "digit", "space"]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def build(tok, rng):
    c = {}
    c["prose"] = ("text", json.load(open(
        os.path.join(DATA, "census-docs-phi-2.json"), encoding="utf-8"))[:N_DOCS])
    c["code"] = ("text", json.load(open(
        os.path.join(DATA, "census-docs-code-repo.json"), encoding="utf-8"))[:N_DOCS])
    big = json.load(open(os.path.join(DATA, "pile-big-80000.json"),
                         encoding="utf-8"))
    c["prose_alt"] = ("text", big[50000:50000 + N_DOCS])
    del big
    V = len(tok)
    c["random_tok"] = ("ids", [rng.integers(0, V, size=MAXLEN).tolist()
                               for _ in range(N_DOCS)])
    rep = tok(" the", add_special_tokens=False)["input_ids"]
    c["repeat_tok"] = ("ids", [(rep * MAXLEN)[:MAXLEN] for _ in range(N_DOCS)])
    return c


@torch.no_grad()
def capture(m, tok, kind, docs):
    store = {L: [] for L in LAYERS}
    cls, docid = [], []
    for di, doc in enumerate(docs):
        if kind == "text":
            ids = tok(doc, truncation=True, max_length=MAXLEN,
                      return_tensors="pt")["input_ids"]
        else:
            ids = torch.tensor([doc[:MAXLEN]])
        if ids.shape[1] < SKIP + 48:
            continue
        ids = ids.to(DEV)
        cap, hs = {}, []
        for L in LAYERS:
            def mk(L=L):
                def f(mod, inp):
                    cap[L] = inp[0][0].detach().float()
                return f
            hs.append(m.model.layers[L].mlp.fc1.register_forward_pre_hook(mk()))
        m(input_ids=ids)
        for h in hs:
            h.remove()
        toks = [tok.decode([i]) for i in ids[0].tolist()]
        for L in LAYERS:
            store[L].append(cap[L][SKIP:].cpu())
        for p in range(SKIP, len(toks)):
            cls.append(tok_class(toks[p]))
            docid.append(di)
        del cap
    return ({L: torch.cat(v, 0).float().numpy().astype(np.float64)
             for L, v in store.items()},
            np.array(cls), np.array(docid))


def profile(X, cls, dd):
    """Class proportions, class means, and the between/within variance split."""
    n = len(cls)
    pi, mu, dev = {}, {}, {}
    g = X.mean(0)
    for c in CLASSES:
        sel = cls == c
        k = int(sel.sum())
        pi[c] = k / n
        if k >= MIN_CLASS:
            mu[c] = X[sel].mean(0)
            dev[c] = float(np.linalg.norm(mu[c] - g) /
                           max(np.linalg.norm(g), 1e-12))
    tot = float(((X - g) ** 2).sum(1).mean())
    betw = 0.0
    for c, m_ in mu.items():
        betw += pi[c] * float(((m_ - g) ** 2).sum())
    return {"pi": pi, "mu": mu, "dev": dev, "global": g,
            "between_frac": round(betw / max(tot, 1e-12), 4), "n": n}


@torch.no_grad()
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    corp = build(tok, rng)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True).to(DEV).eval()

    acts, meta = {}, {}
    for name, (kind, docs) in corp.items():
        acts[name], c_, d_ = capture(m, tok, kind, docs)
        meta[name] = (c_, d_)
        print(f"  {name}: {len(c_)} positions")
    del m
    gc.collect()
    torch.cuda.empty_cache()

    rec = {"config": {"n_docs": N_DOCS, "layers": LAYERS, "classes": CLASSES,
                      "min_class": MIN_CLASS, "seed": SEED}}

    # ---- mixture stability, corpus-level (layer-independent)
    print("\n" + "=" * 88)
    print("CLASS MIX pi_c per corpus (this is what must be stable)")
    print("=" * 88)
    print(f"{'corpus':>11} " + " ".join(f"{c:>10}" for c in CLASSES))
    pis = {}
    for name in corp:
        c_, d_ = meta[name]
        pis[name] = np.array([float((c_ == c).mean()) for c in CLASSES])
        print(f"{name:>11} " + " ".join(f"{v:>10.4f}" for v in pis[name]))
    print(f"\n{'corpus':>11} {'splithalf TVD(pi)':>20}")
    for name in corp:
        c_, d_ = meta[name]
        a = np.array([float((c_[d_ % 2 == 0] == c).mean()) for c in CLASSES])
        b = np.array([float((c_[d_ % 2 == 1] == c).mean()) for c in CLASSES])
        print(f"{name:>11} {0.5*np.abs(a-b).sum():>20.4f}")
    rec["pi"] = {n: {c: round(float(v), 5) for c, v in zip(CLASSES, pis[n])}
                 for n in corp}

    out = {}
    for L in LAYERS:
        prof = {n: profile(acts[n][L], *meta[n]) for n in corp}
        e = {"between_frac": {n: prof[n]["between_frac"] for n in corp},
             "dev": {n: {c: round(v, 4) for c, v in prof[n]["dev"].items()}
                     for n in corp},
             "line_cos": {}, "reweight": {}}

        print("\n" + "=" * 88)
        print(f"L{L}  between-class variance fraction: " +
              "  ".join(f"{n}:{prof[n]['between_frac']:.3f}" for n in corp))
        print("=" * 88)

        # 1 -- are the LINES shared across corpora?
        print("  (1) cos(mu_c^A, mu_c^B) per class -- are the lines themselves shared?")
        for a, b in [("prose", "prose_alt"), ("prose", "code"),
                     ("prose", "random_tok")]:
            row = {}
            for c in CLASSES:
                if c in prof[a]["mu"] and c in prof[b]["mu"]:
                    row[c] = round(float(unit(prof[a]["mu"][c]) @
                                         unit(prof[b]["mu"][c])), 4)
            e["line_cos"][f"{a}|{b}"] = row
            base = float(unit(prof[a]["global"]) @ unit(prof[b]["global"]))
            print(f"      {a:>9} vs {b:<11} cos(b_A,b_B)={base:+.3f}   " +
                  "  ".join(f"{c}:{v:+.3f}" for c, v in row.items()))

        # 2 -- REWEIGHTING: A's lines under B's mix
        print("  (2) reweight A's lines by B's mix: does it close the gap to b_B?")
        for a, b in [("prose", "code"), ("code", "prose"),
                     ("prose", "random_tok"), ("prose", "prose_alt")]:
            shared = [c for c in CLASSES
                      if c in prof[a]["mu"] and c in prof[b]["mu"]]
            wa = np.array([prof[a]["pi"][c] for c in shared])
            wb = np.array([prof[b]["pi"][c] for c in shared])
            wa, wb = wa / wa.sum(), wb / wb.sum()
            M = np.stack([prof[a]["mu"][c] for c in shared])
            b_self = wa @ M                      # A's lines, A's mix
            b_re = wb @ M                        # A's lines, B's mix
            gb = unit(prof[b]["global"])
            base = float(unit(b_self) @ gb)
            new = float(unit(b_re) @ gb)
            closed = (new - base) / max(1.0 - base, 1e-9)
            e["reweight"][f"{a}->{b}"] = {
                "cos_before": round(base, 4), "cos_after": round(new, 4),
                "gap_closed": round(float(closed), 4),
                "tvd_pi": round(float(0.5 * np.abs(wa - wb).sum()), 4)}
            print(f"      {a:>9} lines @ {b:<11} mix: cos {base:+.3f} -> "
                  f"{new:+.3f}   gap closed {closed*100:>5.1f}%   "
                  f"TVD(pi)={0.5*np.abs(wa-wb).sum():.3f}")
        out[str(L)] = e

    rec["layers"] = out
    json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
