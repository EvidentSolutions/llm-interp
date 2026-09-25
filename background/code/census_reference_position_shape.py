"""Is b_L a standing component present AT EACH POSITION, or only a centroid of a
cloud that no individual position resembles? And does it have internal shape?
(Olli, 2026-08-13: "we identified it as an average entity -- give me examples at
random positions, across different corpora; does it have a shape/meaning beyond
a bias term at discrete points?")

b_L is DEFINED as a corpus mean, so every claim about it being "carried" and
"maintained" is, so far, a claim about an average. An average has a shape only if
the population concentrates around it. Five legs:

  A PRESENCE     per-position cos(x_ln, b_hat_L): full distribution, not just the
                 mean. If the 1st percentile is high, b_L is at every position.
                 If the spread is wide, it is a centroid and "carried constant"
                 is the wrong description.
  B EXAMPLES     literal random positions: token, context, cos, coefficient.
  C IDENTITY     cos(b_hat^corpusA, b_hat^corpusB) for every pair, against the
                 SPLIT-HALF floor within each corpus (the sampling noise on a
                 direction estimate). Includes two controls that decide whether
                 b_L is a MODEL object or a TEXT statistic:
                   random_tok  uniform-random token ids (no natural text at all)
                   repeat_tok  one token repeated
                 If b_hat survives those at high cosine to prose, it is intrinsic.
  D SHAPE        register-conditioned means. Partition positions by token class
                 (punct / word-initial / subword / digit / newline) and by quote
                 state; per-class mean direction; cos(class, global) and pairwise
                 cos, each against the split-half floor for that class. This is
                 the paper's open "within-layer atomicity" question: one atomic
                 direction, or several register-conditioned bias lines under a
                 rank-1 mean?
  E MEANING      top-k W_U rows by cosine for b_hat per corpus per layer (the
                 paper's own S2 method), and for each class DEVIATION direction
                 (class_mean - global_mean), which is where any per-register
                 content would live.

Usage: .venv/Scripts/python.exe superposition/code/census_reference_position_shape.py
       SMOKE=1 fast pass
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
SMOKE = os.environ.get("SMOKE", "") not in ("", "0", "false")
LAYERS = [6, 10, 14, 22]
N_DOCS = 5 if SMOKE else 40
MAXLEN = 256
SKIP = 16
N_EXAMPLES = 12
TOPK_DECODE = 8

DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
TWIN = os.path.join(DATA, "phi2-random-twin")
OUT = os.path.join(DATA, "census-reference-position-shape.json")
PUNCT = set(string.punctuation)


def build_corpora(tok, rng):
    """Five corpora: the paper's prose, a disjoint Pile sample, code, and two
    synthetic controls fed as raw token ids."""
    c = {}
    prose = json.load(open(os.path.join(DATA, "census-docs-phi-2.json"),
                           encoding="utf-8"))[:N_DOCS]
    c["prose"] = ("text", prose)
    code = json.load(open(os.path.join(DATA, "census-docs-code-repo.json"),
                          encoding="utf-8"))[:N_DOCS]
    c["code"] = ("text", code)
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


@torch.no_grad()
def capture(m, tok, kind, docs, n_layers):
    """Return per-layer (N,d) fc1 inputs over far positions, plus per-position
    metadata (token string, class, quote state, doc/pos index)."""
    store = {L: [] for L in LAYERS if L < n_layers}
    meta = []
    for di, doc in enumerate(docs):
        if kind == "text":
            ids = tok(doc, truncation=True, max_length=MAXLEN,
                      return_tensors="pt")["input_ids"]
        else:
            ids = torch.tensor([doc[:MAXLEN]])
        if ids.shape[1] < SKIP + 48:
            continue
        ids = ids.to(DEV)
        cap = {}
        hs = []
        for L in store:
            def mk(L=L):
                def f(mod, inp):
                    cap[L] = inp[0][0].detach().float()
                return f
            hs.append(m.model.layers[L].mlp.fc1.register_forward_pre_hook(mk()))
        m(input_ids=ids)
        for h in hs:
            h.remove()
        toks = [tok.decode([i]) for i in ids[0].tolist()]
        q = 0
        qstate = []
        for s in toks:
            if '"' in s:
                q ^= 1
            qstate.append(q)
        for L in store:
            store[L].append(cap[L][SKIP:].cpu())
        for p in range(SKIP, len(toks)):
            meta.append({"doc": di, "pos": p, "tok": toks[p],
                         "cls": tok_class(toks[p]), "inq": qstate[p],
                         "ctx": "".join(toks[max(0, p - 7):p])})
        del cap
    store = {L: torch.cat(v, 0) for L, v in store.items()}
    return store, meta


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def summarize(X, b):
    """Per-position cosine to b, as a distribution."""
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    c = Xn @ unit(b)
    return {"mean": round(float(c.mean()), 4), "std": round(float(c.std()), 4),
            "min": round(float(c.min()), 4),
            "p01": round(float(np.percentile(c, 1)), 4),
            "p50": round(float(np.percentile(c, 50)), 4),
            "p99": round(float(np.percentile(c, 99)), 4),
            "max": round(float(c.max()), 4),
            "frac_above_0.5": round(float((c > 0.5).mean()), 4)}, c


@torch.no_grad()
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    corpora = build_corpora(tok, rng)
    print(f"Device {DEV} SMOKE={SMOKE} docs/corpus={N_DOCS} layers={LAYERS}")
    print("corpora: " + ", ".join(corpora))

    rec = {"config": {"smoke": SMOKE, "n_docs": N_DOCS, "skip": SKIP,
                      "layers": LAYERS, "seed": SEED, "corpora": list(corpora)}}

    for which in (["trained"] if SMOKE else ["trained", "random"]):
        print(f"\n{'='*78}\n{which.upper()}\n{'='*78}")
        if which == "random":
            m = AutoModelForCausalLM.from_pretrained(
                TWIN, dtype=torch.float16,
                low_cpu_mem_usage=False).to(DEV).eval()
        else:
            m = AutoModelForCausalLM.from_pretrained(
                MODEL, dtype=torch.float16,
                low_cpu_mem_usage=True).to(DEV).eval()
        n_layers = len(m.model.layers)
        WU = m.lm_head.weight.detach().float()
        WUn = (WU / WU.norm(dim=1, keepdim=True).clamp(min=1e-9))

        acts, metas = {}, {}
        for name, (kind, docs) in corpora.items():
            acts[name], metas[name] = capture(m, tok, kind, docs, n_layers)
            print(f"  {name}: {len(metas[name])} far positions")

        out = {}
        for L in [x for x in LAYERS if x < n_layers]:
            e = {"presence": {}, "identity": {}, "splithalf": {},
                 "classes": {}, "decode": {}, "examples": []}
            bhat = {}
            for name in corpora:
                X = acts[name][L].float().numpy().astype(np.float64)
                b = X.mean(0)
                bhat[name] = unit(b)
                stats, cvec = summarize(X, b)
                e["presence"][name] = stats
                e["presence"][name]["b_norm"] = round(float(np.linalg.norm(b)), 3)
                e["presence"][name]["x_norm_mean"] = round(
                    float(np.linalg.norm(X, axis=1).mean()), 3)
                # split-half floor (even/odd documents)
                d_idx = np.array([mm["doc"] for mm in metas[name]])
                h1 = X[d_idx % 2 == 0].mean(0)
                h2 = X[d_idx % 2 == 1].mean(0)
                e["splithalf"][name] = round(float(unit(h1) @ unit(h2)), 4)

            e["identity"] = {a: {b_: round(float(bhat[a] @ bhat[b_]), 4)
                                 for b_ in corpora} for a in corpora}

            # presence of the PROSE reference in every other corpus
            e["prose_ref_presence"] = {}
            for name in corpora:
                X = acts[name][L].float().numpy().astype(np.float64)
                s, _ = summarize(X, bhat["prose"])
                e["prose_ref_presence"][name] = s

            # ---- D: register-conditioned means, on prose+code pooled
            for cname in ("prose", "code"):
                X = acts[cname][L].float().numpy().astype(np.float64)
                mm = metas[cname]
                g = unit(X.mean(0))
                d_idx = np.array([q["doc"] for q in mm])
                cls = np.array([q["cls"] for q in mm])
                ent = {}
                for c in sorted(set(cls.tolist())):
                    sel = cls == c
                    if sel.sum() < 40:
                        continue
                    Xc = X[sel]
                    dc = d_idx[sel]
                    cm = unit(Xc.mean(0))
                    a = Xc[dc % 2 == 0]
                    b_ = Xc[dc % 2 == 1]
                    floor = (round(float(unit(a.mean(0)) @ unit(b_.mean(0))), 4)
                             if len(a) > 5 and len(b_) > 5 else None)
                    dev = Xc.mean(0) - X.mean(0)
                    ent[c] = {"n": int(sel.sum()),
                              "cos_to_global": round(float(cm @ g), 4),
                              "splithalf_floor": floor,
                              "dev_norm_over_b": round(
                                  float(np.linalg.norm(dev) /
                                        max(np.linalg.norm(X.mean(0)), 1e-9)), 4)}
                    dv = torch.tensor(unit(dev), dtype=torch.float32, device=DEV)
                    top = (WUn @ dv).topk(TOPK_DECODE)
                    ent[c]["dev_decode"] = [tok.decode([i]) for i in
                                            top.indices.tolist()]
                    ent[c]["dev_decode_maxcos"] = round(float(top.values[0]), 4)
                keys = [k for k in ent]
                ent["_pairwise"] = {}
                for i, k1 in enumerate(keys):
                    for k2 in keys[i + 1:]:
                        s1 = cls == k1
                        s2 = cls == k2
                        ent["_pairwise"][f"{k1}|{k2}"] = round(float(
                            unit(X[s1].mean(0)) @ unit(X[s2].mean(0))), 4)
                e["classes"][cname] = ent

            # ---- F: pooled mean vs POSITION-CONDITIONAL mean.
            # b_L is the mean pooled over all far positions. If a mean taken at
            # a FIXED position index (over documents) captures much more of the
            # norm, then the standing structure has position-dependent shape
            # that the pooled rank-1 b_L cannot represent -- which is the direct
            # form of "is it more than a bias term at discrete points?"
            for cname in ("prose", "code"):
                X = acts[cname][L].float().numpy().astype(np.float64)
                mm = metas[cname]
                pos = np.array([q["pos"] for q in mm])
                dd = np.array([q["doc"] for q in mm])
                g = X.mean(0)
                gh = unit(g)
                ndoc = len(set(dd.tolist()))
                shares, cosg, resid, floors = [], [], [], []
                for t in sorted(set(pos.tolist())):
                    sel = pos == t
                    if sel.sum() < max(4, 0.6 * ndoc):
                        continue
                    Xt = X[sel]
                    pm = Xt.mean(0)
                    shares.append(np.linalg.norm(pm) /
                                  max(np.linalg.norm(Xt, axis=1).mean(), 1e-9))
                    cosg.append(unit(pm) @ gh)
                    r = pm - (pm @ gh) * gh
                    resid.append(np.linalg.norm(r) /
                                 max(np.linalg.norm(pm), 1e-9))
                    dt = dd[sel]
                    a, b_ = Xt[dt % 2 == 0], Xt[dt % 2 == 1]
                    if len(a) > 2 and len(b_) > 2:
                        floors.append(unit(a.mean(0)) @ unit(b_.mean(0)))
                if shares:
                    e.setdefault("position_conditional", {})[cname] = {
                        "n_positions": len(shares),
                        "pooled_share": round(float(
                            np.linalg.norm(g) /
                            np.linalg.norm(X, axis=1).mean()), 4),
                        "poscond_share_mean": round(float(np.mean(shares)), 4),
                        "poscond_share_p05": round(float(np.percentile(shares, 5)), 4),
                        "poscond_share_p95": round(float(np.percentile(shares, 95)), 4),
                        "cos_poscond_to_pooled_mean": round(float(np.mean(cosg)), 4),
                        "resid_after_removing_bL_mean": round(float(np.mean(resid)), 4),
                        "splithalf_floor_mean": (round(float(np.mean(floors)), 4)
                                                 if floors else None),
                    }

            # ---- E: what b_hat decodes to, per corpus
            for name in corpora:
                dv = torch.tensor(bhat[name], dtype=torch.float32, device=DEV)
                top = (WUn @ dv).topk(TOPK_DECODE)
                e["decode"][name] = {
                    "top": [tok.decode([i]) for i in top.indices.tolist()],
                    "maxcos": round(float(top.values[0]), 4)}

            # ---- B: literal examples at random positions
            for name in ("prose", "code", "prose_alt", "random_tok"):
                X = acts[name][L].float().numpy().astype(np.float64)
                mm = metas[name]
                idx = rng.choice(len(mm), size=min(3, len(mm)), replace=False)
                for i in idx:
                    x = X[i]
                    e["examples"].append({
                        "corpus": name, "tok": mm[i]["tok"],
                        "cls": mm[i]["cls"], "pos": mm[i]["pos"],
                        "ctx": mm[i]["ctx"][-40:],
                        "cos_own_bhat": round(float(unit(x) @ bhat[name]), 4),
                        "cos_prose_bhat": round(float(unit(x) @ bhat["prose"]), 4),
                        "coef": round(float(x @ bhat[name]), 3),
                        "x_norm": round(float(np.linalg.norm(x)), 3)})
            out[str(L)] = e

            p = e["presence"]
            print(f"\n-- L{L} -- per-position cos(x, b_hat_own): "
                  f"mean [p01 .. p99]")
            for name in corpora:
                s = p[name]
                print(f"   {name:>11}: {s['mean']:.3f} "
                      f"[{s['p01']:.3f} .. {s['p99']:.3f}] "
                      f"min {s['min']:+.3f}  >0.5: {s['frac_above_0.5']:.3f}"
                      f"   splithalf {e['splithalf'][name]:.4f}")
            print(f"   cos(b_hat_prose, b_hat_X):  " + "  ".join(
                f"{n}:{e['identity']['prose'][n]:+.3f}" for n in corpora))
            print(f"   prose-ref present in X (mean cos): " + "  ".join(
                f"{n}:{e['prose_ref_presence'][n]['mean']:+.3f}" for n in corpora))
            print(f"   decode b_hat: " + " | ".join(
                f"{n}: {' '.join(repr(t) for t in e['decode'][n]['top'][:5])}"
                for n in ("prose", "code", "random_tok")))
            for cname, pc in e.get("position_conditional", {}).items():
                print(f"   [{cname}] POOLED share {pc['pooled_share']:.3f} vs "
                      f"POSITION-CONDITIONAL {pc['poscond_share_mean']:.3f} "
                      f"[p05 {pc['poscond_share_p05']:.3f}, p95 "
                      f"{pc['poscond_share_p95']:.3f}]  "
                      f"cos(poscond, b_L)={pc['cos_poscond_to_pooled_mean']:.3f}  "
                      f"non-b_L residue={pc['resid_after_removing_bL_mean']:.3f}  "
                      f"floor={pc['splithalf_floor_mean']}")
            for cname, ent in e["classes"].items():
                rows = [(k, v) for k, v in ent.items() if k != "_pairwise"]
                print(f"   [{cname}] register-conditioned means "
                      f"(cos to global / split-half floor / dev norm):")
                for k, v in rows:
                    print(f"      {k:>10} n={v['n']:>5}  "
                          f"cos={v['cos_to_global']:.3f}  "
                          f"floor={v['splithalf_floor']}  "
                          f"dev={v['dev_norm_over_b']:.3f}  "
                          f"-> {' '.join(repr(t) for t in v['dev_decode'][:5])}")

        rec[which] = out
        del m, acts
        gc.collect()
        torch.cuda.empty_cache()

    json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSaved {OUT} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
