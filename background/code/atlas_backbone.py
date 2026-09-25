# -*- coding: utf-8 -*-
"""E92 -- the atlas, applied to the non-token structural backbone (Olli).

E91 settled that the massive activations + bias are a non-token structural
backbone. What does its DYNAMIC content compute? Apply the atlas method (E88):
correlate the backbone's values with candidate functions, and do the
context-resolved causal ablation to see WHERE removing it hurts.

Candidate functions: CONFIDENCE (next-token entropy; the softmax-null/entropy-
neuron family), FREQUENCY/REGISTER (the function-word crust the writers decode
to), POSITION.

PART 1 -- correlate per-position: each massive-dim value and the bias-projection
  (h.b_L) against next-token entropy, position index, current-token log-freq.
PART 2 -- causal, context-resolved: in the REAL residual at L, ablate the
  backbone dynamic content (massive dims -> corpus constant; and separately
  remove the b_L component); substitute; measure per-position dNLL and dEntropy,
  bucketed by entropy tercile / position band / function-vs-content token. Does
  the backbone carry CONFIDENCE (dEntropy) or CONTENT (dNLL/top-1)?

Usage: .venv/Scripts/python.exe superposition/code/atlas_backbone.py
"""
import sys, os, json, math
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
torch.manual_seed(0)
DATA = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data"))
OUT = os.path.join(DATA, "atlas-backbone.json")
L = 16
N_MASSIVE = 8

FUNCTION_WORDS = set("the a an of to and in is was were are be been for on at "
                     "with as by that this it he she they we you i but or not "
                     "his her their its from which who".split())

CORPUS = [
    "The scientist carefully measured the temperature of the boiling water today.",
    "After the long storm finally passed, the sky slowly turned a brilliant orange.",
    "She opened the ancient leather book and began to read the very first chapter.",
    "The company announced that its quarterly profits had risen sharply this year.",
    "He walked into the crowded kitchen and immediately smelled the fresh bread.",
    "The children played happily in the park until the sun began to set slowly.",
    "A group of researchers published their surprising findings in a major journal.",
    "The old wooden bridge creaked loudly as the heavy truck rolled slowly across.",
    "Investors grew nervous as the stock market fell for the third straight week.",
    "The chef added a pinch of salt and a little pepper to the tomato sauce.",
    "Doctors recommend regular exercise and a balanced diet for good health daily.",
    "The spacecraft entered orbit around the distant planet after many long years.",
    "Quantum computers may one day solve problems classical machines cannot handle.",
    "The museum displayed rare paintings from several different historical periods.",
    "Farmers worried that the unexpected frost would damage their delicate crops.",
    "The orchestra tuned their instruments before the conductor raised his baton.",
]


def load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, attn_implementation="sdpa").to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def cap_resid(m, ids, L):
    cap = {}

    def hook(mod, i, o):
        cap["h"] = (o[0] if isinstance(o, tuple) else o)[0].detach().float()
    hd = m.model.layers[L].register_forward_hook(hook)
    with torch.no_grad():
        out = m(torch.tensor([ids], device=DEV), use_cache=False)
    hd.remove()
    return cap["h"], out.logits[0].float()


def sub(m, ids, L, Hnew):
    done = [False]

    def hook(mod, i, o):
        if done[0]:
            return o
        done[0] = True
        tup = isinstance(o, tuple)
        h = (o[0] if tup else o).clone()
        h[0] = Hnew.to(h.dtype)
        return ((h,) + o[1:]) if tup else h
    hd = m.model.layers[L].register_forward_hook(hook)
    with torch.no_grad():
        out = m(torch.tensor([ids], device=DEV), use_cache=False)
    hd.remove()
    return out.logits[0].float()


def spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 8:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    tok, m = load()
    d = m.config.hidden_size

    # freq for log-freq of current token (quick corpus count)
    from collections import Counter
    cnt = Counter()
    idsets = []
    for t in CORPUS:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        idsets.append(ids); cnt.update(ids)

    # first pass: identify massive dims, gather per-position features
    mag = torch.zeros(d, device=DEV); npos = 0
    allh = []
    for ids in idsets:
        H, _ = cap_resid(m, ids, L)
        mag += H[1:].abs().sum(0); npos += H.shape[0] - 1
        allh.append(H[1:])
    mag /= npos
    massive = mag.topk(N_MASSIVE).indices
    const_massive = torch.cat(allh, 0).mean(0)          # corpus mean (all dims)
    b_L = unit(torch.cat(allh, 0).mean(0))

    # per-position features
    feats = {"entropy": [], "position": [], "logfreq": [],
             "biasproj": [], "isfunc": []}
    massvals = {int(j): [] for j in massive.tolist()}
    for ids in idsets:
        H, lg = cap_resid(m, ids, L)
        p = torch.softmax(lg, -1)
        ent = -(p * (p + 1e-12).log()).sum(-1)
        for t in range(1, H.shape[0] - 1):
            feats["entropy"].append(float(ent[t]))
            feats["position"].append(t)
            feats["logfreq"].append(math.log(cnt[ids[t]] + 1))
            feats["biasproj"].append(float(H[t] @ b_L))
            w = tok.decode([ids[t]]).strip().lower()
            feats["isfunc"].append(1 if w in FUNCTION_WORDS else 0)
            for j in massive.tolist():
                massvals[int(j)].append(float(H[t][j]))

    # PART 1 -- correlations
    print("PART 1 -- what does the backbone track? (Spearman)")
    print(f"  {'component':>12} {'entropy':>8} {'position':>9} {'logfreq':>8}")
    out = {"massive_dims": massive.tolist(), "correlations": {}}
    for j in massive.tolist():
        v = massvals[int(j)]
        ce = spearman(v, feats["entropy"])
        cp = spearman(v, feats["position"])
        cf = spearman(v, feats["logfreq"])
        out["correlations"][f"dim{j}"] = dict(entropy=round(ce, 3),
                                              position=round(cp, 3), logfreq=round(cf, 3))
        print(f"  {'dim'+str(j):>12} {ce:>+8.2f} {cp:>+9.2f} {cf:>+8.2f}")
    be = spearman(feats["biasproj"], feats["entropy"])
    bp = spearman(feats["biasproj"], feats["position"])
    bf = spearman(feats["biasproj"], feats["logfreq"])
    print(f"  {'bias-proj':>12} {be:>+8.2f} {bp:>+9.2f} {bf:>+8.2f}")
    out["correlations"]["bias_proj"] = dict(entropy=round(be, 3),
                                            position=round(bp, 3), logfreq=round(bf, 3))

    # PART 2 -- causal, context-resolved ablation
    print("\nPART 2 -- ablate the backbone (real residual); per-context effect")
    ent_t = np.array(feats["entropy"])
    lo, hi = np.percentile(ent_t, [33, 67])
    rows = {"dNLL": {"const_massive": [], "no_bias": []},
            "dEnt": {"const_massive": [], "no_bias": []},
            "top1_kept": {"const_massive": [], "no_bias": []},
            "bucket": [], "isfunc": [], "pos": []}
    for ids in idsets:
        H, real_lg = cap_resid(m, ids, L)
        rp = torch.softmax(real_lg, -1)
        rent = -(rp * (rp + 1e-12).log()).sum(-1)
        tgt = torch.tensor(ids[1:], device=DEV)
        rnll = -torch.log_softmax(real_lg[:-1], -1).gather(1, tgt[:, None]).squeeze(1)
        rtop1 = real_lg[:-1].argmax(-1)
        for name in ("const_massive", "no_bias"):
            Hh = H.clone()
            if name == "const_massive":
                Hh[1:, massive] = const_massive[massive]
            else:
                for t in range(1, H.shape[0]):
                    Hh[t] = Hh[t] - (Hh[t] @ b_L) * b_L
            lg = sub(m, ids, L, Hh)
            pp = torch.softmax(lg, -1)
            aent = -(pp * (pp + 1e-12).log()).sum(-1)
            anll = -torch.log_softmax(lg[:-1], -1).gather(1, tgt[:, None]).squeeze(1)
            atop1 = lg[:-1].argmax(-1)
            for t in range(1, H.shape[0] - 1):
                rows["dNLL"][name].append(float(anll[t] - rnll[t]))
                rows["dEnt"][name].append(float(aent[t] - rent[t]))
                rows["top1_kept"][name].append(float(atop1[t] == rtop1[t]))
        for t in range(1, H.shape[0] - 1):
            e = float(rent[t])
            rows["bucket"].append("lo" if e < lo else "hi" if e > hi else "mid")
            w = tok.decode([ids[t]]).strip().lower()
            rows["isfunc"].append(1 if w in FUNCTION_WORDS else 0)
            rows["pos"].append(t)

    def agg(name, metric, mask=None):
        v = np.array(rows[metric][name])
        if mask is not None:
            v = v[mask]
        return round(float(v.mean()), 3) if len(v) else float("nan")

    buckets = np.array(rows["bucket"]); isf = np.array(rows["isfunc"])
    print(f"  ablation      dNLL   dEntropy  top1-kept  | dNLL(loEnt/hiEnt) | dNLL(func/content)")
    out["ablation"] = {}
    for name in ("const_massive", "no_bias"):
        dn = agg(name, "dNLL"); de = agg(name, "dEnt"); tk = agg(name, "top1_kept")
        dn_lo = agg(name, "dNLL", buckets == "lo")
        dn_hi = agg(name, "dNLL", buckets == "hi")
        dn_f = agg(name, "dNLL", isf == 1)
        dn_c = agg(name, "dNLL", isf == 0)
        print(f"  {name:>13} {dn:>+6.3f} {de:>+8.3f} {tk:>9.2f}  | "
              f"{dn_lo:>+.3f}/{dn_hi:>+.3f} | {dn_f:>+.3f}/{dn_c:>+.3f}")
        out["ablation"][name] = dict(dNLL=dn, dEntropy=de, top1_kept=tk,
                                     dNLL_loEnt=dn_lo, dNLL_hiEnt=dn_hi,
                                     dNLL_func=dn_f, dNLL_content=dn_c)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
