"""
Apply the visibility threshold f* (Tuomi 2026) as a detection bar to the paper's
own contrastive readouts.

For a difference vector x and a token t with unit unembedding direction u_t, the
energy fraction f = cos^2(x, u_t) must exceed
    f* = 2 ln(2V) / (d + 2 ln(2V))
for t to be distinguishable from an isotropic bath in the top-k readout. The
top-k always returns k tokens, so the bar separates a genuine read (top token's
f >> f*) from soup (top token's f ~ f*, i.e. bath noise).

We report, per (example, layer): the top token the readout surfaces, its energy
fraction f, the bar f*, the margin f/f*, and a verdict. Two studies:
  A. Phi-2 "The hot dog was" - "The cold dog was" at the prediction site, per
     layer (does the mid-layer collapse fall below the bar?).
  B. Cross-model noun read, single pair vs triangulation (does triangulation
     push the noun read across the bar, as the paper claims by citation?).

Phi-2 / Pythia-1.4B / Qwen2.5-1.5B, fp32.
Usage: .venv/Scripts/python.exe public/contrastive/code/detection_bar.py
"""
import sys, os, json, math
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def wu(m):
    return m.get_output_embeddings().weight.detach().float()


def fstar(d, V):
    return 2 * math.log(2 * V) / (d + 2 * math.log(2 * V))


def hidden(m, tok, text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
    return ids, [o.hidden_states[L][0] for L in range(m.config.num_hidden_layers + 1)]


def noun_pos(tok, ids):
    toks = [tok.decode([x]) for x in ids]
    return max(i for i, t in enumerate(toks) if "dog" in t.lower())


def read_and_bar(x, Wn, Wu, tok, fs, k=4):
    """x: [d] difference vector. Wn: row-normalised W_U, Wu: raw W_U.
    Returns raw-logit top-k (what the paper prints) with each token's energy
    fraction f=cos^2, plus the max-cos token (what the bar is about). Verdict is
    by the max energy fraction over all tokens vs f*."""
    xn = x / x.norm()
    cos = Wn @ xn                      # cos(x, u_t) for every token
    logit = Wu @ x                     # raw logit-lens score (paper's readout)
    top = torch.topk(logit, k).indices
    items = [(tok.decode([int(t)]).strip()[:12], float(cos[t] ** 2)) for t in top]
    ci = int(torch.argmax(cos))        # most-aligned token (bar's readout)
    fmax = float(cos[ci] ** 2)
    return items, (tok.decode([ci]).strip()[:12], fmax), fmax / fs


def main():
    out = {}

    # ---- A. Phi-2 hot dog, prediction site, per layer ----
    tok, m = load("microsoft/phi-2")
    d, V = m.config.hidden_size, m.config.vocab_size
    fs = fstar(d, V)
    Wu = m.lm_head.weight.detach().float()
    Wn = Wu / Wu.norm(dim=1, keepdim=True)
    _, ht = hidden(m, tok, "The hot dog was")
    _, cd = hidden(m, tok, "The cold dog was")
    print(f"\n=== A. Phi-2 prediction site  (d={d}, V={V}, f*={fs*100:.2f}%) ===")
    print(f"{'L':>3} | {'raw-logit top-4 (token:f%)':52} | max-cos tok:f%   /f*  verdict")
    A = {}
    for L in [4, 6, 8, 10, 12, 16, 20, 24, 28]:
        x = ht[L][-1] - cd[L][-1]
        items, (cw, fmax), ratio = read_and_bar(x, Wn, Wu, tok, fs)
        s = "  ".join(f"{w}:{f*100:.2f}" for w, f in items)
        verdict = "VISIBLE" if ratio > 1 else "soup"
        print(f"{L:>3} | {s:52} | {cw}:{fmax*100:.2f}  {ratio:4.1f}x  {verdict}")
        A[L] = dict(raw_top=items, maxcos_tok=cw, f_max=round(fmax, 5),
                    ratio=round(ratio, 2), visible=ratio > 1)
    out["A_phi2_hotdog_predsite"] = dict(fstar=fs, layers=A)

    # ---- B. Cross-model noun read: single vs triangulation ----
    # target hot dog; single baseline cold dog; tri baselines below.
    TRI = ["cold", "angry", "old", "pet", "stray"]
    def crossmodel(name, layers):
        tk, mm = load(name)
        dd, VV = mm.config.hidden_size, mm.config.vocab_size
        f2 = fstar(dd, VV)
        W = wu(mm)
        Wnn = W / W.norm(dim=1, keepdim=True)
        idt, hts = hidden(mm, tk, "The hot dog was")
        p = noun_pos(tk, idt)
        bases = []
        for a in TRI:
            _, hb = hidden(mm, tk, f"The {a} dog was")
            bases.append(hb)
        res = {"fstar": f2, "layers": {}}
        print(f"\n=== B. {name}  noun position (d={dd}, V={VV}, f*={f2*100:.2f}%) ===")
        print(f"{'L':>3} | {'single max-cos:f%':26} r/f* | {'tri max-cos:f%':26} r/f*")
        for L in layers:
            xs = hts[L][p] - bases[0][L][p]
            xt = hts[L][p] - sum(b[L][p] for b in bases) / len(bases)
            (si, (scw, sf), sr) = read_and_bar(xs, Wnn, W, tk, f2, k=3)
            (ti, (tcw, tf), tr) = read_and_bar(xt, Wnn, W, tk, f2, k=3)
            print(f"{L:>3} | {scw+':'+format(sf*100,'.2f'):26} {sr:4.1f} | "
                  f"{tcw+':'+format(tf*100,'.2f'):26} {tr:4.1f}")
            res["layers"][L] = dict(single_maxcos=scw, single_f=round(sf, 5),
                                    single_ratio=round(sr, 2),
                                    tri_maxcos=tcw, tri_f=round(tf, 5),
                                    tri_ratio=round(tr, 2))
        del mm
        torch.cuda.empty_cache()
        return res

    out["B_phi2"] = crossmodel("microsoft/phi-2", [4, 6, 8])
    out["B_pythia"] = crossmodel("EleutherAI/pythia-1.4b", [12, 15, 17, 20])
    out["B_qwen"] = crossmodel("Qwen/Qwen2.5-1.5B", [12, 16, 20])

    path = os.path.join(os.path.dirname(__file__), "..", "data", "detection_bar.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
