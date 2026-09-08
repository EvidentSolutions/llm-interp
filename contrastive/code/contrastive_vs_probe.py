"""
Contrastive projection vs a trained linear probe (paper: probing related work).

Concept: sentiment (positive vs negative), read at a NEUTRAL final token so the
sentiment must be carried in the representation, not read off the word itself.
Frame: "The <subject> was absolutely <adj> and I"  -> read at "I".

For each layer we compare, on a held-out split:
  - accuracy of the CONTRASTIVE direction (mean_pos - mean_neg), used as a
    classifier (sign of projection past the train midpoint);
  - accuracy of a TRAINED probe (logistic regression on hidden states);
  - what each direction reads through W_U (does a sentiment token surface?).

The question: is the probe more accurate/sensitive, and does it recover the
concept where the contrastive top-k readout does not (e.g. at early layers)?
Phi-2, fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/contrastive_vs_probe.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
SUBJ = ["film", "book", "meal", "trip", "show", "concert", "hotel", "restaurant",
        "painting", "novel", "lecture", "game", "album", "speech", "party"]
POS = ["wonderful", "great", "amazing", "fantastic", "brilliant", "excellent",
       "superb", "delightful", "marvelous", "outstanding"]
NEG = ["terrible", "awful", "horrible", "dreadful", "atrocious", "dismal",
       "appalling", "lousy", "miserable", "disappointing"]
LAYERS = [4, 12, 20, 28]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float().cpu().numpy()
    NL = m.config.num_hidden_layers

    texts, ys = [], []
    for s in SUBJ:
        for a in POS:
            texts.append(f"The {s} was absolutely {a} and I"); ys.append(1)
        for a in NEG:
            texts.append(f"The {s} was absolutely {a} and I"); ys.append(0)
    ys = np.array(ys)

    # hidden states at final token for every layer
    H = {L: [] for L in LAYERS}
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            out = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        for L in LAYERS:
            H[L].append(out.hidden_states[L][0, -1].float().cpu().numpy())
    H = {L: np.stack(v) for L, v in H.items()}

    def reads(vec, k=4):
        s = vec @ W_U.T
        pos = ", ".join(tok.decode([int(i)]).strip()[:10] for i in np.argsort(-s)[:k])
        neg = ", ".join(tok.decode([int(i)]).strip()[:10] for i in np.argsort(s)[:k])
        return pos, neg

    idx_tr, idx_te = train_test_split(np.arange(len(ys)), test_size=0.3,
                                      random_state=0, stratify=ys)
    out = {}
    print(f"{'L':>3} | {'contrast acc':>12} {'probe acc':>10} | direction reads (+ pole / - pole)")
    for L in LAYERS:
        X, y = H[L], ys
        Xtr, Xte, ytr, yte = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
        # contrastive direction = mean(pos) - mean(neg) on train
        cdir = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        proj_tr = Xtr @ cdir
        thr = 0.5 * (proj_tr[ytr == 1].mean() + proj_tr[ytr == 0].mean())
        cacc = float(((Xte @ cdir > thr).astype(int) == yte).mean())
        # trained probe
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
        pacc = float(clf.score(Xte, yte))
        cp, cn = reads(cdir)
        pp, pn = reads(clf.coef_[0])
        print(f"{L:>3} | {cacc:>11.2f} {pacc:>10.2f} | contrast +[{cp}] -[{cn}]")
        print(f"    |                          | probe    +[{pp}] -[{pn}]")
        out[L] = dict(contrast_acc=round(cacc, 3), probe_acc=round(pacc, 3),
                      contrast_reads=[cp, cn], probe_reads=[pp, pn])

    path = os.path.join(os.path.dirname(__file__), "..", "data", "contrastive_vs_probe.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
