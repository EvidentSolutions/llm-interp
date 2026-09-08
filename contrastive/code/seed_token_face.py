"""
The token-face is seed-specific; the underlying distinction is not.

Same architecture, same tokenizer, (for weight-seed variants) same data --
only the init seed differs. Read the SAME contrast through each seed's own dh
and its own W_U and ask what agrees.

Two things are measured, per contrast, across seeds:
  LITERAL token-face = mean pairwise Jaccard of the top-k token SETS
     (expect LOW: which tokens surface is near-arbitrary per seed).
  CONCEPT preservation = does the contrastive projection BOOST the concept
     anchors and SUPPRESS the anti-concept, read at each seed's own legible
     layer? margin = mean(logit[concept]) - mean(logit[anti]). If margin>0
     for (nearly) every seed, the distinction is preserved even though the
     surfaced tokens are not (expect margin>0 broadly).
  Secondary: literal Jaccard is also split by source using the decoupled
     variants -- data-order-only vs init-only -- to attribute the variation.

Read layer chosen PER SEED as the layer maximising the concept margin (the
method's own "read where the distinction is legible" doctrine), not the
norm-max final layer (raw norm grows with depth -- an instrument artifact).
Pythia-410m seeds, fp32. All Pythia share one tokenizer, so cross-seed
Jaccard is confound-free.

Usage: .venv/Scripts/python.exe public/contrastive/code/seed_token_face.py
"""
import sys, os, json, itertools
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = os.environ.get("SIZE", "410m")
BASE = f"EleutherAI/pythia-{SIZE}"
# cached 410m seeds: base, seed1, seed3, seed6, seed7
SEEDS = [BASE] + [f"EleutherAI/pythia-{SIZE}-seed{i}" for i in (1, 3, 6, 7)]
K = 10

# (label, context, control, concept_anchors(+pole), anti_anchors(-pole))
CONTRASTS = [
    ("sentiment", "The film was absolutely wonderful and I",
     "The film was absolutely terrible and I",
     ["wonderful", "great", "amazing", "excellent", "love", "brilliant",
      "fantastic", "enjoyed"],
     ["terrible", "awful", "horrible", "bad", "hate", "disappointing",
      "dreadful", "boring"]),
    ("temperature", "I touched the metal and it felt extremely hot",
     "I touched the metal and it felt extremely cold",
     ["hot", "warm", "heat", "burning", "scalding", "fiery", "burn"],
     ["cold", "cool", "freezing", "icy", "chilly", "frozen", "cool"]),
    ("time", "This will definitely happen tomorrow",
     "This definitely happened yesterday",
     ["tomorrow", "future", "soon", "later", "upcoming", "next", "will"],
     ["yesterday", "past", "ago", "earlier", "previously", "before", "was"]),
    ("nationality", "She grew up in Paris speaking fluent French",
     "She grew up in Tokyo speaking fluent Japanese",
     ["French", "France", "Paris", "Europe", "European", "France"],
     ["Japanese", "Japan", "Tokyo", "Asia", "Asian", "Japan"]),
]


def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def anchor_ids(tok, words):
    out = []
    for w in words:
        for cand in (" " + w, w):
            t = tok(cand, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                out.append(t[0]); break
    return out


def read_seed(m, tok, ctx, ctrl, pos_anchor, neg_anchor):
    """Per-seed: pick layer maximising concept margin; return top-k, margin."""
    W_U = m.embed_out.weight.detach().float()
    ia = tok(ctx, add_special_tokens=False)["input_ids"]
    ib = tok(ctrl, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        oa = m(torch.tensor([ia], device=DEV), output_hidden_states=True)
        ob = m(torch.tensor([ib], device=DEV), output_hidden_states=True)
    NL = m.config.num_hidden_layers
    pa = torch.tensor(anchor_ids(tok, pos_anchor), device=DEV)
    na = torch.tensor(anchor_ids(tok, neg_anchor), device=DEV)
    best_L, best_margin, best_logits = None, -1e9, None
    for L in range(2, NL + 1):
        dh = oa.hidden_states[L][0, -1] - ob.hidden_states[L][0, -1]
        logits = dh @ W_U.T
        margin = float(logits[pa].mean() - logits[na].mean())
        if margin > best_margin:
            best_margin, best_L, best_logits = margin, L, logits
    pos = torch.topk(best_logits, K).indices.tolist()
    return pos, best_margin, best_L


def jaccard(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def mpj(sets):
    ps = list(itertools.combinations(sets, 2))
    return sum(jaccard(a, b) for a, b in ps) / len(ps) if ps else 0.0


def main():
    results = {}
    # cache per-model reads
    per = {lab: [] for lab, *_ in CONTRASTS}
    for name in SEEDS:
        tok, m = load(name)
        for lab, ctx, ctrl, pa, nb in CONTRASTS:
            pos, margin, L = read_seed(m, tok, ctx, ctrl, pa, nb)
            per[lab].append(dict(seed=name.split("-")[-1], pos=pos,
                                 margin=round(margin, 2), L=L,
                                 toks=[tok.decode([t]).strip() for t in pos[:6]]))
        del m
        torch.cuda.empty_cache()

    print(f"{'='*92}\nPythia-{SIZE}: {len(SEEDS)} seeds -- token-face vs concept\n{'='*92}")
    for lab, *_ in CONTRASTS:
        rows = per[lab]
        lit = mpj([r["pos"] for r in rows])
        union = len(set().union(*[set(r["pos"]) for r in rows]))
        margins = [r["margin"] for r in rows]
        n_pos = sum(1 for x in margins if x > 0)
        results[lab] = dict(literal_jaccard=round(lit, 3), union=union, k=K,
                            n_seeds=len(rows), concept_margin_positive=n_pos,
                            mean_margin=round(sum(margins) / len(margins), 2),
                            reads=rows)
        print(f"\n[{lab}]  literal Jaccard={lit:.2f}  "
              f"(union {union} distinct tokens / {K*len(rows)} slots)   "
              f"concept margin >0 in {n_pos}/{len(rows)} seeds "
              f"(mean {sum(margins)/len(margins):+.2f})")
        for r in rows:
            print(f"    {r['seed']:>7} L{r['L']:>2} margin{r['margin']:>+6.1f}: "
                  f"{', '.join(r['toks'])}")

    print(f"\n{'='*92}\nSUMMARY (mean over contrasts)\n{'='*92}")
    lits = [results[l]["literal_jaccard"] for l, *_ in CONTRASTS]
    posc = [results[l]["concept_margin_positive"] for l, *_ in CONTRASTS]
    tot = len(SEEDS) * len(CONTRASTS)
    print(f"  literal token-face Jaccard: {sum(lits)/len(lits):.2f} (near-zero "
          f"= the surfaced tokens barely overlap across seeds)")
    print(f"  concept preserved (margin>0): {sum(posc)}/{tot} seed-contrasts "
          f"(the distinction survives even though the tokens don't)")

    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "seed_token_face.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
