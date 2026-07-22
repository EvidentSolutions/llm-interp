"""
Compound-noun circuit: neuron- and head-level mechanism (paper Section 4.1).

Establishes that the readable food content of "hot dog" is genuinely computed,
by a DISTRIBUTED MLP write at the "dog" position plus a LOCALIZED routing head
to "was" -- not by any single neuron. Run in the stable (preamble) regime.

Part A -- MLP write at "dog" (L6-12). Decompose the MLP-output contrast
  (hot dog vs cold dog) neuron by neuron: each neuron's push along the food
  direction is dgate_n * (fc2[:,n] . food_dir). Reports what fraction the top
  neurons carry (distribution) and the top food-writers / animal-suppressors.

Part B -- routing head to "was". Decompose the attention output at "was" per
  head; report heads whose per-head output delta aligns with the food
  direction, together with their attention weight onto "dog".

Observed (Phi-2, 2026-07-04, preamble "Everyone agreed that "):
  A: top-10 neurons carry ~8-12% of the food push (spread over hundreds);
     food-writers e.g. N1958 "cooked, foods, dishes" (L8),
     N8098 "breakfast, lunch, eaten" (L12); animal-suppressor N6665
     "dogs, puppies" (L12).
  B: L5.H19 attends "was"->"dog" at ~0.85 and writes "cooked, eaten"
     (the paper's routing head); later heads (L15-17) refine.

Usage: .venv/Scripts/python.exe contrastive/code/compound_noun_mechanism.py
"""
import sys, os
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
PRE = "Everyone agreed that "
FOOD = ["fried", "cooked", "delicious", "tasty", "crispy", "grilled", "edible", "flavor"]


def main():
    # eager attention so output_attentions returns weights (Part B)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager").to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(MODEL)
    layers = model.model.layers
    W_U = model.lm_head.weight.detach().float().cpu()
    H = model.config.num_attention_heads
    DH = model.config.hidden_size // H

    def toks(p):
        return tok(p, add_special_tokens=False)["input_ids"]

    def sid(w):
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        return t[0] if len(t) == 1 else None

    def tk(v, k=6):
        i = torch.topk(v, k).indices
        return " ".join(tok.decode([int(x)]).strip()[:8] for x in i)

    hp, cp = PRE + "the hot dog was", PRE + "the cold dog was"
    DOG = toks(hp).index(sid("dog"))
    WAS = len(toks(hp)) - 1
    food = [x for x in (sid(w) for w in FOOD) if x is not None]
    fd = W_U[food].mean(0)
    fd = fd / fd.norm()

    # ---------- Part A: MLP write at dog ----------
    def pregelu(p, L):
        st = {}
        h = layers[L].mlp.fc1.register_forward_hook(
            lambda m, i, o: st.__setitem__("x", o[0, DOG].float().cpu()))
        with torch.no_grad():
            model(torch.tensor([toks(p)], device=DEV))
        h.remove()
        return st["x"]

    print(f"Model {MODEL}   preamble {PRE!r}   dog@{DOG} was@{WAS}\n")
    print("=== A. MLP write at 'dog' (distributed) ===")
    for L in [6, 8, 10, 12]:
        fc2 = layers[L].mlp.fc2.weight.detach().float().cpu()   # [d_model, d_ff]
        dg = F.gelu(pregelu(hp, L)) - F.gelu(pregelu(cp, L))
        contrib = dg[:, None] * fc2.T                           # per-neuron residual push
        fc = contrib @ fd
        order = torch.argsort(fc, descending=True)
        tot = fc[fc > 0].sum().item()
        t10 = 100 * fc[order[:10]].sum().item() / tot
        t30 = 100 * fc[order[:30]].sum().item() / tot
        print(f"L{L:>2}: top-10 carry {t10:.0f}% / top-30 {t30:.0f}% of food push")
        for n in order[:3]:
            n = int(n)
            print(f"     N{n}: push={fc[n]:+.2f}  fc2=[{tk(fc2[:, n] @ W_U.T)}]")

    # ---------- Part B: routing heads to was ----------
    def attn_out_and_pattern(p):
        ao = {}
        hs = [layers[L].self_attn.dense.register_forward_pre_hook(
                (lambda L: lambda m, x: ao.__setitem__(L, x[0][0, WAS].float().cpu()))(L))
              for L in range(len(layers))]
        with torch.no_grad():
            out = model(torch.tensor([toks(p)], device=DEV), output_attentions=True)
        for h in hs:
            h.remove()
        return ao, out.attentions

    aoH, attH = attn_out_and_pattern(hp)
    aoC, _ = attn_out_and_pattern(cp)

    rows = []
    for L in range(4, 18):
        Wd = layers[L].self_attn.dense.weight.detach().float().cpu()
        for h in range(H):
            sl = slice(h * DH, (h + 1) * DH)
            dhead = (aoH[L][sl] - aoC[L][sl]) @ Wd[:, sl].T     # per-head output delta at was
            rows.append((float(dhead @ fd), L, h, float(attH[L][0, h, WAS, DOG]), dhead))
    rows.sort(key=lambda r: -r[0])
    print("\n=== B. Routing heads to 'was' (food-writing + attention to 'dog') ===")
    print(f"  {'head':<8} {'food':>6} {'attn->dog':>10}  reads")
    for fs, L, h, a2d, dhead in rows[:6]:
        print(f"  L{L}.H{h:<3} {fs:>+6.2f} {a2d:>10.2f}  [{tk(dhead @ W_U.T, 5)}]")


if __name__ == "__main__":
    main()
