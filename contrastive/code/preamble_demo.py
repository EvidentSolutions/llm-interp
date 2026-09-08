"""
Prompt design and short-prompt instability (paper Sec 2.4, preamble).

Reads the SAME food contrast (hot vs cold dog) at the final token under
several prompt designs, and separately reads a pure-capitalization contrast,
to show (a) that a shared preamble stabilizes the read and (b) that at very
short prompts a semantically irrelevant change (capitalization) leaks into
the readout. Phi-2, fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/preamble_demo.py
"""
import sys, os, json
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"
FOOD = ["fried", "cooked", "delicious", "tasty", "crispy", "grilled", "edible",
        "flavor", "food", "sausage", "meat", "eaten", "juicy", "sauce", "bun",
        "ketchup", "snack", "vendor"]

DESIGNS = {
    "very short (Hot dog / Cold dog)": ("Hot dog", "Cold dog"),
    "bare matched (The hot / The cold)": ("The hot dog was", "The cold dog was"),
    "lowercased matched (the hot / the cold)": ("the hot dog was", "the cold dog was"),
    "MISMATCH (The hot / the cold)": ("The hot dog was", "the cold dog was"),
    "preamble (My grandmother said...)":
        ("My grandmother said that the hot dog was",
         "My grandmother said that the cold dog was"),
}
# the danger: a pair that differs ONLY in capitalization of the first word
CAP = ("The hot dog was", "the hot dog was")
LAYERS = [12, 16, 20, 24]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    W_U = m.lm_head.weight.detach().float()
    NL = m.config.num_hidden_layers

    def food_ids():
        ids = set()
        for w in FOOD:
            for c in (" " + w, w, " " + w.capitalize()):
                t = tok(c, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    ids.add(t[0])
        return ids
    FID = food_ids()

    def dh_final(a, b):
        ia = tok(a, add_special_tokens=False)["input_ids"]
        ib = tok(b, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            oa = m(torch.tensor([ia], device=DEV), output_hidden_states=True)
            ob = m(torch.tensor([ib], device=DEV), output_hidden_states=True)
        return [oa.hidden_states[L][0, -1] - ob.hidden_states[L][0, -1]
                for L in range(NL + 1)]

    def reads(v, k=8):
        i = torch.topk(v @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)

    def food_rank(v):
        order = torch.argsort(v @ W_U.T, descending=True).tolist()[:300]
        rr = [order.index(t) for t in FID if t in order]
        return min(rr) if rr else None

    results = {}
    print(f"{'='*95}\nFood contrast under prompt designs (top-8 at each layer; food rank in [])\n{'='*95}")
    dh_by_design = {}
    for name, (a, b) in DESIGNS.items():
        dh = dh_final(a, b)
        dh_by_design[name] = dh
        print(f"\n### {name}")
        rec = {}
        for L in LAYERS:
            fr = food_rank(dh[L])
            print(f"  L{L}: [{str(fr):>4}] {reads(dh[L])}")
            rec[L] = dict(food_rank=fr, top=reads(dh[L]))
        results[name] = rec

    # stability: cosine of the food direction across designs vs the preamble one
    print(f"\n{'='*95}\nStability: cosine of dh(L20) between designs and the preamble design\n{'='*95}")
    ref = dh_by_design["preamble (My grandmother said...)"][20]
    ref = ref / ref.norm()
    cos = {}
    for name, dh in dh_by_design.items():
        v = dh[20] / dh[20].norm()
        c = float(v @ ref)
        cos[name] = round(c, 3)
        print(f"  {name:<40} cos(L20, preamble) = {c:+.3f}")

    # the danger: capitalization-only contrast
    print(f"\n{'='*95}\nDANGER: a pair differing ONLY in first-word capitalization\n"
          f"  '{CAP[0]}'  vs  '{CAP[1]}'\n{'='*95}")
    dcap = dh_final(*CAP)
    caprec = {}
    for L in LAYERS:
        print(f"  L{L}: {reads(dcap[L])}   (||dh||={float(dcap[L].norm()):.1f})")
        caprec[L] = dict(top=reads(dcap[L]), norm=round(float(dcap[L].norm()), 1))

    out = dict(designs=results, cos_to_preamble=cos, capitalization=caprec)
    path = os.path.join(os.path.dirname(__file__), "..", "data", "preamble_demo.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
