"""
Activation patching of the hot-dog prompt (paper Sec 3.1), with greedy readout.

Replace the residual at one position and one layer of "The hot dog was" with the
corresponding "The cold dog was" value, then let generation continue greedily.
P(food) is the food mass at the next token. This traces the two-hop chain:
"dog" is causal early (patching it there yields animal continuations), "was"
is causal late, and "hot" never matters (L0 attention copied it to "dog").

Phi-2, fp32.
Usage: .venv/Scripts/python.exe public/contrastive/code/hotdog_patch.py
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
HOT, COLD = "The hot dog was", "The cold dog was"   # The(0) hot/cold(1) dog(2) was(3)
FOOD = ["cooked", "delicious", "tasty", "fried", "grilled", "eaten", "edible",
        "meat", "sausage", "bun", "yummy", "spicy", "served", "juicy"]
# (position, layer) patches; position 1=hot, 2=dog, 3=was
PATCHES = [("baseline", None), ("dog", (2, 2)), ("dog", (2, 8)), ("dog", (2, 20)),
           ("hot", (1, 2)), ("was", (3, 2)), ("was", (3, 12)), ("was", (3, 20))]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    fid = set()
    for w in FOOD:
        for c in (" " + w, w):
            t = tok(c, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                fid.add(t[0])
    fid = list(fid)

    cold_ids = tok(COLD, add_special_tokens=False)["input_ids"]
    cold_hs = m(torch.tensor([cold_ids], device=DEV),
                output_hidden_states=True).hidden_states

    def run(patch, n=6):
        ids = tok(HOT, add_special_tokens=False)["input_ids"][:]
        hnd = None
        if patch:
            pos, L = patch
            cv = cold_hs[L + 1][0, pos].clone()
            def hook(mod, i, o):
                if isinstance(o, tuple):
                    o[0][0, pos] = cv; return o
                o[0, pos] = cv; return o
            hnd = m.model.layers[L].register_forward_hook(hook)
        pfood = None
        for step in range(n):
            with torch.no_grad():
                lo = m(torch.tensor([ids], device=DEV)).logits[0, -1]
            if step == 0:
                pfood = float(torch.softmax(lo.float(), -1)[fid].sum())
            ids.append(int(lo.argmax()))
        if hnd:
            hnd.remove()
        return pfood, tok.decode(ids[4:]).strip()

    out = {}
    print(f"{'patch':>10} {'layer':>5} {'P(food)':>8}  greedy continuation")
    for name, patch in PATCHES:
        pf, gen = run(patch)
        key = name if patch is None else f"{name}@L{patch[1]}"
        out[key] = dict(pos=None if patch is None else patch[0],
                        layer=None if patch is None else patch[1],
                        pfood=round(pf, 3), greedy=gen)
        L = "" if patch is None else patch[1]
        print(f"{name:>10} {str(L):>5} {pf:>8.3f}  {gen!r}")

    path = os.path.join(os.path.dirname(__file__), "..", "data", "hotdog_patch.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
