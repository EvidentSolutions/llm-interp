"""
Worked example: decomposing a contrast into concept axes and testing each for
causality (paper Sec: "using a contrast").

The single pair "The hot dog was" - "The cold dog was" is not one axis. We build
three data-derived concept directions at the read position ("was", "The X was"):
  edibility  = mean(edible nouns)  - mean(objects)
  petness    = mean(pet animals)   - mean(objects)
  temperature= mean("hot X")       - mean("cold X")   over neutral nouns
Project the hot-cold difference onto each (it has components on all three), then
inject each component (and the joint planes) into the cold-dog run at "was" and
measure how far the next-token prediction moves from animal toward food.

food fraction = sum P(food tokens) / (sum P(food) + sum P(animal)) at the next
token; baseline cold ~ 0, hot ~ 1. Phi-2, fp32.

Usage: .venv/Scripts/python.exe public/contrastive/code/hotdog_concept_axes.py
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
EDIBLE = ["steak", "sausage", "burger", "bacon", "pizza", "sandwich", "cutlet", "omelet"]
OBJECTS = ["rock", "brick", "chair", "table", "box", "stone", "lamp", "clock"]
PETS = ["puppy", "kitten", "poodle", "hamster", "parrot", "rabbit", "kitten", "beagle"]
NEUTRAL = ["soup", "coffee", "plate", "drink", "morning", "water", "room", "towel"]
FOOD = ["cooked", "delicious", "tasty", "fried", "grilled", "eaten", "edible",
        "sausage", "meat", "bun", "yummy", "spicy", "served", "juicy"]
ANIMAL = ["shivering", "shaking", "barking", "whimpering", "wagging", "panting",
          "scared", "trembling", "growling", "sniffing", "shy", "nervous"]
LAYERS = [8, 12, 16, 20, 24]


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    NL = m.config.num_hidden_layers

    def ids_of(words):
        out = set()
        for w in words:
            for c in (" " + w, w):
                t = tok(c, add_special_tokens=False)["input_ids"]
                if len(t) == 1:
                    out.add(t[0])
        return list(out)
    FID, AID = ids_of(FOOD), ids_of(ANIMAL)

    def final_hidden(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [o.hidden_states[L][0, -1] for L in range(NL + 1)]

    def mean_hidden(nouns, tmpl="The %s was"):
        acc = None
        for n in nouns:
            h = final_hidden(tmpl % n)
            acc = h if acc is None else [a + b for a, b in zip(acc, h)]
        return [a / len(nouns) for a in acc]

    obj = mean_hidden(OBJECTS)
    edible = mean_hidden(EDIBLE)
    pets = mean_hidden(PETS)
    hotN = mean_hidden(NEUTRAL, "The hot %s was")
    coldN = mean_hidden(NEUTRAL, "The cold %s was")
    E = [edible[L] - obj[L] for L in range(NL + 1)]
    P = [pets[L] - obj[L] for L in range(NL + 1)]
    T = [hotN[L] - coldN[L] for L in range(NL + 1)]

    hot = final_hidden("The hot dog was")
    cold = final_hidden("The cold dog was")

    def unit(v):
        return v / v.norm()

    def food_frac_from_logits(logits):
        p = torch.softmax(logits.float(), -1)
        f = float(p[FID].sum()); a = float(p[AID].sum())
        return f / (f + a) if (f + a) > 0 else 0.0

    cold_ids = tok("The cold dog was", add_special_tokens=False)["input_ids"]

    def inject(delta, L):
        done = [False]
        def hook(mod, i, o):
            if done[0]:
                return o
            done[0] = True
            if isinstance(o, tuple):
                h = o[0].clone(); h[0, -1] += delta; return (h,) + o[1:]
            h = o.clone(); h[0, -1] += delta; return h
        layer = m.model.layers[L] if L < NL else m.model.final_layernorm
        hd = layer.register_forward_hook(hook)
        with torch.no_grad():
            out = m(torch.tensor([cold_ids], device=DEV))
        hd.remove()
        return food_frac_from_logits(out.logits[0, -1])

    def plane_comp(d, basis):
        M = torch.stack([unit(b) for b in basis], 1)
        Q, _ = torch.linalg.qr(M)
        return Q @ (Q.T @ d)

    base_cold = food_frac_from_logits(
        m(torch.tensor([cold_ids], device=DEV)).logits[0, -1])
    base_hot = food_frac_from_logits(
        m(torch.tensor([tok("The hot dog was", add_special_tokens=False)["input_ids"]],
          device=DEV)).logits[0, -1])
    print(f"food fraction baseline: cold={base_cold:.2f}  hot={base_hot:.2f}\n")

    out = {"baseline": {"cold": round(base_cold, 3), "hot": round(base_hot, 3)},
           "layers": {}}
    print(f"{'L':>3} | {'projE':>6} {'projP':>6} {'projT':>6} | "
          f"{'injE':>5} {'injP':>5} {'injT':>5} {'injEP':>6} {'injEPT':>6} {'injFull':>7}")
    for L in LAYERS:
        d = hot[L] - cold[L]
        pe = float(d @ unit(E[L])); pp = float(d @ unit(P[L])); pt = float(d @ unit(T[L]))
        injE = inject(pe * unit(E[L]), L)
        injP = inject(pp * unit(P[L]), L)
        injT = inject(pt * unit(T[L]), L)
        injEP = inject(plane_comp(d, [E[L], P[L]]), L)
        injEPT = inject(plane_comp(d, [E[L], P[L], T[L]]), L)
        injFull = inject(d, L)
        print(f"{L:>3} | {pe:>+6.1f} {pp:>+6.1f} {pt:>+6.1f} | "
              f"{injE:>5.2f} {injP:>5.2f} {injT:>5.2f} {injEP:>6.2f} {injEPT:>6.2f} {injFull:>7.2f}")
        out["layers"][L] = dict(projE=round(pe, 1), projP=round(pp, 1), projT=round(pt, 1),
                                injE=round(injE, 3), injP=round(injP, 3), injT=round(injT, 3),
                                injEP=round(injEP, 3), injEPT=round(injEPT, 3),
                                injFull=round(injFull, 3))
    # axis correlation and what each axis reads at L20
    L = 20
    ce = float(torch.nn.functional.cosine_similarity(E[L][None], P[L][None]))
    out["cosEP_L20"] = round(ce, 2)
    W_U = m.lm_head.weight.detach().float()

    def rd(v, k=4):
        i = torch.topk(v @ W_U.T, k).indices
        return ", ".join(tok.decode([int(x)]).strip()[:10] for x in i)
    out["axis_reads_L20"] = {"edibility": rd(E[L]), "petness": rd(P[L]),
                             "temperature": rd(T[L])}
    print(f"\ncos(edibility, petness) at L20 = {ce:+.2f}")
    for k, v in out["axis_reads_L20"].items():
        print(f"  {k:12} reads: {v}")
    path = os.path.join(os.path.dirname(__file__), "..", "data", "hotdog_concept_axes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"Saved {path}\nDONE")


if __name__ == "__main__":
    main()
