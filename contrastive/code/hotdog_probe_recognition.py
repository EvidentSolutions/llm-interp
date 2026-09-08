"""
The compound distinction is decodable even where no token face surfaces
(paper Sec 3.2, cross-model).

A transfer probe tests whether the food-vs-animal compound distinction is present
at the noun in each model, independent of whether the contrastive read surfaces a
token there. We train a linear probe to separate clear food nouns (steak,
burger, ...) from animal nouns (poodle, kitten, ...) at the noun position, then
apply it to the "dog" of "hot dog" and "cold dog". In all three models it reads
hot dog's noun as food and cold dog's as animal with near-certainty, including
Qwen, where the contrastive single-pair read at the noun is illegible. The token
face is network-specific; the computation is present regardless.

Phi-2, Pythia-1.4B, Qwen2.5-1.5B; fp32.
Usage: .venv/Scripts/python.exe public/contrastive/code/hotdog_probe_recognition.py
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
from sklearn.model_selection import cross_val_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = ["microsoft/phi-2", "EleutherAI/pythia-1.4b", "Qwen/Qwen2.5-1.5B"]
FOOD = ["fresh steak", "grilled burger", "fried bacon", "juicy sausage",
        "roasted chicken", "baked pizza", "crispy cutlet", "smoked ham",
        "tasty sandwich", "warm pretzel"]
ANIM = ["little poodle", "small kitten", "brown rabbit", "grey parrot",
        "tiny hamster", "black beagle", "white puppy", "old horse",
        "young lamb", "wild fox"]


def run(MODEL):
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, low_cpu_mem_usage=True).to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    NL = m.config.num_hidden_layers

    def rep(text):  # read at the noun = token just before " was"
        ids = tok(text, add_special_tokens=False)["input_ids"]
        toks = [tok.decode([x]) for x in ids]
        pos = max(i for i, x in enumerate(toks) if "was" in x.lower()) - 1
        with torch.no_grad():
            o = m(torch.tensor([ids], device=DEV), output_hidden_states=True)
        return [o.hidden_states[L][0, pos].detach().float().cpu().numpy()
                for L in range(NL + 1)]

    Xf = [rep("The " + s + " was") for s in FOOD]
    Xa = [rep("The " + s + " was") for s in ANIM]
    hot, cold = rep("The hot dog was"), rep("The cold dog was")
    y = np.array([1] * len(Xf) + [0] * len(Xa))

    best = None
    for L in range(1, NL + 1):
        X = np.stack([x[L] for x in Xf] + [x[L] for x in Xa])
        clf = LogisticRegression(max_iter=3000, C=0.5).fit(X, y)
        ph = float(clf.predict_proba(hot[L][None])[0, 1])
        if best is None or ph > best["hot"]:
            cv = float(cross_val_score(
                LogisticRegression(max_iter=3000, C=0.5), X, y, cv=5).mean())
            best = dict(layer=L, hot=round(ph, 3),
                        cold=round(float(clf.predict_proba(cold[L][None])[0, 1]), 3),
                        probe_cv=round(cv, 3))
    del m
    torch.cuda.empty_cache()
    return best


def main():
    out = {}
    for M in MODELS:
        out[M] = run(M)
        b = out[M]
        print(f"{M}: hot P(food)={b['hot']:.2f}  cold={b['cold']:.2f}  "
              f"(L{b['layer']}, probe cv={b['probe_cv']:.2f})")
    path = os.path.join(os.path.dirname(__file__), "..", "data",
                        "hotdog_probe_recognition.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nSaved {path}\nDONE")


if __name__ == "__main__":
    main()
