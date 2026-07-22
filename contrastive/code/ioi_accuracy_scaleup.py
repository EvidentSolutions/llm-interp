"""
IOI behavioral accuracy at scale (paper Section 4.2 / IOI).

Scales the original "30/30" claim to a larger templated set. For each
(name pair, template), builds an IOI prompt where the subject is repeated
and the indirect object is the other name, and checks logit(IO) > logit(S).

Result on Phi-2 (2026-07-03): 160/160 = 100.0% over 40 name pairs x 4
templates; mean logit(IO)-logit(S) = 6.43 (min 2.44, always positive).

Usage: .venv/Scripts/python.exe contrastive/code/ioi_accuracy_scaleup.py
       MODEL env var overrides the model (default microsoft/phi-2).
"""
import sys, os, itertools, statistics as st
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)


def tid(w):
    t = tok(" " + w, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None


def toks(p):
    return tok(p, add_special_tokens=False)["input_ids"]


NAME_POOL = ["John", "Mary", "Tom", "Sarah", "James", "Anna", "Paul", "Emma",
             "Mark", "Lucy", "David", "Kate", "Peter", "Laura", "Mike", "Susan"]
names = [n for n in NAME_POOL if tid(n)]

TEMPLATES = [
    "{A} and {B} went to the store. {A} gave a book to",
    "{A} and {B} were at the park. {A} handed the ball to",
    "When {A} and {B} arrived, {A} passed the keys to",
    "{A} and {B} had lunch together. {A} gave the menu to",
]


def last_logits(p):
    ids = toks(p)
    with torch.no_grad():
        return model(torch.tensor([ids], device=DEV)).logits[0, -1].float()


def main():
    print(f"Model: {MODEL}  single-token names: {len(names)}")
    pairs = [(a, b) for a, b in itertools.permutations(names, 2)]
    # deterministic sample (no RNG) to ~40 pairs
    sel = pairs[:: len(pairs) // 40][:40]

    correct, tot, diffs = 0, 0, []
    for tmpl in TEMPLATES:
        for A, B in sel:  # subject A repeated; IO = B
            lg = last_logits(tmpl.format(A=A, B=B))
            d = lg[tid(B)].item() - lg[tid(A)].item()
            diffs.append(d)
            correct += (d > 0)
            tot += 1

    print(f"IOI accuracy (logit(IO)>logit(S)): {correct}/{tot} = "
          f"{100 * correct / tot:.1f}%")
    print(f"mean logit(IO)-logit(S) = {st.mean(diffs):.2f} (min {min(diffs):.2f})")
    print(f"N = {len(TEMPLATES)} templates x {len(sel)} name pairs")


if __name__ == "__main__":
    main()
