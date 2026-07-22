"""
Factual-recall localization at scale (paper Section 4.3).

Scales the original "six country/capital pairs" to 20. Measures:
  (1) clean P(capital) for "The capital of <C> is" (should be high), and
  (2) the subject->END crossover layer via position-resolved patching:
      restore the clean residual at the subject or END position into a
      corrupt (different-country) run and measure recovery of P(capital);
      the crossover is the first layer where END-only recovery overtakes
      subject-only.

Requires single-token country and capital so the clean/corrupt prompts
differ in exactly one token.

Result on Phi-2 (2026-07-03): 20 usable pairs; clean P(capital) mean 0.877
(min 0.727, all >= 0.5); subject->END crossover at L22-L24 for every pair
(median L24).

Usage: .venv/Scripts/python.exe contrastive/code/factual_recall_scaleup.py
"""
import sys, os, statistics as st
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
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
layers = model.model.layers

SUBJ, END = 3, 4          # "The capital of <C> is" -> [The, capital, of, C, is]
PATCH_LAYERS = [14, 16, 18, 20, 22, 24, 26, 28]

CANDIDATES = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Russia", "Moscow"), ("Egypt", "Cairo"),
    ("China", "Beijing"), ("Cuba", "Havana"), ("Peru", "Lima"),
    ("Iran", "Tehran"), ("Poland", "Warsaw"), ("Greece", "Athens"),
    ("Germany", "Berlin"), ("Norway", "Oslo"), ("Iraq", "Baghdad"),
    ("Austria", "Vienna"), ("Portugal", "Lisbon"), ("Ireland", "Dublin"),
    ("Finland", "Helsinki"), ("Hungary", "Budapest"),
]


def tid(w):
    t = tok(" " + w, add_special_tokens=False)["input_ids"]
    return t[0] if len(t) == 1 else None


def toks(p):
    return tok(p, add_special_tokens=False)["input_ids"]


def prompt(c):
    return "The capital of " + c + " is"


def clean_P(c, k):
    with torch.no_grad():
        lg = model(torch.tensor([toks(prompt(c))], device=DEV)).logits[0, -1].float()
    return torch.softmax(lg, -1)[tid(k)].item()


def hidden(c):
    with torch.no_grad():
        o = model(torch.tensor([toks(prompt(c))], device=DEV),
                  output_hidden_states=True)
    return o.hidden_states


def patch_recovery(corrupt_c, clean_hs, pos, L, kid, Pc, Pk):
    def hook(m, i, o):
        oo = o[0].clone()
        oo[:, pos, :] = clean_hs[L][0, pos]
        return (oo,) + o[1:]
    h = layers[L - 1].register_forward_hook(hook)
    with torch.no_grad():
        lg = model(torch.tensor([toks(prompt(corrupt_c))], device=DEV)).logits[0, -1].float()
    h.remove()
    P = torch.softmax(lg, -1)[kid].item()
    return 100 * (P - Pc) / (Pk - Pc + 1e-9)


def main():
    pairs = [(c, k) for c, k in CANDIDATES
             if tid(c) and tid(k) and len(toks(prompt(c))) == 5]
    print(f"Model: {MODEL}  usable pairs: {len(pairs)}")

    Pcl = [clean_P(c, k) for c, k in pairs]
    print(f"clean P(capital): mean {sum(Pcl)/len(Pcl):.3f}, "
          f"min {min(Pcl):.3f}, frac>=0.5: {sum(x >= .5 for x in Pcl)}/{len(Pcl)}")

    cross = []
    for i, (c, k) in enumerate(pairs):
        c2, _ = pairs[(i + 1) % len(pairs)]          # corrupt country
        Hcl = hidden(c)
        kid = tid(k)
        with torch.no_grad():
            Pc = torch.softmax(model(torch.tensor([toks(prompt(c2))], device=DEV))
                               .logits[0, -1].float(), -1)[kid].item()
        Pk = Pcl[i]
        xover = None
        for L in PATCH_LAYERS:
            s = patch_recovery(c2, Hcl, SUBJ, L, kid, Pc, Pk)
            e = patch_recovery(c2, Hcl, END, L, kid, Pc, Pk)
            if e > s and xover is None:
                xover = L
        if xover:
            cross.append(xover)

    print(f"subject->END crossover across {len(cross)} pairs: "
          f"median L{st.median(cross):.0f}, range L{min(cross)}-L{max(cross)}")
    print("crossover layers:", sorted(cross))


if __name__ == "__main__":
    main()
