"""
Truth-direction consistency and bidirectional causality at scale
(paper Section 3.2).

Scales the original five fact pairs to 15. For each (true, false) fact,
the truth direction is h[true]-h[false] at the final position of
"<fact>. This statement is" (layer L=20). Measures:
  (1) mean pairwise cosine across the 15 directions (consistency), and
  (2) leave-one-out bidirectional causality: extract the direction from
      the other 14 pairs, inject +dir into the held-out FALSE statement
      (should flip the verdict to "true") and -dir into the held-out TRUE
      statement (should flip to "false").

Result on Phi-2 (2026-07-03): mean pairwise cos 0.36; at 2x natural
magnitude all 15 false->true and 15 true->false flips succeed (10/15 and
7/15 at 1x). LOO => held-out, not circular.

Usage: .venv/Scripts/python.exe contrastive/code/truth_direction_scaleup.py
"""
import sys, os, itertools
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
L = 20
SCALES = [1.0, 2.0]

model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)
tok = AutoTokenizer.from_pretrained(MODEL)
layers = model.model.layers

TRUE = tok(" true", add_special_tokens=False)["input_ids"][0]
FALSE = tok(" false", add_special_tokens=False)["input_ids"][0]

PAIRS = [
    ("Dogs are mammals", "Dogs are reptiles"),
    ("The Sun is a star", "The Sun is a planet"),
    ("Ice is cold", "Ice is hot"),
    ("Fish live in water", "Fish live in fire"),
    ("Gold is a metal", "Gold is a gas"),
    ("Birds have feathers", "Birds have scales"),
    ("The Earth is round", "The Earth is flat"),
    ("Lions are animals", "Lions are plants"),
    ("Snow is white", "Snow is black"),
    ("Fire is hot", "Fire is cold"),
    ("Humans breathe air", "Humans breathe water"),
    ("Iron is a metal", "Iron is a liquid"),
    ("Cats are animals", "Cats are vegetables"),
    ("The ocean is salty", "The ocean is sweet"),
    ("Bees make honey", "Bees make milk"),
]


def toks(p):
    return tok(p, add_special_tokens=False)["input_ids"]


def last_vec(statement):
    p = statement + ". This statement is"
    with torch.no_grad():
        o = model(torch.tensor([toks(p)], device=DEV), output_hidden_states=True)
    return o.hidden_states[L][0, -1].float()


def predicts_true(statement, d, sign):
    p = statement + ". This statement is"

    def hook(m, i, o):
        oo = o[0].clone()
        oo[:, -1, :] = oo[:, -1, :] + sign * d.to(oo.dtype)
        return (oo,) + o[1:]
    h = layers[L - 1].register_forward_hook(hook)
    with torch.no_grad():
        lg = model(torch.tensor([toks(p)], device=DEV)).logits[0, -1].float()
    h.remove()
    return lg[TRUE].item() > lg[FALSE].item()


def main():
    print(f"Model: {MODEL}  layer L{L}  N={len(PAIRS)} pairs")
    dirs = [last_vec(t) - last_vec(f) for t, f in PAIRS]
    D = torch.stack(dirs)
    coss = [torch.nn.functional.cosine_similarity(D[i], D[j], 0).item()
            for i, j in itertools.combinations(range(len(PAIRS)), 2)]
    print(f"truth direction consistency: mean pairwise cos {sum(coss)/len(coss):.2f}")

    for scale in SCALES:
        f2t = t2f = 0
        for i, (t, f) in enumerate(PAIRS):
            loo = torch.stack([dirs[j] for j in range(len(PAIRS)) if j != i]).mean(0) * scale
            if predicts_true(f, loo, +1):      # false + truth -> true
                f2t += 1
            if not predicts_true(t, loo, -1):  # true - truth -> false
                t2f += 1
        print(f"scale {scale}x (LOO): false->true {f2t}/{len(PAIRS)}, "
              f"true->false {t2f}/{len(PAIRS)}")


if __name__ == "__main__":
    main()
