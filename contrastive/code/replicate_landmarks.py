"""
Replicate landmark mech interp findings with contrastive method.
1. IOI (indirect object identification)
2. Factual recall
3. Successor heads

Usage: .venv/Scripts/python.exe contrastive/code/replicate_landmarks.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer
import os

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(
        tok.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(
        tok.decode([int(i[j])]).strip()[:12] for j in range(k)
    )


def run_pair(model, tok, pa, pb, NL, W_U, layers=None):
    if layers is None:
        layers = _sl(8, 16, 20, 24, 28, 32)
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV),
            output_hidden_states=True,
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV),
            output_hidden_states=True,
        )
        gen_a = model.generate(
            torch.tensor([ids_a], device=DEV),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        gen_b = model.generate(
            torch.tensor([ids_b], device=DEV),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    ans_a = tok.decode(gen_a[0][len(ids_a) :]).strip()[:30]
    ans_b = tok.decode(gen_b[0][len(ids_b) :]).strip()[:30]
    print(f'  A: "{pa}" -> "{ans_a}"')
    print(f'  B: "{pb}" -> "{ans_b}"')
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.float().T
        print(f"    L{L:>2} ({norm:.3f}) A=[{tk(ld)}]  B=[{bk(ld)}]")
    del out_a, out_b
    return ans_a, ans_b


print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
for p in model.parameters():
    p.requires_grad_(False)
NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

def _sl(*layers):
    """Scale layer indices from 32-layer base to current NL."""
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))

# ============================================================
print("=" * 100)
print("1. IOI")
print("=" * 100)

ioi = [
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to"),
    ("Alice and Bob went to the park. Alice gave a gift to",
     "Bob and Alice went to the park. Bob gave a gift to"),
    ("Dan and Eve ate dinner together. Dan passed the salt to",
     "Eve and Dan ate dinner together. Eve passed the salt to"),
]

for pa, pb in ioi:
    print()
    run_pair(model, tok, pa, pb, NL, W_U)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. FACTUAL RECALL")
print("=" * 100)

facts = [
    ("The Eiffel Tower is located in",
     "The Colosseum is located in"),
    ("Mount Fuji is located in",
     "Big Ben is located in"),
    ("The iPhone was made by",
     "Windows was made by"),
    ("The theory of relativity was developed by",
     "The laws of motion were developed by"),
    ("The capital of France is",
     "The capital of Japan is"),
    ("Shakespeare wrote",
     "Tolstoy wrote"),
]

for pa, pb in facts:
    print()
    run_pair(model, tok, pa, pb, NL, W_U)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. SUCCESSOR HEADS")
print("=" * 100)

successors = [
    ("After Monday comes",
     "After Tuesday comes"),
    ("After January comes",
     "After February comes"),
    ("After summer comes",
     "After winter comes"),
    ("Monday, Tuesday, Wednesday, Thursday,",
     "Tuesday, Wednesday, Thursday, Friday,"),
    ("1, 2, 3, 4,",
     "2, 3, 4, 5,"),
]

for pa, pb in successors:
    print()
    run_pair(model, tok, pa, pb, NL, W_U,
             layers=_sl(16, 24, 28, 32))

torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
