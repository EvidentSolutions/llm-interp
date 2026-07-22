"""
Verify claim: "the contrastive projection reveals content that neither
constituent's logit lens can see."

For each poster case, show at key layers:
  - Logit lens on h_c (top-5 tokens)
  - Logit lens on h_k (top-5 tokens)
  - Contrastive projection on h_c - h_k (top-5 positive, top-5 negative)

If the claim is true, the contrastive top tokens should NOT appear in
either constituent's logit lens top tokens at mid-layers.

Usage: .venv/Scripts/python.exe contrastive/code/logit_lens_comparison.py
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

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach().float()


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return [(tok.decode([int(i[j])]).strip()[:15], int(i[j])) for j in range(k)]


def tk_str(logits, k=5):
    return ", ".join(t[0] for t in tk(logits, k))


def compare_lenses(pa, pb, layers=None, label=""):
    if layers is None:
        layers = [4, 8, 12, 16, 20, 24, 28, 32]
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )

    print(f'\n  {label}')
    print(f'  A: "{pa}"')
    print(f'  B: "{pb}"')

    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b

        # Logit lens on each constituent
        logits_a = h_a @ W_U.T
        logits_b = h_b @ W_U.T
        # Contrastive
        logits_d = dh @ W_U.T

        top_a = tk(logits_a, 10)
        top_b = tk(logits_b, 10)
        top_d_pos = tk(logits_d, 5)
        top_d_neg = tk(-logits_d, 5)  # bottom = most negative

        # Check overlap: are contrastive top tokens in either constituent's top-20?
        top_a_ids = set(t[1] for t in tk(logits_a, 20))
        top_b_ids = set(t[1] for t in tk(logits_b, 20))
        contrast_pos_ids = set(t[1] for t in top_d_pos)
        contrast_neg_ids = set(t[1] for t in top_d_neg)

        overlap_a = contrast_pos_ids & top_a_ids
        overlap_b = contrast_neg_ids & top_b_ids

        print(f"\n  L{L:>2}:")
        print(f"    Logit lens A: [{', '.join(t[0] for t in top_a[:5])}]")
        print(f"    Logit lens B: [{', '.join(t[0] for t in top_b[:5])}]")
        print(f"    Contrastive:  + [{', '.join(t[0] for t in top_d_pos)}]"
              f"  - [{', '.join(t[0] for t in top_d_neg)}]")
        print(f"    Overlap: {len(overlap_a)}/5 pos in A's top-20, "
              f"{len(overlap_b)}/5 neg in B's top-20")

    # Summary: what fraction of contrastive tokens appear in constituent top-20?
    print(f"\n  Overlap summary (contrastive top-5 found in constituent top-20):")
    print(f"  {'Layer':>5} {'pos∩A':>6} {'neg∩B':>6}")
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        logits_a = h_a @ W_U.T
        logits_b = h_b @ W_U.T
        logits_d = dh @ W_U.T

        top_a_ids = set(t[1] for t in tk(logits_a, 20))
        top_b_ids = set(t[1] for t in tk(logits_b, 20))
        top_d_pos = tk(logits_d, 5)
        top_d_neg = tk(-logits_d, 5)
        contrast_pos_ids = set(t[1] for t in top_d_pos)
        contrast_neg_ids = set(t[1] for t in top_d_neg)

        oa = len(contrast_pos_ids & top_a_ids)
        ob = len(contrast_neg_ids & top_b_ids)
        print(f"  L{L:>3}   {oa}/5    {ob}/5")

    del out_a, out_b
    torch.cuda.empty_cache()


# ============================================================
print("=" * 100)
print("LOGIT LENS vs CONTRASTIVE PROJECTION")
print("=" * 100)

compare_lenses(
    "The hot dog was",
    "The cold dog was",
    layers=[4, 8, 12, 16, 20, 24, 28, 32],
    label="HOT DOG"
)

compare_lenses(
    "John and Mary went to the store. John gave a book to",
    "Mary and John went to the store. Mary gave a book to",
    layers=[8, 16, 20, 24, 28, 32],
    label="IOI"
)

compare_lenses(
    "The Eiffel Tower is located in",
    "The Colosseum is located in",
    layers=[8, 16, 20, 24, 28, 32],
    label="FACTUAL RECALL"
)

compare_lenses(
    "After Monday comes",
    "After Tuesday comes",
    layers=[8, 16, 20, 24, 28, 32],
    label="SUCCESSOR"
)

compare_lenses(
    "He caught a cold and",
    "He caught a fish and",
    layers=[8, 16, 20, 24, 28, 32],
    label="COLD/FISH"
)

print(f"\n{'='*100}")
print("DONE")
