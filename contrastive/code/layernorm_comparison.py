"""
Compare three contrastive projection variants:
  A) Raw:    (h_c - h_k) @ W_U^T           (what we do now)
  B) PostLN: (LN(h_c) - LN(h_k)) @ W_U^T  (each state normalized first)
  C) DiffLN: LN(h_c - h_k) @ W_U^T         (normalize the difference)

For each, show top-5 tokens and compare rankings across our poster cases.

Usage: .venv/Scripts/python.exe contrastive/code/layernorm_comparison.py
"""
import sys
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/phi-2"

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
LN = model.model.final_layernorm
# Also get bias if present
bias = model.lm_head.bias.detach().float() if model.lm_head.bias is not None else None


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(f"{tok.decode([int(i[j])]).strip()[:12]}" for j in range(k))


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(f"{tok.decode([int(i[j])]).strip()[:12]}" for j in range(k))


def compare_variants(pa, pb, layers=None):
    if layers is None:
        layers = [4, 8, 16, 24, 28, 32]
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )

    print(f'\n  A: "{pa}"')
    print(f'  B: "{pb}"')

    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()

        # Variant A: Raw difference
        dh_raw = h_a - h_b
        logits_A = dh_raw @ W_U.T

        # Variant B: Post-LN difference (LN each, then subtract)
        h_a_ln = LN(h_a.unsqueeze(0).half()).float().squeeze(0)
        h_b_ln = LN(h_b.unsqueeze(0).half()).float().squeeze(0)
        dh_postln = h_a_ln - h_b_ln
        logits_B = dh_postln @ W_U.T

        # Variant C: LN of the difference
        dh_diffln = LN(dh_raw.unsqueeze(0).half()).float().squeeze(0)
        logits_C = dh_diffln @ W_U.T

        # Cosines between variants
        cos_AB = float(torch.nn.functional.cosine_similarity(
            logits_A.unsqueeze(0), logits_B.unsqueeze(0)))
        cos_AC = float(torch.nn.functional.cosine_similarity(
            logits_A.unsqueeze(0), logits_C.unsqueeze(0)))
        cos_BC = float(torch.nn.functional.cosine_similarity(
            logits_B.unsqueeze(0), logits_C.unsqueeze(0)))

        # Rank correlation: how many of top-10 overlap?
        top10_A = set(torch.topk(logits_A, 10).indices.tolist())
        top10_B = set(torch.topk(logits_B, 10).indices.tolist())
        top10_C = set(torch.topk(logits_C, 10).indices.tolist())
        overlap_AB = len(top10_A & top10_B)
        overlap_AC = len(top10_A & top10_C)

        print(f"\n  L{L:>2}  cos(A,B)={cos_AB:.4f}  cos(A,C)={cos_AC:.4f}  "
              f"top10_overlap: A∩B={overlap_AB}/10  A∩C={overlap_AC}/10")
        print(f"    Raw:    + [{tk(logits_A)}]  - [{bk(logits_A)}]")
        print(f"    PostLN: + [{tk(logits_B)}]  - [{bk(logits_B)}]")
        print(f"    DiffLN: + [{tk(logits_C)}]  - [{bk(logits_C)}]")

    del out_a, out_b


# ============================================================
print("=" * 100)
print("1. HOT DOG — compound noun disambiguation")
print("=" * 100)
compare_variants(
    "The hot dog was",
    "The cold dog was",
    layers=[4, 8, 16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("2. IOI — indirect object identification")
print("=" * 100)
compare_variants(
    "John and Mary went to the store. John gave a book to",
    "Mary and John went to the store. Mary gave a book to",
    layers=[8, 16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("3. FACTUAL RECALL — Eiffel Tower vs Colosseum")
print("=" * 100)
compare_variants(
    "The Eiffel Tower is located in",
    "The Colosseum is located in",
    layers=[16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("4. SUCCESSOR — Monday vs Tuesday")
print("=" * 100)
compare_variants(
    "After Monday comes",
    "After Tuesday comes",
    layers=[16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. TRUTH — true vs false statement")
print("=" * 100)
compare_variants(
    "Paris is the capital of France. This is",
    "Paris is the capital of Germany. This is",
    layers=[16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("6. NEGATION — positive vs negated")
print("=" * 100)
compare_variants(
    "The dog ran quickly through the park",
    "The dog did not run quickly through the park",
    layers=[16, 24, 28, 32]
)

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("7. SYSTEMATIC: cosine across ALL layers for hot dog case")
print("=" * 100)

pa = "The hot dog was"
pb = "The cold dog was"
ids_a = tok(pa, add_special_tokens=False)["input_ids"]
ids_b = tok(pb, add_special_tokens=False)["input_ids"]
with torch.no_grad():
    out_a = model(torch.tensor([ids_a], device=DEV), output_hidden_states=True)
    out_b = model(torch.tensor([ids_b], device=DEV), output_hidden_states=True)

print(f"  {'L':>3}  cos(Raw,PostLN)  cos(Raw,DiffLN)  "
      f"top10(A∩B)  ||dh||  ||dh_ln||  ratio")
for L in range(NL + 1):
    h_a = out_a.hidden_states[L][0, -1, :].float()
    h_b = out_b.hidden_states[L][0, -1, :].float()
    dh = h_a - h_b
    h_a_ln = LN(h_a.unsqueeze(0).half()).float().squeeze(0)
    h_b_ln = LN(h_b.unsqueeze(0).half()).float().squeeze(0)
    dh_ln = h_a_ln - h_b_ln

    lA = dh @ W_U.T
    lB = dh_ln @ W_U.T
    dh_normed = LN(dh.unsqueeze(0).half()).float().squeeze(0)
    lC = dh_normed @ W_U.T

    cos_AB = float(torch.nn.functional.cosine_similarity(
        lA.unsqueeze(0), lB.unsqueeze(0)))
    cos_AC = float(torch.nn.functional.cosine_similarity(
        lA.unsqueeze(0), lC.unsqueeze(0)))
    top10_A = set(torch.topk(lA, 10).indices.tolist())
    top10_B = set(torch.topk(lB, 10).indices.tolist())
    overlap = len(top10_A & top10_B)
    norm_raw = float(dh.norm())
    norm_ln = float(dh_ln.norm())
    ratio = norm_raw / norm_ln if norm_ln > 0 else float('inf')

    print(f"  {L:>3}  {cos_AB:>+.4f}          {cos_AC:>+.4f}          "
          f"  {overlap:>2}/10     {norm_raw:>6.1f}  {norm_ln:>8.4f}  {ratio:>7.1f}")

del out_a, out_b
torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
