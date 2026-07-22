"""
Scalar implicature experiment.

Question: When the model processes "some," does it compute "not all" as an
inference in the residual stream, or does the pragmatic narrowing happen
only at prediction time (different continuations)?

"Some students passed" pragmatically implies "not all passed."
If the model computes this, Δh("some" - "all") should contain negation
tokens or restriction tokens at mid-layers, not just different entity
predictions.

Design:
  1. Basic trajectory: "Some X" vs "All X" — what does Δh read?
  2. Scalar scale: none < few < some < most < all — is there a monotonic
     ordering in a consistent direction? (Like the epistemic gradient)
  3. Cross-content consistency: does the some/all axis generalise across
     different noun phrases? (students, cookies, employees, countries...)
  4. Implicature vs. literal: "Some but not all X" vs "Some X" — does
     the explicit "but not all" change the trajectory, or is the
     implicature already computed?
  5. Embedded contexts that cancel implicature: "If some students passed,
     then..." (downward entailing — implicature suspended). Does the
     trajectory change?
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
W_U = model.lm_head.weight.detach()

cos = F.cosine_similarity


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_str(logits, k=6):
    vals, idxs = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(idxs[j])]).strip()[:12] for j in range(k))


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out.hidden_states


def show_contrast(label, text_a, text_b, layers=None):
    if layers is None:
        layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)

    print(f"\n  {label}")
    print(f"    A: \"{text_a}\"")
    print(f"    B: \"{text_b}\"")

    hs_a = get_hidden_states(text_a)
    hs_b = get_hidden_states(text_b)

    deltas = {}
    for L in layers:
        ha = hs_a[L][0, -1, :].float()
        hb = hs_b[L][0, -1, :].float()
        dh = ha - hb
        deltas[L] = dh.cpu()
        norm = float(dh.norm() / ha.norm())
        ld = dh @ W_U.float().T
        pos = topk_str(ld)
        neg = topk_str(-ld)
        print(f"    L{L:>2} ({norm:.3f}) +[{pos}]  -[{neg}]")

    del hs_a, hs_b
    torch.cuda.empty_cache()
    return deltas


# ── PART 1: BASIC SOME vs ALL TRAJECTORIES ───────────────────
print("\n" + "="*70)
print("PART 1: BASIC TRAJECTORIES — some vs all")
print("="*70)
print("Looking for: does 'some' read negation/restriction tokens?")
print("  +side = 'some' pole, -side = 'all' pole")

layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)
ref_L = _sl(28)[0]

basic_pairs = [
    ("students/passed",
     "Some of the students passed the exam, so",
     "All of the students passed the exam, so"),
    ("cookies/eaten",
     "Some of the cookies were eaten at the party, and",
     "All of the cookies were eaten at the party, and"),
    ("employees/promoted",
     "Some of the employees were promoted this year, which",
     "All of the employees were promoted this year, which"),
    ("countries/signed",
     "Some of the countries signed the agreement, meaning",
     "All of the countries signed the agreement, meaning"),
    ("books/translated",
     "Some of the books were translated into French, so",
     "All of the books were translated into French, so"),
    ("houses/damaged",
     "Some of the houses were damaged in the storm, and",
     "All of the houses were damaged in the storm, and"),
]

all_deltas = {}
for label, some_text, all_text in basic_pairs:
    deltas = show_contrast(label, some_text, all_text, layers)
    all_deltas[label] = deltas


# ── PART 2: CROSS-CONTENT CONSISTENCY ────────────────────────
print("\n" + "="*70)
print(f"PART 2: CROSS-CONTENT CONSISTENCY (cosine at L{ref_L})")
print("="*70)
print("If some/all is a consistent axis, cross-pair cosines should be high.\n")

pair_labels = [lb for lb, _, _ in basic_pairs]
n = len(pair_labels)

print(f"  {'':>16}", end="")
for lb in pair_labels:
    print(f" {lb[:10]:>10}", end="")
print()

cosines = []
for i, li in enumerate(pair_labels):
    print(f"  {li:>16}", end="")
    for j, lj in enumerate(pair_labels):
        c = float(cos(all_deltas[li][ref_L].unsqueeze(0),
                       all_deltas[lj][ref_L].unsqueeze(0)))
        if i != j:
            cosines.append(c)
        print(f" {c:>+10.3f}", end="")
    print()

print(f"\n  Mean off-diagonal cosine: {sum(cosines)/len(cosines):.3f}")
print(f"  Min: {min(cosines):.3f}  Max: {max(cosines):.3f}")


# ── PART 3: SCALAR SCALE — none < few < some < most < all ────
print("\n" + "="*70)
print("PART 3: SCALAR GRADIENT — is there a monotonic ordering?")
print("="*70)
print("Project each quantifier onto the some↔all axis.\n")

quantifiers = ["None", "Few", "Some", "Most", "All"]
template = "{q} of the students passed the exam, so"

# Use first pair's delta as the axis direction
axis_label = pair_labels[0]
axis_dir = all_deltas[axis_label][ref_L].to(DEV)  # some - all direction

q_vecs = {}
for q in quantifiers:
    text = template.format(q=q)
    hs = get_hidden_states(text)
    q_vecs[q] = hs[ref_L][0, -1, :].float().cpu()
    del hs
    torch.cuda.empty_cache()

# Project onto axis
baseline = q_vecs["All"].to(DEV)
axis_norm = axis_dir.norm()

print(f"  Projection onto some→all axis (L{ref_L}):")
print(f"  (positive = toward 'some', negative = toward 'all')\n")
for q in quantifiers:
    v = q_vecs[q].to(DEV)
    proj = float(torch.dot(v - baseline, axis_dir) / axis_norm)
    bar_pos = "█" * int(max(0, proj) / 5)
    bar_neg = "█" * int(max(0, -proj) / 5)
    print(f"  {q:>6}: {proj:>+8.1f}  {'':>{20-len(bar_neg)}}{bar_neg}|{bar_pos}")

# Also check pairwise trajectory: each adjacent quantifier pair
print(f"\n  Adjacent-pair contrastive readout:")
for i in range(len(quantifiers) - 1):
    q_lo, q_hi = quantifiers[i], quantifiers[i+1]
    text_lo = template.format(q=q_lo)
    text_hi = template.format(q=q_hi)
    hs_lo = get_hidden_states(text_lo)
    hs_hi = get_hidden_states(text_hi)
    dh = hs_lo[ref_L][0, -1, :].float() - hs_hi[ref_L][0, -1, :].float()
    ld = dh @ W_U.float().T
    pos = topk_str(ld)
    neg = topk_str(-ld)
    print(f"    {q_lo}→{q_hi}: +[{pos}]  -[{neg}]")
    del hs_lo, hs_hi
    torch.cuda.empty_cache()


# ── PART 4: EXPLICIT vs IMPLICIT IMPLICATURE ─────────────────
print("\n" + "="*70)
print("PART 4: EXPLICIT 'but not all' vs BARE 'some'")
print("="*70)
print("If implicature is already computed, adding 'but not all' should")
print("change little. If it's not computed, this should add negation.\n")

explicit_pairs = [
    ("bare_some",
     "Some of the students passed the exam, so",
     "Some but not all of the students passed the exam, so"),
    ("bare_some_cookies",
     "Some of the cookies were eaten at the party, and",
     "Some but not all of the cookies were eaten at the party, and"),
]

for label, bare, explicit in explicit_pairs:
    deltas = show_contrast(f"{label}: bare_some − explicit_some", bare, explicit, layers)
    # Check: is this delta small (implicature already present) or large?
    ref_dh = deltas[ref_L]
    print(f"    Relative norm at L{ref_L}: {float(ref_dh.norm()):.1f}")


# ── PART 5: DOWNWARD-ENTAILING CANCELLATION ──────────────────
print("\n" + "="*70)
print("PART 5: IMPLICATURE CANCELLATION IN DOWNWARD-ENTAILING CONTEXT")
print("="*70)
print("'If some X, then...' suspends the 'not all' implicature.")
print("Contrast: bare assertion vs if-clause. Does the trajectory change?\n")

cancel_pairs = [
    ("assertion_vs_if",
     "Some of the students passed the exam, so",
     "If some of the students passed the exam, then"),
    ("assertion_vs_doubt",
     "Some of the students passed the exam, so",
     "I doubt that some of the students passed the exam, but"),
]

for label, assertion, embedded in cancel_pairs:
    show_contrast(label, assertion, embedded, layers)


# ── PART 6: LAYER-RESOLVED CONSISTENCY ───────────────────────
print("\n" + "="*70)
print("PART 6: MEAN CROSS-PAIR COSINE BY LAYER (some/all axis)")
print("="*70)

for L in layers:
    pair_cos = []
    for i, li in enumerate(pair_labels):
        for j, lj in enumerate(pair_labels):
            if i < j:
                c = float(cos(all_deltas[li][L].unsqueeze(0),
                               all_deltas[lj][L].unsqueeze(0)))
                pair_cos.append(c)
    mean_c = sum(pair_cos) / len(pair_cos)
    bar = "█" * int(max(0, mean_c) * 40)
    print(f"  L{L:>2}: mean cos = {mean_c:+.3f}  {bar}")


print("\n" + "="*70)
print("DONE")
print("="*70)
