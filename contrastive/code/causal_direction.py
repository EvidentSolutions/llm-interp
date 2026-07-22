"""
Causal direction experiment.

Question: Does the model encode causal directionality as a consistent axis
(like past/future), or is "A causes B" vs "B causes A" symmetric in the
residual stream with direction only in surface syntax?

Design:
  - Multiple cause-effect pairs with natural causal asymmetry
  - For each pair, construct forward ("rain causes wet streets") and
    reversed ("wet streets cause rain") prompts
  - 2x2 factorial: cross entity pairs with direction to separate
    causal-direction axis from entity content
  - Consistency check: cosine between Δh across different entity pairs
    at the same causal direction tells us if there's a universal axis

If causal direction IS an axis: high cosine across pairs (like past/future).
If it's NOT: low cosine, meaning direction is only in the syntax, not a
separable representation.
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


# ── CAUSAL PAIRS ──────────────────────────────────────────────
# Each entry: (cause, effect, forward_template, reverse_template)
# Templates end at the prediction point so we read the last token.

causal_pairs = [
    ("rain",        "wet streets",
     "Heavy rain causes",
     "Wet streets are caused by"),

    ("fire",        "smoke",
     "A large fire causes",
     "Thick smoke is caused by"),

    ("exercise",    "weight loss",
     "Regular exercise causes",
     "Significant weight loss is caused by"),

    ("deforestation", "flooding",
     "Massive deforestation causes",
     "Severe flooding is caused by"),

    ("stress",      "insomnia",
     "Chronic stress causes",
     "Persistent insomnia is caused by"),

    ("vaccination", "immunity",
     "Proper vaccination causes",
     "Strong immunity is caused by"),

    ("overheating", "engine failure",
     "Severe overheating causes",
     "Sudden engine failure is caused by"),

    ("poverty",     "crime",
     "Widespread poverty causes",
     "Rising crime is caused by"),
]

# ── PART 1: TRAJECTORY FOR EACH PAIR ─────────────────────────
print("\n" + "="*70)
print("PART 1: CONTRASTIVE TRAJECTORIES (forward - reverse)")
print("="*70)

layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)
ref_L = _sl(28)[0]

all_deltas = {}  # (pair_idx, layer) -> delta vector

for i, (cause, effect, fwd, rev) in enumerate(causal_pairs):
    print(f"\n  Pair {i}: {cause} → {effect}")
    print(f"    FWD: \"{fwd}\"")
    print(f"    REV: \"{rev}\"")

    hs_fwd = get_hidden_states(fwd)
    hs_rev = get_hidden_states(rev)

    for L in layers:
        h_fwd = hs_fwd[L][0, -1, :].float()
        h_rev = hs_rev[L][0, -1, :].float()
        dh = h_fwd - h_rev
        all_deltas[(i, L)] = dh.cpu()

        norm = float(dh.norm() / h_fwd.norm())
        ld = dh @ W_U.float().T
        pos = topk_str(ld)
        neg = topk_str(-ld)
        print(f"    L{L:>2} ({norm:.3f}) +[{pos}]  -[{neg}]")

    del hs_fwd, hs_rev
    torch.cuda.empty_cache()

# ── PART 2: CROSS-PAIR CONSISTENCY ───────────────────────────
print("\n" + "="*70)
print(f"PART 2: CROSS-PAIR CONSISTENCY (cosine between Δh at L{ref_L})")
print("="*70)
print("If causal direction is an axis, off-diagonal cosines should be high.")
print("If it's entity-specific, they should be near zero.\n")

n = len(causal_pairs)
labels = [f"{c[:8]}→{e[:8]}" for c, e, _, _ in causal_pairs]

# Header
print(f"  {'':>18}", end="")
for lb in labels:
    print(f" {lb[:10]:>10}", end="")
print()

cosines = []
for i in range(n):
    print(f"  {labels[i]:>18}", end="")
    for j in range(n):
        c = float(cos(all_deltas[(i, ref_L)].unsqueeze(0),
                       all_deltas[(j, ref_L)].unsqueeze(0)))
        if i != j:
            cosines.append(c)
        print(f" {c:>+10.3f}", end="")
    print()

print(f"\n  Mean off-diagonal cosine: {sum(cosines)/len(cosines):.3f}")
print(f"  Min: {min(cosines):.3f}  Max: {max(cosines):.3f}")


# ── PART 3: 2x2 FACTORIAL ────────────────────────────────────
# Cross entity content with causal direction to check separability.
# Use first 4 pairs. For each pair of pairs (i,j), build:
#   A = fwd_i, B = rev_i, C = fwd_j, D = rev_j
#   Direction axis: (A-B) vs (C-D) should be consistent
#   Content axis: (A-C) vs (B-D) should be consistent
#   Direction ⊥ Content: (A-B) vs (A-C) should be low

print("\n" + "="*70)
print(f"PART 3: 2x2 FACTORIAL — direction vs content separability (L{ref_L})")
print("="*70)

# Collect hidden states at ref_L for first 4 pairs
ref_h = {}  # (pair_idx, "fwd"/"rev") -> h
for i in range(min(4, n)):
    cause, effect, fwd, rev = causal_pairs[i]
    hs_fwd = get_hidden_states(fwd)
    hs_rev = get_hidden_states(rev)
    ref_h[(i, "fwd")] = hs_fwd[ref_L][0, -1, :].float().cpu()
    ref_h[(i, "rev")] = hs_rev[ref_L][0, -1, :].float().cpu()
    del hs_fwd, hs_rev
    torch.cuda.empty_cache()

print("\n  Checking all pairs-of-pairs:\n")
for i in range(min(4, n)):
    for j in range(i+1, min(4, n)):
        A = ref_h[(i, "fwd")]
        B = ref_h[(i, "rev")]
        C = ref_h[(j, "fwd")]
        D = ref_h[(j, "rev")]

        d_dir_i = A - B  # direction axis, pair i
        d_dir_j = C - D  # direction axis, pair j
        d_cont_fwd = A - C  # content axis, forward
        d_cont_rev = B - D  # content axis, reverse

        dir_cos = float(cos(d_dir_i.unsqueeze(0), d_dir_j.unsqueeze(0)))
        cont_cos = float(cos(d_cont_fwd.unsqueeze(0), d_cont_rev.unsqueeze(0)))
        orth_cos = float(cos(d_dir_i.unsqueeze(0), d_cont_fwd.unsqueeze(0)))

        ci, ei, _, _ = causal_pairs[i]
        cj, ej, _, _ = causal_pairs[j]
        print(f"  [{ci[:8]}] × [{cj[:8]}]")
        print(f"    Direction consistency: {dir_cos:+.3f}")
        print(f"    Content consistency:   {cont_cos:+.3f}")
        print(f"    Direction ⊥ Content:   {orth_cos:+.3f}")
        print()


# ── PART 4: LAYER-RESOLVED CONSISTENCY ───────────────────────
print("="*70)
print("PART 4: MEAN CROSS-PAIR COSINE BY LAYER")
print("="*70)
print("Where does the causal direction axis emerge (if it exists)?\n")

for L in layers:
    pair_cosines = []
    for i in range(n):
        for j in range(i+1, n):
            c = float(cos(all_deltas[(i, L)].unsqueeze(0),
                           all_deltas[(j, L)].unsqueeze(0)))
            pair_cosines.append(c)
    mean_c = sum(pair_cosines) / len(pair_cosines)
    bar = "█" * int(max(0, mean_c) * 40)
    print(f"  L{L:>2}: mean cos = {mean_c:+.3f}  {bar}")

print("\n" + "="*70)
print("DONE")
print("="*70)
