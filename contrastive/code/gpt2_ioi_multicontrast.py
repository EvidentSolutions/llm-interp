"""
Multi-contrast triangulation on per-head IOI contributions in GPT-2.

Single-pair readout produces gibberish for some high-norm heads.
Average the per-head contribution vectors across many name pairs,
project through W_U. If the gibberish was superposition, the shared
component should become readable. If not, the signal isn't token-shaped.

Also report: norm of averaged vector vs mean single-pair norm.
If the averaged norm drops → signal was pair-specific (cancels).
If it persists → signal is shared across pairs.

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_ioi_multicontrast.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.n_layer
NH = model.config.n_head
HD = model.config.n_embd // NH
d_model = model.config.n_embd
W_U = model.lm_head.weight.detach().float()


def tk(logits, k=5):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def run_capture(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}
    hooks = []
    for L in range(NL):
        def make_hook(li):
            def hook_fn(m, inp, out):
                captured[li] = inp[0].detach().float().cpu()
            return hook_fn
        h = model.transformer.h[L].attn.c_proj.register_forward_hook(make_hook(L))
        hooks.append(h)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    for h in hooks:
        h.remove()
    return out, ids, captured


def get_head_contribution(cap, L, h, pos):
    """Get head h's contribution to residual stream at position pos."""
    W = model.transformer.h[L].attn.c_proj.weight.float().cpu()
    inp = cap[L][0, pos, h*HD:(h+1)*HD]
    return inp @ W[h*HD:(h+1)*HD, :]  # (d_model,)


# ============================================================
# Name pairs — many, to average over
# ============================================================
TEMPLATE = "When {A} and {B} went to the store, {A} gave a drink to"

# Cross-gender pairs (IO name = B)
NAME_PAIRS = [
    ("John", "Mary"),
    ("Alice", "Bob"),
    ("Dan", "Eve"),
    ("Tom", "Sarah"),
    ("Mike", "Lisa"),
    ("James", "Anna"),
    ("Peter", "Jane"),
    ("David", "Emma"),
    ("Chris", "Laura"),
    ("Steve", "Karen"),
]

# Verify all pairs have same tokenization length
ref_len = None
valid_pairs = []
for a, b in NAME_PAIRS:
    pa = TEMPLATE.format(A=a, B=b)
    pb = TEMPLATE.format(A=b, B=a)
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    if ref_len is None:
        ref_len = len(ids_a)
    if len(ids_a) == ref_len and len(ids_b) == ref_len:
        valid_pairs.append((a, b))
        tokens = [tok.decode([t]) for t in ids_a]
    else:
        print(f"  Skipping {a}/{b}: length {len(ids_a)}/{len(ids_b)} != {ref_len}")

print(f"Valid pairs ({len(valid_pairs)}): {[f'{a}/{b}' for a, b in valid_pairs]}")
print(f"Token length: {ref_len}")

# Show reference tokenization
pa_ref = TEMPLATE.format(A=valid_pairs[0][0], B=valid_pairs[0][1])
ids_ref = tok(pa_ref, add_special_tokens=False)["input_ids"]
tokens_ref = [tok.decode([t]) for t in ids_ref]
print(f"Reference: {list(enumerate(tokens_ref))}")
END = ref_len - 1
print(f"END position: {END}")


# ============================================================
# Collect per-head contributions for each pair
# ============================================================
print(f"\n{'='*110}")
print("Collecting per-head contributions across all pairs...")
print("=" * 110)

# For each pair: per-head contrastive contribution at END
# Store as (L, H) → list of contribution vectors across pairs
head_contributions = {}  # (L, H) → list of (d_model,) tensors
head_norms = {}  # (L, H) → list of norms

for a, b in valid_pairs:
    pa = TEMPLATE.format(A=a, B=b)
    pb = TEMPLATE.format(A=b, B=a)

    out_a, ids_a, cap_a = run_capture(pa)
    out_b, ids_b, cap_b = run_capture(pb)

    pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
    pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
    print(f"  {a}/{b}: A→'{pred_a}', B→'{pred_b}'")

    for L in range(NL):
        for h in range(NH):
            c_a = get_head_contribution(cap_a, L, h, END)
            c_b = get_head_contribution(cap_b, L, h, END)
            delta = c_a - c_b

            key = (L, h)
            if key not in head_contributions:
                head_contributions[key] = []
                head_norms[key] = []
            head_contributions[key].append(delta)
            head_norms[key].append(float(delta.norm()))

    del out_a, out_b
    torch.cuda.empty_cache()


# ============================================================
# Compute averaged contributions and compare
# ============================================================
print(f"\n{'='*110}")
print("MULTI-CONTRAST RESULTS: averaged per-head contributions")
print("=" * 110)

results = []
for L in range(NL):
    for h in range(NH):
        key = (L, h)
        contribs = head_contributions[key]
        norms = head_norms[key]

        # Average contribution
        avg = torch.stack(contribs).mean(dim=0)
        avg_norm = float(avg.norm())
        mean_single_norm = sum(norms) / len(norms)

        # Norm retention: how much of the single-pair norm survives averaging?
        retention = avg_norm / mean_single_norm if mean_single_norm > 0 else 0

        # W_U readout of averaged contribution
        avg_logits = avg @ W_U.cpu().T

        # W_U readout of a single pair (first pair) for comparison
        single_logits = contribs[0] @ W_U.cpu().T

        results.append((L, h, mean_single_norm, avg_norm, retention,
                         avg_logits, single_logits))

# Sort by mean single-pair norm (the heads we care about)
results.sort(key=lambda x: -x[2])

print(f"\n  Top 25 heads by mean single-pair norm:")
print(f"  {'Head':>6} {'SingleNorm':>10} {'AvgNorm':>8} {'Retain%':>8}  "
      f"{'Single-pair readout (pair 1)':>45}  {'Multi-contrast averaged readout':>45}")
print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*8}  {'─'*45}  {'─'*45}")

for L, h, snorm, anorm, ret, avg_lg, single_lg in results[:25]:
    s_read = tk(single_lg, 5)
    a_read = tk(avg_lg, 5)
    flag = ""
    if ret > 0.5:
        flag = " ◄ shared"
    elif ret < 0.15:
        flag = " ◄ pair-specific"
    print(f"  L{L}H{h:<2} {snorm:>10.1f} {anorm:>8.1f} {ret:>7.1%}  "
          f"{s_read:>45}  {a_read:>45}{flag}")


# ============================================================
# Focus on the interesting cases
# ============================================================
print(f"\n{'='*110}")
print("FOCUS: heads where averaging CHANGES readability")
print("=" * 110)

print(f"\n  Heads with high norm retention (>40%) — shared signal:")
print(f"  These heads carry something consistent across all name pairs.")
print(f"  {'Head':>6} {'Retain%':>8} {'AvgNorm':>8}  {'Averaged readout':>50}")
print(f"  {'─'*6} {'─'*8} {'─'*8}  {'─'*50}")

for L, h, snorm, anorm, ret, avg_lg, single_lg in results[:40]:
    if ret > 0.40 and snorm > 4.0:
        a_read = tk(avg_lg, 6)
        print(f"  L{L}H{h:<2} {ret:>7.1%} {anorm:>8.1f}  {a_read}")

print(f"\n  Heads with low norm retention (<15%) — pair-specific signal:")
print(f"  The single-pair norm was high but cancels under averaging.")
print(f"  {'Head':>6} {'SingleNorm':>10} {'Retain%':>8}  {'Single readout (pair 1)':>50}")
print(f"  {'─'*6} {'─'*10} {'─'*8}  {'─'*50}")

for L, h, snorm, anorm, ret, avg_lg, single_lg in results[:40]:
    if ret < 0.15 and snorm > 4.0:
        s_read = tk(single_lg, 6)
        print(f"  L{L}H{h:<2} {snorm:>10.1f} {ret:>7.1%}  {s_read}")


# ============================================================
# Now repeat for C2 (dup vs no-dup) at S2
# ============================================================
print(f"\n{'='*110}")
print("C2: MULTI-CONTRAST at S2 position (dup vs no-dup)")
print("  Average across pairs where S1 varies but S2 is the same name")
print("=" * 110)

# For dup vs no-dup: A has John...John, B has Pete...John
# We vary the names across pairs
DUP_PAIRS = [
    ("When John and Mary went to the store, John gave a drink to",
     "When Pete and Mary went to the store, John gave a drink to"),
    ("When Alice and Bob went to the park, Alice gave a gift to",
     "When Carol and Bob went to the park, Alice gave a gift to"),
    ("When Dan and Eve ate dinner together, Dan passed the salt to",
     "When Sam and Eve ate dinner together, Dan passed the salt to"),
    ("When Tom and Sarah went to the mall, Tom gave a present to",
     "When Rick and Sarah went to the mall, Tom gave a present to"),
    ("When Mike and Lisa went to the park, Mike gave a book to",
     "When Fred and Lisa went to the park, Mike gave a book to"),
]

# Verify lengths match and find S2 position
dup_valid = []
for pa, pb in DUP_PAIRS:
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    if len(ids_a) == len(ids_b) == ref_len:
        dup_valid.append((pa, pb))
    else:
        print(f"  Skipping: len {len(ids_a)}/{len(ids_b)} != {ref_len}")

S2_POS = 9  # position of the repeated subject in the template

head_c2_contribs = {}
head_c2_norms = {}

for pa, pb in dup_valid:
    out_a, ids_a, cap_a = run_capture(pa)
    out_b, ids_b, cap_b = run_capture(pb)

    for L in range(NL):
        for h in range(NH):
            c_a = get_head_contribution(cap_a, L, h, S2_POS)
            c_b = get_head_contribution(cap_b, L, h, S2_POS)
            delta = c_a - c_b
            key = (L, h)
            if key not in head_c2_contribs:
                head_c2_contribs[key] = []
                head_c2_norms[key] = []
            head_c2_contribs[key].append(delta)
            head_c2_norms[key].append(float(delta.norm()))

    del out_a, out_b
    torch.cuda.empty_cache()

# Compute averages
c2_results = []
for L in range(NL):
    for h in range(NH):
        key = (L, h)
        contribs = head_c2_contribs[key]
        norms = head_c2_norms[key]
        avg = torch.stack(contribs).mean(dim=0)
        avg_norm = float(avg.norm())
        mean_norm = sum(norms) / len(norms)
        retention = avg_norm / mean_norm if mean_norm > 0 else 0
        avg_logits = avg @ W_U.cpu().T
        single_logits = contribs[0] @ W_U.cpu().T
        c2_results.append((L, h, mean_norm, avg_norm, retention,
                           avg_logits, single_logits))

c2_results.sort(key=lambda x: -x[2])

print(f"\n  Top 15 heads at S2 by mean single-pair norm:")
print(f"  {'Head':>6} {'SingleNorm':>10} {'AvgNorm':>8} {'Retain%':>8}  "
      f"{'Single-pair readout':>45}  {'Averaged readout':>45}")
print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*8}  {'─'*45}  {'─'*45}")

for L, h, snorm, anorm, ret, avg_lg, single_lg in c2_results[:15]:
    s_read = tk(single_lg, 5)
    a_read = tk(avg_lg, 5)
    flag = ""
    if ret > 0.5:
        flag = " ◄ shared"
    elif ret < 0.15:
        flag = " ◄ pair-specific"
    print(f"  L{L}H{h:<2} {snorm:>10.1f} {anorm:>8.1f} {ret:>7.1%}  "
          f"{s_read:>45}  {a_read:>45}{flag}")


print(f"\n{'='*110}")
print("DONE")
print("=" * 110)
