"""
Strict GPT-2 IOI circuit recovery via contrastive projection.

Goal: recover ALL 13 known circuit heads AND classify them by
functional signature (input attention × output W_U readout),
not just rank by norm.

Known circuit (Wang et al. 2023):
  Name mover (pos):   L9H9, L10H0, L9H6
  Name mover (neg):   L10H7, L11H10
  S-inhibition:       L7H3, L7H9, L8H6, L8H10
  Duplicate token:    L0H1, L3H0
  Induction:          L5H5, L6H9

Contrast designs:
  C1: Name-swap (A=John-subj, B=Mary-subj) at END
      → surfaces name movers + S-inhibition
  C2: Duplicate vs no-duplicate at S2 position
      → surfaces duplicate-token heads
  C3: Repeated-subject vs novel-subject at END
      → surfaces induction + S-inhibition

For each head, classify by:
  - INPUT: attention pattern (attends to IO? S2? S1? position-after-S1?)
  - OUTPUT: W_U projection (name tokens? gender? structural/unreadable?)

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_ioi_circuit_strict.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading GPT-2 (eager attention)...")
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", dtype=torch.float32, low_cpu_mem_usage=True,
    attn_implementation="eager",
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained("gpt2")
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.n_layer   # 12
NH = model.config.n_head    # 12
HD = model.config.n_embd // NH  # 64
d_model = model.config.n_embd   # 768
W_U = model.lm_head.weight.detach().float()

KNOWN = {
    (9, 9): "nm+", (10, 0): "nm+", (9, 6): "nm+",
    (10, 7): "nm-", (11, 10): "nm-",
    (7, 3): "s-inh", (7, 9): "s-inh", (8, 6): "s-inh", (8, 10): "s-inh",
    (0, 1): "dup", (3, 0): "dup",
    (5, 5): "ind", (6, 9): "ind",
}


def tk(logits, k=5):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def run_and_capture(text):
    """Run model, return output with hidden states + attentions,
    plus per-layer per-head c_proj inputs for decomposition."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}
    hooks = []

    for L in range(NL):
        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                # inp[0] is concatenated head outputs before c_proj
                captured[layer_idx] = inp[0].detach().float().cpu()
            return hook_fn
        h = model.transformer.h[L].attn.c_proj.register_forward_hook(make_hook(L))
        hooks.append(h)

    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True, output_attentions=True)

    for h in hooks:
        h.remove()

    return out, ids, captured


def per_head_decomposition(captured_a, captured_b, read_pos):
    """Compute per-head contrastive contribution at a given position.
    Returns list of (layer, head, norm, logits_vector)."""
    results = []
    for L in range(NL):
        c_proj_w = model.transformer.h[L].attn.c_proj.weight.float().cpu()
        for h in range(NH):
            inp_a = captured_a[L][0, read_pos, h*HD:(h+1)*HD]
            inp_b = captured_b[L][0, read_pos, h*HD:(h+1)*HD]
            d_inp = inp_a - inp_b
            w_h = c_proj_w[h*HD:(h+1)*HD, :]
            contribution = d_inp @ w_h
            norm = float(contribution.norm())
            logits = contribution @ W_U.cpu().T
            results.append((L, h, norm, logits))
    return results


def classify_head(layer, head, attn_a, attn_b, logits, read_pos,
                  io_pos, s1_pos, s2_pos, after_s1_pos):
    """Classify a head by its attention pattern and W_U readout."""
    # Attention from read_pos
    a_weights = attn_a[layer][0, head, read_pos, :]
    b_weights = attn_b[layer][0, head, read_pos, :]

    # Average attention to key positions
    attn_io = (float(a_weights[io_pos]) + float(b_weights[io_pos])) / 2
    attn_s1 = (float(a_weights[s1_pos]) + float(b_weights[s1_pos])) / 2
    attn_s2 = (float(a_weights[s2_pos]) + float(b_weights[s2_pos])) / 2
    attn_after_s1 = (float(a_weights[after_s1_pos]) + float(b_weights[after_s1_pos])) / 2

    # Contrastive attention (difference matters for some heads)
    d_attn_io = float(a_weights[io_pos]) - float(b_weights[io_pos])
    d_attn_s2 = float(a_weights[s2_pos]) - float(b_weights[s2_pos])

    # W_U readout
    top_toks = tk(logits, 5)
    bot_toks = tk(-logits, 5)

    # Check if readout contains name-like tokens
    top5_str = top_toks.lower()

    return {
        'attn_io': attn_io, 'attn_s1': attn_s1,
        'attn_s2': attn_s2, 'attn_after_s1': attn_after_s1,
        'd_attn_io': d_attn_io, 'd_attn_s2': d_attn_s2,
        'top_toks': top_toks, 'bot_toks': bot_toks,
    }


# ============================================================
# Prompt setup
# ============================================================
# Template: "When {S1=IO_name} and {S2_name} went to the store, {S2=S2_name} gave a drink to"
# In name-swap: swap S1 and S2
# In duplicate-removed: replace S1 with a third name
# Positions are fixed by this template

PROMPT_A = "When John and Mary went to the store, John gave a drink to"
PROMPT_B_NAMESWAP = "When Mary and John went to the store, Mary gave a drink to"
PROMPT_B_NODUP = "When Pete and Mary went to the store, John gave a drink to"
PROMPT_B_NOVELSUBJ = "When John and Mary went to the store, Sam gave a drink to"

# Tokenize reference prompt to find positions
ids_ref = tok(PROMPT_A, add_special_tokens=False)["input_ids"]
tokens_ref = [tok.decode([t]) for t in ids_ref]
print(f"Reference tokens: {list(enumerate(tokens_ref))}")

# Position identification
# "When John and Mary went to the store , John gave a drink to"
#   0     1    2    3     4    5   6    7  8   9    10  11   12  13
S1_POS = 1       # first "John" (= IO name in standard IOI framing)
IO_POS = 1       # alias: the indirect object name
AFTER_S1_POS = 2  # "and" — what follows first subject
S2_NAME_POS = 3   # "Mary" in A (this is the OTHER name)
S2_POS = 9        # second "John" (= subject of "gave")
END_POS = 13      # "to" — prediction position

print(f"S1/IO={IO_POS}('{tokens_ref[IO_POS]}'), "
      f"after_S1={AFTER_S1_POS}('{tokens_ref[AFTER_S1_POS]}'), "
      f"S2_name={S2_NAME_POS}('{tokens_ref[S2_NAME_POS]}'), "
      f"S2={S2_POS}('{tokens_ref[S2_POS]}'), "
      f"END={END_POS}('{tokens_ref[END_POS]}')")

# Wait — standard IOI naming:
# "When [Name1] and [Name2] went..., [Name1] gave a drink to" → predicts Name2
# S1 = first occurrence of the repeated name (pos 1 = "John")
# IO = the non-repeated name (pos 3 = "Mary") — the indirect object
# S2 = second occurrence of the repeated name (pos 9 = "John")
# Let me fix the position labels:
S1_POS = 1   # First "John"
IO_POS = 3   # "Mary" — the indirect object (predicted answer)
S2_POS = 9   # Second "John" — the repeated subject
AFTER_S1_POS = 2  # Token after S1 ("and")
END_POS = 13

print(f"\nCorrected IOI positions:")
print(f"  S1={S1_POS}('{tokens_ref[S1_POS]}') — first occurrence of repeated name")
print(f"  IO={IO_POS}('{tokens_ref[IO_POS]}') — indirect object (predicted)")
print(f"  S2={S2_POS}('{tokens_ref[S2_POS]}') — second occurrence of repeated name")
print(f"  after_S1={AFTER_S1_POS}('{tokens_ref[AFTER_S1_POS]}')")
print(f"  END={END_POS}('{tokens_ref[END_POS]}') — prediction position")


# ============================================================
# CONTRAST 1: Name-swap at END — surfaces name movers + S-inhibition
# ============================================================
print(f"\n{'='*100}")
print("CONTRAST 1: NAME-SWAP at END position")
print("  A: ...John and Mary..., John gave... → Mary")
print("  B: ...Mary and John..., Mary gave... → John")
print("  Surfaces: name movers, S-inhibition")
print("=" * 100)

out_a, ids_a, cap_a = run_and_capture(PROMPT_A)
out_b, ids_b, cap_b = run_and_capture(PROMPT_B_NAMESWAP)

pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

heads_c1 = per_head_decomposition(cap_a, cap_b, END_POS)
heads_c1.sort(key=lambda x: -x[2])

# Classify top heads
print(f"\n  {'Rank':>4} {'Head':>6} {'Known':>6} {'Norm':>6} "
      f"{'attn→IO':>8} {'attn→S2':>8} {'attn→S1':>8} "
      f"{'A-pole (writes)':>40}")
print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*40}")

for rank, (L, h, norm, logits) in enumerate(heads_c1[:20], 1):
    info = classify_head(L, h, out_a.attentions, out_b.attentions, logits, END_POS,
                         IO_POS, S1_POS, S2_POS, AFTER_S1_POS)
    known = KNOWN.get((L, h), "")
    print(f"  {rank:>4} L{L}H{h:<2} {known:>6} {norm:>6.1f} "
          f"{info['attn_io']:>8.3f} {info['attn_s2']:>8.3f} {info['attn_s1']:>8.3f} "
          f"[{info['top_toks'][:38]}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# CONTRAST 2: Duplicate vs no-duplicate at S2 position
#   → surfaces duplicate-token heads
# ============================================================
print(f"\n{'='*100}")
print("CONTRAST 2: DUPLICATE vs NO-DUPLICATE at S2 position")
print("  A: When John and Mary..., John gave...")
print("  B: When Pete and Mary..., John gave...")
print("  Difference: S1='John' vs S1='Pete', S2='John' in both")
print("  At S2, duplicate-token heads fire in A (John matches) but not B (Pete≠John)")
print("=" * 100)

out_a, ids_a, cap_a = run_and_capture(PROMPT_A)
out_b, ids_b, cap_b = run_and_capture(PROMPT_B_NODUP)

tokens_b_nodup = [tok.decode([t]) for t in ids_b]
print(f"  Tokens B: {list(enumerate(tokens_b_nodup))}")
pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

# Read at S2 position — where duplicate-token heads write
heads_c2_s2 = per_head_decomposition(cap_a, cap_b, S2_POS)
heads_c2_s2.sort(key=lambda x: -x[2])

print(f"\n  Per-head decomposition at S2 (pos {S2_POS}):")
print(f"  {'Rank':>4} {'Head':>6} {'Known':>6} {'Norm':>6} "
      f"{'attn→S1':>8} {'attn→IO':>8} "
      f"{'A-pole (writes)':>40}")
print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*40}")

for rank, (L, h, norm, logits) in enumerate(heads_c2_s2[:20], 1):
    # Attention FROM S2 position
    a_weights = out_a.attentions[L][0, h, S2_POS, :]
    b_weights = out_b.attentions[L][0, h, S2_POS, :]
    attn_s1_a = float(a_weights[S1_POS])
    attn_s1_b = float(b_weights[S1_POS])  # to "Pete" position
    attn_io = (float(a_weights[IO_POS]) + float(b_weights[IO_POS])) / 2

    known = KNOWN.get((L, h), "")
    toks = tk(logits, 5)
    print(f"  {rank:>4} L{L}H{h:<2} {known:>6} {norm:>6.1f} "
          f" {attn_s1_a:>.3f}/{attn_s1_b:>.3f} {attn_io:>8.3f} "
          f"[{toks[:38]}]")

# Also read at END — duplicate signal propagates
heads_c2_end = per_head_decomposition(cap_a, cap_b, END_POS)
heads_c2_end.sort(key=lambda x: -x[2])

print(f"\n  Per-head decomposition at END (pos {END_POS}):")
print(f"  {'Rank':>4} {'Head':>6} {'Known':>6} {'Norm':>6} "
      f"{'attn→IO':>8} {'attn→S2':>8} {'attn→S1':>8} "
      f"{'A-pole (writes)':>40}")
print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*40}")

for rank, (L, h, norm, logits) in enumerate(heads_c2_end[:20], 1):
    a_weights = out_a.attentions[L][0, h, END_POS, :]
    b_weights = out_b.attentions[L][0, h, END_POS, :]
    attn_io = (float(a_weights[IO_POS]) + float(b_weights[IO_POS])) / 2
    attn_s2 = (float(a_weights[S2_POS]) + float(b_weights[S2_POS])) / 2
    attn_s1 = (float(a_weights[S1_POS]) + float(b_weights[S1_POS])) / 2

    known = KNOWN.get((L, h), "")
    toks = tk(logits, 5)
    print(f"  {rank:>4} L{L}H{h:<2} {known:>6} {norm:>6.1f} "
          f"{attn_io:>8.3f} {attn_s2:>8.3f} {attn_s1:>8.3f} "
          f"[{toks[:38]}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# CONTRAST 3: Repeated subject vs novel subject at END
#   → surfaces induction heads + downstream effects
# ============================================================
print(f"\n{'='*100}")
print("CONTRAST 3: REPEATED vs NOVEL SUBJECT at END position")
print("  A: When John and Mary..., John gave...")
print("  B: When John and Mary..., Sam gave...")
print("  Difference: S2='John' (repeated) vs S2='Sam' (novel)")
print("  Induction heads fire for repeated 'John' (A...A→ pattern)")
print("=" * 100)

out_a, ids_a, cap_a = run_and_capture(PROMPT_A)
out_b, ids_b, cap_b = run_and_capture(PROMPT_B_NOVELSUBJ)

tokens_b_novel = [tok.decode([t]) for t in ids_b]
print(f"  Tokens B: {list(enumerate(tokens_b_novel))}")
pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

# Read at S2 position first — induction and duplicate heads write here
# But S2 content differs between A and B (John vs Sam), so we need to be careful
# Actually the S2 position has different TOKENS in A and B, so the L0 difference
# is the token embedding difference, not a circuit effect.
# Better: read at END, where the induction signal has propagated.

heads_c3_end = per_head_decomposition(cap_a, cap_b, END_POS)
heads_c3_end.sort(key=lambda x: -x[2])

# For induction heads, check attention from END to after_S1
# (induction pattern: "John...John→" should attend to what follows first "John")
print(f"\n  Per-head decomposition at END:")
print(f"  {'Rank':>4} {'Head':>6} {'Known':>6} {'Norm':>6} "
      f"{'attn→aftS1':>10} {'attn→S2':>8} {'attn→IO':>8} "
      f"{'A-pole (writes)':>40}")
print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*40}")

for rank, (L, h, norm, logits) in enumerate(heads_c3_end[:20], 1):
    a_weights = out_a.attentions[L][0, h, END_POS, :]
    b_weights = out_b.attentions[L][0, h, END_POS, :]
    attn_after_s1 = (float(a_weights[AFTER_S1_POS]) + float(b_weights[AFTER_S1_POS])) / 2
    attn_s2 = (float(a_weights[S2_POS]) + float(b_weights[S2_POS])) / 2
    attn_io = (float(a_weights[IO_POS]) + float(b_weights[IO_POS])) / 2

    known = KNOWN.get((L, h), "")
    toks = tk(logits, 5)
    print(f"  {rank:>4} L{L}H{h:<2} {known:>6} {norm:>6.1f} "
          f"{attn_after_s1:>10.3f} {attn_s2:>8.3f} {attn_io:>8.3f} "
          f"[{toks[:38]}]")

# Also read at S2 — but note tokens differ, so L0 embedding diff is large
heads_c3_s2 = per_head_decomposition(cap_a, cap_b, S2_POS)
heads_c3_s2.sort(key=lambda x: -x[2])

print(f"\n  Per-head decomposition at S2 (pos {S2_POS}, John vs Sam):")
print(f"  {'Rank':>4} {'Head':>6} {'Known':>6} {'Norm':>6} "
      f"{'attn→S1(A/B)':>14} "
      f"{'A-pole (writes)':>40}")
print(f"  {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*14} {'─'*40}")

for rank, (L, h, norm, logits) in enumerate(heads_c3_s2[:20], 1):
    a_weights = out_a.attentions[L][0, h, S2_POS, :]
    b_weights = out_b.attentions[L][0, h, S2_POS, :]
    attn_s1_a = float(a_weights[S1_POS])
    attn_s1_b = float(b_weights[S1_POS])

    known = KNOWN.get((L, h), "")
    toks = tk(logits, 5)
    print(f"  {rank:>4} L{L}H{h:<2} {known:>6} {norm:>6.1f} "
          f" {attn_s1_a:>.3f}/{attn_s1_b:>.3f}   "
          f"[{toks[:38]}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# CROSS-CONTRAST SUMMARY: which heads appear across which contrasts?
# ============================================================
print(f"\n{'='*100}")
print("CROSS-CONTRAST SUMMARY")
print("=" * 100)

# Rerun all three to get consistent data
out_a_ref, _, cap_a_ref = run_and_capture(PROMPT_A)

out_b_ns, _, cap_b_ns = run_and_capture(PROMPT_B_NAMESWAP)
out_b_nd, _, cap_b_nd = run_and_capture(PROMPT_B_NODUP)
out_b_nv, _, cap_b_nv = run_and_capture(PROMPT_B_NOVELSUBJ)

c1_end = per_head_decomposition(cap_a_ref, cap_b_ns, END_POS)
c2_s2 = per_head_decomposition(cap_a_ref, cap_b_nd, S2_POS)
c2_end = per_head_decomposition(cap_a_ref, cap_b_nd, END_POS)
c3_end = per_head_decomposition(cap_a_ref, cap_b_nv, END_POS)
c3_s2 = per_head_decomposition(cap_a_ref, cap_b_nv, S2_POS)

# Build lookup: (L,H) → norm for each contrast
def norm_lookup(heads_list):
    return {(L, h): norm for L, h, norm, _ in heads_list}

c1_norms = norm_lookup(c1_end)
c2s2_norms = norm_lookup(c2_s2)
c2e_norms = norm_lookup(c2_end)
c3e_norms = norm_lookup(c3_end)
c3s2_norms = norm_lookup(c3_s2)

# Collect all heads, compute max norm across contrasts
all_heads = set()
for d in [c1_norms, c2s2_norms, c2e_norms, c3e_norms, c3s2_norms]:
    all_heads.update(d.keys())

# For each head, get attention pattern from the reference run
# Use name-swap attention for name movers, no-dup for duplicate heads
print(f"\n  All 13 known circuit heads — norm across contrasts and functional signature:")
print(f"  {'Head':>6} {'Role':>6} {'C1:END':>7} {'C2:S2':>7} {'C2:END':>7} "
      f"{'C3:END':>7} {'C3:S2':>7} {'Max':>7}  Attention & Readout")
print(f"  {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*7}  {'─'*50}")

for L, h in sorted(KNOWN.keys()):
    role = KNOWN[(L, h)]
    n1 = c1_norms.get((L, h), 0)
    n2s = c2s2_norms.get((L, h), 0)
    n2e = c2e_norms.get((L, h), 0)
    n3e = c3e_norms.get((L, h), 0)
    n3s = c3s2_norms.get((L, h), 0)
    mx = max(n1, n2s, n2e, n3e, n3s)

    # Attention from name-swap contrast (C1) at END
    a_end = out_a_ref.attentions[L][0, h, END_POS, :]
    attn_io_end = float(a_end[IO_POS])
    attn_s2_end = float(a_end[S2_POS])
    attn_s1_end = float(a_end[S1_POS])

    # Attention from C1 at S2
    a_s2 = out_a_ref.attentions[L][0, h, S2_POS, :]
    attn_s1_from_s2 = float(a_s2[S1_POS])

    # Best readout
    best_contrast = max([(n1, c1_end), (n2s, c2_s2), (n2e, c2_end),
                         (n3e, c3_end), (n3s, c3_s2)], key=lambda x: x[0])
    best_logits = None
    for ll, hh, nn, lg in best_contrast[1]:
        if ll == L and hh == h:
            best_logits = lg
            break

    readout = tk(best_logits, 4) if best_logits is not None else "?"

    # Determine which position this head primarily attends to
    if role in ("dup",):
        attn_desc = f"S2→S1: {attn_s1_from_s2:.2f}"
    elif role in ("s-inh",):
        attn_desc = f"END→S2: {attn_s2_end:.2f}"
    elif role in ("nm+", "nm-"):
        attn_desc = f"END→IO: {attn_io_end:.2f}"
    elif role in ("ind",):
        a_after = float(out_a_ref.attentions[L][0, h, END_POS, AFTER_S1_POS])
        attn_desc = f"END→aft_S1: {a_after:.2f}"
    else:
        attn_desc = ""

    star = " ✓" if mx > 5.0 else " ✗"
    print(f"  L{L}H{h:<2} {role:>6} {n1:>7.1f} {n2s:>7.1f} {n2e:>7.1f} "
          f"{n3e:>7.1f} {n3s:>7.1f} {mx:>7.1f}  {attn_desc:<20} [{readout}]{star}")


# Now: can we IDENTIFY these heads without knowing ground truth?
# Use a combined score: appear in top-N of ANY contrast
print(f"\n\n  BLIND IDENTIFICATION: top heads from each contrast, classified by signature")
print(f"  {'─'*100}")

# Collect top-10 from each contrast + position
top_from_c1_end = set((L, h) for L, h, _, _ in sorted(c1_end, key=lambda x: -x[2])[:12])
top_from_c2_s2 = set((L, h) for L, h, _, _ in sorted(c2_s2, key=lambda x: -x[2])[:12])
top_from_c2_end = set((L, h) for L, h, _, _ in sorted(c2_end, key=lambda x: -x[2])[:12])
top_from_c3_end = set((L, h) for L, h, _, _ in sorted(c3_end, key=lambda x: -x[2])[:12])
top_from_c3_s2 = set((L, h) for L, h, _, _ in sorted(c3_s2, key=lambda x: -x[2])[:12])

all_surfaced = top_from_c1_end | top_from_c2_s2 | top_from_c2_end | top_from_c3_end | top_from_c3_s2

print(f"\n  Total unique heads surfaced (top-12 from each of 5 readings): {len(all_surfaced)}")
print(f"  Known circuit heads surfaced: {len(all_surfaced & set(KNOWN.keys()))}/13")
print(f"  Known heads missed: {set(KNOWN.keys()) - all_surfaced}")

# Classify each surfaced head
print(f"\n  {'Head':>6} {'Known':>6} {'Inferred role':>20}  Evidence")
print(f"  {'─'*6} {'─'*6} {'─'*20}  {'─'*60}")

for L, h in sorted(all_surfaced):
    known = KNOWN.get((L, h), "?")

    # Evidence collection
    in_c1_end = (L, h) in top_from_c1_end
    in_c2_s2 = (L, h) in top_from_c2_s2
    in_c2_end = (L, h) in top_from_c2_end
    in_c3_end = (L, h) in top_from_c3_end
    in_c3_s2 = (L, h) in top_from_c3_s2

    # Attention pattern
    a_from_end = out_a_ref.attentions[L][0, h, END_POS, :]
    a_from_s2 = out_a_ref.attentions[L][0, h, S2_POS, :]

    attn_io_from_end = float(a_from_end[IO_POS])
    attn_s2_from_end = float(a_from_end[S2_POS])
    attn_s1_from_end = float(a_from_end[S1_POS])
    attn_s1_from_s2 = float(a_from_s2[S1_POS])
    attn_after_s1_from_end = float(a_from_end[AFTER_S1_POS])

    # Classification rules
    inferred = "?"
    evidence_parts = []

    if in_c1_end and attn_io_from_end > 0.25 and L >= 9:
        inferred = "name_mover"
        evidence_parts.append(f"C1:END, END→IO={attn_io_from_end:.2f}")
    elif in_c1_end and attn_s2_from_end > 0.25 and 7 <= L <= 8:
        inferred = "s_inhibition"
        evidence_parts.append(f"C1:END, END→S2={attn_s2_from_end:.2f}")
    elif in_c2_s2 and attn_s1_from_s2 > 0.15 and L <= 5:
        inferred = "duplicate_token"
        evidence_parts.append(f"C2:S2, S2→S1={attn_s1_from_s2:.2f}")
    elif in_c3_end and attn_after_s1_from_end > 0.05 and 4 <= L <= 7:
        inferred = "induction"
        evidence_parts.append(f"C3:END, END→aft_S1={attn_after_s1_from_end:.2f}")
    elif in_c2_s2 and L <= 3:
        inferred = "duplicate_token?"
        evidence_parts.append(f"C2:S2, L{L} (early)")
    elif in_c3_s2 and L <= 6:
        inferred = "induction?"
        evidence_parts.append(f"C3:S2, L{L}")
    else:
        contrasts = []
        if in_c1_end: contrasts.append("C1:END")
        if in_c2_s2: contrasts.append("C2:S2")
        if in_c2_end: contrasts.append("C2:END")
        if in_c3_end: contrasts.append("C3:END")
        if in_c3_s2: contrasts.append("C3:S2")
        evidence_parts.append(f"in {', '.join(contrasts)}")
        inferred = "other"

    # Add W_U readout from best contrast
    best_norm = 0
    best_logits = None
    for contrast_list in [c1_end, c2_s2, c2_end, c3_end, c3_s2]:
        for ll, hh, nn, lg in contrast_list:
            if ll == L and hh == h and nn > best_norm:
                best_norm = nn
                best_logits = lg

    if best_logits is not None:
        evidence_parts.append(f"writes: [{tk(best_logits, 3)}]")

    match = "✓" if known != "?" and (
        (known in ("nm+", "nm-") and "name_mover" in inferred) or
        (known == "s-inh" and "inhibition" in inferred) or
        (known == "dup" and "duplicate" in inferred) or
        (known == "ind" and "induction" in inferred)
    ) else ("·" if known == "?" else "✗")

    print(f"  L{L}H{h:<2} {known:>6} {inferred:>20}  {match} {'; '.join(evidence_parts)}")

del out_a_ref, out_b_ns, out_b_nd, out_b_nv
torch.cuda.empty_cache()

print(f"\n{'='*100}")
print("DONE")
print("=" * 100)
