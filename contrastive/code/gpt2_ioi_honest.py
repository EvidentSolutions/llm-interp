"""
What can contrastive projection tell us about IOI in GPT-2,
using ONLY W_U readability?

No attention patterns. No known-circuit labels. No classification rules.

Three contrasts designed from the TASK structure (not circuit knowledge):
  C1: Name-swap — what changes when we swap who gave and who received?
  C2: Dup vs no-dup — what changes when the giver's name isn't repeated?
  C3: Novel subject — what changes when a different person is the giver?

For each: per-head decomposition, W_U readout, that's it.

Usage: .venv/Scripts/python.exe contrastive/code/gpt2_ioi_honest.py
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


def per_head_at(cap_a, cap_b, pos):
    results = []
    for L in range(NL):
        W = model.transformer.h[L].attn.c_proj.weight.float().cpu()
        for h in range(NH):
            d = cap_a[L][0, pos, h*HD:(h+1)*HD] - cap_b[L][0, pos, h*HD:(h+1)*HD]
            contrib = d @ W[h*HD:(h+1)*HD, :]
            norm = float(contrib.norm())
            logits = contrib @ W_U.cpu().T
            results.append((L, h, norm, logits))
    results.sort(key=lambda x: -x[2])
    return results


def show_heads(heads, n=15, label=""):
    print(f"\n  {label}")
    print(f"  {'#':>2} {'Head':>6} {'Norm':>6}  {'A-pole (top 5)':>45}  {'B-pole (top 5)':>45}")
    print(f"  {'─'*2} {'─'*6} {'─'*6}  {'─'*45}  {'─'*45}")
    for rank, (L, h, norm, logits) in enumerate(heads[:n], 1):
        a_pole = tk(logits, 5)
        b_pole = tk(-logits, 5)
        print(f"  {rank:>2} L{L}H{h:<2} {norm:>6.1f}  {a_pole:>45}  {b_pole:>45}")


# ============================================================
# Prompts — designed from task structure
# ============================================================

# The IOI task: "When A and B went..., A gave... to" → predict B
# A appears twice (subject), B appears once (indirect object)

PA = "When John and Mary went to the store, John gave a drink to"
ids_ref = tok(PA, add_special_tokens=False)["input_ids"]
tokens = [tok.decode([t]) for t in ids_ref]
print(f"Prompt A: {PA}")
print(f"Tokens:   {list(enumerate(tokens))}")

# Positions: S1=1(John), IO=3(Mary), S2=9(John), END=13(to)
S1, IO, S2, END = 1, 3, 9, 13
print(f"S1={S1}({tokens[S1]}), IO={IO}({tokens[IO]}), S2={S2}({tokens[S2]}), END={END}({tokens[END]})")


# ============================================================
# C1: NAME-SWAP — what if we swap who is the giver vs receiver?
# ============================================================
print(f"\n{'='*110}")
print("C1: NAME-SWAP")
print("  A: When John and Mary went..., John gave... to  →  predicts Mary")
print("  B: When Mary and John went..., Mary gave... to  →  predicts John")
print("  Question: which heads carry the name that differs?")
print("=" * 110)

PB1 = "When Mary and John went to the store, Mary gave a drink to"
out_a, _, cap_a = run_capture(PA)
out_b, _, cap_b = run_capture(PB1)
pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

heads_c1 = per_head_at(cap_a, cap_b, END)
show_heads(heads_c1, n=15, label="Per-head at END — read through W_U:")

# Also show trajectory
print(f"\n  Layer-by-layer contrastive trajectory at END:")
for L in range(NL + 1):
    h_a = out_a.hidden_states[L][0, -1, :].float()
    h_b = out_b.hidden_states[L][0, -1, :].float()
    dh = h_a - h_b
    ld = dh @ W_U.T
    print(f"    L{L:>2}  A-pole=[{tk(ld)}]  B-pole=[{tk(-ld)}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# C2: DUPLICATE vs NO-DUPLICATE
# ============================================================
print(f"\n{'='*110}")
print("C2: DUPLICATE vs NO-DUPLICATE")
print("  A: When John and Mary went..., John gave... to     (John repeated)")
print("  B: When Pete and Mary went..., John gave... to     (Pete ≠ John, no repeat)")
print("  Question: which heads respond to the repeated name?")
print("=" * 110)

PB2 = "When Pete and Mary went to the store, John gave a drink to"
out_a, _, cap_a = run_capture(PA)
out_b, _, cap_b = run_capture(PB2)
pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

print(f"\n  Reading at S2 (pos {S2}) — where repetition is detected:")
heads_c2_s2 = per_head_at(cap_a, cap_b, S2)
show_heads(heads_c2_s2, n=15, label="Per-head at S2:")

print(f"\n  Reading at END (pos {END}) — does the repetition signal propagate?")
heads_c2_end = per_head_at(cap_a, cap_b, END)
show_heads(heads_c2_end, n=15, label="Per-head at END:")

# Trajectory at END
print(f"\n  Layer-by-layer contrastive trajectory at END:")
for L in range(NL + 1):
    h_a = out_a.hidden_states[L][0, -1, :].float()
    h_b = out_b.hidden_states[L][0, -1, :].float()
    dh = h_a - h_b
    ld = dh @ W_U.T
    print(f"    L{L:>2}  A-pole=[{tk(ld)}]  B-pole=[{tk(-ld)}]")

# Trajectory at S2
print(f"\n  Layer-by-layer contrastive trajectory at S2:")
for L in range(NL + 1):
    h_a = out_a.hidden_states[L][0, S2, :].float()
    h_b = out_b.hidden_states[L][0, S2, :].float()
    dh = h_a - h_b
    ld = dh @ W_U.T
    print(f"    L{L:>2}  A-pole=[{tk(ld)}]  B-pole=[{tk(-ld)}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# C3: REPEATED SUBJECT vs NOVEL SUBJECT
# ============================================================
print(f"\n{'='*110}")
print("C3: REPEATED SUBJECT vs NOVEL SUBJECT")
print("  A: When John and Mary went..., John gave... to     (John repeated)")
print("  B: When John and Mary went..., Sam gave... to      (Sam is new)")
print("  Question: what changes when the giver is a new name vs a repeated one?")
print("=" * 110)

PB3 = "When John and Mary went to the store, Sam gave a drink to"
out_a, _, cap_a = run_capture(PA)
out_b, _, cap_b = run_capture(PB3)
pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()
print(f"  Predictions: A→'{pred_a}', B→'{pred_b}'")

heads_c3_end = per_head_at(cap_a, cap_b, END)
show_heads(heads_c3_end, n=15, label="Per-head at END:")

heads_c3_s2 = per_head_at(cap_a, cap_b, S2)
show_heads(heads_c3_s2, n=15, label="Per-head at S2 (John vs Sam — different tokens):")

# Trajectory at END
print(f"\n  Layer-by-layer contrastive trajectory at END:")
for L in range(NL + 1):
    h_a = out_a.hidden_states[L][0, -1, :].float()
    h_b = out_b.hidden_states[L][0, -1, :].float()
    dh = h_a - h_b
    ld = dh @ W_U.T
    print(f"    L{L:>2}  A-pole=[{tk(ld)}]  B-pole=[{tk(-ld)}]")

del out_a, out_b
torch.cuda.empty_cache()


# ============================================================
# CROSS-CONTRAST: consistency check
# ============================================================
print(f"\n{'='*110}")
print("CROSS-CONTRAST: do the same heads appear across name pairs?")
print("=" * 110)

pairs = [
    ("When John and Mary went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to",
     "John/Mary"),
    ("When Alice and Bob went to the park, Alice gave a gift to",
     "When Bob and Alice went to the park, Bob gave a gift to",
     "Alice/Bob"),
    ("When Dan and Eve ate dinner together, Dan passed the salt to",
     "When Eve and Dan ate dinner together, Eve passed the salt to",
     "Dan/Eve"),
]

# For each pair: find top-10 heads by norm, record which heads appear
from collections import Counter
head_counts = Counter()
head_readouts = {}

for pa, pb, label in pairs:
    out_a, _, cap_a = run_capture(pa)
    out_b, _, cap_b = run_capture(pb)
    pred_a = tok.decode([out_a.logits[0, -1].argmax().item()]).strip()
    pred_b = tok.decode([out_b.logits[0, -1].argmax().item()]).strip()

    # Find END position (last token)
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    end = len(ids_a) - 1

    heads = per_head_at(cap_a, cap_b, end)

    print(f"\n  {label}: A→'{pred_a}', B→'{pred_b}'")
    print(f"  Top-10 heads at END:")
    for rank, (L, h, norm, logits) in enumerate(heads[:10], 1):
        a_pole = tk(logits, 4)
        b_pole = tk(-logits, 4)
        head_counts[(L, h)] += 1
        head_readouts[(L, h, label)] = (a_pole, b_pole)
        print(f"    {rank:>2}. L{L}H{h:<2} ({norm:>5.1f})  A=[{a_pole}]  B=[{b_pole}]")

    del out_a, out_b
    torch.cuda.empty_cache()

# Heads that appear in top-10 for ALL three pairs
print(f"\n  Heads in top-10 for all 3 pairs:")
for (L, h), count in head_counts.most_common():
    if count == 3:
        readouts = []
        for label in ["John/Mary", "Alice/Bob", "Dan/Eve"]:
            key = (L, h, label)
            if key in head_readouts:
                a, b = head_readouts[key]
                readouts.append(f"{label}: A=[{a}]")
        print(f"  L{L}H{h}")
        for r in readouts:
            print(f"    {r}")

print(f"\n{'='*110}")
print("DONE")
print("=" * 110)
