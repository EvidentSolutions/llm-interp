"""
Per-head contrastive analysis for IOI, factual recall, and successor heads.
Decompose the attention output by head to identify which heads carry
the critical contrastive information.

Approach: hook each layer's attention dense (output) projection input
to get the concatenated per-head outputs, then split by head and multiply
by the corresponding W_O slice.

Usage: .venv/Scripts/python.exe contrastive/code/per_head_analysis.py
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

def _sl(*layers):
    """Scale layer indices from 32-layer base to current NL."""
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))

NH = model.config.num_attention_heads
HD = model.config.hidden_size // NH
HIDDEN = model.config.hidden_size
W_U = model.lm_head.weight.detach()

# Phi-2 uses 'dense', Phi-3/4 uses 'o_proj' for the attention output projection
_sample_attn = model.model.layers[0].self_attn
DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"


def tk(logits, k=5):
    v, i = torch.topk(logits, k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def get_per_head_contributions(input_ids, layers_of_interest):
    """
    Get per-head attention contributions to the residual stream.
    Hook the dense (output projection) input to capture concatenated
    head outputs, then split and project through W_O per head.
    """
    captured = {}
    hooks = []

    for L in layers_of_interest:
        layer_mod = model.model.layers[L]
        dense = getattr(layer_mod.self_attn, DENSE_ATTR)  # output projection

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                # inp[0] is the concatenated head output: (batch, seq, hidden)
                captured[layer_idx] = inp[0].detach().float()
            return hook_fn

        h = dense.register_forward_hook(make_hook(L))
        hooks.append(h)

    ids_tensor = torch.tensor([input_ids], device=DEV)
    with torch.no_grad():
        out = model(ids_tensor, output_hidden_states=True)

    for h in hooks:
        h.remove()

    # Now decompose per head
    result = {}
    for L in layers_of_interest:
        dense_input = captured[L][0]  # (seq, hidden)
        W_O = getattr(model.model.layers[L].self_attn, DENSE_ATTR).weight.float()  # (hidden, hidden)
        b_O = getattr(model.model.layers[L].self_attn, DENSE_ATTR).bias
        if b_O is not None:
            b_O = b_O.float()

        per_head = []
        for h in range(NH):
            # Each head contributes dense_input[:, h*HD:(h+1)*HD] @ W_O[:, h*HD:(h+1)*HD].T
            head_in = dense_input[:, h * HD : (h + 1) * HD]  # (seq, HD)
            W_O_h = W_O[:, h * HD : (h + 1) * HD]  # (hidden, HD)
            head_out = head_in @ W_O_h.T  # (seq, hidden)
            per_head.append(head_out)

        result[L] = torch.stack(per_head)  # (NH, seq, hidden)

    return result, out


def per_head_contrastive(pa, pb, layers, read_pos=-1, top_heads=8):
    """Run both prompts, get per-head outputs, compute contrastive projection."""
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    heads_a, out_a = get_per_head_contributions(ids_a, layers)
    heads_b, out_b = get_per_head_contributions(ids_b, layers)

    print(f'\n  A: "{pa}"')
    print(f'  B: "{pb}"')

    for L in layers:
        ha = heads_a[L][:, read_pos, :]  # (NH, hidden)
        hb = heads_b[L][:, read_pos, :]
        dh_heads = ha - hb  # (NH, hidden)
        norms = dh_heads.norm(dim=1)

        # Full layer contrast for reference
        h_a_full = out_a.hidden_states[L + 1][0, read_pos, :].float()
        h_b_full = out_b.hidden_states[L + 1][0, read_pos, :].float()
        dh_full = h_a_full - h_b_full
        ld_full = dh_full @ W_U.float().T
        print(f"\n  L{L:>2} full: [{tk(ld_full)}]  "
              f"(norm={float(dh_full.norm()):.1f})")

        sorted_heads = torch.argsort(norms, descending=True)
        for rank in range(min(top_heads, NH)):
            h = int(sorted_heads[rank])
            n = float(norms[h])
            ld_h = dh_heads[h] @ W_U.float().T
            print(f"    H{h:>2} (norm={n:>6.1f}): [{tk(ld_h)}]")

    del out_a, out_b
    torch.cuda.empty_cache()


# ============================================================
print("=" * 100)
print("1. IOI — Per-head decomposition")
print("=" * 100)

per_head_contrastive(
    "John and Mary went to the store. John gave a book to",
    "Mary and John went to the store. Mary gave a book to",
    layers=_sl(8, 16, 24, 28, 31),
)

per_head_contrastive(
    "Alice and Bob went to the park. Alice gave a gift to",
    "Bob and Alice went to the park. Bob gave a gift to",
    layers=_sl(24, 28, 31),
)

per_head_contrastive(
    "Dan and Eve ate dinner together. Dan passed the salt to",
    "Eve and Dan ate dinner together. Eve passed the salt to",
    layers=_sl(24, 28, 31),
)

# ============================================================
print(f"\n{'='*100}")
print("2. FACTUAL RECALL — Per-head decomposition")
print("=" * 100)

per_head_contrastive(
    "The Eiffel Tower is located in",
    "The Colosseum is located in",
    layers=_sl(16, 20, 24, 28, 31),
)

per_head_contrastive(
    "The capital of France is",
    "The capital of Japan is",
    layers=_sl(16, 20, 24, 28, 31),
)

per_head_contrastive(
    "Shakespeare wrote",
    "Tolstoy wrote",
    layers=_sl(16, 20, 24, 28, 31),
)

# ============================================================
print(f"\n{'='*100}")
print("3. SUCCESSOR — Per-head decomposition")
print("=" * 100)

per_head_contrastive(
    "After Monday comes",
    "After Tuesday comes",
    layers=_sl(16, 20, 24, 28, 31),
)

per_head_contrastive(
    "After Saturday comes",
    "After Sunday comes",
    layers=_sl(16, 20, 24, 28, 31),
)

per_head_contrastive(
    "After December comes",
    "After January comes",
    layers=_sl(16, 20, 24, 28, 31),
)

# ============================================================
print(f"\n{'='*100}")
print("4. SUCCESSOR DISCONTINUITY — head norms across all day pairs")
print("=" * 100)

days = ["Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"]

# Find which heads are consistently important across day pairs
all_day_norms = {}
for i in range(len(days)):
    d1 = days[i]
    d2 = days[(i + 1) % len(days)]
    pa = f"After {d1} comes"
    pb = f"After {d2} comes"
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]

    heads_a, out_a = get_per_head_contributions(ids_a, _sl(28))
    heads_b, out_b = get_per_head_contributions(ids_b, _sl(28))

    dh = heads_a[_sl(28)[0]][:, -1, :] - heads_b[_sl(28)[0]][:, -1, :]
    norms = dh.norm(dim=1)
    all_day_norms[f"{d1}->{d2}"] = norms

    del out_a, out_b

# Print as table — only top 10 heads by mean norm
mean_norms = torch.stack(list(all_day_norms.values())).mean(dim=0)
top_heads = torch.argsort(mean_norms, descending=True)[:12]

print(f"\n  L{_sl(28)[0]} head norms (top 12 heads by mean, all day pairs):")
print(f"  {'Pair':<20}", end="")
for h in top_heads:
    print(f"  H{int(h):>2}", end="")
print("  | total")

for pair_name, norms in all_day_norms.items():
    print(f"  {pair_name:<20}", end="")
    for h in top_heads:
        print(f" {float(norms[int(h)]):>4.0f}", end="")
    print(f"  | {float(norms.sum()):.0f}")

print(f"\n  Mean:              ", end="")
for h in top_heads:
    print(f" {float(mean_norms[int(h)]):>4.0f}", end="")
print()

# Which heads read what for the wrap-around?
print(f"\n  Sat->Sun vs Mon->Tue head content comparison (L{_sl(28)[0]}):")
for label, pair in [("Mon->Tue", ("After Monday comes", "After Tuesday comes")),
                     ("Sat->Sun", ("After Saturday comes", "After Sunday comes")),
                     ("Sun->Mon", ("After Sunday comes", "After Monday comes"))]:
    pa, pb = pair
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    heads_a, out_a = get_per_head_contributions(ids_a, _sl(28))
    heads_b, out_b = get_per_head_contributions(ids_b, _sl(28))
    dh = heads_a[_sl(28)[0]][:, -1, :] - heads_b[_sl(28)[0]][:, -1, :]
    norms = dh.norm(dim=1)
    top3 = torch.argsort(norms, descending=True)[:3]
    print(f"\n  {label}:")
    for h in top3:
        h = int(h)
        ld = dh[h] @ W_U.float().T
        print(f"    H{h:>2} (norm={float(norms[h]):.0f}): [{tk(ld)}]")
    del out_a, out_b

torch.cuda.empty_cache()

# ============================================================
print(f"\n{'='*100}")
print("5. MONTH WRAP — December->January per-head")
print("=" * 100)

months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Compare adjacent vs wrap
for label, m1, m2 in [("Jan->Feb", "January", "February"),
                       ("Jun->Jul", "June", "July"),
                       ("Nov->Dec", "November", "December"),
                       ("Dec->Jan", "December", "January")]:
    pa = f"After {m1} comes"
    pb = f"After {m2} comes"
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    heads_a, out_a = get_per_head_contributions(ids_a, _sl(28))
    heads_b, out_b = get_per_head_contributions(ids_b, _sl(28))
    dh = heads_a[_sl(28)[0]][:, -1, :] - heads_b[_sl(28)[0]][:, -1, :]
    norms = dh.norm(dim=1)
    top3 = torch.argsort(norms, descending=True)[:3]
    print(f"\n  {label}:")
    for h in top3:
        h = int(h)
        ld = dh[h] @ W_U.float().T
        print(f"    H{h:>2} (norm={float(norms[h]):.0f}): [{tk(ld)}]")
    del out_a, out_b

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
