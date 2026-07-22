"""
MLP desuperposition analysis.

Question: Does the 4x expansion in the MLP separate superposed signals
into different neurons?

Method:
1. Pick cases with known superposed signals (hot dog: structural "becoming"
   + content "edible"; grief/joy: emotional + social)
2. Hook into the MLP intermediate layer (after fc1 + GELU, before fc2)
   for BOTH inputs. Compute Δ_intermediate = act_c - act_k (10240-dim).
3. For each intermediate neuron, measure its contrastive activation.
4. Group neurons by what they "write" — each neuron's fc2 column is
   a direction in residual stream space. Project that column through W_U
   to see what token each neuron contributes.
5. Check: do neurons that write "food" tokens activate differently from
   neurons that write "animal" tokens? Do they use separate subsets?
6. Iterative: for a multi-signal case, do the top-k neurons for signal A
   overlap with top-k neurons for signal B?

If the MLP separates signals: different neurons should carry different
contrastive content, with minimal overlap between signal groups.
If not: the same neurons should carry mixed content.
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
d_model = model.config.hidden_size
d_inter = model.config.intermediate_size  # 10240
W_U = model.lm_head.weight.detach().float()

cos = F.cosine_similarity


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [tok.decode([int(idxs[j])]).strip()[:14] for j in range(k)]


def get_mlp_intermediates(text, layer_idx):
    """
    Hook into MLP at layer_idx, capture activations AFTER fc1+GELU
    (before fc2). Returns the 10240-dim intermediate activation at
    the last token position.
    """
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}

    def hook_fn(module, input, output):
        # input to MLP is the hidden state after attention+layernorm
        # We need the intermediate: fc1(input) -> GELU
        h_in = input[0] if isinstance(input, tuple) else input
        mlp = model.model.layers[layer_idx].mlp
        intermediate = mlp.activation_fn(mlp.fc1(h_in))
        captured['intermediate'] = intermediate[0, -1, :].detach().float()
        captured['input'] = h_in[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.register_forward_pre_hook(
        lambda m, inp: None  # dummy, we use forward hook instead
    )
    handle.remove()

    # Use forward hook on fc1 to get pre-GELU, or hook on MLP itself
    # Actually, let's hook on fc1's output and apply GELU manually
    fc1_out = {}

    def fc1_hook(module, input, output):
        fc1_out['pre_gelu'] = output[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(fc1_hook)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    handle.remove()

    # Apply GELU to get post-activation
    gelu = model.model.layers[layer_idx].mlp.activation_fn
    post_gelu = gelu(fc1_out['pre_gelu'])

    return post_gelu, fc1_out['pre_gelu'], out


ref_L = _sl(28)[0]  # Use L28 as before, but also test the MLP that writes at this layer
# The MLP at layer L writes INTO layer L's output (which becomes L+1's input)
# So the MLP at layer 27 writes the content we read at L28's hidden state
# Actually, hidden_states[L] is AFTER layer L-1's full processing
# Let's test multiple layers

# ══════════════════════════════════════════════════════════════
# CASE 1: HOT DOG — food vs animal signals
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("CASE 1: HOT DOG — do MLP neurons separate food from animal?")
print("=" * 70)

text_a = "The hot dog was"
text_b = "The cold dog was"

# Test at layers where we know the food signal is being written
test_layers = _sl(4, 8, 12, 16, 20, 24, 28)

for L in test_layers:
    print(f"\n  Layer {L}:")

    # Get fc2 weight for this layer: (2560, 10240) — each column is a neuron's write vector
    fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()  # (d_model, d_inter)

    # Get intermediate activations
    act_a, pre_a, out_a = get_mlp_intermediates(text_a, L)
    act_b, pre_b, out_b = get_mlp_intermediates(text_b, L)

    delta_act = act_a - act_b  # (10240,) — contrastive activation per neuron

    # Which neurons have the largest contrastive activation?
    abs_delta = delta_act.abs()
    topk_vals, topk_idx = torch.topk(abs_delta, 20)

    print(f"    Top 20 contrastive neurons (of {d_inter}):")
    print(f"    {'Neuron':>8} {'Δact':>8} {'act_a':>8} {'act_b':>8}  writes_to_W_U")

    for rank in range(20):
        n_idx = int(topk_idx[rank])
        d_a = float(act_a[n_idx])
        d_b = float(act_b[n_idx])
        d_diff = float(delta_act[n_idx])

        # What does this neuron write? Its fc2 column projected through W_U
        write_vec = fc2_w[:, n_idx]  # (d_model,)
        write_logits = write_vec @ W_U.T  # (vocab,)

        # Sign matters: if neuron activates more for A, it writes in + direction
        if d_diff > 0:
            write_toks = topk_tok(write_logits, 5)
        else:
            write_toks = topk_tok(-write_logits, 5)

        print(f"    {n_idx:>8} {d_diff:>+8.2f} {d_a:>8.2f} {d_b:>8.2f}  {write_toks}")

    # ── Neuron groups: food-writing vs animal-writing ──
    # Define "food neurons" and "animal neurons" by what they write
    # Project all neuron write vectors through W_U, find which ones
    # write food-related vs animal-related tokens

    # Get contrastive contribution of each neuron: delta_act[n] * fc2[:, n]
    # This is the actual vector each neuron contributes to the residual stream difference
    # Total MLP contrastive output = sum over all neurons of (delta_act[n] * fc2[:, n])

    # Verify: does sum of neuron contributions = total MLP Δ output?
    mlp_delta_reconstructed = (delta_act.to(DEV).unsqueeze(0) @ fc2_w.T.to(DEV)).squeeze()
    # Compare to actual MLP output difference
    h_a_post = out_a.hidden_states[L+1][0, -1, :].float() if L+1 <= NL else out_a.hidden_states[L][0, -1, :].float()
    h_b_post = out_b.hidden_states[L+1][0, -1, :].float() if L+1 <= NL else out_b.hidden_states[L][0, -1, :].float()

    # Read the neuron-reconstructed delta through W_U
    recon_logits = mlp_delta_reconstructed @ W_U.T
    print(f"\n    Reconstructed MLP Δ via neurons:")
    print(f"      +[{', '.join(topk_tok(recon_logits, 6))}]")
    print(f"      -[{', '.join(topk_tok(-recon_logits, 6))}]")

    # ── How many neurons account for 80% of the contrastive signal? ──
    sorted_abs, sorted_idx = torch.sort(abs_delta, descending=True)
    cumsum = torch.cumsum(sorted_abs, dim=0)
    total = cumsum[-1]

    for threshold in [0.5, 0.8, 0.9, 0.95]:
        n_needed = int((cumsum >= total * threshold).nonzero()[0]) + 1
        print(f"    Neurons for {threshold*100:.0f}% of |Δact|: {n_needed} / {d_inter} "
              f"({n_needed/d_inter*100:.1f}%)")

    del out_a, out_b
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# CASE 2: MULTI-SIGNAL — do different signals use different neurons?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 2: MULTI-SIGNAL SEPARATION")
print("=" * 70)
print("Two contrasts that should activate different MLP neurons:")
print("  Contrast A: hot dog vs cold dog (food vs animal)")
print("  Contrast B: hot dog vs hot cat (dog vs cat)")
print("If MLP separates signals, the neuron sets should differ.\n")

L = _sl(20)[0]  # mid-to-late layer where both signals are active
print(f"  Analyzing at Layer {L}:")

fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()

act_hotdog, _, _ = get_mlp_intermediates("The hot dog was", L)
act_colddog, _, _ = get_mlp_intermediates("The cold dog was", L)
act_hotcat, _, _ = get_mlp_intermediates("The hot cat was", L)

delta_food = act_hotdog - act_colddog    # food vs animal axis
delta_species = act_hotdog - act_hotcat  # dog vs cat axis

torch.cuda.empty_cache()

# Top neurons for each contrast
K = 100
_, top_food_idx = torch.topk(delta_food.abs(), K)
_, top_species_idx = torch.topk(delta_species.abs(), K)

food_set = set(top_food_idx.tolist())
species_set = set(top_species_idx.tolist())
overlap = food_set & species_set

print(f"  Top {K} food neurons ∩ Top {K} species neurons: {len(overlap)} overlap")
print(f"  Food-only: {len(food_set - species_set)}")
print(f"  Species-only: {len(species_set - food_set)}")
print(f"  Shared: {len(overlap)}")

# What do the overlapping neurons write?
if overlap:
    print(f"\n  Overlapping neurons write:")
    for n_idx in list(overlap)[:10]:
        write_vec = fc2_w[:, n_idx]
        write_logits = write_vec @ W_U.T
        toks = topk_tok(write_logits, 5)
        f_act = float(delta_food[n_idx])
        s_act = float(delta_species[n_idx])
        print(f"    N{n_idx:>5}: food Δ={f_act:>+6.2f}  species Δ={s_act:>+6.2f}  writes={toks}")

# What do the food-only neurons write?
print(f"\n  Food-only neurons write (top 10 by |Δact|):")
food_only = sorted(food_set - species_set, key=lambda n: -abs(float(delta_food[n])))
for n_idx in food_only[:10]:
    write_vec = fc2_w[:, n_idx]
    write_logits = write_vec @ W_U.T
    d = float(delta_food[n_idx])
    toks = topk_tok(write_logits if d > 0 else -write_logits, 5)
    print(f"    N{n_idx:>5}: Δ={d:>+6.2f}  writes={toks}")

print(f"\n  Species-only neurons write (top 10 by |Δact|):")
species_only = sorted(species_set - food_set, key=lambda n: -abs(float(delta_species[n])))
for n_idx in species_only[:10]:
    write_vec = fc2_w[:, n_idx]
    write_logits = write_vec @ W_U.T
    d = float(delta_species[n_idx])
    toks = topk_tok(write_logits if d > 0 else -write_logits, 5)
    print(f"    N{n_idx:>5}: Δ={d:>+6.2f}  writes={toks}")

# ── Cosine between the two neuron activation patterns ──
c = float(cos(delta_food.unsqueeze(0), delta_species.unsqueeze(0)))
print(f"\n  Cosine between food and species neuron patterns: {c:+.3f}")

# ── Causal verification: inject only food-neurons' contribution ──
print(f"\n  Causal test: inject only food-exclusive neurons' contribution")
food_only_contribution = torch.zeros(d_model, device=DEV)
for n_idx in food_only:
    food_only_contribution += float(delta_food[n_idx]) * fc2_w[:, n_idx].to(DEV)

species_only_contribution = torch.zeros(d_model, device=DEV)
for n_idx in species_only:
    species_only_contribution += float(delta_species[n_idx]) * fc2_w[:, n_idx].to(DEV)

# Read through W_U
food_logits = food_only_contribution @ W_U.T
species_logits = species_only_contribution @ W_U.T
print(f"  Food-only neurons write: +[{', '.join(topk_tok(food_logits, 8))}]")
print(f"                          -[{', '.join(topk_tok(-food_logits, 8))}]")
print(f"  Species-only neurons write: +[{', '.join(topk_tok(species_logits, 8))}]")
print(f"                              -[{', '.join(topk_tok(-species_logits, 8))}]")


# ══════════════════════════════════════════════════════════════
# CASE 3: SCALING — does 4x expansion give room for 4 signals?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 3: CAPACITY — how many independent signals fit?")
print("=" * 70)

# Use 4 contrasts against "The hot dog was" and check neuron overlap
L = _sl(24)[0]
fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()

signals = [
    ("food",    "The hot dog was", "The cold dog was"),
    ("species", "The hot dog was", "The hot cat was"),
    ("age",     "The hot dog was", "The old dog was"),
    ("temp",    "The hot dog was", "The hot rod was"),
]

signal_deltas = {}
for label, ta, tb in signals:
    act_a, _, _ = get_mlp_intermediates(ta, L)
    act_b, _, _ = get_mlp_intermediates(tb, L)
    signal_deltas[label] = act_a - act_b
    torch.cuda.empty_cache()

K = 200
signal_sets = {}
for label in signal_deltas:
    _, top_idx = torch.topk(signal_deltas[label].abs(), K)
    signal_sets[label] = set(top_idx.tolist())

print(f"\n  Top {K} neuron overlap matrix at L{L}:")
slabels = list(signal_sets.keys())
print(f"  {'':>10}", end="")
for sl in slabels:
    print(f" {sl:>8}", end="")
print()
for sl1 in slabels:
    print(f"  {sl1:>10}", end="")
    for sl2 in slabels:
        overlap = len(signal_sets[sl1] & signal_sets[sl2])
        print(f" {overlap:>8}", end="")
    print()

# Pairwise cosine in neuron activation space
print(f"\n  Cosine between neuron activation patterns:")
for i, sl1 in enumerate(slabels):
    for j, sl2 in enumerate(slabels):
        if j > i:
            c = float(cos(signal_deltas[sl1].unsqueeze(0),
                           signal_deltas[sl2].unsqueeze(0)))
            print(f"    {sl1:>8} × {sl2:<8}: {c:+.3f}")

# How many neurons are UNIQUE to each signal?
all_neurons = set()
for s in signal_sets.values():
    all_neurons |= s
print(f"\n  Total unique neurons across all 4 signals: {len(all_neurons)} / {d_inter}")

for label in slabels:
    others = set()
    for other_label, other_set in signal_sets.items():
        if other_label != label:
            others |= other_set
    unique = signal_sets[label] - others
    print(f"    {label:>10}-exclusive: {len(unique)} neurons")
    if unique:
        # What do these exclusive neurons write?
        sorted_unique = sorted(unique, key=lambda n: -abs(float(signal_deltas[label][n])))
        top_writes = []
        for n_idx in sorted_unique[:5]:
            write_vec = fc2_w[:, n_idx]
            d = float(signal_deltas[label][n_idx])
            write_logits = write_vec @ W_U.T
            toks = topk_tok(write_logits if d > 0 else -write_logits, 3)
            top_writes.append(f"N{n_idx}→{toks[0]}")
        print(f"                       top writes: {', '.join(top_writes)}")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
