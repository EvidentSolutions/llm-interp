"""
MLP read→gate→write decomposition.

Hypothesis: fc1 rows are input DETECTORS, GELU is an if-block,
fc2 columns are token-shaped WRITES. The expansion creates a bank
of detectors, each of which fires when a specific input pattern
is present and writes a specific output.

Test:
1. For each neuron, compute what it READS (fc1 row projected against
   known input directions) and what it WRITES (fc2 column through W_U)
2. Check: do neurons that read "food input" write "food output"?
3. Check: does GELU gate them correctly? (fire for hot_dog, zero for cold_dog)
4. Cross-signal: do neurons that read "food" vs "species" use different
   detectors and writers?
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
d_inter = model.config.intermediate_size
W_U = model.lm_head.weight.detach().float()

cos = F.cosine_similarity


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [tok.decode([int(idxs[j])]).strip()[:14] for j in range(k)]


def get_h_and_mlp_input(text, layer_idx):
    """Get hidden states and the actual input to the MLP at this layer."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    mlp_input = {}

    def hook_fn(module, args):
        # Pre-hook: args[0] is the input to the MLP
        inp = args[0] if isinstance(args, tuple) else args
        mlp_input['x'] = inp[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    handle.remove()
    return out, mlp_input['x']


def get_post_gelu(text, layer_idx):
    """Get post-GELU activations (10240-dim) at last token."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    fc1_out = {}

    def hook_fn(module, input, output):
        fc1_out['pre_gelu'] = output[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV), output_hidden_states=True)
    handle.remove()
    gelu = model.model.layers[layer_idx].mlp.activation_fn
    return gelu(fc1_out['pre_gelu']), fc1_out['pre_gelu'], out


ref_L = _sl(20)[0]  # Layer where food signal is being actively written

# ══════════════════════════════════════════════════════════════
# STEP 1: Get the input contrastive direction at the MLP input
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print(f"STEP 1: MLP INPUT ANALYSIS at Layer {ref_L}")
print("=" * 70)

text_a = "The hot dog was"
text_b = "The cold dog was"
text_c = "The hot cat was"

out_a, mlp_in_a = get_h_and_mlp_input(text_a, ref_L)
out_b, mlp_in_b = get_h_and_mlp_input(text_b, ref_L)
out_c, mlp_in_c = get_h_and_mlp_input(text_c, ref_L)

# Input contrastive directions
delta_in_food = mlp_in_a - mlp_in_b      # food vs animal input
delta_in_species = mlp_in_a - mlp_in_c   # dog vs cat input

# What does the input contrast read as through W_U?
food_in_logits = delta_in_food @ W_U.T
species_in_logits = delta_in_species @ W_U.T

print(f"\n  MLP input Δ (food axis) through W_U:")
print(f"    +[{', '.join(topk_tok(food_in_logits, 8))}]")
print(f"    -[{', '.join(topk_tok(-food_in_logits, 8))}]")
print(f"\n  MLP input Δ (species axis) through W_U:")
print(f"    +[{', '.join(topk_tok(species_in_logits, 8))}]")
print(f"    -[{', '.join(topk_tok(-species_in_logits, 8))}]")

input_cos = float(cos(delta_in_food.unsqueeze(0), delta_in_species.unsqueeze(0)))
print(f"\n  Cosine between food and species input directions: {input_cos:+.3f}")

del out_a, out_b, out_c
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# STEP 2: For each neuron, measure READ alignment and WRITE content
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"STEP 2: READ→GATE→WRITE decomposition at Layer {ref_L}")
print("=" * 70)

fc1_w = model.model.layers[ref_L].mlp.fc1.weight.detach().float()  # (10240, 2560)
fc1_b = model.model.layers[ref_L].mlp.fc1.bias.detach().float()    # (10240,)
fc2_w = model.model.layers[ref_L].mlp.fc2.weight.detach().float()  # (2560, 10240)

# Get post-GELU activations
act_a, pre_a, _ = get_post_gelu(text_a, ref_L)
act_b, pre_b, _ = get_post_gelu(text_b, ref_L)
act_c, pre_c, _ = get_post_gelu(text_c, ref_L)

torch.cuda.empty_cache()

# For each neuron:
# READ = fc1_w[n, :] — what input pattern does it detect?
# PRE_GELU = fc1_w[n, :] @ input + fc1_b[n] — how strongly does it fire pre-gate?
# POST_GELU = GELU(pre_gelu) — does it pass the gate?
# WRITE = fc2_w[:, n] — what direction does it push the output?

# Compute READ alignment with food and species input directions
food_dir_norm = delta_in_food / delta_in_food.norm()
species_dir_norm = delta_in_species / delta_in_species.norm()

# Each fc1 row dotted with the input direction = how much this neuron
# responds to that specific input contrast
read_food = fc1_w @ food_dir_norm.to(DEV)      # (10240,) — sensitivity to food input
read_species = fc1_w @ species_dir_norm.to(DEV) # (10240,) — sensitivity to species input

# Write content: project each neuron's fc2 column through W_U
# fc2_w is (2560, 10240), so fc2_w[:, n] is the write vector for neuron n
write_logits = fc2_w.T @ W_U.T  # (10240, vocab) — each row is a neuron's write in token space

# Contrastive activation (post-GELU)
delta_act_food = act_a - act_b
delta_act_species = act_a - act_c

print(f"\n  Neuron statistics:")
print(f"    Neurons with |read_food| > 0.5:    {int((read_food.abs() > 0.5).sum())}")
print(f"    Neurons with |read_species| > 0.5: {int((read_species.abs() > 0.5).sum())}")

# ── Find neurons that READ food and check what they WRITE ──
print(f"\n  TOP FOOD-READING NEURONS (sorted by |read alignment with food input|):")
print(f"  {'Neuron':>8} {'read_f':>8} {'read_s':>8} {'Δact_f':>8} {'Δact_s':>8} {'pre_a':>7} {'pre_b':>7}  gate_a gate_b  writes")

top_food_readers = torch.topk(read_food.abs(), 30)
for rank in range(30):
    n = int(top_food_readers.indices[rank])
    rf = float(read_food[n])
    rs = float(read_species[n])
    da_f = float(delta_act_food[n])
    da_s = float(delta_act_species[n])
    pa = float(pre_a[n])
    pb = float(pre_b[n])
    ga = float(act_a[n])
    gb = float(act_b[n])

    # What does this neuron write?
    w_logits = write_logits[n]
    # Sign: if neuron reads positive food input and fires, it writes in + direction
    if rf > 0:
        w_toks = topk_tok(w_logits, 4)
    else:
        w_toks = topk_tok(-w_logits, 4)

    gate_a = "ON " if abs(ga) > 0.01 else "off"
    gate_b = "ON " if abs(gb) > 0.01 else "off"

    if rank < 20 or abs(da_f) > 1.0:
        print(f"  {n:>8} {rf:>+8.3f} {rs:>+8.3f} {da_f:>+8.2f} {da_s:>+8.2f} {pa:>+7.2f} {pb:>+7.2f}  {gate_a}    {gate_b}    {w_toks}")


print(f"\n  TOP SPECIES-READING NEURONS:")
print(f"  {'Neuron':>8} {'read_f':>8} {'read_s':>8} {'Δact_f':>8} {'Δact_s':>8} {'pre_a':>7} {'pre_c':>7}  gate_a gate_c  writes")

top_species_readers = torch.topk(read_species.abs(), 20)
for rank in range(20):
    n = int(top_species_readers.indices[rank])
    rf = float(read_food[n])
    rs = float(read_species[n])
    da_f = float(delta_act_food[n])
    da_s = float(delta_act_species[n])
    pa = float(pre_a[n])
    pc = float(pre_c[n])
    ga = float(act_a[n])
    gc = float(act_c[n])

    if rs > 0:
        w_toks = topk_tok(write_logits[n], 4)
    else:
        w_toks = topk_tok(-write_logits[n], 4)

    gate_a = "ON " if abs(ga) > 0.01 else "off"
    gate_c = "ON " if abs(gc) > 0.01 else "off"

    print(f"  {n:>8} {rf:>+8.3f} {rs:>+8.3f} {da_f:>+8.2f} {da_s:>+8.2f} {pa:>+7.2f} {pc:>+7.2f}  {gate_a}    {gate_c}    {w_toks}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Read-write coherence — do food-readers write food?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: READ-WRITE COHERENCE")
print("=" * 70)
print("Do neurons that detect food-input also write food-output?")

# For each neuron, compute:
# - read_food_score = |fc1_row · food_input_direction|
# - write_food_score = fc2_col · food_output_direction (in residual stream)
# If read-write coherent: high read_food → high write_food

# Get the MLP OUTPUT contrastive direction
# MLP output = fc2(act) for each input
mlp_out_a = (act_a.to(DEV) @ fc2_w.T.to(DEV))
mlp_out_b = (act_b.to(DEV) @ fc2_w.T.to(DEV))
delta_mlp_out_food = mlp_out_a - mlp_out_b
food_out_norm = delta_mlp_out_food / delta_mlp_out_food.norm()

# Write alignment: how much does each neuron's fc2 column align with the food output?
write_food = fc2_w.to(DEV).T @ food_out_norm  # (10240,)

# Scatter: read_food vs write_food
# Bin neurons by read_food quartile, check mean write_food
read_abs = read_food.abs().cpu()
write_vals = write_food.cpu()

quartiles = torch.quantile(read_abs, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
print(f"\n  Read-food quartile → mean write-food alignment:")
for i in range(len(quartiles)-1):
    mask = (read_abs >= quartiles[i]) & (read_abs < quartiles[i+1])
    if mask.sum() > 0:
        mean_write = float(write_vals[mask].mean())
        mean_read = float(read_abs[mask].mean())
        n_neurons = int(mask.sum())
        # Also check: in this bin, what fraction have same-sign read and write?
        same_sign = float(((read_food.cpu()[mask] * write_vals[mask]) > 0).float().mean())
        print(f"    read [{quartiles[i]:.3f}, {quartiles[i+1]:.3f}): "
              f"n={n_neurons:>5}  mean |write|={write_vals[mask].abs().mean():.4f}  "
              f"same_sign={same_sign:.2f}")


# ══════════════════════════════════════════════════════════════
# STEP 4: The "if-block" test — does GELU actually gate?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: GELU AS IF-BLOCK — gating analysis")
print("=" * 70)

# For food-reading neurons: how many are gated ON for hot_dog but OFF for cold_dog?
# "ON" = post-GELU activation > threshold (not just nonzero)
threshold = 0.1

# Top 200 food-reading neurons
top200_food = torch.topk(read_food.abs(), 200).indices

on_a_off_b = 0   # ON for hot_dog, OFF for cold_dog
on_b_off_a = 0   # ON for cold_dog, OFF for hot_dog
on_both = 0
off_both = 0

for n in top200_food:
    n = int(n)
    ga = abs(float(act_a[n])) > threshold
    gb = abs(float(act_b[n])) > threshold
    if ga and not gb:
        on_a_off_b += 1
    elif gb and not ga:
        on_b_off_a += 1
    elif ga and gb:
        on_both += 1
    else:
        off_both += 1

print(f"\n  Top 200 food-reading neurons:")
print(f"    ON for hot_dog, OFF for cold_dog: {on_a_off_b}")
print(f"    ON for cold_dog, OFF for hot_dog: {on_b_off_a}")
print(f"    ON for both:                      {on_both}")
print(f"    OFF for both:                     {off_both}")

# Same for species-reading neurons
top200_species = torch.topk(read_species.abs(), 200).indices

on_a_off_c = 0
on_c_off_a = 0
on_both_sp = 0
off_both_sp = 0

for n in top200_species:
    n = int(n)
    ga = abs(float(act_a[n])) > threshold
    gc = abs(float(act_c[n])) > threshold
    if ga and not gc:
        on_a_off_c += 1
    elif gc and not ga:
        on_c_off_a += 1
    elif ga and gc:
        on_both_sp += 1
    else:
        off_both_sp += 1

print(f"\n  Top 200 species-reading neurons:")
print(f"    ON for hot_dog, OFF for hot_cat: {on_a_off_c}")
print(f"    ON for hot_cat, OFF for hot_dog: {on_c_off_a}")
print(f"    ON for both:                     {on_both_sp}")
print(f"    OFF for both:                    {off_both_sp}")


# ══════════════════════════════════════════════════════════════
# STEP 5: Show complete read→gate→write for top gated neurons
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: COMPLETE READ→GATE→WRITE for cleanly gated neurons")
print("=" * 70)
print("Neurons ON for hot_dog, OFF for cold_dog, with food-aligned read:\n")

print(f"  {'Neuron':>8}  {'reads (fc1·W_U)':>40}  {'pre_a':>6} {'pre_b':>6}  {'gateA':>5} {'gateB':>5}  writes (fc2·W_U)")

count = 0
for n in top200_food:
    n = int(n)
    ga = float(act_a[n])
    gb = float(act_b[n])

    # Only show cleanly gated neurons
    if abs(ga) > 0.3 and abs(gb) < 0.05:
        # What does fc1 row read? Project through... it reads from residual stream
        # We can project fc1 row through W_U to see what "input tokens" it's sensitive to
        read_vec = fc1_w[n, :]  # (2560,) — reads from residual stream
        read_logits = read_vec @ W_U.T
        if float(read_food[n]) > 0:
            read_toks = topk_tok(read_logits, 5)
        else:
            read_toks = topk_tok(-read_logits, 5)

        write_vec = fc2_w[:, n]
        write_log = write_vec @ W_U.T
        if ga > 0:
            write_toks = topk_tok(write_log, 5)
        else:
            write_toks = topk_tok(-write_log, 5)

        pa = float(pre_a[n])
        pb = float(pre_b[n])

        print(f"  {n:>8}  {str(read_toks):>40}  {pa:>+6.2f} {pb:>+6.2f}  {ga:>+5.2f} {gb:>+5.2f}  {write_toks}")
        count += 1
        if count >= 20:
            break

print(f"\n  ({count} cleanly gated food-reading neurons found)")

# Same for species
print(f"\n  Neurons ON for hot_dog, OFF for hot_cat, with species-aligned read:\n")
print(f"  {'Neuron':>8}  {'reads (fc1·W_U)':>40}  {'pre_a':>6} {'pre_c':>6}  {'gateA':>5} {'gateC':>5}  writes (fc2·W_U)")

count = 0
for n in top200_species:
    n = int(n)
    ga = float(act_a[n])
    gc = float(act_c[n])

    if abs(ga) > 0.3 and abs(gc) < 0.05:
        read_vec = fc1_w[n, :]
        read_logits = read_vec @ W_U.T
        if float(read_species[n]) > 0:
            read_toks = topk_tok(read_logits, 5)
        else:
            read_toks = topk_tok(-read_logits, 5)

        write_vec = fc2_w[:, n]
        write_log = write_vec @ W_U.T
        if ga > 0:
            write_toks = topk_tok(write_log, 5)
        else:
            write_toks = topk_tok(-write_log, 5)

        pa = float(pre_a[n])
        pc = float(pre_c[n])

        print(f"  {n:>8}  {str(read_toks):>40}  {pa:>+6.2f} {pc:>+6.2f}  {ga:>+5.2f} {gc:>+5.2f}  {write_toks}")
        count += 1
        if count >= 20:
            break

print(f"\n  ({count} cleanly gated species-reading neurons found)")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
