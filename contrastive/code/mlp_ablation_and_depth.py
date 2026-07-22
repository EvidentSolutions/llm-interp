"""
MLP neuron ablation + deeper correspondence analysis.

Part 1: ABLATION
  For each contrast, zero out the strictly gated neurons.
  Does the contrastive projection lose its token readout?
  Does the model's prediction change?
  Compare: ablate strict neurons vs ablate random neurons of same count.

Part 2: VARIANCE CONTRIBUTION
  What fraction of the MLP's contrastive output do the strict neurons
  account for? (Not energy of Δh — energy of the MLP layer's Δ output.)

Part 3: DEEPER CORRESPONDENCE
  - Do the same neurons appear at different layers for the same contrast?
  - How does the number of strict neurons scale with layer depth?
  - Are there neurons that read one feature and write a DIFFERENT feature?
    (compositional neurons: read "hot" + "noun", write "food")
  - Layer sweep: at which layer do the clean detectors first appear?

Full prompts shown with all results.
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
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{float(vals[j]):.3f}")
            for j in range(k)]


def predict(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def predict_with_neuron_ablation(text, layer_idx, neuron_indices):
    """Zero out specific neurons in the MLP intermediate at layer_idx."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    ablated = [False]

    def hook_fn(module, input, output):
        if ablated[0]:
            return output
        ablated[0] = True
        # output is the fc1 output (pre-GELU); we need to hook on fc1
        # Actually, let's hook on the full MLP and ablate inside
        return output

    # Hook after fc1 to zero specific neurons before GELU
    def fc1_hook(module, input, output):
        if not ablated[0]:
            ablated[0] = True
            out = output.clone()
            for n in neuron_indices:
                out[0, -1, n] = 0.0
            return out
        return output

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(fc1_hook)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k=5)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(5)], probs


def get_post_gelu(text, layer_idx):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    fc1_out = {}

    def hook_fn(module, input, output):
        fc1_out['pre_gelu'] = output[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.fc1.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(torch.tensor([ids], device=DEV))
    handle.remove()
    gelu = model.model.layers[layer_idx].mlp.activation_fn
    return gelu(fc1_out['pre_gelu']), fc1_out['pre_gelu']


def get_mlp_input(text, layer_idx):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    mlp_in = {}

    def hook_fn(module, args):
        inp = args[0] if isinstance(args, tuple) else args
        mlp_in['x'] = inp[0, -1, :].detach().float()
        return None

    handle = model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook_fn)
    with torch.no_grad():
        model(torch.tensor([ids], device=DEV))
    handle.remove()
    return mlp_in['x']


def find_strict_neurons(text_a, text_b, layer_idx, top_k=300):
    fc1_w = model.model.layers[layer_idx].mlp.fc1.weight.detach().float()
    act_a, pre_a = get_post_gelu(text_a, layer_idx)
    act_b, pre_b = get_post_gelu(text_b, layer_idx)
    mlp_in_a = get_mlp_input(text_a, layer_idx)
    mlp_in_b = get_mlp_input(text_b, layer_idx)
    delta_in = mlp_in_a - mlp_in_b
    delta_in_norm = delta_in / (delta_in.norm() + 1e-8)
    read_align = fc1_w @ delta_in_norm.to(DEV)
    top_readers = torch.topk(read_align.abs(), top_k).indices

    strict = []
    for n in top_readers:
        n = int(n)
        ga = float(act_a[n])
        gb = float(act_b[n])
        a_on = abs(ga) > 0.3
        b_on = abs(gb) > 0.05
        b_on_rev = abs(gb) > 0.3
        a_on_rev = abs(ga) > 0.05
        if (a_on and not b_on):
            strict.append(n)
        elif (b_on_rev and not a_on_rev):
            strict.append(n)
    return strict, act_a, act_b


# ══════════════════════════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════════════════════════
cases = [
    ("food_compound", "The hot dog was", "The cold dog was", _sl(20)[0]),
    ("grief_joy", "She learned that her mother had passed away. She felt",
     "She learned that her mother had won the lottery. She felt", _sl(24)[0]),
    ("metaphor_cold",
     "The ice in the bucket was extremely cold. The temperature was",
     "The reception at the party was extremely cold. The atmosphere was", _sl(24)[0]),
    ("capital_france", "The capital of France is",
     "The capital of Germany is", _sl(28)[0]),
    ("IOI_names",
     "When Mary and John went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to", _sl(28)[0]),
    ("positive_neg", "The movie was absolutely wonderful and everyone",
     "The movie was absolutely terrible and everyone", _sl(24)[0]),
    ("elephant_mouse", "The elephant walked slowly, its massive body",
     "The mouse scurried quickly, its tiny body", _sl(20)[0]),
    ("english_french", "The dog is in the garden. The animal is a",
     "Le chien est dans le jardin. L'animal est un", _sl(20)[0]),
]


# ══════════════════════════════════════════════════════════════
# PART 1: ABLATION
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1: NEURON ABLATION — does zeroing strict neurons affect output?")
print("=" * 70)

import random

for name, text_a, text_b, layer in cases:
    print(f"\n{'─'*70}")
    print(f"  {name} (L{layer})")
    print(f"    A: \"{text_a[-50:]}\"")
    print(f"    B: \"{text_b[-50:]}\"")

    strict, act_a, act_b = find_strict_neurons(text_a, text_b, layer)
    n_strict = len(strict)

    if n_strict == 0:
        print(f"    No strict neurons found, skipping")
        continue

    # Baseline predictions
    preds_a_base, probs_a_base = predict(text_a)
    preds_b_base, probs_b_base = predict(text_b)

    # Ablate strict neurons
    preds_a_abl, probs_a_abl = predict_with_neuron_ablation(text_a, layer, strict)
    preds_b_abl, probs_b_abl = predict_with_neuron_ablation(text_b, layer, strict)

    # Ablate random neurons (same count, 5 trials)
    random_effects_a = []
    random_effects_b = []
    for trial in range(5):
        rand_neurons = random.sample(range(d_inter), n_strict)
        _, probs_a_rand = predict_with_neuron_ablation(text_a, layer, rand_neurons)
        _, probs_b_rand = predict_with_neuron_ablation(text_b, layer, rand_neurons)
        # KL divergence from baseline
        kl_a = float(F.kl_div(probs_a_rand.log(), probs_a_base, reduction='sum'))
        kl_b = float(F.kl_div(probs_b_rand.log(), probs_b_base, reduction='sum'))
        random_effects_a.append(kl_a)
        random_effects_b.append(kl_b)

    # KL from ablating strict neurons
    kl_a_strict = float(F.kl_div(probs_a_abl.log(), probs_a_base, reduction='sum'))
    kl_b_strict = float(F.kl_div(probs_b_abl.log(), probs_b_base, reduction='sum'))
    mean_kl_a_rand = sum(random_effects_a) / len(random_effects_a)
    mean_kl_b_rand = sum(random_effects_b) / len(random_effects_b)

    print(f"    Strict neurons: {strict[:10]}{'...' if n_strict > 10 else ''} (n={n_strict})")
    print(f"    Input A baseline: {preds_a_base[:3]}")
    print(f"    Input A ablated:  {preds_a_abl[:3]}")
    print(f"    Input B baseline: {preds_b_base[:3]}")
    print(f"    Input B ablated:  {preds_b_abl[:3]}")
    print(f"    KL(ablated || baseline):")
    print(f"      Strict neurons:  A={kl_a_strict:.4f}  B={kl_b_strict:.4f}")
    print(f"      Random neurons:  A={mean_kl_a_rand:.4f}  B={mean_kl_b_rand:.4f}")
    print(f"      Ratio (strict/random): A={kl_a_strict/max(mean_kl_a_rand,1e-6):.1f}x  "
          f"B={kl_b_strict/max(mean_kl_b_rand,1e-6):.1f}x")

    # Does ablation change the contrastive readout?
    # Get hidden states with and without ablation, compute Δh, read through W_U
    # For this we need to hook deeper — get hidden states after the layer
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 2: VARIANCE CONTRIBUTION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 2: VARIANCE — what fraction of MLP Δ output do strict neurons carry?")
print("=" * 70)

for name, text_a, text_b, layer in cases:
    fc2_w = model.model.layers[layer].mlp.fc2.weight.detach().float()
    strict, act_a, act_b = find_strict_neurons(text_a, text_b, layer)

    if not strict:
        print(f"\n  {name}: no strict neurons")
        continue

    # MLP contrastive output = sum over all neurons of delta_act[n] * fc2[:, n]
    delta_act = act_a - act_b

    # Total MLP Δ output
    total_mlp_delta = delta_act.to(DEV) @ fc2_w.T.to(DEV)  # (d_model,)
    total_norm = float(total_mlp_delta.norm())

    # Strict neurons' contribution
    strict_contribution = torch.zeros(d_model, device=DEV)
    for n in strict:
        strict_contribution += float(delta_act[n]) * fc2_w[:, n].to(DEV)
    strict_norm = float(strict_contribution.norm())

    # Cosine between strict contribution and total
    if total_norm > 0 and strict_norm > 0:
        alignment = float(cos(strict_contribution.unsqueeze(0),
                               total_mlp_delta.unsqueeze(0)))
    else:
        alignment = 0.0

    # What does the strict contribution read as through W_U?
    strict_logits = strict_contribution @ W_U.T
    strict_toks = topk_tok(strict_logits, 5)

    # What does the total MLP Δ read as?
    total_logits = total_mlp_delta @ W_U.T
    total_toks = topk_tok(total_logits, 5)

    print(f"\n  {name} (L{layer}, {len(strict)} strict neurons):")
    print(f"    ||strict contribution||: {strict_norm:.1f}  "
          f"||total MLP Δ||: {total_norm:.1f}  "
          f"ratio: {strict_norm/max(total_norm,1e-6):.3f}")
    print(f"    cos(strict, total): {alignment:+.3f}")
    print(f"    Strict reads:  {strict_toks}")
    print(f"    Total MLP reads: {total_toks}")

    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 3: LAYER SWEEP — where do clean detectors appear?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 3: LAYER SWEEP — strict neuron count by layer")
print("=" * 70)

sweep_cases = [
    ("food_compound", "The hot dog was", "The cold dog was"),
    ("grief_joy", "She learned that her mother had passed away. She felt",
     "She learned that her mother had won the lottery. She felt"),
    ("capital_france", "The capital of France is",
     "The capital of Germany is"),
]

layers = _sl(4, 8, 12, 16, 20, 24, 28, 32)

for name, text_a, text_b in sweep_cases:
    print(f"\n  {name}:")
    for L in layers:
        strict, _, _ = find_strict_neurons(text_a, text_b, L)

        # What do the strict neurons read/write?
        if strict:
            fc1_w = model.model.layers[L].mlp.fc1.weight.detach().float()
            fc2_w = model.model.layers[L].mlp.fc2.weight.detach().float()
            # Show top strict neuron's read/write
            act_a, _ = get_post_gelu(text_a, L)
            n = strict[0]
            read_logits = fc1_w[n] @ W_U.T
            ga = float(act_a[n])
            if ga > 0:
                r_toks = [t[0] for t in topk_tok(read_logits, 3)]
            else:
                r_toks = [t[0] for t in topk_tok(-read_logits, 3)]

            write_logits = fc2_w[:, n] @ W_U.T
            if ga > 0:
                w_toks = [t[0] for t in topk_tok(write_logits, 3)]
            else:
                w_toks = [t[0] for t in topk_tok(-write_logits, 3)]
            detail = f"  top: N{n} reads={r_toks} writes={w_toks}"
        else:
            detail = ""

        bar = "█" * len(strict)
        print(f"    L{L:>2}: {len(strict):>3} strict{detail}  {bar}")
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# PART 4: READ≠WRITE NEURONS — compositional detectors
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PART 4: READ≠WRITE — do any neurons read one feature, write another?")
print("=" * 70)

for name, text_a, text_b, layer in cases[:4]:
    fc1_w = model.model.layers[layer].mlp.fc1.weight.detach().float()
    fc2_w = model.model.layers[layer].mlp.fc2.weight.detach().float()
    strict, act_a, act_b = find_strict_neurons(text_a, text_b, layer)

    if not strict:
        continue

    print(f"\n  {name} (L{layer}):")
    for n in strict[:10]:
        ga = float(act_a[n])
        gb = float(act_b[n])

        read_logits = fc1_w[n] @ W_U.T
        write_logits = fc2_w[:, n] @ W_U.T

        if ga > 0.3:  # fires for A
            r = [t[0] for t in topk_tok(read_logits, 4)]
            w = [t[0] for t in topk_tok(write_logits, 4)]
        elif gb > 0.3:  # fires for B
            r = [t[0] for t in topk_tok(-read_logits, 4)]
            w = [t[0] for t in topk_tok(-write_logits, 4)]
        else:
            continue

        # Cosine between read and write directions
        read_dir = fc1_w[n].to(DEV)
        write_dir = fc2_w[:, n].to(DEV)
        rw_cos = float(cos(read_dir.unsqueeze(0), write_dir.unsqueeze(0)))

        same = "SAME" if rw_cos > 0.3 else ("DIFF" if rw_cos < 0.1 else "weak")
        print(f"    N{n:>5}: reads={r}  writes={w}  "
              f"cos(read,write)={rw_cos:+.3f} [{same}]")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
