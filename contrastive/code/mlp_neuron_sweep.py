"""
Broad sweep: do we consistently find clean read→gate→write neurons
across many different contrasts?

For each contrast pair:
1. Get MLP input contrastive direction
2. Find neurons whose fc1 row aligns with that direction
3. Check GELU gating (ON for A, OFF for B)
4. Read what the cleanly gated neurons detect (fc1·W_U) and write (fc2·W_U)
5. Report: how many clean detectors per contrast? Are they interpretable?

Test across many domains: food, emotion, syntax, factual, code,
register, temporal, spatial, etc.
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


def find_clean_neurons(text_a, text_b, layer_idx, gate_thresh=0.1, top_k=200):
    """
    Find neurons that are cleanly gated: ON for text_a, OFF for text_b.
    Returns list of (neuron_idx, read_toks, write_toks, pre_a, pre_b, act_a, act_b)
    """
    fc1_w = model.model.layers[layer_idx].mlp.fc1.weight.detach().float()
    fc2_w = model.model.layers[layer_idx].mlp.fc2.weight.detach().float()

    act_a, pre_a = get_post_gelu(text_a, layer_idx)
    act_b, pre_b = get_post_gelu(text_b, layer_idx)

    # Input contrastive direction
    mlp_in_a = get_mlp_input(text_a, layer_idx)
    mlp_in_b = get_mlp_input(text_b, layer_idx)
    delta_in = mlp_in_a - mlp_in_b
    delta_in_norm = delta_in / (delta_in.norm() + 1e-8)

    # Read alignment
    read_align = fc1_w @ delta_in_norm.to(DEV)  # (10240,)

    # Find top readers
    top_readers = torch.topk(read_align.abs(), top_k).indices

    clean_neurons = []
    for n in top_readers:
        n = int(n)
        ga = float(act_a[n])
        gb = float(act_b[n])
        pa = float(pre_a[n])
        pb = float(pre_b[n])
        ra = float(read_align[n])

        # Clean gate: ON for A, OFF for B (or vice versa)
        a_on = abs(ga) > gate_thresh
        b_on = abs(gb) > gate_thresh

        if (a_on and not b_on) or (b_on and not a_on):
            # Determine which direction to read
            read_vec = fc1_w[n, :]
            read_logits = read_vec @ W_U.T
            if ra > 0:
                read_toks = topk_tok(read_logits, 5)
            else:
                read_toks = topk_tok(-read_logits, 5)

            write_vec = fc2_w[:, n]
            write_logits = write_vec @ W_U.T
            if ga > 0 or (not a_on and gb > 0):
                active_val = ga if a_on else gb
                if active_val > 0:
                    write_toks = topk_tok(write_logits, 5)
                else:
                    write_toks = topk_tok(-write_logits, 5)
            else:
                active_val = ga if a_on else gb
                if active_val > 0:
                    write_toks = topk_tok(write_logits, 5)
                else:
                    write_toks = topk_tok(-write_logits, 5)

            active_for = "A" if a_on else "B"
            clean_neurons.append({
                'idx': n,
                'read': read_toks,
                'write': write_toks,
                'pre_a': pa, 'pre_b': pb,
                'act_a': ga, 'act_b': gb,
                'read_align': ra,
                'active_for': active_for,
            })

    return clean_neurons


# ── TEST CASES ────────────────────────────────────────────────
cases = [
    # (name, text_a, text_b, layer, description)
    ("food_compound", "The hot dog was", "The cold dog was",
     _sl(20)[0], "food vs animal"),
    ("caught_cold", "She caught a cold and went to",
     "She caught a fish and went to",
     _sl(20)[0], "illness vs fishing"),
    ("grief_joy", "She learned that her mother had passed away. She felt",
     "She learned that her mother had won the lottery. She felt",
     _sl(24)[0], "grief vs joy"),
    ("positive_neg", "The movie was absolutely wonderful and everyone",
     "The movie was absolutely terrible and everyone",
     _sl(24)[0], "positive vs negative sentiment"),
    ("theft_moral", "He slipped a bottle under his coat and walked out without paying. He",
     "He picked up a bottle, went to the register and paid. He",
     _sl(24)[0], "theft vs honest purchase"),
    ("metaphor_cold", "The ice in the bucket was extremely cold. The temperature was",
     "The reception at the party was extremely cold. The atmosphere was",
     _sl(24)[0], "literal vs metaphorical cold"),
    ("capital_france", "The capital of France is",
     "The capital of Germany is",
     _sl(28)[0], "France vs Germany"),
    ("IOI_names", "When Mary and John went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to",
     _sl(28)[0], "Mary vs John"),
    ("some_all", "Some of the students passed the exam, so",
     "All of the students passed the exam, so",
     _sl(24)[0], "partial vs universal quantifier"),
    ("past_future", "Yesterday it rained heavily and the streets were",
     "Tomorrow it will rain heavily and the streets will be",
     _sl(20)[0], "past vs future tense"),
    ("english_french", "The dog is in the garden. The animal is a",
     "Le chien est dans le jardin. L'animal est un",
     _sl(20)[0], "English vs French"),
    ("formal_informal", "Patient presented with acute chest pain. The diagnosis was",
     "Guy came in, chest really hurt. The diagnosis was",
     _sl(24)[0], "formal vs informal register"),
    ("know_doubt", "I know that the capital of France is",
     "I doubt that the capital of France is",
     _sl(24)[0], "certainty vs doubt"),
    ("agent_swap", "The dog chased the cat through the park. The animal that was exhausted was the",
     "The cat chased the dog through the park. The animal that was exhausted was the",
     _sl(28)[0], "agent-patient role swap"),
    ("code_natural", "def calculate_sum(a, b):\n    return a +",
     "The total sum of a and b equals a plus",
     _sl(20)[0], "code vs natural language"),
    ("fire_flood", "The building was on fire, with flames on the roof. The damage was",
     "The river was flooding, with water on the second floor. The damage was",
     _sl(24)[0], "fire vs flood disaster"),
    ("elephant_mouse", "The elephant walked slowly, its massive body",
     "The mouse scurried quickly, its tiny body",
     _sl(20)[0], "large vs small animal"),
    ("knife_book", "The knife was on the table, so she",
     "The book was on the table, so she",
     _sl(20)[0], "dangerous vs benign object"),
]

# ── RUN SWEEP ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MLP NEURON SWEEP — clean read→gate→write detectors across 18 contrasts")
print("=" * 70)

summary = []

for name, text_a, text_b, layer, desc in cases:
    print(f"\n{'─'*70}")
    print(f"  {name} (L{layer}): {desc}")
    print(f"    A: \"{text_a[-55:]}\"")
    print(f"    B: \"{text_b[-55:]}\"")

    clean = find_clean_neurons(text_a, text_b, layer, gate_thresh=0.1, top_k=300)

    # Also count with stricter threshold
    strict = [n for n in clean if abs(n['act_a'] if n['active_for']=='A' else n['act_b']) > 0.3
              and abs(n['act_b'] if n['active_for']=='A' else n['act_a']) < 0.05]

    print(f"    Clean gated (>0.1 / <0.1): {len(clean)}")
    print(f"    Strict gated (>0.3 / <0.05): {len(strict)}")

    # Show top 5 strict neurons
    if strict:
        print(f"    Top strict neurons:")
        for neuron in sorted(strict, key=lambda n: -abs(n['read_align']))[:8]:
            active = neuron['active_for']
            act_val = neuron['act_a'] if active == 'A' else neuron['act_b']
            inactive_val = neuron['act_b'] if active == 'A' else neuron['act_a']
            print(f"      N{neuron['idx']:>5} [{active}] "
                  f"act={act_val:>+5.2f}/{inactive_val:>+5.2f}  "
                  f"reads={neuron['read'][:3]}  "
                  f"writes={neuron['write'][:3]}")
    elif clean:
        print(f"    Top gated neurons (relaxed threshold):")
        for neuron in sorted(clean, key=lambda n: -abs(n['read_align']))[:5]:
            active = neuron['active_for']
            act_val = neuron['act_a'] if active == 'A' else neuron['act_b']
            inactive_val = neuron['act_b'] if active == 'A' else neuron['act_a']
            print(f"      N{neuron['idx']:>5} [{active}] "
                  f"act={act_val:>+5.2f}/{inactive_val:>+5.2f}  "
                  f"reads={neuron['read'][:3]}  "
                  f"writes={neuron['write'][:3]}")

    summary.append((name, len(clean), len(strict), desc))

    torch.cuda.empty_cache()

# ── SUMMARY TABLE ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"  {'Case':>20} {'Clean':>6} {'Strict':>7}  Description")
print(f"  {'─'*20} {'─'*6} {'─'*7}  {'─'*30}")
for name, clean_n, strict_n, desc in summary:
    print(f"  {name:>20} {clean_n:>6} {strict_n:>7}  {desc}")

total_clean = sum(c for _, c, _, _ in summary)
total_strict = sum(s for _, _, s, _ in summary)
print(f"\n  Total clean across {len(cases)} contrasts: {total_clean}")
print(f"  Total strict across {len(cases)} contrasts: {total_strict}")
print(f"  Mean clean per contrast: {total_clean/len(cases):.1f}")
print(f"  Mean strict per contrast: {total_strict/len(cases):.1f}")

# ── NEURON REUSE: do the same neurons appear across contrasts? ──
print("\n" + "=" * 70)
print("NEURON REUSE — do different contrasts use different neurons?")
print("=" * 70)

# Collect all strict neuron sets
all_strict_sets = {}
for name, text_a, text_b, layer, desc in cases:
    clean = find_clean_neurons(text_a, text_b, layer, gate_thresh=0.1, top_k=300)
    strict_ids = set(n['idx'] for n in clean
                     if abs(n['act_a'] if n['active_for']=='A' else n['act_b']) > 0.3
                     and abs(n['act_b'] if n['active_for']=='A' else n['act_a']) < 0.05)
    all_strict_sets[name] = (strict_ids, layer)
    torch.cuda.empty_cache()

# Group by layer and check overlap
layer_groups = {}
for name, (ids, layer) in all_strict_sets.items():
    if layer not in layer_groups:
        layer_groups[layer] = []
    layer_groups[layer].append((name, ids))

for layer, groups in sorted(layer_groups.items()):
    if len(groups) < 2:
        continue
    print(f"\n  Layer {layer} — {len(groups)} contrasts:")
    all_neurons = set()
    for name, ids in groups:
        all_neurons |= ids

    print(f"    Total unique neurons: {len(all_neurons)}")
    for i, (n1, s1) in enumerate(groups):
        for j, (n2, s2) in enumerate(groups):
            if j > i and s1 and s2:
                overlap = len(s1 & s2)
                print(f"    {n1:>20} ∩ {n2:<20}: {overlap} / min({len(s1)},{len(s2)})")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
