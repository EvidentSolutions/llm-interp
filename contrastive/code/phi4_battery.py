"""
Run the paper's core experiments on Phi-4 (14B) in 4-bit quantization.

Covers:
  1. Landmark replication (§5): IOI, factual recall, successor heads
  2. Null model + causal injection (§3.1)
  3. MLP neuron sweep - SwiGLU (§3.3)

Usage: .venv/Scripts/python.exe contrastive/code/phi4_battery.py
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEV = "cuda"
MODEL = "microsoft/phi-4"

print(f"Loading {MODEL} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb_config, low_cpu_mem_usage=True
).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
d_model = model.config.hidden_size
d_inter = model.config.intermediate_size
W_U = model.lm_head.weight.detach().float()

print(f"Layers: {NL}, Hidden: {d_model}, Intermediate: {d_inter}")
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")


def _sl(*layers):
    """Scale layer indices from 32-layer base to current NL."""
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def tk(logits, k=5):
    v, i = torch.topk(logits.float(), k)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def bk(logits, k=5):
    v, i = torch.topk(logits, k, largest=False)
    return ", ".join(tok.decode([int(i[j])]).strip()[:12] for j in range(k))


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    return out, ids


# ============================================================
# PART 1: LANDMARK REPLICATION
# ============================================================
print("\n" + "=" * 100)
print("PART 1: LANDMARK REPLICATION")
print("=" * 100)


def run_pair(pa, pb, layers=None):
    if layers is None:
        layers = _sl(8, 16, 20, 24, 28, 32)
    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )
        gen_a = model.generate(
            torch.tensor([ids_a], device=DEV),
            max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id,
        )
        gen_b = model.generate(
            torch.tensor([ids_b], device=DEV),
            max_new_tokens=5, do_sample=False, pad_token_id=tok.eos_token_id,
        )
    ans_a = tok.decode(gen_a[0][len(ids_a):]).strip()[:30]
    ans_b = tok.decode(gen_b[0][len(ids_b):]).strip()[:30]
    print(f'  A: "{pa}" -> "{ans_a}"')
    print(f'  B: "{pb}" -> "{ans_b}"')
    for L in layers:
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norm = float(dh.norm() / h_a.norm())
        ld = dh @ W_U.T
        print(f"    L{L:>2} ({norm:.3f}) A=[{tk(ld)}]  B=[{bk(ld)}]")
    del out_a, out_b
    torch.cuda.empty_cache()


print("\n--- IOI ---")
ioi = [
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to"),
    ("Alice and Bob went to the park. Alice gave a gift to",
     "Bob and Alice went to the park. Bob gave a gift to"),
    ("Dan and Eve ate dinner together. Dan passed the salt to",
     "Eve and Dan ate dinner together. Eve passed the salt to"),
]
for pa, pb in ioi:
    print()
    run_pair(pa, pb)

print("\n--- FACTUAL RECALL ---")
facts = [
    ("The Eiffel Tower is located in",
     "The Colosseum is located in"),
    ("The capital of France is",
     "The capital of Japan is"),
    ("Shakespeare wrote",
     "Tolstoy wrote"),
]
for pa, pb in facts:
    print()
    run_pair(pa, pb)

print("\n--- SUCCESSORS ---")
successors = [
    ("After Monday comes",
     "After Tuesday comes"),
    ("Monday, Tuesday, Wednesday, Thursday,",
     "Tuesday, Wednesday, Thursday, Friday,"),
]
for pa, pb in successors:
    print()
    run_pair(pa, pb, layers=_sl(16, 24, 28, 32))

torch.cuda.empty_cache()

# ============================================================
# PART 2: NULL MODEL + CAUSAL INJECTION
# ============================================================
print(f"\n{'='*100}")
print("PART 2: NULL MODEL + CAUSAL INJECTION")
print("=" * 100)


def consecutive_cosine(logit_trajectory):
    cosines = []
    for i in range(len(logit_trajectory) - 1):
        cos = float(F.cosine_similarity(
            logit_trajectory[i].unsqueeze(0),
            logit_trajectory[i + 1].unsqueeze(0),
        ))
        cosines.append(cos)
    return cosines


def run_null_model(pa, pb, n_random=50, label=""):
    out_a, ids_a = get_hidden_states(pa)
    out_b, ids_b = get_hidden_states(pb)
    print(f'\n  {label}')
    print(f'  A: "{pa}"')
    print(f'  B: "{pb}"')

    real_logits = []
    norms = []
    W_U_f = W_U.float()
    for L in range(NL + 1):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norms.append(float(dh.norm()))
        logits = dh @ W_U_f.T
        real_logits.append(logits.cpu())

    real_cos = consecutive_cosine(real_logits)
    mean_real = sum(real_cos) / len(real_cos)

    random_means = []
    for trial in range(n_random):
        rand_logits = []
        for L in range(NL + 1):
            rand_dir = torch.randn(d_model, device=DEV)
            rand_dir = rand_dir / rand_dir.norm() * norms[L]
            logits = rand_dir @ W_U_f.T
            rand_logits.append(logits.cpu())
        rand_cos = consecutive_cosine(rand_logits)
        random_means.append(sum(rand_cos) / len(rand_cos))

    mean_random = sum(random_means) / len(random_means)
    std_random = (sum((x - mean_random)**2 for x in random_means)
                  / len(random_means)) ** 0.5
    z = (mean_real - mean_random) / std_random if std_random > 0 else float('inf')

    print(f"  Mean consecutive cosine:")
    print(f"    Real trajectory:     {mean_real:.4f}")
    print(f"    Random directions:   {mean_random:.4f} ± {std_random:.4f}")
    print(f"    z vs random:         {z:.1f}")

    # Per-layer summary (groups of 10)
    for start in range(0, len(real_cos), 10):
        end = min(start + 10, len(real_cos))
        segment = ' '.join(f'{c:.3f}' for c in real_cos[start:end])
        print(f"    L{start}-{end}: {segment}")

    del out_a, out_b
    torch.cuda.empty_cache()
    return mean_real, mean_random, z


null_cases = [
    ("The hot dog was", "The cold dog was", "hot dog"),
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to", "IOI"),
    ("The Eiffel Tower is located in",
     "The Colosseum is located in", "factual recall"),
    ("After Monday comes", "After Tuesday comes", "successor"),
    ("He caught a cold and", "He caught a fish and", "disambiguation"),
    ("The bank was steep and", "The bank was closed and", "bank ambiguity"),
]

null_results = []
for pa, pb, label in null_cases:
    r = run_null_model(pa, pb, n_random=50, label=label)
    null_results.append((label, *r))

print(f"\n  Summary:")
print(f"  {'Case':<20} {'Real cos':>10} {'Random cos':>10} {'z':>6}")
for label, real, rand, z in null_results:
    print(f"  {label:<20} {real:>10.4f} {rand:>10.4f} {z:>6.1f}")

# --- Causal injection ---
print(f"\n{'='*100}")
print("CAUSAL INJECTION")
print("=" * 100)


def run_injection(pa, pb, target_token, layers_to_inject=None, label=""):
    if layers_to_inject is None:
        layers_to_inject = _sl(4, 8, 12, 16, 20, 24, 28, 31)

    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    target_id = tok(target_token, add_special_tokens=False)["input_ids"]
    if len(target_id) == 1:
        target_id = target_id[0]
    else:
        target_id = tok(" " + target_token, add_special_tokens=False)["input_ids"]
        target_id = target_id[0] if len(target_id) == 1 else target_id[-1]

    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True)
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True)

    p_a = float(torch.softmax(out_a.logits[0, -1].float(), -1)[target_id])
    p_b = float(torch.softmax(out_b.logits[0, -1].float(), -1)[target_id])
    gap = p_a - p_b

    print(f'\n  {label}')
    print(f'  A: "{pa}"')
    print(f'  B: "{pb}"')
    print(f'  Target: "{target_token}" (id={target_id})')
    print(f"  P(target|A) = {p_a:.4f}")
    print(f"  P(target|B) = {p_b:.4f}")
    print(f"  Gap = {gap:.4f}")

    if abs(gap) < 0.001:
        print(f"  [SKIP — gap too small for meaningful injection test]")
        del out_a, out_b
        torch.cuda.empty_cache()
        return

    print(f"\n  {'Layer':>5}  {'P(target)':>10}  {'Recovery':>10}  {'z vs rand':>10}")

    for L in layers_to_inject:
        dh = (out_a.hidden_states[L][0, -1, :] -
              out_b.hidden_states[L][0, -1, :]).detach()

        def make_hook(delta):
            injected = [False]
            def hook_fn(module, input, output):
                if injected[0]:
                    return output
                if isinstance(output, tuple):
                    h = output[0].clone()
                    h[0, -1, :] += delta
                    injected[0] = True
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, -1, :] += delta
                    injected[0] = True
                    return h
            return hook_fn, injected

        hook_fn, injected = make_hook(dh)
        handle = model.model.layers[L].register_forward_hook(hook_fn)
        injected[0] = False
        with torch.no_grad():
            out_inj = model(torch.tensor([ids_b], device=DEV))
        handle.remove()

        p_inj = float(torch.softmax(out_inj.logits[0, -1].float(), -1)[target_id])
        recovery = (p_inj - p_b) / gap * 100

        # Random control
        n_rand = 10
        rand_recoveries = []
        for _ in range(n_rand):
            rand_dir = torch.randn_like(dh)
            rand_dir = rand_dir / rand_dir.norm() * dh.norm()
            hook_fn_r, injected_r = make_hook(rand_dir)
            handle = model.model.layers[L].register_forward_hook(hook_fn_r)
            injected_r[0] = False
            with torch.no_grad():
                out_rand = model(torch.tensor([ids_b], device=DEV))
            handle.remove()
            p_rand = float(torch.softmax(out_rand.logits[0, -1].float(), -1)[target_id])
            rand_recoveries.append((p_rand - p_b) / gap * 100)

        mean_rand = sum(rand_recoveries) / len(rand_recoveries)
        std_rand = (sum((x - mean_rand)**2 for x in rand_recoveries)
                    / len(rand_recoveries)) ** 0.5
        z = (recovery - mean_rand) / std_rand if std_rand > 0 else 0

        print(f"  L{L:>3}  {p_inj:>10.4f}  {recovery:>9.1f}%  {z:>10.1f}")

    del out_a, out_b
    torch.cuda.empty_cache()


# IOI
run_injection(
    "John and Mary went to the store. John gave a book to",
    "Mary and John went to the store. Mary gave a book to",
    "Mary",
    layers_to_inject=_sl(4, 12, 20, 24, 28, 31),
    label="IOI → Mary"
)

# Factual recall
run_injection(
    "The Eiffel Tower is located in",
    "The Colosseum is located in",
    "Paris",
    layers_to_inject=_sl(4, 12, 20, 24, 28, 31),
    label="Eiffel Tower → Paris"
)

# Hot dog — use "cooked" since Phi-4 reads food at L16+
run_injection(
    "The hot dog was",
    "The cold dog was",
    "cooked",
    layers_to_inject=_sl(4, 12, 16, 20, 24, 28, 31),
    label="hot dog → cooked"
)

torch.cuda.empty_cache()

# ============================================================
# PART 3: MLP NEURON SWEEP (SwiGLU)
# ============================================================
print(f"\n{'='*100}")
print("PART 3: MLP NEURON SWEEP (SwiGLU)")
print("=" * 100)


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [tok.decode([int(idxs[j])]).strip()[:14] for j in range(k)]


def get_swiglu_activations(text, layer_idx):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    captured = {}

    def hook_fn(module, input, output):
        captured['gate_up'] = output[0, -1, :].detach().float()

    handle = model.model.layers[layer_idx].mlp.gate_up_proj.register_forward_hook(hook_fn)
    with torch.no_grad():
        model(torch.tensor([ids], device=DEV))
    handle.remove()

    gate_up = captured['gate_up']
    gate_pre = gate_up[:d_inter]
    up_vals = gate_up[d_inter:]
    gate_post = F.silu(gate_pre)
    neuron_act = gate_post * up_vals
    return gate_pre, gate_post, up_vals, neuron_act


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


def dequantize_weight(layer_idx, proj_name):
    """Get weight matrix, dequantizing if needed (4-bit)."""
    proj = getattr(model.model.layers[layer_idx].mlp, proj_name)
    w = proj.weight
    if hasattr(w, 'data'):
        # Try to get the dequantized version
        try:
            import bitsandbytes as bnb
            if isinstance(proj, bnb.nn.Linear4bit):
                return bnb.functional.dequantize_4bit(
                    w.data, w.quant_state
                ).float()
        except Exception:
            pass
    return w.detach().float()


def find_clean_neurons(text_a, text_b, layer_idx, gate_thresh=0.1, top_k=200):
    gate_up_w = dequantize_weight(layer_idx, 'gate_up_proj')
    gate_w = gate_up_w[:d_inter, :]
    down_w = dequantize_weight(layer_idx, 'down_proj')

    gate_pre_a, gate_post_a, up_a, act_a = get_swiglu_activations(text_a, layer_idx)
    gate_pre_b, gate_post_b, up_b, act_b = get_swiglu_activations(text_b, layer_idx)

    mlp_in_a = get_mlp_input(text_a, layer_idx)
    mlp_in_b = get_mlp_input(text_b, layer_idx)
    delta_in = mlp_in_a - mlp_in_b
    delta_in_norm = delta_in / (delta_in.norm() + 1e-8)

    read_align = gate_w @ delta_in_norm.to(gate_w.device)
    top_readers = torch.topk(read_align.abs(), top_k).indices

    clean_neurons = []
    for n in top_readers:
        n = int(n)
        aa = float(act_a[n])
        ab = float(act_b[n])
        ra = float(read_align[n])

        a_on = abs(aa) > gate_thresh
        b_on = abs(ab) > gate_thresh

        if (a_on and not b_on) or (b_on and not a_on):
            read_vec = gate_w[n, :]
            read_logits = read_vec.cpu().float() @ W_U.cpu().T
            read_toks = topk_tok(read_logits if ra > 0 else -read_logits, 5)

            write_vec = down_w[:, n]
            write_logits = write_vec.cpu().float() @ W_U.cpu().T
            active_val = aa if a_on else ab
            write_toks = topk_tok(write_logits if active_val > 0 else -write_logits, 5)

            clean_neurons.append({
                'idx': n, 'read': read_toks, 'write': write_toks,
                'act_a': aa, 'act_b': ab, 'read_align': ra,
                'active_for': "A" if a_on else "B",
            })

    return clean_neurons


neuron_cases = [
    ("food_compound", "The hot dog was", "The cold dog was",
     _sl(20)[0], "food vs animal"),
    ("caught_cold", "She caught a cold and went to",
     "She caught a fish and went to",
     _sl(20)[0], "illness vs fishing"),
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
    ("english_french", "The dog is in the garden. The animal is a",
     "Le chien est dans le jardin. L'animal est un",
     _sl(20)[0], "English vs French"),
    ("code_natural", "def calculate_sum(a, b):\n    return a +",
     "The total sum of a and b equals a plus",
     _sl(20)[0], "code vs natural language"),
    ("elephant_mouse", "The elephant walked slowly, its massive body",
     "The mouse scurried quickly, its tiny body",
     _sl(20)[0], "large vs small animal"),
]

summary = []
for name, text_a, text_b, layer, desc in neuron_cases:
    print(f"\n{'─'*70}")
    print(f"  {name} (L{layer}): {desc}")
    print(f"    A: \"{text_a[-55:]}\"")
    print(f"    B: \"{text_b[-55:]}\"")

    clean = find_clean_neurons(text_a, text_b, layer, gate_thresh=0.1, top_k=300)
    strict = [n for n in clean if abs(n['act_a'] if n['active_for']=='A' else n['act_b']) > 0.3
              and abs(n['act_b'] if n['active_for']=='A' else n['act_a']) < 0.05]

    print(f"    Clean gated (>0.1 / <0.1): {len(clean)}")
    print(f"    Strict gated (>0.3 / <0.05): {len(strict)}")

    neurons_to_show = strict if strict else clean
    if neurons_to_show:
        print(f"    Top {'strict' if strict else 'gated'} neurons:")
        for neuron in sorted(neurons_to_show, key=lambda n: -abs(n['read_align']))[:6]:
            active = neuron['active_for']
            act_val = neuron['act_a'] if active == 'A' else neuron['act_b']
            inactive_val = neuron['act_b'] if active == 'A' else neuron['act_a']
            print(f"      N{neuron['idx']:>5} [{active}] "
                  f"act={act_val:>+6.2f}/{inactive_val:>+6.2f}  "
                  f"reads={neuron['read'][:3]}  "
                  f"writes={neuron['write'][:3]}")

    summary.append((name, len(clean), len(strict), desc))
    torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("NEURON SWEEP SUMMARY")
print("=" * 70)
print(f"  {'Case':>20} {'Clean':>6} {'Strict':>7}  Description")
print(f"  {'─'*20} {'─'*6} {'─'*7}  {'─'*30}")
for name, clean_n, strict_n, desc in summary:
    print(f"  {name:>20} {clean_n:>6} {strict_n:>7}  {desc}")

total_clean = sum(c for _, c, _, _ in summary)
total_strict = sum(s for _, _, s, _ in summary)
print(f"\n  Total clean across {len(neuron_cases)} contrasts: {total_clean}")
print(f"  Total strict across {len(neuron_cases)} contrasts: {total_strict}")
print(f"  Mean clean per contrast: {total_clean/len(neuron_cases):.1f}")
print(f"  Mean strict per contrast: {total_strict/len(neuron_cases):.1f}")

print(f"\n{'='*100}")
print("ALL DONE")
print(f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
print("=" * 100)
