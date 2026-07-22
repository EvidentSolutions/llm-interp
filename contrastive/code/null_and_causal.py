"""
Null model validation + causal injection for poster cases.

1. Null model: compare real Δh trajectory smoothness (consecutive-layer cosine)
   against random directions of matched norm projected through W_U.

2. Injection recovery: inject Δh from context into control's residual stream
   and measure how much of the prediction gap it recovers.

Usage: .venv/Scripts/python.exe contrastive/code/null_and_causal.py
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
W_U = model.lm_head.weight.detach()


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(
            torch.tensor([ids], device=DEV), output_hidden_states=True
        )
    return out, ids


def consecutive_cosine(logit_trajectory):
    """Mean cosine between consecutive layers' logit vectors."""
    cosines = []
    for i in range(len(logit_trajectory) - 1):
        cos = float(torch.nn.functional.cosine_similarity(
            logit_trajectory[i].unsqueeze(0),
            logit_trajectory[i + 1].unsqueeze(0),
        ))
        cosines.append(cos)
    return cosines


def run_null_model(pa, pb, n_random=50, label=""):
    """Compare real trajectory smoothness against random directions."""
    out_a, ids_a = get_hidden_states(pa)
    out_b, ids_b = get_hidden_states(pb)

    print(f'\n  {label}')
    print(f'  A: "{pa}"')
    print(f'  B: "{pb}"')

    # Real trajectory: Δh at each layer, projected through W_U
    real_logits = []
    norms = []
    for L in range(NL + 1):
        h_a = out_a.hidden_states[L][0, -1, :].float()
        h_b = out_b.hidden_states[L][0, -1, :].float()
        dh = h_a - h_b
        norms.append(float(dh.norm()))
        logits = dh @ W_U.float().T
        real_logits.append(logits.cpu())

    real_cos = consecutive_cosine(real_logits)
    mean_real = sum(real_cos) / len(real_cos)

    # Random null: random directions of matched norm at each layer
    # Do all trials in a batch on GPU for speed
    W_U_f = W_U.float()
    random_means = []
    for trial in range(n_random):
        rand_logits = []
        for L in range(NL + 1):
            rand_dir = torch.randn(W_U.shape[1], device=DEV)
            rand_dir = rand_dir / rand_dir.norm() * norms[L]
            logits = rand_dir @ W_U_f.T
            rand_logits.append(logits.cpu())
        rand_cos = consecutive_cosine(rand_logits)
        random_means.append(sum(rand_cos) / len(rand_cos))

    mean_random = sum(random_means) / len(random_means)
    std_random = (sum((x - mean_random)**2 for x in random_means)
                  / len(random_means)) ** 0.5
    z = (mean_real - mean_random) / std_random if std_random > 0 else float('inf')

    # Also: shuffled-layer null (real vectors, scrambled order)
    import random
    shuffle_means = []
    for trial in range(n_random):
        perm = list(range(NL + 1))
        random.shuffle(perm)
        shuffled = [real_logits[p] for p in perm]
        shuf_cos = consecutive_cosine(shuffled)
        shuffle_means.append(sum(shuf_cos) / len(shuf_cos))

    mean_shuffle = sum(shuffle_means) / len(shuffle_means)

    print(f"  Mean consecutive cosine:")
    print(f"    Real trajectory:     {mean_real:.4f}")
    print(f"    Random directions:   {mean_random:.4f} ± {std_random:.4f}")
    print(f"    Shuffled layers:     {mean_shuffle:.4f}")
    print(f"    z vs random:         {z:.1f}")
    print(f"  Per-layer cosines (real):")
    print(f"    L0-8:  {' '.join(f'{c:.3f}' for c in real_cos[:8])}")
    print(f"    L8-16: {' '.join(f'{c:.3f}' for c in real_cos[8:16])}")
    print(f"    L16-24:{' '.join(f'{c:.3f}' for c in real_cos[16:24])}")
    print(f"    L24-32:{' '.join(f'{c:.3f}' for c in real_cos[24:])}")

    del out_a, out_b
    return mean_real, mean_random, z


def run_injection(pa, pb, target_token, layers_to_inject=None, label=""):
    """
    Inject Δh from context into control at each layer.
    Measure recovery of prediction gap.
    """
    if layers_to_inject is None:
        layers_to_inject = [4, 8, 12, 16, 20, 24, 28, 31]

    ids_a = tok(pa, add_special_tokens=False)["input_ids"]
    ids_b = tok(pb, add_special_tokens=False)["input_ids"]
    target_id = tok(target_token, add_special_tokens=False)["input_ids"]
    if len(target_id) == 1:
        target_id = target_id[0]
    else:
        # Try with space prefix
        target_id = tok(" " + target_token, add_special_tokens=False)["input_ids"]
        target_id = target_id[0] if len(target_id) == 1 else target_id[-1]

    # Baseline predictions
    with torch.no_grad():
        out_a = model(
            torch.tensor([ids_a], device=DEV), output_hidden_states=True
        )
        out_b = model(
            torch.tensor([ids_b], device=DEV), output_hidden_states=True
        )

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

    # For each layer: inject Δh into control, measure new P(target)
    print(f"\n  {'Layer':>5}  {'P(target)':>10}  {'Recovery':>10}  {'z vs rand':>10}")

    for L in layers_to_inject:
        dh = (out_a.hidden_states[L][0, -1, :] -
              out_b.hidden_states[L][0, -1, :]).detach()

        # Hook to add Δh at the target layer
        handle = None
        injected = [False]

        def make_hook(delta, layer_idx):
            def hook_fn(module, input, output):
                if injected[0]:
                    return output
                # output is (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    h = output[0]
                    h = h.clone()
                    h[0, -1, :] += delta
                    injected[0] = True
                    return (h,) + output[1:]
                else:
                    h = output.clone()
                    h[0, -1, :] += delta
                    injected[0] = True
                    return h
            return hook_fn

        # Register hook on the appropriate layer
        if L < NL:
            layer_mod = model.model.layers[L]
            handle = layer_mod.register_forward_hook(make_hook(dh, L))
        else:
            # Final layer — inject before lm_head
            handle = model.model.final_layernorm.register_forward_hook(
                make_hook(dh, L))

        injected[0] = False
        with torch.no_grad():
            out_inj = model(torch.tensor([ids_b], device=DEV))

        if handle:
            handle.remove()

        p_inj = float(torch.softmax(out_inj.logits[0, -1].float(), -1)[target_id])
        recovery = (p_inj - p_b) / gap * 100 if gap != 0 else 0

        # Random control: inject random direction of same norm
        n_rand = 20
        rand_recoveries = []
        for _ in range(n_rand):
            rand_dir = torch.randn_like(dh)
            rand_dir = rand_dir / rand_dir.norm() * dh.norm()

            if L < NL:
                handle = model.model.layers[L].register_forward_hook(
                    make_hook(rand_dir, L))
            else:
                handle = model.model.final_layernorm.register_forward_hook(
                    make_hook(rand_dir, L))

            injected[0] = False
            with torch.no_grad():
                out_rand = model(torch.tensor([ids_b], device=DEV))
            handle.remove()

            p_rand = float(torch.softmax(
                out_rand.logits[0, -1].float(), -1)[target_id])
            rand_recoveries.append((p_rand - p_b) / gap * 100 if gap != 0 else 0)

        mean_rand = sum(rand_recoveries) / len(rand_recoveries)
        std_rand = (sum((x - mean_rand)**2 for x in rand_recoveries)
                    / len(rand_recoveries)) ** 0.5
        z = (recovery - mean_rand) / std_rand if std_rand > 0 else 0

        print(f"  L{L:>3}  {p_inj:>10.4f}  {recovery:>9.1f}%  {z:>10.1f}")

    del out_a, out_b
    torch.cuda.empty_cache()


# ============================================================
print("=" * 100)
print("1. NULL MODEL — trajectory smoothness vs random")
print("=" * 100)

cases = [
    ("The hot dog was", "The cold dog was", "hot dog"),
    ("John and Mary went to the store. John gave a book to",
     "Mary and John went to the store. Mary gave a book to", "IOI"),
    ("The Eiffel Tower is located in",
     "The Colosseum is located in", "factual recall"),
    ("After Monday comes", "After Tuesday comes", "successor"),
    ("He caught a cold and", "He caught a fish and", "disambiguation"),
    ("The bank was steep and", "The bank was closed and", "bank ambiguity"),
]

results = []
for pa, pb, label in cases:
    r = run_null_model(pa, pb, n_random=50, label=label)
    results.append((label, *r))
    torch.cuda.empty_cache()

print(f"\n  Summary:")
print(f"  {'Case':<20} {'Real cos':>10} {'Random cos':>10} {'z':>6}")
for label, real, rand, z in results:
    print(f"  {label:<20} {real:>10.4f} {rand:>10.4f} {z:>6.1f}")

# ============================================================
print(f"\n{'='*100}")
print("2. INJECTION RECOVERY — causal validation")
print("=" * 100)

# Hot dog
run_injection(
    "The hot dog was",
    "The cold dog was",
    "delicious",
    layers_to_inject=[0, 4, 8, 12, 16, 20, 24, 28, 31],
    label="hot dog → delicious"
)

# IOI
run_injection(
    "John and Mary went to the store. John gave a book to",
    "Mary and John went to the store. Mary gave a book to",
    "Mary",
    layers_to_inject=[4, 8, 12, 16, 20, 24, 28, 31],
    label="IOI → Mary"
)

# Factual recall
run_injection(
    "The Eiffel Tower is located in",
    "The Colosseum is located in",
    "Paris",
    layers_to_inject=[4, 8, 12, 16, 20, 24, 28, 31],
    label="Eiffel Tower → Paris"
)

# Successor
run_injection(
    "After Monday comes",
    "After Tuesday comes",
    "Tuesday",
    layers_to_inject=[4, 8, 12, 16, 20, 24, 28, 31],
    label="Monday successor → Tuesday"
)

torch.cuda.empty_cache()
print(f"\n{'='*100}")
print("DONE")
