"""
Generate greedy continuations for the dose-response cases in §3.2.
Shows what the model actually says, not just top-1 token.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
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


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def generate(text, max_new=30):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def generate_with_injection(text, delta, layer, pos=-1, max_new=30):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    injected = [False]

    def hook_fn(module, input, output):
        if injected[0]:
            return output
        injected[0] = True
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, pos, :] += delta.half().to(DEV)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, pos, :] += delta.half().to(DEV)
            return h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
    handle.remove()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# ── Extract eatability direction (same as §3.2) ──
print("Extracting eatability direction...")
contrasts = ["cold", "angry", "old", "pet", "stray"]
target = "The hot dog was"
dog_pos = 2  # "The hot dog was" → dog is at position 2
L = 4  # L4 post, at dog position

dirs = []
for adj in contrasts:
    text_t = "The hot dog was"
    text_c = f"The {adj} dog was"
    ids_t = tok(text_t, add_special_tokens=False)["input_ids"]
    ids_c = tok(text_c, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_t = model(torch.tensor([ids_t], device=DEV), output_hidden_states=True)
        out_c = model(torch.tensor([ids_c], device=DEV), output_hidden_states=True)
    h_t = out_t.hidden_states[L][0, dog_pos, :].float()
    h_c = out_c.hidden_states[L][0, dog_pos, :].float()
    dirs.append((h_t - h_c).cpu())
    del out_t, out_c

mean_dir = torch.stack(dirs).mean(dim=0).to(DEV)
mean_dir_norm = mean_dir / mean_dir.norm()
base_scale = float(mean_dir.norm())
print(f"  Direction extracted, ||mean||={base_scale:.1f}")

torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# EATABILITY DOSE-RESPONSE WITH GENERATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EATABILITY DIRECTION — dose-response with greedy generation")
print("=" * 70)

# Forward injection into non-food prompts
injection_targets = [
    ("The cold dog was", dog_pos),
    ("The angry dog was", dog_pos),
    ("The old dog was", dog_pos),
]

for text, pos in injection_targets:
    print(f"\n  Prompt: \"{text}\"")
    print(f"  Baseline: {generate(text)}")
    for frac in [0.25, 0.5, 1.0]:
        delta = mean_dir_norm * base_scale * frac
        gen = generate_with_injection(text, delta, L, pos)
        print(f"  +{frac:.2f}×:   {gen}")

# Reverse injection — subtract from hot dog
print(f"\n  Prompt: \"The hot dog was\"")
print(f"  Baseline: {generate('The hot dog was')}")
for frac in [0.5, 1.0, 1.5]:
    delta = mean_dir_norm * base_scale * (-frac)
    gen = generate_with_injection("The hot dog was", delta, L, dog_pos)
    print(f"  −{frac:.2f}×:   {gen}")

torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# TRUTH DIRECTION WITH GENERATION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TRUTH DIRECTION — dose-response with greedy generation")
print("=" * 70)

# Extract truth direction
ref_L = _sl(28)[0]
truth_pairs = [
    ("Paris is the capital of France. This statement is",
     "Paris is not the capital of France. This statement is"),
    ("Water boils at 100 degrees Celsius. This statement is",
     "Water boils at 50 degrees Celsius. This statement is"),
    ("The Sun is a star. This statement is",
     "The Sun is a planet. This statement is"),
    ("Dogs are mammals. This statement is",
     "Dogs are reptiles. This statement is"),
    ("Tokyo is the capital of Japan. This statement is",
     "Tokyo is the capital of China. This statement is"),
]

truth_dirs = []
for true_text, false_text in truth_pairs:
    ids_t = tok(true_text, add_special_tokens=False)["input_ids"]
    ids_f = tok(false_text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out_t = model(torch.tensor([ids_t], device=DEV), output_hidden_states=True)
        out_f = model(torch.tensor([ids_f], device=DEV), output_hidden_states=True)
    h_t = out_t.hidden_states[ref_L][0, -1, :].float()
    h_f = out_f.hidden_states[ref_L][0, -1, :].float()
    truth_dirs.append((h_t - h_f).cpu())
    del out_t, out_f

truth_mean = torch.stack(truth_dirs).mean(dim=0).to(DEV)
truth_norm = truth_mean / truth_mean.norm()
truth_scale = float(truth_mean.norm())
print(f"  Truth direction extracted, ||mean||={truth_scale:.1f}")

torch.cuda.empty_cache()

# Inject truth into false statements
false_prompts = [
    "Paris is not the capital of France. This statement is",
    "Dogs are reptiles. This statement is",
]

for prompt in false_prompts:
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Baseline: {generate(prompt)}")
    for frac in [0.5, 1.0]:
        delta = truth_norm * truth_scale * frac
        gen = generate_with_injection(prompt, delta, ref_L)
        print(f"  +{frac:.2f}× truth: {gen}")

# Subtract truth from true statements
true_prompts = [
    "Paris is the capital of France. This statement is",
    "The Sun is a star. This statement is",
]

for prompt in true_prompts:
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Baseline: {generate(prompt)}")
    for frac in [1.0, 1.5]:
        delta = truth_norm * truth_scale * (-frac)
        gen = generate_with_injection(prompt, delta, ref_L)
        print(f"  −{frac:.2f}× truth: {gen}")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
