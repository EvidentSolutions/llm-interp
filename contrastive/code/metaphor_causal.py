"""
Metaphor causal injection test.

§6.3 shows metaphor is domain routing: "cold reception" routes to
emotion tokens, "cold ice" routes to temperature tokens. This is
observational. Here we test causally:

1. Extract the routing direction (literal - metaphorical) for "cold"
   from multiple pairs
2. Inject it into a metaphorical context → does it flip to literal?
3. Inject the reverse into a literal context → does it flip to metaphorical?
4. Cross-domain: does the "cold" routing direction transfer to "sharp"?
   (If domain routing is per-word, it shouldn't. If there's a shared
   literal/figurative axis, it should.)
5. Dose-response: minimum injection to flip the domain

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
W_U = model.lm_head.weight.detach().float()

cos = F.cosine_similarity


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{float(vals[j]):.3f}")
            for j in range(k)]


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out


def predict(text, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)]


def inject_and_predict(text, delta, layer, k=5):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    injected = [False]

    def hook_fn(module, input, output):
        if injected[0]:
            return output
        injected[0] = True
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, -1, :] += delta.half().to(DEV)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, -1, :] += delta.half().to(DEV)
            return h

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV))
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)]


# ══════════════════════════════════════════════════════════════
# STEP 1: Extract routing directions from multiple pairs
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 1: EXTRACT DOMAIN ROUTING DIRECTIONS")
print("=" * 70)

ref_L = _sl(24)[0]  # L24 — where domain routing is active per §6.3

# COLD: literal - metaphorical pairs
cold_pairs = [
    ("The ice in the bucket was extremely cold. The temperature was",
     "The reception at the party was extremely cold. The atmosphere was"),
    ("The water in the lake was extremely cold. The temperature was",
     "The welcome from the host was extremely cold. The atmosphere was"),
    ("The metal railing was extremely cold. The temperature was",
     "The response from the audience was extremely cold. The atmosphere was"),
    ("The wind outside was extremely cold. The temperature was",
     "The tone of the email was extremely cold. The atmosphere was"),
]

cold_dirs = []
for lit, met in cold_pairs:
    out_lit = get_hidden_states(lit)
    out_met = get_hidden_states(met)
    h_lit = out_lit.hidden_states[ref_L][0, -1, :].float()
    h_met = out_met.hidden_states[ref_L][0, -1, :].float()
    cold_dirs.append((h_lit - h_met).cpu())
    del out_lit, out_met
    torch.cuda.empty_cache()

# Check pairwise consistency
print(f"\n  COLD routing direction consistency (L{ref_L}):")
for i in range(len(cold_dirs)):
    for j in range(i+1, len(cold_dirs)):
        c = float(cos(cold_dirs[i].unsqueeze(0), cold_dirs[j].unsqueeze(0)))
        print(f"    pair {i} × pair {j}: cos = {c:+.3f}")

cold_mean = torch.stack(cold_dirs).mean(dim=0).to(DEV)
cold_mean_norm = cold_mean / cold_mean.norm()
cold_logits = cold_mean @ W_U.T
print(f"\n  Mean cold routing direction reads:")
print(f"    literal pole:      {topk_tok(cold_logits, 8)}")
print(f"    metaphorical pole: {topk_tok(-cold_logits, 8)}")

# SHARP: literal - metaphorical pairs
sharp_pairs = [
    ("The knife on the counter was extremely sharp. The blade was",
     "The criticism in the review was extremely sharp. The tone was"),
    ("The scissors were extremely sharp. The blade was",
     "The rebuke from the manager was extremely sharp. The tone was"),
    ("The razor was extremely sharp. The blade was",
     "The wit of the comedian was extremely sharp. The tone was"),
    ("The needle was extremely sharp. The blade was",
     "The sarcasm in her voice was extremely sharp. The tone was"),
]

sharp_dirs = []
for lit, met in sharp_pairs:
    out_lit = get_hidden_states(lit)
    out_met = get_hidden_states(met)
    h_lit = out_lit.hidden_states[ref_L][0, -1, :].float()
    h_met = out_met.hidden_states[ref_L][0, -1, :].float()
    sharp_dirs.append((h_lit - h_met).cpu())
    del out_lit, out_met
    torch.cuda.empty_cache()

print(f"\n  SHARP routing direction consistency (L{ref_L}):")
for i in range(len(sharp_dirs)):
    for j in range(i+1, len(sharp_dirs)):
        c = float(cos(sharp_dirs[i].unsqueeze(0), sharp_dirs[j].unsqueeze(0)))
        print(f"    pair {i} × pair {j}: cos = {c:+.3f}")

sharp_mean = torch.stack(sharp_dirs).mean(dim=0).to(DEV)
sharp_mean_norm = sharp_mean / sharp_mean.norm()
sharp_logits = sharp_mean @ W_U.T
print(f"\n  Mean sharp routing direction reads:")
print(f"    literal pole:      {topk_tok(sharp_logits, 8)}")
print(f"    metaphorical pole: {topk_tok(-sharp_logits, 8)}")

# Cross-domain cosine
cross_cos = float(cos(cold_mean.unsqueeze(0), sharp_mean.unsqueeze(0)))
print(f"\n  Cross-domain cosine (cold vs sharp routing): {cross_cos:+.3f}")


# ══════════════════════════════════════════════════════════════
# STEP 2: Causal injection — flip the domain
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: CAUSAL INJECTION — flip domain routing")
print("=" * 70)

# Inject COLD literal direction into metaphorical context
met_target = "The reception at the party was extremely cold. The atmosphere was"
lit_target = "The ice in the bucket was extremely cold. The temperature was"

print(f"\n  Baseline predictions:")
print(f"    Metaphorical: \"{met_target}\"")
print(f"      → {predict(met_target)}")
print(f"    Literal: \"{lit_target}\"")
print(f"      → {predict(lit_target)}")

base_scale = float(cold_mean.norm())

print(f"\n  Inject LITERAL direction into metaphorical context:")
print(f"    (positive = toward literal/temperature, scale={base_scale:.1f})")
for frac in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    delta = cold_mean * frac
    preds = inject_and_predict(met_target, delta, ref_L)
    print(f"    +{frac:.2f}×: {preds}")

print(f"\n  Inject METAPHORICAL direction into literal context:")
print(f"    (negative = toward metaphorical/emotion)")
for frac in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    delta = cold_mean * (-frac)
    preds = inject_and_predict(lit_target, delta, ref_L)
    print(f"    −{frac:.2f}×: {preds}")

# Same for SHARP
print(f"\n  SHARP domain flip:")
sharp_met = "The criticism in the review was extremely sharp. The tone was"
sharp_lit = "The knife on the counter was extremely sharp. The blade was"

print(f"    Baseline metaphorical: {predict(sharp_met)}")
print(f"    Baseline literal: {predict(sharp_lit)}")

sharp_scale = float(sharp_mean.norm())
print(f"\n  Inject LITERAL direction into sharp-metaphorical:")
for frac in [0.5, 1.0, 1.5, 2.0]:
    delta = sharp_mean * frac
    preds = inject_and_predict(sharp_met, delta, ref_L)
    print(f"    +{frac:.2f}×: {preds}")

print(f"\n  Inject METAPHORICAL direction into sharp-literal:")
for frac in [0.5, 1.0, 1.5, 2.0]:
    delta = sharp_mean * (-frac)
    preds = inject_and_predict(sharp_lit, delta, ref_L)
    print(f"    −{frac:.2f}×: {preds}")


# ══════════════════════════════════════════════════════════════
# STEP 3: Cross-domain transfer
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: CROSS-DOMAIN TRANSFER")
print("=" * 70)
print("Does the COLD routing direction affect SHARP contexts?")
print("If domain routing is per-word, it shouldn't transfer.")
print("If there's a shared literal/figurative axis, it should.\n")

print(f"  Inject COLD literal direction into SHARP metaphorical:")
for frac in [0.5, 1.0, 1.5, 2.0]:
    delta = cold_mean * frac
    preds = inject_and_predict(sharp_met, delta, ref_L)
    print(f"    cold +{frac:.2f}×: {preds}")

print(f"\n  Inject SHARP literal direction into COLD metaphorical:")
for frac in [0.5, 1.0, 1.5, 2.0]:
    delta = sharp_mean * frac
    preds = inject_and_predict(met_target, delta, ref_L)
    print(f"    sharp +{frac:.2f}×: {preds}")


# ══════════════════════════════════════════════════════════════
# STEP 4: Dose-response — minimum flip dose
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: DOSE-RESPONSE — minimum dose to flip domain")
print("=" * 70)

baseline_met = predict(met_target, 1)[0][0]
print(f"\n  Cold metaphorical baseline top-1: {baseline_met}")
print(f"  Sweeping literal injection dose:")

for pct in range(5, 205, 5):
    frac = pct / 100.0
    delta = cold_mean * frac
    preds = inject_and_predict(met_target, delta, ref_L, 1)
    top1 = preds[0][0]
    if top1 != baseline_met:
        print(f"    FLIPS at {frac:.2f}× ({base_scale*frac:.1f} norm): "
              f"{baseline_met} → {top1}")
        # Show full top-5 at flip point
        preds5 = inject_and_predict(met_target, delta, ref_L, 5)
        print(f"    Top-5 at flip: {preds5}")
        break
else:
    print(f"    No flip up to 2.0×")

baseline_lit = predict(lit_target, 1)[0][0]
print(f"\n  Cold literal baseline top-1: {baseline_lit}")
print(f"  Sweeping metaphorical injection dose:")

for pct in range(5, 205, 5):
    frac = pct / 100.0
    delta = cold_mean * (-frac)
    preds = inject_and_predict(lit_target, delta, ref_L, 1)
    top1 = preds[0][0]
    if top1 != baseline_lit:
        print(f"    FLIPS at −{frac:.2f}× ({base_scale*frac:.1f} norm): "
              f"{baseline_lit} → {top1}")
        preds5 = inject_and_predict(lit_target, delta, ref_L, 5)
        print(f"    Top-5 at flip: {preds5}")
        break
else:
    print(f"    No flip up to −2.0×")


# ══════════════════════════════════════════════════════════════
# STEP 5: Additional metaphor domains (bright, heavy)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 5: ADDITIONAL DOMAINS — bright, heavy")
print("=" * 70)

extra_domains = [
    ("bright", [
        ("The lamp in the corner was extremely bright. The light was",
         "The student in the class was extremely bright. The child was"),
        ("The spotlight was extremely bright. The light was",
         "The idea she proposed was extremely bright. The child was"),
    ]),
    ("heavy", [
        ("The boulder on the trail was extremely heavy. The weight was",
         "The news from the hospital was extremely heavy. The mood was"),
        ("The barbell was extremely heavy. The weight was",
         "The silence in the room was extremely heavy. The mood was"),
    ]),
]

all_domain_dirs = {"cold": cold_mean_norm, "sharp": sharp_mean_norm}

for word, pairs in extra_domains:
    dirs = []
    for lit, met in pairs:
        out_lit = get_hidden_states(lit)
        out_met = get_hidden_states(met)
        h_lit = out_lit.hidden_states[ref_L][0, -1, :].float()
        h_met = out_met.hidden_states[ref_L][0, -1, :].float()
        dirs.append((h_lit - h_met).cpu())
        del out_lit, out_met
        torch.cuda.empty_cache()

    pair_cos = float(cos(dirs[0].unsqueeze(0), dirs[1].unsqueeze(0)))
    mean_dir = torch.stack(dirs).mean(dim=0).to(DEV)
    mean_norm = mean_dir / mean_dir.norm()
    all_domain_dirs[word] = mean_norm

    logits = mean_dir @ W_U.T
    print(f"\n  {word.upper()} routing direction (pair cos = {pair_cos:+.3f}):")
    print(f"    literal:      {topk_tok(logits, 6)}")
    print(f"    metaphorical: {topk_tok(-logits, 6)}")

    # Causal test
    met_text = pairs[0][1]
    lit_text = pairs[0][0]
    scale = float(mean_dir.norm())

    print(f"    Baseline literal:      {predict(lit_text)}")
    print(f"    Baseline metaphorical: {predict(met_text)}")
    print(f"    Inject literal +1.0×: {inject_and_predict(met_text, mean_dir, ref_L)}")
    print(f"    Inject metaph  −1.0×: {inject_and_predict(lit_text, -mean_dir, ref_L)}")

# Cross-domain cosine matrix
print(f"\n  Cross-domain routing cosine matrix:")
labels = list(all_domain_dirs.keys())
print(f"  {'':>8}", end="")
for lb in labels:
    print(f" {lb:>8}", end="")
print()
for l1 in labels:
    print(f"  {l1:>8}", end="")
    for l2 in labels:
        c = float(cos(all_domain_dirs[l1].unsqueeze(0), all_domain_dirs[l2].unsqueeze(0)))
        print(f" {c:>+8.3f}", end="")
    print()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
