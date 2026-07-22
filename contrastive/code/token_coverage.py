"""
Token-shaped coverage measurement.

Three questions:
1. What fraction of Δh energy is captured by top-k W_U directions?
   (If high → computation is token-shaped. If low → opaque substrate.)
2. How does this compare to random vectors?
   (W_U smoothness baseline — how much is method artifact?)
3. Does coverage vary by phenomenon type?
   (Prediction-relevant vs structural vs abstract.)

Metric: "W_U coverage at rank k"
  - Project Δh through W_U: logits = Δh @ W_U.T
  - Take top-k and bottom-k token directions from W_U rows
  - Reconstruct: Δh_reconstructed = sum of projections onto those 2k directions
  - Coverage = ||Δh_reconstructed||² / ||Δh||²
  This measures how much of the Δh vector lives in the subspace
  spanned by the tokens it's "trying to say."

Baseline: same measurement on random unit vectors in R^d_model.
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
import json

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
W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)

# Normalise W_U rows for projection
W_U_norm = F.normalize(W_U, dim=1)  # (vocab, d_model)


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out.hidden_states


def wu_coverage(vec, k=10):
    """
    Measure what fraction of vec's energy is captured by the top-k + bottom-k
    W_U directions.

    Returns: coverage ratio (0-1), top tokens, bottom tokens
    """
    vec_f = vec.float().to(DEV)
    vec_norm_sq = (vec_f @ vec_f).item()
    if vec_norm_sq < 1e-10:
        return 0.0, [], []

    # Project onto all W_U rows
    logits = vec_f @ W_U.T  # (vocab,)

    # Top-k and bottom-k indices
    topk_vals, topk_idx = torch.topk(logits, k)
    botk_vals, botk_idx = torch.topk(-logits, k)

    all_idx = torch.cat([topk_idx, botk_idx])

    # Get the W_U rows for these tokens (unnormalized — we want actual directions)
    wu_rows = W_U[all_idx]  # (2k, d_model)

    # Project vec onto subspace spanned by these rows
    # Use QR to get orthonormal basis
    Q, R = torch.linalg.qr(wu_rows.T)  # Q: (d_model, 2k)
    proj = Q @ (Q.T @ vec_f)  # projection onto subspace
    proj_norm_sq = (proj @ proj).item()

    coverage = proj_norm_sq / vec_norm_sq

    top_toks = [tok.decode([int(i)]).strip()[:12] for i in topk_idx[:5]]
    bot_toks = [tok.decode([int(i)]).strip()[:12] for i in botk_idx[:5]]

    return coverage, top_toks, bot_toks


def wu_coverage_topk_only(vec, k=10):
    """Coverage using only top-k (positive direction), not bottom-k."""
    vec_f = vec.float().to(DEV)
    vec_norm_sq = (vec_f @ vec_f).item()
    if vec_norm_sq < 1e-10:
        return 0.0

    logits = vec_f @ W_U.T
    topk_vals, topk_idx = torch.topk(logits.abs(), k)
    wu_rows = W_U[topk_idx]
    Q, R = torch.linalg.qr(wu_rows.T)
    proj = Q @ (Q.T @ vec_f)
    return (proj @ proj).item() / vec_norm_sq


# ── TEST CASES ────────────────────────────────────────────────
# Organized by expected readability: high → medium → low

cases = {
    # HIGH READABILITY (prediction-shaped, lexical)
    "hotdog_food": (
        "The hot dog was",
        "The cold dog was",
        "high",
    ),
    "caught_cold_fish": (
        "She caught a cold and went to",
        "She caught a fish and went to",
        "high",
    ),
    "IOI_names": (
        "When Mary and John went to the store, John gave a drink to",
        "When Mary and John went to the store, Mary gave a drink to",
        "high",
    ),
    "capital_france": (
        "The capital of France is",
        "The capital of Germany is",
        "high",
    ),
    "positive_negative": (
        "The movie was absolutely wonderful and everyone",
        "The movie was absolutely terrible and everyone",
        "high",
    ),
    "past_future": (
        "Yesterday it rained heavily and the streets were",
        "Tomorrow it will rain heavily and the streets will be",
        "high",
    ),
    "some_all": (
        "Some of the students passed the exam, so",
        "All of the students passed the exam, so",
        "high",
    ),
    "english_french": (
        "The dog is in the garden. The animal is a",
        "Le chien est dans le jardin. L'animal est un",
        "high",
    ),

    # MEDIUM READABILITY (semantic, compositional)
    "metaphor_cold": (
        "The ice in the bucket was extremely cold. The temperature was",
        "The reception at the party was extremely cold. The atmosphere was",
        "medium",
    ),
    "know_doubt": (
        "I know that the capital of France is",
        "I doubt that the capital of France is",
        "medium",
    ),
    "grief_joy": (
        "She learned that her mother had passed away. She felt",
        "She learned that her mother had won the lottery. She felt",
        "medium",
    ),
    "formal_informal": (
        "Patient presented with acute chest pain. The diagnosis was",
        "Guy came in, chest really hurt. The diagnosis was",
        "medium",
    ),
    "thought_speech": (
        "He thought that the food was",
        "He said that the food was",
        "medium",
    ),
    "theft_moral": (
        "He slipped a bottle under his coat and walked out without paying. He",
        "He picked up a bottle, went to the register and paid. He",
        "medium",
    ),
    "causal_rain": (
        "Heavy rain causes",
        "Wet streets are caused by",
        "medium",
    ),

    # LOW READABILITY (structural, abstract, role-based)
    "agent_swap": (
        "The dog chased the cat through the park. The animal that was exhausted was the",
        "The cat chased the dog through the park. The animal that was exhausted was the",
        "low",
    ),
    "active_passive": (
        "The dog chased the cat. The one that was caught was the",
        "The cat was chased by the dog. The one that was caught was the",
        "low",
    ),
    "pronoun_bind": (
        "John told Mary that he had been promoted. She said she was happy for",
        "Mary told John that she had been promoted. He said he was happy for",
        "low",
    ),
    "determiner_def_indef": (
        "The hot dog was",
        "A hot dog was",
        "low",
    ),
    "person_1st_3rd": (
        "I woke up feeling happy. I decided to",
        "She woke up feeling happy. She decided to",
        "low",
    ),
    "order_only": (
        "Alice is taller than Bob. Bob is taller than Carol. The tallest is",
        "Carol is taller than Bob. Bob is taller than Alice. The tallest is",
        "low",
    ),
}

# ── RANDOM BASELINE ──────────────────────────────────────────
print("="*70)
print("RANDOM BASELINE: W_U coverage of random unit vectors")
print("="*70)

n_random = 200
ks = [5, 10, 20, 50, 100]

for k in ks:
    coverages = []
    for _ in range(n_random):
        rv = torch.randn(d_model, device=DEV)
        rv = rv / rv.norm()
        cov, _, _ = wu_coverage(rv, k=k)
        coverages.append(cov)
    mean_cov = sum(coverages) / len(coverages)
    std_cov = (sum((c - mean_cov)**2 for c in coverages) / len(coverages)) ** 0.5
    print(f"  k={k:>3}: mean coverage = {mean_cov:.4f} ± {std_cov:.4f}")

print(f"\n  (Vocab size: {W_U.shape[0]}, d_model: {d_model})")
print(f"  Expected if uniform: k/V = {100/W_U.shape[0]:.6f} for k=100")
print(f"  Expected if W_U rows span d_model: 2k/d_model = {200/d_model:.4f} for k=100")


# ── MAIN MEASUREMENT ─────────────────────────────────────────
print("\n" + "="*70)
print("CONTRASTIVE COVERAGE BY CASE AND LAYER")
print("="*70)

layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)
ref_L = _sl(28)[0]
k = 20  # top+bottom 20 tokens = 40 directions

results = {}  # case_name -> {layer: coverage}

for case_name, (text_a, text_b, expected) in cases.items():
    hs_a = get_hidden_states(text_a)
    hs_b = get_hidden_states(text_b)

    case_results = {}
    for L in layers:
        ha = hs_a[L][0, -1, :].float()
        hb = hs_b[L][0, -1, :].float()
        dh = ha - hb
        cov, top, bot = wu_coverage(dh, k=k)
        case_results[L] = cov

    results[case_name] = case_results

    # Print trajectory
    print(f"\n  {case_name} [{expected}]")
    print(f"    A: \"{text_a[-60:]}\"")
    print(f"    B: \"{text_b[-60:]}\"")
    for L in layers:
        cov = case_results[L]
        bar = "█" * int(cov * 50)
        print(f"    L{L:>2}: {cov:.3f} {bar}")

    del hs_a, hs_b
    torch.cuda.empty_cache()


# ── SUMMARY BY EXPECTED CATEGORY ─────────────────────────────
print("\n" + "="*70)
print(f"SUMMARY: MEAN COVERAGE AT L{ref_L} (k={k}, {2*k} directions)")
print("="*70)

for category in ["high", "medium", "low"]:
    cat_cases = [(name, res) for name, (_, _, exp) in cases.items()
                 for res in [results[name]] if exp == category]
    if not cat_cases:
        continue
    covs = [res[ref_L] for _, res in cat_cases]
    mean_c = sum(covs) / len(covs)
    print(f"\n  {category.upper()} readability (n={len(covs)}): mean = {mean_c:.3f}")
    for name, res in sorted(cat_cases, key=lambda x: -x[1][ref_L]):
        print(f"    {name:>25}: {res[ref_L]:.3f}")


# ── LAYER-RESOLVED SUMMARY ───────────────────────────────────
print("\n" + "="*70)
print("LAYER-RESOLVED MEAN COVERAGE BY CATEGORY")
print("="*70)

for category in ["high", "medium", "low"]:
    cat_names = [name for name, (_, _, exp) in cases.items() if exp == category]
    if not cat_names:
        continue
    print(f"\n  {category.upper()}:")
    for L in layers:
        covs = [results[n][L] for n in cat_names]
        mean_c = sum(covs) / len(covs)
        bar = "█" * int(mean_c * 50)
        print(f"    L{L:>2}: {mean_c:.3f} {bar}")


# ── COVERAGE vs RANDOM COMPARISON AT REF LAYER ───────────────
print("\n" + "="*70)
print(f"COVERAGE vs RANDOM BASELINE AT L{ref_L} (k={k})")
print("="*70)

# Recompute random at k=20
random_covs = []
for _ in range(500):
    rv = torch.randn(d_model, device=DEV)
    rv = rv / rv.norm()
    cov, _, _ = wu_coverage(rv, k=k)
    random_covs.append(cov)
random_mean = sum(random_covs) / len(random_covs)
random_std = (sum((c - random_mean)**2 for c in random_covs) / len(random_covs)) ** 0.5

print(f"\n  Random baseline: {random_mean:.4f} ± {random_std:.4f}")
print(f"\n  Contrastive Δh values:")
all_covs = []
for name in sorted(results.keys(), key=lambda n: -results[n][ref_L]):
    cov = results[name][ref_L]
    sigma = (cov - random_mean) / random_std
    all_covs.append(cov)
    _, _, exp = cases[name]
    print(f"    {name:>25} [{exp[0]}]: {cov:.3f}  ({sigma:>+6.1f}σ)")

print(f"\n  Overall mean: {sum(all_covs)/len(all_covs):.3f}")
print(f"  Overall mean vs random: {sum(all_covs)/len(all_covs) / random_mean:.1f}x")


# ── TOP TOKENS AT REF LAYER FOR EACH CASE ────────────────────
print("\n" + "="*70)
print(f"TOKEN READOUT AT L{ref_L} (for interpretability check)")
print("="*70)

for case_name, (text_a, text_b, expected) in cases.items():
    hs_a = get_hidden_states(text_a)
    hs_b = get_hidden_states(text_b)
    ha = hs_a[ref_L][0, -1, :].float()
    hb = hs_b[ref_L][0, -1, :].float()
    dh = ha - hb
    cov, top, bot = wu_coverage(dh, k=k)
    print(f"  {case_name:>25} [{expected[0]}] cov={cov:.3f}  +[{', '.join(top)}]  -[{', '.join(bot)}]")
    del hs_a, hs_b
    torch.cuda.empty_cache()


print("\n" + "="*70)
print("DONE")
print("="*70)
