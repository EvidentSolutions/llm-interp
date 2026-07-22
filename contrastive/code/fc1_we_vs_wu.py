"""
Does fc1 read in W_E space or W_U space?

fc1 rows are detectors that read from the residual stream.
W_E maps tokens → residual (input embedding).
W_U maps residual → tokens (output unembedding).
In Phi-2, cos(W_E[tok], W_U[tok]) ≈ -0.01 — they're orthogonal.

Test: project fc1 rows through both W_E.T and W_U.T.
Which gives more interpretable token labels for the neurons
we know are clean detectors (N925, N7828, N5082, etc.)?

Also: project the contrastive Δh through W_E.T — does it
read differently from W_U.T?
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
W_U = model.lm_head.weight.detach().float()  # (vocab, d_model)
W_E = model.model.embed_tokens.weight.detach().float()  # (vocab, d_model)

cos = F.cosine_similarity


def topk_tok(logits, k=5):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{float(vals[j]):.2f}")
            for j in range(k)]


def get_hidden_states(text):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out


def _sl(*layers):
    return sorted(set(min(round(l * NL / 32), NL) for l in layers))


# ══════════════════════════════════════════════════════════════
# STEP 0: W_E vs W_U basic comparison
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("STEP 0: W_E vs W_U geometry")
print("=" * 70)

# Sample cosines
sample_words = ["food", "dog", "cold", "hot", "the", "Mary", "Paris", "doctor",
                "fried", "sharp", "heavy", "bright"]
print(f"\n  Per-token cos(W_E[tok], W_U[tok]):")
for word in sample_words:
    ids = tok.encode(word, add_special_tokens=False)
    if len(ids) == 1:
        tid = ids[0]
        c = float(cos(W_E[tid].unsqueeze(0), W_U[tid].unsqueeze(0)))
        print(f"    {word:>10}: {c:+.4f}")

# Global stats
all_cos = []
for i in range(0, W_E.shape[0], 100):
    c = float(cos(W_E[i].unsqueeze(0), W_U[i].unsqueeze(0)))
    all_cos.append(c)
print(f"\n  Global cos(W_E, W_U) stats (sampled every 100th token):")
print(f"    mean: {sum(all_cos)/len(all_cos):.4f}")
print(f"    min:  {min(all_cos):.4f}  max: {max(all_cos):.4f}")


# ══════════════════════════════════════════════════════════════
# STEP 1: Known clean neurons — fc1 through W_E vs W_U
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 1: KNOWN CLEAN NEURONS — fc1 read through W_E vs W_U")
print("=" * 70)

known_neurons = [
    (20, 925,  "food detector"),
    (20, 7828, "food/edible detector"),
    (20, 5123, "taste detector"),
    (20, 2133, "size/bulky detector"),
    (28, 841,  "gender/himself detector"),
    (20, 2226, "French language detector"),
    (24, 5082, "temperature change detector"),
]

for layer, n_idx, label in known_neurons:
    fc1_w = model.model.layers[layer].mlp.fc1.weight.detach().float()
    fc2_w = model.model.layers[layer].mlp.fc2.weight.detach().float()

    read_vec = fc1_w[n_idx]  # (d_model,)
    write_vec = fc2_w[:, n_idx]  # (d_model,)

    # Project read through both
    read_wu = read_vec @ W_U.T
    read_we = read_vec @ W_E.T

    # Project write through both
    write_wu = write_vec @ W_U.T
    write_we = write_vec @ W_E.T

    print(f"\n  N{n_idx} (L{layer}) — {label}:")
    print(f"    fc1 READ via W_U: {topk_tok(read_wu, 6)}")
    print(f"    fc1 READ via W_E: {topk_tok(read_we, 6)}")
    print(f"    fc2 WRITE via W_U: {topk_tok(write_wu, 6)}")
    print(f"    fc2 WRITE via W_E: {topk_tok(write_we, 6)}")


# ══════════════════════════════════════════════════════════════
# STEP 2: Contrastive Δh through W_E vs W_U
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 2: CONTRASTIVE Δh THROUGH W_E vs W_U")
print("=" * 70)

contrasts = [
    ("hot_dog", "The hot dog was", "The cold dog was", _sl(20)[0]),
    ("hot_dog_late", "The hot dog was", "The cold dog was", _sl(28)[0]),
    ("IOI", "When Mary and John went to the store, John gave a drink to",
     "When Mary and John went to the store, Mary gave a drink to", _sl(28)[0]),
    ("caught_cold", "She caught a cold and went to",
     "She caught a fish and went to", _sl(20)[0]),
    ("caught_cold_late", "She caught a cold and went to",
     "She caught a fish and went to", _sl(28)[0]),
    ("capital", "The capital of France is",
     "The capital of Germany is", _sl(28)[0]),
    ("metaphor", "The ice in the bucket was extremely cold. The temperature was",
     "The reception at the party was extremely cold. The atmosphere was", _sl(24)[0]),
]

for name, text_a, text_b, layer in contrasts:
    out_a = get_hidden_states(text_a)
    out_b = get_hidden_states(text_b)
    ha = out_a.hidden_states[layer][0, -1, :].float()
    hb = out_b.hidden_states[layer][0, -1, :].float()
    dh = ha - hb

    logits_wu = dh @ W_U.T
    logits_we = dh @ W_E.T

    print(f"\n  {name} (L{layer}):")
    print(f"    Δh via W_U: +{[t[0] for t in topk_tok(logits_wu, 6)]}  "
          f"-{[t[0] for t in topk_tok(-logits_wu, 6)]}")
    print(f"    Δh via W_E: +{[t[0] for t in topk_tok(logits_we, 6)]}  "
          f"-{[t[0] for t in topk_tok(-logits_we, 6)]}")

    del out_a, out_b
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# STEP 3: Layer-by-layer — does W_E readability change with depth?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 3: W_E vs W_U READABILITY BY LAYER")
print("=" * 70)

text_a = "The hot dog was"
text_b = "The cold dog was"

layers = _sl(0, 4, 8, 12, 16, 20, 24, 28, 32)

out_a = get_hidden_states(text_a)
out_b = get_hidden_states(text_b)

print(f"\n  Hot dog contrast, layer by layer:")
for L in layers:
    if L > NL:
        continue
    ha = out_a.hidden_states[L][0, -1, :].float()
    hb = out_b.hidden_states[L][0, -1, :].float()
    dh = ha - hb

    wu_logits = dh @ W_U.T
    we_logits = dh @ W_E.T

    wu_top = [t[0] for t in topk_tok(wu_logits, 5)]
    we_top = [t[0] for t in topk_tok(we_logits, 5)]

    print(f"    L{L:>2}  W_U: {wu_top}")
    print(f"         W_E: {we_top}")

del out_a, out_b
torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# STEP 4: fc1 alignment — is fc1 closer to W_E or W_U?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("STEP 4: fc1 ROW ALIGNMENT — closer to W_E or W_U?")
print("=" * 70)

for L in [_sl(4)[0], _sl(12)[0], _sl(20)[0], _sl(28)[0]]:
    if L >= NL:
        continue
    fc1_w = model.model.layers[L].mlp.fc1.weight.detach().float()

    # For each fc1 row, compute max cosine with any W_E row vs any W_U row
    # That's expensive. Instead: for each fc1 row, project through both and
    # measure how "peaky" the result is (max logit / mean logit)
    # Higher peak = more aligned with one specific token

    # Sample 500 neurons
    sample_idx = torch.randperm(fc1_w.shape[0])[:500]

    wu_peaks = []
    we_peaks = []
    for n in sample_idx:
        row = fc1_w[int(n)]
        wu_logits = row @ W_U.T
        we_logits = row @ W_E.T

        # Peak-to-mean ratio
        wu_peak = float(wu_logits.max() - wu_logits.mean()) / float(wu_logits.std() + 1e-6)
        we_peak = float(we_logits.max() - we_logits.mean()) / float(we_logits.std() + 1e-6)
        wu_peaks.append(wu_peak)
        we_peaks.append(we_peak)

    mean_wu = sum(wu_peaks) / len(wu_peaks)
    mean_we = sum(we_peaks) / len(we_peaks)
    print(f"  L{L:>2}: mean peak z-score  W_U={mean_wu:.2f}  W_E={mean_we:.2f}  "
          f"ratio W_U/W_E={mean_wu/max(mean_we,0.01):.2f}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
