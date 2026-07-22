"""
Adjacent-position contrastive sweep on natural text.

For each adjacent token pair (pos, pos+1) in continuous text, compute:
  Δh[L] = h[L, pos+1] - h[L, pos]
and project through W_U to read what changes between positions.

Decomposes by: layer, attention head, and sub-layer (attn vs MLP).

The hypothesis: there is a CONSTANT part of the representation that
encodes accumulated context, and a VARIABLE part that processes the
current token and forms the next prediction.  Heads with low coefficient
of variation across positions are "context heads"; heads with high CV
are "prediction heads".

Usage: .venv\Scripts\python.exe contrastive\code\adjacent_position_sweep.py

Outputs saved to contrastive/results/adjacent_sweep_<model>.pt
"""
import sys
import os
import time
import json
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")
CHUNK_SIZE = 1024          # tokens per forward pass
TARGET_PAIRS = 1000        # aim for this many adjacent pairs
TOP_K = 10                 # top-k tokens to store per contrast
CORPUS_DIR = Path("rosetta/corpus")
OUT_DIR = Path("contrastive/results")

# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
print(f"Loading {MODEL} on {DEV}...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token
for p in model.parameters():
    p.requires_grad_(False)
print(f"  loaded in {time.time()-t0:.1f}s")

NL = model.config.num_hidden_layers
NH = model.config.num_attention_heads
HD = model.config.hidden_size // NH
HIDDEN = model.config.hidden_size
VOCAB = model.config.vocab_size

W_U = model.lm_head.weight.detach().float()  # (vocab, hidden)

_sample_attn = model.model.layers[0].self_attn
DENSE_ATTR = "dense" if hasattr(_sample_attn, "dense") else "o_proj"

print(f"  NL={NL}, NH={NH}, HD={HD}, hidden={HIDDEN}, vocab={VOCAB}")

# -------------------------------------------------------------------
# Load and tokenize corpus
# -------------------------------------------------------------------
print(f"\nLoading corpora from {CORPUS_DIR}...")
corpus_files = sorted(CORPUS_DIR.glob("*.txt"))
all_text = []
for f in corpus_files:
    text = f.read_text(encoding="utf-8").strip()
    all_text.append(text)
    print(f"  {f.name}: {len(text)} chars")

combined = "\n\n".join(all_text)
all_ids = tok(combined, add_special_tokens=False)["input_ids"]
print(f"  total tokens: {len(all_ids)}")

# Take first CHUNK_SIZE+1 tokens (to get CHUNK_SIZE adjacent pairs)
n_tokens = min(len(all_ids), TARGET_PAIRS + 1)
input_ids = all_ids[:n_tokens]
n_pairs = n_tokens - 1
token_strs = [tok.decode([t]) for t in input_ids]

print(f"  using {n_tokens} tokens -> {n_pairs} adjacent pairs")
print(f"  first 20 tokens: {token_strs[:20]}")

# -------------------------------------------------------------------
# Forward pass with hooks
# -------------------------------------------------------------------
print(f"\nRunning forward pass ({n_tokens} tokens)...")

# We need:
# 1. hidden_states at all layers (from output_hidden_states=True)
# 2. per-head dense inputs at all layers (for head decomposition)
# 3. attention outputs at all layers (for attn vs MLP split)

dense_inputs = {}   # layer -> (seq, hidden)  input to O-proj
attn_outputs = {}   # layer -> (seq, hidden)  output of O-proj
hooks = []

for L in range(NL):
    dense = getattr(model.model.layers[L].self_attn, DENSE_ATTR)

    def make_hook(layer_idx):
        def hook_fn(module, inp, out):
            # inp[0] = concatenated head outputs before O-proj: (batch, seq, hidden)
            # out    = after O-proj: (batch, seq, hidden)
            dense_inputs[layer_idx] = inp[0][0].detach().float().cpu()
            attn_outputs[layer_idx] = out[0].detach().float().cpu() if isinstance(out, tuple) else out.detach().float().cpu()
        return hook_fn

    h = dense.register_forward_hook(make_hook(L))
    hooks.append(h)

ids_tensor = torch.tensor([input_ids], device=DEV)
t0 = time.time()
with torch.no_grad():
    out = model(ids_tensor, output_hidden_states=True)
print(f"  forward pass: {time.time()-t0:.1f}s")

for h in hooks:
    h.remove()

# Extract hidden states to CPU
hidden_states = []
for L in range(NL + 1):
    hidden_states.append(out.hidden_states[L][0].detach().float().cpu())
    # (seq, hidden)

# Final logits for predictions
final_logits = out.logits[0].detach().float().cpu()  # (seq, vocab)

del out
torch.cuda.empty_cache()

# -------------------------------------------------------------------
# Compute W_O matrices per head (needed for head decomposition)
# -------------------------------------------------------------------
print("\nPrecomputing W_O slices...")
W_O_per_layer = {}
for L in range(NL):
    W_O = getattr(model.model.layers[L].self_attn, DENSE_ATTR).weight.float().cpu()
    # W_O shape: (hidden, hidden)
    W_O_per_layer[L] = W_O

W_U_cpu = W_U.cpu()

# -------------------------------------------------------------------
# Compute all adjacent-position contrasts
# -------------------------------------------------------------------
print(f"\nComputing contrasts for {n_pairs} adjacent pairs...")
t0 = time.time()

# Storage tensors
layer_delta_norms = torch.zeros(NL, n_pairs)          # ||Δh[L]||
layer_top_ids = torch.zeros(NL, n_pairs, TOP_K, dtype=torch.long)
layer_top_vals = torch.zeros(NL, n_pairs, TOP_K)

head_delta_norms = torch.zeros(NL, NH, n_pairs)       # ||Δh_head[L,h]||
head_top_ids = torch.zeros(NL, NH, n_pairs, TOP_K, dtype=torch.long)
head_top_vals = torch.zeros(NL, NH, n_pairs, TOP_K)

attn_delta_norms = torch.zeros(NL, n_pairs)            # ||Δattn[L]||
mlp_delta_norms = torch.zeros(NL, n_pairs)             # ||Δmlp[L]||

# Predictions at each position (from final logits)
pred_top_ids = torch.zeros(n_tokens, TOP_K, dtype=torch.long)
pred_top_vals = torch.zeros(n_tokens, TOP_K)

for pos in range(n_tokens):
    vals, ids = torch.topk(final_logits[pos], TOP_K)
    pred_top_ids[pos] = ids
    pred_top_vals[pos] = vals

# Logit lens at each position per layer
logit_lens_top_ids = torch.zeros(NL + 1, n_tokens, TOP_K, dtype=torch.long)
logit_lens_top_vals = torch.zeros(NL + 1, n_tokens, TOP_K)

print("  computing logit lens per position per layer...")
for L in range(NL + 1):
    hs = hidden_states[L]  # (seq, hidden)
    ll = hs @ W_U_cpu.T    # (seq, vocab)
    vals, ids = torch.topk(ll, TOP_K, dim=1)
    logit_lens_top_ids[L] = ids
    logit_lens_top_vals[L] = vals

print("  computing adjacent contrasts per layer...")
for L in range(NL):
    # Layer-level contrast: use post-layer hidden states (hidden_states[L+1])
    hs = hidden_states[L + 1]  # (seq, hidden)
    # Δh for adjacent pairs: h[pos+1] - h[pos]
    dh = hs[1:] - hs[:-1]  # (n_pairs, hidden)
    layer_delta_norms[L] = dh.norm(dim=1)

    # Project through W_U
    dl = dh @ W_U_cpu.T  # (n_pairs, vocab)
    vals, ids = torch.topk(dl, TOP_K, dim=1)
    layer_top_ids[L] = ids
    layer_top_vals[L] = vals

    # Sub-layer: attn vs MLP
    # Phi-2 parallel: post = pre + attn + mlp
    # pre = hidden_states[L], post = hidden_states[L+1]
    # attn_out from hook
    a_out = attn_outputs[L]  # (seq, hidden)
    da = a_out[1:] - a_out[:-1]  # (n_pairs, hidden)
    attn_delta_norms[L] = da.norm(dim=1)

    # mlp = post - pre - attn
    pre = hidden_states[L]
    mlp_out = hs - pre - a_out
    dm = mlp_out[1:] - mlp_out[:-1]
    mlp_delta_norms[L] = dm.norm(dim=1)

    # Per-head decomposition
    di = dense_inputs[L]  # (seq, hidden) - input to O-proj
    W_O = W_O_per_layer[L]

    for h_idx in range(NH):
        # Head h's contribution: dense_input[:, h*HD:(h+1)*HD] @ W_O[:, h*HD:(h+1)*HD].T
        h_in = di[:, h_idx * HD : (h_idx + 1) * HD]  # (seq, HD)
        W_O_h = W_O[:, h_idx * HD : (h_idx + 1) * HD]  # (hidden, HD)
        h_out = h_in @ W_O_h.T  # (seq, hidden)

        dh_head = h_out[1:] - h_out[:-1]  # (n_pairs, hidden)
        head_delta_norms[L, h_idx] = dh_head.norm(dim=1)

        # Project per-head contrast through W_U
        dl_head = dh_head @ W_U_cpu.T  # (n_pairs, vocab)
        vals, ids = torch.topk(dl_head, TOP_K, dim=1)
        head_top_ids[L, h_idx] = ids
        head_top_vals[L, h_idx] = vals

    if (L + 1) % 8 == 0:
        print(f"    layer {L+1}/{NL} done")

elapsed = time.time() - t0
print(f"  all contrasts computed in {elapsed:.1f}s")

# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------
model_short = MODEL.split("/")[-1]
out_path = OUT_DIR / f"adjacent_sweep_{model_short}.pt"

results = {
    "model": MODEL,
    "n_tokens": n_tokens,
    "n_pairs": n_pairs,
    "n_layers": NL,
    "n_heads": NH,
    "head_dim": HD,
    "hidden_size": HIDDEN,
    "top_k": TOP_K,
    "token_ids": torch.tensor(input_ids, dtype=torch.long),
    "token_strs": token_strs,
    # Per-layer adjacent contrasts
    "layer_delta_norms": layer_delta_norms,          # (NL, n_pairs)
    "layer_top_ids": layer_top_ids,                  # (NL, n_pairs, K)
    "layer_top_vals": layer_top_vals,                # (NL, n_pairs, K)
    # Per-head adjacent contrasts
    "head_delta_norms": head_delta_norms,            # (NL, NH, n_pairs)
    "head_top_ids": head_top_ids,                    # (NL, NH, n_pairs, K)
    "head_top_vals": head_top_vals,                  # (NL, NH, n_pairs, K)
    # Sub-layer split
    "attn_delta_norms": attn_delta_norms,            # (NL, n_pairs)
    "mlp_delta_norms": mlp_delta_norms,              # (NL, n_pairs)
    # Predictions at each position
    "pred_top_ids": pred_top_ids,                    # (n_tokens, K)
    "pred_top_vals": pred_top_vals,                  # (n_tokens, K)
    # Logit lens at each position per layer
    "logit_lens_top_ids": logit_lens_top_ids,        # (NL+1, n_tokens, K)
    "logit_lens_top_vals": logit_lens_top_vals,      # (NL+1, n_tokens, K)
}

torch.save(results, out_path)
file_size = out_path.stat().st_size / 1024 / 1024
print(f"\nSaved to {out_path} ({file_size:.1f} MB)")

# -------------------------------------------------------------------
# Print summary statistics
# -------------------------------------------------------------------
print(f"\n{'='*100}")
print("SUMMARY STATISTICS")
print(f"{'='*100}")

# 1. Per-layer mean contrastive norm
print(f"\n--- Per-layer mean adjacent-contrastive norm ---")
print(f"{'Layer':>6} {'mean||Δh||':>10} {'std':>8} {'||Δattn||':>10} {'||Δmlp||':>10} {'attn%':>7}")
for L in range(NL):
    mn = float(layer_delta_norms[L].mean())
    sd = float(layer_delta_norms[L].std())
    ma = float(attn_delta_norms[L].mean())
    mm = float(mlp_delta_norms[L].mean())
    apct = ma / (ma + mm) * 100 if (ma + mm) > 0 else 0
    print(f"  L{L:>3} {mn:>10.1f} {sd:>8.1f} {ma:>10.1f} {mm:>10.1f} {apct:>6.1f}%")

# 2. Per-head: mean, std, and CV of contrastive norms
print(f"\n--- Head coefficient of variation (CV = std/mean) across positions ---")
print(f"  Low CV = consistent contribution (context processing)")
print(f"  High CV = variable contribution (token/prediction processing)")

head_means = head_delta_norms.mean(dim=2)  # (NL, NH)
head_stds = head_delta_norms.std(dim=2)
head_cv = head_stds / (head_means + 1e-8)

# Show top-10 most constant and most variable heads
all_heads = []
for L in range(NL):
    for h in range(NH):
        all_heads.append((L, h, float(head_means[L, h]),
                          float(head_stds[L, h]),
                          float(head_cv[L, h])))

# Filter to heads with meaningful norm (mean > 1.0)
active_heads = [x for x in all_heads if x[2] > 1.0]
active_heads.sort(key=lambda x: x[4])

print(f"\n  MOST CONSTANT heads (lowest CV, mean norm > 1.0):")
print(f"  {'Head':>8} {'mean':>8} {'std':>8} {'CV':>8}")
for L, h, mn, sd, cv in active_heads[:15]:
    print(f"  L{L:>2}H{h:>2} {mn:>8.1f} {sd:>8.1f} {cv:>8.3f}")

print(f"\n  MOST VARIABLE heads (highest CV, mean norm > 1.0):")
print(f"  {'Head':>8} {'mean':>8} {'std':>8} {'CV':>8}")
for L, h, mn, sd, cv in active_heads[-15:]:
    print(f"  L{L:>2}H{h:>2} {mn:>8.1f} {sd:>8.1f} {cv:>8.3f}")

# 3. Per-layer: what fraction of contrastive norm comes from top-3 heads?
print(f"\n--- Head concentration: fraction of total norm from top-3 heads per layer ---")
print(f"{'Layer':>6} {'top3/total':>10} {'top3 heads':>20} {'top3 norms':>20}")
for L in range(NL):
    mean_norms = head_means[L]  # (NH,)
    total = float(mean_norms.sum())
    top3_idx = torch.argsort(mean_norms, descending=True)[:3]
    top3_sum = float(mean_norms[top3_idx].sum())
    frac = top3_sum / total if total > 0 else 0
    heads_str = ",".join(f"H{int(i)}" for i in top3_idx)
    norms_str = ",".join(f"{float(mean_norms[int(i)]):.1f}" for i in top3_idx)
    print(f"  L{L:>3} {frac:>9.1%} {heads_str:>20} {norms_str:>20}")

# 4. Sample adjacent contrasts (show a few)
print(f"\n--- Sample adjacent contrasts (last 3 layers) ---")
sample_positions = [10, 50, 100, 200, 500, 800]
for pos in sample_positions:
    if pos >= n_pairs:
        break
    t_left = token_strs[pos].replace('\n', '\\n')
    t_right = token_strs[pos + 1].replace('\n', '\\n')
    pred_left = tok.decode([int(pred_top_ids[pos, 0])])
    pred_right = tok.decode([int(pred_top_ids[pos + 1, 0])])
    print(f"\n  pos {pos}: [{t_left!r}] -> [{t_right!r}]")
    print(f"    predictions: [{pred_left!r}] -> [{pred_right!r}]")
    for L in [NL - 3, NL - 2, NL - 1]:
        top_toks = [tok.decode([int(layer_top_ids[L, pos, k])]).strip()[:12]
                    for k in range(5)]
        norm = float(layer_delta_norms[L, pos])
        print(f"    L{L}: norm={norm:.1f}  Δ=[{', '.join(top_toks)}]")
        # Top-3 contributing heads
        h_norms = head_delta_norms[L, :, pos]
        top3h = torch.argsort(h_norms, descending=True)[:3]
        for hi in top3h:
            hi = int(hi)
            hn = float(h_norms[hi])
            h_toks = [tok.decode([int(head_top_ids[L, hi, pos, k])]).strip()[:12]
                      for k in range(5)]
            print(f"      H{hi:>2} (norm={hn:.1f}): [{', '.join(h_toks)}]")

# 5. Shared prediction analysis: how often do adjacent positions predict
#    similar tokens?
print(f"\n--- Prediction overlap between adjacent positions ---")
overlap_counts = []
for pos in range(n_pairs):
    set_a = set(pred_top_ids[pos, :5].tolist())
    set_b = set(pred_top_ids[pos + 1, :5].tolist())
    overlap_counts.append(len(set_a & set_b))
overlap_counts = torch.tensor(overlap_counts, dtype=torch.float)
print(f"  Mean top-5 prediction overlap: {float(overlap_counts.mean()):.2f} / 5")
print(f"  Positions with 0 overlap: {int((overlap_counts == 0).sum())} / {n_pairs}")
print(f"  Positions with 5 overlap: {int((overlap_counts == 5).sum())} / {n_pairs}")

# 6. Layer-by-layer prediction stability (logit lens)
print(f"\n--- Logit lens prediction stability across adjacent positions ---")
print(f"  (fraction of positions where top-1 logit-lens token is same for pos and pos+1)")
print(f"{'Layer':>6} {'same_top1':>10}")
for L in range(0, NL + 1, max(1, NL // 8)):
    same = 0
    for pos in range(n_pairs):
        if logit_lens_top_ids[L, pos, 0] == logit_lens_top_ids[L, pos + 1, 0]:
            same += 1
    print(f"  L{L:>3} {same/n_pairs:>9.1%}")

print(f"\n{'='*100}")
print("DONE")
print(f"Results saved to: {out_path}")
print(f"Load with: r = torch.load('{out_path}')")
