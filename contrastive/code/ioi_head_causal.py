"""
Causal verification of IOI name-mover heads identified by contrastive projection.

§5.1 identifies H14, H1, H16 at L24 as candidate name-mover heads by
contrastive norm. We test causally:

1. ABLATION: Zero each head's output at L24. Does the model lose the
   ability to predict the correct IO name?
2. INJECTION: Take the contrastive head output from one prompt and
   inject it into the other. Does the prediction flip?
3. GENERATION: Show greedy continuations with and without ablation.
4. SPECIFICITY: Ablating non-name-mover heads should have less effect.

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
n_heads = model.config.num_attention_heads
d_model = model.config.hidden_size
d_head = d_model // n_heads


def predict(text, k=5):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(ids)
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def generate(text, max_new=20):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def predict_with_head_ablation(text, layer, heads_to_ablate, k=5):
    """Zero out specific attention heads' output at a layer."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    ablated = [False]

    def hook_fn(module, input, output):
        if ablated[0]:
            return output
        ablated[0] = True
        # output is attn_output: (batch, seq, d_model)
        # Each head contributes d_head dimensions after the output projection
        # We need to hook on the attention module's output before o_proj
        # Actually, the attention layer output includes o_proj already
        # We need to zero heads BEFORE o_proj, or zero their contribution after
        # Simpler: hook on self_attn, zero specific head slices in the
        # pre-projection output
        return output

    # Better approach: hook on the attention output projection input
    # to zero specific heads before they're projected
    def attn_hook(module, args, output):
        if ablated[0]:
            return output
        ablated[0] = True

        if isinstance(output, tuple):
            attn_out = output[0]  # (batch, seq, d_model)
        else:
            attn_out = output

        # The attention output is already mixed by o_proj
        # We need to intervene earlier. Let's use a different approach:
        # Capture the full attention output, then subtract the contribution
        # of the ablated heads.
        return output

    # Cleanest approach: compute the model twice, once normal, once with
    # the head's contribution subtracted from the layer output
    # First pass: capture per-head outputs
    head_outputs = {}

    def capture_hook(module, args):
        # Pre-hook on self_attn to capture input
        inp = args[0] if isinstance(args, tuple) else args
        head_outputs['input'] = inp.detach()
        return None

    # Actually, the simplest causal test: run the model, capture hidden states
    # at the layer, compute what each head contributes, subtract specific heads,
    # and re-run from that layer onward.
    # But that requires careful hook management.

    # Simpler: use the dense output projection. In Phi-2, the attention
    # output passes through dense (o_proj). Each head's contribution to the
    # final output is: attn_weights @ V @ W_o[head_slice]
    # We can zero the head's slice of the attention output before o_proj.

    # Let's hook on the output of the attention's internal computation
    # before the output projection.

    # For Phi-2: model.model.layers[L].self_attn
    # The forward pass computes query, key, value, attention weights,
    # context = attn_weights @ value, then output = dense(context)
    # We want to zero context[:, :, head*d_head:(head+1)*d_head] = 0

    # Hook on dense (o_proj) input
    def dense_pre_hook(module, args):
        if not ablated[0]:
            ablated[0] = True
            inp = args[0].clone()
            for h in heads_to_ablate:
                inp[0, -1, h*d_head:(h+1)*d_head] = 0.0
            return (inp,) + args[1:] if len(args) > 1 else (inp,)
        return args

    # Find the dense/o_proj layer
    attn = model.model.layers[layer].self_attn
    # Phi-2 uses 'dense' for output projection
    if hasattr(attn, 'dense'):
        handle = attn.dense.register_forward_pre_hook(dense_pre_hook)
    elif hasattr(attn, 'o_proj'):
        handle = attn.o_proj.register_forward_pre_hook(dense_pre_hook)
    else:
        print("  WARNING: cannot find output projection layer")
        return None, None

    with torch.no_grad():
        out = model(ids)
    handle.remove()

    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def generate_with_head_ablation(text, layer, heads_to_ablate, max_new=20):
    """Generate with heads ablated."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    call_count = [0]

    def dense_pre_hook(module, args):
        call_count[0] += 1
        if call_count[0] <= 1:  # Only ablate on first forward pass
            inp = args[0].clone()
            for h in heads_to_ablate:
                inp[0, -1, h*d_head:(h+1)*d_head] = 0.0
            return (inp,) + args[1:] if len(args) > 1 else (inp,)
        return args

    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    handle = proj.register_forward_pre_hook(dense_pre_hook)

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    handle.remove()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


# ══════════════════════════════════════════════════════════════
# IOI PROMPTS
# ══════════════════════════════════════════════════════════════

ioi_cases = [
    ("John/Mary",
     "When John and Mary went to the store, John gave a drink to",
     "Mary", "John"),
    ("Alice/Bob",
     "When Alice and Bob went to the park, Alice gave a gift to",
     "Bob", "Alice"),
    ("Dan/Eve",
     "When Dan and Eve went to the office, Dan gave a letter to",
     "Eve", "Dan"),
]

L = 24  # Layer where name-mover heads were identified
name_mover_heads = [14, 1, 16]  # Identified in §5.1
control_heads = [0, 5, 10]  # Heads NOT identified as name-movers

# ══════════════════════════════════════════════════════════════
# TEST 1: BASELINE PREDICTIONS
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: BASELINE PREDICTIONS")
print("=" * 70)

for name, prompt, io_name, subj_name in ioi_cases:
    preds, probs = predict(prompt)
    io_id = tok.encode(f" {io_name}", add_special_tokens=False)[0]
    subj_id = tok.encode(f" {subj_name}", add_special_tokens=False)[0]
    p_io = float(probs[io_id])
    p_subj = float(probs[subj_id])
    gen = generate(prompt)
    print(f"\n  {name}: \"{prompt}\"")
    print(f"    Top-5: {preds}")
    print(f"    P({io_name})={p_io:.3f}  P({subj_name})={p_subj:.3f}")
    print(f"    Generation: {gen}")


# ══════════════════════════════════════════════════════════════
# TEST 2: ABLATE NAME-MOVER HEADS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"TEST 2: ABLATE NAME-MOVER HEADS (H{name_mover_heads}) at L{L}")
print("=" * 70)

for name, prompt, io_name, subj_name in ioi_cases:
    io_id = tok.encode(f" {io_name}", add_special_tokens=False)[0]
    subj_id = tok.encode(f" {subj_name}", add_special_tokens=False)[0]

    # Baseline
    _, probs_base = predict(prompt)
    p_io_base = float(probs_base[io_id])
    p_subj_base = float(probs_base[subj_id])

    print(f"\n  {name}: \"{prompt}\"")
    print(f"    Baseline: P({io_name})={p_io_base:.3f}  P({subj_name})={p_subj_base:.3f}")

    # Ablate all three name-movers
    preds_abl, probs_abl = predict_with_head_ablation(prompt, L, name_mover_heads)
    if probs_abl is not None:
        p_io_abl = float(probs_abl[io_id])
        p_subj_abl = float(probs_abl[subj_id])
        gen_abl = generate_with_head_ablation(prompt, L, name_mover_heads)
        print(f"    Ablate H{name_mover_heads}: P({io_name})={p_io_abl:.3f}  "
              f"P({subj_name})={p_subj_abl:.3f}")
        print(f"    Top-5: {preds_abl}")
        print(f"    Generation: {gen_abl}")

    # Ablate each individually
    for h in name_mover_heads:
        preds_h, probs_h = predict_with_head_ablation(prompt, L, [h])
        if probs_h is not None:
            p_io_h = float(probs_h[io_id])
            p_subj_h = float(probs_h[subj_id])
            print(f"    Ablate H{h} only: P({io_name})={p_io_h:.3f}  "
                  f"P({subj_name})={p_subj_h:.3f}  top={preds_h[:2]}")


# ══════════════════════════════════════════════════════════════
# TEST 3: ABLATE CONTROL HEADS (should have less effect)
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"TEST 3: ABLATE CONTROL HEADS (H{control_heads}) at L{L}")
print("=" * 70)

for name, prompt, io_name, subj_name in ioi_cases:
    io_id = tok.encode(f" {io_name}", add_special_tokens=False)[0]
    subj_id = tok.encode(f" {subj_name}", add_special_tokens=False)[0]

    _, probs_base = predict(prompt)
    p_io_base = float(probs_base[io_id])

    preds_ctrl, probs_ctrl = predict_with_head_ablation(prompt, L, control_heads)
    if probs_ctrl is not None:
        p_io_ctrl = float(probs_ctrl[io_id])
        p_subj_ctrl = float(probs_ctrl[subj_id])
        print(f"\n  {name}:")
        print(f"    Baseline: P({io_name})={p_io_base:.3f}")
        print(f"    Ablate H{control_heads}: P({io_name})={p_io_ctrl:.3f}  "
              f"P({subj_name})={p_subj_ctrl:.3f}  top={preds_ctrl[:3]}")


# ══════════════════════════════════════════════════════════════
# TEST 4: ABLATE AT DIFFERENT LAYERS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: LAYER SWEEP — ablate H14,H1,H16 at different layers")
print("=" * 70)

prompt = ioi_cases[0][1]  # John/Mary
io_name = "Mary"
io_id = tok.encode(f" {io_name}", add_special_tokens=False)[0]
subj_id = tok.encode(" John", add_special_tokens=False)[0]

_, probs_base = predict(prompt)
p_io_base = float(probs_base[io_id])

print(f"\n  Prompt: \"{prompt}\"")
print(f"  Baseline: P(Mary)={p_io_base:.3f}")

for test_L in [8, 12, 16, 20, 24, 28]:
    preds_l, probs_l = predict_with_head_ablation(prompt, test_L, name_mover_heads)
    if probs_l is not None:
        p_io_l = float(probs_l[io_id])
        p_subj_l = float(probs_l[subj_id])
        print(f"  Ablate at L{test_L:>2}: P(Mary)={p_io_l:.3f}  "
              f"P(John)={p_subj_l:.3f}  top={preds_l[:3]}")


# ══════════════════════════════════════════════════════════════
# TEST 5: ALL-HEADS SWEEP AT L24 — which heads matter most?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: PER-HEAD ABLATION SWEEP at L24 — which heads matter?")
print("=" * 70)

prompt = ioi_cases[0][1]
io_id = tok.encode(" Mary", add_special_tokens=False)[0]
subj_id = tok.encode(" John", add_special_tokens=False)[0]

_, probs_base = predict(prompt)
p_io_base = float(probs_base[io_id])
p_subj_base = float(probs_base[subj_id])

print(f"\n  Baseline: P(Mary)={p_io_base:.3f}  P(John)={p_subj_base:.3f}")
print(f"\n  {'Head':>6}  {'P(Mary)':>8}  {'P(John)':>8}  {'ΔP(Mary)':>9}  {'Effect':>8}")

results = []
for h in range(n_heads):
    _, probs_h = predict_with_head_ablation(prompt, 24, [h])
    if probs_h is not None:
        p_io = float(probs_h[io_id])
        p_subj = float(probs_h[subj_id])
        delta = p_io - p_io_base
        results.append((h, p_io, p_subj, delta))

# Sort by impact
results.sort(key=lambda x: x[3])
for h, p_io, p_subj, delta in results:
    marker = " ← NAME-MOVER" if h in name_mover_heads else ""
    if abs(delta) > 0.01 or h in name_mover_heads:
        print(f"  H{h:>4}  {p_io:>8.3f}  {p_subj:>8.3f}  {delta:>+9.3f}{marker}")


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
