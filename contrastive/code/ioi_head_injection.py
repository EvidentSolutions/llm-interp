"""
IOI causal tests: injection and activation patching.

Ablation showed name-mover heads aren't necessary (redundancy).
Now test sufficiency and patching:

1. INJECTION: Extract contrastive head output (H14+H1+H16) from
   John/Mary prompt. Inject into a prompt where a different name
   should be predicted. Does it shift toward Mary?

2. ACTIVATION PATCHING: Replace the name-mover heads' output in
   prompt B with the output from prompt A. Does B now predict like A?

3. NAME INJECTION: Inject just the W_U direction of a specific name
   token at L24. Does it shift prediction to that name?

4. CROSS-PAIR: Extract head output from John/Mary, inject into
   Alice/Bob. Does it shift from Bob toward Mary?

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
W_U = model.lm_head.weight.detach().float()


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


def capture_head_outputs(text, layer):
    """Capture pre-projection head outputs at the last position."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    captured = {}

    def dense_pre_hook(module, args):
        # args[0] shape: (batch, seq, d_model) — concatenated head outputs
        captured['pre_proj'] = args[0][0, -1, :].detach().float()
        return args

    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    handle = proj.register_forward_pre_hook(dense_pre_hook)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    handle.remove()
    return captured['pre_proj'], out


def inject_at_dense(text, layer, delta_pre_proj, k=5):
    """Inject a delta into the pre-projection output at last position."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    injected = [False]

    def dense_pre_hook(module, args):
        if not injected[0]:
            injected[0] = True
            inp = args[0].clone()
            inp[0, -1, :] += delta_pre_proj.half().to(DEV)
            return (inp,) + args[1:] if len(args) > 1 else (inp,)
        return args

    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    handle = proj.register_forward_pre_hook(dense_pre_hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def patch_heads_at_dense(text_target, text_source, layer, heads, k=5):
    """Replace specific heads in text_target with heads from text_source."""
    # First capture source head outputs
    source_pre, _ = capture_head_outputs(text_source, layer)

    ids = tok.encode(text_target, return_tensors="pt").to(DEV)
    patched = [False]

    def dense_pre_hook(module, args):
        if not patched[0]:
            patched[0] = True
            inp = args[0].clone()
            for h in heads:
                inp[0, -1, h*d_head:(h+1)*d_head] = source_pre[h*d_head:(h+1)*d_head].half().to(DEV)
            return (inp,) + args[1:] if len(args) > 1 else (inp,)
        return args

    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    handle = proj.register_forward_pre_hook(dense_pre_hook)
    with torch.no_grad():
        out = model(ids)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def inject_at_residual(text, layer, delta, k=5):
    """Inject delta into the residual stream after a layer."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
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
        out = model(ids)
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


L = 24
name_mover_heads = [14, 1, 16]

# ══════════════════════════════════════════════════════════════
# IOI prompts — A predicts Mary, B predicts John
# ══════════════════════════════════════════════════════════════
prompt_A = "When John and Mary went to the store, John gave a drink to"
prompt_B = "When Mary and John went to the store, Mary gave a drink to"
# Different pair
prompt_C = "When Alice and Bob went to the park, Alice gave a gift to"
prompt_D = "When Bob and Alice went to the park, Bob gave a gift to"

mary_id = tok.encode(" Mary", add_special_tokens=False)[0]
john_id = tok.encode(" John", add_special_tokens=False)[0]
bob_id = tok.encode(" Bob", add_special_tokens=False)[0]
alice_id = tok.encode(" Alice", add_special_tokens=False)[0]


# ══════════════════════════════════════════════════════════════
# TEST 1: BASELINES
# ══════════════════════════════════════════════════════════════
print("=" * 70)
print("TEST 1: BASELINES")
print("=" * 70)

for label, prompt in [("A (→Mary)", prompt_A), ("B (→John)", prompt_B),
                       ("C (→Bob)", prompt_C), ("D (→Alice)", prompt_D)]:
    preds, probs = predict(prompt)
    gen = generate(prompt)
    print(f"\n  {label}: \"{prompt}\"")
    print(f"    Top-3: {preds[:3]}")
    print(f"    P(Mary)={float(probs[mary_id]):.3f}  P(John)={float(probs[john_id]):.3f}  "
          f"P(Bob)={float(probs[bob_id]):.3f}  P(Alice)={float(probs[alice_id]):.3f}")
    print(f"    Generation: {gen}")


# ══════════════════════════════════════════════════════════════
# TEST 2: HEAD-LEVEL INJECTION
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 2: INJECT NAME-MOVER HEAD CONTRASTIVE OUTPUT")
print("=" * 70)
print("Extract H14+H1+H16 output difference (A−B), inject into B.")
print("If sufficient, B should shift from John toward Mary.\n")

pre_A, _ = capture_head_outputs(prompt_A, L)
pre_B, _ = capture_head_outputs(prompt_B, L)

# Full contrastive of all heads
delta_full = pre_A - pre_B

# Name-mover heads only
delta_nm = torch.zeros_like(delta_full)
for h in name_mover_heads:
    delta_nm[h*d_head:(h+1)*d_head] = delta_full[h*d_head:(h+1)*d_head]

# Non-name-mover heads
delta_other = delta_full - delta_nm

_, probs_base = predict(prompt_B)
print(f"  Baseline B: P(Mary)={float(probs_base[mary_id]):.3f}  "
      f"P(John)={float(probs_base[john_id]):.3f}")

for label, delta in [("all heads (A−B)", delta_full),
                      ("name-movers only", delta_nm),
                      ("non-name-movers only", delta_other)]:
    preds, probs = inject_at_dense(prompt_B, L, delta)
    print(f"  + {label}: P(Mary)={float(probs[mary_id]):.3f}  "
          f"P(John)={float(probs[john_id]):.3f}  top={preds[:3]}")

# Dose-response for name-mover injection
print(f"\n  Dose-response (name-mover heads into B):")
for frac in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
    preds, probs = inject_at_dense(prompt_B, L, delta_nm * frac)
    print(f"    +{frac:.2f}×: P(Mary)={float(probs[mary_id]):.3f}  "
          f"P(John)={float(probs[john_id]):.3f}  top-1={preds[0]}")

torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# TEST 3: ACTIVATION PATCHING — swap heads between A and B
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 3: ACTIVATION PATCHING — replace B's heads with A's")
print("=" * 70)

print(f"\n  Baseline B: P(John)={float(probs_base[john_id]):.3f}  "
      f"P(Mary)={float(probs_base[mary_id]):.3f}")

# Patch name-mover heads
preds_nm, probs_nm = patch_heads_at_dense(prompt_B, prompt_A, L, name_mover_heads)
print(f"  Patch name-movers (A→B): P(Mary)={float(probs_nm[mary_id]):.3f}  "
      f"P(John)={float(probs_nm[john_id]):.3f}  top={preds_nm[:3]}")

# Patch control heads
preds_ctrl, probs_ctrl = patch_heads_at_dense(prompt_B, prompt_A, L, [0, 5, 10])
print(f"  Patch controls (A→B):    P(Mary)={float(probs_ctrl[mary_id]):.3f}  "
      f"P(John)={float(probs_ctrl[john_id]):.3f}  top={preds_ctrl[:3]}")

# Patch ALL heads
preds_all, probs_all = patch_heads_at_dense(prompt_B, prompt_A, L, list(range(n_heads)))
print(f"  Patch ALL heads (A→B):   P(Mary)={float(probs_all[mary_id]):.3f}  "
      f"P(John)={float(probs_all[john_id]):.3f}  top={preds_all[:3]}")

# Reverse: patch A's heads with B's
_, probs_A_base = predict(prompt_A)
preds_rev, probs_rev = patch_heads_at_dense(prompt_A, prompt_B, L, name_mover_heads)
print(f"\n  Baseline A: P(Mary)={float(probs_A_base[mary_id]):.3f}  "
      f"P(John)={float(probs_A_base[john_id]):.3f}")
print(f"  Patch name-movers (B→A): P(Mary)={float(probs_rev[mary_id]):.3f}  "
      f"P(John)={float(probs_rev[john_id]):.3f}  top={preds_rev[:3]}")


# ══════════════════════════════════════════════════════════════
# TEST 4: CROSS-PAIR — inject John/Mary head output into Alice/Bob
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 4: CROSS-PAIR — inject John/Mary heads into Alice/Bob")
print("=" * 70)

_, probs_C_base = predict(prompt_C)
print(f"\n  Baseline C (→Bob): P(Bob)={float(probs_C_base[bob_id]):.3f}  "
      f"P(Alice)={float(probs_C_base[alice_id]):.3f}  "
      f"P(Mary)={float(probs_C_base[mary_id]):.3f}")

# Inject John/Mary name-mover delta into Alice/Bob prompt
for frac in [0.5, 1.0, 2.0]:
    preds, probs = inject_at_dense(prompt_C, L, delta_nm * frac)
    print(f"  + John/Mary NM heads {frac:.1f}×: "
          f"P(Bob)={float(probs[bob_id]):.3f}  "
          f"P(Alice)={float(probs[alice_id]):.3f}  "
          f"P(Mary)={float(probs[mary_id]):.3f}  "
          f"top={preds[:3]}")

# Patch: replace Alice/Bob's name-mover heads with John/Mary's
preds_xp, probs_xp = patch_heads_at_dense(prompt_C, prompt_A, L, name_mover_heads)
print(f"  Patch NM (JohnMary→AliceBob): "
      f"P(Bob)={float(probs_xp[bob_id]):.3f}  "
      f"P(Alice)={float(probs_xp[alice_id]):.3f}  "
      f"P(Mary)={float(probs_xp[mary_id]):.3f}  "
      f"top={preds_xp[:3]}")


# ══════════════════════════════════════════════════════════════
# TEST 5: FULL Δh INJECTION AT RESIDUAL STREAM
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TEST 5: FULL Δh INJECTION (residual stream, not per-head)")
print("=" * 70)

# Get hidden states
ids_A = tok.encode(prompt_A, return_tensors="pt").to(DEV)
ids_B = tok.encode(prompt_B, return_tensors="pt").to(DEV)
with torch.no_grad():
    out_A = model(ids_A, output_hidden_states=True)
    out_B = model(ids_B, output_hidden_states=True)

for test_L in [20, 24, 28]:
    h_A = out_A.hidden_states[test_L][0, -1, :].float()
    h_B = out_B.hidden_states[test_L][0, -1, :].float()
    dh = h_A - h_B

    preds, probs = inject_at_residual(prompt_B, test_L, dh)
    print(f"  L{test_L} full Δh into B: P(Mary)={float(probs[mary_id]):.3f}  "
          f"P(John)={float(probs[john_id]):.3f}  top={preds[:3]}")

del out_A, out_B
torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
