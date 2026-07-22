"""
IOI causal path via contrastive tools — three moves the original §5.1 missed.

The original name-swap contrast (A: "...John...John gave to"→Mary vs
B: "...Mary...Mary gave to"→John) cancels the SHARED inhibition algorithm
and keeps only name identity. So it found name-content carriers, and
ablating them under a redundant/distributed copy scheme showed nothing.

This script tests whether a better contrast design + attention + denoising
recovers an actual causal path:

  MOVE 1 — Contrast design comparison.
     (a) name-swap contrast (original): cancels inhibition.
     (b) no-duplicate contrast: clean IOI vs third-name giver. Keeps the
         duplicate-detection / inhibition computation in the subtraction.
     Per-head delta decomposition (residual-contribution space) + W_U reads,
     layer sweep. Does (b) light up different heads than (a)?

  MOVE 2 — Attention-pattern confirmation.
     For the candidate heads, where does END attend? A genuine mover should
     read from the IO token position; an inhibition head from S2.

  MOVE 3 — Denoising / path patching.
     Corrupt = the swapped prompt (predicts John). Patch CLEAN head outputs
     into the corrupted run, per head and cumulative, and measure how much
     P(Mary) is RESTORED. Denoising is sensitive where ablation is blind.

Full prompts shown with all results.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = os.environ.get("MODEL", "microsoft/phi-2")

print(f"Loading {MODEL} on {DEV}...")
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True
).to(DEV).eval()  # default SDPA; eager (needed for output_attentions) breaks
# fp16 logits, so we switch to eager only transiently for MOVE 2.
for p in model.parameters():
    p.requires_grad_(False)

NL = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
d_model = model.config.hidden_size
d_head = d_model // n_heads
W_U = model.lm_head.weight.detach().float()


def tid(name):
    return tok.encode(f" {name}", add_special_tokens=False)[0]


def predict(text, k=5):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(ids)
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    topk_v, topk_i = torch.topk(probs, k)
    return [(tok.decode([int(topk_i[j])]).strip()[:14], float(topk_v[j]))
            for j in range(k)], probs


def generate(text, max_new=18):
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def topk_tok(vec, k=5):
    """Top-k tokens for a d_model vector projected through W_U."""
    logits = vec.float().to(W_U.device) @ W_U.T
    idx = torch.topk(logits, k).indices
    return [tok.decode([int(i)]).strip()[:10] for i in idx]


def capture_context(text, layer):
    """Capture pre-dense (concatenated head) context at all positions."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    captured = {}

    def dense_pre_hook(module, args):
        captured['ctx'] = args[0][0].detach().float()  # (seq, d_model)
        return args

    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    handle = proj.register_forward_pre_hook(dense_pre_hook)
    with torch.no_grad():
        model(ids)
    handle.remove()
    return captured['ctx'], ids


def per_head_delta(ctx_a, ctx_b, layer, pos=-1):
    """Per-head contribution of (a-b) to the residual stream at a position.

    o_proj is linear: out = ctx @ W_o.T. Head h contributes
    ctx[:, h*dh:(h+1)*dh] @ W_o[:, h*dh:(h+1)*dh].T.
    Returns list of (head, norm, delta_vec).
    """
    attn = model.model.layers[layer].self_attn
    proj = attn.dense if hasattr(attn, 'dense') else attn.o_proj
    W_o = proj.weight.detach().float()  # (d_model, d_model)
    da = ctx_a[pos]
    db = ctx_b[pos]
    out = []
    for h in range(n_heads):
        s, e = h * d_head, (h + 1) * d_head
        dvec = (da[s:e] - db[s:e]).to(W_o.device) @ W_o[:, s:e].T  # (d_model,)
        out.append((h, float(dvec.norm()), dvec.cpu()))
    return out


def attention_from_end(text, layer):
    """Return attention distribution from END over key positions, per head."""
    ids = tok.encode(text, return_tensors="pt").to(DEV)
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    att = out.attentions[layer][0]  # (n_heads, seq, seq)
    return att[:, -1, :].float().cpu(), ids


def patch_heads(text_target, text_source, layer, heads, k=5):
    """Denoising: replace target's head outputs at END with source's."""
    src_ctx, _ = capture_context(text_source, layer)
    src_end = src_ctx[-1]  # (d_model,)

    ids = tok.encode(text_target, return_tensors="pt").to(DEV)
    patched = [False]

    def dense_pre_hook(module, args):
        if not patched[0]:
            patched[0] = True
            inp = args[0].clone()
            for h in heads:
                s, e = h * d_head, (h + 1) * d_head
                inp[0, -1, s:e] = src_end[s:e].half().to(DEV)
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


# ══════════════════════════════════════════════════════════════
# PROMPTS  (token-aligned: only the giver token differs)
# ══════════════════════════════════════════════════════════════
CLEAN   = "When John and Mary went to the store, John gave a drink to"   # → Mary
SWAPPED = "When John and Mary went to the store, Mary gave a drink to"   # → John
NODUPE  = "When John and Mary went to the store, Sam gave a drink to"    # ambiguous

mary_id, john_id, sam_id = tid("Mary"), tid("John"), tid("Sam")

# token alignment check (denoising needs identical length & END token)
for label, p in [("CLEAN", CLEAN), ("SWAPPED", SWAPPED), ("NODUPE", NODUPE)]:
    ids = tok.encode(p, add_special_tokens=False)
    print(f"  {label}: {len(ids)} tokens, END token = "
          f"{tok.decode([ids[-1]])!r}")


# ══════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BASELINES")
print("=" * 70)
for label, p in [("CLEAN  (→Mary)", CLEAN), ("SWAPPED(→John)", SWAPPED),
                 ("NODUPE (ambig)", NODUPE)]:
    preds, probs = predict(p)
    print(f"\n  {label}: \"{p}\"")
    print(f"    P(Mary)={float(probs[mary_id]):.3f}  "
          f"P(John)={float(probs[john_id]):.3f}  "
          f"P(Sam)={float(probs[sam_id]):.3f}")
    print(f"    top-3: {preds[:3]}   gen: {generate(p)}")


# ══════════════════════════════════════════════════════════════
# MOVE 1 — CONTRAST DESIGN COMPARISON
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MOVE 1: per-head delta decomposition at END, two contrast designs")
print("  (a) name-swap : CLEAN - SWAPPED  (cancels inhibition)")
print("  (b) no-dupe   : CLEAN - NODUPE   (keeps inhibition)")
print("=" * 70)

sweep_layers = [12, 16, 18, 20, 22, 24, 26, 28]
for L in sweep_layers:
    ctx_clean, _ = capture_context(CLEAN, L)
    ctx_swap, _ = capture_context(SWAPPED, L)
    ctx_nodupe, _ = capture_context(NODUPE, L)

    da = per_head_delta(ctx_clean, ctx_swap, L)    # name-swap
    db = per_head_delta(ctx_clean, ctx_nodupe, L)  # no-dupe

    da_top = sorted(da, key=lambda x: -x[1])[:5]
    db_top = sorted(db, key=lambda x: -x[1])[:5]

    print(f"\n  L{L}")
    print(f"    (a) name-swap top heads:")
    for h, norm, vec in da_top:
        print(f"        H{h:>2}  norm={norm:6.2f}  reads={topk_tok(vec, 5)}")
    print(f"    (b) no-dupe   top heads:")
    for h, norm, vec in db_top:
        print(f"        H{h:>2}  norm={norm:6.2f}  reads={topk_tok(vec, 5)}")

    del ctx_clean, ctx_swap, ctx_nodupe
    gc.collect()
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# MOVE 2 — ATTENTION PATTERNS
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MOVE 2: attention from END for candidate heads (CLEAN prompt)")
print("=" * 70)

ids_clean = tok.encode(CLEAN, add_special_tokens=False)
pos_tokens = [tok.decode([t]).strip() for t in ids_clean]
print(f"\n  positions: " +
      "  ".join(f"{i}:{t!r}" for i, t in enumerate(pos_tokens)))

model.set_attn_implementation("eager")  # required for output_attentions

# inspect the heads that the no-dupe contrast surfaces most, plus the
# original name-movers, at a couple of layers
inspect = {
    20: None, 24: None,  # filled below with each layer's top no-dupe heads
}
for L in [20, 24]:
    ctx_clean, _ = capture_context(CLEAN, L)
    ctx_nodupe, _ = capture_context(NODUPE, L)
    db = per_head_delta(ctx_clean, ctx_nodupe, L)
    cand = [h for h, _, _ in sorted(db, key=lambda x: -x[1])[:4]]
    cand = sorted(set(cand + [14, 1, 16]))  # union with original name-movers

    att, _ = attention_from_end(CLEAN, L)
    print(f"\n  L{L}  candidate heads {cand}")
    for h in cand:
        dist = att[h]
        top = torch.topk(dist, 3)
        desc = ", ".join(f"{pos_tokens[int(i)]!r}={float(v):.2f}"
                         for v, i in zip(top.values, top.indices))
        print(f"    H{h:>2}: attends → {desc}")
    del ctx_clean, ctx_nodupe
    gc.collect()
    torch.cuda.empty_cache()

model.set_attn_implementation("sdpa")  # restore: eager gives fp16 NaN logits


# ══════════════════════════════════════════════════════════════
# MOVE 3 — DENOISING / PATH PATCHING
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("MOVE 3: denoising — patch CLEAN head outputs into SWAPPED run")
print("  corrupted = SWAPPED (predicts John). How much P(Mary) is restored?")
print("=" * 70)

_, probs_corrupt = predict(SWAPPED)
_, probs_clean = predict(CLEAN)
p_corrupt = float(probs_corrupt[mary_id])
p_clean = float(probs_clean[mary_id])
print(f"\n  Corrupted (SWAPPED) P(Mary) = {p_corrupt:.3f}")
print(f"  Clean     (CLEAN)   P(Mary) = {p_clean:.3f}")
print(f"  Recovery% = (patched - corrupt) / (clean - corrupt) * 100\n")


def recov(p):
    denom = (p_clean - p_corrupt)
    return 100.0 * (p - p_corrupt) / denom if abs(denom) > 1e-6 else float('nan')


name_movers = [14, 1, 16]
for L in [20, 22, 24, 26, 28]:
    # per-layer candidate set from the no-dupe contrast
    ctx_clean, _ = capture_context(CLEAN, L)
    ctx_nodupe, _ = capture_context(NODUPE, L)
    db = per_head_delta(ctx_clean, ctx_nodupe, L)
    cand = [h for h, _, _ in sorted(db, key=lambda x: -x[1])[:4]]
    del ctx_clean, ctx_nodupe
    gc.collect(); torch.cuda.empty_cache()

    # cumulative denoising of name-movers
    _, probs_nm = patch_heads(SWAPPED, CLEAN, L, name_movers)
    p_nm = float(probs_nm[mary_id])

    # cumulative denoising of no-dupe candidates
    _, probs_cd = patch_heads(SWAPPED, CLEAN, L, cand)
    p_cd = float(probs_cd[mary_id])

    # all heads (upper bound for END-only patch)
    _, probs_all = patch_heads(SWAPPED, CLEAN, L, list(range(n_heads)))
    p_all = float(probs_all[mary_id])

    print(f"  L{L}:")
    print(f"    name-movers {name_movers}: P(Mary)={p_nm:.3f}  recov={recov(p_nm):6.1f}%")
    print(f"    no-dupe cand {cand}: P(Mary)={p_cd:.3f}  recov={recov(p_cd):6.1f}%")
    print(f"    ALL heads @END:          P(Mary)={p_all:.3f}  recov={recov(p_all):6.1f}%")

    # per-head denoising at this layer for the union set
    union = sorted(set(cand + name_movers))
    cells = []
    for h in union:
        _, probs_h = patch_heads(SWAPPED, CLEAN, L, [h])
        cells.append(f"H{h}:{recov(float(probs_h[mary_id])):.0f}%")
    print(f"    per-head recov: " + "  ".join(cells))
    gc.collect(); torch.cuda.empty_cache()


print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
