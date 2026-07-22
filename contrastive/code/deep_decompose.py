"""
Deep decomposition of contrastive signals.

Strategy:
1. Multi-contrast triangulation: same target vs many baselines,
   extract shared direction (the invariant causal signal)
2. Peel the shared direction, re-read remainder through W_U
3. For the W_U-unreadable residual: try PCA to find structure,
   then inject each PC to test causality
4. Cross-contrast residual analysis: do residuals from different
   contrasts share structure? (If yes, there's a hidden axis)
5. Axis projection: project residuals onto known axes from
   the taxonomy to see if "structural" signals decompose into
   known linguistic features

Do NOT accept "token-shaped vs opaque" without exhausting
decomposition methods.
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


def topk_str(logits, k=6):
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(idxs[j])]).strip()[:14], f"{vals[j]:.1f}")
            for j in range(k)]


def get_h(text, layer):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        out = model(torch.tensor([ids], device=DEV),
                    output_hidden_states=True)
    return out.hidden_states[layer][0, -1, :].float(), out


def inject_and_measure(text_b, delta, layer, pos=-1):
    ids_b = tok(text_b, add_special_tokens=False)["input_ids"]
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
        out = model(torch.tensor([ids_b], device=DEV))
    handle.remove()
    probs = torch.softmax(out.logits[0, -1].float(), -1)
    return probs


def recovery(probs_inj, target_id, p_a, p_b):
    gap = p_a - p_b
    if abs(gap) < 0.001:
        return float('nan')
    return (float(probs_inj[target_id]) - p_b) / gap * 100


def project_out(vec, directions):
    """Project vec orthogonal to all directions (list of vectors)."""
    if not directions:
        return vec.clone()
    # Stack and orthogonalize
    D = torch.stack(directions).float().to(DEV)
    Q, R = torch.linalg.qr(D.T)
    proj = Q @ (Q.T @ vec.float().to(DEV))
    return vec.float().to(DEV) - proj


def project_onto(vec, directions):
    """Project vec onto subspace spanned by directions."""
    if not directions:
        return torch.zeros_like(vec)
    D = torch.stack(directions).float().to(DEV)
    Q, R = torch.linalg.qr(D.T)
    return Q @ (Q.T @ vec.float().to(DEV))


ref_L = _sl(28)[0]

# ══════════════════════════════════════════════════════════════
# CASE 1: HOT DOG — multiple contrasts available
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 1: HOT DOG — multi-contrast decomposition")
print("=" * 70)

target = "The hot dog was"
contrasts = [
    ("cold_dog", "The cold dog was"),
    ("hot_cat", "The hot cat was"),
    ("old_dog", "The old dog was"),
    ("angry_dog", "The angry dog was"),
    ("hot_rod", "The hot rod was"),
    ("hot_choc", "The hot chocolate was"),
    ("the_dog", "The dog was"),
    ("a_hot_dog", "A hot dog was"),
]

h_target, out_target = get_h(target, ref_L)
probs_target = torch.softmax(out_target.logits[0, -1].float(), -1)
target_top1_id = torch.argmax(probs_target).item()
target_top1_tok = tok.decode([target_top1_id]).strip()
p_target = float(probs_target[target_top1_id])

# Compute all Δh vectors
deltas = {}
p_contrasts = {}
for label, text in contrasts:
    h_c, out_c = get_h(text, ref_L)
    deltas[label] = (h_target - h_c).cpu()
    probs_c = torch.softmax(out_c.logits[0, -1].float(), -1)
    p_contrasts[label] = float(probs_c[target_top1_id])
    del out_c
    torch.cuda.empty_cache()

del out_target
torch.cuda.empty_cache()

# ── Step 1: Pairwise cosine between all Δh ──
print("\n  Pairwise cosine between Δh vectors:")
labels = list(deltas.keys())
print(f"  {'':>12}", end="")
for lb in labels:
    print(f" {lb[:8]:>8}", end="")
print()
for i, li in enumerate(labels):
    print(f"  {li:>12}", end="")
    for j, lj in enumerate(labels):
        c = float(cos(deltas[li].unsqueeze(0), deltas[lj].unsqueeze(0)))
        print(f" {c:>+8.3f}", end="")
    print()

# ── Step 2: Shared direction via mean ──
all_dh = torch.stack([d.to(DEV) for d in deltas.values()])
mean_dir = all_dh.mean(dim=0)
mean_dir_norm = mean_dir / mean_dir.norm()

print(f"\n  Mean direction (shared across {len(contrasts)} contrasts):")
logits_mean = mean_dir @ W_U.T
print(f"    +[{', '.join(t[0] for t in topk_str(logits_mean, 8))}]")
print(f"    -[{', '.join(t[0] for t in topk_str(-logits_mean, 8))}]")
print(f"    ||mean_dir||: {mean_dir.norm():.1f}")

# ── Step 3: For each contrast, decompose into shared + unique ──
print(f"\n  Decomposing each Δh into shared_direction + unique:")
unique_dirs = {}
for label in labels:
    dh = deltas[label].to(DEV)
    shared_proj = (dh @ mean_dir_norm) * mean_dir_norm
    unique = dh - shared_proj
    unique_dirs[label] = unique.cpu()

    shared_logits = shared_proj @ W_U.T
    unique_logits = unique @ W_U.T
    s_tok = [t[0] for t in topk_str(shared_logits, 4)]
    u_tok = [t[0] for t in topk_str(unique_logits, 4)]

    # Causal test
    gap = p_target - p_contrasts[label]
    text_b = dict(contrasts)[label]
    if abs(gap) > 0.001:
        p_full = inject_and_measure(text_b, dh, ref_L)
        p_shared = inject_and_measure(text_b, shared_proj, ref_L)
        p_unique = inject_and_measure(text_b, unique, ref_L)
        rec_f = recovery(p_full, target_top1_id, p_target, p_contrasts[label])
        rec_s = recovery(p_shared, target_top1_id, p_target, p_contrasts[label])
        rec_u = recovery(p_unique, target_top1_id, p_target, p_contrasts[label])
        print(f"    {label:>12}: shared=[{', '.join(s_tok)}] rec={rec_s:>+6.1f}%  "
              f"unique=[{', '.join(u_tok)}] rec={rec_u:>+6.1f}%  "
              f"full={rec_f:>+6.1f}%")
    else:
        print(f"    {label:>12}: shared=[{', '.join(s_tok)}]  "
              f"unique=[{', '.join(u_tok)}]  (gap too small)")

# ── Step 4: PCA on the set of Δh vectors ──
print(f"\n  PCA on {len(contrasts)} Δh vectors:")
centered = all_dh - all_dh.mean(dim=0, keepdim=True)
U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
print(f"    Singular values: {', '.join(f'{s:.1f}' for s in S[:6])}")
print(f"    Variance explained: {', '.join(f'{(s**2/((S**2).sum()))*100:.1f}%' for s in S[:6])}")

for pc_idx in range(min(4, len(S))):
    pc_dir = Vt[pc_idx]
    pc_logits = pc_dir @ W_U.T
    pc_top = [t[0] for t in topk_str(pc_logits, 6)]
    pc_bot = [t[0] for t in topk_str(-pc_logits, 6)]
    print(f"    PC{pc_idx}: +[{', '.join(pc_top)}]  -[{', '.join(pc_bot)}]")

    # Causal test: inject PC direction (scaled to mean Δh norm)
    pc_scaled = pc_dir * all_dh[0].norm()  # scale to natural magnitude
    text_b0 = contrasts[0][1]  # use first contrast as baseline
    gap0 = p_target - p_contrasts[contrasts[0][0]]
    if abs(gap0) > 0.001:
        p_pc = inject_and_measure(text_b0, pc_scaled, ref_L)
        rec_pc = recovery(p_pc, target_top1_id, p_target, p_contrasts[contrasts[0][0]])
        print(f"          causal recovery: {rec_pc:>+6.1f}%")

# ── Step 5: Iterative multi-contrast peel ──
# Peel the mean direction, then PCA the residuals, peel PC0, etc.
print(f"\n  Iterative peel using multi-contrast structure:")
remainder = deltas[labels[0]].to(DEV)  # start with first contrast
text_b0 = contrasts[0][1]
gap0 = p_target - p_contrasts[labels[0]]
peeled_dirs = []

# Round 0: peel mean direction
mean_proj_on_dh = (remainder @ mean_dir_norm) * mean_dir_norm
remainder = remainder - mean_proj_on_dh
peeled_dirs.append(mean_dir_norm.clone())

logits_r = remainder @ W_U.T
r_top = topk_str(logits_r, 6)
r_bot = topk_str(-logits_r, 6)
print(f"    After removing mean_dir (||remainder||={remainder.norm():.1f}):")
print(f"      +[{', '.join(t[0] for t in r_top)}]")
print(f"      -[{', '.join(t[0] for t in r_bot)}]")
if abs(gap0) > 0.001:
    p_r = inject_and_measure(text_b0, remainder, ref_L)
    print(f"      causal recovery: {recovery(p_r, target_top1_id, p_target, p_contrasts[labels[0]]):>+6.1f}%")
    p_m = inject_and_measure(text_b0, mean_proj_on_dh, ref_L)
    print(f"      mean_dir recovery: {recovery(p_m, target_top1_id, p_target, p_contrasts[labels[0]]):>+6.1f}%")

# Rounds 1-3: find dominant direction in remainder via W_U, peel, repeat
for round_idx in range(1, 5):
    logits_r = remainder @ W_U.T
    # Find the single strongest token direction
    top_idx = torch.argmax(logits_r.abs())
    top_dir = W_U[top_idx]
    top_dir_norm = top_dir / top_dir.norm()
    top_tok_name = tok.decode([int(top_idx)]).strip()

    proj_on_top = (remainder @ top_dir_norm) * top_dir_norm
    remainder = remainder - proj_on_top
    peeled_dirs.append(top_dir_norm.clone())

    logits_r = remainder @ W_U.T
    r_top = topk_str(logits_r, 6)
    r_bot = topk_str(-logits_r, 6)
    print(f"    Round {round_idx}: peeled '{top_tok_name}' (||remainder||={remainder.norm():.1f}):")
    print(f"      +[{', '.join(t[0] for t in r_top)}]")
    print(f"      -[{', '.join(t[0] for t in r_bot)}]")
    if abs(gap0) > 0.001:
        p_r = inject_and_measure(text_b0, remainder, ref_L)
        p_p = inject_and_measure(text_b0, proj_on_top, ref_L)
        print(f"      remainder recovery: {recovery(p_r, target_top1_id, p_target, p_contrasts[labels[0]]):>+6.1f}%  "
              f"peeled_component recovery: {recovery(p_p, target_top1_id, p_target, p_contrasts[labels[0]]):>+6.1f}%")


# ══════════════════════════════════════════════════════════════
# CASE 2: CAUGHT COLD/FISH — the big residual case
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 2: CAUGHT COLD/FISH — decomposing the causal residual")
print("=" * 70)

target2 = "She caught a cold and went to"
contrasts2 = [
    ("fish", "She caught a fish and went to"),
    ("ball", "She caught a ball and went to"),
    ("bus", "She caught a bus and went to"),
    ("thief", "She caught a thief and went to"),
    ("glimpse", "She caught a glimpse and went to"),
]

h_t2, out_t2 = get_h(target2, ref_L)
probs_t2 = torch.softmax(out_t2.logits[0, -1].float(), -1)
t2_id = torch.argmax(probs_t2).item()
t2_tok = tok.decode([t2_id]).strip()
p_t2 = float(probs_t2[t2_id])

deltas2 = {}
p_c2 = {}
for label, text in contrasts2:
    h_c, out_c = get_h(text, ref_L)
    deltas2[label] = (h_t2 - h_c).cpu()
    probs_c = torch.softmax(out_c.logits[0, -1].float(), -1)
    p_c2[label] = float(probs_c[t2_id])
    del out_c
    torch.cuda.empty_cache()

del out_t2
torch.cuda.empty_cache()

print(f"  Target top-1: '{t2_tok}' (p={p_t2:.3f})")

# Pairwise cosine
print(f"\n  Pairwise cosine:")
labels2 = list(deltas2.keys())
for i, li in enumerate(labels2):
    for j, lj in enumerate(labels2):
        if j > i:
            c = float(cos(deltas2[li].unsqueeze(0), deltas2[lj].unsqueeze(0)))
            print(f"    {li:>8} × {lj:<8}: {c:+.3f}")

# Mean direction + decompose
all_dh2 = torch.stack([d.to(DEV) for d in deltas2.values()])
mean_dir2 = all_dh2.mean(dim=0)
mean_dir2_norm = mean_dir2 / mean_dir2.norm()

logits_m2 = mean_dir2 @ W_U.T
print(f"\n  Mean direction:")
print(f"    +[{', '.join(t[0] for t in topk_str(logits_m2, 8))}]")
print(f"    -[{', '.join(t[0] for t in topk_str(-logits_m2, 8))}]")

# Shared/unique decomposition + causality
print(f"\n  Shared vs unique causality:")
for label in labels2:
    dh = deltas2[label].to(DEV)
    shared = (dh @ mean_dir2_norm) * mean_dir2_norm
    unique = dh - shared
    text_b = dict(contrasts2)[label]
    gap = p_t2 - p_c2[label]
    if abs(gap) > 0.001:
        p_f = inject_and_measure(text_b, dh, ref_L)
        p_s = inject_and_measure(text_b, shared, ref_L)
        p_u = inject_and_measure(text_b, unique, ref_L)
        s_logits = shared @ W_U.T
        u_logits = unique @ W_U.T
        s_tok = [t[0] for t in topk_str(s_logits, 4)]
        u_tok = [t[0] for t in topk_str(u_logits, 4)]
        print(f"    {label:>8}: shared=[{', '.join(s_tok)}] rec={recovery(p_s, t2_id, p_t2, p_c2[label]):>+6.1f}%  "
              f"unique=[{', '.join(u_tok)}] rec={recovery(p_u, t2_id, p_t2, p_c2[label]):>+6.1f}%  "
              f"full={recovery(p_f, t2_id, p_t2, p_c2[label]):>+6.1f}%")

# PCA on residuals (after removing mean)
print(f"\n  PCA on residuals (after removing mean direction):")
residuals2 = []
for label in labels2:
    dh = deltas2[label].to(DEV)
    res = dh - (dh @ mean_dir2_norm) * mean_dir2_norm
    residuals2.append(res)
res_stack = torch.stack(residuals2)
res_centered = res_stack - res_stack.mean(dim=0, keepdim=True)
U2, S2, Vt2 = torch.linalg.svd(res_centered, full_matrices=False)
print(f"    Singular values: {', '.join(f'{s:.1f}' for s in S2[:5])}")
for pc_idx in range(min(3, len(S2))):
    pc_dir = Vt2[pc_idx]
    pc_logits = pc_dir @ W_U.T
    print(f"    ResidPC{pc_idx}: +[{', '.join(t[0] for t in topk_str(pc_logits, 6))}]  "
          f"-[{', '.join(t[0] for t in topk_str(-pc_logits, 6))}]")


# ══════════════════════════════════════════════════════════════
# CASE 3: SOME/ALL — the case where tokens hurt
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 3: SOME/ALL — multi-contrast + residual decomposition")
print("=" * 70)

target3 = "Some of the students passed the exam, so"
contrasts3 = [
    ("all", "All of the students passed the exam, so"),
    ("none", "None of the students passed the exam, so"),
    ("most", "Most of the students passed the exam, so"),
    ("few", "Few of the students passed the exam, so"),
    ("many", "Many of the students passed the exam, so"),
]

h_t3, out_t3 = get_h(target3, ref_L)
probs_t3 = torch.softmax(out_t3.logits[0, -1].float(), -1)
t3_id = torch.argmax(probs_t3).item()
t3_tok = tok.decode([t3_id]).strip()
p_t3 = float(probs_t3[t3_id])

deltas3 = {}
p_c3 = {}
for label, text in contrasts3:
    h_c, out_c = get_h(text, ref_L)
    deltas3[label] = (h_t3 - h_c).cpu()
    probs_c = torch.softmax(out_c.logits[0, -1].float(), -1)
    p_c3[label] = float(probs_c[t3_id])
    del out_c
    torch.cuda.empty_cache()
del out_t3
torch.cuda.empty_cache()

print(f"  Target top-1: '{t3_tok}' (p={p_t3:.3f})")

# Pairwise cosine
print(f"\n  Pairwise cosine:")
labels3 = list(deltas3.keys())
for i, li in enumerate(labels3):
    for j, lj in enumerate(labels3):
        if j > i:
            c = float(cos(deltas3[li].unsqueeze(0), deltas3[lj].unsqueeze(0)))
            print(f"    {li:>8} × {lj:<8}: {c:+.3f}")

# Mean direction
all_dh3 = torch.stack([d.to(DEV) for d in deltas3.values()])
mean_dir3 = all_dh3.mean(dim=0)
mean_dir3_norm = mean_dir3 / mean_dir3.norm()

logits_m3 = mean_dir3 @ W_U.T
print(f"\n  Mean direction:")
print(f"    +[{', '.join(t[0] for t in topk_str(logits_m3, 8))}]")
print(f"    -[{', '.join(t[0] for t in topk_str(-logits_m3, 8))}]")

# Per-contrast shared/unique + causality
print(f"\n  Shared vs unique causality:")
for label in labels3:
    dh = deltas3[label].to(DEV)
    shared = (dh @ mean_dir3_norm) * mean_dir3_norm
    unique = dh - shared
    text_b = dict(contrasts3)[label]
    gap = p_t3 - p_c3[label]
    if abs(gap) > 0.001:
        p_f = inject_and_measure(text_b, dh, ref_L)
        p_s = inject_and_measure(text_b, shared, ref_L)
        p_u = inject_and_measure(text_b, unique, ref_L)
        s_logits = shared @ W_U.T
        u_logits = unique @ W_U.T
        s_tok = [t[0] for t in topk_str(s_logits, 4)]
        u_tok = [t[0] for t in topk_str(u_logits, 4)]
        print(f"    {label:>8}: shared=[{', '.join(s_tok)}] rec={recovery(p_s, t3_id, p_t3, p_c3[label]):>+6.1f}%  "
              f"unique=[{', '.join(u_tok)}] rec={recovery(p_u, t3_id, p_t3, p_c3[label]):>+6.1f}%  "
              f"full={recovery(p_f, t3_id, p_t3, p_c3[label]):>+6.1f}%")
    else:
        s_logits = shared @ W_U.T
        u_logits = unique @ W_U.T
        s_tok = [t[0] for t in topk_str(s_logits, 4)]
        u_tok = [t[0] for t in topk_str(u_logits, 4)]
        print(f"    {label:>8}: shared=[{', '.join(s_tok)}]  unique=[{', '.join(u_tok)}]  (gap={gap:.4f})")


# ══════════════════════════════════════════════════════════════
# CASE 4: THEFT/MORAL — iterative multi-contrast peel
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("CASE 4: THEFT/MORAL — multi-contrast decomposition")
print("=" * 70)

target4 = "He slipped a bottle under his coat and walked out without paying. He"
contrasts4 = [
    ("paid", "He picked up a bottle, went to the register and paid. He"),
    ("returned", "He slipped a bottle under his coat then returned it to the shelf. He"),
    ("forgot", "He slipped a bottle under his coat, forgetting he already paid. He"),
    ("browsed", "He browsed the shelves, picked up a bottle, and put it back. He"),
]

h_t4, out_t4 = get_h(target4, ref_L)
probs_t4 = torch.softmax(out_t4.logits[0, -1].float(), -1)
t4_id = torch.argmax(probs_t4).item()
t4_tok = tok.decode([t4_id]).strip()
p_t4 = float(probs_t4[t4_id])

deltas4 = {}
p_c4 = {}
for label, text in contrasts4:
    h_c, out_c = get_h(text, ref_L)
    deltas4[label] = (h_t4 - h_c).cpu()
    probs_c = torch.softmax(out_c.logits[0, -1].float(), -1)
    p_c4[label] = float(probs_c[t4_id])
    del out_c
    torch.cuda.empty_cache()
del out_t4
torch.cuda.empty_cache()

print(f"  Target top-1: '{t4_tok}' (p={p_t4:.3f})")

# Pairwise cosine
labels4 = list(deltas4.keys())
print(f"\n  Pairwise cosine:")
for i, li in enumerate(labels4):
    for j, lj in enumerate(labels4):
        if j > i:
            c = float(cos(deltas4[li].unsqueeze(0), deltas4[lj].unsqueeze(0)))
            print(f"    {li:>8} × {lj:<8}: {c:+.3f}")

# Mean + decompose
all_dh4 = torch.stack([d.to(DEV) for d in deltas4.values()])
mean_dir4 = all_dh4.mean(dim=0)
mean_dir4_norm = mean_dir4 / mean_dir4.norm()

logits_m4 = mean_dir4 @ W_U.T
print(f"\n  Mean direction:")
print(f"    +[{', '.join(t[0] for t in topk_str(logits_m4, 8))}]")
print(f"    -[{', '.join(t[0] for t in topk_str(-logits_m4, 8))}]")

print(f"\n  Shared vs unique causality:")
for label in labels4:
    dh = deltas4[label].to(DEV)
    shared = (dh @ mean_dir4_norm) * mean_dir4_norm
    unique = dh - shared
    text_b = dict(contrasts4)[label]
    gap = p_t4 - p_c4[label]
    if abs(gap) > 0.001:
        p_f = inject_and_measure(text_b, dh, ref_L)
        p_s = inject_and_measure(text_b, shared, ref_L)
        p_u = inject_and_measure(text_b, unique, ref_L)
        s_logits = shared @ W_U.T
        u_logits = unique @ W_U.T
        s_tok = [t[0] for t in topk_str(s_logits, 4)]
        u_tok = [t[0] for t in topk_str(u_logits, 4)]
        print(f"    {label:>8}: shared=[{', '.join(s_tok)}] rec={recovery(p_s, t4_id, p_t4, p_c4[label]):>+6.1f}%  "
              f"unique=[{', '.join(u_tok)}] rec={recovery(p_u, t4_id, p_t4, p_c4[label]):>+6.1f}%  "
              f"full={recovery(p_f, t4_id, p_t4, p_c4[label]):>+6.1f}%")
    else:
        print(f"    {label:>8}: gap too small ({gap:.4f})")

# ── Final: try known axes on the residual ──
print(f"\n  Projecting theft residual onto known linguistic axes:")
# Build some reference axes at ref_L
axis_pairs = [
    ("positive/negative",
     "The movie was absolutely wonderful and everyone",
     "The movie was absolutely terrible and everyone"),
    ("past/future",
     "Yesterday it rained heavily and the streets were",
     "Tomorrow it will rain heavily and the streets will be"),
    ("know/doubt",
     "I know that the capital of France is",
     "I doubt that the capital of France is"),
    ("formal/informal",
     "Patient presented with acute chest pain. The diagnosis was",
     "Guy came in, chest really hurt. The diagnosis was"),
]

dh_theft = deltas4["paid"].to(DEV)  # theft - paid
# Get token-unreadable residual
logits_theft = dh_theft @ W_U.T
topk_idx = torch.topk(logits_theft.abs(), 20).indices
wu_rows = W_U[topk_idx]
Q, R = torch.linalg.qr(wu_rows.T)
theft_token_comp = Q @ (Q.T @ dh_theft)
theft_residual = dh_theft - theft_token_comp

for axis_name, text_pos, text_neg in axis_pairs:
    h_pos, _ = get_h(text_pos, ref_L)
    h_neg, _ = get_h(text_neg, ref_L)
    axis_dir = (h_pos - h_neg)
    axis_dir_norm = axis_dir / axis_dir.norm()

    proj_full = float(dh_theft @ axis_dir_norm)
    proj_res = float(theft_residual @ axis_dir_norm)
    cos_full = float(cos(dh_theft.unsqueeze(0), axis_dir.unsqueeze(0)))
    cos_res = float(cos(theft_residual.unsqueeze(0), axis_dir.unsqueeze(0)))

    print(f"    {axis_name:>20}: full cos={cos_full:+.3f} proj={proj_full:>+7.1f}  "
          f"residual cos={cos_res:+.3f} proj={proj_res:>+7.1f}")

torch.cuda.empty_cache()

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
